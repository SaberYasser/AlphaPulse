"""Startup warmup over REST: baselines, static levels, and mid-session catch-up.

Loads at least `baseline_sessions` completed sessions of 1-minute bars for
minute-of-day baselines, the prior-day daily bar, today's premarket range and —
when starting mid-session — today's regular-hours minutes so levels and VWAP
are correct before the stream takes over.
"""

from __future__ import annotations

import logging
import time

from .clock import MarketClock
from .config import Config
from .data.models import Bar, Minute, SessionInfo
from .data.rest import AlpacaRest, AlpacaRestError
from .engine.baseline import MinuteBaseline
from .engine.symbol_engine import SymbolEngine

log = logging.getLogger(__name__)

# Free-tier accounts may query historical SIP data except the most recent
# minutes; keep a safety margin beyond Alpaca's 15-minute restriction.
_SIP_RECENCY_S = 16 * 60


async def _levels_bars(
    rest: AlpacaRest,
    symbols: list[str],
    timeframe: str,
    t0: float,
    t1: float,
    hist_feed: str,
    live_feed: str,
) -> dict[str, list[Bar]]:
    """Price-level bars from the history feed, falling back to the live feed."""
    try:
        return await rest.bars(symbols, timeframe, t0, t1, hist_feed)
    except AlpacaRestError as e:
        if hist_feed == live_feed:
            raise
        log.warning(
            "history feed %s unavailable for %s bars (%s) — falling back to %s",
            hist_feed,
            timeframe,
            e,
            live_feed,
        )
        return await rest.bars(symbols, timeframe, t0, t1, live_feed)


def bars_to_minutes(bars: list[Bar]) -> list[Minute]:
    return [
        Minute(
            ts=int(b.ts) - (int(b.ts) % 60),
            open=b.open,
            high=b.high,
            low=b.low,
            close=b.close,
            vol=b.volume,
            n=b.trade_count,
            dollar=(b.vwap if b.vwap > 0 else b.close) * b.volume,
        )
        for b in bars
    ]


async def warmup(
    rest: AlpacaRest,
    clock: MarketClock,
    cfg: Config,
    engines: dict[str, SymbolEngine],
    baseline: MinuteBaseline,
    session: SessionInfo,
    now_ts: float,
) -> None:
    symbols = list(engines)  # warm exactly the engines passed, not the full universe
    feed = cfg.data.feed
    # Price LEVELS come from the consolidated history feed (historical SIP is
    # available on every plan and sees the whole market — IEX alone often has
    # no premarket tape at all). Volume-bearing warmup (baselines, VWAP seed)
    # must stay on the live feed so its units match the live stream.
    hist_feed = cfg.data.history_feed or feed

    # 1) Minute-of-day baselines from completed sessions.
    hist = clock.completed_sessions_before(session.open_ts, cfg.data.baseline_sessions)
    if len(hist) < cfg.data.baseline_sessions:
        log.warning(
            "only %d completed sessions available for baselines (wanted %d)",
            len(hist),
            cfg.data.baseline_sessions,
        )
    if hist:
        bars = await rest.bars(symbols, "1Min", hist[0].open_ts, hist[-1].close_ts, feed)
        baseline.build(bars, hist)
        log.info("baselines built from %d sessions", len(hist))

    # 2) Prior-day high/low/close from daily bars. Alpaca includes today's
    # PARTIAL daily bar (timestamped midnight ET) in this range — exclude it.
    daily = await _levels_bars(
        rest, symbols, "1Day", session.open_ts - 14 * 86400, session.open_ts - 1, hist_feed, feed
    )
    prior_cutoff = session.open_ts - 10 * 3600
    daily = {sym: [b for b in bars if b.ts < prior_cutoff] for sym, bars in daily.items()}

    # 3) Today's premarket range (04:00 ET onward). The consolidated part is
    # clipped to the free-tier SIP recency boundary; the freshest slice tops
    # up from the live feed (prices only, so mixing feeds is safe).
    pm_start = session.open_ts - 5.5 * 3600
    pm_end = min(now_ts, session.open_ts)
    pm_bars: dict[str, list[Bar]] = {s: [] for s in symbols}
    if pm_end > pm_start:
        sip_safe_end = time.time() - _SIP_RECENCY_S if hist_feed == "sip" else pm_end
        hist_end = min(pm_end, sip_safe_end)
        if hist_end > pm_start:
            pm_bars = await _levels_bars(rest, symbols, "1Min", pm_start, hist_end, hist_feed, feed)
        if hist_end < pm_end and feed != hist_feed:
            tail = await rest.bars(symbols, "1Min", max(pm_start, hist_end), pm_end, feed)
            for tail_sym, tail_bars in tail.items():
                pm_bars.setdefault(tail_sym, []).extend(tail_bars)

    # 4) Today's regular-hours minutes when starting mid-session.
    rth_bars: dict[str, list[Bar]] = {s: [] for s in symbols}
    if now_ts > session.open_ts + 60:
        rth_bars = await rest.bars(symbols, "1Min", session.open_ts, now_ts, feed)

    for sym in symbols:
        eng = engines[sym]
        eng.set_session(session.open_ts)

        pdh = pdl = pdc = 0.0
        if daily.get(sym):
            last_day = daily[sym][-1]
            pdh, pdl, pdc = last_day.high, last_day.low, last_day.close
        pmh = pml = 0.0
        pm = [b for b in pm_bars.get(sym, []) if b.ts < session.open_ts]
        if pm:
            pmh = max(b.high for b in pm)
            pml = min(b.low for b in pm)
        eng.seed_static_levels(pdh=pdh, pdl=pdl, pdc=pdc, pmh=pmh, pml=pml)
        # Fresh start at/near the bell: seed the ring with the last premarket
        # print so velocity/acceleration are live from the first RTH second.
        if pm and now_ts <= session.open_ts + 60:
            eng.agg.seed_preopen(pm[-1].close, session.open_ts)

        today = [b for b in rth_bars.get(sym, []) if b.ts >= session.open_ts]
        if today:
            minutes = bars_to_minutes(today)
            eng.seed_minutes(minutes)
            pv = sum(m.dollar for m in minutes)
            v = float(sum(m.vol for m in minutes))
            eng.agg.seed_vwap(pv, v)
            eng.agg.session_high = max(m.high for m in minutes)
            eng.agg.session_low = min(m.low for m in minutes)
            or_end = session.open_ts + cfg.session.opening_range_min * 60
            or_minutes = [m for m in minutes if m.ts < or_end]
            if or_minutes and now_ts >= or_end:
                eng.seed_opening_range(
                    max(m.high for m in or_minutes), min(m.low for m in or_minutes)
                )
        log.info(
            "warmup %s: PDH=%.2f PDL=%.2f PMH=%.2f PML=%.2f rth_minutes=%d",
            sym,
            pdh,
            pdl,
            pmh,
            pml,
            len(today),
        )
