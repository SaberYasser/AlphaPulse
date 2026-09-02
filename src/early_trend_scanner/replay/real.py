"""Replay a real historical session slice from Alpaca REST (no lookahead).

Trades (and optionally NBBO quotes) are fetched for an ET time window, merged
in timestamp order and pushed through the identical live pipeline.

Two modes:
- combined (default for <=3 symbols): one runner, one merged event stream —
  preserves the cross-symbol global alert limiter exactly.
- per-symbol (default for bigger universes / quote-less runs): each symbol is
  fetched and replayed independently so memory stays bounded; results are
  re-aggregated into one report. The global alert limiter is per-run in this
  mode, so universe-wide alert caps are not applied across symbols.

Quotes are the heavy part of full-day tick history (tens of millions of NBBO
updates per liquid symbol). `with_quotes=False` skips them; trade sides then
fall back to the tick rule, while volume, price and baselines are unaffected.
"""

from __future__ import annotations

import heapq
import logging
from collections.abc import Iterable
from datetime import date

from ..clock import MarketClock
from ..config import Config, Secrets
from ..data.models import Quote, SessionInfo, Trade
from ..data.rest import AlpacaRest
from ..engine.baseline import MinuteBaseline
from ..store.metrics import MetricsTracker
from ..timeutil import parse_rfc3339
from ..warmup import warmup
from .engine import ReplayResult, ReplayRunner

log = logging.getLogger(__name__)

# Pagination caps sized for one full RTH session of a very liquid symbol.
_TRADE_PAGES_FULL_DAY = 400
_QUOTE_PAGES_FULL_DAY = 400


async def replay_real(
    cfg: Config,
    secrets: Secrets,
    day: date,
    symbols: list[str],
    start_min: int = 0,
    end_min: int = 390,
    with_quotes: bool = True,
    per_symbol: bool | None = None,
) -> ReplayResult:
    if per_symbol is None:
        per_symbol = len(symbols) > 3 or not with_quotes

    async with AlpacaRest(
        secrets.alpaca_key,
        secrets.alpaca_secret,
        cfg.data.data_base_url,
        cfg.data.trading_base_url,
    ) as rest:
        sessions = await rest.calendar(day.isoformat(), day.isoformat())
        target = [s for s in sessions if s.date_str == day.isoformat()]
        if not target:
            raise SystemExit(f"{day} is not a trading session")
        session: SessionInfo = target[0]
        t0 = session.open_ts + start_min * 60
        t1 = min(session.open_ts + end_min * 60, session.close_ts)

        clock = MarketClock(rest)
        await clock.load_sessions(back_days=40, fwd_days=1)

        # Regime-context tape (SPY/VXX): fetched once, merged into every stream.
        ctx_streams: dict[str, list[Trade]] = {}
        ctx_syms = list(cfg.data.context_symbols)
        if ctx_syms:
            log.info("fetching context tape %s", ctx_syms)
            ctx_raw = await rest.trades(ctx_syms, t0, t1, cfg.data.feed, max_pages=200)
            ctx_streams = {
                c: sorted(ctx_raw.get(c, []), key=lambda e: e.ts) for c in ctx_syms
            }

        if not per_symbol:
            runner = ReplayRunner(cfg, session, symbols, baseline=MinuteBaseline())
            await warmup(rest, clock, cfg, runner.engines, runner.baseline, session, t0)
            events = await _fetch_events(rest, cfg, symbols, t0, t1, with_quotes, ctx_streams)
            return runner.run(events)

        combined = ReplayResult()
        agg = MetricsTracker(session.open_ts, session.close_ts)
        for i, sym in enumerate(symbols, 1):
            log.info("[%d/%d] %s: warmup + fetch (feed=%s)", i, len(symbols), sym, cfg.data.feed)
            runner = ReplayRunner(cfg, session, [sym], baseline=MinuteBaseline())
            await warmup(rest, clock, cfg, runner.engines, runner.baseline, session, t0)
            events = await _fetch_events(rest, cfg, [sym], t0, t1, with_quotes, ctx_streams)
            res = runner.run(events)
            log.info(
                "[%d/%d] %s: %d events, %d signals",
                i,
                len(symbols),
                sym,
                res.events_processed,
                len([s for s in res.signals if not s.suppressed]),
            )
            combined.signals.extend(res.signals)
            combined.labels.extend(res.labels)
            combined.messages.extend(res.messages)
            combined.events_processed += res.events_processed
            combined.wall_seconds += res.wall_seconds
            combined.peak_rss_mb = max(combined.peak_rss_mb, res.peak_rss_mb)
            combined.ring_sizes.update(res.ring_sizes)
            combined.rejections.update(res.rejections)
            for sig in res.signals:
                agg.on_alert(sig)
                if sig.resolution in ("CONFIRMED", "FAILED"):
                    agg.on_resolution(sig)
            for lr in res.labels:
                agg.on_label(lr)
        combined.messages.sort(key=lambda m: m[0])
        combined.metrics = agg.summary()
        return combined


async def _fetch_events(
    rest: AlpacaRest,
    cfg: Config,
    symbols: list[str],
    t0: float,
    t1: float,
    with_quotes: bool,
    ctx_streams: dict[str, list[Trade]] | None = None,
) -> Iterable[Trade | Quote]:
    trades = await rest.trades(symbols, t0, t1, cfg.data.feed, max_pages=_TRADE_PAGES_FULL_DAY)
    quotes_raw = (
        await rest.quotes(symbols, t0, t1, cfg.data.feed, max_pages=_QUOTE_PAGES_FULL_DAY)
        if with_quotes
        else {}
    )

    def quote_stream(sym: str) -> list[Quote]:
        out = []
        for r in quotes_raw.get(sym, []):
            bid = float(r.get("bp", 0.0))
            ask = float(r.get("ap", 0.0))
            if bid <= 0 or ask <= 0:
                continue
            out.append(
                Quote(
                    symbol=sym,
                    ts=parse_rfc3339(r["t"]),
                    bid=bid,
                    bid_size=int(r.get("bs", 0)),
                    ask=ask,
                    ask_size=int(r.get("as", 0)),
                )
            )
        return out

    streams: list[Iterable[Trade | Quote]] = []
    for sym in symbols:
        streams.append(sorted(trades.get(sym, []), key=lambda e: e.ts))
        if with_quotes:
            streams.append(quote_stream(sym))
    scanned = set(symbols)
    for context_symbol, context_events in (ctx_streams or {}).items():
        if context_symbol not in scanned:
            streams.append(context_events)
    return heapq.merge(*streams, key=lambda e: e.ts)
