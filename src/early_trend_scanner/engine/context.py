"""Market-regime context: index/volatility proxies feeding signal features.

Motivated by live observation (2026-08-31): signals that fight the broad
market's direction stall even when confirmed, and rising volatility steers
everything. The tape of one or two context symbols — a market proxy (SPY)
and a fear proxy (VXX, since the VIX index itself is not on the equities
feed) — is aggregated with the same 1-second machinery as scan symbols, and
each Snapshot gets direction-aligned regime velocities. They are features
for the score/model layer, never hard gates: all of them read 0.0 (neutral)
when context data is absent or stale, so replays, tests and outages degrade
gracefully.
"""

from __future__ import annotations

from ..data.models import Trade
from .aggregator import SymbolAggregator

_CLAMP_BPS_S = 10.0


class ContextTracker:
    def __init__(self, symbols: list[str], stale_after_s: float = 45.0) -> None:
        # Positional convention: symbols[0] = market proxy (direction-aligned
        # features), symbols[1] = volatility/fear proxy (unaligned).
        self.symbols = list(symbols)
        self.stale_after_s = stale_after_s
        self.aggs = {s: SymbolAggregator(s) for s in self.symbols}
        self._last_ts: dict[str, float] = {}

    @property
    def market_symbol(self) -> str | None:
        return self.symbols[0] if self.symbols else None

    @property
    def vol_symbol(self) -> str | None:
        return self.symbols[1] if len(self.symbols) > 1 else None

    def on_trade(self, t: Trade) -> None:
        agg = self.aggs.get(t.symbol)
        if agg is None:
            return
        agg.on_trade(t)
        if t.updates_last:
            self._last_ts[t.symbol] = t.ts

    def on_second_tick(self, now_ts: float) -> None:
        for agg in self.aggs.values():
            agg.roll_to(int(now_ts) - 1)

    def vel_bps_s(self, symbol: str | None, now_ts: float, window_s: int) -> float:
        """Signed velocity in bps/s over window_s; 0.0 when unknown or stale."""
        if symbol is None:
            return 0.0
        agg = self.aggs.get(symbol)
        if agg is None:
            return 0.0
        if now_ts - self._last_ts.get(symbol, 0.0) > self.stale_after_s:
            return 0.0
        last = agg.last_price
        ago = agg.price_ago(window_s)
        if last <= 0.0 or ago is None or ago <= 0.0:
            return 0.0
        vel = (last - ago) / ago * 1e4 / window_s
        return max(-_CLAMP_BPS_S, min(vel, _CLAMP_BPS_S))

    def features(self, now_ts: float, direction: int) -> tuple[float, float, float, float]:
        """(market 1m x dir, market 5m x dir, fear 1m, fear 5m), all bps/s.

        Positive aligned values mean the market is moving WITH the candidate
        signal. Fear-proxy velocities are unaligned (positive = fear rising);
        both a fast 1m pulse and a 5m trend are exposed because the volatility
        regime can flip mid-session and recent prints must dominate.
        """
        spy1 = self.vel_bps_s(self.market_symbol, now_ts, 60) * direction
        spy5 = self.vel_bps_s(self.market_symbol, now_ts, 300) * direction
        fear1 = self.vel_bps_s(self.vol_symbol, now_ts, 60)
        fear5 = self.vel_bps_s(self.vol_symbol, now_ts, 300)
        return spy1, spy5, fear1, fear5
