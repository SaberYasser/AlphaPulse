"""Deterministic synthetic session generator for replay tests.

Produces a realistic-enough microstructure stream (trades + NBBO quotes) with
scripted ground-truth events: an upside breakout after compression, a downside
breakdown, a failed sweep (fakeout) and a quiet control symbol. Expansions
start with a sharp impulse (40% of the move in the first 15s) — the shape the
scanner is meant to catch early. Determinism comes from an explicit seed and a
stable CRC-based per-symbol stream; no wall clock, no global RNG.
"""

from __future__ import annotations

import heapq
import random
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field, replace

from ..config import EngineCfg
from ..data.models import Quote, Trade

Event = Trade | Quote

_IMPULSE_S = 15.0
_IMPULSE_SHARE = 0.40
# The scripted impulse runs at this multiple of the configured minimum
# velocity, so the "unmistakable breakout" stays unmistakable under any
# engine profile (matches compute_score's velocity saturation at 3x).
_IMPULSE_VEL_MULT = 2.5
_MOVE_BPS_CAP = 300.0


@dataclass
class Truth:
    symbol: str
    kind: str  # breakout_up | breakout_down | fakeout_up | quiet
    event_ts: float = 0.0
    direction: int = 0
    move_end_ts: float = 0.0


@dataclass
class ScriptedSymbol:
    symbol: str
    base_price: float
    kind: str
    breakout_at_s: float = 660.0  # seconds after session open
    quiet_rate: float = 3.0  # trades/sec during quiet phases
    burst_rate: float = 25.0  # trades/sec during the event
    move_bps: float = 60.0  # total move over the expansion
    move_duration_s: float = 120.0
    sweep_bps: float = 20.0  # fakeout: sweep impulse size
    sweep_s: float = 12.0  # fakeout: sweep impulse duration
    truth: Truth = field(init=False)

    def __post_init__(self) -> None:
        self.truth = Truth(symbol=self.symbol, kind=self.kind)


DEFAULT_SCRIPT = [
    ScriptedSymbol("SYNUP", 100.0, "breakout_up", breakout_at_s=660.0),
    ScriptedSymbol("SYNDN", 250.0, "breakout_down", breakout_at_s=780.0, move_bps=70.0),
    ScriptedSymbol("SYNFK", 50.0, "fakeout_up", breakout_at_s=720.0),
    ScriptedSymbol("SYNQT", 75.0, "quiet"),
]


class SyntheticSession:
    def __init__(
        self,
        open_ts: float = 1_756_000_000.0 - (1_756_000_000.0 % 60),
        duration_s: float = 1500.0,
        script: list[ScriptedSymbol] | None = None,
        seed: int = 7,
        engine_cfg: EngineCfg | None = None,
    ) -> None:
        self.open_ts = open_ts
        self.close_ts = open_ts + duration_s
        self.duration_s = duration_s
        # Copy scripted symbols so DEFAULT_SCRIPT (and its Truth objects) are
        # never shared or mutated across sessions.
        self.script = [replace(s) for s in (script if script is not None else DEFAULT_SCRIPT)]
        self.seed = seed
        if engine_cfg is not None:
            self._scale_to_profile(engine_cfg)

    def _scale_to_profile(self, cfg: EngineCfg) -> None:
        """Size the scripted moves relative to the configured trigger minima.

        The synthetic session's purpose is regression-testing the pipeline
        mechanics, so its ground-truth expansions must stay clearly above
        whatever thresholds the profile sets — otherwise a stricter live
        tuning silently turns the detection contract into a coin flip.
        """
        floor_vel = _IMPULSE_VEL_MULT * cfg.vel_min_bps_s
        for sc in self.script:
            if sc.kind in ("breakout_up", "breakout_down"):
                cur_vel = sc.move_bps * _IMPULSE_SHARE / _IMPULSE_S
                if floor_vel > cur_vel:
                    sc.move_bps = min(floor_vel * _IMPULSE_S / _IMPULSE_SHARE, _MOVE_BPS_CAP)
            elif sc.kind == "fakeout_up":
                sc.sweep_bps = max(sc.sweep_bps, floor_vel * sc.sweep_s)

    @property
    def truths(self) -> list[Truth]:
        return [s.truth for s in self.script]

    def events(self) -> Iterator[Event]:
        """All symbols' events merged in strict timestamp order."""
        streams = [self._symbol_events(s) for s in self.script]
        return heapq.merge(*streams, key=lambda e: e.ts)

    # ------------------------------------------------------------- per symbol

    def _drift_per_s(self, sc: ScriptedSymbol, t: float, price: float, direction: int) -> float:
        """Deterministic price slope ($/s) at time t."""
        event_start = self.open_ts + sc.breakout_at_s
        move_total = sc.base_price * sc.move_bps / 1e4
        elapsed = t - event_start

        if sc.kind == "quiet" or elapsed < -240.0:
            return (sc.base_price - price) * 0.01  # loose mean reversion
        if elapsed < 0.0:
            return (sc.base_price - price) * 0.05  # compression: tight reversion
        if sc.kind == "fakeout_up":
            if elapsed < sc.sweep_s:  # sweep impulse beyond the level
                return sc.base_price * (sc.sweep_bps / 1e4) / sc.sweep_s
            if elapsed < 60.0:  # sharp reversal back through the level
                rev_bps = sc.sweep_bps * 1.75
                return -sc.base_price * (rev_bps / 1e4) / (60.0 - sc.sweep_s)
            return (sc.base_price - price) * 0.02
        if elapsed < _IMPULSE_S:
            return direction * move_total * _IMPULSE_SHARE / _IMPULSE_S
        if elapsed < sc.move_duration_s:
            rest = sc.move_duration_s - _IMPULSE_S
            return direction * move_total * (1.0 - _IMPULSE_SHARE) / rest
        target = sc.base_price + direction * move_total
        return (target - price) * 0.02  # hold the new level

    def _symbol_events(self, sc: ScriptedSymbol) -> Iterator[Event]:
        rng = random.Random(zlib.crc32(sc.symbol.encode()) ^ self.seed)
        price = sc.base_price
        t = self.open_ts
        direction = 1 if sc.kind in ("breakout_up", "fakeout_up") else -1
        event_start = self.open_ts + sc.breakout_at_s
        event_end = event_start + sc.move_duration_s
        next_quote = self.open_ts
        seq = 0

        sc.truth.event_ts = event_start
        sc.truth.direction = direction if sc.kind != "quiet" else 0
        sc.truth.move_end_ts = event_end

        while t < self.close_ts:
            elapsed = t - event_start
            in_event = sc.kind != "quiet" and 0.0 <= elapsed < sc.move_duration_s
            in_fake_reversal = sc.kind == "fakeout_up" and sc.sweep_s <= elapsed < 60.0
            in_compression = sc.kind != "quiet" and -240.0 <= elapsed < 0.0

            rate = sc.burst_rate if in_event else sc.quiet_rate
            dt = max(rng.expovariate(rate), 0.001)
            t += dt
            if t >= self.close_ts:
                break

            noise_sigma = sc.base_price * (0.12e-4 if in_compression else 0.30e-4)
            price = max(
                0.5,
                price
                + self._drift_per_s(sc, t, price, direction) * dt
                + rng.gauss(0.0, noise_sigma),
            )

            # NBBO quotes ~4/s, emitted BEFORE the trade so the per-symbol
            # stream stays timestamp-sorted (heapq.merge contract).
            spread = price * 1.2e-4
            while next_quote <= t:
                yield Quote(
                    symbol=sc.symbol,
                    ts=next_quote,
                    bid=round(price - spread / 2, 4),
                    bid_size=rng.randint(1, 20),
                    ask=round(price + spread / 2, 4),
                    ask_size=rng.randint(1, 20),
                )
                next_quote += 0.25

            if in_fake_reversal:
                buy_p = 0.20
                size = int(rng.choice((100, 200, 300)))
            elif in_event:
                buy_p = 0.85 if direction > 0 else 0.15
                size = int(rng.choice((100, 200, 300, 500, 800)))
            else:
                buy_p = 0.5
                size = int(rng.choice((100, 100, 100, 200, 300)))
            is_buy = rng.random() < buy_p
            trade_price = round(price + (spread / 2 if is_buy else -spread / 2), 4)
            seq += 1
            yield Trade(
                symbol=sc.symbol,
                ts=t,
                price=trade_price,
                size=size,
                conditions=("@",),
                exchange="X",
                tape="C",
                trade_id=seq,
            )
