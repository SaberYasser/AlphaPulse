"""Key price levels, swing structure, compression state and sweep/reclaim tracking.

Static levels (prior day, premarket) are set at warmup; the opening range locks
after its window; dynamic levels (rolling ranges, swing fractals) refresh on
each completed minute. Everything is small, bounded and recomputed cheaply.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum

from ..data.models import Minute


class LevelKind(IntEnum):
    PDH = 0  # prior-day high
    PDL = 1  # prior-day low
    PDC = 2  # prior-day close
    PMH = 3  # premarket high
    PML = 4  # premarket low
    ORH = 5  # opening-range high
    ORL = 6  # opening-range low
    VWAP = 7
    RANGE_H = 8  # 10-minute rolling high
    RANGE_L = 9
    MICRO_H = 10  # 3-minute rolling high (short consolidations)
    MICRO_L = 11
    SWING_H = 12  # 1-minute fractal swing high
    SWING_L = 13
    SWING5_H = 14  # 5-minute fractal swing high
    SWING5_L = 15


# Break direction implied when price escapes upward through the level.
_UPPER_KINDS = {
    LevelKind.PDH,
    LevelKind.PMH,
    LevelKind.ORH,
    LevelKind.RANGE_H,
    LevelKind.MICRO_H,
    LevelKind.SWING_H,
    LevelKind.SWING5_H,
}

_PRIORITY = {
    LevelKind.PDH: 0,
    LevelKind.PDL: 0,
    LevelKind.PDC: 1,
    LevelKind.PMH: 1,
    LevelKind.PML: 1,
    LevelKind.ORH: 2,
    LevelKind.ORL: 2,
    LevelKind.VWAP: 3,
    LevelKind.SWING5_H: 4,
    LevelKind.SWING5_L: 4,
    LevelKind.SWING_H: 5,
    LevelKind.SWING_L: 5,
    LevelKind.RANGE_H: 6,
    LevelKind.RANGE_L: 6,
    LevelKind.MICRO_H: 7,
    LevelKind.MICRO_L: 7,
}

BREAK_VERB = {
    LevelKind.VWAP: "cross",
    LevelKind.SWING_H: "pivot",
    LevelKind.SWING_L: "pivot",
    LevelKind.SWING5_H: "pivot",
    LevelKind.SWING5_L: "pivot",
}


@dataclass(slots=True)
class Level:
    price: float
    kind: LevelKind
    ts: float = 0.0

    @property
    def key(self) -> tuple[int, float]:
        return (int(self.kind), round(self.price, 2))


@dataclass(slots=True)
class _SweepState:
    beyond_since: float = 0.0
    direction: int = 0  # +1 price went above the level, -1 below
    extreme: float = 0.0


@dataclass(slots=True)
class ReclaimEvent:
    level: Level
    direction: int  # trade direction implied by the reclaim
    ts: float
    swept_extreme: float


@dataclass(slots=True)
class Compression:
    ratio: float = 1.0  # last-3m avg range / prior-10m avg range
    tightening: int = 0  # consecutive minutes of lower highs AND higher lows
    active: bool = False


class LevelBook:
    def __init__(
        self,
        symbol: str,
        sweep_window_s: float = 90.0,
        sweep_max_bps: float = 20.0,
        compression_max_ratio: float = 0.75,
        merge_bps: float = 2.0,
    ) -> None:
        self.symbol = symbol
        self.sweep_window_s = sweep_window_s
        self.sweep_max_bps = sweep_max_bps
        self.compression_max_ratio = compression_max_ratio
        self.merge_bps = merge_bps

        self.static_levels: list[Level] = []
        self.dynamic_levels: list[Level] = []
        self.or_levels: list[Level] = []
        self.compression = Compression()
        self.range5m: float = 0.0  # high-low of the last 5 completed minutes

        self._sweeps: dict[tuple[int, float], _SweepState] = {}
        self._reclaims: deque[ReclaimEvent] = deque(maxlen=8)
        self._cache: list[Level] = []
        self._cache_sec: int = -1
        self._cache_vwap: float = 0.0

    # ------------------------------------------------------------------ setup

    def set_static(
        self,
        pdh: float = 0.0,
        pdl: float = 0.0,
        pdc: float = 0.0,
        pmh: float = 0.0,
        pml: float = 0.0,
    ) -> None:
        self.static_levels = [
            Level(p, k)
            for p, k in (
                (pdh, LevelKind.PDH),
                (pdl, LevelKind.PDL),
                (pdc, LevelKind.PDC),
                (pmh, LevelKind.PMH),
                (pml, LevelKind.PML),
            )
            if p > 0.0
        ]
        self._cache_sec = -1

    def set_opening_range(self, or_high: float, or_low: float) -> None:
        self.or_levels = []
        self._cache_sec = -1
        if or_high > 0.0:
            self.or_levels.append(Level(or_high, LevelKind.ORH))
        if or_low > 0.0:
            self.or_levels.append(Level(or_low, LevelKind.ORL))

    # ----------------------------------------------------------- minute close

    def on_minute(self, minutes: deque[Minute]) -> None:
        """Recompute dynamic levels + compression from completed minutes."""
        n = len(minutes)
        self._cache_sec = -1
        if n == 0:
            return
        recent = list(minutes)[-13:]  # at most 13 minutes examined
        self.dynamic_levels = []

        last10 = recent[-10:]
        self.dynamic_levels.append(
            Level(max(m.high for m in last10), LevelKind.RANGE_H, last10[-1].ts)
        )
        self.dynamic_levels.append(
            Level(min(m.low for m in last10), LevelKind.RANGE_L, last10[-1].ts)
        )
        last3 = recent[-3:]
        self.dynamic_levels.append(
            Level(max(m.high for m in last3), LevelKind.MICRO_H, last3[-1].ts)
        )
        self.dynamic_levels.append(
            Level(min(m.low for m in last3), LevelKind.MICRO_L, last3[-1].ts)
        )
        last5 = recent[-5:]
        self.range5m = max(m.high for m in last5) - min(m.low for m in last5)

        self._add_swings(minutes)
        self._update_compression(recent)
        self._prune_sweep_states()

    def _add_swings(self, minutes: deque[Minute]) -> None:
        ms = list(minutes)[-40:]
        swings_h: list[Level] = []
        swings_l: list[Level] = []
        for i in range(2, len(ms) - 2):
            m = ms[i]
            if (
                m.high >= ms[i - 1].high
                and m.high >= ms[i - 2].high
                and m.high > ms[i + 1].high
                and m.high > ms[i + 2].high
            ):
                swings_h.append(Level(m.high, LevelKind.SWING_H, m.ts))
            if (
                m.low <= ms[i - 1].low
                and m.low <= ms[i - 2].low
                and m.low < ms[i + 1].low
                and m.low < ms[i + 2].low
            ):
                swings_l.append(Level(m.low, LevelKind.SWING_L, m.ts))
        self.dynamic_levels.extend(swings_h[-3:])
        self.dynamic_levels.extend(swings_l[-3:])

        # 5-minute swings from aggregated minutes
        fives: list[tuple[float, float, float]] = []  # (ts, high, low)
        for i in range(0, len(ms) - 4, 5):
            chunk = ms[i : i + 5]
            fives.append((chunk[-1].ts, max(m.high for m in chunk), min(m.low for m in chunk)))
        for i in range(1, len(fives) - 1):
            ts, hi, lo = fives[i]
            if hi > fives[i - 1][1] and hi > fives[i + 1][1]:
                self.dynamic_levels.append(Level(hi, LevelKind.SWING5_H, ts))
            if lo < fives[i - 1][2] and lo < fives[i + 1][2]:
                self.dynamic_levels.append(Level(lo, LevelKind.SWING5_L, ts))

    def _update_compression(self, recent: list[Minute]) -> None:
        if len(recent) < 6:
            self.compression = Compression()
            return
        last3 = recent[-3:]
        prior = recent[:-3][-10:]
        avg3 = sum(m.range_ for m in last3) / len(last3)
        avgp = sum(m.range_ for m in prior) / max(len(prior), 1)
        ratio = avg3 / avgp if avgp > 0 else 1.0
        tight = 0
        for i in range(len(recent) - 1, 0, -1):
            if recent[i].high <= recent[i - 1].high and recent[i].low >= recent[i - 1].low:
                tight += 1
            else:
                break
        self.compression = Compression(
            ratio=ratio, tightening=tight, active=ratio <= self.compression_max_ratio
        )

    # ------------------------------------------------------------------ query

    def levels_cached(self, vwap: float, ts: float) -> list[Level]:
        """Merged levels, rebuilt at most once per second (hot-path variant)."""
        sec = int(ts)
        if sec != self._cache_sec or (
            self._cache_vwap > 0.0
            and vwap > 0.0
            and abs(vwap - self._cache_vwap) / self._cache_vwap * 1e4 > 1.0
        ):
            self._cache = self.levels(vwap)
            self._cache_sec = sec
            self._cache_vwap = vwap
        return self._cache

    def levels(self, vwap: float = 0.0) -> list[Level]:
        """Merged level list, higher-priority kind wins within merge_bps."""
        all_levels = list(self.static_levels) + list(self.or_levels) + list(self.dynamic_levels)
        if vwap > 0.0:
            all_levels.append(Level(vwap, LevelKind.VWAP))
        all_levels.sort(key=lambda lv: (lv.price, _PRIORITY[lv.kind]))
        merged: list[Level] = []
        for lv in all_levels:
            if (
                merged
                and abs(lv.price - merged[-1].price) / merged[-1].price * 1e4 < self.merge_bps
            ):
                if _PRIORITY[lv.kind] < _PRIORITY[merged[-1].kind]:
                    merged[-1] = lv
                continue
            merged.append(lv)
        return merged

    @staticmethod
    def break_direction(level: Level, price: float) -> int:
        """+1 when price is above the level, -1 below."""
        return 1 if price > level.price else -1

    @staticmethod
    def is_upper(level: Level) -> bool:
        return level.kind in _UPPER_KINDS

    # ------------------------------------------------------- sweeps / reclaims

    def observe(self, price: float, ts: float, levels: list[Level] | None = None) -> None:
        """Track cross/reclaim state for every current level (called per print)."""
        if levels is None:
            levels = self.levels_cached(0.0, ts)
        for lv in levels:
            key = lv.key
            st = self._sweeps.get(key)
            beyond = price > lv.price
            if st is None or st.direction == 0:
                if st is None:
                    st = _SweepState()
                    self._sweeps[key] = st
                st.direction = 1 if beyond else -1
                st.beyond_since = ts
                st.extreme = price
                continue
            same_side = (st.direction == 1) == beyond
            if same_side:
                if st.direction == 1 and price > st.extreme:
                    st.extreme = price
                elif st.direction == -1 and price < st.extreme:
                    st.extreme = price
                continue
            # Crossed back through the level: was the excursion a sweep?
            held_for = ts - st.beyond_since
            excursion_bps = abs(st.extreme - lv.price) / lv.price * 1e4
            if held_for <= self.sweep_window_s and excursion_bps <= self.sweep_max_bps:
                self._reclaims.append(
                    ReclaimEvent(level=lv, direction=-st.direction, ts=ts, swept_extreme=st.extreme)
                )
            st.direction = 1 if beyond else -1
            st.beyond_since = ts
            st.extreme = price

    def active_reclaims(self, ts: float, max_age_s: float = 5.0) -> list[ReclaimEvent]:
        return [r for r in self._reclaims if ts - r.ts <= max_age_s]

    def consume_reclaim(self, event: ReclaimEvent) -> None:
        try:
            self._reclaims.remove(event)
        except ValueError:
            pass

    def _prune_sweep_states(self) -> None:
        live_keys = {lv.key for lv in self.levels()}
        for key in list(self._sweeps):
            if key not in live_keys:
                del self._sweeps[key]
