"""Local aggregation: trades -> contiguous 1-second bars -> 1-minute bars.

Also maintains session VWAP, session high/low and the opening range. Raw ticks
are never retained: every print folds into the current second accumulator and
is discarded.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from ..data.models import Minute, Sec, Trade
from .rolling import SecRing

# Guard against pathological clock jumps: never fill more than this many empty
# seconds in one step (ring is reset instead).
_MAX_GAP_FILL = 900


class _Acc:
    """Mutable accumulator for the second currently being built."""

    __slots__ = ("buy_vol", "close", "dollar", "high", "low", "n", "open", "sell_vol", "ts", "vol")

    def __init__(self, ts: int, ref_price: float) -> None:
        self.ts = ts
        self.open = ref_price
        self.high = ref_price
        self.low = ref_price
        self.close = ref_price
        self.vol = 0
        self.buy_vol = 0
        self.sell_vol = 0
        self.n = 0
        self.dollar = 0.0

    def to_sec(self) -> Sec:
        return Sec(
            ts=self.ts,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            vol=self.vol,
            buy_vol=self.buy_vol,
            sell_vol=self.sell_vol,
            n=self.n,
            dollar=self.dollar,
        )


class SymbolAggregator:
    def __init__(
        self,
        symbol: str,
        ring_seconds: int = 600,
        minutes_kept: int = 120,
        on_minute: Callable[[Minute], None] | None = None,
        on_second: Callable[[Sec], None] | None = None,
    ) -> None:
        self.symbol = symbol
        self.ring = SecRing(ring_seconds)
        self.minutes: deque[Minute] = deque(maxlen=minutes_kept)
        self.on_minute = on_minute
        self.on_second = on_second

        self._acc: _Acc | None = None
        self._min_acc: Minute | None = None
        self.last_price: float = 0.0  # last price-forming print
        self.last_flow_price: float = 0.0  # last flow-eligible print (tick rule)

        # Minute-trend tracker: EMA20 of completed 1-min closes, premarket-
        # seeded by warmup. hist keeps the last 4 EMA values so the verdict can
        # read the slope over a 3-bar span in O(1).
        self._ema20: float | None = None
        self._ema20_hist: deque[float] = deque(maxlen=4)

        # Session context (set by warmup/engine)
        self.session_open_ts: float = 0.0
        self.opening_range_end_ts: float = 0.0
        self.vwap_pv: float = 0.0
        self.vwap_v: float = 0.0
        self.session_high: float = 0.0
        self.session_low: float = 0.0
        self.or_high: float = 0.0
        self.or_low: float = 0.0

    # ----------------------------------------------------------------- session

    def set_session(self, open_ts: float, opening_range_min: int) -> None:
        self.session_open_ts = open_ts
        self.opening_range_end_ts = open_ts + opening_range_min * 60

    def seed_vwap(self, pv: float, v: float) -> None:
        self.vwap_pv = pv
        self.vwap_v = v

    def seed_preopen(self, price: float, open_ts: float, seconds: int = 60) -> None:
        """Warmup: flat zero-volume seconds at the premarket close price.

        Without this the ring is empty at the opening bell, velocity and
        acceleration are incomputable for the first ~15 seconds, and a
        gap-and-go mover is already past the extension cap by the time they
        exist. Prices are real (last premarket print); volumes are zero so no
        volume window is distorted.
        """
        if price <= 0.0 or self.ring.newest is not None or self._acc is not None:
            return
        start = int(open_ts) - seconds
        for s in range(start, int(open_ts)):
            self.ring.append(
                Sec(
                    ts=s,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    vol=0,
                    buy_vol=0,
                    sell_vol=0,
                    n=0,
                    dollar=0.0,
                )
            )
        self.last_price = price
        self.last_flow_price = price

    @property
    def vwap(self) -> float:
        return self.vwap_pv / self.vwap_v if self.vwap_v > 0 else 0.0

    # ------------------------------------------------------------------ events

    def on_trade(self, t: Trade) -> None:
        bucket = int(t.ts)
        # Never reopen a finalized second (the wall-clock ticker may run ahead
        # of a slightly late feed event): fold such prints into the next bucket.
        newest = self.ring.newest
        if newest is not None and bucket <= newest.ts:
            bucket = newest.ts + 1
        self.roll_to(bucket)
        if self._acc is None:
            ref = t.price if t.updates_last else (self.last_price or t.price)
            self._acc = _Acc(bucket, ref)

        acc = self._acc
        if t.updates_volume:
            acc.vol += t.size
        if t.updates_last:
            if t.price > acc.high:
                acc.high = t.price
            if t.price < acc.low:
                acc.low = t.price
            acc.close = t.price
            self.last_price = t.price
        if t.flow_eligible:
            acc.n += 1
            acc.dollar += t.price * t.size
            if t.side > 0:
                acc.buy_vol += t.size
            elif t.side < 0:
                acc.sell_vol += t.size
            self.last_flow_price = t.price

        in_session = t.ts >= self.session_open_ts > 0
        if in_session and t.flow_eligible:
            self.vwap_pv += t.price * t.size
            self.vwap_v += t.size
        if in_session and t.updates_last:
            if self.session_high == 0.0 or t.price > self.session_high:
                self.session_high = t.price
            if self.session_low == 0.0 or t.price < self.session_low:
                self.session_low = t.price
            if t.ts < self.opening_range_end_ts:
                if self.or_high == 0.0 or t.price > self.or_high:
                    self.or_high = t.price
                if self.or_low == 0.0 or t.price < self.or_low:
                    self.or_low = t.price

    def roll_to(self, bucket: int) -> None:
        """Finalize every second before `bucket`, filling quiet gaps."""
        if self._acc is None:
            return
        if bucket <= self._acc.ts:
            return
        gap = bucket - self._acc.ts
        if gap > _MAX_GAP_FILL:
            # A gap this large means a stopped clock or a huge outage: keep the
            # last price but do not synthesize 15+ minutes of empty history.
            last_close = self._acc.close
            self._finalize_current()
            self._acc = _Acc(bucket - 1, last_close)
            self._finalize_current()
            return
        while self._acc is not None and self._acc.ts < bucket:
            last_close = self._acc.close
            next_ts = self._acc.ts + 1
            self._finalize_current()
            if next_ts < bucket:
                self._acc = _Acc(next_ts, last_close)
        # current bucket accumulator is created lazily on the next trade

    def _finalize_current(self) -> None:
        assert self._acc is not None
        sec = self._acc.to_sec()
        self._acc = None
        self.ring.append(sec)
        if self.on_second is not None:
            self.on_second(sec)
        self._fold_minute(sec)

    def _fold_minute(self, sec: Sec) -> None:
        m_ts = sec.ts - (sec.ts % 60)
        m = self._min_acc
        if m is not None and m.ts != m_ts:
            self.minutes.append(m)
            self._ema_update(m.close)
            if self.on_minute is not None:
                self.on_minute(m)
            m = None
        if m is None:
            m = Minute(ts=m_ts, open=sec.open, high=sec.high, low=sec.low, close=sec.close)
            self._min_acc = m
        else:
            if sec.high > m.high:
                m.high = sec.high
            if sec.low < m.low:
                m.low = sec.low
            m.close = sec.close
        m.vol += sec.vol
        m.buy_vol += sec.buy_vol
        m.sell_vol += sec.sell_vol
        m.n += sec.n
        m.dollar += sec.dollar

    # ----------------------------------------------------------------- queries

    def partial(self) -> _Acc | None:
        return self._acc

    def window(self, w: int) -> tuple[float, float, float, float, float]:
        """(vol, buy, sell, n, dollar) over the last w seconds including the partial one."""
        vol, buy, sell, n, dollar = self.ring.sums(w - 1)
        acc = self._acc
        if acc is not None:
            vol += acc.vol
            buy += acc.buy_vol
            sell += acc.sell_vol
            n += acc.n
            dollar += acc.dollar
        return (vol, buy, sell, n, dollar)

    def price_ago(self, seconds: int) -> float | None:
        return self.ring.close_ago(seconds)

    def range_high_low(self, seconds: int) -> tuple[float, float] | None:
        hl = self.ring.high_low(seconds)
        acc = self._acc
        if acc is None or acc.n == 0:
            return hl
        if hl is None:
            return (acc.high, acc.low)
        return (max(hl[0], acc.high), min(hl[1], acc.low))

    def seed_minutes(self, minutes: list[Minute]) -> None:
        """Warmup: pre-fill completed minutes (levels are rebuilt by the caller)."""
        for m in minutes:
            self.minutes.append(m)
            self._ema_update(m.close)
        if minutes:
            self.last_price = minutes[-1].close
            self.last_flow_price = minutes[-1].close

    # ------------------------------------------------------------ minute trend

    _EMA_ALPHA = 2.0 / 21.0  # EMA20

    def _ema_update(self, close: float) -> None:
        if close <= 0.0:
            return
        prev = self._ema20
        a = self._EMA_ALPHA
        self._ema20 = close if prev is None else a * close + (1 - a) * prev
        self._ema20_hist.append(self._ema20)

    def seed_ema(self, closes: list[float]) -> None:
        """Warmup: fold premarket minute closes into the trend EMA so the
        slope is defined from the first regular-hours verdicts."""
        for c in closes:
            self._ema_update(c)

    def ema_slope_bps(self) -> float | None:
        """EMA20 slope over the last 3 completed minutes, in bps of price.

        None until enough history exists — callers must treat that as neutral
        (missing data never fails a signal)."""
        if len(self._ema20_hist) < 4 or self.last_price <= 0.0:
            return None
        return (self._ema20_hist[-1] - self._ema20_hist[0]) / self.last_price * 1e4
