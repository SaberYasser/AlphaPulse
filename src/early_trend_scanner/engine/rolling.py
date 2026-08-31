"""Fixed-size ring of one-second aggregates with O(1) rolling-window sums.

Seconds are contiguous (the aggregator fills gaps with empty seconds), so
"k seconds ago" is simply the k-th element from the end. Memory is bounded by
`size` regardless of session length or trade rate.
"""

from __future__ import annotations

from collections.abc import Iterator

from ..data.models import Sec

# Windows are maintained both at w and w-1 so callers can combine (w-1) finalized
# seconds with the current partial second and still cover exactly w seconds.
DEFAULT_WINDOWS = (4, 5, 14, 15, 29, 30, 59, 60)


class SecRing:
    __slots__ = ("_buf", "_count", "_sums", "_write", "size", "windows")

    def __init__(self, size: int, windows: tuple[int, ...] = DEFAULT_WINDOWS) -> None:
        if size < max(windows) + 1:
            raise ValueError("ring size must exceed the largest window")
        self.size = size
        self.windows = windows
        self._buf: list[Sec | None] = [None] * size
        self._write = 0
        self._count = 0
        # per window: [vol, buy, sell, n, dollar]
        self._sums: dict[int, list[float]] = {w: [0.0] * 5 for w in windows}

    def __len__(self) -> int:
        return self._count

    @property
    def newest(self) -> Sec | None:
        return self.from_end(0)

    def from_end(self, k: int) -> Sec | None:
        """k=0 is the newest finalized second."""
        if k < 0 or k >= self._count:
            return None
        return self._buf[(self._write - 1 - k) % self.size]

    def append(self, sec: Sec) -> None:
        for w in self.windows:
            s = self._sums[w]
            s[0] += sec.vol
            s[1] += sec.buy_vol
            s[2] += sec.sell_vol
            s[3] += sec.n
            s[4] += sec.dollar
            if self._count >= w:
                old = self.from_end(w - 1)
                if old is not None:
                    s[0] -= old.vol
                    s[1] -= old.buy_vol
                    s[2] -= old.sell_vol
                    s[3] -= old.n
                    s[4] -= old.dollar
        self._buf[self._write] = sec
        self._write = (self._write + 1) % self.size
        if self._count < self.size:
            self._count += 1

    def sums(self, window: int) -> tuple[float, float, float, float, float]:
        """(vol, buy_vol, sell_vol, n, dollar) over the last `window` finalized seconds."""
        s = self._sums[window]
        return (s[0], s[1], s[2], s[3], s[4])

    def close_ago(self, seconds: int) -> float | None:
        """Close of the finalized second `seconds` before the current partial one."""
        sec = self.from_end(seconds - 1)
        return sec.close if sec is not None else None

    def high_low(self, window: int) -> tuple[float, float] | None:
        hi = -1.0
        lo = float("inf")
        n = min(window, self._count)
        if n == 0:
            return None
        for k in range(n):
            sec = self.from_end(k)
            assert sec is not None
            if sec.high > hi:
                hi = sec.high
            if sec.low < lo:
                lo = sec.low
        return (hi, lo)

    def iter_between(self, ts0: int, ts1: int) -> Iterator[Sec]:
        """Finalized seconds with ts0 <= ts < ts1, oldest first (label resolution)."""
        newest = self.newest
        if newest is None:
            return
        start_k = min(newest.ts - ts0, self._count - 1)
        for k in range(int(start_k), -1, -1):
            sec = self.from_end(k)
            if sec is None:
                continue
            if ts0 <= sec.ts < ts1:
                yield sec
