"""Symbol x minute-of-day activity baselines from completed historical sessions.

Median 1-minute volume and trade count per session minute, built once at warmup
from REST bars. Bounded: symbols x 390 floats.
"""

from __future__ import annotations

from statistics import median

from ..data.models import Bar, SessionInfo


class MinuteBaseline:
    def __init__(self) -> None:
        self._vol: dict[str, list[float]] = {}
        self._n: dict[str, list[float]] = {}
        self._fallback_vol: dict[str, float] = {}
        self._fallback_n: dict[str, float] = {}
        self.sessions_used = 0

    def build(
        self,
        bars_by_symbol: dict[str, list[Bar]],
        sessions: list[SessionInfo],
        minutes_per_session: int = 390,
    ) -> None:
        self.sessions_used = len(sessions)
        for symbol, bars in bars_by_symbol.items():
            per_minute: list[list[float]] = [[] for _ in range(minutes_per_session)]
            per_minute_n: list[list[float]] = [[] for _ in range(minutes_per_session)]
            for bar in bars:
                for s in sessions:
                    if s.open_ts <= bar.ts < s.close_ts:
                        idx = int((bar.ts - s.open_ts) // 60)
                        if 0 <= idx < minutes_per_session:
                            per_minute[idx].append(float(bar.volume))
                            per_minute_n[idx].append(float(bar.trade_count))
                        break
            vol_med = [median(v) if v else 0.0 for v in per_minute]
            n_med = [median(v) if v else 0.0 for v in per_minute_n]
            nonzero = [v for v in vol_med if v > 0]
            nonzero_n = [v for v in n_med if v > 0]
            self._vol[symbol] = vol_med
            self._n[symbol] = n_med
            self._fallback_vol[symbol] = median(nonzero) if nonzero else 0.0
            self._fallback_n[symbol] = median(nonzero_n) if nonzero_n else 0.0

    def minute_volume(self, symbol: str, minute_idx: int) -> float:
        arr = self._vol.get(symbol)
        if arr and 0 <= minute_idx < len(arr) and arr[minute_idx] > 0:
            return arr[minute_idx]
        return self._fallback_vol.get(symbol, 0.0)

    def minute_trades(self, symbol: str, minute_idx: int) -> float:
        arr = self._n.get(symbol)
        if arr and 0 <= minute_idx < len(arr) and arr[minute_idx] > 0:
            return arr[minute_idx]
        return self._fallback_n.get(symbol, 0.0)

    def vol_per_5s(self, symbol: str, minute_idx: int) -> float:
        return self.minute_volume(symbol, minute_idx) / 12.0

    def midday_vol_per_5s(self, symbol: str) -> float:
        """Typical mid-session 5s volume (median of minutes 30..360).

        The stable ruler for judging opening-minute participation: relative-
        to-prior-window comparisons are self-defeating while the opening flood
        IS the prior window.
        """
        arr = self._vol.get(symbol)
        if arr:
            mid = [v for v in arr[30:360] if v > 0]
            if mid:
                return median(mid) / 12.0
        return self._fallback_vol.get(symbol, 0.0) / 12.0

    def ready(self, symbol: str) -> bool:
        return self._fallback_vol.get(symbol, 0.0) > 0.0
