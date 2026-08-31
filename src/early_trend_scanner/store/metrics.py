"""In-memory performance tracking with bounded storage.

Tracks exactly what the spec asks for: precision, false-alert rate, median
lead time, share of the move remaining at alert time, confirmed/failed rate,
MFE/MAE — sliced by symbol, direction, time-of-day bucket and setup verb.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from ..engine.state import Signal
from ..ml.gate import tod_bucket
from ..ml.labeler import LabelResult


@dataclass
class _Slice:
    alerts: int = 0
    confirmed: int = 0
    failed: int = 0
    labeled_pos: int = 0
    labeled_neg: int = 0
    late_neg: int = 0
    lead_times: deque[float] = field(default_factory=lambda: deque(maxlen=256))
    remaining: deque[float] = field(default_factory=lambda: deque(maxlen=256))
    mfe: deque[float] = field(default_factory=lambda: deque(maxlen=256))
    mae: deque[float] = field(default_factory=lambda: deque(maxlen=256))

    def summary(self) -> dict[str, Any]:
        labeled = self.labeled_pos + self.labeled_neg
        followups = self.confirmed + self.failed
        return {
            "alerts": self.alerts,
            "confirmed": self.confirmed,
            "failed": self.failed,
            "confirm_rate": round(self.confirmed / followups, 3) if followups else None,
            "precision": round(self.labeled_pos / labeled, 3) if labeled else None,
            "false_alert_rate": round(self.labeled_neg / labeled, 3) if labeled else None,
            "late_alert_share": round(self.late_neg / labeled, 3) if labeled else None,
            "median_lead_s": round(median(self.lead_times), 1) if self.lead_times else None,
            "median_remaining": round(median(self.remaining), 3) if self.remaining else None,
            "median_mfe": round(median(self.mfe), 4) if self.mfe else None,
            "median_mae": round(median(self.mae), 4) if self.mae else None,
        }


class MetricsTracker:
    def __init__(self, session_open_ts: float = 0.0, session_close_ts: float = 0.0) -> None:
        self.session_open_ts = session_open_ts
        self.session_close_ts = session_close_ts
        self.total = _Slice()
        self.by_symbol: dict[str, _Slice] = defaultdict(_Slice)
        self.by_direction: dict[str, _Slice] = defaultdict(_Slice)
        self.by_bucket: dict[int, _Slice] = defaultdict(_Slice)
        self.by_verb: dict[str, _Slice] = defaultdict(_Slice)
        self.suppressed = 0

    def _slices(self, sig: Signal) -> list[_Slice]:
        return [
            self.total,
            self.by_symbol[sig.symbol],
            self.by_direction[sig.dir_str],
            self.by_bucket[tod_bucket(sig.alert_ts, self.session_open_ts, self.session_close_ts)],
            self.by_verb[sig.trigger_verb],
        ]

    def on_alert(self, sig: Signal) -> None:
        if sig.suppressed:
            self.suppressed += 1
            return
        for s in self._slices(sig):
            s.alerts += 1

    def on_resolution(self, sig: Signal) -> None:
        if sig.suppressed:
            return
        for s in self._slices(sig):
            if sig.resolution == "CONFIRMED":
                s.confirmed += 1
            elif sig.resolution == "FAILED":
                s.failed += 1

    def on_label(self, r: LabelResult) -> None:
        if r.signal.suppressed:
            return
        for s in self._slices(r.signal):
            if r.label:
                s.labeled_pos += 1
            else:
                s.labeled_neg += 1
                if r.was_late:
                    s.late_neg += 1
            if r.lead_time_s is not None:
                s.lead_times.append(r.lead_time_s)
            s.remaining.append(r.remaining_frac)
            s.mfe.append(r.mfe)
            s.mae.append(r.mae)

    def summary(self) -> dict[str, Any]:
        return {
            "total": self.total.summary(),
            "suppressed_by_model": self.suppressed,
            "by_symbol": {k: v.summary() for k, v in sorted(self.by_symbol.items())},
            "by_direction": {k: v.summary() for k, v in self.by_direction.items()},
            "by_time_bucket": {
                {0: "first_hour", 1: "midday", 2: "last_hour"}.get(k, str(k)): v.summary()
                for k, v in sorted(self.by_bucket.items())
            },
            "by_setup": {k: v.summary() for k, v in sorted(self.by_verb.items())},
        }
