"""Bounded adaptive thresholds + revert-on-deterioration guardrails.

The learning layer may only nudge rule thresholds through multipliers clamped
to [bound_low, bound_high], per symbol and time-of-day bucket. A frozen
rule-only baseline precision (measured while the gate was inactive) is the
comparison anchor: if rolling precision materially deteriorates, all
adaptation reverts and ML influence is disabled until re-earned.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field

from ..config import MlCfg

log = logging.getLogger(__name__)

_BUCKETS = 3  # 0: first hour, 1: midday, 2: last hour


def tod_bucket(ts: float, session_open_ts: float, session_close_ts: float) -> int:
    if session_open_ts <= 0:
        return 1
    if ts < session_open_ts + 3600:
        return 0
    if ts > session_close_ts - 3600:
        return 2
    return 1


@dataclass
class LabelOutcome:
    signal_id: str
    symbol: str
    direction: int
    label: bool
    was_late: bool  # negative specifically because of excessive displacement
    ts: float


@dataclass
class AdaptiveGate:
    cfg: MlCfg
    session_open_ts: float = 0.0
    session_close_ts: float = 0.0

    vol_mult: dict[str, float] = field(default_factory=dict)  # key: f"{sym}:{bucket}"
    score_mult: dict[str, float] = field(default_factory=dict)
    n_labels: int = 0
    baseline_precision: float = -1.0  # frozen precision of the rule-only phase
    _baseline_hits: int = 0
    _baseline_total: int = 0
    reverted: bool = False
    relearn_needed: int = 0
    version: int = 1
    recent: deque[bool] = field(default_factory=lambda: deque(maxlen=64))

    def __post_init__(self) -> None:
        self.recent = deque(self.recent, maxlen=max(8, self.cfg.revert_window))

    # ------------------------------------------------------------------ state

    @property
    def active(self) -> bool:
        """True when ML may influence live thresholds / gate by probability."""
        if self.reverted:
            return False
        return self.n_labels >= self.cfg.min_labels

    def set_session(self, open_ts: float, close_ts: float) -> None:
        self.session_open_ts = open_ts
        self.session_close_ts = close_ts

    def _key(self, symbol: str, ts: float) -> str:
        return f"{symbol}:{tod_bucket(ts, self.session_open_ts, self.session_close_ts)}"

    def multipliers(self, symbol: str, ts: float) -> tuple[float, float]:
        """(vol_accel multiplier, score-threshold multiplier), both bounded."""
        if not self.active:
            return 1.0, 1.0
        key = self._key(symbol, ts)
        return (
            self._clamp(self.vol_mult.get(key, 1.0)),
            self._clamp(self.score_mult.get(key, 1.0)),
        )

    def _clamp(self, v: float) -> float:
        return max(self.cfg.bound_low, min(self.cfg.bound_high, v))

    # --------------------------------------------------------------- learning

    def on_label(self, outcome: LabelOutcome) -> None:
        self.n_labels += 1
        self.recent.append(outcome.label)

        # The first min_labels outcomes were produced while the gate was
        # inactive (rule-only): they define the frozen comparison baseline.
        if self.baseline_precision < 0 and self._baseline_total < self.cfg.min_labels:
            self._baseline_total += 1
            if outcome.label:
                self._baseline_hits += 1
            if self._baseline_total >= self.cfg.min_labels:
                self.baseline_precision = self._baseline_hits / self._baseline_total
                log.info("rule-only baseline precision frozen at %.2f", self.baseline_precision)

        key = self._key(outcome.symbol, outcome.ts)
        step = self.cfg.adapt_step
        if outcome.label:
            self.vol_mult[key] = self._clamp(self.vol_mult.get(key, 1.0) * (1.0 - step))
            self.score_mult[key] = self._clamp(self.score_mult.get(key, 1.0) * (1.0 - step))
        else:
            factor = 1.0 + step * (2.0 if outcome.was_late else 1.0)
            self.vol_mult[key] = self._clamp(self.vol_mult.get(key, 1.0) * factor)
            self.score_mult[key] = self._clamp(self.score_mult.get(key, 1.0) * factor)

        if self.reverted:
            self.relearn_needed -= 1
            if self.relearn_needed <= 0:
                self.reverted = False
                log.info("adaptive gate re-enabled after relearning period")
        self._check_revert()

    def _check_revert(self) -> None:
        if not self.active or self.baseline_precision < 0:
            return
        if len(self.recent) < max(8, self.cfg.revert_window // 2):
            return
        precision = sum(self.recent) / len(self.recent)
        if precision < self.baseline_precision - self.cfg.revert_precision_drop:
            log.warning(
                "REVERT: rolling precision %.2f fell below baseline %.2f - %.2f; "
                "resetting adaptive thresholds and disabling ML influence",
                precision,
                self.baseline_precision,
                self.cfg.revert_precision_drop,
            )
            self.vol_mult.clear()
            self.score_mult.clear()
            self.reverted = True
            self.relearn_needed = max(self.cfg.min_labels // 2, 10)
            self.version += 1

    # ------------------------------------------------------------ persistence

    def to_json(self) -> str:
        return json.dumps(
            {
                "vol_mult": self.vol_mult,
                "score_mult": self.score_mult,
                "n_labels": self.n_labels,
                "baseline_precision": self.baseline_precision,
                "baseline_hits": self._baseline_hits,
                "baseline_total": self._baseline_total,
                "reverted": self.reverted,
                "relearn_needed": self.relearn_needed,
                "version": self.version,
                "recent": list(self.recent),
            }
        )

    @classmethod
    def from_json(cls, cfg: MlCfg, raw: str) -> AdaptiveGate:
        d = json.loads(raw)
        gate = cls(cfg=cfg)
        gate.vol_mult = {k: float(v) for k, v in d.get("vol_mult", {}).items()}
        gate.score_mult = {k: float(v) for k, v in d.get("score_mult", {}).items()}
        gate.n_labels = int(d.get("n_labels", 0))
        gate.baseline_precision = float(d.get("baseline_precision", -1.0))
        gate._baseline_hits = int(d.get("baseline_hits", 0))
        gate._baseline_total = int(d.get("baseline_total", 0))
        gate.reverted = bool(d.get("reverted", False))
        gate.relearn_needed = int(d.get("relearn_needed", 0))
        gate.version = int(d.get("version", 1))
        gate.recent = deque(
            (bool(x) for x in d.get("recent", [])), maxlen=max(8, cfg.revert_window)
        )
        return gate
