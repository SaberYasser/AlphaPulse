"""Outcome labeling: resolve each signal after the configured outcome window.

Only the compact feature vector captured at alert time is kept; the label is
computed from the 1-second ring (bounded) once the window closes. Definitions:

  invalidation distance  d  = |trigger - invalidation|
  MFE(pre-invalidation)     = best favorable excursion from the ALERT price
                              before price ever crosses the invalidation
  positive label            = MFE_pre >= pos_multiple * d
                              AND favorable excursion reached d within the window
                              AND remaining_frac >= min_remaining_frac
  remaining_frac            = (peak - alert_price) / (peak - trigger_price),
                              the share of the trigger->peak move still ahead
                              when the alert went out (penalizes late alerts)
  lead_time_s               = seconds from alert until favorable excursion
                              first reached d (the move became evident)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from ..config import MlCfg
from ..engine.rolling import SecRing
from ..engine.state import Signal

log = logging.getLogger(__name__)


@dataclass(slots=True)
class LabelResult:
    signal: Signal
    label: bool
    mfe: float  # favorable excursion from alert price (abs $)
    mae: float  # adverse excursion from alert price (abs $)
    lead_time_s: float | None
    remaining_frac: float
    was_late: bool  # failed only the remaining_frac requirement
    reason: str


@dataclass(slots=True)
class _Pending:
    signal: Signal
    due_ts: float


class Labeler:
    def __init__(self, cfg: MlCfg, on_result: Callable[[LabelResult], None]) -> None:
        self.cfg = cfg
        self.on_result = on_result
        self._pending: list[_Pending] = []

    def track(self, sig: Signal) -> None:
        self._pending.append(_Pending(sig, sig.alert_ts + self.cfg.outcome_window_s))

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def on_tick(self, now_ts: float, rings: dict[str, SecRing]) -> None:
        if not self._pending:
            return
        still: list[_Pending] = []
        for p in self._pending:
            if now_ts < p.due_ts:
                still.append(p)
                continue
            ring = rings.get(p.signal.symbol)
            if ring is None:
                continue
            result = self.resolve(p.signal, ring)
            if result is not None:
                self.on_result(result)
        self._pending = still

    def flush(self, rings: dict[str, SecRing]) -> int:
        """Session close: resolve whatever already has a full window; drop the rest."""
        resolved = 0
        for p in self._pending:
            ring = rings.get(p.signal.symbol)
            newest = ring.newest if ring is not None else None
            if ring is None or newest is None or newest.ts < p.due_ts - 1:
                log.info("label window truncated by close: %s", p.signal.signal_id)
                continue
            result = self.resolve(p.signal, ring)
            if result is not None:
                self.on_result(result)
                resolved += 1
        self._pending = []
        return resolved

    # ------------------------------------------------------------- resolution

    def resolve(self, sig: Signal, ring: SecRing) -> LabelResult | None:
        d = sig.direction
        inv_dist = abs(sig.trigger_price - sig.invalidation)
        if inv_dist <= 0:
            return None
        t0 = int(sig.alert_ts)
        t1 = int(sig.alert_ts + self.cfg.outcome_window_s) + 1

        fav_extreme = 0.0  # signed favorable excursion from alert price
        adv_extreme = 0.0
        fav_pre_inval = 0.0
        peak = sig.alert_price  # favorable extreme in absolute price
        invalidated = False
        lead_time: float | None = None
        seen = 0

        for sec in ring.iter_between(t0, t1):
            seen += 1
            fav_px = sec.high if d > 0 else sec.low
            adv_px = sec.low if d > 0 else sec.high
            fav = (fav_px - sig.alert_price) * d
            adv = (sig.alert_price - adv_px) * d
            if fav > fav_extreme:
                fav_extreme = fav
                if (fav_px - peak) * d > 0:
                    peak = fav_px
            if adv > adv_extreme:
                adv_extreme = adv
            if not invalidated:
                if fav > fav_pre_inval:
                    fav_pre_inval = fav
                if (adv_px - sig.invalidation) * d < 0:
                    invalidated = True
            if lead_time is None and fav >= inv_dist:
                lead_time = sec.ts + 1 - sig.alert_ts

        if seen == 0:
            log.warning("no ring data to label %s", sig.signal_id)
            return None

        move_total = (peak - sig.trigger_price) * d
        move_after_alert = (peak - sig.alert_price) * d
        remaining = move_after_alert / move_total if move_total > 1e-9 else 0.0

        expanded = fav_pre_inval >= self.cfg.pos_multiple * inv_dist
        started = lead_time is not None
        enough_left = remaining >= self.cfg.min_remaining_frac
        label = expanded and started and enough_left
        was_late = expanded and started and not enough_left

        if label:
            reason = "expanded"
        elif not started:
            reason = "no_expansion"
        elif not expanded:
            reason = "insufficient_expansion"
        else:
            reason = "alert_too_late"

        return LabelResult(
            signal=sig,
            label=label,
            mfe=fav_extreme,
            mae=adv_extreme,
            lead_time_s=lead_time,
            remaining_frac=max(0.0, min(remaining, 1.0)),
            was_late=was_late,
            reason=reason,
        )
