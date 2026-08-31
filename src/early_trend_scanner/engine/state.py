"""Per-symbol signal state machine.

    SCANNING -> READY -> EARLY_SIGNAL(FIRED) -> CONFIRMED | FAILED -> COOLDOWN

The EARLY alert is emitted the instant minimum price+volume conditions hold —
never waiting for a completed candle, retest or higher-timeframe agreement.
Follow-through is analyzed only after the alert is already on its way.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..config import EngineCfg
from .features import Snapshot, fake_start_penalty

log = logging.getLogger(__name__)


class Phase(Enum):
    SCANNING = "SCANNING"
    READY = "READY"
    FIRED = "EARLY_SIGNAL"
    COOLDOWN = "COOLDOWN"


@dataclass
class Signal:
    signal_id: str
    symbol: str
    direction: int  # +1 up, -1 down
    alert_ts: float
    alert_price: float
    trigger_price: float
    level_kind: int
    trigger_verb: str
    invalidation: float
    vol_ratio: float
    features: dict[str, float]
    prob: float | None = None
    model_version: str = ""
    score: float = 0.0
    vol5: float = 0.0  # absolute 5-second volume (shares) at trigger
    suppressed: bool = False  # ML-gated: tracked/learned but not alerted
    # observation state
    micro_extreme: float = 0.0
    followups_sent: int = 0
    resolution: str = ""  # CONFIRMED | FAILED
    resolution_ts: float = 0.0
    resolution_reason: str = ""

    @property
    def dir_str(self) -> str:
        return "UP" if self.direction > 0 else "DOWN"


@dataclass
class Rejection:
    """Why a candidate did not fire (diagnostics/metrics only)."""

    reason: str
    snapshot: Snapshot | None = None


class GlobalLimiter:
    """Shared sanity caps across all symbols.

    Two windows: the hourly budget, and a short burst window so a market-wide
    event (the whole universe re-pricing at the bell) cannot fire the entire
    hourly budget into the user's phone in a few seconds.
    """

    def __init__(
        self, max_per_hour: int, max_per_burst: int = 3, burst_window_s: float = 60.0
    ) -> None:
        self.max_per_hour = max_per_hour
        self.max_per_burst = max_per_burst
        self.burst_window_s = burst_window_s
        self._alerts: deque[float] = deque(maxlen=max(max_per_hour * 2, 16))

    def allow(self, ts: float) -> bool:
        while self._alerts and ts - self._alerts[0] > 3600.0:
            self._alerts.popleft()
        if len(self._alerts) >= self.max_per_hour:
            return False
        in_burst = sum(1 for t in self._alerts if ts - t <= self.burst_window_s)
        return in_burst < self.max_per_burst

    def note(self, ts: float) -> None:
        self._alerts.append(ts)


@dataclass
class MachineHooks:
    """Callbacks wired by the symbol engine / app."""

    emit: Callable[[Signal, str, dict[str, Any]], None]
    ml_predict: Callable[[dict[str, float], int], tuple[float | None, str]]
    gate_multipliers: Callable[[str, float], tuple[float, float]]  # (vol_mult, score_mult)
    ml_gate_active: Callable[[], bool]
    prob_gate_min: float
    prob_bypass_score: float = 1.01  # score at/above this is never model-suppressed
    on_signal_fired: Callable[[Signal], None] = field(default=lambda s: None)
    on_signal_final: Callable[[Signal], None] = field(default=lambda s: None)


class StateMachine:
    def __init__(
        self,
        symbol: str,
        cfg: EngineCfg,
        hooks: MachineHooks,
        limiter: GlobalLimiter,
    ) -> None:
        self.symbol = symbol
        self.cfg = cfg
        self.hooks = hooks
        self.limiter = limiter

        self.phase = Phase.SCANNING
        self.signal: Signal | None = None
        self.cooldown_until = 0.0
        self.last_trigger_price = 0.0
        self.last_direction = 0
        self.last_alert_ts = 0.0
        self.alerts_today = 0
        self.suppressed_today = 0
        self.trend_alerts_today = 0
        self.rejections: dict[str, int] = {}

    # ------------------------------------------------------------- evaluation

    def consider(self, snap: Snapshot, threshold_score: float) -> Signal | Rejection | None:
        """Evaluate one candidate snapshot; fire or explain why not.

        `threshold_score` already includes gate multipliers and compression relax.
        Returns the fired Signal, a Rejection, or None when simply not triggered.
        """
        cfg = self.cfg
        ts = snap.ts

        if self.phase == Phase.FIRED:
            return None
        if self.phase == Phase.COOLDOWN:
            if ts < self.cooldown_until:
                return None
            self.phase = Phase.SCANNING
        if (
            self.last_trigger_price > 0.0
            and snap.direction == self.last_direction
            and abs(snap.level_price - self.last_trigger_price) / self.last_trigger_price * 1e4
            < cfg.re_arm_bps
        ):
            return None  # same structure as the previous setup — needs a new level

        # --- price trigger gates -----------------------------------------
        if snap.break_bps < cfg.break_min_bps:
            self.phase = Phase.READY
            return None
        # Opening minutes get a wider (still bounded) extension cap: gap-and-go
        # movers travel further past their level per second than midday tape,
        # and treating that speed as "too late" forfeits the day's best entries.
        in_open_phase = snap.minute_frac * 390.0 < cfg.open_phase_min
        break_cap = cfg.break_max_open_bps if in_open_phase else cfg.break_max_bps
        if snap.break_bps > break_cap:
            return self._reject("extended", snap)
        dir_vel = snap.vel5_bps_s * snap.direction
        if dir_vel < cfg.vel_min_bps_s:
            return self._reject("velocity", snap)
        if not snap.accelerating:
            return self._reject("not_accelerating", snap)

        # --- volume trigger gates ----------------------------------------
        if snap.vol_ratio_prev < cfg.vol_accel_min * self._vol_mult(ts):
            return self._reject("volume_accel", snap)
        if snap.vol_ratio_base > 0.0 and snap.vol_ratio_base < cfg.vol_base_min:
            return self._reject("volume_baseline", snap)
        if snap.imb5 * snap.direction < cfg.imb_min:
            return self._reject("imbalance", snap)

        # --- persistence: more than an isolated print ---------------------
        if snap.n2s < cfg.min_trades_2s:
            return self._reject("single_print", snap)
        if snap.dominant_frac > cfg.single_print_max_frac:
            return self._reject("single_print", snap)
        if snap.persist_trades < cfg.persist_min_trades:
            return self._reject("persistence_count", snap)
        if snap.persist_span_s < cfg.persist_min_span_s:
            return self._reject("persistence_span", snap)

        # --- quality: score minus fake-start penalty ----------------------
        # The chase-evidence penalty (see features.fake_start_penalty) weighs
        # against the trigger score rather than hard-gating, so overwhelming
        # trigger evidence can still fire. When the penalty alone is what
        # sinks a candidate below threshold, attribute it as "fake_start".
        penalty = 0.0
        if cfg.fresh_break_veto:
            penalty = fake_start_penalty(
                snap,
                comp_veto_ratio=cfg.comp_veto_ratio,
                imb15_blowoff_max=cfg.imb15_blowoff_max,
                fresh_break_vol_max=cfg.fresh_break_vol_max,
                weight=cfg.fake_start_weight,
            )
        if snap.score - penalty < threshold_score:
            reason = "fake_start" if snap.score >= threshold_score else "score"
            return self._reject(reason, snap)

        # --- rate caps ----------------------------------------------------
        if self.alerts_today >= cfg.max_alerts_symbol_day:
            return self._reject("symbol_day_cap", snap)
        if ts - self.last_alert_ts < cfg.min_gap_same_symbol_s:
            return None
        if not self.limiter.allow(ts):
            return self._reject("global_hour_cap", snap)

        return self._fire(snap)

    def _vol_mult(self, ts: float) -> float:
        vol_mult, _ = self.hooks.gate_multipliers(self.symbol, ts)
        return vol_mult

    def _reject(self, reason: str, snap: Snapshot) -> Rejection:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1
        return Rejection(reason, snap)

    # ------------------------------------------------- sustained-pressure path

    def consider_trend(self, snap: Snapshot) -> Signal | Rejection | None:
        """Fire a sustained-pressure (trend-onset) signal through shared caps.

        A separate entry that bypasses the micro-burst gates (the evaluator in
        the symbol engine already established the escalator conditions) but
        shares every discipline mechanism: phase/cooldown, structural re-arm,
        per-symbol and global rate caps, and the follow-up/labeling pipeline.
        Rule-pure while `trend_model_gated` is false: the model records its
        probability and learns the class, but cannot suppress it yet.
        """
        cfg = self.cfg
        ts = snap.ts
        if self.phase == Phase.FIRED:
            return None
        if self.phase == Phase.COOLDOWN:
            if ts < self.cooldown_until:
                return None
            self.phase = Phase.SCANNING
        if (
            self.last_trigger_price > 0.0
            and snap.direction == self.last_direction
            and abs(snap.level_price - self.last_trigger_price) / self.last_trigger_price * 1e4
            < cfg.re_arm_bps
        ):
            return None
        if self.trend_alerts_today >= cfg.trend_max_per_day:
            return self._reject("trend_day_cap", snap)
        if self.alerts_today >= cfg.max_alerts_symbol_day:
            return self._reject("symbol_day_cap", snap)
        if ts - self.last_alert_ts < cfg.min_gap_same_symbol_s:
            return None
        if not self.limiter.allow(ts):
            return self._reject("global_hour_cap", snap)
        out = self._fire(snap, allow_suppress=cfg.trend_model_gated)
        if isinstance(out, Signal):
            self.trend_alerts_today += 1
        return out

    # ----------------------------------------------------------------- firing

    def _fire(self, snap: Snapshot, allow_suppress: bool = True) -> Signal | Rejection:
        cfg = self.cfg
        # Opening phase: the 5m-range estimate is degenerate (no completed
        # minutes) while realized whip is at its daily maximum — floor the
        # range input so invalidations breathe with opening volatility.
        range_bps = self._range5m_bps_value
        if snap.minute_frac * 390.0 < cfg.open_phase_min:
            range_bps = max(range_bps, cfg.open_range_floor_bps)
        inv_dist_bps = min(
            max(
                cfg.invalidation_min_bps,
                cfg.invalidation_range_frac * range_bps,
            ),
            cfg.invalidation_max_bps,
        )
        inv_price = snap.level_price * (1.0 - snap.direction * inv_dist_bps / 1e4)

        features = snap.to_features()
        prob, model_version = self.hooks.ml_predict(features, snap.direction)
        suppressed = (
            allow_suppress
            and self.hooks.ml_gate_active()
            and prob is not None
            and prob < self.hooks.prob_gate_min
            # Exceptional raw evidence always delivers — the model (and the
            # market context it reads) informs, it never vetoes the strongest
            # price+volume signals.
            and snap.score < self.hooks.prob_bypass_score
        )

        sig = Signal(
            signal_id=f"{self.symbol}-{int(snap.ts * 1000)}-{'U' if snap.direction > 0 else 'D'}",
            symbol=self.symbol,
            direction=snap.direction,
            alert_ts=snap.ts,
            alert_price=snap.price,
            trigger_price=snap.level_price,
            level_kind=snap.level_kind,
            trigger_verb=snap.trigger_verb,
            invalidation=round(inv_price, 4),
            vol_ratio=snap.vol_ratio_prev,
            features=features,
            prob=prob,
            model_version=model_version,
            score=snap.score,
            vol5=snap.vol5,
            suppressed=suppressed,
            micro_extreme=snap.price,
        )

        if suppressed:
            # Shadow signal: no notification, but tracked/labeled so the model
            # keeps learning about what it suppressed.
            self.suppressed_today += 1
            self.signal = sig
            self.phase = Phase.FIRED
            self.last_alert_ts = snap.ts
            self.last_trigger_price = snap.level_price
            self.last_direction = snap.direction
            log.info("suppressed by model gate p=%.2f %s", prob or -1.0, sig.signal_id)
            self.hooks.on_signal_fired(sig)
            return sig

        self.alerts_today += 1
        self.last_alert_ts = snap.ts
        self.last_trigger_price = snap.level_price
        self.last_direction = snap.direction
        self.limiter.note(snap.ts)
        self.signal = sig
        self.phase = Phase.FIRED
        # Highest-priority path first: emit() only enqueues the Telegram message.
        self.hooks.emit(sig, "EARLY", {})
        self.hooks.on_signal_fired(sig)
        return sig

    _range5m_bps_value: float = 15.0

    def set_range5m_bps(self, value: float) -> None:
        """Recent 5-minute range in bps; sizes the invalidation distance."""
        self._range5m_bps_value = max(value, 1.0)

    # ------------------------------------------------------------- follow-up

    def observe(
        self,
        ts: float,
        price: float,
        imb5: float,
        share_up5: float,
        vol5: float,
    ) -> None:
        """Called on every print and every second tick while FIRED.

        `share_up5` is the buy share of classified volume; it is converted to
        the share in the signal's direction here.
        """
        sig = self.signal
        if self.phase != Phase.FIRED or sig is None:
            return
        cfg = self.cfg
        d = sig.direction
        dir_share5 = share_up5 if d > 0 else 1.0 - share_up5

        if (price - sig.micro_extreme) * d > 0:
            sig.micro_extreme = price

        # Hard invalidation: through the invalidation price.
        if (price - sig.invalidation) * d < 0:
            self._resolve(sig, ts, "FAILED", "hit invalidation")
            return

        # Trigger recross + directional flow gone. Suspended during the opening
        # phase: opening tape routinely whips back through the trigger before
        # launching, so only the hard invalidation or the progress deadline may
        # fail an opening signal.
        in_open = sig.features.get("minute", 1.0) * 390.0 < cfg.open_phase_min
        if not in_open:
            recross = (
                price - sig.trigger_price
            ) * d < -cfg.fail_buffer_bps / 1e4 * sig.trigger_price
            flow_dead = imb5 * d <= 0.0 or dir_share5 < 0.40
            if recross and flow_dead:
                self._resolve(sig, ts, "FAILED", "directional volume reversed")
                return

        # Confirmation requires real expansion progress, not mere survival:
        # the micro-extreme must travel confirm_min_r x the invalidation
        # distance beyond the trigger. Anything less by the deadline is a
        # FAILED follow-up — a confirmation should predict continuation.
        inv_dist = abs(sig.trigger_price - sig.invalidation)
        progress_r = (sig.micro_extreme - sig.trigger_price) * d / inv_dist if inv_dist > 0 else 0.0

        elapsed = ts - sig.alert_ts
        if elapsed >= cfg.confirm_min_s:
            beyond = (price - sig.trigger_price) * d > 0
            flow_ok = dir_share5 >= 0.5 or imb5 * d > 0
            if beyond and progress_r >= cfg.confirm_min_r and flow_ok and vol5 > 0:
                self._resolve(sig, ts, "CONFIRMED", "volume sustained")
                return
        if elapsed >= cfg.observe_max_s:
            if (price - sig.trigger_price) * d > 0 and progress_r >= cfg.confirm_min_r:
                self._resolve(sig, ts, "CONFIRMED", "held at deadline")
            else:
                self._resolve(sig, ts, "FAILED", "no expansion progress")

    def _resolve(self, sig: Signal, ts: float, state: str, reason: str) -> None:
        sig.resolution = state
        sig.resolution_ts = ts
        sig.resolution_reason = reason
        # Hard cap: EARLY + at most 2 follow-ups per setup (we send exactly 1).
        if not sig.suppressed and sig.followups_sent < 2:
            sig.followups_sent += 1
            self.hooks.emit(sig, state, {"reason": reason})
        self.hooks.on_signal_final(sig)
        if sig.suppressed:
            # Nothing was sent, so there is no alert spam to throttle — a
            # short cooldown only, or the model silencing one marginal setup
            # locks the symbol out of the next (possibly deliverable) move.
            cooldown = self.cfg.cooldown_suppressed_s
        elif state == "CONFIRMED":
            cooldown = self.cfg.cooldown_confirmed_s
        else:
            cooldown = self.cfg.cooldown_failed_s
        self.cooldown_until = ts + cooldown
        self.phase = Phase.COOLDOWN
        self.signal = None

    # ------------------------------------------------------------------- tick

    def on_tick(self, ts: float, price: float, imb5: float, share_up5: float, vol5: float) -> None:
        if self.phase == Phase.FIRED:
            self.observe(ts, price, imb5, share_up5, vol5)
        elif self.phase == Phase.COOLDOWN and ts >= self.cooldown_until:
            self.phase = Phase.SCANNING
