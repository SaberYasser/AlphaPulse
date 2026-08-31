"""Per-symbol signal state machine — PUBLIC INTERFACE.

    SCANNING -> READY -> EARLY_SIGNAL(FIRED) -> CONFIRMED | FAILED -> COOLDOWN

The EARLY alert is emitted the instant minimum price+volume conditions hold —
never waiting for a completed candle, retest or higher-timeframe agreement.
Follow-through is analyzed only after the alert is already on its way.

NOTE — proprietary core withheld from this public repository.
The decision methods below (`consider`, `observe`, `_fire`) implement the
calibrated gate sequence: price-trigger gates, dual-ruler volume acceleration
(prior-window midday-baseline switching for the opening phase), directional
imbalance, dense-tape persistence, the continuous fake-start penalty, the
model probability gate with a strong-evidence bypass, opening-phase risk
sizing and the notification rate limits. Their signatures, data contracts and
the full test/replay methodology are shown; the tuned logic itself is the
project's edge and lives in a private overlay. See README ("Protected core").
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..config import EngineCfg
from .features import Snapshot

log = logging.getLogger(__name__)

_PRIVATE = (
    "Proprietary detection core: this method is withheld from the public "
    "repository. See README — 'Protected core'."
)


class Phase(Enum):
    SCANNING = "SCANNING"
    READY = "READY"
    FIRED = "EARLY_SIGNAL"
    COOLDOWN = "COOLDOWN"


@dataclass
class Signal:
    """Everything captured at alert time (features frozen for later learning)."""

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
        self.rejections: dict[str, int] = {}

    # ------------------------------------------------------------- evaluation

    def consider(self, snap: Snapshot, threshold_score: float) -> Signal | Rejection | None:
        """Evaluate one candidate snapshot; fire, or explain why not.

        Sequenced funnel (withheld): structural re-arm rules -> price-trigger
        gates with an opening-phase extension cap -> dual-ruler volume
        acceleration -> directional imbalance -> dense-tape persistence and
        single-print dominance -> trigger score minus the continuous
        fake-start penalty (attributed rejection) -> per-symbol / global /
        burst rate caps -> fire (or model-suppress with a strong-score bypass).
        Every rejection is counted by reason for replay diagnostics.
        """
        raise NotImplementedError(_PRIVATE)

    def observe(
        self,
        ts: float,
        price: float,
        imb5: float,
        share_up5: float,
        vol5: float,
        env: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Follow a FIRED signal to CONFIRMED/FAILED (withheld).

        The verdict runs ~1 minute after the alert. CONFIRMED requires real
        expansion progress in invalidation-distance units, not mere survival;
        `env` carries resolution-time environment (market alignment, fear
        velocity, event-tape volume) used as a bounded tiebreaker on marginal
        progress and quoted as a brief justification in the follow-up message.
        The opening phase suspends the twitch-sensitive recross fail.
        """
        raise NotImplementedError(_PRIVATE)

    def consider_trend(self, snap: Snapshot) -> Signal | Rejection | None:
        """Sustained-pressure ("trend onset") entry — the band between
        micro-burst and grind: a 60-90s escalator on dominant one-sided
        volume printing new local extremes. Shares every discipline mechanism
        (cooldowns, structural re-arm, per-symbol/global/burst caps) and runs
        rule-pure until the model has learned the class. Withheld.
        """
        raise NotImplementedError(_PRIVATE)

    def set_range5m_bps(self, value: float) -> None:
        """Recent 5-minute range in bps; sizes the invalidation distance."""
        self._range5m_bps_value = max(value, 1.0)

    _range5m_bps_value: float = 15.0

    def on_tick(
        self,
        ts: float,
        price: float,
        imb5: float,
        share5: float,
        vol5: float,
        env: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Wall-clock upkeep for FIRED deadlines on quiet tape (withheld)."""
        raise NotImplementedError(_PRIVATE)
