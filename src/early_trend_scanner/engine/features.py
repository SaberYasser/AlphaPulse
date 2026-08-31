"""Trigger snapshot: every price/volume feature evaluated at one instant.

The same snapshot drives the rule gates, the trigger score, the ML feature
vector and the Telegram message — computed once per candidate evaluation,
all O(1) against the rolling windows.
"""

from __future__ import annotations

from dataclasses import dataclass

from .levels import BREAK_VERB, Level, LevelBook, LevelKind


@dataclass(slots=True)
class Snapshot:
    ts: float
    symbol: str
    direction: int  # +1 up candidate, -1 down candidate
    price: float
    level_price: float
    level_kind: int
    trigger_verb: str  # break | reclaim | reject | cross | pivot

    break_bps: float  # distance beyond the level, in bps (>=0 once crossed)
    vel5_bps_s: float  # signed 5s velocity, bps/s
    vel15_bps_s: float
    accelerating: bool
    vol5: float
    vol_ratio_prev: float  # 5s volume vs avg-5s over the prior 60s
    vol_ratio_base: float  # 5s volume vs minute-of-day baseline per-5s
    imb5: float  # signed volume imbalance -1..1 over 5s
    imb15: float
    n5: float
    n_ratio_prev: float
    dollar5: float
    range_exp: float  # current 15s range vs avg 15s range of prior 5m
    comp_ratio: float
    comp_active: bool
    dist_vwap_bps: float
    spread_bps: float
    quote_imb: float  # (bid_size - ask_size) / (bid_size + ask_size)
    minute_frac: float  # session progress 0..1
    persist_trades: int
    persist_span_s: float
    n2s: int
    dominant_frac: float = 0.0  # largest single print / directional volume (persist window)
    # Market-regime context (0.0 = neutral/unknown). Aligned = x direction:
    # positive means the market proxy is moving WITH this candidate.
    mkt_al_1m: float = 0.0  # market-proxy 1m velocity (bps/s), aligned
    mkt_al_5m: float = 0.0  # market-proxy 5m velocity (bps/s), aligned
    fear_1m: float = 0.0  # fear-proxy 1m velocity (bps/s), unaligned fast pulse
    fear_5m: float = 0.0  # fear-proxy 5m velocity (bps/s), unaligned trend
    is_trend: float = 0.0  # 1.0 = sustained-pressure (trend-onset) detector class
    score: float = 0.0

    def to_features(self) -> dict[str, float]:
        """Compact, model-facing feature vector (no post-alert information)."""
        return {
            "dir": float(self.direction),
            "break_bps": self.break_bps,
            "vel5": self.vel5_bps_s * self.direction,
            "vel15": self.vel15_bps_s * self.direction,
            "accel": 1.0 if self.accelerating else 0.0,
            "vol_prev": min(self.vol_ratio_prev, 20.0),
            "vol_base": min(self.vol_ratio_base, 20.0),
            "imb5": self.imb5 * self.direction,
            "imb15": self.imb15 * self.direction,
            "n_prev": min(self.n_ratio_prev, 20.0),
            "range_exp": min(self.range_exp, 10.0),
            "comp": min(self.comp_ratio, 3.0),
            "dist_vwap": max(-100.0, min(self.dist_vwap_bps * self.direction, 100.0)),
            "spread": min(self.spread_bps, 50.0),
            "q_imb": self.quote_imb * self.direction,
            "minute": self.minute_frac,
            "persist": float(self.persist_trades),
            "dominance": self.dominant_frac,
            "mkt_al_1m": self.mkt_al_1m,
            "mkt_al_5m": self.mkt_al_5m,
            "fear_1m": self.fear_1m,
            "fear_5m": self.fear_5m,
            "level_major": 1.0 if self.level_kind in (0, 1, 3, 4, 5, 6) else 0.0,
            "is_reclaim": 1.0 if self.trigger_verb == "reclaim" else 0.0,
            "trend": self.is_trend,
        }


_SCORE_WEIGHTS = {
    "vol_prev": 0.25,
    "vol_base": 0.15,
    "imb": 0.20,
    "vel": 0.20,
    "fresh": 0.10,
    "range_exp": 0.05,
    "comp": 0.05,
}


def fake_start_penalty(
    s: Snapshot,
    comp_veto_ratio: float,
    imb15_blowoff_max: float,
    fresh_break_vol_max: float,
    weight: float,
) -> float:
    """Continuous chase-evidence penalty, subtracted from the trigger score.

    Mined from labeled sessions (2026-08-24..28): failed "early" signals
    disproportionately fire from a range that is still expanding (no coiled
    energy behind the break) or into a 15-second tape that is already
    one-sided (the crowd chased; the earliest stage of the move is gone).
    Both are evidence, not verdicts, so they weigh against the score instead
    of hard-gating: strong volume/velocity/imbalance can outvote marginal
    staleness. The whole penalty is damped to zero as 5s volume acceleration
    approaches fresh_break_vol_max — a fresh volume explosion is never
    penalized, whatever the range looked like before it.
    """
    damp = 1.0 - s.vol_ratio_prev / max(fresh_break_vol_max, 1e-9)
    if damp <= 0.0:
        return 0.0
    stale_range = min(max(s.comp_ratio - comp_veto_ratio, 0.0) / 0.5, 1.0)
    stale_tape = min(max(s.imb15 * s.direction - imb15_blowoff_max, 0.0) / 0.25, 1.0)
    return weight * damp * max(stale_range, stale_tape)


def compute_score(s: Snapshot, vol_accel_min: float, vel_min: float, break_max_bps: float) -> float:
    """0..1 weighted trigger quality; components saturate to keep it bounded."""
    v_vol = min(s.vol_ratio_prev / (2.0 * vol_accel_min), 1.0)
    if s.vol_ratio_base > 0.0:
        v_base = min(s.vol_ratio_base / (2.0 * max(0.5, vol_accel_min * 0.75)), 1.0)
    else:
        v_base = v_vol  # baseline unknown (e.g. opening minutes): neutral, not a penalty
    v_imb = min(abs(s.imb5) / 0.6, 1.0)
    v_vel = min((s.vel5_bps_s * s.direction) / (3.0 * vel_min), 1.0) if vel_min > 0 else 0.0
    v_fresh = 1.0 - min(s.break_bps / max(break_max_bps, 1.0), 1.0)
    v_range = min(s.range_exp / 3.0, 1.0)
    v_comp = 1.0 - min(s.comp_ratio, 1.5) / 1.5
    score = (
        _SCORE_WEIGHTS["vol_prev"] * v_vol
        + _SCORE_WEIGHTS["vol_base"] * v_base
        + _SCORE_WEIGHTS["imb"] * v_imb
        + _SCORE_WEIGHTS["vel"] * max(v_vel, 0.0)
        + _SCORE_WEIGHTS["fresh"] * v_fresh
        + _SCORE_WEIGHTS["range_exp"] * v_range
        + _SCORE_WEIGHTS["comp"] * max(v_comp, 0.0)
    )
    return max(0.0, min(score, 1.0))


def trigger_verb(level: Level, is_reclaim: bool, direction: int) -> str:
    if is_reclaim:
        return "reclaim"
    if level.kind in BREAK_VERB:
        return BREAK_VERB[level.kind]
    upper = LevelBook.is_upper(level)
    if level.kind in (LevelKind.PDC, LevelKind.VWAP):
        return "cross"
    if (upper and direction < 0) or (not upper and direction > 0):
        return "reject"
    return "break"
