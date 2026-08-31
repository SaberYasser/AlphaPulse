from early_trend_scanner.config import MlCfg
from early_trend_scanner.data.models import Sec
from early_trend_scanner.engine.rolling import SecRing
from early_trend_scanner.engine.state import Signal
from early_trend_scanner.ml.labeler import Labeler, LabelResult

T0 = 1_756_000_000


def sig(direction: int = 1, alert_price: float = 100.05, trigger: float = 100.0) -> Signal:
    inv = trigger - direction * 0.10  # invalidation distance = 10 cents
    return Signal(
        signal_id="s1",
        symbol="X",
        direction=direction,
        alert_ts=float(T0),
        alert_price=alert_price,
        trigger_price=trigger,
        level_kind=0,
        trigger_verb="break",
        invalidation=inv,
        vol_ratio=3.0,
        features={},
    )


def ring_from_path(path: list[float], start: int = T0) -> SecRing:
    ring = SecRing(600)
    for i, px in enumerate(path):
        ring.append(Sec(ts=start + i, open=px, high=px + 0.01, low=px - 0.01, close=px))
    return ring


def collect(cfg: MlCfg, s: Signal, ring: SecRing) -> LabelResult:
    results: list[LabelResult] = []
    lab = Labeler(cfg, results.append)
    lab.track(s)
    lab.on_tick(s.alert_ts + cfg.outcome_window_s + 2, {"X": ring})
    assert results, "label not resolved"
    return results[0]


def test_positive_label_clean_expansion() -> None:
    cfg = MlCfg(outcome_window_s=120, pos_multiple=1.5, min_remaining_frac=0.5)
    # steady climb: +40 cents over the window (inv dist = 10c, need MFE >= 15c)
    path = [100.05 + i * 0.4 / 120 for i in range(125)]
    r = collect(cfg, sig(), ring_from_path(path))
    assert r.label is True
    assert r.reason == "expanded"
    assert r.lead_time_s is not None and r.lead_time_s < 60
    assert r.remaining_frac > 0.5
    assert r.mfe > 0.15


def test_negative_label_immediate_failure() -> None:
    cfg = MlCfg(outcome_window_s=120)
    # collapses through invalidation (99.90) then chops
    path = [100.05, 100.0, 99.92, 99.85, 99.8] + [99.85] * 120
    r = collect(cfg, sig(), ring_from_path(path))
    assert r.label is False
    assert r.reason in ("no_expansion", "insufficient_expansion")


def test_expansion_after_invalidation_does_not_count() -> None:
    cfg = MlCfg(outcome_window_s=120, pos_multiple=1.5)
    # dies through invalidation first, THEN rips: must stay negative
    path = [100.05, 99.95, 99.85] + [99.85 + i * 0.01 for i in range(120)]
    r = collect(cfg, sig(), ring_from_path(path))
    assert r.label is False


def test_late_alert_negative_via_remaining_frac() -> None:
    cfg = MlCfg(outcome_window_s=120, pos_multiple=1.5, min_remaining_frac=0.5)
    # alert taken far above trigger: move mostly over (peak 100.60, alert 100.45)
    s = sig(alert_price=100.45)
    path = [100.45 + i * 0.15 / 120 for i in range(125)]
    r = collect(cfg, s, ring_from_path(path))
    assert r.label is False
    assert r.was_late is True
    assert r.reason == "alert_too_late"


def test_downside_symmetry() -> None:
    cfg = MlCfg(outcome_window_s=120, pos_multiple=1.5, min_remaining_frac=0.5)
    s = sig(direction=-1, alert_price=99.95, trigger=100.0)  # inv at 100.10
    path = [99.95 - i * 0.4 / 120 for i in range(125)]
    r = collect(cfg, s, ring_from_path(path))
    assert r.label is True
    assert r.mfe > 0.15


def test_flush_truncated_window_dropped() -> None:
    cfg = MlCfg(outcome_window_s=300)
    results: list[LabelResult] = []
    lab = Labeler(cfg, results.append)
    lab.track(sig())
    ring = ring_from_path([100.05] * 30)  # only 30s of data
    n = lab.flush({"X": ring})
    assert n == 0 and results == []
    assert lab.pending_count == 0
