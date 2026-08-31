import random

import pytest

from early_trend_scanner.config import MlCfg
from early_trend_scanner.ml.gate import AdaptiveGate, LabelOutcome, tod_bucket
from early_trend_scanner.ml.online import BuiltinModel, OnlineModel, make_model


def _learnable_stream(n: int, seed: int = 3):
    rng = random.Random(seed)
    for _ in range(n):
        good = rng.random() < 0.5
        feats = {
            "vol_prev": (4.0 if good else 1.2) + rng.gauss(0, 0.3),
            "imb5": (0.5 if good else 0.05) + rng.gauss(0, 0.05),
            "vel5": (2.0 if good else 0.3) + rng.gauss(0, 0.2),
            "break_bps": 5.0 + rng.gauss(0, 1.0),
        }
        yield feats, good


def test_builtin_model_learns_separable_pattern() -> None:
    model = BuiltinModel()
    for feats, good in _learnable_stream(400):
        model.learn(feats, 1, good)
    good_feats = {"vol_prev": 4.0, "imb5": 0.5, "vel5": 2.0, "break_bps": 5.0}
    bad_feats = {"vol_prev": 1.2, "imb5": 0.05, "vel5": 0.3, "break_bps": 5.0}
    assert model.predict(good_feats, 1) > 0.7
    assert model.predict(bad_feats, 1) < 0.3
    assert model.n_learned[1] == 400 and model.n_learned[-1] == 0
    assert model.ready(min_labels=100)


def test_directions_are_independent() -> None:
    model = BuiltinModel()
    for feats, good in _learnable_stream(200):
        model.learn(feats, 1, good)
    neutral = {"vol_prev": 4.0, "imb5": 0.5, "vel5": 2.0, "break_bps": 5.0}
    assert abs(model.predict(neutral, -1) - 0.5) < 0.1  # untouched down model


def test_builtin_drift_reset() -> None:
    model = BuiltinModel()
    for feats, good in _learnable_stream(200, seed=1):
        model.learn(feats, 1, good)
    # concept flips: previously-good features now always fail
    for feats, good in _learnable_stream(300, seed=2):
        model.learn(feats, 1, not good)
    assert model.drift_events >= 1


def test_model_persistence_roundtrip(tmp_path) -> None:
    model = BuiltinModel()
    for feats, good in _learnable_stream(100):
        model.learn(feats, 1, good)
    data = model.to_bytes()
    loaded = OnlineModel.from_bytes(data)
    f = {"vol_prev": 4.0, "imb5": 0.5, "vel5": 2.0, "break_bps": 5.0}
    assert abs(loaded.predict(f, 1) - model.predict(f, 1)) < 1e-9
    assert loaded.version == model.version


def test_river_model_if_available() -> None:
    pytest.importorskip("river")
    model = make_model("river")
    assert model.engine_name == "river"
    for feats, good in _learnable_stream(300):
        model.learn(feats, 1, good)
    assert model.predict({"vol_prev": 4.0, "imb5": 0.5, "vel5": 2.0, "break_bps": 5.0}, 1) > 0.6
    assert model.predict({"vol_prev": 1.2, "imb5": 0.05, "vel5": 0.3, "break_bps": 5.0}, 1) < 0.4
    loaded = OnlineModel.from_bytes(model.to_bytes())
    assert loaded.engine_name == "river"
    assert loaded.predict({"vol_prev": 4.0, "imb5": 0.5, "vel5": 2.0, "break_bps": 5.0}, 1) > 0.6


# ------------------------------------------------------------------ gate


def _outcome(label: bool, sym: str = "TSLA", ts: float = 0.0) -> LabelOutcome:
    return LabelOutcome("id", sym, 1, label, was_late=not label, ts=ts)


def test_gate_inactive_until_min_labels() -> None:
    cfg = MlCfg(min_labels=10)
    gate = AdaptiveGate(cfg=cfg)
    assert not gate.active
    assert gate.multipliers("TSLA", 0.0) == (1.0, 1.0)
    for _ in range(10):
        gate.on_label(_outcome(True))
    assert gate.active
    assert gate.baseline_precision == 1.0


def test_multipliers_bounded() -> None:
    cfg = MlCfg(min_labels=5, bound_low=0.75, bound_high=1.5, adapt_step=0.1)
    gate = AdaptiveGate(cfg=cfg)
    for _ in range(200):
        gate.on_label(_outcome(True))
    vol, score = gate.multipliers("TSLA", 0.0)
    assert vol >= 0.75 and score >= 0.75
    gate2 = AdaptiveGate(cfg=MlCfg(min_labels=5, adapt_step=0.1, revert_precision_drop=2.0))
    for i in range(5):
        gate2.on_label(_outcome(True, ts=float(i)))
    for i in range(200):
        gate2.on_label(_outcome(False, ts=float(i)))
    vol2, score2 = gate2.multipliers("TSLA", 0.0)
    assert vol2 <= 1.5 and score2 <= 1.5


def test_revert_on_precision_deterioration() -> None:
    cfg = MlCfg(min_labels=10, revert_window=20, revert_precision_drop=0.15)
    gate = AdaptiveGate(cfg=cfg)
    for _ in range(10):
        gate.on_label(_outcome(True))  # baseline precision = 1.0
    assert gate.active
    v0 = gate.version
    for _ in range(20):
        gate.on_label(_outcome(False))  # precision collapses
    assert gate.reverted
    assert not gate.active  # ML influence disabled
    # may revert again after an unsuccessful relearn period — version only grows
    assert gate.version > v0
    assert gate.multipliers("TSLA", 0.0) == (1.0, 1.0)


def test_gate_json_roundtrip() -> None:
    cfg = MlCfg(min_labels=5)
    gate = AdaptiveGate(cfg=cfg)
    for i in range(7):
        gate.on_label(_outcome(i % 2 == 0, ts=float(i)))
    raw = gate.to_json()
    loaded = AdaptiveGate.from_json(cfg, raw)
    assert loaded.n_labels == gate.n_labels
    assert loaded.baseline_precision == gate.baseline_precision
    assert loaded.multipliers("TSLA", 0.0) == gate.multipliers("TSLA", 0.0)


def test_tod_bucket() -> None:
    open_ts, close_ts = 1000.0, 1000.0 + 6.5 * 3600
    assert tod_bucket(open_ts + 60, open_ts, close_ts) == 0
    assert tod_bucket(open_ts + 3 * 3600, open_ts, close_ts) == 1
    assert tod_bucket(close_ts - 60, open_ts, close_ts) == 2
