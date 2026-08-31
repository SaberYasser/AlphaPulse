"""Streaming learners: separate upside/downside probability models.

Preferred engine is River's Adaptive Random Forest (the strongest free
streaming classifier: an ensemble of adaptive Hoeffding trees) plus an outer
ADWIN concept-drift monitor. A dependency-free builtin engine implements the
same contract (Welford scaler, SGD logistic regression, EWMA drift detector)
so the scanner never loses its learning loop if River is absent.

Invariant: `learn` is only ever called with the feature vector captured at
prediction time and a label derived from the later outcome window — no
post-alert information leaks into features.
"""

from __future__ import annotations

import logging
import math
import pickle
from abc import ABC, abstractmethod
from typing import Any

log = logging.getLogger(__name__)


class OnlineModel(ABC):
    engine_name = "base"

    def __init__(self) -> None:
        self.n_learned = {1: 0, -1: 0}
        self.drift_events = 0

    @abstractmethod
    def predict(self, features: dict[str, float], direction: int) -> float:
        """P(positive-label expansion) for the given direction."""

    @abstractmethod
    def learn(self, features: dict[str, float], direction: int, label: bool) -> None: ...

    @property
    def version(self) -> str:
        up, dn = self.n_learned[1], self.n_learned[-1]
        return f"{self.engine_name}:u{up}+d{dn}:drift{self.drift_events}"

    def ready(self, min_labels: int) -> bool:
        return (self.n_learned[1] + self.n_learned[-1]) >= min_labels

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)

    @staticmethod
    def from_bytes(data: bytes) -> OnlineModel:
        obj = pickle.loads(data)
        if not isinstance(obj, OnlineModel):
            raise TypeError("model file does not contain an OnlineModel")
        return obj


# --------------------------------------------------------------------- builtin


class _WelfordScaler:
    """Online per-feature standardization."""

    def __init__(self) -> None:
        self.n: dict[str, int] = {}
        self.mean: dict[str, float] = {}
        self.m2: dict[str, float] = {}

    def update(self, x: dict[str, float]) -> None:
        for k, v in x.items():
            n = self.n.get(k, 0) + 1
            mean = self.mean.get(k, 0.0)
            delta = v - mean
            mean += delta / n
            self.n[k] = n
            self.mean[k] = mean
            self.m2[k] = self.m2.get(k, 0.0) + delta * (v - mean)

    def transform(self, x: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in x.items():
            n = self.n.get(k, 0)
            if n < 2:
                out[k] = 0.0
                continue
            var = self.m2[k] / (n - 1)
            out[k] = (v - self.mean[k]) / math.sqrt(var) if var > 1e-12 else 0.0
        return out


class _LogReg:
    """SGD logistic regression with decaying learning rate and L2."""

    def __init__(self, lr: float = 0.05, l2: float = 1e-4) -> None:
        self.w: dict[str, float] = {}
        self.b = 0.0
        self.t = 0
        self.lr0 = lr
        self.l2 = l2

    def _raw(self, x: dict[str, float]) -> float:
        return self.b + sum(self.w.get(k, 0.0) * v for k, v in x.items())

    def predict(self, x: dict[str, float]) -> float:
        z = max(-30.0, min(30.0, self._raw(x)))
        return 1.0 / (1.0 + math.exp(-z))

    def learn(self, x: dict[str, float], y: float) -> float:
        p = self.predict(x)
        self.t += 1
        lr = self.lr0 / math.sqrt(1.0 + 0.01 * self.t)
        g = p - y
        for k, v in x.items():
            w = self.w.get(k, 0.0)
            self.w[k] = w - lr * (g * v + self.l2 * w)
        self.b -= lr * g
        eps = 1e-9
        return -(y * math.log(p + eps) + (1.0 - y) * math.log(1.0 - p + eps))


class _EwmaDrift:
    """Fast/slow EWMA log-loss comparison (drift fallback when River is absent)."""

    def __init__(self, fast: float = 0.30, slow: float = 0.02, margin: float = 0.25) -> None:
        self.fast_a = fast
        self.slow_a = slow
        self.margin = margin
        self.fast = 0.0
        self.slow = 0.0
        self.n = 0

    def update(self, loss: float) -> bool:
        self.n += 1
        if self.n == 1:
            self.fast = self.slow = loss
            return False
        self.fast += self.fast_a * (loss - self.fast)
        self.slow += self.slow_a * (loss - self.slow)
        return self.n > 30 and self.fast > self.slow + self.margin


class BuiltinModel(OnlineModel):
    engine_name = "builtin"

    def __init__(self) -> None:
        super().__init__()
        self._models = {1: _LogReg(), -1: _LogReg()}
        self._scalers = {1: _WelfordScaler(), -1: _WelfordScaler()}
        self._drift = {1: _EwmaDrift(), -1: _EwmaDrift()}

    def predict(self, features: dict[str, float], direction: int) -> float:
        d = 1 if direction >= 0 else -1
        return self._models[d].predict(self._scalers[d].transform(features))

    def learn(self, features: dict[str, float], direction: int, label: bool) -> None:
        d = 1 if direction >= 0 else -1
        self._scalers[d].update(features)
        x = self._scalers[d].transform(features)
        loss = self._models[d].learn(x, 1.0 if label else 0.0)
        self.n_learned[d] += 1
        if self._drift[d].update(loss):
            log.warning("builtin drift detected (dir=%+d) — resetting learner", d)
            self._models[d] = _LogReg()
            self._drift[d] = _EwmaDrift()
            self.drift_events += 1


# ----------------------------------------------------------------------- river


def _river_model() -> Any:
    """Best free streaming classifier available: Adaptive Random Forest.

    ARF is River's strongest general online learner (an ensemble of adaptive
    Hoeffding trees, each with its own drift/warning detectors). 10 trees keeps
    CPU negligible at our label rates. Falls back to standardized logistic
    regression if the forest module is unavailable.
    """
    try:
        from river import forest

        return forest.ARFClassifier(n_models=10, seed=42, leaf_prediction="nba")
    except ImportError:  # pragma: no cover - depends on river version
        from river import linear_model, preprocessing

        return preprocessing.StandardScaler() | linear_model.LogisticRegression()


class RiverModel(OnlineModel):
    engine_name = "river"

    def __init__(self) -> None:
        super().__init__()
        from river import drift

        self._models: dict[int, Any] = {1: _river_model(), -1: _river_model()}
        self._adwin: dict[int, Any] = {1: drift.ADWIN(delta=0.002), -1: drift.ADWIN(delta=0.002)}

    def predict(self, features: dict[str, float], direction: int) -> float:
        d = 1 if direction >= 0 else -1
        proba = self._models[d].predict_proba_one(features)
        return float(proba.get(True, 0.5))

    def learn(self, features: dict[str, float], direction: int, label: bool) -> None:
        d = 1 if direction >= 0 else -1
        p = self.predict(features, d)
        self._models[d].learn_one(features, label)
        self.n_learned[d] += 1
        det = self._adwin[d]
        det.update(abs((1.0 if label else 0.0) - p))
        if det.drift_detected:
            log.warning("ADWIN drift detected (dir=%+d) — resetting learner", d)
            from river import drift

            self._models[d] = _river_model()
            self._adwin[d] = drift.ADWIN(delta=0.002)
            self.drift_events += 1


def make_model(engine: str = "auto") -> OnlineModel:
    if engine in ("auto", "river"):
        try:
            model = RiverModel()
            log.info("ML engine: river")
            return model
        except ImportError:
            if engine == "river":
                raise
            log.warning("river not installed — using builtin online learner")
    log.info("ML engine: builtin")
    return BuiltinModel()
