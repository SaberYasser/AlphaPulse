"""Shared fixtures: default config, engine harness with captured notifications."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from early_trend_scanner.config import Config  # noqa: E402
from early_trend_scanner.engine.baseline import MinuteBaseline  # noqa: E402
from early_trend_scanner.engine.state import (  # noqa: E402
    GlobalLimiter,
    MachineHooks,
    Signal,
)
from early_trend_scanner.engine.symbol_engine import SymbolEngine  # noqa: E402

BASE_TS = 1_756_000_000.0 - (1_756_000_000.0 % 60)  # minute-aligned epoch


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(root=tmp_path)


class Harness:
    """One SymbolEngine wired with recording hooks and no ML gating."""

    def __init__(self, cfg: Config, symbol: str = "TEST") -> None:
        self.cfg = cfg
        self.emitted: list[tuple[Signal, str, dict[str, Any]]] = []
        self.fired: list[Signal] = []
        self.final: list[Signal] = []
        hooks = MachineHooks(
            emit=lambda s, k, e: self.emitted.append((s, k, e)),
            ml_predict=lambda f, d: (None, "test"),
            gate_multipliers=lambda s, t: (1.0, 1.0),
            ml_gate_active=lambda: False,
            prob_gate_min=0.35,
            on_signal_fired=self.fired.append,
            on_signal_final=self.final.append,
        )
        self.engine = SymbolEngine(
            symbol, cfg, MinuteBaseline(), hooks, GlobalLimiter(100), lambda: True
        )
        self.engine.set_session(BASE_TS)

    def messages(self, kind: str | None = None) -> list[str]:
        return [k for _s, k, _e in self.emitted if kind is None or k == kind]


@pytest.fixture
def harness(cfg: Config) -> Harness:
    return Harness(cfg)
