"""Replay harness: run the identical live pipeline over historical events.

Strictly no lookahead — events are consumed in timestamp order and every
component only ever sees data at or before the virtual clock. Used for the
bundled synthetic regression sessions and for real Alpaca historical data.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from ..data.filters import TradeFilter
from ..data.models import Quote, SessionInfo, Trade
from ..engine.baseline import MinuteBaseline
from ..engine.context import ContextTracker
from ..engine.state import GlobalLimiter, MachineHooks, Signal
from ..engine.symbol_engine import SymbolEngine
from ..ml.gate import AdaptiveGate, LabelOutcome
from ..ml.labeler import Labeler, LabelResult
from ..ml.online import OnlineModel, make_model
from ..notify.telegram import format_message
from ..status import rss_bytes
from ..store.metrics import MetricsTracker
from ..timeutil import et_hms

log = logging.getLogger(__name__)

Event = Trade | Quote


@dataclass
class ReplayResult:
    signals: list[Signal] = field(default_factory=list)
    labels: list[LabelResult] = field(default_factory=list)
    messages: list[tuple[float, str]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    events_processed: int = 0
    wall_seconds: float = 0.0
    peak_rss_mb: float = 0.0
    ring_sizes: dict[str, int] = field(default_factory=dict)
    rejections: dict[str, dict[str, int]] = field(default_factory=dict)

    def report(self) -> dict[str, Any]:
        total = self.metrics.get("total", {})
        return {
            "events_processed": self.events_processed,
            "wall_seconds": round(self.wall_seconds, 2),
            "events_per_sec": (
                round(self.events_processed / self.wall_seconds) if self.wall_seconds else None
            ),
            "signals": len([s for s in self.signals if not s.suppressed]),
            "suppressed": len([s for s in self.signals if s.suppressed]),
            "precision": total.get("precision"),
            "false_alert_rate": total.get("false_alert_rate"),
            "median_lead_s": total.get("median_lead_s"),
            "median_remaining": total.get("median_remaining"),
            "confirm_rate": total.get("confirm_rate"),
            "peak_rss_mb": self.peak_rss_mb,
            "ring_sizes": self.ring_sizes,
            "by_symbol": self.metrics.get("by_symbol", {}),
            "by_direction": self.metrics.get("by_direction", {}),
            "gate_rejections": self.rejections,
        }


class ReplayRunner:
    def __init__(
        self,
        cfg: Config,
        session: SessionInfo,
        symbols: list[str],
        model: OnlineModel | None = None,
        gate: AdaptiveGate | None = None,
        baseline: MinuteBaseline | None = None,
    ) -> None:
        self.cfg = cfg
        self.session = session
        self.result = ReplayResult()
        self.model = model or make_model(cfg.ml.engine)
        self.gate = gate or AdaptiveGate(cfg=cfg.ml)
        self.gate.set_session(session.open_ts, session.close_ts)
        self.baseline = baseline or MinuteBaseline()
        self.metrics = MetricsTracker(session.open_ts, session.close_ts)
        self.labeler = Labeler(cfg.ml, self._on_label)
        self.tfilter = TradeFilter()
        self._virtual_now = 0.0

        hooks = MachineHooks(
            emit=self._emit,
            ml_predict=self._predict,
            gate_multipliers=self.gate.multipliers,
            ml_gate_active=lambda: self.gate.active and self.model.ready(cfg.ml.min_labels),
            prob_gate_min=cfg.ml.prob_gate_min,
            prob_bypass_score=cfg.ml.prob_bypass_score,
            on_signal_fired=self._on_fired,
        )
        limiter = GlobalLimiter(cfg.engine.max_alerts_hour_total, cfg.engine.max_alerts_burst)
        self.context = ContextTracker(list(cfg.data.context_symbols))
        self.engines: dict[str, SymbolEngine] = {
            sym: SymbolEngine(
                sym, cfg, self.baseline, hooks, limiter, lambda: True, context=self.context
            )
            for sym in symbols
        }
        for eng in self.engines.values():
            eng.set_session(session.open_ts)

    # ------------------------------------------------------------------ hooks

    def _emit(self, sig: Signal, kind: str, extras: dict[str, Any]) -> None:
        self.result.messages.append((self._virtual_now, format_message(sig, kind, extras)))
        if kind in ("CONFIRMED", "FAILED"):
            self.metrics.on_resolution(sig)

    def _on_fired(self, sig: Signal) -> None:
        self.result.signals.append(sig)
        self.metrics.on_alert(sig)
        self.labeler.track(sig)

    def _predict(self, features: dict[str, float], direction: int) -> tuple[float | None, str]:
        return self.model.predict(features, direction), self.model.version

    def _on_label(self, r: LabelResult) -> None:
        self.result.labels.append(r)
        self.model.learn(r.signal.features, r.signal.direction, r.label)
        self.gate.on_label(
            LabelOutcome(
                signal_id=r.signal.signal_id,
                symbol=r.signal.symbol,
                direction=r.signal.direction,
                label=r.label,
                was_late=r.was_late,
                ts=r.signal.alert_ts,
            )
        )
        self.metrics.on_label(r)

    # -------------------------------------------------------------------- run

    def run(self, events: Iterable[Event]) -> ReplayResult:
        started = time.perf_counter()
        peak_rss = rss_bytes()
        last_tick = 0
        n = 0
        last_ts = 0.0
        for ev in events:
            if ev.ts < last_ts - 5.0:
                raise AssertionError("replay events out of order (lookahead risk)")
            last_ts = max(last_ts, ev.ts)
            self._virtual_now = ev.ts
            n += 1
            if isinstance(ev, Trade):
                ft = self.tfilter.apply(ev)
                if ft is not None:
                    eng = self.engines.get(ft.symbol)
                    if eng is not None:
                        eng.on_trade(ft)
                    self.context.on_trade(ft)
            else:
                eng = self.engines.get(ev.symbol)
                if eng is not None:
                    eng.on_quote(ev)

            sec = int(ev.ts)
            if sec > last_tick:
                last_tick = sec
                rings = {sym: e.agg.ring for sym, e in self.engines.items()}
                self.context.on_second_tick(float(sec))
                for e in self.engines.values():
                    e.on_second_tick(float(sec))
                self.labeler.on_tick(float(sec), rings)
            if n % 20000 == 0:
                peak_rss = max(peak_rss, rss_bytes())

        rings = {sym: e.agg.ring for sym, e in self.engines.items()}
        self.labeler.flush(rings)
        peak_rss = max(peak_rss, rss_bytes())

        self.result.events_processed = n
        self.result.wall_seconds = time.perf_counter() - started
        self.result.peak_rss_mb = round(peak_rss / 1e6, 1)
        self.result.metrics = self.metrics.summary()
        self.result.ring_sizes = {sym: len(e.agg.ring) for sym, e in self.engines.items()}
        self.result.rejections = {
            sym: dict(e.machine.rejections)
            for sym, e in self.engines.items()
            if e.machine.rejections
        }
        return self.result


def print_report(result: ReplayResult, header: str) -> None:
    import json

    print(f"\n=== {header} ===")
    print(json.dumps(result.report(), indent=2))
    if result.messages:
        print("\n--- notifications (as they would have been sent) ---")
        for ts, msg in result.messages:
            print(f"[{et_hms(ts)} ET] {msg}")
    if result.labels:
        print("\n--- labeled outcomes ---")
        for r in result.labels:
            lead = f"{r.lead_time_s:.0f}s" if r.lead_time_s is not None else "-"
            print(
                f"{r.signal.signal_id}: {'POS' if r.label else 'NEG'} ({r.reason}) "
                f"lead={lead} remaining={r.remaining_frac:.2f} "
                f"mfe={r.mfe:.3f} mae={r.mae:.3f}"
            )
