"""Live scanner orchestration: session gating, stream, engines, learning, teardown.

One process handles one or more sessions (after_close: exit | pause). Regular
trading hours only — outside them there is no scanning, no training and no
notification traffic.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from . import __version__
from .clock import ClockState, MarketClock
from .config import Config, Secrets
from .data.filters import TradeFilter
from .data.models import Quote, SessionInfo, Trade
from .data.rest import AlpacaRest
from .data.stream import AlpacaStream, StreamAuthError
from .engine.baseline import MinuteBaseline
from .engine.context import ContextTracker
from .engine.levels import LevelKind
from .engine.state import GlobalLimiter, MachineHooks, Signal
from .engine.symbol_engine import SymbolEngine
from .ml.gate import AdaptiveGate, LabelOutcome
from .ml.labeler import Labeler, LabelResult
from .ml.online import OnlineModel, make_model
from .notify.telegram import TelegramNotifier
from .power import KeepAwake
from .status import StatusWriter, rss_bytes
from .store.db import SignalStore
from .store.metrics import MetricsTracker
from .timeutil import et_date
from .warmup import warmup

log = logging.getLogger(__name__)


class ScannerApp:
    def __init__(self, cfg: Config, secrets: Secrets) -> None:
        self.cfg = cfg
        self.secrets = secrets
        self.keepawake = KeepAwake()
        self.status = StatusWriter(cfg.path(cfg.storage.status_path))
        self.store = SignalStore(cfg.path(cfg.storage.db_path), cfg.storage.retention_days)
        self.model = self._load_model()
        self.gate = self._load_gate()
        self.notifier = TelegramNotifier(
            token=secrets.telegram_token,
            chat_id=secrets.telegram_chat_id,
            max_retries=cfg.telegram.max_retries,
            send_timeout_s=cfg.telegram.send_timeout_s,
            dedupe_size=cfg.telegram.dedupe_size,
            prefix="DEMO " if cfg.data.demo_mode else "",
        )
        self._bg_tasks: set[asyncio.Task[Any]] = set()
        self._stopping = asyncio.Event()
        self._skew = 0.0  # server-minus-local clock skew, refreshed per session

        # per-session objects, rebuilt each session
        self.engines: dict[str, SymbolEngine] = {}
        self.metrics = MetricsTracker()
        self.labeler: Labeler | None = None
        self.stream: AlpacaStream | None = None
        self.baseline = MinuteBaseline()
        self.context = ContextTracker(list(cfg.data.context_symbols))
        self.tfilter = TradeFilter()
        self.session: SessionInfo | None = None
        self._consumers_running = asyncio.Event()
        self._consumers_running.set()

    # ------------------------------------------------------------- persistence

    def _model_path(self) -> Path:
        d = self.cfg.path(self.cfg.storage.model_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d / "online_model.pkl"

    def _load_model(self) -> OnlineModel:
        path = self._model_path()
        if path.exists():
            try:
                model = OnlineModel.from_bytes(path.read_bytes())
                log.info("loaded model %s", model.version)
                return model
            except Exception as e:
                log.warning("could not load persisted model (%s) — starting fresh", e)
        return make_model(self.cfg.ml.engine)

    def _load_gate(self) -> AdaptiveGate:
        raw = None
        try:
            raw = self.store.get_meta("adaptive_gate")
        except Exception:
            pass
        if raw:
            try:
                return AdaptiveGate.from_json(self.cfg.ml, raw)
            except Exception as e:
                log.warning("could not load gate state (%s) — starting fresh", e)
        return AdaptiveGate(cfg=self.cfg.ml)

    def _persist_learning_state(self) -> None:
        try:
            self._model_path().write_bytes(self.model.to_bytes())
            self.store.set_meta("adaptive_gate", self.gate.to_json())
            self.store.set_meta("model_version", self.model.version)
            log.info("persisted model %s and gate v%d", self.model.version, self.gate.version)
        except Exception:
            log.exception("failed to persist learning state")

    def _bg(self, coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    # ------------------------------------------------------------------- hooks

    def _build_hooks(self) -> MachineHooks:
        def emit(sig: Signal, kind: str, extras: dict[str, Any]) -> None:
            # Telegram first — nothing may run before the enqueue.
            self.notifier.publish_signal(sig, kind, extras)
            if kind in ("CONFIRMED", "FAILED"):
                self.metrics.on_resolution(sig)
                self._bg(asyncio.to_thread(self.store.record_resolution, sig))

        def on_signal_fired(sig: Signal) -> None:
            self.metrics.on_alert(sig)
            if self.labeler is not None:
                self.labeler.track(sig)
            date_str = et_date(sig.alert_ts).isoformat()
            self._bg(asyncio.to_thread(self.store.record_signal, sig, date_str, self.gate.version))
            log.info(
                "SIGNAL %s %s @%.2f trigger=%.2f inv=%.2f vol=%.1fx score=%.2f p=%s%s",
                sig.dir_str,
                sig.symbol,
                sig.alert_price,
                sig.trigger_price,
                sig.invalidation,
                sig.vol_ratio,
                sig.score,
                f"{sig.prob:.2f}" if sig.prob is not None else "-",
                " [suppressed]" if sig.suppressed else "",
            )

        def on_signal_final(sig: Signal) -> None:
            if sig.suppressed:
                self._bg(asyncio.to_thread(self.store.record_resolution, sig))

        def ml_predict(features: dict[str, float], direction: int) -> tuple[float | None, str]:
            try:
                return self.model.predict(features, direction), self.model.version
            except Exception:
                log.exception("model predict failed")
                return None, self.model.version

        return MachineHooks(
            emit=emit,
            ml_predict=ml_predict,
            gate_multipliers=self.gate.multipliers,
            ml_gate_active=lambda: self.gate.active and self.model.ready(self.cfg.ml.min_labels),
            prob_gate_min=self.cfg.ml.prob_gate_min,
            prob_bypass_score=self.cfg.ml.prob_bypass_score,
            on_signal_fired=on_signal_fired,
            on_signal_final=on_signal_final,
        )

    def _on_label(self, r: LabelResult) -> None:
        sig = r.signal
        try:
            self.model.learn(sig.features, sig.direction, r.label)
        except Exception:
            log.exception("model learn failed")
        self.gate.on_label(
            LabelOutcome(
                signal_id=sig.signal_id,
                symbol=sig.symbol,
                direction=sig.direction,
                label=r.label,
                was_late=r.was_late,
                ts=sig.alert_ts,
            )
        )
        self.metrics.on_label(r)
        self._bg(asyncio.to_thread(self.store.record_label, r))
        log.info(
            "LABEL %s -> %s (%s) lead=%s remaining=%.2f",
            sig.signal_id,
            "POS" if r.label else "NEG",
            r.reason,
            f"{r.lead_time_s:.0f}s" if r.lead_time_s is not None else "-",
            r.remaining_frac,
        )

    # ------------------------------------------------------------------ health

    def _alerts_enabled(self) -> bool:
        s = self.stream
        if s is None or not s.connected:
            return False
        if time.monotonic() - s.last_rx_mono > 3.0:
            return False
        return s.latency_ewma_s <= self.cfg.data.max_feed_latency_s

    # --------------------------------------------------------------------- run

    def _acquire_instance_lock(self) -> bool:
        """Bind a localhost port for the process lifetime (single instance)."""
        port = self.cfg.storage.instance_lock_port
        if port <= 0:
            return True
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            sock.close()
            return False
        self._lock_sock = sock  # held (and OS-released on exit/crash)
        return True

    async def run(self) -> int:
        if not self._acquire_instance_lock():
            log.error(
                "another scanner instance is already running "
                "(lock port %d busy) — refusing to start a duplicate",
                self.cfg.storage.instance_lock_port,
            )
            return 3
        creds_ok = bool(self.secrets.alpaca_key and self.secrets.alpaca_secret)
        if not creds_ok:
            log.error("APCA_API_KEY_ID / APCA_API_SECRET_KEY are not set")
            return 2
        if not self.notifier.enabled:
            log.warning("Telegram not configured — running in shadow mode (log-only alerts)")

        await self.notifier.start()
        try:
            async with AlpacaRest(
                self.secrets.alpaca_key,
                self.secrets.alpaca_secret,
                self.cfg.data.data_base_url,
                self.cfg.data.trading_base_url,
            ) as rest:
                clock = MarketClock(rest)
                while not self._stopping.is_set():
                    ran = await self._run_one_session(rest, clock)
                    if self.cfg.session.after_close == "exit" or self._stopping.is_set():
                        break
                    if not ran:
                        await self._pause_until_next_open(clock)
            return 0
        finally:
            self.keepawake.release()
            await self.notifier.stop()
            self._persist_learning_state()
            self.store.close()
            self.status.write({"state": "stopped", "version": __version__})

    async def _pause_until_next_open(self, clock: MarketClock) -> None:
        state = await clock.fetch()
        wake = state.next_open_ts - self.cfg.session.start_lead_min * 60
        wait_s = max(30.0, wake - time.time())
        log.info("paused (zero-work) for %.0f min until next session", wait_s / 60)
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=wait_s)
        except TimeoutError:
            pass

    async def _run_one_session(self, rest: AlpacaRest, clock: MarketClock) -> bool:
        """Returns True if a session was actually scanned."""
        state = await clock.fetch()
        await clock.load_sessions()
        session = clock.session_for(state.now_ts)
        self._skew = state.skew_s
        now = time.time() + state.skew_s

        if session is None or now >= session.close_ts:
            nxt = state.next_open_ts
            log.info(
                "market closed (holiday/weekend or after close); next open in %.1f h",
                max(0.0, nxt - now) / 3600,
            )
            self.status.write({"state": "market_closed", "next_open_ts": nxt})
            return False

        self.session = session
        log.info(
            "session %s: open in %.1f min, close in %.1f min",
            session.date_str,
            (session.open_ts - now) / 60,
            (session.close_ts - now) / 60,
        )

        # Operational window begins: prevent idle sleep until teardown.
        self.keepawake.acquire()
        self._build_session_objects(session)
        self.status.write({"state": "warmup", "session": session.date_str})
        await warmup(rest, clock, self.cfg, self.engines, self.baseline, session, now)

        # Wait for the opening bell (stream connects ~2 min early for continuity).
        pre_connect = max(session.open_ts - 120.0, now)
        await self._sleep_until(pre_connect, state)
        if self._stopping.is_set():
            return True
        if session.open_ts - (time.time() + state.skew_s) > 30.0:
            await self._refresh_premarket(rest, session)

        close_reason = "close"
        attempt = 0
        tasks: list[asyncio.Task[Any]] = []
        try:
            while True:
                tasks = self._spawn_session_tasks(rest, clock)
                try:
                    await self._sleep_until(session.close_ts, state, tasks)
                    break
                except StreamDied as e:
                    attempt += 1
                    gap_start = self.stream.last_event_ts if self.stream else 0.0
                    await self._cancel_tasks(tasks)
                    tasks = []
                    fatal_auth = isinstance(e.__cause__, StreamAuthError)
                    remaining = session.close_ts - (time.time() + state.skew_s)
                    max_restarts = self.cfg.data.stream_restart_max
                    if fatal_auth or attempt > max_restarts or remaining < 120:
                        close_reason = f"fatal: {e}"
                        log.error("session aborted: %s", e)
                        break
                    log.error(
                        "stream died (%s) — restarting session tasks (attempt %d/%d)",
                        e,
                        attempt,
                        max_restarts,
                    )
                    await asyncio.sleep(min(30.0, 10.0 * attempt))
                    if gap_start > 0:
                        self._bg(self._recover(rest, gap_start, time.time()))
        finally:
            log.info("session teardown (%s)", close_reason)
            await self._teardown_session(tasks)
        return True

    def _spawn_session_tasks(self, rest: AlpacaRest, clock: MarketClock) -> list[asyncio.Task[Any]]:
        trade_q: asyncio.Queue[Trade] = asyncio.Queue(maxsize=self.cfg.data.queue_trades)
        quote_q: asyncio.Queue[Quote] = asyncio.Queue(maxsize=self.cfg.data.queue_quotes)
        self.stream = AlpacaStream(
            key=self.secrets.alpaca_key,
            secret=self.secrets.alpaca_secret,
            feed=self.cfg.data.feed,
            symbols=self.cfg.symbols + list(self.cfg.data.context_symbols),
            quote_symbols=self.cfg.symbols,
            trade_queue=trade_q,
            quote_queue=quote_q,
            stream_base_url=self.cfg.data.stream_base_url,
            gap_reconnect_s=self.cfg.data.gap_reconnect_s,
            on_gap=self._make_gap_handler(rest),
        )
        return [
            asyncio.create_task(self.stream.run(), name="stream"),
            asyncio.create_task(self._trade_consumer(trade_q), name="trades"),
            asyncio.create_task(self._quote_consumer(quote_q), name="quotes"),
            asyncio.create_task(self._ticker(), name="ticker"),
            asyncio.create_task(self._status_loop(), name="status"),
            asyncio.create_task(self._checkpoint_loop(), name="checkpoint"),
            asyncio.create_task(self._clock_refresh(clock), name="clockref"),
        ]

    async def _cancel_tasks(self, tasks: list[asyncio.Task[Any]]) -> None:
        if self.stream is not None:
            self.stream.stop()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _build_session_objects(self, session: SessionInfo) -> None:
        self.metrics = MetricsTracker(session.open_ts, session.close_ts)
        self.gate.set_session(session.open_ts, session.close_ts)
        self.labeler = Labeler(self.cfg.ml, self._on_label)
        self.tfilter = TradeFilter()
        self.baseline = MinuteBaseline()
        self.context = ContextTracker(list(self.cfg.data.context_symbols))
        hooks = self._build_hooks()
        limiter = GlobalLimiter(
            self.cfg.engine.max_alerts_hour_total, self.cfg.engine.max_alerts_burst
        )
        self.engines = {
            sym: SymbolEngine(
                sym,
                self.cfg,
                self.baseline,
                hooks,
                limiter,
                self._alerts_enabled,
                context=self.context,
            )
            for sym in self.cfg.symbols
        }

    async def _sleep_until(
        self, target_ts: float, state: ClockState, tasks: list[asyncio.Task[Any]] | None = None
    ) -> None:
        """Sleep in short slices; fail fast if a critical task dies."""
        while not self._stopping.is_set():
            now = time.time() + state.skew_s
            remaining = target_ts - now
            if remaining <= 0:
                return
            if tasks:
                for t in tasks:
                    if t.done() and not t.cancelled():
                        exc = t.exception()
                        if exc is not None:
                            raise StreamDied(str(exc)) from exc
                        if t.get_name() == "stream":
                            raise StreamDied("stream task exited")
            try:
                await asyncio.wait_for(self._stopping.wait(), timeout=min(remaining, 5.0))
                return
            except TimeoutError:
                continue

    async def _teardown_session(self, tasks: list[asyncio.Task[Any]]) -> None:
        await self._cancel_tasks(tasks)
        if self._bg_tasks:
            await asyncio.gather(*list(self._bg_tasks), return_exceptions=True)

        if self.labeler is not None:
            rings = {sym: e.agg.ring for sym, e in self.engines.items()}
            resolved = self.labeler.flush(rings)
            log.info("labeler flush: %d pending resolved at close", resolved)

        summary = self.metrics.summary()
        date_str = self.session.date_str if self.session else "unknown"
        try:
            await asyncio.to_thread(self.store.record_daily_metrics, date_str, "_ALL", summary)
            for sym, eng in self.engines.items():
                await asyncio.to_thread(
                    self.store.record_daily_metrics,
                    date_str,
                    sym,
                    {
                        "alerts": eng.machine.alerts_today,
                        "suppressed": eng.machine.suppressed_today,
                        "rejections": eng.machine.rejections,
                    },
                )
        except Exception:
            log.exception("failed writing daily metrics")
        self._persist_learning_state()
        self.keepawake.release()
        self.stream = None
        self.status.write({"state": "closed", "summary": summary.get("total", {})})
        log.info("daily summary: %s", summary.get("total"))

    # ------------------------------------------------------------------ loops

    async def _trade_consumer(self, q: asyncio.Queue[Trade]) -> None:
        while True:
            t = await q.get()
            await self._consumers_running.wait()
            ft = self.tfilter.apply(t)
            if ft is None:
                continue
            eng = self.engines.get(ft.symbol)
            if eng is not None:
                try:
                    eng.on_trade(ft)
                except Exception:
                    log.exception("engine error on trade %s", ft.symbol)
            else:
                self.context.on_trade(ft)

    async def _quote_consumer(self, q: asyncio.Queue[Quote]) -> None:
        while True:
            quote = await q.get()
            await self._consumers_running.wait()
            eng = self.engines.get(quote.symbol)
            if eng is not None:
                eng.on_quote(quote)

    async def _ticker(self) -> None:
        """1 Hz upkeep: roll quiet symbols, follow-up deadlines, labeling."""
        while True:
            await asyncio.sleep(1.0)
            now = time.time() + self._skew
            self.context.on_second_tick(now)
            for eng in self.engines.values():
                try:
                    eng.on_second_tick(now)
                except Exception:
                    log.exception("tick error %s", eng.symbol)
            if self.labeler is not None:
                rings = {sym: e.agg.ring for sym, e in self.engines.items()}
                self.labeler.on_tick(now, rings)

    async def _status_loop(self) -> None:
        while True:
            await asyncio.sleep(15.0)
            self.status.write(self.status_payload())
            self.keepawake.acquire()  # refresh the execution-state flag

    async def _checkpoint_loop(self) -> None:
        interval = max(self.cfg.ml.checkpoint_min, 1) * 60
        while True:
            await asyncio.sleep(interval)
            await asyncio.to_thread(self._persist_learning_state)

    async def _clock_refresh(self, clock: MarketClock) -> None:
        while True:
            await asyncio.sleep(300.0)
            try:
                state = await clock.fetch()
                if self.session and not state.is_open and time.time() > self.session.open_ts:
                    # Unexpected mid-day close per the authoritative clock (halt of
                    # the whole market is extremely rare; trust the calendar).
                    log.warning("clock reports market closed before calendar close")
            except Exception as e:
                log.warning("clock refresh failed: %s", e)

    async def _refresh_premarket(self, rest: AlpacaRest, session: SessionInfo) -> None:
        """Warmup ran minutes before the bell; refresh the premarket extremes."""
        try:
            pm_bars = await rest.bars(
                self.cfg.symbols,
                "1Min",
                session.open_ts - 5.5 * 3600,
                session.open_ts,
                self.cfg.data.feed,
            )
        except Exception as e:
            log.warning("premarket refresh failed (%s) — keeping warmup levels", e)
            return
        for sym, bars in pm_bars.items():
            eng = self.engines.get(sym)
            if eng is None or not bars:
                continue
            book = eng.book
            static = {lv.kind: lv.price for lv in book.static_levels}
            book.set_static(
                pdh=static.get(LevelKind.PDH, 0.0),
                pdl=static.get(LevelKind.PDL, 0.0),
                pdc=static.get(LevelKind.PDC, 0.0),
                pmh=max(b.high for b in bars),
                pml=min(b.low for b in bars),
            )

    # --------------------------------------------------------------- recovery

    def _make_gap_handler(self, rest: AlpacaRest) -> Any:
        async def on_gap(gap_start: float, now: float) -> None:
            # Run the backfill in the background so the stream reconnects
            # immediately; consumers are paused while it replays.
            self._bg(self._recover(rest, gap_start, now))

        return on_gap

    async def _recover(self, rest: AlpacaRest, gap_start: float, now: float) -> None:
        gap = now - gap_start
        if gap < self.cfg.data.recovery_min_gap_s:
            return
        log.info("recovering %.1fs gap via REST backfill", gap)
        for eng in self.engines.values():
            eng.catchup = True
        self._consumers_running.clear()
        try:
            all_syms = self.cfg.symbols + list(self.cfg.data.context_symbols)
            trades = await rest.trades(all_syms, gap_start, now, self.cfg.data.feed)
            merged: list[Trade] = []
            for rows in trades.values():
                merged.extend(rows)
            merged.sort(key=lambda t: t.ts)
            for t in merged:
                ft = self.tfilter.apply(t)
                if ft is None:
                    continue
                target = self.engines.get(ft.symbol)
                if target is not None:
                    target.on_trade(ft)
                else:
                    self.context.on_trade(ft)
            log.info("backfill complete: %d trades", len(merged))
        except Exception:
            log.exception("gap recovery failed — continuing with live data only")
        finally:
            self._consumers_running.set()
            for eng in self.engines.values():
                eng.catchup = False

    # ----------------------------------------------------------------- status

    def status_payload(self) -> dict[str, Any]:
        s = self.stream
        alerts = sum(e.machine.alerts_today for e in self.engines.values())
        suppressed = sum(e.machine.suppressed_today for e in self.engines.values())
        return {
            "state": "scanning",
            "version": __version__,
            "session": self.session.date_str if self.session else None,
            "feed": self.cfg.data.feed,
            "demo_mode": self.cfg.data.demo_mode,
            "ws_connected": bool(s and s.connected),
            "feed_latency_s": round(s.latency_ewma_s, 3) if s else None,
            "dropped_trades": s.dropped_trades if s else 0,
            "dropped_quotes": s.dropped_quotes if s else 0,
            "alerts_today": alerts,
            "suppressed_today": suppressed,
            "labels_pending": self.labeler.pending_count if self.labeler else 0,
            "model_version": self.model.version,
            "gate_active": self.gate.active,
            "gate_version": self.gate.version,
            "keepawake": self.keepawake.active,
            "telegram_enabled": self.notifier.enabled,
            "telegram_sent": self.notifier.sent_count,
            "telegram_failed": self.notifier.failed_count,
            "filter": vars(self.tfilter.stats),
            "rss_mb": round(rss_bytes() / 1e6, 1),
        }

    def request_stop(self) -> None:
        self._stopping.set()


class StreamDied(RuntimeError):
    pass
