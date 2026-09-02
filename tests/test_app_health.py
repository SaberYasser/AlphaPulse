"""Application-level health reporting and background-task supervision."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from types import SimpleNamespace
from typing import Any, cast

import pytest

from early_trend_scanner.app import ScannerApp
from early_trend_scanner.config import Config, Secrets
from early_trend_scanner.status import StatusWriter


def _app(cfg: Config) -> ScannerApp:
    return ScannerApp(cfg, Secrets(telegram_token="token", telegram_chat_id="chat"))


def test_status_payload_reports_runtime_identity_and_alert_health(cfg: Config) -> None:
    app = _app(cfg)
    now = time.monotonic()
    app.stream = cast(
        Any,
        SimpleNamespace(
            connected=True,
            last_rx_mono=now,
            latency_ewma_s=0.25,
            dropped_trades=0,
            dropped_quotes=0,
        ),
    )

    payload = app.status_payload()

    assert payload["process_id"] == os.getpid()
    assert payload["project_root"] == str(cfg.root.resolve())
    assert payload["symbols"] == cfg.symbols
    assert payload["context_symbols"] == list(cfg.data.context_symbols)
    assert payload["alerts_enabled"] is True
    assert payload["last_rx_age_s"] < 1.0
    app.store.close()


def test_status_payload_disables_alerts_for_stale_stream(cfg: Config) -> None:
    app = _app(cfg)
    app.stream = cast(
        Any,
        SimpleNamespace(
            connected=True,
            last_rx_mono=time.monotonic() - 4.0,
            latency_ewma_s=0.25,
            dropped_trades=0,
            dropped_quotes=0,
        ),
    )

    payload = app.status_payload()

    assert payload["alerts_enabled"] is False
    assert payload["last_rx_age_s"] >= 3.0
    app.store.close()


@pytest.mark.asyncio
async def test_background_task_failure_is_logged(
    cfg: Config, caplog: pytest.LogCaptureFixture
) -> None:
    app = _app(cfg)

    async def fail() -> None:
        raise RuntimeError("write failed")

    with caplog.at_level(logging.ERROR, logger="early_trend_scanner.app"):
        app._bg(fail())
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    assert "background task failed" in caplog.text
    assert not app._bg_tasks
    app.store.close()


def test_status_writer_round_trip_is_atomic(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "status.json"
    writer = StatusWriter(path)
    monkeypatch.setattr(time, "time", lambda: 123.5)

    writer.write({"state": "scanning"})

    assert StatusWriter.read(path) == {"written_at": 123.5, "state": "scanning"}
    assert not path.with_suffix(".tmp").exists()
    path.write_text("not json", encoding="utf-8")
    assert StatusWriter.read(path) is None
