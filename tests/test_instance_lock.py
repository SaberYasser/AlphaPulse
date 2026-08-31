"""Single-instance lock: a second scanner must refuse to start."""

from __future__ import annotations

import socket
from dataclasses import replace

from early_trend_scanner.app import ScannerApp
from early_trend_scanner.config import Config, Secrets


def _app(cfg: Config) -> ScannerApp:
    return ScannerApp(cfg, Secrets())


def test_lock_acquired_and_blocks_second_instance(cfg: Config) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    cfg2 = replace(cfg, storage=replace(cfg.storage, instance_lock_port=port))
    first = _app(cfg2)
    assert first._acquire_instance_lock() is True
    second = _app(cfg2)
    assert second._acquire_instance_lock() is False
    first._lock_sock.close()
    first.store.close()
    second.store.close()


def test_lock_disabled_with_port_zero(cfg: Config) -> None:
    cfg2 = replace(cfg, storage=replace(cfg.storage, instance_lock_port=0))
    app = _app(cfg2)
    assert app._acquire_instance_lock() is True
    app.store.close()
