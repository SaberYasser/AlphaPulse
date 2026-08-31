"""Windows sleep prevention via SetThreadExecutionState.

ES_SYSTEM_REQUIRED keeps the machine from idling into sleep while the market is
open. It does NOT block a deliberate user action: manual sleep, shutdown and
lid-close policies still apply. The flag is cleared at market close.
"""

from __future__ import annotations

import ctypes
import logging
import sys

log = logging.getLogger(__name__)

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


class KeepAwake:
    def __init__(self) -> None:
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def _set(self, flags: int) -> int:
        if sys.platform != "win32":  # pragma: no cover - non-Windows dev machines
            return 1
        return int(ctypes.windll.kernel32.SetThreadExecutionState(flags))  # type: ignore[attr-defined]

    def acquire(self) -> None:
        """Idempotent; also called periodically as a refresh."""
        prev = self._set(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        if prev == 0:
            log.warning("SetThreadExecutionState(acquire) failed")
            return
        if not self._active:
            log.info("sleep prevention ON (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)")
        self._active = True

    def release(self) -> None:
        if not self._active:
            return
        prev = self._set(ES_CONTINUOUS)
        if prev == 0:
            log.warning("SetThreadExecutionState(release) failed")
        self._active = False
        log.info("sleep prevention OFF (ES_CONTINUOUS)")
