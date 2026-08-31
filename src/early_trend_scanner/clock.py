"""Market session gating built on the Alpaca clock + calendar (never weekday math).

Handles holidays, early closes and DST because the calendar reports each
session's actual open/close in Eastern wall-clock time.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta

from .data.models import SessionInfo
from .data.rest import AlpacaRest
from .timeutil import parse_rfc3339

log = logging.getLogger(__name__)


@dataclass
class ClockState:
    now_ts: float
    is_open: bool
    next_open_ts: float
    next_close_ts: float
    skew_s: float  # server minus local, informational


class MarketClock:
    def __init__(self, rest: AlpacaRest) -> None:
        self._rest = rest
        self._sessions: list[SessionInfo] = []

    async def fetch(self) -> ClockState:
        raw = await self._rest.clock()
        server_ts = parse_rfc3339(raw["timestamp"])
        return ClockState(
            now_ts=server_ts,
            is_open=bool(raw["is_open"]),
            next_open_ts=parse_rfc3339(raw["next_open"]),
            next_close_ts=parse_rfc3339(raw["next_close"]),
            skew_s=server_ts - time.time(),
        )

    async def load_sessions(self, back_days: int = 21, fwd_days: int = 7) -> list[SessionInfo]:
        today = date.today()
        start = (today - timedelta(days=back_days)).isoformat()
        end = (today + timedelta(days=fwd_days)).isoformat()
        self._sessions = await self._rest.calendar(start, end)
        return self._sessions

    def session_for(self, ts: float) -> SessionInfo | None:
        """The session whose [open-6h, close] window contains ts (pre-open included)."""
        for s in self._sessions:
            if s.open_ts - 6 * 3600 <= ts <= s.close_ts:
                return s
        return None

    def completed_sessions_before(self, ts: float, n: int) -> list[SessionInfo]:
        done = [s for s in self._sessions if s.close_ts < ts]
        return done[-n:]

    def prior_session(self, ts: float) -> SessionInfo | None:
        done = [s for s in self._sessions if s.close_ts < ts]
        return done[-1] if done else None
