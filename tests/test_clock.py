"""Market-clock behavior with a stubbed Alpaca REST layer (no network)."""

from __future__ import annotations

from datetime import date

from early_trend_scanner.clock import MarketClock
from early_trend_scanner.timeutil import et_dt, et_time_ts


class StubRest:
    """Mimics AlpacaRest.clock()/calendar() responses."""

    def __init__(self, now_ts: float, is_open: bool, sessions: list[tuple[str, str, str]]):
        self._now = now_ts
        self._open = is_open
        self._sessions = sessions

    async def clock(self):
        from early_trend_scanner.timeutil import rfc3339

        return {
            "timestamp": rfc3339(self._now),
            "is_open": self._open,
            "next_open": rfc3339(self._now + 3600),
            "next_close": rfc3339(self._now + 7 * 3600),
        }

    async def calendar(self, start: str, end: str):
        # reuse the real parsing path
        from early_trend_scanner.data.rest import AlpacaRest

        real = AlpacaRest("k", "s")

        async def fake_get(url, params):
            return [{"date": d, "open": o, "close": c} for d, o, c in self._sessions]

        real._get = fake_get  # type: ignore[method-assign]
        return await real.calendar(start, end)


SESSIONS = [
    ("2026-11-25", "09:30", "16:00"),
    ("2026-11-27", "09:30", "13:00"),  # day after Thanksgiving: early close
    ("2026-11-30", "09:30", "16:00"),
]
# 2026-11-26 (Thanksgiving) intentionally absent — holiday.


async def make_clock(now_ts: float, is_open: bool = True) -> MarketClock:
    clock = MarketClock(StubRest(now_ts, is_open, SESSIONS))  # type: ignore[arg-type]
    await clock.load_sessions()
    return clock


async def test_calendar_parses_eastern_times() -> None:
    clock = await make_clock(et_time_ts(date(2026, 11, 25), 10, 0))
    s = clock.session_for(et_time_ts(date(2026, 11, 25), 10, 0))
    assert s is not None
    assert et_dt(s.open_ts).strftime("%H:%M") == "09:30"
    assert et_dt(s.close_ts).strftime("%H:%M") == "16:00"
    assert s.minutes == 390


async def test_early_close_recognized() -> None:
    ts = et_time_ts(date(2026, 11, 27), 10, 0)
    clock = await make_clock(ts)
    s = clock.session_for(ts)
    assert s is not None
    assert s.minutes == 210  # 09:30-13:00
    assert et_dt(s.close_ts).strftime("%H:%M") == "13:00"


async def test_holiday_has_no_session() -> None:
    ts = et_time_ts(date(2026, 11, 26), 10, 0)  # Thanksgiving
    clock = await make_clock(ts)
    assert clock.session_for(ts) is None


async def test_premarket_maps_to_todays_session() -> None:
    ts = et_time_ts(date(2026, 11, 25), 8, 0)  # 8:00 ET, pre-open
    clock = await make_clock(ts)
    s = clock.session_for(ts)
    assert s is not None and s.date_str == "2026-11-25"


async def test_after_close_has_no_session() -> None:
    ts = et_time_ts(date(2026, 11, 25), 17, 0)
    clock = await make_clock(ts)
    assert clock.session_for(ts) is None


async def test_completed_sessions_before() -> None:
    ts = et_time_ts(date(2026, 11, 30), 9, 0)
    clock = await make_clock(ts)
    done = clock.completed_sessions_before(ts, 5)
    assert [s.date_str for s in done] == ["2026-11-25", "2026-11-27"]
    prior = clock.prior_session(ts)
    assert prior is not None and prior.date_str == "2026-11-27"


async def test_clock_state_fields() -> None:
    now = et_time_ts(date(2026, 11, 25), 10, 0)
    clock = MarketClock(StubRest(now, True, SESSIONS))  # type: ignore[arg-type]
    state = await clock.fetch()
    assert state.is_open is True
    assert state.next_open_ts > state.now_ts
    assert abs(state.now_ts - now) < 1e-6
