"""Time helpers: RFC-3339 parsing (nanosecond tolerant) and Eastern Time conversions."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = UTC


def parse_rfc3339(ts: str) -> float:
    """Parse an RFC-3339 timestamp (SIP feeds use nanosecond precision) to epoch seconds.

    datetime.fromisoformat rejects more than 6 fractional digits, so the
    fractional part is truncated to microseconds before parsing.
    """
    s = ts
    if s.endswith(("Z", "z")):
        s = s[:-1] + "+00:00"
    dot = s.find(".")
    if dot != -1:
        tzpos = len(s)
        for i in range(dot + 1, len(s)):
            if s[i] in "+-":
                tzpos = i
                break
        frac = s[dot + 1 : tzpos]
        if len(frac) > 6:
            frac = frac[:6]
        s = f"{s[: dot + 1]}{frac}{s[tzpos:]}"
    return datetime.fromisoformat(s).timestamp()


def rfc3339(ts: float) -> str:
    """Epoch seconds -> RFC-3339 UTC string accepted by Alpaca REST."""
    return (
        datetime.fromtimestamp(ts, UTC).replace(tzinfo=None).isoformat(timespec="microseconds")
        + "Z"
    )


def et_dt(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, ET)


def et_hms(ts: float) -> str:
    return et_dt(ts).strftime("%H:%M:%S")


def et_date(ts: float) -> date:
    return et_dt(ts).date()


def et_time_ts(d: date, hour: int, minute: int) -> float:
    """Epoch seconds of an Eastern wall-clock time on a given date (DST safe)."""
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=ET).timestamp()


def minute_of_session(ts: float, session_open_ts: float) -> int:
    """0-based minute index since the session open (0..389 on a full session)."""
    return int((ts - session_open_ts) // 60.0)


def prior_days(d: date, n: int) -> date:
    return d - timedelta(days=n)
