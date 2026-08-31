from datetime import date

from early_trend_scanner.timeutil import (
    ET,
    et_dt,
    et_time_ts,
    minute_of_session,
    parse_rfc3339,
    rfc3339,
)


def test_parse_nanosecond_timestamp() -> None:
    ts = parse_rfc3339("2026-01-02T14:30:00.123456789Z")
    assert abs(ts - (parse_rfc3339("2026-01-02T14:30:00Z") + 0.123456)) < 1e-6


def test_parse_variants() -> None:
    base = parse_rfc3339("2026-01-02T14:30:00Z")
    assert parse_rfc3339("2026-01-02T14:30:00+00:00") == base
    assert parse_rfc3339("2026-01-02T09:30:00-05:00") == base
    assert parse_rfc3339("2026-01-02T14:30:00.5Z") == base + 0.5


def test_roundtrip() -> None:
    ts = 1_760_000_000.123456
    assert abs(parse_rfc3339(rfc3339(ts)) - ts) < 1e-6


def test_dst_boundaries() -> None:
    # US DST 2026: begins Mar 8, ends Nov 1.
    winter = et_time_ts(date(2026, 1, 15), 9, 30)
    summer = et_time_ts(date(2026, 6, 15), 9, 30)
    assert et_dt(winter).strftime("%H:%M") == "09:30"
    assert et_dt(summer).strftime("%H:%M") == "09:30"
    assert et_dt(winter).utcoffset().total_seconds() == -5 * 3600  # type: ignore[union-attr]
    assert et_dt(summer).utcoffset().total_seconds() == -4 * 3600  # type: ignore[union-attr]


def test_minute_of_session() -> None:
    open_ts = et_time_ts(date(2026, 6, 15), 9, 30)
    assert minute_of_session(open_ts, open_ts) == 0
    assert minute_of_session(open_ts + 59.9, open_ts) == 0
    assert minute_of_session(open_ts + 60.0, open_ts) == 1
    assert minute_of_session(open_ts + 389 * 60, open_ts) == 389
    assert ET.key == "America/New_York"
