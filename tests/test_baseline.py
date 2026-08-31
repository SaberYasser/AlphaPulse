from early_trend_scanner.data.models import Bar, SessionInfo
from early_trend_scanner.engine.baseline import MinuteBaseline

OPEN = 1_756_000_000.0 - (1_756_000_000.0 % 60)


def session(day: int) -> SessionInfo:
    o = OPEN + day * 86_400
    return SessionInfo(f"d{day}", o, o + 390 * 60)


def bars_for(sess: SessionInfo, vol: int) -> list[Bar]:
    return [
        Bar(
            ts=sess.open_ts + i * 60,
            open=10,
            high=10,
            low=10,
            close=10,
            volume=vol + i,
            trade_count=50,
        )
        for i in range(390)
    ]


def test_median_across_sessions() -> None:
    sessions = [session(i) for i in range(5)]
    bars = {"X": []}
    for k, s in enumerate(sessions):
        bars["X"].extend(bars_for(s, vol=1000 * (k + 1)))
    b = MinuteBaseline()
    b.build(bars, sessions)
    assert b.sessions_used == 5
    # minute 0 volumes: 1000..5000 -> median 3000
    assert b.minute_volume("X", 0) == 3000
    assert b.minute_volume("X", 10) == 3010
    assert b.vol_per_5s("X", 0) == 3000 / 12
    assert b.minute_trades("X", 5) == 50
    assert b.ready("X")


def test_fallback_for_missing_minutes() -> None:
    s = session(0)
    bars = {"X": bars_for(s, vol=1200)[:100]}  # only first 100 minutes present
    b = MinuteBaseline()
    b.build(bars, [s])
    assert b.minute_volume("X", 350) > 0  # falls back to session median
    assert b.minute_volume("UNKNOWN", 5) == 0.0
    assert not b.ready("UNKNOWN")


def test_out_of_session_bars_ignored() -> None:
    s = session(0)
    bars = {"X": bars_for(s, vol=1000)}
    bars["X"].append(Bar(ts=s.close_ts + 3600, open=1, high=1, low=1, close=1, volume=9_999_999))
    b = MinuteBaseline()
    b.build(bars, [s])
    assert b.minute_volume("X", 0) == 1000


def test_midday_ruler_ignores_opening_flood() -> None:
    from early_trend_scanner.data.models import Bar, SessionInfo

    open_ts = 1_756_000_020.0
    session = SessionInfo("d", open_ts, open_ts + 390 * 60)
    bars = []
    for m in range(390):
        vol = 1_000_000 if m == 0 else (12_000 if 30 <= m < 360 else 30_000)
        bars.append(
            Bar(ts=open_ts + m * 60, open=1, high=1, low=1, close=1, volume=vol, trade_count=10)
        )
    b = MinuteBaseline()
    b.build({"X": bars}, [session])
    # midday ruler reflects typical tape (~1000/5s), not the auction minute
    assert abs(b.midday_vol_per_5s("X") - 1000.0) < 1.0
    assert b.vol_per_5s("X", 0) > 80_000  # the self-defeating opening figure
