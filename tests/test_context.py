"""Market-regime context tracker: velocities, alignment, staleness, neutrality."""

from __future__ import annotations

from conftest import BASE_TS
from early_trend_scanner.data.models import Trade
from early_trend_scanner.engine.context import ContextTracker


def tr(sym: str, ts: float, price: float, size: int = 100) -> Trade:
    return Trade(symbol=sym, ts=ts, price=price, size=size, conditions=("@",))


def steady_tape(ctx: ContextTracker, sym: str, start: float, seconds: int, p0: float, slope: float):
    for s in range(seconds):
        ctx.on_trade(tr(sym, start + s + 0.5, p0 + slope * s))
    ctx.on_second_tick(start + seconds)


def test_velocity_and_alignment() -> None:
    ctx = ContextTracker(["SPY", "VXX"])
    # SPY rising 0.05/s from 500 -> ~1 bps/s; VXX falling
    steady_tape(ctx, "SPY", BASE_TS, 400, 500.0, 0.05)
    steady_tape(ctx, "VXX", BASE_TS, 400, 50.0, -0.005)
    now = BASE_TS + 400
    up1, up5, fear1, fear5 = ctx.features(now, direction=1)
    dn1, dn5, fear1b, fear5b = ctx.features(now, direction=-1)
    assert up1 > 0.5 and up5 > 0.5  # market moving WITH an UP candidate
    assert dn1 < -0.5 and dn5 < -0.5  # and AGAINST a DOWN candidate
    assert up1 == -dn1 and up5 == -dn5
    assert fear5 < -0.5 and fear5 == fear5b  # fear proxy unaligned
    assert fear1 < 0.0 and fear1 == fear1b


def test_stale_context_is_neutral() -> None:
    ctx = ContextTracker(["SPY", "VXX"], stale_after_s=45.0)
    steady_tape(ctx, "SPY", BASE_TS, 120, 500.0, 0.05)
    fresh = ctx.features(BASE_TS + 120, direction=1)
    assert fresh[0] != 0.0
    stale = ctx.features(BASE_TS + 120 + 300, direction=1)
    assert stale == (0.0, 0.0, 0.0, 0.0)


def test_missing_symbols_are_neutral() -> None:
    ctx = ContextTracker([])
    assert ctx.features(BASE_TS, 1) == (0.0, 0.0, 0.0, 0.0)
    ctx1 = ContextTracker(["SPY"])  # no vol proxy configured
    steady_tape(ctx1, "SPY", BASE_TS, 90, 500.0, 0.05)
    m1, _m5, fear1, fear5 = ctx1.features(BASE_TS + 90, 1)
    assert m1 > 0.0 and fear1 == 0.0 and fear5 == 0.0


def test_clamped_extreme_velocity() -> None:
    ctx = ContextTracker(["SPY"])
    steady_tape(ctx, "SPY", BASE_TS, 90, 500.0, 5.0)  # absurd slope
    m1, _m5, _f1, _f5 = ctx.features(BASE_TS + 90, 1)
    assert m1 == 10.0  # clamped


def test_intra_session_fear_flip_hits_fast_window_first() -> None:
    """A VIX-direction change mid-session must show up in fear_1m before fear_5m."""
    ctx = ContextTracker(["SPY", "VXX"])
    steady_tape(ctx, "VXX", BASE_TS, 300, 50.0, 0.01)  # fear rising for 5 min
    # regime flips: fear falls hard for 90 seconds
    steady_tape(ctx, "VXX", BASE_TS + 300, 90, 53.0, -0.02)
    _m1, _m5, fear1, fear5 = ctx.features(BASE_TS + 390, direction=1)
    assert fear1 < 0.0  # fast pulse already negative
    assert fear5 > fear1  # slow trend lags the flip
