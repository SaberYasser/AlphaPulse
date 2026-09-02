"""Warmup orchestration: feed fallback and session seeding."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

import pytest

from early_trend_scanner.config import Config
from early_trend_scanner.data.models import Bar, Minute, SessionInfo
from early_trend_scanner.data.rest import AlpacaRestError
from early_trend_scanner.engine.baseline import MinuteBaseline
from early_trend_scanner.warmup import _levels_bars, bars_to_minutes, warmup

OPEN = 1_800_000_000.0


def bar(ts: float, close: float, *, high: float | None = None, low: float | None = None) -> Bar:
    return Bar(
        ts=ts,
        open=close - 0.5,
        high=high if high is not None else close + 1.0,
        low=low if low is not None else close - 1.0,
        close=close,
        volume=1_200,
        trade_count=120,
        vwap=close - 0.25,
    )


def test_bars_to_minutes_preserves_market_fields() -> None:
    result = bars_to_minutes([bar(OPEN + 61.5, 101.0)])

    assert result == [
        Minute(
            ts=int(OPEN + 60),
            open=100.5,
            high=102.0,
            low=100.0,
            close=101.0,
            vol=1_200,
            n=120,
            dollar=100.75 * 1_200,
        )
    ]


@pytest.mark.asyncio
async def test_level_history_falls_back_to_live_feed() -> None:
    class Rest:
        def __init__(self) -> None:
            self.feeds: list[str] = []

        async def bars(self, symbols, timeframe, t0, t1, feed):
            self.feeds.append(feed)
            if feed == "sip":
                raise AlpacaRestError(403, "not entitled")
            return {"TSLA": [bar(OPEN, 100.0)]}

    rest = Rest()

    result = await _levels_bars(cast(Any, rest), ["TSLA"], "1Day", OPEN - 100, OPEN, "sip", "iex")

    assert rest.feeds == ["sip", "iex"]
    assert result["TSLA"][0].close == 100.0


@dataclass
class FakeAgg:
    ema_closes: list[float] = field(default_factory=list)
    vwap_seed: tuple[float, float] | None = None
    session_high: float = 0.0
    session_low: float = 0.0

    def seed_ema(self, closes: list[float]) -> None:
        self.ema_closes = closes

    def seed_preopen(self, price: float, open_ts: float) -> None:
        raise AssertionError("mid-session warmup must not seed a pre-open second")

    def seed_vwap(self, pv: float, volume: float) -> None:
        self.vwap_seed = (pv, volume)


@dataclass
class FakeEngine:
    agg: FakeAgg = field(default_factory=FakeAgg)
    session_open: float = 0.0
    static_levels: dict[str, float] = field(default_factory=dict)
    minutes: list[Minute] = field(default_factory=list)
    opening_range: tuple[float, float] | None = None

    def set_session(self, open_ts: float) -> None:
        self.session_open = open_ts

    def seed_static_levels(self, **levels: float) -> None:
        self.static_levels = levels

    def seed_minutes(self, minutes: list[Minute]) -> None:
        self.minutes = minutes

    def seed_opening_range(self, high: float, low: float) -> None:
        self.opening_range = (high, low)


@pytest.mark.asyncio
async def test_mid_session_warmup_seeds_all_engine_inputs(
    cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = SessionInfo("2027-01-15", OPEN, OPEN + 390 * 60)
    prior = SessionInfo("2027-01-14", OPEN - 86_400, OPEN - 86_400 + 390 * 60)
    monkeypatch.setattr("early_trend_scanner.warmup.time.time", lambda: OPEN + 20 * 60)

    class Rest:
        async def bars(self, symbols, timeframe, t0, t1, feed):
            if timeframe == "1Day":
                return {"TSLA": [bar(OPEN - 86_400, 100.0, high=110.0, low=90.0)]}
            if t1 < OPEN - 1_000:
                return {"TSLA": [bar(prior.open_ts, 99.0)]}
            if t0 < OPEN:
                return {
                    "TSLA": [
                        bar(OPEN - 120, 103.0, high=105.0, low=102.0),
                        bar(OPEN - 60, 104.0, high=106.0, low=103.0),
                    ]
                }
            return {
                "TSLA": [
                    bar(OPEN, 105.0, high=107.0, low=104.0),
                    bar(OPEN + 60, 106.0, high=108.0, low=105.0),
                ]
            }

    clock = SimpleNamespace(completed_sessions_before=lambda _ts, _n: [prior])
    engine = FakeEngine()
    baseline = MinuteBaseline()

    await warmup(
        cast(Any, Rest()),
        cast(Any, clock),
        cfg,
        cast(Any, {"TSLA": engine}),
        baseline,
        session,
        OPEN + 20 * 60,
    )

    assert baseline.ready("TSLA")
    assert engine.session_open == OPEN
    assert engine.static_levels == {
        "pdh": 110.0,
        "pdl": 90.0,
        "pdc": 100.0,
        "pmh": 106.0,
        "pml": 102.0,
    }
    assert engine.agg.ema_closes == [103.0, 104.0]
    assert len(engine.minutes) == 2
    assert engine.agg.vwap_seed == (104.75 * 1_200 + 105.75 * 1_200, 2_400.0)
    assert engine.agg.session_high == 108.0
    assert engine.agg.session_low == 104.0
    assert engine.opening_range == (108.0, 104.0)
