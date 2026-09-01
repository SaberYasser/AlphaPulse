from early_trend_scanner.data.models import Trade
from early_trend_scanner.engine.aggregator import SymbolAggregator

T0 = 1_756_000_020  # minute-aligned epoch second


def trade(ts: float, price: float, size: int = 100, side: int = 1, **kw) -> Trade:
    t = Trade(symbol="X", ts=ts, price=price, size=size, conditions=("@",))
    t.side = side
    for k, v in kw.items():
        setattr(t, k, v)
    return t


def test_second_bars_and_gap_fill() -> None:
    agg = SymbolAggregator("X", ring_seconds=300)
    agg.on_trade(trade(T0 + 0.2, 10.0))
    agg.on_trade(trade(T0 + 0.7, 10.05))
    agg.on_trade(trade(T0 + 4.1, 10.10))  # 3 empty seconds between
    assert len(agg.ring) == 4  # seconds T0..T0+3 finalized
    first = agg.ring.from_end(3)
    assert first is not None
    assert first.high == 10.05 and first.vol == 200
    empty = agg.ring.from_end(1)
    assert empty is not None
    assert empty.vol == 0 and empty.close == 10.05  # carried close


def test_minute_folding() -> None:
    minutes = []
    agg = SymbolAggregator("X", on_minute=minutes.append)
    start = T0  # T0 is minute-aligned
    assert start % 60 == 0
    for i in range(130):
        agg.on_trade(trade(start + i, 10.0 + i * 0.01, size=50))
    assert len(minutes) == 2
    m = minutes[0]
    assert m.ts == start and m.vol == 50 * 60
    assert m.open == 10.0 and abs(m.close - 10.59) < 1e-9
    assert minutes[1].ts == start + 60 and abs(minutes[1].open - 10.60) < 1e-9


def test_flags_respected() -> None:
    agg = SymbolAggregator("X")
    agg.on_trade(trade(T0 + 0.1, 10.0))
    # volume-only print: huge price must not touch OHLC
    agg.on_trade(trade(T0 + 0.2, 99.0, size=500, updates_last=False, flow_eligible=False))
    agg.on_trade(trade(T0 + 1.2, 10.02))
    sec = agg.ring.from_end(0)
    assert sec is not None
    assert sec.high == 10.0 and sec.vol == 600
    assert agg.last_price == 10.02


def test_vwap_and_session_extremes() -> None:
    agg = SymbolAggregator("X")
    agg.set_session(float(T0), opening_range_min=1)
    agg.on_trade(trade(T0 + 1, 10.0, size=100, side=1))
    agg.on_trade(trade(T0 + 2, 20.0, size=100, side=-1))
    assert abs(agg.vwap - 15.0) < 1e-9
    assert agg.session_high == 20.0 and agg.session_low == 10.0
    assert agg.or_high == 20.0 and agg.or_low == 10.0
    # pre-session trades must not affect vwap
    agg2 = SymbolAggregator("Y")
    agg2.set_session(float(T0) + 1000, opening_range_min=5)
    agg2.on_trade(trade(T0 + 1, 10.0))
    assert agg2.vwap == 0.0


def test_window_includes_partial_second() -> None:
    agg = SymbolAggregator("X")
    for i in range(10):
        agg.on_trade(trade(T0 + i, 10.0, size=100))
    agg.on_trade(trade(T0 + 10.3, 10.0, size=999))  # partial second
    vol5, *_ = agg.window(5)
    assert vol5 == 4 * 100 + 999


def test_buy_sell_split() -> None:
    agg = SymbolAggregator("X")
    agg.on_trade(trade(T0 + 0.1, 10.0, size=100, side=1))
    agg.on_trade(trade(T0 + 0.2, 10.0, size=40, side=-1))
    agg.on_trade(trade(T0 + 0.3, 10.0, size=10, side=0))
    agg.on_trade(trade(T0 + 1.5, 10.0))
    sec = agg.ring.from_end(0)
    assert sec is not None
    assert sec.buy_vol == 100 and sec.sell_vol == 40 and sec.vol == 150


def test_preopen_seed_enables_immediate_velocity() -> None:
    agg = SymbolAggregator("X")
    agg.seed_preopen(100.0, T0)
    assert agg.price_ago(5) == 100.0 and agg.price_ago(15) == 100.0
    assert agg.last_price == 100.0
    # first live seconds can now measure velocity against the premarket close
    agg.on_trade(trade(T0 + 0.5, 100.10, size=50))
    agg.on_trade(trade(T0 + 1.5, 100.20, size=50))
    assert agg.price_ago(1) is not None
    # zero-volume seed must not distort volume windows
    vol, *_ = agg.window(60)
    assert vol == 100


def test_preopen_seed_noop_when_ring_has_data() -> None:
    agg = SymbolAggregator("X")
    agg.on_trade(trade(T0, 99.0, size=10))
    agg.on_trade(trade(T0 + 1.0, 99.1, size=10))
    agg.seed_preopen(50.0, T0 + 2.0)  # must refuse: real data present
    assert agg.last_price != 50.0


def test_ema_slope_none_until_enough_history() -> None:
    agg = SymbolAggregator("X")
    assert agg.ema_slope_bps() is None
    agg.seed_ema([100.0, 100.1, 100.2])
    agg.last_price = 100.2
    assert agg.ema_slope_bps() is None  # 3 EMA values: no 3-bar span yet
    agg.seed_ema([100.3])
    assert agg.ema_slope_bps() is not None


def test_ema_slope_sign_tracks_minute_trend() -> None:
    agg = SymbolAggregator("X")
    agg.seed_ema([100.0 + 0.05 * i for i in range(25)])  # rising premarket
    agg.last_price = 101.2
    up = agg.ema_slope_bps()
    assert up is not None and up > 0
    # live minutes now fall: folding completed falling minutes flips the slope
    start = T0
    for i in range(6 * 60):
        agg.on_trade(trade(start + i, 101.2 - i * 0.01, size=10))
    down = agg.ema_slope_bps()
    assert down is not None and down < 0


def test_seed_minutes_feeds_ema() -> None:
    from early_trend_scanner.data.models import Minute

    agg = SymbolAggregator("X")
    mins = [
        Minute(ts=T0 + i * 60, open=100.0, high=100.2, low=99.9, close=100.0 + 0.1 * i)
        for i in range(6)
    ]
    agg.seed_minutes(mins)
    slope = agg.ema_slope_bps()
    assert slope is not None and slope > 0
