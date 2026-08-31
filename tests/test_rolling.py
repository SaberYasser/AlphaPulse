from early_trend_scanner.data.models import Sec
from early_trend_scanner.engine.rolling import SecRing


def sec(ts: int, close: float = 10.0, vol: int = 100, buy: int = 60, sell: int = 40) -> Sec:
    return Sec(
        ts=ts,
        open=close,
        high=close + 0.01,
        low=close - 0.01,
        close=close,
        vol=vol,
        buy_vol=buy,
        sell_vol=sell,
        n=5,
        dollar=close * vol,
    )


def test_window_sums_match_bruteforce() -> None:
    ring = SecRing(120)
    secs = [sec(1000 + i, vol=100 + i, buy=50 + i, sell=50) for i in range(90)]
    for s in secs:
        ring.append(s)
    for w in (5, 15, 30, 60):
        vol, buy, sell, n, dollar = ring.sums(w)
        tail = secs[-w:]
        assert vol == sum(s.vol for s in tail)
        assert buy == sum(s.buy_vol for s in tail)
        assert sell == sum(s.sell_vol for s in tail)
        assert n == sum(s.n for s in tail)
        assert abs(dollar - sum(s.dollar for s in tail)) < 1e-6


def test_wraparound_keeps_sums_exact() -> None:
    ring = SecRing(64)
    total = 300  # force multiple wraps
    secs = [sec(2000 + i, vol=(i * 7) % 50 + 1) for i in range(total)]
    for s in secs:
        ring.append(s)
    vol, *_ = ring.sums(30)
    assert vol == sum(s.vol for s in secs[-30:])
    assert len(ring) == 64


def test_close_ago_and_from_end() -> None:
    ring = SecRing(100)
    for i in range(20):
        ring.append(sec(3000 + i, close=10.0 + i))
    assert ring.newest is not None and ring.newest.ts == 3019
    assert ring.close_ago(1) == 29.0  # newest finalized
    assert ring.close_ago(5) == 25.0
    assert ring.from_end(19).ts == 3000  # type: ignore[union-attr]
    assert ring.from_end(20) is None
    assert ring.close_ago(21) is None


def test_high_low_and_iter_between() -> None:
    ring = SecRing(100)
    for i in range(10):
        ring.append(sec(4000 + i, close=10.0 + (i % 3)))
    hl = ring.high_low(5)
    assert hl is not None
    tail = [ring.from_end(k) for k in range(5)]
    assert hl[0] == max(s.high for s in tail)  # type: ignore[union-attr]
    assert hl[1] == min(s.low for s in tail)  # type: ignore[union-attr]
    got = list(ring.iter_between(4002, 4005))
    assert [s.ts for s in got] == [4002, 4003, 4004]


def test_ring_is_bounded() -> None:
    ring = SecRing(70)
    for i in range(10_000):
        ring.append(sec(i))
    assert len(ring) == 70
