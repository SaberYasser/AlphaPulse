from collections import deque

from early_trend_scanner.data.models import Minute
from early_trend_scanner.engine.levels import Level, LevelBook, LevelKind

T0 = 1_756_000_000


def minute(i: int, high: float, low: float, close: float | None = None) -> Minute:
    c = close if close is not None else (high + low) / 2
    return Minute(ts=T0 + i * 60, open=c, high=high, low=low, close=c, vol=1000)


def filled(book: LevelBook, bars: list[Minute]) -> deque[Minute]:
    d: deque[Minute] = deque(bars, maxlen=120)
    book.on_minute(d)
    return d


def test_static_levels_and_merge() -> None:
    book = LevelBook("X", merge_bps=2.0)
    book.set_static(pdh=101.0, pdl=99.0, pdc=100.0, pmh=101.001, pml=98.5)
    levels = book.levels()
    # PMH within 2 bps of PDH merges; PDH (higher priority) wins
    kinds = [lv.kind for lv in levels]
    assert LevelKind.PDH in kinds and LevelKind.PMH not in kinds
    assert LevelKind.PDL in kinds and LevelKind.PML in kinds


def test_rolling_ranges_and_swings() -> None:
    book = LevelBook("X")
    bars = [minute(i, 100.2, 99.8) for i in range(8)]
    bars += [minute(8, 100.9, 99.9)]  # swing-high candidate
    bars += [minute(9 + i, 100.3, 99.9) for i in range(4)]
    filled(book, bars)
    kinds = {lv.kind: lv for lv in book.dynamic_levels}
    assert kinds[LevelKind.RANGE_H].price == 100.9
    assert LevelKind.MICRO_H in kinds and kinds[LevelKind.MICRO_H].price == 100.3
    swing_hs = [lv for lv in book.dynamic_levels if lv.kind == LevelKind.SWING_H]
    assert any(abs(lv.price - 100.9) < 1e-9 for lv in swing_hs)


def test_compression_detection() -> None:
    book = LevelBook("X", compression_max_ratio=0.75)
    wide = [minute(i, 100.5, 99.5) for i in range(10)]
    tight = [minute(10 + i, 100.06, 99.98) for i in range(3)]
    filled(book, wide + tight)
    assert book.compression.active
    assert book.compression.ratio < 0.25
    # no compression when ranges stay wide
    book2 = LevelBook("X")
    filled(book2, [minute(i, 100.5, 99.5) for i in range(13)])
    assert not book2.compression.active


def test_range5m() -> None:
    book = LevelBook("X")
    bars = [minute(i, 100.0 + i * 0.1, 99.0 + i * 0.1) for i in range(6)]
    filled(book, bars)
    assert abs(book.range5m - (100.5 - 99.1)) < 1e-9


def test_sweep_and_reclaim() -> None:
    book = LevelBook("X", sweep_window_s=30.0, sweep_max_bps=20.0)
    book.set_static(pdh=100.0)
    ts = float(T0)
    book.observe(99.95, ts)  # below the level
    book.observe(100.05, ts + 5)  # sweep above (5 bps)
    book.observe(100.08, ts + 8)
    book.observe(99.97, ts + 12)  # reclaimed below within window
    events = book.active_reclaims(ts + 13)
    assert len(events) == 1
    ev = events[0]
    assert ev.direction == -1 and ev.level.kind == LevelKind.PDH
    assert abs(ev.swept_extreme - 100.08) < 1e-9
    book.consume_reclaim(ev)
    assert not book.active_reclaims(ts + 13)


def test_deep_break_is_not_a_sweep() -> None:
    book = LevelBook("X", sweep_window_s=30.0, sweep_max_bps=20.0)
    book.set_static(pdh=100.0)
    ts = float(T0)
    book.observe(99.95, ts)
    book.observe(100.50, ts + 5)  # 50 bps beyond: a real break
    book.observe(99.97, ts + 10)
    assert not book.active_reclaims(ts + 11)


def test_levels_cache_invalidation() -> None:
    book = LevelBook("X")
    book.set_static(pdh=100.0)
    lv1 = book.levels_cached(0.0, float(T0))
    assert [lv.kind for lv in lv1] == [LevelKind.PDH]
    book.set_static(pdh=100.0, pdl=90.0)
    lv2 = book.levels_cached(0.0, float(T0))
    assert len(lv2) == 2  # cache was invalidated within the same second


def test_break_direction_helpers() -> None:
    lv = Level(100.0, LevelKind.PDH)
    assert LevelBook.break_direction(lv, 100.5) == 1
    assert LevelBook.break_direction(lv, 99.5) == -1
    assert LevelBook.is_upper(lv)
    assert not LevelBook.is_upper(Level(99.0, LevelKind.PDL))
