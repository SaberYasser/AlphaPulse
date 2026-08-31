from early_trend_scanner.data.filters import TradeFilter, classify_conditions
from early_trend_scanner.data.models import Trade


def make_trade(
    ts: float = 100.0, price: float = 10.0, size: int = 100, conds: tuple = ("@",)
) -> Trade:
    return Trade(symbol="X", ts=ts, price=price, size=size, conditions=conds)


def test_regular_trade_full_eligibility() -> None:
    assert classify_conditions(("@",)) == (True, True, True)
    assert classify_conditions(()) == (True, True, True)
    assert classify_conditions(("@", "F")) == (True, True, True)  # intermarket sweep


def test_odd_lot_counts_volume_not_last() -> None:
    last, vol, flow = classify_conditions(("@", "I"))
    assert (last, vol, flow) == (False, True, True)


def test_average_price_and_derivative_excluded_from_flow() -> None:
    assert classify_conditions(("W",)) == (False, True, False)
    assert classify_conditions(("4",)) == (False, True, False)
    assert classify_conditions(("@", "7")) == (False, True, False)


def test_admin_prints_dropped_entirely() -> None:
    f = TradeFilter()
    assert f.apply(make_trade(conds=("M",))) is None  # official close
    assert f.apply(make_trade(conds=("Q",))) is None  # official open
    assert f.apply(make_trade(conds=("9",))) is None
    assert f.stats.admin_excluded == 3


def test_extended_hours_and_out_of_sequence_not_price_forming() -> None:
    for c in ("T", "U", "Z", "L", "P", "R", "C", "H", "G", "N", "V"):
        last, vol, flow = classify_conditions((c,))
        assert last is False, c
        assert vol is True, c


def test_bad_fields_dropped() -> None:
    f = TradeFilter()
    assert f.apply(make_trade(price=0.0)) is None
    assert f.apply(make_trade(size=0)) is None
    assert f.stats.dropped_bad_fields == 2


def test_late_out_of_sequence_demoted() -> None:
    f = TradeFilter(late_grace_s=2.0)
    t1 = f.apply(make_trade(ts=100.0))
    assert t1 is not None and t1.updates_last
    late = f.apply(make_trade(ts=97.0))  # 3s behind watermark
    assert late is not None
    assert late.updates_last is False and late.flow_eligible is False
    assert late.updates_volume is True
    assert f.stats.late_out_of_sequence == 1
    # within grace: untouched
    ok = f.apply(make_trade(ts=99.0))
    assert ok is not None and ok.updates_last is True


def test_watermark_is_per_symbol() -> None:
    f = TradeFilter()
    a = make_trade(ts=100.0)
    b = Trade(symbol="Y", ts=50.0, price=5.0, size=10, conditions=("@",))
    assert f.apply(a) is not None
    fb = f.apply(b)
    assert fb is not None and fb.updates_last  # other symbol unaffected
