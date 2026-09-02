import time
from pathlib import Path

from early_trend_scanner.engine.state import Signal
from early_trend_scanner.ml.labeler import LabelResult
from early_trend_scanner.store.db import SignalStore


def sig(sid: str = "X-1-U", ts: float | None = None) -> Signal:
    return Signal(
        signal_id=sid,
        symbol="X",
        direction=1,
        alert_ts=ts or time.time(),
        alert_price=100.05,
        trigger_price=100.0,
        level_kind=0,
        trigger_verb="break",
        invalidation=99.9,
        vol_ratio=2.5,
        features={"vol_prev": 3.0},
        prob=0.61,
        model_version="builtin:u1+d0:drift0",
        score=0.7,
    )


def test_signal_lifecycle_roundtrip(tmp_path: Path) -> None:
    store = SignalStore(tmp_path / "t.db")
    s = sig()
    store.record_signal(s, "2026-08-28", gate_version=1)
    s.resolution = "CONFIRMED"
    s.resolution_ts = s.alert_ts + 20
    s.resolution_price = 100.25
    s.resolution_reason = "volume sustained"
    store.record_resolution(s)
    store.record_label(
        LabelResult(
            signal=s,
            label=True,
            mfe=0.4,
            mae=0.05,
            lead_time_s=18.0,
            remaining_frac=0.8,
            was_late=False,
            reason="expanded",
        )
    )
    rows = store.recent_signals()
    assert len(rows) == 1
    row = rows[0]
    assert row["resolution"] == "CONFIRMED"
    assert row["label"] == 1
    assert row["lead_time_s"] == 18.0
    assert row["prob"] == 0.61
    store.record_efficacy(s.signal_id, True, 37.5)
    row = store.recent_signals()[0]
    assert row["efficacy_label"] == 1
    assert row["efficacy_move_bps"] == 37.5
    assert store.efficacy_label_counts() == (1, 0)
    training = store.efficacy_training_rows()
    assert training == [({"vol_prev": 3.0}, 1, True)]
    assert store.label_counts() == (1, 0)
    store.close()


def test_retention_removes_old_rows(tmp_path: Path) -> None:
    store = SignalStore(tmp_path / "t.db", retention_days=30)
    old = sig("OLD-1-U", ts=time.time() - 40 * 86400)
    new = sig("NEW-1-U")
    store.record_signal(old, "2026-07-01", 1)
    store.record_signal(new, "2026-08-28", 1)
    store.close()
    store2 = SignalStore(tmp_path / "t.db", retention_days=30)  # retention on open
    ids = [r["signal_id"] for r in store2.recent_signals()]
    assert ids == ["NEW-1-U"]
    store2.close()


def test_meta_and_daily_metrics(tmp_path: Path) -> None:
    store = SignalStore(tmp_path / "t.db")
    store.set_meta("adaptive_gate", '{"n_labels": 3}')
    assert store.get_meta("adaptive_gate") == '{"n_labels": 3}'
    assert store.get_meta("missing") is None
    store.record_daily_metrics("2026-08-28", "_ALL", {"alerts": 5})
    store.record_daily_metrics("2026-08-28", "_ALL", {"alerts": 6})  # upsert
    assert store.get_daily_metrics("2026-08-28") == {"alerts": 6}
    assert store.get_daily_metrics("2026-08-29") is None
    store.close()
