"""Daily recap: efficacy arithmetic, formatting, and quiet-hour delivery gate."""

from __future__ import annotations

from early_trend_scanner.recap import Recap, RecapRow, build_recap, format_recap

T0 = 1_756_000_000.0


def rows3() -> list[RecapRow]:
    return [
        RecapRow("AAA", 1, T0, 100.0, T0 + 60),  # long, keeps rising  -> favorable
        RecapRow("BBB", -1, T0, 50.0, T0 + 60),  # short, price rises  -> unfavorable
        RecapRow("CCC", 1, T0, 200.0, T0 + 60),  # long, fades         -> unfavorable
    ]


def price_at(sym: str, ts: float) -> float | None:
    entry = {"AAA": 100.0, "BBB": 50.0, "CCC": 200.0}
    final = {"AAA": 101.0, "BBB": 50.5, "CCC": 199.0}
    return entry[sym] if ts < T0 + 900 else final[sym]


def test_build_recap_math() -> None:
    r = build_recap(rows3(), price_at, cutoff_ts=T0 + 1800)
    assert r.confirmed == 3 and r.favorable == 1
    assert r.efficacy is not None and abs(r.efficacy - 1 / 3) < 1e-9
    # moves: +100, -100, -50 bps -> avg ~ -16.7
    assert abs(r.avg_move_bps + 16.7) < 1.0


def test_missing_price_falls_back_or_skips() -> None:
    def sparse(sym: str, ts: float) -> float | None:
        return None if sym == "AAA" else price_at(sym, ts)

    r = build_recap(rows3(), sparse, cutoff_ts=T0 + 1800)
    assert r.confirmed == 2  # AAA skipped: no final price at cutoff


def test_format_is_short() -> None:
    msg = format_recap("2026-08-31", Recap(confirmed=5, favorable=3, avg_move_bps=23.4))
    assert "60%" in msg and len(msg.split()) < 40
    empty = format_recap("2026-08-31", Recap(0, 0, 0.0))
    assert "no confirmed" in empty


def test_quiet_hour_blocks_new_alert_delivery(cfg) -> None:
    from early_trend_scanner.app import ScannerApp
    from early_trend_scanner.config import Secrets
    from early_trend_scanner.engine.state import Signal

    app = ScannerApp(cfg, Secrets())
    app._cutoff_ts = T0 + 1000.0
    hooks = app._build_hooks()

    def sig(ts: float) -> Signal:
        return Signal(
            signal_id=f"X-{int(ts)}",
            symbol="TSLA",
            direction=1,
            alert_ts=ts,
            alert_price=100.1,
            trigger_price=100.0,
            level_kind=0,
            trigger_verb="break",
            invalidation=99.7,
            vol_ratio=3.0,
            features={},
        )

    hooks.emit(sig(T0 + 500.0), "EARLY", {})  # before cutoff: delivered
    hooks.emit(sig(T0 + 1500.0), "EARLY", {})  # in the quiet hour: silent
    delivered = [m for m in app.notifier.captured if "EARLY" in m]
    assert len(delivered) == 1
    # follow-up for the pre-cutoff signal still delivers after the cutoff
    s = sig(T0 + 500.0)
    s.resolution_ts = T0 + 1600.0
    hooks.emit(s, "CONFIRMED", {"reason": "volume sustained", "env": ""})
    assert any("CONFIRMED" in m for m in app.notifier.captured)
    app.store.close()
