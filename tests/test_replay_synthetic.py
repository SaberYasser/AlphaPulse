"""Replay regression: full pipeline over deterministic synthetic sessions.

Asserts the properties the product is judged on: early detection of both
directions, silence on quiet tape, bounded memory, and no lookahead.
"""

from __future__ import annotations

import pytest

from early_trend_scanner.config import Config
from early_trend_scanner.data.models import SessionInfo
from early_trend_scanner.replay.engine import ReplayResult, ReplayRunner
from early_trend_scanner.replay.synthetic import SyntheticSession


@pytest.fixture(scope="module")
def outcome() -> tuple[SyntheticSession, ReplayResult]:
    synth = SyntheticSession(seed=7)
    cfg = Config()
    session = SessionInfo("synthetic", synth.open_ts, synth.close_ts)
    runner = ReplayRunner(cfg, session, [s.symbol for s in synth.script])
    result = runner.run(synth.events())
    return synth, result


def test_breakouts_detected_both_directions(outcome) -> None:
    synth, result = outcome
    truths = {t.symbol: t for t in synth.truths}
    for sym, _kind in (("SYNUP", "breakout_up"), ("SYNDN", "breakout_down")):
        truth = truths[sym]
        hits = [
            s
            for s in result.signals
            if s.symbol == sym and not s.suppressed and s.direction == truth.direction
        ]
        assert hits, f"{sym}: expansion missed entirely"
        first = min(hits, key=lambda s: s.alert_ts)
        delay = first.alert_ts - truth.event_ts
        assert -5.0 <= delay <= 60.0, f"{sym}: alert {delay:.1f}s after expansion start"


def test_alert_lands_before_most_of_move(outcome) -> None:
    synth, result = outcome
    labeled = {r.signal.symbol: r for r in result.labels if r.signal.symbol == "SYNUP"}
    assert "SYNUP" in labeled
    r = labeled["SYNUP"]
    assert r.label is True, f"SYNUP labeled {r.reason}"
    assert r.remaining_frac >= 0.5, "most of the move was already gone at alert time"


def test_quiet_symbol_stays_silent(outcome) -> None:
    _, result = outcome
    quiet = [s for s in result.signals if s.symbol == "SYNQT" and not s.suppressed]
    assert quiet == [], f"false alerts on quiet tape: {len(quiet)}"


def test_fakeout_fails_fast_if_alerted(outcome) -> None:
    _, result = outcome
    fk = [s for s in result.signals if s.symbol == "SYNFK" and not s.suppressed]
    for s in fk:
        assert s.resolution in ("FAILED", "CONFIRMED", "")
    # if the sweep tricked us, the follow-up must not be CONFIRMED-up-and-hold
    confirmed_up = [s for s in fk if s.direction == 1 and s.resolution == "CONFIRMED"]
    assert len(confirmed_up) == 0, "fakeout was CONFIRMED — follow-through check broken"


def test_max_three_messages_per_setup(outcome) -> None:
    _, result = outcome
    by_setup: dict[str, int] = {}
    for _ts, msg in result.messages:
        parts = msg.split()
        key = f"{parts[2]}"  # symbol
        by_setup[key] = by_setup.get(key, 0) + 1
    for s in result.signals:
        if not s.suppressed:
            assert s.followups_sent <= 2


def test_memory_bounded(outcome) -> None:
    _, result = outcome
    assert 0 < result.peak_rss_mb < 500, f"peak RSS {result.peak_rss_mb} MB"
    for sym, size in result.ring_sizes.items():
        assert size <= Config().engine.ring_seconds, f"{sym} ring grew past its cap"


def test_metrics_report_complete(outcome) -> None:
    _, result = outcome
    report = result.report()
    assert report["events_processed"] > 10_000
    assert report["signals"] >= 2
    assert report["precision"] is not None
    assert report["median_lead_s"] is not None
    assert "SYNUP" in report["by_symbol"]


def test_determinism_same_seed() -> None:
    cfg = Config()

    def run(seed: int) -> list[str]:
        synth = SyntheticSession(seed=seed)
        session = SessionInfo("synthetic", synth.open_ts, synth.close_ts)
        runner = ReplayRunner(cfg, session, [s.symbol for s in synth.script])
        res = runner.run(synth.events())
        return [s.signal_id for s in res.signals]

    assert run(11) == run(11)


def test_no_lookahead_guard() -> None:
    """The runner rejects an out-of-order stream instead of silently using it."""
    from early_trend_scanner.data.models import Trade

    cfg = Config()
    synth = SyntheticSession(seed=7)
    session = SessionInfo("synthetic", synth.open_ts, synth.close_ts)
    runner = ReplayRunner(cfg, session, ["SYNUP"])
    t0 = synth.open_ts
    bad = [
        Trade(symbol="SYNUP", ts=t0 + 100, price=100.0, size=100, conditions=("@",)),
        Trade(symbol="SYNUP", ts=t0 + 10, price=100.0, size=100, conditions=("@",)),
    ]
    with pytest.raises(AssertionError, match="out of order"):
        runner.run(bad)
