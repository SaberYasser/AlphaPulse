"""End-to-end state machine scenarios through a real SymbolEngine."""

from __future__ import annotations

from conftest import BASE_TS, Harness
from early_trend_scanner.data.models import Quote, Trade
from early_trend_scanner.engine.state import Phase


def q(ts: float, price: float, spread: float = 0.012) -> Quote:
    return Quote("TEST", ts, round(price - spread / 2, 4), 5, round(price + spread / 2, 4), 5)


def tr(ts: float, price: float, size: int = 100) -> Trade:
    return Trade(symbol="TEST", ts=ts, price=price, size=size, conditions=("@",))


def pump_quiet(h: Harness, start: float, seconds: int, price: float = 100.0) -> float:
    """Steady two-sided tape: ~300 sh/s, flat price, builds minutes and levels."""
    ts = start
    for s in range(seconds):
        h.engine.on_quote(q(ts, price))
        for k in range(3):
            up = (s + k) % 2 == 0
            px = price + (0.006 if up else -0.006)
            h.engine.on_trade(tr(ts + 0.1 + 0.25 * k, px))
        ts += 1.0
    return ts


def pump_burst(
    h: Harness,
    start: float,
    price0: float,
    direction: int,
    seconds: float = 8.0,
    rate: int = 20,
    slope_bps_s: float = 2.0,
    buy_frac: float = 0.9,
) -> float:
    """Directional impulse: fast tape, one-sided flow, accelerating price."""
    n = int(seconds * rate)
    px = price0
    ts = start
    for i in range(n):
        ts = start + i / rate
        px = price0 * (1 + direction * slope_bps_s / 1e4 * (ts - start))
        h.engine.on_quote(q(ts, px))
        aggressive = (i % 10) < int(buy_frac * 10)
        side_up = aggressive if direction > 0 else not aggressive
        trade_px = round(px + (0.006 if side_up else -0.006), 4)
        h.engine.on_trade(tr(ts + 0.001, trade_px, size=300))
    return ts


def warmed(h: Harness, minutes: int = 4) -> float:
    return pump_quiet(h, BASE_TS, minutes * 60)


def test_early_signal_fires_up(harness: Harness) -> None:
    ts = warmed(harness)
    pump_burst(harness, ts, 100.0, direction=1)
    kinds = harness.messages()
    assert "EARLY" in kinds, f"no EARLY, emitted={kinds}"
    sig = harness.fired[0]
    assert sig.direction == 1
    assert sig.alert_ts - ts < 6.0, "alert not early enough into the impulse"
    assert sig.invalidation < sig.trigger_price
    assert harness.engine.machine.phase == Phase.FIRED


def test_early_signal_fires_down(harness: Harness) -> None:
    ts = warmed(harness)
    pump_burst(harness, ts, 100.0, direction=-1)
    assert "EARLY" in harness.messages()
    sig = harness.fired[0]
    assert sig.direction == -1
    assert sig.invalidation > sig.trigger_price


def test_confirmed_followup(harness: Harness) -> None:
    ts = warmed(harness)
    end = pump_burst(harness, ts, 100.0, direction=1, seconds=6.0)
    assert "EARLY" in harness.messages()
    # keep grinding higher past the 1-minute verdict clock
    pump_burst(harness, end, 100.0 * (1 + 12e-4), direction=1, seconds=70.0, slope_bps_s=0.8)
    kinds = harness.messages()
    assert "CONFIRMED" in kinds
    assert harness.engine.machine.phase == Phase.COOLDOWN
    sig = harness.final[0]
    assert sig.resolution == "CONFIRMED"
    assert (sig.micro_extreme - sig.alert_price) * sig.direction > 0


def test_failed_followup_on_reversal(harness: Harness) -> None:
    ts = warmed(harness, minutes=12)  # past the opening phase: recross-fail active
    end = pump_burst(harness, ts, 100.0, direction=1, seconds=6.0)
    assert "EARLY" in harness.messages()
    sig = harness.fired[0]
    # immediate hard reversal through the trigger on sell flow
    pump_burst(
        harness,
        end,
        sig.alert_price,
        direction=-1,
        seconds=10.0,
        slope_bps_s=3.0,
        buy_frac=0.9,
    )
    kinds = harness.messages()
    assert "FAILED" in kinds
    assert harness.final[0].resolution == "FAILED"


def test_max_three_notifications_per_setup(harness: Harness) -> None:
    ts = warmed(harness)
    end = pump_burst(harness, ts, 100.0, direction=1, seconds=6.0)
    pump_burst(harness, end, 100.0 * (1 + 12e-4), direction=1, seconds=70.0, slope_bps_s=0.8)
    per_setup = [k for k in harness.messages()]
    assert 1 <= len(per_setup) <= 3
    assert per_setup.count("EARLY") == 1


def test_cooldown_and_re_arm_block_repeat(harness: Harness) -> None:
    ts = warmed(harness, minutes=12)  # past the opening phase: recross-fail active
    end = pump_burst(harness, ts, 100.0, direction=1, seconds=6.0)
    sig = harness.fired[0]
    pump_burst(harness, end, sig.alert_price, direction=-1, seconds=10.0, slope_bps_s=3.0)
    assert harness.engine.machine.phase == Phase.COOLDOWN
    early_count = harness.messages("EARLY")
    # a second identical burst right away must not alert again
    end2 = pump_quiet(harness, end + 12, 20, price=100.0)
    pump_burst(harness, end2, 100.0, direction=1, seconds=8.0)
    assert harness.messages("EARLY") == early_count


def test_no_alert_without_volume_acceleration(harness: Harness) -> None:
    ts = warmed(harness)
    # price drifts up fast but tape stays at quiet rate: volume gate must reject
    n = 40
    for i in range(n):
        t = ts + i * 0.25
        px = 100.0 * (1 + 2.5e-4 / 4 * i)
        harness.engine.on_quote(q(t, px))
        harness.engine.on_trade(tr(t + 0.01, round(px + 0.006, 4), size=80))
    assert "EARLY" not in harness.messages()
    assert harness.engine.machine.rejections.get("volume_accel", 0) >= 1


def test_no_alert_when_extended(harness: Harness) -> None:
    ts = warmed(harness)
    # burst starting ~45 bps past the nearest level: beyond even the wider
    # opening-phase cap (40 bps), so still rejected as extended
    harness.engine.on_quote(q(ts, 100.45))
    pump_burst(harness, ts + 0.5, 100.45, direction=1, seconds=4.0)
    assert "EARLY" not in harness.messages()
    assert harness.engine.machine.rejections.get("extended", 0) >= 1


def test_single_print_rejected(harness: Harness) -> None:
    ts = warmed(harness)
    harness.engine.on_quote(q(ts, 100.05))
    harness.engine.on_trade(tr(ts + 0.1, 100.09, size=5000))  # one anomalous print
    assert "EARLY" not in harness.messages()
    assert harness.engine.machine.rejections.get("single_print", 0) >= 1


def test_stale_data_blocks_alerts(cfg) -> None:
    h = Harness(cfg)
    h.engine.alerts_enabled = lambda: False  # simulates disconnect/latency
    ts = pump_quiet(h, BASE_TS, 240)
    pump_burst(h, ts, 100.0, direction=1)
    assert h.messages() == []


def test_observation_deadline_confirms_when_held(harness: Harness) -> None:
    ts = warmed(harness)
    end = pump_burst(harness, ts, 100.0, direction=1, seconds=6.0)
    sig = harness.fired[0]
    held = sig.trigger_price * (1 + 6e-4)
    # quiet hold above trigger; no new micro-high burst, resolution at deadline
    t = end
    for i in range(95):
        t = end + i
        harness.engine.on_quote(q(t, held))
        harness.engine.on_trade(tr(t + 0.2, held))
        harness.engine.on_second_tick(t)
    kinds = harness.messages()
    assert "CONFIRMED" in kinds or "FAILED" in kinds  # resolved, not hanging
    assert harness.engine.machine.phase == Phase.COOLDOWN


def test_confirmation_requires_expansion_progress(harness: Harness) -> None:
    m = harness.engine.machine
    m.set_range5m_bps(100.0)  # inv distance = 35 bps of trigger
    sig = m.consider(_passing_snap(), 0.55)
    inv_dist = abs(sig.trigger_price - sig.invalidation)
    assert (sig.alert_price - sig.trigger_price) / inv_dist < 0.5  # not yet progressed
    barely = sig.trigger_price + 0.15 * inv_dist
    # past confirm_min_s, flow fine, but no 0.5R progress -> keeps observing
    m.observe(sig.alert_ts + 61.0, barely, imb5=0.5, share_up5=0.8, vol5=1000.0)
    assert sig.resolution == ""
    # deadline without progress -> FAILED, phrased as a stall
    m.observe(sig.alert_ts + 85.0, barely, imb5=0.5, share_up5=0.8, vol5=1000.0)
    assert sig.resolution == "FAILED"
    assert sig.resolution_reason == "no expansion progress"


def test_confirmation_fires_on_real_progress(harness: Harness) -> None:
    m = harness.engine.machine
    m.set_range5m_bps(100.0)
    sig = m.consider(_passing_snap(), 0.55)
    inv_dist = abs(sig.trigger_price - sig.invalidation)
    progressed = sig.trigger_price + 0.6 * inv_dist
    m.observe(sig.alert_ts + 61.0, progressed, imb5=0.5, share_up5=0.8, vol5=1000.0)
    assert sig.resolution == "CONFIRMED"


# ------------------------------------------------------------ fake-start veto


def _passing_snap(ts: float = BASE_TS + 1800.0, **over):
    from early_trend_scanner.engine.features import Snapshot

    base = dict(
        ts=ts,
        symbol="TEST",
        direction=1,
        price=100.10,
        level_price=100.05,
        level_kind=2,
        trigger_verb="break",
        break_bps=5.0,
        vel5_bps_s=3.0,
        vel15_bps_s=1.5,
        accelerating=True,
        vol5=5000.0,
        vol_ratio_prev=6.0,
        vol_ratio_base=4.0,
        imb5=0.5,
        imb15=0.35,
        n5=20.0,
        n_ratio_prev=4.0,
        dollar5=500_000.0,
        range_exp=1.5,
        comp_ratio=0.7,
        comp_active=True,
        dist_vwap_bps=5.0,
        spread_bps=2.0,
        quote_imb=0.1,
        minute_frac=0.3,
        persist_trades=6,
        persist_span_s=1.2,
        n2s=8,
        dominant_frac=0.2,
        score=0.95,
    )
    base.update(over)
    return Snapshot(**base)


def test_penalty_math() -> None:
    from early_trend_scanner.engine.features import fake_start_penalty

    # fully stale range, half-damped volume: 0.30 * (1 - 3/6) * 1.0
    s = _passing_snap(comp_ratio=1.6, vol_ratio_prev=3.0)
    p = fake_start_penalty(s, 1.0, 0.55, 6.0, 0.30)
    assert abs(p - 0.15) < 1e-9
    # fresh volume forgives everything
    s2 = _passing_snap(comp_ratio=2.0, imb15=0.9, vol_ratio_prev=6.0)
    assert fake_start_penalty(s2, 1.0, 0.55, 6.0, 0.30) == 0.0
    # clean setup carries no penalty
    assert fake_start_penalty(_passing_snap(vol_ratio_prev=2.5), 1.0, 0.55, 6.0, 0.30) == 0.0


def test_penalty_rejects_marginal_uncompressed_break(harness: Harness) -> None:
    from early_trend_scanner.engine.state import Rejection

    # marginal score + fully stale range + quiet volume -> attributed fake_start
    snap = _passing_snap(comp_ratio=1.6, vol_ratio_prev=2.5, score=0.70)
    out = harness.engine.machine.consider(snap, 0.55)
    assert isinstance(out, Rejection) and out.reason == "fake_start"
    assert harness.engine.machine.rejections["fake_start"] == 1


def test_penalty_rejects_marginal_one_sided_tape(harness: Harness) -> None:
    from early_trend_scanner.engine.state import Rejection

    out = harness.engine.machine.consider(
        _passing_snap(imb15=0.85, vol_ratio_prev=2.5, score=0.70), 0.55
    )
    assert isinstance(out, Rejection) and out.reason == "fake_start"
    # aligned check for the DOWN side too
    out2 = harness.engine.machine.consider(
        _passing_snap(
            direction=-1,
            imb5=-0.5,
            imb15=-0.85,
            vel5_bps_s=-3.0,
            vel15_bps_s=-1.5,
            vol_ratio_prev=2.5,
            score=0.70,
        ),
        0.55,
    )
    assert isinstance(out2, Rejection) and out2.reason == "fake_start"


def test_strong_evidence_outvotes_staleness(harness: Harness) -> None:
    from early_trend_scanner.engine.state import Signal

    # same stale conditions, but overwhelming trigger quality still fires
    out = harness.engine.machine.consider(
        _passing_snap(comp_ratio=1.6, vol_ratio_prev=2.5, score=0.95), 0.55
    )
    assert isinstance(out, Signal)


def test_fresh_volume_never_penalized(harness: Harness) -> None:
    from early_trend_scanner.engine.state import Signal

    out = harness.engine.machine.consider(
        _passing_snap(comp_ratio=2.0, imb15=0.9, vol_ratio_prev=20.0, score=0.70), 0.55
    )
    assert isinstance(out, Signal)


def test_score_failure_not_misattributed(harness: Harness) -> None:
    from early_trend_scanner.engine.state import Rejection

    # below threshold on raw score alone -> reason is "score", not "fake_start"
    out = harness.engine.machine.consider(
        _passing_snap(comp_ratio=1.6, vol_ratio_prev=2.5, score=0.50), 0.55
    )
    assert isinstance(out, Rejection) and out.reason == "score"


def test_veto_off_switch(cfg) -> None:
    from dataclasses import replace

    from early_trend_scanner.engine.state import Signal

    cfg2 = replace(cfg, engine=replace(cfg.engine, fresh_break_veto=False))
    h = Harness(cfg2)
    out = h.engine.machine.consider(
        _passing_snap(comp_ratio=1.6, vol_ratio_prev=2.5, score=0.70), 0.55
    )
    assert isinstance(out, Signal)


def test_clean_snap_fires(harness: Harness) -> None:
    from early_trend_scanner.engine.state import Signal

    out = harness.engine.machine.consider(_passing_snap(), 0.55)
    assert isinstance(out, Signal)


def test_suppressed_setup_uses_short_cooldown(cfg) -> None:
    from dataclasses import replace

    h = Harness(replace(cfg, engine=replace(cfg.engine, cooldown_suppressed_s=60.0)))
    m = h.engine.machine
    m.hooks.ml_predict = lambda f, d: (0.10, "test")  # low conviction
    m.hooks.ml_gate_active = lambda: True  # gate live -> suppression path
    m.set_range5m_bps(100.0)
    sig = m.consider(_passing_snap(), 0.55)
    assert sig.suppressed
    # resolve via deadline; suppressed setups get the short cooldown
    m.observe(sig.alert_ts + 85.0, sig.trigger_price, 0.0, 0.5, 100.0)
    assert sig.resolution != ""
    assert m.cooldown_until - sig.alert_ts - 85.0 == 60.0


def test_open_phase_allows_wider_extension(harness: Harness) -> None:
    from early_trend_scanner.engine.state import Rejection, Signal

    m = harness.engine.machine
    m.set_range5m_bps(100.0)
    # 30 bps past the level: allowed in the opening phase (cap 40)...
    early_snap = _passing_snap(break_bps=30.0, minute_frac=0.005)
    out = m.consider(early_snap, 0.55)
    assert isinstance(out, Signal)
    # ...rejected as extended at midday (cap 25 by dataclass default)
    h2 = Harness(harness.cfg)
    late = h2.engine.machine.consider(_passing_snap(break_bps=30.0, minute_frac=0.3), 0.55)
    assert isinstance(late, Rejection) and late.reason == "extended"


def test_opening_invalidation_breathes(harness: Harness) -> None:
    from early_trend_scanner.engine.state import Signal

    m = harness.engine.machine  # default range5m estimate is tiny at the open
    sig = m.consider(_passing_snap(minute_frac=0.005), 0.55)
    assert isinstance(sig, Signal)
    inv_bps = abs(sig.trigger_price - sig.invalidation) / sig.trigger_price * 1e4
    # 0.35 x the 100 bps opening floor = 35 bps, not the 6 bps minimum
    assert 30.0 <= inv_bps <= 40.5

    h2 = Harness(harness.cfg)
    sig2 = h2.engine.machine.consider(_passing_snap(minute_frac=0.3), 0.55)
    assert isinstance(sig2, Signal)
    inv2 = abs(sig2.trigger_price - sig2.invalidation) / sig2.trigger_price * 1e4
    assert inv2 < 10.0  # midday keeps the tight default sizing


def test_global_burst_cap() -> None:
    from early_trend_scanner.engine.state import GlobalLimiter

    lim = GlobalLimiter(10, max_per_burst=3, burst_window_s=60.0)
    t0 = 1_756_000_000.0
    for i in range(3):
        assert lim.allow(t0 + i)
        lim.note(t0 + i)
    assert not lim.allow(t0 + 5)  # 4th within the minute: blocked
    assert lim.allow(t0 + 65)  # burst window passed, hourly budget remains


def test_strong_score_bypasses_model_suppression(harness: Harness) -> None:
    from early_trend_scanner.engine.state import Signal

    m = harness.engine.machine
    m.hooks.ml_predict = lambda f, d: (0.10, "test")  # model hates it
    m.hooks.ml_gate_active = lambda: True
    m.hooks.prob_bypass_score = 0.80
    m.set_range5m_bps(100.0)
    sig = m.consider(_passing_snap(score=0.85), 0.55)
    assert isinstance(sig, Signal) and not sig.suppressed  # delivered anyway

    h2 = Harness(harness.cfg)
    m2 = h2.engine.machine
    m2.hooks.ml_predict = lambda f, d: (0.10, "test")
    m2.hooks.ml_gate_active = lambda: True
    m2.hooks.prob_bypass_score = 0.80
    m2.set_range5m_bps(100.0)
    sig2 = m2.consider(_passing_snap(score=0.72), 0.55)
    assert sig2.suppressed  # ordinary signals still gated


# ------------------------------------------------- sustained-pressure detector


def escalator(h: Harness, start: float, p0: float, direction: int, seconds: int = 90) -> float:
    """60-90s escalator: ~0.6 bps/s drift on steady one-sided volume.

    Below the micro-burst velocity gate (2 bps/s) yet a meaningful net move —
    exactly the band the sustained-pressure detector exists for.
    """
    ts = start
    for s in range(seconds):
        px = p0 * (1 + direction * 0.6e-4 * s)  # ~0.6 bps/s
        h.engine.on_quote(q(ts, px))
        for k in range(6):  # 6 prints/s, ~80% directional
            side_up = (k < 5) if direction > 0 else (k >= 5)
            tp = round(px + (0.006 if side_up else -0.006), 4)
            h.engine.on_trade(tr(ts + 0.05 + 0.15 * k, tp, size=120))
        h.engine.on_second_tick(ts + 1.0)
        ts += 1.0
    return ts


def test_trend_detector_fires_on_escalator(harness: Harness) -> None:
    ts = warmed(harness, minutes=16)  # 15-minute extreme lookback needs history
    escalator(harness, ts, 100.0, direction=1)
    kinds = harness.messages()
    assert "EARLY" in kinds, f"no trend EARLY, got {kinds}"
    sig = harness.fired[0]
    assert sig.trigger_verb == "trend"
    assert sig.features.get("trend") == 1.0
    assert not sig.suppressed  # rule-pure class


def test_trend_detector_fires_downside(harness: Harness) -> None:
    ts = warmed(harness, minutes=16)
    escalator(harness, ts, 100.0, direction=-1)
    sig = harness.fired[0]
    assert sig.direction == -1 and sig.trigger_verb == "trend"


def test_trend_needs_volume(harness: Harness) -> None:
    ts = warmed(harness, minutes=16)
    # same drift but tape thinner than the quiet baseline: volume gate blocks
    t = ts
    for s in range(90):
        px = 100.0 * (1 + 1.5e-4 / 10 * s)
        harness.engine.on_quote(q(t, px))
        harness.engine.on_trade(tr(t + 0.1, round(px + 0.006, 4), size=40))
        harness.engine.on_second_tick(t + 1.0)
        t += 1.0
    assert "EARLY" not in harness.messages()


def test_trend_daily_cap(harness: Harness) -> None:
    m = harness.engine.machine
    ts = warmed(harness, minutes=16)
    end = escalator(harness, ts, 100.0, direction=1)
    assert m.trend_alerts_today == 1
    # resolve + pass cooldown, then run a second and a third escalator
    end = pump_quiet(harness, end + 700, 60, price=101.0)
    end = escalator(harness, end, 101.0, direction=1)
    end = pump_quiet(harness, end + 700, 60, price=102.2)
    escalator(harness, end, 102.2, direction=1)
    assert m.trend_alerts_today <= harness.cfg.engine.trend_max_per_day


def test_opening_trend_continuation_uses_premarket_anchor(harness: Harness) -> None:
    """A premarket trend continuing through the bell fires without 15 minutes
    of session history, anchored to the premarket high."""
    from early_trend_scanner.engine.state import Signal

    harness.engine.seed_static_levels(pdh=101.5, pdl=98.0, pdc=99.0, pmh=100.05, pml=99.2)
    ts = warmed(harness, minutes=4)  # only 4 session minutes
    escalator(harness, ts, 100.0, direction=1)  # runs through the premarket high
    trend_sigs = [s for s in harness.fired if s.trigger_verb == "trend"]
    assert trend_sigs and isinstance(trend_sigs[0], Signal)


def test_opening_trend_needs_premarket_data(cfg) -> None:
    from dataclasses import replace

    h = Harness(replace(cfg, engine=replace(cfg.engine, fresh_break_veto=False)))
    # no premarket extremes seeded: young session must stay silent on trend
    ts = warmed(h, minutes=4)
    escalator(h, ts, 100.0, direction=1)
    assert not [s for s in h.fired if s.trigger_verb == "trend"]


def test_env_tiebreaker_confirms_marginal_progress(harness: Harness) -> None:
    m = harness.engine.machine
    m.set_range5m_bps(100.0)
    sig = m.consider(_passing_snap(), 0.55)
    inv_dist = abs(sig.trigger_price - sig.invalidation)
    marginal = sig.trigger_price + 0.4 * inv_dist  # 0.8x the required progress
    # deadline with supportive environment (market flowing with the signal)
    m.observe(sig.alert_ts + 85.0, marginal, 0.5, 0.8, 1000.0, env=(0.05, 0.0, 1.0))
    assert sig.resolution == "CONFIRMED"
    assert sig.resolution_reason == "progress with tailwind"


def test_env_neutral_marginal_progress_fails(harness: Harness) -> None:
    m = harness.engine.machine
    m.set_range5m_bps(100.0)
    sig = m.consider(_passing_snap(), 0.55)
    inv_dist = abs(sig.trigger_price - sig.invalidation)
    marginal = sig.trigger_price + 0.4 * inv_dist
    m.observe(sig.alert_ts + 85.0, marginal, 0.5, 0.8, 1000.0, env=(0.0, 0.0, 1.0))
    assert sig.resolution == "FAILED"
