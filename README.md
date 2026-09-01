# ⚡ AlphaPulse

### Self-learning early trend detection — an online Adaptive Random Forest that fine-tunes itself on every live market outcome

**A production real-time equity scanner that detects the earliest stage of 1–5 minute price
expansions — long and short, symmetrically — distilling 9M+ ticks per session into a handful of
high-conviction alerts, and sharpening its own selectivity every single trading day.**

Built in Python on Alpaca SIP/IEX market data, running live during US regular trading hours on
Windows, delivering sub-40-word signals to Telegram within seconds of trigger conditions being met.

> **Live case study.** On 2026-08-31, TSLA opened at \$347.82 and ran ~\$13 intraday. The scanner's
> replay of that session (after a day of instrumented debugging described below) produces:
>
> ```
> EARLY UP TSLA 09:30:01 ET | 347.94 cross | volume 40.1x | velocity accelerating | invalidation 346.73
> CONFIRMED UP TSLA 09:30:27 ET | held 347.94 | volume sustained | new micro-high 349.45
> → outcome label: POSITIVE — 98% of the eventual move still ahead at alert time
> ```
>
> One second after the opening bell, with 40× mid-day volume behind it.

---

## Architecture

```
                        Alpaca WebSocket (trades + NBBO quotes)
                        14 scan symbols + SPY/VXX context tape
                                       │
             ┌─────────────────────────▼──────────────────────────┐
  LAYER 1    │  DATA INTEGRITY                                    │
             │  SIP condition-code filtering (price-forming vs    │
             │  volume-only vs excluded), out-of-sequence         │
             │  demotion, corrections/cancels, bounded queues,    │
             │  reconnect w/ lifetime-aware backoff, silence      │
             │  watchdog, REST gap backfill, single-instance lock │
             └─────────────────────────┬──────────────────────────┘
             ┌─────────────────────────▼──────────────────────────┐
  LAYER 2    │  AGGREGATION (all O(1) per print, fixed memory)    │
             │  1-second ring (600s) → 5/15/30/60s rolling sums,  │
             │  1m/5m bars, session VWAP, minute-of-day volume    │
             │  baselines from 5 historical sessions, premarket-  │
             │  seeded ring so velocity exists at the bell        │
             └───────────┬─────────────────────────┬──────────────┘
             ┌───────────▼───────────┐ ┌───────────▼──────────────┐
  LAYER 3    │  STRUCTURE            │ │  MARKET CONTEXT          │
             │  prior-day H/L/C,     │ │  SPY velocity (1m/5m,    │
             │  premarket H/L (SIP   │ │  direction-aligned),     │
             │  history), opening    │ │  VXX fear velocity       │
             │  range, swing pivots, │ │  (fast 1m + 5m trend).   │
             │  rolling ranges,      │ │  Features only — context │
             │  compression state,   │ │  informs the model, it   │
             │  sweep-and-reclaim    │ │  NEVER gates a signal    │
             └───────────┬───────────┘ └───────────┬──────────────┘
             ┌───────────▼─────────────────────────▼──────────────┐
  LAYER 4    │  SELECTION (per-symbol state machine)              │
             │  SCANNING → READY → EARLY_SIGNAL →                 │
             │            CONFIRMED | FAILED → COOLDOWN           │
             │  hard trigger gates → continuous quality score     │
             │  minus mined fake-start penalty → ML probability   │
             │  gate (with strong-evidence bypass) → rate caps    │
             └─────────────────────────┬──────────────────────────┘
             ┌─────────────────────────▼──────────────────────────┐
  LAYER 5    │  DELIVERY & LEARNING                               │
             │  Telegram (priority queue: EARLY jumps the line,   │
             │  retries, dedupe, <40 words) ─ then SQLite record, │
             │  5-minute outcome labeling, Adaptive Random Forest │
             │  learn_one, bounded threshold adaptation, nightly  │
             │  model persistence                                 │
             └────────────────────────────────────────────────────┘
```

The notification path is deliberately the shortest one in the system: a signal formats a string
and enqueues it before any database write, learning update, or follow-up analysis happens.

## The selection mechanism

Selection is a funnel where each stage has a distinct epistemological job. On a real session the
system processes **~9.4M events** (26k–50k events/sec in replay) and delivers a handful of alerts:

1. **Hard trigger gates — "did something objectively happen?"** Price must escape a real level
   (range, pivot, VWAP, prior-day/premarket extreme, opening range, or a sweep-and-reclaim) with
   directional velocity accelerating across 5s/15s windows, and volume accelerating against *two
   rulers*: the immediately preceding minute and the same-minute-of-day historical baseline.
   During the opening minutes both relative rulers are statistically self-defeating (see
   *What I learned*), so participation is judged against the symbol's typical **mid-day** tape
   instead. Persistence requirements (multiple directional prints spanning real time, no single
   print dominating) kill isolated-print artifacts.

2. **Continuous quality scoring — "how good is the evidence?"** Surviving candidates get a
   transparent 0–1 score aggregating all trigger evidence. A **fake-start penalty**, mined from
   labeled sessions, weighs against the score when a break shows chase signatures (a range still
   expanding rather than compressed, a tape already one-sided) — but damps to zero when volume is
   freshly exploding. Evidence weighs against evidence; nothing here is a cliff, so overwhelming
   trigger quality can outvote marginal staleness. Rejections caused by the penalty are
   attributed separately (`fake_start`) in the diagnostics.

3. **Learned probability gate — "does this look like past winners?"** The Adaptive Random Forest
   scores each candidate; below-threshold signals are *suppressed but still tracked and labeled*,
   so the model keeps learning about what it silenced. Two safety valves: a **strong-evidence
   bypass** (exceptional raw scores always deliver — market context informs, it never vetoes),
   and the entire gate deactivates automatically if its rolling precision deteriorates against a
   frozen rule-only baseline.

4. **Rate discipline — "respect the human."** Per-symbol daily caps, a global hourly budget, and
   a burst cap (the whole universe re-prices at the opening bell; three messages in a minute is
   signal, ten in six seconds is noise). Suppressed setups cool down for less time than delivered
   ones — nothing was sent, so there is no spam to throttle.

Follow-ups are equally strict: the verdict runs **one minute after the alert** and
**CONFIRMED requires real expansion progress** (a configurable fraction of the invalidation
distance beyond the trigger), not mere survival. Resolution-time environment — market
alignment, fear velocity, an event-volume news proxy — acts as a bounded tiebreaker on
marginal progress, and every follow-up carries a brief justification. "Held the level but
went nowhere" resolves as FAILED — and so does a pop whose **minute-scale trend still points
the other way**: the verdict reads an incrementally maintained, premarket-seeded EMA20 of
1-minute closes and refuses to confirm against its slope. That one condition came out of a
six-session study of EMA/ATR candidates in which everything else (EMA crosses, ATR-normalized
progress, extra model features) failed honest holdout tests; the survivor lifted the share of
confirmations still moving favorably an hour before the close from 53% to 59% on all six of
six sessions, roughly doubling the average post-confirmation move. Confirmations carry
market-context awareness: `…| against market, fear rising.`

**A second detector class covers the burst detector's blind band.** Instrumented replay of a
missed midday mover revealed moves that are neither micro-bursts nor grinds: 60–90-second
*escalators* at 1–2 bps/s on steadily building one-sided volume. A dedicated sustained-pressure
detector fires on meaningful net movement over ~75 s carried by dominant directional volume above
baseline while price prints new local extremes — evaluated at 1 Hz from the same O(1) aggregates,
capped at 2/symbol/day, sharing every cooldown and rate cap, and kept rule-pure until the forest
has learned the class (its signals are recorded and labeled like all others). On its first
evaluated session it went 2-for-4 on strict 1.5R labels with ~90% of the move still ahead on its
winners — versus ~34% for the burst class that day.

## ML training — online Adaptive Random Forest

The learning layer is [River](https://riverml.xyz)'s `ARFClassifier` — an online ensemble of
adaptive Hoeffding trees with per-tree drift detectors — one independent model per direction
(upside/downside), wrapped in an outer ADWIN concept-drift monitor and a dependency-free online
logistic fallback so the learning loop survives any environment.

**Training methodology (the part I'm most careful about):**

- **Real labels only.** Every fired signal freezes its ~22-feature vector at alert time; a
  5-minute outcome window labels it later (positive = reached ≥1.5× the invalidation distance
  before invalidation, expansion started within the window, and ≥50% of the move was still ahead
  at alert — *late alerts are labeled negative by construction*, so the model cannot become
  "accurate" by alerting after moves are obvious).
- **No lookahead, enforced.** The replay harness consumes events in strict timestamp order and
  raises on out-of-order input; `learn_one` only ever sees features from alert time plus the
  later label. Training replays run the *identical* live pipeline.
- **Train/holdout discipline.** The model trains sequentially over cached tick sessions
  (Mon–Thu, ~180 labeled signals from ~45M consolidated-SIP events) and is **frozen** for
  evaluation on the held-out Friday — a wrapper makes learning a no-op during evaluation so
  later symbols can't peek at earlier outcomes from the same day.
- **Threshold selection by sweep.** The suppression threshold is chosen from the holdout
  probability sweep (pass-rate / precision / winners-kept at each cut), not by feel. The
  production operating point cuts ~70% of rule-level signals.
- **Bounded self-modification.** The model may only nudge live thresholds through multipliers
  clamped to [0.75, 1.5] per symbol and time-of-day bucket; every signal records the model
  version and probability that judged it; the model persists at close and reloads at open; and a
  precision-collapse guardrail reverts all adaptation and disables ML influence until it re-earns
  trust. Cold start runs on transparent rules until a minimum label count is reached.

Honest status: with a few hundred labels the forest reliably *ranks volume* (cutting alert count
dramatically) but has not yet proven it beats the base rate on precision — which is exactly why
the guardrails, the bypass, and the shadow-learning of suppressed signals exist. Every live
session adds ~40–50 labels with full market-context features attached.

## What I learned from this project

The most valuable lessons came from instrumenting *why* the system stayed silent when it
shouldn't have. The flagship example: TSLA's +\$13 opening run produced **zero** signals, and the
cause turned out to be **five independent blockers stacked on top of each other**, each hidden
behind the previous one, found by monkey-patching the rejection path and logging every gate's
value at every evaluation:

1. The single-exchange feed had **no premarket tape** for TSLA → the premarket-high level that
   the breakout crossed didn't exist. (Fix: consolidated-SIP history for price levels — free on
   every plan — while volume baselines stay on the live feed for unit consistency.)
2. The 1-second ring was **empty at the bell** → velocity was incomputable for the first ~15
   seconds, precisely when the move happened. (Fix: seed the ring with the last premarket print.)
3. The anti-chase extension cap treated **gap-and-go speed as lateness**. (Fix: a wider, still
   bounded, opening-phase cap.)
4. **Both relative volume rulers self-defeat at the open** — the baseline's first minute contains
   the auction print, and the "prior 60 seconds" *is* the opening flood, so no burst can ever
   look accelerated against them. (Fix: judge opening participation against typical mid-day tape
   — 40× on that TSLA open.)
5. A genuine bug: the persistence check's flow memory held 64 prints ≈ 0.3s of dense opening tape
   — **shorter than the 0.6s span it demanded**, making the gate unpassable exactly on the most
   important tape. (Fix: size buffers for the densest tape, not the average.)

Broader lessons I'll carry to every future system:

- **Measure, don't theorize.** Every one of those five fixes came from reading actual gate values
  on actual tape, not from reasoning about what "should" happen. The same instrumentation later
  proved a suspected "duplicate process" was my own venv launcher's two-process anatomy.
- **Statistical rulers have domains of validity.** A comparison that is exactly right at 11:00
  (volume vs the prior minute) is structurally meaningless at 09:30. Regime-aware measurement
  beats one-size-fits-all thresholds.
- **Every filter kills some winners; make the trade visible.** Each gate ships with counted,
  attributed rejections, and calibration was done on train/holdout kill-ratios (fakes killed per
  winner killed), not on vibes.
- **Guard the learner against itself.** Frozen baselines, bounded adaptation, drift detection,
  revert-on-deterioration, and never training on post-alert information — online learning in
  production is mostly safety engineering.
- **The environment is part of the system.** Feed entitlements (SIP vs IEX), WebSocket
  subscription limits, clock skew, PowerShell 5.1 argument-splatting quirks, venv process
  anatomy, Task Scheduler behavior on failure — production reliability lives in these details as
  much as in the algorithm.
- **Asymmetries hide in defaults.** Long/short symmetry was a hard requirement: every feature is
  direction-aligned by multiplication, both directions get their own learner, and the label
  corpus is monitored for balance (currently ~48/52 up/down).

## Protected core

This repository is a **portfolio publication**: the full architecture, data engineering, ML
plumbing, replay/verification harness, operations tooling and test suite are real and complete.
Two modules ship as documented interfaces with the implementation withheld, and the config
thresholds are illustrative placeholders:

| Withheld | What it contains |
|---|---|
| `engine/state.py` decision methods | The calibrated gate sequence and firing/follow-up logic |
| `engine/features.py` scoring functions | Score weights and the mined fake-start penalty formula |
| `config.yaml` production values | Session-replay-calibrated thresholds |

Everything else — including the feature definitions the model learns from — is published as it
runs. The withheld pieces are the project's edge; the published pieces are the engineering.
(Consequently the live scanner and the two tests that exercise the detection core are not
runnable from this repository alone; the remaining 102 unit tests pass as-is.)

## Repository tour

```
src/early_trend_scanner/
  data/        stream client (reconnect, watchdog, gap recovery), REST client
               (retrying, paginated), SIP condition-code filtering
  engine/      1s aggregation + rolling windows, level book, compression &
               sweep tracking, market-context tracker, baselines, snapshot
               features, state machine (interface), symbol engine
  ml/          River ARF + builtin fallback, outcome labeler, bounded
               adaptive gate with revert guardrails
  replay/      no-lookahead replay harness, profile-scaled synthetic session
               generator, real-session replay (per-symbol, cached)
  notify/      Telegram: priority queue, retries, dedupe, <40-word formatting
  store/       SQLite persistence, bounded metrics (precision, lead time,
               remaining-move share, MFE/MAE by symbol/direction/hour/setup)
  app.py       session orchestration: market clock/calendar gating, warmup,
               in-session stream restarts, checkpointing, graceful teardown
tests/         102 tests: filters, aggregation, levels, context, ML, gate,
               labeler, telegram, clock/DST, reliability, instance lock
*.ps1          setup / run / health / scheduled-task installation (wake-to-run)
docs/          operational runbook
```

**Stack:** Python 3.12 · asyncio + aiohttp · River (online ML) · SQLite · Alpaca market data ·
Windows Task Scheduler + SetThreadExecutionState · ruff + mypy + pytest.

---

*Yasser Saber — [github.com/SaberYasser](https://github.com/SaberYasser)*
