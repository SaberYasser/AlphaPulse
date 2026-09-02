# early_trend_scanner

Detects the earliest usable sign that one of 13 liquid US stocks and ETFs
(TSLA, NVDA, AAPL, QQQ, SPY, PLTR, AMD, AMZN, GOOGL, INTC, AVGO, MSTR, HOOD) is
beginning a potentially substantial 1–5 minute expansion — **up or down,
symmetrically** — and immediately sends a short Telegram signal.

Built for Windows, runs only during official U.S. regular trading hours,
learns online from every completed signal, and keeps memory bounded.

## How it works

```
Alpaca SIP WebSocket (trades + NBBO quotes)
        │  condition-code filtering, out-of-sequence demotion
        ▼
1-second local aggregates ──► 1m/5m bars, session VWAP, opening range
        │                      prior-day & premarket levels, swings,
        │                      rolling ranges, compression state
        ▼
per-symbol state machine:  SCANNING → READY → EARLY_SIGNAL → CONFIRMED/FAILED → COOLDOWN
        │
        ├─ EARLY fires the instant price escapes a level WITH volume/velocity
        │  acceleration and directional flow — never waits for a candle close
        ▼
Telegram (priority queue, retries, dedupe)  +  SQLite record
        ├─ 5m expansion outcome → River model + bounded adaptive gate
        └─ CONFIRMED price → session close outcome → efficacy model (≥40 labels to gate)
```

**Price trigger**: break/cross of a recent range, swing pivot, VWAP,
prior-day/premarket/opening-range level, or a sweep-and-reclaim — with
accelerating directional velocity (1s/5s/15s windows).

**Volume trigger**: 5-second volume vs the prior 60 seconds AND vs the
same-minute-of-day baseline built from ≥5 completed sessions; signed
(ask-initiated vs bid-initiated) imbalance must point the same way over 5
seconds and retain aligned participation over 15 seconds for micro-breaks.

**Persistence**: several qualifying prints across ≥ ~1 s — one anomalous print
never alerts. Alerts are rejected when volume doesn't accelerate, when price is
already extended past the trigger, or when the feed is disconnected/stale.

**Quality layer (evidence, not cliffs)**: candidates that survive the hard
integrity and trigger gates are decided by a transparent weighted score. A
*fake-start penalty*, mined from labeled real sessions (2026-08-24..28),
weighs against the score when the break shows chase evidence — a range still
expanding instead of compressed, or a 15-second tape already one-sided —
damped to zero as 5-second volume acceleration rises (a fresh volume
explosion is never penalized). Strong trigger evidence can outvote marginal
staleness; when the penalty is what sinks a candidate, it is counted as a
`fake_start` rejection. The trained model's probability then gates the
survivors (`ml.prob_gate_min`), suppressing low-conviction alerts while still
tracking and labeling them so learning continues.

**Session efficacy layer**: the north-star metric is the share of delivered,
CONFIRMED signals whose price moves in the signaled direction from the
confirmation price to the official session close. A separate River model
learns that exact label and may gate below `ml.efficacy_prob_gate_min` only
after `ml.efficacy_min_labels` outcomes; this avoids pretending one session is
enough training data. Raw trigger scores above `ml.prob_bypass_score` retain
the existing exceptional-evidence bypass.

**Learning**: each signal stores its compact feature vector at alert time; the
outcome window labels it (positive = reached ≥1.5× the invalidation distance
before invalidation, expansion started inside the window, and ≥50% of the move
was still ahead at alert time — late alerts are penalized). A River **Adaptive
Random Forest** per direction (10 adaptive Hoeffding trees, the strongest free
streaming classifier River offers; builtin online-logistic fallback if River
is absent) learns from every label with ADWIN drift detection; it may only
nudge thresholds inside hard bounds `[0.75, 1.5]` and is auto-reverted if
rolling precision deteriorates vs the frozen rule-only baseline.

## Requirements

- Windows 10/11, Python **3.11+**
- Alpaca account & API keys. **SIP consolidated data requires a paid Alpaca
  market-data subscription (Algo Trader Plus).** The free IEX feed covers one
  exchange only and produces wrong consolidated volume — the scanner refuses
  to run on IEX unless you explicitly opt into labeled `DEMO` mode.
- A Telegram bot (free, via @BotFather) and the @YassirSaber account.

## Install (exact commands)

```powershell
# 0) if Python is missing:
winget install --id Python.Python.3.12 --scope user

# 1) from the project folder:
cd $env:USERPROFILE\Documents\early_trend_scanner
.\setup.ps1               # creates .venv, installs pinned deps, creates .env

# 2) put your keys in .env  (never committed; do not share)
notepad .env              # APCA_API_KEY_ID, APCA_API_SECRET_KEY, TELEGRAM_BOT_TOKEN

# 3) Telegram wiring (one time):
#    - @BotFather -> /newbot -> paste token into .env
#    - from the @YassirSaber account, open the bot and press START
.\telegram_test.ps1 -Setup   # finds the numeric chat id, saves TELEGRAM_CHAT_ID, sends a test

# 4) sanity checks (offline + online):
.\run.ps1 replay             # deterministic synthetic replay, prints metrics
.\run.ps1 clock              # verifies market clock/calendar against Alpaca
.\run.ps1 power-selftest     # verifies sleep-prevention acquire/release

# 5) schedule daily start (~09:20 ET, wakes the PC):
.\install_task.ps1
```

### Run manually

```powershell
.\run.ps1                    # live scanner; exits after the close
.\run.ps1 status             # heartbeat JSON (state, latency, alerts, RSS)
.\healthcheck.ps1            # human-friendly health summary + exit code
```

### Demo mode (no paid data plan)

In `config.yaml` set:

```yaml
data:
  feed: iex
  demo_mode: true
```

Every notification is then prefixed `DEMO`. Volume-based quality will be
degraded — do not treat DEMO signals as tradeable.

## Telegram messages (max 3 per setup, each < 40 words)

```
EARLY UP TSLA 10:32:14 ET | 350.22 break | volume 2.1x | velocity accelerating | invalidation 349.88 | possible 1-5m expansion.
CONFIRMED UP TSLA 10:32:25 ET | held 350.22 | volume sustained | new micro-high 350.61.
FAILED UP TSLA 10:32:23 ET | lost 350.22 trigger | directional volume reversed.
```

One EARLY, then exactly one follow-up (CONFIRMED or FAILED) after 15–50 s.
A configurable cooldown plus a "new structure required" rule prevents repeats
off the same level.

## Replay (no lookahead)

```powershell
.\run.ps1 replay                                        # synthetic, offline, deterministic
.\run.ps1 replay --date 2026-08-27 --symbols TSLA,NVDA --start-min 0 --end-min 90
```

Real-date replay pulls historical SIP trades+quotes over REST, feeds them
through the *identical* live pipeline in timestamp order, and prints precision,
false-alert rate, median lead time, median share-of-move-remaining, confirm
rate and per-symbol/direction slices. Keep windows modest — full-day tick data
for the full universe is heavy over REST.

## Operations

- **Task Scheduler**: `install_task.ps1` registers *EarlyTrendScanner*
  (Mon–Fri, ~09:20 ET local equivalent, "Wake the computer to run this task"
  enabled, runs on battery, and retries unexpected failures up to three times
  at one-minute intervals). The app itself decides everything from the Alpaca
  clock/calendar: holidays and early closes are respected, DST is handled in
  `America/New_York`, and on non-trading days it exits immediately.
  Re-run `install_task.ps1` after a DST switch for exact timing (the app
  tolerates an off-by-an-hour trigger by waiting/catching up).
- **Sleep**: while the market is open the process holds
  `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` — the machine
  won't idle-sleep, but deliberate sleep/shutdown/lid actions still work.
  The flag is released at the close (verify with `run.ps1 power-selftest`).
- **At the close**: streams unsubscribe, pending outcomes are labeled, the
  confirmation-to-close efficacy outcomes are evaluated, the recap is sent,
  both models + adaptive gate + daily metrics are persisted, the sleep flag
  is released, and the process exits (`session.after_close: pause` keeps a
  zero-work idle process instead).
- **Resilience**: automatic WS reconnect with exponential backoff + jitter,
  silence watchdog (5 s), REST backfill of trade gaps > 2 s (alerts suppressed
  during catch-up), bounded queues everywhere, rotating logs in `logs/`,
  Telegram retry with 429 handling and duplicate suppression, feed-latency
  monitoring (alerts pause when the feed lags > 3 s), model checkpoint every
  15 min.
- **Memory**: raw ticks are never retained; per symbol only a 600-second ring
  of 1-second aggregates, ≤420 one-minute bars, a 64-print flow window and a
  handful of levels. SQLite holds compact signal rows with a 90-day retention.
  Live RSS is reported in `status.json` / `healthcheck.ps1`.

## Configuration

Everything lives in `config.yaml` (thresholds, cooldowns, ML bounds, caps —
all documented inline). Secrets live only in `.env`:

| variable | purpose |
|---|---|
| `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` | Alpaca market data + clock |
| `TELEGRAM_BOT_TOKEN` | bot from @BotFather |
| `TELEGRAM_CHAT_ID` | numeric id discovered by `telegram_test.ps1 -Setup` |

Credentials are never logged; the log layer additionally masks anything that
looks like a bot-token URL.

## Development

```powershell
.\.venv\Scripts\python.exe -m pytest -q          # unit + replay tests (offline)
.\.venv\Scripts\python.exe -m ruff format .      # formatting
.\.venv\Scripts\python.exe -m ruff check .       # lint
.\.venv\Scripts\python.exe -m mypy               # type check
```

Project layout: `src/early_trend_scanner/` (package), `tests/` (pytest,
no network needed), PowerShell entry points at the repo root. The repo is
GitHub-ready: MIT licensed, only free/open-source dependencies, secrets and
runtime state are gitignored.

## Disclaimers

Signals are informational pattern detections, not investment advice. Past
detection quality does not guarantee future results. DEMO/IEX mode is for
evaluation only.
