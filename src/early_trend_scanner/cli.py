"""Command-line entry points: run, replay, clock, telegram setup/test, self-tests."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

from . import __version__
from .config import Config, load_config, load_secrets
from .logging_setup import setup_logging
from .timeutil import et_dt

log = logging.getLogger(__name__)


def _project_root() -> Path:
    # src/early_trend_scanner/cli.py -> project root two levels up from src/
    return Path(__file__).resolve().parents[2]


def _load_env(root: Path) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env")
    except ImportError:
        pass


def _load(config_arg: str | None) -> Config:
    root = _project_root()
    _load_env(root)
    cfg_path = Path(config_arg) if config_arg else root / "config.yaml"
    return load_config(cfg_path, root=root)


# ---------------------------------------------------------------------- run


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    setup_logging(cfg.path(cfg.storage.log_dir), cfg.storage.log_level, console=True)
    log.info(
        "early_trend_scanner %s starting (feed=%s demo=%s symbols=%d)",
        __version__,
        cfg.data.feed,
        cfg.data.demo_mode,
        len(cfg.symbols),
    )
    from .app import ScannerApp

    app = ScannerApp(cfg, load_secrets())
    try:
        return asyncio.run(app.run())
    except KeyboardInterrupt:
        log.info("interrupted — shutting down")
        return 0


# -------------------------------------------------------------------- replay


def cmd_replay(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    from .replay.engine import ReplayRunner, print_report

    if args.date:
        setup_logging(cfg.path(cfg.storage.log_dir), "INFO", console=True)
        from .replay.real import replay_real

        secrets = load_secrets()
        if not secrets.alpaca_key:
            print("APCA_API_KEY_ID / APCA_API_SECRET_KEY required for real-data replay")
            return 2
        if args.feed:
            from dataclasses import replace

            cfg = replace(cfg, data=replace(cfg.data, feed=args.feed))
        symbols = [s.strip().upper() for s in (args.symbols or "TSLA,NVDA").split(",")]
        result = asyncio.run(
            replay_real(
                cfg,
                secrets,
                date.fromisoformat(args.date),
                symbols,
                start_min=args.start_min,
                end_min=args.end_min,
                with_quotes=not args.no_quotes,
            )
        )
        print_report(result, f"real replay {args.date} feed={cfg.data.feed} {symbols}")
        return 0
    setup_logging(cfg.path(cfg.storage.log_dir), "WARNING", console=True)

    # Synthetic deterministic session (no credentials, offline).
    from .data.models import SessionInfo
    from .replay.synthetic import SyntheticSession

    synth = SyntheticSession(seed=args.seed, engine_cfg=cfg.engine)
    session = SessionInfo("synthetic", synth.open_ts, synth.close_ts)
    symbols = [s.symbol for s in synth.script]
    runner = ReplayRunner(cfg, session, symbols)
    result = runner.run(synth.events())
    print_report(result, f"synthetic replay (seed={args.seed})")

    truths = {t.symbol: t for t in synth.truths}
    print("\n--- ground truth vs detection ---")
    ok = True
    for sym, truth in truths.items():
        sigs = [s for s in result.signals if s.symbol == sym and not s.suppressed]
        if truth.kind == "quiet":
            verdict = "OK (no alert)" if not sigs else f"FALSE ALERTS: {len(sigs)}"
            ok = ok and not sigs
        else:
            hit = [
                s
                for s in sigs
                if s.direction == truth.direction
                and truth.event_ts - 5 <= s.alert_ts <= truth.move_end_ts
            ]
            if hit:
                first = min(hit, key=lambda s: s.alert_ts)
                delay = first.alert_ts - truth.event_ts
                verdict = f"detected {delay:.1f}s after expansion start"
            else:
                verdict = "MISSED"
                ok = ok and truth.kind == "fakeout_up"
        print(f"{sym:6s} {truth.kind:14s} -> {verdict}")
    return 0 if ok else 1


# --------------------------------------------------------------------- clock


def cmd_clock(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    secrets = load_secrets()
    if not secrets.alpaca_key:
        print("APCA_API_KEY_ID / APCA_API_SECRET_KEY not set")
        return 2

    async def go() -> int:
        from .clock import MarketClock
        from .data.rest import AlpacaRest

        async with AlpacaRest(
            secrets.alpaca_key,
            secrets.alpaca_secret,
            cfg.data.data_base_url,
            cfg.data.trading_base_url,
        ) as rest:
            clock = MarketClock(rest)
            state = await clock.fetch()
            sessions = await clock.load_sessions()
            print(f"server time : {et_dt(state.now_ts):%Y-%m-%d %H:%M:%S} ET")
            print(f"market open : {state.is_open}")
            print(f"next open   : {et_dt(state.next_open_ts):%Y-%m-%d %H:%M:%S} ET")
            print(f"next close  : {et_dt(state.next_close_ts):%Y-%m-%d %H:%M:%S} ET")
            print(f"clock skew  : {state.skew_s:+.2f}s vs local")
            et_today = f"{et_dt(state.now_ts):%Y-%m-%d}"
            today = next((s for s in sessions if s.date_str == et_today), None)
            if today:
                print(
                    f"session     : {today.date_str} "
                    f"{et_dt(today.open_ts):%H:%M}-{et_dt(today.close_ts):%H:%M} ET "
                    f"({today.minutes} min{' — early close' if today.minutes < 390 else ''})"
                )
            else:
                print("session     : none today (holiday/weekend)")
            print(f"calendar    : {len(sessions)} sessions loaded around today")
        return 0

    return asyncio.run(go())


# ------------------------------------------------------------------ telegram


def cmd_telegram_setup(args: argparse.Namespace) -> int:
    root = _project_root()
    _load_env(root)
    cfg = _load(args.config)
    import os

    from .notify.telegram import TelegramNotifier, discover_chat_id

    expected = cfg.telegram.expected_username
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("TELEGRAM_BOT_TOKEN is not set.")
        print("1. In Telegram, talk to @BotFather -> /newbot -> copy the token.")
        print("2. Put it in .env as TELEGRAM_BOT_TOKEN=... and re-run this command.")
        return 2

    print(f"Now, FROM THE @{expected} ACCOUNT, open the bot in Telegram and send: /start")
    try:
        input("Press Enter here once /start has been sent... ")
    except EOFError:
        # Non-interactive shell (Windows NUL stdin still claims to be a tty).
        print("(non-interactive shell — assuming /start was already sent)")

    async def go() -> int:
        import aiohttp

        chat_id = None
        seen: list[str] = []
        for attempt in range(6):
            try:
                chat_id, seen = await discover_chat_id(token, expected)
            except (TimeoutError, aiohttp.ClientError, OSError) as exc:
                print(f"  network hiccup ({type(exc).__name__}) — retrying ({attempt + 1}/6)")
                await asyncio.sleep(5)
                continue
            if chat_id:
                break
            print(f"  not found yet (saw: {seen or 'nobody'}) — retrying ({attempt + 1}/6)")
            await asyncio.sleep(5)
        if not chat_id:
            print(f"Could not find a private chat from @{expected}.")
            print("Make sure that exact account pressed Start on the bot, then re-run.")
            return 1
        print(f"Verified @{expected}; numeric chat id: {chat_id}")
        _write_env_var(root / ".env", "TELEGRAM_CHAT_ID", chat_id)
        print("Saved TELEGRAM_CHAT_ID to .env")
        notifier = TelegramNotifier(token=token, chat_id=chat_id)
        if await notifier.send_test():
            print("Test message delivered. Telegram is ready.")
            return 0
        print("Test message FAILED — see logs.")
        return 1

    return asyncio.run(go())


def _write_env_var(env_path: Path, key: str, value: str) -> None:
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        example = env_path.with_name(".env.example")
        if example.exists():
            lines = example.read_text(encoding="utf-8").splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_telegram_test(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    secrets = load_secrets()
    if not secrets.telegram_token or not secrets.telegram_chat_id:
        print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — run: ets telegram-setup")
        return 2
    from .notify.telegram import TelegramNotifier

    notifier = TelegramNotifier(
        token=secrets.telegram_token,
        chat_id=secrets.telegram_chat_id,
        prefix="DEMO " if cfg.data.demo_mode else "",
    )
    ok = asyncio.run(notifier.send_test())
    print("Telegram test message sent." if ok else "Telegram test FAILED.")
    return 0 if ok else 1


# ---------------------------------------------------------------- self tests


def cmd_power_selftest(args: argparse.Namespace) -> int:
    from .power import KeepAwake

    ka = KeepAwake()
    print(f"platform: {sys.platform}")
    ka.acquire()
    print(f"acquired: active={ka.active} (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)")
    time.sleep(1.0)
    ka.release()
    print(f"released: active={ka.active} (ES_CONTINUOUS)")
    print("sleep-prevention selftest OK")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = _load(args.config)
    from .status import StatusWriter

    payload = StatusWriter.read(cfg.path(cfg.storage.status_path))
    if payload is None:
        print("no status file yet — is the scanner running?")
        return 1
    age = time.time() - float(payload.get("written_at", 0))
    payload["_age_s"] = round(age, 1)
    print(json.dumps(payload, indent=2))
    return 0


# ---------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ets", description="early trend scanner")
    p.add_argument("--config", help="path to config.yaml", default=None)
    p.add_argument("--version", action="version", version=f"early-trend-scanner {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="run the live scanner (regular hours only)")
    rp = sub.add_parser("replay", help="replay a session without lookahead")
    rp.add_argument("--date", help="YYYY-MM-DD real session (requires Alpaca keys)")
    rp.add_argument("--symbols", help="comma list for real replay (default TSLA,NVDA)")
    rp.add_argument("--start-min", type=int, default=0, help="minutes after open")
    rp.add_argument("--end-min", type=int, default=120, help="minutes after open")
    rp.add_argument("--seed", type=int, default=7, help="seed for synthetic replay")
    rp.add_argument("--feed", choices=["sip", "iex"], help="historical feed override")
    rp.add_argument(
        "--no-quotes",
        action="store_true",
        help="skip NBBO quotes (tick-rule sides; much lighter for full days)",
    )
    sub.add_parser("clock", help="show Alpaca market clock/calendar")
    sub.add_parser("telegram-setup", help="discover and store the numeric chat id")
    sub.add_parser("telegram-test", help="send a test message")
    sub.add_parser("power-selftest", help="verify sleep-prevention acquire/release")
    sub.add_parser("status", help="print the live status heartbeat")

    args = p.parse_args(argv)
    handlers = {
        "run": cmd_run,
        "replay": cmd_replay,
        "clock": cmd_clock,
        "telegram-setup": cmd_telegram_setup,
        "telegram-test": cmd_telegram_test,
        "power-selftest": cmd_power_selftest,
        "status": cmd_status,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
