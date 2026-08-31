"""Typed configuration: config.yaml for behavior, environment (.env) for secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

DEFAULT_SYMBOLS = [
    "TSLA",
    "NVDA",
    "AAPL",
    "PLTR",
    "AMD",
    "AMZN",
    "META",
    "MSFT",
    "GOOGL",
    "INTC",
    "MU",
    "AVGO",
    "MSTR",
    "HOOD",
]


@dataclass(frozen=True)
class DataCfg:
    feed: str = "sip"
    demo_mode: bool = False
    history_feed: str = "sip"  # price-level history (historical SIP is on every plan)
    trading_base_url: str = "https://paper-api.alpaca.markets"
    data_base_url: str = "https://data.alpaca.markets"
    stream_base_url: str = "wss://stream.data.alpaca.markets"
    baseline_sessions: int = 5
    gap_reconnect_s: float = 5.0
    recovery_min_gap_s: float = 2.0
    stream_restart_max: int = 3
    # Regime context tape: [market proxy, volatility proxy]. Streamed and
    # aggregated but never scanned for signals; feeds Snapshot features.
    context_symbols: tuple[str, ...] = ("SPY", "VXX")
    max_feed_latency_s: float = 3.0
    queue_trades: int = 20000
    queue_quotes: int = 10000


@dataclass(frozen=True)
class SessionCfg:
    start_lead_min: int = 8
    after_close: str = "exit"
    opening_range_min: int = 5


@dataclass(frozen=True)
class EngineCfg:
    ring_seconds: int = 600
    minutes_kept: int = 120
    ready_bps: float = 8.0
    break_min_bps: float = 3.0
    break_max_bps: float = 25.0
    break_max_open_bps: float = 40.0  # wider anti-chase cap during the opening minutes
    open_phase_min: float = 10.0  # minutes after the open treated as the opening phase
    vel_min_bps_s: float = 1.0
    vel15_max_bps_s: float = 2.0  # 15s velocity already hotter = the move is underway (chase)
    range_exp_max: float = 3.0  # 15s range already fully expanded = late entry
    accel_ratio: float = 1.10
    vol_accel_min: float = 2.0
    vol_base_min: float = 1.5
    baseline_skip_open_min: int = 2  # minute-of-day baseline is meaningless this early
    imb_min: float = 0.22
    persist_min_trades: int = 4
    persist_window_s: float = 2.5
    persist_min_span_s: float = 0.6
    min_trades_2s: int = 3
    single_print_max_frac: float = 0.70
    fresh_break_veto: bool = True
    fresh_break_vol_max: float = 6.0
    comp_veto_ratio: float = 1.0
    imb15_blowoff_max: float = 0.55
    fake_start_weight: float = 0.30
    score_min: float = 0.55
    sweep_window_s: float = 90.0
    sweep_max_bps: float = 20.0
    compression_max_ratio: float = 0.75
    compression_relax: float = 0.90
    invalidation_min_bps: float = 6.0
    open_range_floor_bps: float = 100.0  # opening-phase floor for the range feeding invalidation
    invalidation_range_frac: float = 0.35
    invalidation_max_bps: float = 40.0
    confirm_min_s: float = 60.0  # verdict ~1 minute after the alert (owner directive)
    confirm_min_r: float = 0.5  # progress beyond trigger required, in invalidation-distance units
    observe_max_s: float = 80.0
    fail_buffer_bps: float = 2.0
    cooldown_failed_s: float = 120.0
    cooldown_confirmed_s: float = 240.0
    cooldown_suppressed_s: float = 180.0  # setups the model silenced re-arm sooner
    re_arm_bps: float = 10.0
    max_alerts_symbol_day: int = 12
    max_alerts_hour_total: int = 30
    max_alerts_burst: int = 3  # global cap inside any rolling minute (bell clusters)
    # --- sustained-pressure ("trend onset") detector -----------------------
    # Catches the band between micro-burst and grind: a 60-90s escalator with
    # dominant one-sided volume making new local extremes. Rule-pure while the
    # model learns the class (never model-suppressed when ungated).
    trend_detector: bool = True
    trend_window_s: int = 75  # net-move lookback
    trend_min_bps: float = 25.0  # net directional move over the window
    trend_dir_share_min: float = 0.60  # one-sided share of classified volume (60s)
    trend_vol_base_min: float = 1.8  # 60s volume vs minute-of-day baseline
    trend_extreme_min: int = 15  # must make a new N-minute high/low
    trend_max_per_day: int = 2  # per symbol
    trend_model_gated: bool = False  # enable after the model has seen the class
    min_gap_same_symbol_s: float = 30.0


@dataclass(frozen=True)
class MlCfg:
    engine: str = "auto"
    min_labels: int = 40
    prob_gate_min: float = 0.45
    # Signals at/above this raw trigger score are never model-suppressed:
    # market context informs the model, but exceptional price+volume evidence
    # (e.g. a counter-market opening mover) must always reach the user.
    prob_bypass_score: float = 0.80
    outcome_window_s: int = 300
    pos_multiple: float = 1.5
    min_remaining_frac: float = 0.50
    bound_low: float = 0.75
    bound_high: float = 1.50
    adapt_step: float = 0.02
    revert_window: int = 30
    revert_precision_drop: float = 0.15
    checkpoint_min: int = 15


@dataclass(frozen=True)
class TelegramCfg:
    expected_username: str = "YourTelegramUsername"
    # Optional label prepended to every message (e.g. "DEMO "). Decoupled from
    # data.demo_mode at the owner's direction 2026-08-31: the feed flag stays
    # an explicit acknowledgment of single-exchange data, not a branding.
    prefix: str = ""
    max_retries: int = 4
    send_timeout_s: float = 5.0
    dedupe_size: int = 512


@dataclass(frozen=True)
class StorageCfg:
    db_path: str = "data/scanner.db"
    retention_days: int = 90
    status_path: str = "data/status.json"
    model_dir: str = "data/models"
    log_dir: str = "logs"
    log_level: str = "INFO"
    # Held for the process lifetime; a second scanner instance fails fast
    # instead of fighting for Alpaca's single-connection stream slot. 0 = off.
    instance_lock_port: int = 47113


@dataclass(frozen=True)
class Secrets:
    """Loaded from environment only. Never logged, never serialized."""

    alpaca_key: str = ""
    alpaca_secret: str = ""
    telegram_token: str = ""
    telegram_chat_id: str = ""

    def __repr__(self) -> str:  # defensive: keep secrets out of tracebacks/logs
        return "Secrets(<redacted>)"


@dataclass(frozen=True)
class Config:
    symbols: list[str] = field(default_factory=lambda: list(DEFAULT_SYMBOLS))
    data: DataCfg = field(default_factory=DataCfg)
    session: SessionCfg = field(default_factory=SessionCfg)
    engine: EngineCfg = field(default_factory=EngineCfg)
    ml: MlCfg = field(default_factory=MlCfg)
    telegram: TelegramCfg = field(default_factory=TelegramCfg)
    storage: StorageCfg = field(default_factory=StorageCfg)
    root: Path = field(default_factory=Path.cwd)

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p


T = TypeVar("T")


def _build(cls: type[T], raw: dict[str, Any], where: str) -> T:
    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - set(known)
    if unknown:
        raise ValueError(f"unknown config keys in '{where}': {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, val in raw.items():
        f = known[name]
        if is_dataclass(f.type) if isinstance(f.type, type) else False:
            kwargs[name] = _build(f.type, val or {}, f"{where}.{name}")  # type: ignore[arg-type]
        else:
            kwargs[name] = val
    return cls(**kwargs)


_SECTIONS: dict[str, type] = {
    "data": DataCfg,
    "session": SessionCfg,
    "engine": EngineCfg,
    "ml": MlCfg,
    "telegram": TelegramCfg,
    "storage": StorageCfg,
}


def load_config(config_path: Path | str, root: Path | None = None) -> Config:
    config_path = Path(config_path)
    raw: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    root = root or config_path.resolve().parent

    kwargs: dict[str, Any] = {"root": root}
    if "symbols" in raw:
        symbols = [str(s).upper() for s in raw["symbols"]]
        if not symbols:
            raise ValueError("config: symbols list is empty")
        kwargs["symbols"] = symbols
    for key, cls in _SECTIONS.items():
        if key in raw:
            kwargs[key] = _build(cls, raw[key] or {}, key)
    unknown = set(raw) - set(_SECTIONS) - {"symbols"}
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")

    cfg = Config(**kwargs)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    d = cfg.data
    if d.feed not in ("sip", "iex"):
        raise ValueError(f"data.feed must be 'sip' or 'iex', got {d.feed!r}")
    if d.history_feed not in ("sip", "iex"):
        raise ValueError(f"data.history_feed must be 'sip' or 'iex', got {d.history_feed!r}")
    if d.feed == "iex" and not d.demo_mode:
        raise ValueError(
            "data.feed=iex covers a single exchange and cannot produce accurate "
            "consolidated volume. Set data.demo_mode=true to acknowledge and "
            "run anyway, or use feed=sip."
        )
    if cfg.session.after_close not in ("exit", "pause"):
        raise ValueError("session.after_close must be 'exit' or 'pause'")
    if cfg.ml.engine not in ("auto", "river", "builtin"):
        raise ValueError("ml.engine must be auto|river|builtin")
    ctx = list(d.context_symbols)
    if any(s != str(s).upper() for s in ctx):
        raise ValueError("data.context_symbols must be uppercase tickers")
    overlap = set(ctx) & set(cfg.symbols)
    if overlap:
        raise ValueError(f"data.context_symbols overlap scan symbols: {sorted(overlap)}")
    if not 60 <= cfg.ml.outcome_window_s <= 300:
        raise ValueError("ml.outcome_window_s must be within 60..300 (1-5 minutes)")
    if cfg.engine.ring_seconds < cfg.ml.outcome_window_s + 60:
        raise ValueError("engine.ring_seconds must exceed ml.outcome_window_s by >= 60")
    if not 0 < cfg.ml.bound_low <= 1.0 <= cfg.ml.bound_high:
        raise ValueError("ml bounds must satisfy 0 < bound_low <= 1 <= bound_high")


def load_secrets() -> Secrets:
    """Read secrets from process environment (run.ps1 / dotenv load them first)."""
    return Secrets(
        alpaca_key=os.environ.get("APCA_API_KEY_ID", "").strip(),
        alpaca_secret=os.environ.get("APCA_API_SECRET_KEY", "").strip(),
        telegram_token=os.environ.get("TELEGRAM_BOT_TOKEN", "").strip(),
        telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID", "").strip(),
    )
