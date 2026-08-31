from pathlib import Path

import pytest

from early_trend_scanner.config import Config, load_config, load_secrets


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_defaults_without_file(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "missing.yaml", root=tmp_path)
    assert cfg.data.feed == "sip"
    assert len(cfg.symbols) == 14
    assert "TSLA" in cfg.symbols and "HOOD" in cfg.symbols


def test_load_real_project_config() -> None:
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config.yaml", root=root)
    assert cfg.symbols[0] == "TSLA"
    assert cfg.engine.ring_seconds >= cfg.ml.outcome_window_s + 60


def test_iex_requires_demo_mode(tmp_path: Path) -> None:
    p = write(tmp_path, "data:\n  feed: iex\n")
    with pytest.raises(ValueError, match="demo_mode"):
        load_config(p, root=tmp_path)
    p2 = write(tmp_path, "data:\n  feed: iex\n  demo_mode: true\n")
    cfg = load_config(p2, root=tmp_path)
    assert cfg.data.demo_mode is True


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(write(tmp_path, "engine:\n  breka_min_bps: 3\n"), root=tmp_path)
    with pytest.raises(ValueError, match="unknown top-level"):
        load_config(write(tmp_path, "engnie: {}\n"), root=tmp_path)


def test_outcome_window_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outcome_window_s"):
        load_config(write(tmp_path, "ml:\n  outcome_window_s: 900\n"), root=tmp_path)


def test_symbols_uppercased(tmp_path: Path) -> None:
    cfg = load_config(write(tmp_path, "symbols: [tsla, nvda]\n"), root=tmp_path)
    assert cfg.symbols == ["TSLA", "NVDA"]


def test_secrets_never_repr(monkeypatch) -> None:
    monkeypatch.setenv("APCA_API_KEY_ID", "AKSECRETSECRET")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:tok")
    s = load_secrets()
    assert s.alpaca_key == "AKSECRETSECRET"
    assert "AKSECRETSECRET" not in repr(s)
    assert "tok" not in repr(s)


def test_path_resolution(tmp_path: Path) -> None:
    cfg = Config(root=tmp_path)
    assert cfg.path("data/scanner.db") == tmp_path / "data" / "scanner.db"
