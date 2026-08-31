"""Rotating file + console logging. Secrets never reach the log layer by design."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FMT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


class _RedactFilter(logging.Filter):
    """Belt-and-braces: masks anything that looks like a Telegram bot token in a URL."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "api.telegram.org/bot" in msg:
            record.msg = _mask_token(msg)
            record.args = ()
        return True


def _mask_token(msg: str) -> str:
    out: list[str] = []
    i = 0
    marker = "api.telegram.org/bot"
    while True:
        j = msg.find(marker, i)
        if j == -1:
            out.append(msg[i:])
            break
        j_end = j + len(marker)
        out.append(msg[i:j_end])
        k = j_end
        while k < len(msg) and msg[k] not in "/ \"'":
            k += 1
        out.append("<token>")
        i = k
    return "".join(out)


def setup_logging(log_dir: Path, level: str = "INFO", console: bool = True) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)

    fh = RotatingFileHandler(
        log_dir / "scanner.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter(_FMT))
    fh.addFilter(_RedactFilter())
    root.addHandler(fh)

    if console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(logging.Formatter(_FMT))
        ch.addFilter(_RedactFilter())
        root.addHandler(ch)

    logging.getLogger("aiohttp").setLevel(logging.WARNING)
