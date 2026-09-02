"""Defensive logging redaction."""

from __future__ import annotations

import logging

from early_trend_scanner.logging_setup import _mask_token, _RedactFilter


def test_mask_token_redacts_every_telegram_url() -> None:
    message = (
        "first https://api.telegram.org/bot123456:secret/sendMessage "
        "second https://api.telegram.org/bot999:other/getMe"
    )

    masked = _mask_token(message)

    assert "secret" not in masked
    assert "other" not in masked
    assert masked.count("<token>") == 2


def test_redaction_filter_handles_logging_arguments() -> None:
    record = logging.LogRecord(
        "test",
        logging.ERROR,
        __file__,
        1,
        "request failed: %s",
        ("https://api.telegram.org/bot123:secret/sendMessage",),
        None,
    )

    assert _RedactFilter().filter(record) is True
    assert "secret" not in record.getMessage()
    assert "<token>" in record.getMessage()
