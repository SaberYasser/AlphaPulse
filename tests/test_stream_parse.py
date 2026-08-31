"""Stream message dispatch without a network: parsing, queues, error codes."""

from __future__ import annotations

import asyncio

import pytest

from early_trend_scanner.data.models import Quote, Trade
from early_trend_scanner.data.stream import AlpacaStream, StreamAuthError


def make_stream(trade_size: int = 100, quote_size: int = 100) -> AlpacaStream:
    return AlpacaStream(
        key="k",
        secret="s",
        feed="sip",
        symbols=["TSLA"],
        trade_queue=asyncio.Queue(maxsize=trade_size),
        quote_queue=asyncio.Queue(maxsize=quote_size),
    )


def test_trade_and_quote_parsing() -> None:
    s = make_stream()
    s._dispatch(
        [
            {
                "T": "t",
                "S": "TSLA",
                "i": 5,
                "x": "V",
                "p": 350.22,
                "s": 130,
                "t": "2026-08-28T14:30:01.123456789Z",
                "c": ["@", "I"],
                "z": "C",
            },
            {
                "T": "q",
                "S": "TSLA",
                "bx": "V",
                "bp": 350.20,
                "bs": 4,
                "ax": "V",
                "ap": 350.24,
                "as": 6,
                "t": "2026-08-28T14:30:01.2Z",
            },
        ]
    )
    trade = s.trade_queue.get_nowait()
    assert isinstance(trade, Trade)
    assert trade.price == 350.22 and trade.size == 130
    assert trade.conditions == ("@", "I")
    quote = s.quote_queue.get_nowait()
    assert isinstance(quote, Quote)
    assert quote.bid == 350.20 and quote.ask == 350.24
    assert s.latency_ewma_s > 0.0
    assert s.last_event_ts > 0.0


def test_zero_bid_quote_skipped() -> None:
    s = make_stream()
    s._dispatch(
        [
            {
                "T": "q",
                "S": "TSLA",
                "bp": 0.0,
                "bs": 0,
                "ap": 350.0,
                "as": 1,
                "t": "2026-08-28T14:30:01Z",
            },
        ]
    )
    assert s.quote_queue.empty()


def test_queue_overflow_counts_drops() -> None:
    s = make_stream(trade_size=2, quote_size=1)
    msg = {
        "T": "t",
        "S": "TSLA",
        "i": 1,
        "p": 1.0,
        "s": 1,
        "t": "2026-08-28T14:30:01Z",
        "c": [],
        "z": "C",
    }
    s._dispatch([msg, msg, msg, msg])
    assert s.trade_queue.qsize() == 2
    assert s.dropped_trades == 2


def test_auth_error_codes_fatal() -> None:
    s = make_stream()
    with pytest.raises(StreamAuthError, match="check API keys"):
        s._dispatch([{"T": "error", "code": 402, "msg": "auth failed"}])
    with pytest.raises(StreamAuthError, match="subscription"):
        s._dispatch([{"T": "error", "code": 409, "msg": "insufficient subscription"}])


def test_connection_limit_retryable() -> None:
    s = make_stream()
    with pytest.raises(ConnectionError, match="connection limit"):
        s._dispatch([{"T": "error", "code": 406, "msg": "connection limit exceeded"}])


def test_corrections_and_cancels_ignored_gracefully() -> None:
    s = make_stream()
    s._dispatch(
        [
            {"T": "c", "S": "TSLA"},
            {"T": "x", "S": "TSLA"},
            {"T": "success", "msg": "authenticated"},
            {"T": "subscription", "trades": ["TSLA"]},
        ]
    )
    assert s.trade_queue.empty() and s.quote_queue.empty()
