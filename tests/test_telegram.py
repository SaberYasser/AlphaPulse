import asyncio

from early_trend_scanner.engine.state import Signal
from early_trend_scanner.notify.telegram import TelegramNotifier, format_message

T0 = 1_756_000_000.0


def sig(direction: int = 1) -> Signal:
    s = Signal(
        signal_id="TSLA-1-U",
        symbol="TSLA",
        direction=direction,
        alert_ts=T0,
        alert_price=350.30,
        trigger_price=350.22,
        level_kind=0,
        trigger_verb="break",
        invalidation=349.88,
        vol_ratio=2.1,
        features={},
    )
    s.micro_extreme = 350.61
    s.resolution_ts = T0 + 11
    return s


def words(text: str) -> int:
    return len(text.split())


def test_early_format_under_40_words() -> None:
    text = format_message(sig(), "EARLY", {})
    assert words(text) < 40
    for token in (
        "EARLY",
        "UP",
        "TSLA",
        "ET",
        "350.22",
        "break",
        "2.1x",
        "invalidation",
        "349.88",
        "1-5m",
    ):
        assert token in text, f"{token} missing from: {text}"


def test_confirmed_format() -> None:
    text = format_message(sig(), "CONFIRMED", {})
    assert words(text) < 40
    assert "held 350.22" in text and "micro-high 350.61" in text


def test_failed_format() -> None:
    text = format_message(sig(), "FAILED", {"reason": "directional volume reversed"})
    assert words(text) < 40
    assert "lost 350.22 trigger" in text and "directional volume reversed" in text


def test_down_direction_and_demo_prefix() -> None:
    s = sig(direction=-1)
    text = format_message(s, "EARLY", {}, prefix="DEMO ")
    assert text.startswith("DEMO EARLY DOWN TSLA")
    c = format_message(s, "CONFIRMED", {})
    assert "micro-low" in c


def test_disabled_notifier_captures() -> None:
    n = TelegramNotifier(token="", chat_id="", enabled=False)
    n.publish_signal(sig(), "EARLY", {})
    assert len(n.captured) == 1 and "EARLY" in n.captured[0]


def test_duplicate_prevention() -> None:
    n = TelegramNotifier(token="", chat_id="", enabled=False)
    s = sig()
    n.publish_signal(s, "EARLY", {})
    n.publish_signal(s, "EARLY", {})
    n.publish_signal(s, "CONFIRMED", {})
    assert len(n.captured) == 2  # EARLY once + CONFIRMED once


async def test_retry_then_success(monkeypatch) -> None:
    """The real retry loop against a fake HTTP layer: two 500s then success."""
    n = TelegramNotifier(token="t", chat_id="1", max_retries=4)

    class FakeResp:
        def __init__(self, status: int) -> None:
            self.status = status

        async def json(self, content_type=None):
            return {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, url, json=None, timeout=None):
            self.calls += 1
            return FakeResp(500 if self.calls < 3 else 200)

        async def close(self) -> None:
            pass

    fake = FakeSession()
    n._session = fake  # type: ignore[assignment]
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    await n._send_with_retry("hello")
    assert fake.calls == 3
    assert n.sent_count == 1 and n.failed_count == 0


async def _instant_sleep(_delay: float) -> None:
    return None


async def test_gives_up_after_max_retries(monkeypatch) -> None:
    n = TelegramNotifier(token="t", chat_id="1", max_retries=3)

    class DeadResp:
        status = 500

        async def json(self, content_type=None):
            return {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class DeadSession:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, url, json=None, timeout=None):
            self.calls += 1
            return DeadResp()

        async def close(self) -> None:
            pass

    dead = DeadSession()
    n._session = dead  # type: ignore[assignment]
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
    await n._send_with_retry("hello")
    assert dead.calls == 3
    assert n.failed_count == 1


async def test_non_retryable_4xx_stops_immediately() -> None:
    n = TelegramNotifier(token="t", chat_id="1", max_retries=5)

    class BadReqResp:
        status = 400

        async def json(self, content_type=None):
            return {"description": "chat not found"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class S:
        def __init__(self) -> None:
            self.calls = 0

        def post(self, url, json=None, timeout=None):
            self.calls += 1
            return BadReqResp()

        async def close(self) -> None:
            pass

    s = S()
    n._session = s  # type: ignore[assignment]
    await n._send_with_retry("hello")
    assert s.calls == 1 and n.failed_count == 1


def test_confirmed_carries_market_context() -> None:
    s = sig()
    s.features["mkt_al_5m"] = -1.2  # market moving against the signal
    s.features["fear_5m"] = 0.8  # fear rising
    msg = format_message(s, "CONFIRMED", {})
    assert "against market, fear rising" in msg
    assert len(msg.split()) < 40

    s.features["mkt_al_5m"] = 0.0  # neutral context stays silent
    s.features["fear_5m"] = 0.0
    assert "market" not in format_message(s, "CONFIRMED", {})
