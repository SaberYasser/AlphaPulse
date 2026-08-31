"""REST client retry behavior: timeouts and transient errors must not escape."""

from __future__ import annotations

import pytest

from early_trend_scanner.data import rest as rest_mod
from early_trend_scanner.data.rest import AlpacaRest, AlpacaRestError


class _Resp:
    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.headers: dict[str, str] = {}

    async def json(self):
        return {"ok": True}

    async def text(self):
        return "err"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _Session:
    """Raises TimeoutError for the first `fail_times` calls, then succeeds."""

    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def get(self, url, params=None, headers=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TimeoutError("stalled connection")
        return _Resp()


@pytest.fixture()
def fast_sleep(monkeypatch):
    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(rest_mod.asyncio, "sleep", _no_sleep)


async def test_timeout_retried_then_succeeds(fast_sleep) -> None:
    session = _Session(fail_times=3)
    rest = AlpacaRest("k", "s", session=session)  # type: ignore[arg-type]
    out = await rest._get("http://x/y", {})
    assert out == {"ok": True}
    assert session.calls == 4


async def test_timeout_exhausts_to_resterror(fast_sleep) -> None:
    session = _Session(fail_times=99)
    rest = AlpacaRest("k", "s", session=session)  # type: ignore[arg-type]
    with pytest.raises(AlpacaRestError):
        await rest._get("http://x/y", {})
    assert session.calls == 6  # initial + 5 retries
