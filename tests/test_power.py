import sys

from early_trend_scanner.power import ES_CONTINUOUS, ES_SYSTEM_REQUIRED, KeepAwake


def test_flag_values_match_windows_api() -> None:
    assert ES_CONTINUOUS == 0x80000000
    assert ES_SYSTEM_REQUIRED == 0x00000001


def test_acquire_release_flag_sequence(monkeypatch) -> None:
    calls: list[int] = []
    ka = KeepAwake()
    monkeypatch.setattr(ka, "_set", lambda flags: (calls.append(flags), 1)[1])
    ka.acquire()
    assert ka.active
    ka.acquire()  # idempotent refresh
    ka.release()
    assert not ka.active
    ka.release()  # no double-clear call
    assert calls == [
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED,
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED,
        ES_CONTINUOUS,
    ]


def test_real_acquire_release_on_windows() -> None:
    """Exercise the actual SetThreadExecutionState call and its release."""
    if sys.platform != "win32":
        return
    ka = KeepAwake()
    ka.acquire()
    assert ka.active
    ka.release()
    assert not ka.active


def test_failure_leaves_inactive(monkeypatch) -> None:
    ka = KeepAwake()
    monkeypatch.setattr(ka, "_set", lambda flags: 0)  # API failure
    ka.acquire()
    assert not ka.active


def test_rss_probe_returns_real_value() -> None:
    from early_trend_scanner.status import rss_bytes

    rss = rss_bytes()
    assert rss > 10_000_000, f"rss_bytes() returned {rss} — memory probe broken"
