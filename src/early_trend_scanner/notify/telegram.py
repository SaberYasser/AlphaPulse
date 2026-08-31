"""Telegram delivery: highest-priority async path, retries, duplicate prevention.

The signal path only formats a short string and enqueues it — database writes,
learning updates and follow-up analysis all happen elsewhere. EARLY messages
jump the queue ahead of everything else.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import aiohttp

from ..engine.state import Signal
from ..timeutil import et_hms

log = logging.getLogger(__name__)

_API = "https://api.telegram.org"


def _context_clause(sig: Signal) -> str:
    """Market-regime awareness for the confirmation, from alert-time features.

    Purely informational — context never gates a signal. A counter-market
    mover confirms with 'against market, fear rising' so the reader knows the
    tide it is fighting.
    """
    parts = []
    mkt = sig.features.get("mkt_al_5m", 0.0)
    if mkt <= -0.3:
        parts.append("against market")
    elif mkt >= 0.3:
        parts.append("with market")
    fear = sig.features.get("fear_5m", 0.0)
    if fear >= 0.3:
        parts.append("fear rising")
    elif fear <= -0.3:
        parts.append("fear easing")
    return f" | {', '.join(parts)}" if parts else ""


def format_message(sig: Signal, kind: str, extras: dict[str, Any], prefix: str = "") -> str:
    """< 40 words, exactly the fields the spec allows."""
    t = et_hms(sig.resolution_ts if kind != "EARLY" else sig.alert_ts)
    head = f"{prefix}{kind} {sig.dir_str} {sig.symbol} {t} ET"
    if kind == "EARLY":
        if sig.trigger_verb == "trend":
            return (
                f"{head} | {sig.trigger_price:.2f} trend | "
                f"volume {min(sig.vol_ratio, 99.0):.1f}x baseline | pressure sustained | "
                f"invalidation {sig.invalidation:.2f} | possible 1-5m expansion."
            )
        return (
            f"{head} | {sig.trigger_price:.2f} {sig.trigger_verb} | "
            f"volume {min(sig.vol_ratio, 99.0):.1f}x | velocity accelerating | "
            f"invalidation {sig.invalidation:.2f} | possible 1-5m expansion."
        )
    env = str(extras.get("env", "")) or _context_clause(sig).lstrip(" |")
    env_part = f" | {env}" if env else ""
    if kind == "CONFIRMED":
        micro = "micro-high" if sig.direction > 0 else "micro-low"
        reason = str(extras.get("reason", "volume sustained"))
        return (
            f"{head} | held {sig.trigger_price:.2f} | {reason} | "
            f"new {micro} {sig.micro_extreme:.2f}{env_part}."
        )
    reason = str(extras.get("reason", "trigger lost"))
    if reason in ("no expansion progress", "environment against"):
        return f"{head} | {sig.trigger_price:.2f} stalled | {reason}{env_part}."
    return f"{head} | lost {sig.trigger_price:.2f} trigger | {reason}{env_part}."


@dataclass(order=True)
class _QueueItem:
    priority: int
    seq: int
    text: str = ""
    key: str = ""


class TelegramNotifier:
    def __init__(
        self,
        token: str,
        chat_id: str,
        max_retries: int = 4,
        send_timeout_s: float = 5.0,
        dedupe_size: int = 512,
        prefix: str = "",
        enabled: bool = True,
    ) -> None:
        self._token = token
        self.chat_id = chat_id
        self.max_retries = max_retries
        self.send_timeout_s = send_timeout_s
        self.prefix = prefix
        self.enabled = enabled and bool(token) and bool(chat_id)

        self._queue: asyncio.PriorityQueue[_QueueItem] = asyncio.PriorityQueue(maxsize=200)
        self._seq = 0
        self._sent_keys: OrderedDict[str, None] = OrderedDict()
        self._dedupe_size = dedupe_size
        self._task: asyncio.Task[None] | None = None
        self._session: aiohttp.ClientSession | None = None
        self.sent_count = 0
        self.failed_count = 0
        self.captured: list[str] = []  # populated when disabled (replay / shadow)

    # ---------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        if self._task is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.send_timeout_s + 25)
            )
            self._task = asyncio.create_task(self._sender(), name="telegram-sender")

    async def stop(self) -> None:
        if self._task is not None:
            try:
                await asyncio.wait_for(self._queue.join(), timeout=10)
            except TimeoutError:
                log.warning("telegram queue not drained before shutdown")
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._session is not None:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------ publish

    def publish_signal(self, sig: Signal, kind: str, extras: dict[str, Any]) -> None:
        """Non-blocking; called directly from the signal hot path."""
        text = format_message(sig, kind, extras, prefix=self.prefix)
        key = f"{sig.signal_id}:{kind}"
        priority = 0 if kind == "EARLY" else 1
        self._publish(text, key, priority)

    def publish_ops(self, text: str, key: str) -> None:
        self._publish(f"{self.prefix}{text}", key, priority=2)

    def _publish(self, text: str, key: str, priority: int) -> None:
        if key in self._sent_keys:
            log.info("duplicate suppressed: %s", key)
            return
        self._sent_keys[key] = None
        while len(self._sent_keys) > self._dedupe_size:
            self._sent_keys.popitem(last=False)
        if not self.enabled:
            self.captured.append(text)
            log.info("TELEGRAM (disabled) %s", text)
            return
        self._seq += 1
        try:
            self._queue.put_nowait(_QueueItem(priority, self._seq, text, key))
        except asyncio.QueueFull:
            self.failed_count += 1
            log.error("telegram queue full — dropping %s", key)

    # ------------------------------------------------------------------- sender

    async def _sender(self) -> None:
        assert self._session is not None
        while True:
            item = await self._queue.get()
            try:
                await self._send_with_retry(item.text)
            finally:
                self._queue.task_done()

    async def _send_with_retry(self, text: str) -> None:
        assert self._session is not None
        url = f"{_API}/bot{self._token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text}
        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                async with self._session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=self.send_timeout_s)
                ) as resp:
                    if resp.status == 200:
                        self.sent_count += 1
                        return
                    body = await resp.json(content_type=None)
                    if resp.status == 429:
                        retry_after = float(
                            (body.get("parameters") or {}).get("retry_after", delay)
                        )
                        log.warning("telegram 429, waiting %.1fs", retry_after)
                        await asyncio.sleep(retry_after + 0.5)
                        continue
                    if 400 <= resp.status < 500:
                        log.error("telegram rejected message (%s): %s", resp.status, body)
                        self.failed_count += 1
                        return  # non-retryable
            except asyncio.CancelledError:
                raise
            except (TimeoutError, aiohttp.ClientError) as e:
                log.warning("telegram send attempt %d failed: %s", attempt, e)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 15.0)
        self.failed_count += 1
        log.error("telegram delivery failed after %d attempts", self.max_retries)

    # ---------------------------------------------------------------- utilities

    async def send_test(self) -> bool:
        """Direct send (bypasses queue) used by `ets telegram-test`."""
        if not self._token or not self.chat_id:
            return False
        own_session = self._session is None
        session = self._session or aiohttp.ClientSession()
        try:
            url = f"{_API}/bot{self._token}/sendMessage"
            text = f"{self.prefix}Scanner test OK. Live signals will look like the README examples."
            async with session.post(
                url,
                json={"chat_id": self.chat_id, "text": text},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                ok = resp.status == 200
                if not ok:
                    log.error("test message failed: %s %s", resp.status, await resp.text())
                return ok
        finally:
            if own_session:
                await session.close()


async def discover_chat_id(token: str, expected_username: str) -> tuple[str | None, list[str]]:
    """Poll getUpdates for a /start (or any) message from the expected user.

    Returns (chat_id, usernames_seen). Only private chats are considered.
    """
    seen: list[str] = []
    async with aiohttp.ClientSession() as session:
        url = f"{_API}/bot{token}/getUpdates"
        async with session.get(
            url, params={"timeout": 20}, timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"getUpdates failed: HTTP {resp.status} {await resp.text()}")
            data = await resp.json()
    for update in data.get("result", []):
        msg = update.get("message") or update.get("edited_message") or {}
        chat = msg.get("chat") or {}
        user = msg.get("from") or {}
        username = str(user.get("username") or "")
        if username and username not in seen:
            seen.append(username)
        if chat.get("type") == "private" and username.lower() == expected_username.lower():
            return str(chat.get("id")), seen
    return None, seen
