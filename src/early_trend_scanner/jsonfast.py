"""JSON with an optional orjson fast path (stream parsing is the hot spot)."""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - depends on optional install
    import orjson

    def loads(data: str | bytes) -> Any:
        return orjson.loads(data)

    def dumps(obj: Any) -> str:
        return orjson.dumps(obj).decode()

except ImportError:  # pragma: no cover
    import json

    def loads(data: str | bytes) -> Any:
        return json.loads(data)

    def dumps(obj: Any) -> str:
        return json.dumps(obj, separators=(",", ":"))
