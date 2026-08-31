"""Daily efficacy recap: the north-star metric the learning loop must improve.

Efficacy = among delivered EARLY signals that were CONFIRMED on Telegram
(alerted before the quiet cutoff), the share whose price kept moving in the
signal's direction from the confirmation until the cutoff. Computed from
1-minute closes, pure and stateless: the app supplies the rows and a
price-at-minute lookup, this module supplies the arithmetic and the message.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True)
class RecapRow:
    symbol: str
    direction: int
    alert_ts: float
    alert_price: float
    resolution_ts: float


@dataclass(slots=True)
class Recap:
    confirmed: int
    favorable: int
    avg_move_bps: float

    @property
    def efficacy(self) -> float | None:
        return self.favorable / self.confirmed if self.confirmed else None


def build_recap(
    rows: list[RecapRow],
    price_at: Callable[[str, float], float | None],
    cutoff_ts: float,
) -> Recap:
    """`price_at(symbol, ts)` returns the 1-minute close at/just before ts."""
    confirmed = 0
    favorable = 0
    moves: list[float] = []
    for r in rows:
        entry = price_at(r.symbol, r.resolution_ts) or r.alert_price
        final = price_at(r.symbol, cutoff_ts)
        if entry is None or final is None or entry <= 0.0:
            continue
        confirmed += 1
        move_bps = (final - entry) / entry * 1e4 * r.direction
        moves.append(move_bps)
        if move_bps > 0.0:
            favorable += 1
    avg = sum(moves) / len(moves) if moves else 0.0
    return Recap(confirmed=confirmed, favorable=favorable, avg_move_bps=avg)


def format_recap(date_str: str, r: Recap) -> str:
    """Compact Telegram recap (well under the 40-word alert budget)."""
    if r.confirmed == 0:
        return f"RECAP {date_str}: no confirmed signals before the quiet hour."
    pct = round(100.0 * (r.efficacy or 0.0))
    return (
        f"RECAP {date_str}: {r.confirmed} confirmed, {r.favorable} still moving "
        f"favorably at cutoff ({pct}% efficacy) | avg {r.avg_move_bps:+.0f} bps "
        f"confirmation-to-cutoff."
    )
