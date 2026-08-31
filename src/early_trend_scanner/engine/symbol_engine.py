"""Per-symbol composition: aggregation, levels, NBBO, snapshot building, state machine.

This is the live hot path. Everything here is O(1) or O(small-constant) per
print; no raw tick history is retained beyond a 64-print flow deque.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

from ..config import Config
from ..data.models import Minute, Quote, Trade
from ..timeutil import minute_of_session
from .aggregator import SymbolAggregator
from .baseline import MinuteBaseline
from .context import ContextTracker
from .features import Snapshot, compute_score, trigger_verb
from .levels import Level, LevelBook
from .state import GlobalLimiter, MachineHooks, Phase, Rejection, Signal, StateMachine

_EPS_VOL = 1e-9


class SymbolEngine:
    def __init__(
        self,
        symbol: str,
        cfg: Config,
        baseline: MinuteBaseline,
        hooks: MachineHooks,
        limiter: GlobalLimiter,
        alerts_enabled: Callable[[], bool],
        context: ContextTracker | None = None,
    ) -> None:
        self.symbol = symbol
        self.cfg = cfg
        self.baseline = baseline
        self.alerts_enabled = alerts_enabled
        self.context = context

        e = cfg.engine
        self.agg = SymbolAggregator(
            symbol,
            ring_seconds=e.ring_seconds,
            minutes_kept=e.minutes_kept,
            on_minute=self._on_minute,
        )
        self.book = LevelBook(
            symbol,
            sweep_window_s=e.sweep_window_s,
            sweep_max_bps=e.sweep_max_bps,
            compression_max_ratio=e.compression_max_ratio,
        )
        self.machine = StateMachine(symbol, e, hooks, limiter)
        self.nbbo: Quote | None = None
        # Sized for the DENSEST tape: the span gate needs persist_window_s of
        # history at up to ~300 prints/s (opening bursts). A 64-print deque
        # held ~0.3s there — less than the required span, making the gate
        # unpassable exactly when it mattered most.
        self.recent_flow: deque[tuple[float, int, int]] = deque(maxlen=1024)
        self.catchup = False  # True while replaying a recovery backfill
        self._or_locked = False
        self.session_open_ts = 0.0

    # ------------------------------------------------------------------ setup

    def set_session(self, open_ts: float) -> None:
        self.session_open_ts = open_ts
        self.agg.set_session(open_ts, self.cfg.session.opening_range_min)

    def seed_static_levels(
        self, pdh: float, pdl: float, pdc: float, pmh: float, pml: float
    ) -> None:
        self.book.set_static(pdh=pdh, pdl=pdl, pdc=pdc, pmh=pmh, pml=pml)

    def seed_minutes(self, minutes: list[Minute]) -> None:
        self.agg.seed_minutes(minutes)
        if minutes:
            self.book.on_minute(self.agg.minutes)
            self._update_range5m()

    def seed_opening_range(self, or_high: float, or_low: float) -> None:
        if or_high > 0.0 and or_low > 0.0:
            self.book.set_opening_range(or_high, or_low)
            self._or_locked = True

    # ------------------------------------------------------------------ events

    def on_quote(self, q: Quote) -> None:
        if q.ask >= q.bid > 0.0:
            self.nbbo = q

    def on_trade(self, t: Trade) -> None:
        if t.flow_eligible:
            t.side = self._classify_side(t.price)
        self.agg.on_trade(t)
        if t.flow_eligible:
            self.recent_flow.append((t.ts, t.side, t.size))

        if not t.updates_last:
            return
        self._maybe_lock_opening_range(t.ts)
        levels = self.book.levels_cached(self.agg.vwap, t.ts)
        self.book.observe(t.price, t.ts, levels)

        if self.machine.phase == Phase.FIRED:
            imb5, share5, vol5 = self._flow5()
            self.machine.observe(t.ts, t.price, imb5, share5, vol5)
            return

        if not self._may_evaluate(t):
            return
        self._evaluate(t, levels)

    def on_second_tick(self, now_ts: float) -> None:
        """Wall-clock driven upkeep for quiet symbols and FIRED deadlines."""
        self.agg.roll_to(int(now_ts) - 1)
        self._maybe_lock_opening_range(now_ts)
        price = self.agg.last_price
        if price <= 0.0:
            return
        imb5, share5, vol5 = self._flow5()
        self.machine.on_tick(now_ts, price, imb5, share5, vol5)

    # -------------------------------------------------------------- internals

    def _classify_side(self, price: float) -> int:
        q = self.nbbo
        if q is not None:
            if price >= q.ask:
                return 1
            if price <= q.bid:
                return -1
            mid = (q.ask + q.bid) * 0.5
            if price > mid:
                return 1
            if price < mid:
                return -1
        last = self.agg.last_flow_price
        if last > 0.0:
            if price > last:
                return 1
            if price < last:
                return -1
        return 0

    def _maybe_lock_opening_range(self, ts: float) -> None:
        if (
            not self._or_locked
            and self.session_open_ts > 0.0
            and ts >= self.agg.opening_range_end_ts
            and self.agg.or_high > 0.0
        ):
            self.book.set_opening_range(self.agg.or_high, self.agg.or_low)
            self._or_locked = True

    def _on_minute(self, m: Minute) -> None:
        self.book.on_minute(self.agg.minutes)
        self._update_range5m()

    def _update_range5m(self) -> None:
        price = self.agg.last_price
        if price > 0.0 and self.book.range5m > 0.0:
            self.machine.set_range5m_bps(self.book.range5m / price * 1e4)

    def _may_evaluate(self, t: Trade) -> bool:
        return (
            t.flow_eligible
            and not self.catchup
            and self.session_open_ts > 0.0
            and t.ts >= self.session_open_ts
            and self.alerts_enabled()
        )

    def _flow5(self) -> tuple[float, float, float]:
        vol, buy, sell, _n, _d = self.agg.window(5)
        classified = buy + sell
        if classified <= 0:
            return 0.0, 0.5, vol
        imb = (buy - sell) / classified
        share_up = buy / classified
        return imb, share_up, vol

    # ------------------------------------------------------------- evaluation

    def _evaluate(self, t: Trade, levels: list[Level]) -> None:
        price = t.price
        cfg = self.cfg.engine
        max_dist_frac = (cfg.ready_bps + cfg.break_max_bps) / 1e4

        # Freshest crossed level per direction: for UP the highest level below
        # price, for DOWN the lowest level above price (the last one escaped).
        best_up: Level | None = None
        best_dn: Level | None = None
        near_any = False
        for lv in levels:
            d = price - lv.price
            if abs(d) > price * max_dist_frac:
                continue
            near_any = True
            if d > 0 and (best_up is None or lv.price > best_up.price):
                best_up = lv
            elif d < 0 and (best_dn is None or lv.price < best_dn.price):
                best_dn = lv
        reclaims = self.book.active_reclaims(t.ts)
        if not near_any and not reclaims:
            if self.machine.phase == Phase.READY:
                self.machine.phase = Phase.SCANNING
            return

        fired = False
        for cand, direction in ((best_up, 1), (best_dn, -1)):
            if cand is None or fired:
                continue
            snap = self._snapshot(t, cand, direction, is_reclaim=False)
            fired = self._consider(snap)

        for r in reclaims:
            if fired:
                break
            beyond = (price - r.level.price) * r.direction
            if beyond <= 0:
                continue
            snap = self._snapshot(t, r.level, r.direction, is_reclaim=True)
            if self._consider(snap):
                self.book.consume_reclaim(r)
                fired = True

        if not fired and near_any and self.machine.phase == Phase.SCANNING:
            self.machine.phase = Phase.READY

    def _consider(self, snap: Snapshot) -> bool:
        cfg = self.cfg.engine
        _, score_mult = self.machine.hooks.gate_multipliers(self.symbol, snap.ts)
        threshold = cfg.score_min * score_mult
        if snap.comp_active:
            threshold *= cfg.compression_relax
        threshold = max(threshold, 0.30)
        result = self.machine.consider(snap, threshold)
        return isinstance(result, Signal) and not isinstance(result, Rejection)

    def _snapshot(self, t: Trade, level: Level, direction: int, is_reclaim: bool) -> Snapshot:
        agg = self.agg
        price = t.price
        ts = t.ts

        v5, b5, s5, n5, d5 = agg.window(5)
        v15, b15, s15, n15, _ = agg.window(15)
        v60, b60, s60, n60, _ = agg.window(60)

        p5 = agg.price_ago(5)
        p15 = agg.price_ago(15)
        vel5 = (price - p5) / p5 * 1e4 / 5.0 if p5 else 0.0
        vel15 = (price - p15) / p15 * 1e4 / 15.0 if p15 else 0.0
        dvel5 = vel5 * direction
        dvel15 = vel15 * direction
        accelerating = dvel5 > 0 and (dvel15 <= 0 or dvel5 >= self.cfg.engine.accel_ratio * dvel15)

        prev_avg5 = (v60 - v5) / 11.0
        vol_ratio_prev = v5 / prev_avg5 if prev_avg5 > _EPS_VOL else (20.0 if v5 > 0 else 0.0)
        minute_idx = minute_of_session(ts, self.session_open_ts) if self.session_open_ts > 0 else 0
        if minute_idx < self.cfg.engine.baseline_skip_open_min:
            # Opening minutes: both relative volume rulers are self-defeating —
            # the minute-of-day baseline contains the auction flood, and the
            # prior-60s window IS the opening burst. Judge participation
            # against the symbol's typical MID-DAY tape instead: an opening
            # move that dwarfs midday volume is genuinely accelerating.
            vol_ratio_base = 0.0
            midday5 = self.baseline.midday_vol_per_5s(self.symbol)
            if midday5 > 0.0:
                vol_ratio_prev = v5 / midday5
        else:
            base5 = self.baseline.vol_per_5s(self.symbol, minute_idx)
            vol_ratio_base = v5 / base5 if base5 > 0 else 0.0

        cls5 = b5 + s5
        imb5 = (b5 - s5) / cls5 if cls5 > 0 else 0.0
        cls15 = b15 + s15
        imb15 = (b15 - s15) / cls15 if cls15 > 0 else 0.0
        prev_n5 = (n60 - n5) / 11.0
        n_ratio_prev = n5 / prev_n5 if prev_n5 > _EPS_VOL else (20.0 if n5 > 0 else 0.0)

        hl15 = agg.range_high_low(15)
        range15_bps = (hl15[0] - hl15[1]) / price * 1e4 if hl15 else 0.0
        minutes = agg.minutes
        if len(minutes) >= 5:
            recent = list(minutes)[-5:]
            avg1m_bps = sum(m.range_ for m in recent) / 5.0 / price * 1e4
        else:
            avg1m_bps = 0.0
        prior15_avg = avg1m_bps / 2.0  # diffusion scaling of a 1m range to 15s
        range_exp = range15_bps / prior15_avg if prior15_avg > 0 else 1.0

        vwap = agg.vwap
        dist_vwap = (price - vwap) / vwap * 1e4 if vwap > 0 else 0.0
        q = self.nbbo
        if q is not None and q.bid > 0:
            spread_bps = (q.ask - q.bid) / q.bid * 1e4
            qsum = q.bid_size + q.ask_size
            quote_imb = (q.bid_size - q.ask_size) / qsum if qsum > 0 else 0.0
        else:
            spread_bps = 0.0
            quote_imb = 0.0

        persist_trades, persist_span, n2s, dominant_frac = self._persistence(ts, direction)

        mkt_al_1m = mkt_al_5m = fear_1m = fear_5m = 0.0
        if self.context is not None:
            mkt_al_1m, mkt_al_5m, fear_1m, fear_5m = self.context.features(ts, direction)

        break_bps = (price - level.price) / level.price * 1e4 * direction

        snap = Snapshot(
            ts=ts,
            symbol=self.symbol,
            direction=direction,
            price=price,
            level_price=level.price,
            level_kind=int(level.kind),
            trigger_verb=trigger_verb(level, is_reclaim, direction),
            break_bps=break_bps,
            vel5_bps_s=vel5,
            vel15_bps_s=vel15,
            accelerating=accelerating,
            vol5=v5,
            vol_ratio_prev=vol_ratio_prev,
            vol_ratio_base=vol_ratio_base,
            imb5=imb5,
            imb15=imb15,
            n5=n5,
            n_ratio_prev=n_ratio_prev,
            dollar5=d5,
            range_exp=range_exp,
            comp_ratio=self.book.compression.ratio,
            comp_active=self.book.compression.active,
            dist_vwap_bps=dist_vwap,
            spread_bps=spread_bps,
            quote_imb=quote_imb,
            minute_frac=min(max(minute_idx / 390.0, 0.0), 1.0),
            persist_trades=persist_trades,
            persist_span_s=persist_span,
            n2s=n2s,
            dominant_frac=dominant_frac,
            mkt_al_1m=mkt_al_1m,
            mkt_al_5m=mkt_al_5m,
            fear_1m=fear_1m,
            fear_5m=fear_5m,
        )
        snap.score = compute_score(
            snap,
            vol_accel_min=self.cfg.engine.vol_accel_min,
            vel_min=self.cfg.engine.vel_min_bps_s,
            break_max_bps=self.cfg.engine.break_max_bps,
        )
        return snap

    def _persistence(self, ts: float, direction: int) -> tuple[int, float, int, float]:
        """(directional prints, span, prints in 2s, largest-print share of dir volume)."""
        cfg = self.cfg.engine
        count = 0
        first_q = 0.0
        last_q = 0.0
        n2s = 0
        dir_vol = 0
        max_print = 0
        for f_ts, side, size in reversed(self.recent_flow):
            age = ts - f_ts
            if age > cfg.persist_window_s:
                break
            if age <= 2.0:
                n2s += 1
            if side == direction:
                count += 1
                dir_vol += size
                if size > max_print:
                    max_print = size
                if last_q == 0.0:
                    last_q = f_ts
                first_q = f_ts
        span = (last_q - first_q) if count >= 2 else 0.0
        dominant = max_print / dir_vol if dir_vol > 0 else 1.0
        return count, span, n2s, dominant
