"""
Synthetic Market Feed Simulator
================================

Converts raw Hawkes-process event timestamps into a realistic stream of
limit-order, market-order, and cancel events, then pipes them through the
limit order book.

Price Dynamics
--------------
The reference (fundamental) price follows a discretized arithmetic Brownian
motion with a volatility parameter σ.  Limit-order prices are drawn from a
distribution centered on the reference price ± half-spread, then snapped to
the nearest tick.  This mimics how liquidity providers quote around their
private estimate of fair value.

Order Types
-----------
Each Hawkes event is one of three types (sampled via Dirichlet weights):
  • Limit order  (default weight 0.70) — rests on the book
  • Market order (default weight 0.20) — immediate execution
  • Cancel       (default weight 0.10) — cancels a random live order

Quantity Distribution
---------------------
Order sizes are drawn from a log-normal distribution (matches empirical LOB
data; see Gould et al. 2013) and rounded to the nearest lot size.

Usage
-----
from feed.simulator import MarketSimulator
sim = MarketSimulator(symbol="AAPL", ref_price=150.0, seed=42)
result = sim.run(n_events=50_000)
# result.book, result.trades, result.snapshots, result.event_log
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from engine.order import Order, OrderType, Side
from engine.order_book import OrderBook
from hawkes.process import simulate_bivariate


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    symbol: str
    book: OrderBook
    event_log: List[Dict]            # one entry per event
    n_limit_orders: int
    n_market_orders: int
    n_cancels: int
    elapsed_wall_time_s: float
    sim_duration_s: float
    seed: int

    @property
    def trades(self):
        return self.book.trades

    @property
    def snapshots(self):
        return self.book.snapshots

    @property
    def n_events(self) -> int:
        return self.n_limit_orders + self.n_market_orders + self.n_cancels

    def summary(self) -> str:
        thr = self.n_events / max(self.elapsed_wall_time_s, 1e-9)
        lines = [
            f"{'='*60}",
            f"  Simulation: {self.symbol}",
            f"{'='*60}",
            f"  Events processed : {self.n_events:>10,}",
            f"    Limit orders   : {self.n_limit_orders:>10,}",
            f"    Market orders  : {self.n_market_orders:>10,}",
            f"    Cancels        : {self.n_cancels:>10,}",
            f"  Trades generated : {len(self.trades):>10,}",
            f"  LOB snapshots    : {len(self.snapshots):>10,}",
            f"  Sim duration     : {self.sim_duration_s:>10.1f} s (sim-time)",
            f"  Wall-clock time  : {self.elapsed_wall_time_s:>10.3f} s",
            f"  Throughput       : {thr:>10,.0f} events/s",
            f"{'='*60}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class MarketSimulator:
    """
    Drives the limit order book with a synthetic Hawkes-generated order flow.

    Parameters
    ----------
    symbol        : ticker symbol (cosmetic)
    ref_price     : starting reference (mid) price
    tick_size     : minimum price increment (e.g. 0.01 for penny stocks)
    lot_size      : minimum order quantity
    half_spread   : initial half bid-ask spread in price units
    sigma         : annualised price volatility (used to scale GBM steps)
    mu_bid/ask    : Hawkes baseline intensity (events/second)
    alpha_self    : Hawkes self-excitation coefficient
    alpha_cross   : Hawkes cross-excitation coefficient
    beta          : Hawkes decay rate  (must satisfy α_self + α_cross < β)
    p_market      : probability an event becomes a market order
    p_cancel      : probability an event becomes a cancel
    seed          : master RNG seed
    """

    def __init__(
        self,
        symbol: str = "SYM",
        ref_price: float = 100.0,
        tick_size: float = 0.01,
        lot_size: float = 1.0,
        half_spread: float = 0.05,
        sigma: float = 0.20,             # annualised
        mu_bid: float = 5.0,
        mu_ask: float = 5.0,
        alpha_self: float = 3.0,
        alpha_cross: float = 0.5,
        beta: float = 10.0,
        p_market: float = 0.15,
        p_cancel: float = 0.10,
        seed: int = 42,
    ) -> None:
        self.symbol     = symbol
        self.ref_price  = ref_price
        self.tick_size  = tick_size
        self.lot_size   = lot_size
        self.half_spread = half_spread
        self.sigma      = sigma
        self.mu_bid     = mu_bid
        self.mu_ask     = mu_ask
        self.alpha_self  = alpha_self
        self.alpha_cross = alpha_cross
        self.beta        = beta
        self.p_market    = p_market
        self.p_cancel    = p_cancel
        self.seed        = seed
        self._rng        = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def run(self, duration_s: float = 3600.0) -> SimulationResult:
        """
        Run the full simulation for `duration_s` seconds of simulated time.

        Steps
        -----
        1. Generate bid and ask event timestamps via bivariate Hawkes.
        2. Merge into a single chronological event stream.
        3. For each event decide type (limit / market / cancel).
        4. Compute a reference-price update (arithmetic BM).
        5. Draw order price from N(ref_price ± half_spread, σ_quote).
        6. Submit order to the LOB and record everything.
        """
        t0_wall = time.perf_counter()

        book = OrderBook(self.symbol)

        # --- 1. Generate Hawkes timestamps --------------------------------
        bid_times, ask_times = simulate_bivariate(
            mu_bid=self.mu_bid,
            alpha_bb=self.alpha_self,
            alpha_ab=self.alpha_cross,
            mu_ask=self.mu_ask,
            alpha_aa=self.alpha_self,
            alpha_ba=self.alpha_cross,
            beta=self.beta,
            T=duration_s,
            seed=int(self._rng.integers(0, 2**31)),
        )

        # --- 2. Merge event streams ----------------------------------------
        events: List[Tuple[float, Side]] = (
            [(t, Side.BID) for t in bid_times]
            + [(t, Side.ASK) for t in ask_times]
        )
        events.sort(key=lambda x: x[0])

        # --- 3-6. Process events ------------------------------------------
        n_limit = n_market = n_cancel = 0
        event_log: List[Dict] = []

        # Running state
        ref_price = self.ref_price
        active_order_ids: List[str] = []    # pool of cancellable order IDs
        dt_year = 1.0 / (252 * 6.5 * 3600) # one-second in trading-year units

        for sim_t, side in events:
            # Update reference price with a Brownian step proportional to dt
            # (approximate: use 1-sec intervals between events for simplicity)
            ref_price += self._rng.normal(
                0.0, self.sigma * ref_price * np.sqrt(dt_year)
            )
            ref_price = max(ref_price, self.tick_size)  # floor

            # Decide event type
            r = self._rng.uniform()
            if r < self.p_cancel and active_order_ids:
                # --- Cancel -------------------------------------------------
                cancel_id = active_order_ids.pop(
                    int(self._rng.integers(0, len(active_order_ids)))
                )
                book.cancel_order(cancel_id)
                n_cancel += 1
                event_log.append({"type": "cancel", "t": sim_t, "side": side.value})

            elif r < self.p_cancel + self.p_market:
                # --- Market order -------------------------------------------
                qty = self._draw_quantity()
                order = Order(
                    side=side,
                    order_type=OrderType.MARKET,
                    quantity=qty,
                )
                book.add_market_order(order)
                n_market += 1
                event_log.append({
                    "type": "market", "t": sim_t, "side": side.value, "qty": qty
                })

            else:
                # --- Limit order -------------------------------------------
                price = self._draw_limit_price(ref_price, side)
                qty   = self._draw_quantity()
                order = Order(
                    side=side,
                    order_type=OrderType.LIMIT,
                    price=price,
                    quantity=qty,
                )
                trades = book.add_limit_order(order)
                if order.is_active:
                    active_order_ids.append(order.order_id)
                    # Trim pool to avoid unbounded growth
                    if len(active_order_ids) > 5000:
                        active_order_ids = active_order_ids[-5000:]
                n_limit += 1
                event_log.append({
                    "type": "limit", "t": sim_t, "side": side.value,
                    "price": price, "qty": qty, "trades": len(trades)
                })

        elapsed = time.perf_counter() - t0_wall

        return SimulationResult(
            symbol=self.symbol,
            book=book,
            event_log=event_log,
            n_limit_orders=n_limit,
            n_market_orders=n_market,
            n_cancels=n_cancel,
            elapsed_wall_time_s=elapsed,
            sim_duration_s=duration_s,
            seed=self.seed,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_limit_price(self, ref_price: float, side: Side) -> float:
        """
        Draw a limit price from a Gaussian centered on the relevant side
        of the book.  Standard deviation equals 2× the half-spread, which
        places orders both inside and outside the current spread.
        """
        if side == Side.BID:
            center = ref_price - self.half_spread
        else:
            center = ref_price + self.half_spread

        raw = self._rng.normal(center, 2.0 * self.half_spread)
        # Snap to tick grid and enforce positivity
        ticked = round(raw / self.tick_size) * self.tick_size
        return max(ticked, self.tick_size)

    def _draw_quantity(self) -> float:
        """
        Log-normal quantity distribution.
        Mean lot size ≈ 100, heavy right tail (block trades).
        """
        log_qty = self._rng.normal(np.log(100 * self.lot_size), 1.0)
        qty = np.exp(log_qty)
        return max(round(qty / self.lot_size) * self.lot_size, self.lot_size)
