from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from engine.order import Order, OrderType, Side
from engine.order_book import OrderBook
from hawkes.process import simulate_bivariate


@dataclass
class SimulationResult:
    symbol: str
    book: OrderBook
    event_log: List[Dict]
    n_limit_orders: int
    n_market_orders: int
    n_cancels: int
    elapsed_wall_time_s: float
    sim_duration_s: float
    seed: int

    @property
    def trades(self): return self.book.trades

    @property
    def snapshots(self): return self.book.snapshots

    @property
    def n_events(self): return self.n_limit_orders + self.n_market_orders + self.n_cancels

    def summary(self):
        thr = self.n_events / max(self.elapsed_wall_time_s, 1e-9)
        lines = [
            f"{'='*60}", f"  Simulation: {self.symbol}", f"{'='*60}",
            f"  Events processed : {self.n_events:>10,}",
            f"  Trades generated : {len(self.trades):>10,}",
            f"  Wall-clock time  : {self.elapsed_wall_time_s:>10.3f} s",
            f"  Throughput       : {thr:>10,.0f} events/s",
            f"{'='*60}",
        ]
        return "\n".join(lines)


class MarketSimulator:
    def __init__(self, symbol="SYM", ref_price=100.0, tick_size=0.01,
                 lot_size=1.0, half_spread=0.05, sigma=0.20,
                 mu_bid=5.0, mu_ask=5.0, alpha_self=3.0, alpha_cross=0.5,
                 beta=10.0, p_market=0.15, p_cancel=0.10, seed=42):
        self.symbol = symbol
        self.ref_price = ref_price
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.half_spread = half_spread
        self.sigma = sigma
        self.mu_bid = mu_bid
        self.mu_ask = mu_ask
        self.alpha_self = alpha_self
        self.alpha_cross = alpha_cross
        self.beta = beta
        self.p_market = p_market
        self.p_cancel = p_cancel
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def run(self, duration_s=3600.0) -> SimulationResult:
        raise NotImplementedError("run() not yet implemented")
