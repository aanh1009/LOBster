"""
Event-Driven Backtesting Engine
=================================

Replays the synthetic order stream through any pluggable Strategy.
The backtester provides a clean API boundary — strategies never touch
the LOB internals directly; they interact only through the StrategyContext.

Architecture
------------
                ┌──────────────┐
 event_log ──►  │ BacktestEngine│──► strategy.on_event(ctx, event)
                └──────┬───────┘
                       │
               StrategyContext
               ├── submit_limit(side, price, qty)
               ├── submit_market(side, qty)
               ├── cancel(order_id)
               ├── position        (net shares held)
               ├── cash            (cash P&L)
               └── pnl_series      (mark-to-market over time)

Included Strategies
-------------------
1. MarketMaker   — quotes a symmetric spread around the mid-price,
                   hedges inventory with market orders.
2. MomentumTrader — uses a short-window OFI signal to trade direction.

Both strategies are intentionally simple — their purpose is to validate
the LOB engine and demonstrate the backtester API, not to be profitable.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from engine.order import Order, OrderType, Side
from engine.order_book import OrderBook


# ---------------------------------------------------------------------------
# Context: what the strategy sees
# ---------------------------------------------------------------------------

@dataclass
class StrategyContext:
    book: OrderBook
    sim_time: float          # current simulation time (seconds)
    position: float          # net shares (+ = long, - = short)
    cash: float              # realised cash flows
    open_orders: Dict[str, Order] = field(default_factory=dict)
    pnl_series: List[Tuple[float, float]] = field(default_factory=list)  # (time, pnl)

    @property
    def mark_to_market(self) -> float:
        """Unrealised P&L on current position at midprice."""
        mid = self.book.mid_price
        if mid is None or self.position == 0:
            return self.cash
        return self.cash + self.position * mid

    def submit_limit(self, side: Side, price: float, quantity: float) -> Optional[str]:
        """Place a limit order and track it."""
        order = Order(side=side, order_type=OrderType.LIMIT,
                      price=price, quantity=quantity, trader_id="strategy")
        trades = self.book.add_limit_order(order)
        self._process_fills(trades)
        if order.is_active:
            self.open_orders[order.order_id] = order
        return order.order_id

    def submit_market(self, side: Side, quantity: float) -> None:
        """Place a market order."""
        order = Order(side=side, order_type=OrderType.MARKET,
                      quantity=quantity, trader_id="strategy")
        trades = self.book.add_market_order(order)
        self._process_fills(trades)

    def cancel(self, order_id: str) -> None:
        """Cancel an open order."""
        self.book.cancel_order(order_id)
        self.open_orders.pop(order_id, None)

    def cancel_all(self) -> None:
        for oid in list(self.open_orders.keys()):
            self.cancel(oid)

    def _process_fills(self, trades) -> None:
        """Update cash and position when our orders are filled."""
        for trade in trades:
            is_buyer  = trade.buyer_trader_id  == "strategy"
            is_seller = trade.seller_trader_id == "strategy"
            if is_buyer:
                self.cash     -= trade.price * trade.quantity
                self.position += trade.quantity
            if is_seller:
                self.cash     += trade.price * trade.quantity
                self.position -= trade.quantity

    def snapshot_pnl(self) -> None:
        self.pnl_series.append((self.sim_time, self.mark_to_market))


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------

class Strategy(ABC):
    @abstractmethod
    def on_event(self, ctx: StrategyContext, event: dict) -> None:
        """Called for every event in the event log."""
        ...

    def on_start(self, ctx: StrategyContext) -> None:
        """Called once before the first event."""
        pass

    def on_end(self, ctx: StrategyContext) -> None:
        """Called once after the last event."""
        pass


# ---------------------------------------------------------------------------
# Strategy 1: Simple Market Maker
# ---------------------------------------------------------------------------

class MarketMaker(Strategy):
    """
    Quotes a two-sided market around the current midprice.

    Logic
    -----
    Every `quote_interval` seconds (sim time):
      1. Cancel all open quotes.
      2. If |inventory| > max_inventory, hedge with a market order.
      3. Otherwise quote bid at (mid - half_spread) and ask at (mid + half_spread).

    Risk
    ----
    The maker earns the spread on balanced flow and loses on directional flow
    (adverse selection captured by Kyle's λ).
    """

    def __init__(
        self,
        half_spread: float = 0.03,
        quote_qty: float = 100.0,
        max_inventory: float = 500.0,
        quote_interval: float = 2.0,   # sim seconds between re-quoting
        tick_size: float = 0.01,
    ) -> None:
        self.half_spread    = half_spread
        self.quote_qty      = quote_qty
        self.max_inventory  = max_inventory
        self.quote_interval = quote_interval
        self.tick_size      = tick_size
        self._last_quote_t  = -np.inf

    def on_event(self, ctx: StrategyContext, event: dict) -> None:
        if ctx.sim_time - self._last_quote_t < self.quote_interval:
            return

        mid = ctx.book.mid_price
        if mid is None:
            return

        ctx.cancel_all()

        # Inventory hedge
        if abs(ctx.position) > self.max_inventory:
            hedge_side = Side.ASK if ctx.position > 0 else Side.BID
            ctx.submit_market(hedge_side, abs(ctx.position) / 2)

        # Refresh quotes
        bid_price = round((mid - self.half_spread) / self.tick_size) * self.tick_size
        ask_price = round((mid + self.half_spread) / self.tick_size) * self.tick_size

        ctx.submit_limit(Side.BID, bid_price, self.quote_qty)
        ctx.submit_limit(Side.ASK, ask_price, self.quote_qty)
        ctx.snapshot_pnl()
        self._last_quote_t = ctx.sim_time


# ---------------------------------------------------------------------------
# Strategy 2: OFI Momentum Trader
# ---------------------------------------------------------------------------

class MomentumTrader(Strategy):
    """
    Trades in the direction of recent Order Flow Imbalance (OFI).

    A positive OFI (more aggressive bids) predicts a short-term price rise;
    a negative OFI predicts a fall.  This exploits the finding in Cont et al.
    (2014) that OFI is the dominant predictor of short-horizon price changes.

    Logic
    -----
    Maintain a rolling window of (bid_vol − ask_vol) from LOB snapshots.
    If rolling mean OFI > threshold → buy.
    If rolling mean OFI < −threshold → sell.
    Flat otherwise.
    """

    def __init__(
        self,
        window: int = 20,
        threshold: float = 50.0,    # OFI units (shares)
        trade_qty: float = 100.0,
        max_position: float = 500.0,
    ) -> None:
        self.window       = window
        self.threshold    = threshold
        self.trade_qty    = trade_qty
        self.max_position = max_position
        self._ofi_window: deque = deque(maxlen=window)

    def on_event(self, ctx: StrategyContext, event: dict) -> None:
        snaps = ctx.book.snapshots
        if len(snaps) < 2:
            return

        # Compute instantaneous OFI from the two most recent snapshots
        s_prev, s_curr = snaps[-2], snaps[-1]
        ofi = (s_curr.bid_volume_top5 - s_prev.bid_volume_top5) \
            - (s_curr.ask_volume_top5 - s_prev.ask_volume_top5)
        self._ofi_window.append(ofi)

        if len(self._ofi_window) < self.window:
            return

        signal = float(np.mean(self._ofi_window))
        ctx.cancel_all()

        if signal > self.threshold and ctx.position < self.max_position:
            qty = min(self.trade_qty, self.max_position - ctx.position)
            ctx.submit_market(Side.BID, qty)
        elif signal < -self.threshold and ctx.position > -self.max_position:
            qty = min(self.trade_qty, self.max_position + ctx.position)
            ctx.submit_market(Side.ASK, qty)

        ctx.snapshot_pnl()


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    strategy_name: str
    final_pnl: float
    max_drawdown: float
    n_trades: int
    sharpe_ratio: float
    pnl_series: List[Tuple[float, float]]
    elapsed_s: float

    def summary(self) -> str:
        lines = [
            f"\n--- Backtest: {self.strategy_name} ---",
            f"  Final P&L      : ${self.final_pnl:>12,.2f}",
            f"  Max Drawdown   : ${self.max_drawdown:>12,.2f}",
            f"  Trades         : {self.n_trades:>12,}",
            f"  Sharpe Ratio   : {self.sharpe_ratio:>12.3f}",
            f"  Wall-clock     : {self.elapsed_s:>12.3f} s",
        ]
        return "\n".join(lines)


class BacktestEngine:
    """
    Replay the event log through a Strategy.

    The engine does NOT re-run the Hawkes simulation; it replays the already-
    generated event_log from SimulationResult so the LOB state is identical.
    The strategy's own orders interact with the same order book.
    """

    def run(
        self,
        book: OrderBook,
        event_log: List[Dict],
        strategy: Strategy,
        initial_cash: float = 0.0,
    ) -> BacktestResult:
        t0 = time.perf_counter()

        ctx = StrategyContext(
            book=book,
            sim_time=0.0,
            position=0.0,
            cash=initial_cash,
        )
        strategy.on_start(ctx)

        for event in event_log:
            ctx.sim_time = event["t"]
            strategy.on_event(ctx, event)

        strategy.on_end(ctx)

        # --- P&L analytics ---
        pnl_values = [p for _, p in ctx.pnl_series]
        if pnl_values:
            arr = np.array(pnl_values)
            running_max = np.maximum.accumulate(arr)
            drawdowns = running_max - arr
            max_dd = float(drawdowns.max())

            # Daily-ish Sharpe on the P&L changes
            pnl_changes = np.diff(arr)
            sharpe = (
                float(pnl_changes.mean() / (pnl_changes.std() + 1e-12) * np.sqrt(252))
                if len(pnl_changes) > 1
                else 0.0
            )
        else:
            max_dd = 0.0
            sharpe = 0.0

        elapsed = time.perf_counter() - t0

        return BacktestResult(
            strategy_name=strategy.__class__.__name__,
            final_pnl=ctx.mark_to_market,
            max_drawdown=max_dd,
            n_trades=len(book.trades),
            sharpe_ratio=sharpe,
            pnl_series=ctx.pnl_series,
            elapsed_s=elapsed,
        )
