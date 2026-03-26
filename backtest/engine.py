from __future__ import annotations
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from engine.order import Order, OrderType, Side
from engine.order_book import OrderBook


@dataclass
class StrategyContext:
    book: OrderBook
    sim_time: float
    position: float
    cash: float
    open_orders: Dict[str, Order] = field(default_factory=dict)
    pnl_series: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def mark_to_market(self):
        mid = self.book.mid_price
        if mid is None or self.position == 0:
            return self.cash
        return self.cash + self.position * mid

    def submit_limit(self, side, price, quantity):
        order = Order(side=side, order_type=OrderType.LIMIT,
                      price=price, quantity=quantity, trader_id="strategy")
        trades = self.book.add_limit_order(order)
        self._process_fills(trades)
        if order.is_active:
            self.open_orders[order.order_id] = order
        return order.order_id

    def submit_market(self, side, quantity):
        order = Order(side=side, order_type=OrderType.MARKET,
                      quantity=quantity, trader_id="strategy")
        trades = self.book.add_market_order(order)
        self._process_fills(trades)

    def cancel(self, order_id):
        self.book.cancel_order(order_id)
        self.open_orders.pop(order_id, None)

    def cancel_all(self):
        for oid in list(self.open_orders.keys()):
            self.cancel(oid)

    def _process_fills(self, trades):
        for trade in trades:
            if trade.buyer_trader_id == "strategy":
                self.cash -= trade.price * trade.quantity
                self.position += trade.quantity
            if trade.seller_trader_id == "strategy":
                self.cash += trade.price * trade.quantity
                self.position -= trade.quantity

    def snapshot_pnl(self):
        self.pnl_series.append((self.sim_time, self.mark_to_market))


class Strategy(ABC):
    @abstractmethod
    def on_event(self, ctx: StrategyContext, event: dict) -> None: ...
    def on_start(self, ctx: StrategyContext) -> None: pass
    def on_end(self, ctx: StrategyContext) -> None: pass


class MarketMaker(Strategy):
    """Quotes a two-sided market around the current midprice."""
    def __init__(self, half_spread=0.03, quote_qty=100.0,
                 max_inventory=500.0, quote_interval=2.0, tick_size=0.01):
        self.half_spread   = half_spread
        self.quote_qty     = quote_qty
        self.max_inventory = max_inventory
        self.quote_interval = quote_interval
        self.tick_size     = tick_size
        self._last_quote_t = -np.inf

    def on_event(self, ctx, event):
        if ctx.sim_time - self._last_quote_t < self.quote_interval:
            return
        mid = ctx.book.mid_price
        if mid is None:
            return
        ctx.cancel_all()
        if abs(ctx.position) > self.max_inventory:
            side = Side.ASK if ctx.position > 0 else Side.BID
            ctx.submit_market(side, abs(ctx.position) / 2)
        bid = round((mid - self.half_spread) / self.tick_size) * self.tick_size
        ask = round((mid + self.half_spread) / self.tick_size) * self.tick_size
        ctx.submit_limit(Side.BID, bid, self.quote_qty)
        ctx.submit_limit(Side.ASK, ask, self.quote_qty)
        ctx.snapshot_pnl()
        self._last_quote_t = ctx.sim_time
