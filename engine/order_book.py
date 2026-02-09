from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from sortedcontainers import SortedDict
from engine.order import Order, OrderStatus, OrderType, Side, Trade


class PriceLevel:
    __slots__ = ("price", "orders", "total_volume", "order_count")

    def __init__(self, price: float) -> None:
        self.price = price
        self.orders: deque = deque()
        self.total_volume: float = 0.0
        self.order_count: int = 0

    def add(self, order: Order) -> None:
        self.orders.append(order)
        self.total_volume += order.remaining_quantity
        self.order_count += 1

    def peek(self) -> Optional[Order]:
        self._drain_inactive()
        return self.orders[0] if self.orders else None

    def _drain_inactive(self) -> None:
        while self.orders and not self.orders[0].is_active:
            self.orders.popleft()

    @property
    def is_empty(self) -> bool:
        return self.peek() is None


class OrderBook:
    def __init__(self, symbol: str = "") -> None:
        self.symbol = symbol
        self._bids: SortedDict = SortedDict()
        self._asks: SortedDict = SortedDict()
        self._order_index: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.snapshots: List[dict] = []

    @property
    def best_bid(self) -> Optional[float]:
        return -self._bids.keys()[0] if self._bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self._asks.keys()[0] if self._asks else None

    @property
    def mid_price(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        return (bb + ba) / 2.0 if bb is not None and ba is not None else None

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        return ba - bb if bb is not None and ba is not None else None

    def bid_depth(self, levels: int = 10) -> List[Tuple[float, float]]:
        return [(-k, v.total_volume) for k, v in list(self._bids.items())[:levels]
                if not v.is_empty]

    def ask_depth(self, levels: int = 10) -> List[Tuple[float, float]]:
        return [(k, v.total_volume) for k, v in list(self._asks.items())[:levels]
                if not v.is_empty]

    def add_limit_order(self, order: Order) -> List[Trade]:
        self._order_index[order.order_id] = order
        self._rest(order)
        return []  # matching not yet implemented

    def add_market_order(self, order: Order) -> List[Trade]:
        return []  # TODO

    def cancel_order(self, order_id: str) -> bool:
        return False  # TODO

    def _rest(self, order: Order) -> None:
        key, side = self._side_key_and_dict(order)
        if key not in side:
            side[key] = PriceLevel(order.price)
        side[key].add(order)

    def _side_key_and_dict(self, order: Order):
        if order.side == Side.BID:
            return (-order.price, self._bids)
        return (order.price, self._asks)
