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
        trades = self._match_limit(order)
        if order.is_active:
            self._rest(order)
        return trades

    def add_market_order(self, order: Order) -> List[Trade]:
        self._order_index[order.order_id] = order
        return self._match_market(order)

    def cancel_order(self, order_id: str) -> bool:
        return False  # TODO

    def _match_limit(self, order: Order) -> List[Trade]:
        trades: List[Trade] = []
        if order.side == Side.BID:
            while order.is_active and self._asks:
                best_ask = self._asks.keys()[0]
                if order.price < best_ask:
                    break
                level = self._asks[best_ask]
                trades += self._fill_against_level(order, level, best_ask)
                if level.is_empty:
                    del self._asks[best_ask]
        else:
            while order.is_active and self._bids:
                neg_best = self._bids.keys()[0]
                best_bid = -neg_best
                if order.price > best_bid:
                    break
                level = self._bids[neg_best]
                trades += self._fill_against_level(order, level, best_bid)
                if level.is_empty:
                    del self._bids[neg_best]
        return trades

    def _match_market(self, order: Order) -> List[Trade]:
        trades: List[Trade] = []
        if order.side == Side.BID:
            while order.is_active and self._asks:
                best_ask = self._asks.keys()[0]
                level = self._asks[best_ask]
                trades += self._fill_against_level(order, level, best_ask)
                if level.is_empty:
                    del self._asks[best_ask]
        else:
            while order.is_active and self._bids:
                neg_best = self._bids.keys()[0]
                level = self._bids[neg_best]
                trades += self._fill_against_level(order, level, -neg_best)
                if level.is_empty:
                    del self._bids[neg_best]
        return trades

    def _fill_against_level(self, aggressor: Order, level: PriceLevel,
                            exec_price: float) -> List[Trade]:
        trades: List[Trade] = []
        while aggressor.is_active:
            passive = level.peek()
            if passive is None:
                break
            fill_qty = min(aggressor.remaining_quantity, passive.remaining_quantity)
            aggressor.fill(fill_qty)
            passive.fill(fill_qty)
            level.total_volume = max(0.0, level.total_volume - fill_qty)
            if not passive.is_active:
                level.orders.popleft()
                level.order_count -= 1
            if aggressor.side == Side.BID:
                buyer_oid, seller_oid = aggressor.order_id, passive.order_id
                buyer_tid, seller_tid = aggressor.trader_id, passive.trader_id
            else:
                buyer_oid, seller_oid = passive.order_id, aggressor.order_id
                buyer_tid, seller_tid = passive.trader_id, aggressor.trader_id
            trade = Trade(price=exec_price, quantity=fill_qty,
                          aggressor_side=aggressor.side,
                          buyer_order_id=buyer_oid, seller_order_id=seller_oid,
                          buyer_trader_id=buyer_tid, seller_trader_id=seller_tid)
            trades.append(trade)
            self.trades.append(trade)
        return trades

    def _rest(self, order: Order) -> None:
        key, side = self._side_key_and_dict(order)
        if key not in side:
            side[key] = PriceLevel(order.price)
        side[key].add(order)

    def _side_key_and_dict(self, order: Order):
        if order.side == Side.BID:
            return (-order.price, self._bids)
        return (order.price, self._asks)
