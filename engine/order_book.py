"""
Central Limit Order Book (CLOB) with price-time priority matching.

Architecture
------------
Bids  : SortedDict keyed by (-price), so index 0 is always the highest bid.
Asks  : SortedDict keyed by (+price), so index 0 is always the lowest ask.
Each price level holds a collections.deque of Orders in FIFO (time-priority) order.
A flat dict  { order_id -> Order }  gives O(1) cancel lookup.

Complexity
----------
add_limit_order  : O(log P)   P = distinct price levels
cancel_order     : O(1)       hash lookup + lazy deque skip
best_bid / ask   : O(1)
match            : O(k)       k = orders consumed per incoming order
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sortedcontainers import SortedDict

from engine.order import Order, OrderStatus, OrderType, Side, Trade


# ---------------------------------------------------------------------------
# Price level
# ---------------------------------------------------------------------------

class PriceLevel:
    """
    All orders resting at a single price, maintained in FIFO order.
    Cancelled / fully-filled orders are skipped lazily during iteration.
    """
    __slots__ = ("price", "orders", "total_volume", "order_count")

    def __init__(self, price: float) -> None:
        self.price: float = price
        self.orders: deque[Order] = deque()
        self.total_volume: float = 0.0
        self.order_count: int = 0

    # ------------------------------------------------------------------
    def add(self, order: Order) -> None:
        self.orders.append(order)
        self.total_volume += order.remaining_quantity
        self.order_count += 1

    def cancel(self, order: Order) -> None:
        """Mark order cancelled and remove its volume from the level total."""
        self.total_volume = max(0.0, self.total_volume - order.remaining_quantity)
        self.order_count -= 1
        order.cancel()

    # ------------------------------------------------------------------
    def peek(self) -> Optional[Order]:
        """Return the front active order without removing it."""
        self._drain_inactive()
        return self.orders[0] if self.orders else None

    def pop_front(self) -> Optional[Order]:
        """Remove and return the front active order."""
        self._drain_inactive()
        if self.orders:
            return self.orders.popleft()
        return None

    def _drain_inactive(self) -> None:
        """Lazily remove filled/cancelled orders from the front."""
        while self.orders and not self.orders[0].is_active:
            self.orders.popleft()

    @property
    def is_empty(self) -> bool:
        return self.peek() is None


# ---------------------------------------------------------------------------
# LOB snapshot (lightweight, for microstructure analysis)
# ---------------------------------------------------------------------------

@dataclass
class BookSnapshot:
    timestamp: int
    best_bid: Optional[float]
    best_ask: Optional[float]
    mid_price: Optional[float]
    spread: Optional[float]
    bid_volume_top5: float   # summed volume at top-5 bid levels
    ask_volume_top5: float
    bid_levels: int          # number of distinct price levels
    ask_levels: int
    num_trades: int          # cumulative trades so far


# ---------------------------------------------------------------------------
# Order book
# ---------------------------------------------------------------------------

class OrderBook:
    """
    Full central limit order book for a single symbol.

    Usage
    -----
    book = OrderBook("AAPL")
    trades = book.add_limit_order(order)
    trades = book.add_market_order(order)
    book.cancel_order(order_id)
    """

    def __init__(self, symbol: str = "") -> None:
        self.symbol = symbol

        # Bids  keyed by -price  (ascending sort → index 0 = highest bid)
        self._bids: SortedDict = SortedDict()
        # Asks  keyed by +price  (ascending sort → index 0 = lowest ask)
        self._asks: SortedDict = SortedDict()

        # O(1) cancel lookup
        self._order_index: Dict[str, Order] = {}

        # Output streams
        self.trades: List[Trade] = []
        self.snapshots: List[BookSnapshot] = []

        # Snapshot sampling: record every N order events
        self._event_count: int = 0
        self._snapshot_every: int = 50

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_limit_order(self, order: Order) -> List[Trade]:
        """
        Submit a limit order.  If it crosses the spread it matches first;
        any unfilled remainder rests on the book.
        """
        assert order.order_type == OrderType.LIMIT
        self._order_index[order.order_id] = order
        trades = self._match_limit(order)
        if order.is_active:
            self._rest(order)
        self._maybe_snapshot()
        return trades

    def add_market_order(self, order: Order) -> List[Trade]:
        """
        Submit a market order.  Walks the opposite side at any price until
        fully filled or the book is exhausted.
        """
        assert order.order_type == OrderType.MARKET
        self._order_index[order.order_id] = order
        trades = self._match_market(order)
        self._maybe_snapshot()
        return trades

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a resting order in O(1).
        Returns True if the order existed and was active.
        """
        order = self._order_index.get(order_id)
        if order is None or not order.is_active:
            return False

        key, book_side = self._side_key_and_dict(order)
        level: Optional[PriceLevel] = book_side.get(key)
        if level is not None:
            level.cancel(order)
            if level.is_empty:
                del book_side[key]
        return True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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
        """[(price, volume), ...] highest first."""
        result = []
        for neg_p, lvl in self._bids.items():
            if not lvl.is_empty:
                result.append((-neg_p, lvl.total_volume))
            if len(result) >= levels:
                break
        return result

    def ask_depth(self, levels: int = 10) -> List[Tuple[float, float]]:
        """[(price, volume), ...] lowest first."""
        result = []
        for p, lvl in self._asks.items():
            if not lvl.is_empty:
                result.append((p, lvl.total_volume))
            if len(result) >= levels:
                break
        return result

    # ------------------------------------------------------------------
    # Internal matching
    # ------------------------------------------------------------------

    def _match_limit(self, order: Order) -> List[Trade]:
        trades: List[Trade] = []
        if order.side == Side.BID:
            # Incoming buy: match against resting asks (lowest ask first)
            while order.is_active and self._asks:
                best_ask_key = self._asks.keys()[0]
                if order.price < best_ask_key:          # no cross
                    break
                level = self._asks[best_ask_key]
                trades += self._fill_against_level(order, level, exec_price=best_ask_key)
                if level.is_empty:
                    del self._asks[best_ask_key]
        else:
            # Incoming sell: match against resting bids (highest bid first)
            while order.is_active and self._bids:
                neg_best_bid = self._bids.keys()[0]
                best_bid_price = -neg_best_bid
                if order.price > best_bid_price:         # no cross
                    break
                level = self._bids[neg_best_bid]
                trades += self._fill_against_level(order, level, exec_price=best_bid_price)
                if level.is_empty:
                    del self._bids[neg_best_bid]
        return trades

    def _match_market(self, order: Order) -> List[Trade]:
        trades: List[Trade] = []
        if order.side == Side.BID:
            while order.is_active and self._asks:
                best_ask_key = self._asks.keys()[0]
                level = self._asks[best_ask_key]
                trades += self._fill_against_level(order, level, exec_price=best_ask_key)
                if level.is_empty:
                    del self._asks[best_ask_key]
        else:
            while order.is_active and self._bids:
                neg_best_bid = self._bids.keys()[0]
                best_bid_price = -neg_best_bid
                level = self._bids[neg_best_bid]
                trades += self._fill_against_level(order, level, exec_price=best_bid_price)
                if level.is_empty:
                    del self._bids[neg_best_bid]
        return trades

    def _fill_against_level(
        self, aggressor: Order, level: PriceLevel, exec_price: float
    ) -> List[Trade]:
        """
        Walk a price level FIFO, generating Trade objects until the
        aggressor is filled or the level is exhausted.

        The execution price is always the passive order's price
        (standard price-time priority rule).
        """
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

            trade = Trade(
                price=exec_price,
                quantity=fill_qty,
                aggressor_side=aggressor.side,
                buyer_order_id=buyer_oid,
                seller_order_id=seller_oid,
                buyer_trader_id=buyer_tid,
                seller_trader_id=seller_tid,
            )
            trades.append(trade)
            self.trades.append(trade)

        return trades

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rest(self, order: Order) -> None:
        """Insert an unfilled limit order into the resting book."""
        key, book_side = self._side_key_and_dict(order)
        if key not in book_side:
            book_side[key] = PriceLevel(order.price)
        book_side[key].add(order)

    def _side_key_and_dict(self, order: Order):
        if order.side == Side.BID:
            return (-order.price, self._bids)
        return (order.price, self._asks)

    def _maybe_snapshot(self) -> None:
        self._event_count += 1
        if self._event_count % self._snapshot_every == 0:
            bid_depth = self.bid_depth(5)
            ask_depth = self.ask_depth(5)
            snap = BookSnapshot(
                timestamp=time.time_ns(),
                best_bid=self.best_bid,
                best_ask=self.best_ask,
                mid_price=self.mid_price,
                spread=self.spread,
                bid_volume_top5=sum(v for _, v in bid_depth),
                ask_volume_top5=sum(v for _, v in ask_depth),
                bid_levels=len(self._bids),
                ask_levels=len(self._asks),
                num_trades=len(self.trades),
            )
            self.snapshots.append(snap)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"OrderBook({self.symbol!r} | "
            f"bid={self.best_bid} ask={self.best_ask} "
            f"spread={self.spread} trades={len(self.trades)})"
        )
