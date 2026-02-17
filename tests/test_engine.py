import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from engine.order import Order, OrderStatus, OrderType, Side
from engine.order_book import OrderBook

def limit(side, price, qty=100.0):
    return Order(side=side, order_type=OrderType.LIMIT, price=price, quantity=qty)

def market(side, qty=100.0):
    return Order(side=side, order_type=OrderType.MARKET, quantity=qty)


class TestEmptyBook:
    def test_empty_best_bid_is_none(self):
        assert OrderBook().best_bid is None

    def test_empty_best_ask_is_none(self):
        assert OrderBook().best_ask is None

    def test_empty_mid_is_none(self):
        assert OrderBook().mid_price is None

    def test_empty_spread_is_none(self):
        assert OrderBook().spread is None


class TestBasicInsertion:
    def test_bid_sets_best_bid(self):
        book = OrderBook()
        book.add_limit_order(limit(Side.BID, 99.0))
        assert book.best_bid == pytest.approx(99.0)

    def test_ask_sets_best_ask(self):
        book = OrderBook()
        book.add_limit_order(limit(Side.ASK, 101.0))
        assert book.best_ask == pytest.approx(101.0)

    def test_spread_computed_correctly(self):
        book = OrderBook()
        book.add_limit_order(limit(Side.BID, 99.0))
        book.add_limit_order(limit(Side.ASK, 101.0))
        assert book.spread == pytest.approx(2.0)
        assert book.mid_price == pytest.approx(100.0)

    def test_best_bid_is_highest(self):
        book = OrderBook()
        for p in [98.0, 99.0, 97.0]:
            book.add_limit_order(limit(Side.BID, p))
        assert book.best_bid == pytest.approx(99.0)

    def test_best_ask_is_lowest(self):
        book = OrderBook()
        for p in [102.0, 101.0, 103.0]:
            book.add_limit_order(limit(Side.ASK, p))
        assert book.best_ask == pytest.approx(101.0)
