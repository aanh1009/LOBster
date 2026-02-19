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


class TestLimitMatching:
    def test_crossing_limit_generates_trade(self):
        book = OrderBook()
        book.add_limit_order(limit(Side.ASK, 100.0))
        trades = book.add_limit_order(limit(Side.BID, 100.0))
        assert len(trades) == 1
        assert trades[0].price == pytest.approx(100.0)

    def test_trade_at_passive_price(self):
        book = OrderBook()
        book.add_limit_order(limit(Side.ASK, 100.0))
        trades = book.add_limit_order(limit(Side.BID, 101.0))
        assert trades[0].price == pytest.approx(100.0)

    def test_non_crossing_limit_rests_on_book(self):
        book = OrderBook()
        book.add_limit_order(limit(Side.ASK, 101.0))
        trades = book.add_limit_order(limit(Side.BID, 99.0))
        assert len(trades) == 0
        assert book.best_bid == pytest.approx(99.0)

    def test_partial_fill_remainder_rests(self):
        book = OrderBook()
        book.add_limit_order(limit(Side.ASK, 100.0, qty=60.0))
        o = limit(Side.BID, 100.0, qty=100.0)
        trades = book.add_limit_order(o)
        assert trades[0].quantity == pytest.approx(60.0)
        assert o.filled_quantity == pytest.approx(60.0)
        assert book.best_ask is None

    def test_price_time_priority_fifo(self):
        book = OrderBook()
        o1 = limit(Side.ASK, 100.0, qty=50.0)
        o2 = limit(Side.ASK, 100.0, qty=50.0)
        book.add_limit_order(o1)
        book.add_limit_order(o2)
        trades = book.add_limit_order(limit(Side.BID, 100.0, qty=50.0))
        assert trades[0].seller_order_id == o1.order_id

    def test_multi_level_sweep(self):
        book = OrderBook()
        for p in [100.0, 101.0, 102.0]:
            book.add_limit_order(limit(Side.ASK, p))
        trades = book.add_limit_order(limit(Side.BID, 103.0, qty=300.0))
        assert len(trades) == 3
        assert book.best_ask is None


class TestMarketOrders:
    def test_market_buy_executes_at_best_ask(self):
        book = OrderBook()
        book.add_limit_order(limit(Side.ASK, 101.0))
        trades = book.add_market_order(market(Side.BID))
        assert trades[0].price == pytest.approx(101.0)

    def test_market_sell_executes_at_best_bid(self):
        book = OrderBook()
        book.add_limit_order(limit(Side.BID, 99.0))
        trades = book.add_market_order(market(Side.ASK))
        assert trades[0].price == pytest.approx(99.0)

    def test_market_order_against_empty_book_is_no_op(self):
        book = OrderBook()
        assert book.add_market_order(market(Side.BID)) == []
