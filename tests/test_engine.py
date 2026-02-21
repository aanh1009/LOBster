"""
Unit tests for the LOB matching engine.

Covers: price-time priority, partial fills, cancels, market orders,
        spread arithmetic, and edge cases (empty book, self-cross).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from engine.order import Order, OrderStatus, OrderType, Side
from engine.order_book import OrderBook


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def limit(side: Side, price: float, qty: float = 100.0) -> Order:
    return Order(side=side, order_type=OrderType.LIMIT, price=price, quantity=qty)


def market(side: Side, qty: float = 100.0) -> Order:
    return Order(side=side, order_type=OrderType.MARKET, quantity=qty)


# ---------------------------------------------------------------------------
# Basic book properties
# ---------------------------------------------------------------------------

class TestEmptyBook:
    def test_empty_best_bid_is_none(self):
        book = OrderBook()
        assert book.best_bid is None

    def test_empty_best_ask_is_none(self):
        book = OrderBook()
        assert book.best_ask is None

    def test_empty_mid_is_none(self):
        book = OrderBook()
        assert book.mid_price is None

    def test_empty_spread_is_none(self):
        book = OrderBook()
        assert book.spread is None


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
        book.add_limit_order(limit(Side.BID, 98.0))
        book.add_limit_order(limit(Side.BID, 99.0))
        book.add_limit_order(limit(Side.BID, 97.0))
        assert book.best_bid == pytest.approx(99.0)

    def test_best_ask_is_lowest(self):
        book = OrderBook()
        book.add_limit_order(limit(Side.ASK, 102.0))
        book.add_limit_order(limit(Side.ASK, 101.0))
        book.add_limit_order(limit(Side.ASK, 103.0))
        assert book.best_ask == pytest.approx(101.0)


# ---------------------------------------------------------------------------
# Matching: limit orders
# ---------------------------------------------------------------------------

class TestLimitMatching:
    def test_crossing_limit_generates_trade(self):
        book = OrderBook()
        book.add_limit_order(limit(Side.ASK, 100.0, qty=100.0))
        trades = book.add_limit_order(limit(Side.BID, 100.0, qty=100.0))
        assert len(trades) == 1
        assert trades[0].price == pytest.approx(100.0)
        assert trades[0].quantity == pytest.approx(100.0)

    def test_trade_at_passive_price(self):
        """Execution price must be the resting (passive) order's price."""
        book = OrderBook()
        book.add_limit_order(limit(Side.ASK, 100.0))     # passive at 100
        trades = book.add_limit_order(limit(Side.BID, 101.0))  # aggressor
        assert trades[0].price == pytest.approx(100.0)   # passive price wins

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
        assert len(trades) == 1
        assert trades[0].quantity == pytest.approx(60.0)
        assert o.filled_quantity == pytest.approx(60.0)
        assert o.status == OrderStatus.PARTIALLY_FILLED
        # Remaining 40 should rest as bid
        assert book.best_bid == pytest.approx(100.0)
        assert book.best_ask is None                      # ask fully consumed

    def test_price_time_priority_fifo(self):
        """Two resting asks at the same price must fill in submission order."""
        book = OrderBook()
        o1 = limit(Side.ASK, 100.0, qty=50.0)
        o2 = limit(Side.ASK, 100.0, qty=50.0)
        book.add_limit_order(o1)
        book.add_limit_order(o2)

        trades = book.add_limit_order(limit(Side.BID, 100.0, qty=50.0))
        assert len(trades) == 1
        assert trades[0].seller_order_id == o1.order_id   # o1 first

    def test_multi_level_sweep(self):
        """Large aggressive order should sweep multiple price levels."""
        book = OrderBook()
        book.add_limit_order(limit(Side.ASK, 100.0, qty=100.0))
        book.add_limit_order(limit(Side.ASK, 101.0, qty=100.0))
        book.add_limit_order(limit(Side.ASK, 102.0, qty=100.0))

        # BID at 103 can match all three ask levels
        trades = book.add_limit_order(limit(Side.BID, 103.0, qty=300.0))
        assert len(trades) == 3
        prices = sorted(t.price for t in trades)
        assert prices == pytest.approx([100.0, 101.0, 102.0])
        assert book.best_ask is None    # all asks consumed


# ---------------------------------------------------------------------------
# Market orders
# ---------------------------------------------------------------------------

class TestMarketOrders:
    def test_market_buy_executes_at_best_ask(self):
        book = OrderBook()
        book.add_limit_order(limit(Side.ASK, 101.0, qty=100.0))
        trades = book.add_market_order(market(Side.BID, qty=100.0))
        assert len(trades) == 1
        assert trades[0].price == pytest.approx(101.0)

    def test_market_sell_executes_at_best_bid(self):
        book = OrderBook()
        book.add_limit_order(limit(Side.BID, 99.0, qty=100.0))
        trades = book.add_market_order(market(Side.ASK, qty=100.0))
        assert len(trades) == 1
        assert trades[0].price == pytest.approx(99.0)

    def test_market_order_against_empty_book_is_no_op(self):
        book = OrderBook()
        trades = book.add_market_order(market(Side.BID))
        assert trades == []

    def test_market_order_partial_fill_on_thin_book(self):
        book = OrderBook()
        book.add_limit_order(limit(Side.ASK, 101.0, qty=50.0))
        o = market(Side.BID, qty=100.0)
        trades = book.add_market_order(o)
        # Only 50 available
        assert len(trades) == 1
        assert trades[0].quantity == pytest.approx(50.0)
        assert o.filled_quantity == pytest.approx(50.0)
        assert o.status == OrderStatus.PARTIALLY_FILLED


# ---------------------------------------------------------------------------
# Cancel orders
# ---------------------------------------------------------------------------

class TestCancelOrders:
    def test_cancel_removes_from_best_bid(self):
        book = OrderBook()
        o = limit(Side.BID, 99.0)
        book.add_limit_order(o)
        assert book.best_bid == pytest.approx(99.0)
        result = book.cancel_order(o.order_id)
        assert result is True
        assert book.best_bid is None

    def test_cancel_nonexistent_returns_false(self):
        book = OrderBook()
        assert book.cancel_order("nonexistent-id") is False

    def test_cancel_already_filled_returns_false(self):
        book = OrderBook()
        book.add_limit_order(limit(Side.ASK, 100.0))
        o = limit(Side.BID, 100.0)
        book.add_limit_order(o)   # fully matches and fills
        assert o.status == OrderStatus.FILLED
        assert book.cancel_order(o.order_id) is False

    def test_cancelled_order_not_matched(self):
        book = OrderBook()
        o = limit(Side.ASK, 100.0, qty=100.0)
        book.add_limit_order(o)
        book.cancel_order(o.order_id)
        # Now send a crossing bid — should find no ask
        trades = book.add_limit_order(limit(Side.BID, 100.0))
        assert len(trades) == 0

    def test_cancel_one_of_two_at_same_price(self):
        book = OrderBook()
        o1 = limit(Side.ASK, 100.0, qty=100.0)
        o2 = limit(Side.ASK, 100.0, qty=100.0)
        book.add_limit_order(o1)
        book.add_limit_order(o2)

        book.cancel_order(o1.order_id)
        trades = book.add_limit_order(limit(Side.BID, 100.0, qty=100.0))
        assert len(trades) == 1
        assert trades[0].seller_order_id == o2.order_id


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_quantity_order_rejected(self):
        with pytest.raises(ValueError):
            Order(side=Side.BID, order_type=OrderType.LIMIT,
                  price=99.0, quantity=0.0)

    def test_limit_order_without_price_rejected(self):
        with pytest.raises(ValueError):
            Order(side=Side.BID, order_type=OrderType.LIMIT,
                  price=None, quantity=100.0)

    def test_aggressor_side_recorded_correctly(self):
        book = OrderBook()
        book.add_limit_order(limit(Side.ASK, 100.0))
        trades = book.add_limit_order(limit(Side.BID, 100.0))
        assert trades[0].aggressor_side == Side.BID

    def test_depth_returns_correct_volumes(self):
        book = OrderBook()
        book.add_limit_order(limit(Side.BID, 99.0, qty=200.0))
        book.add_limit_order(limit(Side.BID, 98.0, qty=300.0))
        depth = book.bid_depth()
        assert depth[0] == pytest.approx((99.0, 200.0))
        assert depth[1] == pytest.approx((98.0, 300.0))
