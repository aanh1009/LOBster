"""
LOB Performance Benchmark Suite
=================================

Measures throughput (events/second) and latency distribution for the three
core LOB operations: add_limit_order, cancel_order, add_market_order.

Methodology
-----------
* Warm-up phase: 10,000 orders (JIT compilation, branch prediction warm-up)
* Measurement phase: 100,000 orders per operation type
* Timing: time.perf_counter_ns() around each individual operation → latency CDF
* Throughput: total events / wall-clock elapsed

Reported statistics
-------------------
  Throughput   : events/second
  p50 latency  : median
  p95 latency  : 95th percentile
  p99 latency  : 99th percentile
  p99.9 latency: 99.9th percentile ("four nines")

Run
---
    python benchmarks/benchmark_lob.py
    python benchmarks/benchmark_lob.py --n 500000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.order import Order, OrderType, Side
from engine.order_book import OrderBook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_limit_order(price: float, side: Side, qty: float = 100.0) -> Order:
    return Order(side=side, order_type=OrderType.LIMIT, price=price, quantity=qty)


def _make_market_order(side: Side, qty: float = 100.0) -> Order:
    return Order(side=side, order_type=OrderType.MARKET, quantity=qty)


def _format_ns(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.0f} ns"
    if ns < 1_000_000:
        return f"{ns/1_000:.1f} µs"
    return f"{ns/1_000_000:.2f} ms"


def _print_latency_table(name: str, latencies_ns: np.ndarray, n: int) -> None:
    elapsed_s = latencies_ns.sum() / 1e9
    throughput = n / elapsed_s

    p = np.percentile(latencies_ns, [50, 90, 95, 99, 99.9])

    print(f"\n  ┌─ {name} {'─'*(44-len(name))}┐")
    print(f"  │  N            : {n:>12,}               │")
    print(f"  │  Throughput   : {throughput:>12,.0f} events/s       │")
    print(f"  │  p50 latency  : {_format_ns(p[0]):>12}               │")
    print(f"  │  p90 latency  : {_format_ns(p[1]):>12}               │")
    print(f"  │  p95 latency  : {_format_ns(p[2]):>12}               │")
    print(f"  │  p99 latency  : {_format_ns(p[3]):>12}               │")
    print(f"  │  p99.9 latency: {_format_ns(p[4]):>12}               │")
    print(f"  └{'─'*50}┘")
    return throughput


# ---------------------------------------------------------------------------
# Benchmark: add_limit_order (no matching — pure insertion)
# ---------------------------------------------------------------------------

def bench_limit_add(n: int = 100_000, warm_up: int = 10_000) -> np.ndarray:
    """Add non-crossing limit orders (resting only, no matches)."""
    book = OrderBook("BENCH")
    rng  = np.random.default_rng(0)

    # Pre-generate orders outside the timed loop
    prices_bid = rng.uniform(90.0, 99.0, n + warm_up).round(2)
    prices_ask = rng.uniform(101.0, 110.0, n + warm_up).round(2)
    sides = rng.integers(0, 2, n + warm_up)   # 0=BID, 1=ASK

    # Warm-up
    for i in range(warm_up):
        price = prices_bid[i] if sides[i] == 0 else prices_ask[i]
        side  = Side.BID if sides[i] == 0 else Side.ASK
        book.add_limit_order(_make_limit_order(price, side))

    # Measure
    latencies = np.empty(n, dtype=np.int64)
    for i in range(n):
        price = prices_bid[i + warm_up] if sides[i + warm_up] == 0 else prices_ask[i + warm_up]
        side  = Side.BID if sides[i + warm_up] == 0 else Side.ASK
        order = _make_limit_order(price, side)
        t0 = time.perf_counter_ns()
        book.add_limit_order(order)
        latencies[i] = time.perf_counter_ns() - t0

    return latencies


# ---------------------------------------------------------------------------
# Benchmark: cancel_order
# ---------------------------------------------------------------------------

def bench_cancel(n: int = 100_000, warm_up: int = 10_000) -> np.ndarray:
    """Cancel resting limit orders (O(1) hash lookup + lazy deque skip)."""
    book = OrderBook("BENCH")
    rng  = np.random.default_rng(1)

    # Pre-populate the book with orders to cancel
    order_ids: List[str] = []
    for _ in range(n + warm_up):
        price = rng.uniform(90.0, 110.0).round(2)
        side  = Side.BID if rng.integers(0, 2) == 0 else Side.ASK
        # Keep bid below 100 and ask above 100 to avoid matching
        price = price if (side == Side.ASK and price > 100) or \
                         (side == Side.BID and price < 100) \
                else (price + 2 if side == Side.ASK else price - 2)
        order = _make_limit_order(price, side)
        book.add_limit_order(order)
        order_ids.append(order.order_id)

    # Warm-up
    for i in range(warm_up):
        book.cancel_order(order_ids[i])

    # Measure
    latencies = np.empty(n, dtype=np.int64)
    for i in range(n):
        oid = order_ids[i + warm_up]
        t0 = time.perf_counter_ns()
        book.cancel_order(oid)
        latencies[i] = time.perf_counter_ns() - t0

    return latencies


# ---------------------------------------------------------------------------
# Benchmark: add_limit_order with matching (crossing spread)
# ---------------------------------------------------------------------------

def bench_matching(n: int = 50_000, warm_up: int = 5_000) -> np.ndarray:
    """
    Alternating limit BID/ASK that fully cross the spread and consume
    exactly one resting order each time — exercises the full matching path.
    """
    book = OrderBook("BENCH")

    # Seed the book: 1000 resting bids at 99, 1000 resting asks at 101
    for _ in range(1000):
        book.add_limit_order(_make_limit_order(99.0,  Side.BID, 100.0))
        book.add_limit_order(_make_limit_order(101.0, Side.ASK, 100.0))

    # Warm-up: alternate crossing bids and asks
    for i in range(warm_up):
        if i % 2 == 0:
            book.add_limit_order(_make_limit_order(101.0, Side.BID, 100.0))
            book.add_limit_order(_make_limit_order(99.0,  Side.ASK, 100.0))
        # Replenish
        book.add_limit_order(_make_limit_order(99.0,  Side.BID, 100.0))
        book.add_limit_order(_make_limit_order(101.0, Side.ASK, 100.0))

    # Measure
    latencies = np.empty(n, dtype=np.int64)
    for i in range(n):
        # Replenish one level
        book.add_limit_order(_make_limit_order(99.0,  Side.BID, 100.0))
        book.add_limit_order(_make_limit_order(101.0, Side.ASK, 100.0))

        # Crossing order
        aggressor = _make_limit_order(101.0, Side.BID, 100.0)
        t0 = time.perf_counter_ns()
        book.add_limit_order(aggressor)
        latencies[i] = time.perf_counter_ns() - t0

    return latencies


# ---------------------------------------------------------------------------
# Benchmark: end-to-end throughput (mixed workload)
# ---------------------------------------------------------------------------

def bench_mixed_throughput(n: int = 200_000) -> float:
    """
    Realistic mixed workload: 70% limit adds, 15% cancels, 15% market orders.
    Returns events-per-second.
    """
    book = OrderBook("BENCH")
    rng  = np.random.default_rng(99)

    active_ids: List[str] = []

    t0 = time.perf_counter()
    for _ in range(n):
        r = rng.uniform()
        if r < 0.70:
            # Limit order
            side  = Side.BID if rng.integers(0, 2) == 0 else Side.ASK
            price = (rng.uniform(98.0, 99.9) if side == Side.BID
                     else rng.uniform(100.1, 102.0))
            order = _make_limit_order(round(price, 2), side)
            book.add_limit_order(order)
            active_ids.append(order.order_id)
            if len(active_ids) > 2000:
                active_ids = active_ids[-2000:]
        elif r < 0.85 and active_ids:
            # Cancel
            idx = int(rng.integers(0, len(active_ids)))
            book.cancel_order(active_ids.pop(idx))
        else:
            # Market order
            side = Side.BID if rng.integers(0, 2) == 0 else Side.ASK
            book.add_market_order(_make_market_order(side, 100.0))

    elapsed = time.perf_counter() - t0
    return n / elapsed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all(n: int = 100_000) -> None:
    print("\n" + "="*54)
    print("  LOB ENGINE — PERFORMANCE BENCHMARK")
    print("  Python reference implementation")
    print("="*54)

    print("\n[1/4] Limit order insertion (no matching) ...")
    lat_add = bench_limit_add(n)
    _print_latency_table("add_limit_order (resting)", lat_add, n)

    print("\n[2/4] Cancel order ...")
    lat_cancel = bench_cancel(n)
    _print_latency_table("cancel_order", lat_cancel, n)

    print("\n[3/4] Limit order with crossing match ...")
    lat_match = bench_matching(min(n, 50_000))
    _print_latency_table("add_limit_order (matched)", lat_match, min(n, 50_000))

    print("\n[4/4] Mixed workload throughput ...")
    thr = bench_mixed_throughput(n * 2)
    print(f"\n  Mixed throughput: {thr:>12,.0f} events/s")
    print(f"  (70% limit adds, 15% cancels, 15% market orders)\n")

    print("="*54)
    print("  NOTE: Production LOB engines use C++/Rust for the")
    print("  hot path. This Python impl validates correctness and")
    print("  serves as reference for the microstructure models.")
    print("="*54 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOB benchmark")
    parser.add_argument("--n", type=int, default=100_000,
                        help="Number of events per benchmark (default: 100,000)")
    args = parser.parse_args()
    run_all(args.n)
