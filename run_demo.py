# -*- coding: utf-8 -*-
"""
LOB Engine - Full Demo
======================
Runs the complete pipeline end-to-end:

  1. Simulate order flow via bivariate Hawkes process
  2. Process all events through the limit order book
  3. Fit Hawkes parameters via MLE (validate simulation params)
  4. Compute all five microstructure metrics
  5. Run two backtested strategies (MarketMaker, MomentumTrader)
  6. Print a formatted summary report
  7. Launch the interactive Dash dashboard

Usage
-----
    python run_demo.py                  # 30-minute sim, port 8050
    python run_demo.py --duration 3600  # 1-hour sim
    python run_demo.py --no-dashboard   # skip dashboard (terminal only)
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LOB Engine Demo")
    p.add_argument("--duration",      type=float, default=1800.0,
                   help="Simulation duration in seconds (default: 1800 = 30 min)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--symbol",        type=str,   default="SYM")
    p.add_argument("--ref-price",     type=float, default=100.0)
    p.add_argument("--no-dashboard",  action="store_true",
                   help="Skip launching the Dash dashboard")
    p.add_argument("--port",          type=int,   default=8050)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    width = 60
    print(f"\n{'='*width}")
    print(f"  {title}")
    print('='*width)


def _extract_bid_ask_times(event_log, duration_s):
    """Reconstruct per-side event timestamps from the event log."""
    bid_times = np.array([e["t"] for e in event_log
                          if e["type"] in ("limit", "market") and e["side"] == "bid"])
    ask_times = np.array([e["t"] for e in event_log
                          if e["type"] in ("limit", "market") and e["side"] == "ask"])
    return bid_times, ask_times


def _build_microstructure_inputs(result):
    """Convert LOB trade tape and snapshots into numpy arrays for metrics."""
    from engine.order import Side as _Side

    trades = result.trades
    snaps  = result.snapshots

    if not trades:
        return None

    trade_prices  = np.array([t.price    for t in trades])
    trade_volumes = np.array([t.quantity for t in trades])
    trade_sides   = np.array([
        1.0 if t.aggressor_side == _Side.BID else -1.0
        for t in trades
    ])
    trade_times_ns = np.array([t.timestamp for t in trades])

    bid_prices_snap  = np.array([s.best_bid       for s in snaps if s.best_bid  is not None])
    ask_prices_snap  = np.array([s.best_ask       for s in snaps if s.best_ask  is not None])
    bid_vols_snap    = np.array([s.bid_volume_top5 for s in snaps if s.best_bid is not None])
    ask_vols_snap    = np.array([s.ask_volume_top5 for s in snaps if s.best_ask is not None])
    mid_prices_snap  = np.array([s.mid_price       for s in snaps if s.mid_price is not None])

    return {
        "trade_prices":    trade_prices,
        "trade_volumes":   trade_volumes,
        "trade_sides":     trade_sides,
        "trade_times_ns":  trade_times_ns,
        "bid_prices_snap": bid_prices_snap,
        "ask_prices_snap": ask_prices_snap,
        "bid_vols_snap":   bid_vols_snap,
        "ask_vols_snap":   ask_vols_snap,
        "mid_prices_snap": mid_prices_snap,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # STEP 1 — Simulate                                                    #
    # ------------------------------------------------------------------ #
    section("STEP 1 / 5 - Hawkes-Driven Order Flow Simulation")
    from feed.simulator import MarketSimulator

    sim = MarketSimulator(
        symbol=args.symbol,
        ref_price=args.ref_price,
        seed=args.seed,
    )
    print(f"  Simulating {args.duration:.0f} s of market activity ...")
    t0 = time.perf_counter()
    result = sim.run(duration_s=args.duration)
    print(result.summary())

    # ------------------------------------------------------------------ #
    # STEP 2 — Fit Hawkes parameters                                       #
    # ------------------------------------------------------------------ #
    section("STEP 2 / 5 - Hawkes Process MLE Fitting")
    from hawkes.process import fit as hawkes_fit

    bid_times, ask_times = _extract_bid_ask_times(result.event_log, args.duration)

    bid_params = ask_params = None
    if len(bid_times) >= 50:
        print("  Fitting bid-side Hawkes process ...")
        bid_params_raw = hawkes_fit(bid_times, T=args.duration, n_restarts=8, seed=args.seed)
        bid_params = {k: bid_params_raw[k] for k in ("mu", "alpha", "beta")}
        print(f"  Bid  mu={bid_params_raw['mu']:.4f}  alpha={bid_params_raw['alpha']:.4f}  "
              f"beta={bid_params_raw['beta']:.4f}  n={bid_params_raw['branching_ratio']:.4f}  "
              f"(stationary: {bid_params_raw['branching_ratio'] < 1})")
        print(f"       log-L={bid_params_raw['log_likelihood']:.2f}  "
              f"AIC={bid_params_raw['aic']:.2f}  "
              f"E[lam]={bid_params_raw['mean_intensity']:.4f} events/s")

    if len(ask_times) >= 50:
        print("  Fitting ask-side Hawkes process ...")
        ask_params_raw = hawkes_fit(ask_times, T=args.duration, n_restarts=8, seed=args.seed)
        ask_params = {k: ask_params_raw[k] for k in ("mu", "alpha", "beta")}
        print(f"  Ask  mu={ask_params_raw['mu']:.4f}  alpha={ask_params_raw['alpha']:.4f}  "
              f"beta={ask_params_raw['beta']:.4f}  n={ask_params_raw['branching_ratio']:.4f}  "
              f"(stationary: {ask_params_raw['branching_ratio'] < 1})")
        print(f"       log-L={ask_params_raw['log_likelihood']:.2f}  "
              f"AIC={ask_params_raw['aic']:.2f}  "
              f"E[lam]={ask_params_raw['mean_intensity']:.4f} events/s")

    # ------------------------------------------------------------------ #
    # STEP 3 — Microstructure metrics                                      #
    # ------------------------------------------------------------------ #
    section("STEP 3 / 5 - Market Microstructure Metrics")
    from microstructure.metrics import (
        amihud_illiquidity,
        bin_trades,
        kyle_lambda,
        order_flow_imbalance,
        roll_spread,
        vpin,
    )

    ms = _build_microstructure_inputs(result)
    kyle_result = vpin_result = ofi_result = amihud_result = roll_result = None
    mid_prices_binned = signed_volumes_binned = None

    if ms is not None and len(ms["trade_prices"]) >= 10:
        # Compute bin size from actual timestamp span (targeting ~30 bins)
        ts_span_ns = int(ms["trade_times_ns"][-1] - ms["trade_times_ns"][0])
        interval_ns = max(ts_span_ns // 30, 1_000_000)   # at least 1 ms per bin
        binned = bin_trades(
            ms["trade_times_ns"], ms["trade_prices"],
            ms["trade_volumes"],  ms["trade_sides"],
            interval_ns=interval_ns,
        )

        mid_prices_binned    = binned["close_prices"]
        signed_volumes_binned = binned["signed_volume"]

        # Kyle's lambda (needs >= 3 bins)
        if binned["n_bins"] >= 3:
            kyle_result = kyle_lambda(mid_prices_binned, signed_volumes_binned)
            print(f"\n  Kyle's lambda = {kyle_result['lambda']:.6f}  "
                  f"({kyle_result['interpretation']})")
            print(f"            R2 = {kyle_result['r_squared']:.4f}  "
                  f"t-stat = {kyle_result['t_stat']:.3f}  "
                  f"N = {kyle_result['n_obs']}")
        else:
            kyle_result = None
            print("\n  Kyle's lambda: insufficient data (need >= 3 time bins)")

        # VPIN
        vpin_result = vpin(
            ms["trade_prices"], ms["trade_volumes"], ms["trade_sides"]
        )
        print(f"\n  VPIN      = {vpin_result['vpin']:.4f}  "
              f"({vpin_result['interpretation']})")
        print(f"            Buckets filled: {vpin_result['n_buckets_filled']}")

        # OFI
        if len(ms["bid_prices_snap"]) >= 3:
            ofi_result = order_flow_imbalance(
                ms["bid_prices_snap"], ms["ask_prices_snap"],
                ms["bid_vols_snap"],  ms["ask_vols_snap"],
            )
            corr = ofi_result["ofi_price_correlation"]
            print(f"\n  OFI-Price corr = {corr:.4f}  "
                  f"({ofi_result['interpretation']})")

        # Amihud (log_returns has n-1 elements; align volume array)
        if len(binned["log_returns"]) >= 5:
            amihud_result = amihud_illiquidity(
                binned["log_returns"], binned["total_volume"][1:]
            )
            print(f"\n  Amihud ILLIQ = {amihud_result['illiquidity']:.6f}  "
                  f"({amihud_result['interpretation']})")

        # Roll spread
        if len(ms["mid_prices_snap"]) >= 10:
            roll_result = roll_spread(ms["mid_prices_snap"])
            rs = roll_result["roll_spread"]
            print(f"\n  Roll Spread = "
                  f"{'N/A (positive serial cov)' if np.isnan(rs) else f'${rs:.5f}'}")

        # Spread stats from snapshots
        snaps = result.snapshots
        spreads = [s.spread for s in snaps if s.spread is not None]
        if spreads:
            arr = np.array(spreads)
            mid_arr = np.array([s.mid_price for s in snaps if s.mid_price])
            bps = arr / mid_arr[:len(arr)] * 10000
            print(f"\n  Quoted Spread  mean={arr.mean():.4f}  "
                  f"median={np.median(arr):.4f}  "
                  f"({np.median(bps):.2f} bps median)")

    # ------------------------------------------------------------------ #
    # STEP 4 — Backtesting                                                 #
    # ------------------------------------------------------------------ #
    section("STEP 4 / 5 - Strategy Backtesting")
    from backtest.engine import BacktestEngine, MarketMaker, MomentumTrader

    # We run a fresh simulator for backtesting (same seed -> same events)
    print("  Re-simulating for backtesting ...")
    bt_result_mm = bt_result_mo = None

    # Market Maker
    sim_bt = MarketSimulator(symbol=args.symbol, ref_price=args.ref_price, seed=args.seed)
    bt_result_mm_sim = sim_bt.run(duration_s=args.duration)
    engine = BacktestEngine()
    bt_result_mm = engine.run(
        book=bt_result_mm_sim.book,
        event_log=bt_result_mm_sim.event_log,
        strategy=MarketMaker(),
    )
    print(bt_result_mm.summary())

    # Momentum Trader (separate fresh book)
    sim_bt2 = MarketSimulator(symbol=args.symbol, ref_price=args.ref_price, seed=args.seed)
    bt_result_mo_sim = sim_bt2.run(duration_s=args.duration)
    bt_result_mo = engine.run(
        book=bt_result_mo_sim.book,
        event_log=bt_result_mo_sim.event_log,
        strategy=MomentumTrader(),
    )
    print(bt_result_mo.summary())

    # ------------------------------------------------------------------ #
    # STEP 5 — Dashboard                                                   #
    # ------------------------------------------------------------------ #
    section("STEP 5 / 5 - Interactive Dashboard")
    if args.no_dashboard:
        print("  --no-dashboard flag set. Skipping.")
        print(f"\n  Total wall-clock time: {time.perf_counter()-t0:.2f} s")
        return

    from visualization.dashboard import launch_dashboard
    launch_dashboard(
        result=result,
        bid_times=bid_times if len(bid_times) >= 10 else None,
        ask_times=ask_times if len(ask_times) >= 10 else None,
        bid_params=bid_params,
        ask_params=ask_params,
        kyle_result=kyle_result,
        vpin_result=vpin_result,
        mid_prices=mid_prices_binned,
        signed_volumes=signed_volumes_binned,
        backtest_results=[r for r in [bt_result_mm, bt_result_mo] if r is not None],
        port=args.port,
    )


if __name__ == "__main__":
    main()
