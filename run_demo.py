# -*- coding: utf-8 -*-
"""LOB Engine demo - simulation and Hawkes fitting."""
from __future__ import annotations
import argparse, sys, time
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--duration",     type=float, default=1800.0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--symbol",       type=str,   default="SYM")
    p.add_argument("--ref-price",    type=float, default=100.0)
    p.add_argument("--no-dashboard", action="store_true")
    p.add_argument("--port",         type=int,   default=8050)
    return p.parse_args()

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def main():
    args = parse_args()
    from feed.simulator import MarketSimulator
    section("STEP 1 / 5 - Hawkes-Driven Order Flow Simulation")
    sim = MarketSimulator(symbol=args.symbol, ref_price=args.ref_price, seed=args.seed)
    print(f"  Simulating {args.duration:.0f} s of market activity ...")
    result = sim.run(duration_s=args.duration)
    print(result.summary())

    from hawkes.process import fit as hawkes_fit
    section("STEP 2 / 5 - Hawkes Process MLE Fitting")
    bid_times = np.array([e["t"] for e in result.event_log
                           if e["type"] in ("limit","market") and e["side"] == "bid"])
    ask_times = np.array([e["t"] for e in result.event_log
                           if e["type"] in ("limit","market") and e["side"] == "ask"])
    if len(bid_times) >= 50:
        print("  Fitting bid-side ...")
        r = hawkes_fit(bid_times, T=args.duration, n_restarts=8, seed=args.seed)
        print(f"  Bid mu={r['mu']:.4f} alpha={r['alpha']:.4f} "
              f"beta={r['beta']:.4f} n={r['branching_ratio']:.4f}")

if __name__ == "__main__":
    main()
