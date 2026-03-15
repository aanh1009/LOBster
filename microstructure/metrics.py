"""
Market Microstructure Metrics
==============================

Implements five core metrics used by academic researchers and practitioners
to characterize liquidity, price impact, and informed trading:

  1. Kyle's Lambda    — adverse selection / price impact coefficient
  2. VPIN             — volume-synchronized probability of informed trading
  3. Order Flow Imbalance (OFI) — short-horizon price predictor
  4. Amihud Illiquidity Ratio  — illiquidity per unit of volume
  5. Roll's Spread Estimator   — implied spread from serial covariance of returns

All functions accept plain numpy arrays and return dicts for easy inspection.

References
----------
Kyle (1985) "Continuous Auctions and Insider Trading"
Easley, Lopez de Prado, O'Hara (2012) "Flow Toxicity and Liquidity..."
Cont, Kukanov, Stoikov (2014) "The Price Impact of Order Book Events"
Amihud (2002) "Illiquidity and Stock Returns"
Roll (1984) "A Simple Implicit Measure of the Bid-Ask Spread"
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 1. Kyle's Lambda
# ---------------------------------------------------------------------------

def kyle_lambda(
    mid_prices: np.ndarray,
    signed_volumes: np.ndarray,
    interval_seconds: float = 60.0,
) -> Dict:
    """
    Estimate Kyle's (1985) price impact coefficient λ via OLS.

    Model:  ΔP_t = λ · Q_t + ε_t

    where:
        ΔP_t = midprice change over an interval
        Q_t  = net signed order flow (buy volume − sell volume)
        λ    = price impact per unit of order flow
               • Higher λ → more adverse selection → less liquid market
               • Units: $/share per share = $/share²  (commonly bps/share)

    Parameters
    ----------
    mid_prices     : array of midprice observations at each interval boundary
    signed_volumes : array of net signed volume in each interval (length = len(mid_prices)-1)
    interval_seconds : duration of each interval (informational only)

    Returns
    -------
    dict with keys: lambda, intercept, r_squared, t_stat, n_obs
    """
    if len(mid_prices) < 3:
        raise ValueError("Need at least 3 midprice observations")

    delta_p = np.diff(mid_prices)                   # shape (n-1,)
    q = np.asarray(signed_volumes[: len(delta_p)])  # align lengths

    # OLS with intercept:  [1, q] @ [α, λ]ᵀ = ΔP
    X = np.column_stack([np.ones(len(q)), q])
    coeffs, residuals, _, _ = np.linalg.lstsq(X, delta_p, rcond=None)
    alpha_hat, lambda_hat = coeffs

    y_pred = alpha_hat + lambda_hat * q
    ss_res = np.sum((delta_p - y_pred) ** 2)
    ss_tot = np.sum((delta_p - delta_p.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    # t-statistic for λ
    n = len(delta_p)
    se = np.sqrt(ss_res / max(n - 2, 1)) / (np.std(q) * np.sqrt(n))
    t_stat = lambda_hat / se if se > 1e-12 else np.nan

    return {
        "lambda": float(lambda_hat),
        "intercept": float(alpha_hat),
        "r_squared": float(r_squared),
        "t_stat": float(t_stat),
        "n_obs": int(n),
        "interpretation": (
            "high adverse selection" if lambda_hat > 0.005
            else "moderate" if lambda_hat > 0.001
            else "low adverse selection"
        ),
    }


# ---------------------------------------------------------------------------
# 2. VPIN  (Volume-synchronized Probability of Informed Trading)
# ---------------------------------------------------------------------------

def vpin(
    trade_prices: np.ndarray,
    trade_volumes: np.ndarray,
    trade_sides: np.ndarray,     # +1 = buy-initiated,  -1 = sell-initiated
    bucket_size: Optional[float] = None,
    n_buckets: int = 50,
) -> Dict:
    """
    Easley, Lopez de Prado, O'Hara (2012) VPIN estimator.

    VPIN ≈ E[|V_buy − V_sell|] / V_bucket

    Volume buckets of equal size V_bucket are formed sequentially.
    In each bucket the signed imbalance |V_buy − V_sell| is computed.
    VPIN is the sample mean of these imbalances, normalized by bucket size.

    A VPIN close to 1 means almost all volume is one-sided (informed).
    A VPIN close to 0 means buy and sell volume are balanced (uninformed).

    Parameters
    ----------
    trade_prices  : execution prices
    trade_volumes : trade sizes (always positive)
    trade_sides   : +1 (buy aggressor) / -1 (sell aggressor)
    bucket_size   : volume per bucket; if None, derived from n_buckets
    n_buckets     : target number of buckets (used when bucket_size=None)

    Returns
    -------
    dict with keys: vpin, bucket_vpins, bucket_size, n_buckets_filled
    """
    volumes = np.asarray(trade_volumes, dtype=float)
    sides   = np.asarray(trade_sides,   dtype=float)

    total_volume = volumes.sum()
    if bucket_size is None:
        bucket_size = total_volume / n_buckets

    bucket_vpins: List[float] = []
    bucket_buy = bucket_sell = accumulated = 0.0

    for vol, side in zip(volumes, sides):
        if side > 0:
            bucket_buy += vol
        else:
            bucket_sell += vol
        accumulated += vol

        while accumulated >= bucket_size:
            bucket_vpins.append(abs(bucket_buy - bucket_sell) / bucket_size)
            # Reset bucket
            overflow = accumulated - bucket_size
            # Attribute overflow proportionally to current trade direction
            accumulated = overflow
            if side > 0:
                bucket_buy  = overflow
                bucket_sell = 0.0
            else:
                bucket_buy  = 0.0
                bucket_sell = overflow

    arr = np.array(bucket_vpins)
    return {
        "vpin": float(arr.mean()) if len(arr) > 0 else np.nan,
        "vpin_rolling": arr.tolist(),
        "bucket_size": bucket_size,
        "n_buckets_filled": len(arr),
        "interpretation": (
            "high toxicity - possible informed trading"
            if (arr.mean() if len(arr) else 0) > 0.4
            else "moderate" if (arr.mean() if len(arr) else 0) > 0.2
            else "low toxicity"
        ),
    }


# ---------------------------------------------------------------------------
# 3. Order Flow Imbalance (OFI)
# ---------------------------------------------------------------------------

def order_flow_imbalance(
    bid_prices: np.ndarray,      # best bid price at each snapshot
    ask_prices: np.ndarray,      # best ask price at each snapshot
    bid_volumes: np.ndarray,     # total bid volume at best bid at each snapshot
    ask_volumes: np.ndarray,     # total ask volume at best ask at each snapshot
) -> Dict:
    """
    Cont, Kukanov, Stoikov (2014) Order Flow Imbalance.

    OFI_t captures the net change in liquidity supply at the top of the book:

        OFI_t = ΔBid_t − ΔAsk_t

    where
        ΔBid_t = bid_vol[t] · 1(bid_px[t] ≥ bid_px[t-1])
                 − bid_vol[t-1] · 1(bid_px[t] ≤ bid_px[t-1])

        ΔAsk_t = ask_vol[t] · 1(ask_px[t] ≤ ask_px[t-1])
                 − ask_vol[t-1] · 1(ask_px[t] ≥ ask_px[t-1])

    Positive OFI → more buying pressure → midprice tends to rise.
    Negative OFI → more selling pressure → midprice tends to fall.

    Returns
    -------
    dict with keys: ofi_series, mean_ofi, ofi_price_correlation
    """
    bp = np.asarray(bid_prices,  dtype=float)
    ap = np.asarray(ask_prices,  dtype=float)
    bv = np.asarray(bid_volumes, dtype=float)
    av = np.asarray(ask_volumes, dtype=float)

    n = len(bp) - 1

    delta_bid = (
        bv[1:] * (bp[1:] >= bp[:-1]).astype(float)
        - bv[:-1] * (bp[1:] <= bp[:-1]).astype(float)
    )
    delta_ask = (
        av[1:] * (ap[1:] <= ap[:-1]).astype(float)
        - av[:-1] * (ap[1:] >= ap[:-1]).astype(float)
    )
    ofi = delta_bid - delta_ask

    mid = (bp + ap) / 2.0
    delta_mid = np.diff(mid)
    ofi_aligned = ofi[: len(delta_mid)]

    # Correlation of OFI with subsequent midprice change
    if np.std(ofi_aligned) > 1e-12 and np.std(delta_mid) > 1e-12:
        corr = float(np.corrcoef(ofi_aligned, delta_mid)[0, 1])
    else:
        corr = np.nan

    return {
        "ofi_series": ofi.tolist(),
        "mean_ofi": float(ofi.mean()),
        "ofi_price_correlation": corr,
        "interpretation": (
            f"OFI explains {corr**2*100:.1f}% of midprice variance"
            if not np.isnan(corr)
            else "insufficient data"
        ),
    }


# ---------------------------------------------------------------------------
# 4. Amihud Illiquidity Ratio
# ---------------------------------------------------------------------------

def amihud_illiquidity(
    returns: np.ndarray,
    volumes: np.ndarray,
    annualization_factor: float = 252.0,
) -> Dict:
    """
    Amihud (2002) illiquidity ratio.

    ILLIQ = (1/T) Σ_t |R_t| / Vol_t

    Measures price impact per unit of dollar volume traded.
    Higher ILLIQ → more illiquid market.

    Parameters
    ----------
    returns : array of period returns (e.g., log or simple)
    volumes : array of dollar volumes in the same periods

    Returns
    -------
    dict with keys: illiquidity, annualized_illiquidity, n_obs
    """
    r = np.asarray(returns, dtype=float)
    v = np.asarray(volumes, dtype=float)
    valid = v > 0
    ratios = np.abs(r[valid]) / v[valid]

    illiq = float(ratios.mean()) if ratios.size > 0 else np.nan

    return {
        "illiquidity": illiq,
        "annualized_illiquidity": illiq * annualization_factor,
        "n_obs": int(ratios.size),
        "interpretation": (
            "highly illiquid" if illiq > 1e-4
            else "moderately liquid" if illiq > 1e-6
            else "liquid"
        ),
    }


# ---------------------------------------------------------------------------
# 5. Roll's Spread Estimator
# ---------------------------------------------------------------------------

def roll_spread(mid_prices: np.ndarray) -> Dict:
    """
    Roll (1984) implicit spread from the serial covariance of price changes.

    Model: transactions bounce between bid and ask with half-spread c.
        ΔP_t = c(Q_t − Q_{t-1}) + ε_t   where Q_t ∈ {-1, +1}

    This implies:  Cov(ΔP_t, ΔP_{t-1}) = −c²
    Roll Spread = 2·√(−Cov(ΔP_t, ΔP_{t-1}))

    The estimator is purely statistical — no trade direction data needed.

    Returns
    -------
    dict with keys: roll_spread, serial_covariance, n_obs
    """
    p = np.asarray(mid_prices, dtype=float)
    delta_p = np.diff(p)
    # Serial covariance: Cov(ΔP_t, ΔP_{t-1})
    cov = float(np.cov(delta_p[:-1], delta_p[1:])[0, 1])

    if cov < 0:
        roll_s = 2.0 * np.sqrt(-cov)
    else:
        # Positive serial covariance: Roll model doesn't apply cleanly
        roll_s = np.nan

    return {
        "roll_spread": float(roll_s) if not np.isnan(roll_s) else np.nan,
        "serial_covariance": cov,
        "n_obs": len(delta_p),
        "interpretation": (
            f"Implied half-spread: {roll_s/2:.5f}" if not np.isnan(roll_s)
            else "non-negative serial covariance — Roll model inapplicable"
        ),
    }


# ---------------------------------------------------------------------------
# Utility: bin trade tape into fixed time intervals
# ---------------------------------------------------------------------------

def bin_trades(
    timestamps_ns: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    sides: np.ndarray,
    interval_ns: int = 60_000_000_000,   # 1 minute in nanoseconds
) -> Dict[str, np.ndarray]:
    """
    Aggregate a trade tape into fixed time intervals.

    Returns arrays of: open/close mid-price, total volume, signed volume,
    absolute return per interval — ready for Kyle and Amihud estimators.
    """
    t = np.asarray(timestamps_ns, dtype=np.int64)
    p = np.asarray(prices,        dtype=float)
    v = np.asarray(volumes,       dtype=float)
    s = np.asarray(sides,         dtype=float)

    t_start = t[0]
    t_end   = t[-1]
    edges = np.arange(t_start, t_end + interval_ns, interval_ns)

    bin_idx = np.digitize(t, edges) - 1
    n_bins  = len(edges) - 1

    open_prices  = np.full(n_bins, np.nan)
    close_prices = np.full(n_bins, np.nan)
    total_vol    = np.zeros(n_bins)
    signed_vol   = np.zeros(n_bins)

    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            open_prices[b]  = p[mask][0]
            close_prices[b] = p[mask][-1]
            total_vol[b]    = v[mask].sum()
            signed_vol[b]   = (v[mask] * s[mask]).sum()

    # Forward-fill missing open/close
    for arr in (open_prices, close_prices):
        last = arr[0]
        for i in range(len(arr)):
            if np.isnan(arr[i]):
                arr[i] = last
            else:
                last = arr[i]

    log_returns = np.log(close_prices[1:] / close_prices[:-1])
    log_returns = np.where(np.isfinite(log_returns), log_returns, 0.0)

    return {
        "open_prices":  open_prices,
        "close_prices": close_prices,
        "log_returns":  log_returns,
        "total_volume": total_vol,
        "signed_volume": signed_vol,
        "n_bins": n_bins,
    }
