from __future__ import annotations
from typing import Dict
import numpy as np


def kyle_lambda(mid_prices, signed_volumes, interval_seconds=60.0):
    """Kyle (1985) price impact coefficient via OLS."""
    if len(mid_prices) < 3:
        raise ValueError("Need at least 3 midprice observations")
    delta_p = np.diff(mid_prices)
    q = np.asarray(signed_volumes[:len(delta_p)])
    X = np.column_stack([np.ones(len(q)), q])
    coeffs, _, _, _ = np.linalg.lstsq(X, delta_p, rcond=None)
    alpha_hat, lambda_hat = coeffs
    y_pred = alpha_hat + lambda_hat * q
    ss_res = np.sum((delta_p - y_pred) ** 2)
    ss_tot = np.sum((delta_p - delta_p.mean()) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
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


def vpin(trade_prices, trade_volumes, trade_sides, bucket_size=None, n_buckets=50):
    """Easley, Lopez de Prado, O'Hara (2012) VPIN toxicity estimator."""
    from typing import List
    volumes = np.asarray(trade_volumes, dtype=float)
    sides   = np.asarray(trade_sides,   dtype=float)
    if bucket_size is None:
        bucket_size = volumes.sum() / n_buckets
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
            overflow = accumulated - bucket_size
            accumulated = overflow
            if side > 0:
                bucket_buy, bucket_sell = overflow, 0.0
            else:
                bucket_buy, bucket_sell = 0.0, overflow
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
