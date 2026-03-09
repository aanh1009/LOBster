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
