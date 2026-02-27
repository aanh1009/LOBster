from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def _njit(*args, **kwargs):
        def _wrap(fn): return fn
        if len(args) == 1 and callable(args[0]): return args[0]
        return _wrap

@_njit(cache=True)
def _log_likelihood_kernel(timestamps, mu, alpha, beta):
    n = len(timestamps)
    if n == 0:
        return 0.0
    T = timestamps[n - 1]
    EPS = 1e-300
    A = 0.0
    log_sum = np.log(max(mu, EPS))
    for i in range(1, n):
        dt = timestamps[i] - timestamps[i - 1]
        A = np.exp(-beta * dt) * (1.0 + A)
        lam = mu + alpha * A
        log_sum += np.log(max(lam, EPS))
    integral = mu * T
    for i in range(n):
        integral += (alpha / beta) * (1.0 - np.exp(-beta * (T - timestamps[i])))
    return log_sum - integral


def simulate(mu, alpha, beta, T, seed=None):
    if alpha >= beta:
        raise ValueError(
            f"Branching ratio alpha/beta = {alpha/beta:.3f} >= 1 -- process is non-stationary."
        )
    rng = np.random.default_rng(seed)
    events: List[float] = []
    t = 0.0
    A = 0.0
    lambda_upper = mu
    while t < T:
        u = rng.exponential(1.0 / max(lambda_upper, 1e-12))
        t_cand = t + u
        if t_cand > T:
            break
        A = A * np.exp(-beta * u)
        lambda_cand = mu + alpha * A
        if rng.uniform() * lambda_upper <= lambda_cand:
            events.append(t_cand)
            A += 1.0
            lambda_upper = lambda_cand + alpha
        else:
            lambda_upper = lambda_cand
        t = t_cand
    return np.array(events, dtype=float)


def fit(timestamps, T=None, n_restarts=10, seed=42):
    ts = np.sort(np.asarray(timestamps, dtype=float))
    if T is None:
        T = float(ts[-1])
    n = len(ts)
    rng = np.random.default_rng(seed)
    best_result = None
    best_ll = -np.inf

    def neg_ll(params):
        mu, alpha, beta = params
        if mu <= 0 or alpha <= 0 or beta <= 0:
            return 1e10
        if alpha >= beta:
            return 1e10
        ll = _log_likelihood_kernel(ts, mu, alpha, beta)
        return -ll if np.isfinite(ll) else 1e10

    for _ in range(n_restarts):
        mu0   = rng.uniform(0.05, 5.0)
        beta0 = rng.uniform(0.5, 10.0)
        alpha0 = rng.uniform(0.01, 0.95) * beta0
        result = minimize(neg_ll, [mu0, alpha0, beta0], method="L-BFGS-B",
                          bounds=[(1e-6, None)] * 3,
                          options={"maxiter": 1000, "ftol": 1e-12})
        if result.success or result.fun < 1e9:
            ll = -result.fun
            if ll > best_ll:
                best_ll = ll
                best_result = result

    if best_result is None:
        raise RuntimeError("Hawkes MLE failed to converge.")

    mu, alpha, beta = best_result.x
    n_params = 3
    return {
        "mu": float(mu), "alpha": float(alpha), "beta": float(beta),
        "branching_ratio": float(alpha / beta),
        "log_likelihood": float(best_ll),
        "aic": float(2 * n_params - 2 * best_ll),
        "bic": float(np.log(n) * n_params - 2 * best_ll),
        "converged": bool(best_result.success),
        "n_events": int(n),
        "mean_intensity": float(mu / (1.0 - alpha / beta)),
    }
