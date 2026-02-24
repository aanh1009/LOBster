from __future__ import annotations
from typing import List, Optional
import numpy as np

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
    """Ogata's (1981) modified thinning algorithm."""
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
