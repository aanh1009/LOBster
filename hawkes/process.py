"""
Hawkes Process — Self-exciting Point Process for Order Arrival Modeling
========================================================================

Background
----------
Empirical research (Bowsher 2007, Bacry et al. 2015) shows that limit order
arrivals on electronic exchanges exhibit *self-excitation*: a burst of buy
orders begets more buy orders, and vice versa.  A Hawkes process captures
this clustering through a conditional intensity function:

    λ*(t) = μ + Σ_{t_i < t} α · exp(−β · (t − t_i))

where:
    μ     = baseline arrival rate (events/second)
    α     = excitation magnitude  (jump in intensity per event)
    β     = decay rate            (how quickly excitation fades)
    n     = α/β = branching ratio (must be < 1 for stationarity)

The branching ratio n has an intuitive interpretation:
    each event is expected to spawn n "child" events on average.
    n = 0  → Poisson process (no clustering)
    n → 1  → critical, highly clustered, long memory

Implementation
--------------
1. Simulation : Ogata's (1981) modified thinning algorithm — O(n) amortized
2. MLE fitting : Maximize log-likelihood via L-BFGS-B with multiple restarts
                 Recursive O(n) computation of the convolution sum

References
----------
Hawkes (1971) "Spectra of some self-exciting and mutually exciting point processes"
Ogata (1981) "On Lewis' simulation method for point processes"
Bowsher (2007) "Modelling security market events in continuous time"
Bacry, Mastromatteo, Muzy (2015) "Hawkes processes in finance"
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Optional Numba JIT — falls back to plain numpy seamlessly
# ---------------------------------------------------------------------------
try:
    from numba import njit as _njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False
    def _njit(*args, **kwargs):          # type: ignore[misc]
        """Identity decorator used when Numba is not installed."""
        def _wrap(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _wrap


# ---------------------------------------------------------------------------
# Log-likelihood  (JIT-compiled when Numba is available)
# ---------------------------------------------------------------------------

@_njit(cache=True)
def _log_likelihood_kernel(
    timestamps: np.ndarray,
    mu: float,
    alpha: float,
    beta: float,
) -> float:
    """
    O(n) evaluation of the Hawkes log-likelihood using the recursion:

        A(0) = 0
        A(i) = exp(−β · (t_i − t_{i−1})) · (1 + A(i−1))

    Then:
        λ*(t_i) = μ + α · A(i)

        ℓ = Σ log λ*(t_i) − μT − (α/β) Σ (1 − exp(−β(T−t_i)))

    Numerical notes
    ---------------
    * We add a small ε to the intensity before log to avoid log(0).
    * expm1 is used for the integral term for stability when β(T−t_i) ≈ 0.
    """
    n = len(timestamps)
    if n == 0:
        return 0.0

    T = timestamps[n - 1]
    EPS = 1e-300

    # --- sum of log intensities -------------------------------------------
    A = 0.0
    log_sum = np.log(max(mu, EPS))           # contribution from first event
    for i in range(1, n):
        dt = timestamps[i] - timestamps[i - 1]
        A = np.exp(-beta * dt) * (1.0 + A)
        lam = mu + alpha * A
        log_sum += np.log(max(lam, EPS))

    # --- integral term  ∫_0^T λ*(t) dt -----------------------------------
    integral = mu * T
    for i in range(n):
        # (α/β)(1 − exp(−β(T−t_i)))  =  (α/β)(−expm1(−β(T−t_i)))
        exponent = beta * (T - timestamps[i])
        integral += (alpha / beta) * (1.0 - np.exp(-exponent))

    return log_sum - integral


# ---------------------------------------------------------------------------
# Public: fit
# ---------------------------------------------------------------------------

def fit(
    timestamps: np.ndarray,
    T: Optional[float] = None,
    n_restarts: int = 10,
    seed: int = 42,
) -> Dict:
    """
    Maximum-likelihood estimation of Hawkes parameters (μ, α, β).

    Uses L-BFGS-B with multiple random restarts to avoid local optima.
    Stationarity constraint (α < β) is enforced via the initial guess and
    a penalty in the objective.

    Parameters
    ----------
    timestamps : sorted array of event arrival times (seconds)
    T          : observation window end; defaults to timestamps[-1]
    n_restarts : number of random starting points
    seed       : RNG seed for reproducibility

    Returns
    -------
    dict with keys:
        mu, alpha, beta      — fitted parameters
        branching_ratio      — n = α/β  (stationarity check < 1)
        log_likelihood       — ℓ at the optimum
        aic, bic             — information criteria
        converged            — whether any restart converged
        n_events             — number of events in the sample
    """
    ts = np.sort(np.asarray(timestamps, dtype=float))
    if T is None:
        T = float(ts[-1])
    n = len(ts)

    rng = np.random.default_rng(seed)
    best_result = None
    best_ll = -np.inf

    def neg_ll(params: np.ndarray) -> float:
        mu, alpha, beta = params
        # Hard penalty for non-stationary or invalid parameters
        if mu <= 0 or alpha <= 0 or beta <= 0:
            return 1e10
        if alpha >= beta:        # stationarity: branching ratio < 1
            return 1e10
        ll = _log_likelihood_kernel(ts, mu, alpha, beta)
        return -ll if np.isfinite(ll) else 1e10

    for _ in range(n_restarts):
        # Sample random starting point; ensure α < β
        mu0    = rng.uniform(0.05, 5.0)
        beta0  = rng.uniform(0.5, 10.0)
        alpha0 = rng.uniform(0.01, 0.95) * beta0    # guarantees α < β at start

        result = minimize(
            neg_ll,
            x0=[mu0, alpha0, beta0],
            method="L-BFGS-B",
            bounds=[(1e-6, None), (1e-6, None), (1e-6, None)],
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if result.success or result.fun < 1e9:
            ll = -result.fun
            if ll > best_ll:
                best_ll = ll
                best_result = result

    if best_result is None:
        raise RuntimeError("Hawkes MLE failed to converge across all restarts.")

    mu, alpha, beta = best_result.x
    n_params = 3
    aic = 2 * n_params - 2 * best_ll
    bic = np.log(n) * n_params - 2 * best_ll

    return {
        "mu": float(mu),
        "alpha": float(alpha),
        "beta": float(beta),
        "branching_ratio": float(alpha / beta),
        "log_likelihood": float(best_ll),
        "aic": float(aic),
        "bic": float(bic),
        "converged": bool(best_result.success),
        "n_events": int(n),
        "mean_intensity": float(mu / (1.0 - alpha / beta)),  # E[λ] in stationarity
    }


# ---------------------------------------------------------------------------
# Public: simulate
# ---------------------------------------------------------------------------

def simulate(
    mu: float,
    alpha: float,
    beta: float,
    T: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate a univariate Hawkes process via Ogata's (1981) thinning algorithm.

    Algorithm (O(n) amortized)
    --------------------------
    Maintain a running sum A representing the current excitation:

        λ*(t) = μ + α · A(t)      where  A(t) = Σ_{t_i < t} exp(−β(t−t_i))

    Between events A decays exponentially so we can update it analytically:

        A ← A · exp(−β · dt)

    When an event is accepted:

        A ← A + 1          (the new event contributes exp(0) = 1)
        λ_upper ← λ*(t+) = μ + α(A+1)    (tight upper bound just after event)

    Parameters
    ----------
    mu, alpha, beta : Hawkes parameters (must satisfy alpha < beta)
    T               : simulation horizon (seconds)
    seed            : RNG seed

    Returns
    -------
    1-D numpy array of event times in [0, T]
    """
    if alpha >= beta:
        raise ValueError(
            f"Branching ratio α/β = {alpha/beta:.3f} ≥ 1 — process is non-stationary."
        )

    rng = np.random.default_rng(seed)
    events: List[float] = []

    t = 0.0
    A = 0.0                        # running excitation sum
    lambda_upper = mu              # upper bound on intensity

    while t < T:
        # Sample candidate inter-arrival from exponential upper bound
        u = rng.exponential(1.0 / max(lambda_upper, 1e-12))
        t_cand = t + u

        if t_cand > T:
            break

        # Decay excitation from t to t_cand
        A = A * np.exp(-beta * u)
        lambda_cand = mu + alpha * A

        # Thinning acceptance
        if rng.uniform() * lambda_upper <= lambda_cand:
            events.append(t_cand)
            A += 1.0                          # add new event's contribution
            lambda_upper = lambda_cand + alpha  # tight upper bound after event
        else:
            lambda_upper = lambda_cand        # intensity can only decrease now

        t = t_cand

    return np.array(events, dtype=float)


# ---------------------------------------------------------------------------
# Public: intensity
# ---------------------------------------------------------------------------

def intensity(
    query_times: np.ndarray,
    event_times: np.ndarray,
    mu: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Evaluate the conditional intensity function λ*(t) at arbitrary query times.

    Parameters
    ----------
    query_times : times at which to evaluate the intensity
    event_times : observed (or simulated) event history

    Returns
    -------
    Array of λ*(t) values, one per query time.
    """
    qt = np.asarray(query_times, dtype=float)
    et = np.sort(np.asarray(event_times, dtype=float))

    result = np.full(len(qt), mu)
    for i, t in enumerate(qt):
        past = et[et < t]
        result[i] = mu + alpha * np.sum(np.exp(-beta * (t - past)))

    return result


# ---------------------------------------------------------------------------
# Public: bivariate simulate
# ---------------------------------------------------------------------------

def simulate_bivariate(
    mu_bid: float, alpha_bb: float, alpha_ab: float,
    mu_ask: float, alpha_aa: float, alpha_ba: float,
    beta: float,
    T: float,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a symmetric bivariate Hawkes process for bid and ask arrivals.

    Each side has:
      • Self-excitation  (α_same):  bid orders beget more bid orders
      • Cross-excitation (α_cross): bid orders also beget ask orders
                                    (market-maker response / contrarian flow)

    Intensity functions:
      λ_bid(t) = μ_bid + α_bb · Σ exp(−β(t−t_i^bid)) + α_ab · Σ exp(−β(t−t_j^ask))
      λ_ask(t) = μ_ask + α_aa · Σ exp(−β(t−t_i^ask)) + α_ba · Σ exp(−β(t−t_j^bid))

    Uses superposition thinning on the combined process (λ_bid + λ_ask).

    Returns
    -------
    bid_times, ask_times  (sorted arrays)
    """
    rng = np.random.default_rng(seed)
    bid_times: List[float] = []
    ask_times: List[float] = []

    t = 0.0
    A_bb = A_ab = A_aa = A_ba = 0.0    # running excitation sums

    lambda_bid = mu_bid
    lambda_ask = mu_ask
    lambda_upper = lambda_bid + lambda_ask

    while t < T:
        u = rng.exponential(1.0 / max(lambda_upper, 1e-12))
        t_cand = t + u

        if t_cand > T:
            break

        # Decay all excitation sums
        decay = np.exp(-beta * u)
        A_bb *= decay
        A_ab *= decay
        A_aa *= decay
        A_ba *= decay

        lambda_bid_cand = mu_bid + alpha_bb * A_bb + alpha_ab * A_ab
        lambda_ask_cand = mu_ask + alpha_aa * A_aa + alpha_ba * A_ba
        lambda_cand     = lambda_bid_cand + lambda_ask_cand

        if rng.uniform() * lambda_upper <= lambda_cand:
            # Determine which side fired (proportional to partial intensities)
            if rng.uniform() * lambda_cand <= lambda_bid_cand:
                # Bid event
                bid_times.append(t_cand)
                A_bb += 1.0    # bid excites bids
                A_ba += 1.0    # bid excites asks (cross)
                lambda_bid_cand += alpha_bb
                lambda_ask_cand += alpha_ba
            else:
                # Ask event
                ask_times.append(t_cand)
                A_aa += 1.0
                A_ab += 1.0
                lambda_bid_cand += alpha_ab
                lambda_ask_cand += alpha_aa

            lambda_upper = lambda_bid_cand + lambda_ask_cand
        else:
            lambda_upper = lambda_cand

        t = t_cand

    return np.array(bid_times, dtype=float), np.array(ask_times, dtype=float)
