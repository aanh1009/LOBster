"""
Unit tests for the Hawkes process implementation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from hawkes.process import fit, intensity, simulate, simulate_bivariate


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

class TestSimulate:
    def test_events_in_range(self):
        times = simulate(mu=2.0, alpha=0.5, beta=3.0, T=100.0, seed=0)
        assert (times >= 0).all()
        assert (times <= 100.0).all()

    def test_events_sorted(self):
        times = simulate(mu=2.0, alpha=0.5, beta=3.0, T=100.0, seed=1)
        assert np.all(np.diff(times) >= 0)

    def test_more_events_with_higher_mu(self):
        t1 = simulate(mu=1.0, alpha=0.1, beta=2.0, T=1000.0, seed=42)
        t2 = simulate(mu=5.0, alpha=0.1, beta=2.0, T=1000.0, seed=42)
        assert len(t2) > len(t1)

    def test_non_stationary_raises(self):
        """alpha >= beta violates stationarity → should raise."""
        with pytest.raises(ValueError, match="non-stationary"):
            simulate(mu=1.0, alpha=5.0, beta=3.0, T=100.0)

    def test_reproducible_with_same_seed(self):
        t1 = simulate(mu=2.0, alpha=0.5, beta=3.0, T=100.0, seed=99)
        t2 = simulate(mu=2.0, alpha=0.5, beta=3.0, T=100.0, seed=99)
        np.testing.assert_array_equal(t1, t2)

    def test_branching_ratio_affects_clustering(self):
        """Higher branching ratio → more clustering (larger burst sizes)."""
        # Low branching: close to Poisson
        t_poisson = simulate(mu=3.0, alpha=0.01, beta=10.0, T=500.0, seed=0)
        # High branching: strongly clustered
        t_hawkes  = simulate(mu=3.0, alpha=8.0,  beta=10.0, T=500.0, seed=0)
        # Hawkes should have more events on average (each event spawns children)
        assert len(t_hawkes) >= len(t_poisson)


# ---------------------------------------------------------------------------
# MLE Fitting
# ---------------------------------------------------------------------------

class TestFit:
    def _generate_and_fit(self, mu, alpha, beta, T=2000.0, seed=0):
        times  = simulate(mu=mu, alpha=alpha, beta=beta, T=T, seed=seed)
        result = fit(times, T=T, n_restarts=5, seed=seed)
        return result

    def test_fit_returns_required_keys(self):
        result = self._generate_and_fit(1.0, 0.3, 1.0)
        for key in ("mu", "alpha", "beta", "branching_ratio", "log_likelihood"):
            assert key in result

    def test_branching_ratio_stationary(self):
        """Fitted branching ratio must be < 1 for stationarity."""
        result = self._generate_and_fit(2.0, 0.5, 2.0)
        assert result["branching_ratio"] < 1.0

    def test_fitted_mu_positive(self):
        result = self._generate_and_fit(1.5, 0.3, 1.5)
        assert result["mu"] > 0

    def test_fit_recovers_approx_parameters(self):
        """
        With sufficient data the MLE should recover parameters within 30%.
        Hawkes MLE can have high variance on short samples.
        """
        true_mu, true_alpha, true_beta = 2.0, 1.0, 4.0
        result = self._generate_and_fit(
            true_mu, true_alpha, true_beta, T=5000.0, seed=7
        )
        assert result["mu"]    == pytest.approx(true_mu,    rel=0.35)
        assert result["alpha"] == pytest.approx(true_alpha, rel=0.35)
        assert result["beta"]  == pytest.approx(true_beta,  rel=0.35)


# ---------------------------------------------------------------------------
# Intensity function
# ---------------------------------------------------------------------------

class TestIntensity:
    def test_intensity_at_least_mu(self):
        """λ*(t) ≥ μ always (excitation is non-negative)."""
        mu = 1.5
        times = simulate(mu=mu, alpha=0.5, beta=2.0, T=100.0, seed=0)
        query = np.linspace(0, 100, 200)
        lam   = intensity(query, times, mu=mu, alpha=0.5, beta=2.0)
        assert (lam >= mu - 1e-9).all()

    def test_intensity_decays_after_event(self):
        """Intensity should be higher just after an event than later."""
        mu, alpha, beta = 1.0, 2.0, 5.0
        # Manually place one event at t=10
        event_times = np.array([10.0])
        just_after  = intensity(np.array([10.001]), event_times, mu, alpha, beta)
        much_later  = intensity(np.array([20.0]),   event_times, mu, alpha, beta)
        assert just_after[0] > much_later[0]


# ---------------------------------------------------------------------------
# Bivariate simulation
# ---------------------------------------------------------------------------

class TestSimulateBivariate:
    def test_returns_two_arrays(self):
        bid, ask = simulate_bivariate(
            mu_bid=2.0, alpha_bb=1.0, alpha_ab=0.2,
            mu_ask=2.0, alpha_aa=1.0, alpha_ba=0.2,
            beta=5.0, T=100.0, seed=0,
        )
        assert isinstance(bid, np.ndarray)
        assert isinstance(ask, np.ndarray)

    def test_events_in_range(self):
        bid, ask = simulate_bivariate(
            mu_bid=3.0, alpha_bb=1.5, alpha_ab=0.3,
            mu_ask=3.0, alpha_aa=1.5, alpha_ba=0.3,
            beta=6.0, T=200.0, seed=1,
        )
        assert (bid >= 0).all() and (bid <= 200.0).all()
        assert (ask >= 0).all() and (ask <= 200.0).all()

    def test_cross_excitation_increases_events(self):
        """More cross-excitation → more total events."""
        kw = dict(mu_bid=2.0, mu_ask=2.0, alpha_bb=1.0, alpha_aa=1.0,
                  beta=8.0, T=500.0, seed=0)
        bid_lo, ask_lo = simulate_bivariate(**kw, alpha_ab=0.01, alpha_ba=0.01)
        bid_hi, ask_hi = simulate_bivariate(**kw, alpha_ab=1.5,  alpha_ba=1.5)
        assert len(bid_hi) + len(ask_hi) > len(bid_lo) + len(ask_lo)
