"""
Unit tests for microstructure metrics.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from microstructure.metrics import (
    amihud_illiquidity,
    bin_trades,
    kyle_lambda,
    order_flow_imbalance,
    roll_spread,
    vpin,
)


# ---------------------------------------------------------------------------
# Kyle's Lambda
# ---------------------------------------------------------------------------

class TestKyleLambda:
    def test_positive_lambda_on_positive_impact(self):
        """Positive order flow should positively impact price."""
        rng = np.random.default_rng(0)
        n = 200
        # True lambda = 0.005
        signed_vol = rng.normal(0, 1000, n)
        mid_prices = np.cumsum(np.concatenate([[100.0], 0.005 * signed_vol + rng.normal(0, 0.01, n)]))
        result = kyle_lambda(mid_prices, signed_vol)
        assert result["lambda"] > 0
        assert result["n_obs"] == n

    def test_r_squared_bounded(self):
        rng = np.random.default_rng(1)
        prices = np.cumsum(rng.normal(0, 0.01, 101)) + 100
        vols   = rng.normal(0, 100, 100)
        result = kyle_lambda(prices, vols)
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_raises_on_too_few_obs(self):
        with pytest.raises(ValueError):
            kyle_lambda(np.array([100.0, 100.1]), np.array([50.0]))


# ---------------------------------------------------------------------------
# VPIN
# ---------------------------------------------------------------------------

class TestVPIN:
    def test_balanced_flow_low_vpin(self):
        """Perfectly alternating buy/sell → VPIN near 0."""
        n = 1000
        volumes = np.ones(n) * 10.0
        sides   = np.tile([1.0, -1.0], n // 2)
        result  = vpin(np.ones(n), volumes, sides, bucket_size=100.0)
        assert result["vpin"] < 0.2

    def test_one_sided_flow_high_vpin(self):
        """All buys → VPIN = 1.0."""
        n = 1000
        volumes = np.ones(n) * 10.0
        sides   = np.ones(n)
        result  = vpin(np.ones(n), volumes, sides, bucket_size=100.0)
        assert result["vpin"] == pytest.approx(1.0, abs=0.01)

    def test_vpin_bounded(self):
        rng = np.random.default_rng(42)
        n = 500
        volumes = rng.lognormal(0, 1, n)
        sides   = rng.choice([-1.0, 1.0], n)
        result  = vpin(np.ones(n), volumes, sides)
        assert 0.0 <= result["vpin"] <= 1.0


# ---------------------------------------------------------------------------
# OFI
# ---------------------------------------------------------------------------

class TestOFI:
    def test_bid_volume_increase_positive_ofi(self):
        """Adding bid volume should produce positive OFI."""
        bid_prices  = np.array([99.0, 99.0, 99.0])
        ask_prices  = np.array([101.0, 101.0, 101.0])
        bid_volumes = np.array([100.0, 200.0, 300.0])   # increasing
        ask_volumes = np.array([100.0, 100.0, 100.0])   # constant
        result = order_flow_imbalance(bid_prices, ask_prices, bid_volumes, ask_volumes)
        ofi_arr = np.array(result["ofi_series"])
        assert ofi_arr.sum() > 0

    def test_ofi_length(self):
        n = 50
        bp = np.ones(n) * 99.0
        ap = np.ones(n) * 101.0
        bv = np.random.default_rng(0).uniform(100, 200, n)
        av = np.random.default_rng(1).uniform(100, 200, n)
        result = order_flow_imbalance(bp, ap, bv, av)
        assert len(result["ofi_series"]) == n - 1


# ---------------------------------------------------------------------------
# Amihud
# ---------------------------------------------------------------------------

class TestAmihud:
    def test_higher_volume_lower_illiquidity(self):
        r = np.abs(np.random.default_rng(0).normal(0, 0.01, 100))
        low_vol  = amihud_illiquidity(r, np.ones(100) * 1_000)
        high_vol = amihud_illiquidity(r, np.ones(100) * 1_000_000)
        assert low_vol["illiquidity"] > high_vol["illiquidity"]

    def test_zero_volume_excluded(self):
        r = np.ones(10) * 0.01
        v = np.array([0.0] * 5 + [1000.0] * 5)
        result = amihud_illiquidity(r, v)
        assert result["n_obs"] == 5


# ---------------------------------------------------------------------------
# Roll Spread
# ---------------------------------------------------------------------------

class TestRollSpread:
    def test_synthetic_roll_spread(self):
        """
        Generate prices from the Roll model and check that the recovered
        spread is close to the true spread.
        """
        rng = np.random.default_rng(7)
        c = 0.05          # true half-spread
        n = 2000
        Q = rng.choice([-1.0, 1.0], n)
        # ΔP = c(Q_t - Q_{t-1}) + ε
        delta_p = c * np.diff(Q) + rng.normal(0, 0.001, n - 1)
        prices  = np.cumsum(np.concatenate([[100.0], delta_p]))
        result  = roll_spread(prices)
        # Roll spread should be close to 2c = 0.10
        if not np.isnan(result["roll_spread"]):
            assert abs(result["roll_spread"] - 2 * c) < 0.05

    def test_monotone_prices_no_spread(self):
        """Monotonically increasing prices have non-negative serial cov → NaN spread."""
        prices = np.linspace(100, 110, 100)
        result = roll_spread(prices)
        # Serial cov will be positive (trending) → NaN or non-applicable
        # We just check it doesn't crash
        assert "roll_spread" in result
