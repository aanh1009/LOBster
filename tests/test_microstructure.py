import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from microstructure.metrics import kyle_lambda, vpin


class TestKyleLambda:
    def test_positive_lambda_on_positive_impact(self):
        rng = np.random.default_rng(0)
        n = 200
        signed_vol = rng.normal(0, 1000, n)
        mid_prices = np.cumsum(
            np.concatenate([[100.0], 0.005 * signed_vol + rng.normal(0, 0.01, n)])
        )
        result = kyle_lambda(mid_prices, signed_vol)
        assert result["lambda"] > 0

    def test_r_squared_bounded(self):
        rng = np.random.default_rng(1)
        prices = np.cumsum(rng.normal(0, 0.01, 101)) + 100
        vols   = rng.normal(0, 100, 100)
        result = kyle_lambda(prices, vols)
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_raises_on_too_few_obs(self):
        with pytest.raises(ValueError):
            kyle_lambda(np.array([100.0, 100.1]), np.array([50.0]))


class TestVPIN:
    def test_balanced_flow_low_vpin(self):
        n = 1000
        volumes = np.ones(n) * 10.0
        sides   = np.tile([1.0, -1.0], n // 2)
        result  = vpin(np.ones(n), volumes, sides, bucket_size=100.0)
        assert result["vpin"] < 0.2

    def test_one_sided_flow_high_vpin(self):
        n = 1000
        result = vpin(np.ones(n), np.ones(n) * 10.0, np.ones(n), bucket_size=100.0)
        assert result["vpin"] == pytest.approx(1.0, abs=0.01)

    def test_vpin_bounded(self):
        rng = np.random.default_rng(42)
        n = 500
        result = vpin(np.ones(n), rng.lognormal(0, 1, n),
                      rng.choice([-1.0, 1.0], n))
        assert 0.0 <= result["vpin"] <= 1.0
