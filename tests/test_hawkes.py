import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from hawkes.process import simulate, simulate_bivariate


class TestSimulate:
    def test_events_in_range(self):
        times = simulate(mu=2.0, alpha=0.5, beta=3.0, T=100.0, seed=0)
        assert (times >= 0).all() and (times <= 100.0).all()

    def test_events_sorted(self):
        times = simulate(mu=2.0, alpha=0.5, beta=3.0, T=100.0, seed=1)
        assert np.all(np.diff(times) >= 0)

    def test_more_events_with_higher_mu(self):
        t1 = simulate(mu=1.0, alpha=0.1, beta=2.0, T=1000.0, seed=42)
        t2 = simulate(mu=5.0, alpha=0.1, beta=2.0, T=1000.0, seed=42)
        assert len(t2) > len(t1)

    def test_non_stationary_raises(self):
        with pytest.raises(ValueError, match="non-stationary"):
            simulate(mu=1.0, alpha=5.0, beta=3.0, T=100.0)

    def test_reproducible_with_same_seed(self):
        t1 = simulate(mu=2.0, alpha=0.5, beta=3.0, T=100.0, seed=99)
        t2 = simulate(mu=2.0, alpha=0.5, beta=3.0, T=100.0, seed=99)
        np.testing.assert_array_equal(t1, t2)

    def test_branching_ratio_affects_clustering(self):
        t_low  = simulate(mu=3.0, alpha=0.01, beta=10.0, T=500.0, seed=0)
        t_high = simulate(mu=3.0, alpha=8.0,  beta=10.0, T=500.0, seed=0)
        assert len(t_high) >= len(t_low)
