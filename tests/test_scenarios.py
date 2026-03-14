import numpy as np
from rebalancer.data import get_prices, get_returns
from rebalancer.scenarios import historical_bootstrap


def test_bootstrap_shape_and_no_nans():
    """Full pipeline: prices → returns → scenarios; check shape and no NaNs."""
    prices = get_prices()
    returns = get_returns(prices)
    scenarios = historical_bootstrap(returns, n_scenarios=5000)
    assert scenarios.shape == (5000, len(prices.columns))
    assert not np.isnan(scenarios).any()
