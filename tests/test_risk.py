"""
Tests for rebalancer.risk: CVaR/VaR computation and sanity checks.
Uses real data pipeline (prices → returns → scenarios) then compute_cvar().
"""
import numpy as np
import pytest
from rebalancer.data import get_prices, get_returns
from rebalancer.risk import compute_cvar
from rebalancer.scenarios import historical_bootstrap


@pytest.fixture(scope="module")
def scenarios_and_weights():
    """Build scenarios and two weight vectors once for all tests in this module."""
    prices = get_prices()
    returns = get_returns(prices)
    scenarios = historical_bootstrap(returns, n_scenarios=10_000)
    n = len(prices.columns)
    w_equal = np.ones(n) / n
    w_concentrated = np.array([0.9, 0.025, 0.025, 0.025, 0.025])
    return scenarios, w_equal, w_concentrated


def test_compute_cvar_cvar_geq_var(scenarios_and_weights):
    """CVaR must be >= VaR by definition for any portfolio."""
    scenarios, w_equal, w_concentrated = scenarios_and_weights
    eq = compute_cvar(w_equal, scenarios, beta=0.95)
    co = compute_cvar(w_concentrated, scenarios, beta=0.95)
    assert eq["cvar"] >= eq["var"], "CVaR must be >= VaR"
    assert co["cvar"] >= co["var"], "CVaR must be >= VaR"


def test_concentrated_portfolio_higher_cvar_than_equal_weight(scenarios_and_weights):
    """Concentrated portfolio should have higher tail risk than equal weight."""
    scenarios, w_equal, w_concentrated = scenarios_and_weights
    eq = compute_cvar(w_equal, scenarios, beta=0.95)
    co = compute_cvar(w_concentrated, scenarios, beta=0.95)
    assert co["cvar"] > eq["cvar"], "Concentration should increase tail risk"
