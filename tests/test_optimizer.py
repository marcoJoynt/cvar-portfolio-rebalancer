# tests/test_optimizer.py
"""Optimizer tests using synthetic data so they run without network."""
import numpy as np
import pandas as pd
import pytest
from rebalancer.config import OptimizerConfig
from rebalancer.exceptions import InfeasibleConfigError
from rebalancer.scenarios import historical_bootstrap
from rebalancer.optimizer import optimise


def _synthetic_scenarios(n_assets=5, n_scenarios=5_000, seed=42):
    """Build scenarios from random returns (no yfinance)."""
    rng = np.random.default_rng(seed)
    n_days = 504
    returns = pd.DataFrame(
        rng.standard_normal((n_days, n_assets)) * 0.01,
        columns=["VTI", "VXUS", "BND", "GLD", "VNQ"][:n_assets],
    )
    return historical_bootstrap(returns, n_scenarios=n_scenarios)


def test_layer1_tracking_error_only():
    """Layer 1: no CVaR or cost — solution should be close to target."""
    scenarios = _synthetic_scenarios()
    n = 5
    w_target = np.array([0.40, 0.20, 0.25, 0.10, 0.05])
    w_prev = np.ones(n) / n
    config = {"min_weight": 0.05, "max_weight": 0.60}

    result = optimise(w_prev, w_target, scenarios, config)

    assert result["status"] in ("optimal", "optimal_inaccurate")
    np.testing.assert_allclose(result["weights"], w_target, atol=0.01)
    assert result["tracking_error"] >= 0
    assert result["turnover"] >= 0


def test_layer2_cvar_constraint():
    """Layer 2: CVaR constraint must be respected."""
    scenarios = _synthetic_scenarios()
    n = 5
    w_target = np.array([0.40, 0.20, 0.25, 0.10, 0.05])
    w_prev = np.ones(n) / n
    config = {
        "min_weight": 0.05,
        "max_weight": 0.60,
        "cvar_limit": 0.05,
        "cvar_beta": 0.95,
    }

    result = optimise(w_prev, w_target, scenarios, config)

    assert result["status"] in ("optimal", "optimal_inaccurate")
    assert result["cvar"] <= config["cvar_limit"] + 1e-4


def test_layer3_cost_penalty_reduces_turnover():
    """Layer 3: cost penalty should reduce turnover vs layer 2."""
    scenarios = _synthetic_scenarios()
    n = 5
    w_target = np.array([0.40, 0.20, 0.25, 0.10, 0.05])
    w_prev = np.ones(n) / n
    config_no_cost = {
        "min_weight": 0.05,
        "max_weight": 0.60,
        "cvar_limit": 0.05,
        "cvar_beta": 0.95,
    }
    config_with_cost = {**config_no_cost, "lambda_cost": 1.0, "cost_per_unit": 0.001}

    result2 = optimise(w_prev, w_target, scenarios, config_no_cost)
    result3 = optimise(w_prev, w_target, scenarios, config_with_cost)

    assert result3["cvar"] <= config_with_cost["cvar_limit"] + 1e-4
    assert result3["turnover"] < result2["turnover"]


def test_optimise_accepts_OptimizerConfig():
    """optimise() accepts OptimizerConfig and uses it like a dict."""
    scenarios = _synthetic_scenarios()
    w_target = np.array([0.40, 0.20, 0.25, 0.10, 0.05])
    w_prev = np.ones(5) / 5
    config = OptimizerConfig(
        min_weight=0.05,
        max_weight=0.60,
        cvar_limit=0.05,
        cvar_beta=0.95,
    )
    result = optimise(w_prev, w_target, scenarios, config)
    assert result["status"] in ("optimal", "optimal_inaccurate")
    assert result["cvar"] <= 0.05 + 1e-4


def test_invalid_cvar_limit_raises():
    """Passing cvar_limit <= 0 raises InfeasibleConfigError."""
    scenarios = _synthetic_scenarios()
    w_prev = np.ones(5) / 5
    w_target = np.array([0.40, 0.20, 0.25, 0.10, 0.05])
    with pytest.raises(InfeasibleConfigError, match="cvar_limit"):
        optimise(w_prev, w_target, scenarios, {"cvar_limit": 0.0, "min_weight": 0.05, "max_weight": 0.6})
