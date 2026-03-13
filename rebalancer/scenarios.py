# rebalancer/scenarios.py
"""
Build many possible "future" return scenarios by resampling recent history.
Used as input to the CVaR optimizer (no distributional assumption needed).
"""
import numpy as np
import pandas as pd


def historical_bootstrap(
    returns: pd.DataFrame,
    n_scenarios: int = 10_000,
    window: int = 252,
) -> np.ndarray:
    """
    Sample rows from the last `window` trading days of returns, with replacement.
    Returns shape (n_scenarios, n_assets). Each row is one scenario (one possible
    joint return across assets).

    Bootstrap (vs a parametric model) keeps fat tails, skew, and cross-asset
    dependence from real data; the CVaR LP works with any scenario set.
    """
    # Use only the most recent window (default 252 ≈ 1 year of trading days).
    recent = returns.iloc[-window:].values
    # Draw n_scenarios rows at random, with replacement (same day can appear many times).
    idx = np.random.choice(len(recent), size=n_scenarios, replace=True)
    return recent[idx]