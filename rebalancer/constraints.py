# rebalancer/constraints.py
"""
Builds cvxpy constraint lists for the portfolio optimiser.

All constraints are affine in w — this is required for the overall
optimisation problem to remain convex (DCP-compliant).
"""

import cvxpy as cp
import numpy as np


def build_constraints(
    w: cp.Variable,
    w_prev: np.ndarray | None,
    config: dict,
) -> list[cp.Constraint]:
    """
    Parameters
    ----------
    w : cp.Variable
        Portfolio weight vector, shape (n_assets,).
    w_prev : np.ndarray or None
        Current (pre-rebalance) weights. Required for turnover constraint.
        Pass None to skip turnover constraint.
    config : dict
        Supported keys:
            min_weight    : float, default 0.0   — lower bound per asset
            max_weight    : float, default 1.0   — upper bound per asset
            max_turnover  : float, optional      — L1 turnover cap
            min_return    : float, optional      — expected return floor
            mu            : np.ndarray, optional — expected returns vector
                            (required if min_return is set)

    Returns
    -------
    list of cp.Constraint
    """
    constraints = []

    # 1. Fully invested
    constraints.append(cp.sum(w) == 1)

    # 2. Weight bounds (long-only by default)
    lb = config.get("min_weight", 0.0)
    ub = config.get("max_weight", 1.0)
    if lb > ub:
        raise ValueError("min_weight must be <= max_weight")
    constraints.append(w >= lb)
    constraints.append(w <= ub)

    # 3. Turnover cap: L1 norm of weight change
    # L1 = sum of absolute trades = one-way turnover
    if w_prev is not None and "max_turnover" in config:
        mt = config["max_turnover"]
        if mt < 0:
            raise ValueError("max_turnover must be >= 0")
        w_prev_flat = np.asarray(w_prev).ravel()
        constraints.append(cp.norm1(w - w_prev_flat) <= mt)

    # 4. Minimum expected return floor
    # Requires mu (expected returns) to be passed in config
    if "min_return" in config:
        mu = config.get("mu")
        if mu is None:
            raise ValueError("min_return set in config but mu not provided")
        mu_flat = np.asarray(mu).ravel()
        constraints.append(mu_flat @ w >= config["min_return"])

    return constraints
