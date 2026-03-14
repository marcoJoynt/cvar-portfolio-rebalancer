# rebalancer/risk.py
"""
CVaR formulation via Rockafellar-Uryasev (2000).

Key result: CVaR at confidence level beta is equivalent to:

    CVaR_beta(w) = min_z { z + 1/(q*(1-beta)) * sum(max(-r_k @ w - z, 0)) }

where:
    z     = VaR threshold (auxiliary scalar, optimised over)
    q     = number of scenarios
    r_k   = return vector for scenario k  (loss = -r_k @ w)
    beta  = confidence level (e.g. 0.95)

The max(·, 0) term becomes cp.pos() in cvxpy — which is convex,
so the whole expression fits inside a disciplined convex program.

This means CVaR can be a *constraint* inside the optimiser, not just
a metric computed after the fact. That's the core insight.
"""

import cvxpy as cp
import numpy as np

from .constants import DEFAULT_CVAR_BETA


def cvar_expression(
    w: cp.Variable,
    scenarios: np.ndarray,
    beta: float = DEFAULT_CVAR_BETA,
) -> tuple[cp.Expression, cp.Variable]:
    """
    Build the CVaR expression and its auxiliary VaR variable.

    Parameters
    ----------
    w : cp.Variable
        Portfolio weight vector, shape (n_assets,)
    scenarios : np.ndarray
        Return scenarios, shape (q, n_assets).
        Rows are scenarios, columns are assets.
        Simple returns — losses are defined as -scenarios @ w.
    beta : float
        Confidence level. 0.95 = worst 5% of scenarios.

    Returns
    -------
    cvar_expr : cp.Expression
        Scalar CVaR expression — can be used as objective or constraint.
    z : cp.Variable
        The auxiliary VaR variable. Expose so caller can inspect VaR value
        after solving.

    Notes
    -----
    We do NOT introduce the slack variable u here — instead we use cp.pos()
    which lets cvxpy handle the epigraph reformulation internally.
    This keeps the interface clean: one expression, one auxiliary variable.
    """
    q, n = scenarios.shape
    if q == 0 or (1 - beta) <= 0:
        raise ValueError("scenarios must be non-empty and beta must be in (0, 1)")
    z = cp.Variable()  # VaR threshold

    # Scenario losses: shape (q,)
    # loss_k = -r_k @ w  (positive when portfolio loses money)
    losses = -scenarios @ w  # cp.Expression, shape (q,)

    # Excess losses above VaR threshold, floored at 0
    # cp.pos(x) = max(x, 0) — convex, so DCP-compliant
    excess = cp.pos(losses - z)  # shape (q,)

    # Rockafellar-Uryasev CVaR formula
    cvar_expr = z + (1.0 / (q * (1 - beta))) * cp.sum(excess)

    return cvar_expr, z


def compute_cvar(
    weights: np.ndarray,
    scenarios: np.ndarray,
    beta: float = DEFAULT_CVAR_BETA,
) -> dict:
    """
    Compute CVaR and VaR for a *fixed* weight vector.
    Use this for sanity checks and pre/post rebalance reporting —
    not for optimisation (use cvar_expression() for that).

    Parameters
    ----------
    weights : np.ndarray
        Fixed portfolio weights, shape (n_assets,).
    scenarios : np.ndarray
        Return scenarios, shape (q, n_assets).
    beta : float
        Confidence level.

    Returns
    -------
    dict with keys: cvar, var, mean_loss, worst_loss
    """
    if scenarios.size == 0 or not (0 < beta < 1):
        raise ValueError("scenarios must be non-empty and beta must be in (0, 1)")
    weights = np.asarray(weights).ravel()
    losses = -scenarios @ weights  # shape (q,)
    var = np.percentile(losses, beta * 100, method="lower")
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if len(tail) > 0 else float(var)

    return {
        "cvar": cvar,
        "var": var,
        "mean_loss": float(losses.mean()),
        "worst_loss": float(losses.max()),
    }