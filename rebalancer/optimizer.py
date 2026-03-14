# rebalancer/optimizer.py
"""
optimizer.py
------------
CVaR-constrained portfolio optimiser.

Objective:  minimise tracking error vs target weights
            + transaction cost penalty

Constraint: CVaR(w) <= cvar_limit  (Rockafellar-Uryasev)
            + budget, box, turnover  (see constraints.py)

Built in three layers — each is a valid standalone problem:
    1. Tracking error only          (no risk, no cost)
    2. + CVaR hard constraint       (risk-aware)
    3. + transaction cost penalty   (execution-aware)
"""

import cvxpy as cp
import numpy as np

from .constraints import build_constraints
from ._logging import get_logger
from .risk import cvar_expression, compute_cvar

logger = get_logger(__name__)


def optimise(
    w_prev: np.ndarray,
    w_target: np.ndarray,
    scenarios: np.ndarray,
    config: dict,
) -> dict:
    """
    Parameters
    ----------
    w_prev : np.ndarray
        Current portfolio weights, shape (n_assets,).
    w_target : np.ndarray
        Target (benchmark) weights, shape (n_assets,).
    scenarios : np.ndarray
        Return scenarios, shape (n_scenarios, n_assets).
    config : dict
        Supported keys (in addition to constraints.py keys):
            cvar_limit      : float, optional  — CVaR upper bound
            cvar_beta       : float, default 0.95
            lambda_cost     : float, default 0.0  — transaction cost penalty
            cost_per_unit   : float, default 0.001 — cost per unit turnover

    Returns
    -------
    dict with keys:
        weights         : np.ndarray  — optimal weights
        tracking_error  : float
        cvar            : float       — ex-ante CVaR at solution
        var             : float       — ex-ante VaR at solution
        turnover        : float       — L1 turnover vs w_prev
        status          : str         — solver status
    """
    w_prev   = np.asarray(w_prev).ravel()
    w_target = np.asarray(w_target).ravel()
    n = len(w_target)

    w = cp.Variable(n)

    # ------------------------------------------------------------------
    # Layer 1: tracking error
    # sum of squared deviations from target — convex quadratic
    # ------------------------------------------------------------------
    tracking_error = cp.sum_squares(w - w_target)
    objective_terms = [tracking_error]

    # ------------------------------------------------------------------
    # Layer 2: transaction cost penalty
    # lambda_cost * cost_per_unit * L1_turnover
    # ------------------------------------------------------------------
    lambda_cost   = config.get("lambda_cost", 0.0)
    cost_per_unit = config.get("cost_per_unit", 0.001)

    if lambda_cost > 0:
        turnover_expr = cp.norm1(w - w_prev)
        objective_terms.append(lambda_cost * cost_per_unit * turnover_expr)

    objective = cp.Minimize(cp.sum(objective_terms))

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    constraints = build_constraints(w, w_prev, config)

    # Layer 3: CVaR hard constraint
    if "cvar_limit" in config:
        beta = config.get("cvar_beta", 0.95)
        cvar_expr, z_var = cvar_expression(w, scenarios, beta=beta)
        constraints.append(cvar_expr <= config["cvar_limit"])

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, warm_start=True)

    logger.debug("Solver finished: status=%s", prob.status)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Optimiser failed: {prob.status}")

    w_opt = w.value

    # Ex-ante risk at solution
    beta = config.get("cvar_beta", 0.95)
    risk = compute_cvar(w_opt, scenarios, beta=beta)

    return {
        "weights":        w_opt,
        "tracking_error": float(tracking_error.value),
        "cvar":           risk["cvar"],
        "var":            risk["var"],
        "turnover":       float(np.sum(np.abs(w_opt - w_prev))),
        "status":         prob.status,
    }