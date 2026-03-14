# main.py
"""
Demo: rebalance a drifted 5-ETF portfolio toward target allocation.
Run with: python main.py
"""

import numpy as np
from rebalancer.rebalancer import Portfolio, rebalance

# ---------------------------------------------------------------------------
# Client portfolio — drifted after a equity rally
# ---------------------------------------------------------------------------
portfolio = Portfolio(
    tickers     = ["VTI", "VXUS", "BND", "GLD", "VNQ"],
    weights     = np.array([0.52, 0.18, 0.17, 0.08, 0.05]),  # drifted
    value       = 100_000.0,                                   # €100k
    cost_bases  = np.array([180.0, 52.0, 72.0, 165.0, 78.0]), # purchase prices
    prices      = np.array([240.0, 58.0, 74.0, 185.0, 82.0]), # current prices
)

# ---------------------------------------------------------------------------
# Target allocation
# ---------------------------------------------------------------------------
w_target = np.array([0.40, 0.20, 0.25, 0.10, 0.05])

# ---------------------------------------------------------------------------
# Optimizer config
# ---------------------------------------------------------------------------
config = {
    "min_weight":   0.05,
    "max_weight":   0.60,
    "cvar_limit":   0.015,
    "cvar_beta":    0.95,
    "max_turnover": 0.40,
    "lambda_cost":  1.0,
    "cost_per_unit": 0.001,
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
report = rebalance(portfolio, w_target, config, n_scenarios=10_000)
print(report.summary())