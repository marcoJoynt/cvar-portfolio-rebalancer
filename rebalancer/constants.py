"""
Named constants used across the rebalancer.
"""

# Trading days per year (e.g. for scenario bootstrap window)
TRADING_DAYS_PER_YEAR = 252

# CVaR confidence level: 0.95 = worst 5% of scenarios
DEFAULT_CVAR_BETA = 0.95

# Default weight bounds (long-only)
DEFAULT_MIN_WEIGHT = 0.0
DEFAULT_MAX_WEIGHT = 1.0

# Default transaction cost tuning
DEFAULT_COST_PER_UNIT = 0.001
DEFAULT_LAMBDA_COST = 0.0
