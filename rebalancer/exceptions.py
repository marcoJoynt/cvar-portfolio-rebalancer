"""
Custom exceptions for the rebalancer.
"""


class RebalancerError(Exception):
    """Base for rebalancer-specific errors."""

    pass


class InfeasibleConfigError(RebalancerError):
    """Raised when config validation fails (e.g. min_weight > max_weight, cvar_limit <= 0)."""

    pass


class OptimizationError(RebalancerError):
    """Raised when the convex solver fails."""

    pass
