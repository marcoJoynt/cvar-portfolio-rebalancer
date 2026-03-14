"""
Configuration for the portfolio optimiser. OptimizerConfig can be built from a dict (e.g. YAML/JSON).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .constants import (
    DEFAULT_COST_PER_UNIT,
    DEFAULT_CVAR_BETA,
    DEFAULT_LAMBDA_COST,
    DEFAULT_MAX_WEIGHT,
    DEFAULT_MIN_WEIGHT,
)
from .exceptions import InfeasibleConfigError


@dataclass
class OptimizerConfig:
    """
    Typed config for the CVaR optimiser and constraints. All fields have defaults.
    Use from_dict() to build from a plain dict.
    """

    min_weight: float = DEFAULT_MIN_WEIGHT
    max_weight: float = DEFAULT_MAX_WEIGHT
    max_turnover: Optional[float] = None
    min_return: Optional[float] = None
    mu: Optional[np.ndarray] = None

    cvar_limit: Optional[float] = None
    cvar_beta: float = DEFAULT_CVAR_BETA

    lambda_cost: float = DEFAULT_LAMBDA_COST
    cost_per_unit: float = DEFAULT_COST_PER_UNIT

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise InfeasibleConfigError if config is invalid."""
        if self.min_weight > self.max_weight:
            raise InfeasibleConfigError("min_weight must be <= max_weight")
        if self.max_turnover is not None and self.max_turnover < 0:
            raise InfeasibleConfigError("max_turnover must be >= 0")
        if self.cvar_limit is not None and self.cvar_limit <= 0:
            raise InfeasibleConfigError("cvar_limit must be > 0")
        if not (0 < self.cvar_beta < 1):
            raise InfeasibleConfigError("cvar_beta must be in (0, 1)")
        if self.min_return is not None and self.mu is None:
            raise InfeasibleConfigError("min_return set but mu not provided")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for build_constraints and optimise (backward compatible)."""
        d: Dict[str, Any] = {
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "cvar_beta": self.cvar_beta,
            "lambda_cost": self.lambda_cost,
            "cost_per_unit": self.cost_per_unit,
        }
        if self.max_turnover is not None:
            d["max_turnover"] = self.max_turnover
        if self.min_return is not None and self.mu is not None:
            d["min_return"] = self.min_return
            d["mu"] = self.mu
        if self.cvar_limit is not None:
            d["cvar_limit"] = self.cvar_limit
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptimizerConfig":
        """Build from a plain dict (e.g. from YAML/JSON)."""
        return cls(
            min_weight=d.get("min_weight", DEFAULT_MIN_WEIGHT),
            max_weight=d.get("max_weight", DEFAULT_MAX_WEIGHT),
            max_turnover=d.get("max_turnover"),
            min_return=d.get("min_return"),
            mu=d.get("mu"),
            cvar_limit=d.get("cvar_limit"),
            cvar_beta=d.get("cvar_beta", DEFAULT_CVAR_BETA),
            lambda_cost=d.get("lambda_cost", DEFAULT_LAMBDA_COST),
            cost_per_unit=d.get("cost_per_unit", DEFAULT_COST_PER_UNIT),
        )
