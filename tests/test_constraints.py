"""Tests for build_constraints and InfeasibleConfigError."""
import numpy as np
import pytest
import cvxpy as cp
from rebalancer.constraints import build_constraints
from rebalancer.exceptions import InfeasibleConfigError


def test_min_weight_gt_max_weight_raises():
    """min_weight > max_weight raises InfeasibleConfigError."""
    w = cp.Variable(3)
    config = {"min_weight": 0.5, "max_weight": 0.3}
    with pytest.raises(InfeasibleConfigError, match="min_weight must be <= max_weight"):
        build_constraints(w, None, config)


def test_min_return_without_mu_raises():
    """min_return set without mu raises InfeasibleConfigError."""
    w = cp.Variable(3)
    config = {"min_weight": 0.0, "max_weight": 1.0, "min_return": 0.01}
    with pytest.raises(InfeasibleConfigError, match="mu not provided"):
        build_constraints(w, np.ones(3) / 3, config)
