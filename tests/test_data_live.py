"""
Live API tests (skipped in CI). Run with: RUN_INTEGRATION=1 pytest -m integration -v
"""
import os

import pytest
from rebalancer.data import get_prices


@pytest.mark.integration
def test_get_prices_live():
    """Hit Yahoo Finance; skipped unless RUN_INTEGRATION=1."""
    if not os.environ.get("RUN_INTEGRATION"):
        pytest.skip("set RUN_INTEGRATION=1 to run live API tests")
    prices = get_prices()
    assert not prices.empty
    assert list(prices.columns) == ["VTI", "VXUS", "BND", "GLD", "VNQ"]
