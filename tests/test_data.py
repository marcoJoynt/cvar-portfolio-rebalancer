import os

import numpy as np
import pytest
from rebalancer.data import get_prices, get_returns


@pytest.fixture(autouse=True)
def _use_fake_prices(mock_get_prices):
    """Use fake price data in this module so tests don't hit Yahoo Finance."""
    pass


def test_get_prices_shape():
    prices = get_prices()
    assert not prices.empty
    assert prices.isna().sum().sum() == 0


def test_get_returns_simple():
    prices = get_prices()
    returns = get_returns(prices)
    assert returns.shape[0] == prices.shape[0] - 1
    assert returns.isna().sum().sum() == 0


