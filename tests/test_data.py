import numpy as np
from rebalancer.data import get_prices, get_returns


def test_get_prices_shape():
    prices = get_prices()
    assert not prices.empty
    assert prices.isna().sum().sum() == 0
    print(f"  → {len(prices)} days, {list(prices.columns)}, no NaNs")


def test_get_returns_simple():
    prices = get_prices()
    returns = get_returns(prices)
    assert returns.shape[0] == prices.shape[0] - 1
    assert returns.isna().sum().sum() == 0
    print(f"  → {len(returns)} return rows, {returns.shape[1]} assets, no NaNs")
