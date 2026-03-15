"""
Pytest hooks and shared config. VERBOSE=1 enables rebalancer DEBUG logging during tests.

By default, get_prices() is mocked with deterministic fake data so tests don't hit
the Yahoo Finance API (fast CI, no rate-limit flakiness). To run tests against the
live API, use: pytest -m integration (see test_data.py).
"""
import os

import numpy as np
import pandas as pd
import pytest


def pytest_configure(config):
    """Turn on rebalancer logger when VERBOSE=1."""
    if os.environ.get("VERBOSE") == "1":
        import logging
        from rebalancer._logging import configure_logging
        configure_logging(level=logging.DEBUG)


@pytest.fixture(scope="session")
def fake_prices():
    """Deterministic price DataFrame (252 days, 5 tickers) for tests that need get_prices."""
    from rebalancer.data import DEFAULT_TICKERS
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    # Geometric random walk so prices are positive
    log_returns = rng.standard_normal((252, len(DEFAULT_TICKERS))) * 0.01
    prices = 100 * np.exp(np.cumsum(log_returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=DEFAULT_TICKERS)


@pytest.fixture
def mock_get_prices(monkeypatch, fake_prices):
    """
    Replace yf.download() in rebalancer.data so get_prices() returns fake data
    without hitting Yahoo Finance. Works for all callers (import get_prices or
    data.get_prices). Integration tests live in test_data_live.py without this.
    """
    import pandas as pd
    from rebalancer.data import DEFAULT_TICKERS

    def _fake_download(tickers, start=None, end=None, **kwargs):
        cols = tickers if isinstance(tickers, list) else [tickers]
        out = fake_prices[cols].copy()
        # get_prices expects raw["Close"] with MultiIndex columns for multiple tickers
        out.columns = pd.MultiIndex.from_product([["Close"], out.columns])
        return out

    monkeypatch.setattr("rebalancer.data.yf.download", _fake_download)
