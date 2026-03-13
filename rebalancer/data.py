# rebalancer/data.py
"""
Fetch historical prices and compute returns for the assets we use in the
CVaR rebalancer. All data is daily; returns are used later to build scenarios.
"""
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Default portfolio: US stocks, ex-US stocks, bonds, gold, real estate.
DEFAULT_TICKERS = ["VTI", "VXUS", "BND", "GLD", "VNQ"]


def get_prices(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = "2015-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download daily close prices from Yahoo Finance. Returns a DataFrame
    with dates as index and one column per ticker.
    """
    # threads=False avoids SQLite "database is locked" when yfinance uses its cache
    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False, threads=False
    )

    # yfinance returns MultiIndex columns when multiple tickers; get Close only.
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    prices = prices.reindex(columns=tickers)
    # Drop tickers that failed to download (all-NaN columns) and any rows with NaNs.
    prices = prices.dropna(axis=1, how="all")
    prices = prices.dropna()

    if prices.empty:
        raise ValueError(f"No price data returned for: {tickers}")

    return prices


def get_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    """
    Turn price series into daily returns (percent change day-over-day).
    Simple returns by default. Use simple (not log) for CVaR — losses are
    defined as -r, which only makes sense for simple returns.
    """
    if log:
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()