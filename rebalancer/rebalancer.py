# rebalancer/rebalancer.py
"""
rebalancer.py
-------------
Orchestration layer. Takes a real client portfolio state and produces
a concrete rebalancing report with trade list, risk metrics, and tax impact.

This is the main entry point for the library — everything else feeds into here.
"""

from dataclasses import dataclass
from datetime import date
import numpy as np
import pandas as pd

from .data import get_prices, get_returns
from .scenarios import historical_bootstrap
from .optimizer import optimise
from .tax import tax_penalty, CostBasisTracker


# ---------------------------------------------------------------------------
# Input / output types
# ---------------------------------------------------------------------------

@dataclass
class Portfolio:
    """
    Current client portfolio state.

    Parameters
    ----------
    tickers : list[str]
        Asset tickers in consistent order.
    weights : np.ndarray
        Current portfolio weights, shape (n_assets,). Must sum to 1.
    value : float
        Total portfolio value in currency units (EUR).
    cost_bases : np.ndarray
        Average cost basis per unit for each asset, shape (n_assets,).
    prices : np.ndarray
        Current market prices per unit, shape (n_assets,).
    """
    tickers: list[str]
    weights: np.ndarray
    value: float
    cost_bases: np.ndarray
    prices: np.ndarray

    def __post_init__(self):
        self.weights   = np.asarray(self.weights).ravel()
        self.cost_bases = np.asarray(self.cost_bases).ravel()
        self.prices    = np.asarray(self.prices).ravel()

        if not np.isclose(self.weights.sum(), 1.0, atol=1e-3):
            raise ValueError(f"Weights must sum to 1, got {self.weights.sum():.4f}")


@dataclass
class RebalanceReport:
    """Output of the rebalancer — everything needed to execute and explain a rebalance."""
    date: date
    tickers: list[str]

    # Weights
    weights_prev: np.ndarray
    weights_target: np.ndarray
    weights_optimal: np.ndarray

    # Trades
    trade_list: pd.DataFrame   # ticker, direction, units, value_eur

    # Risk
    cvar_before: float
    cvar_after: float
    var_after: float

    # Cost
    turnover: float
    estimated_tax: float       # as fraction of portfolio value
    estimated_tax_eur: float   # in EUR

    # Solver
    tracking_error: float
    status: str

    def summary(self) -> str:
        lines = [
            f"\n{'='*52}",
            f"  Rebalancing Report — {self.date}",
            f"{'='*52}",
            f"  Status:          {self.status}",
            f"  Tracking error:  {self.tracking_error:.6f}",
            f"  CVaR(95%) before: {self.cvar_before:.4f}",
            f"  CVaR(95%) after:  {self.cvar_after:.4f}",
            f"  VaR(95%) after:   {self.var_after:.4f}",
            f"  Turnover:        {self.turnover:.2%}",
            f"  Est. tax cost:   €{self.estimated_tax_eur:.2f} "
            f"({self.estimated_tax:.2%} of portfolio)",
            f"\n  Weights:",
            f"  {'Ticker':<8} {'Current':>10} {'Target':>10} {'Optimal':>10}",
            f"  {'-'*40}",
        ]
        for i, t in enumerate(self.tickers):
            lines.append(
                f"  {t:<8} "
                f"{self.weights_prev[i]:>10.2%} "
                f"{self.weights_target[i]:>10.2%} "
                f"{self.weights_optimal[i]:>10.2%}"
            )
        lines.append(f"\n  Trades:")
        lines.append(self.trade_list.to_string(index=False))
        lines.append(f"{'='*52}\n")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main rebalancer
# ---------------------------------------------------------------------------

def rebalance(
    portfolio: Portfolio,
    w_target: np.ndarray,
    config: dict,
    n_scenarios: int = 10_000,
    start_date: str = "2015-01-01",
) -> RebalanceReport:
    """
    Run a full rebalancing cycle for a client portfolio.

    Parameters
    ----------
    portfolio : Portfolio
        Current client state.
    w_target : np.ndarray
        Target allocation weights, shape (n_assets,).
    config : dict
        Optimizer config — see optimizer.py for supported keys.
    n_scenarios : int
        Number of bootstrap scenarios for CVaR estimation.
    start_date : str
        Start of historical window for scenario generation.

    Returns
    -------
    RebalanceReport
    """
    w_target = np.asarray(w_target).ravel()
    tickers  = portfolio.tickers

    # 1. Fetch returns and generate scenarios
    prices_hist = get_prices(tickers, start=start_date)
    returns     = get_returns(prices_hist)
    scenarios   = historical_bootstrap(returns, n_scenarios=n_scenarios)

    # 2. CVaR before rebalancing
    from .risk import compute_cvar
    risk_before = compute_cvar(portfolio.weights, scenarios, beta=0.95)

    # 3. Optimise
    result = optimise(
        w_prev=portfolio.weights,
        w_target=w_target,
        scenarios=scenarios,
        config=config,
    )
    w_opt = result["weights"]

    # 4. Tax impact
    tax_frac = tax_penalty(
        w_prev=portfolio.weights,
        w_proposed=w_opt,
        current_prices=portfolio.prices,
        cost_bases=portfolio.cost_bases,
        portfolio_value=portfolio.value,
        tax_rate=0.26375,
        allowance=1000.0,
    )

    # 5. Build trade list
    delta_w   = w_opt - portfolio.weights
    trade_rows = []
    for i, ticker in enumerate(tickers):
        if abs(delta_w[i]) < 1e-4:
            continue
        value_eur = delta_w[i] * portfolio.value
        units     = value_eur / portfolio.prices[i]
        trade_rows.append({
            "ticker":    ticker,
            "direction": "BUY" if delta_w[i] > 0 else "SELL",
            "units":     round(units, 4),
            "value_eur": round(value_eur, 2),
        })

    trade_list = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame(
        columns=["ticker", "direction", "units", "value_eur"]
    )

    return RebalanceReport(
        date=date.today(),
        tickers=tickers,
        weights_prev=portfolio.weights,
        weights_target=w_target,
        weights_optimal=w_opt,
        trade_list=trade_list,
        cvar_before=risk_before["cvar"],
        cvar_after=result["cvar"],
        var_after=result["var"],
        turnover=result["turnover"],
        estimated_tax=tax_frac,
        estimated_tax_eur=tax_frac * portfolio.value,
        tracking_error=result["tracking_error"],
        status=result["status"],
    )