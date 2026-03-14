# rebalancer/tax.py
"""
tax.py
------
Cost basis tracking and tax impact estimation for rebalancing decisions.

Covers:
    - FIFO cost basis tracking per asset
    - Realised gain/loss calculation for a proposed trade
    - Wash-sale flag (30-day window)
    - Tax penalty scalar for the optimiser

Scope note: this is a simplified model suitable for a single-account,
single-currency robo advisor. It does not handle:
    - Multi-lot average cost basis
    - Tax-loss harvesting across accounts
    - Country-specific tax rates (though rate is configurable)
"""

from dataclasses import dataclass, field
from datetime import date, timedelta
from collections import deque
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Lot:
    """A single purchase lot: quantity bought at a given price on a given date."""
    quantity: float
    purchase_price: float
    purchase_date: date


@dataclass
class TaxPosition:
    """
    Full tax state for one asset.
    lots: FIFO queue of purchase lots
    realised_losses: list of (date, loss_amount) — for wash-sale tracking
    """
    ticker: str
    lots: deque = field(default_factory=deque)
    realised_losses: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cost basis tracker
# ---------------------------------------------------------------------------

class CostBasisTracker:
    """
    Tracks FIFO cost basis across all assets in the portfolio.

    Usage
    -----
    tracker = CostBasisTracker(tickers)
    tracker.add_lot("VTI", quantity=10, purchase_price=200.0, purchase_date=date.today())
    result = tracker.sell("VTI", quantity=5, sale_price=220.0, sale_date=date.today())
    """

    def __init__(self, tickers: list[str]):
        self.positions = {t: TaxPosition(ticker=t) for t in tickers}

    def add_lot(
        self,
        ticker: str,
        quantity: float,
        purchase_price: float,
        purchase_date: date,
    ) -> None:
        """Record a new purchase lot (FIFO queue)."""
        if ticker not in self.positions:
            raise KeyError(f"Unknown ticker: {ticker}")
        self.positions[ticker].lots.append(
            Lot(quantity=quantity, purchase_price=purchase_price, purchase_date=purchase_date)
        )

    def sell(
        self,
        ticker: str,
        quantity: float,
        sale_price: float,
        sale_date: date,
    ) -> dict:
        """
        Process a sale using FIFO. Returns a dict with:
            realised_gain   : float  — positive = gain, negative = loss
            wash_sale_risk  : bool   — True if a loss was realised within
                                       30 days of a prior loss on same ticker
            lots_consumed   : int    — number of lots fully or partially sold
        """
        if ticker not in self.positions:
            raise KeyError(f"Unknown ticker: {ticker}")

        position = self.positions[ticker]

        if sum(l.quantity for l in position.lots) < quantity - 1e-9:
            raise ValueError(
                f"Cannot sell {quantity} of {ticker} — insufficient lots"
            )

        remaining = quantity
        realised_gain = 0.0
        lots_consumed = 0

        while remaining > 1e-9 and position.lots:
            lot = position.lots[0]

            if lot.quantity <= remaining:
                # Consume entire lot
                realised_gain += lot.quantity * (sale_price - lot.purchase_price)
                remaining -= lot.quantity
                position.lots.popleft()
                lots_consumed += 1
            else:
                # Partial lot
                realised_gain += remaining * (sale_price - lot.purchase_price)
                lot.quantity -= remaining
                remaining = 0.0
                lots_consumed += 1

        # Wash-sale check: was a loss realised on this ticker in the last 30 days?
        wash_sale_risk = False
        if realised_gain < 0:
            cutoff = sale_date - timedelta(days=30)
            recent_losses = [
                (d, amt) for d, amt in position.realised_losses
                if d >= cutoff and amt < 0
            ]
            wash_sale_risk = len(recent_losses) > 0
            position.realised_losses.append((sale_date, realised_gain))

        return {
            "realised_gain":  realised_gain,
            "wash_sale_risk": wash_sale_risk,
            "lots_consumed":  lots_consumed,
        }

    def unrealised_gain(self, ticker: str, current_price: float) -> float:
        """
        Total unrealised gain/loss for a ticker at the current market price.
        Positive = gain, negative = loss.
        """
        position = self.positions[ticker]
        return sum(
            l.quantity * (current_price - l.purchase_price)
            for l in position.lots
        )


# ---------------------------------------------------------------------------
# Tax penalty for the optimiser
# ---------------------------------------------------------------------------

# German Abgeltungsteuer: 25% + 5.5% Solidarity surcharge (Soli) on the tax
ABGELTUNGSTEUER_RATE = 0.25 * (1 + 0.055)  # ~0.26375


def tax_penalty(
    w_prev: np.ndarray,
    w_proposed: np.ndarray,
    current_prices: np.ndarray,
    cost_bases: np.ndarray,
    portfolio_value: float,
    tax_rate: float = ABGELTUNGSTEUER_RATE,
    allowance: float = 1000.0,
) -> float:
    """
    Estimate the tax cost of moving from w_prev to w_proposed.

    Simplified model: for each asset being reduced, estimate the
    realised gain as (current_price - cost_basis) * units_sold.
    Positive gains are summed, then the allowance (Sparer-Pauschbetrag)
    is subtracted, and tax_rate is applied to the remainder. Losses are
    not penalised.

    Defaults are calibrated to German Abgeltungsteuer (25% + 5.5% Soli)
    and Sparer-Pauschbetrag (€1,000 p.a. per person; €2,000 for couples).

    Parameters
    ----------
    w_prev : np.ndarray
        Current weights, shape (n_assets,).
    w_proposed : np.ndarray
        Proposed weights, shape (n_assets,).
    current_prices : np.ndarray
        Current market prices, shape (n_assets,).
    cost_bases : np.ndarray
        Average cost basis per unit, shape (n_assets,).
    portfolio_value : float
        Total portfolio value in currency units.
    tax_rate : float
        Marginal tax rate on capital gains. Default ~26.375% (German
        Abgeltungsteuer + Soli). Church tax not included.
    allowance : float
        Annual tax-free allowance in currency units (e.g. Sparer-Pauschbetrag
        €1,000). Applied to reduce taxable gain for this rebalancing.

    Returns
    -------
    float
        Estimated tax cost as a fraction of portfolio value.
        Can be passed directly as a penalty term in the optimiser config.
    """
    w_prev     = np.asarray(w_prev).ravel()
    w_proposed = np.asarray(w_proposed).ravel()

    delta_w = w_proposed - w_prev  # negative = selling

    total_gross_gain = 0.0
    for i, dw in enumerate(delta_w):
        if dw < 0:  # selling asset i
            units_sold = abs(dw) * portfolio_value / current_prices[i]
            gain_per_unit = current_prices[i] - cost_bases[i]
            if gain_per_unit > 0:
                total_gross_gain += units_sold * gain_per_unit

    taxable_gain = max(0.0, total_gross_gain - allowance)
    total_tax = taxable_gain * tax_rate
    return total_tax / portfolio_value