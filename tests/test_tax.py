# tests/test_tax.py
from datetime import date, timedelta
from rebalancer.tax import ABGELTUNGSTEUER_RATE, CostBasisTracker, tax_penalty
import numpy as np

def test_fifo_gain():
    tracker = CostBasisTracker(["VTI"])
    tracker.add_lot("VTI", quantity=10, purchase_price=100.0, purchase_date=date(2023, 1, 1))
    tracker.add_lot("VTI", quantity=10, purchase_price=150.0, purchase_date=date(2023, 6, 1))

    result = tracker.sell("VTI", quantity=10, sale_price=200.0, sale_date=date(2024, 1, 1))

    # FIFO: first lot sold at 100 -> gain = 10 * (200 - 100) = 1000
    assert result["realised_gain"] == 1000.0
    assert result["wash_sale_risk"] is False

def test_wash_sale_flag():
    tracker = CostBasisTracker(["VTI"])
    tracker.add_lot("VTI", quantity=10, purchase_price=200.0, purchase_date=date(2024, 1, 1))
    tracker.add_lot("VTI", quantity=10, purchase_price=200.0, purchase_date=date(2024, 1, 1))

    # First sale at a loss
    r1 = tracker.sell("VTI", quantity=10, sale_price=150.0, sale_date=date(2024, 6, 1))
    assert r1["realised_gain"] == -500.0
    assert r1["wash_sale_risk"] is False  # no prior losses

    # Second sale at a loss within 30 days — wash sale
    r2 = tracker.sell("VTI", quantity=10, sale_price=150.0, sale_date=date(2024, 6, 15))
    assert r2["realised_gain"] == -500.0
    assert r2["wash_sale_risk"] is True

def test_tax_penalty_only_on_gains():
    w_prev     = np.array([0.5, 0.5])
    w_proposed = np.array([0.3, 0.7])  # selling asset 0, buying asset 1
    prices     = np.array([100.0, 100.0])
    cost_bases = np.array([80.0, 100.0])  # asset 0 has a gain, asset 1 is flat

    penalty = tax_penalty(w_prev, w_proposed, prices, cost_bases,
                          portfolio_value=10_000, tax_rate=0.25, allowance=0)

    assert penalty > 0, "Should have a tax cost on the gain"
    print(f"Tax penalty: {penalty:.4f} ({penalty*100:.2f}% of portfolio)")


def test_tax_penalty_german_allowance():
    """Sparer-Pauschbetrag: gain below allowance => no tax."""
    w_prev     = np.array([0.5, 0.5])
    w_proposed = np.array([0.3, 0.7])
    prices     = np.array([100.0, 100.0])
    cost_bases = np.array([90.0, 100.0])  # asset 0: 10% gain

    # Gross gain = 20% of 5000 = 1000. With allowance=1000, taxable = 0
    penalty = tax_penalty(w_prev, w_proposed, prices, cost_bases,
                          portfolio_value=10_000, allowance=1000.0)
    assert penalty == 0.0

    # Same gain, no allowance => positive penalty at German rate
    # Gross gain = 20% of 10k in asset 0, gain 10/unit => 20*10 = 200
    penalty_no_allowance = tax_penalty(w_prev, w_proposed, prices, cost_bases,
                                       portfolio_value=10_000, allowance=0)
    assert penalty_no_allowance > 0
    expected_tax = 200 * ABGELTUNGSTEUER_RATE / 10_000
    assert abs(penalty_no_allowance - expected_tax) < 1e-9