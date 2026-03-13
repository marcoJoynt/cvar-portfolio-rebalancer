# tests/test_optimizer.py
import numpy as np
from rebalancer.data import get_prices, get_returns
from rebalancer.scenarios import historical_bootstrap
from rebalancer.optimizer import optimise

prices  = get_prices()
returns = get_returns(prices)
scenarios = historical_bootstrap(returns, n_scenarios=5_000)
n = len(prices.columns)

w_target = np.array([0.40, 0.20, 0.25, 0.10, 0.05])  # VTI/VXUS/BND/GLD/VNQ
w_prev   = np.ones(n) / n  # equal weight starting point

config_layer1 = {
    "min_weight": 0.05,
    "max_weight": 0.60,
}

result = optimise(w_prev, w_target, scenarios, config_layer1)

print("Status:         ", result["status"])
print("Weights:        ", result["weights"].round(4))
print("Tracking error: ", round(result["tracking_error"], 6))
print("CVaR(95%):      ", round(result["cvar"], 4))
print("Turnover:       ", round(result["turnover"], 4))

# Layer 1 check: unconstrained, should return close to w_target
assert result["status"] in ("optimal", "optimal_inaccurate")
np.testing.assert_allclose(result["weights"], w_target, atol=0.01)
print("\nLayer 1 passed.")

# Layer 2: CVaR constraint should push weights away from equities
config_layer2 = {
    "min_weight":  0.05,
    "max_weight":  0.60,
    "cvar_limit":  0.013,   # tighter than the 0.0168 we just saw
    "cvar_beta":   0.95,
}

result2 = optimise(w_prev, w_target, scenarios, config_layer2)

print("\nLayer 2:")
print("Status:         ", result2["status"])
print("Weights:        ", result2["weights"].round(4))
print("Tracking error: ", round(result2["tracking_error"], 6))
print("CVaR(95%):      ", round(result2["cvar"], 4))
print("Turnover:       ", round(result2["turnover"], 4))

# CVaR constraint must be respected
assert result2["cvar"] <= config_layer2["cvar_limit"] + 1e-4, \
    "CVaR constraint violated"

# Weights must differ from target (constraint forced a tradeoff)
assert not np.allclose(result2["weights"], w_target, atol=0.01), \
    "CVaR constraint had no effect"

print("Layer 2 passed.")
# Layer 3: cost penalty should reduce turnover
config_layer3 = {
    "min_weight":   0.05,
    "max_weight":   0.60,
    "cvar_limit":   0.013,
    "cvar_beta":    0.95,
    "lambda_cost":  1.0,
    "cost_per_unit": 0.001,
}

result3 = optimise(w_prev, w_target, scenarios, config_layer3)

print("\nLayer 3:")
print("Status:         ", result3["status"])
print("Weights:        ", result3["weights"].round(4))
print("Tracking error: ", round(result3["tracking_error"], 6))
print("CVaR(95%):      ", round(result3["cvar"], 4))
print("Turnover:       ", round(result3["turnover"], 4))

# CVaR constraint still respected
assert result3["cvar"] <= config_layer3["cvar_limit"] + 1e-4, \
    "CVaR constraint violated"

# Cost penalty should reduce turnover vs layer 2
assert result3["turnover"] < result2["turnover"], \
    "Cost penalty had no effect on turnover"

print("Layer 3 passed.")