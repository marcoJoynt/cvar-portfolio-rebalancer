# CVaR Portfolio Rebalancer

**What it does.** This project rebalances a multi-asset portfolio toward target weights while limiting downside risk (CVaR) and turnover. It uses convex optimisation (Rockafellar–Uryasev) and reports trade lists with estimated German capital-gains tax impact.

**Live demo:** [app](https://cvar-portfolio-rebalancer-production.up.railway.app)

---

## Optimization problem

Decision variable: portfolio weights \( w \in \mathbb{R}^n \).

**Objective (minimise):**
\[
\|w - w_{\text{target}}\|_2^2
\;+\;
\lambda_{\text{cost}} \cdot c \cdot \|w - w_{\text{prev}}\|_1
\]
- First term: tracking error (squared \(L_2\) deviation from target).
- Second term: transaction-cost penalty (L1 turnover × cost per unit).

**Constraints:**
\[
\mathbf{1}^\top w = 1, \qquad
w_{\min} \leq w \leq w_{\max}, \qquad
\|w - w_{\text{prev}}\|_1 \leq \tau_{\max}, \qquad
\text{CVaR}_\beta(w) \leq \gamma.
\]
- Budget: weights sum to 1.
- Box: per-asset min/max (e.g. 5%–60%).
- Turnover: L1 change vs current weights ≤ \(\tau_{\max}\).
- Risk: CVaR at level \(\beta\) (e.g. 95%) ≤ \(\gamma\).

**CVaR (Rockafellar–Uryasev):** For scenario returns \(r_1,\ldots,r_q\) and loss \(L_k = -r_k^\top w\),
\[
\text{CVaR}_\beta(w)
\;=\;
\min_{z \in \mathbb{R}} \;
\left\{\;
z \;+\; \frac{1}{q(1-\beta)} \sum_{k=1}^{q} \max(L_k - z,\, 0)
\;\right\}.
\]
Here \(z\) is the VaR threshold; the \(\max(\cdot,0)\) terms are implemented as `cp.pos()` in CVXPY, so the problem stays convex.

---

## How to run

```bash
pip install -r requirements.txt
python main.py                    # CLI demo: rebalance and print report
streamlit run app.py              # Streamlit dashboard
```

- **`main.py`** — Example portfolio, target, and config; prints a text report.
- **`app.py`** — Interactive UI: sliders for weights, CVaR limit, turnover, cost bases and prices; rebalance button; metrics, charts, and trade list.

## Structure

```
cvar-portfolio-rebalancer/
├── rebalancer/
│   ├── data.py          # price fetching, return computation
│   ├── scenarios.py     # historical bootstrap scenario generation
│   ├── risk.py          # Rockafellar-Uryasev CVaR formulation
│   ├── constraints.py   # budget, box, turnover constraints
│   ├── optimizer.py     # cvxpy problem — tracking error + CVaR constraint
│   ├── tax.py           # FIFO cost basis, wash-sale, Abgeltungsteuer penalty
│   └── rebalancer.py    # orchestration → rebalancing report
├── tests/               # pytest test suite
├── app.py               # Streamlit dashboard
├── main.py              # CLI demo
└── Dockerfile
```

---

## Screenshot

<!-- Replace the path below with your actual screenshot (e.g. docs/dashboard.png) -->
![Streamlit dashboard](docs/dashboard.png)

*Streamlit dashboard: sidebar inputs, rebalance button, metrics, allocation and risk charts, trade list.*

*(Add a screenshot to `docs/dashboard.png` or update the path above.)*

---

## Theory

- **CVaR (Conditional Value-at-Risk)** at e.g. 95% is the *average loss in the worst 5% of outcomes*. It’s a coherent risk measure and more stable than VaR for optimisation.

- **Rockafellar–Uryasev (2000)** showed that CVaR can be written as a minimum over an auxiliary variable \(z\). That formulation is convex, so we can minimise other objectives (like tracking error) subject to a CVaR *constraint* — the solver handles both in one convex program.

- **Here:** we minimise tracking error plus a turnover penalty, subject to a CVaR cap, budget, box, and turnover limit. Scenarios are bootstrap samples of historical returns; tax is approximated with German Abgeltungsteuer (flat rate + allowance).

---

## References

**Papers**

- Rockafellar, R.T. & Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk.* Journal of Risk, 2(3), 21–41.
- Diamond, S. & Boyd, S. (2016). *CVXPY: A Python-Embedded Modeling Language for Convex Optimization.* Journal of Machine Learning Research, 17(83), 1–5.

**Textbook**

- Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization.* Cambridge University Press. [Free online](https://web.stanford.edu/~boyd/cvxbook/).

**Practitioner**

- Wealthfront (2022). *Tax-Loss Harvesting White Paper.* https://research.wealthfront.com/whitepapers/tax-loss-harvesting/

**Regulatory / tax (Germany)**

- Bundeszentralamt für Steuern. *Abgeltungsteuer (Capital Gains Tax).* https://www.bzst.de  
- § 20 EStG (Einkommensteuergesetz) — statutory basis for the 25% + Soli rate.
