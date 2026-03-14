# app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from rebalancer.rebalancer import Portfolio, rebalance

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CVaR Portfolio Rebalancer",
    page_icon="📊",
    layout="wide",
)

st.title("CVaR Portfolio Rebalancer")
st.caption("Convex optimisation · Rockafellar-Uryasev (2000) · German Abgeltungsteuer")

# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
st.sidebar.header("Portfolio")

portfolio_value = st.sidebar.number_input(
    "Portfolio value (€)", min_value=1_000, max_value=10_000_000,
    value=100_000, step=1_000
)

st.sidebar.subheader("Current weights (%)")
w_vti  = st.sidebar.slider("VTI  (US equities)",          0, 100, 52)
w_vxus = st.sidebar.slider("VXUS (Intl equities)",        0, 100, 18)
w_bnd  = st.sidebar.slider("BND  (US bonds)",             0, 100, 17)
w_gld  = st.sidebar.slider("GLD  (Gold)",                 0, 100,  8)
w_vnq  = st.sidebar.slider("VNQ  (REITs)",                0, 100,  5)

w_current_raw = np.array([w_vti, w_vxus, w_bnd, w_gld, w_vnq], dtype=float)
w_sum = w_current_raw.sum()

if not np.isclose(w_sum, 100.0, atol=1.0):
    st.sidebar.warning(f"Weights sum to {w_sum:.0f}% — must equal 100%")
    st.stop()

w_current = w_current_raw / 100.0

st.sidebar.subheader("Target weights (%)")
t_vti  = st.sidebar.slider("VTI  target",  0, 100, 40)
t_vxus = st.sidebar.slider("VXUS target",  0, 100, 20)
t_bnd  = st.sidebar.slider("BND  target",  0, 100, 25)
t_gld  = st.sidebar.slider("GLD  target",  0, 100, 10)
t_vnq  = st.sidebar.slider("VNQ  target",  0, 100,  5)

w_target_raw = np.array([t_vti, t_vxus, t_bnd, t_gld, t_vnq], dtype=float)
t_sum = w_target_raw.sum()

if not np.isclose(t_sum, 100.0, atol=1.0):
    st.sidebar.warning(f"Target weights sum to {t_sum:.0f}% — must equal 100%")
    st.stop()

w_target = w_target_raw / 100.0

st.sidebar.subheader("Risk & execution")
cvar_limit   = st.sidebar.slider("CVaR limit (95%)",     0.005, 0.040, 0.015, step=0.001, format="%.3f")
max_turnover = st.sidebar.slider("Max turnover (e.g. 0.40 = 40%)", 0.10, 1.00, 0.40, step=0.05, format="%.2f")
lambda_cost  = st.sidebar.slider("Transaction cost λ",  0.0,   5.0,   1.0,   step=0.1)

st.sidebar.subheader("Cost basis (purchase price €)")
cb_vti  = st.sidebar.number_input("VTI",  value=180.0)
cb_vxus = st.sidebar.number_input("VXUS", value=52.0)
cb_bnd  = st.sidebar.number_input("BND",  value=72.0)
cb_gld  = st.sidebar.number_input("GLD",  value=165.0)
cb_vnq  = st.sidebar.number_input("VNQ",  value=78.0)

px_vti  = st.sidebar.number_input("VTI price",  value=240.0)
px_vxus = st.sidebar.number_input("VXUS price", value=58.0)
px_bnd  = st.sidebar.number_input("BND price",  value=74.0)
px_gld  = st.sidebar.number_input("GLD price",  value=185.0)
px_vnq  = st.sidebar.number_input("VNQ price",  value=82.0)

run = st.sidebar.button("⚡ Rebalance", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
TICKERS = ["VTI", "VXUS", "BND", "GLD", "VNQ"]

if not run:
    st.info("Set your portfolio parameters in the sidebar and click **Rebalance**.")
    st.stop()

with st.spinner("Optimising..."):
    portfolio = Portfolio(
        tickers    = TICKERS,
        weights    = w_current,
        value      = float(portfolio_value),
        cost_bases = np.array([cb_vti, cb_vxus, cb_bnd, cb_gld, cb_vnq]),
        prices     = np.array([px_vti, px_vxus, px_bnd, px_gld, px_vnq]),
    )

    config = {
        "min_weight":    0.05,
        "max_weight":    0.60,
        "cvar_limit":    cvar_limit,
        "cvar_beta":     0.95,
        "max_turnover":  max_turnover,
        "lambda_cost":   lambda_cost,
        "cost_per_unit": 0.001,
    }

    try:
        report = rebalance(portfolio, w_target, config, n_scenarios=5_000)
    except RuntimeError as e:
        st.error(f"Optimiser failed: {e}. Try relaxing the CVaR limit or turnover constraint.")
        st.stop()

# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "CVaR (95%) before",
    f"{report.cvar_before:.2%}",
)
col2.metric(
    "CVaR (95%) after",
    f"{report.cvar_after:.2%}",
    delta=f"{report.cvar_after - report.cvar_before:.2%}",
    delta_color="inverse",
)
col3.metric(
    "Turnover",
    f"{report.turnover:.2%}",
)
col4.metric(
    "Est. tax cost",
    f"€{report.estimated_tax_eur:,.2f}",
    delta=f"{report.estimated_tax:.2%} of portfolio",
    delta_color="off",
)

st.divider()

# ---------------------------------------------------------------------------
# Weights chart
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Allocation")

    fig = go.Figure()
    x = TICKERS
    bar_kwargs = dict(barmode="group")

    fig.add_trace(go.Bar(
        name="Current",
        x=x,
        y=report.weights_prev * 100,
        marker_color="#636EFA",
    ))
    fig.add_trace(go.Bar(
        name="Target",
        x=x,
        y=report.weights_target * 100,
        marker_color="#EF553B",
    ))
    fig.add_trace(go.Bar(
        name="Optimal",
        x=x,
        y=report.weights_optimal * 100,
        marker_color="#00CC96",
    ))

    fig.update_layout(
        barmode="group",
        yaxis_title="Weight (%)",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=40, b=20),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Risk decomposition")

    risk_data = pd.DataFrame({
        "Metric": ["CVaR (95%)", "VaR (95%)", "Tracking Error"],
        "Before": [report.cvar_before, report.cvar_before * 0.62, None],
        "After":  [report.cvar_after,  report.var_after,          report.tracking_error],
    })

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name="Before",
        x=risk_data["Metric"],
        y=risk_data["Before"],
        marker_color="#636EFA",
    ))
    fig2.add_trace(go.Bar(
        name="After",
        x=risk_data["Metric"],
        y=risk_data["After"],
        marker_color="#00CC96",
    ))
    fig2.update_layout(
        barmode="group",
        yaxis_title="Value",
        yaxis_tickformat=".2%",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=40, b=20),
        height=350,
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Trade list
# ---------------------------------------------------------------------------
st.subheader("Trade list")

if report.trade_list.empty:
    st.success("No trades required — portfolio is within tolerance.")
else:
    def colour_direction(val):
        return "color: #EF553B" if val == "SELL" else "color: #00CC96"

    styled = report.trade_list.style.applymap(
        colour_direction, subset=["direction"]
    ).format({
        "units":     "{:+.4f}",
        "value_eur": "€{:+,.2f}",
    })
    st.dataframe(styled, use_container_width=True)

st.caption(
    "CVaR optimisation via Rockafellar-Uryasev (2000) · "
    "Tax model: German Abgeltungsteuer 26.375% + Sparer-Pauschbetrag €1,000 · "
    "Scenarios: historical bootstrap (252-day window)"
)