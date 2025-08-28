# app.py
# Streamlit Monte Carlo Portfolio Tool (lean + fast deploy)
# Requirements (keep it light):
# streamlit, pandas, numpy, scipy, yfinance, plotly, python-dateutil

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize


# ---------- Helpers ----------
TRADING_DAYS = 252


def normalize_tickers(tickers):
    """Normalize user tickers to yfinance format (e.g., BRK.B -> BRK-B)."""
    return [t.replace(".", "-").strip().upper() for t in tickers if t.strip()]


def load_prices(tickers, start, end):
    """
    Batched yfinance download with robust handling of single/multi-index outputs.
    Uses auto_adjust=True (so 'Close' is adjusted) and falls back from 'Adj Close' to 'Close'.
    Returns a DataFrame with columns per original user ticker.
    """
    if not tickers:
        return pd.DataFrame()

    norm = normalize_tickers(tickers)

    df = yf.download(
        " ".join(norm),
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,  # adjusted prices directly in 'Close'
        group_by="ticker",
        threads=True,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    out = {}
    if isinstance(df.columns, pd.MultiIndex):
        # Multi-ticker case
        for t in norm:
            block = df.get(t)
            if block is None or block.empty:
                continue
            if "Adj Close" in block.columns:
                s = block["Adj Close"].dropna().rename(t)
            elif "Close" in block.columns:
                s = block["Close"].dropna().rename(t)
            else:
                continue
            if not s.empty:
                out[t] = s
    else:
        # Single-ticker case
        candidate = None
        if "Adj Close" in df.columns:
            candidate = df["Adj Close"]
        elif "Close" in df.columns:
            candidate = df["Close"]
        if candidate is not None:
            out[norm[0]] = candidate.dropna().rename(norm[0])

    if not out:
        return pd.DataFrame()

    prices = pd.DataFrame(out).dropna(how="all")
    # Map back to original symbols (preserve user input casing)
    rename_map = {n: o for n, o in zip(norm, tickers)}
    prices = prices.rename(columns=rename_map)
    # Drop columns that are all NaN (symbols that failed)
    prices = prices.dropna(axis=1, how="all")
    return prices


def portfolio_performance(weights, mean_daily, cov_daily, rf=0.0):
    """
    Given weights, mean daily returns, and daily covariance:
    returns (annual_return, annual_volatility, sharpe)
    """
    weights = np.array(weights)
    mu = np.dot(weights, mean_daily) * TRADING_DAYS
    var = weights @ cov_daily @ weights.T * TRADING_DAYS
    vol = math.sqrt(var) if var > 0 else 0.0
    sharpe = (mu - rf) / vol if vol > 0 else 0.0
    return mu, vol, sharpe


def simulate_portfolios(n, mean_daily, cov_daily, rf=0.0, allow_short=False):
    """
    Monte Carlo simulation of random portfolios.
    If allow_short=False: weights are on the simplex (Dirichlet).
    If allow_short=True: sample from normal, then normalize to sum to 1 (weights may be negative).
    """
    dim = len(mean_daily)
    rets = np.zeros(n)
    vols = np.zeros(n)
    sharpes = np.zeros(n)
    weights_list = []

    for i in range(n):
        if allow_short:
            w = np.random.normal(size=dim)
            if np.isclose(w.sum(), 0):
                w = np.ones(dim)
            w = w / w.sum()
        else:
            w = np.random.dirichlet(np.ones(dim), size=1).flatten()

        mu, vol, s = portfolio_performance(w, mean_daily, cov_daily, rf)
        rets[i], vols[i], sharpes[i] = mu, vol, s
        weights_list.append(w)

    return rets, vols, sharpes, np.array(weights_list)


def optimize_min_variance(mean_daily, cov_daily, allow_short=False):
    n = len(mean_daily)
    x0 = np.ones(n) / n
    bounds = None if allow_short else tuple((0.0, 1.0) for _ in range(n))
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def obj(w):
        return w @ cov_daily @ w * TRADING_DAYS

    res = minimize(obj, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)
    return res


def optimize_max_sharpe(mean_daily, cov_daily, rf=0.0, allow_short=False):
    n = len(mean_daily)
    x0 = np.ones(n) / n
    bounds = None if allow_short else tuple((0.0, 1.0) for _ in range(n))
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def neg_sharpe(w):
        mu, vol, s = portfolio_performance(w, mean_daily, cov_daily, rf)
        return -s

    res = minimize(neg_sharpe, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)
    return res


def efficient_frontier(mean_daily, cov_daily, targets, allow_short=False):
    """Compute minimum-variance portfolios for a range of target annual returns."""
    n = len(mean_daily)
    bounds = None if allow_short else tuple((0.0, 1.0) for _ in range(n))
    cons_base = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    frontier_vols = []
    frontier_weights = []

    for target in targets:
        # Convert annual target to daily target
        daily_target = target / TRADING_DAYS

        def obj(w):
            return w @ cov_daily @ w * TRADING_DAYS

        def ret_constraint(w, tgt=daily_target):
            return w @ mean_daily - tgt

        cons = tuple(cons_base + [{"type": "eq", "fun": ret_constraint}])
        x0 = np.ones(n) / n
        res = minimize(obj, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            mu, vol, s = portfolio_performance(res.x, mean_daily, cov_daily, rf=0.0)
            frontier_vols.append(vol)
            frontier_weights.append(res.x)
        else:
            frontier_vols.append(np.nan)
            frontier_weights.append(np.full(n, np.nan))

    return np.array(frontier_vols), np.array(frontier_weights)


def format_weights(weights, labels):
    df = pd.DataFrame({"Asset": labels, "Weight": weights})
    df["Weight %"] = (df["Weight"] * 100).round(2)
    return df.sort_values("Weight %", ascending=False).reset_index(drop=True)


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Monte Carlo Portfolio", layout="wide")
st.title("📈 Monte Carlo Portfolio Simulator (Fast Deploy)")

with st.sidebar:
    st.header("Inputs")

    # Common quick-pick tickers for convenience
    default_choices = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "V", "JPM", "XOM"]
    st.caption("Tip: Use '-' not '.' (e.g., BRK-B).")
    tickers_str = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, NVDA, GOOGL",
        help="Enter any symbols yfinance supports (e.g., 'SPY, QQQ, IWM' or 'AAPL, MSFT').",
    )
    st.multiselect("Quick add", options=default_choices, default=[], key="quick_add")

    # Merge the two sources of tickers
    user_tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    all_tickers = sorted(set(user_tickers + st.session_state.quick_add))

    today = date.today()
    default_start = today - relativedelta(years=5)
    start_date = st.date_input("Start date", value=default_start, max_value=today - timedelta(days=1))
    end_date = st.date_input("End date", value=today)

    rf_percent = st.number_input("Risk-free rate (annual, %)", value=0.0, step=0.25, min_value=-5.0, max_value=20.0)
    rf = rf_percent / 100.0

    allow_short = st.checkbox("Allow short selling (weights may be negative)", value=False)

    n_sims = st.slider("Number of Monte Carlo portfolios", min_value=500, max_value=20000, value=5000, step=500)

    st.markdown("---")
    st.caption("This version avoids slow/blocked network calls for ticker lists and uses a batched yfinance loader.")


if not all_tickers:
    st.info("Enter at least one ticker to begin.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

with st.spinner("Downloading price data…"):
    prices = load_prices(all_tickers, start_date, end_date + timedelta(days=1))  # include end date

if prices.empty or prices.shape[1] == 0:
    st.error("Failed to load stock data. Check tickers (e.g., use BRK-B not BRK.B) and the date range.")
    st.caption(f"Tickers: {all_tickers} | Start: {start_date} | End: {end_date}")
    st.stop()

# Align and clean
prices = prices.dropna(how="all").sort_index()
prices = prices.dropna(axis=1, how="any")  # drop assets without full history in the window
assets = list(prices.columns)

if len(assets) < 1:
    st.error("No asset has a complete price history in the selected window. Try a shorter window or different tickers.")
    st.stop()

st.subheader("Price History")
norm_prices = prices / prices.iloc[0] * 100.0
st.plotly_chart(px.line(norm_prices, title="Indexed Prices (100 = start)"), use_container_width=True)

# Returns and stats
returns = prices.pct_change().dropna(how="all")
mean_daily = returns.mean().values  # vector
cov_daily = returns.cov().values    # matrix

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Assets", len(assets))
with col2:
    st.metric("Observations (days)", len(returns))
with col3:
    st.metric("Period", f"{prices.index[0].date()} → {prices.index[-1].date()}")

st.markdown("### Summary (Annualized)")
ann_mu = pd.Series(mean_daily * TRADING_DAYS, index=assets, name="Return")
ann_vol = pd.Series(np.sqrt(np.diag(cov_daily) * TRADING_DAYS), index=assets, name="Volatility")
summary = pd.concat([ann_mu, ann_vol], axis=1).sort_values("Return", ascending=False)
summary["Return %"] = (summary["Return"] * 100).round(2)
summary["Volatility %"] = (summary["Volatility"] * 100).round(2)
st.dataframe(summary[["Return %", "Volatility %"]], use_container_width=True)

# Simulation
with st.spinner("Running Monte Carlo simulation…"):
    sim_rets, sim_vols, sim_sharpes, sim_weights = simulate_portfolios(
        n_sims, mean_daily, cov_daily, rf=rf, allow_short=allow_short
    )

# Optimization
min_var_res = optimize_min_variance(mean_daily, cov_daily, allow_short=allow_short)
max_sharpe_res = optimize_max_sharpe(mean_daily, cov_daily, rf=rf, allow_short=allow_short)

if not (min_var_res.success and max_sharpe_res.success):
    st.warning("Optimization did not fully converge for some targets, results may be approximate.")

w_min_var = min_var_res.x
w_max_sharpe = max_sharpe_res.x

min_mu, min_vol, min_s = portfolio_performance(w_min_var, mean_daily, cov_daily, rf)
max_mu, max_vol, max_s = portfolio_performance(w_max_sharpe, mean_daily, cov_daily, rf)

# Efficient frontier (curve)
target_grid = np.linspace(sim_rets.min(), sim_rets.max(), 40)
frontier_vols, frontier_weights = efficient_frontier(mean_daily, cov_daily, target_grid, allow_short=allow_short)

# Plot Risk-Return
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=sim_vols,
        y=sim_rets,
        mode="markers",
        name="Simulations",
        opacity=0.4,
        marker=dict(size=5),
        hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=frontier_vols,
        y=target_grid,
        mode="lines",
        name="Efficient Frontier",
        line=dict(width=3),
        hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=[min_vol],
        y=[min_mu],
        mode="markers",
        name="Min Variance",
        marker=dict(size=12, symbol="x"),
        hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=[max_vol],
        y=[max_mu],
        mode="markers",
        name="Max Sharpe",
        marker=dict(size=12, symbol="star"),
        hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
    )
)

fig.update_layout(
    title="Risk vs Return",
    xaxis_title="Volatility (σ, annualized)",
    yaxis_title="Expected Return (annualized)",
)
st.plotly_chart(fig, use_container_width=True)

# Allocation tables
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Max Sharpe Portfolio")
    st.write(
        f"Return: **{max_mu:.2%}**, Volatility: **{max_vol:.2%}**, Sharpe: **{max_s:.2f}** (rf={rf_percent:.2f}%)"
    )
    st.dataframe(format_weights(w_max_sharpe, assets), use_container_width=True)

with c2:
    st.markdown("#### Minimum Variance Portfolio")
    st.write(
        f"Return: **{min_mu:.2%}**, Volatility: **{min_vol:.2%}**, Sharpe: **{min_s:.2f}** (rf={rf_percent:.2f}%)"
    )
    st.dataframe(format_weights(w_min_var, assets), use_container_width=True)

# Correlation heatmap
st.markdown("### Asset Return Correlations")
corr = returns[assets].corr()
st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix"), use_container_width=True)

# Downloads
st.markdown("### Downloads")
csv_prices = prices.to_csv().encode("utf-8")
st.download_button("Download Prices (CSV)", data=csv_prices, file_name="prices.csv", mime="text/csv")

results_df = pd.DataFrame(
    {
        "Return": sim_rets,
        "Volatility": sim_vols,
        "Sharpe": sim_sharpes,
    }
)
st.download_button(
    "Download Simulation Results (CSV)",
    data=results_df.to_csv(index=False).encode("utf-8"),
    file_name="simulation_results.csv",
    mime="text/csv",
)

st.caption(
    "Note: This app intentionally avoids external ticker-list APIs (Alpha Vantage / NASDAQ FTP) for reliability on Streamlit Cloud."
)
