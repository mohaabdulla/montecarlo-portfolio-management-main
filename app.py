import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
from scipy.stats import qmc, norm


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


def portfolio_performance(weights, mean_daily, cov_daily, rf=0.0, trading_periods=TRADING_DAYS):
    """
    Given weights, mean daily returns, and daily covariance:
    returns (annual_return, annual_volatility, sharpe)
    """
    weights = np.array(weights)
    mu = np.dot(weights, mean_daily) * trading_periods
    var = weights @ cov_daily @ weights.T * trading_periods
    vol = math.sqrt(var) if var > 0 else 0.0
    sharpe = (mu - rf) / vol if vol > 0 else 0.0
    return mu, vol, sharpe


def simulate_portfolios(n, mean_daily, cov_daily, rf=0.0, allow_short=False, trading_periods=TRADING_DAYS):
    dim = len(mean_daily)
    
    mean_daily = mean_daily.astype(np.float32)
    cov_daily = cov_daily.astype(np.float32)
    
    if allow_short:
        weights_list = np.random.normal(size=(n, dim)).astype(np.float32)
        sums = weights_list.sum(axis=1, keepdims=True)
        sums[np.abs(sums) < 1e-10] = 1.0
        weights_list = weights_list / sums
    else:
        weights_list = np.random.dirichlet(np.ones(dim), size=n).astype(np.float32)
    
    rets = (weights_list @ mean_daily) * trading_periods
    vars = np.einsum('ij,jk,ik->i', weights_list, cov_daily, weights_list) * trading_periods
    vols = np.sqrt(np.maximum(vars, 0))
    sharpes = np.divide(rets - rf, vols, where=vols > 0, out=np.zeros_like(vols))

    return rets, vols, sharpes, weights_list


@st.cache_data(ttl=3600)
def chol_correlated_normals_sobol(n_steps, n_sims, mean_daily_tuple, cov_daily_tuple, seed=0):
    mean_daily = np.array(mean_daily_tuple, dtype=np.float32)
    cov_daily = np.array(cov_daily_tuple, dtype=np.float32)
    n_assets = len(mean_daily)
    
    sobol = qmc.Sobol(d=n_assets, scramble=True, seed=seed)
    n_samples = n_steps * n_sims
    
    batch_size = 100000
    shocks_list = []
    
    for i in range(0, n_samples, batch_size):
        batch_end = min(i + batch_size, n_samples)
        batch_samples = batch_end - i
        
        uniform_samples = sobol.random(batch_samples)
        standard_normals = norm.ppf(uniform_samples).astype(np.float16)
        
        if i == 0:
            L = np.linalg.cholesky(cov_daily).astype(np.float16)
        
        correlated_normals = standard_normals @ L.T
        shocks_batch = correlated_normals + mean_daily.astype(np.float16)
        shocks_list.append(shocks_batch)
    
    shocks = np.vstack(shocks_list).astype(np.float32)
    shocks = shocks.reshape(n_steps, n_sims, n_assets)
    
    return shocks


def run_time_series_simulation(weights, mean_daily, cov_daily, time_horizon, n_sims, initial_investment):
    weights = np.asarray(weights, dtype=np.float32)
    mean_daily_tuple = tuple(mean_daily.astype(np.float32).tolist())
    cov_daily_tuple = tuple(map(tuple, cov_daily.astype(np.float32).tolist()))
    
    shocks = chol_correlated_normals_sobol(time_horizon, n_sims, mean_daily_tuple, cov_daily_tuple)
    
    port_ret = np.einsum('ijk,k->ij', shocks, weights)
    
    cum = np.exp(np.cumsum(port_ret, axis=0))
    
    all_paths = cum * np.float32(initial_investment)
    
    final_values = all_paths[-1]
    
    return all_paths, final_values


def optimize_min_variance(mean_daily, cov_daily, allow_short=False, trading_periods=TRADING_DAYS):
    n = len(mean_daily)
    x0 = np.ones(n) / n
    bounds = None if allow_short else tuple((0.0, 1.0) for _ in range(n))
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def obj(w):
        return w @ cov_daily @ w * trading_periods

    res = minimize(obj, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)
    return res


def optimize_max_sharpe(mean_daily, cov_daily, rf=0.0, allow_short=False, trading_periods=TRADING_DAYS):
    n = len(mean_daily)
    x0 = np.ones(n) / n
    bounds = None if allow_short else tuple((0.0, 1.0) for _ in range(n))
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)

    def neg_sharpe(w):
        mu, vol, s = portfolio_performance(w, mean_daily, cov_daily, rf, trading_periods)
        return -s

    res = minimize(neg_sharpe, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)
    return res


def efficient_frontier(mean_daily, cov_daily, targets, allow_short=False, trading_periods=TRADING_DAYS):
    """Compute minimum-variance portfolios for a range of target annual returns."""
    n = len(mean_daily)
    bounds = None if allow_short else tuple((0.0, 1.0) for _ in range(n))
    cons_base = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    frontier_vols = []
    frontier_weights = []

    for target in targets:
        # Convert annual target to period target
        period_target = target / trading_periods

        def obj(w):
            return w @ cov_daily @ w * trading_periods

        def ret_constraint(w, tgt=period_target):
            return w @ mean_daily - tgt

        cons = tuple(cons_base + [{"type": "eq", "fun": ret_constraint}])
        x0 = np.ones(n) / n
        res = minimize(obj, x0=x0, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            mu, vol, s = portfolio_performance(res.x, mean_daily, cov_daily, rf=0.0, trading_periods=trading_periods)
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
st.title("📈 Monte Carlo Portfolio Simulator")

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

    n_sims = st.number_input("Number of Monte Carlo portfolios", min_value=1000, max_value=10000000, value=10000, step=1000)
    
    time_horizon_days = st.number_input("Time horizon (days)", min_value=3, max_value=1000, value=252, step=3)
    time_horizon = time_horizon_days // 3
    
    initial_investment = st.number_input("Initial investment ($)", min_value=1000, max_value=10000000, value=5000000, step=1000)

    st.markdown("---")
    st.caption("Using 3-day candles for optimized performance. Data is resampled from daily to 3-day periods.")


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

# Resample to 3-day candles for better performance
prices_3d = prices.resample('3D').last().dropna(how="all")
prices_3d = prices_3d.dropna(axis=1, how="any")

st.subheader("Price History")
norm_prices = prices / prices.iloc[0] * 100.0
st.plotly_chart(px.line(norm_prices, title="Indexed Prices (100 = start)"), use_container_width=True)

# Returns and stats using 3-day data
returns_3d = prices_3d.pct_change().dropna(how="all")
mean_3d = returns_3d.mean().values
cov_3d = returns_3d.cov().values

# Convert 3-day statistics to daily equivalent for annualization
mean_daily = mean_3d / 3
cov_daily = cov_3d / 3

# Adjust trading days for 3-day periods
TRADING_PERIODS_PER_YEAR = TRADING_DAYS / 3

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Assets", len(assets))
with col2:
    st.metric("Daily Observations", len(prices))
with col3:
    st.metric("3-Day Periods", len(returns_3d))
with col4:
    st.metric("Period", f"{prices.index[0].date()} → {prices.index[-1].date()}")

# Simulation
with st.spinner("Running Monte Carlo simulation…"):
    sim_rets, sim_vols, sim_sharpes, sim_weights = simulate_portfolios(
        int(n_sims), mean_daily, cov_daily, rf=rf, allow_short=allow_short, trading_periods=TRADING_DAYS
    )

# Optimization
min_var_res = optimize_min_variance(mean_daily, cov_daily, allow_short=allow_short, trading_periods=TRADING_DAYS)
max_sharpe_res = optimize_max_sharpe(mean_daily, cov_daily, rf=rf, allow_short=allow_short, trading_periods=TRADING_DAYS)

if not (min_var_res.success and max_sharpe_res.success):
    st.warning("Optimization did not fully converge for some targets, results may be approximate.")

w_min_var = min_var_res.x
w_max_sharpe = max_sharpe_res.x

min_mu, min_vol, min_s = portfolio_performance(w_min_var, mean_daily, cov_daily, rf, TRADING_DAYS)
max_mu, max_vol, max_s = portfolio_performance(w_max_sharpe, mean_daily, cov_daily, rf, TRADING_DAYS)

# Efficient frontier (curve)
target_grid = np.linspace(sim_rets.min(), sim_rets.max(), 20)
frontier_vols, frontier_weights = efficient_frontier(mean_daily, cov_daily, target_grid, allow_short=allow_short, trading_periods=TRADING_DAYS)

# Plot Risk-Return
fig = go.Figure()

sample_size = min(5000, len(sim_vols))
sample_indices = np.random.choice(len(sim_vols), sample_size, replace=False)

fig.add_trace(
    go.Scattergl(
        x=sim_vols[sample_indices],
        y=sim_rets[sample_indices],
        mode="markers",
        name="Simulations",
        opacity=0.4,
        marker=dict(size=3),
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

# Portfolio strategy selection
st.markdown("---")
st.markdown("### Monte Carlo Simulation - Portfolio Value Over Time")
st.write("Select a portfolio strategy to run the time series simulation:")

col_a, col_b, col_c, col_d, col_e = st.columns(5)

with col_a:
    max_sharpe_btn = st.button("Max Sharpe Portfolio", use_container_width=True)
with col_b:
    min_var_btn = st.button("Minimum Variance Portfolio", use_container_width=True)
with col_c:
    equal_weight_btn = st.button("Equal Weight Portfolio", use_container_width=True)
with col_d:
    max_return_btn = st.button("Maximum Return Portfolio", use_container_width=True)
with col_e:
    balanced_btn = st.button("Balanced Portfolio", use_container_width=True)

selected_weights = None
selected_name = None
selected_perf = None

if max_sharpe_btn:
    selected_weights = w_max_sharpe
    selected_name = "Max Sharpe Portfolio"
    selected_perf = (max_mu, max_vol, max_s)
elif min_var_btn:
    selected_weights = w_min_var
    selected_name = "Minimum Variance Portfolio"
    selected_perf = (min_mu, min_vol, min_s)
elif equal_weight_btn:
    selected_weights = np.ones(len(assets)) / len(assets)
    selected_name = "Equal Weight Portfolio"
    selected_perf = portfolio_performance(selected_weights, mean_daily, cov_daily, rf, TRADING_DAYS)
elif max_return_btn:
    idx_max_return = np.argmax(mean_daily)
    selected_weights = np.zeros(len(assets))
    selected_weights[idx_max_return] = 1.0
    selected_name = f"Maximum Return Portfolio ({assets[idx_max_return]})"
    selected_perf = portfolio_performance(selected_weights, mean_daily, cov_daily, rf, TRADING_DAYS)
elif balanced_btn:
    target_return = (min_mu + max_mu) / 2
    balanced_res = optimize_min_variance(mean_daily, cov_daily, allow_short=allow_short, trading_periods=TRADING_DAYS)
    cons_balanced = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w: np.dot(w, mean_daily) * TRADING_DAYS - target_return}
    )
    n = len(mean_daily)
    x0 = np.ones(n) / n
    bounds = None if allow_short else tuple((0.0, 1.0) for _ in range(n))
    def obj_bal(w):
        return w @ cov_daily @ w * TRADING_DAYS
    balanced_res = minimize(obj_bal, x0=x0, method="SLSQP", bounds=bounds, constraints=cons_balanced)
    if balanced_res.success:
        selected_weights = balanced_res.x
        selected_name = "Balanced Portfolio"
        selected_perf = portfolio_performance(selected_weights, mean_daily, cov_daily, rf, TRADING_DAYS)
    else:
        st.warning("Balanced portfolio optimization failed. Using equal weights instead.")
        selected_weights = np.ones(len(assets)) / len(assets)
        selected_name = "Balanced Portfolio (Equal Weight Fallback)"
        selected_perf = portfolio_performance(selected_weights, mean_daily, cov_daily, rf, TRADING_DAYS)

if selected_weights is not None:
    with st.spinner(f"Running time series simulation for {selected_name}…"):
        all_paths, final_values = run_time_series_simulation(
            selected_weights, mean_daily, cov_daily, int(time_horizon), int(n_sims), initial_investment
        )
    
    st.markdown(f"#### Selected Strategy: {selected_name}")
    st.write(
        f"Return: **{selected_perf[0]:.2%}**, Volatility: **{selected_perf[1]:.2%}**, Sharpe: **{selected_perf[2]:.2f}** (rf={rf_percent:.2f}%)"
    )
    st.dataframe(format_weights(selected_weights, assets), use_container_width=True)
    
    fig2 = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Monte Carlo Simulation - Cumulative Returns', 'Distribution of Final Portfolio Values')
    )

    num_paths_to_plot = min(50, int(n_sims))
    time_steps = np.arange(int(time_horizon))
    
    indices = np.linspace(0, int(n_sims) - 1, num_paths_to_plot, dtype=int)
    
    for idx in indices:
        fig2.add_trace(
            go.Scattergl(
                x=time_steps,
                y=all_paths[:, idx],
                mode='lines',
                line=dict(width=0.8),
                showlegend=False,
                opacity=0.5
            ),
            row=1, col=1
        )

    fig2.update_xaxes(title_text='Time Steps', row=1, col=1)
    fig2.update_yaxes(title_text='Portfolio Value ($)', row=1, col=1)

    mean_final = float(np.mean(final_values))
    var_95 = float(np.percentile(final_values, 5))
    
    hist_sample_size = min(10000, len(final_values))
    hist_indices = np.random.choice(len(final_values), hist_sample_size, replace=False)

    fig2.add_trace(
        go.Histogram(
            x=final_values[hist_indices],
            nbinsx=50,
            marker_color='blue',
            opacity=0.75,
            showlegend=False
        ),
        row=1, col=2
    )

    fig2.add_vline(x=mean_final, line=dict(color='red', dash='dash'), row=1, col=2)
    fig2.add_vline(x=var_95, line=dict(color='green', dash='dash'), row=1, col=2)

    fig2.update_xaxes(title_text='Final Portfolio Value ($)', row=1, col=2)
    fig2.update_yaxes(title_text='Frequency', row=1, col=2)

    fig2.update_layout(height=500, width=1400)

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"""
    **Simulation Statistics:**
    - Mean Final Value: **${mean_final:,.2f}**
    - VaR (95%): **${var_95:,.2f}**
    - Expected Gain: **${mean_final - initial_investment:,.2f}** ({((mean_final/initial_investment - 1) * 100):.2f}%)
    - Number of Simulations: **{int(n_sims):,}**
    - Time Horizon: **{int(time_horizon)} 3-day periods** (~{int(time_horizon) * 3} days)
    """)
    
    del all_paths, final_values
    import gc
    gc.collect()
else:
    st.info("👆 Click a button above to run the Monte Carlo simulation for your chosen portfolio strategy.")

# Correlation heatmap
st.markdown("### Asset Return Correlations")
corr = returns_3d[assets].corr()
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
