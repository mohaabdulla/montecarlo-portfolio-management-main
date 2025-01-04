import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from portfolio_management.data.data_loader import DataLoader
from portfolio_management.portfolio.portfolio import Portfolio
from portfolio_management.monte_carlo.simulation import MonteCarloSimulation
from portfolio_management.utils.helpers import (
    plot_interactive_simulation_results,
    get_simulation_insights,
    display_optimal_weights
)
import os
from dotenv import load_dotenv
from portfolio_management.utils.alpha_vantage import get_alpha_vantage_tickers
from portfolio_management.utils.nasdaq_nyse import get_nasdaq_nyse_tickers
from portfolio_management.utils.utils import merge_tickers

load_dotenv()

class PortfolioOptimizer:
    def __init__(self, expected_returns, covariance_matrix, risk_free_rate=0.0, min_weight=0.01):
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.risk_free_rate = risk_free_rate
        self.min_weight = min_weight

    def maximize_sharpe_ratio(self, partial_weights=None):
        num_assets = len(self.expected_returns)
        initial_weights = np.ones(num_assets) / num_assets

        def negative_sharpe_ratio(weights):
            portfolio_return = np.dot(weights, self.expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))
            return -(portfolio_return - self.risk_free_rate) / portfolio_volatility

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        # Apply partial weights as constraints
        if partial_weights:
            for i, weight in enumerate(partial_weights):
                if weight is not None:
                    constraints.append({'type': 'eq', 'fun': lambda w, i=i, weight=weight: w[i] - weight})

        bounds = [(self.min_weight, 1) for _ in range(num_assets)]
        result = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        return result.x

def main():
    st.title('Portfolio Management with Monte Carlo Simulation')

    st.write("""
    Welcome to the Portfolio Management Application. Define your investment preferences, input stock data, and run a Monte Carlo simulation to analyze portfolio performance.
    """)

    @st.cache_data
    def load_ticker_list():
        alpha_vantage_tickers = get_alpha_vantage_tickers()
        nasdaq_nyse_tickers = get_nasdaq_nyse_tickers()
        all_tickers = merge_tickers([alpha_vantage_tickers, nasdaq_nyse_tickers])
        return all_tickers

    ticker_list = load_ticker_list()

    # Input: Stock Tickers
    st.header('1. Select Stocks and Date Range')
    selected_tickers = st.multiselect('Select Stock Tickers:', options=ticker_list)

    if not selected_tickers:
        st.info('Please select at least one stock ticker to proceed.')
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
    with col2:
        end_date = st.date_input('End Date', value=pd.to_datetime(datetime.today() - relativedelta(days=1)))

    # Input: Investment Options
    st.header('2. Investment Preferences')
    risk_free_rate = st.number_input('Risk-Free Rate (Annualized):', value=0.02, min_value=0.0, max_value=1.0, step=0.01)

    optimize = st.checkbox('Optimize Portfolio', value=False)
    partial_weights = [None] * len(selected_tickers)

    if optimize:
        st.subheader("Partial Weights (Optional)")
        for i, ticker in enumerate(selected_tickers):
            partial_weights[i] = st.number_input(
                f"Weight for {ticker} (leave blank for optimizer to adjust):",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                value=None
            )
    
    initial_investment = st.number_input('Initial Investment ($):', value=1000.0, min_value=0.0)

    # Input: Simulation Parameters
    st.header('3. Simulation Parameters')
    num_simulations = st.number_input('Number of Simulations:', value=10000, min_value=100, step=100)
    time_horizon = st.number_input('Time Horizon (Days):', value=252, min_value=1, step=1)

    # Button to Run Simulation
    run_simulation = st.button('Run Monte Carlo Simulation')
    if run_simulation:
        data_loader = DataLoader()
        stock_data = data_loader.load_data(selected_tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        if stock_data.empty:
            st.error('Failed to load stock data. Please check the tickers and date range.')
            return

        portfolio = Portfolio(stock_data)
        portfolio.calculate_returns()
        expected_returns = portfolio.returns.mean() * 252
        covariance_matrix = portfolio.returns.cov() * 252

        if optimize:
            optimizer = PortfolioOptimizer(expected_returns, covariance_matrix, risk_free_rate=risk_free_rate, min_weight=0.01)
            weights = optimizer.maximize_sharpe_ratio(partial_weights)
        else:
            weights = [1.0 / len(selected_tickers)] * len(selected_tickers)

        st.subheader('Optimized Weights:')
        weight_df = pd.DataFrame({'Ticker': selected_tickers, 'Weight': weights})
        st.dataframe(weight_df)

        simulation = MonteCarloSimulation(portfolio.returns, initial_investment, weights)
        all_cumulative_returns, final_portfolio_values = simulation.run_simulation(int(num_simulations), int(time_horizon))

        st.header('4. Simulation Results')
        insights = get_simulation_insights(final_portfolio_values, initial_investment)
        for key, value in insights.items():
            st.write(f"**{key}:** {value}")

        st.subheader('Interactive Plots')
        plot_interactive_simulation_results(all_cumulative_returns, final_portfolio_values, end_date)

if __name__ == '__main__':
    main()
