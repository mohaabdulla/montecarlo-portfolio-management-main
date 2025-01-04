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

    def minimize_volatility(self, target_return):
        num_assets = len(self.expected_returns)
        initial_weights = np.ones(num_assets) / num_assets

        # Constraints: weights sum to 1, portfolio return equals target return
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Sum of weights = 1
            {'type': 'eq', 'fun': lambda w: np.dot(w, self.expected_returns) - target_return}  # Target return
        )

        # Bounds: weights between min_weight and 1
        bounds = tuple((self.min_weight, 1) for _ in range(num_assets))

        # Objective: Minimize portfolio variance
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.covariance_matrix, weights)))

        result = minimize(portfolio_volatility, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
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
    st.write("Start typing a stock ticker and select from the suggestions.")
    selected_tickers = st.multiselect(
        'Select Stock Tickers:',
        options=ticker_list,
        help='Type to search and select stock tickers.'
    )

    if not selected_tickers:
        st.info('Please select at least one stock ticker to proceed.')
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start Date', value=pd.to_datetime('2020-01-01'), help='The start date for historical data.')
    with col2:
        end_date = st.date_input('End Date', value=pd.to_datetime(datetime.today() - relativedelta(days=1)), help='The end date for historical data.')

    tickers = selected_tickers

    # Input: Investment Options
    st.header('2. Investment Preferences')
    risk_free_rate = st.number_input(
        'Risk-Free Rate (Annualized):',
        value=0.02,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help='The risk-free rate used for calculations, typically a treasury bond yield.'
    )

    optimize = st.checkbox('Optimize Portfolio', value=False, help='Select to optimize portfolio weights.')

    initial_investment = st.number_input(
        'Initial Investment ($):',
        value=1000.0,
        min_value=0.0,
        help='Total amount you plan to invest.'
    )

    # Input: Simulation Parameters
    st.header('3. Simulation Parameters')
    col1, col2 = st.columns(2)
    with col1:
        num_simulations = st.number_input(
            'Number of Simulations:',
            value=10000,
            min_value=100,
            step=100,
            help='Number of Monte Carlo simulations to run.'
        )
    with col2:
        time_horizon = st.number_input(
            'Time Horizon (Days):',
            value=252,
            min_value=1,
            step=1,
            help='Investment period in days (e.g., 252 for one year).'
        )

    # Button to Run Simulation
    run_simulation = st.button('Run Monte Carlo Simulation')
    if run_simulation:
        # Load data
        data_loader = DataLoader()
        stock_data = data_loader.load_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

        # Validate data
        if stock_data.empty:
            st.error('Failed to load stock data. Please check the tickers and date range.')
            return

        portfolio = Portfolio(stock_data)
        portfolio.calculate_returns()

        # Annualized returns and covariance
        expected_returns = portfolio.returns.mean() * 252
        covariance_matrix = portfolio.returns.cov() * 252

        # Optimization
        if optimize:
            optimizer = PortfolioOptimizer(expected_returns, covariance_matrix, risk_free_rate=risk_free_rate, min_weight=0.01)
            target_return = expected_returns.mean()
            st.write(f"Target Return: {target_return:.4f}")
            try:
                weights = optimizer.minimize_volatility(target_return=target_return)
                st.subheader('Optimal Portfolio Weights:')
                display_optimal_weights(tickers, weights, streamlit_display=True)
            except ValueError as e:
                st.error(f"Optimization failed: {e}")
                return
        else:
            weights = [1.0 / len(tickers)] * len(tickers)  # Equal weights

        # Run Monte Carlo simulation
        log_returns = np.log(stock_data / stock_data.shift(1)).dropna()  # Correct log returns
        simulation = MonteCarloSimulation(log_returns, initial_investment, weights)
        all_cumulative_returns, final_portfolio_values = simulation.run_simulation(int(num_simulations), int(time_horizon))

        # Display Results
        st.header('4. Simulation Results')
        insights = get_simulation_insights(final_portfolio_values, initial_investment)
        for key, value in insights.items():
            st.write(f"**{key}:** {value}")

        # Plot results
        st.subheader('Interactive Plots')
        plot_interactive_simulation_results(all_cumulative_returns, final_portfolio_values, end_date)

if __name__ == '__main__':
    main()
