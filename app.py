import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
from portfolio_management.data.data_loader import DataLoader
from portfolio_management.portfolio.portfolio import Portfolio
from portfolio_management.portfolio.optimizer import PortfolioOptimizer
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

def main():
    st.title('Portfolio Management with Monte Carlo Simulation')

    st.write(""" 
        Welcome to the Portfolio Management Application. Define your investment preferences and utilize Monte Carlo simulations to project and analyze potential portfolio performance.
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
    )

    if not selected_tickers:
        st.info('Please select at least one stock ticker to proceed.')
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start Date', value=pd.to_datetime('2014-01-01'), help='The start date for historical data.')
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

    st.write('Choose Investment Input Option: Use Weights and Initial Investment')
    custom_weights = None
    initial_investment = st.number_input(
        'Initial Investment ($):',
        value=1000.0,
        min_value=0.0,
        help='Total amount you plan to invest.'
    )

    # Add weighting method
    weighting_method = st.radio(
        "Choose weighting method:",
        ("Optimize Portfolio", "Custom Weights", "Equal Weights")
    )

    if weighting_method == "Custom Weights":
        custom_weights = {}
        st.write("Enter custom weights (must sum to 1.0)")

        for ticker in selected_tickers:
            weight = st.number_input(
                f"Weight for {ticker}:",
                min_value=0.0,
                max_value=1.0,
                value=1.0 / len(selected_tickers),
                step=0.01
            )
            custom_weights[ticker] = weight

        # Validate weights sum to 1
        total_weight = sum(custom_weights.values())
        if abs(total_weight - 1.0) > 0.0001:
            st.warning(f"Total weights sum to {total_weight:.2f}. Please adjust to sum to 1.0")
            st.stop()

    optimization_choice = st.selectbox(
        'Optimization Strategy',
        ('Maximize Sharpe Ratio', 'Balanced Portfolio'),
        help='Choose an optimization strategy.'
    )

    balanced = (optimization_choice == 'Balanced Portfolio')

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

        
        # Calculate log returns
        portfolio = Portfolio(stock_data)
        portfolio.calculate_returns()  # Ensure this uses log returns internally
        expected_returns = (1 + portfolio.returns.mean()) ** 252 - 1  # Compounded annual returns
        covariance_matrix = portfolio.returns.cov() * 252  # Annualized covariance

        # Scale risk-free rate to daily
        daily_risk_free_rate = (1 + risk_free_rate) ** (1 / 252) - 1

        if weighting_method == "Optimize Portfolio":
            optimizer = PortfolioOptimizer(expected_returns, covariance_matrix, risk_free_rate=daily_risk_free_rate)
            weights, sharpe = optimizer.optimize_weights(num_simulations=10000)
            st.write(f"Optimized Portfolio Sharpe Ratio: {sharpe:.4f}")
            display_optimal_weights(tickers, weights, streamlit_display=True)
        elif weighting_method == "Custom Weights":
            weights = [custom_weights[ticker] for ticker in tickers]
        else:  # Equal Weights
            weights = [1.0 / len(tickers)] * len(tickers)

        # Run Monte Carlo simulation
        simulation = MonteCarloSimulation(portfolio.returns, initial_investment, weights)
        all_cumulative_returns, final_portfolio_values = simulation.run_simulation(int(num_simulations), int(time_horizon))

        st.header('4. Simulation Results')
        insights = get_simulation_insights(final_portfolio_values, initial_investment)
        for key, value in insights.items():
            st.write(f"**{key}:** {value}")

        plot_interactive_simulation_results(all_cumulative_returns, final_portfolio_values, end_date)

if __name__ == '__main__':
    main()
