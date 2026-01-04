"""
Main Module - Portfolio Analysis Orchestration

This is the main entry point for the Monte Carlo portfolio management system.
It orchestrates the complete workflow:

1. Load configuration from config.json
2. Download historical stock data
3. Calculate portfolio statistics
4. Determine optimal weights (via optimization, custom, or equal weighting)
5. Run Monte Carlo simulation
6. Display results and insights

Usage:
    python -m portfolio_management.main
    (Ensure config.json exists in the parent directory)
"""

import json
from portfolio_management.data.data_loader import DataLoader
from portfolio_management.portfolio.portfolio import Portfolio
from portfolio_management.portfolio.optimizer import PortfolioOptimizer
from portfolio_management.monte_carlo.simulation import MonteCarloSimulation
from portfolio_management.utils.helpers import (
    plot_simulation_results,
    print_simulation_insights,
    display_optimal_weights
)


def main():
    """
    Main orchestration function for portfolio analysis.
    
    Workflow:
    1. Loads configuration parameters from config.json
    2. Downloads historical price data for all tickers
    3. Calculates daily returns and statistics
    4. Determines portfolio weights using specified strategy
    5. Runs Monte Carlo simulation with 10,000+ paths
    6. Prints statistical insights about outcomes
    7. Displays visualization of results
    
    Configuration (config.json):
        Required:
            - tickers: List of stock symbols (e.g., ['AAPL', 'MSFT'])
            - start_date: Start date in 'YYYY-MM-DD' format
            - end_date: End date in 'YYYY-MM-DD' format
        
        Optional:
            - initial_investment: Starting amount (default: $1000)
            - num_simulations: Number of Monte Carlo paths (default: 10000)
            - time_horizon: Days to simulate (default: 252 = 1 year)
            - risk_free_rate: Risk-free rate for Sharpe ratio (default: 0.0)
            - weights: Custom weights (overrides optimization)
            - optimization:
                - optimize: True to run optimization (default: False)
                - balanced: True for min volatility, False for max Sharpe
    """
    
    # ==================== STEP 1: Load Configuration ====================
    # Load all parameters from config.json file
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Extract configuration parameters with sensible defaults
    tickers = config.get('tickers', [])
    start_date = config.get('start_date')
    end_date = config.get('end_date')
    initial_investment = config.get('initial_investment', 1000)
    num_simulations = config.get('num_simulations', 10000)
    time_horizon = config.get('time_horizon', 252)  # 252 trading days = 1 year
    risk_free_rate = config.get('risk_free_rate', 0.0)
    custom_weights = config.get('weights')
    
    # Extract optimization settings
    optimization_config = config.get('optimization', {})
    optimize = optimization_config.get('optimize', False)
    balanced = optimization_config.get('balanced', True)

    # ==================== STEP 2: Load Data ====================
    # Download historical price data from Yahoo Finance
    print(f"Loading data for {len(tickers)} assets from {start_date} to {end_date}...")
    data_loader = DataLoader()
    stock_data = data_loader.load_data(tickers, start_date, end_date)
    print(f"Loaded {len(stock_data)} trading days of data")

    # ==================== STEP 3: Calculate Returns ====================
    # Convert prices to daily returns (percentage changes)
    # This is what Monte Carlo simulation uses
    portfolio = Portfolio(stock_data)
    portfolio.calculate_returns()

    # Annualize returns and covariance from historical daily data
    # Multiply by 252 trading days per year
    expected_returns = portfolio.returns.mean() * 252
    covariance_matrix = portfolio.returns.cov() * 252

    # ==================== STEP 4: Determine Weights ====================
    # Use one of three strategies to determine portfolio allocation:
    # A) Optimization (max Sharpe ratio or min volatility)
    # B) Custom weights provided by user
    # C) Equal weighting as default
    
    if optimize:
        # Strategy A: Optimization
        optimizer = PortfolioOptimizer(
            expected_returns,
            covariance_matrix,
            risk_free_rate=risk_free_rate
        )
        if balanced:
            # Minimize volatility approach
            optimal_weights = optimizer.minimize_volatility(
                target_return=expected_returns.mean()
            )
            print("\nOptimal Balanced Portfolio Weights:")
            print("(Minimizes volatility for average return)")
        else:
            # Maximize Sharpe ratio approach
            optimal_weights = optimizer.maximize_sharpe_ratio()
            print("\nOptimal Portfolio Weights to Maximize Sharpe Ratio:")
            print("(Best return per unit of risk)")
        
        display_optimal_weights(stock_data.columns, optimal_weights)
        weights = optimal_weights
        
    elif custom_weights:
        # Strategy B: Use user-provided custom weights
        # Validate that number of weights matches number of assets
        if len(custom_weights) != len(expected_returns):
            raise ValueError(
                f"Number of custom weights ({len(custom_weights)}) must match "
                f"number of assets ({len(expected_returns)})"
            )
        
        # Normalize weights to ensure they sum to 1 (fully invested)
        weights = [w / sum(custom_weights) for w in custom_weights]
        print("\nUsing Custom Weights (Normalized):")
        print(f"Original weights: {custom_weights}")
        print(f"Sum of weights: {sum(custom_weights):.4f} → 1.0000")
        display_optimal_weights(stock_data.columns, weights)
        
    else:
        # Strategy C: Equal weighting (simplest, diversified approach)
        num_assets = len(expected_returns)
        weights = [1.0 / num_assets] * num_assets
        print("\nUsing Equal Weights:")
        print(f"Allocating {100/num_assets:.2f}% to each of {num_assets} assets")
        display_optimal_weights(stock_data.columns, weights)

    # ==================== STEP 5: Monte Carlo Simulation ====================
    # Create simulation engine and run thousands of possible future scenarios
    print(f"\nRunning {num_simulations:,} Monte Carlo simulations...")
    print(f"Time horizon: {time_horizon} trading days")
    
    simulation = MonteCarloSimulation(
        portfolio.returns, 
        initial_investment, 
        weights
    )
    
    # Run simulations and get results
    all_cumulative_returns, final_portfolio_values = simulation.run_simulation(
        num_simulations, time_horizon
    )

    # ==================== STEP 6: Display Results ====================
    # Calculate and print key statistics from simulation results
    print_simulation_insights(final_portfolio_values, initial_investment)

    # ==================== STEP 7: Visualize Results ====================
    # Show interactive plots of simulation paths and outcome distribution
    plot_simulation_results(all_cumulative_returns, final_portfolio_values)
    
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
