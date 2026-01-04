"""
Utility Helpers Module

This module provides helper functions for:
- Converting simulation time steps to real dates
- Calculating and displaying portfolio statistics
- Visualizing Monte Carlo simulation results
- Formatting output for different environments (Streamlit, console, matplotlib)
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dateutil.relativedelta import relativedelta


def convert_time_steps_to_dates(start_date_str, time_steps):
    """
    Convert time step indices to actual calendar dates.
    
    Since Monte Carlo simulations use integer time steps (0, 1, 2, ...),
    this function converts them to real trading dates for visualization.
    
    Args:
        start_date_str (str): Starting date in format 'YYYY-MM-DD'
        time_steps (array-like): Array of integer time steps (e.g., [0, 1, 2, ..., 252])
    
    Returns:
        list: Dates as strings in 'YYYY-MM-DD' format
        Example: ['2024-01-01', '2024-01-02', '2024-01-03', ...]
    """
    # Convert the start date string to a datetime object
    start_date = pd.to_datetime(start_date_str)
    
    # Calculate the actual dates by adding days to start_date
    # Each time_step represents one trading day
    actual_dates = [
        pd.to_datetime(start_date + relativedelta(days=int(step))).strftime('%Y-%m-%d') 
        for step in time_steps
    ]
    
    return actual_dates


def plot_interactive_simulation_results(all_cumulative_returns, final_portfolio_values, start_date):
    """
    Create interactive Plotly visualization of Monte Carlo simulation results.
    
    Displays two subplots:
    1. Multiple simulation paths showing portfolio value over time
    2. Histogram of final values with mean and VaR marked
    
    Args:
        all_cumulative_returns (np.array): Portfolio values over time
                                          Shape: (time_horizon, num_simulations)
                                          Each column is one simulation path
        final_portfolio_values (np.array): Final portfolio value for each simulation
                                          Shape: (num_simulations,)
        start_date (str): Starting date for x-axis in format 'YYYY-MM-DD'
    """
    # Limit number of paths plotted to avoid overwhelming the visualization
    # 100 paths shows the range without cluttering
    num_simulations_to_plot = min(100, all_cumulative_returns.shape[1])
    time_steps = np.arange(all_cumulative_returns.shape[0])

    # Create subplots: left (line plot), right (histogram)
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=(
            'Monte Carlo Simulation - Cumulative Returns',
            'Distribution of Final Portfolio Values'
        )
    )

    # Plot cumulative return paths (left subplot)
    # Each line represents one possible future outcome
    for i in range(num_simulations_to_plot):
        fig.add_trace(
            go.Scatter(
                x=convert_time_steps_to_dates(start_date, time_steps),
                y=all_cumulative_returns[:, i],
                mode='lines',
                line=dict(width=1),
                showlegend=False,  # Don't show individual paths in legend
                hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
            ),
            row=1,
            col=1
        )
    
    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_yaxes(title_text='Portfolio Value ($)', row=1, col=1)

    # Plot histogram of final values (right subplot)
    # Shows the distribution of outcomes - key for risk assessment
    fig.add_trace(
        go.Histogram(
            x=final_portfolio_values,
            nbinsx=50,  # 50 bins to show distribution detail
            marker_color='blue',
            opacity=0.75,
            showlegend=False,
            hovertemplate='Portfolio Value: $%{x:,.0f}<br>Frequency: %{y}<extra></extra>'
        ),
        row=1,
        col=2
    )
    
    # Calculate key statistics
    mean_value = np.mean(final_portfolio_values)
    var_95 = np.percentile(final_portfolio_values, 5)  # 95% confidence VaR

    # Add reference lines to histogram
    # Red line: expected return (mean)
    fig.add_vline(x=mean_value, line=dict(color='red', dash='dash'), row=1, col=2)
    # Green line: Value at Risk (5th percentile - 95% confidence)
    fig.add_vline(x=var_95, line=dict(color='green', dash='dash'), row=1, col=2)

    fig.update_xaxes(title_text='Final Portfolio Value ($)', row=1, col=2)
    fig.update_yaxes(title_text='Frequency', row=1, col=2)

    fig.update_layout(height=500, width=1000)

    st.plotly_chart(fig)


def get_simulation_insights(sim_results, initial_investment):
    """
    Calculate key statistics from Monte Carlo simulation results.
    
    Computes metrics important for investment decisions:
    - Expected value and distribution measures
    - Risk metrics (VaR, CVaR)
    - Probability of loss
    - Risk-adjusted return (Sharpe ratio)
    
    Args:
        sim_results (np.array): Final portfolio values from all simulations
        initial_investment (float): Starting investment amount
    
    Returns:
        dict: Dictionary with formatted statistics
              Keys: 'Initial Investment', 'Expected Final Portfolio Value', etc.
              Values: Formatted strings with currency/percentage
    """
    # Calculate distribution statistics
    mean_return = np.mean(sim_results)
    median_return = np.median(sim_results)
    std_dev = np.std(sim_results)
    
    # Calculate Value at Risk (VaR) at 95% confidence level
    # VaR 95% = the worst expected loss in 5% of scenarios
    # Example: VaR of $1000 means in 5% of cases, you could lose $1000 or more
    percentile_5 = np.percentile(sim_results, 5)  # 5th percentile
    var_95 = initial_investment - percentile_5  # Loss amount
    
    # Calculate Conditional VaR (CVaR) at 95% confidence
    # CVaR = average loss when things go really wrong (in worst 5% of scenarios)
    # This is worse than VaR and gives a sense of tail risk
    worst_5_percent = sim_results[sim_results <= percentile_5]
    cvar_95 = initial_investment - np.mean(worst_5_percent)
    
    # Probability of loss: percentage of scenarios where you lose money
    # If 10% of 10,000 simulations show losses, prob_loss = 10%
    prob_loss = np.mean(sim_results < initial_investment) * 100
    
    # Sharpe ratio: return per unit of risk
    # Higher is better; assumes risk-free rate of 0
    sharpe_ratio = (mean_return - initial_investment) / std_dev

    # Format results for display
    insights = {
        'Initial Investment': f"${initial_investment:,.2f}",
        'Expected Final Portfolio Value': f"${mean_return:,.2f}",
        'Median Final Portfolio Value': f"${median_return:,.2f}",
        'Standard Deviation of Final Portfolio Value': f"${std_dev:,.2f}",
        'Value at Risk (VaR 95%)': f"${var_95:,.2f}",
        'Conditional Value at Risk (CVaR 95%)': f"${cvar_95:,.2f}",
        'Probability of Loss': f"{prob_loss:.2f}%",
        'Sharpe Ratio': f"{sharpe_ratio:.4f}"
    }
    return insights


def print_simulation_insights(sim_results, initial_investment):
    """
    Print simulation insights to console in formatted table.
    
    Displays all key metrics for investment analysis, including:
    - Return metrics (expected value, median)
    - Risk metrics (volatility, VaR, CVaR)
    - Probability of losses
    - Risk-adjusted returns
    
    Args:
        sim_results (np.array): Final portfolio values from all simulations
        initial_investment (float): Starting investment amount
    
    Returns:
        dict: Dictionary of insights for programmatic access
    """
    # Get all insights/metrics
    insights = get_simulation_insights(sim_results, initial_investment)
    
    # Calculate expected return percentage
    mean_return = np.mean(sim_results)
    expected_return = ((mean_return / initial_investment) - 1) * 100
    num_losses = np.sum(sim_results < initial_investment)
    total_sims = len(sim_results)
    
    # Print formatted results
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    
    for key, value in insights.items():
        print(f"{key:45s}: {value}")
    
    print(f"\n{'Expected Return':45s}: {expected_return:.2f}%")
    print(f"{'Number of Loss Scenarios':45s}: {num_losses:,} out of {total_sims:,}")
    print("="*60 + "\n")
    
    return insights


def plot_simulation_results(all_cumulative_returns, final_portfolio_values):
    """
    Plot simulation results using matplotlib (for non-Streamlit environments).
    
    Creates two subplots:
    1. Line plot: Multiple simulation paths
    2. Histogram: Distribution of final values with statistics
    
    Works in Jupyter notebooks, scripts, or standalone Python environments.
    Falls back gracefully if matplotlib is not installed.
    
    Args:
        all_cumulative_returns (np.array): Portfolio values over time
                                          Shape: (time_horizon, num_simulations)
        final_portfolio_values (np.array): Final portfolio values from all simulations
    """
    try:
        import matplotlib.pyplot as plt
        
        # Limit paths to plot for clarity
        num_simulations_to_plot = min(100, all_cumulative_returns.shape[1])
        time_steps = np.arange(all_cumulative_returns.shape[0])
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Left subplot: Cumulative returns paths
        # Each semi-transparent line is one possible future outcome
        for i in range(num_simulations_to_plot):
            ax1.plot(time_steps, all_cumulative_returns[:, i], alpha=0.3, linewidth=0.5)
        ax1.set_xlabel('Trading Days')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Monte Carlo Simulation - 100 Sample Paths')
        ax1.grid(True, alpha=0.3)
        
        # Right subplot: Histogram of final values
        ax2.hist(final_portfolio_values, bins=50, color='blue', alpha=0.75, edgecolor='black')
        mean_value = np.mean(final_portfolio_values)
        var_95 = np.percentile(final_portfolio_values, 5)
        
        # Add reference lines
        ax2.axvline(
            mean_value, color='red', linestyle='--', linewidth=2, 
            label=f'Mean: ${mean_value:,.0f}'
        )
        ax2.axvline(
            var_95, color='green', linestyle='--', linewidth=2, 
            label=f'VaR 95%: ${var_95:,.0f}'
        )
        ax2.set_xlabel('Final Portfolio Value ($)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Final Values')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        # Matplotlib not available - print text summary instead
        print("Matplotlib not available. Showing text summary instead:")
        print(f"Mean Final Value: ${np.mean(final_portfolio_values):,.2f}")
        print(f"VaR (95%): ${np.percentile(final_portfolio_values, 5):,.2f}")


def display_optimal_weights(tickers, weights, streamlit_display=True):
    """
    Display optimal portfolio weights in formatted table.
    
    Shows the recommended allocation for each asset. Useful for:
    - Comparing different optimization strategies
    - Verifying weights sum to 100%
    - Making investment decisions
    
    Args:
        tickers (list): List of asset symbols
                       Example: ['AAPL', 'MSFT', 'GOOGL']
        weights (list/array): Allocation weight for each ticker
                             Example: [0.40, 0.35, 0.25]
        streamlit_display (bool): If True, display in Streamlit app
                                 If False, just return DataFrame
    
    Returns:
        pd.DataFrame: Table with ticker symbols and weights (percentage)
    """
    # Create DataFrame with tickers and weights as percentages
    weights_df = pd.DataFrame({
        "Ticker": tickers, 
        "Weight": [f"{w*100:.2f}%" for w in weights]
    })
    
    if streamlit_display:
        st.write("### Optimal Portfolio Weights")
        st.table(weights_df)
    
    return weights_df
