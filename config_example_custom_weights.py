from datetime import datetime

"""
Example Configuration File - Custom Portfolio Weights
======================================================
This example demonstrates how to set up custom portfolio weights.
Copy this file to config.py and modify the values to suit your needs.
"""

# List of stock tickers to include in the portfolio
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']

# Date range for historical data
START_DATE = '2020-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')  # Dynamic today's date

# Initial investment amount in dollars
INITIAL_INVESTMENT = 10000

# Number of Monte Carlo simulations to run
NUM_SIMULATIONS = 100000

# Investment time horizon in days (e.g., 252 trading days in a year)
TIME_HORIZON = 252

# Risk-free rate for Sharpe Ratio calculation
RISK_FREE_RATE = 0.02

# ===============================================
# CUSTOM WEIGHTS CONFIGURATION
# ===============================================
# Specify custom weights for each stock in the same order as TICKERS
# Weights will be automatically normalized to sum to 1.0
# Set to None to use equal weights or enable optimization

# Example 1: Tech-heavy portfolio
WEIGHTS = [0.35, 0.25, 0.20, 0.15, 0.05]
# AAPL: 35%, MSFT: 25%, GOOGL: 20%, AMZN: 15%, NVDA: 5%

# Example 2: Equal weights (alternative to None)
# WEIGHTS = [0.20, 0.20, 0.20, 0.20, 0.20]

# Example 3: Conservative (favor large-cap stability)
# WEIGHTS = [0.30, 0.30, 0.20, 0.15, 0.05]

# Example 4: Aggressive (favor high-growth stocks)
# WEIGHTS = [0.15, 0.15, 0.20, 0.20, 0.30]

# ===============================================
# OPTIMIZATION SETTINGS
# ===============================================
# Set OPTIMIZE to False when using custom weights
OPTIMIZE = False  # Must be False to use WEIGHTS above

# Set to True for a balanced portfolio, False to maximize Sharpe Ratio
# (Only applies when OPTIMIZE = True)
BALANCED = False

# ===============================================
# NOTES
# ===============================================
# - Number of weights must match the number of tickers
# - Weights can be any positive numbers; they will be normalized
# - For equal weights, you can set WEIGHTS = None and OPTIMIZE = False
# - To use optimized weights instead, set OPTIMIZE = True and WEIGHTS = None

print("Configuration loaded:")
print(f"  Tickers: {TICKERS}")
print(f"  Custom Weights: {WEIGHTS}")
print(f"  Optimization Enabled: {OPTIMIZE}")
print(f"  End Date: {END_DATE}")
