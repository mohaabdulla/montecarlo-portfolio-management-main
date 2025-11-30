# Custom Portfolio Weights Guide

## Overview
The Monte Carlo Portfolio Management system now supports custom portfolio weights, allowing you to specify your own asset allocation strategy instead of relying solely on optimization algorithms.

## Methods to Specify Custom Weights

### Method 1: Streamlit Web Interface (Recommended)

1. **Launch the app:**
   ```bash
   streamlit run app.py
   ```

2. **Enter your tickers** in the sidebar (e.g., "AAPL, MSFT, GOOGL, AMZN")

3. **Configure settings:**
   - Select date range
   - Set initial investment
   - Choose time horizon
   - Configure other parameters

4. **Select Custom Weights:**
   - Click the **"Custom Weights"** button
   - Input fields will appear for each asset
   - Enter desired weight for each asset (e.g., 0.40 for 40%)
   - Weights can be decimals or percentages
   - They will be automatically normalized to sum to 1.0

5. **Run the simulation:**
   - Click **"Run Custom Portfolio"**
   - View results including:
     - Expected return and volatility
     - Sharpe ratio
     - Monte Carlo simulation paths
     - Final portfolio value distribution
     - VaR (Value at Risk)

### Method 2: Configuration File (Command-Line)

1. **Edit config.py:**
   ```python
   # Define your tickers
   TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
   
   # Specify custom weights (must match order of TICKERS)
   WEIGHTS = [0.40, 0.30, 0.20, 0.10]
   # This means: 40% AAPL, 30% MSFT, 20% GOOGL, 10% AMZN
   
   # Disable optimization to use custom weights
   OPTIMIZE = False
   ```

2. **Run the main script:**
   ```bash
   python portfolio_management/main.py
   ```

## Important Notes

### Weight Normalization
- Weights don't need to sum exactly to 1.0
- They will be automatically normalized
- Example: `[40, 30, 20, 10]` becomes `[0.40, 0.30, 0.20, 0.10]`
- Example: `[2, 1, 1]` becomes `[0.50, 0.25, 0.25]`

### Weight Constraints
- By default, weights must be non-negative (long-only positions)
- Enable "Allow short selling" checkbox to permit negative weights
- Each weight should be between 0.0 and 1.0 (or -1.0 to 1.0 if shorting)
- Number of weights must match number of tickers

### Common Portfolio Strategies

#### Equal Weight
```python
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
WEIGHTS = [0.25, 0.25, 0.25, 0.25]
```

#### Market Cap Weighted (example)
```python
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
WEIGHTS = [0.35, 0.30, 0.20, 0.15]  # Approximation
```

#### Risk Parity (equal risk contribution)
```python
# Requires volatility calculation
# Lower volatility assets get higher weights
TICKERS = ['AAPL', 'MSFT', 'BND', 'GLD']
WEIGHTS = [0.25, 0.25, 0.30, 0.20]  # Example
```

#### Sector Rotation
```python
# Focus on specific sectors
TICKERS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY']  # Sector ETFs
WEIGHTS = [0.30, 0.20, 0.15, 0.20, 0.15]
```

#### Core-Satellite
```python
# 70% core broad market, 30% satellite high-conviction
TICKERS = ['SPY', 'NVDA', 'TSLA']
WEIGHTS = [0.70, 0.15, 0.15]
```

## Comparing Custom vs Optimized Portfolios

You can compare your custom allocation against optimized strategies:

1. First, run your custom weights portfolio
2. Note the performance metrics (return, volatility, Sharpe)
3. Then click one of the optimization buttons:
   - **Max Sharpe**: Best risk-adjusted returns
   - **Min Variance**: Lowest volatility
   - **Balanced**: Middle ground approach
4. Compare the results side-by-side

## Tips for Setting Custom Weights

1. **Diversification**: Avoid putting too much weight in a single asset
2. **Risk Tolerance**: Higher volatility stocks should have lower weights if you're risk-averse
3. **Investment Goals**: Align weights with your time horizon and objectives
4. **Rebalancing**: Periodically review and adjust weights as markets change
5. **Starting Point**: Use optimization results as a baseline, then adjust based on your preferences

## Example Workflow

### Conservative Investor
```python
# Mix of stable blue-chips and bonds
TICKERS = ['MSFT', 'JNJ', 'PG', 'BND']
WEIGHTS = [0.25, 0.25, 0.20, 0.30]  # 30% bonds for stability
```

### Growth Investor
```python
# Technology-focused high-growth
TICKERS = ['NVDA', 'TSLA', 'AMD', 'PLTR']
WEIGHTS = [0.35, 0.25, 0.25, 0.15]
```

### Dividend Income
```python
# High dividend yield stocks
TICKERS = ['T', 'VZ', 'XOM', 'KO']
WEIGHTS = [0.25, 0.25, 0.25, 0.25]
```

## Troubleshooting

### "Number of custom weights must match number of assets"
- Ensure WEIGHTS list has same length as TICKERS list
- Check for typos in ticker symbols

### "Total weight must be greater than 0"
- At least one weight must be positive
- If using negative weights, ensure net position is positive

### Optimization overriding custom weights
- Set `OPTIMIZE = False` in config.py
- Make sure WEIGHTS is not None

## Further Reading

- [Modern Portfolio Theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory)
- [Asset Allocation Strategies](https://www.investopedia.com/terms/a/assetallocation.asp)
- [Portfolio Rebalancing](https://www.investopedia.com/terms/r/rebalancing.asp)
