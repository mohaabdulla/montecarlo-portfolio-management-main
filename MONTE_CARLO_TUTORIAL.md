# Monte Carlo Portfolio Simulation - Complete Tutorial

## Table of Contents
1. [What is Monte Carlo Simulation?](#what-is-monte-carlo-simulation)
2. [Core Concepts](#core-concepts)
3. [How This Project Works](#how-this-project-works)
4. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
5. [Understanding the Code](#understanding-the-code)
6. [Key Metrics Explained](#key-metrics-explained)
7. [Running Examples](#running-examples)
8. [Practical Applications](#practical-applications)

---

## What is Monte Carlo Simulation?

### The Intuition
Monte Carlo simulation is a computational technique that uses random sampling to estimate outcomes of uncertain events. It answers the question:

**"Given what happened in the past, what are the possible futures?"**

### How It Works
Instead of predicting a single future outcome (which would be wrong), Monte Carlo generates thousands of possible scenarios by:

1. **Learning from history**: Calculating average returns and volatility from past data
2. **Adding randomness**: Simulating thousands of "what-if" paths using these statistics
3. **Analyzing patterns**: Finding the distribution of outcomes to assess risk

### Real-World Analogy
Think of weather forecasting:
- We observe past weather patterns (historical returns)
- We know humidity, pressure patterns create certain weather conditions (correlations)
- Meteorologists create multiple scenarios (Monte Carlo paths) to show probabilities
- They show you there's a 70% chance of rain, but also might be sunny (distribution of outcomes)

---

## Core Concepts

### 1. Daily Returns
A stock price changes daily. We measure this as a **percentage return**:

$$\text{Daily Return} = \frac{\text{Price Today} - \text{Price Yesterday}}{\text{Price Yesterday}}$$

**Example**:
- Yesterday: AAPL = $150
- Today: AAPL = $152.50
- Return = (152.50 - 150) / 150 = 0.0167 = **1.67%**

Why percentages? Because a $1 gain means different things for a $10 stock vs a $1000 stock.

### 2. Annualized Return
Daily returns vary wildly. To get a yearly expectation:

$$\text{Annual Return} = \text{Average Daily Return} \times 252$$

(252 = typical trading days per year in the US stock market)

**Example**:
- Average daily return = 0.001 (0.1%)
- Annual return = 0.001 × 252 = **25.2% per year**

### 3. Volatility (Risk)
How much does a stock bounce around? We measure this with **standard deviation**:

$$\text{Volatility} = \sqrt{\text{Variance of Daily Returns}}$$

Higher volatility = more risk but also more potential for big gains/losses.

**Example**:
- Tech stock: 30% annual volatility (swings a lot - risky but exciting)
- Bond index: 5% annual volatility (stable - boring but safe)

### 4. Correlation
Do assets move together or independently?

- **Correlation = +1.0**: Assets move in perfect sync (if one goes up, other always goes up)
- **Correlation = 0.0**: Assets are independent (movement of one tells nothing about other)
- **Correlation = -1.0**: Assets move opposite (if one goes up, other always goes down)

**Why it matters**: 
- **High correlation**: Bad for diversification (you're taking same risk twice)
- **Low correlation**: Good for diversification (risks cancel out)

### 5. Covariance Matrix
A table showing correlation between all pairs of assets in your portfolio. Used in simulations to ensure returns are realistic (assets move together as they do in real markets).

### 6. Sharpe Ratio
The most important metric for investment performance:

$$\text{Sharpe Ratio} = \frac{\text{Annual Return} - \text{Risk Free Rate}}{\text{Annual Volatility}}$$

**Interpretation**:
- How much extra return you get per unit of risk taken
- Higher is better (more return for same risk)
- Sharpe ratio of 1.0 is decent; 2.0+ is excellent

**Example**:
- Portfolio A: 10% return, 10% volatility → Sharpe = (10% - 2%) / 10% = **0.80**
- Portfolio B: 12% return, 12% volatility → Sharpe = (12% - 2%) / 12% = **0.83**
- Portfolio B is slightly better risk-adjusted even though volatility is higher

---

## How This Project Works

### The Pipeline
```
┌─────────────────────────────────────────────────────────────────┐
│ 1. LOAD HISTORICAL DATA                                         │
│    - Download 5+ years of daily stock prices from Yahoo Finance │
│    - Convert prices to percentage daily returns                 │
└────────────────────┬────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. CALCULATE STATISTICS                                         │
│    - Average daily return per asset                             │
│    - Covariance matrix (correlation structure)                  │
│    - Annualize to yearly values (× 252)                         │
└────────────────────┬────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. DETERMINE WEIGHTS                                            │
│    - Either: Optimize for max Sharpe ratio                      │
│    - Or: Optimize for min volatility                            │
│    - Or: Use user-provided custom weights                       │
│    - Or: Equal weight all assets                                │
└────────────────────┬────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. RUN MONTE CARLO SIMULATION                                   │
│    For each of 10,000+ simulations:                             │
│      - Generate random daily returns using statistics above     │
│      - Preserve correlations between assets                     │
│      - Apply portfolio weights                                  │
│      - Compound returns over time horizon (e.g., 1 year)        │
│      - Track final portfolio value                              │
└────────────────────┬────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. ANALYZE RESULTS                                              │
│    - Expected value (mean of final values)                      │
│    - Value at Risk (VaR): worst 5% of scenarios                 │
│    - Probability of loss                                        │
│    - Distribution of outcomes                                   │
└────────────────────┬────────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. VISUALIZE RESULTS                                            │
│    - Multiple paths showing possible futures                    │
│    - Histogram of final values                                  │
│    - Risk-return scatter plot                                   │
│    - Efficient frontier curve                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Walkthrough

### Step 1: Load Historical Data
```python
from portfolio_management.data.data_loader import DataLoader

loader = DataLoader()
prices = loader.load_data(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2019-01-01',
    end_date='2024-01-01'
)
```

**Output**: DataFrame with dates as index, tickers as columns, adjusted closing prices as values.

```
            AAPL      MSFT      GOOGL
2019-01-02  154.89    104.52    1050.32
2019-01-03  153.67    104.37    1041.89
...
2024-01-01  189.95    371.05    2752.13
```

### Step 2: Create Portfolio Object and Calculate Returns
```python
from portfolio_management.portfolio.portfolio import Portfolio

portfolio = Portfolio(prices)
portfolio.calculate_returns()

# Now we have daily returns
returns = portfolio.returns
```

**Output**: DataFrame of percentage daily changes.

```
            AAPL      MSFT      GOOGL
2019-01-03  -0.0079   -0.0014   -0.0080
2019-01-04   0.0095    0.0031    0.0145
...
```

### Step 3: Calculate Statistics
```python
# Annualize the statistics (multiply by 252 trading days)
mean_returns = portfolio.returns.mean() * 252
covariance_matrix = portfolio.returns.cov() * 252

# Example output:
# mean_returns = [0.25, 0.28, 0.22]  (25%, 28%, 22% annual expected returns)
# covariance_matrix = 3x3 matrix showing correlation structure
```

### Step 4: Optimize Weights
```python
from portfolio_management.portfolio.optimizer import PortfolioOptimizer

optimizer = PortfolioOptimizer(mean_returns, covariance_matrix)

# Option A: Maximize Sharpe ratio (best risk-adjusted return)
optimal_weights = optimizer.maximize_sharpe_ratio()
# Output: [0.40, 0.35, 0.25]  (40% AAPL, 35% MSFT, 25% GOOGL)

# Option B: Minimize volatility for target return
optimal_weights = optimizer.minimize_volatility(target_return=0.15)
# Output: [0.50, 0.30, 0.20]
```

### Step 5: Run Monte Carlo Simulation
```python
from portfolio_management.monte_carlo.simulation import MonteCarloSimulation

simulation = MonteCarloSimulation(
    returns=portfolio.returns,
    initial_investment=10000,
    weights=optimal_weights
)

all_paths, final_values = simulation.run_simulation(
    num_simulations=10000,
    time_horizon=252  # 1 year
)
```

**What happens internally**:
1. For each of 10,000 simulations:
   - Generate 252 random daily returns for each asset
   - These returns follow the historical distribution (mean, volatility, correlation)
   - Apply weights: portfolio return = 40% * AAPL return + 35% * MSFT return + 25% * GOOGL return
   - Compound returns: portfolio_value = initial * exp(cumsum(returns))
2. Track all 10,000 final values

**Output shapes**:
- `all_paths`: (252, 10000) - portfolio value at each day for each simulation
- `final_values`: (10000,) - final value for each simulation

### Step 6: Analyze Results
```python
import numpy as np

mean_final = np.mean(final_values)  # Expected portfolio value
std_final = np.std(final_values)    # Volatility of outcomes
var_95 = np.percentile(final_values, 5)  # Value at Risk (95% confidence)
prob_loss = np.mean(final_values < 10000) * 100  # % of scenarios with losses

print(f"Expected final value: ${mean_final:,.0f}")
print(f"Worst 5% outcomes: ${var_95:,.0f}")
print(f"Probability of loss: {prob_loss:.1f}%")
```

---

## Understanding the Code

### Key Files and What They Do

#### `portfolio_management/monte_carlo/simulation.py`
**Purpose**: Core simulation engine

**Key Method**: `run_simulation()`
```python
def run_simulation(self, num_simulations, time_horizon):
    """
    1. Initialize arrays to hold results
    2. For each simulation:
       a) Generate random daily returns using multivariate normal distribution
       b) Apply portfolio weights to get portfolio returns
       c) Compound returns over time using exponential growth
       d) Store final values
    3. Return all paths and final values
    """
```

**Important Details**:
- Uses `np.random.multivariate_normal()` to generate realistic correlated returns
- **252 trading days** is the key constant for annualization
- Compound returns using `exp(cumsum(log_returns))` for mathematical accuracy

#### `portfolio_management/portfolio/portfolio.py`
**Purpose**: Manage portfolio data

**Key Method**: `calculate_returns()`
```python
self.returns = self.price_data.pct_change().dropna()
# Converts prices to percentage changes
# Example: $150 → $152 becomes 0.0133 (1.33% return)
```

#### `portfolio_management/portfolio/optimizer.py`
**Purpose**: Find optimal portfolio weights

**Key Methods**:
- `maximize_sharpe_ratio()`: Find best risk-adjusted portfolio
- `minimize_volatility()`: Find lowest-risk portfolio for target return

**How Optimization Works**:
1. Define objective function (what to minimize/maximize)
2. Set constraints (weights sum to 1, no shorting)
3. Use `scipy.optimize.minimize` with SLSQP algorithm
4. Return optimal weights

#### `portfolio_management/data/data_loader.py`
**Purpose**: Download and clean data

**Key Method**: `load_data()`
1. Normalize tickers (BRK.B → BRK-B for yfinance)
2. Batch download from Yahoo Finance
3. Extract adjusted closing prices
4. Handle missing data
5. Return clean price DataFrame

#### `portfolio_management/utils/helpers.py`
**Purpose**: Visualization and analysis utilities

**Key Functions**:
- `get_simulation_insights()`: Calculate key metrics (VaR, mean, median)
- `plot_simulation_results()`: Create visualizations
- `print_simulation_insights()`: Display results

### The Flow Through Code
```
User Input (config.json or Streamlit UI)
        ↓
DataLoader.load_data() → Historical prices
        ↓
Portfolio.calculate_returns() → Daily returns
        ↓
PortfolioOptimizer → Optimal weights
        ↓
MonteCarloSimulation.run_simulation() → 10,000 scenarios
        ↓
helpers.get_simulation_insights() → Risk metrics
        ↓
helpers.plot_simulation_results() → Visualizations
```

---

## Key Metrics Explained

### Expected Value (Mean)
**What it means**: On average, what will your portfolio be worth?

**Formula**: Average of all 10,000 final values

**Example**: If 10,000 simulations end with an average of $12,000:
- Expected return = ($12,000 - $10,000) / $10,000 = **20% return**

**Note**: This is the expected value, not a guarantee.

### Standard Deviation
**What it means**: How much do outcomes vary?

**Formula**: $\sqrt{\frac{\sum (x_i - \text{mean})^2}{n}}$

**Example**:
- Low std dev ($500): Most outcomes cluster near expected value (predictable)
- High std dev ($3000): Outcomes spread wide (uncertain)

### Value at Risk (VaR) at 95% Confidence
**What it means**: In the worst 5% of scenarios, how much could you lose?

**Calculation**: 5th percentile of final values

**Example**:
- VaR 95% = $8,500
- Interpretation: In 5% of scenarios, you could have $8,500 or less
- Maximum expected loss = $10,000 - $8,500 = **$1,500 (15% loss)**

**Important**: VaR doesn't tell you how bad it could be in the worst 1% scenario!

### Conditional Value at Risk (CVaR) at 95% Confidence
**What it means**: In the worst 5% of scenarios, what's the **average** loss?

**More conservative than VaR because** it looks at the worst-case tail.

**Example**:
- VaR 95% = $8,500 (worst 5% scenarios have ≤ $8,500)
- CVaR 95% = $7,500 (average of those worst 5% scenarios is $7,500)

### Probability of Loss
**What it means**: In what % of scenarios do you lose money?

**Calculation**: Number of simulations < initial investment / total simulations × 100%

**Example**:
- 250 out of 10,000 simulations show losses
- Probability of loss = 250/10,000 = **2.5%**

**This answers**: "What are the odds I lose money?"

### Sharpe Ratio
**What it means**: Return per unit of risk (best metric for comparing portfolios)

**Formula**: $\frac{\text{Annual Return} - \text{Risk Free Rate}}{\text{Annual Volatility}}$

**Example**:
- Portfolio A: 10% return, 15% volatility → Sharpe = (10% - 2%) / 15% = **0.53**
- Portfolio B: 9% return, 10% volatility → Sharpe = (9% - 2%) / 10% = **0.70**
- **Portfolio B is better** despite lower return (same risk gets more return)

**Benchmarks**:
- Sharpe < 0.5: Poor
- Sharpe 0.5-1.0: Decent
- Sharpe 1.0-2.0: Good
- Sharpe > 2.0: Excellent

---

## Running Examples

### Example 1: Simple 3-Stock Portfolio (Jupyter Notebook)

```python
import numpy as np
import pandas as pd
from portfolio_management.data.data_loader import DataLoader
from portfolio_management.portfolio.portfolio import Portfolio
from portfolio_management.portfolio.optimizer import PortfolioOptimizer
from portfolio_management.monte_carlo.simulation import MonteCarloSimulation
from portfolio_management.utils.helpers import get_simulation_insights

# 1. Load data
loader = DataLoader()
prices = loader.load_data(
    tickers=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2020-01-01',
    end_date='2024-01-01'
)

# 2. Calculate returns
portfolio = Portfolio(prices)
portfolio.calculate_returns()

# 3. Get statistics
mean_returns = portfolio.returns.mean() * 252
cov_matrix = portfolio.returns.cov() * 252

# 4. Optimize
optimizer = PortfolioOptimizer(mean_returns, cov_matrix)
weights = optimizer.maximize_sharpe_ratio()

print("Optimal weights:")
print(dict(zip(['AAPL', 'MSFT', 'GOOGL'], weights)))

# 5. Simulate
simulation = MonteCarloSimulation(
    portfolio.returns,
    initial_investment=100000,
    weights=weights
)
all_paths, final_values = simulation.run_simulation(10000, 252)

# 6. Analyze
insights = get_simulation_insights(final_values, 100000)
for key, value in insights.items():
    print(f"{key}: {value}")
```

### Example 2: Conservative vs Aggressive (Script)

```python
# Conservative: 60% bonds, 40% stocks
conservative_weights = [0.0, 0.4, 0.0, 0.6]  # AAPL, MSFT, GOOGL, BND

# Aggressive: 30% bonds, 70% stocks
aggressive_weights = [0.35, 0.35, 0.3, 0.0]  # AAPL, MSFT, GOOGL, BND

# Compare simulations
for name, w in [('Conservative', conservative_weights), 
                ('Aggressive', aggressive_weights)]:
    sim = MonteCarloSimulation(portfolio.returns, 100000, w)
    paths, finals = sim.run_simulation(10000, 252)
    
    insights = get_simulation_insights(finals, 100000)
    print(f"\n{name} Portfolio:")
    for k, v in insights.items():
        print(f"  {k}: {v}")
```

### Example 3: Using the Streamlit App

```bash
# Install dependencies
pip install streamlit pandas numpy yfinance plotly scipy

# Run the app
streamlit run app.py

# Open browser to http://localhost:8501
```

Then in the UI:
1. Enter tickers: "AAPL, MSFT, GOOGL, AMZN"
2. Set date range: Last 5 years
3. Set initial investment: $50,000
4. Set time horizon: 52 weeks (1 year)
5. Click "Run Simulation"

---

## Practical Applications

### 1. Retirement Planning
**Question**: "If I invest $500,000 for 30 years, will it be enough to retire?"

**How to use Monte Carlo**:
- Run simulation with 30-year horizon (≈7,500 trading days)
- Calculate how many simulations result in portfolio > $2,000,000
- If 90% of scenarios exceed target, good plan

### 2. Risk Assessment
**Question**: "What's the worst 5% case for my portfolio?"

**How to use Monte Carlo**:
- Run simulation
- Look at VaR metric: "5% of the time, you could lose this much"
- Plan for this scenario in emergency fund

### 3. Portfolio Allocation Decisions
**Question**: "Should I be 60/40 stocks/bonds or 70/30?"

**How to use Monte Carlo**:
- Run simulation for each allocation
- Compare Sharpe ratios
- Choose allocation with best risk-adjusted returns for your risk tolerance

### 4. Performance Benchmarking
**Question**: "Is my portfolio performing better than random?"

**How to use Monte Carlo**:
- Simulate equal-weight portfolio
- Simulate Sharpe-optimal portfolio
- Compare actual returns to both
- See if you beat the alternatives

### 5. Time Horizon Analysis
**Question**: "How does risk change if I invest for 1 year vs 5 years vs 30 years?"

**How to use Monte Carlo**:
- Run simulations with different time horizons
- Plot probability of loss vs time horizon
- Longer horizons = lower risk (time diversification)

### 6. Stress Testing
**Question**: "What if volatility doubles (market crash scenario)?"

**How to use Monte Carlo**:
- Double the covariance matrix
- Re-run simulation
- See how portfolio responds to increased volatility

---

## Common Misconceptions

### ❌ "Monte Carlo predicts the future"
✅ **Truth**: Monte Carlo estimates the **range of possible futures** based on historical patterns. Past performance doesn't guarantee future results.

### ❌ "The expected value will happen"
✅ **Truth**: Expected value is the **average** of all outcomes. Most individual outcomes differ from it.

**Analogy**: A die has expected value of 3.5, but you never roll a 3.5!

### ❌ "VaR tells you the worst that can happen"
✅ **Truth**: VaR tells you the worst in 95% of scenarios. The worst 5% could be worse.

**Solution**: Also look at CVaR (average of the worst 5%).

### ❌ "More simulations = more accurate predictions"
✅ **Truth**: More simulations = cleaner statistics, but not more predictive.

**10,000 simulations** is usually enough; diminishing returns after that.

### ❌ "If correlation is low, portfolios are safe"
✅ **Truth**: Low correlation helps, but high individual volatility still creates risk.

**Example**: Two 50% volatile assets with 0 correlation:
- Portfolio volatility = ~35% (still significant!)

---

## Advanced Topics

### Sobol Sequences (in app.py)
The Streamlit app uses **Sobol sequences** instead of pure random numbers for faster convergence.

- **Random numbers**: Hit randomly distributed points
- **Sobol numbers**: Deliberately fill space more evenly
- **Result**: Better estimates with fewer samples

### Cholesky Decomposition
Used to generate correlated random returns:

```
correlated = random_normals @ cholesky(covariance_matrix).T
```

Ensures generated returns have the right correlations.

### SLSQP Optimization
Algorithm used for weight optimization:
- **SLSQP** = Sequential Least Squares Programming
- Handles constraints (weights sum to 1, no shorting)
- Local optimization (may not find global optimum with difficult problems)

---

## Troubleshooting

### Problem: "Failed to load stock data"
**Solution**: 
- Check ticker format (MSFT not MSFT-USD, but BRK-B not BRK.B)
- Ensure date range has data (stocks haven't always existed)
- Try a shorter date range

### Problem: "Optimization did not converge"
**Solution**:
- Usually just a warning, results are still usable
- Try a shorter date range
- Try different optimization strategy

### Problem: "Division by zero in Sharpe ratio"
**Cause**: Zero volatility portfolio (all in risk-free asset)
**Solution**: Allow some volatility; pure cash has low Sharpe anyway

---

## Next Steps

1. **Run the Streamlit app**: `streamlit run app.py`
2. **Experiment with different tickers** and time horizons
3. **Compare strategies**: Which allocation has best risk-return?
4. **Read the code comments**: Each module is heavily documented
5. **Modify and extend**: Add new metrics, risk measures, or optimization strategies

---

## Recommended Resources

### Learning Resources
- **Modern Portfolio Theory**: Markowitz, "Portfolio Selection" (1952)
- **Sharpe Ratio**: Sharpe, "The Sharpe Ratio" (1994)
- **Value at Risk**: Jorion, "Value at Risk" (2006)

### Python Libraries
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **SciPy**: Optimization and statistics
- **Plotly**: Interactive visualizations
- **Streamlit**: Web app framework
- **yfinance**: Stock data access

### Online Tools
- [Efficient Frontier Visualization](https://en.wikipedia.org/wiki/Efficient_frontier)
- [Investopedia - Sharpe Ratio](https://www.investopedia.com/terms/s/sharperatio.asp)
- [Modern Portfolio Theory Explained](https://www.investopedia.com/terms/m/modernportfoliotheory.asp)

---

## Questions?

Each Python module has extensive docstrings explaining:
- What the function does
- Why it works that way
- Examples of inputs/outputs
- Mathematical details

Read the comments in the code for deep dives into implementation details!

**Key files to explore**:
- `portfolio_management/monte_carlo/simulation.py` - Core simulation engine (✨ most important)
- `portfolio_management/portfolio/optimizer.py` - Optimization logic
- `portfolio_management/utils/helpers.py` - Analysis and visualization
