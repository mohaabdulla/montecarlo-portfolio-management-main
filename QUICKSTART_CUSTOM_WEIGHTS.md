# Quick Start: Custom Portfolio Weights

## 🚀 Get Started in 3 Minutes

### Option 1: Web Interface (Easiest)

```bash
# Step 1: Launch the app
streamlit run app.py

# Step 2: In the browser
# - Enter tickers: AAPL, MSFT, GOOGL, AMZN
# - Click "Custom Weights" button
# - Enter your desired allocation:
#   AAPL: 0.40 (40%)
#   MSFT: 0.30 (30%)
#   GOOGL: 0.20 (20%)
#   AMZN: 0.10 (10%)
# - Click "Run Custom Portfolio"
# - Done! View your results
```

### Option 2: Configuration File (Advanced)

```bash
# Step 1: Edit config.py
```

```python
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
WEIGHTS = [0.40, 0.30, 0.20, 0.10]
OPTIMIZE = False
```

```bash
# Step 2: Run the script
python portfolio_management/main.py
```

## 📊 Example Portfolios

### Conservative (Low Risk)
```python
TICKERS = ['VTI', 'BND', 'GLD']  # Stocks, Bonds, Gold
WEIGHTS = [0.50, 0.40, 0.10]     # 50% stocks, 40% bonds, 10% gold
```

### Balanced (Medium Risk)
```python
TICKERS = ['SPY', 'QQQ', 'IWM', 'BND']
WEIGHTS = [0.40, 0.25, 0.15, 0.20]
```

### Aggressive (High Risk)
```python
TICKERS = ['NVDA', 'TSLA', 'AMD', 'PLTR']
WEIGHTS = [0.35, 0.30, 0.20, 0.15]
```

### Tech-Focused
```python
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']
WEIGHTS = [0.25, 0.25, 0.20, 0.15, 0.15]
```

## ✅ Quick Tips

1. **Weights don't need to be exact** - They're automatically normalized
   - `[40, 30, 20, 10]` → `[0.40, 0.30, 0.20, 0.10]`
   - `[2, 1, 1]` → `[0.50, 0.25, 0.25]`

2. **Number of weights must match tickers**
   - ✅ 4 tickers → 4 weights
   - ❌ 4 tickers → 3 weights (Error!)

3. **Compare with optimized portfolios**
   - Run your custom weights first
   - Then try "Max Sharpe" or "Min Variance"
   - See which performs better!

## 🎯 What You'll See

After running your custom portfolio:
- **Expected Return**: Annual return percentage
- **Volatility**: Risk level (standard deviation)
- **Sharpe Ratio**: Risk-adjusted performance
- **VaR**: Worst-case scenario (5th percentile)
- **Charts**: 
  - Portfolio value over time (50 sample paths)
  - Distribution of final values

## 🔧 Troubleshooting

**"Number of custom weights must match number of assets"**
- Check that you have one weight per ticker
- Example: 3 tickers need exactly 3 weights

**"Total weight must be greater than 0"**
- Make sure at least one weight is positive
- All zeros won't work!

**Optimization overriding my custom weights**
- Set `OPTIMIZE = False` in config.py

## 📚 Learn More

- See `CUSTOM_WEIGHTS_GUIDE.md` for detailed documentation
- Check `config_example_custom_weights.py` for more examples
- Read `README.md` for full feature list

## 💡 Pro Tips

1. Start with equal weights to understand baseline performance
2. Gradually adjust weights based on your conviction
3. Use the correlation matrix to avoid over-concentration
4. Regularly rebalance to maintain target weights
5. Test different scenarios with the simulation

---

**Need Help?** Check the full documentation in `CUSTOM_WEIGHTS_GUIDE.md`
