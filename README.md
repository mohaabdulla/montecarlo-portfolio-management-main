# Monte Carlo Portfolio Optimization

## 📘 **Project Overview**
This project leverages Monte Carlo simulations to optimize investment portfolio allocation. The system enables users to determine optimal sector weights, expected returns, and Value at Risk (VaR) for each investment. The project includes a user-friendly Graphical User Interface (GUI) to simplify data input, analysis, and decision-making.

## 🚀 **Key Features**
- **Monte Carlo Simulation**: Uses random sampling to generate thousands of portfolio weight combinations for optimal allocation.
- **Expected Return & Probability of Loss**: Calculate and display expected returns (annualized and period-based) and probability of losing money.
- **Automated Portfolio Analysis**: Automatically calculates key metrics like sector weights, expected returns, and VaR for each sector and ticker.
- **User-Friendly GUI**: Provides a visual interface to input, update, and analyze portfolio data.
- **Custom Portfolio Weights**: Users can manually specify their desired portfolio allocation weights for each asset.
- **Multiple Portfolio Strategies**: Choose from Max Sharpe, Min Variance, Equal Weight, Max Return, Balanced, or Custom Weights.
- **Data Extraction and Analysis**: Extracts up to 10 years of historical data to analyze past performance.
- **Customizable Inputs**: Users can input custom tickers, set risk tolerance, and define target returns.

## 📂 **Project Structure**
```
MonteCarlo/
├── .git/                   # Git configuration files for version control
├── portfolio_management/   # Main package for portfolio management logic
│   ├── data/               # Data processing and storage modules
│   ├── monte_carlo/       # Monte Carlo simulation logic
│   ├── portfolio/         # Portfolio allocation logic
│   ├── utils/             # Helper utilities and shared functions
│   ├── __init__.py        # Package initialization file
│   └── main.py            # Main entry point for the system
├── tests/                 # Unit and integration tests to ensure system reliability
├── app.py                 # Main application script to launch the GUI
├── config.py              # Configuration file to set system parameters
├── requirements.txt      # List of required Python dependencies
└── setup.py               # Installation script for the project
```

## 🛠️ **Installation Instructions**

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/username/MonteCarlo.git
cd MonteCarlo
```

### **2️⃣ Set up a Virtual Environment**
(Optional but recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4️⃣ Run the Application**
```bash
python app.py
```

## 💡 **How to Use**

### **1️⃣ Launch the GUI**
Start the application by running the following command:
```bash
python app.py
```
Or launch the Streamlit web interface:
```bash
streamlit run app.py
```

### **2️⃣ Input Portfolio Data**
- Enter stock tickers (comma-separated) or use the quick-add dropdown
- Select the date range for historical data analysis
- Set the risk-free rate, initial investment amount, and time horizon
- Choose whether to allow short selling

### **3️⃣ Select Portfolio Strategy**
Choose from multiple portfolio strategies:
- **Max Sharpe Portfolio**: Maximizes the Sharpe ratio (risk-adjusted returns)
- **Minimum Variance Portfolio**: Minimizes portfolio volatility
- **Equal Weight Portfolio**: Distributes investment equally across all assets
- **Maximum Return Portfolio**: Invests 100% in the highest-return asset
- **Balanced Portfolio**: Targets a balanced risk-return profile
- **Custom Weights**: Specify your own allocation percentages for each asset

### **4️⃣ Using Custom Weights**
To specify custom portfolio weights:
1. Click the **"Custom Weights"** button
2. Enter the desired weight for each asset (e.g., 0.40 for 40%)
3. Weights will be automatically normalized to sum to 1.0
4. Click **"Run Custom Portfolio"** to execute the simulation

Alternatively, for command-line usage, edit `config.py`:
```python
# Set custom weights (must match the order of TICKERS)
TICKERS = ['AAPL', 'MSFT', 'GOOG']
WEIGHTS = [0.5, 0.3, 0.2]  # 50% AAPL, 30% MSFT, 20% GOOG
OPTIMIZE = False  # Disable optimization to use custom weights
```

### **5️⃣ Run Monte Carlo Simulation**
- After selecting a strategy, the system will automatically run the simulation
- View cumulative returns over time and the distribution of final portfolio values

### **6️⃣ Review Results**
- View the recommended portfolio allocation and associated risk metrics
- Analyze expected returns, volatility, and Sharpe ratio
- Examine Value at Risk (VaR) and expected portfolio gains
- Download results as CSV files for further analysis

## ⚙️ **Configuration**
Modify `config.py` to customize system parameters, including:
- **Tickers**: Set the list of stock symbols to analyze (e.g., `['AAPL', 'MSFT', 'GOOG']`)
- **Custom Weights**: Specify portfolio allocation weights (e.g., `[0.4, 0.3, 0.3]`)
- **Date Range**: Set `START_DATE` and `END_DATE` for historical data
- **Simulation Parameters**: Adjust `NUM_SIMULATIONS`, `TIME_HORIZON`, and `RISK_FREE_RATE`
- **Optimization Settings**: Toggle `OPTIMIZE` and `BALANCED` flags to control portfolio optimization

Example configuration for custom weights:
```python
TICKERS = ['AAPL', 'MSFT', 'GOOG', 'AMZN']
WEIGHTS = [0.40, 0.30, 0.20, 0.10]  # Custom allocation: 40%, 30%, 20%, 10%
OPTIMIZE = False  # Set to False to use custom weights
INITIAL_INVESTMENT = 10000
NUM_SIMULATIONS = 1000000
TIME_HORIZON = 252  # One year of trading days
```

## 📑 **Key Files Explained**

- **`app.py`**: Main entry point to run the GUI for user interaction.
- **`portfolio_management/monte_carlo/`**: Contains core logic for Monte Carlo simulations.
- **`portfolio_management/portfolio/`**: Handles portfolio allocation, weight calculations, and optimization.
- **`portfolio_management/utils/`**: Provides utility functions, including logging, file I/O, and data validation.

## 🔍 **Testing Instructions**

### **1️⃣ Run Unit Tests**
```bash
pytest tests/
```

### **2️⃣ Check Coverage**
```bash
pytest --cov=portfolio_management tests/
```

### **3️⃣ View Test Results**
Check for any test failures, errors, or skipped tests in the output.

## 📈 **Future Enhancements**
- **Advanced Risk Analysis**: Add support for Conditional VaR (CVaR) to enhance risk assessments.
- **Interactive Visualizations**: Include charts and graphs to visualize portfolio performance and risk analysis.
- **Real-Time Data Updates**: Integrate live market data to provide up-to-date financial insights.
