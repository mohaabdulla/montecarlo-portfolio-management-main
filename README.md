# Monte Carlo Portfolio Optimization

## 📘 **Project Overview**
This project leverages Monte Carlo simulations to optimize investment portfolio allocation. The system enables users to determine optimal sector weights, expected returns, and Value at Risk (VaR) for each investment. The project includes a user-friendly Graphical User Interface (GUI) to simplify data input, analysis, and decision-making.

## 🚀 **Key Features**
- **Monte Carlo Simulation**: Uses random sampling to generate thousands of portfolio weight combinations for optimal allocation.
- **Automated Portfolio Analysis**: Automatically calculates key metrics like sector weights, expected returns, and VaR for each sector and ticker.
- **User-Friendly GUI**: Provides a visual interface to input, update, and analyze portfolio data.
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

### **2️⃣ Input Portfolio Data**
- Add custom tickers, expected returns, and risk tolerance.
- Set the total target return for the portfolio (e.g., 20% to 25%).

### **3️⃣ Run Monte Carlo Simulation**
- Click the **Run Simulation** button in the GUI.
- The system will generate optimal weights, expected returns, and VaR for each sector.

### **4️⃣ Review Results**
- View the recommended portfolio allocation and associated risk metrics.

## ⚙️ **Configuration**
Modify `config.py` to customize system parameters, including:
- **Data Sources**: Set API keys and data sources for market data.
- **Simulation Parameters**: Adjust the number of iterations, constraints, or target return range.

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
