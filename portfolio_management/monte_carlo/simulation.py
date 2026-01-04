"""
Monte Carlo Simulation Module

This module performs Monte Carlo simulations to model portfolio performance over time.
It uses historical return distributions and covariance to generate thousands of possible
future portfolio paths, helping investors understand potential outcomes and risks.
"""

import numpy as np


class MonteCarloSimulation:
    """
    Monte Carlo Simulation Engine for Portfolio Analysis
    
    This class simulates multiple potential paths a portfolio could take by:
    1. Using historical mean returns and covariance from past data
    2. Generating random returns based on these statistics
    3. Compounding returns over a specified time horizon
    4. Tracking portfolio values across all simulations
    
    The key insight: With many simulations, we can estimate the probability of 
    different outcomes and measure risk (VaR, drawdowns, etc.)
    """
    
    def __init__(self, returns, initial_investment=1, weights=None):
        """
        Initialize the Monte Carlo Simulation with historical returns data.
        
        Args:
            returns (pd.DataFrame): Historical daily returns for each asset in the portfolio.
                                   Shape: (num_days, num_assets)
            initial_investment (float): Starting portfolio value in dollars. Default is 1.
            weights (list/array): Portfolio allocation weights for each asset.
                                 Must sum to 1. If None, uses equal weighting.
                                 Example: [0.4, 0.3, 0.3] for 3 assets
        
        Attributes:
            self.mean (np.array): Annualized expected return for each asset
                                 Calculated as: daily_mean * 252 trading days
            self.covariance (np.array): Annualized covariance matrix showing how assets
                                       move together: covariance * 252
            self.initial_investment (float): Starting portfolio value
            self.weights (np.array): Normalized portfolio weights
        """
        self.returns = returns
        
        # Annualize the mean and covariance using 252 trading days per year
        # 252 is the standard number of trading days in a US stock market year
        self.mean = returns.mean() * 252  # Convert daily returns to annual
        self.covariance = returns.cov() * 252  # Convert daily covariance to annual
        
        self.initial_investment = initial_investment
        
        num_assets = len(self.mean)
        
        # If no weights provided, use equal weighting (diversified baseline)
        if weights is None:
            self.weights = np.ones(num_assets) / num_assets
        else:
            self.weights = np.array(weights)

    def run_simulation(self, num_simulations, time_horizon):
        """
        Execute Monte Carlo simulation to generate portfolio paths.
        
        For each simulation:
        1. Generate random daily returns from a multivariate normal distribution
           (matching historical mean/covariance)
        2. Apply portfolio weights to blend asset returns into portfolio returns
        3. Compound returns over time using log returns for mathematical accuracy
        4. Track total portfolio value at each time step
        
        Args:
            num_simulations (int): Number of portfolio paths to simulate.
                                  More simulations = more accurate but slower.
                                  Typical: 10,000 to 100,000
            time_horizon (int): Number of trading days to simulate forward.
                               Typical: 252 (1 year), 1260 (5 years), etc.
        
        Returns:
            tuple: 
                - all_cumulative_returns (np.array): Shape (time_horizon, num_simulations)
                  Portfolio value at each time step for each simulation path
                - final_portfolio_values (np.array): Shape (num_simulations,)
                  Final portfolio value after time_horizon days for each simulation.
                  These values are used to calculate Value at Risk (VaR),
                  probability of loss, and expected returns.
        
        Mathematical Details:
        - Uses multivariate normal distribution to generate correlated returns
        - Log returns are used: ln(1 + daily_return)
        - Cumulative log returns compound via: portfolio_value = initial * exp(sum(log_returns))
        """
        # Initialize arrays to store results
        # Shape: (time_horizon, num_simulations) - one value per day per simulation
        all_cumulative_returns = np.zeros((time_horizon, num_simulations))
        # Shape: (num_simulations,) - final value for each simulation
        final_portfolio_values = np.zeros(num_simulations)

        # Convert annualized parameters to daily (divide by 252 trading days)
        # This allows us to generate realistic daily return movements
        daily_mean = self.mean / 252
        daily_cov = self.covariance / 252

        # Run each simulation independently
        for sim in range(num_simulations):
            # Generate random daily returns for all assets following historical distribution
            # multivariate_normal(mean, cov, num_days) returns shape (num_days, num_assets)
            # This captures both the expected return and volatility of each asset,
            # plus the correlation between assets (they tend to move together)
            simulated_returns = np.random.multivariate_normal(
                daily_mean, daily_cov, time_horizon
            )
            
            # Apply portfolio weights to combine individual asset returns into one portfolio return
            # Example: if weights = [0.4, 0.3, 0.3] and returns = [0.01, 0.02, -0.01],
            # portfolio_return = 0.4*0.01 + 0.3*0.02 + 0.3*(-0.01) = 0.007 or 0.7%
            portfolio_returns = simulated_returns.dot(self.weights)
            
            # Compound returns using exp(cumsum(log_returns))
            # This is mathematically more accurate than (1 + r1) * (1 + r2) * ...
            # Example: if returns are [0.01, 0.02], cumulative is [0.01, 0.0302...]
            # Then we convert back: portfolio_value = exp(cumulative) for growth factor
            cumulative_returns = np.exp(np.cumsum(portfolio_returns))
            
            # Scale by initial investment and store results
            # All values are indexed starting at initial_investment
            all_cumulative_returns[:, sim] = cumulative_returns * self.initial_investment
            final_portfolio_values[sim] = cumulative_returns[-1] * self.initial_investment
            
        return all_cumulative_returns, final_portfolio_values