"""
Portfolio Optimization Module

This module finds optimal portfolio allocations using Modern Portfolio Theory.
It helps answer: "What weights should I assign to each stock to maximize returns
for a given risk level (or minimize risk for a given return)?"

Key Concepts:
- Sharpe Ratio: Return per unit of risk (higher is better)
- Volatility: Standard deviation of returns (lower is better/less risky)
- Efficient Frontier: Set of portfolios with best return-to-risk tradeoff
"""

import numpy as np
from scipy.optimize import minimize


class PortfolioOptimizer:
    """
    Portfolio Optimization Engine using Modern Portfolio Theory
    
    This class optimizes portfolio weights using mathematical techniques:
    1. Maximize Sharpe Ratio: Best risk-adjusted return
    2. Minimize Volatility: Lowest risk for target return
    
    The optimizer uses constraints:
    - Weights must sum to 1 (fully invested)
    - Each weight between 0 and 1 (no shorting by default)
    """
    
    def __init__(self, expected_returns, covariance_matrix, risk_free_rate=0.0):
        """
        Initialize the optimizer with asset statistics.
        
        Args:
            expected_returns (np.array): Expected annual return for each asset.
                                        Shape: (num_assets,)
                                        Example: [0.08, 0.12, 0.10] for 3 stocks
                                        (8%, 12%, 10% expected annual returns)
            
            covariance_matrix (np.array): Annual covariance matrix showing how assets
                                         move together.
                                         Shape: (num_assets, num_assets)
                                         Diagonal: variance of each asset
                                         Off-diagonal: covariance between assets
                                         Higher covariance = more correlated (less diversification)
            
            risk_free_rate (float): Annual risk-free rate (e.g., 0.02 for 2% T-bills).
                                   Used in Sharpe ratio: (return - risk_free) / volatility
                                   Default: 0.0 (assume no risk-free rate)
        """
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.risk_free_rate = risk_free_rate

    def maximize_sharpe_ratio(self):
        """
        Find portfolio weights that maximize the Sharpe Ratio.
        
        Sharpe Ratio = (Portfolio Return - Risk Free Rate) / Portfolio Volatility
        
        This is the "best bang for your buck" - maximizes return per unit of risk taken.
        Often considered the optimal portfolio for most investors.
        
        Process:
        1. Set up optimization problem to minimize -Sharpe Ratio
           (we minimize the negative because scipy.optimize.minimize finds minimums)
        2. Apply constraints:
           - Weights sum to 1 (fully invested)
           - No short selling (0 <= weight <= 1)
        3. Use SLSQP algorithm (Sequential Least Squares Programming)
        
        Returns:
            np.array: Optimal weights that maximize Sharpe Ratio
                     Example: [0.40, 0.35, 0.25] for 3 assets
        """
        num_assets = len(self.expected_returns)
        args = (self.expected_returns, self.covariance_matrix, self.risk_free_rate)
        
        # Constraint: weights must sum to 1
        constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
        
        # Bounds: each weight between 0 and 1 (no shorting)
        bounds = tuple((0, 1) for _ in range(num_assets))

        # Minimize negative Sharpe ratio (equivalent to maximizing Sharpe ratio)
        result = minimize(
            self._neg_sharpe_ratio,
            x0=num_assets * [1.0 / num_assets],  # Start with equal weights
            args=args,
            method='SLSQP',  # Constrained optimization algorithm
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x

    def minimize_volatility(self, target_return):
        """
        Find portfolio weights that achieve target return with minimum volatility.
        
        This answers: "What's the lowest-risk way to achieve my desired return?"
        Useful for conservative investors with specific return targets.
        
        Process:
        1. Minimize portfolio volatility (standard deviation)
        2. Subject to constraints:
           - Weights sum to 1
           - Portfolio return equals target_return
           - No short selling
        
        Args:
            target_return (float): Desired annual return (e.g., 0.10 for 10%)
        
        Returns:
            np.array: Optimal weights achieving target return with min volatility
                     Example: [0.45, 0.30, 0.25] for 3 assets
        """
        num_assets = len(self.expected_returns)
        args = (self.covariance_matrix,)
        
        # Two constraints:
        # 1. Weights sum to 1 (fully invested)
        # 2. Portfolio return equals target (wp^T * expected_returns = target_return)
        constraints = (
            {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
            {'type': 'eq', 'fun': lambda weights: np.dot(weights, self.expected_returns) - target_return}
        )
        
        # Bounds: each weight between 0 and 1
        bounds = tuple((0, 1) for _ in range(num_assets))

        # Minimize volatility
        result = minimize(
            self._portfolio_volatility,
            x0=num_assets * [1.0 / num_assets],  # Start with equal weights
            args=args,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x

    @staticmethod
    def _neg_sharpe_ratio(weights, expected_returns, covariance_matrix, risk_free_rate):
        """
        Calculate negative Sharpe Ratio for optimization.
        
        We return the negative because scipy.optimize.minimize finds minimums,
        but we want to maximize Sharpe ratio.
        
        Sharpe Ratio = (Portfolio Return - Risk Free Rate) / Portfolio Volatility
        
        Where:
        - Portfolio Return = weights^T * expected_returns (weighted sum of returns)
        - Portfolio Volatility = sqrt(weights^T * Cov * weights) (portfolio std dev)
        """
        # Calculate portfolio expected return: weighted sum of individual returns
        portfolio_return = np.dot(weights, expected_returns)
        
        # Calculate portfolio volatility: sqrt(w^T * Cov * w)
        # This is the standard deviation of the portfolio's returns
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
        
        # Calculate Sharpe ratio
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        # Return negative (for minimization to work)
        return -sharpe_ratio

    @staticmethod
    def _portfolio_volatility(weights, covariance_matrix):
        """
        Calculate portfolio volatility (standard deviation of returns).
        
        Formula: volatility = sqrt(weights^T * Covariance * weights)
        
        This measures how much a portfolio's returns vary day-to-day.
        Higher volatility = more risk = wider swings in value
        
        Args:
            weights: Portfolio allocation weights
            covariance_matrix: Annual covariance of asset returns
        
        Returns:
            float: Annual volatility (standard deviation of annual returns)
        """
        return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
