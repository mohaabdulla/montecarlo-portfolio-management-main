"""
Portfolio Module

This module manages portfolio data and calculations. It handles:
- Loading and storing historical price data for multiple assets
- Calculating daily returns (percentage changes in price)
- Computing portfolio statistics (mean returns, covariance)
"""

import pandas as pd


class Portfolio:
    """
    Portfolio class to manage asset prices and calculate returns.
    
    A portfolio is a collection of assets (stocks) with historical price data.
    This class processes that data to calculate returns, which are essential
    for Monte Carlo simulations and portfolio optimization.
    """
    
    def __init__(self, price_data):
        """
        Initialize a Portfolio with historical price data.
        
        Args:
            price_data (pd.DataFrame): Daily closing prices for assets.
                                      Index: dates (datetime)
                                      Columns: asset tickers (strings)
                                      Values: closing prices (floats)
                Example:
                        AAPL    MSFT    GOOGL
                2020-01-01  80.0   160.0   1400.0
                2020-01-02  81.5   161.3   1410.5
                ...
        """
        self.price_data = price_data
        self.returns = None  # Will be calculated in calculate_returns()

    def calculate_returns(self):
        """
        Convert prices to percentage daily returns.
        
        This method calculates what investors care about: how much did each stock
        grow/shrink each day, as a percentage?
        
        Formula: daily_return = (price_today - price_yesterday) / price_yesterday
        
        Why this matters:
        - Raw prices are hard to compare (a $10 stock and $1000 stock move differently)
        - Returns normalize this: both show percentage changes
        - Monte Carlo simulations work on returns, not prices
        - Correlations between assets are measured via returns
        
        The .dropna() removes the first row (no previous price to calculate return from)
        
        Example output:
                    AAPL      MSFT     GOOGL
        2020-01-02  0.0188   0.0081   0.0075    (1.88%, 0.81%, 0.75% gains)
        2020-01-03  -0.0045  0.0120   -0.0020   (-0.45%, 1.20%, -0.20%)
        ...
        """
        self.returns = self.price_data.pct_change().dropna()
