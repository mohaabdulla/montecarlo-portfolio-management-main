"""
Data Loader Module

This module handles downloading and cleaning historical stock price data.
It uses yfinance to fetch real-time market data from Yahoo Finance.

Key responsibilities:
- Download daily closing prices for multiple stocks
- Handle data normalization (ticker formatting)
- Clean and validate data (remove missing values)
- Return a clean DataFrame ready for analysis
"""

import pandas as pd
import yfinance as yf
from typing import List, Dict, Union


class DataLoader:
    """
    Data Loader for fetching historical stock price data.
    
    This class downloads OHLC (Open, High, Low, Close) data from Yahoo Finance
    and prepares it for portfolio analysis by extracting adjusted closing prices.
    
    Attributes:
        Handles both single and multiple ticker downloads efficiently using yfinance.
    """
    
    def load_data(
        self,
        tickers: List[str],
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
    ) -> pd.DataFrame:
        """
        Download historical stock price data for multiple tickers.
        
        This method:
        1. Normalizes ticker symbols (e.g., BRK.B -> BRK-B)
        2. Batch downloads all tickers in one efficient call
        3. Extracts adjusted closing prices (accounts for stock splits, dividends)
        4. Cleans missing data
        5. Returns aligned price data ready for analysis
        
        Args:
            tickers (List[str]): List of stock ticker symbols
                                Example: ['AAPL', 'MSFT', 'GOOGL', 'BRK.B']
                                Note: '.' is automatically converted to '-' for yfinance
            
            start_date (str or pd.Timestamp): Start date for historical data
                                             Format: 'YYYY-MM-DD'
                                             Example: '2019-01-01'
            
            end_date (str or pd.Timestamp): End date for historical data
                                           Format: 'YYYY-MM-DD'
                                           Example: '2024-01-01'
        
        Returns:
            pd.DataFrame: Clean price data indexed by date
                         Index: trading dates (datetime)
                         Columns: ticker symbols (original user input)
                         Values: adjusted closing prices (float)
                Example:
                            AAPL      MSFT      GOOGL
                2019-01-02  154.89    104.52    1050.32
                2019-01-03  153.67    104.37    1041.89
                ...
                2024-01-01  189.95    371.05    2752.13
        
        Notes:
            - Returns empty DataFrame if no data is available or tickers are empty
            - Automatically handles both single and multi-ticker downloads
            - Uses adjusted close prices (accounts for splits/dividends)
            - Falls back to regular close if adjusted close unavailable
            - Removes tickers with insufficient or missing data
        """
        # Return empty DataFrame if no tickers provided
        if not tickers:
            return pd.DataFrame()

        # Normalize ticker symbols for yfinance compatibility
        # yfinance prefers hyphens: BRK.B -> BRK-B
        norm = [t.replace(".", "-").strip().upper() for t in tickers]

        # Batch download all tickers at once (more efficient than individual downloads)
        # auto_adjust=True: automatically adjusts prices for splits/dividends in 'Close' column
        # group_by='ticker': organizes output by ticker (useful for multi-ticker downloads)
        # threads=True: uses multi-threading for faster downloads
        df = yf.download(
            " ".join(norm),
            start=start_date,
            end=end_date,
            progress=False,  # Don't show download progress
            auto_adjust=True,  # Gives you adjusted prices directly in 'Close'
            group_by="ticker",
            threads=True,
        )

        # Handle empty or None result
        if df is None or df.empty:
            return pd.DataFrame()

        # Build a clean price matrix
        out: Dict[str, pd.Series] = {}

        # Case 1: MultiIndex columns (occurs with multiple tickers)
        # yfinance returns shape (dates, num_tickers * num_columns)
        if isinstance(df.columns, pd.MultiIndex):
            for t in norm:
                cols = df.get(t)  # Get columns for this ticker
                if cols is not None and not cols.empty:
                    # Prefer Adj Close if present (accounts for splits/dividends),
                    # otherwise use Close
                    if "Adj Close" in cols.columns:
                        s = cols["Adj Close"].rename(t)
                    elif "Close" in cols.columns:
                        s = cols["Close"].rename(t)
                    else:
                        continue  # Skip if neither column exists
                    
                    # Only include if we have valid data (not all NaN)
                    if not s.dropna().empty:
                        out[t] = s
        else:
            # Case 2: Single ticker (flat columns, not MultiIndex)
            if "Adj Close" in df.columns:
                out[norm[0]] = df["Adj Close"].rename(norm[0])
            elif "Close" in df.columns:
                out[norm[0]] = df["Close"].rename(norm[0])

        # Combine all ticker series into one DataFrame
        prices = pd.DataFrame(out)
        
        # Map normalized tickers back to original user input
        # (preserves original casing/formatting)
        rename_map = {n: o for n, o in zip(norm, tickers)}
        prices = prices.rename(columns=rename_map)

        # Remove columns that are completely missing (ticker failed to download)
        prices = prices.dropna(axis=1, how="all")

        return prices
