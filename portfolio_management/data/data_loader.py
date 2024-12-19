import pandas as pd
import yfinance as yf
from typing import List, Dict, Union

class DataLoader:
    def load_data(self, tickers: List[str], start_date: Union[str, pd.Timestamp], end_date: Union[str, pd.Timestamp]) -> pd.DataFrame:
        stock_data: Dict[str, pd.Series] = {}
        for ticker in tickers:
            try:
                # Download stock data from Yahoo Finance
                data: pd.DataFrame = yf.download(ticker, start=start_date, end=end_date, progress=False)
                print(f"Downloaded data for {ticker}: {data.head()}")  # Debug print
                if not data.empty:
                    stock_data[ticker] = data['Adj Close'].tolist()  # Store the adjusted close prices as lists
                else:
                    print(f"No data found for {ticker}")
            except Exception as e:
                print(f"Error loading data for {ticker}: {e}")

        if stock_data:
            return pd.DataFrame(stock_data)
        else:
            return pd.DataFrame()  # Return an empty DataFrame if no data is found

    
    def get_sector_data(self, tickers: List[str]) -> Dict[str, str]:
        """Fetches sector data for a list of stock tickers using Yahoo Finance."""
        sector_data = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                sector_data[ticker] = stock.info.get('sector', 'Unknown')
            except Exception as e:
                print(f"Error fetching sector data for {ticker}: {e}")
                sector_data[ticker] = 'Unknown'
        return sector_data

if __name__ == "__main__":
    # Initialize DataLoader
    data_loader = DataLoader()

    # Sample tickers and date range
    tickers = ['AAPL', 'MSFT', 'GOOG']
    start_date = '2016-01-01'
    end_date = '2023-01-01'

    # Fetch the stock data
    stock_data = data_loader.load_data(tickers, start_date, end_date)

    # Print the first few rows of the stock data
    print(stock_data.head())
