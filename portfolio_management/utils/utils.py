
import pandas as pd

def merge_tickers(tickers_list):
    merged_tickers = list(set().union(*tickers_list))
    print(f"Total unique tickers: {len(merged_tickers)}")
    return merged_tickers

def save_tickers_to_csv(tickers, filename='tickers.csv'):
    df = pd.DataFrame(tickers, columns=['ticker'])
    df.to_csv(filename, index=False)
    print(f"Saved final tickers to {filename}")
