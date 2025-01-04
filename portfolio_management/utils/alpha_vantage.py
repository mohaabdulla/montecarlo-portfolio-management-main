import requests
import pandas as pd
import os

def get_alpha_vantage_tickers():

    url = f'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey=VXS2UK4QL31BBHLO'
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            with open('tickers_alpha.csv', 'w') as file:
                file.write(response.text)
            print("Saved Alpha Vantage tickers to tickers_alpha.csv")
            
            try:
                df = pd.read_csv('tickers_alpha.csv')
                
                if 'symbol' not in df.columns:
                    print("Error: 'symbol' column not found in the CSV file. Available columns:", df.columns)
                    return []
                
                tickers = df['symbol'].dropna().tolist()
                print(f"Extracted {len(tickers)} tickers from Alpha Vantage")
                return tickers
            
            except pd.errors.EmptyDataError:
                print("Error: The CSV file 'tickers_alpha.csv' is empty or improperly formatted.")
                return []
            except Exception as e:
                print(f"Error while reading the CSV file: {e}")
                return []

        else:
            print(f"Failed to fetch tickers from Alpha Vantage. Status code: {response.status_code}")
            return []

    except requests.exceptions.RequestException as e:
        print(f"Error during API request: {e}")
        return []
