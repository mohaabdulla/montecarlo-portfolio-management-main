import ftplib
import pandas as pd

def get_nasdaq_nyse_tickers():
    ftp_server = 'ftp.nasdaqtrader.com'
    nasdaq_file = '/symboldirectory/nasdaqlisted.txt'
    nyse_file = '/symboldirectory/otherlisted.txt'

    # Connect to FTP server
    ftp = ftplib.FTP(ftp_server)
    ftp.login()

    # Download NASDAQ tickers
    with open('nasdaqlisted.txt', 'wb') as f:
        ftp.retrbinary(f"RETR {nasdaq_file}", f.write)
    print("Downloaded nasdaqlisted.txt")

    # Download NYSE tickers
    with open('otherlisted.txt', 'wb') as f:
        ftp.retrbinary(f"RETR {nyse_file}", f.write)
    print("Downloaded otherlisted.txt")

    ftp.quit()

    # Parse NASDAQ tickers
    nasdaq_tickers = pd.read_csv('nasdaqlisted.txt', sep='|')
    nasdaq_tickers = nasdaq_tickers['Symbol'].dropna().tolist()

    # Parse NYSE tickers
    nyse_tickers = pd.read_csv('otherlisted.txt', sep='|')
    nyse_tickers = nyse_tickers['ACT Symbol'].dropna().tolist()

    # Combine all tickers
    all_tickers = list(set(nasdaq_tickers + nyse_tickers))

    # Save the final combined ticker list
    df = pd.DataFrame(all_tickers, columns=['ticker'])
    df.to_csv('tickers_nasdaq_nyse.csv', index=False)

    print(f"Saved NASDAQ & NYSE tickers to tickers_nasdaq_nyse.csv")
    print(f"Extracted {len(all_tickers)} tickers from NASDAQ & NYSE")
    return all_tickers
