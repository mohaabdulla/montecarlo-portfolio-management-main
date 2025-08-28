import pandas as pd
import yfinance as yf
from typing import List, Dict, Union

class DataLoader:
    def load_data(
        self,
        tickers: List[str],
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
    ) -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()

        # yfinance likes BRK.B as BRK-B, etc.
        norm = [t.replace(".", "-").strip().upper() for t in tickers]

        # Batch download is faster & more reliable
        df = yf.download(
            " ".join(norm),
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,       # gives you adjusted prices directly
            group_by="ticker",
            threads=True,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # Build a tidy price matrix
        out: Dict[str, pd.Series] = {}

        # MultiIndex (per-ticker columns)
        if isinstance(df.columns, pd.MultiIndex):
            for t in norm:
                cols = df.get(t)
                if cols is not None and not cols.empty:
                    # Prefer Adj Close if present, else Close
                    if "Adj Close" in cols.columns:
                        s = cols["Adj Close"].rename(t)
                    elif "Close" in cols.columns:
                        s = cols["Close"].rename(t)
                    else:
                        continue
                    if not s.dropna().empty:
                        out[t] = s
        else:
            # Single ticker case
            if "Adj Close" in df.columns:
                out[norm[0]] = df["Adj Close"].rename(norm[0])
            elif "Close" in df.columns:
                out[norm[0]] = df["Close"].rename(norm[0])

        prices = pd.DataFrame(out)
        # Map back to original user tickers if you want:
        rename_map = {n: o for n, o in zip(norm, tickers)}
        prices = prices.rename(columns=rename_map)

        # Drop columns that are all-NaN
        prices = prices.dropna(axis=1, how="all")

        return prices
