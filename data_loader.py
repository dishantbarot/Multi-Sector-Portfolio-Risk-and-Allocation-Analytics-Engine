# ===============================
# DATA LOADER MODULE
# ===============================
# Purpose:
# 1. Download historical stock price data
# 2. Clean the data
# 3. Convert prices into log returns
# 4. Store predefined sector stock lists

import yfinance as yf
import numpy as np
import pandas as pd


# ====================================
# PREDEFINED SECTOR STOCK COLLECTIONS
# ====================================
# These are grouped stock lists for each sector.
# User can select these instead of entering custom tickers.

PREDEFINED_INDICES = {
    "NIFTY BANK": [...],   # (kept same as your code)
    "NIFTY AUTO": [...],
    "NIFTY IT": [...],
    "NIFTY FIN SERVICE": [...],
    "NIFTY PHARMA": [...],
    "NIFTY FMCG": [...],
    "NIFTY METAL": [...],
    "NIFTY ENERGY": [...]
}


# ====================================
# FUNCTION 1: LOAD PRICE DATA
# ====================================
def load_data(tickers, start="2015-01-01", end=None):
    """
    Safely downloads price data.
    Skips invalid tickers automatically.
    """

    all_data = []

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end)

            if df.empty:
                continue

            df = df[["Close"]]
            df.columns = [ticker]

            all_data.append(df)

        except Exception:
            continue

    if not all_data:
        raise ValueError("No valid tickers found. Please check symbols.")

    # Merge all valid data
    data = pd.concat(all_data, axis=1)

    # Clean data
    data = data.dropna(how="all")
    data = data.ffill().dropna()

    return data


# ====================================
# FUNCTION 2: COMPUTE LOG RETURNS
# ====================================
def compute_log_returns(price_df):
    """
    Calculates daily log returns.
    Formula:
        log_return = ln(P_t / P_t-1)
    """

    log_returns = np.log(price_df / price_df.shift(1))

    # First row becomes NaN due to shift → remove it
    return log_returns.dropna()


# ====================================
# FUNCTION 3: FORMAT TICKERS
# ====================================
def format_tickers(tickers, exchange="NS"):
    """
    Adds exchange suffix (.NS for NSE)
    if not already present.
    """

    formatted = []

    for t in tickers:
        t = t.upper().strip()

        # Add .NS if missing
        if "." not in t:
            t = f"{t}.{exchange}"

        formatted.append(t)

    return formatted
