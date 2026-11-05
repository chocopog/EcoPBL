"""Data loading (Yahoo Finance + synthetic fallback) and feature engineering."""

import numpy as np
import pandas as pd
from config import SEED, get_stock_meta


def make_synthetic(ticker, start_date, n=2800):
    """Generate realistic simulated market data using geometric Brownian motion."""
    meta = get_stock_meta(ticker)
    rng = np.random.default_rng(SEED)
    idx = pd.bdate_range(start_date, periods=n)
    vol = meta.get("volatility", 0.018)
    base = meta.get("base_price", 100.0)
    r = rng.normal(0.0004, vol, n)
    price = base * np.cumprod(1 + r)
    volume = rng.lognormal(15, 0.3, n) * (1 + 50 * np.abs(r))
    return pd.DataFrame({"Close": price, "Volume": volume}, index=idx)


def load_data(ticker, start_date, end_date):
    """Download real market data from Yahoo Finance; fall back to synthetic if unavailable.

    Returns: (df, label_string, is_simulated_bool)
    """
    meta = get_stock_meta(ticker)
    label = f"{ticker} ({meta['name']})"
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start_date, end=end_date,
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Close", "Volume"]].dropna()
        if df.empty or len(df) < 100:
            raise ValueError("Insufficient data")
        return df, label, False
    except Exception:
        df = make_synthetic(ticker, start_date)
        return df, f"{ticker} (SIMULATED: {meta['name']})", True


def build_dataset(df):
    """Build economics-grounded features from raw price/volume data."""
    price = df["Close"]
    quantity = df["Volume"].clip(lower=1)

    out = pd.DataFrame(index=df.index)
    out["close"] = price
    out["price_change"] = price.pct_change()
    out["quantity_gap"] = quantity / quantity.rolling(10).mean() - 1
    out["price_gap"] = price / price.rolling(10).mean() - 1

    up_q = quantity.where(out["price_change"] > 0, 0.0)
    down_q = quantity.where(out["price_change"] < 0, 0.0)
    out["excess_demand"] = (
        (up_q.rolling(10).sum() - down_q.rolling(10).sum())
        / quantity.rolling(10).sum()
    )

    out["target"] = out["price_change"].shift(-1)
    out = out.dropna()
    next_dates = df.index[df.index.get_indexer(out.index) + 1]
    return out, next_dates

