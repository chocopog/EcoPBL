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
        if df.empty or len(df) < 80:
            raise ValueError("Insufficient data")
        return df, label, False
    except Exception:
        df = make_synthetic(ticker, start_date)
        return df, f"{ticker} (SIMULATED: {meta['name']})", True


def build_dataset(df, benchmark=None, horizon=5, tc_rate=0.001):
    """Build causal features, classification targets, and regression targets.

    Works robustly whether a market benchmark is supplied or not.
    """
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

    for window in (5, 10, 20, 50):
        out[f"momentum_{window}"] = price / price.shift(window) - 1
        out[f"volatility_{window}"] = out["price_change"].rolling(window).std()

    gain = out["price_change"].clip(lower=0).rolling(14).mean()
    loss = out["price_change"].clip(upper=0).abs().rolling(14).mean().replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + (gain / loss).fillna(1.0)))
    out["volume_momentum"] = quantity / quantity.rolling(20).mean() - 1
    out["drawdown_50"] = price / price.rolling(50).max() - 1
    out["ma_slope_20"] = price.rolling(20).mean().pct_change(5)

    # Robust benchmark handling
    if benchmark is None or (isinstance(benchmark, (pd.Series, pd.DataFrame)) and (benchmark.dropna().empty or (benchmark == 0).all())):
        # If no benchmark is provided, use a 50-day smoothed moving average of the asset as the market baseline
        benchmark = price.rolling(50, min_periods=1).mean()

    benchmark_series = benchmark.reindex(df.index).ffill().bfill()
    benchmark_return = benchmark_series.pct_change().fillna(0.0)
    out["market_return"] = benchmark_return
    out["relative_strength_20"] = (out["momentum_20"] - benchmark_return.rolling(20).sum()).fillna(0.0)
    b_ma50 = benchmark_series.rolling(50, min_periods=1).mean()
    out["market_trend"] = (benchmark_series / b_ma50.replace(0, np.nan) - 1).fillna(0.0)
    out["market_volatility"] = benchmark_return.rolling(20, min_periods=1).std().fillna(0.0)

    # Dynamic market regime classification
    vol_median = out["market_volatility"].rolling(100, min_periods=20).median().fillna(0.01)
    out["regime"] = np.select(
        [
            out["market_trend"] < -0.02,
            out["market_volatility"] > vol_median * 1.25,
            out["market_trend"] > 0.01,
        ],
        ["defensive", "high_volatility", "trend"],
        default="neutral",
    )

    # Forward horizon returns and excess return target
    out["forward_return"] = price.shift(-horizon) / price - 1
    out["market_forward_return"] = benchmark_series.shift(-horizon) / benchmark_series - 1
    out["excess_forward_return"] = out["forward_return"] - out["market_forward_return"]
    out["target"] = (out["excess_forward_return"] > 2 * tc_rate).astype(float)

    # 1-day percentage change target for single-day regression
    out["target_return"] = out["price_change"].shift(-1)

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    target_dates = df.index[df.index.get_indexer(out.index) + horizon]
    return out, target_dates
