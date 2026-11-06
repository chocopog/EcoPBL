"""Configuration: Stock catalog and default hyperparameters."""

AVAILABLE_STOCKS = {
    "AAPL":        {"name": "Apple Inc.",                "currency": "$", "base_price": 180.0, "volatility": 0.015},
    "META":        {"name": "Meta Platforms Inc.",       "currency": "$", "base_price": 300.0, "volatility": 0.022},
    "NVDA":        {"name": "NVIDIA Corporation",       "currency": "$", "base_price": 120.0, "volatility": 0.028},
    "MSFT":        {"name": "Microsoft Corporation",    "currency": "$", "base_price": 350.0, "volatility": 0.014},
    "AMZN":        {"name": "Amazon.com Inc.",          "currency": "$", "base_price": 150.0, "volatility": 0.018},
    "GOOGL":       {"name": "Alphabet Inc. (Google)",   "currency": "$", "base_price": 170.0, "volatility": 0.016},
    "TSLA":        {"name": "Tesla Inc.",               "currency": "$", "base_price": 220.0, "volatility": 0.032},
    "RELIANCE.NS": {"name": "Reliance Industries Ltd",  "currency": "₹", "base_price": 2500.0, "volatility": 0.012},
    "TCS.NS":      {"name": "Tata Consultancy Services","currency": "₹", "base_price": 3500.0, "volatility": 0.011},
    "INFY.NS":     {"name": "Infosys Ltd",              "currency": "₹", "base_price": 1500.0, "volatility": 0.013},
    "^NSEI":       {"name": "NIFTY 50 Index",           "currency": "₹", "base_price": 20000.0,"volatility": 0.009},
    "^GSPC":       {"name": "S&P 500 Index",            "currency": "$", "base_price": 5000.0, "volatility": 0.009},
}

# Model architecture defaults
LOOKBACK_WINDOW = 60
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
PATIENCE = 10
SEED = 42

# Data split
TEST_FRAC = 0.2
VAL_FRAC = 0.15

# Trading economics
TC_RATE = 0.001
SCALE_W = 100_000
RF_ANNUAL = {"^NSEI": 0.065, "^GSPC": 0.04}

# Feature columns
FEATURES = [
    "price_change", "quantity_gap", "price_gap", "excess_demand",
    "momentum_5", "momentum_10", "momentum_20", "momentum_50",
    "volatility_5", "volatility_10", "volatility_20", "volatility_50",
    "rsi_14", "volume_momentum", "drawdown_50", "ma_slope_20",
    "market_return", "relative_strength_20", "market_trend",
    "market_volatility",
]


def get_stock_meta(ticker):
    """Get metadata for a ticker symbol."""
    if ticker in AVAILABLE_STOCKS:
        return AVAILABLE_STOCKS[ticker]
    curr = "₹" if ticker.endswith((".NS", ".BO")) else "$"
    return {"name": ticker, "currency": curr, "base_price": 100.0, "volatility": 0.018}
