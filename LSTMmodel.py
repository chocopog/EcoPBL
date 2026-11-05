"""
Stock predictor using an LSTM: Economics PBL (basic-economics version)
=======================================================================

Setup:   pip install yfinance torch numpy pandas scikit-learn matplotlib
Run:     python LSTMmodel.py                     (uses settings configured below)
         python LSTMmodel.py --ticker NVDA --epochs 50   (or pass via CLI)

HOW THE ECONOMICS COURSE CONNECTS TO THIS PROJECT
-------------------------------------------------
1) DEMAND AND SUPPLY
   The price of a share is set where buyers (demand) meet sellers (supply).
   - Law of demand / law of supply: price and quantity move together as the curves say.
   - Equilibrium: the price where quantity demanded = quantity supplied.
   - Shortage (demand > supply at the current price)  -> price is pushed UP.
   - Surplus  (supply > demand at the current price)  -> price is pushed DOWN.
   - A shift of the demand curve (news, mood, income) moves the equilibrium price.
   - Elasticity, consumer surplus and producer surplus are shown in the illustration plot.
   In the code we MEASURE this from market data:
   - quantity traded (volume)                        -> "quantity"
   - volume on up-days vs volume on down-days        -> rough proxy for buying vs selling pressure
   - excess demand = (up-day volume - down-day volume) / total volume over 10 days
   - price_gap = price vs its recent average         -> how far the price is from its "equilibrium" level
   NOTE: these are simple PROXIES. A stock exchange does not publish the real demand and supply curves.

2) PRODUCTION COST AND PROFIT (applied to the trading strategy as if it were a small firm)
   - Every trade has a cost (brokerage, taxes). We assume a constant cost per trade, so:
        Total Cost (TC) = cost per trade x number of trades
        Marginal Cost (MC) = Average Cost (AC) = cost of one more trade   (fixed cost assumed zero)
   - Total Revenue (TR) = gross money gained from holding the share on the days the model said "buy".
   - Profit = TR - TC.   A trade is worth making only if its extra revenue (marginal revenue) > MC.

3) UTILITY
   - An investor does not value wealth linearly. We use the utility function U(W) = sqrt(W / SCALE).
   - Total utility (TU) rises with wealth, but marginal utility (MU = dU/dW) FALLS:
     diminishing marginal utility. So losing an amount hurts more than gaining the same amount helps.

MACHINE-LEARNING TERMS (not economics, just so the code is easy to read)
   Training data = what the model learns from.  Validation data = used to stop training at the right time.
   Test data = unseen future days used to judge the model.  Overfitting = memorising the past instead of learning.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==============================================================================
# 🎯 STOCK CATALOG: POPULAR COMPANIES & ASSETS
# ==============================================================================
AVAILABLE_STOCKS = {
    "AAPL":        {"name": "Apple Inc.",                "currency": "$", "base_price": 180.0, "volatility": 0.015},
    "META":        {"name": "Meta Platforms Inc.",       "currency": "$", "base_price": 300.0, "volatility": 0.022},
    "NVDA":        {"name": "NVIDIA Corporation",        "currency": "$", "base_price": 120.0, "volatility": 0.028},
    "MSFT":        {"name": "Microsoft Corporation",     "currency": "$", "base_price": 350.0, "volatility": 0.014},
    "AMZN":        {"name": "Amazon.com Inc.",           "currency": "$", "base_price": 150.0, "volatility": 0.018},
    "GOOGL":       {"name": "Alphabet Inc. (Google)",    "currency": "$", "base_price": 170.0, "volatility": 0.016},
    "TSLA":        {"name": "Tesla Inc.",                "currency": "$", "base_price": 220.0, "volatility": 0.032},
    "RELIANCE.NS": {"name": "Reliance Industries Ltd",   "currency": "₹", "base_price": 2500.0, "volatility": 0.012},
    "TCS.NS":      {"name": "Tata Consultancy Services", "currency": "₹", "base_price": 3500.0, "volatility": 0.011},
    "INFY.NS":     {"name": "Infosys Ltd",               "currency": "₹", "base_price": 1500.0, "volatility": 0.013},
    "^NSEI":       {"name": "NIFTY 50 Index",            "currency": "₹", "base_price": 20000.0, "volatility": 0.009},
    "^GSPC":       {"name": "S&P 500 Index",             "currency": "$", "base_price": 5000.0, "volatility": 0.009},
}

# ==============================================================================
# ⚙️ CONFIGURATION: EDIT YOUR PREFERENCES HERE
# ==============================================================================
# 1. STOCK PICKER: Choose any ticker above or enter your own Yahoo Finance symbol
TICKER = "AAPL"               # e.g., "AAPL", "META", "NVDA", "MSFT", "TSLA", "RELIANCE.NS"

# 2. TRAINING EPOCHS: Number of training passes through the dataset
EPOCHS = 50                   # Recommended: 30 - 100 (early stopping guards against overfitting)

# 3. DATE RANGE & ARCHITECTURE
START, END = "2015-01-01", "2026-10-31"
WINDOW = 60                   # Days the LSTM looks back to forecast tomorrow
TEST_FRAC = 0.2               # Last 20% of data reserved for test period (unseen)
VAL_FRAC = 0.15               # Last 15% of training data used for validation early stopping
HIDDEN = 64                   # Number of LSTM hidden units
LAYERS = 2                    # Number of LSTM layers
DROPOUT = 0.2                 # Dropout rate
BATCH = 64                    # Mini-batch size
LR = 1e-3                     # Learning rate (Adam optimizer)
PATIENCE = 10                 # Early stopping patience
SEED = 42

# 4. TRADING & ECONOMICS SETTINGS
CAPITAL = 100_000             # Initial capital invested in currency units
TC_RATE = 0.001               # Cost per trade = 0.1% of capital (10 basis points)
SCALE_W = 100_000             # Scale factor for wealth utility calculation U(W) = sqrt(W / SCALE_W)
OUT_DIR = "eco_plots"
SHOW = os.environ.get("SHOW_PLOTS", "1") == "1"

# The four features the LSTM examines (grounded in Demand & Supply proxies)
FEATURES = ["price_change", "quantity_gap", "price_gap", "excess_demand"]

# Initialize device and seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass


def get_stock_meta(ticker):
    """Retrieve metadata (name, currency, base price) for a ticker."""
    if ticker in AVAILABLE_STOCKS:
        return AVAILABLE_STOCKS[ticker]
    curr = "₹" if ticker.endswith((".NS", ".BO")) else "$"
    return {"name": ticker, "currency": curr, "base_price": 100.0, "volatility": 0.018}


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved {path}")


# ============================== DATA LOADING ==============================
def make_synthetic(ticker, n=2800):
    """Fallback if Yahoo Finance is unreachable: realistic simulated price and volume."""
    meta = get_stock_meta(ticker)
    rng = np.random.default_rng(SEED)
    idx = pd.bdate_range("2015-01-01", periods=n)
    vol = meta.get("volatility", 0.015)
    base = meta.get("base_price", 100.0)
    r = rng.normal(0.0004, vol, n)
    price = base * np.cumprod(1 + r)
    volume = rng.lognormal(15, 0.3, n) * (1 + 50 * np.abs(r))
    return pd.DataFrame({"Close": price, "Volume": volume}, index=idx)


def load_raw(ticker):
    """Attempt download from Yahoo Finance, otherwise generate realistic synthetic series."""
    meta = get_stock_meta(ticker)
    display_label = f"{ticker} ({meta['name']})"
    try:
        import yfinance as yf
        print(f"[*] Downloading market data for {display_label} from Yahoo Finance...")
        df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Close", "Volume"]].dropna()
        if df.empty:
            raise ValueError("empty dataframe returned")
        return df, display_label
    except Exception as e:
        print(f"[!] Could not download {ticker} ({e}). Using simulated market data instead.")
        return make_synthetic(ticker), f"{ticker} (SIMULATED: {meta['name']})"


def build_dataset(df):
    """Turn price and volume into demand/supply-style features, one row per day."""
    price = df["Close"]
    quantity = df["Volume"].clip(lower=1)                     # quantity of shares traded

    out = pd.DataFrame(index=df.index)
    out["close"] = price
    out["price_change"] = price.pct_change()                  # % change in price today

    # Quantity traded today compared with a normal day (last 10 days average)
    out["quantity_gap"] = quantity / quantity.rolling(10).mean() - 1

    # Price compared with its recent average (the "equilibrium level" it tends to return to)
    out["price_gap"] = price / price.rolling(10).mean() - 1

    # Buying vs selling pressure: volume on up-days is treated as "demand",
    # volume on down-days as "supply". Excess demand > 0 means a shortage, < 0 means a surplus.
    up_q = quantity.where(out["price_change"] > 0, 0.0)
    down_q = quantity.where(out["price_change"] < 0, 0.0)
    out["excess_demand"] = (up_q.rolling(10).sum() - down_q.rolling(10).sum()) / quantity.rolling(10).sum()

    out["target"] = out["price_change"].shift(-1)             # what we want to predict: tomorrow's % change
    out = out.dropna()
    # the date on which each row's target (tomorrow's change) actually happens
    next_dates = df.index[df.index.get_indexer(out.index) + 1]
    return out, next_dates


# ============================== PLOTS: HISTORY AND DEMAND/SUPPLY ==============================
def plot_price_history(data, split_date, label, currency):
    p = data["close"]
    avg50 = p.rolling(50).mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(p.index, p.values, color="grey", lw=1, label="Market price")
    ax.plot(avg50.index, avg50.values, color="tab:blue", lw=1.5,
            label="50-day average price (rough 'equilibrium' level)")
    ax.axvspan(split_date, p.index[-1], color="orange", alpha=0.12, label="Test period (unseen by the model)")
    ax.set_title(f"{label}: Price History & Equilibrium Trend")
    ax.set_ylabel(f"Price ({currency})")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "01_price_history.png")


def plot_demand_supply_pressure(data, label, currency):
    d = data.iloc[-250:]                                       # about the last year
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    ax[0].plot(d.index, d["close"], color="tab:blue", lw=1.3)
    ax[0].set_ylabel(f"Price ({currency})")
    ax[0].set_title(f"{label}: Market Price over the Last Year")
    colors = np.where(d["excess_demand"] >= 0, "tab:green", "tab:red")
    ax[1].bar(d.index, d["excess_demand"], color=colors, width=1.0)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("Excess Demand")
    ax[1].set_title("Demand-Supply Imbalance: Green = Buyers Dominate (Shortage, Price UP) | Red = Sellers Dominate (Surplus, Price DOWN)")
    save(fig, "02_excess_demand.png")


def plot_supply_demand_illustration(currency):
    """Textbook diagram with theoretical numbers: equilibrium, shortage/surplus, demand shift,
    consumer and producer surplus, and elasticity."""
    Pd0, Ps0, m, n = 150.0, 50.0, 0.5, 0.5          # demand: P = Pd0 - m*Q ; supply: P = Ps0 + n*Q
    Qe = (Pd0 - Ps0) / (m + n)                      # equilibrium quantity
    Pe = Pd0 - m * Qe                               # equilibrium price
    s = 30.0                                        # demand curve shifts right by 30 units
    Qe2 = (Pd0 + m * s - Ps0) / (m + n)
    Pe2 = Ps0 + n * Qe2
    cs = 0.5 * Qe * (Pd0 - Pe)                      # consumer surplus = triangle under demand, above price
    ps = 0.5 * Qe * (Pe - Ps0)                      # producer surplus = triangle above supply, under price
    ed = (-1 / m) * Pe / Qe                         # price elasticity of demand at equilibrium
    es = (1 / n) * Pe / Qe                          # price elasticity of supply at equilibrium

    Q = np.linspace(0, 220, 200)
    q1 = np.linspace(0, Qe, 100)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(Q, Pd0 - m * Q, color="tab:blue", lw=2, label="Demand curve D (slopes down)")
    ax.plot(Q, Ps0 + n * Q, color="tab:red", lw=2, label="Supply curve S (slopes up)")
    ax.plot(Q, Pd0 + m * s - m * Q, color="tab:blue", lw=2, ls="--", label="New demand D2 (demand increases)")
    ax.fill_between(q1, Pd0 - m * q1, Pe, color="tab:blue", alpha=0.18, label=f"Consumer surplus = {cs:.0f}")
    ax.fill_between(q1, Pe, Ps0 + n * q1, color="tab:red", alpha=0.18, label=f"Producer surplus = {ps:.0f}")
    ax.scatter([Qe], [Pe], color="k", zorder=5)
    ax.annotate(f"Equilibrium E\nP = {Pe:.0f}, Q = {Qe:.0f}", (Qe, Pe), xytext=(Qe + 8, Pe - 28), fontsize=9)
    ax.scatter([Qe2], [Pe2], color="green", zorder=5)
    ax.annotate(f"New equilibrium E2\nP = {Pe2:.1f}, Q = {Qe2:.0f}\n(price rises)", (Qe2, Pe2),
                xytext=(Qe2 + 6, Pe2 + 10), fontsize=9, color="green")
    for price, text in [(75, "Shortage: price rises"), (125, "Surplus: price falls")]:
        qs_, qd_ = (price - Ps0) / n, (Pd0 - price) / m
        ax.annotate("", xy=(qd_, price), xytext=(qs_, price), arrowprops=dict(arrowstyle="<->", color="grey"))
        ax.text((qs_ + qd_) / 2, price + 3, text, ha="center", fontsize=9, color="dimgrey")
    ax.text(0.98, 0.04, f"Price elasticity at E:\nof demand = {ed:.1f}\nof supply = {es:.1f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set_xlabel("Quantity of shares traded (Q)")
    ax.set_ylabel(f"Price (P in {currency})")
    ax.set_xlim(0, 220)
    ax.set_ylim(0, 170)
    ax.set_title("Microeconomic Theory: Demand and Supply of Shares")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "03_supply_demand_illustration.png")


# ============================== LSTM MODEL ==============================
class LSTMModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.lstm = nn.LSTM(n_features, HIDDEN, LAYERS, batch_first=True, dropout=DROPOUT)
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(HIDDEN, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def make_sequences(X, y):
    xs, ys = [], []
    for i in range(WINDOW, len(X) + 1):
        xs.append(X[i - WINDOW:i])
        ys.append(y[i - 1])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def batches(X, y, shuffle):
    idx = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
    for i in range(0, len(X), BATCH):
        j = idx[i:i + BATCH]
        yield torch.tensor(X[j]).to(device), torch.tensor(y[j]).to(device)


def predict(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in batches(X, np.zeros(len(X), dtype=np.float32), False):
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def train_model(X_tr, y_tr, X_val, y_val, num_epochs):
    model = LSTMModel(len(FEATURES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    best, best_state, wait = np.inf, None, 0
    tr_hist, val_hist = [], []
    print(f"Training LSTM for {num_epochs} epochs (Early stopping patience = {PATIENCE})...")
    for ep in range(num_epochs):
        model.train()
        losses = []
        for xb, yb in batches(X_tr, y_tr, True):
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        vl = float(np.mean((predict(model, X_val) - y_val) ** 2))
        tr_hist.append(float(np.mean(losses)))
        val_hist.append(vl)
        if (ep + 1) % 5 == 0 or ep == 0 or (ep + 1) == num_epochs:
            print(f"epoch {ep + 1:3d}/{num_epochs:3d} | train error: {tr_hist[-1]:.4f} | validation error: {vl:.4f}")
        if vl < best:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"Early stopping at epoch {ep + 1}: validation error stopped improving.")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(tr_hist, label="Training error (MSE)")
    ax.plot(val_hist, label="Validation error (MSE)")
    ax.axvline(int(np.argmin(val_hist)), color="k", ls="--", lw=0.8, label="Best stopping point")
    ax.set_title("LSTM Learning Curve: Avoiding Overfitting via Early Stopping")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error")
    ax.legend()
    save(fig, "04_training_curve.png")
    return model


# ============================== PREDICTION PLOTS ==============================
def plot_predicted_vs_actual(data, split, dates_next, p_true, p_lstm, p_naive, label, currency):
    train_dates = data.index[:split + 1]
    fig, ax = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [1.2, 1]})
    ax[0].plot(train_dates, data["close"].values[:split + 1], color="grey", lw=1, label="Past Price (Training Period)")
    ax[0].plot(dates_next, p_true, color="tab:blue", lw=1.3, label="Actual Price (Test Period)")
    ax[0].plot(dates_next, p_lstm, color="tab:orange", lw=1.1, alpha=0.9, label="LSTM Forecast")
    ax[0].axvline(dates_next[0], color="k", ls=":", lw=1)
    ax[0].set_title(f"{label}: Test Forecast vs Actual Market Trajectory")
    ax[0].set_ylabel(f"Price ({currency})")
    ax[0].legend(loc="upper left")

    z = min(90, len(dates_next))
    ax[1].plot(dates_next[-z:], p_true[-z:], "o-", color="tab:blue", ms=3, lw=1.2, label="Actual Price Tomorrow")
    ax[1].plot(dates_next[-z:], p_lstm[-z:], "s-", color="tab:orange", ms=3, lw=1.1, label="LSTM Forecast")
    ax[1].plot(dates_next[-z:], p_naive[-z:], "--", color="green", lw=1,
               label="Naive Baseline: Tomorrow = Today")
    ax[1].set_title(f"Detailed Zoom (Last {z} Trading Days): LSTM vs Naive Baseline")
    ax[1].set_ylabel(f"Price ({currency})")
    ax[1].legend()
    save(fig, "05_predicted_vs_actual_price.png")


def plot_errors(dates_next, true_ret, pred_ret, p_true, p_lstm, currency):
    err = p_lstm - p_true
    hit = np.sign(pred_ret) == np.sign(true_ret)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].plot(dates_next, err, color="tab:purple", lw=0.8)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_title(f"Forecast Error = Predicted Price - Actual Price ({currency})")
    ax[0].set_ylabel(f"Error ({currency})")
    ax[1].scatter(pred_ret * 100, true_ret * 100, s=8, alpha=0.4)
    lim = max(np.abs(true_ret).max(), np.abs(pred_ret).max()) * 100
    ax[1].plot([-lim, lim], [-lim, lim], "k--", lw=1, label="Perfect Forecast (45° line)")
    ax[1].axhline(0, color="grey", lw=0.6)
    ax[1].axvline(0, color="grey", lw=0.6)
    ax[1].set_xlabel("Predicted % change in price")
    ax[1].set_ylabel("Actual % change in price")
    ax[1].set_title(f"Predicted vs Actual Change (Direction Accuracy: {hit.mean():.1%})")
    ax[1].legend()
    save(fig, "06_forecast_errors.png")


# ============================== COST, REVENUE AND PROFIT ==============================
def summarise(gross, cost, trades):
    tr, tc = float(gross.sum()), float(cost.sum())
    return dict(tr=tr, tc=tc, profit=tr - tc, trades=int(trades.sum()),
                cum_tr=np.cumsum(gross), cum_tc=np.cumsum(cost),
                wealth=CAPITAL + np.cumsum(gross - cost))


def backtest(dates_next, true_ret, pred_ret, currency):
    """Strategy: hold the share on days the model predicts a rise, otherwise stay in cash.
    Every buy or sell is a trade and pays transaction costs."""
    pos = (pred_ret > 0).astype(float)                         # 1 = hold the share, 0 = cash
    trades = np.abs(np.diff(pos, prepend=0.0))                 # 1 whenever we switch (buy or sell)
    gross = CAPITAL * pos * true_ret                           # total revenue each day
    cost = CAPITAL * TC_RATE * trades                          # total cost each day
    s = summarise(gross, cost, trades)

    bh_trades = np.zeros_like(true_ret)
    bh_trades[0] = 1                                           # buy once on day 1 and never sell
    b = summarise(CAPITAL * true_ret, CAPITAL * TC_RATE * bh_trades, bh_trades)

    fig, ax = plt.subplots(3, 1, figsize=(13, 11), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.3, 0.6]})
    ax[0].plot(dates_next, s["cum_tr"], color="tab:green", lw=1.6, label="Total Revenue TR (Gross Gains)")
    ax[0].plot(dates_next, s["cum_tc"], color="tab:red", lw=1.6, label="Total Cost TC (Cumulative Trading Costs)")
    ax[0].fill_between(dates_next, s["cum_tc"], s["cum_tr"], where=s["cum_tr"] >= s["cum_tc"],
                       color="green", alpha=0.15, label="Profit (TR > TC)")
    ax[0].fill_between(dates_next, s["cum_tc"], s["cum_tr"], where=s["cum_tr"] < s["cum_tc"],
                       color="red", alpha=0.15, label="Loss (TR < TC)")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_title("Firm Economics: Total Revenue vs Total Cost (Profit = TR - TC)")
    ax[0].set_ylabel(f"Amount ({currency})")
    ax[0].legend(fontsize=8)

    ax[1].plot(dates_next, s["wealth"], color="tab:orange", lw=1.6, label="LSTM Strategy")
    ax[1].plot(dates_next, b["wealth"], color="tab:blue", lw=1.6, label="Buy & Hold (Passive Benchmark)")
    ax[1].axhline(CAPITAL, color="k", ls="--", lw=0.8, label="Initial Capital")
    ax[1].set_title(f"Cumulative Wealth Growth from {currency}{CAPITAL:,} during Test Period")
    ax[1].set_ylabel(f"Portfolio Wealth ({currency})")
    ax[1].legend(fontsize=8)

    ax[2].fill_between(dates_next, pos, 0, step="post", color="tab:orange", alpha=0.6)
    ax[2].set_yticks([0, 1])
    ax[2].set_yticklabels(["Cash", "Hold Stock"])
    ax[2].set_title(f"Daily Model Allocation ({s['trades']} Trades executed, each cost {currency}{CAPITAL * TC_RATE:,.0f})")
    save(fig, "07_revenue_cost_profit.png")
    return s, b


# ============================== UTILITY ==============================
def plot_utility(w_start, w_strat, w_bh, currency):
    # Total utility U(W) = sqrt(W / SCALE_W); Marginal utility MU = 0.5 / sqrt(W / SCALE_W)
    u = lambda w: np.sqrt(np.asarray(w) / SCALE_W)
    mu = lambda w: 0.5 / np.sqrt(np.asarray(w) / SCALE_W)
    pts = [("Initial Capital", w_start, "k"), ("LSTM Strategy", w_strat, "tab:orange"), ("Buy & Hold", w_bh, "tab:blue")]
    lo = min(0.6 * w_start, 0.9 * min(w_strat, w_bh))
    hi = 1.15 * max(w_start, w_strat, w_bh)
    W = np.linspace(lo, hi, 300)

    step = 0.1 * w_start
    gain = float(u(w_start + step) - u(w_start))
    loss = float(u(w_start) - u(w_start - step))

    fig, ax = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)
    ax[0].plot(W, u(W), color="tab:green", lw=2)
    for name, w, c in pts:
        ax[0].scatter([w], [float(u(w))], color=c, s=70, zorder=5,
                      label=f"{name}: {currency}{w:,.0f}, TU = {float(u(w)):.3f}")
    ax[0].set_ylabel("Total Utility TU (utils)")
    ax[0].set_title("Total Utility: Rises with Wealth at a Decreasing Rate (Risk-Aversion)")
    ax[0].legend(fontsize=8, loc="lower right")

    ax[1].plot(W, mu(W), color="tab:purple", lw=2)
    for name, w, c in pts:
        ax[1].scatter([w], [float(mu(w))], color=c, s=70, zorder=5)
    ax[1].set_xlabel(f"Wealth W ({currency})")
    ax[1].set_ylabel("Marginal Utility MU (utils/wealth unit)")
    ax[1].set_title("Diminishing Marginal Utility: MU Falls as Wealth Increases")
    ax[1].text(0.98, 0.95,
               f"+{currency}{step:,.0f} gain adds {gain:.4f} utils\n-{currency}{step:,.0f} loss removes {loss:.4f} utils\n"
               "Asymmetry: An equal loss hurts more than an equal gain rewards",
               transform=ax[1].transAxes, ha="right", va="top", fontsize=9,
               bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    save(fig, "08_utility.png")
    return float(u(w_strat) - u(w_start)), float(u(w_bh) - u(w_start))


# ============================== MAIN ==============================
def parse_arguments():
    parser = argparse.ArgumentParser(description="Stock predictor using LSTM: Economics PBL")
    parser.add_argument("--ticker", type=str, default=TICKER,
                        help=f"Stock ticker symbol (default: {TICKER}). Available presets: {list(AVAILABLE_STOCKS.keys())}")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS})")
    # If run in environments like Jupyter or test harness with extra args, avoid error:
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_arguments()
    selected_ticker = args.ticker.upper()
    selected_epochs = args.epochs

    meta = get_stock_meta(selected_ticker)
    currency = meta["currency"]

    print("=" * 72)
    print("📈 STOCK PICKER & LSTM ECONOMIC FORECASTING")
    print("=" * 72)
    print(f"Selected Ticker : {selected_ticker} ({meta['name']})")
    print(f"Currency        : {currency}")
    print(f"Training Epochs : {selected_epochs}")
    print(f"Lookback Window : {WINDOW} trading days")
    print(f"Device          : {device}")
    print("=" * 72)

    raw, label = load_raw(selected_ticker)
    data, next_dates = build_dataset(raw)
    n = len(data)
    split = int(n * (1 - TEST_FRAC))
    print(f"\n{label}: {n} trading days ({data.index[0].date()} to {data.index[-1].date()})")
    print(f"Training days : {split} (up to {data.index[split - 1].date()})")
    print(f"Test days     : {n - split} (after {data.index[split - 1].date()})\n")

    print("== 1. Generating Demand and Supply Plots ==")
    plot_price_history(data, data.index[split], label, currency)
    plot_demand_supply_pressure(data, label, currency)
    plot_supply_demand_illustration(currency)

    # Scale features using TRAINING days only, so the model never "sees" the future
    sx = StandardScaler().fit(data[FEATURES].values[:split])
    sy = StandardScaler().fit(data[["target"]].values[:split])
    X_all = sx.transform(data[FEATURES].values)
    y_all = sy.transform(data[["target"]].values).ravel()
    X_seq, y_seq = make_sequences(X_all, y_all)
    end_rows = np.arange(WINDOW - 1, n)
    tr_mask = end_rows < split
    X_full, y_full = X_seq[tr_mask], y_seq[tr_mask]
    X_te, y_te = X_seq[~tr_mask], y_seq[~tr_mask]
    v = int(len(X_full) * (1 - VAL_FRAC))

    print("\n== 2. Training the LSTM Model ==")
    model = train_model(X_full[:v], y_full[:v], X_full[v:], y_full[v:], selected_epochs)

    # ---- Forecasts on the test days ----
    pred_ret = sy.inverse_transform(predict(model, X_te).reshape(-1, 1)).ravel()   # predicted % change
    true_ret = sy.inverse_transform(y_te.reshape(-1, 1)).ravel()                   # actual % change
    test_rows = end_rows[~tr_mask]
    dates_next = next_dates[test_rows]
    p_today = data["close"].values[test_rows]
    p_true = p_today * (1 + true_ret)
    p_lstm = p_today * (1 + pred_ret)
    p_naive = p_today.copy()                                # simple guess: tomorrow = today

    print("\n== 3. Generating Forecast Plots ==")
    plot_predicted_vs_actual(data, split, dates_next, p_true, p_lstm, p_naive, label, currency)
    plot_errors(dates_next, true_ret, pred_ret, p_true, p_lstm, currency)

    print("\n== 4. Generating Revenue, Cost and Profit Plots ==")
    s, b = backtest(dates_next, true_ret, pred_ret, currency)

    print("\n== 5. Generating Investor Utility Plots ==")
    du_s, du_b = plot_utility(CAPITAL, s["wealth"][-1], b["wealth"][-1], currency)

    # ---- Report Statistics ----
    rmse = lambda a, c: float(np.sqrt(np.mean((a - c) ** 2)))
    mae = lambda a, c: float(np.mean(np.abs(a - c)))
    hits = float(np.mean(np.sign(pred_ret) == np.sign(true_ret)))
    up_share = float(np.mean(true_ret > 0))

    train = data.iloc[:split]
    after_shortage = train.loc[train["excess_demand"] > 0, "target"].mean() * 100
    after_surplus = train.loc[train["excess_demand"] < 0, "target"].mean() * 100

    print("\n" + "=" * 72)
    print(f" 📊 ECONOMICS PBL REPORT: {label} (Test Days: {len(true_ret)})")
    print("=" * 72)

    print("1) DEMAND AND SUPPLY: Does excess demand push prices up?")
    print(f"   - Average next-day change after SHORTAGE (excess demand > 0): {after_shortage:+.3f}%")
    print(f"   - Average next-day change after SURPLUS  (excess demand < 0): {after_surplus:+.3f}%")
    if after_shortage > after_surplus:
        print("   -> Finding: Shortage days were followed by higher prices than surplus days (aligning with theory).")
        print("      However, competitive arbitrage narrows the gap as participants respond swiftly.")
    else:
        print("   -> Finding: High market efficiency rapidly eliminates supply/demand imbalances,")
        print("      making single-day price shifts difficult to exploit purely via aggregate volume.")

    print("\n2) FORECAST EVALUATION (Returns in %):")
    print(f"   - RMSE: LSTM {rmse(true_ret, pred_ret) * 100:.4f}% vs Naive Baseline {rmse(true_ret, 0) * 100:.4f}%")
    print(f"   - MAE : LSTM {mae(true_ret, pred_ret) * 100:.4f}% vs Naive Baseline {mae(true_ret, 0) * 100:.4f}%")
    print(f"   - Price-level RMSE ({currency}): LSTM {rmse(p_true, p_lstm):.2f} vs Naive Baseline {rmse(p_true, p_naive):.2f}")
    print(f"   - Directional Accuracy: {hits:.1%} (Market upward rate: {up_share:.1%})")

    print(f"\n3) FIRM ECONOMICS: Revenue, Cost & Profit (Initial Capital {currency}{CAPITAL:,}):")
    print(f"   {'Strategy':18s}{'Revenue':>12s}{'Cost':>10s}{'Profit':>12s}{'Trades':>8s}{'Final Wealth':>16s}")
    for nm, d in [("LSTM Strategy", s), ("Buy & Hold", b)]:
        print(f"   {nm:18s}{currency}{d['tr']:>11,.0f}{currency}{d['tc']:>9,.0f}{currency}{d['profit']:>11,.0f}{d['trades']:>8d}"
              f"{currency}{d['wealth'][-1]:>15,.0f}")
    if s["trades"] > 0:
        print(f"   - LSTM Strategy Average Revenue per Trade: {currency}{s['tr'] / s['trades']:,.0f}")
        print(f"   - Marginal Cost per Trade (MC):            {currency}{CAPITAL * TC_RATE:,.0f}")
        print("   -> Economic Rule: A trade is viable only when Marginal Revenue (MR) exceeds Marginal Cost (MC).")

    print(f"\n4) INVESTOR UTILITY THEORY: U(W) = sqrt(W / {SCALE_W:,})")
    print(f"   - Utility change (LSTM Strategy): {du_s:+.4f} utils")
    print(f"   - Utility change (Buy & Hold)   : {du_b:+.4f} utils")
    print("   -> Diminishing Marginal Utility demonstrates risk aversion: portfolio volatility incurs utility penalties.")

    print("\n" + "=" * 72)
    print("🎓 CORE PBL TAKEAWAYS:")
    print(" 1. Prices are formed at the intersection of supply and demand.")
    print(" 2. Trading costs represent real economic friction; active strategies must cover marginal costs.")
    print(" 3. Investors exhibit concave utility (loss aversion), making drawdown control critical.")
    print(f"All figures successfully saved to ./{OUT_DIR}")
    print("=" * 72)

    if SHOW:
        plt.show()


if __name__ == "__main__":
    main()
