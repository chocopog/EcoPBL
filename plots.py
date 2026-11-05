"""All eight visualization functions plus strategy and walk-forward diagnostics."""

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (thread-safe for Flask)

import numpy as np
import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass


# ---------- helpers ----------
def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------- 1. Price History ----------
def plot_price_history(data, split_date, label, currency, path):
    p = data["close"]
    avg50 = p.rolling(50).mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(p.index, p.values, color="grey", lw=1, label="Market Price")
    ax.plot(avg50.index, avg50.values, color="tab:blue", lw=1.5,
            label="50-day MA (Equilibrium Level)")
    ax.axvspan(split_date, p.index[-1], color="orange", alpha=0.15,
               label="Unseen Test Period")
    ax.set_title(f"{label}: Price Trajectory & Equilibrium Trend", fontsize=12)
    ax.set_ylabel(f"Price ({currency})")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    _save(fig, path)
    return fig


# ---------- 2. Excess Demand ----------
def plot_excess_demand(data, label, currency, path):
    recent = data.iloc[-250:]
    fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                           gridspec_kw={"height_ratios": [1.3, 1]})
    ax[0].plot(recent.index, recent["close"], color="tab:blue", lw=1.3)
    ax[0].set_ylabel(f"Price ({currency})")
    ax[0].set_title(f"{label}: Market Price (Past ~1 Year)", fontsize=11)
    colors = np.where(recent["excess_demand"] >= 0, "tab:green", "tab:red")
    ax[1].bar(recent.index, recent["excess_demand"], color=colors, width=1.0)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("Excess Demand")
    ax[1].set_title(
        "Green = Buyers Dominate (Shortage) | Red = Sellers Dominate (Surplus)",
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, path)
    return fig


# ---------- 3. Textbook Supply & Demand ----------
def plot_supply_demand(currency, path):
    Pd0, Ps0, m, n = 150.0, 50.0, 0.5, 0.5
    Qe = (Pd0 - Ps0) / (m + n)
    Pe = Pd0 - m * Qe
    s = 30.0
    Qe2 = (Pd0 + m * s - Ps0) / (m + n)
    Pe2 = Ps0 + n * Qe2
    cs = 0.5 * Qe * (Pd0 - Pe)
    ps = 0.5 * Qe * (Pe - Ps0)
    ed = (-1 / m) * Pe / Qe
    es = (1 / n) * Pe / Qe

    Q = np.linspace(0, 220, 200)
    q1 = np.linspace(0, Qe, 100)
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.plot(Q, Pd0 - m * Q, "tab:blue", lw=2, label="Demand D")
    ax.plot(Q, Ps0 + n * Q, "tab:red", lw=2, label="Supply S")
    ax.plot(Q, Pd0 + m * s - m * Q, "tab:blue", lw=2, ls="--", label="New Demand D2")
    ax.fill_between(q1, Pd0 - m * q1, Pe, color="tab:blue", alpha=.18,
                     label=f"Consumer Surplus = {cs:.0f}")
    ax.fill_between(q1, Pe, Ps0 + n * q1, color="tab:red", alpha=.18,
                     label=f"Producer Surplus = {ps:.0f}")
    ax.scatter([Qe], [Pe], color="k", zorder=5)
    ax.annotate(f"E\nP={Pe:.0f}, Q={Qe:.0f}", (Qe, Pe),
                xytext=(Qe + 8, Pe - 28), fontsize=9)
    ax.scatter([Qe2], [Pe2], color="green", zorder=5)
    ax.annotate(f"E2\nP={Pe2:.1f}, Q={Qe2:.0f}", (Qe2, Pe2),
                xytext=(Qe2 + 6, Pe2 + 10), fontsize=9, color="green")
    for price, text in [(75, "Shortage → Price UP"),
                        (125, "Surplus → Price DOWN")]:
        qs_ = (price - Ps0) / n
        qd_ = (Pd0 - price) / m
        ax.annotate("", xy=(qd_, price), xytext=(qs_, price),
                    arrowprops=dict(arrowstyle="<->", color="grey"))
        ax.text((qs_ + qd_) / 2, price + 3, text, ha="center",
                fontsize=9, color="dimgrey")
    ax.text(0.98, 0.04, f"Elasticity at E:\nDemand={ed:.1f}\nSupply={es:.1f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=.85))
    ax.set(xlabel="Quantity (Q)", ylabel=f"Price ({currency})",
           xlim=(0, 220), ylim=(0, 170))
    ax.set_title("Microeconomic Model: Demand, Supply & Equilibrium", fontsize=12)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    _save(fig, path)
    return fig


# ---------- 4. Training curve (RMSE) ----------
def plot_training_curve(train_rmse, val_rmse, path):
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(train_rmse, label="Train RMSE (%)", color="tab:blue")
    ax.plot(val_rmse, label="Val RMSE (%)", color="tab:orange")
    best = int(np.argmin(val_rmse)) if len(val_rmse) else 0
    ax.axvline(best, color="k", ls="--", lw=.8,
               label=f"Best (Epoch {best + 1})")
    ax.set_title("LSTM Learning Curve", fontsize=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (% Daily Return)")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)
    return fig


# ---------- 5. Predicted vs Actual ----------
def plot_predicted_vs_actual(data, split_idx, dates_next, p_true, p_lstm,
                              p_naive, label, currency, path):
    train_dates = data.index[: split_idx + 1]
    fig, ax = plt.subplots(2, 1, figsize=(13, 9),
                           gridspec_kw={"height_ratios": [1.2, 1]})
    ax[0].plot(train_dates, data["close"].values[: split_idx + 1],
               color="grey", lw=1, label="Training Period")
    ax[0].plot(dates_next, p_true, color="tab:blue", lw=1.3,
               label="Actual (Test)")
    ax[0].plot(dates_next, p_lstm, color="tab:orange", lw=1.1, alpha=.9,
               label="LSTM Forecast")
    ax[0].axvline(dates_next[0], color="k", ls=":", lw=1)
    ax[0].set_title(f"{label}: Test Forecast vs Actual", fontsize=12)
    ax[0].set_ylabel(f"Price ({currency})")
    ax[0].legend(loc="upper left")

    z = min(90, len(dates_next))
    ax[1].plot(dates_next[-z:], p_true[-z:], "o-", color="tab:blue", ms=3,
               lw=1.2, label="Actual")
    ax[1].plot(dates_next[-z:], p_lstm[-z:], "s-", color="tab:orange", ms=3,
               lw=1.1, label="LSTM")
    ax[1].plot(dates_next[-z:], p_naive[-z:], "--", color="green", lw=1,
               label="Naive (Tomorrow=Today)")
    ax[1].set_title(f"Last {z} Days: LSTM vs Naive Baseline", fontsize=11)
    ax[1].set_ylabel(f"Price ({currency})")
    ax[1].legend()
    fig.tight_layout()
    _save(fig, path)
    return fig


# ---------- 6. Error analysis ----------
def plot_errors(dates_next, true_ret, pred_ret, p_true, p_lstm, currency,
                path):
    err = p_lstm - p_true
    hit = np.sign(pred_ret) == np.sign(true_ret)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].plot(dates_next, err, color="tab:purple", lw=.8)
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].set_title(f"Forecast Error ({currency})", fontsize=11)
    ax[0].set_ylabel(f"Error ({currency})")

    ax[1].scatter(pred_ret * 100, true_ret * 100, s=9, alpha=.4,
                  color="tab:blue")
    lim = max(np.abs(true_ret).max(), np.abs(pred_ret).max()) * 100
    ax[1].plot([-lim, lim], [-lim, lim], "k--", lw=1, label="Perfect")
    ax[1].axhline(0, color="grey", lw=.6)
    ax[1].axvline(0, color="grey", lw=.6)
    ax[1].set_xlabel("Predicted Return (%)")
    ax[1].set_ylabel("Actual Return (%)")
    ax[1].set_title(f"Direction Accuracy: {hit.mean():.1%}", fontsize=11)
    ax[1].legend()
    fig.tight_layout()
    _save(fig, path)
    return fig


# ---------- 7. Revenue / Cost / Profit (returns metrics too) ----------
def make_strategy_position(pred_ret, tc_rate, cost_multiplier=2.0,
                           volatility_multiplier=0.15, strategy_mode="hybrid",
                           baseline_exposure=0.75):
    """Create cost-aware position.

    In 'hybrid' mode, keeps a high baseline exposure (e.g. 75%) to capture upward bull markets
    while using the model to trim exposure on severe downturn signals.
    """
    pred_ret = np.asarray(pred_ret, dtype=float)
    if len(pred_ret) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if strategy_mode == "buy_and_hold":
        return np.ones(len(pred_ret), dtype=float), np.full(len(pred_ret), 0.0)

    recent_vol = np.full(len(pred_ret), np.nan, dtype=float)
    for i in range(len(pred_ret)):
        history = pred_ret[max(0, i - 19):i + 1]
        recent_vol[i] = float(np.std(history, ddof=1)) if len(history) > 1 else 0.0

    entry_threshold = np.maximum(cost_multiplier * tc_rate,
                                 volatility_multiplier * recent_vol)
    pos = np.zeros(len(pred_ret), dtype=float)
    for i, forecast in enumerate(pred_ret):
        if i > 0:
            pos[i] = pos[i - 1]
        if pos[i] == 0.0 and forecast > entry_threshold[i]:
            pos[i] = 1.0
        elif pos[i] == 1.0 and forecast <= 0.0:
            pos[i] = 0.0

    if strategy_mode == "hybrid":
        # 75% baseline market exposure + 25% tactical alpha from LSTM signals
        # Ensures strategy remains highly competitive with Buy & Hold in strong bull markets
        pos = baseline_exposure + (1.0 - baseline_exposure) * pos

    return pos, entry_threshold


def select_strategy_params(pred_ret, true_ret, tc_rate, capital):
    """Select trading thresholds on validation data only."""
    candidates = [
        (cost_multiplier, volatility_multiplier)
        for cost_multiplier in (1.0, 2.0, 3.0)
        for volatility_multiplier in (0.0, 0.1, 0.2)
    ]
    candidates_with_mode = [("hybrid", *c) for c in candidates] + [("active", *c) for c in candidates]
    candidates_with_mode.append(("buy_and_hold", 0.0, 0.0))

    best_score, best_params = -np.inf, {
        "strategy_mode": "hybrid", "cost_multiplier": 1.5, "volatility_multiplier": 0.1
    }

    for strategy_mode, cost_multiplier, volatility_multiplier in candidates_with_mode:
        pos, _ = make_strategy_position(
            pred_ret, tc_rate, cost_multiplier, volatility_multiplier, strategy_mode
        )
        trades = np.abs(np.diff(pos, prepend=0.0))
        wealth = capital + np.cumsum(
            capital * pos * true_ret - capital * tc_rate * trades
        )
        peak = np.maximum.accumulate(wealth)
        max_drawdown = float(np.max((peak - wealth) / np.maximum(peak, 1e-6)))
        score = float(wealth[-1]) - capital * max_drawdown
        if score > best_score:
            best_score = score
            best_params = {
                "strategy_mode": strategy_mode,
                "cost_multiplier": cost_multiplier,
                "volatility_multiplier": volatility_multiplier,
            }
    return best_params


def plot_backtest(dates_next, true_ret, pred_ret, capital, tc_rate, currency,
                  path, strategy_params=None):
    strategy_params = strategy_params or {}
    pos, entry_threshold = make_strategy_position(
        pred_ret,
        tc_rate,
        strategy_params.get("cost_multiplier", 1.5),
        strategy_params.get("volatility_multiplier", 0.1),
        strategy_params.get("strategy_mode", "hybrid"),
    )
    trades = np.abs(np.diff(pos, prepend=0.0))
    gross = capital * pos * true_ret
    cost = capital * tc_rate * trades

    def _summarise(g, c, t):
        tr, tc = float(g.sum()), float(c.sum())
        return dict(tr=tr, tc=tc, profit=tr - tc, trades=int(t.sum()),
                    cum_tr=np.cumsum(g), cum_tc=np.cumsum(c),
                    wealth=capital + np.cumsum(g - c))

    strat = _summarise(gross, cost, trades)
    bh_t = np.zeros_like(true_ret); bh_t[0] = 1
    bh = _summarise(capital * true_ret, capital * tc_rate * bh_t, bh_t)

    fig, ax = plt.subplots(3, 1, figsize=(13, 11), sharex=True,
                           gridspec_kw={"height_ratios": [1.3, 1.3, .6]})
    ax[0].plot(dates_next, strat["cum_tr"], "tab:green", lw=1.6, label="TR (LSTM Strategy)")
    ax[0].plot(dates_next, strat["cum_tc"], "tab:red", lw=1.6, label="TC (Friction)")
    ax[0].fill_between(dates_next, strat["cum_tc"], strat["cum_tr"],
                       where=strat["cum_tr"] >= strat["cum_tc"],
                       color="green", alpha=.15, label="Profit")
    ax[0].fill_between(dates_next, strat["cum_tc"], strat["cum_tr"],
                       where=strat["cum_tr"] < strat["cum_tc"],
                       color="red", alpha=.15, label="Loss")
    ax[0].axhline(0, color="k", lw=.6)
    ax[0].set_title("Firm Economics: Total Revenue (TR) vs Total Cost (TC)", fontsize=12)
    ax[0].set_ylabel(f"Amount ({currency})")
    ax[0].legend(fontsize=8.5)

    ax[1].plot(dates_next, strat["wealth"], "tab:orange", lw=1.6, label="Hybrid LSTM Strategy")
    ax[1].plot(dates_next, bh["wealth"], "tab:blue", lw=1.6, label="Buy & Hold (Benchmark)")
    ax[1].axhline(capital, color="k", ls="--", lw=.8, label="Initial Capital")
    ax[1].set_title(f"Portfolio Wealth: Hybrid LSTM vs Buy & Hold ({currency}{capital:,})", fontsize=12)
    ax[1].set_ylabel(f"Wealth ({currency})")
    ax[1].legend(fontsize=8.5)

    ax[2].fill_between(dates_next, pos, 0, step="post",
                       color="tab:orange", alpha=.6)
    ax[2].set_ylim(-0.05, 1.05)
    ax[2].set_yticks([0, 0.5, 1.0])
    ax[2].set_yticklabels(["Cash", "50% Exposure", "100% Invested"])
    ax[2].set_title(
        f"Dynamic Allocation (Trades: {strat['trades']}, Est. Cost per Trade: {currency}{capital * tc_rate:,.0f})",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, path)

    # strip numpy arrays before returning as JSON-safe dict
    s = {k: v for k, v in strat.items() if k not in ("cum_tr", "cum_tc", "wealth")}
    s["final_wealth"] = float(strat["wealth"][-1])
    s["strategy"] = "Hybrid Trend-Following LSTM (Bull-Market Benchmark)" if strategy_params.get("strategy_mode", "hybrid") == "hybrid" else "Validation-adaptive Active LSTM"
    s["entry_threshold_pct"] = round(float(np.mean(entry_threshold)) * 100, 4) if len(entry_threshold) else 0.0
    s["cost_multiplier"] = strategy_params.get("cost_multiplier", 1.5)
    s["volatility_multiplier"] = strategy_params.get("volatility_multiplier", 0.1)

    b = {k: v for k, v in bh.items() if k not in ("cum_tr", "cum_tc", "wealth")}
    b["final_wealth"] = float(bh["wealth"][-1])
    return fig, s, b


# ---------- 8. Utility ----------
def plot_utility(w_start, w_strat, w_bh, currency, scale_w, path):
    u = lambda w: np.sqrt(np.asarray(w) / scale_w)
    mu = lambda w: 0.5 / np.sqrt(np.asarray(w) / scale_w)
    pts = [("Start", w_start, "k"),
           ("LSTM Strategy", w_strat, "tab:orange"),
           ("Buy & Hold", w_bh, "tab:blue")]
    lo = min(0.6 * w_start, 0.9 * min(w_strat, w_bh))
    hi = 1.15 * max(w_start, w_strat, w_bh)
    W = np.linspace(lo, hi, 300)
    step = 0.1 * w_start
    gain = float(u(w_start + step) - u(w_start))
    loss = float(u(w_start) - u(w_start - step))

    fig, ax = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)
    ax[0].plot(W, u(W), "tab:green", lw=2)
    for name, w, c in pts:
        ax[0].scatter([w], [float(u(w))], color=c, s=70, zorder=5,
                      label=f"{name}: {currency}{w:,.0f}  TU={float(u(w)):.3f}")
    ax[0].set_ylabel("Total Utility")
    ax[0].set_title("Total Utility (concave → risk aversion)", fontsize=11)
    ax[0].legend(fontsize=8.5, loc="lower right")

    ax[1].plot(W, mu(W), "tab:purple", lw=2)
    for name, w, c in pts:
        ax[1].scatter([w], [float(mu(w))], color=c, s=70, zorder=5)
    ax[1].set_xlabel(f"Wealth ({currency})")
    ax[1].set_ylabel("Marginal Utility")
    ax[1].set_title("Diminishing Marginal Utility", fontsize=11)
    ax[1].text(0.98, 0.95,
               f"+{currency}{step:,.0f} → +{gain:.4f} utils\n"
               f"−{currency}{step:,.0f} → −{loss:.4f} utils\n"
               "Loss hurts more than equal gain!",
               transform=ax[1].transAxes, ha="right", va="top", fontsize=9,
               bbox=dict(boxstyle="round", fc="white", alpha=.85))
    fig.tight_layout()
    _save(fig, path)

    du_s = float(u(w_strat) - u(w_start))
    du_b = float(u(w_bh) - u(w_start))
    return fig, du_s, du_b


def plot_walk_forward_diagnostics(folds, metrics, currency, path):
    """Plot model probabilities, regimes, allocation, wealth, and drawdown."""
    if not folds or len(folds) == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Diagnostics unavailable for the selected range", ha="center", va="center", fontsize=12)
        ax.axis("off")
        _save(fig, path)
        return fig

    dates = np.concatenate([fold["dates"] for fold in folds if len(fold["dates"])])
    actual = np.concatenate([fold["actual_label"] for fold in folds if len(fold["actual_label"])])
    lstm_prob = np.concatenate([fold["lstm_probability"] for fold in folds if len(fold["lstm_probability"])])
    rf_prob = np.concatenate([fold["random_forest_probability"] for fold in folds if len(fold["random_forest_probability"])])
    lstm_pos = np.concatenate([fold["hybrid_lstm_position"] for fold in folds if len(fold["hybrid_lstm_position"])])
    rf_pos = np.concatenate([fold["hybrid_random_forest_position"] for fold in folds if len(fold["hybrid_random_forest_position"])])
    returns = np.concatenate([fold["forward_return"] for fold in folds if len(fold["forward_return"])])

    if len(dates) == 0 or len(returns) == 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Insufficient observations across folds", ha="center", va="center")
        ax.axis("off")
        _save(fig, path)
        return fig

    def wealth(position):
        return 100_000 * np.cumprod(1 + position * returns)

    lstm_wealth = wealth(lstm_pos)
    rf_wealth = wealth(rf_pos)
    bh_wealth = wealth(np.ones(len(returns)))
    fig, ax = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    ax[0].plot(dates, lstm_prob, label="LSTM probability", color="tab:orange")
    ax[0].plot(dates, rf_prob, label="Random Forest probability", color="tab:green")
    ax[0].scatter(dates, actual, s=8, alpha=.25, label="Actual profitable label")
    ax[0].axhline(.5, color="k", ls="--", lw=.8)
    ax[0].set_ylabel("Probability")
    ax[0].set_title("Predicted Probability vs Actual Cost-Clearing Outcome", fontsize=11)
    ax[0].legend(fontsize=8)

    ax[1].plot(dates, lstm_pos, label="Hybrid LSTM allocation", color="tab:orange")
    ax[1].plot(dates, rf_pos, label="Hybrid Random Forest allocation", color="tab:green")
    ax[1].set_ylim(-.02, 1.02)
    ax[1].set_ylabel("Allocation")
    ax[1].set_title("Hybrid Allocation (75% baseline market exposure + 25% model adjustment)", fontsize=11)
    ax[1].legend(fontsize=8)

    ax[2].plot(dates, bh_wealth, label="Buy & hold", color="tab:blue")
    ax[2].plot(dates, lstm_wealth, label="Hybrid LSTM", color="tab:orange")
    ax[2].plot(dates, rf_wealth, label="Hybrid Random Forest", color="tab:green")
    ax[2].set_ylabel(f"Wealth ({currency})")
    ax[2].set_title("Compounded Wealth Comparison", fontsize=11)
    ax[2].legend(fontsize=8)

    for label, series, color in (
        ("Buy & hold drawdown", bh_wealth, "tab:blue"),
        ("Hybrid LSTM drawdown", lstm_wealth, "tab:orange"),
        ("Hybrid Random Forest drawdown", rf_wealth, "tab:green"),
    ):
        peak = np.maximum.accumulate(series)
        ax[3].plot(dates, (peak - series) / np.maximum(peak, 1e-6) * 100, label=label, color=color)
    ax[3].set_ylabel("Drawdown (%)")
    ax[3].set_title("Drawdown Comparison", fontsize=11)
    ax[3].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, path)
    return fig
