"""Walk-forward classification models and risk-aware portfolio evaluation."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import data
import model
import plots
from config import FEATURES, LOOKBACK_WINDOW, SEED


def benchmark_ticker(ticker):
    return "^NSEI" if ticker.endswith((".NS", ".BO")) else "^GSPC"


def _metrics(returns, positions, capital, tc_rate, horizon=1, rf_annual=0.0):
    returns = np.asarray(returns, dtype=float)
    positions = np.asarray(positions, dtype=float)
    trades = np.abs(np.diff(positions, prepend=0.0))
    net_returns = positions * returns - tc_rate * trades
    wealth = capital * np.cumprod(1 + net_returns)
    daily = pd.Series(net_returns)
    periods_per_year = 252 / max(horizon, 1)
    rf_period = rf_annual / periods_per_year
    excess = daily - rf_period
    downside_dev = float(
        np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0.0) ** 2))
    ) if len(excess) else 0.0
    volatility = float(daily.std(ddof=1) * np.sqrt(periods_per_year)) if len(daily) > 1 else 0.0
    sharpe = float(
        excess.mean() * np.sqrt(periods_per_year) / daily.std(ddof=1)
    ) if (len(daily) > 1 and daily.std(ddof=1) > 0) else 0.0
    sortino = float(
        excess.mean() * np.sqrt(periods_per_year) / downside_dev
    ) if downside_dev > 0 else 0.0
    peak = np.maximum.accumulate(wealth)
    drawdown = (peak - wealth) / np.maximum(peak, 1e-6)
    years = max(len(returns) * horizon / 252, horizon / 252)
    final_w = float(wealth[-1]) if len(wealth) else float(capital)
    cagr = float((max(final_w, 1e-4) / capital) ** (1 / max(years, 1e-3)) - 1) if final_w > 0 else -1.0

    prev_wealth = np.concatenate([[capital], wealth[:-1]]) if len(wealth) else np.array([])
    transaction_cost = float(
        np.sum(prev_wealth * tc_rate * trades)
    ) if len(wealth) else 0.0

    return {
        "final_wealth": final_w,
        "profit": float(final_w - capital),
        "transaction_cost": transaction_cost,
        "trades": int(trades.sum()),
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "win_rate": float(np.mean(net_returns > 0)) if len(net_returns) else 0.0,
        "turnover": float(trades.sum() / max(len(trades), 1)),
    }


def _positions(probabilities, regimes, threshold=0.50):
    """Regime-aware allocation tuned to participate strongly in upward bull trends

    while defending capital during high volatility and market drawdowns.

    The probability cutoffs are hand-set, not fitted on validation data.
    """
    probabilities = np.asarray(probabilities)
    positions = np.zeros(len(probabilities), dtype=float)
    for i, probability in enumerate(probabilities):
        reg = regimes[i] if i < len(regimes) else "neutral"
        if reg == "trend":
            # In upward trending markets: stay heavily invested to match/beat Buy & Hold
            positions[i] = 1.0 if probability >= 0.46 else 0.7
        elif reg == "defensive":
            # Macro downtrend: only enter if high probability of a relief bounce
            positions[i] = 0.5 if probability >= 0.60 else 0.0
        elif reg == "high_volatility":
            # Choppy / volatile: scale exposure
            positions[i] = 0.5 if probability >= threshold else 0.0
        else:  # neutral
            positions[i] = 1.0 if probability >= threshold else (0.5 if probability >= 0.48 else 0.0)
    return positions


def run_walk_forward(df, benchmark_df, capital, tc_rate, epochs=10,
                     horizon=5, folds=3, baseline_exposure=0.75,
                     rf_annual=0.0):
    """Evaluate LSTM and Random Forest classifiers on expanding walk-forward folds."""
    benchmark = benchmark_df["Close"].reindex(df.index).ffill() if benchmark_df is not None else None
    dataset, target_dates = data.build_dataset(df, benchmark, horizon, tc_rate)
    feature_values = dataset[FEATURES].to_numpy()
    labels = dataset["target"].to_numpy()
    returns = dataset["forward_return"].to_numpy()
    regimes = dataset["regime"].to_numpy()
    n = len(dataset)
    first_test = max(int(n * 0.55), LOOKBACK_WINDOW + 50)
    block = max((n - first_test) // max(folds, 1), 1)

    evaluation_rows = np.arange(first_test, n, horizon)
    outputs = {
        "lstm": [], "random_forest": [], "hybrid_lstm": [],
        "hybrid_random_forest": [], "buy_and_hold": [], "momentum": [],
    }
    diagnostics = []

    for fold in range(folds):
        test_start = first_test + fold * block
        test_end = n if fold == folds - 1 else min(test_start + block, n)
        if test_end <= test_start:
            continue
        train_end = test_start
        train_cut = train_end - horizon
        scaler = StandardScaler().fit(feature_values[:train_end])
        scaled = scaler.transform(feature_values)
        x_seq, y_seq = model.make_sequences(scaled, labels, LOOKBACK_WINDOW)
        end_rows = np.arange(LOOKBACK_WINDOW - 1, n)
        train_mask = end_rows < train_cut

        test_rows = evaluation_rows[
            (evaluation_rows >= test_start) & (evaluation_rows < test_end)
        ]
        test_mask = np.isin(end_rows, test_rows)
        x_train, y_train = x_seq[train_mask], y_seq[train_mask]
        x_test = x_seq[test_mask]
        if len(x_train) < 20 or len(x_test) == 0:
            continue
        val_cut = max(int(len(x_train) * 0.85), 1)
        lstm, _ = model.train_lstm_classifier(
            x_train[:val_cut], y_train[:val_cut],
            x_train[val_cut:], y_train[val_cut:], epochs
        )
        lstm_prob = model.predict_classifier(lstm, x_test)
        flat_train = scaled[LOOKBACK_WINDOW - 1:train_cut]
        flat_test = scaled[test_rows]
        rf = RandomForestClassifier(
            n_estimators=150, max_depth=6, min_samples_leaf=6,
            class_weight="balanced", random_state=SEED, n_jobs=1
        )
        rf.fit(flat_train, labels[LOOKBACK_WINDOW - 1:train_cut])
        rf_probabilities = rf.predict_proba(flat_test)
        positive_class = np.flatnonzero(rf.classes_ == 1)
        rf_prob = (
            rf_probabilities[:, positive_class[0]]
            if len(positive_class) else np.zeros(len(flat_test))
        )

        test_returns = returns[test_rows]
        test_regimes = regimes[test_rows]
        for name, probabilities in (("lstm", lstm_prob), ("random_forest", rf_prob)):
            positions = _positions(probabilities, test_regimes)
            outputs[name].append((test_returns, positions))
            hybrid_positions = baseline_exposure + (1 - baseline_exposure) * positions
            outputs[f"hybrid_{name}"].append((test_returns, hybrid_positions))

        diagnostics.append({
            "dates": target_dates[test_rows],
            "lstm_probability": lstm_prob,
            "random_forest_probability": rf_prob,
            "actual_label": labels[test_rows],
            "regime": test_regimes,
            "hybrid_lstm_position": baseline_exposure + (1 - baseline_exposure) * _positions(lstm_prob, test_regimes),
            "hybrid_random_forest_position": baseline_exposure + (1 - baseline_exposure) * _positions(rf_prob, test_regimes),
            "forward_return": test_returns,
        })
        outputs["buy_and_hold"].append(
            (test_returns, np.ones(len(test_returns)))
        )
        momentum = (dataset["momentum_20"].to_numpy()[test_rows] > 0).astype(float)
        outputs["momentum"].append((test_returns, momentum))

    summary = {}
    for name, fold_metrics in outputs.items():
        if not fold_metrics:
            summary[name] = {
                "final_wealth": float(capital), "profit": 0.0, "transaction_cost": 0.0,
                "trades": 0, "cagr": 0.0, "volatility": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "max_drawdown": 0.0, "win_rate": 0.0, "turnover": 0.0, "folds": 0
            }
            continue
        all_returns = np.concatenate([item[0] for item in fold_metrics])
        all_positions = np.concatenate([item[1] for item in fold_metrics])
        summary[name] = _metrics(
            all_returns, all_positions, capital, tc_rate, horizon, rf_annual
        )
        summary[name]["folds"] = len(fold_metrics)

    return {
        "horizon": horizon,
        "baseline_exposure": baseline_exposure,
        "features": FEATURES,
        "benchmark": "market-relative",
        "walk_forward": summary,
        "_diagnostics": diagnostics,
    }
