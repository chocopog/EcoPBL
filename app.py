"""Flask Web Application for JIIT Eco PBL: Interactive LSTM Stock Simulation.

Authors: dhruv sarthak usaid A8
"""

import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.preprocessing import StandardScaler

import config
import data
import model
import plots
import strategy

app = Flask(__name__)
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "static", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


@app.route("/")
def index():
    """Render main interactive simulation dashboard."""
    return render_template(
        "index.html",
        stocks=config.AVAILABLE_STOCKS,
        default_ticker="NVDA",
        default_epochs=30,
        default_capital=100_000,
        default_start="2018-01-01",
        default_end="2026-10-31",
    )


@app.route("/api/stocks")
def get_stocks():
    """Return available stock catalog."""
    return jsonify(config.AVAILABLE_STOCKS)


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Run full LSTM training pipeline, generate 8 plots, walk-forward diagnostics, and economic report."""
    try:
        req = request.get_json() or {}

        # 1. Parse parameters
        custom_ticker = req.get("custom_ticker", "").strip().upper()
        ticker = custom_ticker if custom_ticker else req.get("ticker", "NVDA").strip().upper()
        epochs = max(5, min(int(req.get("epochs", 30)), 100))
        start_date = req.get("start_date", "2018-01-01").strip()
        end_date = req.get("end_date", "2026-10-31").strip()
        capital = float(req.get("capital", config.SCALE_W))
        tc_rate = float(req.get("tc_rate", config.TC_RATE))

        meta = config.get_stock_meta(ticker)
        currency = meta["currency"]
        run_id = f"{ticker}_{int(time.time() * 1000)}"

        # 2. Ingest market data (Live Yahoo Finance with simulated fallback)
        df_raw, stock_label, is_simulated = data.load_data(ticker, start_date, end_date)
        if len(df_raw) < 80:
            return jsonify({
                "success": False,
                "error": f"Insufficient historical records for {ticker} ({len(df_raw)} trading days found). Please select an earlier start date."
            }), 400

        # Market Benchmark
        benchmark_symbol = strategy.benchmark_ticker(ticker)
        benchmark_raw, _, benchmark_simulated = data.load_data(
            benchmark_symbol, start_date, end_date
        )

        # 3. Walk-Forward Multi-Horizon Evaluation (5-day & 10-day)
        horizons = [5, 10]
        walk_forward = {}
        diagnostic_plots = {}
        try:
            for horizon in horizons:
                wf_result = strategy.run_walk_forward(
                    df_raw, benchmark_raw, capital, tc_rate, epochs=max(3, min(epochs, 15)), horizon=horizon
                )
                diag_items = wf_result.pop("_diagnostics", [])
                walk_forward[str(horizon)] = wf_result
                filename = f"{run_id}_{horizon}_day_diagnostics.png"
                plots.plot_walk_forward_diagnostics(
                    diag_items,
                    wf_result["walk_forward"],
                    currency,
                    os.path.join(PLOTS_DIR, filename),
                )
                diagnostic_plots[str(horizon)] = f"/static/plots/{filename}"
        except Exception as wf_err:
            print(f"Notice: Walk-forward evaluation encountered: {wf_err}")
            walk_forward = {}
            diagnostic_plots = {}

        # 4. Feature engineering for full dataset
        benchmark_series = benchmark_raw["Close"] if benchmark_raw is not None else None
        dataset, target_dates = data.build_dataset(df_raw, benchmark=benchmark_series, horizon=5, tc_rate=tc_rate)
        n_total = len(dataset)
        split_idx = int(n_total * (1 - config.TEST_FRAC))
        split_date = dataset.index[split_idx]

        # 5. Save initial market & theory plots
        p1_file = f"{run_id}_01_price_history.png"
        p2_file = f"{run_id}_02_excess_demand.png"
        p3_file = f"{run_id}_03_supply_demand.png"

        plots.plot_price_history(dataset, split_date, stock_label, currency, os.path.join(PLOTS_DIR, p1_file))
        plots.plot_excess_demand(dataset, stock_label, currency, os.path.join(PLOTS_DIR, p2_file))
        plots.plot_supply_demand(currency, os.path.join(PLOTS_DIR, p3_file))

        # 6. Data preprocessing (strict chronological split, no lookahead bias)
        scaler_x = StandardScaler().fit(dataset[config.FEATURES].values[:split_idx])
        # Use target_return (% price change tomorrow) for regression predictions
        target_col = "target_return" if "target_return" in dataset.columns else "price_change"
        scaler_y = StandardScaler().fit(dataset[[target_col]].values[:split_idx])
        target_scale = float(scaler_y.scale_[0])

        X_scaled = scaler_x.transform(dataset[config.FEATURES].values)
        y_scaled = scaler_y.transform(dataset[[target_col]].values).ravel()

        X_seq, y_seq = model.make_sequences(X_scaled, y_scaled, config.LOOKBACK_WINDOW)
        end_rows = np.arange(config.LOOKBACK_WINDOW - 1, n_total)
        train_mask = end_rows < split_idx

        X_train_full, y_train_full = X_seq[train_mask], y_seq[train_mask]
        X_test, y_test = X_seq[~train_mask], y_seq[~train_mask]

        val_cutoff = int(len(X_train_full) * (1 - config.VAL_FRAC))
        X_train, y_train = X_train_full[:val_cutoff], y_train_full[:val_cutoff]
        X_val, y_val = X_train_full[val_cutoff:], y_train_full[val_cutoff:]

        # 7. Train PyTorch LSTM Model
        trained_model, train_stats = model.train_lstm(
            X_train, y_train, X_val, y_val, epochs, target_scale
        )

        p4_file = f"{run_id}_04_training_curve.png"
        plots.plot_training_curve(train_stats["train_rmse"], train_stats["val_rmse"], os.path.join(PLOTS_DIR, p4_file))

        # 8. Model evaluation on unseen test period
        raw_preds = model.predict(trained_model, X_test)
        pred_returns = scaler_y.inverse_transform(raw_preds.reshape(-1, 1)).ravel()
        actual_returns = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

        test_row_indices = end_rows[~train_mask]
        test_dates = target_dates[test_row_indices]
        today_prices = dataset["close"].values[test_row_indices]

        actual_prices = today_prices * (1 + actual_returns)
        predicted_prices = today_prices * (1 + pred_returns)
        naive_prices = today_prices.copy()

        # Compute regression metrics
        rmse_func = lambda actual, pred: float(np.sqrt(np.mean((actual - pred) ** 2)))
        mae_func = lambda actual, pred: float(np.mean(np.abs(actual - pred)))

        return_rmse_lstm = rmse_func(actual_returns, pred_returns) * 100
        return_rmse_naive = rmse_func(actual_returns, 0) * 100
        return_mae_lstm = mae_func(actual_returns, pred_returns) * 100
        price_rmse_lstm = rmse_func(actual_prices, predicted_prices)
        price_rmse_naive = rmse_func(actual_prices, naive_prices)
        directional_accuracy = float(np.mean(np.sign(pred_returns) == np.sign(actual_returns))) * 100
        market_up_rate = float(np.mean(actual_returns > 0)) * 100

        p5_file = f"{run_id}_05_predicted_vs_actual.png"
        p6_file = f"{run_id}_06_forecast_errors.png"

        plots.plot_predicted_vs_actual(
            dataset, split_idx, test_dates, actual_prices, predicted_prices,
            naive_prices, stock_label, currency, os.path.join(PLOTS_DIR, p5_file)
        )
        plots.plot_errors(
            test_dates, actual_returns, pred_returns, actual_prices, predicted_prices,
            currency, os.path.join(PLOTS_DIR, p6_file)
        )

        # 9. Validation-adaptive strategy parameter selection (with hybrid bull-market participation)
        val_raw_preds = model.predict(trained_model, X_val)
        val_pred_ret = scaler_y.inverse_transform(val_raw_preds.reshape(-1, 1)).ravel()
        val_true_ret = scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel()
        strategy_params = plots.select_strategy_params(val_pred_ret, val_true_ret, tc_rate, capital)

        # 10. Backtest: Revenue, Cost and Profit (Firm Economics)
        p7_file = f"{run_id}_07_backtest.png"
        _, strat_summary, bh_summary = plots.plot_backtest(
            test_dates, actual_returns, pred_returns, capital, tc_rate, currency,
            os.path.join(PLOTS_DIR, p7_file), strategy_params=strategy_params
        )

        # 11. Investor Utility Analysis
        p8_file = f"{run_id}_08_utility.png"
        _, du_strat, du_bh = plots.plot_utility(
            capital, strat_summary["final_wealth"], bh_summary["final_wealth"],
            currency, config.SCALE_W, os.path.join(PLOTS_DIR, p8_file)
        )

        # 12. PBL Economics findings
        train_data = dataset.iloc[:split_idx]
        shortage_ret = float(train_data.loc[train_data["excess_demand"] > 0, "target_return"].mean() * 100) if "target_return" in train_data else 0.0
        surplus_ret = float(train_data.loc[train_data["excess_demand"] < 0, "target_return"].mean() * 100) if "target_return" in train_data else 0.0

        return jsonify({
            "success": True,
            "run_id": run_id,
            "ticker": ticker,
            "label": stock_label,
            "is_simulated": is_simulated,
            "currency": currency,
            "capital": capital,
            "benchmark": benchmark_symbol,
            "benchmark_is_simulated": benchmark_simulated,
            "transaction_cost_rate": tc_rate,
            "data_summary": {
                "total_days": n_total,
                "train_days": split_idx,
                "test_days": n_total - split_idx,
                "start_date": str(dataset.index[0].date()),
                "end_date": str(dataset.index[-1].date()),
            },
            "training_metrics": {
                "epochs_run": train_stats["epochs_run"],
                "best_epoch": train_stats["best_epoch"],
                "final_train_rmse": train_stats["final_train_rmse"],
                "best_val_rmse": train_stats["best_val_rmse"],
                "train_rmse_history": train_stats["train_rmse"],
                "val_rmse_history": train_stats["val_rmse"],
            },
            "forecast_metrics": {
                "return_rmse_lstm": round(return_rmse_lstm, 4),
                "return_rmse_naive": round(return_rmse_naive, 4),
                "return_mae_lstm": round(return_mae_lstm, 4),
                "price_rmse_lstm": round(price_rmse_lstm, 2),
                "price_rmse_naive": round(price_rmse_naive, 2),
                "directional_accuracy": round(directional_accuracy, 1),
                "market_up_rate": round(market_up_rate, 1),
            },
            "economic_backtest": {
                "lstm": strat_summary,
                "buy_and_hold": bh_summary,
                "avg_revenue_per_trade": round(strat_summary["tr"] / max(strat_summary["trades"], 1), 2),
                "marginal_cost_per_trade": round(capital * tc_rate, 2),
                "round_trip_cost_rate": round(2 * tc_rate, 6),
                "strategy_params": strategy_params,
            },
            "utility_metrics": {
                "delta_u_lstm": round(du_strat, 4),
                "delta_u_bh": round(du_bh, 4),
            },
            "pbl_findings": {
                "shortage_ret": round(shortage_ret, 3),
                "surplus_ret": round(surplus_ret, 3),
                "law_of_demand_supported": shortage_ret > surplus_ret,
            },
            "plots": {
                "price_history": f"/static/plots/{p1_file}",
                "excess_demand": f"/static/plots/{p2_file}",
                "supply_demand": f"/static/plots/{p3_file}",
                "training_curve": f"/static/plots/{p4_file}",
                "predicted_vs_actual": f"/static/plots/{p5_file}",
                "forecast_errors": f"/static/plots/{p6_file}",
                "backtest": f"/static/plots/{p7_file}",
                "utility": f"/static/plots/{p8_file}",
            },
            "walk_forward_results": walk_forward,
            "diagnostic_plots": diagnostic_plots,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host="0.0.0.0", port=port, debug=debug)
