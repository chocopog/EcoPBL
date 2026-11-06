# EcoPBL: LSTM Stock Market Simulation

EcoPBL is an economics project (B.Tech II year, Economics PBL) that studies stock-market data using concepts from the course: demand and supply, firm theory (revenue, cost, profit), utility and consumer choice, regression, and forecasting. It trains an LSTM (and a Random Forest) on stock data, compares them with simple baselines, and shows everything in a Jupyter notebook and an interactive Flask dashboard.

> **Important:** this project is for education and experimentation. It is not financial advice, and model results do not guarantee future performance.

## What the project does

- Downloads daily price and volume data from Yahoo Finance for a chosen stock.
- Builds features from economics ideas (price change, quantity gap, price gap, excess demand) plus common market indicators.
- Trains an LSTM to forecast the next-day return and compares it with the naive forecast (tomorrow = today).
- Simulates a trading strategy as a firm: total revenue (TR), total cost (TC, transaction cost) and profit.
- Compares strategies using utility (diminishing marginal utility, risk aversion), and draws an investor-choice diagram with indifference curves and a budget line.
- Runs a walk-forward test with market-relative 5-day and 10-day targets, using an LSTM classifier and a Random Forest classifier, against buy-and-hold and momentum.

The model is not expected to beat buy-and-hold consistently. The aim is to compare returns, costs, losses and risk fairly, not to promise profit.

## Course concepts used

| Course area | Where it appears |
|---|---|
| Demand and supply | Supply and demand diagram (equilibrium, surplus, shortage, elasticity); `excess_demand` feature (buyers vs sellers, based on traded volume) |
| Production / firm theory | Strategy treated as a firm: TR (gross gains), TC (transaction cost x trades), profit = TR - TC; average revenue per trade vs cost per trade |
| Consumer choice | Utility function `U(W) = sqrt(W / C)` and diminishing marginal utility; investor-choice plot (indifference curves and budget line between the stock and a safe asset) |
| Regression | RMSE and MAE of the LSTM vs the naive forecast; regression of next-day return on excess demand; regression of today's return on yesterday's; regression of actual on forecast (a perfect forecast has a = 0, b = 1); t-tests on the slopes and on the error difference |
| Forecasting / time series | Naive forecast as the baseline; 50-day moving average as the trend (equilibrium level); forecast error over time |

Other terms used in the results tables:

- **Opportunity cost**: what the money would have earned in a safe asset (a rough annual rate, set per market).
- **Risk**: standard deviation of returns.
- **Return per unit of risk**: (average return - opportunity cost) / risk. The downside version only counts the bad periods as risk.
- **Worst fall from peak**: the largest drop from the highest wealth reached so far.
- **Hybrid strategy**: 75% always held in the stock (passive) and 25% moved by the model.
- **Market-relative target**: whether the stock beats the market index (S&P 500 or Nifty) over the next 5 or 10 days by more than twice the transaction-cost rate.

## Features

- Yahoo Finance price and volume download, with a deterministic synthetic-data fallback when the download fails. Simulated data is labelled `SIMULATED` in the output.
- Features built only from past data:
  - price change, quantity gap, price gap, excess demand
  - momentum over 5, 10, 20 and 50 trading days
  - rolling volatility
  - RSI and volume momentum
  - drawdown and moving-average slope
  - benchmark return, trend, volatility and relative strength
- Next-day return regression with an LSTM, compared with the naive forecast.
- Market-relative classification targets for 5-day and 10-day forward returns.
- LSTM classifier and Random Forest classifier.
- Expanding walk-forward evaluation (3 folds) with non-overlapping holding periods.
- Market regime labels: `trend`, `neutral`, `high_volatility`, `defensive`.
- Transaction-cost-aware positions and a hybrid strategy with 75% baseline exposure.
- Performance measures: final wealth, profit, total cost and trade count, average annual return, risk, return per unit of risk (and downside risk), worst fall from peak, share of periods with profit, and trading frequency.
- Flask dashboard with model plots and diagnostics.
- Jupyter notebook for the step-by-step analysis.
- Docker configuration for container-based hosting.

## Project structure

| Path | Purpose |
|------|---------|
| `app.py` | Flask application and `/api/simulate` endpoint |
| `config.py` | Tickers, model defaults, trading costs, risk-free rates and feature list |
| `data.py` | Data loading, synthetic fallback, features, targets and regimes |
| `model.py` | PyTorch LSTM models, training, sequence creation and inference |
| `strategy.py` | Walk-forward evaluation, Random Forest, allocation and metrics |
| `plots.py` | Dashboard and diagnostic plot generation |
| `templates/index.html` | Dashboard UI |
| `LSTMmodel.ipynb` | Notebook with the full analysis (self-contained, see below) |
| `Dockerfile` | Container image definition |
| `.dockerignore` | Files excluded from the Docker build context |
| `requirements.txt` | Python dependencies |

Generated plots for the dashboard are written to `static/plots/`. They are ignored by Git because they are regenerated for each simulation. The notebook saves its figures in `eco_plots/`, which are kept as project assets.

## Local setup

Python 3.12 is recommended.

```bash
git clone <repository-url>
cd EcoPBL

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The code needs PyTorch, scikit-learn, SciPy, pandas, NumPy, Matplotlib, yfinance and Flask. For the notebook you also need Jupyter.

Start the Flask application:

```bash
python app.py
```

Open http://localhost:5000.

The application reads `PORT` when provided. Debug mode is controlled with `FLASK_DEBUG`:

```bash
FLASK_DEBUG=1 python app.py
```

For production-style local execution, install a WSGI server separately and run:

```bash
gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 2 --timeout 120 app:app
```

One worker is recommended because each simulation performs CPU- and memory-intensive model training.

## Docker

Build the image:

```bash
docker build -t ecopbl .
```

Run it:

```bash
docker run --rm -p 5002:5002 ecopbl
```

Then open http://localhost:5002.

The image uses the Python slim base image, exposes port 5002, and starts the app with Gunicorn on `0.0.0.0:5002`.

Alternatively, if Docker Compose is installed:

```bash
docker compose up --build
```

The Compose service is then available at http://localhost:5002. Stop it with Ctrl+C, or run `docker compose down` from another terminal.

## Dashboard API

### `GET /api/stocks`

Returns the configured stock catalog.

### `POST /api/simulate`

Runs the full pipeline and returns JSON containing plot URLs, regression metrics, economic findings, and walk-forward results.

Example request:

```bash
curl -X POST http://localhost:5000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "epochs": 10,
    "capital": 100000,
    "start_date": "2018-01-01",
    "end_date": "2026-10-31"
  }'
```

Available configured tickers include AAPL, META, NVDA, MSFT, AMZN, GOOGL, TSLA, several `.NS` Indian equities, `^NSEI`, and `^GSPC`. Custom Yahoo Finance tickers can also be supplied through the dashboard.

In the walk-forward results, `cagr`, `volatility`, `max_drawdown`, `win_rate` and `turnover` are returned as fractions (0.63 means 63%). The dashboard multiplies them by 100 for display. The JSON keys keep their original names; only the labels shown in the dashboard use the plain-language names above.

## Notebook workflow

`LSTMmodel.ipynb` is self-contained: the data, model, strategy and plotting code used by the Flask app is embedded in one cell (names start with `embedded_` / `EMBEDDED_`), so the notebook does not import the `.py` files. Run the cells in order. It uses a GPU if one is available and the CPU otherwise.

The notebook has two parts:

1. **Step-by-step analysis** on one stock (set `TICKER` and `EPOCHS` in the configuration cell):
   - price history with the 50-day moving average, excess demand plot, supply and demand diagram
   - 60-day sliding windows, LSTM training, learning curve
   - test-period forecast vs actual price and vs the naive forecast, forecast errors
   - strategy as a firm (TR, TC, profit) vs buy-and-hold, and the utility analysis
2. **Printed report and walk-forward results**:
   - demand and supply regression, RMSE/MAE vs naive, firm economics, utility, and regression checks
   - 5-day and 10-day walk-forward tables with diagnostic plots
   - investor-choice plot (indifference curves and budget line) built from the 5-day results

Figures are saved in `eco_plots/` (`01_price_history.png` to `08_utility.png`, `5_day_diagnostics.png`, `10_day_diagnostics.png`, `09_investor_choice.png`).

If a cell shows stale results, restart the kernel and run all cells again from the top. The walk-forward cell stops with an error if the benchmark index cannot be downloaded, because the market-relative target would be meaningless on simulated data.

## Evaluation notes

The forward target is based on the stock's return relative to an automatically selected benchmark:

- `.NS` and `.BO` tickers use `^NSEI`.
- Other tickers use `^GSPC`.

The target is positive when:

```
stock forward return - benchmark forward return > 2 x transaction-cost rate
```

Other points about how the evaluation is done:

- **Past only.** Scalers are fit on training data only. In each walk-forward fold the last `horizon` rows before the test period are left out of training, because their labels look `horizon` days ahead and would use test-period prices.
- **No overlapping periods.** The evaluator samples holding periods on a global horizon grid, so overlapping 5-day or 10-day forward returns are not compounded as though they were independent daily returns.
- **Opportunity cost.** Return-per-unit-of-risk measures subtract an assumed annual safe-asset rate (about 6.5% for the Nifty benchmark and 4% for the S&P 500; edit it in the configuration).
- **Costs.** Transaction cost is charged on every change in position, on the wealth held at that time.
- **Reproducibility.** Random seeds are fixed, so repeating a run with the same settings gives the same numbers.

Known limitations:

- Results come from one test window per stock, so they depend on that period.
- The position sizes and probability cutoffs used by the allocation rule (for example 0.46, 0.48, 0.50, 0.60) are set by hand and are not fitted on validation data.
- The notebook's daily-forecast strategy uses a fixed position size (profits are not reinvested), while the walk-forward wealth is compounded, so the two are not directly comparable.
- If the stock download fails, the app falls back to simulated data, which is only for testing the code.

## Hosting and memory

The full simulation is deliberately computationally heavy. A request can:

- train multiple LSTM classifiers,
- fit Random Forest models,
- create several Matplotlib figures, and
- run both 5-day and 10-day walk-forward evaluations.

Small free hosting instances may run out of memory. For a constrained host:

- use fewer epochs,
- evaluate one horizon at a time,
- reduce walk-forward folds,
- reduce Random Forest trees,
- generate fewer plots, or
- move training to a background job and serve cached results.

If the service is containerized, allocate at least 1-2 GB RAM for the current pipeline when possible.

## Development checks

Compile the Python modules:

```bash
python -m py_compile app.py config.py data.py model.py strategy.py plots.py
```

Validate the notebook JSON:

```bash
python -m json.tool LSTMmodel.ipynb > /dev/null
```
