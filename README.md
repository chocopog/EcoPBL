# EcoPBL: LSTM Stock Market Simulation

EcoPBL is an economics project that combines stock-market data, demand-and-supply
proxies, machine learning, trading costs, and investor utility in an interactive
Flask dashboard.

The application evaluates an LSTM model and a Random Forest classifier against
buy-and-hold and momentum baselines. It supports market-relative 5-trading-day
and 10-trading-day targets, regime-aware allocation, walk-forward testing, and
diagnostic plots.

> **Important:** This project is for education and experimentation. It is not
> financial advice, and model results do not guarantee future performance. m

## Features

- Yahoo Finance price and volume ingestion with a deterministic synthetic-data
  fallback when data cannot be downloaded.
- Causal feature engineering:
  - price changes and moving-average gaps
  - momentum over 5, 10, 20, and 50 trading days
  - rolling volatility
  - RSI and volume momentum
  - drawdown and moving-average slope
  - benchmark return, trend, volatility, and relative strength
- Market-relative classification targets for 5-day and 10-day forward returns.
- LSTM classifier and Random Forest classifier.
- Expanding walk-forward evaluation with non-overlapping holding periods.
- Regime labels:
  - `trend`
  - `neutral`
  - `high_volatility`
  - `defensive`
- Transaction-cost-aware positions and hybrid baseline exposure.
- Risk and performance metrics including:
  - final wealth and profit
  - transaction costs and trade count
  - CAGR, volatility, Sharpe, and Sortino ratios
  - maximum drawdown, win rate, and turnover
- Flask dashboard with model plots, economic interpretation, and diagnostics.
- Jupyter notebook for reproducible analysis and report generation.
- Docker configuration for container-based hosting.

## Project structure

| Path | Purpose |
| --- | --- |
| [`app.py`](./app.py) | Flask application and `/api/simulate` endpoint |
| [`config.py`](./config.py) | Tickers, model defaults, trading costs, and feature list |
| [`data.py`](./data.py) | Data loading, synthetic fallback, features, targets, and regimes |
| [`model.py`](./model.py) | PyTorch LSTM models, training, sequence creation, and inference |
| [`strategy.py`](./strategy.py) | Walk-forward evaluation, Random Forest, allocation, and metrics |
| [`plots.py`](./plots.py) | Dashboard and diagnostic plot generation |
| [`templates/index.html`](./templates/index.html) | Dashboard UI |
| [`LSTMmodel.ipynb`](./LSTMmodel.ipynb) | Notebook-based analysis and formatted report |
| [`Dockerfile`](./Dockerfile) | Container image definition |
| [`.dockerignore`](./.dockerignore) | Files excluded from Docker build context |
| [`requirements.txt`](./requirements.txt) | Python dependencies |

Generated plots are written to `static/plots/` for the Flask dashboard. They are
ignored by Git because they are regenerated for each simulation. The static
economic illustrations in `eco_plots/` are retained as project assets.

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

Start the Flask application:

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000).

The application reads `PORT` when provided. Debug mode is controlled with
`FLASK_DEBUG`:

```bash
FLASK_DEBUG=1 python app.py
```

For production-style local execution, install a WSGI server separately and
run:

```bash
gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 2 --timeout 120 app:app
```

One worker is recommended because each simulation performs CPU- and
memory-intensive model training.

## Docker

Build the image:

```bash
docker build -t ecopbl .
```

Run it:

```bash
docker run --rm -p 5002:5002 ecopbl
```

Then open [http://localhost:5002](http://localhost:5002).

The image uses the Python slim base image, exposes port `5002`, and starts the
app with Gunicorn on `0.0.0.0:5002`.

Alternatively, if Docker Compose is installed:

```bash
docker compose up --build
```

The Compose service is then available at
[http://localhost:5002](http://localhost:5002). Stop it with `Ctrl+C`, or run
`docker compose down` from another terminal.

## Dashboard API

### `GET /api/stocks`

Returns the configured stock catalog.

### `POST /api/simulate`

Runs the full pipeline and returns JSON containing plot URLs, regression
metrics, economic findings, and walk-forward results.

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

Available configured tickers include `AAPL`, `META`, `NVDA`, `MSFT`, `AMZN`,
`GOOGL`, `TSLA`, several `.NS` Indian equities, `^NSEI`, and `^GSPC`.
Custom Yahoo Finance tickers can also be supplied through the dashboard.

## Notebook workflow

Open [`LSTMmodel.ipynb`](./LSTMmodel.ipynb) after installing the dependencies.
Run the cells in order. The notebook imports the shared implementation from
`data.py`, `model.py`, `strategy.py`, and `plots.py`, so results are consistent
with the Flask application.

If a notebook cell shows stale results or missing diagnostics, restart the
kernel and rerun the walk-forward evaluation cell before the reporting cell.

## Evaluation notes

The forward target is based on the stock's return relative to an automatically
selected benchmark:

- `.NS` and `.BO` tickers use `^NSEI`.
- Other tickers use `^GSPC`.

The target is positive when:

```text
stock forward return - benchmark forward return > 2 × transaction-cost rate
```

The evaluator samples holding periods on a global horizon grid. This prevents
overlapping 5-day or 10-day forward returns from being compounded as though
they were independent daily returns.

The model is not expected to outperform buy-and-hold consistently. The
evaluation is intended to compare returns, costs, drawdowns, and risk-adjusted
performance fairly rather than promise alpha.

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

If the service is containerized, allocate at least 1–2 GB RAM for the current
pipeline when possible.

## Development checks

Compile the Python modules:

```bash
python -m py_compile app.py config.py data.py model.py strategy.py plots.py
```

Validate the notebook JSON:

```bash
python -m json.tool LSTMmodel.ipynb > /dev/null
```
