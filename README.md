# LSTM STOCK ANALYZER

This is an economics project for 3rd semester Economics class (3 credit for some reason) which analyzes stock market data using concepts from our course: demand and supply, firm theory, cost and production, utility and consumer choice, and forecasting methods. We have trained an LSTM and a random forest model on stock data, compared it with simple baselines (elaborated later) and built both a notebook and a flask app for presentation purposes. 

> **THIS PROJECT IS NOT FINANCIAL ADVICE, MATTER OF FACT ITS PROBABLY WORSE THAN JUST DOING NOTHING**

## What it does exactly

- imports price, volume, trade etc data from Yfinance for a chosen ticker.
- builds its features from economic concepts (price change, quanitity gap, price gap, excess demand) plus some common market indicators.
- Trains the LSTM on these features, use it to forecast next day return and compare it with a naive forecast (meaning tomorrows prices are equal to todays prices)
- it also predicts a 5 and 10 day period(using walk forward test), in which it is often more successful
- **imp for the presentation** simulates a trading strategy as a firm. calculates total revenue, total cost and profit. we have assumed a 0.1% transaction cost on each trade
- compares the different strategies (LSTM, random forest, buy and hold, momentum) on marginal utility and risk aversion.
- draws an investor-choice diagram with indifference curves and a budget line (dicey)

### model is not expected to beat buy and hold strategies, only to compare returns losses costs and risk fairly. ***Goal is not maximize profit.*** 

## Concepts used which were also taught in class

| Course area | Where it appears |
|---|---|
| Demand and supply | Supply and demand diagram (equilibrium, surplus, shortage, elasticity); `excess_demand` feature (buyers vs sellers, based on traded volume) |
| Production / firm theory | Strategy treated as a firm: TR (gross gains), TC (transaction cost x trades), profit = TR - TC; average revenue per trade vs cost per trade |
| Consumer choice | Utility function `U(W) = sqrt(W / C)` and diminishing marginal utility; investor-choice plot (indifference curves and budget line between the stock and a safe asset) |
| Regression | RMSE and MAE of the LSTM vs the naive forecast; regression of next-day return on excess demand; regression of today's return on yesterday's; regression of actual on forecast (a perfect forecast has a = 0, b = 1); t-tests on the slopes and on the error difference |
| Forecasting / time series | Naive forecast as the baseline; 50-day moving average as the trend (equilibrium level); forecast error over time |

Other terms used:
- Opportunity cost
- Risk
- Return per unit of risk: (average return - opportunity cost) / risk
- Worst fall from peak: largest drop from highest wealth seen so far
- Hybrid strategy: 75% always held, and 25% moved (by the model)
- Market-relative target: wether stock beats the market index (like nifty or s&p) over 5 to 10 days by more than twice the transaction cost (0.01%)

## Features
- Yahoo finance price and volume download with open and close, theres a synthetic data fallback as well if the download fails (copilot added it, so im not sure of its efficacy) and simulated data will be labeled `SIMULATED` in the output.
- The features built from historical data:
    - price change, quantity gap, excess demand, price gap
    - volatility (rolling)
    - RSI and volume momentum
    - drawdown and moving average graphs (slopes)
    - benchmark return (naive), trend and relative strength
- next-day return regression with LSTM, compared with tomorrow = today forecast
- Market relative classification targets for 5-day and 10-day returns
- LSTM classifier and random forest classifier (different from regression)
- Walk forward evaluation with 3 folds and non overlapping holding periods
- market regime labels: trend, netural, high_volatility, defensive.
- transaction-cost aware poisitions and the aforementioned hybrid strategy with 75% holding
- Performance metrics:
    - final wealth
    - profit
    - total cost
    - trade count
    - average annual return
    - risk
    -return per unit of risk 
    - worst fall from peak
    - trading frequency
    - share of periods with profit (added by copilot, dont quite understand it)
- Both a website/app for polished display and a notebook for stepwise analysis
- Docker config for containerized hosting (Redundant, ignore)

| Path | Purpose |
|------|---------|
| `app.py` | Web application and simulation endpoint for the api|
| `config.py` | Tickers, hyperparameters, trading costs, risk-free rates and feature list |
| `data.py` | Data loading, fake fallback, features, targets and regimes |
| `model.py` | LSTM model, training, sequence creation and inference |
| `strategy.py` | Walk-forward evaluation, Random Forest, and metrics |
| `plots.py` | All the plots |
| `templates/index.html` | the UI|
| `LSTMmodel.ipynb` | Notebook with the full analysis (self-contained, contains code from all the other files |
| `Dockerfile` | Container image definition (redundant) |
| `.dockerignore` | Files excluded from the Docker build context (redundant) |
| `requirements.txt` | Python dependencies |

#Plots are stored in eco_plots as well as the notebook

**TBH ITS NOT REALLY NECESSARY BUT IF U WANT TO YOU CAN DO THIS**
# SETUP FOR INSTALL
### Step1:
make sure you have python installed, i used python 3.14 and it worked on it
### Step2:
open my github, jo link i sent and fork the repo, (between watch and star)
### Step3:
oull have to connect the fork you made, so go to your own accounts repos and copy the forked repo's link
### Step4: 
Create a project folder, then open it in VsCode or whatever IDE you use and in the terminal
` git clone [the link you copied]` this will fork the repo in a new folder
then move into the folder in terminal by `cd [foldername]`
### Step5:
Abhi you got the repo but if i make any changes in the future so wo bhi aap tk pohochne ke liye youll have to connect the fork you made with my original repo
in the terminal: `git remote add upstream https://github.com/chocopog/EcoPBL.git`

### Step6:
to pull future changes: 
`git fetch upstream`
to merge the changes into ur fork
`git checkout main`
`git merge upstream/main`
`git push origin main`

# SETUP TO RUN IT (make sure ur in the same folder as the project)
` python -m venv .venv`
for mac: 
` source .venv/bin/activate`
for windows
` .venv\Scripts\activate`
pip install -r requirements.txt

### you need Vscode extensions for python and jupyter so when it prompts you install those

## For the flask app:
` python app.py` 
and open it on whatevr localhost port it launches



## Notebook workflow
LSTMmodel.ipynb is self contained; it includes all the necessary code. **Warning** will run slow on devices without a GPU so wait will be long

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




## KNOWN LIMITATIONS
- Results come from one test window per stock, so they depend on that period.
- The position sizes and probability cutoffs used by the allocation rule (for example 0.46, 0.48, 0.50, 0.60) are set by hand and are not fitted on validation data.
- The notebook's daily-forecast strategy uses a fixed position size (profits are not reinvested), while the walk-forward wealth is compounded, so the two are not directly comparable.
- If the stock download fails, the app falls back to simulated data, which is only for testing the code.


