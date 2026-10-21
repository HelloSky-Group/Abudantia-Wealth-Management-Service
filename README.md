# Abudantia Wealth Management Service
A wealth management service that trades the stock market based on data and machine learning models.

> **Status: early research stage.** The project can currently download price data and train an LSTM
> model that predicts the next day's price. There is no backtesting, trading strategy or broker
> integration yet, and nothing here should be used to trade real money.

## How it works
1. **Get prices** – `get_prices.py` downloads daily adjusted close prices from Yahoo Finance and
   saves them to a csv.
2. **Preprocess** – `preprocessing.py` splits the prices chronologically into a training and a test
   set, cuts each into sliding windows of `seq_len` days paired with the price on the following day,
   and min-max scales them using the training set only (so no test data leaks into training).
3. **Train and evaluate** – `train.py` trains a 2-layer LSTM (`model.py`) on the training windows,
   predicts the test set and compares the result with a naive baseline that predicts tomorrow's
   price is today's price.

## Project structure
| File | Purpose |
|---|---|
| `get_prices.py` | Downloads prices for a ticker (`get_stock_data`) or for SPY, the S&P 500 ETF (`get_sp500`), with retries |
| `preprocessing.py` | `DataProcessing` class: train/test split, windowing and scaling |
| `model.py` | PyTorch `LSTMModel`, plus `train_model` and `predict` helpers |
| `train.py` | Command-line script that trains, evaluates and saves the model |
| `tests/` | pytest tests, run on synthetic prices |
| `requirements.txt` | Python dependencies |

## Setup
Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
### 1. Download prices
```bash
python get_prices.py
```
By default this downloads AAPL from 2010-01-01 to 2020-01-01 into `stock_data.csv`. To use another
ticker or date range, call the function directly:

```python
from get_prices import get_stock_data, get_sp500

get_stock_data('MSFT', '2015-01-01', '2024-12-31')   # -> stock_data.csv
get_sp500('2015-01-01', '2024-12-31')                # -> sp500_data.csv
```

### 2. Train and evaluate the model
```bash
python train.py --file stock_data.csv
```

| Option | Default | Meaning |
|---|---|---|
| `--file` | `stock_data.csv` | Price csv with the date in the first column and prices in the second |
| `--train` | `0.8` | Fraction of rows used for training; the rest is the test set |
| `--seq-len` | `50` | Number of past days in each input window |
| `--epochs` | `50` | Training epochs |
| `--batch-size` | `32` | Mini-batch size |
| `--lr` | `0.001` | Adam learning rate |
| `--model-path` | `model.pt` | Where the trained model and its scaling parameters are saved |

Example output (from synthetic test prices, not real market data):
```
final train loss: 0.001385
 lstm: rmse=1.7205 mae=1.3286 direction_acc=0.509
naive: rmse=0.7609 mae=0.6121 direction_acc=n/a
```

### Reading the results
- **rmse / mae** – average prediction error in price units (lower is better).
- **direction_acc** – how often the model correctly predicts whether the price goes up or down.
  0.5 is a coin flip. The naive baseline never predicts a move, so it shows `n/a`.
- **The model is only useful if it beats the naive baseline.** For daily prices that baseline is
  very hard to beat, and the current price-level model usually doesn't.

## Running the tests
```bash
python -m pytest
```
The tests use synthetic prices and need no internet access.

## Known limitations
- The model predicts raw price levels with min-max scaling fit on the training period. Prices in
  the test period that move outside that range are extrapolated badly.
- The model uses past prices of a single ticker as its only input.
- The results don't account for trading costs, slippage or taxes.
- Yahoo Finance data is unofficial and can be rate-limited or change without notice.

## Roadmap
- [ ] Predict daily returns instead of price levels
- [ ] Add more features (volume, technical indicators, market index)
- [ ] Walk-forward validation across several time periods
- [ ] Backtest a trading strategy built on the predictions, including costs
- [ ] Portfolio construction and risk management across several tickers
- [ ] Paper trading through a broker API
- [ ] Package it as a service (configuration, scheduling, API)

## Disclaimer
This project is for research and education. It is not financial advice, and past performance does
not predict future results.
