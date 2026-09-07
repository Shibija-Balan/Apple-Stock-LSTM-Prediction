# Apple Stock LSTM Prediction

A time-series forecasting project using stacked Long Short-Term Memory (LSTM) neural networks to model Apple stock mid-prices from 2020 to 2025.

## Project overview

The project calculates daily Apple mid-prices from historical high and low prices, constructs 60-day rolling sequences and trains a stacked LSTM network to predict the next trading day's mid-price.

The repository now includes a separate leakage-free evaluation pipeline that downloads AAPL data programmatically, preserves chronological ordering and benchmarks the LSTM against a naive persistence forecast.

## Model

- 60-day rolling input window
- Two stacked LSTM layers with 70 units each
- 20% dropout after each LSTM layer
- Dense one-unit output layer
- Adam optimiser
- Mean-squared-error training loss
- 25 epochs and batch size of 32

## Evaluation methodology

`evaluate_model.py` provides a more rigorous out-of-sample evaluation than the original exploratory notebook:

1. Downloads AAPL OHLC data from 1 January 2020 to 17 January 2025 using `yfinance`
2. Calculates daily mid-price as `(High + Low) / 2`
3. Uses a chronological 80/20 train-test split rather than random sampling
4. Fits the MinMax scaler **only on the training period**, preventing test-period information from leaking into preprocessing
5. Trains the same two-layer LSTM architecture used in the notebook
6. Evaluates predictions in original US-dollar price units
7. Reports out-of-sample RMSE, MAE and directional accuracy
8. Compares the model against a naive persistence benchmark where the next day's predicted mid-price equals the previous day's observed mid-price
9. Reports RMSE and MAE skill relative to that benchmark

This benchmark matters because stock-price levels are highly persistent. A prediction line can visually track observed prices while still failing to outperform the simple assumption that tomorrow's price will equal today's price.

## Performance outputs

Running:

```bash
python evaluate_model.py
```

produces `metrics.json` containing:

- LSTM RMSE (USD)
- LSTM MAE (USD)
- Naive persistence RMSE (USD)
- Naive persistence MAE (USD)
- RMSE skill versus persistence
- MAE skill versus persistence
- Directional accuracy
- Final training loss
- Train/test observation counts and modelling parameters

The GitHub Actions workflow `.github/workflows/evaluate.yml` also runs this evaluation automatically and uploads `metrics.json` as a workflow artifact.

## Visualisations

### Apple mid-prices
![Mid-stock Prices](images/mid_stock_prices.png)

### Predicted vs actual prices
![Predicted vs Actual](images/predicted_vs_actual.png)

### Training loss
![Training Loss](images/training_loss.png)

## Repository structure

- `Apple_Stock_LSTM.ipynb` — original exploratory model notebook
- `evaluate_model.py` — reproducible leakage-free out-of-sample evaluation
- `.github/workflows/evaluate.yml` — automated model evaluation
- `images/` — generated plots
- `requirements.txt` — Python dependencies

## Tools

Python, TensorFlow/Keras, pandas, NumPy, matplotlib, scikit-learn, yfinance

## Key learning

- Preparing time-series data for supervised sequence models
- Building and training stacked LSTM networks
- Preventing preprocessing leakage in chronological forecasting problems
- Evaluating predictions in economically interpretable price units
- Benchmarking complex models against simple persistence forecasts
- Distinguishing visually plausible predictions from genuine out-of-sample forecasting improvement

## CV-ready project description

Built a two-layer LSTM model for Apple mid-price forecasting using 60-day rolling sequences and chronological train-test splitting; developed a leakage-free out-of-sample evaluation pipeline reporting RMSE, MAE and directional accuracy against a naive persistence benchmark
