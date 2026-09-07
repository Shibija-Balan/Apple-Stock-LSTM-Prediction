# Apple Stock LSTM Prediction

A time-series forecasting project using stacked Long Short-Term Memory (LSTM) neural networks to model Apple stock mid-prices from 2020 to 2025.

## Project overview

The project calculates daily Apple mid-prices from historical high and low prices, constructs 60-day rolling sequences and trains a stacked LSTM network to predict the next trading day's mid-price.

The repository includes a separate leakage-free evaluation pipeline using a pinned public AAPL OHLC dataset, chronological validation and comparison against a naive persistence forecast.

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

1. Loads a pinned public AAPL OHLC dataset covering 1 January 2020 to 17 January 2025
2. Calculates daily mid-price as `(High + Low) / 2`
3. Uses a chronological 80/20 train-test split rather than random sampling
4. Fits the MinMax scaler **only on the training period**, preventing test-period information from leaking into preprocessing
5. Trains the same two-layer LSTM architecture used in the notebook
6. Evaluates predictions in original US-dollar price units
7. Reports out-of-sample RMSE, MAE and directional accuracy
8. Compares the model against a naive persistence benchmark where the next day's predicted mid-price equals the previous day's observed mid-price
9. Reports RMSE and MAE skill relative to that benchmark

This benchmark matters because stock-price levels are highly persistent. A prediction line can visually track observed prices while still failing to outperform the simple assumption that tomorrow's price will equal today's price.

## Out-of-sample performance

Evaluation window: chronological 80/20 holdout across 1,264 daily observations, with 1,011 training observations and 253 test observations.

| Metric | LSTM | Naive persistence |
|---|---:|---:|
| RMSE | **$6.23** | **$2.80** |
| MAE | **$5.17** | **$2.00** |
| Directional accuracy | **49.8%** | n/a |

Additional results:

- Mean test-period mid-price: **$208.16**
- Normalised LSTM RMSE: **3.0%** of the mean test-period mid-price
- LSTM RMSE skill vs persistence: **-122.9%**
- LSTM MAE skill vs persistence: **-158.6%**

### Interpretation

The LSTM produces predictions that are relatively close to the observed price level in absolute terms: an RMSE of $6.23 corresponds to roughly 3.0% of the mean test-period mid-price. However, it does **not** outperform the naive persistence benchmark. The persistence forecast achieves a materially lower RMSE of $2.80 and MAE of $2.00.

The model's 49.8% directional accuracy is also approximately chance-level, indicating that the LSTM does not demonstrate useful next-day directional forecasting ability in this specification.

The main conclusion is therefore not that the LSTM is a profitable forecasting model. Instead, the project demonstrates why visually close stock-price predictions are insufficient evidence of predictive value and why complex financial machine-learning models must be evaluated against simple baselines using leakage-free out-of-sample testing.

## Performance outputs

Running:

```bash
python evaluate_model.py
```

produces `metrics.json` containing:

- LSTM RMSE and MAE in USD
- Naive persistence RMSE and MAE in USD
- RMSE and MAE skill versus persistence
- Directional accuracy
- Normalised RMSE
- Final training loss
- Train/test observation counts and modelling parameters

The GitHub Actions workflow `.github/workflows/evaluate.yml` runs this evaluation automatically and uploads `metrics.json` as a workflow artifact.

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

Python, TensorFlow/Keras, pandas, NumPy, matplotlib, scikit-learn

## Key learning

- Preparing time-series data for supervised sequence models
- Building and training stacked LSTM networks
- Preventing preprocessing leakage in chronological forecasting problems
- Evaluating predictions in economically interpretable price units
- Benchmarking complex models against simple persistence forecasts
- Distinguishing visually plausible predictions from genuine out-of-sample forecasting improvement

## CV-ready project description

Built and evaluated a two-layer LSTM for Apple mid-price forecasting using 60-day sequences and leakage-free chronological validation; achieved $6.23 out-of-sample RMSE and 49.8% directional accuracy, with benchmark testing showing the model underperformed a $2.80-RMSE persistence forecast
