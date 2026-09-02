# Apple Stock LSTM Prediction

A time-series forecasting project using stacked Long Short-Term Memory (LSTM) neural networks to model Apple stock mid-prices from 2020 to 2025.

## Project overview

The notebook preprocesses historical OHLC data, calculates a mid-price from daily high and low values, constructs 60-day rolling sequences and trains a stacked LSTM network to predict the next value in the series.

## Methods

- Chronological filtering and preprocessing of historical Apple stock data
- Mid-price calculation from high and low prices
- Min-max normalisation
- 60-day rolling input sequences
- Stacked LSTM architecture with dropout regularisation
- Time-ordered train/test handling to avoid future-data leakage
- Visual comparison of predicted and observed prices
- Training-loss monitoring

## Model

The project uses TensorFlow/Keras with stacked LSTM layers, dropout and the Adam optimiser. LSTMs were chosen as an experiment in sequence modelling because they can represent non-linear temporal dependencies across rolling windows.

## Visualisations

### Apple mid-prices
![Mid-stock Prices](images/mid_stock_prices.png)

### Predicted vs actual prices
![Predicted vs Actual](images/predicted_vs_actual.png)

### Training loss
![Training Loss](images/training_loss.png)

## Repository structure

- `Apple_Stock_LSTM.ipynb` — full analysis and model notebook
- `images/` — generated plots
- `requirements.txt` — Python dependencies

## Reproducibility note

The notebook currently expects a local file named `aapl_us_2025.csv`. That source file is not committed to this repository, so the analysis is not yet fully reproducible from a fresh clone. A future revision should either download the price data programmatically from a documented public source or provide explicit instructions for obtaining the same dataset.

## Evaluation limitation

A visually close predicted-price line is not sufficient evidence that a financial forecasting model is useful. Stock-price levels are strongly persistent, meaning a simple baseline such as "tomorrow's price equals today's price" can already look convincing on a chart.

A stronger evaluation should compare the LSTM against naive persistence and simpler statistical or regression baselines using out-of-sample metrics such as MAE and RMSE. Until that comparison is added, this project should be interpreted as a sequence-modelling exercise rather than evidence of profitable market prediction.

## Tools

Python, TensorFlow/Keras, pandas, NumPy, matplotlib, scikit-learn

## Key learning

- Preparing time-series data for supervised sequence models
- Building and training stacked LSTM networks
- Applying normalisation and rolling windows
- Avoiding obvious future-data leakage through chronological splitting
- Recognising the importance of baseline models and rigorous out-of-sample evaluation in financial machine learning
