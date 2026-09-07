import json
from pathlib import Path

NOTEBOOK = Path('Apple_Stock_LSTM.ipynb')
MARKER = '# Model 2: Improving the Model'


def md(text):
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in text.strip().splitlines()]
    }


def code(text):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in text.strip().splitlines()]
    }


nb = json.loads(NOTEBOOK.read_text(encoding='utf-8'))

# Make script idempotent: remove any prior embedded Model 1/Model 2 section.
cut = None
for i, cell in enumerate(nb.get('cells', [])):
    src = ''.join(cell.get('source', []))
    if src.startswith('# Model 1: Proper Out-of-Sample Evaluation') or src.startswith(MARKER):
        cut = i
        break
if cut is not None:
    nb['cells'] = nb['cells'][:cut]

new_cells = [
    md('''# Model 1: Proper Out-of-Sample Evaluation

The section above is retained as the **original prototype**. Its predicted-vs-actual plot looks reasonably close to the price path, but a visual fit is not enough to establish forecasting performance.

There is also an important preprocessing issue in the original prototype: `MinMaxScaler` is fitted to the full price series before the train/test split. That allows information about the test-period price range to enter preprocessing.

To assess the original architecture fairly, the same 60-day, two-layer 70-unit LSTM was re-evaluated in `evaluate_model.py` using a chronological 80/20 holdout, training-only scaling, no shuffling, RMSE, MAE, directional accuracy and a persistence benchmark.

### Leakage-free Model 1 results

| Metric | Original LSTM | Persistence benchmark |
| --- | ---: | ---: |
| RMSE | **$6.23** | **$2.80** |
| MAE | **$5.17** | **$2.00** |
| Directional accuracy | **49.8%** | — |
| Normalized RMSE | **3.0%** of mean test price | — |

The original LSTM therefore underperformed persistence by **122.9% on RMSE**. Its directional accuracy was approximately chance-level.

### Why can the graph still look good?

Stock **price levels are highly persistent**. A model can learn the broad level and trend of the series, producing a visually convincing plot, while still making larger next-day errors than simply carrying forward today's price.

This failure motivates redesigning the forecasting problem rather than merely increasing network complexity.'''),

    md('''# Model 2: Improving the Model

Instead of predicting the next absolute price level, Model 2 predicts the **next-day return**:

\[
r_{t+1}=\frac{P_{t+1}}{P_t}-1
\]

This focuses the model on day-to-day movement rather than the highly persistent price level.

The redesign introduces:

1. **Return target instead of price level**
2. **Multivariate features** describing momentum, volatility, range and volume
3. **20-day sequences** instead of 60-day price-only sequences
4. A chronological **70% train / 15% validation / 15% test** split
5. `StandardScaler` fitted **only on training data**
6. A smaller **32-unit LSTM** followed by a 16-unit dense layer
7. **Early stopping** on validation loss
8. A comparison against a zero-return / persistence forecast on the **same final test period**

The reproducible Model 2 section uses the pinned public AAPL OHLCV dataset used by the repository evaluation scripts. It is not claimed to be byte-for-byte identical to the missing local `aapl_us_2025.csv` used in the original prototype.'''),

    md('## Load reproducible AAPL data for Model 2'),
    code('''import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODEL2_DATA_URL = (
    "https://raw.githubusercontent.com/Buicongbang04/Stock-Prediction/"
    "c6a7f945640aff9c80db10b8419888e72f1f5cc8/data/AAPL.csv"
)

model2_df = pd.read_csv(MODEL2_DATA_URL, parse_dates=["Date"])
model2_df = model2_df.loc[
    (model2_df["Date"] >= "2020-01-01") &
    (model2_df["Date"] <= "2025-01-17")
].copy()
model2_df = model2_df.sort_values("Date").reset_index(drop=True)
model2_df[["Date", "Open", "High", "Low", "Close", "Volume"]].head()'''),

    md('''## Feature engineering

Every feature at time \(t\) uses information available by time \(t\). The target is the return from \(t\) to \(t+1\).

Features include 1-day and 5-day returns, 10-day momentum, 5-day and 20-day rolling volatility, intraday range, open-close movement, volume change, and moving-average gaps.'''),
    code('''model2_df["MidPrice"] = (model2_df["High"] + model2_df["Low"]) / 2.0
model2_df["Return_1d"] = model2_df["MidPrice"].pct_change()
model2_df["Return_5d"] = model2_df["MidPrice"].pct_change(5)
model2_df["Momentum_10d"] = model2_df["MidPrice"] / model2_df["MidPrice"].shift(10) - 1.0
model2_df["Volatility_5d"] = model2_df["Return_1d"].rolling(5).std()
model2_df["Volatility_20d"] = model2_df["Return_1d"].rolling(20).std()
model2_df["RangePct"] = (model2_df["High"] - model2_df["Low"]) / model2_df["MidPrice"]
model2_df["OpenClosePct"] = (model2_df["Close"] - model2_df["Open"]) / model2_df["Open"]
model2_df["VolumeChange"] = model2_df["Volume"].pct_change()
model2_df["MA5Gap"] = model2_df["MidPrice"] / model2_df["MidPrice"].rolling(5).mean() - 1.0
model2_df["MA20Gap"] = model2_df["MidPrice"] / model2_df["MidPrice"].rolling(20).mean() - 1.0

model2_df["TargetReturn"] = model2_df["MidPrice"].shift(-1) / model2_df["MidPrice"] - 1.0
model2_df["NextMidPrice"] = model2_df["MidPrice"].shift(-1)

model2_features = [
    "Return_1d", "Return_5d", "Momentum_10d", "Volatility_5d",
    "Volatility_20d", "RangePct", "OpenClosePct", "VolumeChange",
    "MA5Gap", "MA20Gap"
]

model2_clean = (
    model2_df[["Date", "MidPrice", "NextMidPrice", "TargetReturn"] + model2_features]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .reset_index(drop=True)
)
model2_clean.head()'''),

    md('''## Leakage-free chronological train / validation / test split

The first 70% of observations are used to fit the scalers and train the network. The next 15% form the validation period used by early stopping. The final 15% remain untouched until final evaluation.'''),
    code('''SEQUENCE_LENGTH = 20
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

n_model2 = len(model2_clean)
train_end = int(TRAIN_FRACTION * n_model2)
val_end = int((TRAIN_FRACTION + VALIDATION_FRACTION) * n_model2)

x_scaler_2 = StandardScaler().fit(model2_clean.loc[:train_end - 1, model2_features])
y_scaler_2 = StandardScaler().fit(model2_clean.loc[:train_end - 1, ["TargetReturn"]])

features_scaled = x_scaler_2.transform(model2_clean[model2_features])
target_scaled = y_scaler_2.transform(model2_clean[["TargetReturn"]]).ravel()

def build_model2_sequences(features, target, current_price, next_price, dates, window=20):
    X_seq, y_seq, base_prices, actual_prices, seq_dates = [], [], [], [], []
    for i in range(window - 1, len(features)):
        X_seq.append(features[i-window+1:i+1])
        y_seq.append(target[i])
        base_prices.append(current_price[i])
        actual_prices.append(next_price[i])
        seq_dates.append(dates[i])
    return (np.asarray(X_seq, dtype=np.float32),
            np.asarray(y_seq, dtype=np.float32),
            np.asarray(base_prices, dtype=np.float64),
            np.asarray(actual_prices, dtype=np.float64),
            np.asarray(seq_dates))

X2, y2, base_prices_2, actual_prices_2, dates_2 = build_model2_sequences(
    features_scaled, target_scaled,
    model2_clean["MidPrice"].to_numpy(),
    model2_clean["NextMidPrice"].to_numpy(),
    model2_clean["Date"].to_numpy(),
    window=SEQUENCE_LENGTH
)

row_indices = np.arange(SEQUENCE_LENGTH - 1, n_model2)
train_mask = row_indices < train_end
val_mask = (row_indices >= train_end) & (row_indices < val_end)
test_mask = row_indices >= val_end

X2_train, y2_train = X2[train_mask], y2[train_mask]
X2_val, y2_val = X2[val_mask], y2[val_mask]
X2_test, y2_test = X2[test_mask], y2[test_mask]
base_test_2 = base_prices_2[test_mask]
actual_price_test_2 = actual_prices_2[test_mask]
test_dates_2 = dates_2[test_mask]

print("Training sequences:", len(X2_train))
print("Validation sequences:", len(X2_val))
print("Test sequences:", len(X2_test))'''),

    md('''## Smaller LSTM + early stopping

The original model used two stacked 70-unit LSTM layers. Model 2 deliberately reduces complexity because roughly 1,200 daily observations is a relatively small dataset for a deep recurrent network.'''),
    code('''model2 = Sequential([
    Input(shape=(SEQUENCE_LENGTH, len(model2_features))),
    LSTM(32, return_sequences=False),
    Dropout(0.20),
    Dense(16, activation="relu"),
    Dense(1),
])

model2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss="mse"
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    min_delta=1e-6
)

history2 = model2.fit(
    X2_train, y2_train,
    validation_data=(X2_val, y2_val),
    epochs=60,
    batch_size=32,
    shuffle=False,
    callbacks=[early_stopping],
    verbose=1
)

best_epoch_2 = int(np.argmin(history2.history["val_loss"]) + 1)
print("Best validation epoch:", best_epoch_2)'''),

    md('''## Evaluate Model 2

The network predicts standardized next-day returns. These are transformed back into returns and then converted into next-day prices using:

\[
\hat P_{t+1}=P_t(1+\hat r_{t+1})
\]

The fair benchmark is a zero-return forecast, equivalent to predicting that tomorrow's price equals today's price.'''),
    code('''pred_return_scaled_2 = model2.predict(X2_test, verbose=0).ravel()
pred_return_2 = y_scaler_2.inverse_transform(pred_return_scaled_2.reshape(-1, 1)).ravel()
actual_return_2 = y_scaler_2.inverse_transform(y2_test.reshape(-1, 1)).ravel()

predicted_price_2 = base_test_2 * (1.0 + pred_return_2)
persistence_price_2 = base_test_2.copy()

model2_rmse = np.sqrt(mean_squared_error(actual_price_test_2, predicted_price_2))
model2_mae = mean_absolute_error(actual_price_test_2, predicted_price_2)
persistence_rmse_2 = np.sqrt(mean_squared_error(actual_price_test_2, persistence_price_2))
persistence_mae_2 = mean_absolute_error(actual_price_test_2, persistence_price_2)
directional_accuracy_2 = np.mean(np.sign(actual_return_2) == np.sign(pred_return_2))
skill_vs_persistence_2 = 1 - model2_rmse / persistence_rmse_2

print(f"Model 2 price RMSE: ${model2_rmse:.2f}")
print(f"Model 2 price MAE: ${model2_mae:.2f}")
print(f"Persistence RMSE: ${persistence_rmse_2:.2f}")
print(f"Persistence MAE: ${persistence_mae_2:.2f}")
print(f"Directional accuracy: {directional_accuracy_2:.1%}")
print(f"RMSE skill vs persistence: {skill_vs_persistence_2:.1%}")'''),

    md('## Model 2: Actual vs predicted next-day price'),
    code('''plt.figure(figsize=(12, 6))
plt.plot(test_dates_2, actual_price_test_2, label="Actual next-day mid-price")
plt.plot(test_dates_2, predicted_price_2, label="Model 2 prediction")
plt.plot(test_dates_2, persistence_price_2, label="Persistence benchmark", alpha=0.7)
plt.title("Model 2: Apple Next-Day Mid-Price Reconstruction")
plt.xlabel("Date")
plt.ylabel("Mid Price ($)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()'''),

    md('## Model 2: Training and validation loss'),
    code('''plt.figure(figsize=(10, 5))
plt.plot(history2.history["loss"], label="Training loss")
plt.plot(history2.history["val_loss"], label="Validation loss")
plt.axvline(best_epoch_2 - 1, linestyle="--", label=f"Best epoch = {best_epoch_2}")
plt.title("Model 2 Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE on standardized returns")
plt.legend()
plt.tight_layout()
plt.show()'''),

    md('''# Results: Model 1 vs Model 2

The repository's reproducible GitHub Actions evaluation produced these verified results:

| Metric | Model 1: price-level LSTM | Model 2: return-based LSTM |
| --- | ---: | ---: |
| Price RMSE | **$6.23** | **$2.82** |
| Price MAE | **$5.17** | **$1.89** |
| Directional accuracy | **49.8%** | **66.3%** |
| Sequence length | 60 days | 20 days |
| Inputs | Mid-price only | 10 engineered features |
| Validation design | 80/20 chronological holdout | 70/15/15 chronological split |
| Best epoch | fixed at 25 epochs | **17** via early stopping |

Across the project's reported holdouts, price RMSE fell from **$6.23 to $2.82**, a **54.8% reduction**.

For the apples-to-apples benchmark inside Model 2's final test period (18 April 2024 to 16 January 2025):

| Model 2 test-period metric | Return-based LSTM | Persistence |
| --- | ---: | ---: |
| RMSE | **$2.82** | **$3.00** |
| MAE | **$1.89** | **$2.11** |

Model 2 beat persistence by **6.1% on RMSE** on the same held-out period.

> **Comparison caveat:** Model 1 and Model 2 use different holdout definitions, so the 54.8% reduction describes the change in each model's reported out-of-sample RMSE rather than a same-dates head-to-head test. The 6.1% persistence comparison *is* a same-test-period comparison.

## Final interpretation

The main improvement came from redesigning the forecasting problem rather than simply changing the number of LSTM units. Model 1 mostly learned the persistent Apple price level. Model 2 instead focuses on next-day returns, introduces market-derived features, separates validation from testing, prevents preprocessing leakage and uses early stopping.

The redesign produced **$2.82 price RMSE**, **$1.89 price MAE**, **66.3% directional accuracy**, and **6.1% lower RMSE than persistence on the same final holdout**.

This is encouraging evidence for the redesigned specification, but one successful holdout does **not** establish a durable trading edge. A stronger next step would be walk-forward / expanding-window validation across multiple market regimes and, if a trading strategy is defined, evaluation after transaction costs.''')
]

nb['cells'].extend(new_cells)
NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(f'Embedded {len(new_cells)} cells into {NOTEBOOK}')
