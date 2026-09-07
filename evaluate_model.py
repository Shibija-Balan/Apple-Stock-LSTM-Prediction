"""Leakage-free evaluation for the Apple mid-price LSTM project.

Downloads AAPL daily OHLC data from Stooq, builds the same 60-day/two-layer
LSTM used in Apple_Stock_LSTM.ipynb, and evaluates it on a chronological
holdout period. The scaler is fitted only on the training period.
Performance is compared against a naive persistence baseline:
tomorrow's mid-price = today's mid-price.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

SEED = 42
START_DATE = "2020-01-01"
END_DATE = "2025-01-17"
DATA_URL = "https://stooq.com/q/d/l/?s=aapl.us&d1=20200101&d2=20250117&i=d"
WINDOW_SIZE = 60
TRAIN_FRACTION = 0.80
EPOCHS = 25
BATCH_SIZE = 32


def set_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, parse_dates=["Date"])
    required = ["Date", "Open", "High", "Low", "Close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns: {missing}")

    df = (
        df[required]
        .dropna()
        .query("Date >= @START_DATE and Date <= @END_DATE")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    if df.empty:
        raise RuntimeError("No AAPL observations returned from Stooq")

    df["MidPrice"] = (df["High"] + df["Low"]) / 2.0
    return df


def make_sequences(scaled: np.ndarray, target_indices: np.ndarray):
    X, y = [], []
    for target_idx in target_indices:
        X.append(scaled[target_idx - WINDOW_SIZE : target_idx, 0])
        y.append(scaled[target_idx, 0])
    X = np.asarray(X, dtype=np.float32).reshape(-1, WINDOW_SIZE, 1)
    y = np.asarray(y, dtype=np.float32)
    return X, y


def build_model() -> Sequential:
    model = Sequential(
        [
            Input(shape=(WINDOW_SIZE, 1)),
            LSTM(units=70, return_sequences=True),
            Dropout(0.2),
            LSTM(units=70, return_sequences=False),
            Dropout(0.2),
            Dense(units=1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray, previous: np.ndarray) -> float:
    actual_direction = np.sign(actual - previous)
    predicted_direction = np.sign(predicted - previous)
    return float(np.mean(actual_direction == predicted_direction))


def main() -> None:
    set_seeds()
    df = load_data()
    prices = df["MidPrice"].to_numpy(dtype=np.float64)

    split_idx = int(TRAIN_FRACTION * len(prices))
    if split_idx <= WINDOW_SIZE:
        raise RuntimeError("Training set is too short for the selected window")

    # Leakage prevention: estimate scaling parameters on training data only.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(prices[:split_idx].reshape(-1, 1))
    scaled = scaler.transform(prices.reshape(-1, 1))

    train_targets = np.arange(WINDOW_SIZE, split_idx)
    test_targets = np.arange(split_idx, len(prices))
    X_train, y_train = make_sequences(scaled, train_targets)
    X_test, y_test = make_sequences(scaled, test_targets)

    model = build_model()
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        shuffle=False,
    )

    pred_scaled = model.predict(X_test, verbose=0)
    predicted = scaler.inverse_transform(pred_scaled).ravel()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    previous = prices[test_targets - 1]

    # Persistence benchmark: next day's mid-price equals previous day's actual mid-price.
    naive = previous.copy()

    lstm_rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    lstm_mae = float(mean_absolute_error(actual, predicted))
    naive_rmse = float(np.sqrt(mean_squared_error(actual, naive)))
    naive_mae = float(mean_absolute_error(actual, naive))
    direction_acc = directional_accuracy(actual, predicted, previous)
    rmse_skill = float(1 - lstm_rmse / naive_rmse)
    mae_skill = float(1 - lstm_mae / naive_mae)

    test_mean_price = float(np.mean(actual))
    normalized_rmse = float(lstm_rmse / test_mean_price)

    metrics = {
        "ticker": "AAPL",
        "data_source": "Stooq daily OHLC",
        "start_date": START_DATE,
        "end_date_inclusive": END_DATE,
        "observations": int(len(prices)),
        "train_observations": int(split_idx),
        "test_observations": int(len(test_targets)),
        "window_size": WINDOW_SIZE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lstm_rmse_usd": lstm_rmse,
        "lstm_mae_usd": lstm_mae,
        "naive_rmse_usd": naive_rmse,
        "naive_mae_usd": naive_mae,
        "rmse_skill_vs_naive": rmse_skill,
        "mae_skill_vs_naive": mae_skill,
        "directional_accuracy": direction_acc,
        "test_mean_mid_price_usd": test_mean_price,
        "normalized_rmse": normalized_rmse,
        "final_training_loss": float(history.history["loss"][-1]),
        "methodology": "chronological 80/20 holdout; scaler fit on training period only; no shuffle",
    }

    Path("metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== AAPL LSTM OUT-OF-SAMPLE EVALUATION ===")
    print(json.dumps(metrics, indent=2))
    print("\nInterpretation:")
    if rmse_skill > 0:
        print(f"LSTM beats naive persistence on RMSE by {rmse_skill:.1%}.")
    else:
        print(f"LSTM underperforms naive persistence on RMSE by {-rmse_skill:.1%}.")
    if mae_skill > 0:
        print(f"LSTM beats naive persistence on MAE by {mae_skill:.1%}.")
    else:
        print(f"LSTM underperforms naive persistence on MAE by {-mae_skill:.1%}.")
    print(f"Directional accuracy: {direction_acc:.1%}")
    print(f"Normalized RMSE: {normalized_rmse:.1%} of mean test-period mid-price")


if __name__ == "__main__":
    main()
