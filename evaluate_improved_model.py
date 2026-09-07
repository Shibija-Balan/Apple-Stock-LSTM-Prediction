"""Improvement experiment for the Apple stock LSTM project.

The original model predicts the next-day price level and is benchmarked against
persistence. This experiment reframes the problem as next-day return prediction,
adds market-derived features, uses a train/validation/test split, and reconstructs
price forecasts so its RMSE can be compared directly with the original model.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

SEED = 42
START_DATE = "2020-01-01"
END_DATE = "2025-01-17"
DATA_URL = (
    "https://raw.githubusercontent.com/Buicongbang04/Stock-Prediction/"
    "c6a7f945640aff9c80db10b8419888e72f1f5cc8/data/AAPL.csv"
)
SEQUENCE_LENGTH = 20
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
MAX_EPOCHS = 60
BATCH_SIZE = 32

ORIGINAL_LSTM_RMSE = 6.234461886534269
ORIGINAL_LSTM_MAE = 5.169205665588379
ORIGINAL_PERSISTENCE_RMSE = 2.796607170455091


def set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_and_engineer_features():
    df = pd.read_csv(DATA_URL, parse_dates=["Date"])
    df = df.loc[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    df["MidPrice"] = (df["High"] + df["Low"]) / 2.0
    df["Return_1d"] = df["MidPrice"].pct_change()
    df["Return_5d"] = df["MidPrice"].pct_change(5)
    df["Momentum_10d"] = df["MidPrice"] / df["MidPrice"].shift(10) - 1.0
    df["Volatility_5d"] = df["Return_1d"].rolling(5).std()
    df["Volatility_20d"] = df["Return_1d"].rolling(20).std()
    df["RangePct"] = (df["High"] - df["Low"]) / df["MidPrice"]
    df["OpenClosePct"] = (df["Close"] - df["Open"]) / df["Open"]
    df["VolumeChange"] = df["Volume"].pct_change()
    df["MA5Gap"] = df["MidPrice"] / df["MidPrice"].rolling(5).mean() - 1.0
    df["MA20Gap"] = df["MidPrice"] / df["MidPrice"].rolling(20).mean() - 1.0

    # Target known only at t+1. Each row's features use information available at t.
    df["TargetReturn"] = df["MidPrice"].shift(-1) / df["MidPrice"] - 1.0
    df["NextMidPrice"] = df["MidPrice"].shift(-1)

    feature_cols = [
        "Return_1d",
        "Return_5d",
        "Momentum_10d",
        "Volatility_5d",
        "Volatility_20d",
        "RangePct",
        "OpenClosePct",
        "VolumeChange",
        "MA5Gap",
        "MA20Gap",
    ]

    clean = df[["Date", "MidPrice", "NextMidPrice", "TargetReturn"] + feature_cols].replace(
        [np.inf, -np.inf], np.nan
    ).dropna().reset_index(drop=True)
    return clean, feature_cols


def build_sequences(features, target, current_price, next_price, dates):
    X, y, base_price, actual_price, out_dates = [], [], [], [], []
    for i in range(SEQUENCE_LENGTH - 1, len(features)):
        X.append(features[i - SEQUENCE_LENGTH + 1 : i + 1])
        y.append(target[i])
        base_price.append(current_price[i])
        actual_price.append(next_price[i])
        out_dates.append(dates[i])
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(base_price, dtype=np.float64),
        np.asarray(actual_price, dtype=np.float64),
        np.asarray(out_dates),
    )


def make_model(n_features):
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, n_features)),
        LSTM(32, return_sequences=False),
        Dropout(0.20),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4), loss="mse")
    return model


def directional_accuracy(actual_return, predicted_return):
    return float(np.mean(np.sign(actual_return) == np.sign(predicted_return)))


def main():
    set_seeds()
    df, feature_cols = load_and_engineer_features()

    n = len(df)
    train_end = int(TRAIN_FRACTION * n)
    val_end = int((TRAIN_FRACTION + VALIDATION_FRACTION) * n)

    # Fit transforms only on the training rows.
    x_scaler = StandardScaler().fit(df.loc[: train_end - 1, feature_cols])
    y_scaler = StandardScaler().fit(df.loc[: train_end - 1, ["TargetReturn"]])

    X_all = x_scaler.transform(df[feature_cols])
    y_all = y_scaler.transform(df[["TargetReturn"]]).ravel()

    X, y, base_price, actual_price, seq_dates = build_sequences(
        X_all,
        y_all,
        df["MidPrice"].to_numpy(),
        df["NextMidPrice"].to_numpy(),
        df["Date"].to_numpy(),
    )

    # Sequence indices correspond to original cleaned row index i = seq_idx + SEQUENCE_LENGTH - 1.
    row_indices = np.arange(SEQUENCE_LENGTH - 1, n)
    train_mask = row_indices < train_end
    val_mask = (row_indices >= train_end) & (row_indices < val_end)
    test_mask = row_indices >= val_end

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    base_test, actual_price_test = base_price[test_mask], actual_price[test_mask]
    test_dates = seq_dates[test_mask]

    model = make_model(len(feature_cols))
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, min_delta=1e-6)
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        verbose=0,
        callbacks=callbacks,
    )

    pred_scaled = model.predict(X_test, verbose=0).ravel()
    pred_return = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    actual_return = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    predicted_price = base_test * (1.0 + pred_return)
    zero_return_price = base_test.copy()  # exactly the persistence price benchmark

    improved_rmse = float(np.sqrt(mean_squared_error(actual_price_test, predicted_price)))
    improved_mae = float(mean_absolute_error(actual_price_test, predicted_price))
    persistence_rmse = float(np.sqrt(mean_squared_error(actual_price_test, zero_return_price)))
    persistence_mae = float(mean_absolute_error(actual_price_test, zero_return_price))
    return_rmse = float(np.sqrt(mean_squared_error(actual_return, pred_return)))
    zero_return_rmse = float(np.sqrt(mean_squared_error(actual_return, np.zeros_like(actual_return))))
    direction_acc = directional_accuracy(actual_return, pred_return)

    metrics = {
        "experiment": "Return-target multivariate LSTM",
        "features": feature_cols,
        "sequence_length": SEQUENCE_LENGTH,
        "split": "70% train / 15% validation / 15% test, chronological",
        "train_sequences": int(train_mask.sum()),
        "validation_sequences": int(val_mask.sum()),
        "test_sequences": int(test_mask.sum()),
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
        "improved_price_rmse_usd": improved_rmse,
        "improved_price_mae_usd": improved_mae,
        "persistence_price_rmse_usd_on_same_test": persistence_rmse,
        "persistence_price_mae_usd_on_same_test": persistence_mae,
        "return_rmse": return_rmse,
        "zero_return_rmse": zero_return_rmse,
        "directional_accuracy": direction_acc,
        "rmse_reduction_vs_original_lstm": float(1 - improved_rmse / ORIGINAL_LSTM_RMSE),
        "mae_reduction_vs_original_lstm": float(1 - improved_mae / ORIGINAL_LSTM_MAE),
        "skill_vs_persistence_same_test": float(1 - improved_rmse / persistence_rmse),
        "original_lstm_rmse_usd": ORIGINAL_LSTM_RMSE,
        "original_persistence_rmse_usd": ORIGINAL_PERSISTENCE_RMSE,
        "test_start": str(pd.Timestamp(test_dates[0]).date()),
        "test_end": str(pd.Timestamp(test_dates[-1]).date()),
    }

    Path("improved_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== IMPROVED RETURN-BASED LSTM ===")
    print(json.dumps(metrics, indent=2))
    print("\nInterpretation:")
    print(f"Price RMSE changed from ${ORIGINAL_LSTM_RMSE:.2f} to ${improved_rmse:.2f}.")
    print(f"That is a {metrics['rmse_reduction_vs_original_lstm']:.1%} reduction versus Model 1.")
    if metrics["skill_vs_persistence_same_test"] > 0:
        print(f"Model 2 beats persistence on the same test period by {metrics['skill_vs_persistence_same_test']:.1%} RMSE.")
    else:
        print(f"Model 2 still trails persistence on the same test period by {-metrics['skill_vs_persistence_same_test']:.1%} RMSE.")
    print(f"Directional accuracy: {direction_acc:.1%}")


if __name__ == "__main__":
    main()
