#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def _safe_std(x):
    std = x.std(axis=0, ddof=0)
    return np.where(std == 0, 1.0, std)
def _zscore(x, mean, std):
    return (x - mean) / std
def train_test_split_like_cpp(X, y, test_ratio, seed):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    X = X[idx]
    y = y[idx]
    n_test = int(n * test_ratio)
    n_train = n - n_test
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:n_train + n_test]
    y_test = y[n_train:n_train + n_test]
    return X_train, y_train, X_test, y_test
def main():
    parser = argparse.ArgumentParser(
        description="Sklearn LinearRegression benchmark (project-like preprocessing)."
    )
    parser.add_argument("csv_path", type=str)
    parser.add_argument("target_column", type=int, help="1-based index")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    target_idx = args.target_column - 1
    if target_idx < 0 or target_idx >= df.shape[1]:
        raise ValueError("target_column out of range for this dataset")
    y = df.iloc[:, target_idx].to_numpy(dtype=float)
    X = df.drop(df.columns[target_idx], axis=1).to_numpy(dtype=float)
    X_train, y_train, X_test, y_test = train_test_split_like_cpp(
        X, y, args.test_ratio, args.seed
    )
    X_train_mean = X_train.mean(axis=0)
    X_train_std = _safe_std(X_train)
    X_train_scaled = _zscore(X_train, X_train_mean, X_train_std)
    X_test_mean = X_test.mean(axis=0)
    X_test_std = _safe_std(X_test)
    X_test_scaled = _zscore(X_test, X_test_mean, X_test_std)
    y_train_mean = y_train.mean()
    y_train_std = float(_safe_std(y_train))
    y_train_scaled = _zscore(y_train, y_train_mean, y_train_std)
    y_test_mean = y_test.mean()
    y_test_std = float(_safe_std(y_test))
    y_test_scaled = _zscore(y_test, y_test_mean, y_test_std)
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train_scaled, y_train_scaled)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test_scaled, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test_scaled, y_pred)
    r2 = r2_score(y_test_scaled, y_pred)
    tolerance = 0.5
    accuracy = float((np.abs(y_test_scaled - y_pred) < tolerance).mean() * 100.0)
    result = {
        "model": "sklearn.LinearRegression",
        "dataset": str(csv_path),
        "target_column_1_based": args.target_column,
        "test_ratio": args.test_ratio,
        "preprocessing": {
            "x_scale": "zscore",
            "x_test_scaling": "test_stats",
            "y_scaled": True
        },
        "metrics": {
            "accuracy_within_0_5": accuracy,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "mse": mse
        }
    }
    print(json.dumps(result, indent=2))
if __name__ == "__main__":
    main()
