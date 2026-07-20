"""
Phase A-1 §7.1 — seed variance calibration.

Reruns window_length=10 (current ablation "winner") twice with different
random seeds, same 5-district smoke config, to establish what magnitude
of RMSE difference is actually meaningful vs. training noise.

If seed-to-seed variance is comparable to (or larger than) the gap between
the four candidate window lengths, the ablation as currently sized cannot
distinguish them — informs whether to trust window=10 as a real finding
or fall back to the tie-break rule (prefer shortest).
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import torch
import yaml
import pandas as pd

from src.data.windowing import build_windows, SEQUENCE_FEATURE_COLUMNS
from src.models.lstm_tier3 import LSTMForecastModel

WINDOW_LENGTH = 10  # current winner from the first ablation run
TRAIN_START, TRAIN_END = "1984-01-01", "2010-12-31"
VAL_START, VAL_END = "2011-01-01", "2018-12-31"
HORIZON_DAYS = 10
SEEDS = [0, 1, 2]  # original run had no explicit seed set — add 3 controlled runs

with open("configs/data.yaml") as f:
    all_districts = [d["district_id"] for d in yaml.safe_load(f)["districts"]]
DISTRICTS_TO_USE = all_districts[:5]  # same smoke-test set as before

PLACEHOLDER_HPARAMS = dict(
    hidden_units=64, dropout=0.2, dense_units=32,
    batch_size=128, max_epochs=30, early_stopping_patience=15,
)


def build_split_windows(window_length: int):
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []

    for district_id in DISTRICTS_TO_USE:
        df = pd.read_parquet(f"data/engineered/{district_id}.parquet")
        ws = build_windows(df, window_length, SEQUENCE_FEATURE_COLUMNS, HORIZON_DAYS)

        for i in range(len(ws.input_start_dates)):
            t_start, t_end = ws.target_start_dates[i], ws.target_end_dates[i]
            if TRAIN_START <= t_start and t_end <= TRAIN_END:
                X_train_list.append(ws.X[i])
                y_train_list.append(ws.y[i])
            elif VAL_START <= t_start and t_end <= VAL_END:
                X_val_list.append(ws.X[i])
                y_val_list.append(ws.y[i])

    return (np.stack(X_train_list), np.stack(y_train_list),
            np.stack(X_val_list), np.stack(y_val_list))


def rmse_per_horizon_day(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


X_train, y_train, X_val, y_val = build_split_windows(WINDOW_LENGTH)
print(f"Train windows: {len(X_train)}  |  Val windows: {len(X_val)}")

results = {}

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"Seed: {seed}")
    print(f"{'='*50}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_features = X_train.shape[2]
    model = LSTMForecastModel(n_features=n_features, **PLACEHOLDER_HPARAMS)
    model.fit(X_train, y_train, X_val, y_val)

    val_pred = model.predict(X_val)
    rmse_by_day = rmse_per_horizon_day(y_val, val_pred)
    rmse_7_10 = rmse_by_day[6:10].mean()
    rmse_1_10 = rmse_by_day.mean()

    print(f"RMSE avg (days 7-10): {rmse_7_10:.4f}")
    print(f"RMSE avg (days 1-10): {rmse_1_10:.4f}")

    results[seed] = {"rmse_7_10": float(rmse_7_10), "rmse_1_10": float(rmse_1_10)}

print(f"\n{'='*50}")
print("SEED VARIANCE SUMMARY")
print(f"{'='*50}")

vals_7_10 = [r["rmse_7_10"] for r in results.values()]
print(f"rmse_7_10 across seeds: {[round(v,4) for v in vals_7_10]}")
print(f"  range (max-min): {max(vals_7_10) - min(vals_7_10):.4f}")
print(f"  std: {np.std(vals_7_10):.4f}")

print(f"\nFor comparison, the original 4-candidate ablation spread was:")
print(f"  window=10 vs window=15: {2.3225 - 2.2993:.4f}")
print(f"  window=10 vs window=30: {2.3133 - 2.2993:.4f}")
print(f"  full candidate spread (max-min): {2.3225 - 2.2993:.4f}")

seed_range = max(vals_7_10) - min(vals_7_10)
candidate_range = 2.3225 - 2.2993
if seed_range >= candidate_range * 0.5:
    print(f"\nWARNING: seed-to-seed variance ({seed_range:.4f}) is comparable to "
          f"or larger than the window-length candidate spread ({candidate_range:.4f}). "
          f"The original ablation cannot reliably distinguish these window lengths "
          f"at this sample size (5 districts). Do not trust window=10 as a real "
          f"finding without either (a) scaling to more districts, or (b) falling "
          f"back to the tie-break rule (prefer shortest window).")
else:
    print(f"\nSeed variance ({seed_range:.4f}) is small relative to the candidate "
          f"spread ({candidate_range:.4f}) — the window=10 result is more likely "
          f"a real signal, not noise.")