"""
Tier 2 GBM — quick sanity run + light hyperparameter search on the fixed
reference split. CPU-only, no Kaggle/GPU needed (GBM trains fast on this
data volume). Uses TREE_FEATURE_COLUMNS, window=15 (locked, master §8.6).
"""

import sys
sys.path.insert(0, ".")

import itertools
import time
import numpy as np
import pandas as pd
import yaml

from src.data.windowing import build_windows, TREE_FEATURE_COLUMNS
from src.models.gbm_tier2 import GBMForecastModel

TRAIN_START, TRAIN_END = "1984-01-01", "2010-12-31"
VAL_START, VAL_END = "2011-01-01", "2018-12-31"
WINDOW_LENGTH = 15
HORIZON_DAYS = 10

with open("configs/data.yaml") as f:
    all_districts = [d["district_id"] for d in yaml.safe_load(f)["districts"]]


def build_split_windows():
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []

    print(f"Building windows for {len(all_districts)} districts...")
    for i, did in enumerate(all_districts, 1):
        if i % 15 == 0:
            print(f"  {i}/{len(all_districts)}")
        df = pd.read_parquet(f"data/engineered/{did}.parquet")
        ws = build_windows(df, WINDOW_LENGTH, TREE_FEATURE_COLUMNS, HORIZON_DAYS)

        for j in range(len(ws.input_start_dates)):
            t_start, t_end = ws.target_start_dates[j], ws.target_end_dates[j]
            if TRAIN_START <= t_start and t_end <= TRAIN_END:
                X_train_list.append(ws.X[j])
                y_train_list.append(ws.y[j])
            elif VAL_START <= t_start and t_end <= VAL_END:
                X_val_list.append(ws.X[j])
                y_val_list.append(ws.y[j])

    return (np.stack(X_train_list), np.stack(y_train_list),
            np.stack(X_val_list), np.stack(y_val_list))


def rmse_per_horizon_day(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


print("Building pooled dataset (this takes a few minutes)...")
X_train, y_train, X_val, y_val = build_split_windows()
print(f"X_train: {X_train.shape}  X_val: {X_val.shape}")

# --- Light random search: a handful of reasonable configs, not exhaustive ---
SEARCH_GRID = [
    dict(n_estimators=100, max_depth=5,  learning_rate=0.05, num_leaves=31),
    dict(n_estimators=200, max_depth=5,  learning_rate=0.05, num_leaves=31),
    dict(n_estimators=200, max_depth=7,  learning_rate=0.05, num_leaves=63),
    dict(n_estimators=200, max_depth=-1, learning_rate=0.03, num_leaves=31),
    dict(n_estimators=300, max_depth=7,  learning_rate=0.03, num_leaves=63),
    dict(n_estimators=300, max_depth=-1, learning_rate=0.05, num_leaves=127),
]

results = []
for i, params in enumerate(SEARCH_GRID, 1):
    print(f"\n{'='*60}")
    print(f"Config {i}/{len(SEARCH_GRID)}: {params}")
    print(f"{'='*60}")

    start = time.time()
    model = GBMForecastModel(horizon_days=HORIZON_DAYS, **params)
    model.fit(X_train, y_train, X_val, y_val)
    elapsed = time.time() - start

    val_pred = model.predict(X_val)
    rmse_by_day = rmse_per_horizon_day(y_val, val_pred)
    rmse_7_10 = float(rmse_by_day[6:10].mean())
    rmse_1_10 = float(rmse_by_day.mean())

    print(f"RMSE(7-10): {rmse_7_10:.4f}  RMSE(1-10): {rmse_1_10:.4f}  Time: {elapsed:.1f}s")

    results.append({**params, "rmse_7_10": rmse_7_10, "rmse_1_10": rmse_1_10,
                     "train_time_sec": elapsed})

results_df = pd.DataFrame(results).sort_values("rmse_7_10")
print(f"\n{'='*60}")
print("ALL RESULTS, sorted by rmse_7_10 (best first)")
print(f"{'='*60}")
print(results_df.to_string(index=False))

results_df.to_csv("scripts/gbm_output/tuning_results.csv", index=False)
print("\nSaved: scripts/gbm_output/tuning_results.csv")