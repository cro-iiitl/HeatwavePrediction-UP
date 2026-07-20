"""
Phase A-1 Step 2 — empirical window-length ablation (§4).
Trains Tier 3 (LSTM) four times, window in {10,15,20,30}, fixed reference
split, placeholder hyperparameters (§4.1 — not final).

SMOKE_TEST_DISTRICTS: set to a small list first to validate mechanics
cheaply before running against all 75.
"""

import sys
sys.path.insert(0, ".")

import json
from pathlib import Path

import numpy as np
import yaml

from src.data.windowing import build_windows, SEQUENCE_FEATURE_COLUMNS
from src.models.lstm_tier3 import LSTMForecastModel

# --- Config ---
CANDIDATE_WINDOW_LENGTHS = [10, 15, 20, 30]
TRAIN_START, TRAIN_END = "1984-01-01", "2010-12-31"
VAL_START, VAL_END = "2011-01-01", "2018-12-31"
HORIZON_DAYS = 10

# SMOKE TEST: set to a handful of districts first. Switch to full 75 once
# mechanics are confirmed correct.
with open("configs/data.yaml") as f:
    all_districts = [d["district_id"] for d in yaml.safe_load(f)["districts"]]

SMOKE_TEST_DISTRICTS = all_districts
DISTRICTS_TO_USE = SMOKE_TEST_DISTRICTS

PLACEHOLDER_HPARAMS = dict(
    hidden_units=64, dropout=0.2, dense_units=32,
    batch_size=128, max_epochs=30, early_stopping_patience=15,
)

OUTPUT_DIR = Path("scripts/window_ablation_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_split_windows(window_length: int):
    """Pool windows across all target districts, split by target-date range."""
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []

    for district_id in DISTRICTS_TO_USE:
        import pandas as pd
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
            # windows whose target spans the boundary are silently excluded
            # (correct: neither split can legitimately claim them)

    X_train = np.stack(X_train_list)
    y_train = np.stack(y_train_list)
    X_val = np.stack(X_val_list)
    y_val = np.stack(y_val_list)
    return X_train, y_train, X_val, y_val


def rmse_per_horizon_day(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


results = {}

for window_length in CANDIDATE_WINDOW_LENGTHS:
    print(f"\n{'='*60}")
    print(f"Window length: {window_length} days")
    print(f"{'='*60}")

    X_train, y_train, X_val, y_val = build_split_windows(window_length)
    print(f"Train windows: {len(X_train)}  |  Val windows: {len(X_val)}")

    n_features = X_train.shape[2]
    model = LSTMForecastModel(n_features=n_features, **PLACEHOLDER_HPARAMS)
    model.fit(X_train, y_train, X_val, y_val)

    val_pred = model.predict(X_val)
    rmse_by_day = rmse_per_horizon_day(y_val, val_pred)

    print(f"Validation RMSE per horizon day: {np.round(rmse_by_day, 3)}")
    print(f"RMSE avg (days 7-10): {rmse_by_day[6:10].mean():.4f}")
    print(f"RMSE avg (days 1-10): {rmse_by_day.mean():.4f}")

    results[window_length] = {
        "rmse_per_day": rmse_by_day.tolist(),
        "rmse_avg_7_10": float(rmse_by_day[6:10].mean()),
        "rmse_avg_1_10": float(rmse_by_day.mean()),
        "n_train_windows": len(X_train),
        "n_val_windows": len(X_val),
    }

# --- Decision rule (§5) ---
print(f"\n{'='*60}")
print("DECISION")
print(f"{'='*60}")

best_window = min(results, key=lambda w: results[w]["rmse_avg_7_10"])
print(f"Primary criterion (lowest RMSE, horizon days 7-10): window = {best_window}")
for w, r in sorted(results.items()):
    print(f"  {w}: rmse_7_10={r['rmse_avg_7_10']:.4f}  rmse_1_10={r['rmse_avg_1_10']:.4f}")

with open(OUTPUT_DIR / "ablation_results.json", "w") as f:
    json.dump({"results": results, "chosen_window_length": best_window}, f, indent=2)

print(f"\nSaved: {OUTPUT_DIR / 'ablation_results.json'}")