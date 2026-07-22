"""
One-off verification: does GBMForecastModel.save()/load() actually
round-trip correctly? Trains once on a small subset (not the full
75-district pool -- this only needs to test serialization, not
performance), saves, reloads, and confirms identical predictions.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import yaml

from src.data.windowing import build_windows, TREE_FEATURE_COLUMNS
from src.models.gbm_tier2 import GBMForecastModel

TRAIN_START, TRAIN_END = "1984-01-01", "2010-12-31"
VAL_START, VAL_END = "2011-01-01", "2018-12-31"
WINDOW_LENGTH = 15
HORIZON_DAYS = 10

# Small subset -- 5 districts, just enough to exercise the code path
with open("configs/data.yaml") as f:
    all_districts = [d["district_id"] for d in yaml.safe_load(f)["districts"]]
test_districts = all_districts[:5]

X_train_list, y_train_list = [], []
X_val_list, y_val_list = [], []

for did in test_districts:
    df = pd.read_parquet(f"data/engineered/{did}.parquet")
    ws = build_windows(df, WINDOW_LENGTH, TREE_FEATURE_COLUMNS, HORIZON_DAYS)
    for j in range(len(ws.input_start_dates)):
        t_start, t_end = ws.target_start_dates[j], ws.target_end_dates[j]
        if TRAIN_START <= t_start and t_end <= TRAIN_END:
            X_train_list.append(ws.X[j]); y_train_list.append(ws.y[j])
        elif VAL_START <= t_start and t_end <= VAL_END:
            X_val_list.append(ws.X[j]); y_val_list.append(ws.y[j])

X_train = np.stack(X_train_list)
y_train = np.stack(y_train_list)
X_val = np.stack(X_val_list)
y_val = np.stack(y_val_list)

print(f"Training on {len(X_train)} samples (5-district subset, verification only)...")

model = GBMForecastModel(
    n_estimators=200, max_depth=5, learning_rate=0.05, num_leaves=31,
)
model.fit(X_train, y_train, X_val, y_val)

original_predictions = model.predict(X_val)
print(f"Original predictions shape: {original_predictions.shape}")

SAVE_PATH = "scripts/gbm_output/verify_save_load_test.pkl"
model.save(SAVE_PATH)
print(f"Saved to {SAVE_PATH}")

reloaded_model = GBMForecastModel.load(SAVE_PATH)
reloaded_predictions = reloaded_model.predict(X_val)
print(f"Reloaded predictions shape: {reloaded_predictions.shape}")

identical = np.array_equal(original_predictions, reloaded_predictions)
max_diff = np.max(np.abs(original_predictions - reloaded_predictions))

print(f"\nPredictions identical: {identical}")
print(f"Max absolute difference: {max_diff}")

if identical:
    print("\nPASS -- save/load round-trip produces identical predictions.")
else:
    print("\nFAIL -- reloaded model differs from original. Investigate before "
          "trusting save/load in the walk-forward runner.")

import os
os.remove(SAVE_PATH)
print(f"\nCleaned up: {SAVE_PATH}")