"""
GBM seed-variance check — closes a gap flagged early in Tier 2's
evaluation and never verified: does the locked GBM config's reported
RMSE hold up across different random seeds, or does it carry meaningful
unmeasured variance the way LSTM's window-length "win" once did?

Trains the locked GBM config (n_estimators=200, max_depth=5, lr=0.05,
num_leaves=31) 3 times with different random_state values, across all
3 walk-forward folds, full 75-district pool.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import yaml

from src.data.windowing import build_windows, TREE_FEATURE_COLUMNS
from src.eval.walk_forward import FOLDS, split_windows_by_fold
from src.models.gbm_tier2 import GBMForecastModel

WINDOW_LENGTH = 15
HORIZON_DAYS = 10
SEEDS = [42, 43, 44]  # 42 is the original locked config's seed

with open("configs/data.yaml") as f:
    district_ids = [d["district_id"] for d in yaml.safe_load(f)["districts"]]

with open("configs/model_gbm.yaml") as f:
    gbm_cfg = yaml.safe_load(f)["hyperparameters"]


def rmse_per_horizon_day(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


print("Building windowed data for all 3 folds (tree features only)...")
fold_windows = {}
for fold in FOLDS:
    fold_id = fold["fold_id"]
    print(f"  Fold {fold_id}...")
    window_sets = []
    for did in district_ids:
        df = pd.read_parquet(f"data/engineered/{did}.parquet")
        ws = build_windows(df, WINDOW_LENGTH, TREE_FEATURE_COLUMNS, HORIZON_DAYS)
        window_sets.append(ws)
    X_train, y_train, _, X_test, y_test, _ = split_windows_by_fold(window_sets, fold)
    fold_windows[fold_id] = (X_train, y_train, X_test, y_test)
    print(f"    train={X_train.shape}  test={X_test.shape}")

print("\nRunning seed variance check...")
results = []

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"Seed: {seed}")
    print(f"{'='*50}")

    fold_rmses = []
    for fold in FOLDS:
        fold_id = fold["fold_id"]
        X_train, y_train, X_test, y_test = fold_windows[fold_id]

        cfg_with_seed = {**gbm_cfg, "random_state": seed}
        model = GBMForecastModel(horizon_days=HORIZON_DAYS, **cfg_with_seed)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse_by_day = rmse_per_horizon_day(y_test, preds)
        rmse_7_10 = float(rmse_by_day[6:10].mean())
        fold_rmses.append(rmse_7_10)
        print(f"  Fold {fold_id}: rmse_7_10={rmse_7_10:.4f}")

    avg_rmse = float(np.mean(fold_rmses))
    print(f"  AVERAGE across folds: {avg_rmse:.4f}")

    results.append({
        "seed": seed,
        "fold1_rmse": fold_rmses[0], "fold2_rmse": fold_rmses[1], "fold3_rmse": fold_rmses[2],
        "avg_rmse_7_10": avg_rmse,
    })

results_df = pd.DataFrame(results)
print(f"\n{'='*60}")
print("GBM SEED VARIANCE — SUMMARY")
print(f"{'='*60}")
print(results_df.to_string(index=False))

avg_vals = results_df["avg_rmse_7_10"].values
seed_mean = avg_vals.mean()
seed_std = avg_vals.std()
seed_range = avg_vals.max() - avg_vals.min()

print(f"\nMean across seeds: {seed_mean:.4f}")
print(f"Std across seeds: {seed_std:.4f}")
print(f"Range across seeds: {seed_range:.4f}")

lstm_gap = 2.6385 - seed_mean  # gap to LSTM's confirmed walk-forward result
print(f"\nGap between GBM and LSTM (walk-forward): {lstm_gap:.4f}")
print(f"GBM seed range as fraction of the GBM-LSTM gap: {seed_range / lstm_gap:.2%}")

if seed_range < lstm_gap * 0.2:
    print("\nCONCLUSION: GBM seed variance is small relative to the GBM-LSTM "
          "gap. The 'GBM beats LSTM' conclusion is robust to GBM's own "
          "training randomness.")
else:
    print("\nWARNING: GBM seed variance is non-trivial relative to the "
          "GBM-LSTM gap. Investigate further before treating 'GBM wins' "
          "as fully settled.")

results_df.to_csv("scripts/gbm_output/seed_variance_results.csv", index=False)
print("\nSaved: scripts/gbm_output/seed_variance_results.csv")