"""
Test: does adding lat/lon (district identity) improve LSTM's walk-forward
performance? Compares the locked LSTM config with vs. without lat/lon,
across all 3 walk-forward folds.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import yaml

from src.data.windowing import build_windows
from src.eval.walk_forward import FOLDS, split_windows_by_fold
from src.models.lstm_tier3 import LSTMForecastModel

WINDOW_LENGTH = 15
HORIZON_DAYS = 10

SEQUENCE_FEATURES_BASELINE = [
    "T2M_MAX",
    "SOIL_DRYNESS_lag1", "ATM_DRYNESS_lag1", "RADIATION_EFFECTIVE_lag1",
    "CLOUD_AMT_lag1", "WS10M_lag1", "PRECTOTCORR_lag1", "PS_lag1",
    "DOY_SIN", "DOY_COS", "WD10M_SIN", "WD10M_COS",
]
SEQUENCE_FEATURES_WITH_LOCATION = SEQUENCE_FEATURES_BASELINE + ["lat", "lon"]

with open("configs/data.yaml") as f:
    district_ids = [d["district_id"] for d in yaml.safe_load(f)["districts"]]

with open("configs/model_lstm.yaml") as f:
    lstm_cfg = yaml.safe_load(f)["hyperparameters"]
    lstm_cfg["lr"] = lstm_cfg.pop("learning_rate")


def rmse_per_horizon_day(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def run_experiment(feature_cols: list, label: str):
    print(f"\n{'#'*60}\n{label} ({len(feature_cols)} features)\n{'#'*60}")

    fold_rmses = []
    for fold in FOLDS:
        fold_id = fold["fold_id"]
        window_sets = []
        for did in district_ids:
            df = pd.read_parquet(f"data/engineered/{did}.parquet")
            ws = build_windows(df, WINDOW_LENGTH, feature_cols, HORIZON_DAYS)
            window_sets.append(ws)

        X_train, y_train, _, X_test, y_test, _ = split_windows_by_fold(window_sets, fold)

        model = LSTMForecastModel(n_features=X_train.shape[2], **lstm_cfg)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse_by_day = rmse_per_horizon_day(y_test, preds)
        rmse_7_10 = float(rmse_by_day[6:10].mean())
        fold_rmses.append(rmse_7_10)
        print(f"  Fold {fold_id}: rmse_7_10={rmse_7_10:.4f}")

    avg_rmse = float(np.mean(fold_rmses))
    print(f"  AVERAGE: {avg_rmse:.4f}")
    return fold_rmses, avg_rmse


baseline_folds, baseline_avg = run_experiment(SEQUENCE_FEATURES_BASELINE, "WITHOUT lat/lon (current)")
location_folds, location_avg = run_experiment(SEQUENCE_FEATURES_WITH_LOCATION, "WITH lat/lon")

print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
print(f"Without lat/lon: {baseline_avg:.4f}")
print(f"With lat/lon:    {location_avg:.4f}")
print(f"Difference:      {baseline_avg - location_avg:.4f} "
      f"({'improvement' if location_avg < baseline_avg else 'no improvement'})")

results_df = pd.DataFrame([
    {"variant": "without_lat_lon", "fold1": baseline_folds[0], "fold2": baseline_folds[1],
     "fold3": baseline_folds[2], "avg": baseline_avg},
    {"variant": "with_lat_lon", "fold1": location_folds[0], "fold2": location_folds[1],
     "fold3": location_folds[2], "avg": location_avg},
])
results_df.to_csv("scripts/kaggle_output/lstm_lat_lon_feature_test.csv", index=False)
print("\nSaved: scripts/kaggle_output/lstm_lat_lon_feature_test.csv")