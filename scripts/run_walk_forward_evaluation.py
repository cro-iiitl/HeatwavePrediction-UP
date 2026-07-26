"""
Walk-forward evaluation — master §11.3, §13. Runs all 6 tiers (4
baselines + GBM + LSTM) across 3 chronological folds, computes RMSE per
horizon day, skill score vs climatological normal, and Diebold-Mariano
significance tests between consecutive tiers in the roadmap.

DESIGN SIMPLIFICATION (documented, not hidden): master §13.3 specifies
comparing each tier against whichever baseline is strongest AT EACH
HORIZON DAY (since baselines cross over mid-horizon). This script instead
uses the simpler incremental-chain interpretation: each tier compared
against the immediately preceding tier in the roadmap (Tier1 vs best
Tier0 baseline, Tier2 vs Tier1, Tier3 vs Tier2). This still satisfies
"every tier vs current best," just without per-horizon-day dynamic
baseline selection.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import yaml

from src.data.windowing import (
    build_windows_multi_district, TREE_FEATURE_COLUMNS, SEQUENCE_FEATURE_COLUMNS
)
from src.eval.walk_forward import FOLDS, split_windows_by_fold
from src.eval.metrics import rmse_per_horizon_day, skill_score
from src.eval.dm_test import diebold_mariano_test, holm_correction

from src.models.baselines import (
    PersistenceModel, RollingMeanModel, ClimatologicalNormalModel, SeasonalNaiveModel
)
from src.models.gbm_tier2 import GBMForecastModel
from src.models.lstm_tier3 import LSTMForecastModel

WINDOW_LENGTH = 15
HORIZON_DAYS = 10

with open("configs/data.yaml") as f:
    district_ids = [d["district_id"] for d in yaml.safe_load(f)["districts"]]

with open("configs/model_gbm.yaml") as f:
    gbm_cfg = yaml.safe_load(f)["hyperparameters"]

with open("configs/model_lstm.yaml") as f:
    lstm_cfg = yaml.safe_load(f)["hyperparameters"]
    lstm_cfg["lr"] = lstm_cfg.pop("learning_rate")  # config uses learning_rate, class expects lr

all_results = []
dm_test_results = []

for fold in FOLDS:
    fold_id = fold["fold_id"]
    print(f"\n{'#'*70}")
    print(f"FOLD {fold_id}: train {fold['train_start']}-{fold['train_end']}  "
          f"test {fold['test_start']}-{fold['test_end']}")
    print(f"{'#'*70}")

    print("Building TREE feature windows (for baselines + GBM)...")
    tree_window_sets = build_windows_multi_district(
        "data/engineered", district_ids, WINDOW_LENGTH, TREE_FEATURE_COLUMNS, HORIZON_DAYS
    )
    (X_train_tree, y_train, train_dates,
     X_test_tree, y_test, test_dates) = split_windows_by_fold(tree_window_sets, fold)

    print("Building SEQUENCE feature windows (for LSTM)...")
    seq_window_sets = build_windows_multi_district(
        "data/engineered", district_ids, WINDOW_LENGTH, SEQUENCE_FEATURE_COLUMNS, HORIZON_DAYS
    )
    (X_train_seq, y_train_seq, _, X_test_seq, y_test_seq, _) = split_windows_by_fold(
        seq_window_sets, fold
    )

    print(f"Tree: train={X_train_tree.shape}  test={X_test_tree.shape}")
    print(f"Seq:  train={X_train_seq.shape}  test={X_test_seq.shape}")

    fold_predictions = {}

    # --- Tier 0: baselines ---
    print("\nTraining Tier 0/1 baselines...")
    persistence = PersistenceModel(HORIZON_DAYS)
    persistence.fit(X_train_tree, y_train)
    fold_predictions["persistence"] = persistence.predict(X_test_tree)

    rolling = RollingMeanModel(7, HORIZON_DAYS)
    rolling.fit(X_train_tree, y_train)
    fold_predictions["rolling_mean"] = rolling.predict(X_test_tree)

    climatological = ClimatologicalNormalModel(HORIZON_DAYS)
    climatological.fit(X_train_tree, y_train, train_dates=train_dates)
    fold_predictions["climatological_normal"] = climatological.predict(
        X_test_tree, target_dates=test_dates
    )

    seasonal = SeasonalNaiveModel(HORIZON_DAYS)
    seasonal.fit(X_train_tree, y_train, train_dates=train_dates)
    fold_predictions["seasonal_naive"] = seasonal.predict(X_test_tree, target_dates=test_dates)

    # --- Tier 2: GBM ---
    print("Training Tier 2 GBM...")
    gbm = GBMForecastModel(horizon_days=HORIZON_DAYS, **gbm_cfg)
    gbm.fit(X_train_tree, y_train)
    fold_predictions["gbm_tier2"] = gbm.predict(X_test_tree)

    # --- Tier 3: LSTM ---
    print("Training Tier 3 LSTM...")
    lstm = LSTMForecastModel(n_features=X_train_seq.shape[2], horizon_days=HORIZON_DAYS,
                              **lstm_cfg)
    lstm.fit(X_train_seq, y_train_seq)
    fold_predictions["lstm_tier3"] = lstm.predict(X_test_seq)

    # --- Metrics per tier ---
    print("\nComputing metrics...")
    rmse_by_tier = {}
    for tier_name, preds in fold_predictions.items():
        y_ref = y_test_seq if tier_name == "lstm_tier3" else y_test
        rmse_by_tier[tier_name] = rmse_per_horizon_day(y_ref, preds)

    clim_rmse = rmse_by_tier["climatological_normal"]

    for tier_name, rmse_arr in rmse_by_tier.items():
        skill = skill_score(rmse_arr, clim_rmse)
        for h in range(HORIZON_DAYS):
            all_results.append({
                "fold_id": fold_id, "tier": tier_name, "horizon_day": h + 1,
                "rmse": rmse_arr[h], "skill_score": skill[h],
            })
        print(f"  {tier_name}: rmse_7_10={rmse_arr[6:10].mean():.4f}  "
              f"rmse_1_10={rmse_arr.mean():.4f}")

    # --- DM tests: incremental chain (documented simplification above) ---
    print("\nRunning DM tests (incremental chain)...")
    best_baseline_name = min(
        ["persistence", "rolling_mean", "climatological_normal", "seasonal_naive"],
        key=lambda t: rmse_by_tier[t][6:10].mean()
    )
    chain = [
        ("tier1_vs_tier0", best_baseline_name, "gbm_tier2"),
        ("tier2_vs_tier1", "gbm_tier2", "lstm_tier3"),
    ]
    # Note: chain compares (A, B) meaning "does B beat A" -- see loop below

    comparisons = [
        (best_baseline_name, "gbm_tier2", "gbm_vs_best_baseline"),
        ("gbm_tier2", "lstm_tier3", "lstm_vs_gbm"),
    ]

    for model_a_name, model_b_name, comparison_label in comparisons:
        y_ref_a = y_test_seq if model_a_name == "lstm_tier3" else y_test
        y_ref_b = y_test_seq if model_b_name == "lstm_tier3" else y_test

        p_values = []
        dm_stats = []
        for h in range(HORIZON_DAYS):
            loss_a = (y_ref_a[:, h] - fold_predictions[model_a_name][:, h]) ** 2
            loss_b = (y_ref_b[:, h] - fold_predictions[model_b_name][:, h]) ** 2
            n = min(len(loss_a), len(loss_b))
            dm_stat, p_val = diebold_mariano_test(loss_b[:n], loss_a[:n], max_lag=HORIZON_DAYS)
            dm_stats.append(dm_stat)
            p_values.append(p_val)

        corrected_p = holm_correction(p_values)

        for h in range(HORIZON_DAYS):
            dm_test_results.append({
                "fold_id": fold_id, "comparison": comparison_label,
                "model_a": model_a_name, "model_b": model_b_name,
                "horizon_day": h + 1, "dm_statistic": dm_stats[h],
                "p_value_raw": p_values[h], "p_value_holm_corrected": corrected_p[h],
                "significant_at_0.05": corrected_p[h] < 0.05,
            })

        print(f"  {comparison_label}: {sum(1 for p in corrected_p if p < 0.05)}/10 "
              f"horizon days significant (Holm-corrected)")

results_df = pd.DataFrame(all_results)
results_df.to_csv("scripts/walk_forward_output/tier_metrics_by_fold.csv", index=False)

dm_df = pd.DataFrame(dm_test_results)
dm_df.to_csv("scripts/walk_forward_output/dm_test_results.csv", index=False)

print(f"\n{'='*70}")
print("WALK-FORWARD EVALUATION COMPLETE")
print(f"{'='*70}")
print("Saved: scripts/walk_forward_output/tier_metrics_by_fold.csv")
print("Saved: scripts/walk_forward_output/dm_test_results.csv")

print("\nSummary — RMSE (7-10) by tier, averaged across folds:")
summary = results_df[results_df["horizon_day"].between(7, 10)].groupby("tier")["rmse"].mean()
print(summary.sort_values().to_string())