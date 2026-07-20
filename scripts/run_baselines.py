"""
Tier 0/1 baseline models — master §7.

Tier 0: persistence, rolling-mean, climatological-normal
Tier 1: seasonal-naive

No learned parameters, no windowing.py dependency. Evaluated on the same
fixed reference split (train 1984-2010, val 2011-2018) and same
validation target-date range as the Tier 3 LSTM ablation, so RMSE
numbers are directly comparable -- this is the floor any learned model
must beat (master §8.8, §13.2).
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

TRAIN_START, TRAIN_END = "1984-01-01", "2010-12-31"
VAL_START, VAL_END = "2011-01-01", "2018-12-31"
HORIZON_DAYS = 10
ROLLING_WINDOW = 7

with open("configs/data.yaml") as f:
    all_districts = [d["district_id"] for d in yaml.safe_load(f)["districts"]]


def compute_climatological_normal(train_df: pd.DataFrame) -> dict:
    dates = pd.to_datetime(train_df["date"])
    key = list(zip(dates.dt.month, dates.dt.day))
    temp = pd.DataFrame({"key": key, "T2M_MAX": train_df["T2M_MAX"].values})
    return temp.groupby("key")["T2M_MAX"].mean().to_dict()


def evaluate_district(district_id: str):
    df = pd.read_parquet(f"data/engineered/{district_id}.parquet")
    df = df.sort_values("date").reset_index(drop=True)
    dates_parsed = pd.to_datetime(df["date"])

    train_df = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)]
    normal = compute_climatological_normal(train_df)

    date_to_idx = {d: i for i, d in enumerate(df["date"])}
    t2m = df["T2M_MAX"].to_numpy()
    dates = df["date"].to_numpy()

    errors = {k: [] for k in ["persistence", "rolling_mean", "climatological_normal", "seasonal_naive"]}

    n = len(df)
    for i in range(ROLLING_WINDOW - 1, n - HORIZON_DAYS):
        target_start = dates[i + 1]
        target_end = dates[i + HORIZON_DAYS]
        if not (VAL_START <= target_start and target_end <= VAL_END):
            continue

        y_true = t2m[i + 1:i + 1 + HORIZON_DAYS]
        if np.isnan(y_true).any():
            continue

        persistence_pred = np.full(HORIZON_DAYS, t2m[i])

        roll_val = t2m[i - ROLLING_WINDOW + 1:i + 1].mean()
        rolling_pred = np.full(HORIZON_DAYS, roll_val)

        clim_pred = np.array([
            normal.get((dates_parsed.iloc[i + h].month, dates_parsed.iloc[i + h].day), np.nan)
            for h in range(1, HORIZON_DAYS + 1)
        ])
        if np.isnan(clim_pred).any():
            continue

        seasonal_pred = []
        ok = True
        for h in range(1, HORIZON_DAYS + 1):
            lookup_date = dates_parsed.iloc[i + h] - pd.DateOffset(years=1)
            lookup_str = lookup_date.strftime("%Y-%m-%d")
            if lookup_str not in date_to_idx:
                ok = False
                break
            seasonal_pred.append(t2m[date_to_idx[lookup_str]])
        if not ok:
            continue
        seasonal_pred = np.array(seasonal_pred)
        if np.isnan(seasonal_pred).any():
            continue

        errors["persistence"].append((y_true - persistence_pred) ** 2)
        errors["rolling_mean"].append((y_true - rolling_pred) ** 2)
        errors["climatological_normal"].append((y_true - clim_pred) ** 2)
        errors["seasonal_naive"].append((y_true - seasonal_pred) ** 2)

    return errors


all_errors = {k: [] for k in ["persistence", "rolling_mean", "climatological_normal", "seasonal_naive"]}

print(f"Evaluating baselines across {len(all_districts)} districts...")
for i, did in enumerate(all_districts, 1):
    if i % 15 == 0:
        print(f"  {i}/{len(all_districts)}")
    errs = evaluate_district(did)
    for method, sq_errs in errs.items():
        all_errors[method].extend(sq_errs)

print(f"\n{'='*60}")
print("BASELINE RESULTS (pooled across all 75 districts)")
print(f"{'='*60}")

results = {}
for method, sq_errs in all_errors.items():
    sq_errs = np.array(sq_errs)  # shape (n_samples, 10)
    rmse_per_day = np.sqrt(sq_errs.mean(axis=0))
    rmse_7_10 = rmse_per_day[6:10].mean()
    rmse_1_10 = rmse_per_day.mean()

    print(f"\n{method}:")
    print(f"  n_samples: {len(sq_errs)}")
    print(f"  RMSE per horizon day: {np.round(rmse_per_day, 3)}")
    print(f"  RMSE avg (days 7-10): {rmse_7_10:.4f}")
    print(f"  RMSE avg (days 1-10): {rmse_1_10:.4f}")

    results[method] = {
        "rmse_per_day": rmse_per_day.tolist(),
        "rmse_avg_7_10": float(rmse_7_10),
        "rmse_avg_1_10": float(rmse_1_10),
    }

print(f"\n{'='*60}")
print("COMPARISON: baselines vs. Tier 3 LSTM (window=15, from prior run)")
print(f"{'='*60}")
lstm_rmse_7_10 = 2.3646
lstm_rmse_1_10 = 2.1697
print(f"LSTM (window=15):        rmse_7_10={lstm_rmse_7_10:.4f}  rmse_1_10={lstm_rmse_1_10:.4f}")
for method, r in results.items():
    beats = "BEATS LSTM" if r["rmse_7_10"] < lstm_rmse_7_10 else "LSTM beats this"
    print(f"{method:25s} rmse_7_10={r['rmse_avg_7_10']:.4f}  rmse_1_10={r['rmse_avg_1_10']:.4f}   [{beats}]")

import json
Path("scripts/baseline_output").mkdir(exist_ok=True)
with open("scripts/baseline_output/baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: scripts/baseline_output/baseline_results.json")