"""
Phase A-1 Step 1 — ACF/PACF diagnostic on the anomaly series.

Computes climatological normal from 1984-2010 training years only
(fixed reference split, master §5.3/§11.3), grouped by (month, day)
to avoid leap-year day-of-year misalignment. Anomaly = T2M_MAX(t) -
climatological_normal(month, day, district).

Produces ACF/PACF plots and a written note identifying the lag beyond
which PACF is no longer distinguishable from noise, per district.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import pacf

DISTRICTS = ["banda", "kushinagar"]
TRAIN_START = "1984-01-01"
TRAIN_END = "2010-12-31"
MAX_LAG_DAYS = 40
OUTPUT_DIR = "scripts/acf_pacf_output"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_climatological_normal(train_df: pd.DataFrame) -> dict:
    """(month, day) -> mean T2M_MAX across training years only."""
    dates = pd.to_datetime(train_df["date"])
    key = list(zip(dates.dt.month, dates.dt.day))
    temp = pd.DataFrame({"key": key, "T2M_MAX": train_df["T2M_MAX"].values})
    normal = temp.groupby("key")["T2M_MAX"].mean().to_dict()
    return normal


def compute_anomaly_series(df: pd.DataFrame, normal: dict) -> pd.Series:
    dates = pd.to_datetime(df["date"])
    key = list(zip(dates.dt.month, dates.dt.day))
    normal_vals = np.array([normal[k] for k in key])
    return df["T2M_MAX"].values - normal_vals


def find_pacf_cutoff_lag(anomaly: np.ndarray, max_lag: int) -> int:
    """First lag beyond which PACF stays within the 95% noise band."""
    n = len(anomaly)
    conf_bound = 1.96 / np.sqrt(n)
    pacf_vals = pacf(anomaly, nlags=max_lag)

    # pacf_vals[0] is lag 0 (always 1.0) — skip it
    for lag in range(1, max_lag + 1):
        if abs(pacf_vals[lag]) < conf_bound:
            # Check the next few lags also stay inside the band, to avoid
            # calling a single noisy crossing the "cutoff" prematurely
            following = pacf_vals[lag:min(lag + 5, max_lag + 1)]
            if all(abs(v) < conf_bound for v in following):
                return lag
    return max_lag  # never clearly settles within max_lag


results = {}

for district_id in DISTRICTS:
    print(f"\n{'='*60}")
    print(f"District: {district_id}")
    print(f"{'='*60}")

    df = pd.read_parquet(f"data/engineered/{district_id}.parquet")
    df = df.sort_values("date").reset_index(drop=True)

    train_df = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].copy()
    print(f"Training rows (1984-2010): {len(train_df)}")

    normal = compute_climatological_normal(train_df)
    print(f"Climatological normal computed for {len(normal)} calendar days")

    # Anomaly series computed on the SAME training-year window only —
    # this diagnostic operates entirely within the fixed reference split.
    anomaly = compute_anomaly_series(train_df, normal)

    print(f"Anomaly series: mean={anomaly.mean():.4f} (expect ~0), "
          f"std={anomaly.std():.4f}")

    cutoff_lag = find_pacf_cutoff_lag(anomaly, MAX_LAG_DAYS)
    print(f"PACF cutoff lag (first lag where signal drops into noise "
          f"band and stays there): {cutoff_lag}")

    results[district_id] = {
        "n_train_days": len(train_df),
        "anomaly_mean": float(anomaly.mean()),
        "anomaly_std": float(anomaly.std()),
        "pacf_cutoff_lag": cutoff_lag,
    }

    # Plots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(anomaly, lags=MAX_LAG_DAYS, ax=axes[0])
    axes[0].set_title(f"{district_id} — ACF (anomaly series, secondary/context)")
    plot_pacf(anomaly, lags=MAX_LAG_DAYS, ax=axes[1], method="ywm")
    axes[1].set_title(f"{district_id} — PACF (anomaly series, primary diagnostic)")
    axes[1].axvline(cutoff_lag, color="red", linestyle="--",
                     label=f"cutoff lag = {cutoff_lag}")
    axes[1].legend()
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/{district_id}_acf_pacf.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved plot: {out_path}")


# --- Written note (Phase A-1 §3.4 requires this, not just plots) ---
print(f"\n{'='*60}")
print("WRITTEN NOTE — Step 1 output (Phase A-1 §3.4)")
print(f"{'='*60}")

for district_id, r in results.items():
    print(f"\n{district_id}:")
    print(f"  PACF becomes indistinguishable from noise beyond lag {r['pacf_cutoff_lag']}")

lags = [r["pacf_cutoff_lag"] for r in results.values()]
if len(set(lags)) == 1:
    print(f"\nAgreement: both sampled districts agree — cutoff at lag {lags[0]}.")
else:
    print(f"\nDisagreement: cutoff lags differ across districts "
          f"({dict(zip(results.keys(), lags))}). "
          f"This should be noted, not silently averaged.")

note_path = f"{OUTPUT_DIR}/written_note.txt"
with open(note_path, "w") as f:
    f.write("Phase A-1 Step 1 — ACF/PACF Diagnostic Written Note\n")
    f.write("=" * 55 + "\n\n")
    for district_id, r in results.items():
        f.write(f"{district_id}:\n")
        f.write(f"  Training rows used (1984-2010): {r['n_train_days']}\n")
        f.write(f"  Anomaly series mean: {r['anomaly_mean']:.4f} (sanity check, expect ~0)\n")
        f.write(f"  Anomaly series std: {r['anomaly_std']:.4f}\n")
        f.write(f"  PACF cutoff lag: {r['pacf_cutoff_lag']}\n\n")
    if len(set(lags)) == 1:
        f.write(f"Agreement: both districts agree on cutoff lag {lags[0]}.\n")
    else:
        f.write(f"Disagreement: cutoff lags differ - {dict(zip(results.keys(), lags))}\n")
print(f"\nWritten note saved: {note_path}")