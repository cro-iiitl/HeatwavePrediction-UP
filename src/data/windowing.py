"""
Fold-aware windowing — master §11.4.

Per-district sliding window generation, independent per district (no
cross-district contamination). Produces fixed-length input windows and
10-day direct-forecast targets from the Phase 0 engineered dataset.

Feature sets are tier-aware, not a single global constant:
  - TREE_FEATURE_COLUMNS: full set including T2M_MAX's own engineered
    lags/rolling stats. Needed for Tier 2 (GBM) and any model with no
    internal sequence memory of its own.
  - SEQUENCE_FEATURE_COLUMNS: excludes T2M_MAX_lag1/3/7/roll7_mean/std.
    For LSTM/GRU (Tier 3+), raw T2M_MAX is already present at every
    timestep of the input sequence, so its own engineered lags/rolling
    stats are redundant — the model can learn an equivalent (or better)
    representation internally. Other _lag1 features (SOIL_DRYNESS_lag1,
    ATM_DRYNESS_lag1, etc.) are NOT redundant and are kept in both sets:
    those represent genuinely different physical variables, lagged by
    1 day only because same-day values aren't observable at forecast
    time (Phase 0 §5.2 item 2) — not a summary of information already
    in the sequence.

Two independent exclusion checks applied strictly (drop, don't fill):
  1. has_excluded_gap (Phase 0 §5.1) — raw data gaps >3 days.
  2. NaN in any feature/target column — covers the mechanical lag/rolling
     warm-up period at the start of each district's series. Distinct from
     (1) and checked separately, or warm-up NaNs would silently leak into
     training.

This module does NOT assign windows to folds/splits — it only produces
valid windows with dated metadata. Fold/split boundary logic (train.py,
Phase A-2) decides which windows belong to which split, and must reject
any window whose target span crosses a split boundary (master §11.5).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMN = "T2M_MAX"
HORIZON_DAYS = 10

# Full feature set — GBM / tree-based models (Tier 2), no internal memory.
TREE_FEATURE_COLUMNS = [
    "T2M_MAX",
    "T2M_MAX_lag1", "T2M_MAX_lag3", "T2M_MAX_lag7",
    "T2M_MAX_roll7_mean", "T2M_MAX_roll7_std",
    "SOIL_DRYNESS_lag1", "ATM_DRYNESS_lag1", "RADIATION_EFFECTIVE_lag1",
    "CLOUD_AMT_lag1", "WS10M_lag1", "PRECTOTCORR_lag1", "PS_lag1",
    "DOY_SIN", "DOY_COS", "WD10M_SIN", "WD10M_COS",
]

# Sequence-native feature set — LSTM/GRU (Tier 3+). Drops T2M_MAX's own
# engineered lags/rolling stats since the sequence already carries that
# information at every timestep; keeps all other variables' lag1 features,
# since those are not redundant with anything else in the input.
SEQUENCE_FEATURE_COLUMNS = [
    "T2M_MAX",
    "SOIL_DRYNESS_lag1", "ATM_DRYNESS_lag1", "RADIATION_EFFECTIVE_lag1",
    "CLOUD_AMT_lag1", "WS10M_lag1", "PRECTOTCORR_lag1", "PS_lag1",
    "DOY_SIN", "DOY_COS", "WD10M_SIN", "WD10M_COS",
]


@dataclass
class WindowSet:
    """Materialized windows for one district."""
    district_id: str
    X: np.ndarray                # shape (n_windows, window_length, n_features)
    y: np.ndarray                 # shape (n_windows, horizon_days)
    input_start_dates: list[str]
    target_start_dates: list[str]
    target_end_dates: list[str]


def build_windows(
    df: pd.DataFrame,
    window_length: int,
    feature_cols: list[str],
    horizon_days: int = HORIZON_DAYS,
) -> WindowSet:
    """
    Build valid sliding windows for a single district's engineered dataframe.

    feature_cols is REQUIRED, not defaulted — callers must explicitly pass
    TREE_FEATURE_COLUMNS or SEQUENCE_FEATURE_COLUMNS (or a custom set),
    so the tier-appropriate choice is visible at the call site, not buried
    in a module-level default that's easy to forget is tier-specific.

    df must be sorted chronologically and contain exactly one district's data.
    A window spanning positions [i, i+window_length-1] as input and
    [i+window_length, i+window_length+horizon_days-1] as target is INCLUDED
    only if every row across that full combined span has:
      - has_excluded_gap == False, AND
      - no NaN in any feature_cols or the target column.
    Any violation anywhere in the span drops the whole window (strict
    exclusion, per project policy).
    """
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    total_span = window_length + horizon_days

    district_id = df["district_id"].iloc[0]

    feature_arr = df[feature_cols].to_numpy(dtype=np.float64)
    target_arr = df[TARGET_COLUMN].to_numpy(dtype=np.float64)
    gap_flag = df["has_excluded_gap"].to_numpy(dtype=bool)
    dates = df["date"].to_numpy()

    row_valid = (
        ~np.isnan(feature_arr).any(axis=1)
        & ~np.isnan(target_arr)
        & ~gap_flag
    )

    X_list, y_list = [], []
    input_starts, target_starts, target_ends = [], [], []

    for i in range(0, n - total_span + 1):
        span = slice(i, i + total_span)
        if not row_valid[span].all():
            continue

        input_slice = slice(i, i + window_length)
        target_slice = slice(i + window_length, i + total_span)

        X_list.append(feature_arr[input_slice])
        y_list.append(target_arr[target_slice])
        input_starts.append(str(dates[i]))
        target_starts.append(str(dates[i + window_length]))
        target_ends.append(str(dates[i + total_span - 1]))

    if not X_list:
        X = np.empty((0, window_length, len(feature_cols)))
        y = np.empty((0, horizon_days))
    else:
        X = np.stack(X_list)
        y = np.stack(y_list)

    return WindowSet(
        district_id=district_id,
        X=X,
        y=y,
        input_start_dates=input_starts,
        target_start_dates=target_starts,
        target_end_dates=target_ends,
    )


def build_windows_multi_district(
    engineered_dir: str,
    district_ids: list[str],
    window_length: int,
    feature_cols: list[str],
    horizon_days: int = HORIZON_DAYS,
) -> list[WindowSet]:
    """
    Build windows independently per district, returned as a list of
    WindowSet (one per district) — NOT concatenated here. Callers decide
    how/when to concatenate.
    """
    results = []
    for did in district_ids:
        path = Path(engineered_dir) / f"{did}.parquet"
        df = pd.read_parquet(path)
        ws = build_windows(df, window_length, feature_cols, horizon_days)
        results.append(ws)
    return results