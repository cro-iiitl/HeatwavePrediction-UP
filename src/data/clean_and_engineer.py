"""
Missing-value handling and feature engineering — Phase 0 §5.

Runs per district, independently — no cross-district contamination.
Reads data/raw/{district_id}.parquet, writes data/engineered/{district_id}.parquet
plus a global data/engineered/data_quality_report.csv.

Does NOT compute climatological normal — deferred to Phase A-2, per-fold,
per master §11.5 (leakage prevention). This file must never attach a
single fixed climatological value to the shared engineered output.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
ENGINEERED_DIR = Path("data/engineered")
QUALITY_REPORT_PATH = ENGINEERED_DIR / "data_quality_report.csv"

MAX_GAP_DAYS = 3  # Phase 0 §5.1 / configs/data.yaml missing_value_policy

RAW_VARIABLES = [
    "T2M_MAX", "T2M_MIN", "RH2M", "WS10M", "WD10M",
    "PRECTOTCORR", "PS", "ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN",
    "CLOUD_AMT", "GWETROOT", "GWETTOP",
]


# ---------------------------------------------------------------------------
# Gap detection & interpolation (Phase 0 §5.1)
# ---------------------------------------------------------------------------

def find_nan_runs(series: pd.Series) -> list[tuple[int, int]]:
    """Return list of (start_idx, end_idx) INCLUSIVE positional index pairs
    for each contiguous run of NaN in the series."""
    isna = series.isna().to_numpy()
    runs = []
    start = None
    for i, is_missing in enumerate(isna):
        if is_missing and start is None:
            start = i
        elif not is_missing and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(isna) - 1))
    return runs


def interpolate_and_flag(
    df: pd.DataFrame, col: str, max_gap_days: int = MAX_GAP_DAYS
) -> tuple[pd.Series, int, list[tuple[str, str, int]]]:
    """
    Apply the ≤N-day linear interpolation / >N-day exclusion policy to one
    column, in chronological order (df must already be date-sorted).

    Returns:
        filled_series, interpolated_day_count, excluded_ranges
        (excluded_ranges is a list of (start_date, end_date, gap_length))

    Gaps touching the start or end of the series (no valid value on one
    side) cannot be interpolated regardless of length, and are excluded.
    """
    series = df[col].copy()
    dates = df["date"].to_numpy()
    runs = find_nan_runs(series)

    interpolated_count = 0
    excluded_ranges = []

    for (s, e) in runs:
        length = e - s + 1

        if s == 0 or e == len(series) - 1:
            # Touches series boundary — no valid neighbor on one side.
            excluded_ranges.append((str(dates[s]), str(dates[e]), length))
            continue

        if length <= max_gap_days:
            before_val = series.iloc[s - 1]
            after_val = series.iloc[e + 1]
            for step, idx in enumerate(range(s, e + 1), start=1):
                frac = step / (length + 1)
                series.iloc[idx] = before_val + frac * (after_val - before_val)
            interpolated_count += length
        else:
            excluded_ranges.append((str(dates[s]), str(dates[e]), length))

    return series, interpolated_count, excluded_ranges


# ---------------------------------------------------------------------------
# Derived features — see docstrings for evidentiary status of each formula
# ---------------------------------------------------------------------------

def compute_soil_dryness(gwetroot: pd.Series, gwettop: pd.Series) -> pd.Series:
    """
    ENGINEERING ASSUMPTION, NOT AN ESTABLISHED METEOROLOGICAL FORMULA.

    Equal-weighted average of root-zone and surface soil wetness,
    inverted so higher = drier:

        SOIL_DRYNESS = 1 - (GWETROOT + GWETTOP) / 2

    GWETROOT relates to multi-day heat persistence (water available for
    evapotranspirative cooling over the forecast horizon); GWETTOP
    relates to immediate day-to-day surface energy balance. No calibrated
    or literature-sourced weighting exists for this combination — equal
    weighting is a documented placeholder, not a derived result. Revisit
    with justification if VIF / feature-importance (Phase A-2) shows this
    feature matters; don't treat 0.5/0.5 as validated just because it's simple.
    """
    return 1 - (gwetroot + gwettop) / 2


def compute_atm_dryness(rh2m: pd.Series) -> pd.Series:
    """
    Linear transform of relative humidity — a standard, directly measured
    atmospheric dryness proxy, not an engineering combination guess.

        ATM_DRYNESS = 1 - (RH2M / 100)

    Full vapor pressure deficit (VPD) would be more meteorologically
    complete but requires a temperature term; using T2M_MAX (even lagged)
    here risks subtle target-leakage patterns that are easy to introduce
    and hard to audit. Deliberately kept to RH-only for this reason.
    """
    return 1 - (rh2m / 100)


def compute_radiation_effective(allsky: pd.Series, clrsky: pd.Series) -> pd.Series:
    """
    Clearness index (kt) — an established solar-radiation-literature
    concept, not an ad hoc combination invented for this project.

        RADIATION_EFFECTIVE = ALLSKY_SFC_SW_DWN / CLRSKY_SFC_SW_DWN

    Can exceed 1.0 due to a real, documented physical effect (cloud-edge
    reflection briefly boosting measured radiation above the modeled
    clear-sky value) — clipped to [0, 1.2] rather than treated as an
    error or left unbounded.
    """
    ratio = allsky / clrsky
    return ratio.clip(lower=0, upper=1.2)


# ---------------------------------------------------------------------------
# Circular encodings
# ---------------------------------------------------------------------------

def add_circular_encodings(df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(df["date"])
    doy = dates.dt.dayofyear
    days_in_year = dates.dt.is_leap_year.map({True: 366, False: 365})

    df["DOY_SIN"] = np.sin(2 * np.pi * doy / days_in_year)
    df["DOY_COS"] = np.cos(2 * np.pi * doy / days_in_year)

    wd_rad = np.deg2rad(df["WD10M"])
    df["WD10M_SIN"] = np.sin(wd_rad)
    df["WD10M_COS"] = np.cos(wd_rad)

    return df


# ---------------------------------------------------------------------------
# Per-district processing
# ---------------------------------------------------------------------------

def process_district(raw_path: Path) -> tuple[pd.DataFrame, list[dict]]:
    district_id = raw_path.stem
    df = pd.read_parquet(raw_path)
    df = df.sort_values("date").reset_index(drop=True)

    quality_rows = []
    excluded_dates_all_vars: set[str] = set()

    # --- Step 1: interpolate / exclude, per raw variable, independently ---
    for col in RAW_VARIABLES:
        filled, n_interp, excluded_ranges = interpolate_and_flag(df, col)
        df[col] = filled

        quality_rows.append({
            "district_id": district_id,
            "variable": col,
            "interpolated_day_count": n_interp,
            "excluded_gap_count": len(excluded_ranges),
            "excluded_ranges": "; ".join(
                f"{s}..{e} ({l}d)" for s, e, l in excluded_ranges
            ) if excluded_ranges else "",
        })

        for (start_date, end_date, _length) in excluded_ranges:
            mask = (df["date"] >= start_date) & (df["date"] <= end_date)
            excluded_dates_all_vars.update(df.loc[mask, "date"].tolist())

    # --- Step 2: has_excluded_gap flag — conservative, row-level ---
    df["has_excluded_gap"] = df["date"].isin(excluded_dates_all_vars)

    # --- Step 3: autoregressive lags (on gap-handled T2M_MAX) ---
    df["T2M_MAX_lag1"] = df["T2M_MAX"].shift(1)
    df["T2M_MAX_lag3"] = df["T2M_MAX"].shift(3)
    df["T2M_MAX_lag7"] = df["T2M_MAX"].shift(7)
    df["T2M_MAX_roll7_mean"] = df["T2M_MAX"].rolling(7).mean()
    df["T2M_MAX_roll7_std"] = df["T2M_MAX"].rolling(7).std()

    # --- Step 4: land-surface / atmospheric derived features (1-day lagged) ---
    soil_dryness = compute_soil_dryness(df["GWETROOT"], df["GWETTOP"])
    atm_dryness = compute_atm_dryness(df["RH2M"])
    radiation_effective = compute_radiation_effective(
        df["ALLSKY_SFC_SW_DWN"], df["CLRSKY_SFC_SW_DWN"]
    )

    df["SOIL_DRYNESS_lag1"] = soil_dryness.shift(1)
    df["ATM_DRYNESS_lag1"] = atm_dryness.shift(1)
    df["RADIATION_EFFECTIVE_lag1"] = radiation_effective.shift(1)
    df["CLOUD_AMT_lag1"] = df["CLOUD_AMT"].shift(1)
    df["WS10M_lag1"] = df["WS10M"].shift(1)
    df["PRECTOTCORR_lag1"] = df["PRECTOTCORR"].shift(1)
    df["PS_lag1"] = df["PS"].shift(1)

    # --- Step 5: circular encodings ---
    df = add_circular_encodings(df)

    # --- Step 6: final schema, per Phase 0 §6 ---
    output_cols = [
        "district_id", "date", "lat", "lon",
        "T2M_MAX",
        "T2M_MAX_lag1", "T2M_MAX_lag3", "T2M_MAX_lag7",
        "T2M_MAX_roll7_mean", "T2M_MAX_roll7_std",
        "SOIL_DRYNESS_lag1", "ATM_DRYNESS_lag1", "RADIATION_EFFECTIVE_lag1",
        "CLOUD_AMT_lag1", "WS10M_lag1", "PRECTOTCORR_lag1", "PS_lag1",
        "DOY_SIN", "DOY_COS", "WD10M_SIN", "WD10M_COS",
        "has_excluded_gap",
    ]
    # No climatological normal column — deliberately absent (§5.2 item 4).

    return df[output_cols], quality_rows


def run() -> None:
    ENGINEERED_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(RAW_DIR.glob("*.parquet"))
    print(f"Processing {len(raw_files)} districts...")

    all_quality_rows = []

    for i, raw_path in enumerate(raw_files, 1):
        district_id = raw_path.stem
        print(f"[{i}/{len(raw_files)}] {district_id}")

        engineered_df, quality_rows = process_district(raw_path)
        all_quality_rows.extend(quality_rows)

        out_path = ENGINEERED_DIR / f"{district_id}.parquet"
        engineered_df.to_parquet(out_path, index=False)

    quality_df = pd.DataFrame(all_quality_rows)
    quality_df.to_csv(QUALITY_REPORT_PATH, index=False)

    print(f"\nDone. Engineered files written to {ENGINEERED_DIR}/")
    print(f"Data quality report written to {QUALITY_REPORT_PATH}")

    total_interp = quality_df["interpolated_day_count"].sum()
    total_excluded = quality_df["excluded_gap_count"].sum()
    print(f"\nTotal interpolated days (all districts, all variables): {total_interp}")
    print(f"Total excluded gaps (all districts, all variables): {total_excluded}")

    rows_with_flag = sum(
        pd.read_parquet(f)["has_excluded_gap"].sum()
        for f in ENGINEERED_DIR.glob("*.parquet")
    )
    print(f"Total rows flagged has_excluded_gap=True (across all districts): {rows_with_flag}")


if __name__ == "__main__":
    run()