"""
One-off smoke test for src/data/windowing.py — verifies window shapes,
date alignment, and that the two feature sets produce the expected
column counts, before trusting this against the real ablation.
"""

import sys
sys.path.insert(0, ".")

import pandas as pd
from src.data.windowing import (
    build_windows,
    TREE_FEATURE_COLUMNS,
    SEQUENCE_FEATURE_COLUMNS,
)

df = pd.read_parquet("data/engineered/banda.parquet")
print(f"Banda engineered rows: {len(df)}")

WINDOW_LENGTH = 15

# --- Sequence feature set (what Phase A-1's LSTM ablation will use) ---
ws = build_windows(df, WINDOW_LENGTH, SEQUENCE_FEATURE_COLUMNS)

print(f"\nSequence feature set ({len(SEQUENCE_FEATURE_COLUMNS)} features):")
print(f"  X shape: {ws.X.shape}")   # expect (n_windows, 15, 11)
print(f"  y shape: {ws.y.shape}")   # expect (n_windows, 10)
print(f"  n_windows: {len(ws.input_start_dates)}")

# Sanity: expected window count ~= n_rows - window_length - horizon + 1
# (minus dropped windows from warm-up NaNs / excluded gaps, both ~negligible here)
expected_max = len(df) - WINDOW_LENGTH - 10 + 1
print(f"  theoretical max windows (no drops): {expected_max}")
print(f"  actual windows: {len(ws.input_start_dates)} "
      f"(dropped: {expected_max - len(ws.input_start_dates)})")

# --- Date alignment check on the first window ---
print("\nFirst window:")
print(f"  input_start: {ws.input_start_dates[0]}")
print(f"  target_start: {ws.target_start_dates[0]}")
print(f"  target_end: {ws.target_end_dates[0]}")

# Confirm target_start is exactly window_length days after input_start
input_date = pd.to_datetime(ws.input_start_dates[0])
target_date = pd.to_datetime(ws.target_start_dates[0])
gap_days = (target_date - input_date).days
print(f"  gap between input_start and target_start: {gap_days} days "
      f"(expect {WINDOW_LENGTH})")

# Confirm target spans exactly 10 days
target_end_date = pd.to_datetime(ws.target_end_dates[0])
horizon_span = (target_end_date - target_date).days + 1
print(f"  target span length: {horizon_span} days (expect 10)")

# --- Tree feature set — should have more columns ---
ws_tree = build_windows(df, WINDOW_LENGTH, TREE_FEATURE_COLUMNS)
print(f"\nTree feature set ({len(TREE_FEATURE_COLUMNS)} features):")
print(f"  X shape: {ws_tree.X.shape}")   # expect (n_windows, 15, 17)

# --- No NaNs should exist anywhere in the output ---
import numpy as np
print(f"\nAny NaN in sequence X: {np.isnan(ws.X).any()}")
print(f"Any NaN in sequence y: {np.isnan(ws.y).any()}")
print(f"Any NaN in tree X: {np.isnan(ws_tree.X).any()}")

print("\nSmoke test complete.")