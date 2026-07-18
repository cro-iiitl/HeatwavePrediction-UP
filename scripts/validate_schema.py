import pandas as pd
from pathlib import Path

EXPECTED_COLUMNS = [
    'district_id', 'date', 'lat', 'lon',
    'T2M_MAX',
    'T2M_MAX_lag1', 'T2M_MAX_lag3', 'T2M_MAX_lag7',
    'T2M_MAX_roll7_mean', 'T2M_MAX_roll7_std',
    'SOIL_DRYNESS_lag1', 'ATM_DRYNESS_lag1', 'RADIATION_EFFECTIVE_lag1',
    'CLOUD_AMT_lag1', 'WS10M_lag1', 'PRECTOTCORR_lag1', 'PS_lag1',
    'DOY_SIN', 'DOY_COS', 'WD10M_SIN', 'WD10M_COS',
    'has_excluded_gap',
]
EXPECTED_ROWS = 15341

files = sorted(Path('data/engineered').glob('*.parquet'))
print(f'Checking {len(files)} files...')

failures = []

for f in files:
    df = pd.read_parquet(f)
    district_id = f.stem

    if list(df.columns) != EXPECTED_COLUMNS:
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        extra = set(df.columns) - set(EXPECTED_COLUMNS)
        failures.append(f'{district_id}: COLUMN MISMATCH missing={missing} extra={extra}')
        continue

    if len(df) != EXPECTED_ROWS:
        failures.append(f'{district_id}: ROW COUNT {len(df)} != {EXPECTED_ROWS}')

    if df['district_id'].nunique() != 1 or df['district_id'].iloc[0] != district_id:
        failures.append(f'{district_id}: district_id column mismatch with filename')

    # Unexpected NaNs beyond the known warm-up columns
    warmup_cols = {
        'T2M_MAX_lag1', 'T2M_MAX_lag3', 'T2M_MAX_lag7',
        'T2M_MAX_roll7_mean', 'T2M_MAX_roll7_std',
        'SOIL_DRYNESS_lag1', 'ATM_DRYNESS_lag1', 'RADIATION_EFFECTIVE_lag1',
        'CLOUD_AMT_lag1', 'WS10M_lag1', 'PRECTOTCORR_lag1', 'PS_lag1',
    }
    for col in df.columns:
        if col in warmup_cols:
            continue
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            failures.append(f'{district_id}: unexpected NaN in {col} ({n_nan} rows)')

    # dtype sanity — numeric feature columns should be float, not object
    numeric_cols = [c for c in EXPECTED_COLUMNS if c not in ('district_id', 'date', 'has_excluded_gap')]
    for col in numeric_cols:
        if df[col].dtype not in ('float64', 'float32'):
            failures.append(f'{district_id}: {col} has dtype {df[col].dtype}, expected float')

if failures:
    print(f'\n{len(failures)} FAILURES:')
    for f in failures:
        print(f'  {f}')
else:
    print('\nAll 75 files pass schema validation.')
