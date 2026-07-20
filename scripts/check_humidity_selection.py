import pandas as pd
from pathlib import Path

files = sorted(Path('data/raw').glob('*.parquet'))
records = []

for f in files:
    df = pd.read_parquet(f)
    records.append({
        'district_id': f.stem,
        'mean_rh2m': df['RH2M'].mean(),
    })

summary = pd.DataFrame(records).sort_values('mean_rh2m')
print('Driest 10 districts (annual mean RH2M):')
print(summary.head(10).to_string(index=False))
print()
print('Most humid 10 districts (annual mean RH2M):')
print(summary.tail(10).to_string(index=False))
print()
target = summary[summary['district_id'].isin(['banda', 'muzaffarnagar'])]
print('Selected districts:')
print(target.to_string(index=False))
