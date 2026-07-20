import pandas as pd
from pathlib import Path

files = sorted(Path('data/engineered').glob('*.parquet'))
records = []

for f in files:
    df = pd.read_parquet(f)
    records.append({
        'district_id': f.stem,
        'mean_t2m_max': df['T2M_MAX'].mean(),
    })

summary = pd.DataFrame(records).sort_values('mean_t2m_max')
print('Coolest 10 districts (annual mean T2M_MAX):')
print(summary.head(10).to_string(index=False))
print()
print('Hottest 10 districts (annual mean T2M_MAX):')
print(summary.tail(10).to_string(index=False))
print()
target = summary[summary['district_id'].isin(['banda', 'muzaffarnagar'])]
print('Selected districts:')
print(target.to_string(index=False))
