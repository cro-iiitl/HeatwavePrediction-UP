import pandas as pd
from pathlib import Path

files = sorted(Path('data/engineered').glob('*.parquet'))
records = []

for f in files:
    df = pd.read_parquet(f)
    df['date_parsed'] = pd.to_datetime(df['date'])
    may_data = df[df['date_parsed'].dt.month == 5]

    records.append({
        'district_id': f.stem,
        'may_t2m_max_mean': may_data['T2M_MAX'].mean(),
        'may_t2m_max_max': may_data['T2M_MAX'].max(),
        'may_t2m_max_min': may_data['T2M_MAX'].min(),
        'overall_t2m_max_max': df['T2M_MAX'].max(),
        'overall_t2m_max_min': df['T2M_MAX'].min(),
    })

summary = pd.DataFrame(records).sort_values('may_t2m_max_mean')

print('May T2M_MAX summary across all 75 districts:')
print(summary[['district_id', 'may_t2m_max_mean', 'may_t2m_max_max']].to_string(index=False))

overall_max = summary['overall_t2m_max_max'].max()
overall_min = summary['overall_t2m_max_min'].min()

print()
print('Sanity thresholds for UP May pre-monsoon heat:')
print('  Overall dataset max T2M_MAX:', round(overall_max, 2))
print('  Overall dataset min T2M_MAX:', round(overall_min, 2))
print()

flags = summary[(summary['may_t2m_max_mean'] < 30) | (summary['may_t2m_max_mean'] > 48)]
if not flags.empty:
    print('FLAGGED - May mean T2M_MAX outside plausible [30, 48] C band:')
    print(flags[['district_id', 'may_t2m_max_mean']])
else:
    print('No districts flagged - all May T2M_MAX means fall within plausible range.')

extreme = summary[(summary['overall_t2m_max_max'] > 50) | (summary['overall_t2m_max_min'] < -5)]
if not extreme.empty:
    print()
    print('FLAGGED - extreme all-time values outside [-5, 50] C:')
    print(extreme[['district_id', 'overall_t2m_max_max', 'overall_t2m_max_min']])
else:
    print('No all-time extreme value flags.')
