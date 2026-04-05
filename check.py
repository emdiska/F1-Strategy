import json
import pandas as pd

with open('data/model.json') as f:
    m = json.load(f)

rows = []
for circuit, compounds in m['degradation'].items():
    for compound, d in compounds.items():
        rows.append({
            'compound':  compound,
            'r_squared': d['r_squared'],
            'n_points':  d['n_points'],
        })

df = pd.DataFrame(rows)
print('R² distribution by compound:')
print(df.groupby('compound')['r_squared'].describe().round(3))

print('\nProportion R² > 0.3 by compound:')
for c in ['SOFT', 'MEDIUM', 'HARD']:
    sub = df[df['compound'] == c]
    pct = (sub['r_squared'] > 0.3).mean() * 100
    print(f'  {c}: {pct:.1f}%')