# coding: utf-8
import os, io, pandas as pd

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(base, '出馬表形式5月24日.csv'), 'rb') as f:
    raw = f.read()
try:
    text = raw.decode('cp932')
except:
    text = raw.decode('utf-8', errors='replace')

df = pd.read_csv(io.StringIO(text))

# col[9] = クラス の値を確認
col9 = df.iloc[:, 9]
lines = []
lines.append(f'col[9] unique: {col9.dropna().unique().tolist()[:10]}')
lines.append('')

# col[5] = 頭 (maybe レース名?)
lines.append(f'col[7] unique: {df.iloc[:,7].dropna().unique().tolist()[:10]}')
lines.append(f'col[8] unique: {df.iloc[:,8].dropna().unique().tolist()[:10]}')

# 馬名列を探す: col 26-35 あたり?
lines.append('')
lines.append('col 24-36 samples:')
for i in range(24, 37):
    sample = df.iloc[:, i].dropna().astype(str).unique()[:3]
    lines.append(f'  col[{i}]: {sample.tolist()}')

with open(os.path.join(base, '_tmp_csv_check.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('Written')
