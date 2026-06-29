# coding: utf-8
import pickle, numpy as np, pandas as pd, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

df = pd.read_parquet('data/processed/all_venues_features.parquet')
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')

with open('models/conditional_logit.pkl', 'rb') as f:
    pkg = pickle.load(f)
feat_cols = pkg['feat_cols']

oos    = df[df['日付_num'] >= 230101]
recent = df[df['日付_num'] >= 260101]
print(f'OOS(2023+): {len(oos):,}行')
print(f'直近(2026+): {len(recent):,}行')

diffs = []
for fc in feat_cols:
    oos_nan = oos[fc].isna().mean() if fc in oos.columns else 1.0
    rec_nan = recent[fc].isna().mean() if fc in recent.columns else 1.0
    note = 'COL_MISSING' if fc not in df.columns else ''
    diffs.append((fc, oos_nan, rec_nan, rec_nan - oos_nan, note))

diffs_pos = [d for d in diffs if d[3] > 0.05]
diffs_pos.sort(key=lambda x: -x[3])

print()
print('直近2026でNaN率が増加した特徴量 (差>5%):')
print(f'{"特徴量":<35} {"OOS":>7} {"直近":>7} {"差":>8}')
print('-'*62)
for fc, oos_n, rec_n, diff, note in diffs_pos[:25]:
    print(f'{fc:<35} {oos_n:>7.1%} {rec_n:>7.1%} {diff:>+8.1%} {note}')

diffs_neg = [d for d in diffs if d[3] < -0.05]
diffs_neg.sort(key=lambda x: x[3])
if diffs_neg:
    print()
    print('直近2026でNaN率が減少した特徴量 (差<-5%):')
    print(f'{"特徴量":<35} {"OOS":>7} {"直近":>7} {"差":>8}')
    print('-'*62)
    for fc, oos_n, rec_n, diff, note in diffs_neg[:10]:
        print(f'{fc:<35} {oos_n:>7.1%} {rec_n:>7.1%} {diff:>+8.1%}')

# COL_MISSING（列自体がない）
missing_cols = [d for d in diffs if d[4] == 'COL_MISSING']
if missing_cols:
    print()
    print('parquetに列自体が存在しない特徴量:')
    for fc, *_ in missing_cols:
        print(f'  {fc}')
