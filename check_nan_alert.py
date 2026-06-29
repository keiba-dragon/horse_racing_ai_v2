# coding: utf-8
"""
特徴量を重要度順に並べ、「異常NaN」（データ取得ミス系）を特定する。

正常NaN: 出走回数が足りないからNaN (例: 3走目の馬に「5走前」データなし)
異常NaN: 出走回数は十分なのにNaN (データパイプラインの問題)
"""
import pickle, sys, io, re
import pandas as pd, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── モデル読み込み ──
with open('models/final_model.pkl', 'rb') as f:
    pkg = pickle.load(f)
arts = pkg['artifacts']

# 芝モデルの係数から特徴量重要度を計算
# coef[0:321] = ベース特徴量の係数 (poly2拡張前)
art = arts['芝']
feat_cols = list(art['feat_cols'])
coef_base = art['coef'][:len(feat_cols)]  # 最初の321個がベース特徴量
importance = np.abs(coef_base)
feat_imp = pd.Series(importance, index=feat_cols).sort_values(ascending=False)

# ── 今日のキャッシュ ──
with open('data/raw/cache/20260530.cache.pkl', 'rb') as f:
    cache = pickle.load(f)
df = cache['result'].copy()
df.columns = df.columns.astype(object)

# ── parquet読み込み (出走回数取得用) ──
import pyarrow.parquet as pq
pq_df = pd.read_parquet('data/processed/all_venues_features.parquet',
                         columns=['馬名S', '日付'])
pq_df['日付_num'] = pd.to_numeric(pq_df['日付'], errors='coerce')
pq_df['uma'] = pq_df['馬名S'].astype(str).str.strip()
career_count = pq_df.groupby('uma').size().rename('career')
df['_uma'] = df['馬名S'].astype(str).str.strip()
df = df.join(career_count, on='_uma')
df['career'] = df['career'].fillna(0).astype(int)

# ── 特徴量ごとの「必要出走数」を定義 ──
def required_races(feat_name: str) -> int:
    """この特徴量がNaNになる正当な理由となる最小出走数を返す。
    required_races を超える出走があるのにNaNなら→異常"""
    # N走前系
    m = re.search(r'(\d+)走前', feat_name)
    if m:
        return int(m.group(1))
    # 近N走系
    m = re.search(r'近(\d+)走', feat_name)
    if m:
        return int(m.group(1))
    # 前走系
    if feat_name.startswith('前走') or '前走' in feat_name:
        return 1
    # 2走前系
    if '2走前' in feat_name:
        return 2
    # 3走前系
    if '3走前' in feat_name:
        return 3
    # 近走系
    if '近走' in feat_name:
        return 3
    # 初戦フラグ系: 0走でも正常
    if '初戦' in feat_name or 'デビュー' in feat_name:
        return 0
    # その他: 1走あれば埋まるべき
    return 1

# ── 各特徴量の異常NaN率を計算 ──
results = []
for feat in feat_cols:
    if feat not in df.columns:
        results.append({'feat': feat, 'imp': feat_imp.get(feat, 0),
                        'nan_all': 0, 'nan_normal': 0, 'nan_abnormal': 0,
                        'n_total': len(df), 'req': 0})
        continue

    req = required_races(feat)
    vals = pd.to_numeric(df[feat], errors='coerce')
    is_nan = vals.isna()

    # 正常NaN: 出走数 < required
    normal_nan  = (is_nan & (df['career'] < req)).sum()
    # 異常NaN: 出走数 >= required なのにNaN
    abnormal_nan = (is_nan & (df['career'] >= req)).sum()
    total_nan   = is_nan.sum()

    results.append({
        'feat': feat,
        'imp': feat_imp.get(feat, 0),
        'nan_all': int(total_nan),
        'nan_normal': int(normal_nan),
        'nan_abnormal': int(abnormal_nan),
        'n_total': len(df),
        'req': req,
    })

res = pd.DataFrame(results).sort_values('imp', ascending=False)

print(f'今日の出走馬数: {len(df)}頭')
print(f'特徴量数: {len(feat_cols)}')
print()

# ── 重要度上位50特徴量の異常NaN状況 ──
print('== 重要度上位50特徴量の異常NaN ==')
print(f'{"#":>3} {"特徴量名":<35} {"重要度":>6} {"必要走数":>5} {"全NaN":>6} {"正常NaN":>6} {"異常NaN":>6} {"異常率":>6}')
print('-' * 90)
for i, (_, r) in enumerate(res.head(50).iterrows()):
    abn_pct = r['nan_abnormal'] / r['n_total'] * 100
    alert = ' !! ALERT' if r['nan_abnormal'] > 0 else ''
    print(f'{i+1:>3} {r["feat"]:<35} {r["imp"]:>6.3f} {r["req"]:>5}走  '
          f'{r["nan_all"]:>6} {r["nan_normal"]:>6} {r["nan_abnormal"]:>6} '
          f'{abn_pct:>5.1f}%{alert}')

# ── 異常NaNが多い特徴量 (全体) ──
print()
print('== 異常NaN > 0 の特徴量 (重要度上位から) ==')
abnormal = res[res['nan_abnormal'] > 0].head(30)
for _, r in abnormal.iterrows():
    abn_pct = r['nan_abnormal'] / r['n_total'] * 100
    print(f'  {r["feat"]:<35} imp={r["imp"]:.3f}  異常NaN={r["nan_abnormal"]}頭 ({abn_pct:.1f}%)')

print()
print(f'異常NaN=0の特徴量数: {(res["nan_abnormal"] == 0).sum()} / {len(res)}')
print(f'異常NaN>0の特徴量数: {(res["nan_abnormal"] > 0).sum()} / {len(res)}')

# ── 馬別: 異常NaN数トップ20 ──
print()
print('== 馬別 異常NaN数 (上位20) ==')
horse_abn = []
for feat in feat_cols:
    if feat not in df.columns: continue
    req = required_races(feat)
    vals = pd.to_numeric(df[feat], errors='coerce')
    is_nan = vals.isna()
    abn_mask = is_nan & (df['career'] >= req)
    for idx in df[abn_mask].index:
        horse_abn.append(df.loc[idx, '馬名S'])

from collections import Counter
abn_cnt = Counter(horse_abn)
print(f'{"馬名":<20} {"異常NaN数":>8} {"career":>6}')
for horse, cnt in sorted(abn_cnt.items(), key=lambda x: -x[1])[:20]:
    c = int(df[df['馬名S'].str.strip() == horse.strip()]['career'].values[0]) if len(df[df['馬名S'].str.strip() == horse.strip()]) > 0 else '?'
    print(f'  {horse:<20} {cnt:>8}  {c:>6}走')
