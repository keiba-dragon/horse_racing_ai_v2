# coding: utf-8
"""レースタイプ別ROI分析: rank=1 / EV>0.10 でどのレース属性が得意か"""
import sys, io, os, pickle
import numpy as np
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import add_new_features, segment_softmax, prepare
from save_lambdarank_pace import add_pace_features

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

print('データ読み込み中...')
df = pd.read_parquet(DATA_FILE)
df['日付_num']   = pd.to_numeric(df['日付'], errors='coerce')
df['着順_num']   = pd.to_numeric(df['着順_num'], errors='coerce')
df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
df = df.dropna(subset=['日付_num', '着順_num'])
df = df[df['着順_num'] < 99]
df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                 df['開催'].astype(str).str.strip() + '_' +
                 df['Ｒ'].astype(str).str.strip())
df = add_pace_features(df)
df = add_new_features(df)
df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
df = df[df['surface'].isin(['芝', 'ダ'])].copy()
df['is_maiden'] = (df['クラス_rank'] == 2)

# 距離数値
df['dist_num'] = df['距離'].astype(str).str.extract(r'(\d+)')[0].astype(float)

# 頭数
df['n_horses'] = df.groupby('race_id')['race_id'].transform('count')

# 月
df['month'] = (df['日付_num'] % 10000 // 100).astype(int)

with open(os.path.join(MODEL_DIR, 'roi_model.pkl'), 'rb') as f:
    pkg = pickle.load(f)
artifacts     = pkg['artifacts']
FACTOR_MAIDEN = pkg['factor_maiden']
FACTOR_OTHER  = pkg['factor_other']

def score_df(target: pd.DataFrame) -> pd.DataFrame:
    t = target.sort_values('race_id').reset_index(drop=True)
    calib_arr = np.zeros(len(t))
    odds_arr  = pd.to_numeric(t['単勝オッズ'], errors='coerce').values
    mprob     = 1.0 / np.clip(odds_arr, 1.0, None)
    for surf in ['芝', 'ダ']:
        art  = artifacts[surf]
        mask = (t['surface'] == surf).values
        sub  = t[mask].sort_values('race_id').reset_index(drop=True)
        if len(sub) == 0: continue
        X, _, gs, n, *_ = prepare(
            sub, art['feat_cols'],
            scaler=art['scaler'], poly2=art['poly2'],
            inter_scaler2=art['inter_scaler2'], top_idx=art['top_idx'],
            poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
        raw   = segment_softmax(X @ art['coef'], gs, n)
        calib = art['isotonic'].predict(raw)
        calib_arr[t[mask].index] = calib
    t['calib']    = calib_arr
    t['odds_num'] = odds_arr
    t['mprob']    = mprob
    factor        = np.where(t['is_maiden'], FACTOR_MAIDEN, FACTOR_OTHER)
    t['score']    = t['calib'] - factor * t['mprob']
    t['ev']       = t['calib'] - t['mprob'] * 0.80
    t['rank']     = t.groupby('race_id')['score'].rank(ascending=False, method='first')
    return t

print('OOS スコア計算中...')
oos = score_df(df[df['日付_num'] >= 230101])

def roi_stats(sub):
    if len(sub) < 20:
        return None
    won = sub['着順_num'] == 1
    r   = (sub.loc[won, 'odds_num'] * 100).sum() / (len(sub) * 100) - 1
    return len(sub), won.mean(), r

W = 72

def print_breakdown(title, groups, label_fmt, min_n=30):
    print(f'\n{"="*W}')
    print(title)
    print(f'  {"カテゴリ":<16} {"件数(全)":>8} {"ROI(全)":>9}  {"件数(EV>0.10)":>13} {"ROI(EV>0.10)":>12}')
    print(f'  {"-"*W}')
    for label, mask in groups:
        s_all = oos[(oos['rank'] == 1) & mask]
        s_ev  = oos[(oos['rank'] == 1) & mask & (oos['ev'] > 0.10)]
        r_all = roi_stats(s_all)
        r_ev  = roi_stats(s_ev)
        n_all = f'{r_all[0]:>6}' if r_all else '    --'
        roi_all = f'{r_all[2]:>+8.4f}{"★" if r_all and r_all[2]>0 else " "}' if r_all else '      -- '
        n_ev  = f'{r_ev[0]:>10}' if r_ev else '        --'
        roi_ev = f'{r_ev[2]:>+10.4f}{"★" if r_ev and r_ev[2]>0 else " "}' if r_ev else '          -- '
        print(f'  {label:<16} {n_all}  {roi_all}  {n_ev}  {roi_ev}')

# ── 芝 vs ダ ─────────────────────────────────────────────────────────────
print_breakdown('【コース別 ROI】', [
    ('芝', oos['surface'] == '芝'),
    ('ダート', oos['surface'] == 'ダ'),
], '{}')

# ── 距離帯 ───────────────────────────────────────────────────────────────
dist_bins = [
    ('〜1200m', oos['dist_num'] <= 1200),
    ('1201〜1600m', (oos['dist_num'] > 1200) & (oos['dist_num'] <= 1600)),
    ('1601〜2000m', (oos['dist_num'] > 1600) & (oos['dist_num'] <= 2000)),
    ('2001〜2400m', (oos['dist_num'] > 2000) & (oos['dist_num'] <= 2400)),
    ('2401m〜', oos['dist_num'] > 2400),
]
print_breakdown('【距離帯別 ROI】', dist_bins, '{}')

# ── 頭数 ─────────────────────────────────────────────────────────────────
head_bins = [
    ('〜8頭', oos['n_horses'] <= 8),
    ('9〜12頭', (oos['n_horses'] >= 9) & (oos['n_horses'] <= 12)),
    ('13〜16頭', (oos['n_horses'] >= 13) & (oos['n_horses'] <= 16)),
    ('17〜18頭', oos['n_horses'] >= 17),
]
print_breakdown('【頭数別 ROI】', head_bins, '{}')

# ── クラス ───────────────────────────────────────────────────────────────
# クラス_rank: 1=新馬, 2=未勝利, 3=1勝C, 4=2勝C, 5=3勝C, 6=OP/L, 7=G3, 8=G2, 9=G1 等
class_labels = {1:'新馬', 2:'未勝利', 3:'1勝C', 4:'2勝C', 5:'3勝C', 6:'OP/L', 7:'G3以上'}
class_bins = []
for rk, name in class_labels.items():
    if rk == 7:
        class_bins.append((name, oos['クラス_rank'] >= 7))
    else:
        class_bins.append((name, oos['クラス_rank'] == rk))
print_breakdown('【クラス別 ROI】', class_bins, '{}')

# ── 月別 ────────────────────────────────────────────────────────────────
month_bins = [(f'{m}月', oos['month'] == m) for m in range(1, 13)]
print_breakdown('【月別 ROI】', month_bins, '{}')

# ── 競馬場 ──────────────────────────────────────────────────────────────
venue_list = sorted(oos['開催'].astype(str).str.strip().unique())
venue_bins = [(v, oos['開催'].astype(str).str.strip() == v) for v in venue_list]
print_breakdown('【競馬場別 ROI (件数20以上)】', venue_bins, '{}')

# ── 芝×距離 クロス ───────────────────────────────────────────────────────
print(f'\n{"="*W}')
print('【芝×距離 クロス (rank=1, EV>0.10)】')
print(f'  {"距離":>10}  {"件数":>6}  {"勝率":>6}  {"ROI":>9}')
print(f'  {"-"*45}')
for lo, hi, label in [(0,1200,'〜1200'), (1201,1600,'1201-1600'), (1601,2000,'1601-2000'), (2001,9999,'2001〜')]:
    sub = oos[(oos['rank']==1) & (oos['ev']>0.10) & (oos['surface']=='芝') &
              (oos['dist_num']>=lo) & (oos['dist_num']<=hi)]
    r = roi_stats(sub)
    if r:
        mk = '★' if r[2] > 0 else ''
        print(f'  芝{label+"m":<12} {r[0]:>6}  {r[1]:.3f}  {r[2]:>+8.4f}{mk}')

print(f'  {"-"*45}')
for lo, hi, label in [(0,1200,'〜1200'), (1201,1600,'1201-1600'), (1601,2000,'1601-2000'), (2001,9999,'2001〜')]:
    sub = oos[(oos['rank']==1) & (oos['ev']>0.10) & (oos['surface']=='ダ') &
              (oos['dist_num']>=lo) & (oos['dist_num']<=hi)]
    r = roi_stats(sub)
    if r:
        mk = '★' if r[2] > 0 else ''
        print(f'  ダ{label+"m":<12} {r[0]:>6}  {r[1]:.3f}  {r[2]:>+8.4f}{mk}')

# ── 人気 × モデル乖離（穴狙い vs 本命狙い）───────────────────────────────
print(f'\n{"="*W}')
print('【人気帯別 ROI (rank=1, EV>0.10)】')
print(f'  {"人気帯":>10}  {"件数":>6}  {"勝率":>6}  {"ROI":>9}')
print(f'  {"-"*45}')
oos['pop_rank'] = oos.groupby('race_id')['odds_num'].rank(method='first', ascending=True)
for lo, hi, label in [(1,1,'1番人気'), (2,3,'2-3番人気'), (4,6,'4-6番人気'), (7,99,'7番人気以下')]:
    sub = oos[(oos['rank']==1) & (oos['ev']>0.10) &
              (oos['pop_rank']>=lo) & (oos['pop_rank']<=hi)]
    r = roi_stats(sub)
    if r:
        mk = '★' if r[2] > 0 else ''
        print(f'  {label:<14} {r[0]:>6}  {r[1]:.3f}  {r[2]:>+8.4f}{mk}')
