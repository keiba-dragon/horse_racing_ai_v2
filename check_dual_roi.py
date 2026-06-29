# coding: utf-8
"""
check_dual_roi.py - accuracy_model x final_model の組み合わせ別OOS ROI分析

各フィルター条件でOOS単勝ROIを算出する:
  A: accuracy rank=1 全買い（現状）
  B: ROI rank=1 全買い
  C: 両方 rank=1（◎候補）
  D: acc rank=1 かつ ROI rank<=2
  E: acc rank=1 かつ ROI rank<=3
  F: ROI rank=1 かつ acc rank<=3
  G: acc rank<=2 かつ ROI rank<=2
"""
import sys, os, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE
from save_v3 import add_computed_features

# ─── モデル読み込み ───────────────────────────────────────────────────────────
acc_model = pickle.load(open(os.path.join(BASE_DIR, 'models', 'accuracy_model.pkl'), 'rb'))
roi_raw   = pickle.load(open(os.path.join(BASE_DIR, 'models', 'final_model.pkl'), 'rb'))
roi_arts  = roi_raw.get('artifacts', {})
ROI_KEY_MAP = {'ダ長': 'ダ', 'ダ短': 'ダ短', '芝短': '芝短', '芝中': '芝中', '芝長': '芝長'}
roi_model = {sk: roi_arts[rk] for sk, rk in ROI_KEY_MAP.items() if rk in roi_arts}

# ─── セグメント定義 ───────────────────────────────────────────────────────────
def get_seg_key(surf, dist_m):
    if pd.isna(dist_m): return None
    surf = str(surf).strip()
    if surf == '芝':
        if dist_m <= 1400:  return '芝短'
        elif dist_m <= 2000: return '芝中'
        else:               return '芝長'
    elif surf == 'ダ':
        return 'ダ短' if dist_m <= 1400 else 'ダ長'
    return None

# ─── データ読み込み ───────────────────────────────────────────────────────────
print('データ読み込み中...')
df = pd.read_parquet(DATA_FILE)
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
df = df.dropna(subset=['日付_num', '着順_num'])
df = df[df['着順_num'] < 99]
df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                 df['開催'].astype(str).str.strip() + '_' +
                 df['Ｒ'].astype(str).str.strip())
df = df[df['開催'].notna()].copy()
df['_surf']   = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
df['_dist_m'] = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
df = df[(df['クラス_rank'] != 1.0)].copy()
df['seg_key'] = [get_seg_key(s, d) for s, d in zip(df['_surf'], df['_dist_m'])]
df = df[df['seg_key'].notna()].copy()
df['dist_m'] = df['_dist_m']
df = add_computed_features(df)
baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
for col in df.columns:
    if '馬場状態' in col and col != '馬場状態':
        df[col] = df[col].map(baba_map)

OOS_PERIODS = {
    '2023-24': (230101, 250101),
    '2025':    (250101, 260101),
    '2026':    (260101, 300101),
}

print(f'全データ: {len(df):,}行 / {df["race_id"].nunique():,}R')

# ─── スコア計算 ───────────────────────────────────────────────────────────────
def score_df(df_in, model_dict):
    results = []
    for seg, grp in df_in.groupby('seg_key'):
        if seg not in model_dict:
            continue
        art = model_dict[seg]
        feat_cols = art['feat_cols']
        scaler = art['scaler']
        coef = art['coef']
        rows = []
        for _, row in grp.iterrows():
            fv = []
            for f in feat_cols:
                if f.endswith('_isnan'):
                    fv.append(1.0 if pd.isna(row.get(f[:-6])) else 0.0)
                else:
                    v = row.get(f, np.nan)
                    try:
                        fv.append(float(v) if not pd.isna(v) else 0.0)
                    except:
                        fv.append(0.0)
            rows.append(fv)
        X = np.array(rows, dtype=float)
        try:
            X_sc = scaler.transform(X)
            scores = X_sc @ coef
        except:
            scores = np.zeros(len(grp))
        grp = grp.copy()
        grp['_score'] = scores
        results.append(grp)
    if not results:
        return df_in.copy().assign(_score=np.nan)
    return pd.concat(results)

print('スコア計算中...')
df_acc = score_df(df, acc_model)
df_acc['_acc_rank'] = df_acc.groupby('race_id')['_score'].rank(ascending=False, method='first')

df_roi = score_df(df, roi_model)
df_roi['_roi_rank'] = df_roi.groupby('race_id')['_score'].rank(ascending=False, method='first')

# マージ
df_merged = df_acc[['race_id', '着順_num', '単勝オッズ', '_acc_rank', 'seg_key', '日付_num']].copy()
df_merged['_roi_rank'] = df_roi['_roi_rank']
df_merged['単勝オッズ'] = pd.to_numeric(df_merged['単勝オッズ'], errors='coerce')

# ─── ROI計算関数 ───────────────────────────────────────────────────────────────
def calc_roi(df_sub, acc_max, roi_max, label):
    mask = (df_sub['_acc_rank'] <= acc_max) & (df_sub['_roi_rank'] <= roi_max)
    bets = df_sub[mask].copy()
    # 各レースで条件を満たす馬が複数いる場合 acc_rank 最小を選ぶ
    bets = bets.sort_values('_acc_rank').groupby('race_id').first().reset_index()
    nr = len(bets)
    if nr == 0:
        return label, acc_max, roi_max, float('nan'), 0
    won = bets['着順_num'] == 1
    odds = bets['単勝オッズ']
    payout = (odds[won] * 100).sum()
    roi = payout / (nr * 100) - 1
    win_rate = won.mean()
    return label, win_rate, roi, nr

# ─── 結果表示 ─────────────────────────────────────────────────────────────────
FILTERS = [
    ('A: acc=1 のみ',           1, 99),
    ('B: ROI=1 のみ',           99, 1),
    ('C: 両方=1 (◎)',           1, 1),
    ('D: acc=1 & ROI≤2 (○候補)', 1, 2),
    ('E: acc=1 & ROI≤3',       1, 3),
    ('F: ROI=1 & acc≤2 (▲候補)', 2, 1),
    ('G: ROI=1 & acc≤3',       3, 1),
    ('H: acc≤2 & ROI≤2',       2, 2),
]

print()
print('=' * 90)
print(f'  {"フィルター":30s}  {"的中率":>7}  {"2023-24":>9}  {"2025":>9}  {"2026":>9}  {"25+26":>9}  {"全OOS":>9}  {"R数(OOS)":>8}')
print('=' * 90)

for label, acc_max, roi_max in FILTERS:
    row_parts = [f'  {label:30s}']
    rois = {}
    nr_total = 0
    wr_vals = []
    for period, (d_from, d_to) in OOS_PERIODS.items():
        sub = df_merged[(df_merged['日付_num'] >= d_from) & (df_merged['日付_num'] < d_to)]
        _, wr, roi, nr = calc_roi(sub, acc_max, roi_max, label)
        rois[period] = (roi, nr)
        nr_total += nr
        if not np.isnan(wr):
            wr_vals.append(wr)

    wr_avg = np.mean(wr_vals) if wr_vals else float('nan')
    r2324, n2324 = rois.get('2023-24', (float('nan'), 0))
    r25,   n25   = rois.get('2025',    (float('nan'), 0))
    r26,   n26   = rois.get('2026',    (float('nan'), 0))
    r2526 = (r25 * n25 + r26 * n26) / (n25 + n26) if (n25 + n26) > 0 else float('nan')
    r_all = (r2324 * n2324 + r25 * n25 + r26 * n26) / nr_total if nr_total > 0 else float('nan')

    def fmt(v):
        return f'{v*100:+7.1f}%' if not np.isnan(v) else '     nan'

    row_parts.append(f'  {wr_avg*100:6.1f}%')
    row_parts.append(f'  {fmt(r2324)}')
    row_parts.append(f'  {fmt(r25)}')
    row_parts.append(f'  {fmt(r26)}')
    row_parts.append(f'  {fmt(r2526)}')
    row_parts.append(f'  {fmt(r_all)}')
    row_parts.append(f'  {nr_total:6}R')
    print(''.join(row_parts))

print('=' * 90)
print()

# セグメント別詳細（◎候補のみ）
print('=== セグメント別 ROI [C: 両方rank=1] ===')
print(f'  {"セグメント":8}  {"2023-24":>9}  {"2025":>9}  {"2026":>9}  {"25+26":>9}  {"R数(OOS)":>8}')
for seg in ['ダ長', 'ダ短', '芝短', '芝中', '芝長']:
    sub_all = df_merged[df_merged['seg_key'] == seg]
    row_parts = [f'  {seg:8}']
    rois_s = {}
    nr_s = 0
    for period, (d_from, d_to) in OOS_PERIODS.items():
        sub = sub_all[(sub_all['日付_num'] >= d_from) & (sub_all['日付_num'] < d_to)]
        _, wr, roi, nr = calc_roi(sub, 1, 1, '')
        rois_s[period] = (roi, nr)
        nr_s += nr
    r2324, n2324 = rois_s.get('2023-24', (float('nan'), 0))
    r25,   n25   = rois_s.get('2025',    (float('nan'), 0))
    r26,   n26   = rois_s.get('2026',    (float('nan'), 0))
    r2526 = (r25*n25 + r26*n26)/(n25+n26) if (n25+n26) > 0 else float('nan')
    def fmt(v): return f'{v*100:+7.1f}%' if not np.isnan(v) else '     nan'
    row_parts += [f'  {fmt(r2324)}', f'  {fmt(r25)}', f'  {fmt(r26)}', f'  {fmt(r2526)}', f'  {nr_s:6}R']
    print(''.join(row_parts))
print()
