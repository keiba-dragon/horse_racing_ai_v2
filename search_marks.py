# coding: utf-8
"""
search_marks.py - 印条件のグリッドサーチ

acc_rank / roi_rank / EV閾値 の組み合わせ全パターンを評価し、
OOS ROI が良い条件を探す。
"""
import sys, os, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from save_conditional_logit import BASE_DIR, DATA_FILE
from save_v3 import add_computed_features

# ─── モデル読み込み ───────────────────────────────────────────────────────────
acc_model = pickle.load(open(os.path.join(BASE_DIR, 'models', 'accuracy_model.pkl'), 'rb'))
roi_raw   = pickle.load(open(os.path.join(BASE_DIR, 'models', 'final_model.pkl'), 'rb'))
roi_arts  = roi_raw.get('artifacts', {})
ROI_KEY_MAP = {'ダ長': 'ダ', 'ダ短': 'ダ短', '芝短': '芝短', '芝中': '芝中', '芝長': '芝長'}
roi_model = {sk: roi_arts[rk] for sk, rk in ROI_KEY_MAP.items() if rk in roi_arts}

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
df = df[df['クラス_rank'] != 1.0].copy()
df['seg_key'] = [get_seg_key(s, d) for s, d in zip(df['_surf'], df['_dist_m'])]
df['dist_m'] = df['_dist_m']
df = df[df['seg_key'].notna()].copy()
df = add_computed_features(df)
baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
for col in df.columns:
    if '馬場状態' in col and col != '馬場状態':
        df[col] = df[col].map(baba_map)
df['単勝オッズ'] = pd.to_numeric(df['単勝オッズ'], errors='coerce')
print(f'全データ: {len(df):,}行 / {df["race_id"].nunique():,}R')

# ─── スコア計算 ───────────────────────────────────────────────────────────────
def score_df(df_in, model_dict):
    results = []
    for seg, grp in df_in.groupby('seg_key'):
        if seg not in model_dict: continue
        art = model_dict[seg]
        feat_cols, scaler, coef = art['feat_cols'], art['scaler'], art['coef']
        rows = []
        for _, row in grp.iterrows():
            fv = []
            for f in feat_cols:
                if f.endswith('_isnan'):
                    fv.append(1.0 if pd.isna(row.get(f[:-6])) else 0.0)
                else:
                    v = row.get(f, np.nan)
                    try: fv.append(float(v) if not pd.isna(v) else 0.0)
                    except: fv.append(0.0)
            rows.append(fv)
        X = np.array(rows, dtype=float)
        try:
            scores = scaler.transform(X) @ coef
        except:
            scores = np.zeros(len(grp))
        grp = grp.copy()
        grp['_score'] = scores
        results.append(grp)
    return pd.concat(results) if results else df_in.assign(_score=np.nan)

print('スコア計算中...')
df_acc = score_df(df, acc_model)
df_acc['_acc_rank'] = df_acc.groupby('race_id')['_score'].rank(ascending=False, method='first')
df_roi_df = score_df(df, roi_model)
df_roi_df['_roi_rank'] = df_roi_df.groupby('race_id')['_score'].rank(ascending=False, method='first')

# isotonic calibration で acc_prob を計算（EV用）
df_acc['_acc_prob'] = np.nan
for seg, grp in df_acc.groupby('seg_key'):
    if seg not in acc_model: continue
    art = acc_model[seg]
    iso = art.get('isotonic')
    if iso is None: continue
    e = np.exp(grp['_score'] - grp['_score'].max())
    p_raw = (e / e.sum()).values
    p_cal = np.clip(iso.predict(p_raw), 0.001, 0.999)
    df_acc.loc[grp.index, '_acc_prob'] = p_cal

# マージ
df_m = df_acc[['race_id', '着順_num', '単勝オッズ', '_acc_rank', '_acc_prob', 'seg_key', '日付_num']].copy()
df_m['_roi_rank'] = df_roi_df['_roi_rank']
df_m['_ev'] = df_m['_acc_prob'] * df_m['単勝オッズ']
print('スコア計算完了\n')

# ─── ROI計算 ─────────────────────────────────────────────────────────────────
OOS = {
    '2324': (230101, 250101),
    '25':   (250101, 260101),
    '26':   (260101, 300101),
}

def calc_roi_filter(mask_series, df_sub):
    bets = df_sub[mask_series[df_sub.index]].copy()
    # 同一レースで複数ヒットする場合 acc_rank 最小を選ぶ
    bets = bets.sort_values('_acc_rank').groupby('race_id').first().reset_index()
    nr = len(bets)
    if nr < 10: return float('nan'), nr, float('nan')
    won = bets['着順_num'] == 1
    payout = (bets['単勝オッズ'][won] * 100).sum()
    roi = payout / (nr * 100) - 1
    wr = won.mean()
    return roi, nr, wr

# ─── グリッドサーチ ───────────────────────────────────────────────────────────
print('グリッドサーチ開始...')
results = []

# パターン1: acc_rank <= A かつ roi_rank <= R
for a in range(1, 6):
    for r in range(1, 6):
        # 各レースで「acc_rank<=a かつ roi_rank<=r」の中の acc_rank=最小馬だけ買う
        mask = (df_m['_acc_rank'] <= a) & (df_m['_roi_rank'] <= r)
        rois = {}
        nrs = {}
        for period, (d0, d1) in OOS.items():
            sub = df_m[(df_m['日付_num'] >= d0) & (df_m['日付_num'] < d1)]
            roi, nr, wr = calc_roi_filter(mask, sub)
            rois[period] = roi
            nrs[period] = nr
        n25, n26 = nrs['25'], nrs['26']
        r25, r26 = rois['25'], rois['26']
        r2526 = (r25*n25 + r26*n26)/(n25+n26) if (n25+n26) > 0 else float('nan')
        r2324 = rois['2324']
        nr_total = sum(nrs.values())
        r_all = sum((rois[p]*nrs[p] for p in OOS if not np.isnan(rois[p])), 0) / nr_total if nr_total>0 else float('nan')
        results.append({
            'label': f'acc≤{a} & roi≤{r}',
            'type': 'grid',
            'acc_max': a, 'roi_max': r,
            'r2324': r2324, 'r25': r25, 'r26': r26,
            'r2526': r2526, 'r_all': r_all,
            'nr': nr_total, 'nr25': n25, 'nr26': n26,
        })

# パターン2: EV閾値付き (acc=1 のみ)
for ev_min in [0.8, 1.0, 1.1, 1.2, 1.3, 1.5]:
    for r in range(1, 4):
        mask = (df_m['_acc_rank'] == 1) & (df_m['_roi_rank'] <= r) & (df_m['_ev'] >= ev_min)
        rois = {}; nrs = {}
        for period, (d0, d1) in OOS.items():
            sub = df_m[(df_m['日付_num'] >= d0) & (df_m['日付_num'] < d1)]
            roi, nr, wr = calc_roi_filter(mask, sub)
            rois[period] = roi; nrs[period] = nr
        n25, n26 = nrs['25'], nrs['26']
        r25, r26 = rois['25'], rois['26']
        r2526 = (r25*n25+r26*n26)/(n25+n26) if (n25+n26)>0 else float('nan')
        nr_total = sum(nrs.values())
        r_all = sum((rois[p]*nrs[p] for p in OOS if not np.isnan(rois[p])), 0)/nr_total if nr_total>0 else float('nan')
        results.append({
            'label': f'acc=1 & roi≤{r} & EV≥{ev_min}',
            'type': 'ev',
            'acc_max': 1, 'roi_max': r, 'ev_min': ev_min,
            'r2324': rois['2324'], 'r25': r25, 'r26': r26,
            'r2526': r2526, 'r_all': r_all,
            'nr': nr_total, 'nr25': n25, 'nr26': n26,
        })

# パターン3: オッズ範囲付き
for odds_lo, odds_hi in [(1.5, 10), (2.0, 10), (2.0, 15), (1.5, 20), (3.0, 15), (3.0, 30)]:
    for a in [1, 2]:
        for r in [1, 2]:
            mask = ((df_m['_acc_rank'] <= a) & (df_m['_roi_rank'] <= r) &
                    (df_m['単勝オッズ'] >= odds_lo) & (df_m['単勝オッズ'] <= odds_hi))
            rois = {}; nrs = {}
            for period, (d0, d1) in OOS.items():
                sub = df_m[(df_m['日付_num'] >= d0) & (df_m['日付_num'] < d1)]
                roi, nr, wr = calc_roi_filter(mask, sub)
                rois[period] = roi; nrs[period] = nr
            n25, n26 = nrs['25'], nrs['26']
            r25, r26 = rois['25'], rois['26']
            r2526 = (r25*n25+r26*n26)/(n25+n26) if (n25+n26)>0 else float('nan')
            nr_total = sum(nrs.values())
            r_all = sum((rois[p]*nrs[p] for p in OOS if not np.isnan(rois[p])), 0)/nr_total if nr_total>0 else float('nan')
            results.append({
                'label': f'acc≤{a} & roi≤{r} & odds{odds_lo}-{odds_hi}',
                'type': 'odds',
                'acc_max': a, 'roi_max': r, 'odds_lo': odds_lo, 'odds_hi': odds_hi,
                'r2324': rois['2324'], 'r25': r25, 'r26': r26,
                'r2526': r2526, 'r_all': r_all,
                'nr': nr_total, 'nr25': n25, 'nr26': n26,
            })

# ─── 結果表示 ─────────────────────────────────────────────────────────────────
df_res = pd.DataFrame(results)
df_res = df_res[df_res['nr'] >= 100].copy()

def fmt(v):
    return f'{v*100:+6.1f}%' if not np.isnan(v) else '   nan'

# 25+26 ROI でソート
print('\n=== 全パターン TOP30（25+26 ROI順）===')
print(f'{"条件":38s}  {"2023-24":>8} {"2025":>8} {"2026":>8} {"25+26":>8} {"全OOS":>8} {"R(25+26)":>9}')
print('-' * 95)
for _, row in df_res.nlargest(30, 'r2526').iterrows():
    n2526 = int(row['nr25'] + row['nr26'])
    print(f'{row["label"]:38s}  {fmt(row["r2324"])} {fmt(row["r25"])} {fmt(row["r26"])} '
          f'{fmt(row["r2526"])} {fmt(row["r_all"])} {n2526:>8}R')

# 全OOS ROI でソート
print('\n=== 全パターン TOP20（全OOS ROI順）===')
print(f'{"条件":38s}  {"2023-24":>8} {"2025":>8} {"2026":>8} {"25+26":>8} {"全OOS":>8} {"R数(全)":>8}')
print('-' * 95)
for _, row in df_res.nlargest(20, 'r_all').iterrows():
    print(f'{row["label"]:38s}  {fmt(row["r2324"])} {fmt(row["r25"])} {fmt(row["r26"])} '
          f'{fmt(row["r2526"])} {fmt(row["r_all"])} {int(row["nr"]):>8}R')

# 安定性（2324/25/26が全部プラス）
print('\n=== 全年プラス条件 ===')
plus_all = df_res[(df_res['r2324'] > 0) & (df_res['r25'] > 0) & (df_res['r26'] > 0)]
if len(plus_all) > 0:
    print(f'{"条件":38s}  {"2023-24":>8} {"2025":>8} {"2026":>8} {"25+26":>8} {"全OOS":>8} {"R数(全)":>8}')
    print('-' * 95)
    for _, row in plus_all.nlargest(20, 'r2526').iterrows():
        print(f'{row["label"]:38s}  {fmt(row["r2324"])} {fmt(row["r25"])} {fmt(row["r26"])} '
              f'{fmt(row["r2526"])} {fmt(row["r_all"])} {int(row["nr"]):>8}R')
else:
    print('該当なし')

print('\n完了')
