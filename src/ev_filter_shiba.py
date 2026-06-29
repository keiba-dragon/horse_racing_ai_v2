# coding: utf-8
"""
芝セグメント向け 別アプローチ探索
- オッズ上限フィルタ（穴馬除外）
- 人気一致フィルタ（モデル1位 == 市場1番人気）
- 人気順フィルタ（モデル1位の市場人気 ≤ K）
- EV帯フィルタ（EV下限 + オッズ上限）
- 選択: 2325合算ROI / 検証: 2026
"""
import sys, os, pickle
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE
from save_v3 import add_computed_features

with open(os.path.join(BASE_DIR, 'models', 'hitrate_model.pkl'), 'rb') as f:
    MODEL = pickle.load(f)

print("データ読み込み中...", flush=True)
df = pd.read_parquet(DATA_FILE)
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
df = df.dropna(subset=['日付_num', '着順_num'])
df = df[df['着順_num'] < 99]
df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                 df['開催'].astype(str).str.strip() + '_' + df['Ｒ'].astype(str).str.strip())
df = df[df['開催'].notna()].copy()
df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
dm = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
df = add_computed_features(df)
if '今回_会場' in df.columns and '1走前_開催' in df.columns:
    df['輸送有無'] = (df['今回_会場'].astype(str) != df['1走前_開催'].astype(str).str[1]).astype(float)
    df.loc[df['1走前_開催'].isna(), '輸送有無'] = float('nan')
baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
for col in df.columns:
    if '馬場状態' in col and col != '馬場状態':
        df[col] = df[col].map(baba_map)
df['dist_m'] = dm
df['単勝オッズ_num'] = pd.to_numeric(df['単勝オッズ'], errors='coerce')

# レース内の人気順（オッズ昇順）を計算
df['人気_num'] = df.groupby('race_id')['単勝オッズ_num'].rank(method='first', ascending=True)

s = df['surface']; r = df['クラス_rank']; d = df['dist_m']
SEG_MASKS = {
    'ダ長': (s=='ダ')&(d>1400) &(r!=1.0),
    'ダ短': (s=='ダ')&(d<=1400)&(r!=1.0),
    '芝短': (s=='芝')&(d<=1400)&(r!=1.0),
    '芝中': (s=='芝')&(d>1400) &(d<=2000)&(r!=1.0),
    '芝長': (s=='芝')&(d>2000) &(r!=1.0),
}

print("確率・EV 計算中...", flush=True)
all_rows = []
for seg in ['芝長', '芝中', '芝短', 'ダ長', 'ダ短']:
    pkg = MODEL[seg]
    feat_cols, scaler, coef, iso = pkg['feat_cols'], pkg['scaler'], pkg['coef'], pkg['isotonic']
    seg_df = df[SEG_MASKS[seg]].copy()
    for fc in feat_cols:
        if fc.endswith('_isnan'):
            base = fc[:-6]
            if base in seg_df.columns and fc not in seg_df.columns:
                seg_df[fc] = seg_df[base].isna().astype(float)
    try:
        X, _, gs, n, *_ = prepare(seg_df, feat_cols, scaler=scaler, top_idx=None, top_idx3=None)
    except Exception as e:
        print(f"  {seg} skip: {e}"); continue
    ss = seg_df.sort_values('race_id').reset_index(drop=True)
    raw_prob = segment_softmax(X @ coef, gs, n)
    ss['prob_raw'] = raw_prob
    ss['prob_calib'] = iso.predict(raw_prob)
    ss['rank_pred'] = ss.groupby('race_id')['prob_raw'].rank(ascending=False, method='first')

    rank1 = ss[ss['rank_pred']==1].set_index('race_id')[
        ['prob_raw','prob_calib','単勝オッズ_num','着順_num','人気_num']]
    rank2 = ss[ss['rank_pred']==2].set_index('race_id')[['prob_raw']].rename(columns={'prob_raw':'prob2'})
    merged = rank1.join(rank2, how='left')
    merged['prob_gap'] = merged['prob_raw'] - merged['prob2'].fillna(0)
    merged['ev'] = merged['prob_calib'] * merged['単勝オッズ_num']
    merged['hit'] = (merged['着順_num'] == 1).astype(int)
    merged['seg'] = seg

    date_map = ss.drop_duplicates('race_id').set_index('race_id')['日付_num']
    merged['日付_num'] = merged.index.map(date_map)
    merged = merged.reset_index()
    all_rows.append(merged)

all_df = pd.concat(all_rows, ignore_index=True)
all_df = all_df.dropna(subset=['ev', 'prob_gap', '着順_num', '単勝オッズ_num'])
print(f"全レース: {len(all_df)}R\n")

MIN_R = 50


def ef(data):
    nr = len(data)
    if nr == 0: return 0, 0., 0.
    acc = data['hit'].mean()
    roi = (data[data['hit']==1]['単勝オッズ_num'].sum() - nr) / nr
    return nr, acc, roi


def score_2325(d24, d25):
    r24, _, roi24 = ef(d24)
    r25, _, roi25 = ef(d25)
    if r24 + r25 < MIN_R: return -99.
    roi24c = max(min(roi24, 0.30), -0.30)
    roi25c = max(min(roi25, 0.30), -0.30)
    return (roi24c * r24 + roi25c * r25) / (r24 + r25)


SEGS = ['芝長', '芝中', '芝短', 'ダ長', 'ダ短']

for seg in SEGS:
    seg_d = all_df[all_df['seg']==seg]
    d24 = seg_d[(seg_d['日付_num']>=230101)&(seg_d['日付_num']<250101)]
    d25 = seg_d[(seg_d['日付_num']>=250101)&(seg_d['日付_num']<260101)]
    d26 = seg_d[seg_d['日付_num']>=260101]

    _, _, roi_b24 = ef(d24); _, _, roi_b25 = ef(d25); _, _, roi_b26 = ef(d26)
    sc_base = score_2325(d24, d25)
    print('=' * 75)
    print(f'[{seg}]  全買い: 2324={roi_b24:+.1%} / 2025={roi_b25:+.1%} / 2026={roi_b26:+.1%}  (2325={sc_base:+.1%})')
    print('=' * 75)

    results = []

    # ── A: オッズ上限フィルタ ──────────────────────────────────────────────
    for odds_max in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0]:
        f24 = d24[d24['単勝オッズ_num'] <= odds_max]
        f25 = d25[d25['単勝オッズ_num'] <= odds_max]
        f26 = d26[d26['単勝オッズ_num'] <= odds_max]
        sc = score_2325(f24, f25)
        r24, a24, roi24 = ef(f24); r25, a25, roi25 = ef(f25); r26, a26, roi26 = ef(f26)
        results.append((sc, f'オッズ≤{odds_max:.0f}', r24, a24, roi24, r25, a25, roi25, r26, a26, roi26))

    # ── B: 人気順フィルタ（モデル1位馬の市場人気 ≤ K） ──────────────────────
    for k in [1, 2, 3, 4, 5, 6]:
        f24 = d24[d24['人気_num'] <= k]
        f25 = d25[d25['人気_num'] <= k]
        f26 = d26[d26['人気_num'] <= k]
        sc = score_2325(f24, f25)
        r24, a24, roi24 = ef(f24); r25, a25, roi25 = ef(f25); r26, a26, roi26 = ef(f26)
        results.append((sc, f'人気≤{k}番', r24, a24, roi24, r25, a25, roi25, r26, a26, roi26))

    # ── C: EV帯フィルタ（EV下限 + オッズ上限） ─────────────────────────────
    for ev_lo, odds_hi in [(0.5, 5.0),(0.6, 6.0),(0.7, 8.0),(0.7, 6.0),(0.8, 8.0),(0.8,10.0),(0.9,8.0)]:
        f24 = d24[(d24['ev']>=ev_lo)&(d24['単勝オッズ_num']<=odds_hi)]
        f25 = d25[(d25['ev']>=ev_lo)&(d25['単勝オッズ_num']<=odds_hi)]
        f26 = d26[(d26['ev']>=ev_lo)&(d26['単勝オッズ_num']<=odds_hi)]
        sc = score_2325(f24, f25)
        r24, a24, roi24 = ef(f24); r25, a25, roi25 = ef(f25); r26, a26, roi26 = ef(f26)
        results.append((sc, f'EV≥{ev_lo}+オッズ≤{odds_hi:.0f}', r24, a24, roi24, r25, a25, roi25, r26, a26, roi26))

    # ── D: prob_gap フィルタ ───────────────────────────────────────────────
    for gap in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        f24 = d24[d24['prob_gap'] >= gap]
        f25 = d25[d25['prob_gap'] >= gap]
        f26 = d26[d26['prob_gap'] >= gap]
        sc = score_2325(f24, f25)
        r24, a24, roi24 = ef(f24); r25, a25, roi25 = ef(f25); r26, a26, roi26 = ef(f26)
        results.append((sc, f'gap≥{gap:.2f}', r24, a24, roi24, r25, a25, roi25, r26, a26, roi26))

    # ── E: gap + オッズ上限 ────────────────────────────────────────────────
    for gap, odds_hi in [(0.05,10.0),(0.10,8.0),(0.10,10.0),(0.15,8.0),(0.05,6.0)]:
        f24 = d24[(d24['prob_gap']>=gap)&(d24['単勝オッズ_num']<=odds_hi)]
        f25 = d25[(d25['prob_gap']>=gap)&(d25['単勝オッズ_num']<=odds_hi)]
        f26 = d26[(d26['prob_gap']>=gap)&(d26['単勝オッズ_num']<=odds_hi)]
        sc = score_2325(f24, f25)
        r24, a24, roi24 = ef(f24); r25, a25, roi25 = ef(f25); r26, a26, roi26 = ef(f26)
        results.append((sc, f'gap≥{gap}+オッズ≤{odds_hi:.0f}', r24, a24, roi24, r25, a25, roi25, r26, a26, roi26))

    # ── ソートして上位表示 ────────────────────────────────────────────────
    results.sort(key=lambda x: -x[0])
    print(f"  {'条件':<22} {'2325sc':>7} | {'2324R':>5} {'2324的中率':>8} {'2324ROI':>7} "
          f"| {'2025R':>5} {'2025ROI':>7} | {'2026R':>5} {'2026的中率':>8} {'2026ROI':>7}")
    print('  ' + '-' * 85)
    for sc, label, r24, a24, roi24, r25, a25, roi25, r26, a26, roi26 in results[:12]:
        if sc == -99.: continue
        print(f"  {label:<22} {sc:>+7.1%} | {r24:>5} {a24:>8.1%} {roi24:>+7.1%} "
              f"| {r25:>5} {roi25:>+7.1%} | {r26:>5} {a26:>8.1%} {roi26:>+7.1%}")
    print()
