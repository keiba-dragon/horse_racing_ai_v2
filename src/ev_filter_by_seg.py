# coding: utf-8
"""
セグメント別 EV閾値最適化
- 選択基準: 2325合算ROI（単年±30%キャップ付き）
- 検証: 2026 OOS
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
    rank1 = ss[ss['rank_pred']==1].set_index('race_id')[['prob_raw','prob_calib','単勝オッズ_num','着順_num']]
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
print(f"全レース: {len(all_df)}R")

EV_THS = [0.00, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.50]
MIN_R_2325 = 50


def ef(data, ev_th):
    f = data[data['ev'] >= ev_th]
    nr = len(f)
    if nr == 0:
        return 0, 0., 0.
    acc = f['hit'].mean()
    roi = (f[f['hit']==1]['単勝オッズ_num'].sum() - nr) / nr
    return nr, acc, roi


def score_2325(seg_d, ev_th):
    """2325合算ROI（単年±30%キャップ）"""
    d24 = seg_d[(seg_d['日付_num']>=230101)&(seg_d['日付_num']<250101)]
    d25 = seg_d[(seg_d['日付_num']>=250101)&(seg_d['日付_num']<260101)]
    r24, _, roi24 = ef(d24, ev_th)
    r25, _, roi25 = ef(d25, ev_th)
    if r24 + r25 < MIN_R_2325:
        return -99.
    roi24c = max(min(roi24, 0.30), -0.30)
    roi25c = max(min(roi25, 0.30), -0.30)
    return (roi24c * r24 + roi25c * r25) / (r24 + r25)


print()
print('=' * 90)
print('セグメント別 EV閾値最適化（2325合算ROIで選択 → 2026で検証）')
print('=' * 90)

best_by_seg = {}
for seg in ['芝長', '芝中', '芝短', 'ダ長', 'ダ短']:
    seg_d = all_df[all_df['seg']==seg]
    d24 = seg_d[(seg_d['日付_num']>=230101)&(seg_d['日付_num']<250101)]
    d25 = seg_d[(seg_d['日付_num']>=250101)&(seg_d['日付_num']<260101)]
    d26 = seg_d[seg_d['日付_num']>=260101]

    _, _, roi_b24 = ef(d24, 0.0)
    _, _, roi_b25 = ef(d25, 0.0)
    _, _, roi_b26 = ef(d26, 0.0)
    sc_base = score_2325(seg_d, 0.0)
    print(f'\n[{seg}]  全買い: 2324={roi_b24:+.1%} / 2025={roi_b25:+.1%} / 2026={roi_b26:+.1%}  '
          f'(2325スコア={sc_base:+.1%})')
    print(f"  {'EV≥':>5}  {'2325score':>10} {'2324ROI':>8} {'2025ROI':>8} {'2026ROI':>8}")
    print('  ' + '-' * 50)

    best_score = sc_base
    best_ev = 0.0
    for ev_th in EV_THS[1:]:
        sc = score_2325(seg_d, ev_th)
        if sc == -99.:
            continue
        _, _, roi24 = ef(d24, ev_th)
        _, _, roi25 = ef(d25, ev_th)
        r26, _, roi26 = ef(d26, ev_th)
        mark = ''
        if sc > best_score:
            best_score = sc; best_ev = ev_th; mark = ' ← best'
        print(f'  {ev_th:>5.2f}  {sc:>+10.1%} {roi24:>+8.1%} {roi25:>+8.1%} {roi26:>+8.1%}{mark}')
    best_by_seg[seg] = best_ev
    print(f'  → 最適EV閾値: {best_ev:.2f}  (2325スコア={best_score:+.1%})')

print()
print('=' * 90)
print('最終サマリー: セグメント別最適閾値（2325選択）× 全期間')
print('=' * 90)
print(f"{'セグメント':<6} {'EV閾値':>6} | {'2324R':>5} {'2324的中率':>9} {'2324ROI':>8} "
      f"| {'2025R':>5} {'2025的中率':>9} {'2025ROI':>8} "
      f"| {'2026R':>5} {'2026的中率':>9} {'2026ROI':>8} | {'25+26ROI':>9}")
print('-' * 95)

agg = {'r25': 0, 'pay25': 0, 'r26': 0, 'pay26': 0}
for seg in ['芝長', '芝中', '芝短', 'ダ長', 'ダ短']:
    seg_d = all_df[all_df['seg']==seg]
    ev_th = best_by_seg[seg]
    d24 = seg_d[(seg_d['日付_num']>=230101)&(seg_d['日付_num']<250101)]
    d25 = seg_d[(seg_d['日付_num']>=250101)&(seg_d['日付_num']<260101)]
    d26 = seg_d[seg_d['日付_num']>=260101]
    r24, a24, roi24 = ef(d24, ev_th)
    r25, a25, roi25 = ef(d25, ev_th)
    r26, a26, roi26 = ef(d26, ev_th)
    f25 = d25[d25['ev']>=ev_th]
    f26 = d26[d26['ev']>=ev_th]
    pay25 = f25[f25['hit']==1]['単勝オッズ_num'].sum()
    pay26 = f26[f26['hit']==1]['単勝オッズ_num'].sum()
    roi2526 = (pay25 + pay26 - r25 - r26) / (r25 + r26) if (r25+r26) > 0 else float('nan')
    agg['r25'] += r25; agg['pay25'] += pay25
    agg['r26'] += r26; agg['pay26'] += pay26
    print(f"{seg:<6} {ev_th:>6.2f} | {r24:>5} {a24:>9.1%} {roi24:>+8.1%} "
          f"| {r25:>5} {a25:>9.1%} {roi25:>+8.1%} "
          f"| {r26:>5} {a26:>9.1%} {roi26:>+8.1%} | {roi2526:>+9.1%}")

roi_all2526 = (agg['pay25']+agg['pay26']-agg['r25']-agg['r26']) / (agg['r25']+agg['r26'])
print('-' * 95)
print(f"{'全体計':<13} | {'---':>5} {'---':>9} {'---':>8} "
      f"| {agg['r25']:>5} {'---':>9} {'---':>8} "
      f"| {agg['r26']:>5} {'---':>9} {'---':>8} | {roi_all2526:>+9.1%}")
print()
print(f"全体 25+26合算 ROI: {roi_all2526:+.1%}  ({agg['r25']+agg['r26']}R)")
