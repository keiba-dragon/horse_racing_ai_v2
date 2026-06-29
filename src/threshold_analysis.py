# coding: utf-8
"""5セグメント別 閾値分析 (修正版)
  - OOS: 2023+ (save_conditional_logit.py と同じ定義)
  - 1走前_馬場状態 を BABA_MAP でエンコード
  - 間隔_長_flag を 間隔列から動的生成
  - セグメント分岐: save スクリプトと同一ロジック
"""
import sys, io, pickle, time
import numpy as np
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PARQUET = 'data/processed/all_venues_features.parquet'
MODEL   = 'models/roi_model.pkl'

BABA_MAP = {'良': 0, '稍': 1, '稍重': 1, '重': 2, '不': 3, '不良': 3}

with open(MODEL, 'rb') as f:
    mdl = pickle.load(f)
artifacts = mdl['artifacts']

# ── Parquet ──────────────────────────────────────────────────────────────
print("Parquet読み込み中...")
t0 = time.time()
df_all = pd.read_parquet(PARQUET)
print(f"  全行数: {len(df_all):,}  ({time.time()-t0:.1f}秒)")

# ── 前処理 ────────────────────────────────────────────────────────────────
df_all['_odds']    = pd.to_numeric(df_all['単勝オッズ'], errors='coerce')
df_all['_fuk']     = pd.to_numeric(df_all['複勝配当'],  errors='coerce')
df_all['着順_num'] = pd.to_numeric(df_all['着順_num'],  errors='coerce')
df_all['race_id']  = (df_all['日付_num'].astype(str) + '_'
                      + df_all['開催'].astype(str) + '_'
                      + df_all['Ｒ'].astype(str))

# surface & dist_m (save スクリプトと同じ: 距離文字列から抽出)
df_all['surface'] = df_all['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
df_all['dist_m']  = df_all['距離'].astype(str).str.extract(r'(\d+)')[0].astype(float).fillna(0)

# セグメント (save スクリプトと同一ロジック)
shi = df_all['surface'] == '芝'
da  = df_all['surface'] == 'ダ'
dm  = df_all['dist_m']
df_all['_seg'] = '不明'
df_all.loc[shi & (dm <= 1400),              '_seg'] = '芝短'
df_all.loc[shi & (dm > 1400) & (dm <= 2000),'_seg'] = '芝中'
df_all.loc[shi & (dm > 2000),               '_seg'] = '芝長'
df_all.loc[da  & (dm <= 1400),              '_seg'] = 'ダ短'
df_all.loc[da  & (dm > 1400),               '_seg'] = 'ダ'   # ダ中長距離

# 間隔_長_flag (save スクリプトの prepare 内で生成される)
interval = pd.to_numeric(df_all['間隔'], errors='coerce')
df_all['間隔_長_flag'] = (interval >= 60).astype(float)

# 馬場状態系の文字列 → 数値変換
for baba_col in [c for c in df_all.columns if '馬場状態' in c]:
    encoded = df_all[baba_col].map(BABA_MAP)
    if encoded.notna().any():
        # 数値変換済みの列と置き換え
        df_all[baba_col] = encoded

print(f"  OOS (2023+): {(df_all['日付_num'] >= 230101).sum():,}行")
print("セグメント分布:")
print(df_all[df_all['日付_num'] >= 230101]['_seg'].value_counts().to_string())

# ── clogit スコア計算 ──────────────────────────────────────────────────
def compute_scores(df_seg, art):
    feat_cols = art['feat_cols']
    df_work = df_seg.copy()
    miss = [c for c in feat_cols if c not in df_work.columns]
    for c in miss:
        df_work[c] = 0.0
    if miss:
        print(f"    欠損特徴量 {len(miss)}: {miss}")
    X_raw = df_work[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    X_sc  = art['scaler'].transform(X_raw)
    parts = [X_sc]
    if art.get('top_idx') is not None:
        Xt2 = art['poly2'].transform(X_sc[:, art['top_idx']])
        Xin2 = art['inter_scaler2'].transform(Xt2[:, len(art['top_idx']):])
        parts.append(Xin2)
    if art.get('top_idx3') is not None:
        Xt3 = art['poly3'].transform(X_sc[:, art['top_idx3']])
        Xin3 = art['inter_scaler3'].transform(Xt3[:, len(art['top_idx3']):])
        parts.append(Xin3)
    return np.hstack(parts) @ art['coef']

def softmax_by_race(scores, race_ids):
    out = np.zeros(len(scores))
    for rid, idx_group in pd.Series(race_ids).groupby(race_ids).groups.items():
        idx = list(idx_group)
        s = scores[idx] - scores[idx].max()
        e = np.exp(s); out[idx] = e / e.sum()
    return out

SEG_LABEL = {'ダ': 'ダ中長距離', 'ダ短': 'ダ短', '芝短': '芝短', '芝中': '芝中', '芝長': '芝長'}

# ── 全データにスコア付与 ─────────────────────────────────────────────
all_rows = []
for seg_key, art in artifacts.items():
    if not art.get('feat_cols'): continue
    sub = df_all[df_all['_seg'] == seg_key].copy().reset_index(drop=True)
    if len(sub) == 0: continue
    label = SEG_LABEL.get(seg_key, seg_key)

    scores = compute_scores(sub, art)
    probs  = softmax_by_race(scores, sub['race_id'].values)
    iso    = art.get('isotonic')
    calib  = iso.predict(probs) if iso is not None else probs

    sub['_calib']     = calib
    sub['_prob_raw']  = probs  # raw softmax (save スクリプト準拠)
    # rank は raw softmax で (save スクリプトと同じ)
    sub['_rank']      = sub.groupby('race_id')['_prob_raw'].rank(
                            ascending=False, method='first').fillna(999).astype(int)
    sub['_seg_label'] = label
    all_rows.append(sub)
    oos_sub = sub[sub['日付_num'] >= 230101]
    top1_oos = oos_sub[oos_sub['_rank'] == 1]
    h1 = (top1_oos['着順_num'] == 1).sum()
    n  = len(top1_oos)
    r1 = top1_oos.loc[top1_oos['着順_num']==1, '_odds'].sum()
    roi = r1/n - 1 if n > 0 else float('nan')
    print(f"  {label}: OOS {n:,}R  1着{h1/n*100:.1f}%  単勝ROI(raw)={roi:+.1%}")

df_scored = pd.concat(all_rows, ignore_index=True)
df_top1   = df_scored[df_scored['_rank'] == 1].copy()

# ── 閾値分析 ──────────────────────────────────────────────────────────
THRESHOLDS = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30]

def roi_stats(df):
    n = len(df)
    if n < 10: return None
    h1  = (df['着順_num'] == 1).sum()
    h3  = (df['着順_num'] <= 3).sum()
    r1  = df.loc[df['着順_num'] == 1, '_odds'].sum()
    r3  = df.loc[df['着順_num'] <= 3, '_fuk'].fillna(0).sum()
    return {'n': n, 'h1': h1, 'h3': h3,
            'win%': h1/n*100, 'top3%': h3/n*100,
            'roi1': r1/n - 1, 'roif': r3/(n*100) - 1}

def print_seg(df_period, label, seg_name):
    sub = df_period[df_period['_seg_label'] == seg_name]
    base = roi_stats(sub)
    if base is None:
        print(f"\n  【{seg_name}】データ不足")
        return
    print(f"\n  ┌─ 【{seg_name}】  全買い {base['n']:,}R  "
          f"1着{base['win%']:.1f}%  3着{base['top3%']:.1f}%  "
          f"単勝ROI={base['roi1']:+.1%}  複勝ROI={base['roif']:+.1%}")
    print(f"  │  {'閾値':>6}  {'R数':>6}  {'1着率':>6}  {'3着率':>6}  {'単勝ROI':>9}  {'複勝ROI':>9}")
    prev_n = None
    for thr in THRESHOLDS:
        s2 = sub[sub['_calib'] >= thr]
        st = roi_stats(s2)
        if st is None: continue
        if st['n'] == prev_n: continue
        prev_n = st['n']
        print(f"  │  >={thr:.2f}  {st['n']:>6,}  {st['win%']:>5.1f}%  "
              f"{st['top3%']:>5.1f}%  {st['roi1']:>+9.1%}  {st['roif']:>+9.1%}")
    print(f"  └{'─'*63}")

def print_period(df_period, period_label):
    top1_p = df_period[df_period['_rank'] == 1]
    print(f"\n{'#'*75}")
    print(f"  ■ {period_label}  ({len(top1_p):,}レース)")
    print(f"{'#'*75}")
    for seg in ['ダ中長距離', 'ダ短', '芝短', '芝中', '芝長']:
        print_seg(top1_p, period_label, seg)

# 2023+ 全体 (save スクリプト準拠)
df23p = df_scored[df_scored['日付_num'] >= 230101]
print_period(df23p, "2023+ (save スクリプト準拠 OOS)")

# 年別内訳
for yr, lo, hi in [('2023', 230101, 231231), ('2024', 240101, 241231), ('2025', 250101, 251231)]:
    dfyr = df_scored[(df_scored['日付_num'] >= lo) & (df_scored['日付_num'] <= hi)]
    top1 = dfyr[dfyr['_rank'] == 1]
    base = roi_stats(top1)
    if base:
        print(f"  {yr}:  {base['n']:,}R  1着{base['win%']:.1f}%  単勝ROI={base['roi1']:+.1%}")
