# coding: utf-8
"""
クラス差の走数バリエーションを直接評価して比較する
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
from save_v3 import add_computed_features

L2 = 0.006

def _loss_grad(beta, X, y, gs, n, nr, l2=0.0):
    probs = segment_softmax(X @ beta, gs, n)
    loss  = -np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) / nr + l2 * np.dot(beta, beta)
    grad  = -(X.T @ (y - probs)) / nr + 2 * l2 * beta
    return loss, grad

def adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va, l2=L2):
    d = X_tr.shape[1]
    beta, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t, best_val, best_beta, no_imp = 0, np.inf, np.zeros(d), 0
    for epoch in range(1, N_EPOCHS + 1):
        _, grad = _loss_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr, l2)
        t += 1
        m = b1*m + (1-b1)*grad
        v = b2*v + (1-b2)*grad**2
        beta -= LR * (m/(1-b1**t)) / (np.sqrt(v/(1-b2**t)) + eps)
        if epoch % 10 == 0:
            vl, _ = _loss_grad(beta, X_va, y_va, gs_va, n_va, nr_va, l2=0.0)
            if vl < best_val:
                best_val, best_beta, no_imp = vl, beta.copy(), 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE // 10:
                break
    return best_beta

def roi_from_top1(top1):
    won  = top1['着順_num'] == 1
    odds = pd.to_numeric(top1['単勝オッズ'], errors='coerce')
    if len(top1) == 0:
        return float('nan'), 0
    return (odds[won] * 100).sum() / (len(top1) * 100) - 1, len(top1)

def comb2526(r25, n25, r26, n26):
    if n25 + n26 == 0:
        return 0.0
    return (r25 * n25 + r26 * n26) / (n25 + n26)

def evaluate_set(df_trn, df_val, oos_2324, oos_2025, oos_2026, feats):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va)
    results = {}
    for label, oos in [('2324', oos_2324), ('2025', oos_2025), ('2026', oos_2026)]:
        if len(oos) == 0:
            results[label] = (float('nan'), 0)
            continue
        valid_p = [c for c in valid if c in oos.columns]
        X_p, _, gs_p, n_p, *_ = prepare(oos, valid_p, scaler=scaler, top_idx=None, top_idx3=None)
        scored = oos.sort_values('race_id').reset_index(drop=True)
        scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
        top1 = scored[scored['rank'] == 1]
        results[label] = roi_from_top1(top1)
    return results, valid, beta

# ──────────────────────────────────────────────
print("データ読み込み中...")
df = pd.read_parquet(DATA_FILE)
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
df = df.dropna(subset=['日付_num', '着順_num'])
df = df[df['着順_num'] < 99]
df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                 df['開催'].astype(str).str.strip() + '_' +
                 df['Ｒ'].astype(str).str.strip())
df = df[df['開催'].notna()].copy()
df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
dm = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
df = df[(df['surface'] == 'ダ') & (dm <= 1400)].copy()
if 'クラス_rank' in df.columns:
    df = df[df['クラス_rank'] != 1.0].copy()
df = add_computed_features(df)
baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
for col in df.columns:
    if '馬場状態' in col:
        df[col] = df[col].map(baba_map)
for col in ['1走前_クラス差', '2走前_クラス差', '3走前_クラス差', '4走前_クラス差',
            '1走前_馬場状態', '近5走_上り3F平均', 'コース枠_r200_勝率']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df_trn   = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
df_val   = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
oos_2324 = df[(df['日付_num'] >= 230101) & (df['日付_num'] < 250101)]
oos_2025 = df[(df['日付_num'] >= 250101) & (df['日付_num'] < 260101)]
oos_2026 = df[df['日付_num'] >= 260101]
print(f"train:{len(df_trn):,}  val:{len(df_val):,}  2324:{len(oos_2324):,}\n")

# ──────────────────────────────────────────────
BASE = ['近5走_上り3F平均', 'コース枠_r200_勝率', '1走前_馬場状態']

SETS = {
    'A: 元 (3走前+4走前)':     BASE + ['3走前_クラス差', '4走前_クラス差'],
    'B: 1走前+2走前':           BASE + ['1走前_クラス差', '2走前_クラス差'],
    'C: 1走前のみ':             BASE + ['1走前_クラス差'],
    'D: 1+2+3走前':             BASE + ['1走前_クラス差', '2走前_クラス差', '3走前_クラス差'],
    'E: 1+2+3+4走前(全部)':    BASE + ['1走前_クラス差', '2走前_クラス差', '3走前_クラス差', '4走前_クラス差'],
    'F: 馬場なし (3走前+4走前)': ['近5走_上り3F平均', 'コース枠_r200_勝率', '3走前_クラス差', '4走前_クラス差'],
}

print(f"{'=' * 70}")
print(f"  クラス差バリエーション 直接比較")
print(f"{'=' * 70}")
print(f"  {'セット':30s}  {'2324':>8}  {'2025':>8}  {'2026':>8}  {'25+26':>8}")
print(f"  {'-'*66}")

for name, feats in SETS.items():
    t0 = time.time()
    res, valid, beta = evaluate_set(df_trn, df_val, oos_2324, oos_2025, oos_2026, feats)
    r2324, n2324 = res['2324']
    r25, n25     = res['2025']
    r26, n26     = res['2026']
    rcomb = comb2526(r25, n25, r26, n26)
    print(f"  {name:30s}  {r2324*100:+7.2f}%  {r25*100:+7.2f}%  {r26*100:+7.2f}%  {rcomb*100:+7.2f}%  ({int(time.time()-t0)}s)")

    # クラス差の係数も出力
    klasses = [f for f in valid if 'クラス差' in f]
    if klasses:
        coeffs = []
        for k in klasses:
            idx = valid.index(k)
            coeffs.append(f"{k}:β={beta[idx]:+.4f}")
        print(f"    係数: {', '.join(coeffs)}")
