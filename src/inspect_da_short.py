# coding: utf-8
"""safe_da_short の5特徴モデルの係数を確認する"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
from save_v3 import add_computed_features

L2 = 0.006

FEATS = ['近5走_上り3F平均', '3走前_クラス差', '4走前_クラス差', 'コース枠_r200_勝率', '1走前_馬場状態']

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
for col in FEATS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df_trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
df_val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]

X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(df_trn, FEATS, top_idx=None, top_idx3=None, fit=True)
X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(df_val, FEATS, scaler=scaler, top_idx=None, top_idx3=None)
beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va)

print("=" * 50)
print("  safe_da_short 5特徴モデル 係数")
print("=" * 50)
for f, b in zip(FEATS, beta):
    direction = "↑正(高いほど有利)" if b > 0 else "↓負(低いほど有利)"
    print(f"  {f:30s}  β={b:+.4f}  {direction}")

print()
print("1走前_馬場状態の解釈:")
print("  良=0, 稍重=1, 重=2, 不良=3")
b_baba = beta[FEATS.index('1走前_馬場状態')]
if b_baba > 0:
    print(f"  β={b_baba:+.4f} → 前走が道悪だった馬が有利（重・不良ほどプラス）")
else:
    print(f"  β={b_baba:+.4f} → 前走が良馬場だった馬が有利（良馬場ほどプラス）")

print()
print("クラス差の解釈 (クラス差=今走クラスrank - N走前クラスrank):")
for fname in ['3走前_クラス差', '4走前_クラス差']:
    b = beta[FEATS.index(fname)]
    if b > 0:
        print(f"  {fname}: β={b:+.4f} → 今走のクラスが高い（格上がり）ほど有利")
    else:
        print(f"  {fname}: β={b:+.4f} → 今走のクラスが低い（格下がり）ほど有利")
