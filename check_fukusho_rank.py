# coding: utf-8
"""
check_fukusho_rank.py
  - 既存 accuracy_model の特徴量セットを流用
  - 単勝ロス (y=1位) vs 複勝ロス (y=top3) でそれぞれ再学習
  - OOS のランク別勝率・3着以内率を比較
"""
import sys, os, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from save_v3 import add_computed_features
from save_conditional_logit import segment_softmax, neg_log_lik_fukusho_and_grad

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')

MODEL = pickle.load(open(os.path.join(BASE_DIR, 'models', 'accuracy_model.pkl'), 'rb'))

ORDER = ['芝長', '芝中', '芝短', 'ダ長', 'ダ短']

L2       = 0.006
LR       = 0.001
N_EPOCHS = 600
PATIENCE = 80

# ─── Adam 学習 ──────────────────────────────────────────────────────────────
def adam_train(X_tr, y_tr, gs_tr, X_val, y_val, gs_val, n_tr, n_val, loss_fn):
    beta = np.zeros(X_tr.shape[1])
    m = v = np.zeros_like(beta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_loss, best_beta, wait = np.inf, beta.copy(), 0
    for ep in range(1, N_EPOCHS + 1):
        loss_tr, grad = loss_fn(beta, X_tr, y_tr, gs_tr, len(X_tr), n_tr)
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        mh = m / (1 - b1 ** ep)
        vh = v / (1 - b2 ** ep)
        beta -= LR * mh / (np.sqrt(vh) + eps)
        # val loss（単純 NLL）
        probs_val = segment_softmax(X_val @ beta, gs_val, len(X_val))
        val_loss  = -np.sum(y_val * np.log(np.clip(probs_val, 1e-15, 1))) / n_val
        if val_loss < best_loss - 1e-7:
            best_loss, best_beta, wait = val_loss, beta.copy(), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    return best_beta

# ─── データ準備 ──────────────────────────────────────────────────────────────
def get_group_starts(race_ids):
    _, idx = np.unique(race_ids, return_index=True)
    return np.sort(idx)

def seg_key(surf, dist_m):
    if pd.isna(dist_m): return None
    s = str(surf).strip()
    if s == '芝': return '芝短' if dist_m <= 1400 else ('芝中' if dist_m <= 2000 else '芝長')
    elif s == 'ダ': return 'ダ短' if dist_m <= 1400 else 'ダ長'
    return None

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
df['seg_key'] = [seg_key(s, d) for s, d in zip(df['_surf'], df['_dist_m'])]
df = df[df['seg_key'].notna()].copy()
df['dist_m'] = df['_dist_m']
df = add_computed_features(df)
baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
for col in df.columns:
    if '馬場状態' in col and col != '馬場状態':
        df[col] = df[col].map(baba_map)
print('完了\n')

PERIODS = {
    'train': (130101, 220101),
    'val':   (220101, 230101),
    'oos':   (230101, 990101),
}

def build_X(grp, feat_cols, scaler):
    X = np.zeros((len(grp), len(feat_cols)), dtype=float)
    for j, f in enumerate(feat_cols):
        if f.endswith('_isnan'):
            base = f[:-6]
            X[:, j] = (~grp[base].notna()).astype(float).values if base in grp.columns else 1.0
        else:
            col = grp[f] if f in grp.columns else pd.Series(np.nan, index=grp.index)
            X[:, j] = pd.to_numeric(col, errors='coerce').fillna(0.0).values
    return scaler.transform(X)

# ─── 単勝ロス（参照用） ──────────────────────────────────────────────────────
def win_loss_fn(beta, X, y, gs, n, nr):
    probs = segment_softmax(X @ beta, gs, n)
    loss  = -np.sum(y * np.log(np.clip(probs, 1e-15, 1))) / nr + L2 * np.dot(beta, beta)
    grad  = -(X.T @ (y - probs)) / nr + 2 * L2 * beta
    return loss, grad

def fukusho_loss_fn(beta, X, y, gs, n, nr):
    return neg_log_lik_fukusho_and_grad(beta, X, y, gs, n, nr)

# ─── セグメント別評価 ─────────────────────────────────────────────────────────
print(f'{"="*70}')
print(f'{"セグメント":<6} {"モデル":<8} | {"rank1":>6} {"rank2":>6} {"rank3":>6} {"rank4":>6} {"rank5":>6} | {"3着1":>6} {"3着2":>6} {"3着3":>6} {"3着4":>6} {"3着5":>6}')
print(f'{"="*70}')

for seg in ORDER:
    art = MODEL.get(seg)
    if art is None:
        continue
    feat_cols = art['feat_cols']
    scaler    = art['scaler']

    seg_df = df[df['seg_key'] == seg].sort_values('race_id').reset_index(drop=True)

    def split(period):
        d0, d1 = PERIODS[period]
        g = seg_df[(seg_df['日付_num'] >= d0) & (seg_df['日付_num'] < d1)].copy()
        g = g.sort_values('race_id').reset_index(drop=True)
        X = build_X(g, feat_cols, scaler)
        gs = get_group_starts(g['race_id'].values)
        nr = g['race_id'].nunique()
        y_win = (g['着順_num'] == 1).astype(float).values
        y_p3  = (g['着順_num'] <= 3).astype(float).values
        return X, gs, nr, y_win, y_p3, g

    X_tr, gs_tr, nr_tr, yw_tr, yp_tr, _  = split('train')
    X_va, gs_va, nr_va, yw_va, yp_va, _  = split('val')
    X_oo, gs_oo, nr_oo, yw_oo, yp_oo, g_oo = split('oos')

    def rank_stats(beta, g, X, gs):
        scores = X @ beta
        g = g.copy()
        g['_score'] = scores
        g['_rank']  = g.groupby('race_id')['_score'].rank(ascending=False, method='first').astype(int)
        wr_list, p3_list = [], []
        for r in range(1, 6):
            sub = g[g['_rank'] == r]
            wr_list.append((sub['着順_num'] == 1).mean() if len(sub) else float('nan'))
            p3_list.append((sub['着順_num'] <= 3).mean() if len(sub) else float('nan'))
        return wr_list, p3_list

    # ── 単勝モデル（既存 coef） ────────────────────────────────────────────
    beta_win_orig = art['coef']
    wr_orig, p3_orig = rank_stats(beta_win_orig, g_oo, X_oo, gs_oo)

    # ── 単勝ロスで再学習 ──────────────────────────────────────────────────
    beta_win_new = adam_train(X_tr, yw_tr, gs_tr, X_va, yw_va, gs_va, nr_tr, nr_va, win_loss_fn)
    wr_win, p3_win = rank_stats(beta_win_new, g_oo, X_oo, gs_oo)

    # ── 複勝ロスで学習 ────────────────────────────────────────────────────
    beta_fuk = adam_train(X_tr, yp_tr, gs_tr, X_va, yw_va, gs_va, nr_tr, nr_va, fukusho_loss_fn)
    wr_fuk, p3_fuk = rank_stats(beta_fuk, g_oo, X_oo, gs_oo)

    def fmt(lst):
        return '  '.join(f'{v*100:5.1f}%' if not (v!=v) else '   --' for v in lst)

    sep = '-' * 70
    print(f'{sep}')
    print(f'{seg:<6} 既存win  | 勝率: {fmt(wr_orig)}')
    print(f'{"":6} 再学win  | 勝率: {fmt(wr_win)}')
    print(f'{"":6} 複勝ロス | 勝率: {fmt(wr_fuk)}')
    print(f'{"":6} ---')
    print(f'{"":6} 既存win  | 3着: {fmt(p3_orig)}')
    print(f'{"":6} 再学win  | 3着: {fmt(p3_win)}')
    print(f'{"":6} 複勝ロス | 3着: {fmt(p3_fuk)}')

print('=' * 70)
print('完了')
