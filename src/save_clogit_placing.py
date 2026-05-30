# coding: utf-8
"""
Placing model (top-2 / top-3) via exploded logit (Plackett-Luce).

既存 conditional_logit.pkl のscaler/poly2/top_idxを流用し、
βのみ exploded logit (top-3展開) で再学習。
isotonic calibration を top-2 / top-3 それぞれ個別にfit。

保存: models/final_model_placing.pkl
  artifacts['芝'] = {
      scaler, poly2, inter_scaler2, top_idx,
      coef,         # 再学習済みβ
      feat_cols,
      isotonic_top2,  # raw prob → P(top-2)
      isotonic_top3,  # raw prob → P(top-3)
  }
"""
import os, sys, json, pickle, argparse
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from save_lambdarank_pace import add_pace_features
from save_conditional_logit import (
    add_new_features, segment_softmax, prepare,
    neg_log_lik_and_grad, get_group_starts,
)

DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

LR       = 0.001
N_EPOCHS = 800
PATIENCE = 100
CHECK_EVERY = 10


def make_race_id(df):
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    return df


def get_surface(df):
    return df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')


def expand_topk(df, k=3):
    """
    exploded logit 展開。各レースを最大k個のサブレースに分割。
    サブレースr (0-indexed):
      - 着順_num > r の馬を残す
      - 着順=r+1 の馬を着順_num=1 にリナンバー、他は2
      - race_id に '__r{r}' サフィックスを付加
    """
    parts = []
    for rid, rdf in df.groupby('race_id', sort=False):
        rdf = rdf.sort_values('着順_num').reset_index(drop=True)
        for r in range(k):
            target_rank = r + 1
            remaining = rdf[rdf['着順_num'] >= target_rank].copy()
            if len(remaining) < 2:
                break
            if not (remaining['着順_num'] == target_rank).any():
                break
            remaining['着順_num'] = np.where(
                remaining['着順_num'] == target_rank, 1, 2
            )
            remaining['race_id'] = f"{rid}__r{r}"
            parts.append(remaining)
    return pd.concat(parts, ignore_index=True)


def train_surface(trn_s, val_s, art_base, alpha=1e-4):
    """
    surface別 exploded logit 学習。
    art_base: 既存モデルのscaler/poly2/top_idxを含むdict
    Returns: (beta, train_metrics)
    """
    feat_cols = art_base['feat_cols']

    # ── 訓練データ展開 ────────────────────────────────────────
    print("  訓練データ展開中 (top-3)...")
    trn_exp = expand_topk(trn_s, k=3)
    print(f"  展開後: {len(trn_exp):,}行 ({len(trn_exp)//len(trn_s):.1f}倍)")

    # ── 特徴行列作成（既存スケーラーを使用、交互作用なし） ───
    # 展開後行数×916列でメモリ超過するため poly2/top_idx は無効化
    # 3倍のデータ量が交互作用特徴量の代替となる
    X_tr, y_tr, gs_tr, n_tr, nr_tr, *_ = prepare(
        trn_exp, feat_cols,
        scaler=art_base['scaler'], poly2=None,
        inter_scaler2=None, top_idx=None,
        poly3=None, inter_scaler3=None, top_idx3=None, fit=False)

    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        val_s, feat_cols,
        scaler=art_base['scaler'], poly2=None,
        inter_scaler2=None, top_idx=None,
        poly3=None, inter_scaler3=None, top_idx3=None, fit=False)

    print(f"  特徴量次元: {X_tr.shape[1]}")

    # ── Adam最適化 ────────────────────────────────────────────
    d = X_tr.shape[1]
    beta = np.zeros(d)
    m = np.zeros(d)
    v = np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    best_val = np.inf
    best_beta = beta.copy()
    no_improve = 0

    vl0, _ = neg_log_lik_and_grad(beta, X_va, y_va, gs_va, n_va, nr_va)
    print(f"  epoch=   0  val_loss={vl0:.4f}  (beta=0 baseline)")

    for epoch in range(1, N_EPOCHS + 1):
        loss, grad = neg_log_lik_and_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr)
        t += 1
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        beta -= LR * m_hat / (np.sqrt(v_hat) + eps)

        if epoch % CHECK_EVERY == 0:
            vl, _ = neg_log_lik_and_grad(beta, X_va, y_va, gs_va, n_va, nr_va)
            marker = ''
            if vl < best_val:
                best_val = vl
                best_beta = beta.copy()
                no_improve = 0
                marker = ' ← best'
            else:
                no_improve += 1
            if epoch % 100 == 0 or marker:
                print(f"  epoch={epoch:4d}  tr={loss:.4f}  val={vl:.4f}{marker}")
            if no_improve >= PATIENCE // CHECK_EVERY:
                print(f"  早期停止: epoch={epoch}")
                break

    print(f"  best val_loss={best_val:.4f}")
    return best_beta


def fit_isotonic(raw_probs, labels):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw_probs, labels)
    return iso


def predict_raw(df_s, art):
    """original race (展開なし) でraw softmax確率を返す"""
    s = df_s.sort_values('race_id').reset_index(drop=True)
    X, _, gs, n, *_ = prepare(
        s, art['feat_cols'],
        scaler=art['scaler'], poly2=art.get('poly2'),
        inter_scaler2=art.get('inter_scaler2'), top_idx=art.get('top_idx'),
        poly3=art.get('poly3'), inter_scaler3=art.get('inter_scaler3'),
        top_idx3=art.get('top_idx3'), fit=False)
    return segment_softmax(X @ art['coef'], gs, n), s


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--out-dir',   default=MODEL_DIR)
    ap.add_argument('--data-file', default=None)
    ap.add_argument('--base-dir',  default=MODEL_DIR,
                    help='conditional_logit.pkl の読み込み元')
    args, _ = ap.parse_known_args()
    out_dir  = args.out_dir
    data_file = args.data_file or DATA_FILE
    os.makedirs(out_dir, exist_ok=True)

    # ── 既存モデル読み込み（scaler/poly2/top_idx 流用） ───────
    clogit_path = os.path.join(args.base_dir, 'conditional_logit.pkl')
    print(f"既存モデル読み込み: {clogit_path}")
    with open(clogit_path, 'rb') as f:
        base_pkg = pickle.load(f)

    # surface別artifactを作る（flat形式の場合も対応）
    if 'artifacts' in base_pkg:
        base_arts = base_pkg['artifacts']
    else:
        base_arts = {'芝': base_pkg, 'ダ': base_pkg}

    feat_cols = base_pkg.get('feat_cols') or list(base_arts.values())[0]['feat_cols']

    # ── データロード ─────────────────────────────────────────
    print(f"データ読み込み: {data_file}")
    df = pd.read_parquet(data_file)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df = df[df['開催'].notna()].copy()
    df = make_race_id(df)
    df = add_pace_features(df)
    df = add_new_features(df)
    df['surface'] = get_surface(df)
    df = df[df['surface'].isin(['芝', 'ダ'])].copy()

    for fc in feat_cols:
        if fc not in df.columns:
            df[fc] = np.nan

    trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] <  220101)]
    val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos = df[df['日付_num'] >= 230101].copy()
    print(f"trn={len(trn):,}  val={len(val):,}  oos={len(oos):,}")

    artifacts = {}

    for surf in ['芝', 'ダ']:
        print(f"\n{'='*55}")
        print(f"  surface={surf}")
        print(f"{'='*55}")

        art_base = base_arts.get(surf, base_arts.get('芝'))
        if 'feat_cols' not in art_base:
            art_base = dict(art_base)
            art_base['feat_cols'] = feat_cols

        trn_s = trn[trn['surface'] == surf].copy()
        val_s = val[val['surface'] == surf].sort_values('race_id').reset_index(drop=True)

        print(f"  trn={len(trn_s):,}行  val={len(val_s):,}行")

        beta = train_surface(trn_s, val_s, art_base)

        # val でraw確率を計算してisotonic fit
        art_tmp = dict(art_base)
        art_tmp['coef'] = beta

        raw_val, val_sorted = predict_raw(val_s, art_tmp)
        y_top2_val = (val_sorted['着順_num'] <= 2).astype(float).values
        y_top3_val = (val_sorted['着順_num'] <= 3).astype(float).values

        iso_top2 = fit_isotonic(raw_val, y_top2_val)
        iso_top3 = fit_isotonic(raw_val, y_top3_val)
        print(f"  isotonic_top2 fit: {len(iso_top2.X_thresholds_)}ノード")
        print(f"  isotonic_top3 fit: {len(iso_top3.X_thresholds_)}ノード")

        art_final = {
            'scaler':         art_base['scaler'],
            'poly2':          None,   # 交互作用なし（メモリ節約のため）
            'inter_scaler2':  None,
            'top_idx':        None,
            'poly3':          None,
            'inter_scaler3':  None,
            'top_idx3':       None,
            'coef':           beta,
            'feat_cols':      feat_cols,
            'isotonic_top2':  iso_top2,
            'isotonic_top3':  iso_top3,
        }
        artifacts[surf] = art_final

    # ── OOS評価 ────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("OOS評価 (2023+)")
    print(f"{'='*55}")

    top2_arr = np.zeros(len(oos))
    top3_arr = np.zeros(len(oos))
    oos = oos.sort_values('race_id').reset_index(drop=True)

    for surf in ['芝', 'ダ']:
        art = artifacts[surf]
        mask = (oos['surface'] == surf).values
        oos_s = oos[mask].sort_values('race_id').reset_index(drop=True)
        raw, oos_sorted = predict_raw(oos_s, art)
        top2_arr[oos[mask].index] = art['isotonic_top2'].predict(raw)
        top3_arr[oos[mask].index] = art['isotonic_top3'].predict(raw)

    oos['calib_top2'] = top2_arr
    oos['calib_top3'] = top3_arr
    oos['win'] = (oos['着順_num'] == 1).astype(int)
    oos['top2'] = (oos['着順_num'] <= 2).astype(int)
    oos['top3'] = (oos['着順_num'] <= 3).astype(int)
    oos['rank_top2'] = oos.groupby('race_id')['calib_top2'].rank(ascending=False, method='first')
    oos['rank_top3'] = oos.groupby('race_id')['calib_top3'].rank(ascending=False, method='first')
    oos['odds_num']  = pd.to_numeric(oos['単勝オッズ'], errors='coerce')

    bins   = [0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 1.01]
    labels = ['0-2%','2-5%','5-10%','10-15%','15-20%','20-30%','30-40%','40-60%','60%+']

    for k, col, target in [(2, 'calib_top2', 'top2'), (3, 'calib_top3', 'top3')]:
        oos['_bin'] = pd.cut(oos[col], bins=bins, labels=labels, right=False)
        tbl = oos.groupby('_bin', observed=True).agg(
            N=(target,'count'),
            モデル=(col,'mean'),
            実績=(target,'mean'),
        ).assign(乖離=lambda x: x['実績']-x['モデル'])
        print(f"\n── top-{k} キャリブレーション ──")
        print(tbl.to_string(float_format=lambda x: f'{x:.3f}'))

        # ランク1でのROI（複勝相当：上位k着以内が的中）
        rank_col = f'rank_top{k}'
        top1 = oos[oos[rank_col] == 1]
        hit  = top1[target]
        if len(top1) > 0:
            # 実際の複勝ROIは複勝オッズが必要なので的中率のみ表示
            print(f"  rank=1 的中率: {hit.mean():.3f}  ({len(top1)}レース)")

    # ── 保存 ──────────────────────────────────────────────────
    out_pkl = os.path.join(out_dir, 'final_model_placing.pkl')
    final_pkg = {
        'artifacts': artifacts,
        'feat_cols': feat_cols,
        'note': 'exploded logit top-3; isotonic_top2 / isotonic_top3',
    }
    with open(out_pkl, 'wb') as f:
        pickle.dump(final_pkg, f)
    print(f"\n保存完了: {out_pkl}")


if __name__ == '__main__':
    main()
