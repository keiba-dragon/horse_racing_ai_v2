# coding: utf-8
"""
Placing model (top-2 / top-3) calibration.

勝率モデル (roi_model.pkl) の raw softmax 確率を入力として
isotonic regression で P(top-2) / P(top-3) をキャリブレーション。

P(top2) >= P(win) が保証される。係数の再学習なし。

保存: models/final_model_placing.pkl
  artifacts['芝'] = {
      isotonic_top2,   # raw_prob → P(top-2)
      isotonic_top3,   # raw_prob → P(top-3)
  }
"""
import os, sys, pickle, argparse
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from save_lambdarank_pace import add_pace_features
from save_conditional_logit import (
    add_new_features, segment_softmax, prepare,
    get_group_starts,
)

DATA_FILE  = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')


def make_race_id(df):
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    return df


def get_surface(df):
    return df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')


def compute_raw_probs(df_s, art):
    """勝率モデルの raw softmax 確率を計算（元の feat_cols + poly2/top_idx を使用）"""
    s = df_s.sort_values('race_id').reset_index(drop=True)
    X, _, gs, n, *_ = prepare(
        s, art['feat_cols'],
        scaler=art['scaler'],
        poly2=art.get('poly2'), inter_scaler2=art.get('inter_scaler2'), top_idx=art.get('top_idx'),
        poly3=art.get('poly3'), inter_scaler3=art.get('inter_scaler3'), top_idx3=art.get('top_idx3'),
        fit=False)
    raw = segment_softmax(X @ art['coef'], gs, n)
    return raw, s


def fit_isotonic(raw_probs, labels):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw_probs, labels)
    return iso


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--out-dir',   default=MODEL_DIR)
    ap.add_argument('--data-file', default=None)
    ap.add_argument('--base-dir',  default=MODEL_DIR,
                    help='roi_model.pkl の読み込み元')
    args, _ = ap.parse_known_args()
    out_dir   = args.out_dir
    data_file = args.data_file or DATA_FILE
    os.makedirs(out_dir, exist_ok=True)

    # ── 勝率モデル読み込み（roi_model.pkl を使用） ──────────
    model_path = os.path.join(args.base_dir, 'roi_model.pkl')
    print(f"勝率モデル読み込み: {model_path}")
    with open(model_path, 'rb') as f:
        main_pkg = pickle.load(f)

    if 'artifacts' in main_pkg:
        arts = main_pkg['artifacts']
    else:
        arts = {'芝': main_pkg, 'ダ': main_pkg}

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

    feat_cols = list(arts.values())[0]['feat_cols']
    for fc in feat_cols:
        if fc not in df.columns:
            df[fc] = np.nan

    # val: isotonic fit 用（2022年のみ）
    val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos = df[df['日付_num'] >= 230101].copy()
    print(f"val={len(val):,}  oos={len(oos):,}")

    artifacts = {}

    for surf in ['芝', 'ダ']:
        print(f"\n{'='*55}")
        print(f"  surface={surf}")
        print(f"{'='*55}")

        art = arts.get(surf, arts.get('芝'))
        val_s = val[val['surface'] == surf].sort_values('race_id').reset_index(drop=True)
        print(f"  val={len(val_s):,}行")

        # val で raw確率 → top-2 / top-3 isotonic fit
        raw_val, val_sorted = compute_raw_probs(val_s, art)
        y_top2_val = (val_sorted['着順_num'] <= 2).astype(float).values
        y_top3_val = (val_sorted['着順_num'] <= 3).astype(float).values

        iso_top2 = fit_isotonic(raw_val, y_top2_val)
        iso_top3 = fit_isotonic(raw_val, y_top3_val)
        print(f"  isotonic_top2 fit: {len(iso_top2.X_thresholds_)}ノード")
        print(f"  isotonic_top3 fit: {len(iso_top3.X_thresholds_)}ノード")

        # キャリブ確認（val）
        calib2_val = iso_top2.predict(raw_val)
        calib3_val = iso_top3.predict(raw_val)
        print(f"  val top2 calib mean: {calib2_val.mean():.3f}  actual: {y_top2_val.mean():.3f}")
        print(f"  val top3 calib mean: {calib3_val.mean():.3f}  actual: {y_top3_val.mean():.3f}")

        artifacts[surf] = {
            'isotonic_top2': iso_top2,
            'isotonic_top3': iso_top3,
        }

    # ── OOS評価 ────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("OOS評価 (2023+)")
    print(f"{'='*55}")

    top2_arr = np.zeros(len(oos))
    top3_arr = np.zeros(len(oos))
    oos = oos.sort_values('race_id').reset_index(drop=True)

    for surf in ['芝', 'ダ']:
        art = arts.get(surf, arts.get('芝'))
        piso = artifacts[surf]
        mask = (oos['surface'] == surf).values
        oos_s = oos[mask].sort_values('race_id').reset_index(drop=True)
        raw, oos_sorted = compute_raw_probs(oos_s, art)
        idx_in_oos = np.where(mask)[0]
        top2_arr[idx_in_oos] = piso['isotonic_top2'].predict(raw)
        top3_arr[idx_in_oos] = piso['isotonic_top3'].predict(raw)

    oos['calib_top2'] = top2_arr
    oos['calib_top3'] = top3_arr
    oos['top2'] = (oos['着順_num'] <= 2).astype(int)
    oos['top3'] = (oos['着順_num'] <= 3).astype(int)
    oos['rank_top2'] = oos.groupby('race_id')['calib_top2'].rank(ascending=False, method='first')
    oos['rank_top3'] = oos.groupby('race_id')['calib_top3'].rank(ascending=False, method='first')

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

        rank_col = f'rank_top{k}'
        top1 = oos[oos[rank_col] == 1]
        hit  = top1[target]
        if len(top1) > 0:
            print(f"  rank=1 的中率: {hit.mean():.3f}  ({len(top1)}レース)")

    # ── 保存 ──────────────────────────────────────────────────
    out_pkl = os.path.join(out_dir, 'final_model_placing.pkl')
    final_pkg = {
        'artifacts': artifacts,
        'note': 'win-model raw prob → isotonic top2/top3 calibration',
    }
    with open(out_pkl, 'wb') as f:
        pickle.dump(final_pkg, f)
    print(f"\n保存完了: {out_pkl}")


if __name__ == '__main__':
    main()
