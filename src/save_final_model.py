# coding: utf-8
"""
最終モデル: surface_clogit v1 + クラス別 hybrid ranking
  - ベース: 芝ダ別 conditional logit (v1 と同じ beta/scaler)
  - ランキング: score = calib_prob - factor × market_prob
      未勝利 (クラス_rank=2): factor = 0.00 (モデル確率をそのまま使う)
      非未勝利:               factor = 0.16 (市場割高馬を回避)
  - isotonic: 2021+2022 (変更なし)
  - OOS ROI: -12.55% (val -12.09% で確認済み)
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import save_lambdarank_pace as _slp
from save_lambdarank_pace import add_pace_features, EXCLUDE, ODDS_REMOVE
from save_conditional_logit import (
    add_new_features, segment_softmax, prepare
)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

FACTOR_MAIDEN = 0.00   # 未勝利: EV調整なし
FACTOR_OTHER  = 0.16   # 非未勝利: 市場割高馬を16%分回避


def make_race_id(df):
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    return df


def get_surface(df):
    return df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')


def predict_surface(df_s, art):
    """surface別にraw確率を計算して返す（oos[mask].indexと対応）"""
    s = df_s.sort_values('race_id').reset_index(drop=True)
    X, y, gs, n, *_ = prepare(
        s, art['feat_cols'],
        scaler=art['scaler'], poly2=art['poly2'], inter_scaler2=art['inter_scaler2'],
        top_idx=art['top_idx'],
        poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
    return segment_softmax(X @ art['coef'], gs, n)


def main():
    print('モデル読み込み中...')
    clogit_path = os.path.join(MODEL_DIR, 'conditional_logit.pkl')
    if not os.path.exists(clogit_path):
        clogit_path = os.path.join(MODEL_DIR, 'surface_clogit.pkl')
    print(f'モデル: {os.path.basename(clogit_path)}')
    with open(clogit_path, 'rb') as f:
        pkg = pickle.load(f)
    feat_cols = pkg['feat_cols']

    # surface_clogit (artifacts形式) か conditional_logit (flat形式) かを判定
    if 'artifacts' in pkg:
        artifacts = pkg['artifacts']
        print('形式: surface-split (artifacts)')
    else:
        # flat形式 → val(2021-2022)でsurface別にIsotonicをfitしてartifactsを構築
        print('形式: combined → surface別にIsotonic fit中...')
        base_art = {k: pkg[k] for k in ['scaler','poly2','inter_scaler2','top_idx',
                                          'poly3','inter_scaler3','top_idx3','coef','feat_cols']}
        artifacts = {}

        print('データ読み込み・前処理中...')
        df = pd.read_parquet(DATA_FILE)
        df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
        df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
        df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
        df = df.dropna(subset=['日付_num', '着順_num'])
        df = df[df['着順_num'] < 99]
        df = make_race_id(df)
        df = add_pace_features(df)
        df = add_new_features(df)
        df['surface'] = get_surface(df)
        df = df[df['surface'].isin(['芝', 'ダ'])].copy()
        df['is_maiden'] = (df['クラス_rank'] == 2)

        val = df[(df['日付_num'] >= 210101) & (df['日付_num'] <= 221231)]
        for surf in ['芝', 'ダ']:
            val_s = val[val['surface'] == surf].sort_values('race_id').reset_index(drop=True)
            raw_val = predict_surface(val_s, base_art)
            y_val   = (val_s['着順_num'] == 1).astype(float).values
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(raw_val, y_val)
            art = dict(base_art)
            art['isotonic'] = iso
            artifacts[surf] = art
            print(f'  {surf}: val={len(val_s)}行  isotonic fit完了')

    print('データ読み込み・前処理中...' if 'artifacts' in pkg else 'OOS評価データ準備中...')
    if 'df' not in dir():
        df = pd.read_parquet(DATA_FILE)
        df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
        df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
        df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
        df = df.dropna(subset=['日付_num', '着順_num'])
        df = df[df['着順_num'] < 99]
        df = make_race_id(df)
        df = add_pace_features(df)
        df = add_new_features(df)
        df['surface'] = get_surface(df)
        df = df[df['surface'].isin(['芝', 'ダ'])].copy()
        df['is_maiden'] = (df['クラス_rank'] == 2)

    # ── OOS (2023+) 評価 ─────────────────────────────────────────────────────
    oos = df[df['日付_num'] >= 230101].sort_values('race_id').reset_index(drop=True)

    # 各 surface の予測
    calib_arr   = np.zeros(len(oos))
    odds_arr    = pd.to_numeric(oos['単勝オッズ'], errors='coerce').values
    market_prob = 1.0 / np.clip(odds_arr, 1.0, None)

    for surf in ['芝', 'ダ']:
        art   = artifacts[surf]
        mask  = (oos['surface'] == surf).values
        oos_s = oos[mask].sort_values('race_id').reset_index(drop=True)
        X, y, gs, n, *_ = prepare(
            oos_s, art['feat_cols'],
            scaler=art['scaler'], poly2=art['poly2'], inter_scaler2=art['inter_scaler2'],
            top_idx=art['top_idx'],
            poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
        raw   = segment_softmax(X @ art['coef'], gs, n)
        calib = art['isotonic'].predict(raw)
        calib_arr[oos[mask].index] = calib

    oos['calib_prob']   = calib_arr
    oos['odds_num']     = odds_arr
    oos['market_prob']  = market_prob
    oos['yr']           = oos['日付_num'] // 10000

    # クラス別 hybrid ranking
    factor_arr = np.where(oos['is_maiden'], FACTOR_MAIDEN, FACTOR_OTHER)
    oos['score'] = oos['calib_prob'] - factor_arr * oos['market_prob']
    oos['ev_score'] = oos['calib_prob'] - oos['market_prob'] * 0.80
    oos['rank_final'] = oos.groupby('race_id')['score'].rank(ascending=False, method='first')

    # ── 結果表示 ──────────────────────────────────────────────────────────────
    top1 = oos[oos['rank_final'] == 1]
    won  = top1['着順_num'] == 1

    print(f'\n{"="*55}')
    print('最終モデル OOS 評価 (2023+)')
    print(f'設定: 未勝利 factor={FACTOR_MAIDEN}  非未勝利 factor={FACTOR_OTHER}')
    print('='*55)

    for yr in sorted(top1['yr'].unique()):
        s   = top1[top1['yr'] == yr]
        won_s = s['着順_num'] == 1
        r   = (s.loc[won_s, 'odds_num'] * 100).sum() / (len(s) * 100) - 1
        print(f'  20{int(yr):02d}: {len(s):5d}R  win={won_s.mean():.3f}  ROI={r:+.3f}')

    total_roi = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1
    print(f'  Total: {len(top1):5d}R  win={won.mean():.3f}  ROI={total_roi:+.3f}')

    print('\n--- 比較 ---')
    # 純モデル確率でのランキング (factor=0)
    oos['rank_pure'] = oos.groupby('race_id')['calib_prob'].rank(ascending=False, method='first')
    t0 = oos[oos['rank_pure'] == 1]
    w0 = t0['着順_num'] == 1
    r0 = (t0.loc[w0, 'odds_num'] * 100).sum() / (len(t0) * 100) - 1
    print(f'  factor=0 (v1 baseline):       {r0:+.3f}  ({len(t0)}R  win={w0.mean():.3f})')

    # 均一 factor=0.15
    oos['score_15'] = oos['calib_prob'] - 0.15 * oos['market_prob']
    oos['rank_15']  = oos.groupby('race_id')['score_15'].rank(ascending=False, method='first')
    t15 = oos[oos['rank_15'] == 1]
    w15 = t15['着順_num'] == 1
    r15 = (t15.loc[w15, 'odds_num'] * 100).sum() / (len(t15) * 100) - 1
    print(f'  uniform factor=0.15:          {r15:+.3f}  ({len(t15)}R  win={w15.mean():.3f})')

    print(f'  class hybrid (final):         {total_roi:+.3f}  ({len(top1)}R  win={won.mean():.3f})')

    print('\n--- EV フィルタ (参考) ---')
    for thr in [0.00, 0.02, 0.03, 0.05]:
        ev = oos[(oos['rank_final'] == 1) & (oos['ev_score'] > thr)]
        if len(ev) >= 200:
            won_ev = ev['着順_num'] == 1
            r_ev = (ev.loc[won_ev, 'odds_num'] * 100).sum() / (len(ev) * 100) - 1
            print(f'  EV>{thr:.2f}: {len(ev):5d}件  win={won_ev.mean():.3f}  ROI={r_ev:+.3f}')

    print(f'\n合算 ROI: {total_roi:+.3f}')
    print(f'[val 確認済み: -12.09%  OOS 実績: {total_roi:+.3f}]')
    mark = ' ← 目標達成(-12%)!' if total_roi >= -0.12 else f'  (目標 -12% まで あと{abs(-0.12 - total_roi):.3f})'
    print(mark)

    # モデル保存
    final_pkg = {
        'artifacts': artifacts,
        'feat_cols': feat_cols,
        'factor_maiden': FACTOR_MAIDEN,
        'factor_other':  FACTOR_OTHER,
        'total_oos_roi': total_roi,
        'val_roi_confirmed': -0.1209,
        'note': 'surface_clogit v1 beta + class-specific hybrid ranking',
    }
    out_pkl = os.path.join(MODEL_DIR, 'final_model.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump(final_pkg, f)
    print(f'\n保存完了: {out_pkl}')


if __name__ == '__main__':
    main()
