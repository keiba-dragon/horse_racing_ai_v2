# coding: utf-8
"""
加点減点モデル v1 — clogit(hitrate_model)とは別の、シンプルな点数積み上げ方式の予想手法。

考え方:
  hitrate_model.pkl と同じ特徴量セット・同じセグメント定義を使うが、
  学習係数の代わりに「レース内パーセンタイル順位」を5段階の点数に変換して
  単純合計する（+2/+1/0/-1/-2）。係数が負の特徴量は加減点を反転する。
  合計点が最も高い馬をAI1位として、hitrate_modelと同じOOS期間で
  的中率・ROIを比較する。

usage:
  python src/score_points_model.py
"""
import os, sys, pickle, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from save_v3 import add_computed_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')

with open(os.path.join(BASE_DIR, 'models', 'hitrate_model.pkl'), 'rb') as f:
    MODEL = pickle.load(f)

ORDER = ['芝長', '芝中', '芝短', 'ダ長', 'ダ短']
OOS = {'2324': (230101, 250101), '2025': (250101, 260101), '2026': (260101, 300101)}
PERIOD_LABEL = {'2324': '2023-24', '2025': '2025', '2026': '2026'}


def _seg_key(surf, dist_m):
    if pd.isna(dist_m):
        return None
    surf = str(surf).strip()
    if surf == '芝':
        return '芝短' if dist_m <= 1400 else ('芝中' if dist_m <= 2000 else '芝長')
    elif surf == 'ダ':
        return 'ダ短' if dist_m <= 1400 else 'ダ長'
    return None


def _points_from_pct(pct, invert):
    """0-1のレース内パーセンタイル順位 → 5段階点数（+2〜-2）"""
    pts = np.select(
        [pct >= 0.8, pct >= 0.6, pct >= 0.4, pct >= 0.2],
        [2, 1, 0, -1],
        default=-2,
    )
    return -pts if invert else pts


def _score_seg_points(grp, art):
    """セグメント分の加点減点合計スコアを計算する"""
    feat_cols = [f for f in art['feat_cols'] if not f.endswith('_isnan')]
    coef_map = dict(zip(art['feat_cols'], art['coef']))
    total = pd.Series(0.0, index=grp.index)
    detail = {}
    for f in feat_cols:
        if f not in grp.columns:
            continue
        vals = pd.to_numeric(grp[f], errors='coerce')
        pct = vals.groupby(grp['race_id']).rank(pct=True, na_option='keep')
        invert = coef_map.get(f, 0) < 0
        pts = _points_from_pct(pct.fillna(0.4), invert)  # NaNは中立(0点)寄りの0.4に
        pts = np.where(pct.isna(), 0, pts)  # 欠損は加減点なし
        total += pts
        detail[f] = pts
    return total


def main():
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
    df['_surf'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
    df['_dist_m'] = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
    df = df[df['クラス_rank'] != 1.0].copy()
    df['seg_key'] = [_seg_key(s, d) for s, d in zip(df['_surf'], df['_dist_m'])]
    df = df[df['seg_key'].notna()].copy()
    df['dist_m'] = df['_dist_m']
    df = add_computed_features(df)
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col and col != '馬場状態':
            df[col] = df[col].map(baba_map)
    df['単勝オッズ'] = pd.to_numeric(df['単勝オッズ'], errors='coerce')

    df_oos = df[df['日付_num'] >= 230101].copy()

    print('加点減点スコア計算中...')
    scored = []
    for seg, grp in df_oos.groupby('seg_key'):
        if seg not in MODEL:
            continue
        g = grp.copy()
        g['_score'] = _score_seg_points(g, MODEL[seg])
        scored.append(g)
    df_sc = pd.concat(scored).reset_index(drop=True)
    df_sc['_rank'] = df_sc.groupby('race_id')['_score'].rank(ascending=False, method='first').astype(int)

    print()
    print('=== 加点減点モデル v1 — AI1位（合計スコア最高位）の的中率・ROI ===')
    print(f'{"セグメント":6s} {"期間":8s} {"頭数":>6s} {"勝率":>7s} {"回収率":>8s}')
    rank1 = df_sc[df_sc['_rank'] == 1]
    for seg in ORDER:
        sub_seg = rank1[rank1['seg_key'] == seg]
        for pid, (d0, d1) in OOS.items():
            sub = sub_seg[(sub_seg['日付_num'] >= d0) & (sub_seg['日付_num'] < d1)]
            if len(sub) == 0:
                continue
            hitrate = (sub['着順_num'] == 1).mean() * 100
            hits = sub[sub['着順_num'] == 1]
            ret = hits['単勝オッズ'].sum()
            n = sub['単勝オッズ'].notna().sum()
            roi = (ret - n) / n * 100 if n > 0 else float('nan')
            print(f'{seg:6s} {PERIOD_LABEL[pid]:8s} {len(sub):6d} {hitrate:6.1f}% {roi:+7.1f}%')

    print()
    print('=== 全セグメント合算 ===')
    for pid, (d0, d1) in OOS.items():
        sub = rank1[(rank1['日付_num'] >= d0) & (rank1['日付_num'] < d1)]
        if len(sub) == 0:
            continue
        hitrate = (sub['着順_num'] == 1).mean() * 100
        hits = sub[sub['着順_num'] == 1]
        ret = hits['単勝オッズ'].sum()
        n = sub['単勝オッズ'].notna().sum()
        roi = (ret - n) / n * 100 if n > 0 else float('nan')
        print(f'{PERIOD_LABEL[pid]:8s} n={len(sub):5d} 勝率={hitrate:5.1f}% 回収率={roi:+.1f}%')


if __name__ == '__main__':
    main()
