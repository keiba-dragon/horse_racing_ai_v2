# coding: utf-8
"""
加点減点モデル v2 — 全特徴量（約390列）を自動で二値条件化する版。

考え方:
  各特徴量について、レース内パーセンタイル順位で上位25%/下位25%を
  「ある/なし」の二値条件とする（1特徴量につき2条件 → 合計約780条件）。
  どちら側（高い方 or 低い方）が勝ちに繋がるかは、学習期間(2013-2022)の
  実際の勝率差から自動判定する（リーク防止のためOOS期間は使わない）。
  条件が成立すれば+1点、不成立なら0点（マイナスにはしない = 「ある/なし」方式）。
  合計点が最も高い馬をAI1位として、hitrate_modelと同じOOS期間で
  的中率・ROIを比較する。

usage:
  python src/score_points_model_v2.py
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from save_v3 import add_computed_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')

ORDER = ['芝長', '芝中', '芝短', 'ダ長', 'ダ短']
OOS = {'2324': (230101, 250101), '2025': (250101, 260101), '2026': (260101, 300101)}
PERIOD_LABEL = {'2324': '2023-24', '2025': '2025', '2026': '2026'}

LEAK_EXACT = {
    '着順', '着順_num', '着差', '走破タイム', '走破タイム_sec', '上り3F',
    'PCI', 'PCI3', 'Ave-3F', 'RPCI', '平均速度', '-3F平均速度',
    '上り3F平均速度', '上3F地点差', '単勝オッズ', '賞金', '1角', '2角', '3角', '4角',
    '馬体重変化',
    # 今走オッズ・人気・払戻・当該レースのタイム指数系（CLAUDE.mdルール + 事後にしか分からない値）
    '人気', '単勝配当', '複勝配当', 'タイム指数', '上り3F_指数',
}
INT_KEEP = {'馬番', '季節', '近走連続入着数', '内外枠'}

MIN_TRAIN_N = 2000       # 学習期間での最低サンプル数
MIN_WINRATE_GAP = 0.01   # 上位25%と下位25%の勝率差の最低ライン（ノイズ除去）


def _seg_key(surf, dist_m):
    if pd.isna(dist_m):
        return None
    surf = str(surf).strip()
    if surf == '芝':
        return '芝短' if dist_m <= 1400 else ('芝中' if dist_m <= 2000 else '芝長')
    elif surf == 'ダ':
        return 'ダ短' if dist_m <= 1400 else 'ダ長'
    return None


def get_candidate_features():
    schema = pq.read_schema(DATA_FILE)
    cands = []
    for name, dtype in zip(schema.names, schema.types):
        if name in LEAK_EXACT:
            continue
        if str(dtype) == 'double' or name in INT_KEEP:
            cands.append(name)
    return cands


def main():
    print('データ読み込み中...')
    candidates = get_candidate_features()
    print(f'候補特徴量: {len(candidates)}個')

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
    df['_win'] = (df['着順_num'] == 1).astype(int)

    candidates = [c for c in candidates if c in df.columns]
    print(f'データ内に存在する候補: {len(candidates)}個')

    print('学習期間(2013-2022)でレース内パーセンタイルを計算中...')
    df_train = df[df['日付_num'] < 230101].copy()
    for c in candidates:
        df_train[c] = pd.to_numeric(df_train[c], errors='coerce')
    pct_train = df_train.groupby('race_id')[candidates].rank(pct=True, na_option='keep')

    print('各条件の方向・有効性を判定中...')
    rules = []  # (feature, side, direction) side='top'/'bottom', direction=+1 (採用)
    for c in candidates:
        p = pct_train[c]
        top_mask = p >= 0.75
        bot_mask = p <= 0.25
        n_top, n_bot = top_mask.sum(), bot_mask.sum()
        if n_top < MIN_TRAIN_N or n_bot < MIN_TRAIN_N:
            continue
        wr_top = df_train.loc[top_mask, '_win'].mean()
        wr_bot = df_train.loc[bot_mask, '_win'].mean()
        if wr_top - wr_bot >= MIN_WINRATE_GAP:
            rules.append((c, 'top', wr_top - wr_bot))
        elif wr_bot - wr_top >= MIN_WINRATE_GAP:
            rules.append((c, 'bottom', wr_bot - wr_top))

    rules.sort(key=lambda x: -x[2])
    print(f'採用条件数: {len(rules)}個（上位25%条件: {sum(1 for r in rules if r[1]=="top")} / '
          f'下位25%条件: {sum(1 for r in rules if r[1]=="bottom")}）')
    print('上位10条件（勝率差が大きい順）:')
    for c, side, gap in rules[:10]:
        print(f'  {c} ({"高い方" if side=="top" else "低い方"}が良い)  勝率差={gap*100:.1f}pt')

    del df_train, pct_train

    print('\nOOSデータへ適用中...')
    df_oos = df[df['日付_num'] >= 230101].copy()
    feat_used = sorted(set(r[0] for r in rules))
    for c in feat_used:
        df_oos[c] = pd.to_numeric(df_oos[c], errors='coerce')
    pct_oos = df_oos.groupby('race_id')[feat_used].rank(pct=True, na_option='keep')

    print('加点集計中...')
    total = pd.Series(0, index=df_oos.index, dtype=int)
    for c, side, _ in rules:
        p = pct_oos[c]
        cond = (p >= 0.75) if side == 'top' else (p <= 0.25)
        total += cond.fillna(False).astype(int)
    df_oos['_score'] = total
    df_oos['_rank'] = df_oos.groupby('race_id')['_score'].rank(ascending=False, method='first').astype(int)

    print()
    print(f'=== 加点減点モデル v2（自動二値化・条件数{len(rules)}個） — AI1位の的中率・ROI ===')
    print(f'{"セグメント":6s} {"期間":8s} {"頭数":>6s} {"勝率":>7s} {"回収率":>8s}')
    rank1 = df_oos[df_oos['_rank'] == 1]
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
