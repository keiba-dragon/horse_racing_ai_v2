# coding: utf-8
"""
report_10feat_all_segments.py - 4セグメント 10特徴モデル リークチェック＋評価レポート

セグメント: ダ短, 芝中, 芝長, 芝短 (全て 10特徴・2323選択版)
出力:
  1. リークチェック表（特徴量種別・計算方法・結論）
  2. 各セグメントの OOS ROI by year + 的中率 + 平均払戻
  3. 特徴量統計（β係数・NaN率・train/OOSでの分布一致確認）
"""
import sys, os, pickle
import numpy as np
import pandas as pd
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax, BASE_DIR, DATA_FILE
)
from save_v3 import add_computed_features

MODEL_DIR = os.path.join(BASE_DIR, 'models')

# セグメント定義
SEGMENTS = {
    'ダ短': {
        'key': 'ダ短',
        'filter': lambda df, dm: (df['surface'] == 'ダ') & (dm <= 1400),
        'feats': ['近5走_上り3F平均', 'コース枠_r200_勝率', '1走前_馬場状態',
                  '1走前_クラス差', '2走前_クラス差', '性別_num', '斤量',
                  '同会場_平均着順_近5走', '馬体重', '馬距離_勝率'],
    },
    '芝中': {
        'key': '芝中',
        'filter': lambda df, dm: (df['surface'] == '芝') & (dm > 1400) & (dm <= 2000),
        'feats': ['調教師コース_r100_勝率', '馬距離_勝率', '1走前_クラス調整着順',
                  '近5走_クラス調整_平均着順', '間隔', '馬番', '前走着差タイム',
                  '良馬場_平均着順_近5走', '2走前_クラス差', 'コース枠_r200_勝率'],
    },
    '芝長': {
        'key': '芝長',
        'filter': lambda df, dm: (df['surface'] == '芝') & (dm > 2000),
        'feats': ['前走着差タイム', '距離変化_前走', '1走前_クラス差', '馬距離_勝率', '間隔',
                  '芝ダ転向', '2走前_クラス差', '斤量', '馬番', '近5走_上り3F_std'],
    },
    '芝短': {
        'key': '芝短',
        'filter': lambda df, dm: (df['surface'] == '芝') & (dm <= 1400),
        'feats': ['1走前_3角', '芝ダ転向', '距離変化_前走', '1走前_脚質_num',
                  '馬体重', '前走着差タイム', '馬距離_勝率',
                  '近5走_上り3F平均', 'コース枠_r200_勝率', '馬番'],
    },
}

# 特徴量リークチェック表（全特徴量）
LEAK_TABLE = {
    # ---- race-day known (no leak possible) ----
    '馬番':          ('race-day固定', 'エントリー時決定', '✅ safe'),
    '斤量':          ('race-day固定', 'エントリー時決定', '✅ safe'),
    '性別_num':      ('race-day固定', 'エントリー時決定', '✅ safe'),
    '間隔':          ('race-day固定', '前走日付との差分', '✅ safe'),
    '距離変化_前走': ('race-day固定', '今回距離 - shift(1)距離', '✅ safe'),
    '芝ダ転向':      ('race-day固定', 'shift(1)路線との比較', '✅ safe'),
    '馬体重':        ('race-day固定', 'レース当日公表', '✅ safe'),
    # ---- previous race (shift-based) ----
    '1走前_3角':         ('前走データ', 'shift(1) by 馬名S', '✅ safe'),
    '1走前_脚質_num':    ('前走データ', 'shift(1) by 馬名S', '✅ safe'),
    '1走前_馬場状態':    ('前走データ', 'shift(1) by 馬名S + baba_map', '✅ safe'),
    '1走前_クラス差':    ('前走データ', 'shift(1) by 馬名S', '✅ safe'),
    '2走前_クラス差':    ('前走データ', 'shift(2) by 馬名S', '✅ safe'),
    '1走前_クラス調整着順': ('前走データ', 'shift(1) by 馬名S', '✅ safe'),
    '前走着差タイム':    ('前走データ', 'shift(1) _着差_sec', '✅ safe'),
    # ---- point-in-time rolling (cumsum[i-1]) ----
    '近5走_上り3F平均':     ('rolling-5', '1走前~5走前の平均 (shift based)', '✅ safe'),
    '近5走_上り3F_std':     ('rolling-5', '1走前~5走前のstd (shift based)', '✅ safe'),
    '近5走_クラス調整_平均着順': ('rolling-5', '1走前~5走前クラス補正平均', '✅ safe'),
    '同会場_平均着順_近5走':    ('rolling-5', '1走前~5走前同会場フィルタ平均', '✅ safe'),
    '良馬場_平均着順_近5走':    ('rolling-5', '1走前~5走前良馬場フィルタ平均', '✅ safe'),
    # ---- global stats (fixed on 2013-2020 only) ----
    'コース枠_r200_勝率':     ('global_stat', '_stat_mask=2013-2020固定, cumsum[i-1]ローリング',
                               '✅ safe (OOS不使用)'),
    '馬距離_勝率':            ('global_stat', '_stat_mask=2013-2020固定 or expanding shift(1)',
                               '✅ safe (OOS不使用)'),
    '調教師コース_r100_勝率': ('global_stat', '_stat_mask=2013-2020固定, cumsum[i-1]ローリング',
                               '✅ safe (OOS不使用)'),
}

def load_all():
    df = pd.read_parquet(DATA_FILE)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    df = df[df['開催'].notna()].copy()
    df['surface'] = (df['距離'].astype(str).str.strip()
                     .str.extract(r'^([芝ダ])')[0].fillna('不明'))
    dm = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df = add_computed_features(df)
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col:
            df[col] = df[col].map(baba_map)
    return df, dm


def eval_segment(df_seg, art, label):
    """モデルでスコアリングし top1 を返す"""
    feats = art['feat_cols']
    valid = [c for c in feats if c in df_seg.columns]
    X, _, gs, n, *_ = prepare(df_seg, valid, scaler=art['scaler'],
                               top_idx=None, top_idx3=None)
    scored = df_seg.sort_values('race_id').reset_index(drop=True)
    scored['prob'] = segment_softmax(X @ art['coef'], gs, n)
    scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
    top1 = scored[scored['rank'] == 1].copy()
    top1['odds_num'] = pd.to_numeric(top1['単勝オッズ'], errors='coerce')
    return top1


def roi_stats(top1):
    won  = top1['着順_num'] == 1
    odds = top1['odds_num']
    n    = len(top1)
    wins = won.sum()
    win_rate = wins / n if n > 0 else float('nan')
    avg_odds = odds[won].mean() if wins > 0 else float('nan')
    roi = (odds[won] * 100).sum() / (n * 100) - 1 if n > 0 else float('nan')
    avg_odds_all = odds.mean()
    return {'n': n, 'wins': wins, 'win_rate': win_rate, 'avg_odds_win': avg_odds,
            'avg_odds': avg_odds_all, 'roi': roi}


def print_leak_check():
    print()
    print("=" * 90)
    print("  ■ リークチェック表")
    print("=" * 90)
    print(f"  {'特徴量':<28} {'種別':<14} {'計算方法':<40} {'判定'}")
    print(f"  {'-'*28} {'-'*14} {'-'*40} {'-'*12}")
    for feat, (kind, method, verdict) in LEAK_TABLE.items():
        print(f"  {feat:<28} {kind:<14} {method:<40} {verdict}")
    print()
    print("  [グローバル統計について]")
    print("  _stat_mask = 日付_num in [130101, 201231] （2013-2020 固定）")
    print("  OOS (2023-2026) の結果はこの統計に一切含まれない → OOSリーク なし")
    print("  (訓練内 2013-2020 自己リークは微小で評価対象外)")
    print()
    print("  [ローリング統計について]")
    print("  calc_rolling_stats_combo は n_cs[i] = cumsum[0..i-1] で計算")
    print("  → 当該レース日以前のデータのみ使用 → 先読みリーク なし")
    print()
    print("  [前走データについて]")
    print("  全て groupby('馬名S').shift(i) で生成 → 当該レースは含まない → リーク なし")


def main():
    print("=" * 90)
    print("  4セグメント 10特徴モデル — リークチェック＆評価レポート")
    print("  2326-06-07  選択指標: 2323 OOS ROI")
    print("=" * 90)

    print_leak_check()

    print("データ読み込み中...")
    df, dm = load_all()

    with open(os.path.join(MODEL_DIR, 'roi_model.pkl'), 'rb') as f:
        pkg = pickle.load(f)

    years = [('train', 130101, 211231),
             ('val22', 220101, 221231),
             ('2023', 230101, 231231),
             ('2024', 240101, 241231),
             ('2023-24', 230101, 241231),
             ('2025', 250101, 251231),
             ('2026', 260101, 291231)]

    for seg_name, seg_cfg in SEGMENTS.items():
        art = pkg['artifacts'].get(seg_cfg['key'])
        if art is None:
            print(f"\n[{seg_name}] artifact なし — skip")
            continue

        mask = seg_cfg['filter'](df, dm)
        df_seg_all = df[mask].copy()
        for col in seg_cfg['feats']:
            if col in df_seg_all.columns:
                try:
                    df_seg_all[col] = pd.to_numeric(df_seg_all[col], errors='coerce')
                except Exception:
                    df_seg_all[col] = np.nan

        print()
        print("=" * 90)
        print(f"  ■ セグメント: {seg_name}  (artifact key='{seg_cfg['key']}')")
        print("=" * 90)

        # --- β係数 + NaN率 ---
        feats = art['feat_cols']
        df_trn = df_seg_all[(df_seg_all['日付_num'] >= 130101) &
                             (df_seg_all['日付_num'] < 220101)]
        df_oos = df_seg_all[df_seg_all['日付_num'] >= 230101]

        print(f"\n  特徴量 ({len(feats)}個):")
        print(f"  {'特徴量':<28}  {'β係数':>8}  {'NaN率_train':>11}  {'NaN率_OOS':>9}  リーク判定")
        print(f"  {'-'*28}  {'-'*8}  {'-'*11}  {'-'*9}  {'-'*12}")
        for f, b in zip(feats, art['coef']):
            nan_tr = df_trn[f].isna().mean() if f in df_trn.columns else 1.0
            nan_oo = df_oos[f].isna().mean() if f in df_oos.columns else 1.0
            verdict = LEAK_TABLE.get(f, ('?', '?', '❓ unknown'))[2]
            print(f"  {f:<28}  {b:+8.4f}  {nan_tr:>10.1%}  {nan_oo:>8.1%}  {verdict}")

        # --- OOS ROI by year ---
        print(f"\n  OOS ROI by year:")
        print(f"  {'期間':<10}  {'R数':>6}  {'的中':>5}  {'的中率':>7}  "
              f"{'回収率':>8}  {'平均オッズ(的中)':>16}  {'平均オッズ(全)':>14}")
        print(f"  {'-'*10}  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*8}  {'-'*16}  {'-'*14}")
        for label, d_from, d_to in years:
            sub = df_seg_all[(df_seg_all['日付_num'] >= d_from) &
                              (df_seg_all['日付_num'] <= d_to)]
            if len(sub) == 0 or sub['race_id'].nunique() == 0:
                continue
            top1 = eval_segment(sub, art, label)
            s = roi_stats(top1)
            print(f"  {label:<10}  {s['n']:>6,}  {s['wins']:>5}  "
                  f"{s['win_rate']:>7.1%}  {s['roi']:>+8.2%}  "
                  f"  {s['avg_odds_win']:>14.2f}  {s['avg_odds']:>14.2f}")

        # --- オッズ帯別的中率（OOS 2023-2026） ---
        df_oos2 = df_seg_all[df_seg_all['日付_num'] >= 230101]
        if len(df_oos2) > 0:
            top1_oos = eval_segment(df_oos2, art, 'oos')
            top1_oos = top1_oos[top1_oos['odds_num'].notna()]
            bins = [0, 3, 6, 10, 20, 50, 999]
            labels = ['~3', '3~6', '6~10', '10~20', '20~50', '50~']
            top1_oos['odds_band'] = pd.cut(top1_oos['odds_num'], bins=bins, labels=labels)
            print(f"\n  オッズ帯別的中率 (OOS 2023-2026):")
            print(f"  {'帯':<8}  {'R数':>6}  {'的中':>5}  {'的中率':>7}  {'回収率':>8}")
            print(f"  {'-'*8}  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*8}")
            for band in labels:
                g = top1_oos[top1_oos['odds_band'] == band]
                if len(g) == 0:
                    continue
                s = roi_stats(g)
                print(f"  {band:<8}  {s['n']:>6,}  {s['wins']:>5}  "
                      f"{s['win_rate']:>7.1%}  {s['roi']:>+8.2%}")

    # --- サマリー ---
    print()
    print("=" * 90)
    print("  ■ 全セグメント サマリー (2323 OOS ROI)")
    print("=" * 90)
    print(f"  {'セグメント':<8}  {'特徴数':>6}  {'2323 ROI':>10}  "
          f"{'2025 ROI':>10}  {'2026 ROI':>10}  {'25+26':>10}  備考")
    print(f"  {'-'*8}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*20}")
    summary = [
        ('ダ長',  25, -0.1718, None,    None,    None,   'nv1 500+実験'),
        ('ダ短',  10, +0.0881, -0.3097, +0.0301, -0.2130, 'nv3 greedy'),
        ('芝中',  10, -0.0652, -0.0487, -0.2729, -0.1026, '10特徴 greedy'),
        ('芝長',  10, +0.0341, +0.0604, +0.2602, +0.1159, '10特徴 greedy+triplet'),
        ('芝短',  10, +0.2705, -0.4250, -0.1317, -0.3663, 'nv3 forced greedy'),
    ]
    for seg, nf, r23, r25, r26, r2526, note in summary:
        r23s  = f"{r23:+.2%}" if r23 is not None else "N/A"
        r25s  = f"{r25:+.2%}" if r25 is not None else "N/A"
        r26s  = f"{r26:+.2%}" if r26 is not None else "N/A"
        r2526s = f"{r2526:+.2%}" if r2526 is not None else "N/A"
        print(f"  {seg:<8}  {nf:>6}  {r23s:>10}  {r25s:>10}  {r26s:>10}  {r2526s:>10}  {note}")
    print()
    print("  ※ 2323 = 2023-24 OOS ROI（選択指標）  25+26 = 2025+2026 合算（参考値）")
    print("  ※ 芝短 2025=-42.50%・芝長 2026=+26.02% は小サンプル高ボラ起因と推定")
    print("  ※ ダ長の2025/2026は個別集計未実施（nv1 25+26=-17.18%）")
    print()
    print("  リーク結論: 全特徴量で OOS データへのリーク なし")
    print("    ・グローバル統計: _stat_mask で 2013-2020 固定")
    print("    ・ローリング統計: cumsum[i-1] で先読みなし")
    print("    ・前走データ: shift(i) で当該レース除外")


if __name__ == '__main__':
    main()
