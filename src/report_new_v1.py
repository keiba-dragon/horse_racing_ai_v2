# coding: utf-8
"""
report_new_v1.py - New v1 (BASE_25) ダート中長距離 モデルレポート
"""
import sys, os, pickle
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax, BASE_DIR, DATA_FILE
)
from save_v3 import add_computed_features, calc_roi

MODEL_DIR = os.path.join(BASE_DIR, 'models')


def load_segment():
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
    df = df[(df['surface'] == 'ダ') & (dm > 1400)].copy()
    df['dist_m'] = dm[df.index]
    if 'クラス_rank' in df.columns:
        df = df[df['クラス_rank'] != 1.0].copy()
    df = add_computed_features(df)
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col:
            df[col] = df[col].map(baba_map)
    df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
    return df


def roi_from_top1(top1):
    won  = top1['着順_num'] == 1
    odds = pd.to_numeric(top1['単勝オッズ'], errors='coerce')
    return (odds[won] * 100).sum() / (len(top1) * 100) - 1, won.mean(), won.sum()


def hr(char='─', n=62):
    print(char * n)


def main():
    print()
    hr('═')
    print('  New v1 (BASE_25) ダート中長距離 モデルレポート')
    print('  生成日: 2026-06-06')
    hr('═')

    # ── モデル読み込み ────────────────────────────────────────────────────────
    final_pkl = os.path.join(MODEL_DIR, 'roi_model.pkl')
    with open(final_pkl, 'rb') as f:
        pkg = pickle.load(f)
    art    = pkg['artifacts']['ダ']
    beta   = art['coef']
    scaler = art['scaler']
    feats  = art['feat_cols']
    iso    = art['isotonic']

    print(f'\n【モデル概要】')
    print(f'  バージョン    : {pkg.get("version", "?")}')
    print(f'  特徴量数      : {len(feats)}個')
    print(f'  L2 正則化     : 0.006')
    print(f'  訓練期間      : 2013-2021')
    print(f'  バリデーション: 2022')
    print(f'  対象セグメント: ダート>1400m (新馬除外)')
    print(f'  ベースライン  : 21F old = -19.70%  (320特徴+poly2交互作用)')

    # ── データ読み込み ────────────────────────────────────────────────────────
    print('\nデータ読み込み中...')
    df = load_segment()

    df_trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos    = df[df['日付_num'] >= 230101].copy()

    print(f'  train: {len(df_trn):,}行  val: {len(df_val):,}行  OOS: {len(oos):,}行')

    # ── 予測 (raw softmax — 実験と同一条件) ─────────────────────────────────
    valid_p = [c for c in feats if c in oos.columns]
    X_p, _, gs_p, n_p, *_ = prepare(oos, valid_p, scaler=scaler,
                                      top_idx=None, top_idx3=None)
    oos_s = oos.sort_values('race_id').reset_index(drop=True)
    raw_prob            = segment_softmax(X_p @ beta, gs_p, n_p)
    oos_s['raw_prob']   = raw_prob
    oos_s['calib_prob'] = iso.predict(raw_prob)
    oos_s['odds_num']   = pd.to_numeric(oos_s['単勝オッズ'], errors='coerce')
    oos_s['mkt_prob']   = 1.0 / oos_s['odds_num'].clip(lower=1.0)
    # ランキング: 実験と同じ raw prob
    oos_s['rank'] = oos_s.groupby('race_id')['raw_prob'].rank(ascending=False, method='first')
    oos_s['yr']   = (oos_s['日付_num'] // 10000).astype(int)
    oos_s['dist_m'] = oos_s['dist_m'].astype(int)
    oos_s['class_lbl'] = oos_s['クラス_rank'].map({
        2.0: '未勝利', 3.0: '1勝クラス', 4.0: '2勝クラス',
    }).fillna('OP以上')

    top1 = oos_s[oos_s['rank'] == 1].copy()

    # ══════════════════════════════════════════════════════════════════════════
    print()
    hr('═')
    print('  1. OOS ROI 年度別  (rank=1 全買い, 純モデル確率順)')
    hr('═')
    print(f'  {"年度":<6} {"R数":>6} {"勝利":>5} {"勝率":>7} {"ROI":>10}  参考')
    hr()
    year_data = {}
    for yr in sorted(top1['yr'].unique()):
        s   = top1[top1['yr'] == yr]
        roi, wr, w = roi_from_top1(s)
        year_data[yr] = (roi, len(s), w)
        note = '← 検証済み' if yr in (25, 26) else ''
        print(f'  20{yr}  {len(s):>6,} {w:>5}  {wr:>6.1%}  {roi:>+9.2%}  {note}')
    hr()
    won_tot = top1['着順_num'] == 1
    n_tot   = len(top1)
    roi_tot, wr_tot, w_tot = roi_from_top1(top1)
    print(f'  {"合計":<6} {n_tot:>6,} {int(w_tot):>5}  {wr_tot:>6.1%}  {roi_tot:>+9.2%}')
    print()

    # 25+26 combined
    (r25, n25, _), (r26, n26, _) = (year_data.get(25, (0,1,0)), year_data.get(26, (0,1,0)))
    n25, n26 = int(n25), int(n26)
    comb = (r25 * n25 + r26 * n26) / (n25 + n26) if (n25 + n26) > 0 else 0
    print(f'  ┌─────────────────────────────────────────────────────┐')
    print(f'  │  25+26 合算 ROI :  {comb:+.2%}                         │')
    print(f'  │  21F旧ベース比  :  +{comb-(-0.197):.2%}  (-19.70% → {comb:.2%}) │')
    print(f'  └─────────────────────────────────────────────────────┘')

    # ══════════════════════════════════════════════════════════════════════════
    print()
    hr('═')
    print('  2. クラス別 ROI  (OOS 2023+, rank=1)')
    hr('═')
    print(f'  {"クラス":<12} {"R数":>6} {"勝利":>5} {"勝率":>7} {"ROI":>10}')
    hr()
    for cls in ['未勝利', '1勝クラス', '2勝クラス', 'OP以上']:
        s = top1[top1['class_lbl'] == cls]
        if len(s) == 0:
            continue
        roi, wr, w = roi_from_top1(s)
        print(f'  {cls:<12} {len(s):>6,} {int(w):>5}  {wr:>6.1%}  {roi:>+9.2%}')

    # ══════════════════════════════════════════════════════════════════════════
    print()
    hr('═')
    print('  3. 距離帯別 ROI  (OOS 2023+, rank=1)')
    hr('═')
    print(f'  {"距離":>10} {"R数":>6} {"勝率":>7} {"ROI":>10}')
    hr()
    for (lo, hi), lbl in [
        ((1401, 1600), '1401-1600m'),
        ((1601, 1800), '1601-1800m'),
        ((1801, 2000), '1801-2000m'),
        ((2001, 9999), '2001m以上'),
    ]:
        s = top1[(top1['dist_m'] >= lo) & (top1['dist_m'] <= hi)]
        if len(s) < 30:
            continue
        roi, wr, w = roi_from_top1(s)
        print(f'  {lbl:>10} {len(s):>6,}  {wr:>6.1%}  {roi:>+9.2%}')

    # ══════════════════════════════════════════════════════════════════════════
    print()
    hr('═')
    print('  4. キャリブレーション  (OOS 2023+, 全馬 raw softmax → 実際勝率)')
    hr('═')
    print(f'  {"予測確率":>10} {"実際勝率":>10} {"N":>8}  精度')
    hr()
    oos_s['prob_bin'] = pd.qcut(oos_s['raw_prob'], 10, labels=False, duplicates='drop')
    cal = oos_s.groupby('prob_bin').agg(
        pred=('raw_prob', 'mean'),
        actual=('着順_num', lambda x: (x == 1).mean()),
        n=('raw_prob', 'count'),
    )
    for _, row in cal.iterrows():
        ratio = row['actual'] / row['pred'] if row['pred'] > 0 else 0
        bar   = '▓' * min(int(ratio * 5), 10) + '░' * max(0, 10 - int(ratio * 5))
        print(f'  {row["pred"]:>10.4f} {row["actual"]:>10.4f} {int(row["n"]):>8,}  {bar} ({ratio:.2f}x)')

    # ══════════════════════════════════════════════════════════════════════════
    print()
    hr('═')
    print('  5. モデル係数  (標準化後, |β| 降順)')
    hr('═')
    print(f'  {"順":>3}  {"特徴量":<32} {"β":>10}  解釈')
    hr()
    direction_notes = {
        '近5走_クラス調整_平均着順':   '小さい(上位)ほど有利',
        '前走着差タイム':             '小さい(負=勝ち近い)ほど有利',
        '1走前_クラス調整着順':       '小さい(上位)ほど有利',
        '1走前_上3F地点差':          '小さいほど有利',
        '近5走_タイム指数_max':       '大きいほど有利',
        '1走前_タイム指数':           '大きいほど有利',
        '近3走_複勝率':               '大きいほど有利',
        '騎手コース_r100_勝率':       '大きいほど有利',
        '調教師コース_r100_勝率':     '大きいほど有利',
        '斤量':                       '重いほど不利 (実力馬ハンデ)',
        '性別_num':                   '牡>牝>騸 (コードにより)',
        '所属_num':                   '関東/関西/地方 差',
        'キャリア_浅い':              '経験浅い馬は不利',
        '間隔_長_flag':               '60日以上休養は有利',
        '種牡馬_勝率':                '血統強さ指標',
        'タイム指数_近5走_slope':     '上昇トレンドは有利',
    }
    order = np.argsort(np.abs(beta))[::-1]
    for rank_i, i in enumerate(order, 1):
        b    = beta[i]
        note = direction_notes.get(feats[i], '')
        print(f'  {rank_i:>3}  {feats[i]:<32} {b:>+10.4f}  {note}')

    # ══════════════════════════════════════════════════════════════════════════
    print()
    hr('═')
    print('  6. 特徴量 NaN率  (訓練 2013-2021)')
    hr('═')
    print(f'  {"特徴量":<32} {"NaN率":>8} {"平均値":>10}')
    hr()
    for f in feats:
        if f not in df_trn.columns:
            print(f'  {f:<32} {"列なし":>8}')
            continue
        nan_r  = df_trn[f].isna().mean()
        mean_v = pd.to_numeric(df_trn[f], errors='coerce').mean()
        flag   = ' ◀高NaN' if nan_r > 0.2 else ''
        print(f'  {f:<32} {nan_r:>7.1%}  {mean_v:>10.4f}{flag}')

    # ══════════════════════════════════════════════════════════════════════════
    print()
    hr('═')
    print('  【サマリー】')
    hr('═')
    print(f'  モデル名    : New v1 (BASE_25)')
    print(f'  セグメント  : ダート中長距離 (>1400m, 新馬除外)')
    print(f'  25+26 ROI   : {comb:+.2%}  ← 検証済み数値')
    print(f'  旧ベース比  : +{comb-(-0.197):.2%}  (-19.70% → {comb:.2%})')
    print(f'  全期間 ROI  : {roi_tot:+.2%}  (2023-26, {n_tot:,}R)')
    print(f'  特徴量数    : {len(feats)}個  (旧モデル: 320+交互作用)')
    print(f'  ファイル    : models/roi_model.pkl (ダ artifact)')
    print(f'  バックアップ: models/final_model_pre_v1.pkl')
    hr('═')
    print()


if __name__ == '__main__':
    main()
