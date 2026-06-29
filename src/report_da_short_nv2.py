# coding: utf-8
"""
report_da_short_nv2.py - ダート短距離 nv2 モデルレポート
特徴量: 近5走_上り3F平均, コース枠_r200_勝率, 1走前_馬場状態,
        1走前_クラス差, 2走前_クラス差
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
)
from save_v3 import add_computed_features

L2 = 0.006
FEATS = ['近5走_上り3F平均', 'コース枠_r200_勝率', '1走前_馬場状態',
         '1走前_クラス差', '2走前_クラス差']


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
        return float('nan'), 0.0, 0
    return (odds[won] * 100).sum() / (len(top1) * 100) - 1, won.mean(), int(won.sum())


def hr(char='─', n=62):
    print(char * n)


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
    df = df[(df['surface'] == 'ダ') & (dm <= 1400)].copy()
    df['dist_m'] = dm[df.index]
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
    df['クラス_rank'] = pd.to_numeric(df.get('クラス_rank', np.nan), errors='coerce')
    return df


def main():
    print()
    hr('═')
    print('  ダート短距離 nv2 モデルレポート')
    print('  生成日: 2026-06-06')
    hr('═')

    print('\nデータ読み込み・モデル訓練中...')
    t0 = time.time()
    df = load_segment()

    df_trn   = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val   = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos      = df[df['日付_num'] >= 230101].copy()
    oos_2324 = df[(df['日付_num'] >= 230101) & (df['日付_num'] < 250101)]
    oos_2025 = df[(df['日付_num'] >= 250101) & (df['日付_num'] < 260101)]
    oos_2026 = df[df['日付_num'] >= 260101]

    print(f'  train:{len(df_trn):,}  val:{len(df_val):,}  OOS:{len(oos):,}')
    print(f'  (2324:{len(oos_2324):,}  2025:{len(oos_2025):,}  2026:{len(oos_2026):,})')

    valid = [c for c in FEATS if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va)
    print(f'  訓練完了 ({int(time.time()-t0)}s)\n')

    print(f'\n【モデル概要】')
    print(f'  バージョン    : nv2')
    print(f'  特徴量数      : {len(valid)}個')
    print(f'  L2 正則化     : {L2}')
    print(f'  訓練期間      : 2013-2021')
    print(f'  バリデーション: 2022')
    print(f'  対象セグメント: ダート≤1400m (新馬除外)')
    print(f'  ベースライン  : 旧320特徴モデル = -14.56% (25+26)')

    # OOS スコアリング
    valid_p = [c for c in valid if c in oos.columns]
    X_p, _, gs_p, n_p, *_ = prepare(oos, valid_p, scaler=scaler,
                                      top_idx=None, top_idx3=None)
    oos_s = oos.sort_values('race_id').reset_index(drop=True)
    oos_s['prob']  = segment_softmax(X_p @ beta, gs_p, n_p)
    oos_s['rank']  = oos_s.groupby('race_id')['prob'].rank(ascending=False, method='first')
    oos_s['yr']    = (oos_s['日付_num'] // 10000).astype(int)
    oos_s['dist_m'] = oos_s['dist_m'].astype(int)
    oos_s['class_lbl'] = oos_s['クラス_rank'].map({
        2.0: '未勝利', 3.0: '1勝クラス', 4.0: '2勝クラス',
    }).fillna('OP以上')

    top1 = oos_s[oos_s['rank'] == 1].copy()

    # ══════════════════════════════════════════════════════════════════
    print()
    hr('═')
    print('  1. OOS ROI 年度別  (rank=1 全買い)')
    hr('═')
    print(f'  {"年度":<6} {"R数":>6} {"勝利":>5} {"勝率":>7} {"ROI":>10}')
    hr()
    year_data = {}
    for yr in sorted(top1['yr'].unique()):
        s   = top1[top1['yr'] == yr]
        roi, wr, w = roi_from_top1(s)
        year_data[yr] = (roi, len(s), w)
        blind_mark = ' ← blind' if yr in (25, 26) else ''
        print(f'  20{yr}  {len(s):>6,} {w:>5}  {wr:>6.1%}  {roi:>+9.2%}{blind_mark}')
    hr()
    roi_tot, wr_tot, w_tot = roi_from_top1(top1)
    print(f'  {"合計":<6} {len(top1):>6,} {w_tot:>5}  {wr_tot:>6.1%}  {roi_tot:>+9.2%}')
    print()

    r25, n25, _ = year_data.get(25, (0, 1, 0))
    r26, n26, _ = year_data.get(26, (0, 1, 0))
    n25, n26 = int(n25), int(n26)
    comb = (r25 * n25 + r26 * n26) / (n25 + n26) if (n25 + n26) > 0 else 0
    print(f'  ┌─────────────────────────────────────────────────────┐')
    print(f'  │  25+26 合算 ROI : {comb:+.2%}                          │')
    print(f'  │  旧ベース比     : +{comb-(-0.1456):.2%}  (-14.56% → {comb:.2%})  │')
    print(f'  └─────────────────────────────────────────────────────┘')

    # ══════════════════════════════════════════════════════════════════
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
        print(f'  {cls:<12} {len(s):>6,} {w:>5}  {wr:>6.1%}  {roi:>+9.2%}')

    # ══════════════════════════════════════════════════════════════════
    print()
    hr('═')
    print('  3. 距離帯別 ROI  (OOS 2023+, rank=1)')
    hr('═')
    print(f'  {"距離":>10} {"R数":>6} {"勝率":>7} {"ROI":>10}')
    hr()
    for (lo, hi), lbl in [
        ((0,    1000), '～1000m'),
        ((1001, 1200), '1100-1200m'),
        ((1201, 1400), '1300-1400m'),
    ]:
        s = top1[(top1['dist_m'] >= lo) & (top1['dist_m'] <= hi)]
        if len(s) < 20:
            continue
        roi, wr, w = roi_from_top1(s)
        print(f'  {lbl:>10} {len(s):>6,}  {wr:>6.1%}  {roi:>+9.2%}')

    # ══════════════════════════════════════════════════════════════════
    print()
    hr('═')
    print('  4. キャリブレーション  (OOS 2023+, 全馬 softmax確率 → 実際勝率)')
    hr('═')
    print(f'  {"予測確率":>10} {"実際勝率":>10} {"N":>8}  精度')
    hr()
    oos_s['prob_bin'] = pd.qcut(oos_s['prob'], 10, labels=False, duplicates='drop')
    cal = oos_s.groupby('prob_bin').agg(
        pred=('prob', 'mean'),
        actual=('着順_num', lambda x: (x == 1).mean()),
        n=('prob', 'count'),
    )
    for _, row in cal.iterrows():
        ratio = row['actual'] / row['pred'] if row['pred'] > 0 else 0
        bar   = '▓' * min(int(ratio * 5), 10) + '░' * max(0, 10 - int(ratio * 5))
        print(f'  {row["pred"]:>10.4f} {row["actual"]:>10.4f} {int(row["n"]):>8,}  {bar} ({ratio:.2f}x)')

    # ══════════════════════════════════════════════════════════════════
    print()
    hr('═')
    print('  5. モデル係数')
    hr('═')
    print(f'  {"特徴量":<32} {"β":>10}  解釈')
    hr()
    notes = {
        '近5走_上り3F平均':   '大きい(速い)ほど有利',
        'コース枠_r200_勝率': '枠勝率高いほど有利',
        '1走前_馬場状態':     '重・不良(高値)ほど有利（前走道悪ほど今走巻き返し）',
        '1走前_クラス差':     '正=格上がり, 負=格下がり',
        '2走前_クラス差':     '正=格上がり, 負=格下がり',
    }
    order = np.argsort(np.abs(beta))[::-1]
    for i in order:
        b = beta[i]
        print(f'  {valid[i]:<32} {b:>+10.4f}  {notes.get(valid[i], "")}')

    # ══════════════════════════════════════════════════════════════════
    print()
    hr('═')
    print('  6. 特徴量 NaN率  (訓練 2013-2021)')
    hr('═')
    print(f'  {"特徴量":<32} {"NaN率":>8} {"平均値":>10}')
    hr()
    for f in valid:
        if f not in df_trn.columns:
            print(f'  {f:<32} {"列なし":>8}')
            continue
        nan_r  = df_trn[f].isna().mean()
        mean_v = pd.to_numeric(df_trn[f], errors='coerce').mean()
        flag   = ' ◀高NaN' if nan_r > 0.2 else ''
        print(f'  {f:<32} {nan_r:>7.1%}  {mean_v:>10.4f}{flag}')

    # ══════════════════════════════════════════════════════════════════
    print()
    hr('═')
    print('  【サマリー】')
    hr('═')
    print(f'  モデル名    : ダート短距離 nv2')
    print(f'  セグメント  : ダート≤1400m (新馬除外)')
    print(f'  25+26 ROI   : {comb:+.2%}  ← blind評価')
    print(f'  旧ベース比  : +{comb-(-0.1456):.2%}  (-14.56% → {comb:.2%})')
    print(f'  全期間 ROI  : {roi_tot:+.2%}  (2023-26, {len(top1):,}R)')
    print(f'  特徴量      : {", ".join(valid)}')
    print(f'  選定経緯    : compare_da_short.py セットB (1走前+2走前クラス差)')
    hr('═')
    print()


if __name__ == '__main__':
    main()
