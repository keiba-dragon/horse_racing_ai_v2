# coding: utf-8
"""
add_new_feats.py - 16Fベースへの新規候補全投入
年齢・血統・オッズ・馬場状態・季節・ブリンカー等
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
)
from save_v3 import add_computed_features, calc_roi

BASE_16 = [
    '近5走_クラス調整_平均着順',
    '近5走_タイム指数_max', '1走前_タイム指数', '前走着差タイム',
    '騎手コース_r100_勝率', '1走前_クラス調整着順', '調教師コース_r100_勝率',
    '1走前_RPCI', '1走前_上3F地点差', '斤量', '種牡馬_勝率',
    '間隔_長_flag', '1走前_脚質_num', '騎手変更', '馬番',
    '近3走_複勝率',
    '調教師_r200_複勝率',  # 前ラウンドで採用
]

NEW_CANDIDATES = [
    # 年齢
    '年齢',
    # 血統（未試験系）
    '母父馬_勝率',
    '母父馬_複勝率',
    '種牡馬_複勝率',
    '種牡馬_ダ_勝率',
    '種牡馬_ダ_複勝率',
    '生産者_勝率',
    '生産者_複勝率',
    '産地_勝率',
    '産地_複勝率',
    # クラス変化
    '1走前_クラス差',
    '2走前_クラス差',
    '最大クラス差_近5走',
    '格上経験数_近5走',
    # ブリンカー
    'ブリンカー変更',
    'ブリンカー_装着',
    '前走ブリンカー_装着',
    # 連続着順系
    '近走連続入着数',
    '芝ダ一致数_近5走',
    # 馬場状態系（前走）
    '1走前_馬場状態',
    # 前走オッズ（期待度シグナル）
    '1走前_単勝オッズ',
    '前走単勝オッズ',
    # 季節
    '季節',
    '月',
    # 枠・内外
    '内外枠',
    'コース馬場_r200_勝率',
    'コース枠_r200_複勝率',
    # 体重トレンド
    '近3走_体重増減合計',
    # キャリア
    'キャリア_浅い',
    # 馬の個人成績
    '馬_r20_勝率',
    '馬_r20_複勝率',
    # 同会場系
    '同会場_出走数_近5走',
    '同会場_複勝率_近5走',
    # 所属
    '所属_num',
]


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

    # 文字型カラムを数値化
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col:
            df[col] = df[col].map(baba_map)
    season_map = {'春': 0, '夏': 1, '秋': 2, '冬': 3}
    if '季節' in df.columns:
        df['季節'] = df['季節'].map(season_map)
    if '月' in df.columns:
        df['月'] = pd.to_numeric(df['月'], errors='coerce')
    if '所属_num' in df.columns:
        df['所属_num'] = pd.to_numeric(df['所属_num'], errors='coerce')
    if '内外枠' in df.columns:
        df['内外枠'] = pd.to_numeric(df['内外枠'], errors='coerce')
    return df


def _loss_grad(beta, X, y, gs, n, nr):
    probs = segment_softmax(X @ beta, gs, n)
    res   = y - probs
    loss  = -np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) / nr
    grad  = -(X.T @ res) / nr
    return loss, grad


def adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va):
    d = X_tr.shape[1]
    beta, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t, best_val, best_beta, no_imp = 0, np.inf, np.zeros(d), 0
    for epoch in range(1, N_EPOCHS + 1):
        _, grad = _loss_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr)
        t += 1
        m = b1*m + (1-b1)*grad
        v = b2*v + (1-b2)*grad**2
        beta -= LR * (m/(1-b1**t)) / (np.sqrt(v/(1-b2**t)) + eps)
        if epoch % 10 == 0:
            vl, _ = _loss_grad(beta, X_va, y_va, gs_va, n_va, nr_va)
            if vl < best_val:
                best_val, best_beta, no_imp = vl, beta.copy(), 0
            else:
                no_imp += 1
            if no_imp >= PATIENCE // 10:
                break
    return best_beta, best_val


def evaluate(df_trn, df_val, oos_parts, feats):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    if not valid:
        return None, None, {}
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta, val_nll = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                              X_va, y_va, gs_va, n_va, nr_va)
    oos_roi = {}
    for period, df_p in oos_parts.items():
        if len(df_p) == 0:
            continue
        valid_p = [c for c in valid if c in df_p.columns]
        X_p, _, gs_p, n_p, *_ = prepare(df_p, valid_p, scaler=scaler,
                                          top_idx=None, top_idx3=None)
        scored = df_p.sort_values('race_id').reset_index(drop=True)
        scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        scored['rank'] = scored.groupby('race_id')['prob'].rank(
            ascending=False, method='first')
        top1 = scored[scored['rank'] == 1]
        roi, _ = calc_roi(top1)
        oos_roi[period] = (roi, len(top1))
    return val_nll, beta, oos_roi


def main():
    df = load_segment()
    df_trn = df[(df['日付_num'] >= 130101) & (df['日付_num'] < 220101)]
    df_val = df[(df['日付_num'] >= 220101) & (df['日付_num'] <= 221231)]
    oos    = df[df['日付_num'] >= 230101]
    oos_parts = {
        '2324': oos[oos['日付_num'] < 250101],
        '2025': oos[(oos['日付_num'] >= 250101) & (oos['日付_num'] < 260101)],
        '2026': oos[oos['日付_num'] >= 260101],
    }

    # ベース17F（調教師_r200_複勝率追加済み）
    nll0, _, roi0 = evaluate(df_trn, df_val, oos_parts, BASE_16)
    r25_0 = roi0['2025'][0]
    r26_0 = roi0['2026'][0]
    n25, n26 = roi0['2025'][1], roi0['2026'][1]
    comb0 = (r25_0*n25 + r26_0*n26) / (n25+n26)
    print(f'ベース({len(BASE_16)}F): 2324:{roi0["2324"][0]:+.2%}  '
          f'2025:{r25_0:+.2%}  2026:{r26_0:+.2%}  25+26:{comb0:+.2%}\n')

    results = []
    for cand in NEW_CANDIDATES:
        if cand in BASE_16 or cand not in df.columns:
            continue
        nan_trn = df_trn[cand].isna().mean() if cand in df_trn.columns else 1.0
        if nan_trn > 0.65:
            print(f'  スキップ {cand} (NaN={nan_trn:.0%})')
            continue
        nll, _, roi = evaluate(df_trn, df_val, oos_parts, BASE_16 + [cand])
        if nll is None:
            continue
        r25 = roi.get('2025', (0, 0))[0]
        r26 = roi.get('2026', (0, 0))[0]
        n25_t = roi.get('2025', (0, 0))[1]
        n26_t = roi.get('2026', (0, 0))[1]
        comb = (r25*n25_t + r26*n26_t) / (n25_t + n26_t) if n25_t+n26_t > 0 else 0
        d25 = r25 - r25_0
        d26 = r26 - r26_0
        d_comb = comb - comb0
        both = (d25 > 0 and d26 > 0)
        results.append((cand, d25, d26, d_comb, comb, both, nll))

    results.sort(key=lambda x: -x[3])
    print(f'{"特徴量":<35} {"Δ2025":>8} {"Δ2026":>8} {"Δ25+26":>8}  合意?  最終25+26')
    print('='*80)
    for cand, d25, d26, d_comb, comb, both, nll in results:
        sym = '✓' if d_comb > 0.003 else ('✗' if d_comb < -0.003 else '~')
        agree = '両期間✓' if both else ''
        print(f'{sym} {cand:<33} {d25:>+8.2%} {d26:>+8.2%} {d_comb:>+8.2%}  {agree:<6}  {comb:+.2%}')

    print()
    print('【有望 (Δ25+26 > 0)】:')
    for cand, d25, d26, d_comb, comb, both, nll in results:
        if d_comb > 0.003:
            agree = '[両期間✓]' if both else ''
            print(f'  ✓ {cand}: {d_comb:+.2%} {agree}')
    print()
    print('【両期間合意で改善】:')
    both_agree = [(c, d, comb) for c, _, _, d, comb, b, _ in results if b and d > 0.003]
    if both_agree:
        for c, d, comb in both_agree:
            print(f'  ✓ {c}: {d:+.2%} → 25+26={comb:+.2%}')
    else:
        print('  なし')


if __name__ == '__main__':
    main()
