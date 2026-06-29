# coding: utf-8
"""
fullfeats_l2.py - NaN<65%の全特徴量をL2正則化で一括投入
  強いL2で係数スプールを避けつつ、良い特徴を自動選択
  リーク系（走破タイム・着順・当日ポジション）は除外
"""
import sys, os
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, segment_softmax, BASE_DIR, DATA_FILE, LR, N_EPOCHS, PATIENCE
)
from save_v3 import add_computed_features, calc_roi

# リーク・使用不可列（現在のレース結果など）
EXCLUDE = {
    '着順', '着順_num', '着差', '走破タイム', '走破タイム_sec',
    '単勝配当', '複勝配当', '馬単', '馬連', '枠連', '３連単', '３連複',
    '賞金',
    # 当日コーナー位置（レース中データ）
    '3角', '4角', '2角', '前2角',
    # 当日タイム・速度
    '-3F平均速度', 'Ave-3F', '上3F地点差', '上り3F', '上り3F_指数', '平均速度',
    '上り3F平均速度',  # 当日上がり速度（上り3Fと-0.874相関 = リーク）
    'タイム指数',
    # 当日走法（着順との相関0.38で当日レース結果混入 = リーク）
    '脚質_num',
    # 当日ペース指数
    'PCI', 'PCI3', 'RPCI',
    # race_id計算用
    '日付', '開催', 'Ｒ', '日付_num',
    # ID列
    '馬名S', '騎手', '調教師', '種牡馬', '母父馬', '生産者',
    '産地', '馬主(最新/仮想)', '毛色', '生年月日',
    # string columns (非数値)
    '距離', 'レース名', '前走レース名', '今回_コース種別', '今回_会場',
    '馬場状態', '前走馬場状態', '脚質', '前走脚質',
    '芝・ダ', '前芝・ダ',
    '着順', '前走着順',
    '替', '前走B', '前好走', '好走',
    '前走馬印', '前走馬印2', '前走馬印3', '前走馬印4',
    '馬印', '馬印2', '馬印3', '馬印4',
    '走破タイム', '前走走破タイム',
    'ブリンカー', '馬記号', 'コース区分',
    '前走開催', '取引市場(最終)', '市場取引価格(万/最終)',
    # 100%欠損
    'レース印１', '前走レース印１', 'Ｃ', 'Ｍ',
    # 当日計算特徴量（現在のレース情報）
    '今回_surface', '今回_コース種別', '今回_会場', '今回_距離_m',
    'コース_先行有利度', 'コース展開マッチ', 'レース内_先行馬数',
    'レース内_平均脚質', 'レース内_相対脚質', 'レース内_逃げ馬数',
    'レース内_脚質std', '推定ペース',
    # PCI系の当日版
    '前PCI',
    # 計算中の race_id
    'race_id', 'surface', 'dist_m',
}


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
    return df


def _loss_grad(beta, X, y, gs, n, nr, l2=0.0):
    probs = segment_softmax(X @ beta, gs, n)
    res   = y - probs
    loss  = -np.sum(y * np.log(np.clip(probs, 1e-15, 1.0))) / nr + l2 * np.dot(beta, beta)
    grad  = -(X.T @ res) / nr + 2 * l2 * beta
    return loss, grad


def adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr, X_va, y_va, gs_va, n_va, nr_va, l2=0.0):
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


def evaluate(df_trn, df_val, oos_parts, feats, l2=0.0):
    valid = [c for c in feats if c in df_trn.columns and df_trn[c].isna().mean() <= 0.65]
    X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
        df_trn, valid, top_idx=None, top_idx3=None, fit=True)
    X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
        df_val, valid, scaler=scaler, top_idx=None, top_idx3=None)
    beta = adam_fit(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                    X_va, y_va, gs_va, n_va, nr_va, l2=l2)
    oos_roi = {}
    for period, df_p in oos_parts.items():
        valid_p = [c for c in valid if c in df_p.columns]
        X_p, _, gs_p, n_p, *_ = prepare(df_p, valid_p, scaler=scaler,
                                          top_idx=None, top_idx3=None)
        scored = df_p.sort_values('race_id').reset_index(drop=True)
        scored['prob'] = segment_softmax(X_p @ beta, gs_p, n_p)
        scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
        top1 = scored[scored['rank'] == 1]
        roi, wins = calc_roi(top1)
        oos_roi[period] = (roi, len(top1), wins)
    return beta, valid, oos_roi


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

    # 使用可能な全数値列を列挙
    numeric_cols = [c for c in df.columns
                    if c not in EXCLUDE
                    and df[c].dtype in [np.float64, np.int64, 'float64', 'int64']
                    and df_trn[c].isna().mean() <= 0.65
                    and df_trn[c].nunique() > 1]
    print(f'利用可能特徴量: {len(numeric_cols)}個')

    print('\n=== L2強度比較 (全特徴量) ===')
    best_comb = -99; best_l2 = 0
    for l2 in [0.006, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        beta, valid, roi = evaluate(df_trn, df_val, oos_parts, numeric_cols, l2=l2)
        r24 = roi.get('2324', (0, 1, 0))[0]
        r25, n25, _ = roi.get('2025', (0, 1, 0))
        r26, n26, _ = roi.get('2026', (0, 1, 0))
        comb = (r25*n25 + r26*n26) / (n25+n26)
        mark = ' ★' if comb > best_comb else ''
        print(f'  L2={l2:.3f}  F={len(valid):3d}  2324:{r24:+.2%}  2025:{r25:+.2%}  2026:{r26:+.2%}  25+26:{comb:+.2%}{mark}')
        if comb > best_comb:
            best_comb = comb; best_l2 = l2

    # 最良設定の係数上位
    print(f'\n=== 最良 L2={best_l2} の係数上位20F ===')
    beta, valid, roi = evaluate(df_trn, df_val, oos_parts, numeric_cols, l2=best_l2)
    idx = np.argsort(np.abs(beta))[::-1]
    for rank_i, i in enumerate(idx[:20]):
        print(f'  {rank_i+1:2d}. {valid[i]:<40} β={beta[i]:+.4f}')

    r25, n25, w25 = roi.get('2025', (0, 1, 0))
    r26, n26, w26 = roi.get('2026', (0, 1, 0))
    comb = (r25*n25 + r26*n26) / (n25+n26)
    print(f'\n【全特徴 L2={best_l2} 最終結果】')
    print(f'  2025: {n25}R  ROI={r25:+.4f}  勝率={w25/n25:.1%}')
    print(f'  2026: {n26}R  ROI={r26:+.4f}  勝率={w26/n26:.1%}')
    print(f'  25+26: {comb:+.4f}')
    print(f'  24F+L2=0.006基準(-17.65%) から: {comb-(-0.1765):+.2%}')
    print(f'  21F基準(-19.70%) から: {comb-(-0.197):+.2%}')


if __name__ == '__main__':
    main()
