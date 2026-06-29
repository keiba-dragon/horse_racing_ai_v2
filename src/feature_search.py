# coding: utf-8
"""
Greedy forward+backward feature selection for conditional logit (dist-split 5seg).
- 1ラウンド: forward pass（候補を1本ずつ追加し改善なら採用）→ backward pass（現在特徴量を1本ずつ除き改善なら採用）
- 改善なくなるまで繰り返す
- 結果を logs/feature_search_log.jsonl に追記

注意:
  - 現走データ（上り3F/3角/PCI等）は除外（リーク）
  - NaN率>60%は除外
  - L2正則化 ALPHA=1.0 で過学習抑制
"""
import sys, os, json, datetime, time
import numpy as np
import pandas as pd

# 即時フラッシュ（ファイルリダイレクト時のバッファリング対策）
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import (
    prepare, neg_log_lik_and_grad, segment_softmax,
    BASE_DIR, DATA_FILE, ALPHA, LR, N_EPOCHS, PATIENCE
)

LOG_DIR  = os.path.join(BASE_DIR, 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'feature_search_log.jsonl')
os.makedirs(LOG_DIR, exist_ok=True)

# ── 初期特徴量（v301_blood の 17 本）─────────────────────────────────────
INIT_FEATURES = [
    '1走前_タイム指数', '1走前_上り3F', '前走着差タイム', '1走前_RPCI',
    '近5走_タイム指数平均', '近5走_タイム指数_max', 'タイム指数_近3走_slope',
    '1走前_クラス調整着順', '近5走_クラス調整_平均着順',
    '馬番', 'コース枠_r200_勝率',
    '騎手コース_r100_勝率', '騎手変更', '調教師コース_r100_勝率',
    'コース脚質_r200_勝率', '1走前_脚質_num', '種牡馬_勝率',
]

# ── 追加候補（リーク除外・NaN率<60%を想定）─────────────────────────────
ADD_CANDIDATES = [
    # 馬自身の実績
    '馬_r20_複勝率',       # 馬の近20走複勝率
    '馬コース_r20_複勝率', # 馬のコース別複勝率
    '近走連続入着数',       # 連続複勝圏フィニッシュ数
    # 体重・負担
    '馬体重',              # 現在の馬体重（事前情報）
    '斤量',                # 負担重量
    # 調子・間隔
    '間隔',                # 前走からの間隔（週数）
    # 展開適性
    '展開フィット_v2',     # 脚質×コース展開フィット
    # 転向・距離変化
    '芝ダ転向',            # 芝ダート転向フラグ
    '距離変化_前走',       # 前走からの距離変化
    # 血統
    '母父馬_勝率',         # 母父の全体勝率
    '種牡馬_ダ_勝率',      # 種牡馬ダート勝率
    # キャリア
    'キャリア',            # 通算出走数
    # 騎手系追加
    '騎手距離_r100_勝率',  # 騎手の距離別勝率
    '騎手脚質_r100_勝率',  # 騎手の脚質別勝率
    # 同会場実績
    '同会場_複勝率_近5走', # 同会場の近5走複勝率
    # 近5走上り
    '近5走_上り3F指数平均', # 近5走の上り3F指数平均
    # 前走詳細
    '1走前_3角',           # 前走3角順位（レース展開情報）
    '1走前_上3F地点差',    # 前走上り3F地点差
    '乗替り_近走不振',     # 乗替り×近走不振シグナル
]

# ── 5セグメント定義 ────────────────────────────────────────────────────────
SEGMENTS = [('芝', '短距離'), ('芝', '中距離'), ('芝', '長距離'),
            ('ダ', '短距離'), ('ダ', '中長距離')]


def load_data():
    print(f'データ読み込み: {DATA_FILE}')
    df = pd.read_parquet(DATA_FILE)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    df = df[df['開催'].notna()].copy()

    df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
    df = df[df['surface'].isin(['芝', 'ダ'])].copy()

    if 'クラス_rank' in df.columns:
        df = df[df['クラス_rank'] != 1.0].copy()

    # 距離帯
    df['dist_m'] = pd.to_numeric(
        df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    dm  = df['dist_m']
    shi = df['surface'] == '芝'
    da  = df['surface'] == 'ダ'
    df['dist_band'] = ''
    df.loc[shi & (dm <= 1400),               'dist_band'] = '短距離'
    df.loc[shi & (dm > 1400) & (dm <= 2000), 'dist_band'] = '中距離'
    df.loc[shi & (dm > 2000),                'dist_band'] = '長距離'
    df.loc[da  & (dm <= 1400),               'dist_band'] = '短距離'
    df.loc[da  & (dm > 1400),                'dist_band'] = '中長距離'

    print(f'有効データ: {len(df):,}行')
    return df


def adam_optimize(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                  X_va, y_va, gs_va, n_va, nr_va):
    d    = X_tr.shape[1]
    beta = np.zeros(d)
    m    = np.zeros(d)
    v    = np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    best_val  = np.inf
    best_beta = beta.copy()
    no_improve = 0
    CHECK_EVERY = 10

    for epoch in range(1, N_EPOCHS + 1):
        loss, grad = neg_log_lik_and_grad(beta, X_tr, y_tr, gs_tr, n_tr, nr_tr)
        t += 1
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad ** 2
        beta -= LR * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + eps)

        if epoch % CHECK_EVERY == 0:
            vl, _ = neg_log_lik_and_grad(beta, X_va, y_va, gs_va, n_va, nr_va)
            if vl < best_val:
                best_val  = vl
                best_beta = beta.copy()
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= PATIENCE // CHECK_EVERY:
                break

    return best_beta, best_val


def eval_features(df, feat_cols):
    """指定特徴量でモデルを学習しOOS ROI（全セグメント合計）を返す。"""
    # NaN率チェック
    nan_rates = {c: df[c].isna().mean() for c in feat_cols if c in df.columns}
    high_nan  = [c for c, r in nan_rates.items() if r > 0.6]
    if high_nan:
        print(f'  [スキップ] NaN率>60%: {high_nan}')
        return None, {}

    valid_cols = [c for c in feat_cols if c in df.columns]
    if len(valid_cols) < len(feat_cols):
        missing = [c for c in feat_cols if c not in df.columns]
        print(f'  [警告] 列なし: {missing}')
    if not valid_cols:
        return None, {}

    all_top1  = []
    seg_rois  = {}

    for surf, dist_band in SEGMENTS:
        if dist_band:
            df_s = df[(df['surface'] == surf) & (df['dist_band'] == dist_band)].copy()
        else:
            df_s = df[df['surface'] == surf].copy()

        trn = df_s[(df_s['日付_num'] >= 130101) & (df_s['日付_num'] < 220101)]
        val = df_s[(df_s['日付_num'] >= 220101) & (df_s['日付_num'] <= 221231)]
        oos = df_s[df_s['日付_num'] >= 230101].copy()

        if len(trn) < 500 or len(val) < 50 or len(oos) < 50:
            print(f'  [{surf}_{dist_band}] サンプル不足スキップ')
            continue

        try:
            X_tr, y_tr, gs_tr, n_tr, nr_tr, scaler, *_ = prepare(
                trn, valid_cols, top_idx=None, top_idx3=None, fit=True)
            X_va, y_va, gs_va, n_va, nr_va, *_ = prepare(
                val, valid_cols, scaler=scaler, top_idx=None, top_idx3=None)
            X_oo, y_oo, gs_oo, n_oo, nr_oo, *_ = prepare(
                oos, valid_cols, scaler=scaler, top_idx=None, top_idx3=None)
        except Exception as e:
            print(f'  [{surf}_{dist_band}] prepare失敗: {e}')
            continue

        beta, _ = adam_optimize(X_tr, y_tr, gs_tr, n_tr, nr_tr,
                                 X_va, y_va, gs_va, n_va, nr_va)

        oos = oos.sort_values('race_id').reset_index(drop=True)
        scores = X_oo @ beta
        probs  = segment_softmax(scores, gs_oo, n_oo)
        oos['model_prob'] = probs
        oos['rank_model'] = oos.groupby('race_id')['model_prob'].rank(
            ascending=False, method='first')
        oos['odds_num'] = pd.to_numeric(oos['単勝オッズ'], errors='coerce')

        top1 = oos[oos['rank_model'] == 1].copy()
        all_top1.append(top1)

        won = top1['着順_num'] == 1
        roi = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1
        seg_rois[f'{surf}_{dist_band}'] = round(float(roi), 4)
        print(f'  {surf}_{dist_band}: {len(top1)}R  ROI={roi:+.3f}')

    if not all_top1:
        return None, seg_rois

    combined = pd.concat(all_top1, ignore_index=True)
    won_all = combined['着順_num'] == 1
    total_roi = (combined.loc[won_all, 'odds_num'] * 100).sum() / (len(combined) * 100) - 1
    print(f'  ▶ 合計: {len(combined)}R  ROI={total_roi:+.3f}')
    return float(total_roi), seg_rois


def log_result(entry: dict):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def main():
    df = load_data()

    # 追加候補をNaN率で事前フィルタ
    valid_candidates = []
    for c in ADD_CANDIDATES:
        if c not in df.columns:
            print(f'[候補なし] {c}')
            continue
        nan_rate = df[c].isna().mean()
        if nan_rate > 0.6:
            print(f'[NaN率高] {c}: {nan_rate:.0%}')
            continue
        valid_candidates.append(c)
    print(f'\n有効追加候補: {len(valid_candidates)}本')
    print(valid_candidates)

    current_feats = list(INIT_FEATURES)
    # 現在フィーチャーにない候補のみ残す
    valid_candidates = [c for c in valid_candidates if c not in current_feats]

    # ── ベースライン評価 ─────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'ベースライン評価 ({len(current_feats)}特徴量)')
    print(current_feats)
    print('='*60)
    best_roi, best_seg = eval_features(df, current_feats)
    if best_roi is None:
        print('ベースライン評価失敗')
        return
    print(f'ベースラインROI: {best_roi:+.4f}')
    log_result({
        'ts': datetime.datetime.now().isoformat(),
        'phase': 'baseline',
        'n_feats': len(current_feats),
        'features': current_feats,
        'total_roi': best_roi,
        'seg_rois': best_seg,
    })

    round_num = 0
    while True:
        round_num += 1
        improved_this_round = False
        print(f'\n{"#"*60}')
        print(f'# Round {round_num}  現在ROI={best_roi:+.4f}  特徴量数={len(current_feats)}')
        print(f'{"#"*60}')

        # ── Forward pass ──────────────────────────────────────────────
        print(f'\n--- Forward pass ({len(valid_candidates)}候補) ---')
        add_order = list(valid_candidates)  # 毎ラウンド全候補を試す

        for cand in add_order:
            trial = current_feats + [cand]
            print(f'\n[+追加] {cand}  ({len(trial)}特徴量)')
            roi, seg = eval_features(df, trial)
            if roi is None:
                continue
            log_result({
                'ts': datetime.datetime.now().isoformat(),
                'phase': f'forward_r{round_num}',
                'action': f'+{cand}',
                'n_feats': len(trial),
                'features': trial,
                'total_roi': roi,
                'seg_rois': seg,
            })
            if roi > best_roi + 0.001:  # 0.1pp改善で採用
                print(f'  ★ 採用: {cand}  {best_roi:+.4f} → {roi:+.4f}')
                best_roi = roi
                best_seg = seg
                current_feats = trial
                valid_candidates.remove(cand)
                improved_this_round = True
                log_result({
                    'ts': datetime.datetime.now().isoformat(),
                    'phase': 'adopted',
                    'action': f'+{cand}',
                    'n_feats': len(current_feats),
                    'features': current_feats,
                    'total_roi': best_roi,
                    'seg_rois': best_seg,
                })

        # ── Backward pass ─────────────────────────────────────────────
        print(f'\n--- Backward pass ({len(current_feats)}特徴量を順に除外試験) ---')
        # 削除候補: 現在の全特徴量（最低3本は残す）
        for rem in list(current_feats):
            if len(current_feats) <= 3:
                break
            trial = [c for c in current_feats if c != rem]
            print(f'\n[-除外] {rem}  ({len(trial)}特徴量)')
            roi, seg = eval_features(df, trial)
            if roi is None:
                continue
            log_result({
                'ts': datetime.datetime.now().isoformat(),
                'phase': f'backward_r{round_num}',
                'action': f'-{rem}',
                'n_feats': len(trial),
                'features': trial,
                'total_roi': roi,
                'seg_rois': seg,
            })
            if roi > best_roi + 0.001:
                print(f'  ★ 除外採用: {rem}  {best_roi:+.4f} → {roi:+.4f}')
                best_roi = roi
                best_seg = seg
                current_feats = trial
                valid_candidates.append(rem)  # 除外したものは次round再候補に
                improved_this_round = True
                log_result({
                    'ts': datetime.datetime.now().isoformat(),
                    'phase': 'adopted',
                    'action': f'-{rem}',
                    'n_feats': len(current_feats),
                    'features': current_feats,
                    'total_roi': best_roi,
                    'seg_rois': best_seg,
                })

        if not improved_this_round:
            print(f'\n改善なし → 収束。最終ROI={best_roi:+.4f}  特徴量数={len(current_feats)}')
            print('最終特徴量:', current_feats)
            log_result({
                'ts': datetime.datetime.now().isoformat(),
                'phase': 'converged',
                'n_feats': len(current_feats),
                'features': current_feats,
                'total_roi': best_roi,
                'seg_rois': best_seg,
            })
            break

        # 候補が全て尽きても終了
        if not valid_candidates and not improved_this_round:
            break

    print('\n=== 探索完了 ===')
    print(f'最終ROI: {best_roi:+.4f}')
    print(f'最終特徴量 ({len(current_feats)}本):', current_feats)


if __name__ == '__main__':
    main()
