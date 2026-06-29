# coding: utf-8
"""
5/23・5/24 実績 ROI 計算
- parquet の特徴量 + final_model で予測
- 結果CSV(JV-Link取得)の着順・オッズと照合
"""
import os, sys, pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_lambdarank_pace import add_pace_features
from save_conditional_logit import add_new_features, segment_softmax, prepare
from save_final_model import get_surface

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE  = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_FILE = os.path.join(BASE_DIR, 'models', 'roi_model.pkl')
RES_DIR    = os.path.join(BASE_DIR, 'data', 'raw', 'results')

TARGET_DATES_STR  = ['20260523', '20260524']  # results CSV format
TARGET_DATES_NUM  = [260523, 260524]           # parquet format


def main():
    # ── モデル読み込み ─────────────────────────────────────────────
    print('モデル読み込み中...')
    with open(MODEL_FILE, 'rb') as f:
        pkg = pickle.load(f)
    artifacts     = pkg['artifacts']
    FACTOR_MAIDEN = pkg.get('factor_maiden', 0.0)
    FACTOR_OTHER  = pkg.get('factor_other',  0.16)
    print(f'  factor_maiden={FACTOR_MAIDEN}  factor_other={FACTOR_OTHER}')

    # ── 結果CSV 読み込み（実際の着順・オッズ・会場情報） ──────────
    print('実績CSVを読み込み中...')
    frames = []
    for ds in TARGET_DATES_STR:
        p = os.path.join(RES_DIR, f'{ds}.csv')
        df_r = pd.read_csv(p, encoding='utf-8')
        frames.append(df_r)
    res = pd.concat(frames, ignore_index=True)

    # 日付を6桁に変換してマッチキーに
    res['日付_6'] = res['日付'].astype(str).str[-6:].astype(int)  # 20260523 → 260523
    res['race_id'] = (res['日付'].astype(str) + '_' +
                      res['会場'].astype(str).str.strip() + '_' +
                      res['レースNo'].astype(str).str.strip())
    res = res.rename(columns={'着順': '着順_actual', '単勝オッズ': '単勝オッズ_actual',
                               '馬名': '馬名S'})
    print(f'  実績行数: {len(res)}  ユニーク会場: {res["会場"].unique().tolist()}')
    print(f'  ユニークレース数: {res["race_id"].nunique()}')

    # ── parquet 読み込み・前処理 ──────────────────────────────────
    print('parquet 読み込み中...')
    df = pd.read_parquet(DATA_FILE)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'] if '着順_num' in df.columns
                                   else df['着順'], errors='coerce').fillna(0)
    df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')

    # 5/23・5/24 のみ
    feat = df[df['日付_num'].isin(TARGET_DATES_NUM)].copy()
    print(f'  特徴量行数: {len(feat)}')

    # race_id は results の会場情報を使うため、一旦仮 race_id (merge後に置き換える)
    feat['_tmp_race_id'] = (feat['日付_num'].astype(str) + '_' +
                            feat['Ｒ'].astype(str).str.strip())

    # 特徴量エンジニアリング（add_pace_features は race_id が必要）
    # → 仮 race_id + 馬名 でペースは近似計算
    feat = feat.rename(columns={'_tmp_race_id': 'race_id'})
    feat = add_pace_features(feat)
    feat = add_new_features(feat)
    feat['surface'] = get_surface(feat)

    # ── マージ (馬名S + 日付_6 + Ｒ == レースNo) ──────────────────
    print('マージ中...')
    feat['日付_6'] = feat['日付_num']  # already 260523 format
    feat_cols_needed = list(feat.columns)

    merged = res.merge(
        feat,
        left_on=['馬名S', '日付_6', 'レースNo'],
        right_on=['馬名S', '日付_6', 'Ｒ'],
        how='left',
        suffixes=('', '_feat')
    )
    merged['race_id'] = merged['race_id']  # from results CSV (has proper venue)
    print(f'  マージ後: {len(merged)}行  マッチ率: {merged["日付_num"].notna().mean():.1%}')

    # マッチしなかった行を確認
    unmatched = merged[merged['日付_num'].isna()]
    if len(unmatched) > 0:
        print(f'  ⚠ マッチなし: {len(unmatched)}行')
        print(unmatched[['馬名S', '日付_6', 'レースNo']].head())

    merged = merged[merged['日付_num'].notna()].copy()

    # ── 予測 ────────────────────────────────────────────────────
    print('モデル予測中...')
    merged['クラス_rank'] = pd.to_numeric(merged['クラス_rank'], errors='coerce')
    merged = merged[merged['surface'].isin(['芝', 'ダ'])].copy()

    calib_arr = np.zeros(len(merged))
    for surf in ['芝', 'ダ']:
        art  = artifacts[surf]
        mask = (merged['surface'] == surf).values
        if mask.sum() == 0:
            continue
        m_s = merged[mask].sort_values('race_id').reset_index(drop=True)
        try:
            X, y, gs, n, *_ = prepare(
                m_s, art['feat_cols'],
                scaler=art['scaler'], poly2=art['poly2'],
                inter_scaler2=art['inter_scaler2'], top_idx=art['top_idx'],
                poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
            raw   = segment_softmax(X @ art['coef'], gs, n)
            calib = art['isotonic'].predict(raw)
            calib_arr[np.where(mask)[0]] = calib
        except Exception as e:
            print(f'  [{surf}] 予測エラー: {e}')

    merged['calib_prob'] = calib_arr
    merged['market_prob'] = 1.0 / np.clip(
        pd.to_numeric(merged['単勝オッズ_actual'], errors='coerce').values, 1.0, None)

    is_maiden = (merged['クラス_rank'] == 2).fillna(False)
    factor_arr = np.where(is_maiden, FACTOR_MAIDEN, FACTOR_OTHER)
    merged['score'] = merged['calib_prob'] - factor_arr * merged['market_prob']
    merged['ev_score'] = merged['calib_prob'] - merged['market_prob'] * 0.80
    merged['rank_final'] = merged.groupby('race_id')['score'].rank(
        ascending=False, method='first')

    # ── 結果表示 ──────────────────────────────────────────────────
    top1 = merged[merged['rank_final'] == 1].copy()
    top1['odds'] = pd.to_numeric(top1['単勝オッズ_actual'], errors='coerce')
    top1['won']  = (top1['着順_actual'] == 1)

    n_races = len(top1)
    n_won   = top1['won'].sum()
    total_return = (top1.loc[top1['won'], 'odds'] * 100).sum()
    total_bet    = n_races * 100
    roi = total_return / total_bet - 1

    print()
    print('=' * 55)
    print('5月23日・24日 実績 ROI (final_model 予測)')
    print('=' * 55)
    print(f'対象レース数 : {n_races}')
    print(f'的中数       : {n_won}')
    print(f'勝率         : {n_won / n_races:.3f}')
    print(f'ROI          : {roi:+.3f}  ({roi*100:+.1f}%)')
    print()

    for ds in TARGET_DATES_STR:
        date_6 = int(ds[-6:])
        t = top1[top1['日付_num'] == date_6]
        w = t['won']
        if len(t) == 0:
            continue
        o = pd.to_numeric(t['単勝オッズ_actual'], errors='coerce')
        r = (o[w] * 100).sum() / (len(t) * 100) - 1
        print(f'  {ds}: {len(t)}R  win={w.mean():.3f}  ROI={r:+.3f}')

    # 的中レースの詳細
    print()
    print('--- 的中レース ---')
    won_rows = top1[top1['won']].sort_values(['日付_num', 'race_id'])
    for _, row in won_rows.iterrows():
        print(f"  {row['race_id']}  {row['馬名S']}  オッズ={row['単勝オッズ_actual']}  "
              f"calib={row['calib_prob']:.3f}  EV={row['ev_score']:.3f}")

    print()
    print('--- 上位予測馬の実績 (score順) ---')
    top1_sorted = top1.sort_values('日付_num')
    for _, row in top1_sorted.iterrows():
        mark = '○' if row['won'] else '×'
        print(f"  {mark} {row['race_id']}  {row['馬名S'][:8]:<8}  "
              f"着={int(row['着順_actual']) if pd.notna(row['着順_actual']) else '?'}  "
              f"オッズ={row['単勝オッズ_actual']}  calib={row['calib_prob']:.3f}")

    # EV フィルタ
    print()
    print('--- EV フィルタ ---')
    for thr in [0.00, 0.02, 0.03]:
        ev = top1[top1['ev_score'] > thr]
        if len(ev) > 0:
            w_ev = ev['won']
            o_ev = pd.to_numeric(ev['単勝オッズ_actual'], errors='coerce')
            r_ev = (o_ev[w_ev] * 100).sum() / (len(ev) * 100) - 1
            print(f'  EV>{thr:.2f}: {len(ev)}件  win={w_ev.mean():.3f}  ROI={r_ev:+.3f}')


if __name__ == '__main__':
    main()
