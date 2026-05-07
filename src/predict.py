# -*- coding: utf-8 -*-
"""
予測・推奨リスト生成 (horse_racing_ai_v2)

入力:
  - data/raw/cards/YYYYMMDD.csv  (当日の出馬表)
  - data/processed/features.parquet  (全履歴特徴量)
  - models/  (学習済みモデル)

出力:
  - data/predictions/YYYYMMDD.csv

推奨ロジック:
  - EV = モデル勝率 × 単勝オッズ
  - EV ≥ EV_BUY  → 単勝推奨 (◎/○/▲)
  - EV ≤ EV_KILL → 消し
  - 複勝は 3着以内確率 × 複勝オッズ で同様に評価

実行:
  python src/predict.py 20260503
  python src/predict.py 20260503 --ev-threshold 1.2
  python src/predict.py 20260503 --odds 10.5 11.0 3.2 ...   # オッズ上書き
"""
import sys, io, os, json, pickle, argparse
import numpy as np
import pandas as pd
from itertools import groupby

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_FILE  = os.path.join(BASE_DIR, 'data', 'processed', 'features.parquet')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
CARD_DIR   = os.path.join(BASE_DIR, 'data', 'raw', 'cards')
PRED_DIR   = os.path.join(BASE_DIR, 'data', 'predictions')

EV_BUY_DEFAULT  = 1.2   # EV ≥ ここ → 推奨
EV_KILL_DEFAULT = 0.7   # EV ≤ ここ → 消し
MARK_LEVELS     = [(1.5, '◎'), (1.3, '○'), (1.2, '▲')]

DIST_BANDS = [
    (0,    1400, '短距離'),
    (1401, 1800, 'マイル'),
    (1801, 2200, '中距離'),
    (2201, 9999, '長距離'),
]

FEATURE_COLS = [
    '距離', '頭数', '馬番',
    '斤量', '馬体重', '馬体重変化',
    '出走回数',
    '直近1走_着順', '直近2走_着順', '直近3走_着順', '直近5走_着順',
    '直近3走_着順_平均', '直近5走_着順_平均', '直近3走_着順_slope',
    '連続入着数', '連続連対数', '連続1着数',
    '通算勝率', '通算入着率',
    '前走間隔_日',
    '芝ダ変更', '距離帯変更',
    '同コース_出走数', '同コース_着順_平均', '同コース_勝率',
    '直近1走_オッズ', '直近3走_オッズ_平均',
    '直近3走_走破タイム_平均', '直近1走_走破タイム',
    '直近3走_上り3F_平均', '直近1走_上り3F', '直近3走_上り3F_slope',
    '直近3走_4角_平均', '直近1走_4角',
]


def dist_band(d):
    try:
        v = int(d)
    except Exception:
        return ''
    for lo, hi, name in DIST_BANDS:
        if lo <= v <= hi:
            return name
    return ''


def get_group_key(surface, dist_b):
    if surface == 'ダ' and dist_b in ('中距離', '長距離'):
        dist_b = '中長距離'
    return f"{surface}_{dist_b}"


def load_models():
    info_path = os.path.join(MODEL_DIR, 'model_info.json')
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"モデルが見つかりません: {info_path}")
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)

    models = {}
    for key, minfo in info['models'].items():
        mpath = os.path.join(MODEL_DIR, minfo['path'])
        if os.path.exists(mpath):
            with open(mpath, 'rb') as f:
                models[key] = pickle.load(f)
    print(f"モデル {len(models)}グループ読み込み")
    return models, info['features']


def load_card(date_str):
    """出馬表CSVを読み込む。YYYYMMDD.csv を探す。"""
    # 複数のカードディレクトリを試す
    candidates = [
        os.path.join(CARD_DIR, f'{date_str}.csv'),
        os.path.join(CARD_DIR, 'jvlink', f'{date_str}.csv'),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path, encoding='utf-8', dtype=str)
            print(f"出馬表: {path}  ({len(df)}行)")
            return df
    raise FileNotFoundError(f"出馬表が見つかりません: {date_str}")


def load_history(horse_names, predict_date_num):
    """
    過去特徴量を読み込み、predict_date より前のデータだけを返す。
    horse_names: 予測対象の馬名セット
    """
    if not os.path.exists(FEAT_FILE):
        print(f"WARNING: 特徴量ファイルなし ({FEAT_FILE}) → 過去特徴量は NaN")
        return pd.DataFrame()

    feat = pd.read_parquet(FEAT_FILE)
    feat['日付_num'] = pd.to_numeric(feat['日付_num'], errors='coerce')
    past = feat[
        (feat['馬名'].isin(horse_names)) &
        (feat['日付_num'] < predict_date_num)
    ]
    return past


def build_predict_features(card, history, predict_date_num):
    """
    出馬表の各馬について、predict 用の特徴量 DataFrame を作る。
    """
    rows = []
    for _, horse_row in card.iterrows():
        horse_name = str(horse_row.get('馬名S', horse_row.get('馬名', ''))).strip()
        surface    = str(horse_row.get('芝ダ', '')).strip()
        distance   = horse_row.get('距離', np.nan)
        dist_b     = dist_band(distance)
        course     = f"{surface}_{dist_b}"
        horse_hist = history[history['馬名'] == horse_name].sort_values('日付_num')

        feat = {
            '馬名':    horse_name,
            '芝ダ':    surface,
            '距離帯':  dist_b,
            'コース':  course,
            '距離':    pd.to_numeric(distance, errors='coerce'),
            '頭数':    pd.to_numeric(horse_row.get('頭数', np.nan), errors='coerce'),
            '馬番':    pd.to_numeric(horse_row.get('馬番', np.nan), errors='coerce'),
            '斤量':    pd.to_numeric(horse_row.get('斤量', np.nan), errors='coerce'),
            '馬体重':  pd.to_numeric(horse_row.get('馬体重', np.nan), errors='coerce'),
            '馬体重変化': np.nan,
            '単勝オッズ': pd.to_numeric(horse_row.get('単オッズ', horse_row.get('単勝オッズ', np.nan)), errors='coerce'),
        }

        if len(horse_hist) == 0:
            # 過去データなし: 全 NaN
            for c in FEATURE_COLS:
                if c not in feat:
                    feat[c] = np.nan
            feat['出走回数'] = 0
            rows.append(feat)
            continue

        chakujun_arr = horse_hist['着順'].values.astype(float)
        odds_arr     = horse_hist['単勝オッズ'].values.astype(float)     if '単勝オッズ'   in horse_hist.columns else np.full(len(horse_hist), np.nan)
        soha_arr     = horse_hist['走破タイム_raw'].values.astype(float) if '走破タイム_raw' in horse_hist.columns else np.full(len(horse_hist), np.nan)
        last3f_arr   = horse_hist['上り3F_raw'].values.astype(float)    if '上り3F_raw'    in horse_hist.columns else np.full(len(horse_hist), np.nan)
        corner4_arr  = horse_hist['4角_raw'].values.astype(float)       if '4角_raw'       in horse_hist.columns else np.full(len(horse_hist), np.nan)

        def _tail(arr, n):
            v = arr[-n:]; v = v[~np.isnan(v)]; return v.mean() if len(v) > 0 else np.nan

        def _slope(arr):
            v = arr[~np.isnan(arr)]
            if len(v) < 2: return np.nan
            return np.polyfit(np.arange(len(v), dtype=float), v, 1)[0]

        def _consec(arr, thr):
            cnt = 0
            for v in reversed(arr):
                if np.isnan(v): break
                if v <= thr: cnt += 1
                else: break
            return cnt

        feat['直近1走_着順']   = chakujun_arr[-1] if len(chakujun_arr) >= 1 else np.nan
        feat['直近2走_着順']   = chakujun_arr[-2] if len(chakujun_arr) >= 2 else np.nan
        feat['直近3走_着順']   = chakujun_arr[-3] if len(chakujun_arr) >= 3 else np.nan
        feat['直近5走_着順']   = chakujun_arr[-5] if len(chakujun_arr) >= 5 else np.nan
        feat['直近3走_着順_平均'] = _tail(chakujun_arr, 3)
        feat['直近5走_着順_平均'] = _tail(chakujun_arr, 5)
        feat['直近3走_着順_slope'] = _slope(chakujun_arr[-5:])
        feat['出走回数']       = len(horse_hist)
        feat['連続入着数']     = _consec(chakujun_arr, 3)
        feat['連続連対数']     = _consec(chakujun_arr, 2)
        feat['連続1着数']      = _consec(chakujun_arr, 1)
        valid_c = chakujun_arr[~np.isnan(chakujun_arr)]
        feat['通算勝率']   = (valid_c == 1).mean() if len(valid_c) > 0 else np.nan
        feat['通算入着率'] = (valid_c <= 3).mean() if len(valid_c) > 0 else np.nan

        last_date = horse_hist['日付_num'].iloc[-1]
        try:
            ld = pd.Timestamp(str(int(last_date)))
            cd = pd.Timestamp(str(predict_date_num))
            feat['前走間隔_日'] = (cd - ld).days
        except Exception:
            feat['前走間隔_日'] = np.nan

        prev_surface = str(horse_hist['芝ダ'].iloc[-1]).strip() if '芝ダ' in horse_hist.columns else ''
        prev_dist_b  = str(horse_hist['距離帯'].iloc[-1]).strip() if '距離帯' in horse_hist.columns else ''
        feat['芝ダ変更']   = int(prev_surface != surface) if prev_surface else np.nan
        feat['距離帯変更'] = int(prev_dist_b  != dist_b)  if prev_dist_b  else np.nan

        same_course = horse_hist[horse_hist['コース'] == course] if 'コース' in horse_hist.columns else pd.DataFrame()
        sc_arr = same_course['着順'].values.astype(float) if len(same_course) > 0 else np.array([])
        feat['同コース_出走数']   = len(same_course)
        feat['同コース_着順_平均'] = _tail(sc_arr, 5) if len(sc_arr) > 0 else np.nan
        feat['同コース_勝率']     = (sc_arr == 1).mean() if len(sc_arr) > 0 else np.nan

        feat['直近1走_オッズ']      = odds_arr[-1] if len(odds_arr) >= 1 else np.nan
        feat['直近3走_オッズ_平均'] = _tail(odds_arr, 3)

        feat['直近3走_走破タイム_平均'] = _tail(soha_arr, 3)
        feat['直近1走_走破タイム']      = soha_arr[-1] if len(soha_arr) >= 1 else np.nan
        feat['直近3走_上り3F_平均']     = _tail(last3f_arr, 3)
        feat['直近1走_上り3F']          = last3f_arr[-1] if len(last3f_arr) >= 1 else np.nan
        feat['直近3走_上り3F_slope']    = _slope(last3f_arr[-5:])
        feat['直近3走_4角_平均']        = _tail(corner4_arr, 3)
        feat['直近1走_4角']             = corner4_arr[-1] if len(corner4_arr) >= 1 else np.nan

        rows.append(feat)

    return pd.DataFrame(rows)


def predict_race(card_feat, models, feature_cols):
    """各馬の勝率と EV を計算して返す。"""
    results = card_feat.copy()
    results['prob_win'] = np.nan
    results['group_key'] = results.apply(
        lambda r: get_group_key(str(r.get('芝ダ', '')).strip(),
                                str(r.get('距離帯', '')).strip()),
        axis=1
    )

    for key, clf in models.items():
        mask = results['group_key'] == key
        if mask.sum() == 0:
            continue
        sub = results[mask].copy()
        available = [c for c in feature_cols if c in sub.columns]
        X = sub[available].astype(float)
        proba = clf.predict_proba(X)[:, 1]
        results.loc[mask, 'prob_win'] = proba

    # EV = P(win) × odds
    results['ev'] = results['prob_win'] * results['単勝オッズ'].fillna(0)

    # 推奨マーク
    def _mark(ev):
        if pd.isna(ev): return ''
        for thr, mark in MARK_LEVELS:
            if ev >= thr: return mark
        return ''
    results['推奨'] = results['ev'].apply(_mark)

    # 消し判定
    results['消し'] = (results['ev'] <= EV_KILL_DEFAULT) & results['ev'].notna()

    return results


def format_output(results, date_str):
    """CSV出力用の DataFrame を作る。"""
    out_rows = []
    for _, row in results.iterrows():
        out_rows.append({
            '日付':       date_str,
            '馬名':       row.get('馬名', ''),
            '推奨':       row.get('推奨', ''),
            '消し':       '消し' if row.get('消し', False) else '',
            '勝率':       f"{row['prob_win']:.3f}" if pd.notna(row.get('prob_win')) else '',
            'EV':         f"{row['ev']:.3f}"        if pd.notna(row.get('ev'))       else '',
            '単勝オッズ': row.get('単勝オッズ', ''),
            '馬番':       row.get('馬番', ''),
            '芝ダ':       row.get('芝ダ', ''),
            '距離':       row.get('距離', ''),
            '直近3走_着順_平均': f"{row['直近3走_着順_平均']:.1f}" if pd.notna(row.get('直近3走_着順_平均')) else '',
            '通算勝率':   f"{row['通算勝率']:.1%}" if pd.notna(row.get('通算勝率')) else '',
        })
    return pd.DataFrame(out_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('date', help='予測日 YYYYMMDD')
    ap.add_argument('--ev-threshold', type=float, default=EV_BUY_DEFAULT)
    ap.add_argument('--no-save', action='store_true', help='CSV 保存しない')
    args = ap.parse_args()

    date_str = args.date
    predict_date_num = int(date_str)

    models, feature_cols = load_models()
    card = load_card(date_str)

    # 馬名列を統一
    if '馬名S' in card.columns and '馬名' not in card.columns:
        card = card.rename(columns={'馬名S': '馬名'})

    horse_names = set(card['馬名'].dropna().astype(str).str.strip())
    print(f"出走頭数: {len(horse_names)}頭")

    history = load_history(horse_names, predict_date_num)
    print(f"過去レコード: {len(history):,}行")

    card_feat = build_predict_features(card, history, predict_date_num)
    results   = predict_race(card_feat, models, feature_cols)

    # レース別サマリー表示
    print(f"\n{'='*65}")
    print(f" {date_str} 予測結果")
    print(f"{'='*65}")
    print(f"{'馬名':<16} {'推奨':>4} {'消し':>4} {'勝率':>6} {'EV':>6} {'オッズ':>6}")
    print(f"{'-'*65}")
    for _, r in results.sort_values('ev', ascending=False, na_position='last').iterrows():
        prob_s  = f"{r['prob_win']:.3f}" if pd.notna(r.get('prob_win')) else '   -'
        ev_s    = f"{r['ev']:.3f}"       if pd.notna(r.get('ev'))       else '   -'
        odds_s  = f"{r['単勝オッズ']:.1f}" if pd.notna(r.get('単勝オッズ')) else '   -'
        mark    = r.get('推奨', '') or (' 消' if r.get('消し', False) else '  ')
        print(f"  {str(r.get('馬名','')):<15} {mark:>4} {prob_s:>6} {ev_s:>6} {odds_s:>6}")

    if not args.no_save:
        os.makedirs(PRED_DIR, exist_ok=True)
        out_path = os.path.join(PRED_DIR, f'{date_str}.csv')
        out_df   = format_output(results, date_str)
        out_df.to_csv(out_path, index=False, encoding='utf-8')
        print(f"\n保存: {out_path}")

    # 推奨サマリー
    buy  = results[results['推奨'] != '']
    kill = results[results['消し'] == True]
    print(f"\n推奨: {len(buy)}頭  消し: {len(kill)}頭  (EV閾値={args.ev_threshold})")
    if len(buy) > 0:
        print("推奨馬:")
        for _, r in buy.sort_values('ev', ascending=False).iterrows():
            print(f"  {r.get('推奨','')} {r.get('馬名','')}  EV={r.get('ev',0):.3f}  オッズ={r.get('単勝オッズ','?')}")


if __name__ == '__main__':
    main()
