# -*- coding: utf-8 -*-
"""
特徴量エンジニアリング (horse_racing_ai_v2)

入力 : data/raw/results/*.csv  (fetch.py が出力した日別CSV)
出力 : data/processed/features.parquet

設計方針:
  - 1頭1レースを1行とし、その時点より前の情報だけを使う (look-ahead bias なし)
  - シンプルな特徴セットに絞る (EV model の土台として)
  - 欠損フィールド (走破タイム等) は NaN として保持

実行:
  python src/features.py
  python src/features.py --no-parquet   # parquet不要 → CSV で出力
"""
import sys, io, os, glob, argparse
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR    = os.path.join(BASE_DIR, 'data', 'raw', 'results')
OUT_DIR    = os.path.join(BASE_DIR, 'data', 'processed')
OUT_FILE   = os.path.join(OUT_DIR, 'features.parquet')
OUT_CSV    = os.path.join(OUT_DIR, 'features.csv')

DIST_BANDS = [
    (0,    1400, '短距離'),
    (1401, 1800, 'マイル'),
    (1801, 2200, '中距離'),
    (2201, 9999, '長距離'),
]


# ─────────────────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────────────────
def dist_band(d):
    try:
        v = int(d)
    except Exception:
        return ''
    for lo, hi, name in DIST_BANDS:
        if lo <= v <= hi:
            return name
    return ''


def slope(arr):
    """有効値が2件以上あれば polyfit の傾き、なければ NaN。"""
    valid = arr[~np.isnan(arr)]
    if len(valid) < 2:
        return np.nan
    x = np.arange(len(valid), dtype=float)
    return np.polyfit(x, valid, 1)[0]


def tail_mean(arr, n):
    """直近 n 件の平均（NaN 無視）。"""
    tail = arr[-n:]
    v = tail[~np.isnan(tail)]
    return v.mean() if len(v) > 0 else np.nan


def consecutive_place(arr, threshold=3):
    """末尾からの連続入着数 (着順 ≤ threshold)。"""
    count = 0
    for v in reversed(arr):
        if np.isnan(v):
            break
        if v <= threshold:
            count += 1
        else:
            break
    return count


# ─────────────────────────────────────────────────────────
# データ読み込み
# ─────────────────────────────────────────────────────────
def load_raw():
    files = sorted(glob.glob(os.path.join(RAW_DIR, '????????.csv')))
    if not files:
        raise FileNotFoundError(f"CSVが見つかりません: {RAW_DIR}/YYYYMMDD.csv")
    print(f"CSVファイル: {len(files)}件 ({os.path.basename(files[0])} 〜 {os.path.basename(files[-1])})")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding='utf-8', dtype=str)
            dfs.append(df)
        except Exception as e:
            print(f"  skip {f}: {e}")

    raw = pd.concat(dfs, ignore_index=True)
    print(f"読み込み: {len(raw):,}行")
    return raw


def preprocess(raw):
    df = raw.copy()

    # 数値変換
    num_cols = ['距離', '頭数', '馬番', '着順', '単勝オッズ',
                '斤量', '馬体重', '馬体重変化', '走破タイム', '上り3F',
                '1角', '2角', '3角', '4角']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 着順: 取消・除外・失格は NaN
    df['着順'] = df['着順'].where(df['着順'].between(1, 28), np.nan)

    # 日付を数値 (ソート用)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')

    # 距離帯
    df['距離帯'] = df['距離'].apply(lambda x: dist_band(x) if pd.notna(x) else '')

    # コース識別子 (芝ダ + 距離帯)
    df['コース'] = df['芝ダ'].fillna('') + '_' + df['距離帯'].fillna('')

    # race_id
    df['race_id'] = (df['日付'].fillna('') + '_' +
                     df['会場コード'].fillna('') + '_' +
                     df['レースNo'].fillna(''))

    # ソート
    df = df.sort_values(['馬名', '日付_num', 'race_id']).reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────
# 特徴量生成
# ─────────────────────────────────────────────────────────
def build_features(df):
    """
    各行に対して、その馬の過去レース情報から特徴量を付与する。
    df は馬名・日付でソート済みであること。
    """
    print("特徴量を生成中...")

    feat_rows = []

    for horse, grp in df.groupby('馬名', sort=False):
        grp = grp.reset_index(drop=True)
        n   = len(grp)

        for i in range(n):
            past = grp.iloc[:i]  # その馬の過去レース (look-ahead なし)

            row_feat = {
                # ── 識別子 ──
                'race_id':    grp.at[i, 'race_id'],
                '日付':       grp.at[i, '日付'],
                '日付_num':   grp.at[i, '日付_num'],
                '会場コード': grp.at[i, '会場コード'],
                '会場':       grp.at[i, '会場'],
                'レースNo':   grp.at[i, 'レースNo'],
                '馬名':       horse,
                '馬番':       grp.at[i, '馬番'],

                # ── ターゲット ──
                '着順':       grp.at[i, '着順'],
                '単勝オッズ': grp.at[i, '単勝オッズ'],

                # ── レース条件 ──
                '距離':       grp.at[i, '距離'],
                '距離帯':     grp.at[i, '距離帯'],
                '芝ダ':       grp.at[i, '芝ダ'],
                '馬場状態':   grp.at[i, '馬場状態'],
                'コース':     grp.at[i, 'コース'],
                '頭数':       grp.at[i, '頭数'],
                '斤量':       grp.at[i, '斤量'],
                '馬体重':     grp.at[i, '馬体重'],
                '馬体重変化': grp.at[i, '馬体重変化'],
                # ── 当該レースの生値 (predict.pyが時系列再構築に使う) ──
                '走破タイム_raw': grp.at[i, '走破タイム'] if '走破タイム' in grp.columns else np.nan,
                '上り3F_raw':    grp.at[i, '上り3F']    if '上り3F'    in grp.columns else np.nan,
                '4角_raw':       grp.at[i, '4角']        if '4角'        in grp.columns else np.nan,
            }

            if len(past) == 0:
                # 初出走: 全過去系特徴量 NaN
                feat_rows.append(_fill_nan(row_feat))
                continue

            chakujun_arr = past['着順'].values.astype(float)
            odds_arr     = past['単勝オッズ'].values.astype(float)
            soha_arr     = past['走破タイム'].values.astype(float) if '走破タイム' in past.columns else np.full(len(past), np.nan)
            last3f_arr   = past['上り3F'].values.astype(float)     if '上り3F'   in past.columns else np.full(len(past), np.nan)
            corner4_arr  = past['4角'].values.astype(float)        if '4角'      in past.columns else np.full(len(past), np.nan)

            # ── 直近着順 ──
            row_feat['直近1走_着順']   = chakujun_arr[-1] if len(chakujun_arr) >= 1 else np.nan
            row_feat['直近2走_着順']   = chakujun_arr[-2] if len(chakujun_arr) >= 2 else np.nan
            row_feat['直近3走_着順']   = chakujun_arr[-3] if len(chakujun_arr) >= 3 else np.nan
            row_feat['直近5走_着順']   = chakujun_arr[-5] if len(chakujun_arr) >= 5 else np.nan

            row_feat['直近3走_着順_平均'] = tail_mean(chakujun_arr, 3)
            row_feat['直近5走_着順_平均'] = tail_mean(chakujun_arr, 5)
            row_feat['直近3走_着順_slope'] = slope(chakujun_arr[-5:])

            row_feat['出走回数']       = len(past)
            row_feat['連続入着数']     = consecutive_place(chakujun_arr, 3)
            row_feat['連続連対数']     = consecutive_place(chakujun_arr, 2)
            row_feat['連続1着数']      = consecutive_place(chakujun_arr, 1)

            # 通算勝率・入着率 (全履歴)
            valid_c = chakujun_arr[~np.isnan(chakujun_arr)]
            if len(valid_c) > 0:
                row_feat['通算勝率']   = (valid_c == 1).mean()
                row_feat['通算入着率'] = (valid_c <= 3).mean()
            else:
                row_feat['通算勝率']   = np.nan
                row_feat['通算入着率'] = np.nan

            # ── 前走からの間隔 ──
            last_date = past['日付_num'].iloc[-1]
            curr_date = grp.at[i, '日付_num']
            try:
                # 8桁日付 → 日数差 (近似)
                ld = pd.Timestamp(str(int(last_date)))
                cd = pd.Timestamp(str(int(curr_date)))
                row_feat['前走間隔_日'] = (cd - ld).days
            except Exception:
                row_feat['前走間隔_日'] = np.nan

            # ── コース変更フラグ ──
            prev_surface = past['芝ダ'].iloc[-1] if len(past) > 0 else ''
            prev_dist    = past['距離帯'].iloc[-1] if len(past) > 0 else ''
            row_feat['芝ダ変更']   = int(prev_surface != grp.at[i, '芝ダ']) if prev_surface else np.nan
            row_feat['距離帯変更'] = int(prev_dist    != grp.at[i, '距離帯']) if prev_dist else np.nan

            # ── 同コース成績 ──
            same_course = past[past['コース'] == grp.at[i, 'コース']]
            sc_arr = same_course['着順'].values.astype(float)
            row_feat['同コース_出走数'] = len(same_course)
            row_feat['同コース_着順_平均'] = tail_mean(sc_arr, 5) if len(sc_arr) > 0 else np.nan
            row_feat['同コース_勝率']     = ((sc_arr == 1).mean()) if len(sc_arr) > 0 else np.nan

            # ── 直近オッズ ──
            row_feat['直近1走_オッズ']     = odds_arr[-1] if len(odds_arr) >= 1 else np.nan
            row_feat['直近3走_オッズ_平均'] = tail_mean(odds_arr, 3)

            # ── 走破タイム系 (未確認フィールド) ──
            row_feat['直近3走_走破タイム_平均'] = tail_mean(soha_arr, 3)
            row_feat['直近1走_走破タイム']      = soha_arr[-1] if len(soha_arr) >= 1 else np.nan

            # ── 上り3F系 ──
            row_feat['直近3走_上り3F_平均'] = tail_mean(last3f_arr, 3)
            row_feat['直近1走_上り3F']      = last3f_arr[-1] if len(last3f_arr) >= 1 else np.nan
            row_feat['直近3走_上り3F_slope'] = slope(last3f_arr[-5:])

            # ── 4角位置 ──
            row_feat['直近3走_4角_平均'] = tail_mean(corner4_arr, 3)
            row_feat['直近1走_4角']      = corner4_arr[-1] if len(corner4_arr) >= 1 else np.nan

            feat_rows.append(row_feat)

    features = pd.DataFrame(feat_rows)
    print(f"特徴量DataFrame: {len(features):,}行 × {len(features.columns)}列")
    return features


def _fill_nan(row):
    nan_keys = [
        '直近1走_着順', '直近2走_着順', '直近3走_着順', '直近5走_着順',
        '直近3走_着順_平均', '直近5走_着順_平均', '直近3走_着順_slope',
        '出走回数', '連続入着数', '連続連対数', '連続1着数',
        '通算勝率', '通算入着率', '前走間隔_日', '芝ダ変更', '距離帯変更',
        '同コース_出走数', '同コース_着順_平均', '同コース_勝率',
        '直近1走_オッズ', '直近3走_オッズ_平均',
        '直近3走_走破タイム_平均', '直近1走_走破タイム',
        '直近3走_上り3F_平均', '直近1走_上り3F', '直近3走_上り3F_slope',
        '直近3走_4角_平均', '直近1走_4角',
    ]
    for k in nan_keys:
        row[k] = np.nan
    row['出走回数'] = 0
    row['連続入着数'] = row['連続連対数'] = row['連続1着数'] = 0
    return row


# ─────────────────────────────────────────────────────────
# エントリポイント
# ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-parquet', action='store_true', help='CSV形式で出力')
    args = ap.parse_args()

    raw  = load_raw()
    df   = preprocess(raw)
    feat = build_features(df)

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.no_parquet:
        feat.to_csv(OUT_CSV, index=False, encoding='utf-8')
        print(f"保存: {OUT_CSV}")
    else:
        feat.to_parquet(OUT_FILE, index=False)
        print(f"保存: {OUT_FILE}")

    # 基本統計
    print(f"\n--- 基本統計 ---")
    print(f"日付範囲: {feat['日付_num'].min()} 〜 {feat['日付_num'].max()}")
    print(f"ユニーク馬: {feat['馬名'].nunique():,}頭")
    print(f"ユニークレース: {feat['race_id'].nunique():,}レース")
    print(f"着順あり: {feat['着順'].notna().sum():,}行")

    # 特徴量の欠損率
    feat_cols = [c for c in feat.columns if c not in
                 ['race_id','日付','日付_num','会場コード','会場','レースNo','馬名','馬番','着順','単勝オッズ']]
    miss = feat[feat_cols].isnull().mean().sort_values(ascending=False).head(10)
    print(f"\n欠損率 TOP10:")
    for col, rate in miss.items():
        print(f"  {col:<30} {rate:.1%}")


if __name__ == '__main__':
    main()
