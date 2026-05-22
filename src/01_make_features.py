import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os
import re

FINISH_MAP = {'止': 99, '除': 99, '取': 99, '中': 99, '失': 99, '降': 99}

def calc_rolling_stats(df, group_col, dnum_col, window, min_periods, prefix):
    """点-in-time ローリング勝率/複勝率（shift(1)で当該レース除外・リークなし）"""
    df_s = df.sort_values(dnum_col).copy()
    valid = df_s['着順_num'].notna() & (df_s['着順_num'] < 99)
    df_s['_win_tmp'] = (df_s['着順_num'] == 1).astype(float).where(valid)
    df_s['_plc_tmp'] = (df_s['着順_num'] <= 3).astype(float).where(valid)
    w_col = f'{prefix}_r{window}_勝率'
    p_col = f'{prefix}_r{window}_複勝率'
    df_s[w_col] = (df_s.groupby(group_col)['_win_tmp']
                   .transform(lambda x: x.shift(1).rolling(window, min_periods=min_periods).mean()))
    df_s[p_col] = (df_s.groupby(group_col)['_plc_tmp']
                   .transform(lambda x: x.shift(1).rolling(window, min_periods=min_periods).mean()))
    df_s.drop(columns=['_win_tmp', '_plc_tmp'], inplace=True)
    return df_s.sort_index()[[w_col, p_col]]

def calc_rolling_stats_combo(df, group_cols, dnum_col, window, min_periods, prefix):
    """複合キー（騎手×コース等）のローリング統計"""
    df_s = df.sort_values(dnum_col).copy()
    valid = df_s['着順_num'].notna() & (df_s['着順_num'] < 99)
    df_s['_win_tmp'] = (df_s['着順_num'] == 1).astype(float).where(valid)
    df_s['_plc_tmp'] = (df_s['着順_num'] <= 3).astype(float).where(valid)
    w_col = f'{prefix}_r{window}_勝率'
    p_col = f'{prefix}_r{window}_複勝率'
    df_s[w_col] = (df_s.groupby(group_cols)['_win_tmp']
                   .transform(lambda x: x.shift(1).rolling(window, min_periods=min_periods).mean()))
    df_s[p_col] = (df_s.groupby(group_cols)['_plc_tmp']
                   .transform(lambda x: x.shift(1).rolling(window, min_periods=min_periods).mean()))
    df_s.drop(columns=['_win_tmp', '_plc_tmp'], inplace=True)
    return df_s.sort_index()[[w_col, p_col]]

def clean_finish(series):
    """全角数字・異常着順を数値化（全角→半角変換対応）"""
    def _to_num(x):
        s = str(x).strip()
        s = s.translate(str.maketrans('０１２３４５６７８９', '0123456789'))  # 全角→半角
        s = s.replace('着', '')  # '1着' → '1'
        return FINISH_MAP.get(s, s)
    return series.map(_to_num).pipe(pd.to_numeric, errors='coerce')

def clean_weight(series):
    return series.astype(str).str.replace(r'[☆★◆◇*＊▲△]', '', regex=True).pipe(pd.to_numeric, errors='coerce')

def clean_time_diff(series):
    return series.astype(str).str.replace('----', 'NaN').pipe(pd.to_numeric, errors='coerce')

def clean_race_time(series):
    """走破タイム: 1110 = 1分11秒0 → 71.0秒に変換"""
    def to_sec(x):
        try:
            x = int(float(str(x)))
            m = x // 1000
            s = (x % 1000) / 10
            return m * 60 + s
        except:
            return np.nan
    return series.map(to_sec)

def encode_sex(series):
    return series.map({'牡': 0, '牝': 1, 'セ': 2}).astype(float)

def encode_class(series):
    class_rank = {
        '新馬': 1, '未勝利': 2,
        '１勝': 3, '1勝': 3,
        '２勝': 4, '2勝': 4,
        '３勝': 5, '3勝': 5,
        'OP': 6, 'オープン': 6, 'L': 6,
        'G3': 7, 'G2': 8, 'G1': 9
    }
    def to_rank(val):
        s = str(val)
        for k, v in class_rank.items():
            if k in s:
                return v
        return np.nan
    return series.map(to_rank).astype(float)

def extract_surface(s):
    s = str(s)
    if '芝' in s: return 1.0
    if 'ダ' in s: return 0.0
    if '障' in s: return 2.0
    return np.nan

def extract_dist_m(s):
    m = re.search(r'(\d+)', str(s))
    return float(m.group(1)) if m else np.nan

def extract_venue(kaikai):
    m = re.search(r'\d+([^\d]+)', str(kaikai))
    return m.group(1) if m else str(kaikai)

BABA_MAP = {'良': 0, '稍': 1, '稍重': 1, '重': 2, '不': 3, '不良': 3}
def encode_baba(s):
    s = str(s).strip()
    return BABA_MAP.get(s, BABA_MAP.get(s[:1], np.nan))

def deviation_score_group(series, group_keys, df):
    """グループ内での偏差値を計算（速い=高いに変換済み）"""
    result = pd.Series(np.nan, index=df.index)
    for _, grp_idx in df.groupby(group_keys).groups.items():
        vals = series.loc[grp_idx]
        mean = vals.mean()
        std  = vals.std()
        if std > 0:
            result.loc[grp_idx] = 50 + 10 * (mean - vals) / std  # タイムは小さい方が良いので反転
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--test2012', action='store_true', help='2012年ホールドアウトテスト用')
    args, _ = parser.parse_known_args()

    print("--- 競馬AI データ加工処理を開始します（v3 新データ対応）---")

    base_dir = os.environ.get('KEIBAI_BASE_DIR') or (os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'src' in os.path.abspath(__file__) else '.')
    raw_dir  = os.path.join(base_dir, 'data', 'raw')
    output_file = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.csv')

    # ── ファイルパスの定義 ────────────────────────────────
    master_dir   = os.path.join(raw_dir, 'master')
    file_kihon   = os.path.join(master_dir, 'master_kihon.csv')   # 累積レース結果（毎月追記）
    file_time    = os.path.join(master_dir, 'master_time.csv')    # タイムデータ
    file_horse   = os.path.join(master_dir, 'master_horse.csv')   # 馬データ
    file_odds    = os.path.join(master_dir, 'master_all.csv')     # 全て形式（オッズ・脚質）
    file_odds2   = os.path.join(master_dir, 'recent_all.csv')     # 最新週の全て形式（毎週更新）
    file_card    = os.path.join(raw_dir, 'misc', 'card_extra.csv')  # 出馬表形式（06/10スクリプトが置く）

    # 旧ファイルへのフォールバック（legacy）
    file_kihon_old = os.path.join(master_dir, '2013-20260303　前走 スリム化.csv')
    file_horse_old = os.path.join(master_dir, '2013-20260303　馬データ.csv')

    if args.test2012:
        file_2012   = os.path.join(raw_dir, 'master', 'モデルテスト用2012年（過去含む）.csv')
        file_odds   = '__nonexistent__'  # df_raceに単勝オッズ・脚質が既に含まれるためスキップ
        file_odds2  = '__nonexistent__'
        file_card   = '__nonexistent__'
        output_file = os.path.join(base_dir, 'data', 'processed', 'features_2012_test.csv')
        print(f'=== 2012年ホールドアウトテストモード → {output_file} ===')

    # 1. データ読み込み
    print("データを読み込んでいます...")
    def read_csv(path, **kwargs):
        try:
            return pd.read_csv(path, encoding='cp932', low_memory=False, **kwargs)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding='utf-8', low_memory=False, **kwargs)

    if args.test2012:
        print(f"  2012全データ を使用（全て.csv形式）")
        df_zen_2012 = read_csv(file_2012)
        if '馬名' in df_zen_2012.columns and '馬名S' not in df_zen_2012.columns:
            df_zen_2012 = df_zen_2012.rename(columns={'馬名': '馬名S'})
        # 距離を 芝ダ+数値 形式に変換（例: 1200 → 芝1200）
        if '距離' in df_zen_2012.columns and '芝・ダ' in df_zen_2012.columns:
            df_zen_2012['距離'] = df_zen_2012.apply(
                lambda r: str(r['芝・ダ']).strip() + str(int(r['距離'])) if pd.notna(r.get('距離')) and pd.notna(r.get('芝・ダ')) else r.get('距離'), axis=1
            )
        df_race = df_zen_2012.copy()
        # 馬データ: 2012ファイルから必要列だけ抽出して使う
        uma_cols_need = ['日付', '馬名S', '種牡馬', '母父馬', '馬主(最新/仮想)', '生産者',
                         '毛色', '馬記号', '生年月日', '市場取引価格(万/最終)', '取引市場(最終)', '産地']
        uma_cols_exist = [c for c in uma_cols_need if c in df_zen_2012.columns]
        df_horse = df_zen_2012[uma_cols_exist].drop_duplicates(subset=['日付', '馬名S'])
        use_new = True
    elif os.path.exists(file_kihon):
        print(f"  基本.csv を使用")
        df_race = read_csv(file_kihon)
        # 基本.csvは '馬名' 列 → '馬名S' に統一
        if '馬名' in df_race.columns and '馬名S' not in df_race.columns:
            df_race = df_race.rename(columns={'馬名': '馬名S'})
        # 距離列の整合（基本.csvは距離が数値のため芝ダ情報と結合）
        if '距離' in df_race.columns and '芝・ダ' in df_race.columns:
            # 数値距離 + 芝ダ → 統一表記 (例: ダ1200)
            df_race['距離'] = df_race.apply(
                lambda r: str(r['芝・ダ']).strip() + str(int(r['距離'])) if pd.notna(r.get('距離')) and pd.notna(r.get('芝・ダ')) else r.get('距離'), axis=1
            )
        use_new = True
        # 全て.csv から基本.csv 以降の行を追加（N走前特徴量を正しく生成するため）
        if os.path.exists(file_odds2):
            df_zen = read_csv(file_odds2)
            if '馬名' in df_zen.columns and '馬名S' not in df_zen.columns:
                df_zen = df_zen.rename(columns={'馬名': '馬名S'})
            df_zen['_d'] = pd.to_numeric(df_zen['日付'], errors='coerce')
            last_date = pd.to_numeric(df_race['日付'], errors='coerce').max()
            extra = df_zen[df_zen['_d'] > last_date].drop(columns=['_d'])
            if len(extra) > 0:
                # 距離を基本.csvと同じ形式（芝ダ+数値）に変換してから追加
                if '距離' in extra.columns and '芝・ダ' in extra.columns:
                    extra = extra.copy()
                    extra['距離'] = extra.apply(
                        lambda r: str(r['芝・ダ']).strip() + re.sub(r'\D', '', str(r['距離'])) if pd.notna(r.get('距離')) and pd.notna(r.get('芝・ダ')) and re.sub(r'\D', '', str(r['距離'])) else r.get('距離'), axis=1
                    )
                common_cols = [c for c in df_race.columns if c in extra.columns]
                df_race = pd.concat([df_race, extra[common_cols]], ignore_index=True)
                print(f"  全て.csv から追加: {len(extra):,}行（{int(last_date)+1}以降）")
        # results_supplement.csv（auto_weekly_update.py が自動生成）があれば追加
        file_results_supp = os.path.join(master_dir, 'results_supplement.csv')
        if os.path.exists(file_results_supp):
            df_supp = read_csv(file_results_supp)
            df_supp['_d'] = pd.to_numeric(df_supp['日付'], errors='coerce')
            last_supp = pd.to_numeric(df_race['日付'], errors='coerce').max()
            extra_supp = df_supp[df_supp['_d'] > last_supp].drop(columns=['_d'])
            if len(extra_supp) > 0:
                # ペース指標列を df_race に追加（NaN）して supplement の値を保持
                for _pace_col in ['Ave-3F', 'PCI', 'RPCI', '平均速度', '-3F平均速度', '上り3F平均速度']:
                    if _pace_col in extra_supp.columns and _pace_col not in df_race.columns:
                        df_race[_pace_col] = float('nan')
                common_supp = [c for c in df_race.columns if c in extra_supp.columns]
                df_race = pd.concat([df_race, extra_supp[common_supp]], ignore_index=True)
                print(f"  results_supplement.csv から追加: {len(extra_supp):,}行")

        # card_extra.csv（出馬表形式から変換済み）があれば追加
        if os.path.exists(file_card):
            df_card = read_csv(file_card)
            df_card['_d'] = pd.to_numeric(df_card['日付'], errors='coerce')
            last_all = pd.to_numeric(df_race['日付'], errors='coerce').max()
            extra_c = df_card[df_card['_d'] > last_all].drop(columns=['_d'])
            if len(extra_c) > 0:
                if '距離' in extra_c.columns and '芝・ダ' in extra_c.columns:
                    extra_c = extra_c.copy()
                    extra_c['距離'] = extra_c.apply(
                        lambda r: str(r['芝・ダ']).strip() + str(int(r['距離'])) if pd.notna(r.get('距離')) and pd.notna(r.get('芝・ダ')) else r.get('距離'), axis=1
                    )
                common_c = [c for c in df_race.columns if c in extra_c.columns]
                df_race = pd.concat([df_race, extra_c[common_c]], ignore_index=True)
                print(f"  card_extra.csv から追加: {len(extra_c):,}行")
    else:
        print(f"  旧ファイル（前走スリム化）を使用")
        df_race = read_csv(file_kihon_old)
        use_new = False

    if not args.test2012:
        file_horse_use = file_horse if os.path.exists(file_horse) else file_horse_old
        df_horse = read_csv(file_horse_use)
    # 馬データにも全て.csv の追加行を補完
    if use_new and os.path.exists(file_odds2):
        df_zen_h = read_csv(file_odds2)
        if '馬名' in df_zen_h.columns and '馬名S' not in df_zen_h.columns:
            df_zen_h = df_zen_h.rename(columns={'馬名': '馬名S'})
        df_zen_h['_d'] = pd.to_numeric(df_zen_h['日付'], errors='coerce')
        last_h = pd.to_numeric(df_horse['日付'], errors='coerce').max()
        extra_h = df_zen_h[df_zen_h['_d'] > last_h].drop(columns=['_d'])
        if len(extra_h) > 0:
            common_h = [c for c in df_horse.columns if c in extra_h.columns]
            df_horse = pd.concat([df_horse, extra_h[common_h]], ignore_index=True)
    print(f"レースデータ: {len(df_race):,}行 / 馬データ: {len(df_horse):,}行")

    # タイムデータ（新データある場合）
    df_time = None
    if use_new and os.path.exists(file_time):
        df_time = read_csv(file_time)
        if '馬名' in df_time.columns and '馬名S' not in df_time.columns:
            df_time = df_time.rename(columns={'馬名': '馬名S'})
        print(f"タイムデータ: {len(df_time):,}行")

    # 2. マージ
    overlap = set(df_race.columns) & set(df_horse.columns) - {'日付', '馬名S'}
    df_horse = df_horse.drop(columns=list(overlap))
    # card_extra（未来日付）の行は df_horse に該当日付がないため、馬名のみで結合
    _horse_max_date = pd.to_numeric(df_horse['日付'], errors='coerce').max()
    df_race['_d_num'] = pd.to_numeric(df_race['日付'], errors='coerce')
    df_hist  = df_race[df_race['_d_num'] <= _horse_max_date].drop(columns=['_d_num'])
    df_card_future = df_race[df_race['_d_num'] > _horse_max_date].drop(columns=['_d_num'])
    df = pd.merge(df_hist, df_horse, on=['日付', '馬名S'], how='inner')
    if not df_card_future.empty:
        # 各馬の最新レコードを使って馬名のみで結合
        df_horse_latest = (df_horse.sort_values('日付')
                           .groupby('馬名S', sort=False).last().reset_index()
                           .drop(columns=['日付'], errors='ignore'))
        df_card_merged = df_card_future.merge(df_horse_latest, on='馬名S', how='left')
        df = pd.concat([df, df_card_merged], ignore_index=True)
        print(f"  card_extra 未来行を馬名結合で追加: {len(df_card_future):,}行")

    # オッズ・脚質データのマージ（全て2.csv + 最新補完ファイル）
    odds_cols_use = ['単勝オッズ', '前走単勝オッズ', '脚質', '前走脚質']
    odds_frames = []
    for fp in [file_odds, file_odds2]:
        if os.path.exists(fp):
            d = read_csv(fp)
            cols = ['日付', '馬名S'] + [c for c in odds_cols_use if c in d.columns]
            odds_frames.append(d[cols].copy())
    if odds_frames:
        df_o = pd.concat(odds_frames, ignore_index=True)
        df_o['日付'] = pd.to_numeric(df_o['日付'], errors='coerce')
        df_o = df_o.drop_duplicates(subset=['日付', '馬名S'], keep='last')
        df['日付_merge'] = pd.to_numeric(df['日付'], errors='coerce')
        df = df.merge(df_o.rename(columns={'日付': '日付_merge'}),
                      on=['日付_merge', '馬名S'], how='left')
        df = df.drop(columns=['日付_merge'])
        matched = df['単勝オッズ'].notna().sum() if '単勝オッズ' in df.columns else 0
        print(f"オッズ・脚質マージ完了: {matched:,}件マッチ")
    else:
        print("オッズファイルが見つかりません")

    # タイムCSVの追加列をマージ（Ave-3F, RPCI, 平均速度 など）
    if df_time is not None:
        time_extra = ['Ave-3F', 'RPCI', '平均速度', '-3F平均速度', '上り3F平均速度', '上3F地点差']
        time_extra = [c for c in time_extra if c in df_time.columns]
        if time_extra:
            df_t = df_time[['日付', '馬名S'] + time_extra].copy()
            df = pd.merge(df, df_t, on=['日付', '馬名S'], how='left')
            # _x/_y suffix collision を解消（master_time 優先、supplement フォールバック）
            for _col in time_extra:
                if _col+'_x' in df.columns and _col+'_y' in df.columns:
                    df[_col] = df[_col+'_y'].fillna(df[_col+'_x'])
                    df = df.drop(columns=[_col+'_x', _col+'_y'])
            print(f"タイム追加列: {time_extra}")

    # results_supplement.csv から Ave-3F 等を補完（master_time 範囲外をカバー）
    _file_supp = os.path.join(master_dir, 'results_supplement.csv')
    if use_new and os.path.exists(_file_supp):
        _df_supp = read_csv(_file_supp)
        if '馬名' in _df_supp.columns and '馬名S' not in _df_supp.columns:
            _df_supp = _df_supp.rename(columns={'馬名': '馬名S'})
        _supp_fill = ['Ave-3F', 'PCI', 'RPCI', '平均速度', '-3F平均速度', '上り3F平均速度']
        _supp_fill = [c for c in _supp_fill if c in _df_supp.columns]
        if _supp_fill:
            _df_supp['_dn'] = pd.to_numeric(_df_supp['日付'], errors='coerce')
            _df_sf = _df_supp[['_dn', '馬名S'] + _supp_fill].copy()
            df['_dn'] = pd.to_numeric(df['日付'], errors='coerce')
            df = df.merge(_df_sf, on=['_dn', '馬名S'], how='left', suffixes=('', '_supp'))
            for _c in _supp_fill:
                if _c+'_supp' in df.columns:
                    if _c in df.columns:
                        df[_c] = df[_c].fillna(df[_c+'_supp'])
                    else:
                        df[_c] = df[_c+'_supp']
                    df = df.drop(columns=[_c+'_supp'])
            df = df.drop(columns=['_dn'])
            print(f"supplement Ave-3F 補完完了")

    # 3. 日付・馬名でソート
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df = df.sort_values(by=['馬名S', '日付_num']).reset_index(drop=True)

    # 4. 数値クリーニング
    print("データをクリーニングしています...")
    df['着順_num'] = clean_finish(df['着順'])

    for col in ['斤量', '前走斤量']:
        if col in df.columns: df[col] = clean_weight(df[col])
    if '前走着差タイム' in df.columns:
        df['前走着差タイム'] = clean_time_diff(df['前走着差タイム'])
    if '前走着順' in df.columns:
        df['前走着順_num'] = clean_finish(df['前走着順'])
    for col in ['2角', '3角', '4角', '前2角', '前3角', '前4角']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

    # 走破タイム（秒換算）
    if '走破タイム' in df.columns:
        df['走破タイム_sec'] = clean_race_time(df['走破タイム'])
    if '前走走破タイム' in df.columns:
        df['前走走破タイム_sec'] = clean_race_time(df['前走走破タイム'])

    # 上り3F（数値化）
    for col in ['上り3F', '前走上り3F', 'Ave-3F', 'RPCI', '平均速度', '-3F平均速度', '上り3F平均速度', '上3F地点差']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

    # 単勝・複勝配当（評価用に保持、モデルには使わない）
    for col in ['単勝配当', '複勝配当']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 馬番
    if '馬番' in df.columns:
        df['馬番'] = pd.to_numeric(df['馬番'], errors='coerce')

    # 単勝オッズ数値化（EV計算用・訓練除外）
    if '単勝オッズ' in df.columns:
        df['単勝オッズ'] = pd.to_numeric(df['単勝オッズ'], errors='coerce')
    if '前走単勝オッズ' in df.columns:
        df['前走単勝オッズ'] = pd.to_numeric(df['前走単勝オッズ'], errors='coerce')

    # 5. カテゴリ変数を数値化
    print("カテゴリ変数を数値化しています...")
    if '性別' in df.columns: df['性別_num'] = encode_sex(df['性別'])
    if '年齢' in df.columns: df['年齢'] = pd.to_numeric(df['年齢'], errors='coerce')
    if '前クラス名' in df.columns: df['前クラス_rank'] = encode_class(df['前クラス名'])
    if '所属' in df.columns:
        df['所属_num'] = df['所属'].map(lambda x: 0 if '美' in str(x) else 1 if '栗' in str(x) else 2).astype(float)

    # キャリア（総出走数）
    if 'キャリア' in df.columns:
        df['キャリア'] = pd.to_numeric(df['キャリア'], errors='coerce')

    # 市場取引価格は欠損率75%超のため除外

    # 騎手変更フラグ（前走から騎手が変わったか）
    if '騎手' in df.columns and '前騎手' in df.columns:
        df['騎手変更'] = (
            df['騎手'].notna() & df['前騎手'].notna() &
            (df['騎手'] != df['前騎手'])
        ).astype(float)

    # 脚質数値化（逃げ=0, 先行=1, 中団=2, 後方=3, マクリ=4）
    KASHITSU_MAP = {'逃げ': 0, '先行': 1, '中団': 2, '後方': 3, 'ﾏｸﾘ': 4}
    if '脚質' in df.columns:
        df['脚質_num'] = df['脚質'].map(KASHITSU_MAP).astype(float)
    if '前走脚質' in df.columns:
        df['前走脚質_num'] = df['前走脚質'].map(KASHITSU_MAP).astype(float)

    # ブリンカー（装着=1, なし=0 → 変化も計算）
    if 'ブリンカー' in df.columns:
        df['ブリンカー_装着'] = df['ブリンカー'].notna().astype(float)
    if '前走B' in df.columns:
        df['前走ブリンカー_装着'] = df['前走B'].notna().astype(float)
    if 'ブリンカー_装着' in df.columns and '前走ブリンカー_装着' in df.columns:
        # +1=今回から装着, -1=今回外した, 0=変化なし
        df['ブリンカー変更'] = df['ブリンカー_装着'] - df['前走ブリンカー_装着']

    # 5.5. 基本コース情報（早期計算）
    df['今回_surface']   = df['距離'].map(extract_surface)
    df['今回_距離_m']    = df['距離'].map(extract_dist_m)
    df['今回_馬場_num']  = df['馬場状態'].map(encode_baba)
    df['今回_会場']      = df['開催'].apply(extract_venue)
    df['今回_コース種別'] = df['今回_会場'] + '_' + df['今回_surface'].map(
        {0.0: 'ダ', 1.0: '芝', 2.0: '障'}
    ).fillna('不')

    # 季節・月
    df['月']  = (df['日付_num'] // 100 % 100).astype(float)
    df['季節'] = df['月'].map(lambda m: 2 if m in [3,4,5] else 3 if m in [6,7,8] else 4 if m in [9,10,11] else 1)

    # 5.7. タイム指数（コース×馬場内での偏差値、速い=高い）【NEW】
    print("タイム指数を計算しています...")
    if '走破タイム_sec' in df.columns:
        df['タイム指数'] = deviation_score_group(
            df['走破タイム_sec'],
            ['今回_会場', '今回_surface', '今回_距離_m', '今回_馬場_num'],
            df
        )

    # 上り3F指数（レース内での上がりの速さ、速い=高い）
    if '上り3F' in df.columns:
        df['上り3F_指数'] = np.nan
        group_keys = ['日付_num', '開催', 'Ｒ'] if 'Ｒ' in df.columns else ['日付_num', 'レース名']
        for _, grp in df.groupby(group_keys):
            vals = grp['上り3F'].dropna()
            if len(vals) < 3: continue
            mean, std = vals.mean(), vals.std()
            if std > 0:
                df.loc[grp.index, '上り3F_指数'] = 50 + 10 * (mean - grp['上り3F']) / std

    # 前走の上り3F指数（前走のレース内での上がり順位の代替）
    if '前走上り3F' in df.columns:
        df['前走上り3F'] = pd.to_numeric(df['前走上り3F'], errors='coerce')

    # 6. 騎手統計（ローリング200走・グローバル統計は中間値のみ使用し最終列から除去）
    print("騎手・調教師・血統の成績をエンコードしています...")
    if '騎手' in df.columns and '着順_num' in df.columns:
        jockey_stats = df.groupby('騎手')['着順_num'].agg(
            騎手_勝率   = lambda x: (x == 1).mean(),
            騎手_複勝率  = lambda x: (x <= 3).mean(),
            騎手_平均着順 = lambda x: x[x < 99].mean()
        ).reset_index()
        df = df.merge(jockey_stats, on='騎手', how='left')
        roll_j = calc_rolling_stats(df, '騎手', '日付_num', 200, 20, '騎手')
        df['騎手_r200_勝率']  = roll_j['騎手_r200_勝率'].fillna(df['騎手_勝率'])
        df['騎手_r200_複勝率'] = roll_j['騎手_r200_複勝率'].fillna(df['騎手_複勝率'])
        df.drop(columns=['騎手_勝率', '騎手_複勝率'], inplace=True)

    # 6.5. 調教師統計（ローリング200走・グローバル除去）
    if '調教師' in df.columns and '着順_num' in df.columns:
        trainer_stats = df.groupby('調教師')['着順_num'].agg(
            調教師_勝率  = lambda x: (x == 1).mean(),
            調教師_複勝率 = lambda x: (x <= 3).mean(),
        ).reset_index()
        df = df.merge(trainer_stats, on='調教師', how='left')
        roll_t = calc_rolling_stats(df, '調教師', '日付_num', 200, 20, '調教師')
        df['調教師_r200_勝率']  = roll_t['調教師_r200_勝率'].fillna(df['調教師_勝率'])
        df['調教師_r200_複勝率'] = roll_t['調教師_r200_複勝率'].fillna(df['調教師_複勝率'])
        df.drop(columns=['調教師_勝率', '調教師_複勝率'], inplace=True)

    # 6.7. 騎手×コース種別（ローリング100走・グローバル除去）
    if '騎手' in df.columns and '今回_コース種別' in df.columns:
        jc_stats = df.groupby(['騎手', '今回_コース種別'])['着順_num'].agg(
            騎手コース_勝率  = lambda x: (x == 1).mean() if len(x) >= 10 else np.nan,
            騎手コース_複勝率 = lambda x: (x <= 3).mean() if len(x) >= 10 else np.nan,
        ).reset_index()
        df = df.merge(jc_stats, on=['騎手', '今回_コース種別'], how='left')
        df['騎手コース_勝率']  = df['騎手コース_勝率'].fillna(df['騎手_r200_勝率'])
        df['騎手コース_複勝率'] = df['騎手コース_複勝率'].fillna(df['騎手_r200_複勝率'])
        roll_jc = calc_rolling_stats_combo(df, ['騎手','今回_コース種別'], '日付_num', 100, 15, '騎手コース')
        df['騎手コース_r100_勝率']  = roll_jc['騎手コース_r100_勝率'].fillna(df['騎手コース_勝率'])
        df['騎手コース_r100_複勝率'] = roll_jc['騎手コース_r100_複勝率'].fillna(df['騎手コース_複勝率'])
        df.drop(columns=['騎手コース_勝率', '騎手コース_複勝率'], inplace=True)

    # 6.8. 調教師×コース別統計（ローリング100走・グローバル除去）
    if '調教師' in df.columns and '今回_コース種別' in df.columns and '着順_num' in df.columns:
        tc_stats = df.groupby(['調教師', '今回_コース種別'])['着順_num'].agg(
            調教師コース_勝率  = lambda x: (x == 1).mean() if len(x) >= 10 else np.nan,
            調教師コース_複勝率 = lambda x: (x <= 3).mean() if len(x) >= 10 else np.nan,
        ).reset_index()
        df = df.merge(tc_stats, on=['調教師', '今回_コース種別'], how='left')
        df['調教師コース_勝率']  = df['調教師コース_勝率'].fillna(df['調教師_r200_勝率'])
        df['調教師コース_複勝率'] = df['調教師コース_複勝率'].fillna(df['調教師_r200_複勝率'])
        roll_tc = calc_rolling_stats_combo(df, ['調教師','今回_コース種別'], '日付_num', 100, 15, '調教師コース')
        df['調教師コース_r100_勝率']  = roll_tc['調教師コース_r100_勝率'].fillna(df['調教師コース_勝率'])
        df['調教師コース_r100_複勝率'] = roll_tc['調教師コース_r100_複勝率'].fillna(df['調教師コース_複勝率'])
        df.drop(columns=['調教師コース_勝率', '調教師コース_複勝率'], inplace=True)

    # 6.85. 騎手×馬場別統計（ローリング100走・グローバル除去）
    if '騎手' in df.columns and '今回_馬場_num' in df.columns and '着順_num' in df.columns:
        jb_stats = df.groupby(['騎手', '今回_馬場_num'])['着順_num'].agg(
            騎手馬場_勝率  = lambda x: (x == 1).mean() if len(x) >= 10 else np.nan,
            騎手馬場_複勝率 = lambda x: (x <= 3).mean() if len(x) >= 10 else np.nan,
        ).reset_index()
        df = df.merge(jb_stats, on=['騎手', '今回_馬場_num'], how='left')
        df['騎手馬場_勝率']  = df['騎手馬場_勝率'].fillna(df['騎手_r200_勝率'])
        df['騎手馬場_複勝率'] = df['騎手馬場_複勝率'].fillna(df['騎手_r200_複勝率'])
        roll_jb = calc_rolling_stats_combo(df, ['騎手', '今回_馬場_num'], '日付_num', 100, 15, '騎手馬場')
        df['騎手馬場_r100_勝率']  = roll_jb['騎手馬場_r100_勝率'].fillna(df['騎手馬場_勝率'])
        df['騎手馬場_r100_複勝率'] = roll_jb['騎手馬場_r100_複勝率'].fillna(df['騎手馬場_複勝率'])
        df.drop(columns=['騎手馬場_勝率', '騎手馬場_複勝率'], inplace=True)

    # 6.87. 騎手×会場別統計（ローリング100走・グローバル除去）
    if '騎手' in df.columns and '会場' in df.columns and '着順_num' in df.columns:
        jv_stats = df.groupby(['騎手', '会場'])['着順_num'].agg(
            騎手会場_勝率  = lambda x: (x == 1).mean() if len(x) >= 10 else np.nan,
            騎手会場_複勝率 = lambda x: (x <= 3).mean() if len(x) >= 10 else np.nan,
        ).reset_index()
        df = df.merge(jv_stats, on=['騎手', '会場'], how='left')
        df['騎手会場_勝率']  = df['騎手会場_勝率'].fillna(df['騎手_r200_勝率'])
        df['騎手会場_複勝率'] = df['騎手会場_複勝率'].fillna(df['騎手_r200_複勝率'])
        roll_jv = calc_rolling_stats_combo(df, ['騎手', '会場'], '日付_num', 100, 15, '騎手会場')
        df['騎手会場_r100_勝率']  = roll_jv['騎手会場_r100_勝率'].fillna(df['騎手会場_勝率'])
        df['騎手会場_r100_複勝率'] = roll_jv['騎手会場_r100_複勝率'].fillna(df['騎手会場_複勝率'])
        df.drop(columns=['騎手会場_勝率', '騎手会場_複勝率'], inplace=True)

    # 6.88. 騎手×距離帯別統計（ローリング100走・グローバル除去）
    if '騎手' in df.columns and '距離' in df.columns and '着順_num' in df.columns:
        def _dist_band(d):
            try:
                v = int(str(d).replace('m','').strip())
                if v <= 1400: return '短'
                elif v <= 1800: return 'マイル'
                elif v <= 2200: return '中'
                else: return '長'
            except: return None
        df['_dist_band'] = df['距離'].apply(_dist_band)
        jd_stats = df.groupby(['騎手', '_dist_band'])['着順_num'].agg(
            騎手距離_勝率  = lambda x: (x == 1).mean() if len(x) >= 10 else np.nan,
            騎手距離_複勝率 = lambda x: (x <= 3).mean() if len(x) >= 10 else np.nan,
        ).reset_index()
        df = df.merge(jd_stats, on=['騎手', '_dist_band'], how='left')
        df['騎手距離_勝率']  = df['騎手距離_勝率'].fillna(df['騎手_r200_勝率'])
        df['騎手距離_複勝率'] = df['騎手距離_複勝率'].fillna(df['騎手_r200_複勝率'])
        roll_jd = calc_rolling_stats_combo(df, ['騎手', '_dist_band'], '日付_num', 100, 15, '騎手距離')
        df['騎手距離_r100_勝率']  = roll_jd['騎手距離_r100_勝率'].fillna(df['騎手距離_勝率'])
        df['騎手距離_r100_複勝率'] = roll_jd['騎手距離_r100_複勝率'].fillna(df['騎手距離_複勝率'])
        df.drop(columns=['_dist_band', '騎手距離_勝率', '騎手距離_複勝率'], inplace=True)

    # 6.9. 血統統計
    overall_win   = (df['着順_num'] == 1).mean()
    overall_place = (df['着順_num'] <= 3).mean()
    for stallion_col, prefix in [('種牡馬', '種牡馬'), ('母父馬', '母父馬')]:
        if stallion_col in df.columns:
            stats = df.groupby(stallion_col)['着順_num'].agg(
                **{f'{prefix}_勝率':  lambda x: (x == 1).mean() if len(x) >= 30 else np.nan},
                **{f'{prefix}_複勝率': lambda x: (x <= 3).mean() if len(x) >= 30 else np.nan},
            ).reset_index()
            df = df.merge(stats, on=stallion_col, how='left')
            df[f'{prefix}_勝率']  = df[f'{prefix}_勝率'].fillna(overall_win)
            df[f'{prefix}_複勝率'] = df[f'{prefix}_複勝率'].fillna(overall_place)

    # 6.93. 種牡馬×芝ダ別統計
    if '種牡馬' in df.columns and '今回_surface' in df.columns:
        for surf_val, surf_name in [(0, '芝'), (1, 'ダ')]:
            sub = df[df['今回_surface'] == surf_val]
            stats = sub.groupby('種牡馬')['着順_num'].agg(
                **{f'種牡馬_{surf_name}_勝率':  lambda x: (x == 1).mean() if len(x) >= 20 else np.nan},
                **{f'種牡馬_{surf_name}_複勝率': lambda x: (x <= 3).mean() if len(x) >= 20 else np.nan},
            ).reset_index()
            df = df.merge(stats, on='種牡馬', how='left')
            df[f'種牡馬_{surf_name}_勝率']  = df[f'種牡馬_{surf_name}_勝率'].fillna(overall_win)
            df[f'種牡馬_{surf_name}_複勝率'] = df[f'種牡馬_{surf_name}_複勝率'].fillna(overall_place)

    # 6.95. 産地・生産者統計（min_count=50）
    for grp_col, prefix, min_cnt in [('産地', '産地', 50), ('生産者', '生産者', 30)]:
        if grp_col in df.columns:
            stats = df.groupby(grp_col)['着順_num'].agg(
                **{f'{prefix}_勝率':  lambda x: (x == 1).mean() if len(x) >= min_cnt else np.nan},
                **{f'{prefix}_複勝率': lambda x: (x <= 3).mean() if len(x) >= min_cnt else np.nan},
            ).reset_index()
            df = df.merge(stats, on=grp_col, how='left')
            df[f'{prefix}_勝率']  = df[f'{prefix}_勝率'].fillna(overall_win)
            df[f'{prefix}_複勝率'] = df[f'{prefix}_複勝率'].fillna(overall_place)

    # 6.97. 個馬×コース適性（馬名S基準）
    if '馬名S' in df.columns and '着順_num' in df.columns:
        # (a) 馬の全体ローリング成績（直近20走）: 現在の状態を反映
        roll_h = calc_rolling_stats(df, '馬名S', '日付_num', 20, 5, '馬')
        df['馬_r20_勝率']  = roll_h['馬_r20_勝率'].fillna(overall_win)
        df['馬_r20_複勝率'] = roll_h['馬_r20_複勝率'].fillna(overall_place)

        # (b) 馬×表面（芝/ダ）ローリング成績（直近20走, min3）: 表面適性
        if '今回_surface' in df.columns:
            roll_ms = calc_rolling_stats_combo(df, ['馬名S', '今回_surface'], '日付_num', 20, 3, '馬コース')
            df['馬コース_r20_勝率']  = roll_ms['馬コース_r20_勝率'].fillna(df['馬_r20_勝率'])
            df['馬コース_r20_複勝率'] = roll_ms['馬コース_r20_複勝率'].fillna(df['馬_r20_複勝率'])

        # (c) 馬×距離帯 全期間成績（min3走）: 距離適性（キャリア通じて変わりにくい）
        if '距離' in df.columns:
            def _hband(d):
                try:
                    v = int(str(d).replace('m', '').strip())
                    return '短' if v <= 1400 else 'マイル' if v <= 1800 else '中' if v <= 2200 else '長'
                except Exception:
                    return None
            df['_hband'] = df['距離'].apply(_hband)
            hd_stats = df.groupby(['馬名S', '_hband'])['着順_num'].agg(
                **{'馬距離_勝率':  lambda x: (x == 1).mean() if len(x) >= 3 else np.nan},
                **{'馬距離_複勝率': lambda x: (x <= 3).mean() if len(x) >= 3 else np.nan},
            ).reset_index()
            df = df.merge(hd_stats, on=['馬名S', '_hband'], how='left')
            df['馬距離_勝率']  = df['馬距離_勝率'].fillna(df['馬_r20_勝率'])
            df['馬距離_複勝率'] = df['馬距離_複勝率'].fillna(df['馬_r20_複勝率'])
            df.drop(columns=['_hband'], inplace=True)

    # 7. クラスランク
    df['クラス_rank'] = encode_class(df['レース名'])

    # 7.5. 海外レース行を数珠繋ぎチェーン用に挿入
    # overseas_supplement.csv の行を shift 前に追加し、shift 後に除去する。
    # これにより海外帰り馬の間隔・前走着順が正しく計算される。
    _overseas_path = os.path.join(master_dir, 'overseas_supplement.csv')
    _overseas_mask = pd.Series(False, index=df.index)  # 海外行フラグ（後で除去）
    if os.path.exists(_overseas_path):
        try:
            df_ov = read_csv(_overseas_path)
            if '馬名S' not in df_ov.columns and '馬名' in df_ov.columns:
                df_ov = df_ov.rename(columns={'馬名': '馬名S'})
            df_ov['_overseas'] = 1
            df_ov['日付_num'] = pd.to_numeric(df_ov['日付'], errors='coerce')
            # 着順_num を数値化
            df_ov['着順_num'] = clean_finish(df_ov['着順']) if '着順' in df_ov.columns else np.nan
            # クラス_rank: overseas_supplement の 'クラス推定' 列を使用
            if 'クラス推定' in df_ov.columns:
                df_ov['クラス_rank'] = pd.to_numeric(df_ov['クラス推定'], errors='coerce')
            # 既存 df に無い列を NaN で追加
            for _c in df.columns:
                if _c not in df_ov.columns:
                    df_ov[_c] = np.nan
            _common_ov = [c for c in df.columns if c in df_ov.columns]
            df_ov_trimmed = df_ov[_common_ov].copy()
            df_ov_trimmed['_overseas'] = 1
            df = pd.concat([df, df_ov_trimmed], ignore_index=True)
            df = df.sort_values(by=['馬名S', '日付_num']).reset_index(drop=True)
            _overseas_mask = df['_overseas'].fillna(0).astype(bool)
            print(f"  overseas_supplement: {len(df_ov_trimmed):,}行を数珠繋ぎチェーンに追加")
        except Exception as _e:
            print(f"  overseas_supplement 読み込みスキップ: {_e}")

    # 8. 過去10走のシフト生成
    print("過去10走の履歴データを生成しています...")
    cols_to_shift = [
        '着順_num', 'クラス_rank',
        '開催', '距離', '馬場状態', '頭数', '間隔', '馬体重', '馬体重増減', '斤量',
        '4角',
        '前走着差タイム',   # 2走前以降の着差タイムを取得
    ]
    # 新データの列を追加
    for col in ['タイム指数', '走破タイム_sec', '上り3F', '上り3F_指数', 'PCI', '馬番', 'RPCI',
                '単勝オッズ', '脚質_num']:
        if col in df.columns:
            cols_to_shift.append(col)

    existing_cols = [c for c in cols_to_shift if c in df.columns]
    shift_data = {}
    for i in range(1, 11):
        for col in existing_cols:
            shift_data[f'{i}走前_{col}'] = df.groupby('馬名S')[col].shift(i)
    df = pd.concat([df, pd.DataFrame(shift_data, index=df.index)], axis=1)

    # 海外行を除去（予測・訓練対象でないため）
    if _overseas_mask.any():
        df = df[~_overseas_mask].reset_index(drop=True)
        print(f"  海外行 {_overseas_mask.sum():,}行を除去")
    if '_overseas' in df.columns:
        df = df.drop(columns=['_overseas'])

    # 9. 近走サマリー
    print("近走成績サマリーを計算しています...")
    finish_cols_3  = [f'{i}走前_着順_num' for i in range(1,  4) if f'{i}走前_着順_num' in df.columns]
    finish_cols_5  = [f'{i}走前_着順_num' for i in range(1,  6) if f'{i}走前_着順_num' in df.columns]
    finish_cols_10 = [f'{i}走前_着順_num' for i in range(1, 11) if f'{i}走前_着順_num' in df.columns]
    class_cols_5   = [f'{i}走前_クラス_rank' for i in range(1, 6) if f'{i}走前_クラス_rank' in df.columns]
    heads_cols_5   = [f'{i}走前_頭数' for i in range(1, 6) if f'{i}走前_頭数' in df.columns]

    if finish_cols_3:
        r3 = df[finish_cols_3].replace(99, np.nan)
        df['近3走_平均着順'] = r3.mean(axis=1)
        df['近3走_複勝率']   = (r3 <= 3).sum(axis=1) / r3.notna().sum(axis=1).replace(0, np.nan)
        df['近3走_勝率']     = (r3 == 1).sum(axis=1) / r3.notna().sum(axis=1).replace(0, np.nan)

    if finish_cols_5:
        r5 = df[finish_cols_5].replace(99, np.nan)
        df['近5走_平均着順'] = r5.mean(axis=1)
        df['近5走_複勝率']   = (r5 <= 3).sum(axis=1) / r5.notna().sum(axis=1).replace(0, np.nan)

    if finish_cols_10:
        r10 = df[finish_cols_10].replace(99, np.nan)
        df['近10走_平均着順'] = r10.mean(axis=1)
        df['近10走_勝率']     = (r10 == 1).sum(axis=1) / r10.notna().sum(axis=1).replace(0, np.nan)
        df['近10走_複勝率']   = (r10 <= 3).sum(axis=1) / r10.notna().sum(axis=1).replace(0, np.nan)

    # クラス調整着順
    adj_finish_cols, class_diff_cols = [], []
    curr_class = df['クラス_rank'].replace(0, np.nan).fillna(2)
    for i in range(1, 6):
        fc, cc = f'{i}走前_着順_num', f'{i}走前_クラス_rank'
        if fc not in df.columns or cc not in df.columns: continue
        past_finish = df[fc].replace(99, np.nan)
        past_class  = df[cc].replace(0, np.nan).fillna(2)
        adj_col, diff_col = f'{i}走前_クラス調整着順', f'{i}走前_クラス差'
        df[adj_col]  = (past_finish * curr_class / past_class).clip(upper=20)
        df[diff_col] = past_class - curr_class
        adj_finish_cols.append(adj_col)
        class_diff_cols.append(diff_col)

    if adj_finish_cols:
        adj_df  = df[adj_finish_cols].replace(0, np.nan)
        diff_df = df[class_diff_cols]
        df['近5走_クラス調整_平均着順'] = adj_df.mean(axis=1)
        df['格上経験数_近5走']         = (diff_df > 0).sum(axis=1).astype(float)
        df['最大クラス差_近5走']        = diff_df.max(axis=1)

    if finish_cols_5 and class_cols_5 and len(finish_cols_5) == len(class_cols_5):
        scores = []
        for fc, cc in zip(finish_cols_5, class_cols_5):
            finish = df[fc].replace(99, np.nan)
            cls    = df[cc].fillna(2)
            scores.append(cls / finish)
        df['近5走_クラス補正スコア'] = pd.concat(scores, axis=1).mean(axis=1)

    # 9.4. タイム指数の近走平均【NEW】
    print("タイム指数・上り3F・着差タイムの近走集計を計算しています...")
    for col_base, new_col in [
        ('タイム指数',   '近5走_タイム指数平均'),
        ('走破タイム_sec', '近5走_走破タイム平均'),
        ('上り3F',      '近5走_上り3F平均'),
        ('上り3F_指数', '近5走_上り3F指数平均'),
        ('RPCI',        '近5走_RPCI平均'),
    ]:
        cols = [f'{i}走前_{col_base}' for i in range(1, 6) if f'{i}走前_{col_base}' in df.columns]
        if cols:
            df[new_col] = df[cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)

    # タイム指数の安定性・ベスト値
    ti_cols = [f'{i}走前_タイム指数' for i in range(1, 6) if f'{i}走前_タイム指数' in df.columns]
    if ti_cols:
        ti_df = df[ti_cols].apply(pd.to_numeric, errors='coerce')
        df['近5走_タイム指数_std']  = ti_df.std(axis=1)
        df['近5走_タイム指数_max']  = ti_df.max(axis=1)
        df['近5走_タイム指数_min']  = ti_df.min(axis=1)
        df['近5走_タイム指数_range'] = df['近5走_タイム指数_max'] - df['近5走_タイム指数_min']

    # 近5走上り3Fの安定性
    uri_cols = [f'{i}走前_上り3F' for i in range(1, 6) if f'{i}走前_上り3F' in df.columns]
    if uri_cols:
        uri_df = df[uri_cols].apply(pd.to_numeric, errors='coerce')
        df['近5走_上り3F_std'] = uri_df.std(axis=1)
        df['近5走_上り3F_min'] = uri_df.min(axis=1)  # 最速上り3F（小さいほど速い）

    # 着差タイムの近走平均
    td_cols = []
    if '前走着差タイム' in df.columns:
        td_cols.append(df['前走着差タイム'])
    for i in range(1, 5):
        c = f'{i}走前_前走着差タイム'
        if c in df.columns: td_cols.append(df[c])
    if td_cols:
        df['近5走_着差タイム平均'] = pd.concat(td_cols, axis=1).apply(pd.to_numeric, errors='coerce').mean(axis=1)

    # 着差タイムのクラス補正（G1で0.5秒負け ≠ 未勝利で0.5秒負け）
    # 補正式: 着差タイム × (今回クラス / 前走クラス) → クラスが上がるほど着差を小さく評価
    curr_class_f = df['クラス_rank'].replace(0, np.nan).fillna(2)
    adj_td_cols = []
    for i, td_col in enumerate([('前走着差タイム', None)] + [(f'{i}走前_前走着差タイム', f'{i}走前_クラス_rank') for i in range(1, 5)]):
        td_name, cls_name = td_col
        if td_name not in df.columns: continue
        td = pd.to_numeric(df[td_name], errors='coerce')
        if cls_name and cls_name in df.columns:
            past_cls = df[cls_name].replace(0, np.nan).fillna(2)
        else:
            # 前走着差タイムは前走クラスで補正（前クラス_rankを使用）
            past_cls = df['前クラス_rank'].replace(0, np.nan).fillna(2) if '前クラス_rank' in df.columns else pd.Series(2.0, index=df.index)
        adj_td = (td * curr_class_f / past_cls).clip(upper=5.0)  # 最大5秒でキャップ
        adj_col = f'着差タイム_クラス補正_{i}走前' if i > 0 else '前走着差タイム_クラス補正'
        df[adj_col] = adj_td
        adj_td_cols.append(adj_td)
    if adj_td_cols:
        df['近5走_着差タイム_クラス補正平均'] = pd.concat(adj_td_cols, axis=1).mean(axis=1)

    # 9.5. 頭数補正 × クラス補正着順
    rel_finish_vals = []
    for i, (fc, hc) in enumerate(zip(finish_cols_5, heads_cols_5), start=1):
        cc   = f'{i}走前_クラス_rank'
        fin  = df[fc].replace(99, np.nan)
        head = pd.to_numeric(df[hc], errors='coerce').replace(0, np.nan)
        if cc in df.columns:
            past_class = df[cc].replace(0, np.nan).fillna(2)
            rel_finish_vals.append((fin * curr_class / past_class).clip(upper=20) / head)
        else:
            rel_finish_vals.append(fin / head)
    if rel_finish_vals:
        df['近5走_平均相対着順'] = pd.concat(rel_finish_vals, axis=1).mean(axis=1)

    # 9.6. 脚質派生
    print("脚質・トレンドを計算しています...")
    if '前2角' in df.columns and '前4角' in df.columns:
        df['前走_追い上げ度'] = df['前2角'] - df['前4角']
        df['前走_4角位置']   = df['前4角']

    c4_cols = [f'{i}走前_4角' for i in range(1, 6) if f'{i}走前_4角' in df.columns]
    c2_cols = [f'{i}走前_2角' for i in range(1, 6) if f'{i}走前_2角' in df.columns]
    if c4_cols:
        c4_arr = df[c4_cols].apply(pd.to_numeric, errors='coerce')
        df['近5走_平均4角位置'] = c4_arr.mean(axis=1)
    # 近5走_平均追い上げ度は2角欠損率高いため除外

    # 9.63. 騎手×脚質グループ別統計
    # 近5走_平均4角位置から脚質を推定（当日実脚質はポスト情報のため使用しない）
    if '騎手' in df.columns and '着順_num' in df.columns:
        if '近5走_平均4角位置' in df.columns:
            jt_style = pd.cut(
                df['近5走_平均4角位置'], bins=[-1, 5, 10, 99],
                labels=['先行', '中団', '後方'], right=True
            ).astype(str).replace('nan', np.nan)
        else:
            jt_style = None
        if jt_style is not None:
            tmp = df[['騎手', '着順_num']].copy()
            tmp['_style'] = jt_style.values
            jt_stats = tmp.groupby(['騎手', '_style'])['着順_num'].agg(
                騎手脚質_勝率  = lambda x: (x == 1).mean() if len(x) >= 10 else np.nan,
                騎手脚質_複勝率 = lambda x: (x <= 3).mean() if len(x) >= 10 else np.nan,
            ).reset_index()
            df['_style'] = jt_style.values
            df = df.copy()
            df = df.merge(jt_stats, on=['騎手', '_style'], how='left')
            df['騎手脚質_勝率']  = df['騎手脚質_勝率'].fillna(df['騎手_r200_勝率'])
            df['騎手脚質_複勝率'] = df['騎手脚質_複勝率'].fillna(df['騎手_r200_複勝率'])
            roll_js = calc_rolling_stats_combo(df, ['騎手', '_style'], '日付_num', 100, 15, '騎手脚質')
            df['騎手脚質_r100_勝率']  = roll_js['騎手脚質_r100_勝率'].fillna(df['騎手脚質_勝率'])
            df['騎手脚質_r100_複勝率'] = roll_js['騎手脚質_r100_複勝率'].fillna(df['騎手脚質_複勝率'])
            df.drop(columns=['_style', '騎手脚質_勝率', '騎手脚質_複勝率'], inplace=True)

    # 9.65. コース先行有利度 × 脚質フィット
    # 過去レースの入着馬（3着以内）の4角位置の平均 = そのコースで有利な位置取り
    if '4角' in df.columns and '着順_num' in df.columns and '今回_距離_m' in df.columns:
        course_style = (
            df[df['着順_num'] <= 3]
            .groupby(['今回_会場', '今回_surface', '今回_距離_m'])['4角']
            .mean()
            .reset_index()
            .rename(columns={'4角': 'コース_先行有利度'})
        )
        df = df.merge(course_style, on=['今回_会場', '今回_surface', '今回_距離_m'], how='left')
        # 馬の平均脚質（4角位置）とコース先行有利度の差（0=完全一致、大きい=脚質ミスマッチ）
        if '近5走_平均4角位置' in df.columns:
            df['脚質フィット'] = -(df['近5走_平均4角位置'] - df['コース_先行有利度']).abs()

    # 9.67. 展開予想特徴量（レース内脚質構成 × コース展開との掛け合わせ）
    if '近5走_平均4角位置' in df.columns and '頭数' in df.columns:
        heads = df['頭数'].clip(lower=1)

        # (a) 馬の推定脚質率（0=完全な逃げ, 1=完全な追い込み）
        df['推定_脚質率'] = (df['近5走_平均4角位置'] / heads).clip(0, 1)

        # (b) レース内の先行馬・逃げ馬カウント
        df['_is_nige']    = (df['近5走_平均4角位置'] <= 2).astype(float)
        df['_is_senkou']  = (df['近5走_平均4角位置'] <= 5).astype(float)

        race_grp = df.groupby(['日付', '開催']).agg(
            レース内_逃げ馬数=('_is_nige',   'sum'),
            レース内_先行馬数=('_is_senkou', 'sum'),
            レース内_平均脚質=('近5走_平均4角位置', 'mean'),
            レース内_脚質std=('近5走_平均4角位置',  'std'),
        ).reset_index()
        df = df.merge(race_grp, on=['日付', '開催'], how='left')

        # (c) 推定ペーススコア: 先行馬比率（高い=ハイペース=後方有利）
        df['推定ペース'] = df['レース内_先行馬数'] / heads

        # (d) 展開フィットスコア
        # スロー(推定ペース低)→先行有利, ハイ(高)→後方有利
        # 最適脚質率 = 1 - 推定ペース（スローなら0=先行が最適）
        optimal = 1 - df['推定ペース']
        df['展開フィット_v2'] = 1 - (df['推定_脚質率'] - optimal).abs()

        # (e) レース内相対脚質（自馬がこのフィールドで相対的に前めか後めか）
        df['レース内_相対脚質'] = df['レース内_平均脚質'] - df['近5走_平均4角位置']
        # 正=このフィールドでは相対的に前、負=相対的に後ろ

        # (f) コース先行有利度 × 推定ペース の交互作用
        # コースが先行有利（コース_先行有利度小）かつスロー展開（推定ペース低）→ 先行馬に最大恩恵
        if 'コース_先行有利度' in df.columns:
            course_pace_ratio = df['コース_先行有利度'] / heads
            df['コース展開マッチ'] = 1 - (df['推定ペース'] - course_pace_ratio).abs()
            # 馬の脚質 × コース×展開の3次交互作用
            df['展開_コース_脚質フィット'] = df['展開フィット_v2'] * df['コース展開マッチ']

        df.drop(columns=['_is_nige', '_is_senkou'], inplace=True)

    # 9.7. 近走トレンド（強化版）
    def _calc_slope(arr2d):
        """各行に対してpolyfit slope を計算。有効値2点以上で計算、それ以外はNaN"""
        slopes = np.full(len(arr2d), np.nan)
        for i, row in enumerate(arr2d):
            valid = ~np.isnan(row)
            if valid.sum() >= 2:
                x = np.where(valid)[0].astype(float)
                slopes[i] = np.polyfit(x, row[valid], 1)[0]
        return slopes

    # 着順トレンド（近5走・近3走）
    ord_cols3 = [f'{i}走前_着順_num' for i in range(1, 4) if f'{i}走前_着順_num' in df.columns]
    ord_cols5 = [f'{i}走前_着順_num' for i in range(1, 6) if f'{i}走前_着順_num' in df.columns]
    if len(ord_cols3) >= 2:
        arr = df[ord_cols3].apply(pd.to_numeric, errors='coerce').replace(99, np.nan).values
        df['着順_近3走_slope'] = _calc_slope(arr)   # 負=改善中
    if len(ord_cols5) >= 3:
        arr = df[ord_cols5].apply(pd.to_numeric, errors='coerce').replace(99, np.nan).values
        df['着順_近5走_slope'] = _calc_slope(arr)
    if '1走前_着順_num' in df.columns and '3走前_着順_num' in df.columns:
        f1 = pd.to_numeric(df['1走前_着順_num'], errors='coerce').replace(99, np.nan)
        f3 = pd.to_numeric(df['3走前_着順_num'], errors='coerce').replace(99, np.nan)
        df['近走_改善トレンド'] = f3 - f1   # 後方互換で残す

    # タイム指数トレンド（近3走・近5走）
    ti_cols3 = [f'{i}走前_タイム指数' for i in range(1, 4) if f'{i}走前_タイム指数' in df.columns]
    ti_cols5 = [f'{i}走前_タイム指数' for i in range(1, 6) if f'{i}走前_タイム指数' in df.columns]
    if len(ti_cols3) >= 2:
        arr = df[ti_cols3].apply(pd.to_numeric, errors='coerce').values
        df['タイム指数_近3走_slope'] = _calc_slope(arr)   # 後方互換で残す
    if len(ti_cols5) >= 3:
        arr = df[ti_cols5].apply(pd.to_numeric, errors='coerce').values
        df['タイム指数_近5走_slope'] = _calc_slope(arr)   # 正=上昇傾向

    # タイム指数の加速度（直近2走の改善幅 - 2-3走前の改善幅）
    if all(f'{i}走前_タイム指数' in df.columns for i in range(1, 4)):
        ti1 = pd.to_numeric(df['1走前_タイム指数'], errors='coerce')
        ti2 = pd.to_numeric(df['2走前_タイム指数'], errors='coerce')
        ti3 = pd.to_numeric(df['3走前_タイム指数'], errors='coerce')
        df['タイム指数_加速度'] = (ti1 - ti2) - (ti2 - ti3)  # 正=加速改善

    # 上り3F トレンド（近3走）
    ur_cols3 = [f'{i}走前_上り3F' for i in range(1, 4) if f'{i}走前_上り3F' in df.columns]
    if len(ur_cols3) >= 2:
        arr = df[ur_cols3].apply(pd.to_numeric, errors='coerce').values
        # 上り3Fは小さいほど良い → 負のslopeが改善
        df['上り3F_近3走_slope'] = _calc_slope(arr)

    # 4角位置トレンド（近3走）: 正=後方化, 負=前進化
    k4_cols3 = [f'{i}走前_4角' for i in range(1, 4) if f'{i}走前_4角' in df.columns]
    if len(k4_cols3) >= 2:
        arr = df[k4_cols3].apply(pd.to_numeric, errors='coerce').values
        df['4角位置_近3走_slope'] = _calc_slope(arr)   # 負=前進傾向

    # 近走の着順安定性（近5走着順のstd）
    if len(ord_cols5) >= 3:
        arr = df[ord_cols5].apply(pd.to_numeric, errors='coerce').replace(99, np.nan)
        df['着順_近5走_std'] = arr.std(axis=1)

    # 近走連続入着数（1着〜3着が何走続いているか）
    consec = pd.Series(0, index=df.index)
    for i in range(1, 6):
        col = f'{i}走前_着順_num'
        if col not in df.columns:
            break
        val = pd.to_numeric(df[col], errors='coerce').replace(99, np.nan)
        in_money = (val <= 3).fillna(False)
        # i走前まで全て入着なら加算
        if i == 1:
            consec = in_money.astype(int)
        else:
            prev_col = f'{i-1}走前_着順_num'
            prev_all = pd.to_numeric(df[prev_col], errors='coerce').replace(99, np.nan)
            consec = consec + (in_money & (consec >= i - 1)).astype(int)
    df['近走連続入着数'] = consec.clip(0, 5)

    # 着順加速度（着順での加速度: 負=急改善）
    if all(f'{i}走前_着順_num' in df.columns for i in range(1, 4)):
        o1 = pd.to_numeric(df['1走前_着順_num'], errors='coerce').replace(99, np.nan)
        o2 = pd.to_numeric(df['2走前_着順_num'], errors='coerce').replace(99, np.nan)
        o3 = pd.to_numeric(df['3走前_着順_num'], errors='coerce').replace(99, np.nan)
        df['着順_加速度'] = (o1 - o2) - (o2 - o3)  # 負=加速的に改善中

    # 馬番の内外枠分類【NEW】
    if '馬番' in df.columns:
        df['内外枠'] = df['馬番'].map(lambda x: 0 if x <= 3 else 2 if x >= 14 else 1 if pd.notna(x) else np.nan)

    # 9.8. キャリア段階特徴量
    if 'キャリア' in df.columns:
        df['キャリア_log'] = np.log1p(df['キャリア'])          # 対数変換（初戦付近で感度高め）
        df['キャリア_浅い'] = (df['キャリア'] <= 5).astype(float)  # 5走以下の若駒フラグ

    # 9.9. 騎手乗り替わり時の近走成績変化
    if '騎手変更' in df.columns and '近3走_平均着順' in df.columns:
        df['乗替り_近走不振'] = (
            (df['騎手変更'] == 1) & (df['近3走_平均着順'] > 6)
        ).astype(float)

    # 9.10. 斤量変化（今回 - 前走）
    if '斤量' in df.columns and '前走斤量' in df.columns:
        df['斤量変化'] = pd.to_numeric(df['斤量'], errors='coerce') - pd.to_numeric(df['前走斤量'], errors='coerce')

    # 9.11. 前走クラス変化（昇級=プラス、降級=マイナス）
    if 'クラス_rank' in df.columns and '前クラス_rank' in df.columns:
        df['クラス変化'] = df['クラス_rank'] - df['前クラス_rank'].replace(0, np.nan)
        df['昇級フラグ'] = (df['クラス変化'] > 0).astype(float)
        df['降級フラグ'] = (df['クラス変化'] < 0).astype(float)

    # 9.12. 連闘フラグ（間隔1週以下）・長期休養明けフラグ（13週以上）
    if '間隔' in df.columns:
        interval = pd.to_numeric(df['間隔'], errors='coerce')
        df['連闘フラグ']     = (interval <= 1).astype(float)
        df['休み明けフラグ'] = (interval >= 13).astype(float)

    # 9.13. 初コースフラグ（このコースで近5走以内に出走なし）
    if '同会場_出走数_近5走' in df.columns:
        df['初コースフラグ'] = (df['同会場_出走数_近5走'] == 0).astype(float)

    # 9.14. 馬体重トレンド（近5走の馬体重変化 = 1走前 - 5走前）
    w1 = df.get('1走前_馬体重')
    w5 = df.get('5走前_馬体重')
    if w1 is not None and w5 is not None:
        df['馬体重トレンド_近5走'] = (
            pd.to_numeric(w1, errors='coerce') - pd.to_numeric(w5, errors='coerce')
        )
    # 近3走の馬体重増減の合計（絞れているか太ってきているか）
    inc_cols = [f'{i}走前_馬体重増減' for i in range(1, 4) if f'{i}走前_馬体重増減' in df.columns]
    if inc_cols:
        df['近3走_体重増減合計'] = df[inc_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)

    # 10. コース適性・距離適性・馬場適性
    print("コース・距離・馬場の適性特徴量を計算しています...")
    if '1走前_距離' in df.columns:
        df['前走_surface']  = df['1走前_距離'].map(extract_surface)
        df['前走_距離_m']   = df['1走前_距離'].map(extract_dist_m)
        df['距離変化_前走'] = df['今回_距離_m'] - df['前走_距離_m']
        df['芝ダ転向']      = (df['今回_surface'] != df['前走_surface']).astype(float)

    surfaces, dists, finishes, babas = [], [], [], []
    for i in range(1, 6):
        dc, fc, bc = f'{i}走前_距離', f'{i}走前_着順_num', f'{i}走前_馬場状態'
        if dc in df.columns:
            surfaces.append(df[dc].map(extract_surface))
            dists.append(df[dc].map(extract_dist_m))
        if fc in df.columns: finishes.append(df[fc].replace(99, np.nan))
        if bc in df.columns: babas.append(df[bc].map(encode_baba))

    n = min(len(surfaces), len(finishes))
    if n > 0:
        surf_arr   = np.column_stack([s.values for s in surfaces[:n]])
        finish_arr = np.column_stack([f.values for f in finishes[:n]])
        curr_surf  = df['今回_surface'].values[:, None]
        curr_dist  = df['今回_距離_m'].values[:, None]
        curr_baba  = df['今回_馬場_num'].values[:, None]

        surf_match = (surf_arr == curr_surf)
        df['芝ダ一致数_近5走']       = surf_match.sum(axis=1).astype(float)
        df['芝ダ一致_平均着順_近5走'] = np.nanmean(np.where(surf_match, finish_arr, np.nan), axis=1)

        if len(dists) >= n:
            dist_arr  = np.column_stack([d.values for d in dists[:n]])
            dist_diff = dist_arr - curr_dist
            df['同距離帯_平均着順_近5走']   = np.nanmean(np.where(np.abs(dist_diff) <= 200, finish_arr, np.nan), axis=1)
            df['距離短縮時_平均着順_近5走'] = np.nanmean(np.where(dist_diff > 200,  finish_arr, np.nan), axis=1)
            df['距離延長時_平均着順_近5走'] = np.nanmean(np.where(dist_diff < -200, finish_arr, np.nan), axis=1)

        if len(babas) >= n:
            baba_arr   = np.column_stack([b.values for b in babas[:n]])
            baba_match = (baba_arr == curr_baba)
            df['同馬場_平均着順_近5走'] = np.nanmean(np.where(baba_match, finish_arr, np.nan), axis=1)
            df['良馬場_平均着順_近5走'] = np.nanmean(np.where(baba_arr == 0, finish_arr, np.nan), axis=1)
            df['道悪_平均着順_近5走']   = np.nanmean(np.where(baba_arr >= 2, finish_arr, np.nan), axis=1)
            df['馬場適性差_近5走']      = df['良馬場_平均着順_近5走'] - df['道悪_平均着順_近5走']

    # 11. 同会場成績（クラス補正あり）
    print("同会場成績を計算しています...")
    venue_adj_list, venue_raw_list = [], []
    for i in range(1, 6):
        oc, fc, cc = f'{i}走前_開催', f'{i}走前_着順_num', f'{i}走前_クラス_rank'
        if oc not in df.columns or fc not in df.columns: continue
        match  = (df[oc].apply(extract_venue) == df['今回_会場']).values
        fin    = df[fc].replace(99, np.nan)
        if cc in df.columns:
            past_class = df[cc].replace(0, np.nan).fillna(2)
            adj_fin    = (fin * curr_class / past_class).clip(upper=20)
        else:
            adj_fin = fin
        venue_adj_list.append(np.where(match, adj_fin.values, np.nan))
        venue_raw_list.append(np.where(match, fin.values, np.nan))

    if venue_adj_list:
        adj_arr = np.column_stack(venue_adj_list)
        raw_arr = np.column_stack(venue_raw_list)
        df['同会場_平均着順_近5走'] = np.nanmean(adj_arr, axis=1)
        df['同会場_複勝率_近5走']   = np.nanmean(raw_arr <= 3, axis=1)
        df['同会場_出走数_近5走']   = np.sum(~np.isnan(raw_arr), axis=1).astype(float)

    # 12. 相手レベル
    print("相手レベルを計算しています...")
    if '近5走_クラス調整_平均着順' in df.columns:
        race_key = ['日付_num', 'レース名']
        grp      = df.groupby(race_key)['近5走_クラス調整_平均着順']
        race_sum = grp.transform('sum')
        race_cnt = grp.transform('count')
        horse_val = df['近5走_クラス調整_平均着順'].fillna(race_sum / race_cnt.replace(0, np.nan))
        df['相手レベル_平均着順'] = (race_sum - horse_val) / (race_cnt - 1).replace(0, np.nan)
        df['相手レベル_実力差']   = grp.transform('std')

    # 13. 不要列を削除（高欠損・計算不可）
    drop_cols = ['多頭出し', '市場取引価格_万', '近5走_平均追い上げ度']
    drop_cols += [f'{i}走前_2角' for i in range(1, 11)]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # 14. 保存
    print(f"全会場の統合データを作成しました。列数: {len(df.columns)} / 行数: {len(df):,}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"--- 処理完了！ファイル保存先: {output_file} ---")

if __name__ == "__main__":
    main()
