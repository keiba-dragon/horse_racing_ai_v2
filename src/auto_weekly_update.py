# -*- coding: utf-8 -*-
"""
週次自動更新: JV-Linkから結果を取得してparquetを再生成する。

使い方:
  python src/auto_weekly_update.py            # fetch + 変換 + parquet再生成
  python src/auto_weekly_update.py --no-fetch # fetch済みのresultsを変換するだけ

月曜 06:00 にタスクスケジューラから実行する想定。
ターゲットFrontier（JV-Link）が起動している必要がある。
"""
import sys, io, os, subprocess, argparse, glob as _glob
import pandas as pd
import numpy as np

UM_DATA_DIR    = r'C:\TFJV\UM_DATA'
MASTER_HORSE   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'data', 'raw', 'master', 'master_horse.csv')
UM_REC_LEN     = 1609  # bytes per UM record in TARGET Frontier local DB

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR      = os.path.join(BASE_DIR, 'data', 'raw', 'results')
OVERSEAS_DIR     = os.path.join(BASE_DIR, 'data', 'raw', 'overseas')
SUPPLEMENT_PATH  = os.path.join(BASE_DIR, 'data', 'raw', 'master', 'results_supplement.csv')
OVERSEAS_SUPP    = os.path.join(BASE_DIR, 'data', 'raw', 'master', 'overseas_supplement.csv')
PARQUET_PATH     = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MAKE_FEATURES    = os.path.join(BASE_DIR, 'src', '01_make_features.py')
FETCH_SCRIPT     = os.path.join(BASE_DIR, 'src', 'fetch.py')
FETCH_OVERSEAS   = os.path.join(BASE_DIR, 'src', 'fetch_overseas.py')
TRAIN_CLOGIT     = os.path.join(BASE_DIR, 'src', 'save_conditional_logit.py')
TRAIN_FINAL      = os.path.join(BASE_DIR, 'src', 'save_final_model.py')


def sec_to_jravan(t):
    """走破タイム (小数秒) → clean_race_time が読める整数形式 (1分11秒0 → 1110)"""
    try:
        t = float(t)
        if pd.isna(t) or t <= 0:
            return np.nan
        m = int(t // 60)
        s = t % 60
        return int(m * 1000 + round(s * 10))
    except Exception:
        return np.nan


def jravan_to_sec(v):
    """JRA-VAN整数形式 → 秒 (1458 → 105.8)"""
    try:
        v = int(float(v))
        if v <= 0: return np.nan
        return (v // 1000) * 60 + (v % 1000) / 10
    except Exception:
        return np.nan


def calc_pace_metrics(df):
    """走破タイム・上り3F・距離 からペース指標を計算して列を追加する。"""
    dist_num = df['距離'].astype(str).str.extract(r'(\d+)')[0].pipe(pd.to_numeric, errors='coerce')
    soha_sec = df['走破タイム'].apply(jravan_to_sec)
    last3f   = pd.to_numeric(df['上り3F'], errors='coerce')
    first_sec = soha_sec - last3f
    denom     = dist_num - 600

    df['Ave-3F']         = np.where(denom > 0,    first_sec * 600 / denom, np.nan)
    df['PCI']            = np.where(last3f > 0,   50 + (df['Ave-3F'] - last3f) / last3f * 100, np.nan)
    df['平均速度']        = np.where(soha_sec > 0, dist_num / soha_sec * 3.6, np.nan)
    df['-3F平均速度']     = np.where(first_sec > 0, (dist_num - 600) / first_sec * 3.6, np.nan)
    df['上り3F平均速度']  = np.where(last3f > 0,   600 / last3f * 3.6, np.nan)
    return df


def estimate_style_num(pos4, tosu):
    """4角通過順と頭数から脚質_num を推定 (0=逃げ 1=先行 2=差し 3=追込)"""
    try:
        p, n = float(pos4), float(tosu)
        if pd.isna(p) or pd.isna(n) or n <= 0:
            return np.nan
        r = p / n
        if r <= 0.20: return 0.0
        if r <= 0.40: return 1.0
        if r <= 0.70: return 2.0
        return 3.0
    except Exception:
        return np.nan


def _um_extract_name(rec_bytes, byte_start):
    """UM レコードのバイト位置から馬名を抽出（Shift-JIS 36バイト、空白除去）。"""
    b = rec_bytes[byte_start:byte_start + 36]
    try:
        return b.decode('cp932', errors='replace').replace('　', '').strip()
    except Exception:
        return ''


def update_master_horse():
    """C:\\TFJV\\UM_DATA の UM*.DAT から新馬を master_horse.csv に追記する。"""
    if not os.path.isdir(UM_DATA_DIR):
        print('  UM_DATA ディレクトリが見つかりません。スキップします。')
        return 0

    # 現在の master_horse の馬名セット
    known_horses: set = set()
    if os.path.exists(MASTER_HORSE):
        try:
            mh = pd.read_csv(MASTER_HORSE, usecols=['馬名S'], low_memory=False,
                             encoding='cp932')
        except Exception:
            mh = pd.read_csv(MASTER_HORSE, usecols=['馬名S'], low_memory=False)
        known_horses = set(mh['馬名S'].dropna().unique())
    print(f'  master_horse 既存馬数: {len(known_horses):,}')

    new_rows = []
    scan_years = range(2020, 2026)  # 近年デビュー馬をカバー

    for year in scan_years:
        year_dir = os.path.join(UM_DATA_DIR, str(year))
        if not os.path.isdir(year_dir):
            continue
        for dat_path in sorted(_glob.glob(os.path.join(year_dir, 'UM*.DAT'))):
            try:
                with open(dat_path, 'rb') as f:
                    raw = f.read()
            except Exception as e:
                print(f'  読込失敗: {dat_path}: {e}')
                continue
            for offset in range(0, len(raw) - UM_REC_LEN + 1, UM_REC_LEN):
                rec = raw[offset:offset + UM_REC_LEN]
                if rec[:2] != b'UM':
                    continue
                horse = _um_extract_name(rec, 46)
                if not horse or horse in known_horses:
                    continue
                sire     = _um_extract_name(rec, 214)
                dam_sire = _um_extract_name(rec, 398)
                new_rows.append({'馬名S': horse, '種牡馬': sire, '母父馬': dam_sire})
                known_horses.add(horse)

    if not new_rows:
        print('  新規馬なし')
        return 0

    df_new = pd.DataFrame(new_rows)

    if os.path.exists(MASTER_HORSE):
        # 既存ファイルの列構造を読んで全列を揃える（列順ミスを防ぐ）
        try:
            existing_cols = pd.read_csv(MASTER_HORSE, nrows=0, encoding='cp932').columns.tolist()
        except Exception:
            existing_cols = pd.read_csv(MASTER_HORSE, nrows=0).columns.tolist()
        import numpy as np
        for col in existing_cols:
            if col not in df_new.columns:
                df_new[col] = np.nan
        df_new = df_new[existing_cols]
        df_new.to_csv(MASTER_HORSE, mode='a', header=False, index=False,
                      encoding='cp932', errors='replace')
    else:
        df_new.to_csv(MASTER_HORSE, mode='w', header=True, index=False,
                      encoding='cp932', errors='replace')

    print(f'  master_horse 更新: +{len(df_new):,}頭')
    return len(df_new)


def run_fetch():
    """fetch.py --incremental を実行してdata/raw/results/に保存。"""
    print(f'  fetch.py --incremental を実行中...')
    r = subprocess.run(
        [sys.executable, FETCH_SCRIPT, '--incremental'],
        cwd=BASE_DIR
    )
    return r.returncode == 0


def run_fetch_overseas():
    """fetch_overseas.py --incremental を実行してdata/raw/overseas/に保存。"""
    print(f'  fetch_overseas.py --incremental を実行中...')
    r = subprocess.run(
        [sys.executable, FETCH_OVERSEAS, '--incremental'],
        cwd=BASE_DIR
    )
    if r.returncode != 0:
        print('  海外競走フェッチ失敗（未購読の場合は正常）。スキップします。')
        return False
    return True


def convert_overseas():
    """data/raw/overseas/YYYYMMDD.csv → overseas_supplement.csv に変換・追記。"""
    existing_max = 0
    if os.path.exists(OVERSEAS_SUPP):
        try:
            df_ex = pd.read_csv(OVERSEAS_SUPP, usecols=['日付'], low_memory=False)
            v = pd.to_numeric(df_ex['日付'], errors='coerce').max()
            if pd.notna(v):
                existing_max = int(v)
        except Exception:
            pass
    if existing_max:
        print(f'  既存overseas_supplement 最終日付: {existing_max}')

    if not os.path.isdir(OVERSEAS_DIR):
        print('  overseas/ ディレクトリなし。スキップ。')
        return 0

    result_files = sorted(_glob.glob(os.path.join(OVERSEAS_DIR, '????????.csv')))
    new_frames = []

    for fp in result_files:
        stem = os.path.basename(fp).replace('.csv', '')
        if len(stem) != 8 or not stem.isdigit():
            continue
        year, mmdd = int(stem[:4]), int(stem[4:])
        date_num = (year - 2000) * 10000 + mmdd

        if date_num <= existing_max:
            continue

        try:
            df = pd.read_csv(fp, encoding='utf-8', low_memory=False)
        except Exception:
            continue

        if df.empty:
            continue

        df = df.rename(columns={
            '馬名':    '馬名S',
            'レースNo': 'Ｒ',
        })
        df['日付'] = date_num
        df['_overseas'] = 1

        if '距離' in df.columns and '芝ダ' in df.columns:
            _surf = df['芝ダ'].astype(str).str.strip()
            _dist = df['距離'].astype(str).str.strip()
            df['距離'] = _surf + _dist

        new_frames.append(df)
        print(f'  overseas {stem}: {len(df)}行')

    if not new_frames:
        print('  海外新規データなし')
        return 0

    df_new = pd.concat(new_frames, ignore_index=True)
    mode   = 'a' if existing_max > 0 and os.path.exists(OVERSEAS_SUPP) else 'w'
    header = (mode == 'w')
    df_new.to_csv(OVERSEAS_SUPP, mode=mode, header=header, index=False, encoding='utf-8')
    print(f'  overseas_supplement 更新: +{len(df_new):,}行 (mode={mode})')
    return len(df_new)


def convert_results():
    """data/raw/results/YYYYMMDD.csv → results_supplement.csv に変換・追記。"""
    existing_max = 0
    if os.path.exists(SUPPLEMENT_PATH):
        try:
            df_ex = pd.read_csv(SUPPLEMENT_PATH, usecols=['日付'], low_memory=False)
            v = pd.to_numeric(df_ex['日付'], errors='coerce').max()
            if pd.notna(v):
                existing_max = int(v)
        except Exception:
            pass
    if existing_max:
        print(f'  既存supplement 最終日付: {existing_max}')

    result_files = sorted(_glob.glob(os.path.join(RESULTS_DIR, '????????.csv')))
    new_frames = []

    for fp in result_files:
        stem = os.path.basename(fp).replace('.csv', '')   # '20260510'
        if len(stem) != 8 or not stem.isdigit():
            continue
        year, mmdd = int(stem[:4]), int(stem[4:])
        date_num = (year - 2000) * 10000 + mmdd           # 260510

        if date_num <= existing_max:
            continue

        try:
            df = pd.read_csv(fp, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(fp, encoding='cp932', low_memory=False)

        if df.empty:
            continue

        # ── 列名マッピング ──
        df = df.rename(columns={
            '馬名':      '馬名S',
            '芝ダ':      '芝・ダ',
            '馬体重変化': '馬体重増減',
            'レースNo':  'Ｒ',
            '騎手名':    '騎手',
        })

        # 日付: YYYYMMDD → YYMMDD 整数
        df['日付'] = date_num

        # 距離: 数字のみ → "芝1600" 形式
        if '距離' in df.columns and '芝・ダ' in df.columns:
            _surf = df['芝・ダ'].astype(str).str.strip()
            _dist = df['距離'].astype(str).str.strip()
            df['距離'] = _surf + _dist

        # 走破タイム: fetch.pyがJRA-VAN整数形式で格納するため変換不要
        # (例: "1458" = 1:45.8 → そのまま使用)

        # 脚質_num: 4角通過順から推定
        if '4角' in df.columns and '頭数' in df.columns:
            df['脚質_num'] = df.apply(
                lambda r: estimate_style_num(r.get('4角'), r.get('頭数')), axis=1
            )

        # ペース指標: 走破タイム・上り3F・距離 から計算
        if {'走破タイム', '上り3F', '距離'}.issubset(df.columns):
            df = calc_pace_metrics(df)
        # RPCI: レース内のPCI平均（レースペース指標）
        if 'PCI' in df.columns and '会場コード' in df.columns and 'Ｒ' in df.columns:
            df['RPCI'] = df.groupby(['会場コード', 'Ｒ'])['PCI'].transform('mean').round(1)

        new_frames.append(df)
        print(f'  {stem}: {len(df)}行')

    if not new_frames:
        print('  新規データなし')
        return 0

    df_new = pd.concat(new_frames, ignore_index=True)
    mode   = 'a' if existing_max > 0 and os.path.exists(SUPPLEMENT_PATH) else 'w'
    header = (mode == 'w')
    df_new.to_csv(SUPPLEMENT_PATH, mode=mode, header=header, index=False, encoding='utf-8')
    print(f'  supplement 更新: +{len(df_new):,}行 (mode={mode})')
    return len(df_new)


def rebuild_parquet():
    """01_make_features.py を実行してparquetを再生成する。"""
    print('  01_make_features.py を実行中（25〜30分）...')
    r = subprocess.run(
        [sys.executable, MAKE_FEATURES],
        cwd=BASE_DIR
    )
    if r.returncode != 0:
        return False
    # CSV → parquet 変換
    csv_path = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.csv')
    if os.path.exists(csv_path):
        import pandas as pd
        print('  CSV → parquet 変換中...')
        pd.read_csv(csv_path, low_memory=False).to_parquet(PARQUET_PATH, index=False)
        print('  parquet 更新完了')
    return True


def retrain_model():
    """clogit再学習 → roi_model.pkl 生成（完全自動化パイプライン）"""
    print('  save_conditional_logit.py を実行中（30〜60分）...')
    r = subprocess.run([sys.executable, TRAIN_CLOGIT], cwd=BASE_DIR)
    if r.returncode != 0:
        print('  ERROR: clogit学習失敗')
        return False
    print('  save_final_model.py を実行中...')
    r = subprocess.run([sys.executable, TRAIN_FINAL], cwd=BASE_DIR)
    if r.returncode != 0:
        print('  ERROR: final_model生成失敗')
        return False
    print('  モデル更新完了')
    return True


def delete_parquet():
    if os.path.exists(PARQUET_PATH):
        os.remove(PARQUET_PATH)
        print('  parquet削除（次回予測時に再生成されます）')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-fetch',   action='store_true', help='fetchをスキップ（変換・再生成のみ）')
    ap.add_argument('--no-rebuild', action='store_true', help='parquet再生成をスキップ（削除のみ）')
    ap.add_argument('--no-train',   action='store_true', help='モデル再学習をスキップ')
    args = ap.parse_args()

    print('=' * 50)
    print('  競馬AI 週次自動更新')
    print('=' * 50)

    # Step 1: JV-Link fetch
    if not args.no_fetch:
        print('\n[1/4] JV-Link から結果を取得...')
        if not run_fetch():
            print('  ERROR: fetch失敗。ターゲットFrontierが起動しているか確認してください。')
            sys.exit(1)
        print('\n[1b/4] 海外競走データを取得...')
        run_fetch_overseas()  # 失敗しても続行
    else:
        print('\n[1/4] fetch スキップ')

    # Step 2: 変換
    print('\n[2/4] results → supplement 変換...')
    n = convert_results()
    print('\n[2b/4] overseas → overseas_supplement 変換...')
    n_overseas = convert_overseas()

    # Step 2.5: master_horse 更新（新馬の種牡馬/母父馬を UM_DATA から追記）
    print('\n[2.5/4] master_horse 新馬追記...')
    update_master_horse()

    # Step 3: parquet 再生成
    if n > 0:
        print('\n[3/5] parquet 再生成...')
        if args.no_rebuild:
            delete_parquet()
        else:
            if not rebuild_parquet():
                print('  WARNING: 01_make_features.py が失敗しました。手動で確認してください。')
                sys.exit(1)

        # Step 4: モデル再学習
        if not args.no_train:
            print('\n[4/5] モデル再学習...')
            if not retrain_model():
                print('  WARNING: モデル再学習失敗。前回モデルのまま継続。')
        else:
            print('\n[4/5] モデル再学習 スキップ')

        print('\n[5/5] 完了。')
    else:
        print('\n更新データなし。parquet・モデルはそのまま。')


if __name__ == '__main__':
    main()
