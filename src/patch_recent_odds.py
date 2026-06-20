# coding: utf-8
"""
parquet内の `単勝オッズ` NaN を netkeiba result.html から補完する。

対象: 過去 days_back 日以内（デフォルト90日）で `単勝オッズ=NaN` の行。
取得元: https://race.netkeiba.com/race/result.html?race_id=XXXX の Odds列。

使い方:
  python src/patch_recent_odds.py           # 過去90日分
  python src/patch_recent_odds.py --days 30 # 過去30日分
"""
import os, sys, re, time, argparse
import urllib.request
import pandas as pd
import numpy as np


def _decode_html(r) -> str:
    raw = r.read()
    ct = r.headers.get('Content-Type', '')
    if 'euc-jp' in ct.lower():
        return raw.decode('euc-jp', errors='replace')
    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError:
        return raw.decode('euc-jp', errors='replace')
from datetime import datetime, timedelta

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')

VENUE_LETTER_TO_CODE = {
    '東': '05', '中': '06', '中京': '07', '名': '07',
    '京': '08', '阪': '09', '新': '04', '福': '03',
    '函': '02', '札': '01', '小': '10',
}


def kaisai_to_race_id(date_num, kaisai, r_num) -> str | None:
    """(260509, '4東7', '1') → '202605040701'"""
    date_s = str(int(date_num))          # e.g. '260509'
    year   = '20' + date_s[:2]           # '2026'
    m = re.match(r'^(\d+)([^\d]+)(\d+)$', str(kaisai).strip())
    if not m:
        return None
    kai          = m.group(1).zfill(2)   # '04'
    venue_letter = m.group(2)            # '東'
    day          = m.group(3).zfill(2)   # '07'
    venue_code   = VENUE_LETTER_TO_CODE.get(venue_letter)
    if not venue_code:
        return None
    rr = str(int(float(r_num))).zfill(2)
    return year + venue_code + kai + day + rr


def fetch_race_odds(race_id: str) -> dict | None:
    """result.html から {馬番02d: 単勝オッズ} を返す。未確定または取得失敗は None。"""
    url = f'https://race.netkeiba.com/race/result.html?race_id={race_id}'
    row_pat  = re.compile(r'<tr[^>]*class="[^"]*HorseList[^"]*"[^>]*>(.*?)</tr>', re.DOTALL)
    uma_pat  = re.compile(r'class="Num Txt_C"[^>]*>(.*?)</td>', re.DOTALL)
    odds_pat = re.compile(r'class="Odds\s+Txt_C"[^>]*>\s*([\d.]+)', re.DOTALL)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=12) as r:
            html = _decode_html(r)
        if 'Result_Num' not in html:
            return None   # 未確定レース
        odds_map = {}
        for row_m in row_pat.finditer(html):
            row = row_m.group(1)
            u_m = uma_pat.search(row)
            o_m = odds_pat.search(row)
            if u_m and o_m:
                uma_raw = re.sub(r'<[^>]+>', '', u_m.group(1)).strip()
                if uma_raw.isdigit():
                    try:
                        odds_map[uma_raw.zfill(2)] = float(o_m.group(1))
                    except ValueError:
                        pass
        return odds_map if odds_map else None
    except Exception as e:
        print(f'  [WARN] {race_id}: {e}')
        return None


def patch_recent_odds(parquet_path: str = PARQUET_PATH, days_back: int = 90):
    """parquetの単勝オッズNaNをnetkeiba result.htmlで補完して上書き保存。"""
    cutoff_date = datetime.now() - timedelta(days=days_back)
    cutoff_yymm = int(cutoff_date.strftime('%y%m%d'))

    print(f'parquet読み込み中: {parquet_path}')
    df = pd.read_parquet(parquet_path)
    df['_date_num'] = pd.to_numeric(df['日付'], errors='coerce')

    if '単勝オッズ' not in df.columns:
        print('[ERROR] 単勝オッズ列が見つかりません')
        return

    df['単勝オッズ'] = pd.to_numeric(df['単勝オッズ'], errors='coerce')
    nan_mask = df['単勝オッズ'].isna() & (df['_date_num'] >= cutoff_yymm)

    n_nan = int(nan_mask.sum())
    print(f'対象NaN行: {n_nan}行 (過去{days_back}日, {cutoff_yymm}以降)')
    if n_nan == 0:
        print('補完不要')
        df.drop(columns=['_date_num'], inplace=True, errors='ignore')
        return

    # ユニークレース一覧
    req_cols = ['_date_num', '開催', 'Ｒ']
    nan_df   = df[nan_mask].copy()
    races    = nan_df.drop_duplicates(subset=req_cols)[req_cols].values
    print(f'ユニークレース: {len(races)}R → 取得開始...')

    total_filled = 0
    skipped = 0
    for date_num, kaisai, r_num in races:
        race_id = kaisai_to_race_id(date_num, kaisai, r_num)
        if not race_id:
            print(f'  race_id構築失敗: {date_num} / {kaisai} / {r_num}')
            skipped += 1
            continue

        odds_map = fetch_race_odds(race_id)
        if not odds_map:
            skipped += 1
            time.sleep(0.2)
            continue

        # 馬番列を特定
        uma_col = '馬番' if '馬番' in df.columns else None
        if uma_col is None:
            skipped += 1
            continue

        race_mask = ((df['_date_num'] == date_num) &
                     (df['開催'].astype(str) == str(kaisai)) &
                     (df['Ｒ'].astype(str) == str(r_num)))
        n_before = total_filled
        for idx in df[race_mask].index:
            if pd.notna(df.at[idx, '単勝オッズ']):
                continue
            try:
                uma_s = str(int(float(df.at[idx, uma_col]))).zfill(2)
            except (ValueError, TypeError):
                continue
            if uma_s in odds_map:
                df.at[idx, '単勝オッズ'] = odds_map[uma_s]
                total_filled += 1

        filled_here = total_filled - n_before
        if filled_here > 0:
            print(f'  {race_id}: {filled_here}頭補完')
        time.sleep(0.2)

    df.drop(columns=['_date_num'], inplace=True, errors='ignore')
    print(f'\n補完完了: {total_filled}行 / スキップ{skipped}R')

    if total_filled > 0:
        print('parquet保存中...')
        df.to_parquet(parquet_path, index=False)
        print('保存完了')
    else:
        print('補完なし → parquet未更新')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=90, help='対象期間（日数）')
    ap.add_argument('--parquet', default=PARQUET_PATH, help='parquetパス')
    args = ap.parse_args()
    patch_recent_odds(args.parquet, args.days)
