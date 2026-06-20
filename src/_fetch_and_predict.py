# coding: utf-8
"""
netkeiba shutuba から出馬表を自動取得 → 06_predict_from_card.py → 新聞HTML生成

使い方:
  python src/_fetch_and_predict.py --date 20260530
  python src/_fetch_and_predict.py  # 今日の日付を使用
"""
import sys, io, os, re, csv, time, argparse, subprocess, pickle, json
import urllib.request
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from bs4 import BeautifulSoup

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'cache')

VENUE_CODE_TO_SHORT = {
    '05': '東', '06': '中', '07': '中京', '08': '京', '09': '阪',
    '04': '新', '03': '福', '02': '函', '01': '札', '10': '小',
}
VENUE_FULL_TO_SHORT = {
    '東京': '東', '中山': '中', '中京': '中京', '京都': '京', '阪神': '阪',
    '新潟': '新', '福島': '福', '函館': '函', '札幌': '札', '小倉': '小',
}
CLASS_MAP = {
    '新馬': 1, '未勝利': 2,
    '１勝': 3, '1勝': 3,
    '２勝': 4, '2勝': 4,
    '３勝': 5, '3勝': 5,
    'OP': 6, 'オープン': 6, 'L': 6,
    'G3': 7, 'G2': 8, 'G1': 9,
}


def fetch_html(url: str) -> str:
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept-Encoding': 'identity',
    })
    with urllib.request.urlopen(req, timeout=12) as r:
        raw = r.read()
        # Content-Type ヘッダで charset を確認、なければ UTF-8 を試みる
        ct = r.headers.get('Content-Type', '')
        if 'euc-jp' in ct.lower():
            return raw.decode('euc-jp', errors='replace')
        try:
            return raw.decode('utf-8')
        except UnicodeDecodeError:
            return raw.decode('euc-jp', errors='replace')


def get_race_ids(target_date: str) -> list[str]:
    url = f'https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={target_date}'
    html = fetch_html(url)
    race_ids = list(dict.fromkeys(re.findall(r'race_id=(\d{12})', html)))
    return race_ids


def parse_shutuba(race_id: str) -> dict | None:
    """1レースの出馬表を取得してdictを返す"""
    url = f'https://race.netkeiba.com/race/shutuba.html?race_id={race_id}'
    try:
        html = fetch_html(url)
    except Exception as e:
        print(f'  [WARN] fetch失敗 {race_id}: {e}', file=sys.stderr)
        return None

    soup = BeautifulSoup(html, 'html.parser')

    # ── タイトルからレース情報を抽出 ─────────────────────────
    title = soup.title.text if soup.title else ''
    # 例: "３歳未勝利 出馬表 | 2026年5月30日 東京1R"
    venue_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日\s+([^\d]+)(\d+)R', title)
    if not venue_match:
        print(f'  [WARN] タイトル解析失敗: {title}', file=sys.stderr)
        return None

    year   = int(venue_match.group(1))
    month  = int(venue_match.group(2))
    day    = int(venue_match.group(3))
    venue_full = venue_match.group(4).strip()
    r_num  = int(venue_match.group(5))
    date_s = f'{year}.{month}.{day}'
    venue  = VENUE_FULL_TO_SHORT.get(venue_full, venue_full)

    # クラス（タイトル冒頭部分）
    class_str = title.split('出馬表')[0].strip() if '出馬表' in title else ''

    # ── RaceData01 から距離・芝ダ・馬場状態 ────────────────────
    rd1 = soup.find('div', class_='RaceData01')
    surface, distance_m, baba = 'ダ', 0, '良'
    if rd1:
        text = rd1.get_text()
        m = re.search(r'([芝ダ障])(\d+)m', text)
        if m:
            surface     = m.group(1)
            distance_m  = int(m.group(2))
        bm = re.search(r'馬場:(\S+)', text)
        if bm:
            baba = bm.group(1)

    # ── HorseName span で馬名リスト ────────────────────────────
    horses_spans = soup.find_all('span', class_='HorseName')
    # 先頭が"馬名"（ヘッダ）の場合はスキップ
    horse_names = [sp.get_text(strip=True) for sp in horses_spans
                   if sp.get_text(strip=True) not in ('馬名', '')]

    if not horse_names:
        print(f'  [WARN] 馬名取得失敗 {race_id}', file=sys.stderr)
        return None

    # ── Shutuba_Table から各馬の詳細 ───────────────────────────
    st = soup.find('table', class_='Shutuba_Table')
    rows_data = []
    if st:
        for row in st.find_all('tr')[1:]:
            tds = row.find_all('td')
            if len(tds) < 7:
                continue
            umaban_text  = tds[1].get_text(strip=True) if len(tds) > 1 else ''
            horse_span   = row.find('span', class_='HorseName')
            horse_name   = horse_span.get_text(strip=True) if horse_span else ''
            if not horse_name or horse_name == '馬名':
                continue
            # 馬ID: 出馬表HTMLの馬名リンクから取得
            horse_link = horse_span.find_parent('a') if horse_span else None
            horse_id = ''
            if horse_link and horse_link.get('href'):
                _m = re.search(r'/horse/(\w+)/', str(horse_link.get('href', '')))
                if _m:
                    horse_id = _m.group(1)
            seireй_text  = tds[4].get_text(strip=True) if len(tds) > 4 else ''
            kinryo_text  = tds[5].get_text(strip=True) if len(tds) > 5 else ''
            jockey_text  = tds[6].get_text(strip=True) if len(tds) > 6 else ''
            trainer_text = tds[7].get_text(strip=True) if len(tds) > 7 else ''
            weight_text  = tds[8].get_text(strip=True) if len(tds) > 8 else ''

            # 性齢: "牝3" → 性別='牝', 年齢=3
            sex_match = re.match(r'([牡牝セ])(\d)', seireй_text)
            sex = sex_match.group(1) if sex_match else ''
            # 馬体重: "432(+6)" → weight=432, delta=+6
            wm = re.match(r'(\d+)\(([+\-\d]+)\)', weight_text)
            weight = int(wm.group(1)) if wm else None
            delta  = int(wm.group(2)) if wm else None

            try:
                umaban = int(umaban_text)
            except ValueError:
                continue
            try:
                kinryo = float(kinryo_text)
            except ValueError:
                kinryo = None

            rows_data.append({
                '馬番':    umaban,
                '馬名S':   horse_name,
                '馬ID':    horse_id,
                '性別':    sex,
                '斤量':    kinryo,
                '騎手':    jockey_text,
                '調教師':  trainer_text,
                '馬体重':  weight,
                '馬体重増減': delta,
            })

    return {
        'race_id':    race_id,
        'date_s':     date_s,
        'venue':      venue,
        'venue_full': venue_full,
        'r_num':      r_num,
        'surface':    surface,
        'distance_m': distance_m,
        'baba':       baba,
        'class_str':  class_str,
        'horses':     rows_data if rows_data else [{'馬番': i+1, '馬名S': h, '馬ID': '', '性別': '', '斤量': None, '騎手': '', '馬体重': None, '馬体重増減': None} for i, h in enumerate(horse_names)],
    }


def build_csv_rows(races: list[dict]) -> list[dict]:
    """レース情報リスト → CSVの行リスト（06_predict_from_card.py互換）"""
    rows = []
    for race in races:
        for h in race['horses']:
            rows.append({
                '日付S':   race['date_s'],
                '場 R':   f"{race['venue']}{race['r_num']}",
                '馬番':   h['馬番'],
                '馬名S':  h['馬名S'],
                '馬ID':   h.get('馬ID', ''),
                '性別':   h.get('性別', ''),
                '斤量':   h.get('斤量', ''),
                '騎手':   h.get('騎手', ''),
                '調教師': h.get('調教師', ''),
                '馬体重': h.get('馬体重', ''),
                '馬体重増減': h.get('馬体重増減', ''),
                '芝ダ':   race['surface'],
                '距離':   race['distance_m'],
                'クラス': race['class_str'],
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=None, help='日付 YYYYMMDD (省略時=今日)')
    ap.add_argument('--html-dir', default=r'G:\マイドライブ\競馬AI\予想レポート',
                    help='HTML保存先')
    args = ap.parse_args()

    target_date = args.date or datetime.now().strftime('%Y%m%d')
    print(f'=== {target_date} 出馬表取得 ===')

    # ── 1. race_id 一覧取得 ─────────────────────────────────────
    print('レースID取得中...')
    race_ids = get_race_ids(target_date)
    print(f'  {len(race_ids)}レース')
    if not race_ids:
        print('ERROR: レースが見つかりません。JRA開催日か確認してください。')
        sys.exit(1)

    # ── 2. 各レースの出馬表を取得 ──────────────────────────────
    races = []
    for i, rid in enumerate(race_ids):
        venue_code = rid[4:6]
        r_num_disp = int(rid[10:12])
        venue_short = VENUE_CODE_TO_SHORT.get(venue_code, venue_code)
        print(f'  [{i+1}/{len(race_ids)}] {venue_short}R{r_num_disp} ({rid})...', end=' ', flush=True)
        race = parse_shutuba(rid)
        if race:
            n = len(race['horses'])
            print(f'{n}頭')
            races.append(race)
        else:
            print('スキップ')
        time.sleep(0.5)

    if not races:
        print('ERROR: 出馬表取得失敗')
        sys.exit(1)

    print(f'\n{len(races)}レース / 合計{sum(len(r["horses"]) for r in races)}頭 取得完了')

    # ── 3. CSV保存 ──────────────────────────────────────────────
    csv_cols = ['日付S', '場 R', '馬番', '馬名S', '馬ID', '性別', '斤量', '騎手', '調教師',
                '馬体重', '馬体重増減', '芝ダ', '距離', 'クラス']
    rows = build_csv_rows(races)
    csv_path = os.path.join(BASE_DIR, f'出馬表形式{target_date[4:6]}月{int(target_date[6:8])}日_api.csv')
    with open(csv_path, 'w', encoding='cp932', errors='replace', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f'CSV保存: {csv_path}')

    # ── 3b. JVLink カードが存在すれば調教師名を正式名に上書き ──────
    # jvlink_card_builder.py が SE レコードからフルネームで取得した調教師名を使う
    jvlink_card_path = os.path.join(
        BASE_DIR, 'data', 'raw', 'cards', 'jvlink', f'{target_date}.csv'
    )
    if os.path.exists(jvlink_card_path):
        try:
            import pandas as _pd
            df_netkeiba = _pd.read_csv(csv_path, encoding='cp932')
            df_jvlink   = _pd.read_csv(jvlink_card_path, encoding='cp932')
            if '馬名S' in df_jvlink.columns:
                merged_cols = []
                for col in ['調教師', '騎手']:
                    if col in df_jvlink.columns and col in df_netkeiba.columns:
                        name_map = (df_jvlink.dropna(subset=[col])
                                    .drop_duplicates('馬名S')
                                    .set_index('馬名S')[col])
                        df_netkeiba[col] = df_netkeiba['馬名S'].map(name_map).fillna(df_netkeiba[col])
                        merged_cols.append(col)
                df_netkeiba.to_csv(csv_path, index=False, encoding='cp932', errors='replace')
                print(f'JVLinkマージ完了: {merged_cols} ({jvlink_card_path})')
            else:
                print(f'[WARN] JVLinkカードに調教師列なし: {jvlink_card_path}')
        except Exception as _e:
            print(f'[WARN] JVLink調教師マージ失敗: {_e}')
    else:
        print(f'[INFO] JVLinkカード未生成 ({jvlink_card_path}) → netkeiba略称を使用')

    # ── 4. 06_predict_from_card.py 実行 ────────────────────────
    print('\n=== 予測実行 ===')
    predict_script = os.path.join(BASE_DIR, 'src', '06_predict_from_card.py')
    r = subprocess.run(
        [sys.executable, predict_script, csv_path,
         '--html-dir', args.html_dir],
        cwd=BASE_DIR, text=True,
    )
    if r.returncode != 0:
        print(f'[ERROR] 予測失敗 (code={r.returncode})')
        sys.exit(1)

    print('\n完了')


if __name__ == '__main__':
    main()
