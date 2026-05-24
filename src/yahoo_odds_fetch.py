# coding: utf-8
"""
Yahoo Sports 競馬からライブ単勝オッズをスクレイピング。
JVLink RT の代替（サブスクリプション不要）。

使い方:
  python src/yahoo_odds_fetch.py --date 20260517
  python src/yahoo_odds_fetch.py            # 今日の日付
  python src/yahoo_odds_fetch.py --rediscover

出力:
  data/raw/cache/YYYYMMDD.odds.json  {horse_name: odds_float}
  CHANGED=1 / CHANGED=0 をprint
"""
import sys, io, os, json, re, argparse, pickle, time, glob, ssl, gzip
import urllib.request
from datetime import datetime

try:
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f'[ERROR] 必要パッケージ未インストール: {e}', file=sys.stderr)
    sys.exit(1)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'cache')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3',
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer': 'https://sports.yahoo.co.jp/keiba/',
}
TIMEOUT = 8
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

NAME_TO_CODE = {
    '中京': '07',
    '札': '01', '函': '02', '福': '03', '新': '04', '東': '05',
    '中': '06', '京': '08', '阪': '09', '小': '10',
}


def _get(url: str) -> str | None:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=_SSL_CTX) as res:
            raw = res.read()
        try:
            html = gzip.decompress(raw).decode('utf-8', errors='replace')
        except Exception:
            html = raw.decode('utf-8', errors='replace')
        if len(html) > 5000:
            return html
    except Exception as e:
        print(f'  [WARN] fetch失敗: {url}: {e}', file=sys.stderr)
    return None


def _parse_horses(html: str) -> dict:
    """Yahoo出馬表ページから {horse_name: odds_float}"""
    soup = BeautifulSoup(html, 'html.parser')
    rows = soup.select('table tr')
    result = {}
    for row in rows[1:]:
        tds = row.find_all('td')
        if len(tds) < 8:
            continue
        uma_raw = tds[2].get_text(strip=True)
        uma_name = re.split(r'[牡牝セ]\d|せん\d', uma_raw)[0].strip()
        if not uma_name:
            continue
        odds_raw = tds[7].get_text(strip=True)
        m = re.search(r'\(([0-9]+\.[0-9]+)\)', odds_raw)
        if not m:
            continue
        try:
            result[uma_name] = float(m.group(1))
        except ValueError:
            continue
    return result


def _get_page_horses(key: str) -> set:
    html = _get(f'https://sports.yahoo.co.jp/keiba/race/denma/{key}')
    if not html:
        return set()
    soup = BeautifulSoup(html, 'html.parser')
    names = set()
    for row in soup.select('table tr')[1:]:
        tds = row.find_all('td')
        if len(tds) < 3:
            continue
        uma_raw = tds[2].get_text(strip=True)
        name = re.split(r'[牡牝セ]\d|せん\d', uma_raw)[0].strip()
        if name:
            names.add(name)
    return names


def find_venue_key(venue_code: str, venue_horses: set, year2: str) -> tuple | None:
    print(f'  venue={venue_code} キー検索...', end='', flush=True)
    # R1は障害レースの場合があるので R5 と R1 の両方で試す
    probe_races = ['05', '02', '06', '01']
    for kai in range(1, 7):
        for nichi in range(1, 13):
            matched = False
            for probe in probe_races:
                key = f'{year2}{venue_code}{kai:02d}{nichi:02d}{probe}'
                ph = _get_page_horses(key)
                if not ph:
                    continue
                if len(ph & venue_horses) / len(ph) >= 0.5:
                    print(f' kai={kai} nichi={nichi}')
                    return kai, nichi
                matched = True  # ページあったが馬名不一致 → この kai/nichi は違う
                break
            if matched:
                time.sleep(0.1)
    print(' 見つからず')
    return None


def get_venue_keys(target_date: str, card_df, rediscover: bool = False) -> dict:
    cache_path = os.path.join(CACHE_DIR, f'{target_date}.venue_keys.json')
    if not rediscover and os.path.exists(cache_path):
        with open(cache_path, encoding='utf-8') as f:
            keys = json.load(f)
        print(f'venue_keys キャッシュ: {keys}')
        return keys

    year2 = target_date[2:4]

    # 会場ごとの馬名セットを構築
    venue_horses_map = {}
    for _, row in card_df.iterrows():
        race_col = str(row.get('場 R', ''))
        m = re.match(r'^([^\d]+)', race_col)
        if not m:
            continue
        prefix = m.group(1)
        uma = str(row.get('馬名S', '')).strip()
        if uma:
            venue_horses_map.setdefault(prefix, set()).add(uma)

    # 略称 → venue_code（長い方を優先）
    sorted_names = sorted(NAME_TO_CODE.keys(), key=len, reverse=True)
    result = {}
    for prefix, horses in venue_horses_map.items():
        code = next((NAME_TO_CODE[n] for n in sorted_names if prefix == n), None)
        if not code:
            print(f'  [WARN] 未知の会場略称: {prefix}', file=sys.stderr)
            continue
        kn = find_venue_key(code, horses, year2)
        if kn:
            result[code] = list(kn)

    if result:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def fetch_all_odds(target_date: str, venue_keys: dict, max_races: int = 12) -> dict:
    year2 = target_date[2:4]
    code_to_name = {v: k for k, v in NAME_TO_CODE.items()}
    all_odds = {}
    for venue_code, (kai, nichi) in venue_keys.items():
        name = code_to_name.get(venue_code, venue_code)
        print(f'  [{name}] kai={kai} nichi={nichi}...', end='', flush=True)
        n = 0
        for race_num in range(1, max_races + 1):
            key = f'{year2}{venue_code}{kai:02d}{nichi:02d}{race_num:02d}'
            html = _get(f'https://sports.yahoo.co.jp/keiba/race/denma/{key}')
            if html:
                odds = _parse_horses(html)
                all_odds.update(odds)
                n += 1
            time.sleep(0.15)
        print(f' {n}R {sum(1 for _ in all_odds)}頭')
    return all_odds


def save_odds_json(target_date: str, odds_dict: dict) -> bool:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f'{target_date}.odds.json')
    prev = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            try:
                prev = json.load(f)
            except Exception:
                pass
    if not odds_dict:
        return False
    prev_sum = sum(prev.values())
    new_sum  = sum(odds_dict.values())
    changed = (
        len(odds_dict) != len(prev)
        or (prev_sum > 0 and abs(new_sum - prev_sum) / prev_sum > 0.01)
    )
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(odds_dict, f, ensure_ascii=False)
    print(f'odds保存: {len(odds_dict)}頭 → {os.path.basename(path)}')
    return changed


def load_v2_card_df(target_date: str):
    """キャッシュから card_df を取得。複数形式に対応。"""
    try:
        # 形式1: YYYYMMDD.cache.pkl (predict_from_card の出力)
        p1 = os.path.join(CACHE_DIR, f'{target_date}.cache.pkl')
        if os.path.exists(p1):
            with open(p1, 'rb') as f:
                c = pickle.load(f)
            df = c.get('result')
            if df is None:
                df = c.get('card_df')
            if df is not None:
                return df

        # 形式2: 出馬表形式X月Y日.cache.pkl
        dt = datetime.strptime(target_date, '%Y%m%d')
        label = f'{dt.month}月{dt.day}日'
        pattern = os.path.join(CACHE_DIR, f'出馬表形式{label}*.cache.pkl')
        files = sorted(glob.glob(pattern))
        if files:
            with open(files[-1], 'rb') as f:
                c = pickle.load(f)
            return c.get('card_df')
    except Exception as e:
        print(f'[WARN] cache読み込み失敗: {e}', file=sys.stderr)
    return None


def fetch_odds_now(target_date: str, card_df=None, rediscover: bool = False) -> dict:
    """
    外部から呼び出すメイン関数。
    odds_dict {horse_name: odds_float} を返す。
    """
    if card_df is None:
        card_df = load_v2_card_df(target_date)
    if card_df is None:
        print(f'[ERROR] card_df取得失敗', file=sys.stderr)
        return {}

    venue_keys = get_venue_keys(target_date, card_df, rediscover=rediscover)
    if not venue_keys:
        print('[ERROR] 会場キー取得失敗', file=sys.stderr)
        return {}

    return fetch_all_odds(target_date, venue_keys)


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=None)
    ap.add_argument('--rediscover', action='store_true')
    args = ap.parse_args()

    target_date = args.date or datetime.now().strftime('%Y%m%d')
    print(f'対象日: {target_date}')

    odds = fetch_odds_now(target_date, rediscover=args.rediscover)
    if not odds:
        print('CHANGED=0')
        return
    changed = save_odds_json(target_date, odds)
    print(f'CHANGED={1 if changed else 0}')


if __name__ == '__main__':
    main()
