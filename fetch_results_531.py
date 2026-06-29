# coding: utf-8
"""5/31 結果取得 & 予測照合"""
import sys, io, re, ssl, gzip, time, pickle
import urllib.request
import pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3',
}
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

NAME_TO_CODE = {
    '中京': '07', '札': '01', '函': '02', '福': '03', '新': '04',
    '東': '05', '中': '06', '京': '08', '阪': '09', '小': '10',
}

def _get(url):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=8, context=_SSL_CTX) as res:
            raw = res.read()
        try:
            return gzip.decompress(raw).decode('utf-8', errors='replace')
        except Exception:
            return raw.decode('utf-8', errors='replace')
    except Exception:
        return None

def parse_result_page(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    rows = []
    for table in soup.select('table'):
        headers = [th.get_text(strip=True) for th in table.select('th')]
        if not any('着順' in h for h in headers):
            continue
        idx_chakujun = next(i for i, h in enumerate(headers) if '着順' in h)
        try:
            idx_uma = next(i for i, h in enumerate(headers) if '馬名' in h)
        except StopIteration:
            idx_uma = 3
        for tr in table.select('tr')[1:]:
            tds = tr.find_all('td')
            if len(tds) <= max(idx_chakujun, idx_uma):
                continue
            chakujun = tds[idx_chakujun].get_text(strip=True)
            raw_uma  = tds[idx_uma].get_text(strip=True)
            uma = re.split(r'[牡牝セ]\d|せん\d|/', raw_uma)[0].strip()
            if uma and chakujun.isdigit():
                rows.append({'着順': chakujun, '馬名': uma})
        if rows:
            break
    return rows

def try_result(key):
    url = f'https://sports.yahoo.co.jp/keiba/race/result/{key}'
    html = _get(url)
    if not html or len(html) < 3000:
        return None, None
    if '着順' not in html:
        return None, None
    return html, url

def find_key(code, year2):
    for kai in range(1, 7):
        for nichi in range(1, 12):
            key = f'{year2}{code}{kai:02d}{nichi:02d}01'
            html, url = try_result(key)
            if html:
                return kai, nichi
            time.sleep(0.1)
    return None, None

# ── メイン ──
with open(r'data\raw\cache\出馬表形式05月31日_api.cache.pkl', 'rb') as f:
    cache = pickle.load(f)
result_df = cache['result']

venues = set()
for kaikai in result_df['開催'].dropna().unique():
    m = re.match(r'^\d([^\d]+)', str(kaikai))
    if m:
        venues.add(m.group(1))
print(f'今日の会場: {venues}')

year2 = '26'
all_results = []

for venue_name, code in NAME_TO_CODE.items():
    if venue_name not in venues:
        continue
    kai, nichi = find_key(code, year2)
    if kai is None:
        print(f'{venue_name}: キー見つからず')
        continue
    print(f'\n== {venue_name} kai={kai} nichi={nichi} ==')
    for race_num in range(1, 13):
        key = f'{year2}{code}{kai:02d}{nichi:02d}{race_num:02d}'
        html, url = try_result(key)
        if not html:
            break
        rows = parse_result_page(html)
        if not rows:
            continue
        winner = next((r for r in rows if r['着順'] == '1'), None)
        top3 = [r['馬名'] for r in rows if r['着順'] in ('1', '2', '3')]
        print(f'  {venue_name}{race_num}R: 1着={winner["馬名"] if winner else "?"}  上位3={top3}')
        for r in rows:
            r['会場'] = venue_name
            r['R'] = race_num
            all_results.append(r)
        time.sleep(0.15)

if not all_results:
    print('\n結果取得できず（まだ開催前のレースがある可能性）')
    sys.exit()

df_res = pd.DataFrame(all_results)
print(f'\n合計: {len(df_res)}行取得')

# ── 予測★1位 vs 実際1位 ──
print('\n== clogit ★1位 vs 実際1位 ==')
pred = result_df[result_df['clogit_rank'] == 1][
    ['馬名S','開催','Ｒ','clogit_calib','clogit_rank']].copy()

hits = 0
total = 0
for _, row in pred.iterrows():
    kaikai = str(row['開催'])
    m = re.match(r'^\d([^\d]+)', kaikai)
    venue = m.group(1) if m else ''
    r_num = int(row['Ｒ']) if pd.notna(row['Ｒ']) else 0
    actual = df_res[(df_res['会場'] == venue) & (df_res['R'] == r_num) & (df_res['着順'] == '1')]
    actual_winner = actual['馬名'].values[0] if len(actual) > 0 else '未確定'
    hit = (actual_winner == row['馬名S'])
    if actual_winner != '未確定':
        total += 1
        if hit:
            hits += 1
    mark = '★的中' if hit else ('  未確定' if actual_winner == '未確定' else '  ハズレ')
    calib_pct = pd.to_numeric(row['clogit_calib'], errors='coerce') * 100
    print(f'  {venue}{r_num}R: 予測={row["馬名S"]}({calib_pct:.1f}%)  実際={actual_winner}  {mark}')

if total > 0:
    print(f'\n的中率: {hits}/{total} = {hits/total*100:.1f}%')
