# coding: utf-8
"""6/6 予測 vs 実際結果 照合"""
import sys, io, re, ssl, gzip, time, pickle
import urllib.request
import pandas as pd
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'ja,en-US;q=0.7,en;q=0.3',
}
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode    = ssl.CERT_NONE

NAME_TO_CODE = {
    '東': '05', '阪': '09', '中京': '07', '京': '08',
    '小': '10', '中': '06', '新': '04', '函': '02', '札': '01', '福': '03',
}

def _get(url):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10, context=_SSL_CTX) as res:
            raw = res.read()
        try:
            return gzip.decompress(raw).decode('utf-8', errors='replace')
        except Exception:
            return raw.decode('utf-8', errors='replace')
    except Exception as e:
        return None

def parse_result(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    rows = []
    for table in soup.select('table'):
        headers = [th.get_text(strip=True) for th in table.select('th')]
        if not any('着順' in h for h in headers):
            continue
        idx_c = next(i for i, h in enumerate(headers) if '着順' in h)
        try:
            idx_u = next(i for i, h in enumerate(headers) if '馬名' in h)
        except StopIteration:
            idx_u = 3
        try:
            idx_o = next(i for i, h in enumerate(headers) if 'オッズ' in h or '単勝' in h)
        except StopIteration:
            idx_o = None
        for tr in table.select('tr')[1:]:
            tds = tr.find_all('td')
            if len(tds) <= max(idx_c, idx_u):
                continue
            chak = tds[idx_c].get_text(strip=True)
            raw_u = tds[idx_u].get_text(strip=True)
            uma = re.split(r'[牡牝セ]\d|せん\d|/', raw_u)[0].strip()
            odds_str = tds[idx_o].get_text(strip=True) if idx_o and idx_o < len(tds) else ''
            try:
                odds = float(odds_str)
            except Exception:
                odds = float('nan')
            if uma and chak.isdigit():
                rows.append({'着順': int(chak), '馬名': uma, 'オッズ': odds})
        if rows:
            break
    return rows

def find_kai_nichi(code, year2='26'):
    for kai in range(1, 7):
        for nichi in range(1, 14):
            key = f'{year2}{code}{kai:02d}{nichi:02d}01'
            html = _get(f'https://sports.yahoo.co.jp/keiba/race/result/{key}')
            if html and len(html) > 3000 and '着順' in html:
                return kai, nichi
            time.sleep(0.08)
    return None, None

# ── キャッシュ読み込み ──
with open(r'data\raw\cache\20260606.cache.pkl', 'rb') as f:
    cache = pickle.load(f)
result_df = cache['result']

# clogit 1位 (予測1位) を取得
pred = result_df[result_df['clogit_rank'] == 1][
    ['馬名S', '開催', 'Ｒ', 'clogit_calib', 'clogit_rank']
].copy()
print(f"clogit予測対象: {len(pred)}レース")

# 会場コード取り出し
def venue_from_kaikai(k):
    m = re.match(r'^\d([^\d]+)', str(k))
    return m.group(1) if m else ''

venues_in_pred = set(pred['開催'].apply(venue_from_kaikai).unique())
print(f"会場: {venues_in_pred}")

# ── Yahoo結果取得 ──
all_results = {}
year2 = '26'
for vname in venues_in_pred:
    code = NAME_TO_CODE.get(vname)
    if not code:
        print(f"{vname}: コード不明 skip")
        continue
    print(f"\n{vname} (code={code}) キー探索中...")
    kai, nichi = find_kai_nichi(code, year2)
    if kai is None:
        print(f"  {vname}: 見つからず")
        continue
    print(f"  kai={kai} nichi={nichi}")
    for rnum in range(1, 13):
        key = f'{year2}{code}{kai:02d}{nichi:02d}{rnum:02d}'
        html = _get(f'https://sports.yahoo.co.jp/keiba/race/result/{key}')
        if not html or len(html) < 3000 or '着順' not in html:
            break
        rows = parse_result(html)
        if rows:
            all_results[(vname, rnum)] = rows
            w = next((r for r in rows if r['着順'] == 1), None)
            print(f"  {vname}{rnum}R: 1着={w['馬名'] if w else '?'} ({w['オッズ']:.1f}倍)")
        time.sleep(0.12)

if not all_results:
    print("\nYahoo結果取得失敗")
    sys.exit(1)

# ── 照合 ──
print("\n" + "="*72)
print(f"  {'レース':<8}  {'予測1位':<16}  {'確率':>6}  {'実際1位':<16}  {'オッズ':>6}  結果")
print("="*72)

hits = 0
total = 0
total_return = 0.0
total_bet = 0.0

rows_all = []
for _, row in pred.sort_values(['開催', 'Ｒ']).iterrows():
    venue = venue_from_kaikai(row['開催'])
    rnum  = int(row['Ｒ']) if pd.notna(row['Ｒ']) else 0
    key   = (venue, rnum)

    res = all_results.get(key, [])
    winner = next((r for r in res if r['着順'] == 1), None)
    actual_name  = winner['馬名']  if winner else '?'
    actual_odds  = winner['オッズ'] if winner else float('nan')
    pred_name    = row['馬名S']
    prob         = row['clogit_calib']

    hit = (actual_name == pred_name)
    if hit:
        hits += 1
        total_return += actual_odds * 100
    total_bet += 100
    total += 1

    mark = '★的中' if hit else '  ハズレ'
    print(f"  {venue}{rnum}R  {pred_name:<16}  {prob*100:5.1f}%  "
          f"{actual_name:<16}  {actual_odds:5.1f}倍  {mark}")
    rows_all.append({
        'レース': f'{venue}{rnum}R',
        '予測': pred_name, '確率': prob,
        '実際': actual_name, 'オッズ': actual_odds, '的中': hit,
    })

print("="*72)
roi = total_return / total_bet - 1 if total_bet > 0 else float('nan')
print(f"\n  的中: {hits}/{total}  ({hits/total*100:.1f}%)")
print(f"  回収率: {roi:+.2%}  (投資 {total_bet:.0f}円 / 回収 {total_return:.0f}円)")
