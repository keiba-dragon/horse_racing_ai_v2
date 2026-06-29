# coding: utf-8
"""5/31 予測全体 vs 結果照合（複数条件）"""
import sys, io, re, ssl, gzip, time, pickle
import urllib.request
import pandas as pd, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

NAME_TO_CODE = {
    '東': '05', '京': '08', '中': '06', '阪': '09',
    '中京': '07', '新': '04', '函': '02', '小': '10',
}

def _get(url):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=8, context=_SSL_CTX) as r:
            raw = r.read()
        try: return gzip.decompress(raw).decode('utf-8', errors='replace')
        except: return raw.decode('utf-8', errors='replace')
    except: return None

def parse_result(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    rows = []
    for table in soup.select('table'):
        hdrs = [th.get_text(strip=True) for th in table.select('th')]
        if not any('着順' in h for h in hdrs): continue
        i_c = next(i for i,h in enumerate(hdrs) if '着順' in h)
        try: i_u = next(i for i,h in enumerate(hdrs) if '馬名' in h)
        except: i_u = 3
        for tr in table.select('tr')[1:]:
            tds = tr.find_all('td')
            if len(tds) <= max(i_c, i_u): continue
            rank = tds[i_c].get_text(strip=True)
            raw_name = tds[i_u].get_text(strip=True)
            name = re.split(r'[牡牝セ]\d|せん\d|/', raw_name)[0].strip()
            if name and rank.isdigit():
                rows.append({'rank': int(rank), 'horse': name})
        if rows: break
    return rows

def try_result(key):
    url = f'https://sports.yahoo.co.jp/keiba/race/result/{key}'
    html = _get(url)
    if not html or len(html) < 3000 or '着順' not in html: return None
    return html

def find_kai_nichi(code, year2='26'):
    for kai in range(1, 7):
        for nichi in range(1, 12):
            key = f'{year2}{code}{kai:02d}{nichi:02d}01'
            if try_result(key): return kai, nichi
            time.sleep(0.05)
    return None, None

# ── 結果取得 ──
with open(r'data\raw\cache\出馬表形式05月31日_api.cache.pkl', 'rb') as f:
    cache = pickle.load(f)
pred = cache['result'].copy()

venues = set()
for kk in pred['開催'].dropna().unique():
    m = re.match(r'^\d([^\d]+)', str(kk))
    if m: venues.add(m.group(1))

all_res = {}
for v, code in NAME_TO_CODE.items():
    if v not in venues: continue
    kai, nichi = find_kai_nichi(code)
    if kai is None: continue
    for rn in range(1, 13):
        key = f'26{code}{kai:02d}{nichi:02d}{rn:02d}'
        html = try_result(key)
        if not html: break
        rows = parse_result(html)
        if rows:
            all_res[(v, rn)] = rows
        time.sleep(0.1)

# ── 各馬の実着順を付与 ──
def get_rank(venue, r_num, horse):
    rows = all_res.get((venue, r_num), [])
    for r in rows:
        if r['horse'] == horse: return r['rank']
    return None

pred['_venue'] = pred['開催'].apply(
    lambda x: re.match(r'^\d([^\d]+)', str(x)).group(1)
    if re.match(r'^\d([^\d]+)', str(x)) else '')
pred['_r'] = pd.to_numeric(pred['Ｒ'], errors='coerce').fillna(0).astype(int)
pred['実着順'] = pred.apply(lambda r: get_rank(r['_venue'], r['_r'], r['馬名S']), axis=1)

# 結果が取れたレースのみ
done = pred[pred['実着順'].notna()].copy()
done['実着順'] = done['実着順'].astype(int)

print(f"結果取得済みレース数: {done.groupby(['_venue','_r']).ngroups}")
print(f"結果取得済み頭数: {len(done)}\n")

# ── clogit ★1位の成績 ──
clogit1 = done[done['clogit_rank'] == 1].copy()
print("=== clogit ★1位の成績 ===")
for _, row in clogit1.sort_values(['_venue','_r']).iterrows():
    mark = '★1着' if row['実着順'] == 1 else (f"  {int(row['実着順'])}着")
    calib = pd.to_numeric(row['clogit_calib'], errors='coerce') * 100
    print(f"  {row['_venue']}{row['_r']}R: {row['馬名S']} ({calib:.1f}%)  → {mark}")
wins1 = (clogit1['実着順'] == 1).sum()
top3_1 = (clogit1['実着順'] <= 3).sum()
n1 = len(clogit1)
print(f"\n  勝率={wins1}/{n1}={wins1/n1*100:.1f}%  複勝率={top3_1}/{n1}={top3_1/n1*100:.1f}%")

# ── gap≥0.15 (高信頼) の成績 ──
if 'gap' not in done.columns and 'clogit_calib' in done.columns:
    def calc_gap(g):
        s = g['clogit_calib'].sort_values(ascending=False).values
        return s[0] - s[1] if len(s) >= 2 else np.nan
    done['gap'] = done.groupby(['_venue','_r'], group_keys=False).apply(
        lambda g: g.assign(gap=calc_gap(g)))['gap']

high_conf = done[(done['clogit_rank'] == 1) &
                 (pd.to_numeric(done.get('gap', pd.Series(dtype=float)), errors='coerce') >= 0.15)]
if len(high_conf) > 0:
    wins_hc = (high_conf['実着順'] == 1).sum()
    top3_hc = (high_conf['実着順'] <= 3).sum()
    print(f"\n=== clogit★1位 & gap≥15% ===")
    print(f"  {len(high_conf)}レース  勝率={wins_hc}/{len(high_conf)}={wins_hc/len(high_conf)*100:.1f}%"
          f"  複勝率={top3_hc}/{len(high_conf)}={top3_hc/len(high_conf)*100:.1f}%")
    for _, row in high_conf.sort_values(['_venue','_r']).iterrows():
        mark = '★1着' if row['実着順'] == 1 else f"  {int(row['実着順'])}着"
        print(f"    {row['_venue']}{row['_r']}R: {row['馬名S']} → {mark}")
