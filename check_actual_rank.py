# coding: utf-8
"""予測rank=1馬の実際の着順分布を確認"""
import pickle, re, json, sys, io, time, ssl, gzip
import pandas as pd, numpy as np
import urllib.request
from bs4 import BeautifulSoup
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open('data/raw/cache/20260530.cache.pkl', 'rb') as f:
    cache = pickle.load(f)
df = cache['result'].copy()
df.columns = df.columns.astype(object)

with open('data/raw/cache/20260530.odds.json', encoding='utf-8') as f:
    odds_dict = json.load(f)

df['_horse'] = df['馬名S'].astype(str).str.strip()
df['_venue'] = df['開催'].astype(str).str.extract(r'([^\d]+)')[0]
df['_R']     = pd.to_numeric(df['Ｒ'], errors='coerce')
df['_race']  = df['開催'].astype(str) + '_' + df['Ｒ'].astype(str)
df['_yahoo_odds'] = df['_horse'].map(odds_dict)
df['_mprob']      = 1.0 / df['_yahoo_odds'].clip(lower=1.0)

factor = df['clogit_factor'].fillna(0.16) if 'clogit_factor' in df.columns else pd.Series(0.16, index=df.index)
has_odds = df['_mprob'].notna()
df.loc[has_odds, 'clogit_score'] = df.loc[has_odds, 'clogit_calib'] - factor[has_odds] * df.loc[has_odds, '_mprob']
df['clogit_rank'] = df.groupby('_race')['clogit_score'].rank(ascending=False, method='first')
df['_rank'] = df['clogit_rank'].fillna(99).astype(int)

NAME_TO_CODE = {'京': '08', '東': '05'}
VENUE_KEYS   = {'京': (3, 11), '東': (2, 11)}

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode = ssl.CERT_NONE

def get_html(url):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=8, context=_SSL) as res:
            raw = res.read()
        try: return gzip.decompress(raw).decode('utf-8', errors='replace')
        except: return raw.decode('utf-8', errors='replace')
    except: return None

def parse_all_results(html):
    soup = BeautifulSoup(html, 'html.parser')
    rows = {}
    for table in soup.select('table'):
        headers = [th.get_text(strip=True) for th in table.select('th')]
        if not any('着順' in h for h in headers): continue
        idx_chaku = next(i for i, h in enumerate(headers) if '着順' in h)
        idx_uma   = next((i for i, h in enumerate(headers) if '馬名' in h), 3)
        for tr in table.select('tr')[1:]:
            tds = tr.find_all('td')
            if len(tds) <= max(idx_chaku, idx_uma): continue
            chaku = tds[idx_chaku].get_text(strip=True)
            raw   = tds[idx_uma].get_text(strip=True)
            uma   = re.split(r'[牡牝セ]\d|せん\d|/', raw)[0].strip()
            if uma and chaku:
                rows[uma] = chaku
        if rows: break
    return rows

rank1 = df[df['_rank'] == 1].sort_values(['_venue', '_R'])
results = []

for _, r in rank1.iterrows():
    venue = r['_venue']
    r_num = int(r['_R'])
    horse = r['_horse']
    code  = NAME_TO_CODE.get(venue)
    kai, nichi = VENUE_KEYS.get(venue, (None, None))
    if not code or not kai:
        continue
    key  = f'26{code}{kai:02d}{nichi:02d}{r_num:02d}'
    html = get_html(f'https://sports.yahoo.co.jp/keiba/race/result/{key}')
    if not html or '着順' not in html:
        actual = '中止/未取得'
    else:
        all_res = parse_all_results(html)
        actual  = all_res.get(horse, '不明')
    results.append({'venue': venue, 'R': r_num, 'horse': horse,
                    'calib': r['clogit_calib'], 'actual': actual})
    time.sleep(0.2)

print(f'{"":3} {"R":>2}  {"予測1位馬":<18} {"勝率":>6}  実着順')
print('-' * 55)
for d in results:
    print(f'{d["venue"]:>3}{d["R"]:>2}R  {d["horse"]:<18} {d["calib"]*100:5.1f}%  {d["actual"]}着')

# 着順分布
actual_nums = []
for d in results:
    try:
        actual_nums.append(int(d['actual']))
    except:
        pass

print(f'\n== 予測1位馬の実着順分布 (24レース中 {len(actual_nums)}件数値あり) ==')
s = pd.Series(actual_nums)
dist = s.value_counts().sort_index()
for rank_val, cnt in dist.items():
    bar = '█' * cnt
    print(f'  {rank_val:>2}着: {cnt:>2}件  {bar}')

print(f'\n平均実着順: {s.mean():.1f}')
print(f'中央値:     {s.median():.1f}')
print(f'1着 (的中): {(s==1).sum()}件 / {len(s)}件 = {(s==1).mean()*100:.1f}%')
print(f'3着内:      {(s<=3).sum()}件 / {len(s)}件 = {(s<=3).mean()*100:.1f}%')
print(f'5着内:      {(s<=5).sum()}件 / {len(s)}件 = {(s<=5).mean()*100:.1f}%')
