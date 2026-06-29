# coding: utf-8
"""買い推奨11件 vs 今日の実際の結果を照合"""
import pickle, re, json, sys, io, time, ssl, gzip
import pandas as pd, numpy as np
import urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ── 買い推奨計算 ──
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
df['_ev'] = df['clogit_calib'] - df['_mprob'] * 0.80
df['_gap'] = df.groupby('_race', sort=False)['clogit_calib'].transform(
    lambda x: x.nlargest(2).iloc[0] - x.nlargest(2).iloc[1] if x.dropna().shape[0] >= 2 else 0.0
)

def cls(v):
    v = str(v)
    if '未勝利' in v: return '未勝利'
    if '新馬' in v: return '新馬'
    return '1勝+'
df['_class'] = df['_cls_group'].apply(cls) if '_cls_group' in df.columns else '1勝+'
df['_career'] = pd.to_numeric(df.get('キャリア', 0), errors='coerce').fillna(0)
df['_rank'] = df['clogit_rank'].fillna(99).astype(int)

df['_buy'] = (
    (df['_rank'] == 1) &
    (df['_gap'] >= 0.15) &
    (df['_ev'] >= 0.0) &
    ((df['_class'] == '1勝+') | ((df['_class'] == '未勝利') & (df['_career'] >= 5)))
)

buy = df[df['_buy']].copy()
print(f'買い推奨: {len(buy)}件')

# ── Yahoo結果取得 ──
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
_SSL = ssl.create_default_context()
_SSL.check_hostname = False
_SSL.verify_mode = ssl.CERT_NONE

NAME_TO_CODE = {'京': '08', '東': '05'}
VENUE_KEYS = {'京': (3, 11), '東': (2, 11)}  # 前回調査済み

def get_html(url):
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=8, context=_SSL) as res:
            raw = res.read()
        try: return gzip.decompress(raw).decode('utf-8', errors='replace')
        except: return raw.decode('utf-8', errors='replace')
    except: return None

def parse_winner(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    for table in soup.select('table'):
        headers = [th.get_text(strip=True) for th in table.select('th')]
        if not any('着順' in h for h in headers): continue
        idx_chaku = next(i for i, h in enumerate(headers) if '着順' in h)
        idx_uma = next((i for i, h in enumerate(headers) if '馬名' in h), 3)
        for tr in table.select('tr')[1:]:
            tds = tr.find_all('td')
            if len(tds) <= max(idx_chaku, idx_uma): continue
            chaku = tds[idx_chaku].get_text(strip=True)
            raw = tds[idx_uma].get_text(strip=True)
            uma = re.split(r'[牡牝セ]\d|せん\d|/', raw)[0].strip()
            if uma and chaku == '1':
                return uma
    return None

print()
print('=== 買い推奨 vs 実際の結果 ===')
print(f'{"":3} {"R":>2}  {"予測馬":<18} {"calib":>6} {"gap":>6} {"odds":>6}  実際の1着  判定')
print('-' * 80)

hits = 0
valid = 0
for _, r in buy.sort_values(['_venue', '_R']).iterrows():
    venue = r['_venue']
    r_num = int(r['_R'])
    horse = r['_horse']
    code = NAME_TO_CODE.get(venue)
    kai, nichi = VENUE_KEYS.get(venue, (None, None))
    if not code or not kai:
        print(f'{venue}{r_num}R  {horse} → 会場コード不明')
        continue
    key = f'26{code}{kai:02d}{nichi:02d}{r_num:02d}'
    html = get_html(f'https://sports.yahoo.co.jp/keiba/race/result/{key}')
    if not html or '着順' not in html:
        print(f'{venue:>3}{r_num:>2}R  {horse:<18} {r["clogit_calib"]*100:5.1f}%  {r["_gap"]:+.3f}  {r["_yahoo_odds"]:>5.1f}  → 結果未取得（中止？）')
        time.sleep(0.2)
        continue
    winner = parse_winner(html)
    hit = (winner == horse)
    if hit: hits += 1
    valid += 1
    mark = '★的中' if hit else 'ハズレ'
    print(f'{venue:>3}{r_num:>2}R  {horse:<18} {r["clogit_calib"]*100:5.1f}%  {r["_gap"]:+.3f}  {r["_yahoo_odds"]:>5.1f}  {str(winner):<14}  {mark}')
    time.sleep(0.2)

print(f'\n的中: {hits}/{valid}件 = {hits/valid*100:.1f}%' if valid > 0 else '\n有効レースなし')
