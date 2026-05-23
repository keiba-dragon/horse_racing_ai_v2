# coding: utf-8
import requests, sys, io, re, time
from bs4 import BeautifulSoup
import urllib3; urllib3.disable_warnings()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

import pickle
with open('data/raw/cache/20260517.cache.pkl', 'rb') as f:
    data = pickle.load(f)
df = data['result']
shin_horses = set(df[df['場 R'].str.startswith('新')]['馬名S'].tolist())
print(f'新潟馬 {len(shin_horses)}頭 サンプル: {sorted(shin_horses)[:8]}')

def get_horses(key):
    r = requests.get(f'https://sports.yahoo.co.jp/keiba/race/denma/{key}',
                     headers=HEADERS, timeout=6, verify=False)
    if r.status_code != 200:
        return set(), ''
    r.encoding = 'utf-8'
    soup = BeautifulSoup(r.text, 'html.parser')
    title = soup.find('title')
    t = title.get_text() if title else ''
    names = set()
    for row in soup.select('table tr')[1:]:
        tds = row.find_all('td')
        if len(tds) >= 3:
            uma_raw = tds[2].get_text(strip=True)
            # 性別コード (牡牝セ) の前でカット
            name = re.split(r'[牡牝セ雄雌][\dN]|せん\d', uma_raw)[0].strip()
            if name:
                names.add(name)
    return names, t

# venue=04 (JRA新潟), kai=1, nichi=1 でR2-R12を確認
# 正しいキー形式: year2(2)+venue(2)+kai(2)+nichi(2)+race(2) = 10桁
print('\n=== venue=04, kai=1, nichi=1 ===')
for race in range(1, 13):
    key = f'26040101{race:02d}'   # 10桁
    ph, title = get_horses(key)
    matched = ph & shin_horses
    print(f'  R{race:02d} key={key}({len(key)}桁): {len(ph)}頭 一致={len(matched)} [{title[:30]}] 馬:{list(ph)[:3]}')
    time.sleep(0.2)

# 他のkai/nichiも試す
print('\n新潟 他のkai/nichiを試す')
for nichi in range(1, 4):
    key = f'2604010{nichi:02d}05'  # 10桁
    ph, title = get_horses(key)
    matched = ph & shin_horses
    print(f'  kai=1,nichi={nichi} R5 key={key}({len(key)}桁): {len(ph)}頭 一致={len(matched)} [{title[:30]}] 馬:{list(ph)[:2]}')
    time.sleep(0.2)
