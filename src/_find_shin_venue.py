# coding: utf-8
"""新潟の正しいvenueコードをbrute-forceで探す"""
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

# 新潟の具体的な馬名でページを探す
TARGET = {'エルムラント', 'コンストラクション', 'ティピティーナ', 'エタンセル', 'アデルフィー'}

def get_horses(key):
    try:
        r = requests.get(f'https://sports.yahoo.co.jp/keiba/race/denma/{key}',
                         headers=HEADERS, timeout=5, verify=False)
        if r.status_code != 200 or len(r.text) < 5000:
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
                name = re.split(r'[牡牝セ]\d|せん\d', uma_raw)[0].strip()
                if name:
                    names.add(name)
        return names, t
    except Exception:
        return set(), ''

# 今日のkai/nichiパターンを考慮: Tokyo=kai2,nichi8, Kyoto=kai3,nichi8
# Niigataも今日がnichi=8近辺の可能性 (kai=1か2)
print('venue×kai×nichi のブルートフォース探索 (race=12,11,10で試す)')
found = False
for venue in ['04', '01', '02', '03', '44']:  # 04=JRA新潟、他も試す
    for kai in range(1, 4):
        for nichi in range(1, 10):
            for race in [12, 11, 10, 9, 5]:
                key = f'26{venue}{kai:02d}{nichi:02d}{race:02d}'
                ph, title = get_horses(key)
                if not ph:
                    continue
                matched = ph & TARGET
                if matched:
                    print(f'HIT: venue={venue} kai={kai} nichi={nichi} R{race} '
                          f'key={key} matches={matched}')
                    print(f'     title={title[:50]}')
                    found = True
                time.sleep(0.1)
            if found:
                break
        if found:
            break
    if found:
        break

if not found:
    print('見つからず')
