# coding: utf-8
import sys, io, json, numpy as np
sys.path.insert(0, 'src')
from discord_notify import load_today_cache, prepare_df, get_good_picks, fetch_race_times

target_date = '20260517'
df = load_today_cache(target_date)

with open(f'data/raw/cache/{target_date}.odds.json', encoding='utf-8') as f:
    odds_dict = json.load(f)
with open(f'data/raw/cache/{target_date}.venue_keys.json') as f:
    venue_keys = json.load(f)

race_times = fetch_race_times(venue_keys, target_date[2:4])
df = prepare_df(df, odds_dict, race_times)

picks = get_good_picks(df, ev_thr=0.0)
print(f'EV>0: {len(picks)}頭\n')

cols = ['_race_key', '_horse', '_yahoo_odds', '_pop_rank', '_ev', 'clogit_score', '_time_str']
for _, r in picks[cols].iterrows():
    ev = r['_ev']
    mark = '🟢' if ev > 0.05 else ('🟡' if ev > 0.02 else '🔵')
    rk = int(r['_race_key']) if str(r['_race_key']).isdigit() else r['_race_key']
    odds_v = r['_yahoo_odds']
    pop_v  = r['_pop_rank']
    t = r['_time_str']
    nm = r['_horse']
    cs = r['clogit_score']
    print(f"{mark} {str(rk):<8} {t} {nm:<18} {int(pop_v):>2}番人気 {odds_v:>5.1f}倍  EV={ev:+.3f}  score={cs:.3f}")
