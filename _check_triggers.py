# coding: utf-8
from datetime import datetime, timedelta

race_times_str = ['09:50','10:05','10:20','10:40','11:00','11:20','11:40',
                  '12:10','12:30','12:50','13:10','13:30','13:50',
                  '14:10','14:35','14:55','15:20','15:45','16:15']
d = datetime.today().replace(second=0, microsecond=0)
race_times = [d.replace(hour=int(s[:2]), minute=int(s[3:])) for s in race_times_str]

BEFORE_MINS = 30
AFTER_MINS  = 15

pre_triggers  = {(t - timedelta(minutes=BEFORE_MINS), '前') for t in race_times}
post_triggers = {(t + timedelta(minutes=AFTER_MINS),  '後') for t in race_times}
trigger_map = {}
for t, kind in sorted(pre_triggers | post_triggers):
    trigger_map[t] = kind
triggers = sorted(trigger_map.items())

print("全トリガー（前=馬体重更新、後=結果取得）:")
for t, k in triggers:
    print(f"  {t.strftime('%H:%M')} ({k})")
print(f"\n合計: {len(triggers)}回 (レース数{len(race_times)}×2 から重複除去)")
