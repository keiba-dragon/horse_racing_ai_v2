import sys; sys.stdout.reconfigure(encoding='utf-8')
import pickle, pandas as pd, numpy as np

def analyze(path, label):
    with open(path, 'rb') as f:
        c = pickle.load(f)
    df = c['result']
    calib = pd.to_numeric(df['clogit_calib'], errors='coerce').dropna()
    bands_pct = [0, 1, 2, 5, 10, 20, 101]
    labels = ['0-1%', '1-2%', '2-5%', '5-10%', '10-20%', '20%+']
    cuts = pd.cut(calib * 100, bins=bands_pct, labels=labels)
    vc = cuts.value_counts().sort_index()
    total = len(calib)
    at = c.get('predicted_at', '?')
    print(f'=== {label} ({at}) ===')
    for band, n in vc.items():
        pct = n / total * 100
        print(f'  {band}: {n}頭 ({pct:.1f}%)')
    print(f'  合計: {total}頭')
    print(f'  min={calib.min()*100:.2f}%  max={calib.max()*100:.1f}%  mean={calib.mean()*100:.2f}%')
    return calib

old = analyze(r'data/raw/cache/20260530_old.cache.pkl', '5/30旧（fallback時）')
print()
new = analyze(r'data/raw/cache/出馬表形式05月31日_api.cache.pkl', '5/31今回（fix後）')
