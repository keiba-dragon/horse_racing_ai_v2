import pickle, pandas as pd, sys, json, ssl, gzip, re, time
import urllib.request
from bs4 import BeautifulSoup
sys.stdout.reconfigure(encoding='utf-8')

with open('data/raw/cache/20260530_new.cache.pkl', 'rb') as f:
    cache = pickle.load(f)
df = cache['result'].copy()

# EV列確認
ev_cols = [c for c in df.columns if 'ev' in c.lower() or 'EV' in c or '期待' in c]
print("EV関連列:", ev_cols)
print("列一覧（一部）:", [c for c in df.columns if c.startswith('_')][:20])
