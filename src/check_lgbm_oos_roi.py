# coding: utf-8
"""
既存LGBMモデル（venue×distance別）のOOS ROI測定 + リークチェック
"""
import os, sys, pickle, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# モデル情報読み込み
with open(os.path.join(MODEL_DIR, 'model_info.json'), encoding='utf-8') as f:
    info = json.load(f)

FEAT_COLS   = info['features']
MODELS_MAP  = info['models']       # key: "東_芝1600" → {"win": "lgb_東_芝1600_win.pkl", ...}
EXCLUDE_IN  = info['exclude']['in_race']   # 2角,3角,4角
EXCLUDE_ODS = info['exclude']['odds']      # 人気,前走人気

print(f'特徴量: {len(FEAT_COLS)}列')
print(f'モデル数: {len(MODELS_MAP)}')
print(f'除外(in_race): {EXCLUDE_IN}')
print(f'除外(odds): {EXCLUDE_ODS}')

# ── リークチェック ────────────────────────────────────────────────────────────
print('\n=== リークチェック ===')
# 現在レース内の結果を使う可能性のある列名
SUSPICIOUS = ['2角','3角','4角','着順','人気','単勝オッズ','払戻','タイム']
for col in FEAT_COLS:
    for kw in SUSPICIOUS:
        if kw in col and '前走' not in col and '走前' not in col and '近' not in col:
            print(f'  要確認: {col}')
            break

# ── データ読み込み ────────────────────────────────────────────────────────────
print('\nデータ読み込み中...')
df = pd.read_parquet(DATA_FILE)
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
df = df.dropna(subset=['日付_num', '着順_num'])
df = df[df['着順_num'] < 99]

# race_id
df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                 df['開催'].astype(str).str.strip() + '_' +
                 df['Ｒ'].astype(str).str.strip())

# venue抽出（開催列の中間文字）
def extract_venue_short(kaikai):
    s = str(kaikai).strip()
    # 例: "4東7" → "東", "12阪3" → "阪"
    import re
    m = re.search(r'\d+([^\d]+)', s)
    return m.group(1) if m else s

# 距離帯（モデルキーに合わせた会場略称）
VENUE_MAP = {
    '東京': '東', '中山': '中', '阪神': '阪', '京都': '京',
    '小倉': '小', '新潟': '新', '福島': '福', '中京': '名',
    '函館': '函', '札幌': '札',
}

df['venue_raw'] = df['開催'].astype(str).str.strip()
df['venue_short'] = df['venue_raw'].apply(extract_venue_short)

# 距離数値
df['距離_m'] = df['距離'].astype(str).str.extract(r'(\d+)')[0].astype(float)
df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0]

# モデルキー: "{venue_short}_{surface}{距離_m:.0f}"
df['model_key'] = df['venue_short'] + '_' + df['surface'] + df['距離_m'].astype(int).astype(str)

# OOS
oos = df[df['日付_num'] >= 230101].copy().reset_index(drop=True)
oos['odds_num'] = pd.to_numeric(oos['単勝オッズ'], errors='coerce')
oos['yr'] = oos['日付_num'] // 10000
print(f'OOSレコード: {len(oos):,}  ユニークレース: {oos["race_id"].nunique():,}')

# ── モデル適用 ────────────────────────────────────────────────────────────────
print('\nモデル適用中...')

model_cache = {}
pred_arr = np.full(len(oos), np.nan)

keys_found    = set()
keys_notfound = set()

for key, grp_idx in oos.groupby('model_key').groups.items():
    if key not in MODELS_MAP:
        keys_notfound.add(key)
        continue
    keys_found.add(key)

    win_path = os.path.join(MODEL_DIR, MODELS_MAP[key]['win'])
    if not os.path.exists(win_path):
        keys_notfound.add(key + '(no file)')
        continue

    if key not in model_cache:
        with open(win_path, 'rb') as f:
            model_cache[key] = pickle.load(f)

    m = model_cache[key]
    grp = oos.loc[grp_idx]

    # 特徴量を揃える
    avail = [c for c in FEAT_COLS if c in grp.columns]
    X = grp[avail].copy()
    for c in FEAT_COLS:
        if c not in X.columns:
            X[c] = np.nan
    X = X[FEAT_COLS]
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    try:
        pred_arr[grp_idx] = m.predict(X)
    except Exception as e:
        print(f'  [{key}] predict error: {e}')

print(f'  モデルあり: {len(keys_found)}キー')
print(f'  モデルなし: {len(keys_notfound)}キー  例: {list(keys_notfound)[:5]}')

# ── ROI計算 ──────────────────────────────────────────────────────────────────
oos['pred_win'] = pred_arr
covered = oos['pred_win'].notna()
print(f'\n予測できたレコード: {covered.sum():,} / {len(oos):,} ({covered.mean():.1%})')

oos_cov = oos[covered].copy()
oos_cov['rank_lgbm'] = oos_cov.groupby('race_id')['pred_win'].rank(ascending=False, method='first')

# rank=1の中でpred_winがNaNのレースを除外
races_complete = oos_cov.groupby('race_id')['pred_win'].apply(lambda x: x.notna().all())
complete_races = races_complete[races_complete].index
oos_full = oos_cov[oos_cov['race_id'].isin(complete_races)].copy()
oos_full['rank_lgbm'] = oos_full.groupby('race_id')['pred_win'].rank(ascending=False, method='first')

print(f'全馬予測できたレース: {oos_full["race_id"].nunique():,}')

top1 = oos_full[oos_full['rank_lgbm'] == 1]
won  = top1['着順_num'] == 1
total_roi = (top1.loc[won, 'odds_num'] * 100).sum() / (len(top1) * 100) - 1

print(f'\n{"="*50}')
print('既存LGBMシステム OOS ROI (2023+)')
print('='*50)
for yr in sorted(top1['yr'].unique()):
    s = top1[top1['yr'] == yr]
    w = s['着順_num'] == 1
    r = (s.loc[w, 'odds_num'] * 100).sum() / (len(s) * 100) - 1
    print(f'  20{int(yr):02d}: {len(s):5d}R  win={w.mean():.3f}  ROI={r:+.3f}')
print(f'  Total: {len(top1):5d}R  win={won.mean():.3f}  ROI={total_roi:+.3f}')

print(f'\n[比較]')
print(f'  既存LGBM:        {total_roi:+.3f}')
print(f'  clogit (最終):   -0.125')
