# coding: utf-8
"""
各セグメントの OOS (2023-2026) キャリブレーション分析
予測確率をデシル分けして、bin内平均予測 vs 実際の勝率を出す
"""
import sys, os, pickle
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE
from save_v3 import add_computed_features

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'final_model.pkl')
with open(MODEL_PATH, 'rb') as f:
    pkg = pickle.load(f)
arts = pkg['artifacts']

SEGMENTS = {
    'ダ':   ('ダ',  lambda s, dm: (s == 'ダ') & (dm > 1400)),
    'ダ短': ('ダ短', lambda s, dm: (s == 'ダ') & (dm <= 1400)),
    '芝短': ('芝短', lambda s, dm: (s == '芝') & (dm <= 1400)),
    '芝中': ('芝中', lambda s, dm: (s == '芝') & (dm > 1400) & (dm <= 2000)),
    '芝長': ('芝長', lambda s, dm: (s == '芝') & (dm > 2000)),
}

N_BINS = 10  # デシル

baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}

def load_base():
    df = pd.read_parquet(DATA_FILE)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    df = df[df['開催'].notna()].copy()
    df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
    df['_dist_m'] = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
    df = add_computed_features(df)
    for col in df.columns:
        if '馬場状態' in col and col != '馬場状態':
            df[col] = df[col].map(baba_map)
    return df

print("parquet 読み込み中...")
base = load_base()
oos = base[base['日付_num'] >= 230101].copy()

results = {}

for art_key, (seg_label, dist_fn) in SEGMENTS.items():
    art = arts.get(art_key)
    if art is None:
        print(f"[WARN] artifact '{art_key}' not found")
        continue

    seg_df = oos[(dist_fn(oos['surface'], oos['_dist_m'])) & (oos['クラス_rank'] != 1.0)].copy()
    if len(seg_df) == 0:
        continue

    feat_cols = art['feat_cols']
    # NaN指示変数 を付与
    for f in feat_cols:
        if f.endswith('_isnan'):
            base_f = f[:-6]
            if base_f in seg_df.columns and f not in seg_df.columns:
                seg_df[f] = seg_df[base_f].isna().astype(float)

    valid_p = [c for c in feat_cols if c in seg_df.columns]
    seg_sorted = seg_df.sort_values('race_id').reset_index(drop=True)

    X_p, _, gs_p, n_p, *_ = prepare(seg_sorted, valid_p,
                                      scaler=art['scaler'],
                                      top_idx=None, top_idx3=None)
    raw_prob = segment_softmax(X_p @ art['coef'], gs_p, n_p)
    iso_prob = art['isotonic'].predict(raw_prob)

    seg_sorted['raw_prob'] = raw_prob
    seg_sorted['iso_prob'] = iso_prob
    seg_sorted['win'] = (seg_sorted['着順_num'] == 1).astype(int)

    # デシル bins
    seg_sorted['bin'] = pd.qcut(seg_sorted['iso_prob'], q=N_BINS, duplicates='drop', labels=False)

    bin_stats = seg_sorted.groupby('bin').agg(
        n=('win', 'count'),
        pred_mean=('iso_prob', 'mean'),
        actual_win_rate=('win', 'mean'),
    ).reset_index()

    results[art_key] = {
        'label': seg_label,
        'n_total': len(seg_sorted),
        'bins': bin_stats,
        'global_pred': iso_prob.mean(),
        'global_actual': seg_sorted['win'].mean(),
    }
    print(f"{seg_label}: n={len(seg_sorted):,}  global pred={iso_prob.mean():.3f}  actual={seg_sorted['win'].mean():.3f}")

# JSON形式で保存
import json
out = {}
for k, v in results.items():
    out[k] = {
        'label': v['label'],
        'n_total': int(v['n_total']),
        'global_pred': float(v['global_pred']),
        'global_actual': float(v['global_actual']),
        'bins': [
            {
                'bin': int(row['bin']),
                'n': int(row['n']),
                'pred_mean': float(row['pred_mean']),
                'actual_win_rate': float(row['actual_win_rate']),
            }
            for _, row in v['bins'].iterrows()
        ]
    }

with open('_calib_data.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\n_calib_data.json に保存完了")
