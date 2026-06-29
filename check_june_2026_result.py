# coding: utf-8
"""
2026年6月6日・7日のaccuracy_model.pkl予想をJVLink結果CSVと照合する
"""
import sys, os, pickle
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE
from save_v3 import add_computed_features

TARGET_DATES = [260606, 260607]

# ── 実際の結果をCSVから読み込む ─────────────────────────────────────────────
results_dir = os.path.join(BASE_DIR, 'data', 'raw', 'results')
res_dfs = []
for date_str in ['20260606', '20260607']:
    fpath = os.path.join(results_dir, f'{date_str}.csv')
    if os.path.exists(fpath):
        df_r = pd.read_csv(fpath, encoding='utf-8-sig', low_memory=False)
        res_dfs.append(df_r)

results_df = pd.concat(res_dfs, ignore_index=True)
# 日付_num, 会場コード, レースNo, 着順, 馬名
results_df['日付_num'] = pd.to_numeric(
    results_df['日付'].astype(str).str.replace(r'^\d{2}', '', regex=False),
    errors='coerce')
# 20260606 → 260606
results_df['日付_num'] = results_df['日付'].apply(lambda x: int(str(x)[2:]) if pd.notna(x) else None)
results_df['会場コード'] = results_df['会場コード'].astype(str)
results_df['レースNo'] = pd.to_numeric(results_df['レースNo'], errors='coerce')
results_df['着順'] = pd.to_numeric(results_df['着順'], errors='coerce')

def get_winner(date_num, venue_code, race_no):
    """1着馬名を返す"""
    mask = (results_df['日付_num'] == date_num) & \
           (results_df['会場コード'] == str(venue_code)) & \
           (results_df['レースNo'] == race_no) & \
           (results_df['着順'] == 1)
    rows = results_df[mask]
    return rows['馬名'].values[0] if len(rows) > 0 else None

print(f"結果CSV: {len(results_df)}行 / 1着馬数: {(results_df['着順']==1).sum()}件")

# ── モデルロード ────────────────────────────────────────────────────────────
with open(os.path.join(BASE_DIR, 'models', 'accuracy_model.pkl'), 'rb') as f:
    MODEL = pickle.load(f)

# ── データロード ────────────────────────────────────────────────────────────
print("データ読み込み中...", flush=True)
df = pd.read_parquet(DATA_FILE)
df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
df = df.dropna(subset=['日付_num'])
df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                 df['開催'].astype(str).str.strip() + '_' + df['Ｒ'].astype(str).str.strip())
df = df[df['開催'].notna()].copy()
df['surface'] = df['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
dm = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
df['クラス_rank'] = pd.to_numeric(df['クラス_rank'], errors='coerce')
df = add_computed_features(df)
if '今回_会場' in df.columns and '1走前_開催' in df.columns:
    df['輸送有無'] = (df['今回_会場'].astype(str) != df['1走前_開催'].astype(str).str[1]).astype(float)
    df.loc[df['1走前_開催'].isna(), '輸送有無'] = float('nan')
baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
for col in df.columns:
    if '馬場状態' in col and col != '馬場状態':
        df[col] = df[col].map(baba_map)
df['dist_m'] = dm

df_target = df[df['日付_num'].isin(TARGET_DATES)].copy()
s_t = df_target['surface']
r_t = df_target['クラス_rank']
dm_t = df_target['dist_m']

SEG_MASKS = {
    'ダ長': (s_t=='ダ') & (dm_t>1400)  & (r_t!=1.0),
    'ダ短': (s_t=='ダ') & (dm_t<=1400) & (r_t!=1.0),
    '芝短': (s_t=='芝') & (dm_t<=1400) & (r_t!=1.0),
    '芝中': (s_t=='芝') & (dm_t>1400)  & (dm_t<=2000) & (r_t!=1.0),
    '芝長': (s_t=='芝') & (dm_t>2000)  & (r_t!=1.0),
}

# ── 予想生成 ─────────────────────────────────────────────────────────────────
print("予想生成中...", flush=True)
all_preds = []
name_col = '馬名S' if '馬名S' in df_target.columns else '馬名'

for seg in ['芝長', '芝中', '芝短', 'ダ長', 'ダ短']:
    pkg = MODEL[seg]
    feat_cols, scaler, coef = pkg['feat_cols'], pkg['scaler'], pkg['coef']
    seg_t = df_target[SEG_MASKS[seg]].copy()
    if len(seg_t) == 0:
        continue
    for fc in feat_cols:
        if fc.endswith('_isnan'):
            base = fc[:-6]
            if base in seg_t.columns and fc not in seg_t.columns:
                seg_t[fc] = seg_t[base].isna().astype(float)
    try:
        X_p, _, gs_p, n_p, *_ = prepare(seg_t, feat_cols, scaler=scaler,
                                          top_idx=None, top_idx3=None)
        s_s = seg_t.sort_values('race_id').reset_index(drop=True)
        s_s['prob'] = segment_softmax(X_p @ coef, gs_p, n_p)
        s_s['rank_pred'] = s_s.groupby('race_id')['prob'].rank(ascending=False, method='first')
        pred1 = s_s[s_s['rank_pred'] == 1][['race_id', name_col, 'prob',
                                              '日付_num', '開催', 'Ｒ']].copy()
        pred1.rename(columns={name_col: '予想馬'}, inplace=True)
        pred1['seg'] = seg
        all_preds.append(pred1)
    except Exception as e:
        print(f"  {seg} ERROR: {e}")

preds = pd.concat(all_preds, ignore_index=True)
preds['venue_code'] = preds['開催'].astype(str).str.strip().str.split('_').str[0]
preds['race_no']    = pd.to_numeric(preds['Ｒ'], errors='coerce')
print(f"予想生成完了: {len(preds)}レース\n")

# ── 照合 ────────────────────────────────────────────────────────────────────
VENUE_MAP = {'5':'東京','9':'阪神','7':'中京','8':'京都','1':'札幌','2':'函館',
             '3':'福島','4':'新潟','6':'中山','10':'小倉'}
FAV_ACC = {'ダ長':0.3403,'ダ短':0.3490,'芝短':0.2869,'芝中':0.3321,'芝長':0.3605}

print("="*75)
print(f"{'日':>5} {'会場':>4} {'R':>3} {'セグ':>4} {'予想馬':>18} {'1着馬':>18} {'確率':>6} {'結果':>6}")
print("="*75)

hits = 0; total = 0
seg_stats = {s: [0, 0] for s in FAV_ACC}
all_rows = []

for _, row in preds.sort_values(['日付_num', 'venue_code', 'race_no']).iterrows():
    winner = get_winner(int(row['日付_num']), row['venue_code'], int(row['race_no']))
    date_d = f"6/{int(str(row['日付_num'])[-2:])}"
    venue  = VENUE_MAP.get(str(row['venue_code']), str(row['venue_code']))
    seg    = row['seg']

    if winner is None:
        result_str = '---'
        hit = None
    else:
        hit = (winner.strip() == str(row['予想馬']).strip())
        result_str = '★的中' if hit else 'ハズレ'
        total += 1
        seg_stats[seg][1] += 1
        if hit:
            hits += 1
            seg_stats[seg][0] += 1

    print(f"  {date_d:>5} {venue:>4} {int(row['race_no']):>3}R "
          f"[{seg:>3}] {str(row['予想馬']):>18}  {str(winner or '?'):>18}  "
          f"{row['prob']:.3f}  {result_str}")
    all_rows.append({'date': date_d, 'venue': venue, 'R': int(row['race_no']),
                     'seg': seg, '予想馬': row['予想馬'], '1着馬': winner,
                     'hit': hit, 'prob': row['prob']})

print("="*75)
if total > 0:
    print(f"\n【2日間 合計】 {hits}/{total} = {hits/total*100:.1f}%")
    print()

    print(f"{'セグメント':<6} {'的中':>5} {'R数':>5} {'的中率':>8} {'1番人気基準':>10}")
    print("-"*40)
    for seg in ['芝長','芝中','芝短','ダ長','ダ短']:
        h, t = seg_stats[seg]
        if t == 0:
            print(f"  {seg:<6}  なし")
            continue
        acc = h/t
        fav = FAV_ACC[seg]
        mark = '✅' if acc >= fav else '❌'
        print(f"  {seg:<6} {h:>5}/{t:<5} {acc:>7.1%}  (1番人気={fav:.1%}) {mark}")
    print()

    # 1番人気との比較計算
    fav_hits = 0; fav_total = 0
    for (date_n, venue_c, race_n), grp in \
        preds.sort_values(['日付_num','venue_code','race_no']).groupby(
            ['日付_num','venue_code','race_no']):
        winner = get_winner(int(date_n), venue_c, int(race_n))
        if winner is None:
            continue
        fav_total += 1
        # 1番人気 = 単勝オッズ最低馬
        mask = (results_df['日付_num']==int(date_n)) & \
               (results_df['会場コード']==str(venue_c)) & \
               (results_df['レースNo']==int(race_n))
        race_res = results_df[mask].copy()
        race_res['単勝オッズ'] = pd.to_numeric(race_res['単勝オッズ'], errors='coerce')
        if len(race_res) > 0:
            fav_row = race_res.nsmallest(1, '単勝オッズ')
            fav_name = fav_row['馬名'].values[0]
            if fav_name == winner:
                fav_hits += 1

    if fav_total > 0:
        print(f"【参考】この2日の1番人気的中率: {fav_hits}/{fav_total} = {fav_hits/fav_total*100:.1f}%")
        print(f"【AI予想】                      {hits}/{total} = {hits/total*100:.1f}%")
        diff = hits/total - fav_hits/fav_total
        sign = '+' if diff >= 0 else ''
        print(f"  差: {sign}{diff*100:.1f}pp  {'✅ 1番人気超え!' if diff >= 0 else '❌ 1番人気未満'}")
