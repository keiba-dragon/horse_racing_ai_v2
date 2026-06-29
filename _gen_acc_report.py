# coding: utf-8
import os, sys, pickle
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE
from save_v3 import add_computed_features

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'accuracy_model.pkl')
acc_model  = pickle.load(open(MODEL_PATH, 'rb'))

# 各セグメント情報
SEGS = {
    'ダ長': dict(color='#2980b9', cond='ダ & 距離>1400m',
                 filter_fn=lambda s, dm: (s == 'ダ') & (dm > 1400),
                 acc_2325=0.3009, baseline=0.3403),
    'ダ短': dict(color='#1a6fa0', cond='ダ & 距離≤1400m',
                 filter_fn=lambda s, dm: (s == 'ダ') & (dm <= 1400),
                 acc_2325=0.3089, baseline=0.3490),
    '芝短': dict(color='#27ae60', cond='芝 & 距離≤1400m',
                 filter_fn=lambda s, dm: (s == '芝') & (dm <= 1400),
                 acc_2325=0.2754, baseline=0.2869),
    '芝中': dict(color='#16a085', cond='芝 & 1401-2000m',
                 filter_fn=lambda s, dm: (s == '芝') & (dm > 1400) & (dm <= 2000),
                 acc_2325=0.3088, baseline=None),
    '芝長': dict(color='#8e44ad', cond='芝 & 距離>2000m',
                 filter_fn=lambda s, dm: (s == '芝') & (dm > 2000),
                 acc_2325=0.3534, baseline=None),
}
ORDER = ['ダ長','ダ短','芝短','芝中','芝長']

baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}

# OOS ROI+的中率計算
print("parquet 読み込み...")
df_all = pd.read_parquet(DATA_FILE)
df_all['日付_num'] = pd.to_numeric(df_all['日付'], errors='coerce')
df_all['着順_num'] = pd.to_numeric(df_all['着順_num'], errors='coerce')
df_all = df_all.dropna(subset=['日付_num','着順_num'])
df_all = df_all[df_all['着順_num'] < 99]
df_all['race_id'] = (df_all['日付_num'].astype(int).astype(str) + '_' +
                     df_all['開催'].astype(str).str.strip() + '_' +
                     df_all['Ｒ'].astype(str).str.strip())
df_all = df_all[df_all['開催'].notna()].copy()
df_all['surface'] = df_all['距離'].astype(str).str.strip().str.extract(r'^([芝ダ])')[0].fillna('不明')
df_all['_dist_m'] = pd.to_numeric(df_all['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
df_all['クラス_rank'] = pd.to_numeric(df_all['クラス_rank'], errors='coerce')
df_all = add_computed_features(df_all)
for col in df_all.columns:
    if '馬場状態' in col and col != '馬場状態':
        df_all[col] = df_all[col].map(baba_map)
oos_all = df_all[df_all['日付_num'] >= 230101].copy()

seg_stats = {}
for sk in ORDER:
    d = SEGS[sk]
    art = acc_model[sk]
    seg_df = oos_all[(d['filter_fn'](oos_all['surface'], oos_all['_dist_m'])) &
                     (oos_all['クラス_rank'] != 1.0)].copy()
    if len(seg_df) == 0:
        continue
    feat_cols = art['feat_cols']
    # isnan指示変数を先に追加
    for f in feat_cols:
        if f.endswith('_isnan'):
            base_f = f[:-6]
            if f not in seg_df.columns:
                seg_df[f] = seg_df[base_f].isna().astype(float) if base_f in seg_df.columns else 1.0
    # 欠損カラムをNaNで追加（prepare内で0埋め）
    for f in feat_cols:
        if not f.endswith('_isnan') and f not in seg_df.columns:
            seg_df[f] = np.nan
    seg_sorted = seg_df.sort_values('race_id').reset_index(drop=True)
    X_p, _, gs_p, n_p, *_ = prepare(seg_sorted, feat_cols,
                                      scaler=art['scaler'],
                                      top_idx=None, top_idx3=None)
    raw_prob = segment_softmax(X_p @ art['coef'], gs_p, n_p)
    iso_prob = art['isotonic'].predict(raw_prob)
    seg_sorted['prob'] = iso_prob
    seg_sorted['rank'] = seg_sorted.groupby('race_id')['prob'].rank(ascending=False, method='first')
    top1 = seg_sorted[seg_sorted['rank'] == 1]
    n_races = seg_sorted['race_id'].nunique()
    n_win   = (top1['着順_num'] == 1).sum()
    acc = n_win / len(top1) if len(top1) > 0 else 0
    odds = pd.to_numeric(top1['単勝オッズ'], errors='coerce')
    roi  = (odds[top1['着順_num']==1] * 100).sum() / (len(top1)*100) - 1 if len(top1) > 0 else np.nan

    # 年別
    year_stats = {}
    for yr_label, yr_start, yr_end in [('2023-24', 230101, 250101), ('2025', 250101, 260101), ('2026', 260101, 999999)]:
        ydf = seg_sorted[(seg_sorted['日付_num'] >= yr_start) & (seg_sorted['日付_num'] < yr_end)]
        if len(ydf) == 0: continue
        yt1 = ydf[ydf['rank'] == 1]
        yr_races = ydf['race_id'].nunique()
        yr_acc = (yt1['着順_num']==1).sum() / len(yt1) if len(yt1) > 0 else 0
        yr_odds = pd.to_numeric(yt1['単勝オッズ'], errors='coerce')
        yr_roi  = (yr_odds[yt1['着順_num']==1]*100).sum()/(len(yt1)*100)-1 if len(yt1) > 0 else np.nan
        year_stats[yr_label] = dict(acc=yr_acc, roi=yr_roi, n=yr_races)
    seg_stats[sk] = dict(acc=acc, roi=roi, n=len(top1), n_races=n_races, year=year_stats)
    print(f"{sk}: 的中率={acc:.1%}  ROI={roi:+.2%}  ({n_races}R)")

print("HTML生成中...")

# ---- KPIカード ----
kpi_cards = ''
for sk in ORDER:
    d = SEGS[sk]
    st = seg_stats.get(sk, {})
    acc = st.get('acc', 0)
    roi = st.get('roi', 0) or 0
    baseline = d.get('baseline')
    vs_base = f'（1番人気比 {"+" if baseline and acc>=baseline else ""}{(acc-baseline)*100:.1f}pt）' if baseline else ''
    kpi_cards += f'''
<div class="kpi-card">
  <div class="kpi-seg" style="background:{d["color"]}">{sk}</div>
  <div class="kpi-val" style="color:{"#2e7d32" if acc >= (baseline or 0.30) else "#e65100"}">{acc:.1%}</div>
  <div class="kpi-label">OOS 的中率</div>
  <div class="kpi-sub">{vs_base}</div>
  <div class="kpi-sub" style="margin-top:4px">ROI: <span style="color:{"#2e7d32" if roi>=0 else "#c62828"}">{roi:+.2%}</span></div>
</div>'''

# ---- 年別テーブル ----
tbl_rows = ''
for sk in ORDER:
    d = SEGS[sk]
    st = seg_stats.get(sk, {})
    ys = st.get('year', {})
    def yr_cell(y):
        if y not in ys: return '<td>-</td><td>-</td>'
        a = ys[y]['acc']
        r = ys[y]['roi']
        n = ys[y]['n']
        return (f'<td style="color:{"#2e7d32" if a>=0.30 else "#555"}">{a:.1%}<small>({n}R)</small></td>'
                f'<td style="color:{"#2e7d32" if r>=0 else "#c62828"}">{r:+.2%}</td>')
    tbl_rows += f'''<tr>
  <td style="font-weight:bold;color:{d["color"]}">{sk}</td>
  {yr_cell("2023-24")}{yr_cell("2025")}{yr_cell("2026")}
</tr>'''

# ---- セグメント別詳細タブ ----
tabs_btn = ''
tabs_content = ''
for i, sk in enumerate(ORDER):
    d = SEGS[sk]
    art = acc_model[sk]
    active = 'active' if i == 0 else ''
    tabs_btn += f'<button class="tab-btn {active}" onclick="showTab(\'{sk}\')" style="border-top:3px solid {d["color"]}">{sk}</button>'
    feats_base = [f for f in art['feat_cols'] if not f.endswith('_isnan')]
    feats_nan  = [f for f in art['feat_cols'] if f.endswith('_isnan')]
    feat_chips = ''.join(f'<span class="feat-chip">{f}</span>' for f in feats_base)
    nan_chips  = ''.join(f'<span class="feat-chip nan-chip">{f}</span>' for f in feats_nan)
    nan_sec    = f'<div style="margin:6px 0 4px;font-size:0.82rem;color:#e67e22">NaN指示変数:</div><div>{nan_chips}</div>' if nan_chips else ''
    st = seg_stats.get(sk, {})
    ys = st.get('year', {})
    stats_html = ''
    for yr in ['2023-24','2025','2026']:
        if yr not in ys: continue
        a,r,n = ys[yr]['acc'], ys[yr]['roi'], ys[yr]['n']
        stats_html += f'<span style="margin-right:12px">{yr}: <strong style="color:{"#2e7d32" if a>=0.30 else "#555"}">{a:.1%}</strong> 的中 / ROI <span style="color:{"#2e7d32" if r>=0 else "#c62828"}">{r:+.2%}</span> ({n}R)</span>'
    note = art.get('note','')
    tabs_content += f'''
<div id="tab-{sk}" class="tab-content {active}">
  <div class="seg-header" style="border-left:5px solid {d["color"]}">
    <strong style="color:{d["color"]}">{sk}</strong> — {d["cond"]} &amp; クラス_rank≠1.0（新馬除外）<br>
    {stats_html}
    {"<br><small style='color:#888'>"+note+"</small>" if note else ""}
  </div>
  <div style="margin:10px 0 4px;font-size:0.82rem;color:#555">特徴量 ({len(feats_base)}個):</div>
  <div>{feat_chips}</div>
  {nan_sec}
</div>'''

html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>競馬AI v2 的中率モデルレポート 2026-06-13</title>
<style>
:root{{--green:#1b5e20;--bg:#f0f2f5;}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI','Yu Gothic',sans-serif;background:var(--bg);color:#212121;font-size:14px;line-height:1.6}}
nav{{background:#1b5e20;color:#fff;padding:0 24px;height:52px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;box-shadow:0 2px 8px rgba(0,0,0,.3)}}
nav a{{color:#a5d6a7;text-decoration:none;font-size:0.85rem}}
.hero{{background:linear-gradient(135deg,#1b5e20 0%,#2e7d32 50%,#388e3c 100%);color:#fff;padding:40px 24px;text-align:center}}
.hero h1{{font-size:1.8rem;font-weight:800;margin-bottom:8px}}
.hero .sub{{font-size:0.9rem;opacity:.8;margin-bottom:20px}}
.mission-badge{{display:inline-block;background:rgba(255,255,255,.15);border:2px solid rgba(255,255,255,.4);border-radius:50px;padding:8px 24px;font-size:1rem;font-weight:700}}
.container{{max-width:1100px;margin:0 auto;padding:28px 20px 60px}}
.section{{background:#fff;border-radius:10px;padding:22px 24px;margin-bottom:24px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.section-title{{font-size:1rem;font-weight:700;color:#1b5e20;border-bottom:2px solid #e8f5e9;padding-bottom:8px;margin-bottom:16px}}
.kpi-grid{{display:flex;flex-wrap:wrap;gap:14px}}
.kpi-card{{background:#f8f9fa;border-radius:10px;padding:14px 18px;min-width:160px;flex:1;text-align:center;border:1px solid #e0e0e0}}
.kpi-seg{{font-size:0.78rem;color:white;padding:2px 10px;border-radius:10px;display:inline-block;margin-bottom:6px}}
.kpi-val{{font-size:1.9rem;font-weight:800;margin:4px 0}}
.kpi-label{{font-size:0.75rem;color:#666}}
.kpi-sub{{font-size:0.72rem;color:#999;margin-top:2px}}
table.summary{{border-collapse:collapse;width:100%;font-size:0.88rem}}
table.summary th{{background:#1b5e20;color:#fff;padding:8px 10px;text-align:center}}
table.summary td{{padding:8px 10px;border-bottom:1px solid #eee;text-align:center}}
table.summary tr:hover td{{background:#f5f5f5}}
.tab-btns{{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:16px}}
.tab-btn{{padding:7px 14px;border:none;border-radius:6px 6px 0 0;background:#e8f5e9;cursor:pointer;font-size:0.82rem;font-weight:600;border-top:3px solid transparent}}
.tab-btn.active{{background:#fff;border-bottom:none;font-weight:700}}
.tab-content{{display:none;padding:16px;background:#fff;border:1px solid #e0e0e0;border-radius:0 8px 8px 8px}}
.tab-content.active{{display:block}}
.seg-header{{padding:10px 14px;background:#f8f9fa;border-radius:6px;margin-bottom:12px;line-height:1.7}}
.feat-chip{{display:inline-block;background:#e8f5e9;color:#1b5e20;font-size:0.78rem;padding:2px 8px;border-radius:10px;margin:2px 3px}}
.nan-chip{{background:#fff3e0;color:#e65100}}
.di-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.di-item{{background:#f8f9fa;padding:12px 14px;border-radius:6px;border-left:3px solid #4caf50}}
.di-label{{font-size:0.78rem;color:#666;margin-bottom:2px}}
.di-val{{font-size:0.88rem;font-weight:600}}
small{{font-size:0.8em;color:#888;margin-left:4px}}
.notice-box{{background:#e8f5e9;border:2px solid #4caf50;border-radius:8px;padding:14px 18px;margin-bottom:20px}}
</style>
</head>
<body>
<nav>
  <span style="font-weight:700">🏇 競馬AI v2 — 的中率モデルレポート</span>
  <span style="display:flex;gap:16px">
    <a href="newspaper_260613.html">📰 新聞（6/13）</a>
    <a href="model_report_20260613.html">📈 ROIモデルレポート</a>
    <span style="opacity:.7;font-size:0.82rem">更新: 2026-06-13</span>
  </span>
</nav>
<div class="hero">
  <h1>競馬AI v2 的中率最大化モデル レポート</h1>
  <div class="sub">clogit + isotonic calibration | 5セグメント | OOS 2023-2026 | 新聞予想で使用中</div>
  <div class="mission-badge">🎯 予想新聞はこのモデルのスコア順</div>
</div>
<div class="container">

  <div class="notice-box">
    <strong>このモデルが新聞の予想を担当しています</strong><br>
    ROIモデル（<a href="model_report_20260613.html" style="color:#1565c0">ROIレポート</a>）は指標1位全買いROIの最大化を目的とします。<br>
    的中率モデル（このレポート）は1着的中率の最大化を目的とし、<code>accuracy_model.pkl</code> として保存されています。
    新聞の馬順・特徴量ヒートマップはこちらのモデルスコアに基づきます。
  </div>

  <div class="section">
    <div class="section-title">OOS 的中率（指標1位全買い）</div>
    <div class="kpi-grid">{kpi_cards}</div>
  </div>

  <div class="section">
    <div class="section-title">セグメント別 年別 的中率 / ROI サマリー</div>
    <table class="summary">
      <thead><tr>
        <th>セグメント</th>
        <th>2023-24 的中率</th><th>2023-24 ROI</th>
        <th>2025 的中率</th><th>2025 ROI</th>
        <th>2026 的中率</th><th>2026 ROI</th>
      </tr></thead>
      <tbody>{tbl_rows}</tbody>
    </table>
    <p style="font-size:0.78rem;color:#888;margin-top:8px">
      * 的中率 = 指標1位馬が1着になる確率。1番人気の的中率（参考: ダ長34% / ダ短35% / 芝短29%）との比較<br>
      * OOS = Out-of-Sample。train:2013-2021 / val:2022 / OOS:2023-2026
    </p>
  </div>

  <div class="section" id="segments">
    <div class="section-title">セグメント別 特徴量詳細</div>
    <div class="tab-btns">{tabs_btn}</div>
    {tabs_content}
  </div>

  <div class="section">
    <div class="section-title">設計情報</div>
    <div class="di-grid">
      <div class="di-item"><div class="di-label">モデル</div><div class="di-val">Conditional Logit + Isotonic Calibration</div></div>
      <div class="di-item"><div class="di-label">学習期間</div><div class="di-val">train: 2013-2021 / val: 2022 / OOS: 2023-2026</div></div>
      <div class="di-item"><div class="di-label">選択指標</div><div class="di-val">acc_2325 = 1着的中率（2023-24 と 2025 の加重平均）</div></div>
      <div class="di-item"><div class="di-label">正則化</div><div class="di-val">L2 = 0.003, Adam optimizer, early stopping on val NLL</div></div>
      <div class="di-item"><div class="di-label">特徴量探索</div><div class="di-val">greedy forward selection（候補全数NaN時のみ停止）</div></div>
      <div class="di-item"><div class="di-label">新馬除外</div><div class="di-val">全セグメント クラス_rank ≠ 1.0</div></div>
    </div>
  </div>

</div>
<script>
function showTab(k) {{
  document.querySelectorAll('.tab-content').forEach(e=>e.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(e=>e.classList.remove('active'));
  document.getElementById('tab-'+k).classList.add('active');
  event.target.classList.add('active');
}}
</script>
</body>
</html>"""

out = r'C:\horse_racing_ai_v2\docs\accuracy_model_report_20260613.html'
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"\n生成完了: {out}")
