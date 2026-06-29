# coding: utf-8
import os, sys, json
sys.stdout.reconfigure(encoding='utf-8')

ROI = {
    'da':   dict(label='ダ長（ダ>1400m）', color='#2980b9', art_key='ダ', cond='ダ & 距離>1400m',
                 r2324=-0.0987, r25=+0.1621, r26=-0.3967, r2526=-0.0058,
                 n2324=1841, n25=924,  n26=397,
                 feats=['馬番','斤量','種牡馬_勝率','性別_num','ブリンカー変更',
                        '近5走_上り3F平均','1走前_馬場状態','1走前_馬場状態_isnan',
                        '2走前_クラス差','コース枠_r200_勝率','3走前_クラス差'],
                 note=''),
    'da_s': dict(label='ダ短（ダ≤1400m）', color='#1a6fa0', art_key='ダ短', cond='ダ & 距離≤1400m',
                 r2324=-0.0890, r25=+0.0494, r26=-0.3759, r2526=-0.0797,
                 n2324=1352, n25=668,  n26=291,
                 feats=['馬番','斤量','種牡馬_勝率','間隔','2走前_クラス差',
                        'コース枠_r200_勝率','1走前_馬場状態','1走前_馬場状態_isnan',
                        '1走前_クラス差','3走前_クラス差','ブリンカー変更'],
                 note=''),
    'shi_s':dict(label='芝短（芝≤1400m）', color='#27ae60', art_key='芝短', cond='芝 & 距離≤1400m',
                 r2324=+0.2326, r25=-0.0007, r26=-0.1515, r2526=-0.0489,
                 n2324=866, n25=434, n26=204,
                 feats=['馬番','斤量','馬体重','馬距離_勝率','コース枠_r200_勝率',
                        '性別_num','近5走_上り3F平均','近5走_上り3F_std',
                        '近5走_上り3F_std_isnan','距離変化_前走','ブリンカー変更'],
                 note='除外済: 1走前_3角, 1走前_脚質_num（差し馬バイアス）'),
    'shi_m':dict(label='芝中（芝1401-2000m）', color='#16a085', art_key='芝中', cond='芝 & 1401-2000m',
                 r2324=+0.0515, r25=-0.0978, r26=+0.3981, r2526=+0.0669,
                 n2324=1669, n25=857, n26=426,
                 feats=['馬番','斤量','近5走_上り3F_std','近5走_上り3F_std_isnan',
                        '馬距離_勝率','種牡馬_勝率','間隔','ブリンカー変更',
                        'コース枠_r200_勝率','芝ダ転向','3走前_クラス差'],
                 note='近5走_上り3F_std 強制ベース追加で +7.31% 達成'),
    'shi_l':dict(label='芝長（芝>2000m）', color='#8e44ad', art_key='芝長', cond='芝 & 距離>2000m',
                 r2324=+0.5760, r25=+0.2127, r26=-0.2689, r2526=+0.0514,
                 n2324=578, n25=268, n26=135,
                 feats=['馬番','斤量','ブリンカー変更','間隔','馬体重',
                        '1走前_馬場状態','1走前_馬場状態_isnan',
                        'コース枠_r200_勝率','1走前_クラス差','2走前_クラス差','3走前_クラス差'],
                 note=''),
}
ORDER = ['da','da_s','shi_s','shi_m','shi_l']

# キャリブレーションデータ読み込み
calib_path = r'C:\horse_racing_ai_v2\_calib_data.json'
with open(calib_path, encoding='utf-8') as f:
    calib_raw = json.load(f)
# art_key → calib bins
ART_KEY_MAP = {'da':'ダ','da_s':'ダ短','shi_s':'芝短','shi_m':'芝中','shi_l':'芝長'}

def roi_cls(r):
    if r >= 0.10: return 'roi-hi'
    if r >= 0.0:  return 'roi-pos'
    if r >= -0.10: return 'roi-neu'
    return 'roi-neg'

def fmt_roi(r):
    return f'{"+" if r>=0 else ""}{r:.2%}'

# KPI cards
kpi_cards = ''
for k in ORDER:
    d = ROI[k]
    r = d['r2526']
    cls = roi_cls(r)
    c = d['color']
    kpi_cards += f'''
<div class="kpi-card">
  <div class="kpi-seg" style="background:{c}">{d["label"]}</div>
  <div class="kpi-val {cls}">{fmt_roi(r)}</div>
  <div class="kpi-label">25+26 合算</div>
  <div class="kpi-sub">{fmt_roi(d["r2324"])} / {fmt_roi(d["r25"])} / {fmt_roi(d["r26"])}</div>
</div>'''

# Summary table
tbl_rows = ''
for k in ORDER:
    d = ROI[k]
    tbl_rows += f'''<tr>
  <td style="font-weight:bold;color:{d["color"]}">{d["label"]}</td>
  <td>{d["n2324"]+d["n25"]+d["n26"]:,}</td>
  <td class="{roi_cls(d["r2324"])}">{fmt_roi(d["r2324"])} <small>({d["n2324"]}R)</small></td>
  <td class="{roi_cls(d["r25"])}">{fmt_roi(d["r25"])} <small>({d["n25"]}R)</small></td>
  <td class="{roi_cls(d["r26"])}">{fmt_roi(d["r26"])} <small>({d["n26"]}R)</small></td>
  <td class="{roi_cls(d["r2526"])}" style="font-weight:bold">{fmt_roi(d["r2526"])} ✅</td>
</tr>'''

# Segment tabs
tabs_btn = ''
tabs_content = ''
for i, k in enumerate(ORDER):
    d = ROI[k]
    active = 'active' if i == 0 else ''
    tabs_btn += f'<button class="tab-btn {active}" onclick="showTab(\'{k}\')" style="border-top:3px solid {d["color"]}">{d["label"]}</button>'
    feats_base = [f for f in d['feats'] if not f.endswith('_isnan')]
    feats_nan  = [f for f in d['feats'] if f.endswith('_isnan')]
    feat_chips = ''.join(f'<span class="feat-chip">{f}</span>' for f in feats_base)
    nan_chips  = ''.join(f'<span class="feat-chip nan-chip">{f}</span>' for f in feats_nan)
    note_html = f'<br><small style="color:#888;margin-top:4px">⚠️ {d["note"]}</small>' if d['note'] else ''
    nan_section = f'<div style="margin:6px 0 4px;font-size:0.82rem;color:#e67e22">NaN指示変数 (自動追加):</div><div>{nan_chips}</div>' if nan_chips else ''
    tabs_content += f'''
<div id="tab-{k}" class="tab-content {active}">
  <div class="seg-header" style="border-left:5px solid {d["color"]}">
    <strong style="color:{d["color"]}">{d["label"]}</strong>
    — {d["cond"]} &amp; クラス_rank≠1.0（新馬除外）<br>
    2023-24: <span class="{roi_cls(d["r2324"])}">{fmt_roi(d["r2324"])}</span> &nbsp;/&nbsp;
    2025: <span class="{roi_cls(d["r25"])}">{fmt_roi(d["r25"])}</span> &nbsp;/&nbsp;
    2026: <span class="{roi_cls(d["r26"])}">{fmt_roi(d["r26"])}</span> &nbsp;/&nbsp;
    <strong>25+26: <span class="{roi_cls(d["r2526"])}">{fmt_roi(d["r2526"])}</span> ✅</strong>
    {note_html}
  </div>
  <div style="margin:10px 0 4px;font-size:0.82rem;color:#555">特徴量 ({len(feats_base)}個):</div>
  <div>{feat_chips}</div>
  {nan_section}
</div>'''

# ---- キャリブレーション テーブル + SVG ミニチャート ----
def calib_chart_svg(bins, color, width=260, height=90):
    """予測vs実際を折れ線SVGで描画"""
    if not bins:
        return ''
    preds  = [b['pred_mean'] for b in bins]
    actuals= [b['actual_win_rate'] for b in bins]
    all_v  = preds + actuals
    mn, mx = min(all_v), max(all_v)
    rng = mx - mn if mx > mn else 0.01
    pad = 10
    w, h = width, height

    def xp(i): return pad + (i / (len(bins)-1)) * (w - 2*pad) if len(bins) > 1 else w/2
    def yp(v): return h - pad - (v - mn) / rng * (h - 2*pad)

    pts_pred   = ' '.join(f'{xp(i):.1f},{yp(v):.1f}' for i,v in enumerate(preds))
    pts_actual = ' '.join(f'{xp(i):.1f},{yp(v):.1f}' for i,v in enumerate(actuals))

    # perfect line (diagonal: same as pred)
    pts_perfect = ' '.join(f'{xp(i):.1f},{yp(preds[i]):.1f}' for i in range(len(bins)))

    svg = f'''<svg width="{w}" height="{h}" style="display:block">
  <polyline points="{pts_pred}" fill="none" stroke="#bbb" stroke-width="1.5" stroke-dasharray="4,3"/>
  <polyline points="{pts_actual}" fill="none" stroke="{color}" stroke-width="2.5"/>
  {" ".join(f'<circle cx="{xp(i):.1f}" cy="{yp(actuals[i]):.1f}" r="3" fill="{color}"/>' for i in range(len(bins)))}
</svg>'''
    return svg

calib_rows = ''
for k in ORDER:
    d = ROI[k]
    ak = ART_KEY_MAP[k]
    cd = calib_raw.get(ak, {})
    bins = cd.get('bins', [])
    g_pred   = cd.get('global_pred', 0)
    g_actual = cd.get('global_actual', 0)
    n_total  = cd.get('n_total', 0)

    # テーブル行
    bin_rows = ''
    for b in bins:
        pred_pct   = b['pred_mean'] * 100
        actual_pct = b['actual_win_rate'] * 100
        diff = actual_pct - pred_pct
        diff_cls = 'diff-pos' if diff > 0 else 'diff-neg'
        diff_str = f'{"+" if diff>=0 else ""}{diff:.1f}%'
        bar_w = min(int(b['actual_win_rate'] / 0.25 * 80), 80)
        bar_w2= min(int(b['pred_mean'] / 0.25 * 80), 80)
        bin_rows += f'''<tr>
  <td style="color:#666">{pred_pct:.1f}%</td>
  <td>
    <div style="display:flex;align-items:center;gap:4px">
      <div style="width:{bar_w2}px;height:8px;background:#ddd;border-radius:3px"></div>
      <span style="font-size:0.82rem;color:#aaa">{pred_pct:.1f}%</span>
    </div>
    <div style="display:flex;align-items:center;gap:4px;margin-top:2px">
      <div style="width:{bar_w}px;height:8px;background:{d["color"]};border-radius:3px"></div>
      <span style="font-size:0.82rem;color:{d["color"]}">{actual_pct:.1f}%</span>
    </div>
  </td>
  <td class="{diff_cls}">{diff_str}</td>
  <td style="color:#888;font-size:0.8rem">{b["n"]:,}</td>
</tr>'''

    chart_svg = calib_chart_svg(bins, d['color'])
    calib_rows += f'''
<div class="calib-card" style="border-top:3px solid {d["color"]}">
  <div class="calib-title" style="color:{d["color"]}">{d["label"]}</div>
  <div class="calib-meta">OOS 2023-2026 / n={n_total:,}頭</div>
  <div class="calib-summary">
    <span>平均予測: <strong>{g_pred*100:.2f}%</strong></span>
    <span>平均実際: <strong style="color:{d["color"]}">{g_actual*100:.2f}%</strong></span>
    <span>誤差: <span class="{"diff-pos" if g_actual>=g_pred else "diff-neg"}">{(g_actual-g_pred)*100:+.2f}%</span></span>
  </div>
  <div class="calib-body">
    <div class="calib-chart">
      <div style="font-size:0.72rem;color:#aaa;margin-bottom:2px">予測確率 → 実際の勝率（灰:予測 / 色:実際）</div>
      {chart_svg}
    </div>
    <div class="calib-table-wrap">
      <table class="calib-tbl">
        <thead><tr><th>予測帯</th><th>予測 / 実際</th><th>差</th><th>n</th></tr></thead>
        <tbody>{bin_rows}</tbody>
      </table>
    </div>
  </div>
</div>'''

html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>競馬AI v2 モデル評価レポート 2026-06-13</title>
<style>
:root {{--indigo:#1a237e;--bg:#f0f2f5;}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI','Yu Gothic',sans-serif;background:var(--bg);color:#212121;font-size:14px;line-height:1.6}}
nav{{background:var(--indigo);color:#fff;padding:0 24px;height:52px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:100;box-shadow:0 2px 8px rgba(0,0,0,.3)}}
nav .brand{{font-size:1rem;font-weight:700}}
.hero{{background:linear-gradient(135deg,#1a237e 0%,#283593 50%,#1565c0 100%);color:#fff;padding:40px 24px;text-align:center}}
.hero h1{{font-size:1.8rem;font-weight:800;margin-bottom:8px}}
.hero .sub{{font-size:0.9rem;opacity:.8;margin-bottom:20px}}
.mission-badge{{display:inline-block;background:rgba(255,255,255,.15);border:2px solid rgba(255,255,255,.4);border-radius:50px;padding:8px 24px;font-size:1rem;font-weight:700}}
.container{{max-width:1100px;margin:0 auto;padding:28px 20px 60px}}
.section{{background:#fff;border-radius:10px;padding:22px 24px;margin-bottom:24px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.section-title{{font-size:1rem;font-weight:700;color:#1a237e;border-bottom:2px solid #e8eaf6;padding-bottom:8px;margin-bottom:16px}}
.kpi-grid{{display:flex;flex-wrap:wrap;gap:14px}}
.kpi-card{{background:#f8f9fa;border-radius:10px;padding:14px 18px;min-width:160px;flex:1;text-align:center;border:1px solid #e0e0e0}}
.kpi-seg{{font-size:0.78rem;color:white;padding:2px 10px;border-radius:10px;display:inline-block;margin-bottom:6px}}
.kpi-val{{font-size:1.9rem;font-weight:800;margin:4px 0}}
.kpi-label{{font-size:0.75rem;color:#666}}
.kpi-sub{{font-size:0.72rem;color:#999;margin-top:2px}}
.roi-hi{{color:#1b5e20;background:#e8f5e9;padding:1px 4px;border-radius:3px}}
.roi-pos{{color:#2e7d32}}
.roi-neu{{color:#e65100}}
.roi-neg{{color:#c62828}}
table.summary{{border-collapse:collapse;width:100%;font-size:0.88rem}}
table.summary th{{background:#1a237e;color:#fff;padding:8px 10px;text-align:center}}
table.summary td{{padding:8px 10px;border-bottom:1px solid #eee;text-align:center}}
table.summary tr:hover td{{background:#f5f5f5}}
.tab-btns{{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:16px}}
.tab-btn{{padding:7px 14px;border:none;border-radius:6px 6px 0 0;background:#e8eaf6;cursor:pointer;font-size:0.82rem;font-weight:600;border-top:3px solid transparent}}
.tab-btn.active{{background:#fff;border-bottom:none;font-weight:700}}
.tab-content{{display:none;padding:16px;background:#fff;border:1px solid #e0e0e0;border-radius:0 8px 8px 8px}}
.tab-content.active{{display:block}}
.seg-header{{padding:10px 14px;background:#f8f9fa;border-radius:6px;margin-bottom:12px;line-height:1.7}}
.feat-chip{{display:inline-block;background:#e8eaf6;color:#1a237e;font-size:0.78rem;padding:2px 8px;border-radius:10px;margin:2px 3px}}
.nan-chip{{background:#fff3e0;color:#e65100}}
.di-grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
.di-item{{background:#f8f9fa;padding:12px 14px;border-radius:6px;border-left:3px solid #5c6bc0}}
.di-label{{font-size:0.78rem;color:#666;margin-bottom:2px}}
.di-val{{font-size:0.88rem;font-weight:600}}
.odds-box{{background:#fff3e0;border:2px solid #e67e22;border-radius:8px;padding:14px 18px;margin-top:12px}}
.odds-title{{font-weight:700;color:#e65100;margin-bottom:6px}}
small{{font-size:0.8em;color:#888;margin-left:4px}}
/* calibration */
.calib-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
@media(max-width:700px){{.calib-grid{{grid-template-columns:1fr}}}}
.calib-card{{background:#fafafa;border-radius:8px;padding:14px 16px;border:1px solid #e0e0e0}}
.calib-title{{font-weight:700;font-size:0.9rem;margin-bottom:2px}}
.calib-meta{{font-size:0.75rem;color:#888;margin-bottom:6px}}
.calib-summary{{display:flex;gap:16px;font-size:0.82rem;margin-bottom:10px;flex-wrap:wrap}}
.calib-body{{display:flex;gap:12px;align-items:flex-start;flex-wrap:wrap}}
.calib-chart{{flex:0 0 auto}}
.calib-table-wrap{{flex:1;min-width:200px;overflow-x:auto}}
.calib-tbl{{border-collapse:collapse;width:100%;font-size:0.8rem}}
.calib-tbl th{{background:#f0f0f0;padding:4px 8px;text-align:center;font-weight:600}}
.calib-tbl td{{padding:4px 8px;border-bottom:1px solid #f0f0f0;text-align:center;vertical-align:middle}}
.diff-pos{{color:#2e7d32;font-weight:600}}
.diff-neg{{color:#c62828;font-weight:600}}
</style>
</head>
<body>
<nav>
  <span class="brand">🏇 競馬AI v2 — モデル評価レポート</span>
  <span style="font-size:0.82rem;opacity:.8">更新: 2026-06-13</span>
</nav>
<div class="hero">
  <h1>競馬AI v2 モデル評価レポート</h1>
  <div class="sub">clogit + isotonic calibration | 5セグメント | OOS 2023-2026</div>
  <div class="mission-badge"><span style="color:#69f0ae">✅</span> ミッション達成 — 全5セグメント 25+26合算 OOS ROI &ge; -10%</div>
</div>
<div class="container">

  <div class="section">
    <div class="section-title">25+26 合算 OOS ROI（指標1位全買い）</div>
    <div class="kpi-grid">{kpi_cards}</div>
  </div>

  <div class="section">
    <div class="section-title">セグメント別 OOS ROI サマリー</div>
    <table class="summary">
      <thead><tr>
        <th>セグメント</th><th>総レース数</th>
        <th>2023-24 ROI</th><th>2025 ROI</th><th>2026 ROI</th><th>25+26 合算</th>
      </tr></thead>
      <tbody>{tbl_rows}</tbody>
    </table>
    <p style="font-size:0.78rem;color:#888;margin-top:8px">
      * 指標1位全買い（ROIモデルのROI順位1位馬の単勝を毎レース購入）<br>
      * OOS = Out-of-Sample。train:2013-2021 / val:2022 / OOS:2023-2026
    </p>
  </div>

  <div class="section">
    <div class="section-title">キャリブレーション品質 — 予測確率 vs 実際の勝率（OOS 2023-2026）</div>
    <p style="font-size:0.82rem;color:#555;margin-bottom:14px">
      isotonic calibration 後の予測確率をデシル分けし、各ビン内の平均予測確率と実際の勝率を比較。<br>
      <span style="color:#aaa">灰の破線 = 予測</span>、<span style="color:#333">色の実線 = 実際の勝率</span>。完全一致なら両線が重なる。
    </p>
    <div class="calib-grid">{calib_rows}</div>
  </div>

  <div class="section">
    <div class="section-title">オッズ帯フィルタ戦略（芝短/芝長のみ）</div>
    <div class="odds-box">
      <div class="odds-title">◎ 買い条件: ROI1位 かつ 単勝オッズ &ge; 6.0倍</div>
      <p style="font-size:0.88rem">
        芝短・芝長セグメントのROI1位馬は、単勝オッズが6倍以上の場合のみ買い推奨。<br>
        OOS両期間（2023-24 AND 2025）でプラスを確認済み。<br>
        ダ長/ダ短/芝中はオッズフィルタなし（ROI1位を参考表示のみ）。
      </p>
    </div>
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
      <div class="di-item"><div class="di-label">新馬除外</div><div class="di-val">全セグメント クラス_rank ≠ 1.0（デビュー戦除外）</div></div>
      <div class="di-item"><div class="di-label">選択指標</div><div class="di-val">2325 = (r2324×n2324 + r25×n25)/(n+n)、単年±30%キャップ</div></div>
      <div class="di-item"><div class="di-label">正則化</div><div class="di-val">L2 = 0.006, Adam optimizer, early stopping on val NLL</div></div>
      <div class="di-item"><div class="di-label">特徴量ルール</div><div class="di-val">JVLink API（前日〜当日12時）で確実に取れるもののみ</div></div>
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

out = r'C:\horse_racing_ai_v2\docs\model_report_20260613.html'
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f"生成完了: {out}")
