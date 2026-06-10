# coding: utf-8
"""
gen_accuracy_report.py  - 的中率モデル HTML レポート生成
model_report_20260608.html と同スタイル（インディゴ・KPIカード・タブ・beta値）
usage: python src/gen_accuracy_report.py
"""
import os, sys, pickle
import numpy as np
from datetime import date

sys.stdout.reconfigure(encoding='utf-8')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, 'models', 'accuracy_model.pkl'), 'rb') as f:
    MODEL = pickle.load(f)

TODAY = date.today().strftime('%Y-%m-%d')

FAV  = {'ダ長': 0.3403, 'ダ短': 0.3490, '芝短': 0.2869, '芝中': 0.3321, '芝長': 0.3605}
COND = {'芝長': '芝 &gt;2000m', '芝中': '芝 1401–2000m', '芝短': '芝 ≤1400m',
        'ダ長': 'ダ &gt;1400m', 'ダ短': 'ダ ≤1400m'}
ORDER = ['芝長', '芝中', '芝短', 'ダ長', 'ダ短']

# 改善経緯
TIMELINE = [
    ('2026-06-07', 'greedy forward selection 開始（acc_2325 指標）',
     '全5セグメントで的中率最大化モデルの探索を開始。SEEDはゼロから構築。'),
    ('2026-06-08', '芝長 shiba_long_acc_v1 確定（acc_2325=34.75%）',
     '30特徴で芝長モデルを確立。2025=36.70%と高い精度。'),
    ('2026-06-09', 'ダ短 da_short_acc_v3 確定（acc_2325=30.79%）',
     '40特徴。新候補（タイム指数slope、上り3F slope等）を全探索。'),
    ('2026-06-10 AM', '新候補特徴量追加探索（1走前_単勝オッズ、近5走_タイム指数_min等）',
     '芝中で1走前_単勝オッズ・近5走_タイム指数_minが有効（+0.31pp）。'),
    ('2026-06-10', 'L2=0.003/0.010 変更探索 + 新特徴量（近3走_上り3F_min 等）',
     '前走_1番人気フラグ・前走_人気着順差・近3走_上り3F_minを追加。芝中+0.04pp。'),
    ('2026-06-10', 'ランダムサーチ（局所最適脱出）で芝長 35.11% 達成',
     'greedy の局所最適を remove/add/swap 500回で脱出。芝長が+0.36pp改善。'),
]

def pct(v, digits=2):
    return f'{v*100:.{digits}f}%'

def roi_cls(v):
    if v >= 0: return 'roi-pos'
    if v >= -0.10: return 'roi-neu'
    return 'roi-neg'

def acc_cls(model_acc, fav_acc):
    return 'roi-pos' if model_acc >= fav_acc else ('roi-neu' if model_acc >= fav_acc - 0.02 else 'roi-neg')

# ─── KPI カード ───────────────────────────────────────────────────────────────
def kpi_cards():
    out = ''
    for seg in ORDER:
        pkg = MODEL[seg]
        ma  = pkg.get('acc_2325', 0)
        fa  = FAV[seg]
        gap = ma - fa
        a26 = pkg.get('acc_2526', 0)
        r26 = pkg.get('oos_roi_2526', 0)
        ok  = gap >= 0
        tag = '<span class="kpi-tag tag-ok">目標達成 ✅</span>' if ok else f'<span class="kpi-tag tag-ng">残り {abs(gap)*100:.2f}pp</span>'
        out += f'''
      <div class="kpi-card {'goal-ok' if ok else 'goal-ng'}">
        <div class="kpi-label">AI的中率 2325合算</div>
        <div class="kpi-seg">{seg}</div>
        <div class="kpi-roi {acc_cls(ma,fa)}">{pct(ma)}</div>
        <div class="kpi-sub">{COND[seg]} | 1番人気={pct(fa)} / gap={"+"+pct(gap) if gap>=0 else pct(gap)}</div>
        <div class="kpi-sub" style="margin-top:3px">25+26 ROI <span class="{roi_cls(r26)}">{'+' if r26>=0 else ''}{pct(r26)}</span></div>
        {tag}
      </div>'''
    return out

# ─── サマリーテーブル ─────────────────────────────────────────────────────────
def summary_table():
    rows = ''
    for seg in ORDER:
        pkg = MODEL[seg]
        ma  = pkg.get('acc_2325', 0)
        fa  = FAV[seg]
        gap = ma - fa
        a26 = pkg.get('acc_2526', 0)
        r26 = pkg.get('oos_roi_2526', 0)
        ver = pkg.get('version', '?')
        n   = len([f for f in pkg['feat_cols'] if not f.endswith('_isnan')])
        ok  = gap >= 0
        rows += f'''
          <tr {'class="ok-row"' if ok else ''}>
            <td><strong>{seg}</strong></td>
            <td>{COND[seg]}</td>
            <td>{n}</td>
            <td class="num {acc_cls(ma,fa)}">{pct(ma)}</td>
            <td class="num" style="color:#9c27b0">{pct(fa)}</td>
            <td class="num {'roi-pos' if gap>=0 else 'roi-neg'}">{'+' if gap>=0 else ''}{pct(gap)}</td>
            <td class="num {roi_cls(r26)}">{'+' if r26>=0 else ''}{pct(r26)}</td>
            <td class="num">{pct(a26)}</td>
            <td class="mono" style="font-size:0.8em">{ver}</td>
          </tr>'''
    return rows

# ─── タブボタン ───────────────────────────────────────────────────────────────
def tab_buttons():
    btns = ''
    for i, seg in enumerate(ORDER):
        active = 'active' if i == 0 else ''
        btns += f'<button class="tab-btn {active}" onclick="showTab(\'{seg}\')">{seg}</button>'
    return btns

# ─── 特徴量カード（beta値付き） ────────────────────────────────────────────────
def feat_cards(seg):
    pkg  = MODEL[seg]
    cols = pkg['feat_cols']
    beta = pkg.get('coef', np.zeros(len(cols)))
    cards = ''
    for f, b in zip(cols, beta):
        is_isnan = f.endswith('_isnan')
        cls = 'feat-item isnan' if is_isnan else 'feat-item'
        bcls = 'beta-pos' if b > 0 else 'beta-neg'
        sign = '+' if b > 0 else ''
        label = f.replace('_isnan', ' [NaN指示]') if is_isnan else f
        cards += f'<div class="{cls}"><div class="feat-name">{label}</div><div class="feat-beta">β = <span class="{bcls}">{sign}{b:.4f}</span></div></div>'
    return f'<div class="feat-grid">{cards}</div>'

# ─── セグメント詳細パネル ─────────────────────────────────────────────────────
def seg_panels():
    out = ''
    for i, seg in enumerate(ORDER):
        pkg  = MODEL[seg]
        ma   = pkg.get('acc_2325', 0)
        fa   = FAV[seg]
        gap  = ma - fa
        a26  = pkg.get('acc_2526', 0)
        r26  = pkg.get('oos_roi_2526', 0)
        ver  = pkg.get('version', '?')
        n    = len([f for f in pkg['feat_cols'] if not f.endswith('_isnan')])
        active = 'active' if i == 0 else ''

        # ROI バー（目盛: -40% ~ +40%、幅100px）
        def bar(v):
            pct_bar = min(max(v / 0.4, -1), 1)
            if pct_bar >= 0:
                return f'<div class="roi-bar p" style="width:{pct_bar*50:.1f}%; margin-left:50%"></div>'
            else:
                w = abs(pct_bar) * 50
                return f'<div class="roi-bar n" style="width:{w:.1f}%; margin-left:{50-w:.1f}%"></div>'

        out += f'''
    <div class="tab-panel {active}" id="panel-{seg}">
      <div class="design-grid" style="margin-bottom:16px">
        <div class="design-item">
          <div class="di-label">AI 的中率 (2325合算)</div>
          <div class="di-val" style="font-size:1.6rem;{" color:var(--green)" if gap>=0 else ""}">{pct(ma)}</div>
          <div style="font-size:0.8rem;color:#888;margin-top:4px">1番人気 {pct(fa)} / 差 <span class="{'roi-pos' if gap>=0 else 'roi-neg'}">{'+' if gap>=0 else ''}{pct(gap)}</span></div>
        </div>
        <div class="design-item">
          <div class="di-label">25+26 合算</div>
          <div class="di-val" style="font-size:1.6rem">{pct(a26)}</div>
          <div style="font-size:0.8rem;color:#888;margin-top:4px">ROI <span class="{roi_cls(r26)}">{'+' if r26>=0 else ''}{pct(r26)}</span></div>
        </div>
        <div class="design-item">
          <div class="di-label">条件</div>
          <div class="di-val">{COND[seg]}</div>
          <div style="font-size:0.8rem;color:#888;margin-top:4px">新馬除外 (クラス_rank≠1.0)</div>
        </div>
        <div class="design-item">
          <div class="di-label">バージョン / 特徴数</div>
          <div class="di-val" style="font-family:monospace;font-size:0.9rem">{ver}</div>
          <div style="font-size:0.8rem;color:#888;margin-top:4px">{n} 特徴量</div>
        </div>
      </div>
      <div class="section-title" style="margin-bottom:12px">使用特徴量 ({n}個) と係数</div>
      {feat_cards(seg)}
    </div>'''
    return out

# ─── タイムライン ──────────────────────────────────────────────────────────────
def timeline():
    items = ''
    for dt, title, body in TIMELINE:
        items += f'''
      <li>
        <div class="timeline-title">{dt} — {title}</div>
        <div class="timeline-body">{body}</div>
      </li>'''
    return f'<ul class="timeline">{items}</ul>'

# ─── HTML 本体 ────────────────────────────────────────────────────────────────
html = f'''<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>競馬AI v2 — 的中率最大化モデル レポート {TODAY}</title>
  <style>
    :root {{
      --indigo:    #1a237e;
      --indigo2:   #283593;
      --indigo3:   #3949ab;
      --green:     #2e7d32;
      --green-bg:  #e8f5e9;
      --red:       #c62828;
      --red-bg:    #ffebee;
      --orange:    #e65100;
      --orange-bg: #fff3e0;
      --gray:      #546e7a;
      --bg:        #f0f2f5;
      --card:      #ffffff;
      --border:    #e0e0e0;
    }}
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ font-family:'Segoe UI','Helvetica Neue',sans-serif; background:var(--bg); color:#212121; font-size:14px; line-height:1.6; }}

    nav {{ background:var(--indigo); color:#fff; padding:0 24px; height:52px; display:flex; align-items:center; justify-content:space-between; position:sticky; top:0; z-index:100; box-shadow:0 2px 8px rgba(0,0,0,.3); }}
    nav .brand {{ font-size:1rem; font-weight:700; letter-spacing:.5px; }}
    nav .nav-links {{ display:flex; gap:20px; font-size:0.82rem; }}
    nav .nav-links a {{ color:rgba(255,255,255,.8); text-decoration:none; }}
    nav .nav-links a:hover {{ color:#fff; }}

    .hero {{ background:linear-gradient(135deg,var(--indigo) 0%,#283593 50%,#1565c0 100%); color:#fff; padding:48px 24px 40px; text-align:center; }}
    .hero h1 {{ font-size:1.9rem; font-weight:800; margin-bottom:10px; letter-spacing:-.5px; }}
    .hero .subtitle {{ font-size:0.95rem; opacity:.85; margin-bottom:28px; }}
    .mission-badge {{ display:inline-block; background:rgba(255,255,255,.15); border:2px solid rgba(255,255,255,.4); border-radius:50px; padding:8px 24px; font-size:1.05rem; font-weight:700; backdrop-filter:blur(4px); }}
    .mission-badge .check {{ color:#69f0ae; margin-right:6px; }}
    .mission-badge .pending {{ color:#ffcc80; margin-right:6px; }}

    .container {{ max-width:1100px; margin:0 auto; padding:32px 20px 60px; }}
    .section {{ margin-bottom:40px; }}
    .section-title {{ font-size:1.05rem; font-weight:700; color:var(--indigo2); border-left:4px solid var(--indigo3); padding-left:10px; margin-bottom:16px; }}

    .kpi-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-bottom:32px; }}
    .kpi-card {{ background:var(--card); border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,.08); padding:16px 14px; border-top:4px solid var(--indigo3); transition:transform .15s; }}
    .kpi-card:hover {{ transform:translateY(-2px); box-shadow:0 4px 16px rgba(0,0,0,.12); }}
    .kpi-card.goal-ok {{ border-top-color:var(--green); }}
    .kpi-card.goal-ng {{ border-top-color:var(--red); }}
    .kpi-label {{ font-size:0.72rem; color:#888; margin-bottom:4px; font-weight:600; text-transform:uppercase; letter-spacing:.5px; }}
    .kpi-seg   {{ font-size:0.88rem; font-weight:700; color:var(--indigo2); margin-bottom:6px; }}
    .kpi-roi   {{ font-size:1.7rem; font-weight:800; line-height:1; margin-bottom:8px; }}
    .kpi-sub   {{ font-size:0.75rem; color:#777; }}
    .kpi-tag   {{ display:inline-block; padding:2px 10px; border-radius:20px; font-size:0.72rem; font-weight:700; margin-top:6px; }}
    .tag-ok    {{ background:var(--green-bg); color:var(--green); }}
    .tag-ng    {{ background:var(--red-bg);   color:var(--red); }}

    .roi-pos {{ color:var(--green); }}
    .roi-neg {{ color:var(--red); }}
    .roi-neu {{ color:var(--orange); }}

    .box {{ background:var(--card); border-radius:10px; box-shadow:0 2px 8px rgba(0,0,0,.08); padding:20px 24px; margin-bottom:20px; overflow-x:auto; }}
    table {{ border-collapse:collapse; width:100%; font-size:0.86rem; }}
    th {{ background:var(--indigo2); color:#fff; padding:9px 14px; text-align:left; white-space:nowrap; font-size:0.82rem; }}
    td {{ padding:8px 14px; border-bottom:1px solid var(--border); white-space:nowrap; }}
    tr:last-child td {{ border-bottom:none; }}
    tr:hover td {{ background:#f5f7ff; }}
    .ok-row td {{ background:#f1f8e9 !important; }}
    .ok-row:hover td {{ background:#e8f5e9 !important; }}
    .num {{ font-variant-numeric:tabular-nums; }}
    .mono {{ font-family:monospace; }}

    .tab-bar {{ display:flex; gap:4px; flex-wrap:wrap; margin-bottom:16px; }}
    .tab-btn {{ padding:7px 18px; border:2px solid transparent; border-radius:6px 6px 0 0; background:#e0e4ef; color:var(--gray); font-size:0.84rem; font-weight:600; cursor:pointer; transition:all .15s; }}
    .tab-btn:hover {{ background:#d0d6f0; }}
    .tab-btn.active {{ background:var(--card); color:var(--indigo); border-color:var(--border); border-bottom-color:var(--card); }}
    .tab-panel {{ display:none; }}
    .tab-panel.active {{ display:block; }}

    .feat-grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:10px; margin-bottom:16px; }}
    .feat-item {{ background:#f5f7ff; border:1px solid #c5cae9; border-radius:8px; padding:10px 12px; }}
    .feat-name {{ font-weight:700; font-size:0.85rem; color:var(--indigo); margin-bottom:2px; }}
    .feat-beta {{ font-size:0.78rem; color:#555; }}
    .beta-pos  {{ color:var(--green); font-weight:700; }}
    .beta-neg  {{ color:var(--red); font-weight:700; }}
    .feat-item.isnan {{ background:#fafafa; border-color:#ddd; }}

    .design-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
    .design-item {{ background:#f5f7ff; border-radius:8px; padding:14px 16px; }}
    .di-label {{ font-size:0.75rem; color:#888; font-weight:600; text-transform:uppercase; margin-bottom:4px; }}
    .di-val   {{ font-weight:700; color:var(--indigo); }}

    .timeline {{ list-style:none; position:relative; padding-left:28px; }}
    .timeline::before {{ content:''; position:absolute; left:8px; top:8px; bottom:8px; width:2px; background:var(--border); }}
    .timeline li {{ position:relative; margin-bottom:20px; }}
    .timeline li::before {{ content:''; position:absolute; left:-24px; top:6px; width:12px; height:12px; border-radius:50%; background:var(--indigo3); border:2px solid #fff; box-shadow:0 0 0 2px var(--indigo3); }}
    .timeline-title {{ font-weight:700; color:var(--indigo2); margin-bottom:4px; }}
    .timeline-body  {{ font-size:0.85rem; color:#555; }}

    footer {{ text-align:center; padding:24px; color:#aaa; font-size:0.78rem; background:var(--bg); }}

    @media(max-width:768px) {{
      .kpi-grid {{ grid-template-columns:repeat(2,1fr); }}
      .design-grid {{ grid-template-columns:1fr; }}
      nav .nav-links {{ display:none; }}
    }}
  </style>
</head>
<body>

<nav>
  <span class="brand">🏇 競馬AI v2 — 的中率最大化モデル</span>
  <div class="nav-links">
    <a href="#summary">サマリー</a>
    <a href="#model">モデル設計</a>
    <a href="#segments">セグメント詳細</a>
    <a href="#history">改善経緯</a>
  </div>
</nav>

<div class="hero">
  <h1>的中率最大化モデル レポート</h1>
  <div class="subtitle">競馬AI v2 — clogit + Isotonic Calibration &nbsp;|&nbsp; {TODAY}</div>
  <div class="mission-badge">
    <span class="pending">🎯</span> 目標: 全5セグメントで1番人気の的中率を超える
  </div>
</div>

<div class="container">

  <!-- KPI -->
  <div id="summary" class="section" style="padding-top:8px">
    <div class="section-title">的中率 2323–25 合算（指標1位全買い）</div>
    <div class="kpi-grid">
      {kpi_cards()}
    </div>

    <!-- サマリーテーブル -->
    <div class="box">
      <table>
        <thead>
          <tr>
            <th>セグメント</th><th>条件</th><th>特徴数</th>
            <th>AI的中率<br>2325合算</th><th>1番人気<br>目標</th><th>差</th>
            <th>25+26 ROI</th><th>25+26 的中率</th><th>バージョン</th>
          </tr>
        </thead>
        <tbody>
          {summary_table()}
        </tbody>
      </table>
    </div>
  </div>

  <!-- モデル設計 -->
  <div id="model" class="section">
    <div class="section-title">モデル設計</div>
    <div class="box">
      <div class="design-grid">
        <div class="design-item"><div class="di-label">アーキテクチャ</div><div class="di-val">clogit（条件付きロジット）+ Isotonic Calibration</div></div>
        <div class="design-item"><div class="di-label">選択指標</div><div class="di-val">acc_2325 = (acc2324×n2324 + acc2025×n2025) / (n2324+n2025)</div></div>
        <div class="design-item"><div class="di-label">学習期間 / 評価期間</div><div class="di-val">train: 2013–2021 / val: 2022 / OOS: 2023–2026</div></div>
        <div class="design-item"><div class="di-label">正則化</div><div class="di-val">L2 = 0.006（Adam optimizer、val NLL early stopping）</div></div>
        <div class="design-item"><div class="di-label">新馬除外</div><div class="di-val">クラス_rank ≠ 1.0（全セグメント共通）</div></div>
        <div class="design-item"><div class="di-label">特徴量制約</div><div class="di-val">JVLink API 出馬表（前日〜当日12時）で取得可能なもの限定。賞金・今走オッズ禁止。</div></div>
        <div class="design-item"><div class="di-label">探索手法</div><div class="di-val">greedy forward selection + ランダムサーチ（局所最適脱出）</div></div>
        <div class="design-item"><div class="di-label">NaN 処理</div><div class="di-val">train NaN率 &gt;5% の特徴に _isnan 指示変数を自動追加</div></div>
      </div>
    </div>
  </div>

  <!-- セグメント詳細 -->
  <div id="segments" class="section">
    <div class="section-title">セグメント詳細（特徴量・係数）</div>
    <div class="box">
      <div class="tab-bar">
        {tab_buttons()}
      </div>
      {seg_panels()}
    </div>
  </div>

  <!-- 改善経緯 -->
  <div id="history" class="section">
    <div class="section-title">改善経緯</div>
    <div class="box">
      {timeline()}
    </div>
  </div>

</div>

<footer>
  競馬AI v2 — 的中率最大化モデル | 生成: {TODAY} | accuracy_model.pkl<br>
  clogit + Isotonic Calibration | 新馬除外済み | JVLink API 出馬表データのみ使用
</footer>

<script>
function showTab(seg) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + seg).classList.add('active');
  event.target.classList.add('active');
}}
</script>
</body>
</html>
'''

out = os.path.join(BASE_DIR, 'docs', f'accuracy_model_report_{TODAY.replace("-","")}.html')
with open(out, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'レポート生成完了: {out}')
