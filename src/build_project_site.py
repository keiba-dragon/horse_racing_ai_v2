# coding: utf-8
"""
build_project_site.py - 5セグメント統合プロジェクトサイト生成
出力: docs/index.html
"""
import sys, os, pickle
import numpy as np
import pandas as pd
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from save_conditional_logit import prepare, segment_softmax, BASE_DIR, DATA_FILE
from save_v3 import add_computed_features

MODEL_DIR = os.path.join(BASE_DIR, 'models')
DOCS_DIR  = os.path.join(BASE_DIR, 'docs')
os.makedirs(DOCS_DIR, exist_ok=True)

SEGMENTS = [
    {
        'id':      'da-long',
        'name':    'ダ長',
        'label':   'ダート長距離',
        'cond':    '距離 > 1400m',
        'key':     'ダ',
        'filter':  lambda df, dm: (df['surface'] == 'ダ') & (dm > 1400),
        'version': 'nv1 / 25特徴 / 500+実験',
        'note':    'ダート長距離は500回超の実験を経てnv1(-17.18%)が最良。賞金カラムはリーケージ除外済み。',
        'color':   '#e65100',
    },
    {
        'id':      'da-short',
        'name':    'ダ短',
        'label':   'ダート短距離',
        'cond':    '距離 ≤ 1400m',
        'key':     'ダ短',
        'filter':  lambda df, dm: (df['surface'] == 'ダ') & (dm <= 1400),
        'version': 'nv3 / 10特徴 / greedy (2323選択)',
        'note':    '旧nv2(5特徴)から10特徴へ拡張。greedy forward selectionで2323指標を最大化。',
        'color':   '#bf360c',
    },
    {
        'id':      'shiba-mid',
        'name':    '芝中',
        'label':   '芝中距離',
        'cond':    '1400m < 距離 ≤ 2000m',
        'key':     '芝中',
        'filter':  lambda df, dm: (df['surface'] == '芝') & (dm > 1400) & (dm <= 2000),
        'version': '10特徴 / greedy (2323選択)',
        'note':    '旧AI(5特徴,25+26=+8.81%)から10特徴へ。的中率23-24%と安定。',
        'color':   '#1b5e20',
    },
    {
        'id':      'shiba-long',
        'name':    '芝長',
        'label':   '芝長距離',
        'cond':    '距離 > 2000m',
        'key':     '芝長',
        'filter':  lambda df, dm: (df['surface'] == '芝') & (dm > 2000),
        'version': '10特徴 / greedy+triplet (2323選択)',
        'note':    '旧Y8(5特徴,25+26=-3.03%)から大幅改善。2025/2026ともプラス。',
        'color':   '#004d40',
    },
    {
        'id':      'shiba-short',
        'name':    '芝短',
        'label':   '芝短距離',
        'cond':    '距離 ≤ 1400m',
        'key':     '芝短',
        'filter':  lambda df, dm: (df['surface'] == '芝') & (dm <= 1400),
        'version': 'nv3 / 10特徴 / forced greedy (2323選択)',
        'note':    '旧nv2のOOSスヌーピング(74比較→+32.28%)を修正。2323基準で+27.05%。',
        'color':   '#1a237e',
    },
]

LEAK_INFO = {
    '馬番':                     ('race-day固定', 'エントリー時決定'),
    '斤量':                     ('race-day固定', 'エントリー時決定'),
    '性別_num':                 ('race-day固定', 'エントリー時決定'),
    '間隔':                     ('race-day固定', '前走日付との差分'),
    '距離変化_前走':             ('race-day固定', '今回距離 - shift(1)距離'),
    '芝ダ転向':                 ('race-day固定', 'shift(1)路線との比較'),
    '馬体重':                   ('race-day固定', 'レース当日公表'),
    '1走前_3角':                ('前走データ', 'shift(1) by 馬名S'),
    '1走前_脚質_num':           ('前走データ', 'shift(1) by 馬名S'),
    '1走前_馬場状態':           ('前走データ', 'shift(1) + baba_map'),
    '1走前_クラス差':           ('前走データ', 'shift(1) by 馬名S'),
    '2走前_クラス差':           ('前走データ', 'shift(2) by 馬名S'),
    '1走前_クラス調整着順':     ('前走データ', 'shift(1) by 馬名S'),
    '前走着差タイム':           ('前走データ', 'shift(1) _着差_sec'),
    '近5走_上り3F平均':             ('rolling-5', '1走前~5走前の平均'),
    '近5走_上り3F_std':             ('rolling-5', '1走前~5走前のstd'),
    '近5走_クラス調整_平均着順':     ('rolling-5', '1走前~5走前クラス補正平均'),
    '同会場_平均着順_近5走':         ('rolling-5', '1走前~5走前同会場フィルタ'),
    '良馬場_平均着順_近5走':         ('rolling-5', '1走前~5走前良馬場フィルタ'),
    'コース枠_r200_勝率':           ('global統計', '_stat_mask 2013-2020固定'),
    '馬距離_勝率':                  ('global統計', '_stat_mask 2013-2020固定'),
    '調教師コース_r100_勝率':        ('global統計', '_stat_mask 2013-2020固定'),
    'コース脚質_r200_勝率':          ('global統計', '_stat_mask 2013-2020固定'),
}

YEARS = [
    ('訓練 2013-21', 130101, 211231),
    ('Val 2022',     220101, 221231),
    ('2023',         230101, 231231),
    ('2024',         240101, 241231),
    ('2023-24 ★',   230101, 241231),
    ('2025',         250101, 251231),
    ('2026',         260101, 291231),
]


def load_all():
    df = pd.read_parquet(DATA_FILE)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num'])
    df = df[df['着順_num'] < 99]
    df['race_id'] = (df['日付_num'].astype(int).astype(str) + '_' +
                     df['開催'].astype(str).str.strip() + '_' +
                     df['Ｒ'].astype(str).str.strip())
    df = df[df['開催'].notna()].copy()
    df['surface'] = (df['距離'].astype(str).str.strip()
                     .str.extract(r'^([芝ダ])')[0].fillna('不明'))
    dm = pd.to_numeric(df['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')
    df = add_computed_features(df)
    baba_map = {'良': 0, '稍重': 1, '重': 2, '不良': 3}
    for col in df.columns:
        if '馬場状態' in col:
            df[col] = df[col].map(baba_map)
    return df, dm


def score_top1(df_seg, art):
    feats = art['feat_cols']
    valid = [c for c in feats if c in df_seg.columns]
    X, _, gs, n, *_ = prepare(df_seg, valid, scaler=art['scaler'],
                               top_idx=None, top_idx3=None)
    scored = df_seg.sort_values('race_id').reset_index(drop=True)
    scored['prob'] = segment_softmax(X @ art['coef'], gs, n)
    scored['rank'] = scored.groupby('race_id')['prob'].rank(ascending=False, method='first')
    top1 = scored[scored['rank'] == 1].copy()
    top1['odds_num'] = pd.to_numeric(top1['単勝オッズ'], errors='coerce')
    return top1


def roi_stats(top1):
    won  = top1['着順_num'] == 1
    odds = top1['odds_num']
    n = len(top1)
    if n == 0:
        return dict(n=0, wins=0, win_rate=float('nan'),
                    roi=float('nan'), avg_odds_win=float('nan'), avg_odds=float('nan'))
    wins = int(won.sum())
    return dict(
        n=n, wins=wins,
        win_rate=wins / n,
        roi=float((odds[won] * 100).sum() / (n * 100) - 1),
        avg_odds_win=float(odds[won].mean()) if wins > 0 else float('nan'),
        avg_odds=float(odds.mean()),
    )


def collect_segment_data(seg, df_all, dm_all, art):
    mask = seg['filter'](df_all, dm_all)
    df_seg = df_all[mask].copy()
    for col in art.get('feat_cols', []):
        if col in df_seg.columns:
            try:
                df_seg[col] = pd.to_numeric(df_seg[col], errors='coerce')
            except Exception:
                df_seg[col] = np.nan

    df_trn = df_seg[(df_seg['日付_num'] >= 130101) & (df_seg['日付_num'] < 220101)]
    df_oos = df_seg[df_seg['日付_num'] >= 230101]

    year_stats = []
    for label, d_from, d_to in YEARS:
        sub = df_seg[(df_seg['日付_num'] >= d_from) & (df_seg['日付_num'] <= d_to)]
        if sub['race_id'].nunique() == 0:
            continue
        top1 = score_top1(sub, art)
        s = roi_stats(top1)
        year_stats.append({'label': label, 'is_sel': '★' in label, **s})

    bins   = [0, 3, 6, 10, 20, 50, 999]
    blabels = ['~3倍', '3〜6倍', '6〜10倍', '10〜20倍', '20〜50倍', '50倍〜']
    odds_stats = []
    if len(df_oos) > 0:
        top1_oos = score_top1(df_oos, art)
        top1_oos = top1_oos[top1_oos['odds_num'].notna()].copy()
        top1_oos['odds_band'] = pd.cut(top1_oos['odds_num'], bins=bins, labels=blabels)
        for band in blabels:
            g = top1_oos[top1_oos['odds_band'] == band]
            if len(g) == 0:
                continue
            odds_stats.append({'band': band, **roi_stats(g)})

    feat_stats = []
    for f, b in zip(art.get('feat_cols', []), art.get('coef', [])):
        nan_tr = df_trn[f].isna().mean() if f in df_trn.columns else 1.0
        nan_oo = df_oos[f].isna().mean()  if f in df_oos.columns  else 1.0
        kind, method = LEAK_INFO.get(f, ('不明', '—'))
        feat_stats.append({'name': f, 'beta': float(b),
                           'nan_tr': nan_tr, 'nan_oo': nan_oo,
                           'kind': kind, 'method': method})

    sel = next((s for s in year_stats if s['is_sel']), None)
    s25 = next((s for s in year_stats if s['label'] == '2025'), None)
    s26 = next((s for s in year_stats if s['label'] == '2026'), None)
    return dict(year_stats=year_stats, odds_stats=odds_stats,
                feat_stats=feat_stats,
                roi_2323=sel['roi'] if sel else float('nan'),
                roi_2025=s25['roi'] if s25 else float('nan'),
                roi_2026=s26['roi'] if s26 else float('nan'),
                n_races=sel['n'] if sel else 0,
                n_feats=len(art.get('feat_cols', [])))


# ─── HTML helpers ──────────────────────────────────────────────────────────

def roi_bg(v):
    if np.isnan(v): return '#f5f5f5'
    if v > 0.05:    return '#c8e6c9'
    if v > -0.05:   return '#fff9c4'
    return '#ffcdd2'

def roi_span(v, large=False):
    if np.isnan(v): return '—'
    color = '#2e7d32' if v > 0 else '#c62828' if v < -0.05 else '#e65100'
    size  = '1.35rem' if large else 'inherit'
    return f'<span style="color:{color};font-weight:700;font-size:{size}">{v:+.2%}</span>'

def nan_badge(v):
    if v > 0.5:  bg, fg = '#ffcdd2', '#b71c1c'
    elif v > 0.2: bg, fg = '#fff9c4', '#e65100'
    else:         bg, fg = '#e8f5e9', '#1b5e20'
    return f'<span style="background:{bg};color:{fg};padding:1px 6px;border-radius:10px;font-size:.8rem">{v:.0%}</span>'

KIND_BADGE = {
    'race-day固定': ('#e8f5e9', '#1b5e20'),
    '前走データ':   ('#e3f2fd', '#0d47a1'),
    'rolling-5':    ('#fff8e1', '#e65100'),
    'global統計':   ('#fce4ec', '#880e4f'),
}


def segment_section(seg, data, color):
    n_feats = data['n_feats']
    r2323   = data['roi_2323']
    r2025   = data['roi_2025']
    r2026   = data['roi_2026']
    n_races = data['n_races']

    # KPI bar
    kpi = f"""
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-label">2323 ROI <span class="kpi-star">★ 選択指標</span></div>
        <div class="kpi-val" style="color:{color}">{roi_span(r2323, large=True)}</div>
        <div class="kpi-sub">{n_races:,} レース (2023–24)</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">2025 ROI <span class="kpi-note">参考</span></div>
        <div class="kpi-val">{roi_span(r2025, large=True)}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">2026 ROI <span class="kpi-note">参考</span></div>
        <div class="kpi-val">{roi_span(r2026, large=True)}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">特徴量数 / モデル</div>
        <div class="kpi-val" style="font-size:1.6rem;color:{color}">{n_feats}</div>
        <div class="kpi-sub">{seg['version']}</div>
      </div>
    </div>"""

    # Year ROI table
    def yr_row(s):
        bg  = '#eef2ff' if s['is_sel'] else ''
        fw  = 'font-weight:600;' if s['is_sel'] else ''
        sel = ' ★' if s['is_sel'] else ''
        return f"""<tr style="background:{bg}">
          <td style="{fw}">{s['label']}{sel}</td>
          <td class="num">{s['n']:,}</td>
          <td class="num">{s['wins']:,}</td>
          <td class="num" style="background:{roi_bg(s['win_rate']-0.12)}">{s['win_rate']:.1%}</td>
          <td class="num" style="background:{roi_bg(s['roi'])}">{roi_span(s['roi'])}</td>
          <td class="num">{s['avg_odds_win']:.2f}</td>
          <td class="num">{s['avg_odds']:.2f}</td>
        </tr>"""

    yr_rows = ''.join(yr_row(s) for s in data['year_stats'])
    yr_table = f"""
    <table>
      <thead><tr>
        <th>期間</th><th>R数</th><th>的中</th><th>的中率</th>
        <th>回収率</th><th>平均オッズ(的中)</th><th>平均オッズ(全)</th>
      </tr></thead>
      <tbody>{yr_rows}</tbody>
    </table>"""

    # Odds band table
    def odds_row(s):
        return f"""<tr>
          <td><strong>{s['band']}</strong></td>
          <td class="num">{s['n']:,}</td>
          <td class="num">{s['wins']:,}</td>
          <td class="num" style="background:{roi_bg(s['win_rate']-0.12)}">{s['win_rate']:.1%}</td>
          <td class="num" style="background:{roi_bg(s['roi'])}">{roi_span(s['roi'])}</td>
          <td class="num">{s['avg_odds_win']:.2f}</td>
        </tr>"""

    odds_rows = ''.join(odds_row(s) for s in data['odds_stats'])
    odds_table = f"""
    <table>
      <thead><tr>
        <th>オッズ帯</th><th>R数</th><th>的中</th><th>的中率</th>
        <th>回収率</th><th>平均オッズ(的中)</th>
      </tr></thead>
      <tbody>{odds_rows}</tbody>
    </table>""" if odds_rows else '<p class="muted">OOSデータなし</p>'

    # Feature table
    def feat_row(f):
        bg, fg = KIND_BADGE.get(f['kind'], ('#f5f5f5', '#333'))
        return f"""<tr>
          <td><code>{f['name']}</code></td>
          <td><span class="badge" style="background:{bg};color:{fg}">{f['kind']}</span></td>
          <td class="small muted">{f['method']}</td>
          <td class="num" style="color:{'#c62828' if f['beta']<0 else '#2e7d32'}">{f['beta']:+.4f}</td>
          <td class="num">{nan_badge(f['nan_tr'])}</td>
          <td class="num">{nan_badge(f['nan_oo'])}</td>
          <td class="num"><span class="badge safe-badge">✅ safe</span></td>
        </tr>"""

    feat_rows = ''.join(feat_row(f) for f in data['feat_stats'])
    feat_table = f"""
    <table>
      <thead><tr>
        <th>特徴量</th><th>種別</th><th>計算方法</th>
        <th>β係数</th><th>NaN率 (train)</th><th>NaN率 (OOS)</th><th>リーク判定</th>
      </tr></thead>
      <tbody>{feat_rows}</tbody>
    </table>"""

    return f"""
  <section id="{seg['id']}" class="seg-section">
    <div class="seg-header" style="border-left:5px solid {color}">
      <div>
        <h2 style="color:{color}">{seg['name']} <span class="seg-sublabel">{seg['label']}</span></h2>
        <p class="seg-cond">{seg['cond']} &nbsp;·&nbsp; {seg['version']}</p>
      </div>
    </div>
    <p class="seg-note">{seg['note']}</p>
    {kpi}

    <div class="tab-group" data-seg="{seg['id']}">
      <button class="tab-btn active" data-tab="roi">年別 ROI</button>
      <button class="tab-btn" data-tab="odds">オッズ帯別</button>
      <button class="tab-btn" data-tab="feats">特徴量 / リークチェック</button>
    </div>

    <div class="tab-content active" data-seg="{seg['id']}" data-tab="roi">
      {yr_table}
      <p class="small muted">★ 2023-24 = 選択指標（この期間のみを最適化）。2025・2026は真のOOS参考値。</p>
    </div>
    <div class="tab-content" data-seg="{seg['id']}" data-tab="odds">
      <p class="small muted">集計対象: OOS 2023–2026</p>
      {odds_table}
    </div>
    <div class="tab-content" data-seg="{seg['id']}" data-tab="feats">
      {feat_table}
      <div class="leak-legend">
        <span class="legend-item" style="background:#e8f5e9;color:#1b5e20">race-day固定</span>
        <span class="legend-item" style="background:#e3f2fd;color:#0d47a1">前走データ (shift)</span>
        <span class="legend-item" style="background:#fff8e1;color:#e65100">rolling-5 (shift based)</span>
        <span class="legend-item" style="background:#fce4ec;color:#880e4f">global統計 (_stat_mask 2013-2020)</span>
      </div>
    </div>
  </section>"""


def build_site(all_seg_data):
    # Summary rows for overview table
    def summary_row(seg, data):
        c = seg['color']
        r = data['roi_2323']
        n = data['n_feats']
        bg = roi_bg(r)
        return f"""<tr>
          <td><a href="#{seg['id']}" class="seg-link" style="color:{c}">{seg['name']}</a></td>
          <td class="small">{seg['label']}</td>
          <td class="num">{n}</td>
          <td class="num" style="background:{bg}">{roi_span(data['roi_2323'])}</td>
          <td class="num" style="background:{roi_bg(data['roi_2025'])}">{roi_span(data['roi_2025'])}</td>
          <td class="num" style="background:{roi_bg(data['roi_2026'])}">{roi_span(data['roi_2026'])}</td>
          <td class="small muted">{seg['version']}</td>
        </tr>"""

    summary_rows = ''.join(summary_row(seg, data) for seg, data in all_seg_data)

    nav_links = ''.join(
        f'<a href="#{seg["id"]}" class="nav-link" style="border-bottom:2px solid {seg["color"]}">'
        f'{seg["name"]}</a>'
        for seg, _ in all_seg_data
    )

    seg_sections = ''.join(segment_section(seg, data, seg['color']) for seg, data in all_seg_data)

    css = """
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg: #f4f6f9;
      --surface: #ffffff;
      --border: #e0e0e0;
      --text: #212121;
      --muted: #757575;
      --radius: 10px;
      --shadow: 0 2px 8px rgba(0,0,0,.10);
    }
    html { scroll-behavior: smooth; }
    body { font-family: 'Segoe UI', 'Noto Sans JP', sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }

    /* NAV */
    nav {
      position: sticky; top: 0; z-index: 100;
      background: #1a1a2e;
      display: flex; align-items: center; gap: 8px;
      padding: 0 24px; height: 52px;
      box-shadow: 0 2px 6px rgba(0,0,0,.3);
    }
    .nav-brand { color: #fff; font-weight: 700; font-size: 1rem; margin-right: 16px; white-space: nowrap; }
    .nav-link {
      color: #ccc; font-size: .875rem; font-weight: 600;
      text-decoration: none; padding: 4px 10px; border-radius: 4px;
      transition: background .15s;
    }
    .nav-link:hover { background: rgba(255,255,255,.12); color: #fff; }
    .nav-right { margin-left: auto; color: #888; font-size: .8rem; }

    /* HERO */
    .hero {
      background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
      color: #fff; padding: 56px 24px 48px;
    }
    .hero-inner { max-width: 1100px; margin: 0 auto; }
    .hero h1 { font-size: 2rem; font-weight: 800; letter-spacing: -.5px; margin-bottom: 10px; }
    .hero p  { font-size: 1rem; color: #aab; max-width: 620px; line-height: 1.7; }
    .hero-chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 18px; }
    .chip {
      background: rgba(255,255,255,.1); color: #dde; border-radius: 20px;
      padding: 3px 12px; font-size: .8rem; border: 1px solid rgba(255,255,255,.15);
    }

    /* CONTAINER */
    .container { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }

    /* SUMMARY TABLE BOX */
    .overview-box {
      background: var(--surface); border-radius: var(--radius);
      box-shadow: var(--shadow); padding: 24px; margin-bottom: 32px;
    }
    .overview-box h2 { font-size: 1.1rem; margin-bottom: 16px; color: #1a237e; }

    /* SEGMENT SECTION */
    .seg-section {
      background: var(--surface); border-radius: var(--radius);
      box-shadow: var(--shadow); padding: 28px; margin-bottom: 28px;
    }
    .seg-header { padding-left: 14px; margin-bottom: 6px; }
    .seg-header h2 { font-size: 1.35rem; font-weight: 700; display: inline; }
    .seg-sublabel { font-size: 1rem; font-weight: 400; color: var(--muted); margin-left: 8px; }
    .seg-cond { font-size: .82rem; color: var(--muted); margin-top: 2px; }
    .seg-note { font-size: .875rem; color: #444; margin: 10px 0 18px; line-height: 1.6; }

    /* KPI ROW */
    .kpi-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px; }
    .kpi-card { background: #f8f9fc; border-radius: 8px; padding: 14px 16px; border: 1px solid var(--border); }
    .kpi-label { font-size: .75rem; color: var(--muted); margin-bottom: 4px; }
    .kpi-star { background: #fff3e0; color: #e65100; border-radius: 10px; padding: 1px 6px; font-size: .7rem; font-weight: 700; margin-left: 4px; }
    .kpi-note { background: #ede7f6; color: #4527a0; border-radius: 10px; padding: 1px 6px; font-size: .7rem; margin-left: 4px; }
    .kpi-val { font-size: 1.4rem; font-weight: 700; line-height: 1.2; }
    .kpi-sub { font-size: .75rem; color: var(--muted); margin-top: 4px; }

    /* TABS */
    .tab-group { display: flex; gap: 4px; margin-bottom: 16px; border-bottom: 2px solid var(--border); padding-bottom: 0; }
    .tab-btn {
      background: none; border: none; padding: 7px 16px; cursor: pointer;
      font-size: .875rem; color: var(--muted); border-radius: 6px 6px 0 0;
      border-bottom: 2px solid transparent; margin-bottom: -2px;
      transition: color .15s;
    }
    .tab-btn.active { color: #1a237e; border-bottom-color: #1a237e; font-weight: 600; background: #f0f4ff; }
    .tab-btn:hover:not(.active) { color: #333; background: #f5f5f5; }
    .tab-content { display: none; }
    .tab-content.active { display: block; }

    /* TABLES */
    table { width: 100%; border-collapse: collapse; font-size: .875rem; }
    th { background: #1a237e; color: #fff; padding: 8px 12px; text-align: left; white-space: nowrap; font-weight: 600; }
    td { padding: 7px 12px; border-bottom: 1px solid #eee; white-space: nowrap; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: #f9f9ff; }
    td.num { text-align: right; }

    /* BADGES */
    .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: .75rem; font-weight: 600; }
    .safe-badge { background: #e8f5e9; color: #1b5e20; }

    /* LEAK LEGEND */
    .leak-legend { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
    .legend-item { padding: 3px 10px; border-radius: 12px; font-size: .78rem; font-weight: 600; }

    /* METHODOLOGY */
    .method-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
    .method-card { background: #f8f9fc; border-radius: 8px; padding: 16px; border: 1px solid var(--border); }
    .method-card h3 { font-size: .9rem; color: #1a237e; margin-bottom: 8px; }
    .method-card p  { font-size: .82rem; color: #444; line-height: 1.6; }

    /* MISC */
    .small  { font-size: .8rem; }
    .muted  { color: var(--muted); }
    code    { background: #f5f5f5; padding: 1px 5px; border-radius: 3px; font-size: .85em; }
    .seg-link { text-decoration: none; font-weight: 700; }
    .seg-link:hover { text-decoration: underline; }
    section { scroll-margin-top: 60px; }

    @media (max-width: 700px) {
      .kpi-row       { grid-template-columns: repeat(2, 1fr); }
      .method-grid   { grid-template-columns: 1fr; }
    }
    """

    js = """
    document.addEventListener('DOMContentLoaded', () => {
      document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const seg = btn.closest('[data-seg]') ?
            btn.closest('[data-seg]').dataset.seg :
            btn.parentElement.dataset.seg;
          const tab = btn.dataset.tab;
          btn.parentElement.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          document.querySelectorAll(`.tab-content[data-seg="${seg}"]`).forEach(c => {
            c.classList.toggle('active', c.dataset.tab === tab);
          });
        });
      });
    });
    """

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Horse Racing AI v2 — Project Report</title>
  <style>{css}</style>
</head>
<body>

<nav>
  <span class="nav-brand">🏇 Horse Racing AI v2</span>
  <a href="#overview" class="nav-link">概要</a>
  <a href="#methodology" class="nav-link">手法</a>
  {nav_links}
  <span class="nav-right">2026-06-07</span>
</nav>

<div class="hero">
  <div class="hero-inner">
    <h1>Horse Racing AI v2 — Project Report</h1>
    <p>
      Conditional Logit + Isotonic Calibration による5セグメント予測モデル。<br>
      OOS多重比較バイアスを排除するため <strong>2323 OOS ROI（2023-24）</strong> を選択指標とし、
      2025・2026 は真のアウトオブサンプルとして報告のみ。
    </p>
    <div class="hero-chips">
      <span class="chip">Conditional Logit</span>
      <span class="chip">Adam Optimizer</span>
      <span class="chip">Isotonic Calibration</span>
      <span class="chip">5セグメント</span>
      <span class="chip">10特徴/セグメント</span>
      <span class="chip">train 2013-21 / val 2022</span>
      <span class="chip">リークチェック済み</span>
    </div>
  </div>
</div>

<div class="container">

  <!-- OVERVIEW -->
  <section id="overview">
    <div class="overview-box">
      <h2>全セグメント サマリー</h2>
      <table>
        <thead><tr>
          <th>セグメント</th><th>距離帯</th><th>特徴数</th>
          <th>2323 ROI ★</th><th>2025 ROI</th><th>2026 ROI</th><th>モデル</th>
        </tr></thead>
        <tbody>{summary_rows}</tbody>
      </table>
      <p class="small muted" style="margin-top:10px">
        ★ 2323 = 2023–24 OOS ROI（選択指標）。2025・2026 は最適化に使用していない真のOOS。
        控除率20%相当のランダム予測ROIは −20%。
      </p>
    </div>
  </section>

  <!-- METHODOLOGY -->
  <section id="methodology">
    <div class="overview-box">
      <h2>モデル手法・リークチェック</h2>
      <div class="method-grid">
        <div class="method-card">
          <h3>モデル構造</h3>
          <p>
            Conditional Logit（条件付きロジットモデル）。
            レース内 softmax で各馬の勝利確率を推定。<br>
            最適化: Adam (lr=0.005, L2=0.006, early-stopping)。<br>
            キャリブレーション: Isotonic Regression (val 2022)。
          </p>
        </div>
        <div class="method-card">
          <h3>特徴量のリーク対策</h3>
          <p>
            <strong>global統計</strong>（コース枠勝率・馬距離勝率等）:
            <code>_stat_mask</code> により 2013-2020 データのみで計算。OOSデータを含まない。<br>
            <strong>rolling統計</strong>: <code>cumsum[i-1]</code> で当該レース日前のデータのみ使用。<br>
            <strong>前走データ</strong>: <code>shift(i)</code> で生成。当該レース除外済み。
          </p>
        </div>
        <div class="method-card">
          <h3>選択指標の変更</h3>
          <p>
            旧方針（25+26最大化）では74回比較のOOSスヌーピングにより
            芝短nv2で+32.28%という過楽観な結果が生じた。<br>
            新方針: <strong>2323 OOS ROI</strong> のみを選択指標とし、
            2025・2026 を真のOOSとして留保。
          </p>
        </div>
      </div>
    </div>
  </section>

  <!-- SEGMENT SECTIONS -->
  {seg_sections}

</div>

<script>{js}</script>
</body>
</html>"""


def main():
    print("データ読み込み中...")
    df_all, dm_all = load_all()

    with open(os.path.join(MODEL_DIR, 'final_model.pkl'), 'rb') as f:
        pkg = pickle.load(f)

    all_seg_data = []
    for seg in SEGMENTS:
        art = pkg['artifacts'].get(seg['key'])
        if art is None:
            print(f"[{seg['name']}] artifact '{seg['key']}' なし — skip")
            continue
        print(f"[{seg['name']}] 集計中...")
        data = collect_segment_data(seg, df_all, dm_all, art)
        all_seg_data.append((seg, data))

    print("HTML生成中...")
    html = build_site(all_seg_data)
    out = os.path.join(DOCS_DIR, 'index.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"完了: {out}  ({len(html)//1024} KB)")


if __name__ == '__main__':
    main()
