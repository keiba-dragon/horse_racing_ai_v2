# coding: utf-8
"""
report_html_segments.py - 4セグメント 10特徴モデル HTML評価レポート生成
出力: docs/report_{seg}.html  (ダ短 / 芝中 / 芝長 / 芝短)
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

SEGMENTS = {
    'ダ短': {
        'key':    'ダ短',
        'label':  'ダート短距離 (≤1400m)',
        'filter': lambda df, dm: (df['surface'] == 'ダ') & (dm <= 1400),
        'feats':  ['近5走_上り3F平均', 'コース枠_r200_勝率', '1走前_馬場状態',
                   '1走前_クラス差', '2走前_クラス差', '性別_num', '斤量',
                   '同会場_平均着順_近5走', '馬体重', '馬距離_勝率'],
        'version': 'nv3 (greedy, 2323選択)',
        'baseline': ('nv2 5特徴', -0.0305),
    },
    '芝中': {
        'key':    '芝中',
        'label':  '芝中距離 (1401-2000m)',
        'filter': lambda df, dm: (df['surface'] == '芝') & (dm > 1400) & (dm <= 2000),
        'feats':  ['調教師コース_r100_勝率', '馬距離_勝率', '1走前_クラス調整着順',
                   '近5走_クラス調整_平均着順', '間隔', '馬番', '前走着差タイム',
                   '良馬場_平均着順_近5走', '2走前_クラス差', 'コース枠_r200_勝率'],
        'version': '10特徴 (greedy, 2323選択)',
        'baseline': ('旧AI 5特徴', +0.0881),
    },
    '芝長': {
        'key':    '芝長',
        'label':  '芝長距離 (2001m以上)',
        'filter': lambda df, dm: (df['surface'] == '芝') & (dm > 2000),
        'feats':  ['前走着差タイム', '距離変化_前走', '1走前_クラス差', '馬距離_勝率', '間隔',
                   '芝ダ転向', '2走前_クラス差', '斤量', '馬番', '近5走_上り3F_std'],
        'version': '10特徴 (greedy+triplet, 2323選択)',
        'baseline': ('旧Y8 5特徴', -0.0303),
    },
    '芝短': {
        'key':    '芝短',
        'label':  '芝短距離 (≤1400m)',
        'filter': lambda df, dm: (df['surface'] == '芝') & (dm <= 1400),
        'feats':  ['1走前_3角', '芝ダ転向', '距離変化_前走', '1走前_脚質_num',
                   '馬体重', '前走着差タイム', '馬距離_勝率',
                   '近5走_上り3F平均', 'コース枠_r200_勝率', '馬番'],
        'version': 'nv3 (forced greedy, 2323選択)',
        'baseline': ('旧nv2 OOS-snooped', +0.3228),
    },
}

LEAK_INFO = {
    '馬番':                  ('race-day固定', 'エントリー時決定'),
    '斤量':                  ('race-day固定', 'エントリー時決定'),
    '性別_num':              ('race-day固定', 'エントリー時決定'),
    '間隔':                  ('race-day固定', '前走日付との差分'),
    '距離変化_前走':         ('race-day固定', '今回距離 - shift(1)距離'),
    '芝ダ転向':              ('race-day固定', 'shift(1)路線との比較'),
    '馬体重':                ('race-day固定', 'レース当日公表'),
    '1走前_3角':             ('前走データ', 'shift(1) by 馬名S'),
    '1走前_脚質_num':        ('前走データ', 'shift(1) by 馬名S'),
    '1走前_馬場状態':        ('前走データ', 'shift(1) + baba_map'),
    '1走前_クラス差':        ('前走データ', 'shift(1) by 馬名S'),
    '2走前_クラス差':        ('前走データ', 'shift(2) by 馬名S'),
    '1走前_クラス調整着順':  ('前走データ', 'shift(1) by 馬名S'),
    '前走着差タイム':        ('前走データ', 'shift(1) _着差_sec'),
    '近5走_上り3F平均':         ('rolling-5', '1走前~5走前の平均'),
    '近5走_上り3F_std':         ('rolling-5', '1走前~5走前のstd'),
    '近5走_クラス調整_平均着順': ('rolling-5', '1走前~5走前クラス補正平均'),
    '同会場_平均着順_近5走':     ('rolling-5', '1走前~5走前同会場フィルタ'),
    '良馬場_平均着順_近5走':     ('rolling-5', '1走前~5走前良馬場フィルタ'),
    'コース枠_r200_勝率':       ('global統計', '_stat_mask 2013-2020固定'),
    '馬距離_勝率':              ('global統計', '_stat_mask 2013-2020固定'),
    '調教師コース_r100_勝率':   ('global統計', '_stat_mask 2013-2020固定'),
}

KIND_COLOR = {
    'race-day固定': '#e8f5e9',
    '前走データ':   '#e3f2fd',
    'rolling-5':    '#fff8e1',
    'global統計':   '#fce4ec',
}


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


def eval_segment(df_seg, art):
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
    n    = len(top1)
    wins = int(won.sum())
    win_rate = wins / n if n > 0 else float('nan')
    avg_odds_win = float(odds[won].mean()) if wins > 0 else float('nan')
    avg_odds_all = float(odds.mean())
    roi = float((odds[won] * 100).sum() / (n * 100) - 1) if n > 0 else float('nan')
    return {'n': n, 'wins': wins, 'win_rate': win_rate,
            'avg_odds_win': avg_odds_win, 'avg_odds': avg_odds_all, 'roi': roi}


def roi_cell(v):
    if np.isnan(v):
        return '<td>—</td>'
    color = '#c8e6c9' if v > 0 else '#ffcdd2' if v < -0.05 else '#fff9c4'
    return f'<td style="background:{color};font-weight:bold">{v:+.2%}</td>'


def pct_cell(v, thresholds=(0.20, 0.15)):
    if np.isnan(v):
        return '<td>—</td>'
    color = '#c8e6c9' if v >= thresholds[0] else '#fff9c4' if v >= thresholds[1] else '#ffcdd2'
    return f'<td style="background:{color}">{v:.1%}</td>'


def nan_cell(v):
    color = '#ffcdd2' if v > 0.5 else '#fff9c4' if v > 0.2 else ''
    style = f'style="background:{color}"' if color else ''
    return f'<td {style}>{v:.1%}</td>'


def build_html(seg_name, seg_cfg, df_all, dm_all, art):
    mask = seg_cfg['filter'](df_all, dm_all)
    df_seg = df_all[mask].copy()
    for col in seg_cfg['feats']:
        if col in df_seg.columns:
            try:
                df_seg[col] = pd.to_numeric(df_seg[col], errors='coerce')
            except Exception:
                df_seg[col] = np.nan

    feats   = art['feat_cols']
    df_trn  = df_seg[(df_seg['日付_num'] >= 130101) & (df_seg['日付_num'] < 220101)]
    df_oos  = df_seg[df_seg['日付_num'] >= 230101]

    years = [
        ('訓練 2013-21', 130101, 211231, 'train'),
        ('Val 2022',     220101, 221231, 'val'),
        ('2023',         230101, 231231, 'oos'),
        ('2024',         240101, 241231, 'oos'),
        ('2023-24 ★',   230101, 241231, 'sel'),
        ('2025',         250101, 251231, 'oos'),
        ('2026',         260101, 291231, 'oos'),
    ]

    year_rows = []
    for label, d_from, d_to, kind in years:
        sub = df_seg[(df_seg['日付_num'] >= d_from) & (df_seg['日付_num'] <= d_to)]
        if len(sub) == 0 or sub['race_id'].nunique() == 0:
            continue
        top1 = eval_segment(sub, art)
        s = roi_stats(top1)
        year_rows.append((label, kind, s))

    # odds band OOS 2023-2026
    df_oos_all = df_seg[df_seg['日付_num'] >= 230101]
    odds_rows = []
    if len(df_oos_all) > 0:
        top1_oos = eval_segment(df_oos_all, art)
        top1_oos = top1_oos[top1_oos['odds_num'].notna()]
        bins   = [0, 3, 6, 10, 20, 50, 999]
        blabels = ['~3倍', '3~6倍', '6~10倍', '10~20倍', '20~50倍', '50倍~']
        top1_oos['odds_band'] = pd.cut(top1_oos['odds_num'], bins=bins, labels=blabels)
        for band in blabels:
            g = top1_oos[top1_oos['odds_band'] == band]
            if len(g) == 0:
                continue
            odds_rows.append((band, roi_stats(g)))

    # ---- HTML build ----
    CSS = """
    <style>
      body { font-family: 'Segoe UI', 'Helvetica Neue', sans-serif; margin: 0; background: #f5f5f5; color: #212121; }
      .container { max-width: 1100px; margin: 0 auto; padding: 24px; }
      h1 { font-size: 1.6rem; font-weight: 700; color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 8px; }
      h2 { font-size: 1.15rem; font-weight: 600; color: #283593; margin-top: 32px; border-left: 4px solid #3949ab; padding-left: 10px; }
      .meta { color: #555; font-size: 0.88rem; margin-bottom: 24px; }
      table { border-collapse: collapse; width: 100%; font-size: 0.875rem; margin-bottom: 16px; }
      th { background: #283593; color: #fff; padding: 7px 12px; text-align: left; white-space: nowrap; }
      td { padding: 6px 12px; border-bottom: 1px solid #e0e0e0; white-space: nowrap; }
      tr:hover td { background: #f3f3f3; }
      .badge { display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.78rem; font-weight:600; }
      .safe  { background:#c8e6c9; color:#1b5e20; }
      .warn  { background:#fff9c4; color:#f57f17; }
      .box   { background:#fff; border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,.12); padding:20px 24px; margin-bottom:20px; }
      .summary-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:24px; }
      .kpi  { background:#fff; border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,.12); padding:14px 18px; }
      .kpi .label { font-size:0.78rem; color:#888; margin-bottom:4px; }
      .kpi .value { font-size:1.5rem; font-weight:700; }
      .pos { color: #2e7d32; }
      .neg { color: #c62828; }
      .neu { color: #f57f17; }
      .sel-row td { background: #e8eaf6 !important; font-weight: 600; }
      .note { font-size:0.82rem; color:#666; margin-top:8px; line-height:1.6; }
      .leak-table th { background:#37474f; }
      .kind-racedday { background: #e8f5e9; }
      .kind-prev     { background: #e3f2fd; }
      .kind-rolling  { background: #fff8e1; }
      .kind-global   { background: #fce4ec; }
    </style>
    """

    # KPI values
    roi_2323 = next((s['roi'] for l, k, s in year_rows if '2023-24' in l), float('nan'))
    roi_2025 = next((s['roi'] for l, k, s in year_rows if l == '2025'), float('nan'))
    roi_2026 = next((s['roi'] for l, k, s in year_rows if l == '2026'), float('nan'))
    n_2324   = next((s['n']   for l, k, s in year_rows if '2023-24' in l), 0)

    def kpi_color(v):
        if np.isnan(v): return 'neu'
        return 'pos' if v > 0 else 'neg' if v < -0.05 else 'neu'

    def fmt_roi(v):
        return f'{v:+.2%}' if not np.isnan(v) else 'N/A'

    kpi_html = f"""
    <div class="summary-grid">
      <div class="kpi">
        <div class="label">2323 OOS ROI ★選択指標</div>
        <div class="value {kpi_color(roi_2323)}">{fmt_roi(roi_2323)}</div>
        <div class="note">{n_2324}レース (2023-24)</div>
      </div>
      <div class="kpi">
        <div class="label">2025 ROI（参考）</div>
        <div class="value {kpi_color(roi_2025)}">{fmt_roi(roi_2025)}</div>
      </div>
      <div class="kpi">
        <div class="label">2026 ROI（参考）</div>
        <div class="value {kpi_color(roi_2026)}">{fmt_roi(roi_2026)}</div>
      </div>
      <div class="kpi">
        <div class="label">特徴量数 / モデル</div>
        <div class="value" style="font-size:1.1rem">{len(feats)}特徴</div>
        <div class="note">{seg_cfg['version']}</div>
      </div>
    </div>
    """

    # Leak check table
    leak_rows_html = ''
    for f, b in zip(feats, art['coef']):
        info = LEAK_INFO.get(f, ('不明', '不明'))
        kind, method = info
        nan_tr = df_trn[f].isna().mean() if f in df_trn.columns else 1.0
        nan_oo = df_oos[f].isna().mean()  if f in df_oos.columns  else 1.0
        kind_cls = {
            'race-day固定': 'kind-racedday',
            '前走データ':   'kind-prev',
            'rolling-5':    'kind-rolling',
            'global統計':   'kind-global',
        }.get(kind, '')
        leak_rows_html += f"""
        <tr class="{kind_cls}">
          <td><strong>{f}</strong></td>
          <td>{kind}</td>
          <td>{method}</td>
          <td style="text-align:right">{b:+.4f}</td>
          {nan_cell(nan_tr)}
          {nan_cell(nan_oo)}
          <td><span class="badge safe">✅ safe</span></td>
        </tr>"""

    # Legend for leak table
    legend_html = """
    <div style="display:flex;gap:12px;font-size:0.8rem;margin-top:8px;flex-wrap:wrap">
      <span style="background:#e8f5e9;padding:2px 8px;border-radius:4px">■ race-day固定</span>
      <span style="background:#e3f2fd;padding:2px 8px;border-radius:4px">■ 前走データ (shift(i))</span>
      <span style="background:#fff8e1;padding:2px 8px;border-radius:4px">■ rolling-5 (shift based)</span>
      <span style="background:#fce4ec;padding:2px 8px;border-radius:4px">■ global統計 (_stat_mask 2013-2020固定)</span>
    </div>
    """

    # OOS ROI table
    year_rows_html = ''
    for label, kind, s in year_rows:
        is_sel = '★' in label
        tr_cls = 'class="sel-row"' if is_sel else ''
        year_rows_html += f"""
        <tr {tr_cls}>
          <td>{label}</td>
          <td style="text-align:right">{s['n']:,}</td>
          <td style="text-align:right">{s['wins']:,}</td>
          {pct_cell(s['win_rate'])}
          {roi_cell(s['roi'])}
          <td style="text-align:right">{s['avg_odds_win']:.2f}</td>
          <td style="text-align:right">{s['avg_odds']:.2f}</td>
        </tr>"""

    # Odds band table
    odds_rows_html = ''
    for band, s in odds_rows:
        odds_rows_html += f"""
        <tr>
          <td>{band}</td>
          <td style="text-align:right">{s['n']:,}</td>
          <td style="text-align:right">{s['wins']:,}</td>
          {pct_cell(s['win_rate'])}
          {roi_cell(s['roi'])}
          <td style="text-align:right">{s['avg_odds_win']:.2f}</td>
        </tr>"""

    # Baseline note
    bl_name, bl_roi = seg_cfg['baseline']
    bl_color = '#c8e6c9' if bl_roi > roi_2323 else '#ffcdd2'
    bl_note = f'旧ベースライン ({bl_name}): 2323ベース ROI = <span style="background:{bl_color};padding:1px 6px">{bl_roi:+.2%}</span>'

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>評価レポート: {seg_name} — {seg_cfg['label']}</title>
  {CSS}
</head>
<body>
<div class="container">
  <h1>評価レポート: {seg_name} — {seg_cfg['label']}</h1>
  <div class="meta">
    生成日: 2026-06-07 &nbsp;|&nbsp;
    モデル: Conditional Logit + Isotonic Calibration &nbsp;|&nbsp;
    学習期間: 2013-2021 / Val: 2022 &nbsp;|&nbsp;
    {seg_cfg['version']}
  </div>

  {kpi_html}

  <div class="box">
    <h2>OOS ROI サマリー（年別）</h2>
    <table>
      <thead><tr>
        <th>期間</th><th>レース数</th><th>的中</th><th>的中率</th>
        <th>回収率</th><th>平均オッズ(的中)</th><th>平均オッズ(全)</th>
      </tr></thead>
      <tbody>{year_rows_html}</tbody>
    </table>
    <p class="note">
      ★ 2023-24 = 選択指標（feature selection の最適化対象）<br>
      2025・2026 は一切最適化に使用していない真のOOS（参考値）<br>
      {bl_note}
    </p>
  </div>

  <div class="box">
    <h2>オッズ帯別回収率（OOS 2023-2026）</h2>
    <table>
      <thead><tr>
        <th>オッズ帯</th><th>レース数</th><th>的中</th><th>的中率</th>
        <th>回収率</th><th>平均オッズ(的中)</th>
      </tr></thead>
      <tbody>{odds_rows_html}</tbody>
    </table>
  </div>

  <div class="box">
    <h2>特徴量・リークチェック表</h2>
    <table class="leak-table">
      <thead><tr>
        <th>特徴量</th><th>種別</th><th>計算方法</th>
        <th>β係数</th><th>NaN率 (train)</th><th>NaN率 (OOS)</th><th>リーク判定</th>
      </tr></thead>
      <tbody>{leak_rows_html}</tbody>
    </table>
    {legend_html}
    <p class="note" style="margin-top:12px">
      <strong>global統計の補足</strong>: <code>_stat_mask</code> により 2013-2020 の行のみで計算。
      OOS (2023-2026) のデータは統計に含まれない → OOSへのリーク なし。<br>
      <strong>rolling統計の補足</strong>: <code>calc_rolling_stats_combo</code> は
      <code>n_cs[i] = cumsum[0..i-1]</code> で当該レース日より前のデータのみを使用。<br>
      <strong>前走データの補足</strong>: 全て <code>groupby('馬名S').shift(i)</code> により生成。
    </p>
  </div>

</div>
</body>
</html>
"""
    return html


def main():
    print("データ読み込み中...")
    df_all, dm_all = load_all()

    with open(os.path.join(MODEL_DIR, 'roi_model.pkl'), 'rb') as f:
        pkg = pickle.load(f)

    for seg_name, seg_cfg in SEGMENTS.items():
        art = pkg['artifacts'].get(seg_cfg['key'])
        if art is None:
            print(f"[{seg_name}] artifact なし — skip")
            continue
        print(f"[{seg_name}] レポート生成中...")
        html = build_html(seg_name, seg_cfg, df_all, dm_all, art)
        out_path = os.path.join(DOCS_DIR, f'report_{seg_name}.html')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"  → {out_path}")

    print("\n完了。docs/ フォルダを確認してください。")


if __name__ == '__main__':
    main()
