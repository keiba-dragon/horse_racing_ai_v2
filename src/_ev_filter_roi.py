# coding: utf-8
"""newspaper の★買い推奨レースの◎馬でROIを計算"""
import os, sys, re
import pandas as pd

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res_dir = os.path.join(base, 'data', 'raw', 'results')


def parse_buy_recommendations(date_str):
    """
    newspaper HTML から「★買い推奨」(badge-buy) のレースの◎馬を抽出する。
    """
    html_path = os.path.join(base, 'docs', f'newspaper_{date_str}.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        html = f.read()

    # レースブロックに分割（race-card ごと）
    race_blocks = re.split(r'<div class="race-card">', html)

    picks = []
    for block in race_blocks[1:]:  # 最初は前置きHTML
        # badge-buy があるブロックだけ対象
        if 'badge-buy' not in block:
            continue

        # レース番号
        r_match = re.search(r'class="race-label"[^>]*>.*?(\d+)R', block)
        race_no = int(r_match.group(1)) if r_match else None

        # ◎ 行を探す（row-buy または mark ◎ の行）
        # mark ◎ の馬名・オッズ・EV を抽出
        top_match = re.search(
            r'class="mark"[^>]*>◎</td>\s*'
            r'<td class="umaban">(\d+)</td>\s*'
            r'<td[^>]*>([^<]+)</td>\s*'   # 馬名
            r'<td[^>]*>([^<]*)</td>\s*'   # 人気
            r'<td[^>]*>([^<]*)</td>\s*'   # オッズ
            r'<td[^>]*>([^<]*)</td>\s*'   # calib
            r'<td[^>]*>(?:<span[^>]*>)?(EV[+\-][\d.]+)?',
            block, re.DOTALL
        )

        if top_match:
            horse_no   = int(top_match.group(1))
            horse_name = top_match.group(2).strip()
            pop        = top_match.group(3).strip()
            odds_str   = top_match.group(4).strip()
            calib_str  = top_match.group(5).strip()
            ev_str     = top_match.group(6) or ''

            try:
                odds = float(odds_str) if odds_str and odds_str != '-' else None
            except ValueError:
                odds = None
            try:
                ev_val = float(ev_str.replace('EV','')) if ev_str else None
            except ValueError:
                ev_val = None

            picks.append({
                'date': date_str,
                'race_no': race_no,
                'horse_no': horse_no,
                'horse_name': horse_name,
                'odds_pred': odds,
                'ev': ev_val,
            })

    return picks


# ── データ収集 ─────────────────────────────────────────────────────
all_picks = []
for ds in ['20260523', '20260524']:
    picks = parse_buy_recommendations(ds)
    print(f'{ds}: 買い推奨 {len(picks)} レース')
    for p in picks:
        print(f"  R{p['race_no']:02d}: {p['horse_name']}  EV={p['ev']:+.2f}  予測odds={p['odds_pred']}")
    all_picks.extend(picks)

# 実績CSV
r23 = pd.read_csv(os.path.join(res_dir, '20260523.csv'), encoding='utf-8')
r24 = pd.read_csv(os.path.join(res_dir, '20260524.csv'), encoding='utf-8')
res = pd.concat([r23, r24], ignore_index=True)

# 実績照合
print()
print('=' * 58)
print('★買い推奨の◎馬 実績')
print('=' * 58)

hits = 0
total_return = 0.0
details = []

for p in all_picks:
    date_int = int(p['date'])
    row = res[(res['日付'] == date_int) & (res['馬名'] == p['horse_name'])]
    actual_rank = int(row['着順'].values[0]) if len(row) > 0 else None
    actual_odds = float(row['単勝オッズ'].values[0]) if len(row) > 0 else None
    won = (actual_rank == 1)
    if won:
        hits += 1
        total_return += (actual_odds or 0) * 100
    details.append({**p, 'actual_rank': actual_rank, 'actual_odds': actual_odds, 'won': won})

n = len(details)
roi = total_return / (n * 100) - 1 if n > 0 else 0

for d in details:
    mark = '○' if d['won'] else '×'
    name = d['horse_name'][:10]
    ev_str = f"{d['ev']:+.2f}" if d['ev'] is not None else '  N/A'
    print(f"  {mark} {d['date']} R{d['race_no']:02d}  {name:<10}  "
          f"EV={ev_str}  予測{d['odds_pred']:.1f}倍  "
          f"実際{d['actual_rank']}着 {d['actual_odds']:.1f}倍")

print()
print(f'対象レース: {n}')
print(f'的中数    : {hits}')
print(f'勝率      : {hits/n:.3f}' if n > 0 else '0')
print(f'ROI       : {roi:+.3f}  ({roi*100:+.1f}%)')
