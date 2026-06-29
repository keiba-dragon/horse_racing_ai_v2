# coding: utf-8
"""newspaper HTMLから◎(1位予測)を抽出して実績ROIを計算"""
import sys, os
import pandas as pd

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
docs = os.path.join(base, 'docs')
res_dir = os.path.join(base, 'data', 'raw', 'results')


def parse_newspaper(date_str):
    """newspaper_YYYYMMDD.htmlから各レースの◎(1位予測)行を返す"""
    # すでにパースしたtxtを使う
    txt_path = os.path.join(base, f'_tmp_newspaper_{date_str}.txt')
    if not os.path.exists(txt_path):
        print(f'{txt_path} not found, run _check_card_csv.py first')
        return []

    picks = []
    race_idx = 0
    current_race_horses = []

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [l.rstrip('\n') for l in f.readlines()]

    for line in lines:
        parts = line.split('\t')
        if not parts:
            continue
        mark = parts[0] if parts else ''

        # ヘッダー行（印 馬番 馬名 ...）
        if mark == '印':
            # 前のレースを保存
            if current_race_horses:
                # ◎(1位)を探す
                top = [h for h in current_race_horses if h['mark'] == '◎']
                if top:
                    picks.append({**top[0], 'date': date_str, 'race_no': race_idx})
            race_idx += 1
            current_race_horses = []
            continue

        if len(parts) < 6:
            continue

        try:
            horse_no = int(parts[1]) if parts[1].strip() else None
            horse_name = parts[2].split(' ')[0] if parts[2] else ''
            odds_str = parts[4].strip() if len(parts) > 4 else ''
            odds = float(odds_str) if odds_str else None
            win_rate_str = parts[5].replace('%', '').strip() if len(parts) > 5 else ''
            win_rate = float(win_rate_str) / 100 if win_rate_str else None
        except (ValueError, IndexError):
            continue

        current_race_horses.append({
            'mark': mark,
            'horse_no': horse_no,
            'horse_name': horse_name,
            'odds_pred': odds,
            'win_rate': win_rate,
        })

    # 最後のレース
    if current_race_horses:
        top = [h for h in current_race_horses if h['mark'] == '◎']
        if top:
            picks.append({**top[0], 'date': date_str, 'race_no': race_idx})

    return picks


# 新聞のtxtが存在しない場合はHTMLから再パース
for date_str in ['20260523', '20260524']:
    txt_path = os.path.join(base, f'_tmp_newspaper_{date_str}.txt')
    if not os.path.exists(txt_path):
        # 前のスクリプトを再実行する必要あり
        print(f'Missing: {txt_path}')

# 全◎を収集
all_picks = []
for date_str in ['20260523', '20260524']:
    picks = parse_newspaper(date_str)
    print(f'\n{date_str}: {len(picks)} レースの◎')
    for p in picks[:5]:
        print(f'  R{p["race_no"]}: {p["horse_name"]} odds={p["odds_pred"]} 勝率={p["win_rate"]:.1%}')
    all_picks.extend(picks)

print(f'\n合計 {len(all_picks)} レース')

# 実績CSVで着順確認
r23 = pd.read_csv(os.path.join(res_dir, '20260523.csv'), encoding='utf-8')
r24 = pd.read_csv(os.path.join(res_dir, '20260524.csv'), encoding='utf-8')
res = pd.concat([r23, r24], ignore_index=True)

# 着順1位の馬を抽出
winners = res[res['着順'] == 1][['日付', '会場', 'レースNo', '馬名', '単勝オッズ']].copy()
winners['date'] = winners['日付'].astype(str)
print('\n勝ち馬サンプル:')
print(winners.head(5).to_string())

# マッチング: 馬名でシンプルに
hits = 0
total_return = 0.0
details = []

for p in all_picks:
    date = p['date']
    name = p['horse_name']
    # 実際の勝ち馬を探す
    won = winners[winners['date'] == date]
    won_names = won['馬名'].tolist()

    # 実際の着順（◎の馬が何着か）
    date_int = int(date)
    horse_row = res[(res['日付'] == date_int) & (res['馬名'] == name)]
    actual_rank = horse_row['着順'].values[0] if len(horse_row) > 0 else None
    actual_odds = horse_row['単勝オッズ'].values[0] if len(horse_row) > 0 else None

    won_flag = (actual_rank == 1)
    if won_flag:
        hits += 1
        total_return += (actual_odds or 0) * 100

    details.append({
        'date': date,
        'race_no': p['race_no'],
        'name': name,
        'odds_pred': p['odds_pred'],
        'win_rate': p['win_rate'],
        'actual_rank': actual_rank,
        'actual_odds': actual_odds,
        'won': won_flag,
    })

n_races = len(details)
roi = total_return / (n_races * 100) - 1

print(f'\n{"="*55}')
print(f'新聞◎予測 vs 実績 ROI')
print(f'{"="*55}')
print(f'対象レース: {n_races}')
print(f'的中数    : {hits}')
print(f'勝率      : {hits/n_races:.3f}' if n_races > 0 else '0')
print(f'ROI       : {roi:+.3f}  ({roi*100:+.1f}%)')

print(f'\n--- 的中レース ---')
for d in details:
    if d['won']:
        print(f"  ○ R{d['race_no']:02d}: {d['name']} 予測オッズ={d['odds_pred']} 実績オッズ={d['actual_odds']}")

print(f'\n--- 全◎の実績 ---')
for d in details:
    mark = '○' if d['won'] else '×'
    name = d['name'][:10] if d['name'] else '?'
    print(f"  {mark} {d['date']} R{d['race_no']:02d}: {name:<10} "
          f"着={d['actual_rank']} 予測オッズ={d['odds_pred']} 実績オッズ={d['actual_odds']} "
          f"勝率={d['win_rate']:.1%}")
