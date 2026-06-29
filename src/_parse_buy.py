# coding: utf-8
"""★買い推奨 × 新馬除き ROI（06_predict_from_card.py と同じクラス判定を使用）"""
import os, re
import pandas as pd
from urllib.parse import unquote

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res_dir = os.path.join(base, 'data', 'raw', 'results')


def parse_buy_picks(date_str):
    with open(os.path.join(base, 'docs', f'newspaper_{date_str}.html'), encoding='utf-8') as f:
        html = f.read()
    picks = []
    for card in re.split(r'(?=<div class="race-card">)', html):
        if 'badge-buy' not in card or '★買い' not in card:
            continue
        r_m = re.search(r'class="race-label"[^>]*>.*?(\d+)R', card, re.DOTALL)
        race_no = int(r_m.group(1)) if r_m else None
        gap_m = re.search(r'gap ([\d.]+)', card)
        gap = float(gap_m.group(1)) if gap_m else None
        row_buy = re.search(r'class="row-buy">(.*?)(?=</tr>)', card, re.DOTALL)
        if not row_buy:
            continue
        row_html = row_buy.group(1)
        name_m = re.search(r'class="horse-link"[^>]*>([^<]+)<', row_html)
        horse_name = name_m.group(1).strip() if name_m else None
        if not horse_name:
            continue
        odds_m = re.search(r'class="odds">([\d.]+)', row_html)
        odds = float(odds_m.group(1)) if odds_m else None
        ev_m = re.search(r'>(EV[+\-][\d.]+)<', row_html)
        ev_val = float(ev_m.group(1).replace('EV','')) if ev_m else None
        picks.append({'date': date_str, 'race_no': race_no, 'horse_name': horse_name,
                      'odds_pred': odds, 'ev': ev_val, 'gap': gap})
    return picks


# 出馬表CSVからクラス情報を取得
# 06_predict_from_card.py と同じクラス列('クラス')を使う
def get_class_map_from_card_csv(fname):
    path = os.path.join(base, fname)
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, encoding='utf-8', encoding_errors='replace')
    # 値に '新馬' '未勝利' を含む列（かつ '前' のような前走情報でない列）を探す
    # 7種類の値: 新馬, 未勝利, 1勝, 2勝, 3勝以上, OP, 特別 etc.
    class_col = None
    for c in df.columns:
        vals = df[c].dropna().astype(str).unique()
        has_shinba = any('新馬' in v for v in vals)
        has_mishouri = any('未勝利' in v for v in vals)
        if has_shinba and has_mishouri and len(vals) <= 10:
            class_col = c
            break
    if class_col is None:
        return {}
    # 馬名S列
    name_col = None
    for c in df.columns:
        vals = df[c].dropna().astype(str)
        # 馬名S列は文字数が2-8文字の日本語文字列が多い
        if vals.str.len().between(2, 12).mean() > 0.8 and vals.nunique() > 100:
            # もっと具体的に: 全ての馬名が英数字のみでないもの
            name_col = c
            break
    # 馬名S はインデックス固定で取得（出馬表形式は列番号が固定）
    # 新聞の馬名と一致させるため、馬名S列を探す
    # 馬名S列は通常特定の位置にある
    for c in df.columns:
        if '馬名' in str(c).encode('utf-8', errors='ignore').decode('ascii', errors='ignore'):
            name_col = c
            break
    if name_col is None:
        # フォールバック: 馬名Sっぽい列（文字列で重複が少ない）
        for c in df.columns:
            s = df[c].dropna().astype(str)
            if s.nunique() > 200 and s.str.len().between(2, 10).mean() > 0.7:
                name_col = c
                break
    if name_col and class_col:
        result = {}
        for nm, cls in zip(df[name_col].astype(str), df[class_col].astype(str)):
            result[nm] = cls
        return result
    return {}


# 出馬表CSVを直接読んで馬名→クラスのマップを作る
# 全列を確認して正しい馬名列とクラス列を特定
def get_class_map_v2(fname):
    path = os.path.join(base, fname)
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, encoding='utf-8', encoding_errors='replace')

    # 新聞の馬名（日本語）と一致する列を探す
    # 実績CSVの馬名と照合する
    r23 = pd.read_csv(os.path.join(res_dir, '20260523.csv'), encoding='utf-8')
    r24 = pd.read_csv(os.path.join(res_dir, '20260524.csv'), encoding='utf-8')
    res_names = set(pd.concat([r23, r24])['馬名'].tolist())

    name_col = None
    best_match = 0
    for c in df.columns:
        try:
            vals = set(df[c].dropna().astype(str).tolist())
            overlap = len(vals & res_names)
            if overlap > best_match:
                best_match = overlap
                name_col = c
        except:
            continue

    # クラス列: 新馬 未勝利 を含む
    class_col = None
    for c in df.columns:
        try:
            vals = df[c].dropna().astype(str).unique()
            if any('新馬' in v for v in vals) and any('未勝利' in v for v in vals):
                class_col = c
                break
        except:
            continue

    if name_col and class_col:
        print(f'  馬名列: idx={list(df.columns).index(name_col)}  クラス列: idx={list(df.columns).index(class_col)}')
        return dict(zip(df[name_col].astype(str), df[class_col].astype(str)))
    return {}


print('出馬表CSVから馬名→クラスマップを作成...')
class_map = {}
for fname in ['出馬表形式5月23日.csv', '出馬表形式5月24日.csv', '出馬表形式5月24日2.csv']:
    m = get_class_map_v2(fname)
    if m:
        class_map.update(m)
        print(f'  {fname}: {len(m)}件')

# サンプル確認
sample_keys = list(class_map.keys())[:5]
for k in sample_keys:
    print(f'  {k} -> {class_map[k]}')

# 実績CSV
r23 = pd.read_csv(os.path.join(res_dir, '20260523.csv'), encoding='utf-8')
r24 = pd.read_csv(os.path.join(res_dir, '20260524.csv'), encoding='utf-8')
res = pd.concat([r23, r24], ignore_index=True)

# 全ピック収集
all_picks = []
for ds in ['20260523', '20260524']:
    for p in parse_buy_picks(ds):
        cls = class_map.get(p['horse_name'], None)
        p['class_str'] = cls
        p['is_shinba'] = (cls is not None and '新馬' in str(cls))
        ri = res[(res['日付'] == int(ds)) & (res['馬名'] == p['horse_name'])]
        p['actual_rank'] = int(ri['着順'].values[0]) if len(ri) > 0 else None
        p['actual_odds'] = float(ri['単勝オッズ'].values[0]) if len(ri) > 0 else None
        p['won'] = (p['actual_rank'] == 1)
        all_picks.append(p)


def show_roi(picks, label):
    if not picks:
        print(f'  {label}: 0件')
        return
    n = len(picks)
    hits = sum(p['won'] for p in picks)
    ret = sum(p['actual_odds'] * 100 for p in picks if p['won'])
    roi = ret / (n * 100) - 1
    mark = '★' if roi > 0 else ''
    print(f'  {label:<22s}: {n:3d}件  win={hits/n:.3f}  ROI={roi:+.3f}  ({roi*100:+.1f}%) {mark}')


shinba = [p for p in all_picks if p['is_shinba']]
non_sb = [p for p in all_picks if not p['is_shinba']]

print()
print('=' * 58)
print('★買い推奨 新馬フィルタ結果 (5/23・5/24)')
print('=' * 58)
show_roi(all_picks, '★買い推奨 全て')
show_roi(non_sb,    '新馬除き')
show_roi(shinba,    '新馬のみ')

print()
print('--- 詳細 ---')
for p in all_picks:
    sb = '[新馬]' if p['is_shinba'] else '     '
    won = '○' if p['won'] else '×'
    cls = str(p['class_str'] or '?')[:6]
    print(f"  {won} {p['date']} R{p['race_no']:02d} {sb} {p['horse_name'][:10]:<10} "
          f"class={cls}  実{p['actual_rank']}着/{p['actual_odds']}倍")
