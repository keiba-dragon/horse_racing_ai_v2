import pandas as pd
import re

def parse_amount(val):
    s = str(val).strip()
    if s in ('', 'nan', '―', '-', '×', '\\', '¥', '＼'):
        return 0
    if '／' in s or '/' in s:
        parts = re.split(r'[／/]', s)
        for p in reversed(parts):
            p = p.strip().replace(',', '')
            if re.match(r'^\d+$', p):
                return int(p)
        return 0
    try:
        return float(str(s).replace(',', ''))
    except:
        return 0

for enc in ['cp932', 'utf-8-sig', 'utf-8']:
    try:
        df = pd.read_csv('G:/マイドライブ/horse_racing_ai/data/tohyo/20260322_tohyo.csv', encoding=enc)
        print(f'encoding: {enc}')
        break
    except:
        pass

data_rows = df[df['日付'].notna()].copy()
data_rows['parsed'] = data_rows['購入金額'].apply(parse_amount)

with open('G:/マイドライブ/horse_racing_ai/data/debug_0322.txt', 'w', encoding='utf-8') as fw:
    fw.write(f'行数: {len(data_rows)}, 合計: {data_rows["parsed"].sum():,.0f}円\n\n')
    fw.write('全行の購入金額raw vs parsed:\n')
    for i, row in data_rows.iterrows():
        raw = str(row['購入金額'])
        parsed = parse_amount(raw)
        fw.write(f'  {i:3d}: {str(row["式別"])[:12]:12s} | raw={raw:15s} | parsed={parsed}\n')
print('done')
