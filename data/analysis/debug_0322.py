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
        break
    except:
        pass

# 全行の購入金額を解析
df['購入金額_raw'] = df['購入金額'].astype(str)
df['購入金額_parsed'] = df['購入金額'].apply(parse_amount)

# NaN行（合計行）を確認
nan_rows = df[df['日付'].isna()]
data_rows = df[df['日付'].notna()]

print(f'データ行: {len(data_rows)}行, 合計行: {len(nan_rows)}行')
print(f'CSV合計行の購入: {nan_rows["購入金額"].values}')
print(f'パース済み合計: {data_rows["購入金額_parsed"].sum():,.0f}円')
print(f'\n--- parse=0になった行 ---')
zero_rows = data_rows[data_rows['購入金額_parsed'] == 0]
print(zero_rows[['式別','馬／組番','購入金額_raw','払戻金額']].to_string())

print(f'\n--- 購入金額_rawにスラッシュ以外の区切りがある行 ---')
for i, row in data_rows.iterrows():
    raw = str(row['購入金額']).strip()
    if any(c in raw for c in ['^', '・', '→', '＾']):
        print(f"  行{i}: 式別={row['式別']}, 馬番={row['馬／組番']}, 購入={raw}, parsed={parse_amount(raw)}")
