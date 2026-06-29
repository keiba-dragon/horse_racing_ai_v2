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

data = df[df['日付'].notna()].copy()
data['購入_n'] = data['購入金額'].apply(parse_amount)
data['払戻_n'] = data['払戻金額'].apply(parse_amount)
data['返還_n'] = data['返還金額'].apply(parse_amount)

with open('G:/マイドライブ/horse_racing_ai/data/races_0322.txt', 'w', encoding='utf-8') as fw:
    fw.write(f'2026/03/22 全馬券一覧（{len(data)}行）\n\n')
    fw.write(f'{"場名":4} {"R":3} {"式別":14} {"馬番":20} {"購入":>6} {"払戻":>7} {"返還":>5}\n')
    fw.write('-'*70 + '\n')
    for _, r in data.iterrows():
        fw.write(f'{str(r["場名"]):4} {str(r["レース"]):3} {str(r["式別"])[:14]:14} {str(r["馬／組番"])[:20]:20} {r["購入_n"]:>6,.0f} {r["払戻_n"]:>7,.0f} {r["返還_n"]:>5,.0f}\n')
    fw.write('-'*70 + '\n')
    fw.write(f'{"合計":44} {data["購入_n"].sum():>6,.0f} {data["払戻_n"].sum():>7,.0f} {data["返還_n"].sum():>5,.0f}\n')
    fw.write(f'\n※CSVファイル合計行: 購入49,100円 払戻35,290円\n')
print('done')
