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
data['レース'] = data['レース'].apply(lambda x: int(float(str(x))) if str(x).replace('.','').isdigit() else x)

nakayama = data[data['場名'] == '中山'].sort_values('レース')

with open('G:/マイドライブ/horse_racing_ai/data/nakayama_0322.txt', 'w', encoding='utf-8') as fw:
    for race, grp in nakayama.groupby('レース'):
        buy = grp['購入_n'].sum()
        pay = grp['払戻_n'].sum()
        fw.write(f'【中山 {race}R】 購入:{buy:,}円  払戻:{pay:,}円\n')
        for _, r in grp.iterrows():
            hit = str(r['的中／返還']).strip()
            fw.write(f'  {str(r["式別"])[:16]:16} {str(r["馬／組番"])[:22]:22} 購入:{r["購入_n"]:>6,}円  払戻:{r["払戻_n"]:>7,}円  [{hit}]\n')
        fw.write('\n')
    fw.write(f'中山合計 購入:{nakayama["購入_n"].sum():,}円  払戻:{nakayama["払戻_n"].sum():,}円\n')
print('done')
