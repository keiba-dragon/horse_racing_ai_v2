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
data['レース'] = data['レース'].apply(lambda x: int(float(str(x))) if str(x).replace('.','').isdigit() else x)

with open('G:/マイドライブ/horse_racing_ai/data/races_0322_byrace.txt', 'w', encoding='utf-8') as fw:
    grand_buy = grand_pay = 0
    for (venue, race), grp in data.groupby(['場名', 'レース'], sort=True):
        buy = grp['購入_n'].sum()
        pay = grp['払戻_n'].sum()
        ret = grp['返還_n'].sum()
        net = pay + ret - buy
        grand_buy += buy
        grand_pay += pay
        fw.write(f'【{venue} {race}R】 購入:{buy:,}円  払戻:{pay:,}円  損益:{net:+,}円\n')
        for _, r in grp.iterrows():
            fw.write(f'  {str(r["式別"])[:16]:16} {str(r["馬／組番"])[:20]:20} {r["購入_n"]:>6,}円  払戻:{r["払戻_n"]:>7,}円\n')
        fw.write('\n')
    fw.write(f'【日計】 購入:{grand_buy:,}円  払戻:{grand_pay:,}円  損益:{grand_pay-grand_buy:+,}円\n')
    fw.write(f'※CSVの合計行: 購入49,100円\n')
print('done')
