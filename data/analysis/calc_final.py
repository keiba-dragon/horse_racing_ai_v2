import pandas as pd
import glob, os, re

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

# 日付別CSVのみ（video除外）
files = sorted(glob.glob('G:/マイドライブ/horse_racing_ai/data/tohyo/*_tohyo.csv'))
dfs = []
for f in files:
    if 'video_tohyo_data' in f:
        continue
    for enc in ['cp932', 'utf-8-sig', 'utf-8']:
        try:
            df = pd.read_csv(f, encoding=enc)
            dfs.append(df)
            break
        except:
            pass

all_df = pd.concat(dfs, ignore_index=True)

# 合計行除外
all_df = all_df[all_df['日付'].notna()].copy()

for col in ['購入金額', '払戻金額', '返還金額']:
    all_df[col] = all_df[col].apply(parse_amount)

all_df['日付'] = all_df['日付'].apply(
    lambda x: str(int(float(str(x)))) if str(x).replace('.','').isdigit() else None)
all_df = all_df[all_df['日付'].notna()].copy()

# 03/22 の購入金額差を調べる
df_0322 = all_df[all_df['日付'] == '20260322']
print(f'03/22 購入合計（パース）: {df_0322["購入金額"].sum():,.0f}円')
print(f'03/22 払戻合計（パース）: {df_0322["払戻金額"].sum():,.0f}円')
print(f'※CSVの合計行: 購入49,100円・払戻35,290円')

# 日付別集計
daily = all_df.groupby('日付').agg({'購入金額':'sum','払戻金額':'sum','返還金額':'sum'}).reset_index()
daily['損益'] = daily['払戻金額'] + daily['返還金額'] - daily['購入金額']
daily['ROI(%)'] = (daily['損益'] / daily['購入金額'].replace(0, float('nan')) * 100).round(1)
daily = daily.sort_values('日付').reset_index(drop=True)
daily['累計損益'] = daily['損益'].cumsum()

total_buy = all_df['購入金額'].sum()
total_pay = all_df['払戻金額'].sum()
total_ret = all_df['返還金額'].sum()
net = total_pay + total_ret - total_buy
roi = net / total_buy * 100

print(f'\n=== 全期間（videoなし） ===')
print(f'総投資: {total_buy:,.0f}円 / 総払戻: {total_pay:,.0f}円 / 返還: {total_ret:,.0f}円')
print(f'損益: {net:+,.0f}円 / ROI: {roi:+.1f}%')
print(f'プラス日数: {int((daily["損益"]>0).sum())}/{len(daily)}日\n')
print('=== 日付別 ===')

with open('G:/マイドライブ/horse_racing_ai/data/result_novideo.txt', 'w', encoding='utf-8') as fw:
    fw.write(f'=== 全期間（メイン口のみ） ===\n')
    fw.write(f'総投資: {total_buy:,.0f}円\n')
    fw.write(f'総払戻: {total_pay:,.0f}円\n')
    fw.write(f'総返還: {total_ret:,.0f}円\n')
    fw.write(f'損益: {net:+,.0f}円\n')
    fw.write(f'ROI: {roi:+.1f}%\n')
    fw.write(f'プラス日数: {int((daily["損益"]>0).sum())}/{len(daily)}日\n\n')
    fw.write('=== 日付別 ===\n')
    fw.write(daily.to_string(index=False))

for _, r in daily.iterrows():
    d = str(r['日付'])
    ds = f"{d[:4]}/{d[4:6]}/{d[6:8]}"
    print(f"{ds}  購入:{r['購入金額']:>7,.0f}  払戻:{r['払戻金額']:>7,.0f}  返還:{r['返還金額']:>5,.0f}  損益:{r['損益']:>+8,.0f}  ROI:{r['ROI(%)']:>+7.1f}%  累計:{r['累計損益']:>+9,.0f}")
