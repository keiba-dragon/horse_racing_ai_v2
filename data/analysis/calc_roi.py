import pandas as pd
import glob, os

files = sorted(glob.glob('G:/マイドライブ/horse_racing_ai/data/tohyo/*_tohyo.csv'))
dfs = []
for f in files:
    fname = os.path.basename(f)
    for enc in ['cp932', 'utf-8-sig', 'utf-8']:
        try:
            df = pd.read_csv(f, encoding=enc)
            dfs.append(df)
            print(fname, enc, len(df))
            break
        except Exception as e:
            pass

all_df = pd.concat(dfs, ignore_index=True)

# 数値変換
for col in ['購入金額', '払戻金額', '返還金額']:
    all_df[col] = pd.to_numeric(all_df[col], errors='coerce').fillna(0)

total_purchase = all_df['購入金額'].sum()
total_payout = all_df['払戻金額'].sum()
total_return = all_df['返還金額'].sum()
net = total_payout + total_return - total_purchase
roi = net / total_purchase * 100

out_path = 'G:/マイドライブ/horse_racing_ai/data/roi_check.txt'
with open(out_path, 'w', encoding='utf-8') as fw:
    fw.write(f'=== 全期間合計 ===\n')
    fw.write(f'総購入金額: {total_purchase:,.0f}円\n')
    fw.write(f'総払戻金額: {total_payout:,.0f}円\n')
    fw.write(f'総返還金額: {total_return:,.0f}円\n')
    fw.write(f'損益: {net:,.0f}円\n')
    fw.write(f'ROI: {roi:.1f}%\n\n')
    fw.write('=== 日付別 ===\n')
    daily = all_df.groupby('日付').agg({'購入金額':'sum','払戻金額':'sum','返還金額':'sum'}).reset_index()
    daily['損益'] = daily['払戻金額'] + daily['返還金額'] - daily['購入金額']
    daily['ROI(%)'] = (daily['損益'] / daily['購入金額'] * 100).round(1)
    fw.write(daily.to_string(index=False) + '\n\n')
    fw.write('=== 式別集計 ===\n')
    by_type = all_df.groupby('式別').agg({'購入金額':'sum','払戻金額':'sum','返還金額':'sum'}).reset_index()
    by_type['損益'] = by_type['払戻金額'] + by_type['返還金額'] - by_type['購入金額']
    by_type['ROI(%)'] = (by_type['損益'] / by_type['購入金額'] * 100).round(1)
    fw.write(by_type.to_string(index=False) + '\n')

print('done ->', out_path)
