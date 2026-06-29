import pandas as pd
import glob, os, re

def parse_amount(val):
    s = str(val).strip()
    if s in ('', 'nan', '―', '-', '×', '\\', '¥'):
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

def normalize_shikibetsu(s):
    s = str(s).strip()
    if '3連単' in s or '３連単' in s: return '3連単'
    if '3連複' in s or '３連複' in s: return '3連複'
    if 'ワイド' in s: return 'ワイド'
    if '馬連' in s: return '馬連'
    if '馬単' in s: return '馬単'
    if '枠連' in s: return '枠連'
    if '単勝' in s: return '単勝'
    if '複勝' in s: return '複勝'
    return s

def load_daily_csvs():
    files = sorted(glob.glob('G:/マイドライブ/horse_racing_ai/data/tohyo/*_tohyo.csv'))
    dfs = []
    for f in files:
        fname = os.path.basename(f)
        if fname == 'video_tohyo_data.csv':
            continue
        for enc in ['cp932', 'utf-8-sig', 'utf-8']:
            try:
                df = pd.read_csv(f, encoding=enc)
                dfs.append(df)
                break
            except:
                pass
    return pd.concat(dfs, ignore_index=True)

def load_video_csv():
    path = 'G:/マイドライブ/horse_racing_ai/data/tohyo/video_tohyo_data.csv'
    for enc in ['utf-8-sig', 'utf-8', 'cp932']:
        try:
            return pd.read_csv(path, encoding=enc)
        except:
            pass
    return pd.DataFrame()

def prepare(df):
    for col in ['購入金額', '払戻金額', '返還金額']:
        df[col] = df[col].apply(parse_amount)
    df['日付'] = df['日付'].apply(
        lambda x: str(int(float(str(x)))) if str(x).replace('.','').isdigit() else None)
    df = df[df['日付'].notna()].copy()
    df['式別_norm'] = df['式別'].apply(normalize_shikibetsu)
    return df

def aggregate(df):
    total_buy = df['購入金額'].sum()
    total_pay = df['払戻金額'].sum()
    total_ret = df['返還金額'].sum()
    net = total_pay + total_ret - total_buy
    roi = net / total_buy * 100 if total_buy > 0 else 0

    daily = df.groupby('日付').agg({'購入金額':'sum','払戻金額':'sum','返還金額':'sum'}).reset_index()
    daily['損益'] = daily['払戻金額'] + daily['返還金額'] - daily['購入金額']
    daily['ROI(%)'] = (daily['損益'] / daily['購入金額'].replace(0, float('nan')) * 100).round(1)
    daily = daily.sort_values('日付').reset_index(drop=True)
    daily['累計損益'] = daily['損益'].cumsum()

    by_type = df.groupby('式別_norm').agg({'購入金額':'sum','払戻金額':'sum','返還金額':'sum'}).reset_index()
    by_type['損益'] = by_type['払戻金額'] + by_type['返還金額'] - by_type['購入金額']
    by_type['ROI(%)'] = (by_type['損益'] / by_type['購入金額'].replace(0, float('nan')) * 100).round(1)
    by_type = by_type[by_type['購入金額'] > 0].sort_values('購入金額', ascending=False).reset_index(drop=True)

    return {
        'total_buy': total_buy, 'total_pay': total_pay, 'total_ret': total_ret,
        'net': net, 'roi': roi,
        'daily': daily, 'by_type': by_type,
        'plus_days': int((daily['損益'] > 0).sum()),
        'total_days': len(daily)
    }

df_main = prepare(load_daily_csvs())
df_video = prepare(load_video_csv())
df_combined = prepare(pd.concat([load_daily_csvs(), load_video_csv()], ignore_index=True))

main    = aggregate(df_main)
video   = aggregate(df_video)
combined = aggregate(df_combined)

with open('G:/マイドライブ/horse_racing_ai/data/roi_result.txt', 'w', encoding='utf-8') as fw:
    for label, r in [('メイン口', main), ('別口(video)', video), ('合算', combined)]:
        fw.write(f'=== {label} ===\n')
        fw.write(f'総投資: {r["total_buy"]:,.0f}円\n')
        fw.write(f'総払戻: {r["total_pay"]:,.0f}円\n')
        fw.write(f'総返還: {r["total_ret"]:,.0f}円\n')
        fw.write(f'損益:   {r["net"]:+,.0f}円\n')
        fw.write(f'ROI:    {r["roi"]:+.1f}%\n')
        fw.write(f'プラス日数: {r["plus_days"]}/{r["total_days"]}日\n\n')
        fw.write('--- 式別 ---\n')
        fw.write(r['by_type'].to_string(index=False) + '\n\n')
        fw.write('--- 日付別 ---\n')
        fw.write(r['daily'].to_string(index=False) + '\n\n')

print('done')
