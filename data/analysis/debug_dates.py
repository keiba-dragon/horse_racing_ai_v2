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

target_dates = ['20260228', '20260301', '20260308', '20260314', '20260315', '20260322']

# 日付別CSVを1つずつ確認
files = {
    '20260228': 'G:/マイドライブ/horse_racing_ai/data/tohyo/20260228_tohyo.csv',
    '20260301': 'G:/マイドライブ/horse_racing_ai/data/tohyo/20260301_tohyo.csv',
    '20260308': 'G:/マイドライブ/horse_racing_ai/data/tohyo/20260308_tohyo.csv',
    '20260314': 'G:/マイドライブ/horse_racing_ai/data/tohyo/20260314_tohyo.csv',
    '20260315': 'G:/マイドライブ/horse_racing_ai/data/tohyo/20260315_tohyo.csv',
    '20260322': 'G:/マイドライブ/horse_racing_ai/data/tohyo/20260322_tohyo.csv',
}

video_path = 'G:/マイドライブ/horse_racing_ai/data/tohyo/video_tohyo_data.csv'
df_video = pd.read_csv(video_path, encoding='utf-8-sig')

with open('G:/マイドライブ/horse_racing_ai/data/debug_dates.txt', 'w', encoding='utf-8') as fw:
    for date, fpath in files.items():
        for enc in ['cp932', 'utf-8-sig', 'utf-8']:
            try:
                df = pd.read_csv(fpath, encoding=enc)
                break
            except:
                pass
        # 合計行除外
        df = df[df['日付'].notna()].copy()
        df['購入金額_n'] = df['購入金額'].apply(parse_amount)
        df['払戻金額_n'] = df['払戻金額'].apply(parse_amount)
        df['返還金額_n'] = df['返還金額'].apply(parse_amount)
        total_buy = df['購入金額_n'].sum()
        total_pay = df['払戻金額_n'].sum()
        total_ret = df['返還金額_n'].sum()

        fw.write(f'=== {date} 日付別CSV ({len(df)}行) ===\n')
        fw.write(f'購入: {total_buy:,.0f}円  払戻: {total_pay:,.0f}円  返還: {total_ret:,.0f}円  損益: {total_pay+total_ret-total_buy:+,.0f}円\n')
        fw.write(df[['式別','馬／組番','購入金額','払戻金額','返還金額']].to_string(index=False) + '\n\n')

        # video側
        dv = df_video[df_video['日付'] == int(date)].copy()
        if len(dv) > 0:
            dv['購入金額_n'] = dv['購入金額'].apply(parse_amount)
            dv['払戻金額_n'] = dv['払戻金額'].apply(parse_amount)
            dv['返還金額_n'] = dv['返還金額'].apply(parse_amount)
            vbuy = dv['購入金額_n'].sum()
            vpay = dv['払戻金額_n'].sum()
            vret = dv['返還金額_n'].sum()
            fw.write(f'--- {date} video ({len(dv)}行) ---\n')
            fw.write(f'購入: {vbuy:,.0f}円  払戻: {vpay:,.0f}円  返還: {vret:,.0f}円  損益: {vpay+vret-vbuy:+,.0f}円\n')
            fw.write(dv[['式別','馬／組番','購入金額','払戻金額']].to_string(index=False) + '\n\n')

print('done')
