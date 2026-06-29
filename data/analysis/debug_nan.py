import pandas as pd
import glob, os, re

def parse_amount(val):
    s = str(val).strip()
    if s in ('', 'nan', '―', '-'):
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

files = sorted(glob.glob('G:/マイドライブ/horse_racing_ai/data/tohyo/*_tohyo.csv'))
all_dfs = []
for f in files:
    fname = os.path.basename(f)
    if fname == 'video_tohyo_data.csv':
        continue
    for enc in ['cp932', 'utf-8-sig', 'utf-8']:
        try:
            df = pd.read_csv(f, encoding=enc)
            df['_source'] = fname
            all_dfs.append(df)
            break
        except:
            pass

video_path = 'G:/マイドライブ/horse_racing_ai/data/tohyo/video_tohyo_data.csv'
df_video = pd.read_csv(video_path, encoding='utf-8-sig')
df_video['_source'] = 'video_tohyo_data.csv'
all_dfs.append(df_video)

all_df = pd.concat(all_dfs, ignore_index=True)
all_df['購入金額_num'] = all_df['購入金額'].apply(parse_amount)

# 日付変換
def fix_date(x):
    s = str(x).strip()
    if s in ('nan', '', 'None'):
        return None
    s = s.replace('.0', '')
    if re.match(r'^\d{8}$', s):
        return s
    return None

all_df['日付_fixed'] = all_df['日付'].apply(fix_date)

# NaN日付の行を調査
nan_rows = all_df[all_df['日付_fixed'].isna()]
print(f'NaN日付の行数: {len(nan_rows)}')
print(f'NaN日付の購入金額合計: {nan_rows["購入金額_num"].sum():,.0f}円')
print('\nNaN行のソース別件数:')
print(nan_rows['_source'].value_counts())
if len(nan_rows) > 0:
    with open('G:/マイドライブ/horse_racing_ai/data/nan_rows.txt', 'w', encoding='utf-8') as fw:
        fw.write(nan_rows[['_source','日付','式別','購入金額','払戻金額']].to_string())
    print('\nnan_rows.txtに保存しました')

# 正常行のみ集計
valid_df = all_df[all_df['日付_fixed'].notna()].copy()
print(f'\n有効行数: {len(valid_df)}')
print(f'有効行 総購入: {valid_df["購入金額_num"].sum():,.0f}円')
