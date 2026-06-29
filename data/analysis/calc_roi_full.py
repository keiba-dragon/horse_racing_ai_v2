import pandas as pd
import glob, os

# --- 日付別ファイル読み込み ---
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

# video_tohyo_data.csv
video_path = 'G:/マイドライブ/horse_racing_ai/data/tohyo/video_tohyo_data.csv'
for enc in ['utf-8-sig', 'utf-8', 'cp932']:
    try:
        df_video = pd.read_csv(video_path, encoding=enc)
        dfs.append(df_video)
        print(f'video: {len(df_video)}行 ({enc})')
        break
    except:
        pass

all_df = pd.concat(dfs, ignore_index=True)
print(f'合計: {len(all_df)}行')

# 数値変換
for col in ['購入金額', '払戻金額', '返還金額']:
    all_df[col] = pd.to_numeric(all_df[col], errors='coerce').fillna(0)

# 日付を文字列に統一
all_df['日付'] = all_df['日付'].astype(str).str.replace('.0', '', regex=False).str.strip()

# 式別の正規化（全角スペース等除去）
all_df['式別'] = all_df['式別'].astype(str).str.strip()

# --- 全期間集計 ---
total_buy  = all_df['購入金額'].sum()
total_pay  = all_df['払戻金額'].sum()
total_ret  = all_df['返還金額'].sum()
net        = total_pay + total_ret - total_buy
roi        = net / total_buy * 100 if total_buy > 0 else 0

print(f'\n=== 全期間 ===')
print(f'総投資:   {total_buy:,.0f}円')
print(f'総払戻:   {total_pay:,.0f}円')
print(f'総返還:   {total_ret:,.0f}円')
print(f'損益:     {net:,.0f}円')
print(f'ROI:      {roi:.1f}%')

# --- 日付別 ---
daily = all_df.groupby('日付').agg({'購入金額':'sum','払戻金額':'sum','返還金額':'sum'}).reset_index()
daily['損益'] = daily['払戻金額'] + daily['返還金額'] - daily['購入金額']
daily['ROI(%)'] = (daily['損益'] / daily['購入金額'] * 100).round(1)
daily = daily.sort_values('日付')
print('\n=== 日付別 ===')
print(daily.to_string(index=False))

# --- 式別集計 ---
# フォーメーション等を親式別にまとめる
def normalize_shikibetsu(s):
    s = str(s).strip()
    if '3連単' in s: return '3連単'
    if '3連複' in s or '３連複' in s: return '3連複'
    if 'ワイド' in s: return 'ワイド'
    if '馬連' in s: return '馬連'
    if '馬単' in s: return '馬単'
    if '枠連' in s: return '枠連'
    if '単勝' in s: return '単勝'
    if '複勝' in s: return '複勝'
    return s

all_df['式別_normalized'] = all_df['式別'].apply(normalize_shikibetsu)
by_type = all_df.groupby('式別_normalized').agg({'購入金額':'sum','払戻金額':'sum','返還金額':'sum'}).reset_index()
by_type['損益'] = by_type['払戻金額'] + by_type['返還金額'] - by_type['購入金額']
by_type['ROI(%)'] = (by_type['損益'] / by_type['購入金額'].replace(0, float('nan')) * 100).round(1)
print('\n=== 式別（統合）===')
print(by_type.to_string(index=False))

# --- プラス日数 ---
plus_days = (daily['損益'] > 0).sum()
total_days = len(daily)
print(f'\nプラス日数: {plus_days}/{total_days}日')
