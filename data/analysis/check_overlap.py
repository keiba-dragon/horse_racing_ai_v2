import pandas as pd

# 20260228の日付別CSVとvideo_tohyo_data.csvを比較
df_daily = pd.read_csv('G:/マイドライブ/horse_racing_ai/data/tohyo/20260228_tohyo.csv', encoding='utf-8-sig')
df_video = pd.read_csv('G:/マイドライブ/horse_racing_ai/data/tohyo/video_tohyo_data.csv', encoding='utf-8-sig')
df_video_228 = df_video[df_video['日付'] == 20260228]

with open('G:/マイドライブ/horse_racing_ai/data/overlap_check.txt', 'w', encoding='utf-8') as fw:
    fw.write('=== 20260228_tohyo.csv ===\n')
    fw.write(f'行数: {len(df_daily)}\n')
    fw.write(df_daily.to_string() + '\n\n')
    fw.write('=== video_tohyo_data.csv (20260228のみ) ===\n')
    fw.write(f'行数: {len(df_video_228)}\n')
    fw.write(df_video_228.to_string() + '\n')
print('done')
