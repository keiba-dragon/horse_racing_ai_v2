import pandas as pd

video_path = 'G:/マイドライブ/horse_racing_ai/data/tohyo/video_tohyo_data.csv'
df = pd.read_csv(video_path, encoding='utf-8-sig')

with open('G:/マイドライブ/horse_racing_ai/data/video_check.txt', 'w', encoding='utf-8') as fw:
    fw.write(f'行数: {len(df)}\n')
    fw.write(f'列: {df.columns.tolist()}\n')
    fw.write(f'\n日付ユニーク:\n{df["日付"].unique()}\n')
    fw.write(f'\n式別ユニーク:\n{df["式別"].unique()}\n')
    fw.write(f'\n的中/返還 ユニーク:\n{df["的中／返還"].unique()}\n')
    fw.write(f'\n払戻単価 ユニーク（先頭20）:\n{df["払戻単価"].unique()[:20]}\n')
    fw.write(f'\n--- 全データ ---\n')
    fw.write(df.to_string())
print('done')
