import pandas as pd, glob, os, sys
from datetime import datetime
sys.stdout.reconfigure(encoding='utf-8')

files = glob.glob('data/**/*.parquet', recursive=True)
for f in files:
    mtime = datetime.fromtimestamp(os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M')
    df = pd.read_parquet(f)
    date_col = next((c for c in ['日付','date','レース日'] if c in df.columns), None)
    max_date = df[date_col].max() if date_col else '不明'
    print(f'{f}')
    print(f'  更新日時: {mtime}  行数: {len(df):,}  最終レース日: {max_date}')

    # ダービー出走馬がいるか確認
    derby_horses = ['エムズビギン','コンジェスタス','ロブチェン','リアライズシリウス',
                    'パントルナイーフ','ゴーイントゥスカイ','マテンロウゲイル']
    uma_col = next((c for c in ['馬名S','馬名'] if c in df.columns), None)
    if uma_col:
        found = [h for h in derby_horses if (df[uma_col] == h).any()]
        missing = [h for h in derby_horses if h not in found]
        print(f'  ダービー馬(確認7頭中) 存在: {found}')
        print(f'  不在: {missing}')
