import pickle, pandas as pd, numpy as np, sys
sys.stdout.reconfigure(encoding='utf-8')

results = pd.read_csv('data/raw/results/20260530.csv', encoding='utf-8-sig')

# 着順の分布確認
print("着順分布:")
print(results['着順'].value_counts().head(20))
print()

# 着順1位だけ表示
winners = results[results['着順'] == 1]
print(f"1着馬数: {len(winners)} / {len(results['レースNo'].unique())}R")
print(winners[['会場','レースNo','馬名','単勝オッズ']].head(10).to_string(index=False))
