import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os
import pickle
import json
import re
import lightgbm as lgb

TRAIN_START = 130105
TRAIN_END   = 201231

def get_distance_band(dist):
    try:
        m = re.search(r'\d+', str(dist))
        if not m: return None
        d = int(m.group())
    except: return None
    if d <= 1400:   return '短距離'
    elif d <= 1800: return 'マイル'
    elif d <= 2200: return '中距離'
    else:           return '長距離'

def get_class_group(class_rank):
    try:
        r = int(float(class_rank))
    except: return None
    if np.isnan(float(class_rank)): return None
    if r == 1: return '新馬'
    elif r == 2: return '未勝利'
    elif r == 3: return '1勝'
    elif r == 4: return '2勝'
    elif r >= 5: return '3勝以上'
    return None

def main():
    print("--- サブランカー学習: 芝ダ × 距離帯 × クラスグループ ---")
    print(f"学習データ: {TRAIN_START}–{TRAIN_END}  (2021+はOOSホールドアウト)\n")

    base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'src' in os.path.abspath(__file__) else '.'
    input_file = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.csv')
    model_dir  = os.path.join(base_dir, 'models', 'sub_ranker')
    os.makedirs(model_dir, exist_ok=True)

    # サブモデルの特徴量リストを流用
    sub_info_path = os.path.join(base_dir, 'models', 'submodel', 'submodel_info.json')
    with open(sub_info_path, 'r', encoding='utf-8') as f:
        sub_info = json.load(f)
    features = sub_info['features']

    # データ読み込み
    print("加工済みデータを読み込んでいます...")
    df = pd.read_csv(input_file, low_memory=False)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')

    if '着順_num' in df.columns:
        df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    else:
        df['着順_num'] = (
            df['着順'].astype(str)
            .str.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
            .pipe(pd.to_numeric, errors='coerce')
        )
    df['頭数_num'] = pd.to_numeric(df['頭数'], errors='coerce')
    df = df.dropna(subset=['日付_num', '着順_num']).copy()

    # ランクラベル（1着=頭数、最下位=1）
    df['rank_label'] = (
        np.maximum(0, df['頭数_num'] - df['着順_num'] + 1).fillna(0)
    ).astype(int)

    # グループキー
    df['_surface']  = df['芝・ダ'].astype(str).str.strip()
    df['_dist_band'] = df['距離'].apply(get_distance_band)
    mask_da = (df['_surface'] == 'ダ') & (df['_dist_band'].isin(['中距離', '長距離']))
    df.loc[mask_da, '_dist_band'] = '中長距離'
    df['_class_group'] = df['クラス_rank'].apply(get_class_group) if 'クラス_rank' in df.columns else np.nan
    df['model_key'] = df['_surface'] + '_' + df['_dist_band'].astype(str) + '_' + df['_class_group'].astype(str)
    df = df[~df['model_key'].str.contains('None|nan', na=True)]

    # 特徴量を数値化
    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['race_id'] = df['日付_num'].astype(str) + '_' + df['開催'].astype(str) + '_' + df['レース名'].astype(str)

    MIN_SAMPLES   = 300
    group_counts  = df['model_key'].value_counts()
    target_groups = group_counts[group_counts >= MIN_SAMPLES].index.tolist()
    print(f"学習対象グループ: {len(target_groups)}種類\n")

    trained_rankers = {}
    results = []

    for i, key in enumerate(sorted(target_groups), 1):
        df_g_all = df[df['model_key'] == key].sort_values('日付_num').reset_index(drop=True)
        df_g = df_g_all[
            (df_g_all['日付_num'] >= TRAIN_START) &
            (df_g_all['日付_num'] <= TRAIN_END)
        ].reset_index(drop=True)
        n     = len(df_g)
        split = int(n * 0.8)

        df_train = df_g.iloc[:split].copy().sort_values('race_id')
        df_test  = df_g.iloc[split:].copy().sort_values('race_id')
        if len(df_test) < 50:
            continue

        train_groups = df_train.groupby('race_id').size().values
        test_groups  = df_test.groupby('race_id').size().values

        X_train = df_train[features]
        y_train = df_train['rank_label']
        X_test  = df_test[features]
        y_test  = df_test['rank_label']

        ranker = lgb.LGBMRanker(
            n_estimators=300,
            random_state=42,
            verbosity=-1,
            objective='lambdarank',
            metric='ndcg',
            ndcg_eval_at=[1, 3],
        )
        ranker.fit(
            X_train, y_train,
            group=train_groups,
            eval_set=[(X_test, y_test)],
            eval_group=[test_groups],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20, verbose=False),
                lgb.log_evaluation(period=-1),
            ]
        )

        scores = ranker.predict(X_test)
        df_test = df_test.copy()
        df_test['score'] = scores
        rank1_positions = []
        for _, grp in df_test.groupby('race_id'):
            if len(grp) < 2: continue
            grp2 = grp.sort_values('score', ascending=False).reset_index(drop=True)
            winners = grp2[grp2['着順_num'] == 1]
            if len(winners) > 0:
                rank1_positions.append(winners.index[0] + 1)

        avg_pos   = np.mean(rank1_positions) if rank1_positions else 99
        top1_rate = np.mean([p == 1 for p in rank1_positions]) if rank1_positions else 0

        fn = f'sub_ranker_{key}.pkl'
        with open(os.path.join(model_dir, fn), 'wb') as f:
            pickle.dump(ranker, f)

        trained_rankers[key] = fn
        results.append({'グループ': key, 'サンプル数': n, '1位的中': f'{top1_rate:.1%}'})
        n_all = len(df_g_all)
        print(f"[{i:2d}/{len(target_groups)}] {key:25s} 全{n_all:5,}件→学習{n:5,}件  1位的中:{top1_rate:.1%}")

    ranker_info = {'features': features, 'rankers': trained_rankers}
    info_path = os.path.join(model_dir, 'sub_ranker_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(ranker_info, f, ensure_ascii=False, indent=2)

    print(f"\n=== 学習完了 ===")
    print(f"サブランカー数: {len(trained_rankers)}グループ")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nサブランカー情報: {info_path}")

if __name__ == "__main__":
    main()
