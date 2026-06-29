# coding: utf-8
"""テスト(OOS)と本番の差異を洗い出す"""
import pickle, io, sys, numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, 'src')

# ── 今日のキャッシュから_nan_featuresを取得 ──
with open(r'data\raw\cache\出馬表形式05月30日_api.cache.pkl', 'rb') as f:
    cache = pickle.load(f)
result = cache['result']

print("=" * 60)
print("【差異1】NaN特徴量")
print("=" * 60)
if '_nan_features' in result.columns and '_nan_count' in result.columns:
    # NaN数が0より多い馬だけ
    nan_horses = result[result['_nan_count'] > 0][['馬名S', '_nan_count', '_nan_total', '_nan_features']].copy()
    print(f"NaN特徴量あり: {len(nan_horses)}頭 / {len(result)}頭")
    # 特徴量別集計
    from collections import Counter
    cnt = Counter()
    for row in result['_nan_features'].dropna():
        if row:
            for feat in row.split(','):
                if feat.strip():
                    cnt[feat.strip()] += 1
    print(f"\n特徴量別NaN馬数 (上位20):")
    for feat, n in sorted(cnt.items(), key=lambda x: -x[1])[:20]:
        pct = n / len(result) * 100
        print(f"  {feat:40s}: {n:4d}頭 ({pct:.1f}%)")
    print(f"\nNaN count分布:")
    print(result['_nan_count'].value_counts().sort_index().head(10))
else:
    print("_nan_features 列なし（古いキャッシュ）")

# ── 市場オッズの確認 ──
print("\n" + "=" * 60)
print("【差異2】市場オッズ（OOS=確定オッズ、本番=予測時刻）")
print("=" * 60)
if '単勝オッズ' in result.columns:
    odds = pd.to_numeric(result['単勝オッズ'], errors='coerce')
    nan_odds = odds.isna().sum()
    print(f"オッズNaN: {nan_odds}頭 / {len(result)}頭")
    print(f"オッズ範囲: {odds.min():.1f} ～ {odds.max():.1f}")
    # オッズNaN馬はスコア=calibのみ → factorなし
    print(f"※OOSでは確定オッズ、本番では予測時点オッズ → score差異")

# ── clogit確率の0%馬 ──
print("\n" + "=" * 60)
print("【差異3】clogit_calib=0 の馬 vs OOS統計")
print("=" * 60)
if 'clogit_calib' in result.columns:
    calib = result['clogit_calib'].dropna()
    zero = (calib == 0).sum()
    lt1pct = (calib < 0.01).sum()
    print(f"今日: 0.0% = {zero}頭 / <1% = {lt1pct}頭 / 全体 {len(calib)}頭")
    print(f"今日の1レース平均: {lt1pct/result.groupby('開催').ngroups:.1f}頭/レース")

# ── OOSパーケットのNaN確認 ──
print("\n" + "=" * 60)
print("【差異4】OOS parquetのNaN率（clogit特徴量）")
print("=" * 60)
try:
    with open('models/final_model.pkl', 'rb') as f:
        pkg = pickle.load(f)
    feat_cols = list(pkg['artifacts']['芝']['feat_cols'])
    print(f"clogit特徴量数: {len(feat_cols)}")

    # parquetのOOSデータのNaN率
    import pyarrow.parquet as pq_mod
    feat_pq = 'data/processed/all_venues_features.parquet'
    avail = set(pq_mod.read_schema(feat_pq).names)
    load_cols = [c for c in feat_cols if c in avail] + ['日付']
    df_oos = pd.read_parquet(feat_pq, columns=load_cols)
    df_oos['日付_num'] = pd.to_numeric(df_oos['日付'], errors='coerce')
    df_oos = df_oos[df_oos['日付_num'] >= 230101]
    print(f"OOS行数: {len(df_oos):,}")

    nan_rates = df_oos[feat_cols].isna().mean().sort_values(ascending=False)
    print(f"\nOOS NaN率 >0% の特徴量:")
    high_nan = nan_rates[nan_rates > 0]
    if len(high_nan) == 0:
        print("  なし（全特徴量がOOSで埋まっている）")
    else:
        for col, rate in high_nan.items():
            print(f"  {col:40s}: {rate*100:.1f}%")

    # 今日の本番データのNaN率
    print(f"\n今日の本番データ NaN率 (clogit特徴量):")
    today_feats = [c for c in feat_cols if c in result.columns]
    nan_today = result[today_feats].isna().mean().sort_values(ascending=False)
    high_nan_today = nan_today[nan_today > 0]
    if len(high_nan_today) == 0:
        print("  なし")
    else:
        for col, rate in high_nan_today.items():
            oos_rate = nan_rates.get(col, 0)
            diff = rate - oos_rate
            print(f"  {col:40s}: 本番{rate*100:.0f}%  OOS{oos_rate*100:.0f}%  差{diff*100:+.0f}%")
except Exception as e:
    print(f"エラー: {e}")

# ── クラス_rank分布（factor決定に影響） ──
print("\n" + "=" * 60)
print("【差異5】クラス_rank / factor（未勝利=0.00 / 非未勝利=0.16）")
print("=" * 60)
if 'クラス_rank' in result.columns:
    cr = pd.to_numeric(result['クラス_rank'], errors='coerce')
    print(f"クラス_rank分布:")
    print(cr.value_counts().sort_index())
    maiden = (cr == 2).sum()
    other  = (cr != 2).sum()
    nan_cr = cr.isna().sum()
    print(f"未勝利(=2): {maiden}頭, 非未勝利: {other}頭, NaN→0: {nan_cr}頭")
