import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import pickle
import json

# ============================================================
# サブモデル: 芝ダ × 距離帯 × クラスグループ
# （現行 models/ は一切変更しない → models/submodel/ に保存）
# ============================================================

# 距離帯の定義
def get_distance_band(dist):
    """距離(m) → 距離帯ラベル（'ダ1200'や'芝1800'などの前置き文字も除去）"""
    import re
    try:
        m = re.search(r'\d+', str(dist))
        if not m:
            return None
        d = int(m.group())
    except (ValueError, TypeError):
        return None
    if d <= 1400:
        return '短距離'
    elif d <= 1800:
        return 'マイル'
    elif d <= 2200:
        return '中距離'
    else:
        return '長距離'

# クラスグループの定義（02_train_model.py の encode_class と対応）
# クラス_rank: 新馬=1, 未勝利=2, 1勝=3, 2勝=4, 3勝=5, OP=6, G3=7, G2=8, G1=9, 障害=NaN
def get_class_group(class_rank):
    """クラス_rank → クラスグループラベル"""
    try:
        r = float(class_rank)
    except (ValueError, TypeError):
        return None  # 障害等 → NaN → 除外
    if np.isnan(r):
        return None
    r = int(r)
    if r == 1:
        return '新馬'
    elif r == 2:
        return '未勝利'
    elif r == 3:
        return '1勝'
    elif r == 4:
        return '2勝'
    elif r >= 5:
        return '3勝以上'  # 3勝クラス・OP・G3・G2・G1 すべてまとめる
    return None

def train_one(X_train, y_train, X_test, y_test):
    """LightGBMモデルを1つ学習して返す"""
    model = lgb.LGBMClassifier(n_estimators=300, random_state=42, verbosity=-1)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=-1)
        ]
    )
    return model

TRAIN_START = 130105  # 学習開始: 2013/1/5
TRAIN_END   = 201231  # 学習終了: 2020/12/31 → 2021以降は真のOOSテスト

def main():
    print("--- サブモデル学習: 芝ダ × 距離帯 × クラスグループ ---")
    print("（現行 models/ は変更しません → models/submodel/ に保存）")
    print(f"学習データ: {TRAIN_START}–{TRAIN_END}  (2021+はOOSホールドアウト)\n")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'src' in os.path.abspath(__file__) else '.'
    input_file  = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.csv')
    model_dir   = os.path.join(base_dir, 'models', 'submodel')
    os.makedirs(model_dir, exist_ok=True)

    # 1. データ読み込み
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
    df = df.dropna(subset=['日付_num', '着順_num'])

    df['target_win']   = (df['着順_num'] == 1).astype(int)
    df['target_place'] = (df['着順_num'] <= 3).astype(int)
    print(f"データ読み込み完了: {len(df):,}件")

    # 2. グループキーを作成
    # 芝ダ: '芝・ダ' 列（'芝' or 'ダ'）
    # 距離帯: 距離列から生成
    # クラスグループ: クラス_rank 列から生成
    df['_surface'] = df['芝・ダ'].astype(str).str.strip()
    df['_dist_band'] = df['距離'].apply(get_distance_band)
    # ダートは中距離・長距離を統合（サンプル少のため）
    mask_da_ml = (df['_surface'] == 'ダ') & (df['_dist_band'].isin(['中距離', '長距離']))
    df.loc[mask_da_ml, '_dist_band'] = '中長距離'
    df['_class_group'] = df['クラス_rank'].apply(get_class_group) if 'クラス_rank' in df.columns else None

    if '_class_group' not in df.columns or df['_class_group'].isna().all():
        print("警告: クラス_rank 列が見つかりません。クラスグループを別途推定します...")
        # クラス_rank がなければフォールバック: クラス列から推定
        def encode_class_fallback(s):
            s = str(s)
            if '新馬' in s: return 1
            if '未勝利' in s: return 2
            if '1勝' in s or '500万' in s: return 3
            if '2勝' in s or '1000万' in s: return 4
            if '3勝' in s or '1600万' in s: return 5
            if 'OP' in s or 'オープン' in s: return 6
            if 'G3' in s: return 7
            if 'G2' in s: return 8
            if 'G1' in s: return 9
            return None
        clazz_col = 'クラス' if 'クラス' in df.columns else 'レース名'
        df['_class_rank_fb'] = df[clazz_col].apply(encode_class_fallback)
        df['_class_group'] = df['_class_rank_fb'].apply(get_class_group)

    # グループキー: 例 '芝_マイル_1勝'
    df['model_key'] = (
        df['_surface'].astype(str) + '_' +
        df['_dist_band'].astype(str) + '_' +
        df['_class_group'].astype(str)
    )
    # NaN を含む行を除外（障害 / 距離不明 / 芝ダ不明）
    df = df[~df['model_key'].str.contains('None|nan', na=True)]
    print(f"有効行数（障害・不明除外後）: {len(df):,}件")

    # 3. 特徴量の定義（02_train_model.py と同じ除外ロジック）
    import re
    def extract_venue(kaikai):
        m = re.search(r'\d+([^\d]+)', str(kaikai))
        return m.group(1) if m else str(kaikai)

    in_race_cols = ['2角', '3角', '4角']
    current_race_result_cols = [
        '走破タイム', '走破タイム_sec',
        '上り3F', 'PCI', 'RPCI', 'RPCI_x', 'RPCI_y',
        'タイム指数', '上り3F_指数',
        'Ave-3F', '平均速度', '-3F平均速度', '上り3F平均速度',
        '上3F地点差', '上3F地点差_x', '上3F地点差_y',
        '単勝配当', '複勝配当',
        '賞金',
        '着差',
        '好走',
        'PCI3',
        '馬連', '馬単', '枠連', '３連複', '３連単',
        'レース印１',
    ]
    odds_cols = ['人気', '前走人気'] + [f'{n}走前_人気' for n in range(1, 11)]
    meta_cols = [
        '着順', '着順_num', 'target_win', 'target_place',
        '日付', '日付_num', 'レース名', '馬名S', '馬記号', '生年月日',
        'Ｍ', 'Ｃ', '開催', '距離', '会場', 'model_key',
        '市場取引価格(万/最終)', '取引市場(最終)',
        '前走日付',
        # サブモデル用の一時列
        '_surface', '_dist_band', '_class_group', '_class_rank_fb',
    ]
    string_raw_cols = [
        '騎手', '調教師', '種牡馬', '母父馬', '生産者', '産地',
        '馬主(最新/仮想)', '毛色', '所属', '前騎手',
        '馬場状態', '芝・ダ', '前芝・ダ', 'コース区分',
        '前走開催', '前走レース名',
        '馬印', '馬印2', '馬印3', '馬印4',
        '前走馬印', '前走馬印2', '前走馬印3', '前走馬印4',
        '前走レース印１', '前走B', 'ブリンカー',
        '前走馬場状態', '前好走', '替',
        '脚質グループ',
        '単勝オッズ',
        '脚質_num',
        '脚質',
        '前走脚質',
    ]
    course_id_cols = (
        ['今回_距離_m', '今回_surface', '今回_コース種別', '今回_会場']
        + [f'{n}走前_距離'    for n in range(1, 11)]
        + [f'{n}走前_開催'    for n in range(1, 11)]
        + [f'{n}走前_馬場状態' for n in range(1, 11)]
        + ['前走走破タイム_sec', '前走_4角位置']
    )
    # 現行モデルがコース×距離で学習済みの情報はサブモデルでは特徴量として使える
    # （コース偏差値・レース内偏差値は除外しない）
    exclude_cols = set(in_race_cols + current_race_result_cols + odds_cols + meta_cols + string_raw_cols + course_id_cols)

    print("特徴量を選定しています...")
    for col in df.columns:
        if col not in exclude_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    features = [
        col for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    print(f"使用する特徴量: {len(features)}個\n")

    # 4. グループ別学習
    MIN_SAMPLES  = 300
    group_counts = df['model_key'].value_counts()
    target_groups = group_counts[group_counts >= MIN_SAMPLES].index.tolist()
    skipped = group_counts[group_counts < MIN_SAMPLES]

    print(f"学習対象グループ: {len(target_groups)}種類（{MIN_SAMPLES}件未満 {len(skipped)}グループはスキップ）")
    if len(skipped) > 0:
        print(f"  スキップ: {skipped.index.tolist()}\n")

    trained_models = {}
    results = []

    for i, key in enumerate(sorted(target_groups), 1):
        df_g_all = df[df['model_key'] == key].sort_values('日付_num').reset_index(drop=True)
        # 2013–2020のみを学習プールとする（2021+はOOS）
        df_g = df_g_all[
            (df_g_all['日付_num'] >= TRAIN_START) &
            (df_g_all['日付_num'] <= TRAIN_END)
        ].reset_index(drop=True)
        n    = len(df_g)

        split     = int(n * 0.8)
        X_train   = df_g[features].iloc[:split]
        X_test    = df_g[features].iloc[split:]
        y_win_tr  = df_g['target_win'].iloc[:split]
        y_win_te  = df_g['target_win'].iloc[split:]
        y_plc_tr  = df_g['target_place'].iloc[:split]
        y_plc_te  = df_g['target_place'].iloc[split:]

        if len(X_test) < 50:
            continue

        # 単勝モデル
        m_win   = train_one(X_train, y_win_tr, X_test, y_win_te)
        acc_win = accuracy_score(y_win_te, m_win.predict(X_test))
        fn_win  = f'sub_{key}_win.pkl'
        with open(os.path.join(model_dir, fn_win), 'wb') as f:
            pickle.dump(m_win, f)

        # 複勝モデル
        m_plc   = train_one(X_train, y_plc_tr, X_test, y_plc_te)
        acc_plc = accuracy_score(y_plc_te, m_plc.predict(X_test))
        fn_plc  = f'sub_{key}_place.pkl'
        with open(os.path.join(model_dir, fn_plc), 'wb') as f:
            pickle.dump(m_plc, f)

        # グループ偏差値の基準統計（学習プール全体の分布）
        prob_win_all   = m_win.predict_proba(df_g[features])[:, 1]
        prob_place_all = m_plc.predict_proba(df_g[features])[:, 1]
        group_stats = {
            'win_mean':   float(prob_win_all.mean()),
            'win_std':    float(prob_win_all.std()),
            'place_mean': float(prob_place_all.mean()),
            'place_std':  float(prob_place_all.std()),
        }

        trained_models[key] = {'win': fn_win, 'place': fn_plc, 'stats': group_stats}
        results.append({
            'グループ': key, 'サンプル数': n,
            '単勝正解率': f'{acc_win:.1%}', '複勝正解率': f'{acc_plc:.1%}'
        })
        n_all = len(df_g_all)
        print(f"[{i:3d}/{len(target_groups)}] {key:25s} 全{n_all:5,}件→学習{n:5,}件  単勝:{acc_win:.1%}  複勝:{acc_plc:.1%}")

    # 5. サブモデル情報を保存
    submodel_info = {
        'features': features,
        'models':   trained_models,
        'grouping': {
            'axes': ['芝ダ', '距離帯', 'クラスグループ'],
            '距離帯': {'短距離': '〜1400m', 'マイル': '1500〜1800m', '中距離': '1900〜2200m', '長距離': '2300m〜'},
            'クラスグループ': {'新馬': 1, '未勝利': 2, '1勝': 3, '2勝': 4, '3勝以上': '5〜9(3勝/OP/G3/G2/G1)'},
            '障害': '障害(クラス_rank=NaN)は除外',
        }
    }
    info_path = os.path.join(model_dir, 'submodel_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(submodel_info, f, ensure_ascii=False, indent=2)

    print(f"\n=== サブモデル学習完了 ===")
    print(f"学習済みモデル数: {len(trained_models)}グループ × 2種 = {len(trained_models)*2}個")
    print(f"保存先: {model_dir}")
    print(f"モデル情報: {info_path}\n")
    if results:
        print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    main()
