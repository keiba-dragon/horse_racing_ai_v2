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
import re

def extract_venue(kaikai):
    """開催コードから会場文字を抽出（例: '2中2' → '中', '1東2' → '東'）"""
    m = re.search(r'\d+([^\d]+)', str(kaikai))
    return m.group(1) if m else str(kaikai)

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

TRAIN_START = 130105  # 学習開始: 2013/1/5 (master_kihon.csv の先頭)
TRAIN_END   = 201231  # 学習終了: 2020/12/31 → 2021以降は真のOOSテスト

def main():
    print("--- 競馬AI モデル学習処理を開始します（会場×コース別 / 単勝・複勝2モデル）---")
    print(f"学習データ: {TRAIN_START}–{TRAIN_END}  (2021+はOOSホールドアウト)\n")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'src' in os.path.abspath(__file__) else '.'
    input_file = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.csv')
    model_dir  = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    # 1. データ読み込み
    print("加工済みデータを読み込んでいます...")
    df = pd.read_csv(input_file, low_memory=False)
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    # 着順_numは01_make_features.pyで全角対応済み → そのまま使う
    # 旧データ互換: なければ着順から再生成（全角→半角変換付き）
    if '着順_num' in df.columns:
        df['着順_num'] = pd.to_numeric(df['着順_num'], errors='coerce')
    else:
        df['着順_num'] = (
            df['着順'].astype(str)
            .str.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
            .pipe(pd.to_numeric, errors='coerce')
        )
    df = df.dropna(subset=['日付_num', '着順_num'])

    # 目的変数を2種類作る
    df['target_win']   = (df['着順_num'] == 1).astype(int)   # 単勝：1着か
    df['target_place'] = (df['着順_num'] <= 3).astype(int)   # 複勝：3着以内か

    print(f"データ読み込み完了: {len(df):,}件")

    # 2. 会場×コースのグループキーを作成
    df['会場']      = df['開催'].apply(extract_venue)
    df['model_key'] = df['会場'] + '_' + df['距離'].astype(str)

    # 3. 除外する列を定義
    in_race_cols = ['2角', '3角', '4角']
    # 当レースの結果情報（リーク防止）：過去走のシフト列（N走前_*）は残す
    # マージで生じた _x/_y サフィックスも含む
    current_race_result_cols = [
        '走破タイム', '走破タイム_sec',
        '上り3F', 'PCI', 'RPCI', 'RPCI_x', 'RPCI_y',
        'タイム指数', '上り3F_指数',
        'Ave-3F', '平均速度', '-3F平均速度', '上り3F平均速度',
        '上3F地点差', '上3F地点差_x', '上3F地点差_y',
        '単勝配当', '複勝配当',
        '賞金',
        '着差',  # 当レース着差（勝ち馬との時差）→ほぼ着順そのもの
        '好走',  # 当レース後評価
        'PCI3',  # 当レース指標
        # 払戻金（入着馬情報が直接含まれるためリーク）
        '馬連', '馬単', '枠連', '３連複', '３連単',
        'レース印１',  # レース後評価系
    ]
    odds_cols    = ['人気', '前走人気'] + [f'{n}走前_人気' for n in range(1, 11)]
    meta_cols    = [
        '着順', '着順_num', 'target_win', 'target_place',
        '日付', '日付_num', 'レース名', '馬名S', '馬記号', '生年月日',
        'Ｍ', 'Ｃ', '開催', '距離', '会場', 'model_key',
        '市場取引価格(万/最終)', '取引市場(最終)',
        '前走日付',  # 日付自体は予測力なし（季節性は月・季節特徴量で代替）
    ]
    # 重要度0確認済みの不要列
    # ① 文字列のまま（数値化済み対応列が別途ある）
    string_raw_cols = [
        '騎手', '調教師', '種牡馬', '母父馬', '生産者', '産地',
        '馬主(最新/仮想)', '毛色', '所属', '前騎手',
        '馬場状態', '芝・ダ', '前芝・ダ', 'コース区分',
        '前走開催', '前走レース名',
        '馬印', '馬印2', '馬印3', '馬印4',
        '前走馬印', '前走馬印2', '前走馬印3', '前走馬印4',
        '前走レース印１', '前走B', 'ブリンカー',
        '前走馬場状態', '前好走', '替',
        '脚質グループ',  # 数値化済み（騎手脚質_勝率/複勝率）
        '単勝オッズ',    # EV計算用・訓練に使うとリーク
        '脚質_num',     # 当レースの実走法 → レース後確定のためリーク
        '脚質',         # 文字列（脚質_num として数値化済み）
        '前走脚質',     # 文字列（前走脚質_num として数値化済み）
    ]
    # ② モデルがコース×距離で分割済みのため不要
    course_id_cols = (
        ['今回_距離_m', '今回_surface', '今回_コース種別', '今回_会場']
        + [f'{n}走前_距離'    for n in range(1, 11)]
        + [f'{n}走前_開催'    for n in range(1, 11)]
        + [f'{n}走前_馬場状態' for n in range(1, 11)]
        + ['前走走破タイム_sec']
        # 前走_4角位置 は脚質指標なので除外しない
    )
    zero_importance_cols = string_raw_cols + course_id_cols
    exclude_cols = set(in_race_cols + current_race_result_cols + odds_cols + meta_cols + zero_importance_cols)

    # 4. 数値型の特徴量を抽出
    print("特徴量を選定しています...")
    for col in df.columns:
        if col not in exclude_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # coerce: 変換不可→NaN（LightGBMが自動処理）

    features = [
        col for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    print(f"使用する特徴量: {len(features)}個")
    print(f"  除外（レース中）: {in_race_cols}")
    print(f"  除外（当レース結果）: {len(current_race_result_cols)}列")
    print(f"  除外（人気系）  : {len(odds_cols)}列\n")

    # 5. 会場×コースごとに単勝・複勝の2モデルを学習
    MIN_SAMPLES   = 300
    group_counts  = df['model_key'].value_counts()
    target_groups = group_counts[group_counts >= MIN_SAMPLES].index.tolist()

    print(f"学習対象グループ: {len(target_groups)}種類（{MIN_SAMPLES}件未満はスキップ）\n")

    trained_models = {}  # key → {'win': filename, 'place': filename}
    results = []

    for i, key in enumerate(target_groups, 1):
        df_g_all = df[df['model_key'] == key].sort_values('日付_num').reset_index(drop=True)
        # 2013–2020 のみを学習プールとする（2021+はOOS）
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
        m_win   = train_one(X_train, y_win_tr,   X_test, y_win_te)
        acc_win = accuracy_score(y_win_te, m_win.predict(X_test))
        fn_win  = f'lgb_{key}_win.pkl'
        with open(os.path.join(model_dir, fn_win), 'wb') as f:
            pickle.dump(m_win, f)

        # 複勝モデル
        m_plc   = train_one(X_train, y_plc_tr,   X_test, y_plc_te)
        acc_plc = accuracy_score(y_plc_te, m_plc.predict(X_test))
        fn_plc  = f'lgb_{key}_place.pkl'
        with open(os.path.join(model_dir, fn_plc), 'wb') as f:
            pickle.dump(m_plc, f)

        # コース偏差値の基準統計（学習プール全体の分布）
        prob_win_all   = m_win.predict_proba(df_g[features])[:, 1]
        prob_place_all = m_plc.predict_proba(df_g[features])[:, 1]
        course_stats = {
            'win_mean':   float(prob_win_all.mean()),
            'win_std':    float(prob_win_all.std()),
            'place_mean': float(prob_place_all.mean()),
            'place_std':  float(prob_place_all.std()),
        }

        trained_models[key] = {'win': fn_win, 'place': fn_plc, 'stats': course_stats}
        results.append({
            '会場_コース': key, 'サンプル数': n,
            '単勝正解率': f'{acc_win:.1%}', '複勝正解率': f'{acc_plc:.1%}'
        })
        n_all = len(df_g_all)
        n_oos = len(df_g_all[df_g_all['日付_num'] > TRAIN_END])
        print(f"[{i:3d}/{len(target_groups)}] {key:12s} 全{n_all:5,}件→学習{n:5,}件(OOS:{n_oos})  単勝:{acc_win:.1%}  複勝:{acc_plc:.1%}")

    # 6. モデル情報を保存
    model_info = {
        'features': features,
        'models':   trained_models,
        'exclude':  {'in_race': in_race_cols, 'odds': odds_cols}
    }
    info_path = os.path.join(model_dir, 'model_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    print(f"\n=== 学習完了 ===")
    print(f"学習済みモデル数: {len(trained_models)}グループ × 2種 = {len(trained_models)*2}個")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nモデル情報を保存しました: {info_path}")

if __name__ == "__main__":
    main()
