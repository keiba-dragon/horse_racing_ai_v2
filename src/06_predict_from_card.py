import sys
import io
from datetime import datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os
import pickle
import json
import re
import argparse
import subprocess
from datetime import datetime




def extract_venue(kaikai):
    m = re.search(r'\d+([^\d]+)', str(kaikai))
    return m.group(1) if m else str(kaikai)

def add_pace_features(sub):
    """レース内全馬を使って展開予想特徴量をリアルタイム計算"""
    if '近5走_平均4角位置' not in sub.columns:
        return sub
    pos = pd.to_numeric(sub['近5走_平均4角位置'], errors='coerce')
    n   = max(len(sub), 1)

    sub = sub.copy()
    sub['推定_脚質率']      = (pos / n).clip(0, 1)
    sub['レース内_逃げ馬数'] = float((pos <= 2).sum())
    sub['レース内_先行馬数'] = float((pos <= 5).sum())
    sub['レース内_平均脚質'] = float(pos.mean()) if pos.notna().any() else np.nan
    sub['レース内_脚質std']  = float(pos.std())  if pos.notna().sum() > 1 else 0.0
    sub['推定ペース']        = sub['レース内_先行馬数'] / n
    optimal = 1 - sub['推定ペース']
    sub['展開フィット_v2']   = 1 - (sub['推定_脚質率'] - optimal).abs()
    sub['レース内_相対脚質'] = sub['レース内_平均脚質'] - pos
    if 'コース_先行有利度' in sub.columns:
        course_pace = pd.to_numeric(sub['コース_先行有利度'], errors='coerce') / n
        sub['コース展開マッチ']        = 1 - (sub['推定ペース'] - course_pace).abs()
        sub['展開_コース_脚質フィット'] = sub['展開フィット_v2'] * sub['コース展開マッチ']
    return sub

def get_distance_band(dist):
    m = re.search(r'\d+', str(dist))
    if not m:
        return None
    d = int(m.group())
    if d <= 1400:   return '短距離'
    elif d <= 1800: return 'マイル'
    elif d <= 2200: return '中距離'
    else:           return '長距離'

def get_class_group(class_rank):
    try:
        r = float(class_rank)
    except (ValueError, TypeError):
        return '3勝以上'  # OP等クラス不明は最上位クラスとして扱う
    if np.isnan(r):
        return '3勝以上'  # NaN（一部OP）も同様
    r = int(r)
    if r == 1:   return '新馬'
    elif r == 2: return '未勝利'
    elif r == 3: return '1勝'
    elif r == 4: return '2勝'
    elif r >= 5: return '3勝以上'
    return '3勝以上'


def convert_card_to_base_format(card_path):
    """出馬表形式CSVを基本.csv / 全て.csv 互換形式に変換して返す"""
    try:
        df = pd.read_csv(card_path, encoding='cp932', low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(card_path, encoding='utf-8', low_memory=False)

    # ── 日付変換: 日付S (2026.3.15) → 260315 ──────────
    if '日付S' in df.columns:
        def parse_date_s(s):
            try:
                parts = str(s).replace('/', '.').split('.')
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                return (year - 2000) * 10000 + month * 100 + day
            except Exception:
                return np.nan
        df['日付'] = df['日付S'].apply(parse_date_s)
    elif '日付' in df.columns:
        df['日付'] = pd.to_numeric(df['日付'], errors='coerce')
    else:
        # 日付列なし → ファイル名から推定 (例: 出馬表形式5月2日.csv → 260502)
        import re as _re
        _bn = os.path.basename(card_path)
        _dm = _re.search(r'(\d{1,2})月(\d{1,2})日', _bn)
        if _dm:
            _m, _d = int(_dm.group(1)), int(_dm.group(2))
            _y = datetime.now().year - 2000
            df['日付'] = _y * 10000 + _m * 100 + _d
        else:
            df['日付'] = np.nan

    # ── 開催コード生成: 場 R (京3) → 開催='1京1', Ｒ=3 ─────────────
    if '場 R' in df.columns and '開催' not in df.columns:
        def make_kaikai(bar):
            s = str(bar).strip()
            m = re.match(r'([^\d]+)', s)
            vc = m.group(1) if m else s
            return f'1{vc}1'
        df['開催'] = df['場 R'].apply(make_kaikai)
    if '場 R' in df.columns and 'Ｒ' not in df.columns:
        def extract_r_from_bar(bar):
            m = re.search(r'(\d+)$', str(bar).strip())
            return int(m.group(1)) if m else np.nan
        df['Ｒ'] = df['場 R'].apply(extract_r_from_bar)

    # ── 列名マッピング ─────────────────────────────────
    col_map = {
        '芝ダ':   '芝・ダ',
        '単オッズ': '単勝オッズ',
        '着':     '着順',
        '前着順': '前走着順',
        '前人気': '前走人気',
        '前走騎手': '前騎手',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # 馬名S が無くて馬名だけある場合
    if '馬名S' not in df.columns and '馬名' in df.columns:
        df['馬名S'] = df['馬名']

    # 距離を数値に
    if '距離' in df.columns:
        df['距離'] = pd.to_numeric(df['距離'], errors='coerce')

    # 斤量を数値に（'54㎏' → 54.0）
    if '斤量' in df.columns:
        df['斤量'] = pd.to_numeric(
            df['斤量'].astype(str).str.replace(r'[^\d.]', '', regex=True),
            errors='coerce'
        )

    return df


def merge_results_to_card_extra(base_dir, result_dfs):
    """結果ファイルをcard_extra.csvに反映（feature engineeringは実行しない）"""
    import glob as glob_mod
    card_path = os.path.join(base_dir, 'data', 'raw', 'card_extra.csv')

    if not result_dfs:
        return

    combined = pd.concat(result_dfs, ignore_index=True)
    new_dates = set(pd.to_numeric(combined['日付'], errors='coerce').dropna())

    if os.path.exists(card_path):
        try:
            existing = pd.read_csv(card_path, encoding='utf-8-sig', low_memory=False)
            existing['_d'] = pd.to_numeric(existing['日付'], errors='coerce')
            keep = existing[~existing['_d'].isin(new_dates)].drop(columns=['_d'])
            if len(keep) > 0:
                common = [c for c in keep.columns if c in combined.columns]
                combined = pd.concat([keep[common], combined[common]], ignore_index=True)
        except Exception:
            pass

    combined.to_csv(card_path, index=False, encoding='utf-8-sig')
    print(f"  card_extra.csv 結果反映: {len(result_dfs)}ファイル → {len(combined):,}行")


def run_feature_engineering(base_dir, extra_df):
    """card_extra.csv に変換済みデータを書き、01_make_features.py を実行"""
    card_path = os.path.join(base_dir, 'data', 'raw', 'card_extra.csv')

    # 既存の card_extra.csv と日付単位でマージ（他日付分を消さない）
    new_dates = set(pd.to_numeric(extra_df['日付'], errors='coerce').dropna())
    if os.path.exists(card_path):
        try:
            existing = pd.read_csv(card_path, encoding='utf-8-sig', low_memory=False)
            existing['_d'] = pd.to_numeric(existing['日付'], errors='coerce')
            keep = existing[~existing['_d'].isin(new_dates)].drop(columns=['_d'])
            if len(keep) > 0:
                common = [c for c in keep.columns if c in extra_df.columns]
                extra_df = pd.concat([keep[common], extra_df], ignore_index=True)
                print(f"  既存card_extra: {len(keep)}行を保持（他日付分）")
        except Exception:
            pass

    extra_df.to_csv(card_path, index=False, encoding='utf-8-sig')
    print(f"card_extra.csv 書込: {len(extra_df):,}行")

    script = os.path.join(base_dir, 'src', '01_make_features.py')
    print("特徴量生成中（数分かかります）...")
    import os as _os
    _env = _os.environ.copy()
    _env['KEIBAI_BASE_DIR'] = base_dir
    result = subprocess.run(
        [sys.executable, script],
        cwd=base_dir,
        capture_output=True, text=True, encoding='utf-8',
        env=_env
    )
    if result.returncode != 0:
        print("エラー:", result.stderr[-500:] if result.stderr else "不明")
    else:
        lines = [l for l in result.stdout.strip().split('\n') if l]
        for l in lines[-3:]:
            print(l)
        # Parquet キャッシュを更新
        csv_path = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.csv')
        pq_path  = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.parquet')
        try:
            import pyarrow  # noqa
            print("Parquetキャッシュ更新中...")
            pd.read_csv(csv_path, low_memory=False).to_parquet(pq_path, index=False)
            print("Parquetキャッシュ更新完了")
        except ImportError:
            pass


def predict_date(base_dir, target_date_num, card_df=None):
    """指定日付のレース予測を実行（距離モデル + クラスモデルの両モデル）"""
    import time as _time

    # ── サブモデル情報を読み込む ──────────────────────────
    sub_info_path = os.path.join(base_dir, 'models', 'submodel', 'submodel_info.json')
    sub_features  = None
    sub_models    = {}
    sub_rankers   = {}
    if os.path.exists(sub_info_path):
        with open(sub_info_path, 'r', encoding='utf-8') as f:
            sub_info = json.load(f)
        sub_features = sub_info['features']
        sub_models   = sub_info['models']
        rinfo_sub_path = os.path.join(base_dir, 'models', 'sub_ranker', 'sub_ranker_info.json')
        if os.path.exists(rinfo_sub_path):
            with open(rinfo_sub_path, 'r', encoding='utf-8') as f:
                sub_rankers = json.load(f).get('rankers', {})

    # ── 現行モデル情報を読み込む ──────────────────────────
    cur_info_path = os.path.join(base_dir, 'models', 'model_info.json')
    cur_features  = None
    cur_models    = {}
    cur_rankers   = {}
    if os.path.exists(cur_info_path):
        with open(cur_info_path, 'r', encoding='utf-8') as f:
            cur_info = json.load(f)
        cur_features = cur_info['features']
        cur_models   = cur_info['models']
        rinfo_path   = os.path.join(base_dir, 'models', 'ranker', 'ranker_info.json')
        if os.path.exists(rinfo_path):
            with open(rinfo_path, 'r', encoding='utf-8') as f:
                cur_rankers = json.load(f).get('rankers', {})

    if not cur_models and not sub_models:
        print("エラー: モデルが見つかりません。")
        return None

    # ── clogit (final_model) 読み込み ──────────────────────────────────────
    _clogit_pkg       = None
    _clogit_prepare   = None
    _clogit_softmax   = None
    _clogit_newfeats  = None
    _clogit_pacefeats = None
    _clogit_path = os.path.join(base_dir, 'models', 'final_model.pkl')
    if os.path.exists(_clogit_path):
        try:
            import sys as _sys
            _sys.path.insert(0, os.path.join(base_dir, 'src'))
            # save_conditional_logit はモジュールレベルで sys.stdout を書き換えるので退避・復元する
            _saved_out = _sys.stdout
            _saved_err = _sys.stderr
            from save_conditional_logit import add_new_features as _anf, segment_softmax as _ss, prepare as _prep
            from save_lambdarank_pace import add_pace_features as _apf
            _sys.stdout = _saved_out
            _sys.stderr = _saved_err
            with open(_clogit_path, 'rb') as f:
                _clogit_pkg = pickle.load(f)
            _clogit_prepare   = _prep
            _clogit_softmax   = _ss
            _clogit_newfeats  = _anf
            _clogit_pacefeats = _apf
            print(f'clogitモデル読み込み完了 (OOS ROI={_clogit_pkg.get("total_oos_roi", "?"):.3f})')
        except Exception as _e:
            try:
                print(f'clogit読み込み失敗: {_e}')
            except Exception:
                pass

    # ── 特徴量データを読み込む（Parquet優先 / なければCSV）─────────────
    feat_csv = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.csv')
    feat_pq  = os.path.join(base_dir, 'data', 'processed', 'all_venues_features.parquet')
    t0 = _time.time()
    if os.path.exists(feat_pq) and os.path.getmtime(feat_pq) >= os.path.getmtime(feat_csv):
        # 必要な列だけ読み込んでメモリ節約
        _essential = ['馬名S', '日付', '距離', '開催', '芝・ダ', 'クラス_rank',
                      '単勝オッズ', '馬体重', '馬体重増減', '前距離', '間隔', '性別_num',
                      # N走前シフト補正のために必要な基底列
                      '着順_num', '頭数', '斤量', '4角', '前走着差タイム',
                      'タイム指数', '走破タイム_sec', '上り3F', '上り3F_指数', 'PCI', 'RPCI', '馬番',
                      # add_new_features の中間計算に必要な列
                      '前走斤量', '距離変化_前走']
        _feat_cols = list(set((sub_features or []) + (cur_features or [])))
        _need = set(_essential + _feat_cols)
        # clogit の特徴量列と中間計算列も読み込む（NaN削減）
        if _clogit_pkg is not None:
            for _a in _clogit_pkg['artifacts'].values():
                _need.update(_a['feat_cols'])
            _need.add('前走日付')  # 休養日数計算用
        try:
            import pyarrow.parquet as _pq_mod
            _avail = set(_pq_mod.read_schema(feat_pq).names)
            _load_cols = [c for c in _avail if c in _need]
            df_all = pd.read_parquet(feat_pq, columns=_load_cols)
        except Exception:
            df_all = pd.read_parquet(feat_pq)
        df_all['日付_num'] = pd.to_numeric(df_all['日付'], errors='coerce')
        day = df_all[df_all['日付_num'] == target_date_num].copy()
        print(f"Parquet読み込み: {_time.time()-t0:.1f}秒")
    else:
        chunks = []
        for chunk in pd.read_csv(feat_csv, low_memory=False, chunksize=10000):
            chunk['日付_num'] = pd.to_numeric(chunk['日付'], errors='coerce')
            filtered = chunk[chunk['日付_num'] == target_date_num]
            if not filtered.empty:
                chunks.append(filtered)
        day = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        print(f"CSVチャンク読み込み: {_time.time()-t0:.1f}秒")

    if day.empty:
        if card_df is None:
            print(f"日付 {target_date_num} のデータが見つかりません。")
            return None
        # フォールバック: parquet最新行 + card_df から特徴量構築
        print("フォールバック: parquet最新行 + card_df から特徴量を構築")
        # df_all が未ロードの場合（CSVルート経由）はロード
        if 'df_all' not in locals() and os.path.exists(feat_pq):
            df_all = pd.read_parquet(feat_pq)
            df_all['日付_num'] = pd.to_numeric(df_all['日付'], errors='coerce')
        if 'df_all' not in locals():
            print("エラー: parquetが見つかりません")
            return None
        # 各馬の当日より前の最新行を取得
        df_hist = df_all[df_all['日付_num'] < target_date_num]
        df_latest = (df_hist.sort_values('日付_num')
                     .groupby('馬名S', sort=False).last().reset_index())
        # N走前シフト補正: df_latestの最新行自体が「前走」になるため
        # 1走前_f = latest行のf, 2走前_f = latest行の1走前_f, ... とシフト
        _nmaebo_re = re.compile(r'^(\d+)走前_(.+)$')
        _base_cols = set()
        _max_n_for_base = {}
        for _col in df_latest.columns:
            _m = _nmaebo_re.match(_col)
            if _m:
                _n, _base = int(_m.group(1)), _m.group(2)
                _base_cols.add(_base)
                _max_n_for_base[_base] = max(_max_n_for_base.get(_base, 0), _n)
        for _base in _base_cols:
            _max_n = _max_n_for_base[_base]
            # 大きいNから順に前の値をずらす
            for _n in range(min(_max_n, 9), 0, -1):
                _col_n1 = f'{_n+1}走前_{_base}'
                _col_n  = f'{_n}走前_{_base}'
                if _col_n in df_latest.columns and _col_n1 in df_latest.columns:
                    df_latest[_col_n1] = df_latest[_col_n]
            # 1走前_base = latestの当該レース結果そのもの
            if _base in df_latest.columns and f'1走前_{_base}' in df_latest.columns:
                df_latest[f'1走前_{_base}'] = df_latest[_base]
        # 「前走xxx」列も同様にシフト補正（前走 ≒ 1走前、parquetの最新行は1走前のxxx→今走のxxx）
        _mae_base_map = {
            '前走着順_num':       '着順_num',
            '前走走破タイム_sec':  '走破タイム_sec',
            '前走上り3F':         '上り3F',
            '前走上り3F_指数':    '上り3F_指数',
            '前走馬体重':         '馬体重',
            '前走馬体重増減':     '馬体重増減',
            '前走タイム指数':     'タイム指数',
            '前走脚質_num':       '脚質_num',
            '前走単勝オッズ':     '単勝オッズ',
            '前走PCI':            'PCI',
            '前走RPCI':           'RPCI',
            # 追加: 休養日数・距離変化 の計算に必要
            '前走日付':           '日付',
            '前走_距離_m':        '今回_距離_m',
            '前走_surface':       '今回_surface',
        }
        for _m_col, _b_col in _mae_base_map.items():
            if _m_col in df_latest.columns and _b_col in df_latest.columns:
                df_latest[_m_col] = df_latest[_b_col]
        day = card_df.drop_duplicates('馬名S').copy()
        _clogit_feats_fb = set()
        if _clogit_pkg is not None:
            for _a in _clogit_pkg['artifacts'].values():
                _clogit_feats_fb.update(_a['feat_cols'])
        all_feats_now = list(set((sub_features or []) + (cur_features or []) + list(_clogit_feats_fb)))
        pq_cols = (['馬名S', '日付', '距離', '前走日付'] +
                   [c for c in df_latest.columns
                    if c not in ('馬名S', '日付', '距離', '前走日付') and c in all_feats_now])
        day = day.merge(
            df_latest[list(dict.fromkeys(pq_cols))],
            on='馬名S', how='left', suffixes=('', '_pq'))
        # 前距離・間隔を計算
        def _to_ts(v):
            try:
                v = int(v)
                return pd.Timestamp(2000 + v // 10000, (v // 100) % 100, v % 100)
            except Exception:
                return pd.NaT
        target_ts = _to_ts(target_date_num)
        if '距離_pq' in day.columns:
            day['前距離'] = pd.to_numeric(
                day['距離_pq'].astype(str).str.extract(r'(\d+)')[0],
                errors='coerce')
            day = day.drop(columns=['距離_pq'])
        if '日付_pq' in day.columns:
            day['間隔'] = day['日付_pq'].apply(_to_ts).apply(
                lambda ts: round((target_ts - ts).days / 7) if pd.notna(ts) else np.nan)
            # 104週超（≈2年）は同名別馬の誤マッチの可能性が高いため上限クリップ
            day['間隔'] = pd.to_numeric(day['間隔'], errors='coerce').clip(upper=104)
            day = day.drop(columns=['日付_pq'])
        # card_dfの距離（数値）を "ダ1800" 形式に変換（後続のモデルキー計算と互換）
        surf_col = ('芝・ダ' if '芝・ダ' in day.columns
                    else ('芝ダ' if '芝ダ' in day.columns else None))
        if surf_col and '距離' in day.columns:
            _surf = day[surf_col].astype(str).str.strip()
            _dist = pd.to_numeric(day['距離'], errors='coerce').fillna(0).astype(int).astype(str)
            day['距離'] = _surf + _dist
        # 開催コード生成（card_dfには開催列がない場合あり）
        _VMAP = {'中山': '中', '東京': '東', '阪神': '阪', '中京': '名',
                 '京都': '京', '函館': '函', '新潟': '新', '小倉': '小',
                 '札幌': '札', '福島': '福'}
        if '場所' in day.columns and '開催' not in day.columns:
            _v = day['場所'].map(_VMAP).fillna(day['場所'].str[:1])
            _r = (day['Ｒ'].fillna(1).astype(int).astype(str)
                  if 'Ｒ' in day.columns else '1')
            day['開催'] = '1' + _v + _r
        if '性別' in day.columns and '性別_num' not in day.columns:
            day['性別_num'] = day['性別'].map({'牡': 0, '牝': 1, 'セ': 2}).astype(float)
        n_mae = day['前距離'].notna().sum() if '前距離' in day.columns else 0
        n_kan = day['間隔'].notna().sum() if '間隔' in day.columns else 0
        print(f"フォールバック構築: {len(day)}頭 / 前距離{n_mae}頭 / 間隔{n_kan}頭")
        # _pq 列のNaNフォールバック（card_dfが欠損 → df_latest値で補完）
        for _pq_col in [c for c in day.columns if c.endswith('_pq')]:
            _base_col = _pq_col[:-3]
            if _base_col in day.columns:
                day[_base_col] = day[_base_col].fillna(day[_pq_col])
            day = day.drop(columns=[_pq_col])

        # 今日のレースのクラス_rank をカードデータから補正（fallback時バグ修正）
        # parquetの直近過去レースのクラスが残るため、今日の出走クラスで上書きする
        if 'クラス' in day.columns:
            _cls_map_fb = {
                '新馬': 1, '未勝利': 2,
                '１勝': 3, '1勝': 3,
                '２勝': 4, '2勝': 4,
                '３勝': 5, '3勝': 5,
                'OP': 6, 'オープン': 6, 'L': 6,
                'G3': 7, 'G2': 8, 'G1': 9,
            }
            def _cls_to_rank_fb(val):
                import unicodedata as _ud
                s = _ud.normalize('NFKC', str(val))  # 全角→半角（Ｇ１→G1 等）
                for k, v in _cls_map_fb.items():
                    if k in s:
                        return v
                return np.nan
            _today_cls_rank = day['クラス'].map(_cls_to_rank_fb)
            if _today_cls_rank.notna().any():
                _prev_cls_rank = (day['クラス_rank'].copy()
                                  if 'クラス_rank' in day.columns
                                  else pd.Series(np.nan, index=day.index))
                day['クラス_rank'] = _today_cls_rank
                if 'クラス変化' in day.columns:
                    day['クラス変化'] = _today_cls_rank - _prev_cls_rank

    # 今回_surface / 今回_距離_m をカード情報から直接補完（フォールバック共通）
    if '今回_surface' not in day.columns or day['今回_surface'].isna().all():
        _surf_col = '芝・ダ' if '芝・ダ' in day.columns else ('芝ダ' if '芝ダ' in day.columns else None)
        if _surf_col:
            day['今回_surface'] = day[_surf_col].astype(str).str.strip().map({'芝': 1.0, 'ダ': 0.0, '障': 2.0})
    if '今回_距離_m' not in day.columns or day['今回_距離_m'].isna().all():
        day['今回_距離_m'] = pd.to_numeric(
            day['距離'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')

    # 馬体重・斤量パッチ: card_dfにあれば上書き
    if card_df is not None:
        _weight_cols = [c for c in ['馬体重', '馬体重増減', '斤量'] if c in card_df.columns]
        if _weight_cols:
            _w = card_df[['馬名S'] + _weight_cols].copy()
            for c in _weight_cols:
                _w[c] = pd.to_numeric(
                    _w[c].astype(str).str.replace(r'[^\d.]', '', regex=True),
                    errors='coerce'
                )
            _w = _w.dropna(subset=_weight_cols, how='all')
            if not _w.empty:
                for c in _weight_cols:
                    if c in day.columns:
                        day = day.drop(columns=[c])
                day = day.merge(_w.drop_duplicates('馬名S'), on='馬名S', how='left')
                filled = {c: day[c].notna().sum() for c in _weight_cols if c in day.columns}
                print(f"馬体重・斤量パッチ適用: {filled}")

    # 開催列がNaNの場合、card_dfの「場 R」列でパッチ
    if '開催' not in day.columns or day['開催'].isna().all():
        if card_df is not None and '場 R' in card_df.columns:
            _kaikai = card_df[['馬名S', '場 R']].drop_duplicates('馬名S').rename(columns={'場 R': '_kaikai_tmp'})
            day = day.merge(_kaikai, on='馬名S', how='left')
            day['開催'] = day['_kaikai_tmp'].fillna(day.get('開催', ''))
            day = day.drop(columns=['_kaikai_tmp'])
            print(f"開催列パッチ: {day['開催'].notna().sum()}頭")

    # ── モデルキー計算 ──
    day['会場']       = day['開催'].apply(extract_venue)
    day['cur_key']    = day['会場'] + '_' + day['距離'].astype(str)
    day['_surface']   = day['芝・ダ'].astype(str).str.strip() if '芝・ダ' in day.columns else 'None'
    day['_dist_band'] = day['距離'].apply(get_distance_band)
    mask_da_ml = (day['_surface'] == 'ダ') & (day['_dist_band'].isin(['中距離', '長距離']))
    day.loc[mask_da_ml, '_dist_band'] = '中長距離'
    day['_cls_group'] = day['クラス_rank'].apply(get_class_group) if 'クラス_rank' in day.columns else None
    day['sub_key']    = (day['_surface'].astype(str) + '_' +
                         day['_dist_band'].astype(str) + '_' +
                         day['_cls_group'].astype(str))

    if '単勝オッズ' in day.columns:
        day['単勝オッズ'] = pd.to_numeric(day['単勝オッズ'], errors='coerce')

    all_feats = list(set((sub_features or []) + (cur_features or [])))
    for col in all_feats:
        if col in day.columns:
            day[col] = pd.to_numeric(day[col], errors='coerce')

    print(f"\n日付: {target_date_num}  /  {len(day)}頭\n")

    # ── モデルを一括プリロード ──
    t0 = _time.time()
    model_cache  = {}
    ranker_cache = {}
    cur_model_cache  = {}
    cur_ranker_cache = {}
    for sk in day['sub_key'].dropna().unique():
        if sk in sub_models:
            win_path = os.path.join(base_dir, 'models', 'submodel', sub_models[sk]['win'])
            if os.path.exists(win_path):
                with open(win_path, 'rb') as f:
                    m = pickle.load(f)
                model_cache[sk] = (m, m.booster_.feature_name())
        if sk in sub_rankers:
            rpath = os.path.join(base_dir, 'models', 'sub_ranker', sub_rankers[sk])
            if os.path.exists(rpath):
                with open(rpath, 'rb') as f:
                    ranker_cache[sk] = pickle.load(f)
    if cur_features:
        for ck in day['cur_key'].dropna().unique():
            if ck in cur_models:
                win_path = os.path.join(base_dir, 'models', cur_models[ck]['win'])
                if os.path.exists(win_path):
                    with open(win_path, 'rb') as f:
                        m = pickle.load(f)
                    cur_model_cache[ck] = (m, m.booster_.feature_name())
            if ck in cur_rankers:
                rpath = os.path.join(base_dir, 'models', 'ranker', cur_rankers[ck])
                if os.path.exists(rpath):
                    with open(rpath, 'rb') as f:
                        cur_ranker_cache[ck] = pickle.load(f)
    print(f"モデルプリロード: {_time.time()-t0:.1f}秒 "
          f"(subモデル{len(model_cache)}+ランカー{len(ranker_cache)} / "
          f"現行{len(cur_model_cache)}+ランカー{len(cur_ranker_cache)})")

    # ── レース別に予測 ──────────────────────────────────
    race_keys = ['開催']
    if 'Ｒ' in day.columns:
        day['Ｒ'] = pd.to_numeric(day['Ｒ'], errors='coerce')
        race_keys.append('Ｒ')
    if 'レース名' in day.columns and day['レース名'].notna().any():
        race_keys.append('レース名')
    races = day.groupby(race_keys, sort=True, dropna=False).groups

    all_rows = []
    for group_key, idx in races.items():
        kaikai    = group_key[0]
        race_name = group_key[-1] if 'レース名' in day.columns else str(group_key)
        r_num     = group_key[1] if 'Ｒ' in day.columns else ''
        sub       = day.loc[idx].copy()
        sub_key   = sub['sub_key'].iloc[0]
        cur_key   = sub['cur_key'].iloc[0]

        # ── 展開予想特徴量（レース内集計） ──
        sub = add_pace_features(sub)

        # ── サブモデル予測 ──
        sub['sub_prob_win']       = np.nan
        sub['sub_偏差値の差']      = np.nan
        sub['sub_コース偏差値']    = np.nan
        sub['sub_レース内偏差値']  = np.nan
        sub['sub_ランカー順位']    = np.nan
        if sub_key in model_cache:
            m, wf = model_cache[sub_key]
            for c in wf:
                if c not in sub.columns: sub[c] = np.nan
            sub['sub_prob_win'] = m.predict_proba(sub[wf])[:, 1]
            st  = sub_models[sub_key].get('stats', {})
            w_m = st.get('win_mean', sub['sub_prob_win'].mean())
            w_s = st.get('win_std',  sub['sub_prob_win'].std())
            sub['sub_コース偏差値']  = 50 + 10 * (sub['sub_prob_win'] - w_m) / (w_s if w_s > 0 else 1)
            r_m = sub['sub_prob_win'].mean()
            r_s = sub['sub_prob_win'].std()
            sub['sub_レース内偏差値'] = 50 + 10 * (sub['sub_prob_win'] - r_m) / (r_s if r_s > 0 else 1)
            sub['sub_偏差値の差']    = sub['sub_レース内偏差値'] - sub['sub_コース偏差値']
            if sub_key in ranker_cache:
                scores = ranker_cache[sub_key].predict(sub[wf])
                sub['sub_ランカー順位'] = pd.Series(scores, index=sub.index).rank(
                    ascending=False, method='min').astype(int)

        # ── 現行モデル予測 ──
        sub['cur_prob_win']       = np.nan
        sub['cur_ランカー順位']    = np.nan
        sub['cur_偏差値の差']      = np.nan
        sub['cur_コース偏差値']    = np.nan
        sub['cur_レース内偏差値']  = np.nan
        if cur_key in cur_model_cache:
            m, wf = cur_model_cache[cur_key]
            for c in wf:
                if c not in sub.columns: sub[c] = np.nan
            sub['cur_prob_win'] = m.predict_proba(sub[wf])[:, 1]
            st  = cur_models[cur_key].get('stats', {})
            w_m = st.get('win_mean', sub['cur_prob_win'].mean())
            w_s = st.get('win_std',  sub['cur_prob_win'].std())
            sub['cur_コース偏差値']   = 50 + 10 * (sub['cur_prob_win'] - w_m) / (w_s if w_s > 0 else 1)
            r_m = sub['cur_prob_win'].mean()
            r_s = sub['cur_prob_win'].std()
            sub['cur_レース内偏差値'] = 50 + 10 * (sub['cur_prob_win'] - r_m) / (r_s if r_s > 0 else 1)
            sub['cur_偏差値の差']     = sub['cur_レース内偏差値'] - sub['cur_コース偏差値']
            if cur_key in cur_ranker_cache:
                try:
                    scores = cur_ranker_cache[cur_key].predict(sub[cur_features])
                    sub['cur_ランカー順位'] = pd.Series(scores, index=sub.index).rank(
                        ascending=False, method='min').astype(int)
                except Exception:
                    pass

        # ── clogit 予測 ────────────────────────────────────────────────────
        sub['clogit_score']  = np.nan
        sub['clogit_rank']   = np.nan
        sub['clogit_calib']  = np.nan  # 生のisotonic確率（オッズ補正前）
        sub['clogit_factor'] = np.nan  # 市場補正係数（再計算用）
        if _clogit_pkg is not None:
            try:
                _orig_index = sub.index.tolist()
                _s = sub.reset_index(drop=True).copy()
                _s['race_id'] = 'tmp'
                if '着順_num' not in _s.columns:
                    _s['着順_num'] = 0
                # 展開予想特徴量を追加（pace feats はrace_id単位で集計）
                if _clogit_pacefeats is not None:
                    _s = _clogit_pacefeats(_s)
                # 派生特徴量を追加（休養日数・斤量変化・距離変化・クラス変化・近走トレンド）
                _s = _clogit_newfeats(_s)
                # 列名エイリアス補完（training列名 → prediction列名のマッピング）
                if ('前走走破タイム_sec' not in _s.columns or _s['前走走破タイム_sec'].isna().all()) and '1走前_走破タイム_sec' in _s.columns:
                    _s['前走走破タイム_sec'] = _s['1走前_走破タイム_sec']
                if '今回_surface' not in _s.columns or _s['今回_surface'].isna().all():
                    _surf_src = _s.get('_surface', _s.get('芝・ダ', pd.Series(dtype=str)))
                    _s['今回_surface'] = pd.to_numeric(_surf_src.astype(str).str.strip().map({'芝': 1.0, 'ダ': 0.0, '障': 2.0}), errors='coerce')
                if '今回_距離_m' not in _s.columns:
                    _s['今回_距離_m'] = _s['距離'].astype(str).str.extract(r'(\d+)')[0].astype(float) if '距離' in _s.columns else np.nan
                # 間隔の異常値防止（同名別馬誤マッチで100週超が発生しうる）
                if '間隔' in _s.columns:
                    _s['間隔'] = pd.to_numeric(_s['間隔'], errors='coerce').clip(upper=104)
                # 休養日数: 前走日付なければ 間隔(週) × 7 で近似（間隔=当日基準の週数）
                if _s['休養日数'].isna().all() and '間隔' in _s.columns:
                    _s['休養日数'] = (pd.to_numeric(_s['間隔'], errors='coerce') * 7).clip(0, 365)
                # 距離変化_m: 1走前_距離なければ距離変化_前走を使用
                if _s['距離変化_m'].isna().all() and '距離変化_前走' in _s.columns:
                    _s['距離変化_m'] = pd.to_numeric(_s['距離変化_前走'], errors='coerce')
                _surf = str(_s['距離'].iloc[0]).strip()[:1] if '距離' in _s.columns else None
                if _surf not in ('芝', 'ダ'):
                    _sd = _s['芝・ダ'].iloc[0] if '芝・ダ' in _s.columns else ''
                    _surf = '芝' if str(_sd).strip() in ('芝',) else 'ダ'
                if _surf in _clogit_pkg['artifacts']:
                    _art = _clogit_pkg['artifacts'][_surf]
                    # feat_cols の欠損列を NaN で補完
                    for _fc in _art['feat_cols']:
                        if _fc not in _s.columns:
                            _s[_fc] = np.nan
                    # NaN特徴量カウント＋項目名（impute前に計測）
                    _feat_arr = _art['feat_cols']
                    _total_feats = len(_feat_arr)
                    _nan_mask = _s[_feat_arr].isna()
                    _nan_counts = _nan_mask.sum(axis=1).values
                    _nan_names = [
                        ','.join(_feat_arr[j] for j in range(_total_feats) if _nan_mask.iloc[i, j])
                        for i in range(len(_s))
                    ]
                    for _i, _oi in enumerate(_orig_index):
                        sub.loc[_oi, '_nan_count']    = int(_nan_counts[_i])
                        sub.loc[_oi, '_nan_total']    = _total_feats
                        sub.loc[_oi, '_nan_features'] = _nan_names[_i]
                    _X, _, _gs, _n, *_ = _clogit_prepare(
                        _s, _art['feat_cols'],
                        scaler=_art['scaler'], poly2=_art['poly2'],
                        inter_scaler2=_art['inter_scaler2'], top_idx=_art['top_idx'],
                        poly3=None, inter_scaler3=None, top_idx3=None, fit=False)
                    # NaN率が高い馬は線形スコアを0（中立）にしてsoftmaxに含める
                    _high_nan = np.array(_nan_counts) / _total_feats >= 0.5
                    _lin = _X @ _art['coef']
                    _lin[_high_nan] = 0.0
                    _raw   = _clogit_softmax(_lin, _gs, _n)
                    _calib = _art['isotonic'].predict(_raw)
                    _odds  = pd.to_numeric(_s['単勝オッズ'], errors='coerce').values if '単勝オッズ' in _s.columns else np.full(len(_s), np.nan)
                    _mprob = 1.0 / np.clip(_odds, 1.0, None)  # NaN when odds unavailable
                    _cls   = pd.to_numeric(_s.get('クラス_rank', pd.Series([0]*len(_s))), errors='coerce').fillna(0).values
                    _factor = np.where(_cls == 2, _clogit_pkg['factor_maiden'], _clogit_pkg['factor_other'])
                    # オッズ未確定時は calib_prob のみでランキング（市場補正なし）
                    _score = np.where(np.isnan(_mprob), _calib, _calib - _factor * _mprob)
                    for _i, _oi in enumerate(_orig_index):
                        sub.loc[_oi, 'clogit_score']  = _score[_i]
                        sub.loc[_oi, 'clogit_calib']  = _calib[_i]
                        sub.loc[_oi, 'clogit_factor'] = _factor[_i]
                    sub['clogit_rank'] = sub['clogit_score'].rank(ascending=False, method='first')
            except Exception as _ce:
                pass  # clogit失敗時は既存モデルのみ表示

        # ── combo_gap 計算（1位と2位のスコア差）──
        for score_col, gap_col in [('cur_コース偏差値', 'cur_gap'), ('sub_コース偏差値', 'sub_gap')]:
            sc2 = sub[score_col].dropna().sort_values(ascending=False).values
            sub[gap_col] = (sc2[0] - sc2[1]) if len(sc2) >= 2 else np.nan
        sub['combo_gap'] = sub['cur_gap'].fillna(0) + sub['sub_gap'].fillna(0)

        # ターミナル表示
        venue    = extract_venue(kaikai)
        shiba_da = sub['芝・ダ'].iloc[0] if '芝・ダ' in sub.columns else ''
        kyori    = sub['距離'].iloc[0] if '距離' in sub.columns else ''
        print(f"{'='*70}")
        print(f"  {venue} {r_num}R　{race_name}  [{shiba_da}{kyori}m / {sub_key}]")
        print(f"{'='*70}")
        disp = sub[['馬名S']].copy()
        disp['現行_差']      = sub['cur_偏差値の差'].map(lambda x: f'{x:+.1f}' if pd.notna(x) else '-')
        disp['現行_ランカー'] = sub['cur_ランカー順位'].map(lambda x: f'{int(x)}位' if pd.notna(x) else '-')
        disp['sub_差']       = sub['sub_偏差値の差'].map(lambda x: f'{x:+.1f}' if pd.notna(x) else 'モデルなし')
        disp['sub_ランカー']  = sub['sub_ランカー順位'].map(lambda x: f'{int(x)}位' if pd.notna(x) else '-')
        disp['clogit']       = sub['clogit_rank'].map(lambda x: f'★{int(x)}位' if pd.notna(x) and x == 1 else (f'{int(x)}位' if pd.notna(x) else '-'))
        if '単勝オッズ' in sub.columns:
            disp['オッズ'] = sub['単勝オッズ'].map(lambda x: f'{x:.1f}' if pd.notna(x) else '-')
        print(disp.to_string(index=False))
        print()

        sub['レース名'] = race_name
        all_rows.append(sub)

    if not all_rows:
        print("対応モデルが見つかりませんでした。")
        return None

    return pd.concat(all_rows, ignore_index=True)


def generate_html(result, card_df, target_date_num, out_path):
    """両モデル予測結果をHTMLレポートとして出力（横向き印刷対応）"""

    # ROI統計JSON読み込み
    _roi_stats_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'models', 'roi_stats.json')
    _roi_stats = {}
    if os.path.exists(_roi_stats_path):
        with open(_roi_stats_path, encoding='utf-8') as _f:
            _roi_stats = json.load(_f)
    _roi_cur = _roi_stats.get('distance_model', {})
    _roi_sub = _roi_stats.get('class_model', {})

    def roi_line(stats_dict, key):
        s = stats_dict.get(key)
        if not s:
            return ''
        def r(v): return f'{v:+.1%}' if v is not None else 'N/A'
        n = s.get('n_total', 0)
        return (f"2012テスト {n:,}頭　"
                f"ROI　◎{r(s['diff20']['roi'])}({s['diff20']['n']}頭)　"
                f"〇{r(s['diff15']['roi'])}({s['diff15']['n']}頭)　"
                f"▲{r(s['diff10']['roi'])}({s['diff10']['n']}頭)　"
                f"ランカー1位{r(s['ranker1']['roi'])}({s['ranker1']['n']}頭)")

    d = str(target_date_num)
    date_str = f"20{d[0:2]}/{d[2:4]}/{d[4:6]}" if len(d) == 6 else str(target_date_num)

    # ── カードデータをマージ ──
    card_map = {'枠番':'dc_枠番','馬番':'dc_馬番','性':'dc_性','齢':'dc_齢',
                '性別':'dc_性別','年齢':'dc_年齢','騎手':'dc_騎手',
                '調教師':'dc_調教師','斤量':'dc_斤量','馬体重':'dc_馬体重',
                '増減':'dc_増減','前走着順':'dc_前走着順','前走人気':'dc_前走人気',
                '単勝オッズ':'dc_単勝オッズ'}
    card_cols = ['馬名S'] + [c for c in card_map if c in card_df.columns]
    card_disp = card_df[card_cols].drop_duplicates('馬名S').rename(
        columns={k:v for k,v in card_map.items() if k in card_df.columns})
    df = result.merge(card_disp, on='馬名S', how='left')

    # ── データ品質アラート ──
    _JRA_VENUES  = {'中山','東京','阪神','中京','京都','函館','新潟','小倉','札幌','福島'}
    _NAR_VENUES  = {'大井','川崎','浦和','船橋','門別','帯広','盛岡','水沢',
                    '金沢','笠松','名古屋','園田','姫路','高知','佐賀','荒尾','福山'}
    _JP_VENUES   = _JRA_VENUES | _NAR_VENUES
    _card_extra = {}
    for _, _cr in card_df.drop_duplicates('馬名S').iterrows():
        _card_extra[_cr['馬名S']] = {
            '前場所': str(_cr.get('前場所', '') if not pd.isna(_cr.get('前場所', '')) else ''),
            '間隔_card': _cr.get('間隔', ''),
        }
    alert_rows = []
    for _, r in df.iterrows():
        horse  = r.get('馬名S', '')
        kaikai = str(r.get('開催', ''))
        r_n    = r.get('Ｒ', '')
        ce     = _card_extra.get(horse, {})
        alerts = []
        # 1. 前場所チェック（地方 or 海外 → parquetにデータなし）
        maesho = ce.get('前場所', '')
        if maesho and maesho not in ('', 'nan', '-', 'None'):
            if maesho in _NAR_VENUES:
                alerts.append(f'前走地方競馬「{maesho}」（前走データ未反映の可能性）')
            elif maesho not in _JRA_VENUES:
                alerts.append(f'前場所「{maesho}」が国内場所リスト外（海外帰り疑い）')
        # 2. 間隔ズレ（出馬表 vs parquet計算）
        kan_card = ce.get('間隔_card', '')
        try:
            kan_c = float(str(kan_card).replace('週','').strip())
            kan_p = float(r.get('間隔', np.nan))
            if pd.notna(kan_p) and abs(kan_c - kan_p) > 10:
                alerts.append(f'間隔: 出馬表{int(kan_c)}週 vs 算出{int(kan_p)}週（差{abs(int(kan_c)-int(kan_p))}週）')
        except (ValueError, TypeError):
            pass
        # 3. 重要特徴量欠損（キャリア5走以上なのに近5走平均着順がNaN）
        if pd.isna(r.get('近5走_平均着順')):
            try:
                career = int(float(r.get('キャリア', 0) or 0))
                if career > 5:
                    alerts.append(f'近5走_平均着順が欠損（キャリア{career}走）')
            except (ValueError, TypeError):
                pass
        if alerts:
            venue_disp = extract_venue(kaikai)
            try:  r_disp = int(r_n)
            except: r_disp = r_n
            alert_rows.append({
                'レース': f'{venue_disp}{r_disp}R',
                '馬名': horse,
                'アラート': ' ／ '.join(alerts),
            })

    if alert_rows:
        _arows_html = ''.join(
            f'<tr><td style="white-space:nowrap">{a["レース"]}</td>'
            f'<td style="white-space:nowrap;font-weight:bold">{a["馬名"]}</td>'
            f'<td style="text-align:left">{a["アラート"]}</td></tr>'
            for a in alert_rows
        )
        alert_html = f"""
<div style="page-break-inside:avoid;margin:12px 4px 4px;border:2px solid #e67e22;border-radius:6px;background:#fff8f0;padding:8px 12px">
  <div style="font-size:12px;font-weight:bold;color:#d35400;margin-bottom:6px">⚠ データ品質アラート（最終目視確認）</div>
  <table style="font-size:9px;width:100%;border-collapse:collapse">
    <thead><tr>
      <th style="background:#e67e22;color:white;padding:3px 6px;text-align:left">レース</th>
      <th style="background:#e67e22;color:white;padding:3px 6px;text-align:left">馬名</th>
      <th style="background:#e67e22;color:white;padding:3px 6px;text-align:left">内容</th>
    </tr></thead>
    <tbody>{_arows_html}</tbody>
  </table>
</div>"""
    else:
        alert_html = ''

    # ── 各種フラグ ──
    cur_diff  = pd.to_numeric(df['cur_偏差値の差'], errors='coerce')
    sub_diff  = pd.to_numeric(df['sub_偏差値の差'], errors='coerce')
    combo_gap = pd.to_numeric(df.get('combo_gap'), errors='coerce') if 'combo_gap' in df.columns else pd.Series(np.nan, index=df.index)
    cur_r     = pd.to_numeric(df['cur_ランカー順位'], errors='coerce')
    sub_r     = pd.to_numeric(df['sub_ランカー順位'], errors='coerce')
    cur_r1    = (cur_r == 1)
    sub_r1    = (sub_r == 1)
    both_r1   = cur_r1 & sub_r1
    star      = (cur_r <= 3) & (sub_r <= 3) & ~both_r1  # 片方が2か3
    # オッズ（あれば使う、なければフィルターなし）
    _odds = (pd.to_numeric(df['dc_単勝オッズ'], errors='coerce')
             if 'dc_単勝オッズ' in df.columns
             else pd.Series(np.nan, index=df.index, dtype=float))
    if _odds.isna().all() and '単勝オッズ' in df.columns:
        _odds = pd.to_numeric(df['単勝オッズ'], errors='coerce')
    odds_ok3  = _odds.isna() | (_odds >= 3)   # NaN = フィルターなし扱い
    odds_ok5  = _odds.isna() | (_odds >= 5)
    # 新印マスク（2026-03-28 改訂版）
    # 激熱: 両Rnk=1 & cur_diff≥10 & sub_diff≥10 & odds≥5  推奨: 単1,000円  ROI+253%
    mask_gekiatu = both_r1 & (cur_diff >= 10) & (sub_diff >= 10) & odds_ok5
    # 〇: 両Rnk=1 & sub_diff≥10 & odds≥3 & ~激熱  推奨: 単300円  ROI-24%
    mask_maru2   = both_r1 & (sub_diff >= 10) & odds_ok3 & ~mask_gekiatu
    # ▲: 両Rnk≤2（片方が2）& sd≥10 & odds≥5  推奨: 単500円  ROI+91%
    mask_diamond = (cur_r <= 2) & (sub_r <= 2) & ~both_r1 & (sub_diff >= 10) & odds_ok5
    # ☆: 両Rnk≤3（片方が3）& sd≥10 & odds≥5  推奨: 単200円  ROI-26%
    mask_hoshi   = star & ~((cur_r <= 2) & (sub_r <= 2)) & odds_ok5 & (sub_diff >= 10)
    def mark_of(i):
        if mask_gekiatu.iloc[i]:  return '激熱'
        if mask_maru2.iloc[i]:    return '〇'
        if mask_diamond.iloc[i]:  return '▲'
        if mask_hoshi.iloc[i]:    return '☆'
        return ''
    df['_印'] = [mark_of(i) for i in range(len(df))]

    # ── ※マスク（能力ありだがオッズNG → レース前要確認）──
    _odds_known = _odds.notna()
    # 激熱※: 激熱能力あり & odds<5（〇に降格または無印になっている馬）
    mask_gekiatu_nomi  = both_r1 & (cur_diff >= 10) & (sub_diff >= 10) & _odds_known & (_odds < 5)
    # 〇※: 〇能力あり(激熱能力なし) & odds<3（無印になっている馬）
    mask_maru_nomi     = both_r1 & (sub_diff >= 10) & ~(cur_diff >= 10) & _odds_known & (_odds < 3)
    # ▲※: ▲能力あり & odds<5
    mask_diamond_nomi  = (cur_r <= 2) & (sub_r <= 2) & ~both_r1 & (sub_diff >= 10) & _odds_known & (_odds < 5)
    # ☆※: ☆能力あり & odds<5
    mask_hoshi_nomi    = star & ~((cur_r <= 2) & (sub_r <= 2)) & (sub_diff >= 10) & _odds_known & (_odds < 5)
    mask_any_nomi = mask_gekiatu_nomi | mask_maru_nomi | mask_diamond_nomi | mask_hoshi_nomi

    def nomi_mark_of(i):
        if mask_gekiatu_nomi.iloc[i]: return '激熱※'
        if mask_maru_nomi.iloc[i]:    return '〇※'
        if mask_diamond_nomi.iloc[i]: return '▲※'
        if mask_hoshi_nomi.iloc[i]:   return '☆※'
        return ''
    df['_nomi_印'] = [nomi_mark_of(i) for i in range(len(df))]

    # 旧フラグ（行ハイライト用に残す）
    cur_nota  = cur_diff >= 10
    sub_nota  = sub_diff >= 10
    both_nota = cur_nota & sub_nota

    def fmt(val, fmt_str='+.1f', default='-'):
        try:
            return format(float(val), fmt_str) if pd.notna(val) else default
        except: return default

    def rank_str(val):
        try: return f"{int(float(val))}位" if pd.notna(val) else '-'
        except: return '-'

    def seir(row):
        if 'dc_性別' in row.index and pd.notna(row.get('dc_性別')):
            return f"{row.get('dc_性別','')}{row.get('dc_年齢','')}"
        return f"{row.get('dc_性','')}{row.get('dc_齢','')}"

    # ── CSS ──
    css = """<style>
      @page { size: A4 landscape; margin: 8mm; }
      body { font-family:'Yu Gothic','Hiragino Sans',sans-serif; font-size:11px; margin:6px; background:#f0f0f0; }
      h1 { font-size:16px; margin:6px 0; color:#1a252f; }
      h2 { font-size:13px; background:#2c3e50; color:white; padding:5px 10px; margin:14px 0 0; border-radius:4px 4px 0 0; }
      .race-info { font-size:10px; color:#555; padding:2px 10px; background:#dde4ea; margin:0; }
      .model-stats { font-size:8.5px; padding:1px 10px; background:#f7f9f9; margin:0; }
      .page { page-break-after:always; break-after:page; padding:4px; }
      .page:last-child { page-break-after:avoid; }
      table { border-collapse:collapse; width:100%; background:white; margin-bottom:4px; }
      th { background:#2c3e50; color:white; padding:3px 5px; font-size:9px; text-align:center; white-space:nowrap; border:1px solid #1a252f; }
      td { padding:2px 4px; text-align:center; white-space:nowrap; font-size:10px; border:1px solid #ccc; }
      td.name { text-align:left; font-weight:bold; font-size:11px; }
      td.sub-detail { font-size:9px; color:#555; text-align:left; }
      /* 行ハイライト */
      .both  td { background:#fde8e8 !important; }
      .cur15 td { background:#fff9c4 !important; }
      .sub15 td { background:#e8f4fd !important; }
      /* 数値色 */
      .hi { color:#c0392b; font-weight:bold; }
      .md { color:#e67e22; font-weight:bold; }
      /* セクション見出し */
      .sec-cur  h2 { background:#1e8449; }
      .sec-cur     { background:#f0faf4; }
      .sec-sub  h2 { background:#1f618d; }
      .sec-sub     { background:#eaf4fb; }
      .sec-both h2 { background:#6c3483; }
      .sec-both    { background:#f5eef8; }
      .sec-hot  h2 { background:#b7410e; }
      .sec-hot     { background:#fef5f0; }
      /* 印バッジ */
      .mark { display:inline-block; font-size:13px; font-weight:bold; min-width:28px; text-align:center; }
      .mark-gekiatu  { color:#b7410e; }
      .mark-maru2    { color:#27ae60; }
      .mark-sankaku  { color:#7d3c98; font-weight:bold; }
      .mark-hoshi    { color:#1f618d; }
      /* バッジ */
      .badge { display:inline-block; font-size:8px; font-weight:bold; border-radius:3px;
               padding:1px 3px; margin:1px; }
      .b-both { background:#8e44ad; color:white; }
      .b-cur  { background:#27ae60; color:white; }
      .b-sub  { background:#2980b9; color:white; }
      input.memo { width:120px; border:1px solid #ccc; border-radius:3px; padding:1px 3px; font-size:10px; }
      /* 両モデル一致カード */
      .both-list { display:flex; flex-wrap:wrap; gap:8px; padding:4px; background:white; }
      .both-card { border:2px solid #8e44ad; border-radius:6px; padding:6px 10px; min-width:220px; max-width:320px; background:#fdf5ff; }
      .both-race { font-size:9px; color:#666; margin-bottom:3px; }
      .both-horse { display:flex; align-items:baseline; gap:6px; margin-bottom:4px; }
      .both-banum { font-size:10px; color:#888; }
      .both-name  { font-size:15px; font-weight:bold; color:#1a1a2e; }
      .both-jockey { font-size:10px; color:#555; }
      .both-odds  { font-size:10px; color:#e67e22; margin-left:auto; }
      .both-eval  { display:flex; flex-direction:column; gap:2px; }
      .both-cur   { font-size:10px; color:#27ae60; font-weight:bold; }
      .both-sub   { font-size:10px; color:#2980b9; font-weight:bold; }
      @media print { input { display:none; } body { margin:0; } h2 { margin-top:0; } }
    </style>"""

    def diff_cell(val):
        s = fmt(val)
        try:
            v = float(val)
            if v >= 20: return f'<td class="hi">{s}</td>'
            if v >= 15: return f'<td class="md">{s}</td>'
        except: pass
        return f'<td>{s}</td>'

    # ── サマリーテーブル共通 ──
    def summary_table(mask, show_cur=True, show_sub=True):
        rows = df[mask].copy()
        if rows.empty: return '<p style="padding:6px;background:white;">該当なし</p>'
        race_keys_s = [c for c in ['開催','Ｒ','レース名'] if c in rows.columns]
        rows = rows.sort_values(race_keys_s)
        html = ['<table><thead><tr>']
        html.append('<th>レース</th><th>馬番</th><th>馬名</th><th>騎手</th>')
        if show_cur: html.append('<th>会場×距離<br>ランカー</th><th>会場×距離<br>偏差値差</th>')
        if show_sub: html.append('<th>クラス<br>ランカー</th><th>クラス<br>偏差値差</th>')
        html.append('<th>単勝%</th><th>オッズ</th></tr></thead><tbody>')
        for _, r in rows.iterrows():
            venue    = extract_venue(r.get('開催',''))
            r_n      = int(r['Ｒ']) if 'Ｒ' in rows.columns and pd.notna(r.get('Ｒ')) else ''
            race_lbl = f"{venue}{r_n}R {r.get('レース名','')}"
            try:
                banum = int(float(r['dc_馬番'])) if pd.notna(r.get('dc_馬番')) else '-'
            except (ValueError, TypeError):
                banum = r.get('dc_馬番', '-')
            bc = 'both' if both_nota[r.name] else ('cur15' if cur_nota[r.name] else 'sub15')
            row_html = f'<tr class="{bc}"><td>{race_lbl}</td><td>{banum}</td>'
            row_html += f'<td class="name">{r.get("馬名S","")}</td>'
            row_html += f'<td>{r.get("dc_騎手", r.get("騎手",""))}</td>'
            if show_cur:
                row_html += f'<td>{rank_str(r.get("cur_ランカー順位"))}</td>'
                row_html += diff_cell(r.get('cur_偏差値の差'))
            if show_sub:
                row_html += f'<td>{rank_str(r.get("sub_ランカー順位"))}</td>'
                row_html += diff_cell(r.get('sub_偏差値の差'))
            prob = r.get('cur_prob_win') if show_cur and pd.notna(r.get('cur_prob_win')) else r.get('sub_prob_win')
            row_html += f'<td>{fmt(prob,":.1%","-") if prob else "-"}</td>'
            _odds = r.get('dc_単勝オッズ') if pd.notna(r.get('dc_単勝オッズ')) else r.get('単勝オッズ')
            row_html += f'<td>{fmt(_odds,":.1f")}</td></tr>'
            html.append(row_html)
        html.append('</tbody></table>')
        return ''.join(html)

    GREEN = {'◎': '#1a6e3c', '〇': '#27ae60', '▲': '#58d68d', 'th': '#1e8449', 'bg': '#eafaf1'}
    BLUE  = {'◎': '#1a3f6e', '〇': '#2471a3', '▲': '#5dade2', 'th': '#1f618d', 'bg': '#eaf4fb'}

    def tier_table(label, mask, diff_col, rank_col, palette):
        horses = df[mask].copy()
        if horses.empty:
            return f'<p style="padding:4px 8px;color:#999;font-size:10px">{label}: 該当なし</p>'
        rk = [c for c in ['開催','Ｒ','レース名'] if c in horses.columns]
        horses = horses.sort_values(rk)
        tier_color = palette.get(label[0], palette['▲'])
        rows_html = []
        for _, r in horses.iterrows():
            venue  = extract_venue(r.get('開催',''))
            r_n    = int(r['Ｒ']) if 'Ｒ' in horses.columns and pd.notna(r.get('Ｒ')) else ''
            r_name = r.get('レース名','')
            try:
                banum = int(float(r['dc_馬番'])) if pd.notna(r.get('dc_馬番')) else '-'
            except (ValueError, TypeError):
                banum = r.get('dc_馬番', '-')
            jockey = r.get('dc_騎手', r.get('騎手',''))
            diff_v = r.get(diff_col)
            rank_v = r.get(rank_col)
            odds_v = r.get('dc_単勝オッズ') if pd.notna(r.get('dc_単勝オッズ')) else r.get('単勝オッズ')
            rows_html.append(
                f'<tr>'
                f'<td style="text-align:left;padding:2px 6px">{venue} {r_n}R　{r_name}</td>'
                f'<td>{banum}番</td>'
                f'<td style="text-align:left;font-weight:bold;font-size:12px">{r.get("馬名S","")}</td>'
                f'<td>{jockey}</td>'
                f'<td style="font-weight:bold;color:{palette["th"]}">{rank_str(rank_v)}</td>'
                f'{diff_cell(diff_v)}'
                f'<td>{fmt(odds_v, ".1f")}</td>'
                f'</tr>'
            )
        return (f'<div style="font-size:12px;font-weight:bold;color:{tier_color};'
                f'padding:4px 8px;margin:10px 0 2px;border-left:5px solid {tier_color};'
                f'background:{palette["bg"]}">'
                f'{label}（{len(horses)}頭）</div>'
                f'<table><thead><tr style="background:{palette["th"]}">'
                f'<th style="text-align:left">レース</th><th>馬番</th><th style="text-align:left">馬名</th>'
                f'<th>騎手</th><th>ランカー順位</th><th>偏差値差</th><th>オッズ</th>'
                f'</tr></thead><tbody>{"".join(rows_html)}</tbody></table>')

    # ── 新印カード生成 ──
    def make_mark_cards(mask):
        rows = df[mask].copy()
        if rows.empty:
            return '<p style="padding:8px;background:white;color:#999;font-size:10px">該当なし</p>'
        rk = [c for c in ['開催','Ｒ','レース名'] if c in rows.columns]
        rows = rows.sort_values(rk)
        items = []
        for _, r in rows.iterrows():
            venue  = extract_venue(r.get('開催',''))
            r_n    = int(r['Ｒ']) if 'Ｒ' in rows.columns and pd.notna(r.get('Ｒ')) else ''
            r_name = r.get('レース名','')
            try:
                banum = int(float(r['dc_馬番'])) if pd.notna(r.get('dc_馬番')) else '-'
            except (ValueError, TypeError):
                banum = r.get('dc_馬番', '-')
            horse   = r.get('馬名S','')
            jockey  = r.get('dc_騎手', r.get('騎手',''))
            odds_v  = r.get('dc_単勝オッズ') if pd.notna(r.get('dc_単勝オッズ')) else r.get('単勝オッズ')
            mark    = r.get('_印','')
            cgap    = fmt(r.get('combo_gap'), '.1f')
            sd_v    = fmt(r.get('sub_偏差値の差'), '+.1f')
            cr_s    = rank_str(r.get('cur_ランカー順位'))
            sr_s    = rank_str(r.get('sub_ランカー順位'))
            mark_cls = {'激熱':'mark-gekiatu','〇':'mark-maru2','▲':'mark-sankaku','☆':'mark-hoshi'}.get(mark,'')
            items.append(f"""
            <div class="both-card">
              <div class="both-race">{venue} {r_n}R　{r_name}</div>
              <div class="both-horse">
                <span class="both-banum">{banum}番</span>
                <span class="both-name">{horse}</span>
                <span class="both-jockey">{jockey}</span>
                <span class="both-odds">オッズ {fmt(odds_v,'.1f')}</span>
              </div>
              <div class="both-eval">
                <span class="mark {mark_cls}" style="font-size:18px">{mark}</span>
                <span style="font-size:10px;color:#555">　距離{cr_s} / クラス{sr_s} / gap{cgap} / sd{sd_v}</span>
              </div>
            </div>""")
        return '<div class="both-list">' + ''.join(items) + '</div>'

    # 新印テーブル（一覧用）
    def mark_table(mask, header_color='#2c3e50'):
        rows = df[mask].copy()
        if rows.empty:
            return '<p style="padding:4px 8px;color:#999;font-size:10px">該当なし</p>'
        rk = [c for c in ['開催','Ｒ','レース名'] if c in rows.columns]
        rows = rows.sort_values(rk)
        rows_html = []
        for _, r in rows.iterrows():
            venue  = extract_venue(r.get('開催',''))
            r_n    = int(r['Ｒ']) if 'Ｒ' in rows.columns and pd.notna(r.get('Ｒ')) else ''
            try:
                banum = int(float(r['dc_馬番'])) if pd.notna(r.get('dc_馬番')) else '-'
            except (ValueError, TypeError):
                banum = r.get('dc_馬番', '-')
            jockey  = r.get('dc_騎手', r.get('騎手',''))
            odds_v  = r.get('dc_単勝オッズ') if pd.notna(r.get('dc_単勝オッズ')) else r.get('単勝オッズ')
            mark    = r.get('_印','')
            mark_cls = {'激熱':'mark-gekiatu','〇':'mark-maru2','▲':'mark-sankaku','☆':'mark-hoshi'}.get(mark,'')
            rows_html.append(
                f'<tr>'
                f'<td style="text-align:left">{venue} {r_n}R　{r.get("レース名","")}</td>'
                f'<td>{banum}番</td>'
                f'<td style="text-align:left;font-weight:bold;font-size:12px">{r.get("馬名S","")}</td>'
                f'<td>{jockey}</td>'
                f'<td class="mark {mark_cls}">{mark}</td>'
                f'<td>{rank_str(r.get("cur_ランカー順位"))}</td>'
                f'<td>{rank_str(r.get("sub_ランカー順位"))}</td>'
                f'<td>{fmt(r.get("combo_gap"),".1f")}</td>'
                f'<td>{fmt(r.get("sub_偏差値の差"),"+.1f")}</td>'
                f'<td>{fmt(odds_v,".1f")}</td>'
                f'</tr>'
            )
        return (f'<table><thead><tr style="background:{header_color}">'
                f'<th style="text-align:left">レース</th><th>馬番</th><th style="text-align:left">馬名</th>'
                f'<th>騎手</th><th>印</th><th>距離<br>ランカー</th><th>クラス<br>ランカー</th>'
                f'<th>combo<br>gap</th><th>sub<br>diff</th><th>オッズ</th>'
                f'</tr></thead><tbody>{"".join(rows_html)}</tbody></table>')

    # ── ※一覧テーブル ──
    def nomi_table(label, mask, mark_color):
        rows = df[mask].copy()
        if rows.empty:
            return f'<p style="padding:4px 8px;color:#999;font-size:10px">{label}: 該当なし</p>'
        rk = [c for c in ['開催','Ｒ','レース名'] if c in rows.columns]
        rows = rows.sort_values(rk)
        rows_html = []
        for _, r in rows.iterrows():
            venue   = extract_venue(r.get('開催',''))
            r_n     = int(r['Ｒ']) if 'Ｒ' in rows.columns and pd.notna(r.get('Ｒ')) else ''
            r_name  = r.get('レース名','')
            try:
                banum = int(float(r['dc_馬番'])) if pd.notna(r.get('dc_馬番')) else '-'
            except (ValueError, TypeError):
                banum = r.get('dc_馬番', '-')
            jockey  = r.get('dc_騎手', r.get('騎手',''))
            odds_v  = r.get('dc_単勝オッズ') if pd.notna(r.get('dc_単勝オッズ')) else r.get('単勝オッズ')
            cur_d   = fmt(r.get('cur_偏差値の差'), '+.1f')
            sub_d   = fmt(r.get('sub_偏差値の差'), '+.1f')
            cr_s    = rank_str(r.get('cur_ランカー順位'))
            sr_s    = rank_str(r.get('sub_ランカー順位'))
            cgap    = fmt(r.get('combo_gap'), '.1f')
            current_mark = r.get('_印', '')  # 現在の印（〇に降格している場合など）
            rows_html.append(
                f'<tr>'
                f'<td style="text-align:left">{venue} {r_n}R　{r_name}</td>'
                f'<td>{banum}番</td>'
                f'<td style="text-align:left;font-weight:bold;font-size:13px">{r.get("馬名S","")}</td>'
                f'<td>{jockey}</td>'
                f'<td style="font-weight:bold;color:{mark_color};font-size:14px">{label}</td>'
                f'<td style="font-size:10px;color:#888">{current_mark if current_mark else "無印"}</td>'
                f'<td>{cr_s}</td><td>{sr_s}</td>'
                f'<td>{fmt(odds_v, ".1f")}</td>'
                f'<td>{cur_d}</td><td>{sub_d}</td>'
                f'<td>{cgap}</td>'
                f'</tr>'
            )
        return (f'<div style="font-size:13px;font-weight:bold;color:{mark_color};'
                f'padding:4px 8px;margin:10px 0 2px;border-left:5px solid {mark_color}">'
                f'{label}（{len(rows)}頭）</div>'
                f'<table><thead><tr style="background:#555;color:white">'
                f'<th style="text-align:left">レース</th><th>馬番</th>'
                f'<th style="text-align:left">馬名</th><th>騎手</th>'
                f'<th>能力印</th><th>現在の印</th>'
                f'<th>距離<br>Rnk</th><th>クラス<br>Rnk</th>'
                f'<th>オッズ</th><th>cur<br>diff</th><th>sub<br>diff</th><th>combo<br>gap</th>'
                f'</tr></thead><tbody>{"".join(rows_html)}</tbody></table>')

    # ── ※ページ HTML ──
    _nomi_has_any = mask_any_nomi.any()
    if _nomi_has_any:
        _nomi_body = (
            nomi_table('激熱※', mask_gekiatu_nomi, '#b7410e') +
            nomi_table('〇※',   mask_maru_nomi,    '#27ae60') +
            nomi_table('▲※',   mask_diamond_nomi, '#7d3c98') +
            nomi_table('☆※',   mask_hoshi_nomi,   '#1f618d')
        )
    else:
        _nomi_body = '<p style="padding:12px;background:white;color:#888">該当なし</p>'

    page_nomi = f"""<div class="page" style="background:#fff8f0">
      <h2 style="background:#c0392b">【オッズ要確認】レース前に再チェック推奨</h2>
      <p style="font-size:10px;color:#c0392b;font-weight:bold;margin:4px 8px 8px">
        ⚠ 以下の馬は能力条件（ランカー順位・偏差値差）は満たしていますが、<br>
        　 現在のオッズが購入基準未満です。レース直前にオッズが上昇した場合は購入を検討してください。
      </p>
      <p style="font-size:9px;color:#888;margin:0 8px 8px">
        激熱※: 両Rnk=1 &amp; cur_diff≥10 &amp; sub_diff≥10 だがオッズ&lt;5　／
        〇※: 両Rnk=1 &amp; sub_diff≥10 だがオッズ&lt;3　／
        ▲※: 両Rnk≤2 &amp; sub_diff≥10 だがオッズ&lt;5　／
        ☆※: 両Rnk≤3 &amp; sub_diff≥10 だがオッズ&lt;5
      </p>
      {_nomi_body}
    </div>"""

    # ── Page1: 激熱専用 ──
    page1 = f"""<div class="page sec-hot">
      <h1>競馬AI予測レポート　{date_str}</h1>
      <h2>【激熱】両Rnk=1 &amp; cur_diff≥10 &amp; sub_diff≥10 &amp; odds≥5　単1,000円</h2>
      <p style="font-size:9px;color:#666;margin:2px 0 6px">※ 距離・クラス両モデルで偏差値差10以上＆5倍以上　単勝ROI+253%（2026年1〜3月実績）</p>
      {make_mark_cards(mask_gekiatu)}
    </div>"""

    # ── Page2: 〇/▲/☆ 一覧 ──
    page2 = f"""<div class="page sec-both">
      <h2>【印一覧】〇 両Rnk=1 sd≥10 / ▲ 両Rnk≤2 sd≥10 / ☆ 両Rnk≤3 sd≥10</h2>
      <p style="font-size:9px;color:#666;margin:2px 0 4px">
        〇: 単300円 ROI-24% / ▲: 単500円 ROI+91% / ☆: 単200円 ROI-26%　（2026年1〜3月実績）
      </p>
      {mark_table(mask_maru2 | mask_diamond | mask_hoshi)}
    </div>"""

    # ── Page3 & 4: 両モデル一致 詳細（参考）──
    def make_both_cards(mask):
        rows = df[mask].copy()
        if rows.empty:
            return '<p style="padding:8px;background:white;color:#999;font-size:10px">該当なし</p>'
        rk = [c for c in ['開催','Ｒ','レース名'] if c in rows.columns]
        rows = rows.sort_values(rk)
        items = []
        for _, r in rows.iterrows():
            venue  = extract_venue(r.get('開催',''))
            r_n    = int(r['Ｒ']) if 'Ｒ' in rows.columns and pd.notna(r.get('Ｒ')) else ''
            r_name = r.get('レース名','')
            try:
                banum = int(float(r['dc_馬番'])) if pd.notna(r.get('dc_馬番')) else '-'
            except (ValueError, TypeError):
                banum = r.get('dc_馬番', '-')
            horse  = r.get('馬名S','')
            jockey = r.get('dc_騎手', r.get('騎手',''))
            odds   = fmt(r.get('dc_単勝オッズ') if pd.notna(r.get('dc_単勝オッズ')) else r.get('単勝オッズ'), '.1f')

            cdv = r.get('cur_偏差値の差')
            cur_tier = '◎' if pd.notna(cdv) and cdv >= 20 else ('〇' if pd.notna(cdv) and cdv >= 15 else ('▲' if pd.notna(cdv) and cdv >= 10 else ''))
            cur_lbl_parts = [cur_tier] if cur_tier else []
            if pd.notna(r.get('cur_ランカー順位')):
                cur_lbl_parts.append(f"ランカー{int(r['cur_ランカー順位'])}位")
            if pd.notna(cdv):
                cur_lbl_parts.append(f"偏差値差{fmt(cdv)}")
            cur_lbl = '　'.join(cur_lbl_parts) if cur_lbl_parts else '-'

            sdv = r.get('sub_偏差値の差')
            sub_tier = '◎' if pd.notna(sdv) and sdv >= 20 else ('〇' if pd.notna(sdv) and sdv >= 15 else ('▲' if pd.notna(sdv) and sdv >= 10 else ''))
            sub_lbl_parts = [sub_tier] if sub_tier else []
            if pd.notna(r.get('sub_ランカー順位')):
                sub_lbl_parts.append(f"ランカー{int(r['sub_ランカー順位'])}位")
            if pd.notna(sdv):
                sub_lbl_parts.append(f"クラス差{fmt(sdv)}")
            sub_lbl = '　'.join(sub_lbl_parts) if sub_lbl_parts else '-'

            items.append(f"""
            <div class="both-card">
              <div class="both-race">{venue} {r_n}R　{r_name}</div>
              <div class="both-horse">
                <span class="both-banum">{banum}番</span>
                <span class="both-name">{horse}</span>
                <span class="both-jockey">{jockey}</span>
                <span class="both-odds">オッズ {odds}</span>
              </div>
              <div class="both-eval">
                <span class="both-cur">┗ 距離モデル：{cur_lbl}</span>
                <span class="both-sub">┗ クラスモデル：{sub_lbl}</span>
              </div>
            </div>""")
        return '<div class="both-list">' + ''.join(items) + '</div>'

    # ── レース別詳細（発走時刻順）──
    def horse_tier(row):
        """新印体系: _印列から取得"""
        return row.get('_印', '')

    race_keys_html = [c for c in ['開催','Ｒ','レース名'] if c in df.columns]

    # 発走時刻でソート用のマップを card_df から作成
    race_time_map = {}
    if '発走時刻' in card_df.columns and 'Ｒ' in card_df.columns:
        for _, row in card_df[['開催', 'Ｒ', '発走時刻']].drop_duplicates().iterrows():
            try:
                r_int = int(float(row['Ｒ']))
            except (ValueError, TypeError):
                continue
            t = str(row.get('発走時刻', '') or '').replace(':', '').replace('：', '')
            race_time_map[(str(row['開催']), r_int)] = t

    def race_sort_key(gk):
        kaikai = str(gk[0])
        try:
            r_int = int(float(gk[1])) if len(gk) > 1 else 0
        except (ValueError, TypeError):
            r_int = 0
        t = race_time_map.get((kaikai, r_int), '9999')
        return (t, r_int)

    sorted_race_groups = sorted(
        df.groupby(race_keys_html, sort=False).groups.items(),
        key=lambda x: race_sort_key(x[0])
    )

    race_pages = []
    for gk, idx in sorted_race_groups:
        grp = df.loc[idx].copy()
        if 'dc_馬番' in grp.columns:
            grp = grp.sort_values('dc_馬番', na_position='last')
        kaikai    = gk[0]
        race_name = gk[-1] if 'レース名' in df.columns else ''
        r_n       = int(gk[1]) if 'Ｒ' in df.columns and pd.notna(gk[1]) else ''
        venue     = extract_venue(kaikai)
        shiba_da  = grp['芝・ダ'].iloc[0] if '芝・ダ' in grp.columns else ''
        _kyori_raw = grp['距離'].iloc[0] if '距離' in grp.columns else ''
        try: kyori = int(re.search(r'\d+', str(_kyori_raw)).group())
        except: kyori = _kyori_raw
        sub_key   = grp['sub_key'].iloc[0] if 'sub_key' in grp.columns else ''
        cur_key_r = grp['cur_key'].iloc[0] if 'cur_key' in grp.columns else ''
        sk_parts  = str(sub_key).split('_')
        cls_label = '・'.join(sk_parts[1:]) if len(sk_parts) >= 3 else sub_key
        surface_label = {'芝': '芝', 'ダ': 'ダート'}.get(str(shiba_da).strip(), str(shiba_da))
        cur_roi_txt = roi_line(_roi_cur, cur_key_r)
        sub_roi_txt = roi_line(_roi_sub, sub_key)

        # combo_gap はレース単位で1つの値（全馬同値）→ ヘッダーに表示
        _cgap_val = grp['combo_gap'].dropna().iloc[0] if ('combo_gap' in grp.columns and grp['combo_gap'].notna().any()) else None
        cgap_header = (f'<span style="font-size:9px;color:#888;margin-left:10px">'
                       f'1強度(combo gap): <b>{_cgap_val:.1f}</b>'
                       f'<span style="font-size:8px;color:#aaa"> ※値が大きいほど1強感が強い</span>'
                       f'</span>') if _cgap_val is not None else ''

        rows_html = []
        for _, r in grp.iterrows():
            mark     = horse_tier(r)
            nomi_mark = r.get('_nomi_印', '')
            mark_cls = {'激熱':'mark-gekiatu','〇':'mark-maru2','▲':'mark-sankaku','☆':'mark-hoshi'}.get(mark,'')
            if mark == '激熱':  row_bg = 'background:#fde8e8'
            elif mark == '〇':   row_bg = 'background:#eafaf1'
            elif mark == '▲':   row_bg = 'background:#f5eef8'
            elif mark == '☆':   row_bg = 'background:#eaf4fb'
            elif nomi_mark:     row_bg = 'background:#fff8f0'
            else:               row_bg = ''

            bw   = r.get('dc_馬体重', '')
            bw_d = r.get('dc_増減', '')
            bw_valid = bw not in ('', None) and str(bw) not in ('nan', '-', 'None') and pd.notna(bw)
            if not bw_valid:
                taiju = '-'
            else:
                try: taiju = f"{bw}({'+' if float(bw_d)>=0 else ''}{int(float(bw_d))})" if pd.notna(bw_d) and str(bw_d) not in ('', 'nan') else str(bw)
                except: taiju = str(bw)

            odds_v = r.get('dc_単勝オッズ') if pd.notna(r.get('dc_単勝オッズ')) else r.get('単勝オッズ')

            # 詳細行（距離+クラス 1行）
            detail = (
                f'<span style="color:#1e8449">距離</span> '
                f'{rank_str(r.get("cur_ランカー順位"))} '
                f'絶対{fmt(r.get("cur_コース偏差値"),".1f")} '
                f'内{fmt(r.get("cur_レース内偏差値"),".1f")} '
                f'差{fmt(r.get("cur_偏差値の差"),"+.1f")} '
                f'勝{fmt(r.get("cur_prob_win"),".1%")}'
                f'　／　'
                f'<span style="color:#2471a3">クラス</span> '
                f'{rank_str(r.get("sub_ランカー順位"))} '
                f'絶対{fmt(r.get("sub_コース偏差値"),".1f")} '
                f'内{fmt(r.get("sub_レース内偏差値"),".1f")} '
                f'差{fmt(r.get("sub_偏差値の差"),"+.1f")} '
                f'勝{fmt(r.get("sub_prob_win"),".1%")}'
            )

            # combo_gapは距離ランカー1位の馬のみ表示
            is_rank1 = (pd.notna(r.get('cur_ランカー順位')) and int(r.get('cur_ランカー順位', 99)) == 1)
            cgap_disp = fmt(r.get('combo_gap'), '.1f') if is_rank1 else ''
            # ※印（再チェック馬）
            nomi_disp = (f'<span style="font-size:10px;font-weight:bold;color:#c0392b">{nomi_mark}</span>'
                         if nomi_mark else '')
            # メイン行
            rows_html.append(
                f'<tr style="{row_bg}">'
                f'<td>{r.get("dc_枠番","-")}</td>'
                f'<td>{r.get("dc_馬番","-")}</td>'
                f'<td class="{mark_cls}" style="font-size:13px;font-weight:bold;min-width:28px">{mark}{nomi_disp}</td>'
                f'<td style="text-align:left;font-weight:bold;white-space:nowrap">{r.get("馬名S","")}'
                f' <span style="font-weight:normal;color:#888;font-size:8px">{seir(r)}</span></td>'
                f'<td style="white-space:nowrap;font-size:9px">{r.get("dc_騎手", r.get("騎手",""))}'
                f' <span style="color:#888;font-size:8px">{fmt(r.get("dc_斤量"), ".0f") + "kg" if pd.notna(r.get("dc_斤量")) else ""}</span></td>'
                f'<td style="white-space:nowrap">{taiju}</td>'
                f'<td>{fmt(odds_v, ".1f")}</td>'
                f'<td>{rank_str(r.get("cur_ランカー順位"))}</td>'
                f'<td>{rank_str(r.get("sub_ランカー順位"))}</td>'
                f'<td>{fmt(r.get("sub_偏差値の差"),"+.1f")}</td>'
                f'</tr>'
                # 詳細行（7px・全列結合）
                f'<tr style="{row_bg}">'
                f'<td colspan="10" style="font-size:7px;color:#666;text-align:left;padding:1px 6px;border-top:none">{detail}</td>'
                f'</tr>'
            )

        race_pages.append(f"""<div class="page">
          <h2>{venue} {r_n}R　{race_name}</h2>
          <div class="race-info">{surface_label} {kyori}m　／　{cls_label}{cgap_header}</div>
          <table style="font-size:10px">
            <thead><tr>
              <th>枠</th><th>馬番</th><th>印</th><th style="text-align:left">馬名/性齢</th><th>騎手/斤量</th><th>馬体重</th><th>オッズ</th>
              <th style="color:#58d68d">距離<br>ランカー</th>
              <th style="color:#5dade2">クラス<br>ランカー</th>
              <th>sub<br>diff</th>
            </tr></thead>
            <tbody>{''.join(rows_html)}</tbody>
          </table>
        </div>""")

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>競馬AI予測 {date_str}</title>
  {css}
</head>
<body>
  {page1}
  {page_nomi}
  {page2}
  {''.join(race_pages)}
  {alert_html}
</body>
</html>"""

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"\nHTML出力: {out_path}")

    # ── Chrome ヘッドレスで PDF 生成 ──
    pdf_path = out_path.replace('.html', '.pdf')
    chrome_candidates = [
        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
    ]
    chrome = next((c for c in chrome_candidates if os.path.exists(c)), None)
    if chrome:
        try:
            result_pdf = subprocess.run([
                chrome,
                '--headless=new',
                '--disable-gpu',
                '--no-sandbox',
                '--no-pdf-header-footer',
                f'--print-to-pdf={pdf_path}',
                f'file:///{out_path.replace(chr(92), "/")}',
            ], capture_output=True, timeout=60)
            if os.path.exists(pdf_path):
                print(f"PDF出力: {pdf_path}")
            else:
                print(f"PDF生成失敗: {result_pdf.stderr.decode('utf-8','ignore')[-200:]}")
        except Exception as e:
            print(f"PDF生成エラー: {e}")
    else:
        print("Chrome未検出のためPDF生成スキップ")


def get_base_cache_name(card_file):
    """オッズ確認用などのsuffixを除去したキャッシュ基本名を返す"""
    stem = os.path.splitext(os.path.basename(card_file))[0]
    stem = re.sub(r'オッズ確認用.*$', '', stem).rstrip('_').rstrip()
    return stem


def load_odds_json(odds_path):
    """オッズjsonを読み込む（なければ空dict）"""
    if os.path.exists(odds_path):
        with open(odds_path, encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_odds_json(card_df, odds_path):
    """card_dfの単勝オッズをjsonに保存"""
    if '単勝オッズ' in card_df.columns:
        odds = {str(k): float(v) for k, v in
                card_df.dropna(subset=['単勝オッズ']).set_index('馬名S')['単勝オッズ'].items()
                if pd.notna(v)}
        with open(odds_path, 'w', encoding='utf-8') as f:
            json.dump(odds, f, ensure_ascii=False)
        print(f"オッズ保存: {len(odds)}頭 → {odds_path}")


def apply_odds_to_card(card_df, odds_dict):
    """オッズdictをcard_dfの単勝オッズ列に適用"""
    if not odds_dict:
        return card_df
    card_df = card_df.copy()
    card_df['単勝オッズ'] = card_df['馬名S'].map(
        lambda x: odds_dict.get(str(x), card_df.loc[card_df['馬名S']==x, '単勝オッズ'].values[0]
                  if '単勝オッズ' in card_df.columns and (card_df['馬名S']==x).any() else float('nan'))
    )
    return card_df


def main():
    parser = argparse.ArgumentParser(description='出馬表形式CSVから予測を実行')
    parser.add_argument('card_file', help='出馬表形式CSVのパス（例: data/raw/出馬表形式3月15日.csv）')
    parser.add_argument('--no-rebuild', action='store_true', help='特徴量再生成をスキップ')
    parser.add_argument('--odds-only', action='store_true', help='予測キャッシュを再利用してオッズのみ更新')
    parser.add_argument('--no-html', action='store_true', help='HTML出力をスキップ')
    parser.add_argument('--html-only', action='store_true', help='キャッシュから予測結果を読み込んでHTMLのみ生成')
    parser.add_argument('--html-dir', default=r'G:\マイドライブ\競馬AI\予想レポート', help='HTML保存先フォルダ')
    parser.add_argument('--no-results', action='store_true', help='data/raw/results/の結果ファイル自動読み込みをスキップ')
    args = parser.parse_args()

    base_dir = os.environ.get('KEIBAI_BASE_DIR') or (os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if 'src' in os.path.abspath(__file__) else '.')

    print(f"=== 競馬AI 出馬表予測 ===")
    print(f"入力: {args.card_file}")

    # キャッシュパス（予測pkl + オッズjson に分離）
    cache_dir = os.path.join(base_dir, 'data', 'raw', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    base_name   = get_base_cache_name(args.card_file)
    cache_path  = os.path.join(cache_dir, base_name + '.cache.pkl')
    odds_path   = os.path.join(cache_dir, base_name + '.odds.json')

    if args.odds_only:
        # ── オッズのみ更新（予測キャッシュ再利用）──
        if not os.path.exists(cache_path):
            print(f"[ERROR] 予測キャッシュが見つかりません: {cache_path}")
            print("  先に通常実行してキャッシュを生成してください。")
            sys.exit(1)
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        result      = cached['result']
        card_df     = cached['card_df']
        target_date = cached['target_date']
        # 新しいCSVからオッズだけ抽出して保存
        new_card = convert_card_to_base_format(args.card_file)
        save_odds_json(new_card, odds_path)
        card_df = apply_odds_to_card(card_df, load_odds_json(odds_path))
        print(f"日付: {target_date}  オッズ更新完了")

    elif args.html_only:
        # ── キャッシュからHTML生成のみ ──
        if not os.path.exists(cache_path):
            print(f"[ERROR] キャッシュが見つかりません: {cache_path}")
            print("  先に通常実行してキャッシュを生成してください。")
            sys.exit(1)
        print(f"キャッシュ読み込み: {cache_path}")
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        result      = cached['result']
        card_df     = cached['card_df']
        target_date = cached['target_date']
        card_df = apply_odds_to_card(card_df, load_odds_json(odds_path))
        print(f"日付: {target_date}  ({len(card_df)}頭)")

    else:
        # 1. 出馬表を変換
        card_df = convert_card_to_base_format(args.card_file)

        dates = card_df['日付'].dropna().unique()
        print(f"日付: {[int(d) for d in sorted(dates)]}")
        print(f"レース: {card_df['レース名'].nunique() if 'レース名' in card_df.columns else '?'}レース / {len(card_df)}頭")

        target_date = int(sorted(dates)[-1])

        # 2. 結果ファイルをcard_extraに先行反映（N走前特徴量の精度向上）
        if not args.no_rebuild and not args.no_results:
            import glob as glob_mod
            result_dir = os.path.join(base_dir, 'data', 'raw', 'results')
            result_dfs = []
            if os.path.exists(result_dir):
                for rf in sorted(glob_mod.glob(os.path.join(result_dir, '*.csv'))):
                    try:
                        rdf = convert_card_to_base_format(rf)
                        rdf_dates = pd.to_numeric(rdf['日付'], errors='coerce')
                        rdf = rdf[rdf_dates < target_date]
                        if len(rdf) > 0:
                            result_dfs.append(rdf)
                            dates_in = sorted(rdf_dates[rdf_dates < target_date].dropna().unique().astype(int))
                            print(f"  結果ファイル読込: {os.path.basename(rf)} 日付={dates_in}")
                    except Exception as e:
                        print(f"  [skip] {os.path.basename(rf)}: {e}")
            if result_dfs:
                merge_results_to_card_extra(base_dir, result_dfs)
                print(f"  → 結果ファイル {len(result_dfs)}件をcard_extraに反映（N走前特徴量補完）")

        # 3. 特徴量再生成
        if not args.no_rebuild:
            run_feature_engineering(base_dir, card_df)
        else:
            print("特徴量再生成スキップ")

        # 4. 予測（両モデル）
        result = predict_date(base_dir, target_date, card_df)

        # キャッシュ保存（予測pkl + オッズjson を分離保存）
        if result is not None:
            existing_predicted_at = ''
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        existing_predicted_at = pickle.load(f).get('predicted_at', '')
                except Exception:
                    pass
            predicted_at = existing_predicted_at or datetime.now().strftime('%H:%M')
            with open(cache_path, 'wb') as f:
                pickle.dump({'result': result, 'card_df': card_df, 'target_date': target_date,
                             'predicted_at': predicted_at}, f)
            save_odds_json(card_df, odds_path)
            print(f"キャッシュ保存: {cache_path}")

    # 4. HTML生成
    if result is not None and not args.no_html:
        html_dir = args.html_dir
        if not os.path.exists(os.path.splitdrive(html_dir)[0] + '\\'):
            html_dir = os.path.join(base_dir, 'output')
        now_str = datetime.now().strftime('%Y%m%d_%H%M')
        out_path = os.path.join(html_dir, f'card_{target_date}_{now_str}.html')
        generate_html(result, card_df, target_date, out_path)


if __name__ == '__main__':
    main()
