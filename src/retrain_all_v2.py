# coding: utf-8
"""
retrain_all_v2.py - NaN修正済みparquetで全5セグメントを再訓練

v2.0: NaN修正後の初回再訓練
  - 同じ特徴量・同じL2でparquetを新しくして再訓練
  - 各セグメントの before/after ROI を比較して表示
  - 完了後に models/versions/v2.0/ に最終モデルを保存

実行: python src/retrain_all_v2.py
"""
import sys, os, subprocess, shutil, pickle
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
VER_DIR   = os.path.join(MODEL_DIR, 'versions', 'v2.0')

SCRIPTS = [
    ('ダ長',  'save_new_v1.py'),
    ('ダ短',  'save_da_short_nv3.py'),
    ('芝短',  'save_shiba_short_nv3.py'),
    ('芝中',  'save_shiba_mid_10feat.py'),
    ('芝長',  'save_shiba_long_10feat.py'),
]

# NaN修正前の参照ROI（ハードコード済みのベースライン）
BASELINE = {
    'ダ長': dict(r2323=None,    r25=None,     r26=None,     r2526=-0.1718),
    'ダ短': dict(r2323=+0.0881, r25=-0.3097,  r26=+0.0301,  r2526=-0.2130),
    '芝短': dict(r2323=+0.2705, r25=-0.4250,  r26=-0.1317,  r2526=-0.3663),
    '芝中': dict(r2323=-0.0652, r25=-0.0487,  r26=-0.2729,  r2526=-0.1026),
    '芝長': dict(r2323=+0.0341, r25=+0.0604,  r26=+0.2602,  r2526=+0.1159),
}


def run_script(script_name):
    script_path = os.path.join(BASE_DIR, 'src', script_name)
    print(f'\n{"="*70}')
    print(f'  実行中: {script_name}')
    print(f'{"="*70}')
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        print(f'  [ERROR] {script_name} が失敗 (exit={result.returncode})')
        return False
    return True


def load_roi_from_log(script_name):
    """ログから ROI を読む（暫定：各スクリプトの標準出力から取得）"""
    pass  # 各スクリプトが標準出力にROIを出すのでコンソールで確認


if __name__ == '__main__':
    print('=' * 70)
    print('  v2.0 全セグメント再訓練 (NaN修正済みparquet)')
    print('=' * 70)

    os.makedirs(VER_DIR, exist_ok=True)

    # ベースライン（NaN修正前）表示
    print('\n■ ベースライン ROI (NaN修正前・v2.0_nan_pre):')
    print(f"  {'セグメント':<6}  {'2323':>9}  {'2025':>9}  {'2026':>9}  {'25+26':>9}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")
    for seg, b in BASELINE.items():
        def fmt(v): return f'{v:+.2%}' if v is not None else '   N/A'
        print(f"  {seg:<6}  {fmt(b['r2323']):>9}  {fmt(b['r25']):>9}  {fmt(b['r26']):>9}  {fmt(b['r2526']):>9}")

    print('\n■ 再訓練開始...')
    failed = []
    for seg, script in SCRIPTS:
        ok = run_script(script)
        if not ok:
            failed.append(seg)

    # 完了後にv2.0ディレクトリへ保存
    final_pkl = os.path.join(MODEL_DIR, 'roi_model.pkl')
    v2_pkl    = os.path.join(VER_DIR, 'roi_model.pkl')
    shutil.copy2(final_pkl, v2_pkl)
    print(f'\n■ 新モデル保存: {v2_pkl}')

    if failed:
        print(f'\n[警告] 失敗セグメント: {failed}')
    else:
        print('\n■ 全セグメント再訓練完了')
        print('  → python src/report_10feat_all_segments.py で比較確認してください')
