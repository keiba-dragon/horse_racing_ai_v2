# coding: utf-8
"""
実験: レース内定数特徴量を除外して再学習
- 現行モデルには触れない（models/exp_no_const/ に保存）
- save_conditional_logit.py の EXCLUDE を拡張して再実行
"""
import os, sys

# 除外する定数特徴量（レース内で全馬同じ値 → softmaxで打ち消されて無意味）
RACE_CONST_FEATS = {
    '頭数', 'RPCI', '今回_surface', '今回_距離_m', '今回_馬場_num',
    '月', '季節', 'クラス_rank', 'コース_先行有利度',
    'レース内_逃げ馬数', 'レース内_先行馬数', 'レース内_平均脚質',
    'レース内_脚質std', '推定ペース', 'コース展開マッチ', '相手レベル_実力差',
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

# EXCLUDE を拡張してから save_conditional_logit をインポート
import save_lambdarank_pace as _slp
_slp.EXCLUDE = _slp.EXCLUDE | RACE_CONST_FEATS

import save_conditional_logit as _scl

# save_conditional_logit の EXCLUDE も上書き
_scl_exclude = getattr(_scl, 'EXCLUDE', set())

# main を呼ぶ（--out-dir で実験ディレクトリに保存）
out_dir = os.path.join(BASE_DIR, 'models', 'exp_no_const')
os.makedirs(out_dir, exist_ok=True)

sys.argv = ['_exp_no_const_features.py', '--out-dir', out_dir]
print(f'除外定数特徴量: {len(RACE_CONST_FEATS)}列')
print(f'保存先: {out_dir}')
print()

_scl.main()
