# coding: utf-8
"""D指標プロジェクト - パス設定"""
import os

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # horse_racing_ai/

MODEL_DIR     = os.path.join(BASE_DIR, 'models_2025')
SUBMODEL_DIR  = os.path.join(MODEL_DIR, 'submodel')
RANKER_DIR    = os.path.join(MODEL_DIR, 'ranker')
SUB_RANKER_DIR= os.path.join(MODEL_DIR, 'submodel_ranker')

PARQUET_PATH  = os.path.join(BASE_DIR, 'data', 'processed', 'all_venues_features.parquet')
PAY_CSV_PATH  = os.path.join(BASE_DIR, 'data', 'raw', 'master', '20260315_2013まで_配当.csv')
CACHE_DIR     = os.path.join(BASE_DIR, 'data', 'raw', 'cache')
RESULT_DIR    = os.path.join(BASE_DIR, 'data', 'raw', 'results')

OUTPUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
