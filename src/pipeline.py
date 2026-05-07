# -*- coding: utf-8 -*-
"""
自動パイプライン (horse_racing_ai_v2)

フルパイプライン:
  fetch → features → (train) → predict → CSV

使い方:
  python src/pipeline.py --predict 20260503   # 特定日を予測
  python src/pipeline.py --predict 20260503 --rebuild-features  # 特徴量再生成
  python src/pipeline.py --retrain            # モデル再学習のみ
  python src/pipeline.py --fetch-only         # 最新データ取得のみ
  python src/pipeline.py --full               # fetch + features + retrain + predict(今週末)
"""
import sys, io, os, argparse, subprocess
from datetime import datetime, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON     = sys.executable


def run(cmd, check=True):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check, cwd=BASE_DIR)
    return result.returncode == 0


def upcoming_weekends(n_days=14):
    """今日から n_days 以内の土日を YYYYMMDD リストで返す。"""
    today = datetime.today()
    dates = []
    for i in range(n_days):
        d = today + timedelta(days=i)
        if d.weekday() in (5, 6):  # 土=5, 日=6
            dates.append(d.strftime('%Y%m%d'))
    return dates


def main():
    ap = argparse.ArgumentParser(description='horse_racing_ai_v2 パイプライン')
    ap.add_argument('--predict',           metavar='YYYYMMDD', help='予測日')
    ap.add_argument('--rebuild-features',  action='store_true', help='特徴量を再生成してから予測')
    ap.add_argument('--retrain',           action='store_true', help='モデル再学習')
    ap.add_argument('--fetch-only',        action='store_true', help='fetch だけ実行')
    ap.add_argument('--full',              action='store_true', help='全ステップ実行')
    ap.add_argument('--from-date',         default=None,        help='fetch 開始日 (デフォルト: 今年1月)')
    args = ap.parse_args()

    today_str = datetime.today().strftime('%Y%m%d')
    from_date = args.from_date or (today_str[:4] + '0101')

    # ── fetch ──────────────────────────────────────
    if args.fetch_only or args.full:
        ok = run([PYTHON, 'src/fetch.py',
                  '--from', from_date,
                  '--incremental'])
        if not ok:
            print("fetch 失敗。ターゲットFrontierが起動しているか確認してください。")
            sys.exit(1)

    # ── features ───────────────────────────────────
    feat_file = os.path.join(BASE_DIR, 'data', 'processed', 'features.parquet')
    need_feat = (
        args.full or
        args.rebuild_features or
        not os.path.exists(feat_file)
    )
    if need_feat:
        run([PYTHON, 'src/features.py'])

    # ── retrain ────────────────────────────────────
    model_info = os.path.join(BASE_DIR, 'models', 'model_info.json')
    need_train = (
        args.full or
        args.retrain or
        not os.path.exists(model_info)
    )
    if need_train:
        run([PYTHON, 'src/train.py'])

    # ── predict ────────────────────────────────────
    if args.predict:
        dates = [args.predict]
    elif args.full:
        dates = upcoming_weekends(14)
    else:
        dates = []

    if not dates and not args.fetch_only and not args.retrain and not args.full:
        ap.print_help()
        return

    for date_str in dates:
        card_paths = [
            os.path.join(BASE_DIR, 'data', 'raw', 'cards', f'{date_str}.csv'),
            os.path.join(BASE_DIR, 'data', 'raw', 'cards', 'jvlink', f'{date_str}.csv'),
        ]
        if not any(os.path.exists(p) for p in card_paths):
            print(f"\n{date_str}: 出馬表なし → jvlink_card_builder.py を実行してください")
            print(f"  python src/jvlink/jvlink_card_builder.py --date {date_str}")
            continue

        run([PYTHON, 'src/predict.py', date_str])

    print(f"\n{'='*50}")
    print(f"パイプライン完了")
    if dates:
        out_dir = os.path.join(BASE_DIR, 'data', 'predictions')
        print(f"予測CSV: {out_dir}/")
        for d in dates:
            p = os.path.join(out_dir, f'{d}.csv')
            if os.path.exists(p):
                print(f"  ✓ {d}.csv")
            else:
                print(f"  ✗ {d}.csv (なし)")


if __name__ == '__main__':
    main()
