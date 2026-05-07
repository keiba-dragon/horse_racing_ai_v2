# -*- coding: utf-8 -*-
"""
JV-Link ベース週末予測パイプライン
 1. JV-Link から出馬表カードを取得 (jvlink_card_builder)
 2. 各日付について特徴量再生成 + D指標予測 (06_predict_from_card)
 3. D指標新聞 HTML 生成 (d_core/predict/newspaper)

使い方:
  python src/predict_weekend.py                   # 今週末 (今日〜+14日)
  python src/predict_weekend.py --date 20260503   # 特定日のみ
  python src/predict_weekend.py --from 20260502 --to 20260503
  python src/predict_weekend.py --no-rebuild      # 特徴量再生成スキップ
"""
import sys, io, os, subprocess, argparse
from datetime import datetime, timedelta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run(cmd, desc=''):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(
        cmd, cwd=BASE_DIR,
        capture_output=False, text=True, encoding='utf-8'
    )
    if result.returncode != 0:
        print(f"[ERROR] exit={result.returncode}")
    return result.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date',       default=None, help='特定日 YYYYMMDD')
    ap.add_argument('--from',  dest='from_date', default=None)
    ap.add_argument('--to',    dest='to_date',   default=None)
    ap.add_argument('--no-rebuild', action='store_true', help='特徴量再生成スキップ')
    ap.add_argument('--card-only',  action='store_true', help='カード取得のみ')
    ap.add_argument('--no-newspaper', action='store_true', help='新聞HTML生成スキップ')
    args = ap.parse_args()

    today = datetime.now().strftime('%Y%m%d')

    if args.date:
        from_date = args.date
        to_date   = args.date
    elif args.from_date:
        from_date = args.from_date
        to_date   = args.to_date
    else:
        from_date = today
        to_date   = (datetime.now() + timedelta(days=14)).strftime('%Y%m%d')

    # ── Step 1: カード取得 ──────────────────────────────────────────
    card_builder = os.path.join(BASE_DIR, 'src', 'jvlink', 'jvlink_card_builder.py')
    card_dir     = os.path.join(BASE_DIR, 'data', 'raw', 'cards', 'jvlink')

    rc = run(
        [sys.executable, card_builder, '--from', from_date, '--to', to_date],
        f'STEP 1: JV-Link カード取得  {from_date} 〜 {to_date}'
    )
    if rc != 0:
        print("カード取得に失敗しました。ターゲットFrontierが起動しているか確認してください。")
        sys.exit(1)

    if args.card_only:
        print("\nカード取得完了（--card-only のため予測スキップ）")
        return

    # ── Step 2: 日付ごとに予測 ─────────────────────────────────────
    predict_script = os.path.join(BASE_DIR, 'src', '06_predict_from_card.py')
    newspaper_script = os.path.join(BASE_DIR, 'd_core', 'predict', 'newspaper.py')

    # 生成されたカードファイルを列挙
    import glob
    card_files = sorted(glob.glob(os.path.join(card_dir, '????????.csv')))
    target_files = [
        f for f in card_files
        if from_date <= os.path.splitext(os.path.basename(f))[0] <= to_date
    ]

    if not target_files:
        print(f"\n[WARNING] {card_dir} に対象カードCSVが見つかりません")
        sys.exit(1)

    print(f"\n対象カードファイル: {[os.path.basename(f) for f in target_files]}")

    first = True
    for card_path in target_files:
        date_str = os.path.splitext(os.path.basename(card_path))[0]  # '20260502'

        # 予測 (最初の日付のみ特徴量再生成、以降はスキップ)
        predict_args = [sys.executable, predict_script, card_path]
        if args.no_rebuild or not first:
            predict_args.append('--no-rebuild')

        rc = run(predict_args, f'STEP 2: 予測  {date_str}')
        if rc != 0:
            print(f"[WARNING] {date_str} の予測に失敗（続行）")

        # D指標新聞
        if not args.no_newspaper:
            rc = run(
                [sys.executable, newspaper_script, date_str],  # 8桁YYYYMMDD
                f'STEP 3: D指標新聞  {date_str}'
            )

        first = False

    print("\n" + "="*60)
    print("  全日付 処理完了")
    print("="*60)
    print(f"  カード: {card_dir}/")
    print(f"  予測HTML: G:\\マイドライブ\\競馬AI\\予想レポート\\")
    print(f"  D指標新聞: d_core/output/")


if __name__ == '__main__':
    main()
