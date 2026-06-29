"""
馬券CSV統合スクリプト
- data/tohyo/ に置いた新規 YYYYMMDD_tohyo.csv を all_tohyo.csv に追記する
- all_tohyo.csv が既存の場合: 新規データを追記（重複日付はスキップ）
- all_tohyo.csv がない場合: archiveも含めて全ファイルから再構築

【all_tohyo.csv を消してしまった場合の復元手順】
  git show 2397b9c:data/tohyo/all_tohyo.csv > data/tohyo/all_tohyo.csv
  python data/merge_tohyo.py  # ← archiveの4/4以降も追記される
"""
import pandas as pd
import glob, os, re, shutil, sys

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tohyo_dir   = os.path.join(BASE_DIR, 'data', 'tohyo')
archive_dir = os.path.join(tohyo_dir, 'archive')
out_path    = os.path.join(tohyo_dir, 'all_tohyo.csv')
os.makedirs(archive_dir, exist_ok=True)

def parse_amount(val):
    s = str(val).strip()
    if s in ('', 'nan', '-'):
        return 0
    # 全角・半角スラッシュ対応（フォーメーション: "100／800" → 合計=800）
    if '／' in s or '/' in s:
        parts = re.split(r'[／/]', s)
        for p in reversed(parts):
            p = p.strip().replace(',', '')
            if re.match(r'^\d+$', p):
                return int(p)
        return 0
    try:
        return float(str(s).replace(',', ''))
    except:
        return 0

def read_tohyo_csv(path):
    for enc in ['cp932', 'utf-8-sig', 'utf-8']:
        try:
            df = pd.read_csv(path, encoding=enc)
            if '日付' in df.columns:
                return df
        except Exception:
            pass
    return None

def clean_df(df):
    """数値変換・合計行除去・ソート"""
    df = df[df['日付'].notna()].copy()
    df = df[df['日付'].astype(str).str.match(r'^\d')].copy()
    for col in ['購入金額', '払戻金額', '返還金額']:
        if col in df.columns:
            df[col] = df[col].apply(parse_amount)
    df['日付'] = df['日付'].apply(
        lambda x: int(float(str(x))) if str(x).replace('.', '').isdigit() else None)
    df = df[df['日付'].notna()].copy()
    df['日付'] = df['日付'].astype(int)
    return df

# ── 新規ファイルを収集 ─────────────────────────────────────────
new_files = sorted([
    f for f in glob.glob(os.path.join(tohyo_dir, '*_tohyo.csv'))
    if os.path.basename(f) not in ('all_tohyo.csv', 'video_tohyo_data.csv')
])

# ── all_tohyo.csv の存在確認 ──────────────────────────────────
if os.path.exists(out_path):
    existing = pd.read_csv(out_path, encoding='utf-8-sig')
    existing = clean_df(existing)
    print(f"既存 all_tohyo.csv: {len(existing)}行  最終日: {existing['日付'].max()}")
    # archive内で既存に含まれていない日付のファイルも対象に加える
    exist_dates = set(existing['日付'].unique()) if not existing.empty else set()
    archive_files = sorted(glob.glob(os.path.join(archive_dir, '*_tohyo.csv')))
    extra_archive = []
    for af in archive_files:
        m = re.search(r'(\d{8})_tohyo', os.path.basename(af))
        if m:
            d = int(m.group(1))
            if d not in exist_dates:
                extra_archive.append(af)
    if extra_archive:
        print(f"  archiveに未収録日付あり: {[os.path.basename(f) for f in extra_archive]}")
    source_files = extra_archive + new_files
else:
    # all_tohyo.csv がない → archiveも含めて再構築
    print("all_tohyo.csv が見つかりません。archive + 新規ファイルから再構築します。")
    print("※ 2397b9c以前のデータは git show で復元してください:")
    print("   git show 2397b9c:data/tohyo/all_tohyo.csv > data/tohyo/all_tohyo.csv")
    archive_files = sorted(glob.glob(os.path.join(archive_dir, '*_tohyo.csv')))
    source_files = archive_files + new_files
    existing = pd.DataFrame()

if not source_files:
    print("追加する新規ファイルがありません。")
    if os.path.exists(out_path):
        df = existing
    else:
        sys.exit(0)
else:
    new_dfs = []
    for f in source_files:
        df = read_tohyo_csv(f)
        if df is not None:
            df = clean_df(df)
            new_dfs.append(df)
            print(f"  読込: {os.path.basename(f)} ({len(df)}行)")

    new_data = pd.concat(new_dfs, ignore_index=True) if new_dfs else pd.DataFrame()

    if not existing.empty and not new_data.empty:
        # 既存の日付と重複する分はスキップ
        exist_dates = set(existing['日付'].unique())
        new_dates   = set(new_data['日付'].unique())
        overlap = exist_dates & new_dates
        if overlap:
            print(f"  重複日付をスキップ: {sorted(overlap)}")
            new_data = new_data[~new_data['日付'].isin(overlap)]
        df = pd.concat([existing, new_data], ignore_index=True)
    elif existing.empty:
        df = new_data
    else:
        df = existing

    df = df.sort_values(['日付', '受付番号', '通番']).reset_index(drop=True)

# ── 保存 ────────────────────────────────────────────────────
df.to_csv(out_path, index=False, encoding='utf-8-sig')

buy  = df['購入金額'].sum()
pay  = df['払戻金額'].sum()
ret  = df['返還金額'].sum() if '返還金額' in df.columns else 0
net  = pay + ret - buy
roi  = net / buy * 100 if buy > 0 else 0
dates = sorted(df['日付'].unique())

print(f'\n保存: {out_path}  ({len(df)}行)')
print(f'期間: {dates[0]} 〜 {dates[-1]}  ({len(dates)}日分)')
print(f'購入: {buy:,.0f}円  払戻: {pay:,.0f}円  損益: {net:+,.0f}円  ROI: {roi:+.1f}%')

# ── 新規ファイルをarchiveへ移動 ───────────────────────────────
if new_files:
    print(f'\nアーカイブへ移動 ({len(new_files)}件):')
    for f in new_files:
        fname = os.path.basename(f)
        dest  = os.path.join(archive_dir, fname)
        shutil.move(f, dest)
        print(f'  → archive/{fname}')
