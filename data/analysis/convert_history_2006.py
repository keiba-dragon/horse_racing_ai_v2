# -*- coding: utf-8 -*-
"""
netkeibanote「全て形式」CSVを master_kihon.csv 互換形式に変換する。
2006〜2026 + 1995〜2005 の両ファイルを統合して 1995-2012 を出力。

使い方:
  python data/analysis/convert_history_2006.py          # 1995-2012を出力
  python data/analysis/convert_history_2006.py --all    # 1995-2026全部
  python data/analysis/convert_history_2006.py --check  # 先頭3行確認のみ
"""
import sys, io, os, re, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_FILE      = r"C:/Users/tsuch/Downloads/2006.1.5～2026.4.26(7417日間)"
SRC_FILE_1995 = r"C:/Users/tsuch/Downloads/1995.1.5～2005.12.25(4007日間)"
OUT_HIST = os.path.join(BASE_DIR, 'data', 'raw', 'master', 'master_kihon_2006_2012.csv')
OUT_ALL  = os.path.join(BASE_DIR, 'data', 'raw', 'master', 'master_kihon_2006_2026.csv')

def convert_date(s):
    """'2026. 4.26' → '260426' (YYMMDD 2桁年)"""
    s = str(s).strip()
    m = re.match(r'(\d{4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})', s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y % 100:02d}{mo:02d}{d:02d}"
    return np.nan

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all',   action='store_true', help='2006-2026全部を出力')
    ap.add_argument('--check', action='store_true', help='先頭3行確認のみ')
    args = ap.parse_args()

    def load_and_convert(path):
        print(f"読み込み中: {path}")
        d = pd.read_csv(path, encoding='cp932', low_memory=False)
        print(f"  総行数: {len(d):,}  列数: {len(d.columns)}")
        d['日付'] = d['日付(yyyy.mm.dd)'].apply(convert_date)
        d['日付_num'] = pd.to_numeric(d['日付'], errors='coerce')
        print(f"  日付範囲: {d['日付_num'].min()} 〜 {d['日付_num'].max()}")
        return d

    df_2006 = load_and_convert(SRC_FILE)

    # 1995-2005ファイルも存在すれば結合
    if os.path.exists(SRC_FILE_1995):
        df_1995 = load_and_convert(SRC_FILE_1995)
        df = pd.concat([df_2006, df_1995], ignore_index=True)
        print(f"\n両ファイル結合後: {len(df):,}行")
    else:
        df = df_2006
        print("1995-2005ファイルが見つかりません（2006-2026のみ処理）")

    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    print(f"全体日付範囲: {df['日付_num'].min()} 〜 {df['日付_num'].max()}")

    if args.check:
        print("\n先頭3行の主要列:")
        show = ['日付', '開催', 'Ｒ', '馬名S', '着順.1', '芝・ダ', '距離.1', '単勝オッズ', '脚質', '前走日付']
        for c in show:
            if c in df.columns:
                print(f"  {c}: {df[c].head(3).tolist()}")
        return

    # ── 列マッピング: ダウンロードファイル列名 → master_kihon列名 ──
    COL_MAP = {
        '日付':               '日付',           # 変換済み
        '開催':               '開催',
        'Ｒ':                 'Ｒ',
        'レース名.1':         'レース名',
        '馬名S':              '馬名',           # 01_make_featuresがS→S変換済み
        'Ｃ.1':               'Ｃ',
        '性別.1':             '性別',
        '年齢.1':             '年齢',
        '騎手.1':             '騎手',
        '斤量.1':             '斤量',
        '頭数.1':             '頭数',
        '馬番.1':             '馬番',
        '馬印':               '馬印',
        '馬印2':              '馬印2',
        '馬印3':              '馬印3',
        '馬印4':              '馬印4',
        'レース印１':         'レース印１',
        '馬主(最新/仮想)':    '馬主(最新/仮想)',
        '人気.1':             '人気',
        '着順.1':             '着順',
        '芝・ダ':             '芝・ダ',
        '距離.1':             '距離',           # plain number (e.g. 1150)
        'コース区分':         'コース区分',
        '馬場状態.1':         '馬場状態',
        '賞金':               '賞金',
        '多頭出し.1':         '多頭出し',
        '所属.1':             '所属',
        '調教師.1':           '調教師',
        '走破タイム.1':       '走破タイム',     # numeric seconds (e.g. 1095)
        '着差.1':             '着差',
        '2角.1':              '2角',
        '3角.1':              '3角',
        '4角.1':              '4角',
        '上り3F.1':           '上り3F',
        'PCI':                'PCI',
        '好走':               '好走',
        'PCI3':               'PCI3',
        'RPCI':               'RPCI',
        '上3F地点差':         '上3F地点差',
        '馬体重.1':           '馬体重',
        '馬体重増減.1':       '馬体重増減',
        'ブリンカー.1':       'ブリンカー',
        '単勝配当':           '単勝配当',
        '複勝配当':           '複勝配当',
        '枠連':               '枠連',
        '馬連':               '馬連',
        '馬単':               '馬単',
        '３連複':             '３連複',
        '３連単':             '３連単',
        '間隔':               '間隔',
        '前走日付':           '前走日付',       # 既にYYMMDD形式
        '前走開催':           '前走開催',
        '前走Ｒ':             '前走Ｒ',
        '前走レース名':       '前走レース名',
        '替':                 '替',
        '前騎手':             '前騎手',
        '前走斤量':           '前走斤量',
        '前走頭数':           '前走頭数',
        '前走馬番':           '前走馬番',
        '前走人気':           '前走人気',
        '前走着順':           '前走着順',
        '前芝・ダ':           '前芝・ダ',
        '前距離':             '前距離',
        '前走馬場状態':       '前走馬場状態',
        '前走走破タイム':     '前走走破タイム', # numeric seconds
        '前走着差タイム':     '前走着差タイム',
        '前2角.1':            '前2角',
        '前3角.1':            '前3角',
        '前4角.1':            '前4角',
        '前走上り3F':         '前走上り3F',
        '前PCI':              '前PCI',
        '前好走':             '前好走',
        '前走馬体重':         '前走馬体重',
        '前走馬体重増減':     '前走馬体重増減',
        '前走B':              '前走B',
        '前走馬印':           '前走馬印',
        '前走馬印2':          '前走馬印2',
        '前走馬印3':          '前走馬印3',
        '前走馬印4':          '前走馬印4',
        '前走レース印１':     '前走レース印１',
    }

    # ── 変換実行 ──
    src_cols_present = [c for c in COL_MAP if c in df.columns]
    missing_src = [c for c in COL_MAP if c not in df.columns]
    if missing_src:
        print(f"  WARNING: ソースに存在しない列 ({len(missing_src)}個): {missing_src}")

    out = pd.DataFrame()
    out['Ｍ'] = np.nan  # master_kihonの先頭列
    for src, dst in COL_MAP.items():
        if src in df.columns:
            out[dst] = df[src].values
        else:
            out[dst] = np.nan

    # ── フィルタ ──
    out['日付_num'] = pd.to_numeric(out['日付'], errors='coerce')
    if args.all:
        df_out = out[out['日付_num'].notna()].drop(columns=['日付_num'])
        out_path = OUT_ALL
        tag = '1995-2026全部'
    else:
        # 2006-2012のみ（2000-2005は古すぎてROIが悪化したため除外）
        mask = out['日付_num'].notna() & (out['日付_num'] >= 60105) & (out['日付_num'] < 130101)
        df_out = out[mask].drop(columns=['日付_num'])
        out_path = OUT_HIST
        tag = '2006-2012'

    print(f"\n{tag}: {len(df_out):,}行")
    print(f"  日付範囲: {pd.to_numeric(df_out['日付'], errors='coerce').min()} "
          f"〜 {pd.to_numeric(df_out['日付'], errors='coerce').max()}")
    print(f"  列数: {len(df_out.columns)}")

    df_out.to_csv(out_path, encoding='cp932', index=False)
    print(f"\n保存完了: {out_path}")

    # ── master_horse.csv 互換の馬データも出力 ──
    HORSE_MAP = {
        '日付':               '日付',
        '開催':               '開催',
        'Ｒ':                 'Ｒ',
        'レース名.1':         'レース名',
        '馬名S':              '馬名S',
        'Ｃ.1':               'Ｃ',
        '性別.1':             '性別',
        '年齢.1':             '年齢',
        'キャリア':           'キャリア',
        '騎手.1':             '騎手',
        '人気.1':             '人気',
        '着順.1':             '着順',
        '頭数.1':             '頭数',
        '種牡馬':             '種牡馬',
        '母父馬':             '母父馬',
        '馬主(最新/仮想)':    '馬主(最新/仮想)',
        '生産者':             '生産者',
        '毛色':               '毛色',
        '馬記号':             '馬記号',
        '生年月日':           '生年月日',
        '市場取引価格(万/最終)': '市場取引価格(万/最終)',
        '取引市場(最終)':     '取引市場(最終)',
        '産地':               '産地',
    }
    horse_out = pd.DataFrame(index=range(len(df)))
    horse_out['Ｍ'] = np.nan
    for src, dst in HORSE_MAP.items():
        horse_out[dst] = df['日付'].values if src == '日付' else (df[src].values if src in df.columns else np.nan)

    horse_out['日付_num'] = pd.to_numeric(horse_out['日付'], errors='coerce')
    if args.all:
        horse_df = horse_out[horse_out['日付_num'].notna()].drop(columns=['日付_num'])
        horse_path = os.path.join(BASE_DIR, 'data', 'raw', 'master', 'master_horse_2006_2026.csv')
    else:
        horse_mask = horse_out['日付_num'].notna() & (
            (horse_out['日付_num'] >= 60105) & (horse_out['日付_num'] < 130101)
        )
        horse_df = horse_out[horse_mask].drop(columns=['日付_num'])
        horse_path = os.path.join(BASE_DIR, 'data', 'raw', 'master', 'master_horse_2006_2012.csv')

    horse_df.to_csv(horse_path, encoding='cp932', index=False)
    print(f"馬データ保存完了: {horse_path} ({len(horse_df):,}行)")

    # ── master_kihon.csv との列整合チェック ──
    kihon_path = os.path.join(BASE_DIR, 'data', 'raw', 'master', 'master_kihon.csv')
    if os.path.exists(kihon_path):
        kihon_header = pd.read_csv(kihon_path, encoding='cp932', nrows=0).columns.tolist()
        only_kihon = [c for c in kihon_header if c not in df_out.columns]
        only_new   = [c for c in df_out.columns if c not in kihon_header]
        print(f"\nmaster_kihon.csvにのみある列({len(only_kihon)}個): {only_kihon}")
        print(f"新ファイルにのみある列({len(only_new)}個): {only_new}")

if __name__ == '__main__':
    main()
