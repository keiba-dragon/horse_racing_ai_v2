# coding: utf-8
"""
ROI実験ログ記録ユーティリティ

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
■ 指標定義
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【指標定義】
    rank_prob=1 : レース内でprob_win（モデル勝率予測）が最高の馬
                  = 「モデルが最も強いと判断した馬」= 実力1位
    rank_edge=1 : edge = prob_win - 0.75/odds が最高の馬
                  (longshot選択に偏る傾向あり。avg_odds≈20倍)
    ※ rank_edge/prob は race_id 単位で計算。gk単位だとバグ

    market_P の正しい式 = 0.75 / odds  (控除率25%を考慮)
                                        ※ 1/odds は誤り

【プロジェクト目標】
    rank_prob=1 を全レース・全条件で買い続けたときの
    単勝ROI を -5% 以上にする（控除率 ≈ -20% からの改善）

【真のOOS基準 (2026-05-15計測, 完全OOS: 2025-2026)】
    rank_prob=1 全買い（全条件）:
        N=3,031  勝率=27.6%  avg_OD=4.3倍  単勝ROI=-20.0%
        年別: 2025 N=2,319 ROI=-19.7% / 2026 N=712 ROI=-21.0%

【過去の誤った+ROI結果について】
    旧プロジェクトのrank_edge=1 ダ15頭+ 2025=+20.6%はリーク:
    oos_predictions.parquet に2021-2024年が含まれていた時代に
    その同データでisotonic calibrationを学習→2025テスト（循環リーク）。
    正しくval(2023-24)でcalib学習→oos(2025-26)テストすると -33.3%。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

使い方:
    from roi_logger import log

    log(
        name         = "E指標全レース全買いベースライン",
        hypothesis   = "全条件rank_edge=1全買いのROI確認",
        train_period = "2013-2022",
        test_period  = "2025-2026 OOS",
        cheat_risk   = "低",
        bet_type     = "単勝",
        selection    = {"rank_edge": 1, "filter": "なし（全条件）"},
        results      = {"N": 3036, "win_rate": 0.134, "roi_tan": -0.222},
        conclusion   = "控除率水準と同等。2025が特に悪い(-31.4%)",
        next_action  = "2025悪化の原因調査・条件絞り込み探索",
    )
"""
import json, os
from datetime import datetime

LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'experiments', 'roi_log.jsonl'
)


def log(
    name:         str,
    hypothesis:   str,
    train_period: str,
    test_period:  str,
    cheat_risk:   str,
    bet_type:     str,
    selection:    dict,
    results:      dict,
    conclusion:   str  = '',
    next_action:  str  = '',
    notes:        str  = '',
):
    """実験結果を roi_log.jsonl に1行追記する。"""
    record = {
        'ts':           datetime.now().strftime('%Y-%m-%d %H:%M'),
        'name':         name,
        'hypothesis':   hypothesis,
        'train_period': train_period,
        'test_period':  test_period,
        'cheat_risk':   cheat_risk,
        'bet_type':     bet_type,
        'selection':    selection,
        'results':      results,
        'conclusion':   conclusion,
        'next_action':  next_action,
        'notes':        notes,
    }

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

    roi = results.get('roi_tan', results.get('roi', None))
    roi_s = f'{roi:+.1%}' if roi is not None else '-'
    n     = results.get('N', results.get('n', '-'))
    print(f'[roi_logger] 記録完了: {name}  ROI={roi_s}  N={n}')
