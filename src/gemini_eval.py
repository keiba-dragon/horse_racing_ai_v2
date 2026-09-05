# -*- coding: utf-8 -*-
"""
Gemini API による生成AI評価レイヤー（v3 Phase 3・シャドウモード）。

統計モデル（hitrate_model, 新聞の◎○▲判定に使用）の上位候補馬について、
Gemini APIにレース状況・近走サマリを渡して「注目馬とその理由・0-100点評価」を取得する。

現段階ではシャドウモード: 新聞に参考表示するのみで、買い目（◎○の選定）には一切反映しない。
CLAUDE.md/メモリの「roi_model提案禁止」方針と同様、既存の的中率選定ロジックを
生成AIスコアで上書き・介入することはしない（8週間のシャドウ検証後に別途判断する）。

失敗時は例外を投げず None を返す（新聞生成を絶対に止めない）。

設定ファイル: config/gemini_config.json （config/gemini_config.json.example参照）
  {
    "api_key": "...",
    "model": "gemini-flash-latest",
    "enabled": true
  }
"""
import os
import json
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'gemini_config.json')
EVAL_LOG_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'ai_eval_log.csv')

DEFAULT_MODEL = 'gemini-flash-latest'
MAX_CANDIDATES = 5  # 統計モデル上位何頭までを評価対象にするか（コスト管理）
API_TIMEOUT = 20

_config_cache = None


def load_config() -> dict:
    """config/gemini_config.json を読む。未設定・不正なら enabled=False を返す。"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    if not os.path.exists(CONFIG_PATH):
        _config_cache = {'enabled': False}
        return _config_cache
    try:
        with open(CONFIG_PATH, encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception:
        _config_cache = {'enabled': False}
        return _config_cache
    api_key = str(cfg.get('api_key', '')).strip()
    if not api_key or 'YOUR_API_KEY' in api_key:
        cfg['enabled'] = False
    else:
        cfg.setdefault('enabled', True)
    cfg.setdefault('model', DEFAULT_MODEL)
    _config_cache = cfg
    return _config_cache


def is_enabled() -> bool:
    return bool(load_config().get('enabled'))


def _build_prompt(race_ctx: dict, candidates: list) -> str:
    lines = []
    lines.append(
        f"あなたは日本の競馬（JRA）の分析アシスタントです。"
        f"以下のレース・出走馬データだけを根拠に評価してください。データにない情報は推測しないこと。"
    )
    lines.append(
        f"レース: {race_ctx.get('venue', '?')} {race_ctx.get('r_num', '?')}R "
        f"{race_ctx.get('race_name', '')} "
        f"[{race_ctx.get('surface', '?')}{race_ctx.get('distance_m', '?')} / {race_ctx.get('seg_label', '?')}]"
    )
    lines.append("統計モデル（的中率最適化）上位候補:")
    for c in candidates:
        lines.append(
            f"- {c['horse']} (統計順位{c['stat_rank']}位, 統計予測確率{c.get('prob_pct', '?')}, "
            f"単勝オッズ{c.get('odds', '?')}, 騎手{c.get('jockey', '?')})"
        )
    lines.append(
        "各候補馬について0-100の評価スコア(高いほど推奨)と一言理由をJSONで返してください。"
        '出力形式: {"evaluations": [{"horse": "馬名", "score": 数値, "reason": "20字程度の理由"}], '
        '"note": "レース全体の一言コメント(30字程度)"} '
        "JSON以外の文字列は出力しないこと。"
    )
    return '\n'.join(lines)


def _call_gemini(prompt: str, model: str, api_key: str) -> str:
    import requests
    url = (
        f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
        f'?key={api_key}'
    )
    payload = {
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {
            'temperature': 0.2,
            'responseMimeType': 'application/json',
        },
    }
    r = requests.post(url, json=payload, timeout=API_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data['candidates'][0]['content']['parts'][0]['text']


def evaluate_race(race_ctx: dict, candidates: list) -> dict:
    """race_ctx: {venue, r_num, race_name, surface, distance_m, seg_label}
    candidates: [{horse, stat_rank, prob_pct, odds, jockey}, ...] (MAX_CANDIDATES件まで自動で絞る)

    戻り値: {horse_name: {'score': float, 'reason': str}, ...} + '_note': str
    取得失敗・未設定時は空dict {} を返す（例外は投げない）。
    """
    cfg = load_config()
    if not cfg.get('enabled'):
        return {}
    if not candidates:
        return {}

    candidates = candidates[:MAX_CANDIDATES]
    prompt = _build_prompt(race_ctx, candidates)

    try:
        text = _call_gemini(prompt, cfg.get('model', DEFAULT_MODEL), cfg['api_key'])
        parsed = json.loads(text)
    except Exception as e:
        print(f'  [WARN] gemini_eval: API呼び出し/解析失敗: {e}')
        return {}

    out = {}
    for ev in parsed.get('evaluations', []):
        name = str(ev.get('horse', '')).strip()
        if not name:
            continue
        try:
            score = float(ev.get('score'))
        except (TypeError, ValueError):
            continue
        out[name] = {'score': score, 'reason': str(ev.get('reason', '')).strip()}
    if out:
        out['_note'] = str(parsed.get('note', '')).strip()
    return out


def log_evaluations(date_num: int, race_ctx: dict, candidates: list, evaluations: dict) -> None:
    """予測時点のGemini評価をai_eval_log.csvに記録する（シャドウ検証の材料）。
    同一 日付+開催+R+馬名 のキーで上書き（同日複数回の新聞再生成に対応）。
    実際の着順はここでは記録しない（結果確定後に別途 reconcile する想定、v3 Phase 3範囲外）。
    """
    if not evaluations:
        return
    import pandas as pd

    rows = []
    for c in candidates:
        name = c['horse']
        ev = evaluations.get(name)
        if ev is None:
            continue
        rows.append({
            '日付': date_num,
            '開催': race_ctx.get('venue', ''),
            'R': race_ctx.get('r_num', ''),
            '馬名S': name,
            '統計順位': c.get('stat_rank'),
            '統計予測確率': c.get('prob_pct'),
            'オッズ': c.get('odds'),
            'AI評価スコア': ev['score'],
            'AI理由': ev.get('reason', ''),
            '記録日時': time.strftime('%Y-%m-%d %H:%M:%S'),
        })
    if not rows:
        return
    new_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(EVAL_LOG_PATH), exist_ok=True)
    if os.path.exists(EVAL_LOG_PATH):
        try:
            existing = pd.read_csv(EVAL_LOG_PATH, encoding='utf-8-sig')
            key_cols = ['日付', '開催', 'R', '馬名S']
            new_keys = set(map(tuple, new_df[key_cols].values.tolist()))
            keep = existing[~existing[key_cols].apply(tuple, axis=1).isin(new_keys)]
            new_df = pd.concat([keep, new_df], ignore_index=True)
        except Exception as e:
            print(f'  [WARN] gemini_eval: ai_eval_log.csv 読込失敗、新規作成します: {e}')
    new_df.to_csv(EVAL_LOG_PATH, index=False, encoding='utf-8-sig')
