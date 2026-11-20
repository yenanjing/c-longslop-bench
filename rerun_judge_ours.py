#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 Leaderboard 补跑 LLM 盲判裁判（与 comparison_eval.py 同一裁判、同一 prompt、同一校准）：
  - 对象：paired_10_outputs 中 Ours 两个方法的产物（c_longslop_guided_gate.md）
    * method = clongslop_gate   （Ours + Gate, 1-Pass）
    * method = clongslop_nogate（Ours w/o Gate, 1-Pass；由 gate 版在门禁拦截前保留的段落近似不可得，
      实际脚本以 gate 产物 + original_normalized 未编辑段落混合模拟消融产物 → 本次先盲判 gate 版，
      w/o Gate 行如无独立产物则沿用「—」并注明）
  - 裁判模型：hy4-preview-ioa（comparison_eval.py JUDGE_MODEL，判据已校准 F1=0.895）
  - 输出：results/llm_judge_ours_backfill.jsonl
"""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(ROOT / 'atom-writer'))

from utils.llm_utils import call_llm, parse_json_response  # noqa: E402

JUDGE_MODEL = 'hy4-preview-ioa'

EVAL_SYSTEM = '''你是一位严谨的文本来源鉴定专家。你的任务：判断给定的中文文本是人类撰写的，还是 AI（大语言模型）生成的，并评估其与目标作者（一位以"活人感"著称的科技公众号作者）风格的匹配度。

综合考察：口语化程度、个人色彩、情绪波动、不完美之处；AI 倾向模板化结构、均匀节奏、面面俱到、升华式金句收尾、"首先/其次/最后"、过度长定语。短文本或正式书面文本更易误判，请谨慎给出置信度。'''


def eval_prompt(text: str) -> str:
    return f'''请判断以下中文文本是人类写的还是 AI 生成的。

【待鉴定文本】
{text[:4000]}

【输出要求】
只输出一个 JSON 对象：
{{
  "verdict": "HUMAN 或 AI",
  "confidence": 0-100 的整数,
  "persona_match_score": 0-100 的整数（与目标作者"活人感"风格的匹配度）,
  "ai_tells": ["AI味迹象，无则空数组"],
  "human_tells": ["人类写作迹象，无则空数组"],
  "overall_comment": "一句话总评"
}}'''


def judge_one(task):
    art_dir, path = task
    text = path.read_text(encoding='utf-8')
    body = text
    # 去 frontmatter
    if body.startswith('---'):
        parts = body.split('---', 2)
        body = parts[2] if len(parts) >= 3 else body
    body = body.strip()
    try:
        raw, elapsed = call_llm(eval_prompt(body), system=EVAL_SYSTEM, model=JUDGE_MODEL, timeout=600)
        parsed = parse_json_response(raw)
        parsed['elapsed'] = round(elapsed, 1)
    except Exception as e:
        parsed = {'error': str(e)[:300]}
    parsed['article'] = art_dir.name
    parsed['method'] = 'clongslop_gate'
    parsed['chars'] = len(body)
    return parsed


def main():
    base_dir = BASE / 'leaderboard' / 'paired_10_outputs'
    tasks = []
    for art_dir in sorted(base_dir.iterdir()):
        p = art_dir / 'c_longslop_guided_gate.md'
        if p.exists():
            tasks.append((art_dir, p))
    print(f'共 {len(tasks)} 篇待盲判（method=clongslop_gate, 裁判={JUDGE_MODEL}）')
    results = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(judge_one, t) for t in tasks]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            v = r.get('verdict', r.get('error', '?'))
            print(f"  {r['article'][:36]:38s} -> {v} conf={r.get('confidence', '-')}")
    out = BASE / 'results' / 'llm_judge_ours_backfill.jsonl'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        for r in sorted(results, key=lambda x: x['article']):
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    ok = [r for r in results if r.get('verdict') in ('HUMAN', 'AI')]
    hr = sum(1 for r in ok if r['verdict'] == 'HUMAN') / len(ok) if ok else 0
    print(f'\n盲判完成: {len(ok)}/{len(results)} 成功, HUMAN 通过率 = {hr:.0%}')
    print(f'落盘: {out}')


if __name__ == '__main__':
    main()
