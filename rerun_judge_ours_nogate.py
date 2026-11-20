#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补跑 Ours w/o Gate（消融组）的 LLM 盲判，与全部基线同一裁判、同一 prompt、同一校准：
  Step 1: 复现 run_main_bench_experiments.py 的产物生成逻辑（确定性、无 LLM 调用），
          把此前只在内存中计算、未落盘的 w/o Gate 消融产物 (use_quality_gate=False)
          重新生成并保存为 leaderboard/paired_10_outputs/<article>/c_longslop_guided_nogate.md
  Step 2: 用裁判 hy4-preview-ioa（comparison_eval.py JUDGE_MODEL，判据已校准 F1=0.895）
          对 10 篇产物逐一盲判，落盘 results/llm_judge_ours_nogate.jsonl
"""
import glob
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / 'atom-writer'))

from utils.llm_utils import call_llm, parse_json_response  # noqa: E402
import run_main_bench_experiments as M  # noqa: E402

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
    art_name, path = task
    body = path.read_text(encoding='utf-8').strip()
    try:
        raw, elapsed = call_llm(eval_prompt(body), system=EVAL_SYSTEM, model=JUDGE_MODEL, timeout=600)
        parsed = parse_json_response(raw)
        parsed['elapsed'] = round(elapsed, 1)
    except Exception as e:
        parsed = {'error': str(e)[:300]}
    parsed['article'] = art_name
    parsed['method'] = 'clongslop_nogate'
    parsed['chars'] = len(body)
    return parsed


def main():
    rows = [json.loads(l) for l in open(BASE / 'data/processed/c_longslop_bench_v1_1.jsonl', encoding='utf-8')]
    lm = M.ChineseReferenceLM()
    lm.fit_hc3_human()
    M.AHS.lm_global = lm
    dev_human = [r['text'] for r in rows if r['split'] == 'dev' and r['track'] == 'human']
    dev_ai = [r['text'] for r in rows if r['split'] == 'dev' and r['track'] == 'machine']
    band = M.AHS.HumanBand(dev_human, lm)
    vec, clf = M.AHS.fit_sparse_probe(
        [M.AHS.normalize_for_sparse(t) for t in dev_human],
        [M.AHS.normalize_for_sparse(t) for t in dev_ai],
    )

    comp_dir = ROOT / 'results/comparison_main'
    tasks = []
    for ad in sorted(glob.glob(str(comp_dir / 'article_*'))):
        ad_path = Path(ad)
        art_name = ad_path.name
        orig_candidates = (
            list((ROOT / 'output_hy4v13' / art_name).glob('original.txt'))
            + list((ROOT / 'output_atomv5' / art_name).glob('original.txt'))
            + list((ROOT / 'data/wx_articles').glob(f'{art_name[:25]}*.md'))
        )
        if not orig_candidates:
            print(f'  跳过（找不到原文）: {art_name}')
            continue
        raw_txt = orig_candidates[0].read_text(encoding='utf-8', errors='ignore')
        norm_orig = M.normalize_wx_original(raw_txt)
        rep_nogate, _ = M.apply_general_prescription_repair(norm_orig, band, vec, clf, use_quality_gate=False)
        out_art_dir = BASE / 'leaderboard/paired_10_outputs' / art_name
        out_art_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_art_dir / 'c_longslop_guided_nogate.md'
        out_path.write_text(rep_nogate, encoding='utf-8')
        tasks.append((art_name, out_path))

    print(f'已落盘 {len(tasks)} 篇 w/o Gate 消融产物，开始盲判（裁判={JUDGE_MODEL}，与全部基线同一协议）')
    results = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(judge_one, t) for t in tasks]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            v = r.get('verdict', r.get('error', '?'))
            print(f"  {r['article'][:36]:38s} -> {v} conf={r.get('confidence', '-')}")

    out = BASE / 'results' / 'llm_judge_ours_nogate.jsonl'
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
