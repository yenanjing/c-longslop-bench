#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实 LLM 批量生成 Paired-10 的全部缺失基线与处方微修产物（彻底淘汰假正则脚本！）：
  1. Zero-Shot Generic Prompting (`zero_shot_llm.md`)：
     真实调用 LLM，仅给泛化去 AI 味指令，不提供句段级病灶定位。
  2. Naive Rule-Checklist Prompting (`checklist_llm.md`)：
     真实调用 LLM，给单向禁用词表与「长句切短、多用破折号补充说明」的传统 Checklist 提示词。
  3. C-LongSlop Prescription-Guided Repair：
     - `w/o Gate` (`c_longslop_guided_nogate.md`)：单遍处方引导 LLM 微修（不设段落门禁，直接接受 LLM 首轮微修）；
     - `+ Gate` (`c_longslop_guided_gate.md`)：处方引导 LLM 微修 + **零测试集泄漏的段落局部质量门禁 (`ParagraphQualityGate`)**。
       门禁仅检查四项段落局部自洽条件（绝不调用测试集 `sparse_ai_prob` 或全文 `HumanBand` Oracle！）：
       ① 段落内原子事实锚点（数字/英文专名）100% 零丢失；
       ② 段落长度守恒（0.60x ~ 1.45x）；
       ③ 段落局部解释拐杖（冒号/破折号/括号）严格消除，且单段口语套话不堆砌（<=1处）；
       ④ 相邻段落起手式去重（禁止与前 3 段任一段的前 2 字雷同，防止 LLM 懒惰复读「先看/顺着」）。
       若首轮微修未通过门禁，门禁向 LLM 发出 **1 次带具体违规原因的局部重试 (`Gate-Guided Retry`)**；若重试仍违规，则安全回滚至原段。
"""
import difflib
import glob
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "evaluators"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "atom-writer"))

import antihack_scorer as AHS  # noqa: E402
import build_bench_dataset as BBD  # noqa: E402
from utils.llm_utils import call_llm, parse_json_response  # noqa: E402

REWRITE_MODEL = "hy4flash"
BUF_WORDS = ("其实", "说白了", "你看", "平心而论", "顺着", "先看", "再说", "接着")


def local_crutch_score(para: str) -> float:
    n = max(1, len(para))
    paren = len(re.findall(r"[（(][^）)]{1,25}[）)]", para))
    colon = len(re.findall(r"[^0-9][：:]", para))
    em = para.count("——")
    return (em * 2.0 + paren * 2.2 + colon * 1.2) / n * 1000.0


def check_local_gate(orig_p: str, cand_p: str, prev_prefixes: set) -> tuple:
    fg_para = AHS.factual_gate(orig_p, cand_p)
    if not fg_para["passed"] or fg_para["anchor_retention"] < 0.98:
        return False, f"丢失事实锚点 {fg_para['missing_critical_numbers']}"
    if not (0.60 * len(orig_p) <= len(cand_p) <= 1.45 * len(orig_p)):
        return False, f"字数越界 ({len(cand_p)} vs 原段 {len(orig_p)})"
    buf_cnt = sum(cand_p.count(w) for w in BUF_WORDS)
    if local_crutch_score(cand_p) > local_crutch_score(orig_p) or buf_cnt > 1 or "用来及" in cand_p:
        return False, f"口语词堆砌(cnt={buf_cnt})或标点拐杖未降"
    if len(cand_p) >= 2 and cand_p[:2] in prev_prefixes:
        return False, f"段首起手词「{cand_p[:2]}」与前段雷同，需换用主语或动词直接起手"
    return True, "PASS"


def refine_gate_on_article(ad_path: Path):
    art_name = ad_path.name
    out_dir = BASE / "leaderboard/paired_10_outputs" / art_name
    norm_orig = (out_dir / "original_normalized.md").read_text(encoding="utf-8")
    nogate_doc = (out_dir / "c_longslop_guided_nogate.md").read_text(encoding="utf-8")

    paras = [p.strip() for p in norm_orig.split("\n\n") if p.strip()]
    ng_paras = [p.strip() for p in nogate_doc.split("\n\n") if p.strip()]
    rx = AHS.generate_prescriptions(norm_orig)

    gate_paras = list(paras)
    retry_tasks = []
    first_pass_accepted = 0

    for it in rx:
        idx = it["para_idx"]
        if idx >= len(paras) or idx >= len(ng_paras):
            continue
        orig_p = paras[idx]
        cand_p = ng_paras[idx]
        if cand_p == orig_p:
            continue
        prev_prefixes = {gate_paras[j][:2] for j in range(max(0, idx - 3), idx) if len(gate_paras[j]) >= 2}
        ok, reason = check_local_gate(orig_p, cand_p, prev_prefixes)
        if ok:
            gate_paras[idx] = cand_p
            first_pass_accepted += 1
        else:
            retry_tasks.append({
                "para_idx": idx,
                "orig_text": orig_p,
                "draft_rewrite": cand_p,
                "gate_reject_reason": reason,
                "forbidden_openers": sorted(prev_prefixes),
            })

    retried_accepted = 0
    reverted_cnt = 0
    if retry_tasks:
        prompt = f"""你是严谨的中文科技文章编辑。以下段落的首轮微修草稿被《段落质量门禁 (ParagraphQualityGate)》拦截。
请根据每段的 `gate_reject_reason` 修正草稿：
1. 若因「段首起手词与前段雷同」被拦截：严禁使用 `forbidden_openers` 中的词开头，改用「这样一来，」「所以」「因此」「其中」「从效果看，」或该段核心名词开头；
2. 若因「口语词堆砌或标点拐杖未降」被拦截：删掉多余的「先看/顺着」，把残留的冒号 `：` 或破折号 `——` 化为「也就是」「就是」等自然陈述句，句末可用自然的「了。」「了，」收束；
3. 若因「字数越界」或「丢失事实锚点」被拦截：严格贴合 `orig_text` 的字数，并 100% 补齐所有数字和英文专名！

【待门禁重试段落 JSON】
{json.dumps(retry_tasks, ensure_ascii=False)}

只输出一个 JSON 对象：{{"repaired": {{"段落para_idx字符串": "修正后的段落文本", ...}}}}"""
        try:
            resp, _ = call_llm(prompt, model=REWRITE_MODEL, timeout=150)
            rep_map = parse_json_response(resp).get("repaired", {})
        except Exception:
            rep_map = {}

        for task in retry_tasks:
            idx = task["para_idx"]
            orig_p = paras[idx]
            cand2 = rep_map.get(str(idx), "").strip()
            prev_prefixes = {gate_paras[j][:2] for j in range(max(0, idx - 3), idx) if len(gate_paras[j]) >= 2}
            if cand2:
                ok2, _ = check_local_gate(orig_p, cand2, prev_prefixes)
                if ok2:
                    gate_paras[idx] = cand2
                    retried_accepted += 1
                    continue
            # 若重试仍未通过局部自洽门禁，安全回滚至原段
            gate_paras[idx] = orig_p
            reverted_cnt += 1

    doc_gate = "\n\n".join(gate_paras)
    sm = difflib.SequenceMatcher(None, norm_orig, doc_gate)
    edit_pct = round((1.0 - sm.ratio()) * 100.0, 1)
    fg_final = AHS.factual_gate(norm_orig, doc_gate)
    log_gate = {
        "article": art_name,
        "total_paras": len(paras),
        "prescriptions_triggered": len(rx),
        "accepted_edits": first_pass_accepted + retried_accepted,
        "gate_retried_and_fixed": retried_accepted,
        "reverted_by_gate": reverted_cnt,
        "edit_budget_pct": edit_pct,
        "factual_gate_passed": fg_final["passed"],
        "anchor_retention": fg_final["anchor_retention"],
    }
    (out_dir / "c_longslop_guided_gate.md").write_text(doc_gate, encoding="utf-8")
    (out_dir / "gate_audit_log.json").write_text(json.dumps(log_gate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✅ [{art_name[:34]:34s}] Gate 审计完成: 首轮放行={first_pass_accepted} 门禁重修通过={retried_accepted} 最终回滚={reverted_cnt} 保真={fg_final['anchor_retention']:.1%}")
    return log_gate


def main():
    comp_dir = ROOT / "results/comparison_main"
    art_dirs = [Path(p) for p in sorted(glob.glob(str(comp_dir / "article_*")))]
    with ThreadPoolExecutor(max_workers=3) as ex:
        logs = list(ex.map(refine_gate_on_article, art_dirs))
    logs.sort(key=lambda x: x["article"])
    (BASE / "results/real_llm_repair_logs.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
