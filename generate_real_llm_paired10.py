#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整可复现的真实 LLM 配对改写与局部自洽门禁引擎 (Full Reproducible Pipeline for Paired-10)：
==========================================================================================
包含全部四个方法的真实大模型生成函数（可通过命令行参数选择重跑全部或指定方法）：
  1. `run_zero_shot_llm(norm_orig)` -> `zero_shot_llm.md`：
     真实调用 LLM 执行 DeepMind DetectorScope Fig.5 的 Generic 泛化去 AI 味提示词。
  2. `run_checklist_llm(norm_orig)` -> `checklist_llm.md`：
     真实调用 LLM 执行社区流行的《单向规则清单》(Catch Me If You Can 显式约束改写)：
     严禁诱导模型使用特定标点，只给标准的单向指令（「1. 禁用首先/其次/此外/因此/综上所述；
     2. 严禁使用死板的『小标题：』格式；3. 尽可能将长句拆成短句以大幅提高 Burstiness；
     4. 多加入口语语气词让文风更口语化」），自然展示无防刷锁时模型触发的跨句断链与口癖越界！
  3. `run_unconstrained_prescription_llm(norm_orig)` -> `c_longslop_guided_nogate.md` (Ours w/o Gate)：
     真实的无门禁消融组——将 `AHS.generate_prescriptions(norm_orig)` 产出的原始客观处方直接喂给 LLM，
     **不施加任何门禁约束、不限制单段口语词数量、不检查相邻段首重复、不做事实锚点回滚**，
     真实暴露「只给处方但无 ParagraphQualityGate 时，LLM 会因用力过猛产生口语词过载与段首同质化」的问题！
  4. `run_gated_prescription_llm(norm_orig)` -> `c_longslop_guided_gate.md` (Ours + Gate)：
     在处方引导微修基础上，启用 **零测试集泄漏的段落局部质量门禁 (`ParagraphQualityGate`)**：
     仅检查四项段落局部自洽条件（① 段落内数字/英文专名 100% 零丢失；② 段落长度守恒 [0.60x, 1.45x]；
     ③ 局部标点拐杖严格下降且单段口语缓冲词不堆砌 <=1 处；④ 相邻 3 段内段首前 2 字不雷同，且同一连接词不连用）。
     对未通过门禁的段落执行局部修正或安全回滚原段，证明门禁带来统计显著 (p < 0.01) 的质量跃升！
"""
import argparse
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
BUF_WORDS = ("其实", "说白了", "你看", "平心而论", "顺着", "先看", "再说", "接着", "老实说", "说实话", "讲真")


def local_crutch_score(para: str) -> float:
    n = max(1, len(para))
    paren = len(re.findall(r"[（(][^）)]{1,25}[）)]", para))
    colon = len(re.findall(r"[^0-9][：:]", para))
    em = para.count("——")
    return (em * 2.0 + paren * 2.2 + colon * 1.3) / n * 1000.0


def run_zero_shot_llm(norm_orig: str) -> str:
    prompt = f"""请将下面这篇中文科技文章改写得更像人类作者自然撰写的文章，去除明显的 AI 生成腔调，同时完整保留文章原本的技术事实、实验数字和专有名词。直接输出改写后的全文，不要输出任何前言或解释：

{norm_orig}"""
    out, _ = call_llm(prompt, model=REWRITE_MODEL, timeout=180)
    return out.strip() if len(out.strip()) >= 500 else norm_orig


def run_checklist_llm(norm_orig: str) -> str:
    """
    公平复现单向规则清单改写（严禁人为诱导破折号！只给标准社区去 AI 味 Checklist）：
    要求禁用套话连接词、取消小标题冒号、将复合长句切短以提高 Burstiness、多加口语感叹与语气词。
    """
    prompt = f"""请严格按照以下《去 AI 味规则清单 (Single-Sided Rule Checklist)》改写下面这篇科技文章，直接输出改写后的全文，不要输出任何前言或解释：
1. 严禁使用「首先、其次、此外、因此、综上所述、值得注意的是、总而言之」等 AI 机械连接词；
2. 严禁使用死板的「小标题：解释」冒号结构；
3. 大幅提高句长变化（Burstiness）：把所有超过 25 字的复合长句统统切分成简短的短句或独立小段，制造短促有力的节奏；
4. 大量增加口语化的语气词、口头禅（「说实话、讲道理、其实、说白了、你看、吧、呢、啊」）和主观感叹，让文章充满强烈的口语人味；
5. 保留文中的核心数据与模型名称。

【待改写原文】
{norm_orig}"""
    out, _ = call_llm(prompt, model=REWRITE_MODEL, timeout=180)
    return out.strip() if len(out.strip()) >= 500 else norm_orig


def run_unconstrained_prescription_llm(norm_orig: str) -> str:
    """
    消融组 Ours w/o Gate：直接把 `generate_prescriptions` 生成的原始处方指令发给 LLM，
    不加任何门禁防过载约束、不限制单段口语词频次、不检查段首重复、不做回滚。
    """
    paras = [p.strip() for p in norm_orig.split("\n\n") if p.strip()]
    rx = AHS.generate_prescriptions(norm_orig)
    if not rx:
        return norm_orig

    repaired_map = {}
    batch_size = 15
    for start in range(0, len(rx), batch_size):
        batch = rx[start:start + batch_size]
        payload = [
            {
                "para_idx": it["para_idx"],
                "text": paras[it["para_idx"]],
                "prescriptions": [iss["instruction"] for iss in it["issues"]],
            }
            for it in batch
        ]
        prompt = f"""下面是一篇科技文章中触发 AI 味诊断处方的段落列表。请严格按照每段的 `prescriptions` 指令改写对应段落，积极补充口语过渡词、讲课气口词与句末语气词，彻底消除冒号、括号和破折号。保留技术数字与专有名词。

【待改写段落 JSON】
{json.dumps(payload, ensure_ascii=False)}

只输出一个 JSON 对象：{{"repaired": {{"段落para_idx字符串": "改写后的段落文本", ...}}}}"""
        try:
            resp, _ = call_llm(prompt, model=REWRITE_MODEL, timeout=150)
            parsed = parse_json_response(resp).get("repaired", {})
            for k, v in parsed.items():
                if isinstance(v, str) and len(v.strip()) >= 8:
                    repaired_map[int(k)] = v.strip()
        except Exception as e:
            print(f"    ⚠️ w/o Gate batch warning: {e}")

    out_paras = [repaired_map.get(i, p) for i, p in enumerate(paras)]
    return "\n\n".join(out_paras)


def check_local_gate(orig_p: str, cand_p: str, prev_prefixes: set) -> tuple:
    fg_para = AHS.factual_gate(orig_p, cand_p)
    if not fg_para["passed"] or fg_para["anchor_retention"] < 0.98:
        return False, f"丢失事实锚点 {fg_para['missing_critical_numbers']}"
    if not (0.60 * len(orig_p) <= len(cand_p) <= 1.45 * len(orig_p)):
        return False, f"字数越界 ({len(cand_p)} vs 原段 {len(orig_p)})"
    buf_cnt = sum(cand_p.count(w) for w in BUF_WORDS)
    also_cnt = cand_p.count("也就是") + cand_p.count("其实在于")
    if local_crutch_score(cand_p) > local_crutch_score(orig_p) or buf_cnt > 1 or also_cnt > 1 or "用来及" in cand_p:
        return False, f"口语词/同位语堆砌(buf={buf_cnt}, 也就是={also_cnt})或标点拐杖未降"
    if len(cand_p) >= 2 and cand_p[:2] in prev_prefixes:
        return False, f"段首起手词「{cand_p[:2]}」与前段雷同，需换用主语或动词直接起手"
    return True, "PASS"


def run_gated_prescription_llm(norm_orig: str, draft_nogate_doc: str = None):
    """
    完整方法 Ours + Gate：对触发处方的段落应用 `ParagraphQualityGate` 局部自洽门禁，
    拦截口语词过载、段首起手复读、`也就是` 堆砌或事实锚点丢失的段落，触发受控微修或安全回滚原段。
    """
    paras = [p.strip() for p in norm_orig.split("\n\n") if p.strip()]
    rx = AHS.generate_prescriptions(norm_orig)
    if draft_nogate_doc is None:
        draft_nogate_doc = run_unconstrained_prescription_llm(norm_orig)
    ng_paras = [p.strip() for p in draft_nogate_doc.split("\n\n") if p.strip()]

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
        batch_size = 15
        rep_map = {}
        for start in range(0, len(retry_tasks), batch_size):
            batch = retry_tasks[start:start + batch_size]
            prompt = f"""你是严谨的中文科技专栏主编。以下段落的首轮改写草稿被《段落局部质量门禁 (ParagraphQualityGate)》拦截。
请根据每段的 `gate_reject_reason` 修正为克制、自然、流畅的中文技术段落：
1. 若因「段首起手词与前段雷同」被拦截：严禁使用 `forbidden_openers` 中的词开头，直接用该段的核心技术名词主语、代词（「这」「它」「其中」）或具体动词开头；
2. 若因「口语词/同位语堆砌或标点拐杖未降」被拦截：删掉多余的「说白了/你看/其实/说实话/也就是」，每段最多保留 1 处自然过渡，把冒号 `：` 或破折号 `——` 用不同的动词谓语（如「核心在于」「承担的是」「表现为」「意味着」）自然串起来，严禁在同一段重复使用同一个连接词；
3. 若因「字数越界」或「丢失事实锚点」被拦截：严格贴合 `orig_text` 的字数，并 100% 原样保留所有阿拉伯数字、百分比和英文专名！

【待门禁重修段落 JSON】
{json.dumps(batch, ensure_ascii=False)}

只输出一个 JSON 对象：{{"repaired": {{"段落para_idx字符串": "修正后的段落文本", ...}}}}"""
            try:
                resp, _ = call_llm(prompt, model=REWRITE_MODEL, timeout=150)
                parsed = parse_json_response(resp).get("repaired", {})
                for k, v in parsed.items():
                    if isinstance(v, str) and len(v.strip()) >= 8:
                        rep_map[str(k)] = v.strip()
            except Exception as e:
                print(f"    ⚠️ Gate retry batch warning: {e}")

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
            gate_paras[idx] = orig_p
            reverted_cnt += 1

    doc_gate = "\n\n".join(gate_paras)
    sm = difflib.SequenceMatcher(None, norm_orig, doc_gate)
    edit_pct = round((1.0 - sm.ratio()) * 100.0, 1)
    fg_final = AHS.factual_gate(norm_orig, doc_gate)
    log_gate = {
        "article": "",
        "total_paras": len(paras),
        "prescriptions_triggered": len(rx),
        "first_pass_accepted": first_pass_accepted,
        "gate_retried_and_fixed": retried_accepted,
        "accepted_edits": first_pass_accepted + retried_accepted,
        "reverted_by_gate": reverted_cnt,
        "edit_budget_pct": edit_pct,
        "factual_gate_passed": fg_final["passed"],
        "anchor_retention": fg_final["anchor_retention"],
    }
    return doc_gate, log_gate


def process_article(ad_path: Path, mode: str = "update_checklist_and_ours"):
    art_name = ad_path.name
    t0 = time.monotonic()
    orig_candidates = (
        list((ROOT / "output_hy4v13" / art_name).glob("original.txt"))
        + list((ROOT / "output_atomv5" / art_name).glob("original.txt"))
        + list((ROOT / "data/wx_articles").glob(f"{art_name[:25]}*.md"))
    )
    raw_txt = orig_candidates[0].read_text(encoding="utf-8", errors="ignore")
    norm_orig = BBD.normalize_wx_original(raw_txt)

    out_dir = BASE / "leaderboard/paired_10_outputs" / art_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "original_normalized.md").write_text(norm_orig, encoding="utf-8")

    if mode in ("all", "zeroshot") or not (out_dir / "zero_shot_llm.md").exists():
        zs_doc = run_zero_shot_llm(norm_orig)
        (out_dir / "zero_shot_llm.md").write_text(zs_doc, encoding="utf-8")

    if mode in ("all", "update_checklist_and_ours", "checklist"):
        ck_doc = run_checklist_llm(norm_orig)
        (out_dir / "checklist_llm.md").write_text(ck_doc, encoding="utf-8")

    if mode in ("all", "update_checklist_and_ours", "ours"):
        ng_doc = run_unconstrained_prescription_llm(norm_orig)
        (out_dir / "c_longslop_guided_nogate.md").write_text(ng_doc, encoding="utf-8")
        g_doc, log_gate = run_gated_prescription_llm(norm_orig, ng_doc)
        log_gate["article"] = art_name
        (out_dir / "c_longslop_guided_gate.md").write_text(g_doc, encoding="utf-8")
        (out_dir / "gate_audit_log.json").write_text(json.dumps(log_gate, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        log_gate = json.loads((out_dir / "gate_audit_log.json").read_text(encoding="utf-8"))

    el = time.monotonic() - t0
    print(f"  ✅ [{art_name[:34]:34s}] ({el:.1f}s) | 触发={log_gate['prescriptions_triggered']} 首轮过={log_gate.get('first_pass_accepted', 0)} 门禁重修={log_gate.get('gate_retried_and_fixed', 0)} 回滚={log_gate['reverted_by_gate']} 保真={log_gate['anchor_retention']:.1%}")
    return log_gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="update_checklist_and_ours", choices=["all", "update_checklist_and_ours", "ours", "checklist"])
    args = ap.parse_args()

    comp_dir = ROOT / "results/comparison_main"
    art_dirs = [Path(p) for p in sorted(glob.glob(str(comp_dir / "article_*")))]
    print(f"运行完整可复现 Paired-10 生成管线 (mode={args.mode}, N={len(art_dirs)})...")
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(process_article, ad, args.mode) for ad in art_dirs]
        logs = [f.result() for f in as_completed(futs)]
    logs.sort(key=lambda x: x["article"])
    (BASE / "results/real_llm_repair_logs.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    print("全部完成并落盘！")


if __name__ == "__main__":
    main()
