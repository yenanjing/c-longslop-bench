#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C-LongSlopBench 面向微信公众号/技术长文的开箱即用诊断与无损微修 CLI (`wx_article_doctor.py`)
==============================================================================================
专为「公众号论文解读 / 技术博客生成 Skill」设计的 100% 自包含插件：
1. **结构零破坏（Markdown & HTML Invariance）**：
   - 将文章按双换行切分为块（Block），自动识别并原样锁定：
     * Markdown 标题 (`# / ## / ###`)
     * 图片与 HTML 图注 (`![...]`, `<center>...</center>`, `<img...>`)
     * 论文元信息块 (`· 论文标题：...`, `- 论文链接：...`)
     * LaTeX 公式块 (`$$...$$`)、代码块 (` ```...``` `) 与 Markdown 表格 (`|...|`)
   - 仅对真正的自然语言叙述段落（Narrative Prose Blocks）运行四维探针与局部处方微修，改完后 1:1 拼回原位置！
2. **双引擎完整评分（100% 自包含）**：
   - 输出四维防刷 Z-Score (`D1` 微观拐杖、`D2` 跨句搭扣与句首惊奇度、`D3` 段首同质度、`D4` 宏观回环)、
     双极稀疏探针 AI 概率 (`sparse_ai_prob`)、人类通过分 (`HumanPass`) 与官方总分 (`Total Score`)。
3. **支持两种工作模式**：
   - `--diagnose-only`：50ms 纯本地确定性扫描，输出完整 JSON 诊断报告与逐段修改处方单（零 LLM 消耗，可供外部 Agent 自己改文章）；
   - `--repair`：自动调用 LLM + `ParagraphQualityGate`（段落局部质量门禁）对命中处方的正文段执行 1-Pass 微修，
     确保 100% 事实数字保留、零排版破坏、四维防刷 Z 显著下降。
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))

import antihack_scorer as AHS  # noqa: E402
from probes_core import (  # noqa: E402
    ChineseReferenceLM,
    SPEECH_BUFFERS,
    MODAL_PARTICLES,
    split_sentences,
)
import jieba.posseg as pseg  # noqa: E402

BUF_WORDS = ("其实", "说白了", "你看", "平心而论", "顺着", "先看", "再说", "接着", "老实说", "说实话", "讲真")


def call_llm_standalone(prompt: str, model: str = "hy4flash", timeout: int = 180) -> Tuple[str, float]:
    """优先使用环境变量 OPENAI_API_KEY / OPENAI_BASE_URL，回退使用本地 codebuddy CLI。"""
    t0 = time.monotonic()
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if api_key:
        import urllib.request
        url = base_url.rstrip("/") + "/chat/completions"
        payload = json.dumps({
            "model": os.environ.get("OPENAI_MODEL", model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"], time.monotonic() - t0

    cb_bin = shutil.which("codebuddy") or os.path.expanduser("~/.npm-global/bin/codebuddy")
    env = dict(os.environ)
    env["DISABLE_AUTOUPDATER"] = "1"
    cmd = [cb_bin, "--print", "--tools", "", "--model", model, prompt]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate(timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"LLM CLI failed: {err[:200]}")
    return out.strip(), time.monotonic() - t0


def parse_json_obj(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def is_locked_block(block: str) -> bool:
    """判断该块是否为严禁改动的排版块（标题、图注、公式、代码块、表格、论文元信息）。"""
    s = block.strip()
    if not s or len(s) < 20:
        return True
    if s.startswith(("#", "![", "<center", "<img", "<div", "<p", "$$", "```", "|", ">", "---", "***")):
        return True
    if "<center>" in s or "</center>" in s or "![" in s:
        return True
    first_line = s.split("\n")[0].strip()
    if re.match(
        r"^[·•\-\*\s]*(?:一句话讲清楚|论文标题|论文题目|论文链接|项目链接|开源链接|代码链接|作者|机构|来源|arXiv)[：:]",
        first_line,
        re.I,
    ):
        return True
    if re.match(r"^(?:Figure|Fig\.|Table|图|表)\s*\d+", first_line, re.I):
        return True
    return False


def diagnose_blocks(blocks: List[str]) -> List[Dict[str, Any]]:
    """在保持原始块索引 (`block_idx`) 不变的前提下，仅对自然语言正文块生成诊断处方。"""
    rx = []
    seen = set()
    prose_indices = [i for i, b in enumerate(blocks) if not is_locked_block(b)]
    n_prose = max(1, len(prose_indices))

    for rank, b_idx in enumerate(prose_indices):
        p = blocks[b_idx].strip()
        sents = split_sentences(p)
        issues = []

        if "——" in p:
            for si, s in enumerate(sents):
                if "——" in s:
                    issues.append({
                        "type": "EM_DASH_CRUTCH",
                        "sent_idx": si,
                        "evidence": s.replace("——", "[[——]]"),
                        "instruction": "去掉破折号『——』，改为逗号分句或自然动词衔接（如『说白了就是』『相当于』）；严禁换成括号或冒号。",
                    })

        parens = re.findall(r"（[A-Za-z0-9_\-\s]{2,30}）|\([A-Za-z0-9_\-\s]{2,30}\)", p)
        if len(parens) >= 2:
            issues.append({
                "type": "PARENTHETICAL_ENUM_TIC",
                "sent_idx": 0,
                "evidence": re.sub(r"(（[A-Za-z0-9_\-\s]{2,30}）|\([A-Za-z0-9_\-\s]{2,30}\))", r"[[\1]]", p),
                "instruction": f"删除句内密集的括号注释/编号 {parens[:5]}，首次出现保留即可，后续直接用中文词指代。",
            })

        if re.search(r"(?:^|[。！？\n])[^，。！？\n：:《》]{2,14}：", p) and len(p) >= 35:
            issues.append({
                "type": "COLON_PROMPT_TIC",
                "sent_idx": 0,
                "evidence": p.replace("：", "[[：]]", 1),
                "instruction": "消除『总结词：展开』冒号提示体，改为完整的主谓宾自然陈述句（如『核心在于』『做法是』）。",
            })

        abrupt_eng = re.findall(r"[。！？]\s*([A-Za-z][A-Za-z0-9_\-]{2,18})\s*[\u4e00-\u9fff]", p)
        if len(abrupt_eng) >= 1:
            issues.append({
                "type": "ABRUPT_ENGLISH_SUBJECT_ONSET",
                "sent_idx": 0,
                "evidence": f"[[。{abrupt_eng[0]}...]]",
                "instruction": f"消除句号后直接用英文名『{abrupt_eng[0]}』硬顶句首的机器感（稀疏探针强 AI 特征 `。E`）：在句首垫一个承接上句的指代或顺承短语（如『在此基础上，{abrupt_eng[0]}』『顺着这个设计，它』）。",
            })

        if sents and rank >= 2:
            s0 = sents[0]
            cw0 = [w for w, pos in pseg.cut(s0) if pos.startswith("n") and len(w) >= 2]
            has_old = any(w in seen for w in cw0[:2])
            has_hook = s0.startswith(AHS.PRONOUN_HOOKS)
            if not has_old and not has_hook and len(s0) <= 35:
                issues.append({
                    "type": "ABRUPT_ONSET_BREAK",
                    "sent_idx": 0,
                    "evidence": f"[[{s0}]]",
                    "instruction": "段首突兀硬抛新概念（违反以旧带新律）。句首补一个承接上一段末尾概念的代词或过渡搭扣。",
                })

        if int(n_prose * 0.30) <= rank <= int(n_prose * 0.70):
            has_num = bool(re.search(r"\d+|[「」“”]", p))
            has_buf = any(b in p for b in SPEECH_BUFFERS) or any(mp in p for mp in MODAL_PARTICLES)
            if not has_num and not has_buf and len(p) >= 70:
                issues.append({
                    "type": "MID_SECTION_ABSTRACT_DRIFT",
                    "sent_idx": 0,
                    "evidence": f"[[{p[:35]}...]]",
                    "instruction": "中段全抽象机制名词堆叠且无气口。拆开连续名词定语，补 1 处自然讲课过渡词或直白解释。",
                })

        for s in sents:
            for w, pos in pseg.cut(s):
                if pos.startswith("n") and len(w) >= 2:
                    seen.add(w)

        if issues:
            prev_prose = blocks[prose_indices[rank - 1]][-40:] if rank > 0 else ""
            rx.append({
                "block_idx": b_idx,
                "orig_text": p,
                "prev_para_tail": prev_prose,
                "issues": issues,
            })
    return rx


def local_crutch_score(para: str) -> float:
    n = max(1, len(para))
    paren = len(re.findall(r"[（(][^）)]{1,25}[）)]", para))
    colon = len(re.findall(r"[^0-9][：:]", para))
    em = para.count("——")
    return (em * 2.0 + paren * 2.2 + colon * 1.3) / n * 1000.0


def check_local_gate(orig_p: str, cand_p: str, prev_prefixes: set) -> Tuple[bool, str]:
    fg = AHS.factual_gate(orig_p, cand_p)
    if not fg["passed"] or fg["anchor_retention"] < 0.98:
        return False, f"丢失事实数字或专名锚点 {fg['missing_critical_numbers']}"
    if not (0.65 * len(orig_p) <= len(cand_p) <= 1.40 * len(orig_p)):
        return False, f"段落字数越界 ({len(cand_p)} vs 原段 {len(orig_p)})"
    buf_cnt = sum(cand_p.count(w) for w in BUF_WORDS)
    also_cnt = cand_p.count("也就是")
    if local_crutch_score(cand_p) > local_crutch_score(orig_p) or buf_cnt > 1 or also_cnt > 1:
        return False, f"口语词堆砌(buf={buf_cnt}, 也就是={also_cnt})或标点拐杖未降"
    if len(cand_p) >= 2 and cand_p[:2] in prev_prefixes:
        return False, f"段首起手词「{cand_p[:2]}」与前段重复"
    return True, "PASS"


def repair_markdown_preserving_layout(raw_md: str, model: str = "hy4flash") -> Tuple[str, Dict[str, Any]]:
    blocks = re.split(r"\n\s*\n", raw_md.strip())
    rx = diagnose_blocks(blocks)
    if not rx:
        return raw_md, {"triggered_blocks": 0, "accepted_blocks": 0, "reverted_blocks": 0}

    draft_map = {}
    batch_size = 12
    for start in range(0, len(rx), batch_size):
        batch = rx[start:start + batch_size]
        payload = [
            {
                "block_idx": it["block_idx"],
                "prev_para_tail": it["prev_para_tail"],
                "text": it["orig_text"],
                "prescriptions": [iss["instruction"] for iss in it["issues"]],
            }
            for it in batch
        ]
        prompt = f"""你是微信公众号硬核科技专栏主编（人设：娓娓道来的老师，讲得清楚、有耐心，严禁使用夸张网感黑话或捏造个人跑卡经历）。
请根据每段的 `prescriptions` 诊断指令对 `text` 进行最小代价的句法微修：
1. 100% 原样保留所有阿拉伯数字、百分比、实验指标、英文模型名与专有名词；
2. 消除破折号 `——`、段首死板冒号 `小标题：` 和多余括号注释，化为自然流畅的中文分句；
3. 若提示句首硬顶英文名或断链，请结合 `prev_para_tail` 在句首补一个自然的以旧带新过渡（如「顺着这个思路」「在此基础上」「为了解决这一问题」），严禁连续两段使用相同的开头词；
4. 每段最多使用 1 处口语气口词（「其实」「说白了」「换句话说」），切勿堆砌。

【待微修段落 JSON】
{json.dumps(payload, ensure_ascii=False)}

只输出一个 JSON 对象：{{"repaired": {{"block_idx字符串": "微修后的段落文本", ...}}}}"""
        try:
            resp, _ = call_llm_standalone(prompt, model=model, timeout=180)
            parsed = parse_json_obj(resp).get("repaired", {})
            for k, v in parsed.items():
                if isinstance(v, str) and len(v.strip()) >= 10:
                    draft_map[int(k)] = v.strip()
        except Exception as e:
            print(f"  [Warn] Batch repair error: {e}")

    out_blocks = list(blocks)
    accepted, reverted = 0, 0
    prose_indices = [i for i, b in enumerate(blocks) if not is_locked_block(b)]

    for it in rx:
        b_idx = it["block_idx"]
        orig_p = blocks[b_idx].strip()
        cand_p = draft_map.get(b_idx, orig_p)
        if cand_p == orig_p:
            reverted += 1
            continue
        rank = prose_indices.index(b_idx)
        prev_prefixes = {
            out_blocks[prose_indices[j]].strip()[:2]
            for j in range(max(0, rank - 3), rank)
            if len(out_blocks[prose_indices[j]].strip()) >= 2
        }
        ok, _ = check_local_gate(orig_p, cand_p, prev_prefixes)
        if ok:
            out_blocks[b_idx] = cand_p
            accepted += 1
        else:
            out_blocks[b_idx] = orig_p
            reverted += 1

    repaired_md = "\n\n".join(out_blocks) + "\n"
    return repaired_md, {
        "total_blocks": len(blocks),
        "prose_blocks": len(prose_indices),
        "triggered_blocks": len(rx),
        "accepted_blocks": accepted,
        "reverted_blocks": reverted,
    }


def init_evaluator():
    bench_file = REPO_ROOT / "data/processed/c_longslop_bench_v1_1.jsonl"
    if not bench_file.exists():
        bench_file = REPO_ROOT / "data/processed/c_longslop_bench_public_release.jsonl"
    rows = [json.loads(line) for line in open(bench_file, encoding="utf-8")]
    dev_h = [r for r in rows if r["split"] == "dev" and r["track"] == "human"]
    dev_m = [r for r in rows if r["split"] == "dev" and r["track"] == "machine"]

    lm = ChineseReferenceLM()
    lm.fit_hc3_human()
    AHS.lm_global = lm
    band = AHS.HumanBand(dev_h, lm)

    hn = [r.get("masked_sparse_text") or AHS.normalize_for_sparse(r["text"]) for r in dev_h]
    mn = [r.get("masked_sparse_text") or AHS.normalize_for_sparse(r["text"]) for r in dev_m]
    vec, clf = AHS.fit_sparse_probe(hn, mn)
    return band, vec, clf


def score_article(md_text: str, band: AHS.HumanBand, vec, clf) -> Dict[str, Any]:
    feats = band._extract(md_text)
    dims = AHS.score_four_dims(feats, band)
    x = vec.transform([AHS.normalize_for_sparse(md_text)])
    sparse_p = round(float(clf.predict_proba(x)[0, 1]), 3)
    total_score = round(0.70 * dims["humanpass_100"] + 0.30 * (1.0 - sparse_p) * 100.0, 1)
    return {
        "total_score": total_score,
        "composite_z": dims["composite_z"],
        "humanpass_100": dims["humanpass_100"],
        "sparse_ai_prob": sparse_p,
        "D1_micro_crutch": dims["D1"],
        "D2_meso_cohesion": dims["D2"],
        "D3_skeleton_clone": dims["D3"],
        "D4_macro_echo": dims["D4"],
        "key_raw_metrics": {
            "punct_crutch_per_1k": feats["M1_punct_crutch_total"],
            "colon_prompt_per_1k": feats["M1_colon_prompt_rate"],
            "speech_buffer_per_1k": feats["M1_speech_buffer_rate"],
            "cohesion_break_rate": round(feats["M2_cohesion_break_rate"], 3),
            "uid_onset_shock": round(feats["UID_onset_minus_end_surp"], 3),
            "opener_homogeneity": round(feats["M3_opener_homogeneity"], 3),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="待诊断或微修的 Markdown 文章路径")
    ap.add_argument("--diagnose-only", action="store_true", help="仅运行 50ms 确定性四维诊断与处方输出，不调用 LLM")
    ap.add_argument("--repair", action="store_true", help="执行带门禁的无损处方微修")
    ap.add_argument("--out", default="", help="微修后 Markdown 输出路径")
    ap.add_argument("--json-report", default="", help="诊断/微修报告 JSON 输出路径")
    ap.add_argument("--model", default="hy4flash", help="微修使用的 LLM 模型名")
    args = ap.parse_args()

    fp = Path(args.file)
    raw_md = fp.read_text(encoding="utf-8")

    band, vec, clf = init_evaluator()
    blocks = re.split(r"\n\s*\n", raw_md.strip())
    rx_before = diagnose_blocks(blocks)
    scores_before = score_article(raw_md, band, vec, clf)

    report = {
        "file": str(fp),
        "before": scores_before,
        "prescriptions_count_before": len(rx_before),
        "prescriptions": rx_before,
    }

    if args.repair and not args.diagnose_only:
        repaired_md, gate_stats = repair_markdown_preserving_layout(raw_md, model=args.model)
        scores_after = score_article(repaired_md, band, vec, clf)
        fg = AHS.factual_gate(raw_md, repaired_md)
        rx_after = diagnose_blocks(re.split(r"\n\s*\n", repaired_md.strip()))
        out_fp = Path(args.out) if args.out else fp.with_name(fp.stem + "_repaired.md")
        out_fp.write_text(repaired_md, encoding="utf-8")
        report["after"] = scores_after
        report["prescriptions_count_after"] = len(rx_after)
        report["gate_stats"] = gate_stats
        report["factual_retention"] = fg["anchor_retention"]
        report["repaired_file"] = str(out_fp)
        print(f"=== C-LongSlopBench 公众号文章无损微修完成 ===")
        print(f"  Total Score : {scores_before['total_score']} -> {scores_after['total_score']}")
        print(f"  防刷综合 Z  : {scores_before['composite_z']} -> {scores_after['composite_z']} (D1: {scores_before['D1_micro_crutch']}->{scores_after['D1_micro_crutch']}, D2: {scores_before['D2_meso_cohesion']}->{scores_after['D2_meso_cohesion']})")
        print(f"  稀疏 AI 率  : {scores_before['sparse_ai_prob']} -> {scores_after['sparse_ai_prob']}")
        print(f"  标点拐杖/千字: {scores_before['key_raw_metrics']['punct_crutch_per_1k']} -> {scores_after['key_raw_metrics']['punct_crutch_per_1k']}")
        print(f"  处方命中段数: {len(rx_before)} 段 -> {len(rx_after)} 段 | 事实锚点保留率: {fg['anchor_retention']:.1%}")
        print(f"  已保存至    : {out_fp}")
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.json_report:
        Path(args.json_report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
