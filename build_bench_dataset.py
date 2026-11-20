#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C-LongSlopBench v1.2 数据集标准化构建脚本 (ACL 2027 Rigor & Zero-Leakage Upgrade)
================================================================================
修复独立审计指出的全部数据集硬伤：
1. 彻底消除 Paired-10 数据泄漏：
   - 凡是属于 `results/comparison_main` 配对评测集 (Paired-10) 的文章，100% 强制划入 `heldout` 测试集，
     严禁任何一篇进入 `dev` 训练集（此前有 7/10 混入 dev 导致稀疏探针泄漏）。
2. 统一清洗爬虫断行伪影 (`normalize_wx_original`)：
   - 在入库前对所有 `wx_articles` 统一执行断行合并与元数据清理，消除「英文单词独立成行」的排版伪影，
     确保 Table 1 与 Table 2 文本预处理 100% 一致。
3. 修复 HC3 / M4 子集切分失衡 Bug：
   - 严格按每个子集实际有效样本数的 50% / 50% 对半切分 `dev` 与 `heldout`（`half = len(sub) // 2`），
     彻底消灭 `hc3_baike` / `hc3_open_qa` 在 `heldout` 上单边样本数为 0 的分类器捷径。
"""
import glob
import json
import random
import re
from collections import Counter
from pathlib import Path

BASE = Path("/Users/wenweili/PycharmProjects/Auto-claude-code-research-in-sleep/ai-writing-research/c_longslop_bench")
ROOT = BASE.parent
RAW = BASE / "data/raw"
OUT = BASE / "data/processed/c_longslop_bench_v1.jsonl"
OUT_V11 = BASE / "data/processed/c_longslop_bench_v1_1.jsonl"

random.seed(42)
MIN_LEN = 450


def clean_md_header(t: str) -> str:
    lines = t.strip().split("\n")
    for idx in range(min(12, len(lines))):
        if lines[idx].strip() == "---":
            t = "\n".join(lines[idx + 1:]).strip()
            break
    t = re.sub(r"^#.*\n", "", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
    t = re.sub(r"https?://\S+", "", t)
    return t.strip()


def normalize_wx_original(raw_text: str) -> str:
    """
    统一清理微信公众号爬虫排版伪影（将单词独立成行的碎片还原为正常自然段），
    保证全库 AI 文本与人类文本在自然段切分粒度上完全一致。
    """
    t = clean_md_header(raw_text)
    lines = [ln.strip() for ln in t.split("\n")]
    skip_prefixes = (
        "一句话讲清楚", "论文标题", "论文链接", "项目链接", "开源链接",
        "代码链接", "作者：", "来源：", "📄", "🌐", "⭐️", "👉🏻"
    )

    merged_paras = []
    buf = ""
    for ln in lines:
        if not ln:
            if buf:
                if buf[-1] in "。！？!?：:”」』" and len(buf) >= 25:
                    merged_paras.append(buf)
                    buf = ""
            continue
        if any(ln.startswith(p) for p in skip_prefixes):
            continue
        if re.match(r"^Figure\s*\d+", ln, re.I) or re.match(r"^表\s*\d+", ln):
            continue
        if not buf:
            buf = ln
        else:
            if buf[-1] not in "。！？!?" or len(ln) < 18 or re.match(r"^[A-Za-z0-9_\-\.]+$", ln):
                buf = buf + " " + ln if re.match(r"^[A-Za-z0-9]", ln) else buf + ln
            else:
                merged_paras.append(buf)
                buf = ln
    if buf:
        merged_paras.append(buf)

    valid_paras = [re.sub(r"\s{2,}", " ", p).strip() for p in merged_paras if len(p.strip()) >= 15]
    return "\n\n".join(valid_paras)


def is_valid_article(t: str) -> bool:
    if len(t) < MIN_LEN:
        return False
    t2 = re.sub(r"(?<!\d)\.(?!\d)", "。", t)
    sents = [s.strip() for s in re.findall(r"[^。！？!\?\n；;]+[。！？!\?；;]?", t2) if len(s.strip()) >= 6]
    return len(sents) >= 6 and (t.count("。") + t.count("！") + t.count("？") >= 3)


def add_sample(rows, domain, source, author, era, split, track, text, topic_key=None, normalize_wx=False):
    text = normalize_wx_original(text) if normalize_wx else clean_md_header(text)
    if not is_valid_article(text):
        return False
    rows.append({
        "sample_id": f"csb-{len(rows) + 1:05d}",
        "domain": domain,
        "source_dataset": source,
        "author": author,
        "era": era,
        "split": split,
        "track": track,
        "text": text,
        "char_len": len(text),
        "topic_key": topic_key,
    })
    return True


def get_paired_10_prefixes():
    comp_dir = ROOT / "results/comparison_main"
    art_dirs = sorted(glob.glob(str(comp_dir / "article_*")))
    prefixes = []
    for ad in art_dirs:
        name = Path(ad).name
        # 提取日期+标题前12字作为唯一匹配键
        prefixes.append(name[:22])
    return prefixes


def main():
    rows = []
    paired10_prefixes = get_paired_10_prefixes()

    # ── 1. 六大真实人类博主长文 (Multi-Author Authentic Human Anchors, 109 篇) ──
    kz_files = sorted(glob.glob(str(RAW / "kazike_articles/article_*.md")))
    for i, fp in enumerate(kz_files):
        t = Path(fp).read_text(encoding="utf-8", errors="ignore")
        sp = "dev" if i < 9 else "heldout"
        add_sample(rows, "tech_blog_human", "kazike", "kazike", "post_ai_human", sp, "human", t, topic_key=f"kz_{i:02d}")

    multi_path = RAW / "multi_human_authors/multi_authors_corpus.jsonl"
    if multi_path.exists():
        by_author = {}
        with open(multi_path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                by_author.setdefault(d["author"], []).append(d)
        for auth, items in sorted(by_author.items()):
            random.shuffle(items)
            half = len(items) // 2
            for idx, it in enumerate(items):
                sp = "dev" if idx < half else "heldout"
                add_sample(
                    rows,
                    "tech_blog_human",
                    f"blog_{auth}",
                    auth,
                    it.get("era", "pre_ai"),
                    sp,
                    "human",
                    it["text"],
                    topic_key=f"{auth}_{idx:02d}",
                )

    # ── 2. 原生 AI 公众号长文 (wx_articles, 90 篇)：Paired-10 强制进 heldout 零泄漏！ ──
    wx_files = sorted(glob.glob(str(RAW / "wx_articles/article_*.md")))
    paired_wx, other_wx = [], []
    for fp in wx_files:
        stem = Path(fp).stem
        if any(stem.startswith(pref[:18]) or pref[:18] in stem for pref in paired10_prefixes):
            paired_wx.append(fp)
        else:
            other_wx.append(fp)

    random.shuffle(other_wx)
    # 保证 dev 有 45 篇非配对集的 wx_articles，其余 + 全部 Paired-10 进 heldout
    for i, fp in enumerate(other_wx):
        t = Path(fp).read_text(encoding="utf-8", errors="ignore")
        sp = "dev" if i < 45 else "heldout"
        add_sample(rows, "tech_media_ai", "wx_articles", "wx_llm_pipeline", "post_ai_machine", sp, "machine", t,
                   topic_key=f"wx_other_{i:02d}", normalize_wx=True)

    for i, fp in enumerate(paired_wx):
        t = Path(fp).read_text(encoding="utf-8", errors="ignore")
        add_sample(rows, "tech_media_ai", "wx_articles", "wx_llm_pipeline", "post_ai_machine", "heldout", "machine", t,
                   topic_key=f"wx_paired10_{i:02d}", normalize_wx=True)

    # ── 3. M4-Chinese (qazh_chatgpt + qazh_davinci 同题长文对半切分) ──
    for fn, model_tag in [("qazh_chatgpt.jsonl", "chatgpt"), ("qazh_davinci.jsonl", "davinci")]:
        m4_path = RAW / f"m4_chinese/{fn}"
        if not m4_path.exists():
            continue
        hu_list, ai_list = [], []
        with open(m4_path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                d = json.loads(line)
                h = (d.get("human_text") or "").strip()
                a = (d.get("machine_text") or "").strip()
                if is_valid_article(h) and model_tag == "chatgpt":
                    hu_list.append((f"m4_h_{idx:04d}", h))
                if is_valid_article(a):
                    ai_list.append((f"m4_{model_tag}_{idx:04d}", a))
        random.shuffle(hu_list)
        random.shuffle(ai_list)
        hu_sub = hu_list[:50]
        ai_sub = ai_list[:45]
        for i, (tk, h) in enumerate(hu_sub):
            add_sample(rows, "m4_open_qa", "m4_chinese", "m4_human", "pre_ai",
                       "dev" if i < len(hu_sub) // 2 else "heldout", "human", h, topic_key=tk)
        for i, (tk, a) in enumerate(ai_sub):
            add_sample(rows, f"m4_{model_tag}", "m4_chinese", model_tag, "post_ai_machine",
                       "dev" if i < len(ai_sub) // 2 else "heldout", "machine", a, topic_key=tk)

    # ── 4. HC3-Chinese 多领域长文：按每个子集实际可用对数 1:1 严格五五切分 dev / heldout ──
    for dom in ["baike", "psychology", "open_qa", "finance", "medicine", "nlpcc_dbqa"]:
        fp = RAW / f"hc3_chinese/{dom}.jsonl"
        if not fp.exists():
            continue
        hu, ai = [], []
        with open(fp, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                for h in (d.get("human_answers") or []):
                    if h and is_valid_article(h.strip()):
                        hu.append(h.strip())
                for a in (d.get("chatgpt_answers") or []):
                    if a and is_valid_article(a.strip()):
                        ai.append(a.strip())
        random.shuffle(hu)
        random.shuffle(ai)
        hu_sub = hu[:30]
        ai_sub = ai[:30]
        # 仅保留双边都至少有 4 篇的子集，且各自严格按 len // 2 切分 dev 与 heldout
        if len(hu_sub) >= 4 and len(ai_sub) >= 4:
            for i, t in enumerate(hu_sub):
                add_sample(rows, f"hc3_{dom}", "hc3_chinese", f"hc3_{dom}_human", "pre_ai",
                           "dev" if i < len(hu_sub) // 2 else "heldout", "human", t, topic_key=f"hc3_{dom}_h_{i}")
            for i, t in enumerate(ai_sub):
                add_sample(rows, f"hc3_{dom}", "hc3_chinese", "chatgpt_hc3", "post_ai_machine",
                           "dev" if i < len(ai_sub) // 2 else "heldout", "machine", t, topic_key=f"hc3_{dom}_a_{i}")

    for out_file in [OUT, OUT_V11]:
        with open(out_file, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"=== C-LongSlopBench v1.2 (Zero-Leakage & Clean Splits) Built: {len(rows)} Samples ===")
    stat_split = Counter((r["split"], r["track"]) for r in rows)
    for (sp, tr), n in sorted(stat_split.items()):
        print(f"  Split={sp:8s} | Track={tr:8s} -> {n:3d} 篇")

    print("\n--- 各 Domain 在 dev / heldout 的 Human vs Machine 均衡度核查 ---")
    dom_stat = Counter((r["domain"], r["split"], r["track"]) for r in rows)
    domains = sorted(set(r["domain"] for r in rows))
    for d in domains:
        dh = dom_stat[(d, "dev", "human")]
        dm = dom_stat[(d, "dev", "machine")]
        hh = dom_stat[(d, "heldout", "human")]
        hm = dom_stat[(d, "heldout", "machine")]
        print(f"  Domain={d:20s} | dev (H={dh:2d}, M={dm:2d}) | heldout (H={hh:2d}, M={hm:2d})")


if __name__ == "__main__":
    main()
