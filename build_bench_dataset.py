#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C-LongSlopBench v1.1 数据集标准化构建脚本 (Multi-Author & Multi-Domain Upgrade)
=============================================================================
升级亮点（回应「不要只绑死卡兹克一个博主」的核心诉求）：
1. 引入 6 位风格迥异、真实身份可核验的中国知名技术/随笔真人作者（共 109 篇深度长文）：
   - 陈皓 (左耳朵耗子, CoolShell, 25篇, 2011–2023)：硬核系统架构与工程文化评论，Pre-AI 传奇博主
   - 阮一峰 (科技爱好者周刊刊首语, 22篇, 2019–2021)：100% Pre-ChatGPT 科技与社会观察随笔
   - 云风 (吴云洋, CodingNow, 15篇)：游戏引擎泰斗，20年独立博客亲笔技术随笔
   - piglei (《Python工匠》作者, 20篇)：软件工程哲学与架构漫谈
   - laixintao (卡瓦邦噶, 8篇)：基础设施工程师深度长文
   - 数字生命卡兹克 (19篇)：AI 时代头部科技公众号长文
2. 双实验验证设计：
   - 分层抽样进 dev / heldout，确保 dev 拟合的「人类分布带 [P10, P90]」代表广泛的中文真人优秀写作者，而非单一博主口癖；
   - 同时支持 Leave-One-Author-Out（留一作者交叉验证）：验证任何一位未见过的新真人博主，在我们的探针下是否依然稳稳落在 HUMANLIKE 区间！
3. 均衡扩充 M4-Chinese（ChatGPT + Davinci）与 HC3-Chinese（百科/金融/心理/问答/法律）长文配额。
"""
import glob
import json
import random
import re
from collections import Counter
from pathlib import Path

BASE = Path("/Users/wenweili/PycharmProjects/Auto-claude-code-research-in-sleep/ai-writing-research/c_longslop_bench")
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


def add_sample(rows, domain, source, author, era, split, track, text, topic_key=None):
    text = clean_md_header(text)
    if len(text) < MIN_LEN:
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


def main():
    rows = []

    # ── 1. 六大真实人类博主长文 (Multi-Author Authentic Human Anchors, 109 篇) ──
    # 1A. 卡兹克 (19 篇)：9 篇 dev + 10 篇 heldout
    kz_files = sorted(glob.glob(str(RAW / "kazike_articles/article_*.md")))
    for i, fp in enumerate(kz_files):
        t = Path(fp).read_text(encoding="utf-8", errors="ignore")
        sp = "dev" if i < 9 else "heldout"
        add_sample(rows, "tech_blog_human", "kazike", "kazike", "post_ai_human", sp, "human", t, topic_key=f"kz_{i:02d}")

    # 1B. 新采集的 5 位独立技术博主 (90 篇)：每位作者按 50% dev / 50% heldout 切分，同时保留完整 author 标签供 Leave-One-Author-Out 测试
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

    # ── 2. 原生 AI 公众号长文 (wx_articles, 90 篇)：45 dev + 45 heldout ──
    wx_files = sorted(glob.glob(str(RAW / "wx_articles/article_*.md")))
    random.shuffle(wx_files)
    for i, fp in enumerate(wx_files):
        t = Path(fp).read_text(encoding="utf-8", errors="ignore")
        sp = "dev" if i < 45 else "heldout"
        add_sample(rows, "tech_media_ai", "wx_articles", "wx_llm_pipeline", "post_ai_machine", sp, "machine", t, topic_key=f"wx_{i:02d}")

    # ── 3. M4-Chinese (qazh_chatgpt + qazh_davinci 长文回答) ──
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
                if len(h) >= MIN_LEN and model_tag == "chatgpt":
                    hu_list.append((f"m4_h_{idx:04d}", h))
                if len(a) >= MIN_LEN:
                    ai_list.append((f"m4_{model_tag}_{idx:04d}", a))
        random.shuffle(hu_list)
        random.shuffle(ai_list)
        for i, (tk, h) in enumerate(hu_list[:50]):
            add_sample(rows, "m4_open_qa", "m4_chinese", "m4_human", "pre_ai", "dev" if i < 25 else "heldout", "human", h, topic_key=tk)
        for i, (tk, a) in enumerate(ai_list[:45]):
            add_sample(rows, f"m4_{model_tag}", "m4_chinese", model_tag, "post_ai_machine", "dev" if i < 22 else "heldout", "machine", a, topic_key=tk)

    # ── 4. HC3-Chinese 多领域长文（baike / psychology / open_qa / finance / medicine / nlpcc_dbqa） ──
    for dom in ["baike", "psychology", "open_qa", "finance", "medicine", "nlpcc_dbqa"]:
        fp = RAW / f"hc3_chinese/{dom}.jsonl"
        if not fp.exists():
            continue
        hu, ai = [], []
        with open(fp, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                for h in (d.get("human_answers") or []):
                    if h and len(h.strip()) >= MIN_LEN:
                        hu.append(h.strip())
                for a in (d.get("chatgpt_answers") or []):
                    if a and len(a.strip()) >= MIN_LEN:
                        ai.append(a.strip())
        random.shuffle(hu)
        random.shuffle(ai)
        for i, t in enumerate(hu[:30]):
            add_sample(rows, f"hc3_{dom}", "hc3_chinese", f"hc3_{dom}_human", "pre_ai", "dev" if i < 15 else "heldout", "human", t, topic_key=f"hc3_{dom}_h_{i}")
        for i, t in enumerate(ai[:30]):
            add_sample(rows, f"hc3_{dom}", "hc3_chinese", "chatgpt_hc3", "post_ai_machine", "dev" if i < 15 else "heldout", "machine", t, topic_key=f"hc3_{dom}_a_{i}")

    for out_file in [OUT, OUT_V11]:
        with open(out_file, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"=== C-LongSlopBench v1.1 Built: {len(rows)} Total Long-Form Samples ===")
    stat_split = Counter((r["split"], r["track"]) for r in rows)
    for (sp, tr), n in sorted(stat_split.items()):
        print(f"  Split={sp:8s} | Track={tr:8s} -> {n:3d} 篇")

    print("\n--- 真实人类博主分布 (Authentic Human Bloggers) ---")
    auth_stat = Counter((r["author"], r["era"], r["split"]) for r in rows if r["track"] == "human")
    for (au, era, sp), n in sorted(auth_stat.items()):
        print(f"  Author={au:22s} | Era={era:18s} | Split={sp:8s} : {n:3d} 篇")


if __name__ == "__main__":
    main()
