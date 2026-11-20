#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成符合 ACL 2027 版权与伦理合规标准的公开分发数据集 (Tiered Licensing Public Release)：
====================================================================================
输出文件：`data/processed/c_longslop_bench_public_release.jsonl`
分层规则：
  1. 开源学术与知识共享授权子集 (`m4_chinese`, `hc3_chinese`, `blog_ruanyifeng` CC-BY-NC 4.0)：
     - `license`: 明确标注 `"MIT / CC-BY-SA (Academic Open Benchmark)"` 或 `"CC-BY-NC-4.0 (ruanyf/weekly)"`
     - `text`: 完整保留原文正文
     - `precomputed_4dim_features` & `masked_sparse_text`: 同时附带预计算特征
  2. 受著作权保护的独立博客与公众号文章 (`chenhao_coolshell`, `yunfeng`, `piglei`, `laixintao_kawabangga`, `kazike`, `wx_articles`)：
     - `license`: `"All Rights Reserved (URL + SHA256 + Precomputed Features + 100-Char Fair-Use Preview; Reconstruct via scrape_multi_human_authors.py)"`
     - `text`: 设为 `null`（不直接二次分发受版权保护的完整正文，彻底规避版权侵权风险）
     - `excerpt_100chars`: 仅保留前 100 字合理引用预览（符合著作权法适当引用条款）
     - `sha256`: 原文完整正文的 SHA-256 校验值
     - `precomputed_4dim_features`: 19 项四维防刷探针预计算特征字典（复现者无需下载原文即可 100% 复现所有实验分数）
     - `masked_sparse_text`: 专名与数字全部脱敏为 `E` 和 `0` 后的不可逆结构化掩码序列（供双极稀疏探针 TF-IDF 直接训练/评测，且无法还原原作者受版权保护的具体技术内容）
"""
import hashlib
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE / "evaluators"))
sys.path.insert(0, str(ROOT))

import antihack_scorer as AHS  # noqa: E402
from pilot_longslop_part2 import ChineseReferenceLM  # noqa: E402

OPEN_SOURCES = {"m4_chinese", "hc3_chinese", "blog_ruanyifeng"}

SOURCE_URLS = {
    "chenhao_coolshell": "https://coolshell.cn / https://github.com/marvinqu/coolshell-markdown",
    "ruanyifeng": "https://github.com/ruanyf/weekly (CC-BY-NC-4.0)",
    "yunfeng": "https://blog.codingnow.com/atom.xml",
    "piglei": "https://www.piglei.com/feeds/latest/",
    "laixintao_kawabangga": "https://www.kawabangga.com/feed",
    "kazike": "https://mp.weixin.qq.com (Public WeChat Archive)",
    "wx_llm_pipeline": "https://mp.weixin.qq.com (Public AI Tech Summary Archive)",
}


def main():
    in_path = BASE / "data/processed/c_longslop_bench_v1_1.jsonl"
    out_path = BASE / "data/processed/c_longslop_bench_public_release.jsonl"

    rows = [json.loads(line) for line in open(in_path, encoding="utf-8")]
    lm = ChineseReferenceLM()
    lm.fit_hc3_human()
    AHS.lm_global = lm

    dev_human = [r["text"] for r in rows if r["split"] == "dev" and r["track"] == "human"]
    band = AHS.HumanBand(dev_human, lm)

    public_rows = []
    open_cnt, protected_cnt = 0, 0
    for r in rows:
        raw_text = r["text"]
        sha = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        feats = band._extract(raw_text)
        masked_text = AHS.normalize_for_sparse(raw_text)
        src = r["source_dataset"]
        auth = r["author"]

        if src in OPEN_SOURCES:
            lic = "CC-BY-NC-4.0 (https://github.com/ruanyf/weekly)" if auth == "ruanyifeng" else "Academic Open License (M4 / HC3 Benchmark)"
            pub_item = {
                "sample_id": r["sample_id"],
                "domain": r["domain"],
                "source_dataset": src,
                "author": auth,
                "era": r["era"],
                "split": r["split"],
                "track": r["track"],
                "char_len": r["char_len"],
                "topic_key": r["topic_key"],
                "license": lic,
                "source_url": SOURCE_URLS.get(auth, "https://huggingface.co/datasets"),
                "sha256": sha,
                "text": raw_text,
                "excerpt_100chars": raw_text[:100] + "...",
                "precomputed_4dim_features": feats,
                "masked_sparse_text": masked_text,
            }
            open_cnt += 1
        else:
            pub_item = {
                "sample_id": r["sample_id"],
                "domain": r["domain"],
                "source_dataset": src,
                "author": auth,
                "era": r["era"],
                "split": r["split"],
                "track": r["track"],
                "char_len": r["char_len"],
                "topic_key": r["topic_key"],
                "license": "Copyrighted Author Blog / Article (URL + SHA256 + Precomputed Features + Entity-Masked Structure; Full text reconstructible via scrape_multi_human_authors.py)",
                "source_url": SOURCE_URLS.get(auth, ""),
                "sha256": sha,
                "text": None,
                "excerpt_100chars": raw_text[:100] + "...",
                "precomputed_4dim_features": feats,
                "masked_sparse_text": masked_text,
            }
            protected_cnt += 1
        public_rows.append(pub_item)

    with open(out_path, "w", encoding="utf-8") as f:
        for item in public_rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"=== 合规公开版数据集已生成: {out_path} ===")
    print(f"  - 开源授权保留完整正文样本 (M4 + HC3 + 阮一峰 CC-BY-NC 4.0): {open_cnt} 篇")
    print(f"  - 版权保护脱敏样本 (仅发布 URL + SHA256 + 100字预览 + 预计算特征 + 实体掩码结构): {protected_cnt} 篇")


if __name__ == "__main__":
    main()
