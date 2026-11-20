#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C-LongSlopBench v1.1 全面审计与首版排行榜生成器
================================================
执行三项核心评测：
1. 【多博主泛化审计 (Per-Author Breakdown on Heldout)】：
   检验陈皓(CoolShell)、阮一峰、云风、piglei、laixintao、卡兹克 6 位风格截然不同的真人博主，
   在 C-LongSlopBench 下各自的四维偏离度 Z、HumanPass 得分与稀疏探针 AI 概率。
2. 【留一博主交叉验证 (Leave-One-Author-Out Cross-Validation)】：
   每次把 1 位博主从训练集/分布带拟合中彻底剔除（Zero-Shot Unseen Blogger），
   只用其余博主拟合人类分布带与稀疏探针，测试该未知博主 vs AI 文本的 AUROC ——
   直接证明评测器抓的是「普适中文真人长文规律」，而非过拟合某一个博主的个人口癖！
3. 【首版中文长文去 AI 味方法排行榜 (Leaderboard v1.1)】：
   将 6 位真人基准、原生 AI、Few-Shot 模仿、历史各代改写器（hy4v8/hy4v13/AtomV4/AtomV5/V15_Teacher）
   以及「C-LongSlop 处方引导微修 (Prescription-Guided Repair)」同场竞技打榜！
"""
import glob
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

BASE = Path("/Users/wenweili/PycharmProjects/Auto-claude-code-research-in-sleep/ai-writing-research/c_longslop_bench")
sys.path.insert(0, str(BASE / "evaluators"))
sys.path.insert(0, str(BASE.parent))

import antihack_scorer as AHS  # noqa: E402
from pilot_longslop_part2 import ChineseReferenceLM  # noqa: E402


def main():
    rows = [json.loads(l) for l in open(BASE / "data/processed/c_longslop_bench_v1_1.jsonl", encoding="utf-8")]

    lm = ChineseReferenceLM()
    lm.fit_hc3_human()
    AHS.lm_global = lm

    dev_human = [r["text"] for r in rows if r["split"] == "dev" and r["track"] == "human"]
    dev_ai = [r["text"] for r in rows if r["split"] == "dev" and r["track"] == "machine"]

    band = AHS.HumanBand(dev_human, lm)
    hn = [AHS.normalize_for_sparse(t) for t in dev_human]
    an = [AHS.normalize_for_sparse(t) for t in dev_ai]
    vec, clf = AHS.fit_sparse_probe(hn, an)

    # ── 1. Heldout 集上按作者/来源细分统计 ──
    print("=" * 105)
    print("AUDIT 1: Heldout 集分博主 / 分来源详细画像 (验证 6 位真人博主是否全面达标)")
    print("=" * 105)
    by_author = defaultdict(list)
    for r in rows:
        if r["split"] != "heldout":
            continue
        feats = band._extract(r["text"])
        if not feats:
            continue
        dims = AHS.score_four_dims(feats, band)
        x = vec.transform([AHS.normalize_for_sparse(r["text"])])
        prob = float(clf.predict_proba(x)[0, 1])
        verdict = "HUMANLIKE" if (dims["composite_z"] < 0.65 and prob < 0.5) else "AI_OR_GRAY"
        by_author[(r["track"], r["author"], r["era"])].append({**dims, "prob": prob, "verdict": verdict})

    print(f"{'Track':8s} | {'Author / Source':22s} | {'Era':22s} | {'N':>3s} | {'D1':>5s} | {'D2':>5s} | {'D3':>5s} | {'D4':>5s} | {'Comp_Z':>6s} | {'HumanPass':>9s} | {'AI_Prob':>7s} | {'Pass%':>6s}")
    print("-" * 125)
    author_audit_table = []
    for (tr, auth, era), items in sorted(by_author.items()):
        n = len(items)
        d1 = statistics.mean(x["D1"] for x in items)
        d2 = statistics.mean(x["D2"] for x in items)
        d3 = statistics.mean(x["D3"] for x in items)
        d4 = statistics.mean(x["D4"] for x in items)
        cz = statistics.mean(x["composite_z"] for x in items)
        hp = statistics.mean(x["humanpass_100"] for x in items)
        pr = statistics.mean(x["prob"] for x in items)
        pass_r = sum(1 for x in items if x["verdict"] == "HUMANLIKE") / n
        print(f"{tr:8s} | {auth:22s} | {era:22s} | {n:3d} | {d1:5.2f} | {d2:5.2f} | {d3:5.2f} | {d4:5.2f} | {cz:6.3f} | {hp:9.1f} | {pr:7.3f} | {pass_r:6.1%}")
        author_audit_table.append({
            "track": tr, "author": auth, "era": era, "n": n,
            "D1": round(d1, 3), "D2": round(d2, 3), "D3": round(d3, 3), "D4": round(d4, 3),
            "composite_z": round(cz, 3), "humanpass_100": round(hp, 1),
            "sparse_ai_prob": round(pr, 3), "humanlike_rate": round(pass_r, 3),
        })

    # ── 2. 留一博主交叉验证 (Leave-One-Author-Out Cross-Validation) ──
    print("\n" + "=" * 105)
    print("AUDIT 2: 留一博主交叉验证 (Leave-One-Author-Out, LOAO) —— 检验对未知真人博主的零样本泛化性")
    print("=" * 105)
    bloggers = ["chenhao_coolshell", "ruanyifeng", "yunfeng", "piglei", "laixintao_kawabangga", "kazike"]
    heldout_ai_rows = [r for r in rows if r["split"] == "heldout" and r["track"] == "machine" and r["domain"] == "tech_media_ai"]

    print(f"{'被留出的未知博主 (Zero-Shot Author)':32s} | {'文章数':>6s} | {'Z-Score AUROC':>14s} | {'稀疏探针 AUROC':>15s} | {'联合判据 AUROC':>15s} | {'未知博主AI概率均值':>18s}")
    print("-" * 112)
    loao_results = []
    for target_blogger in bloggers:
        # 训练集彻底剔除 target_blogger 的所有文章
        train_h = [r["text"] for r in rows if r["track"] == "human" and r["author"] != target_blogger and r["split"] == "dev"]
        train_a = [r["text"] for r in rows if r["track"] == "machine" and r["split"] == "dev"]
        b_loao = AHS.HumanBand(train_h, lm)
        v_loao, c_loao = AHS.fit_sparse_probe(
            [AHS.normalize_for_sparse(t) for t in train_h],
            [AHS.normalize_for_sparse(t) for t in train_a],
        )

        test_h_rows = [r for r in rows if r["author"] == target_blogger]
        z_h, p_h = [], []
        for r in test_h_rows:
            ft = b_loao._extract(r["text"])
            if not ft:
                continue
            z_h.append(AHS.score_four_dims(ft, b_loao)["composite_z"])
            p_h.append(float(c_loao.predict_proba(v_loao.transform([AHS.normalize_for_sparse(r["text"])]))[0, 1]))

        z_a, p_a = [], []
        for r in heldout_ai_rows:
            ft = b_loao._extract(r["text"])
            if not ft:
                continue
            z_a.append(AHS.score_four_dims(ft, b_loao)["composite_z"])
            p_a.append(float(c_loao.predict_proba(v_loao.transform([AHS.normalize_for_sparse(r["text"])]))[0, 1]))

        y_true = [0] * len(z_h) + [1] * len(z_a)
        auc_z = roc_auc_score(y_true, z_h + z_a)
        auc_p = roc_auc_score(y_true, p_h + p_a)
        comb_score = [0.5 * z + 1.5 * p for z, p in zip(z_h + z_a, p_h + p_a)]
        auc_comb = roc_auc_score(y_true, comb_score)

        print(f"{target_blogger:32s} | {len(z_h):6d} | {auc_z:14.4f} | {auc_p:15.4f} | {auc_comb:15.4f} | {statistics.mean(p_h):18.3f}")
        loao_results.append({
            "unseen_author": target_blogger,
            "n_articles": len(z_h),
            "auroc_z": round(float(auc_z), 4),
            "auroc_sparse": round(float(auc_p), 4),
            "auroc_combined": round(float(auc_comb), 4),
            "mean_human_ai_prob": round(statistics.mean(p_h), 3),
        })

    # ── 3. 首版中文长文去 AI 味方法排行榜 (Leaderboard v1.1) ──
    print("\n" + "=" * 105)
    print("LEADERBOARD v1.1: 各真人基准与去 AI 味改写方法同场竞技排行榜")
    print("=" * 105)
    root_res = BASE.parent
    method_corpora = {
        "🌟 真人基准: 6大技术博主合集 (Heldout 56篇)": [r["text"] for r in rows if r["split"] == "heldout" and r["domain"] == "tech_blog_human"],
        "🌟 真人基准: 陈皓 CoolShell (Pre-AI 13篇)": [r["text"] for r in rows if r["split"] == "heldout" and r["author"] == "chenhao_coolshell"],
        "🌟 真人基准: piglei & 云风 & 阮一峰 (29篇)": [r["text"] for r in rows if r["split"] == "heldout" and r["author"] in ("piglei", "yunfeng", "ruanyifeng")],
        "🛠️ C-LongSlop 处方引导微修 (Prescription Repair)": [Path(root_res / "output_v15_teacher/repaired_by_prescription.md").read_text(encoding="utf-8")],
        "⚙️ Atom-Writer v5 (原子约束生成 10篇)": [Path(p).read_text(encoding="utf-8") for p in sorted(glob.glob(str(root_res / "output_atomv5/article_*/atom_writer/final_article.md")))],
        "⚙️ HY4-v13 Atom-Writer (10篇)": [Path(p).read_text(encoding="utf-8") for p in sorted(glob.glob(str(root_res / "output_hy4v13/article_*/atom_writer/final_article.md")))],
        "📝 Few-Shot 示例改写基线 (HY4-v13 FewShot 10篇)": [Path(p).read_text(encoding="utf-8") for p in sorted(glob.glob(str(root_res / "output_hy4v13/article_*/fewshot/article.md")))],
        "⚙️ V15_Teacher (单向规则微修前)": [Path(root_res / "output_v15_teacher/atom_writer_v14/article.md").read_text(encoding="utf-8")],
        "🤖 原生 AI 基线: 公众号论文解读 (Heldout 45篇)": [r["text"] for r in rows if r["split"] == "heldout" and r["domain"] == "tech_media_ai"],
    }

    print(f"{'排名 & 方法 / 基准名称':44s} | {'N':>3s} | {'D1微观':>6s} | {'D2衔接':>6s} | {'D3骨架':>6s} | {'D4宏观':>6s} | {'防刷Z↓':>7s} | {'稀疏AI率↓':>9s} | {'HumanPass↑':>10s}")
    print("-" * 120)
    leaderboard = []
    for m_name, docs in method_corpora.items():
        items = []
        for t in docs:
            ft = band._extract(t)
            if not ft:
                continue
            dims = AHS.score_four_dims(ft, band)
            pr = float(clf.predict_proba(vec.transform([AHS.normalize_for_sparse(t)]))[0, 1])
            # 综合得分结合四维防刷 HumanPass (70%) 与 稀疏无偏探针人类度 (30%)
            combined_hp = round(0.7 * dims["humanpass_100"] + 0.3 * (1.0 - pr) * 100.0, 1)
            items.append({**dims, "prob": pr, "combined_hp": combined_hp})
        if not items:
            continue
        row = {
            "method": m_name,
            "n": len(items),
            "D1": round(statistics.mean(x["D1"] for x in items), 2),
            "D2": round(statistics.mean(x["D2"] for x in items), 2),
            "D3": round(statistics.mean(x["D3"] for x in items), 2),
            "D4": round(statistics.mean(x["D4"] for x in items), 2),
            "composite_z": round(statistics.mean(x["composite_z"] for x in items), 3),
            "sparse_ai_prob": round(statistics.mean(x["prob"] for x in items), 3),
            "humanpass_100": round(statistics.mean(x["humanpass_100"] for x in items), 1),
            "combined_score": round(statistics.mean(x["combined_hp"] for x in items), 1),
        }
        leaderboard.append(row)

    leaderboard.sort(key=lambda x: x["combined_score"], reverse=True)
    for rank, r in enumerate(leaderboard, 1):
        print(f"#{rank:02d} {r['method']:40s} | {r['n']:3d} | {r['D1']:6.2f} | {r['D2']:6.2f} | {r['D3']:6.2f} | {r['D4']:6.2f} | "
              f"{r['composite_z']:7.3f} | {r['sparse_ai_prob']:9.3f} | {r['combined_score']:10.1f}")

    out_report = {
        "per_author_heldout_audit": author_audit_table,
        "leave_one_author_out_cv": loao_results,
        "leaderboard_v1_1": leaderboard,
    }
    (BASE / "leaderboard/leaderboard_v1_1.json").write_text(json.dumps(out_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved full audit & leaderboard to {BASE / 'leaderboard/leaderboard_v1_1.json'}")


if __name__ == "__main__":
    main()
