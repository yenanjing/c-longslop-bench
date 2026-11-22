#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C-LongSlopBench 官方评测器 v1.0（Anti-Hacking Evaluator）
=========================================================
这是 Bench 的核心评测引擎，从 pilot 实验脚本固化而来，具有以下契约：
1. 全部探针 100% 确定性（零 LLM 裁判、固定随机种子），毫秒级运行
2. 四锁防刷评分：双边人类分布带 [P10,P90] + 成对张力锁 + 标点拐杖守恒锁 + 事实锚点红线
3. 输出标准《诊断处方单》（精确到段/句/触发片段/修复方向），可直接喂给改写模型
4. 双极稀疏特征探针（中文版 DetectorScope 复现）：字符 2-4 gram + ElasticNet 逻辑回归

用法：
    python3 evaluators/antihack_scorer.py --input data/processed/c_longslop_bench_v1.jsonl \
        --split heldout --output results/heldout_scores.json

输出 JSON：
{
  "per_sample": [ {sample_id, domain, track, D1..D4, composite_z, humanpass_100,
                   sparse_ai_prob, factual_gate, verdict, prescriptions_count}, ... ],
  "summary": {domain 级 / track 级聚合统计},
  "band_definition": {指标名: {mean, std, p10, p90}}  # 人类分布带定义，保证可复现
}
"""
import argparse
import glob
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import jieba
import jieba.posseg as pseg
import numpy as np

# ── 自包含核心探针导入（零外部父目录依赖） ──
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probes_core import (  # noqa: E402
    clean_markdown_header, split_paragraphs, split_sentences,
    compute_probes, SPEECH_BUFFERS, MODAL_PARTICLES,
    ChineseReferenceLM, compute_uid_and_pmi_probes,
)
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


# ──────────────────────────── 人类分布带（Lock 1） ────────────────────────────

class HumanBand:
    """从 dev 集人类长文样本计算每个指标的 [mean, std, P10, P90]，锚定人类分布。支持传入原文或预计算特征字典。"""

    def __init__(self, human_texts: List[Any], lm: ChineseReferenceLM):
        # 优先使用 >=1000 字的深度人类长文拟合长文分布带（支持 str 或含 precomputed_4dim_features 的 dict）
        feats = []
        long_items = [
            it for it in human_texts
            if (isinstance(it, str) and len(it) >= 1000)
            or (isinstance(it, dict) and it.get("char_len", 0) >= 1000)
        ]
        use_items = long_items if len(long_items) >= 20 else human_texts
        for it in use_items:
            if isinstance(it, dict) and it.get("precomputed_4dim_features"):
                feats.append(it["precomputed_4dim_features"])
            elif isinstance(it, dict) and it.get("text"):
                f = self._extract(it["text"])
                if f:
                    feats.append(f)
            elif isinstance(it, str):
                f = self._extract(it)
                if f:
                    feats.append(f)
        self.keys = list(feats[0].keys())
        self.band = {}
        for k in self.keys:
            vals = sorted(d[k] for d in feats)
            self.band[k] = {
                "mean": round(statistics.mean(vals), 4),
                "std": round(max(0.015, statistics.stdev(vals)), 4),
                "p10": vals[max(0, int(len(vals) * 0.10))],
                "p90": vals[min(len(vals) - 1, int(len(vals) * 0.90))],
            }

    def _extract(self, raw: str) -> Dict[str, float]:
        p1 = compute_probes(raw)
        p2 = compute_uid_and_pmi_probes(lm_global, raw)
        if not p1 or not p2:
            return {}
        text = clean_markdown_header(raw)
        n = max(1, len(text))
        paren = len(re.findall(r"[（(][^）)]{1,25}[）)]", text))
        colon = len(re.findall(r"[^0-9][：:]", text))
        em = text.count("——")
        enum_hits = len(re.findall(r"(?:[（(]\s*\d{1,2}\s*[）)]|[•■◆]|[一二三四五六七八九十]+、)", text))
        colon_prompt = len(re.findall(r"(?:^|[。！？\n])[^，。！？\n：:《》]{2,14}[：:]", text))
        crutch = (em * 2.0 + paren * 2.2 + colon * 1.3 + enum_hits * 2.2 + colon_prompt * 2.5) / n * 1000.0
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        short_para_ratio = sum(1 for p in paras if len(p) <= 22) / max(1, len(paras))
        # 表演性元话语与强行第一人称自述密度（抓 Persona Hacking / 过度网感注入，对所有方法一视同仁计算双边 Z）
        perf_words = (
            "说实话", "老实说", "坦率讲", "讲道理", "有一说一", "我个人觉得",
            "我是真", "这里插一句", "顺便提一嘴", "哦对了", "说句实在话", "不夸张地说",
            "人麻了", "有点麻", "一坨", "神之一手", "简单粗暴", "巴拉巴拉", "碎碎念",
            "诈尸", "装死", "干崩", "眼前一黑", "真香", "吓尿", "笑蒙", "邪修"
        )
        perf_hits = sum(text.count(w) for w in perf_words)
        ego_hits = len(re.findall(r"我(?:个人)?(?:当时|之前|那台|上周|真有点|挺爱|最烦|烧了|搓了)", text))
        persona_hack_rate = (perf_hits * 1.5 + ego_hits * 1.5) / n * 1000.0
        return {
            **p1, **p2,
            "M1_punct_crutch_total": round(crutch, 3),
            "M1_colon_prompt_rate": round(colon_prompt / n * 1000.0, 3),
            "M1_persona_hacking_rate": round(persona_hack_rate, 3),
            "M2_short_para_ratio": round(short_para_ratio, 3),
        }

    def two_sided_z(self, key: str, val: float) -> float:
        """双边惩罚：带内仅 0.15 倍软化惩罚，越出 [P10,P90] 后按标准差距离全额惩罚。"""
        b = self.band[key]
        if b["p10"] <= val <= b["p90"]:
            return 0.15 * abs(val - b["mean"]) / b["std"]
        if val < b["p10"]:
            return (b["p10"] - val) / b["std"] + 0.15 * abs(b["p10"] - b["mean"]) / b["std"]
        return (val - b["p90"]) / b["std"] + 0.15 * abs(b["p90"] - b["mean"]) / b["std"]


lm_global = None  # 模块级引用，供 HumanBand 内部调用


# ──────────────────────────── 事实锚点红线（Lock 3） ────────────────────────────

def extract_factual_anchors(text: str) -> Dict[str, set]:
    t = clean_markdown_header(text)
    # 剥离列表编号括号如 (1) (2) （1） （2），避免将排版序号误判为实验事实数字
    t = re.sub(r"[（(]\s*\d{1,2}\s*[）)]", " ", t)
    numbers = set(re.findall(r"\d+(?:\.\d+)?%?", t))
    eng_terms = set(w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_\-\.]{2,}", t))
    return {"numbers": numbers, "eng_terms": eng_terms}


def factual_gate(orig: str, cand: str) -> Dict[str, Any]:
    oa, ca = extract_factual_anchors(orig), extract_factual_anchors(cand)
    critical_nums = {n for n in oa["numbers"] if "." in n or "%" in n or len(n) >= 2}
    missing = critical_nums - ca["numbers"]
    total = max(1, len(oa["numbers"]) + len(oa["eng_terms"]))
    retention = 1.0 - (len(oa["numbers"] - ca["numbers"]) + len(oa["eng_terms"] - ca["eng_terms"])) / total
    return {"passed": len(missing) == 0 and retention >= 0.85,
            "anchor_retention": round(retention, 3),
            "missing_critical_numbers": sorted(missing)}


# ──────────────────────────── 四维防刷 Z 评分 ────────────────────────────

def extract_compensatory_probes(raw: str) -> Dict[str, float]:
    """提取技术长文从零生成中的三类隐蔽代偿特征：一逗到底粘连长句、平均句长膨胀、第三人称论文汇报腔。"""
    t = clean_markdown_header(raw)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
    n = max(1, len(t))
    sents = split_sentences(t)
    if not sents:
        return {
            "M2_overlong_sent_ratio": 0.0,
            "M2_mean_sent_len": 0.0,
            "M1_reportese_rate": 0.0,
            "M1_reader_hook_rate": 0.0,
        }
    sent_lens = [len(s) for s in sents]
    overlong_cnt = sum(
        1 for s in sents
        if len(s) >= 85 or (len(s) >= 65 and (s.count("，") >= 4 or s.count("、") >= 2))
    )
    rep_words = ("作者", "论文中", "原论文", "研究者", "该研究", "研究团队", "实验结果", "实验表明", "结果表明")
    rep_hits = sum(t.count(w) for w in rep_words)
    author_starts = sum(
        1 for s in sents if re.match(r"^(?:作者|论文|研究者|研究团队|本文|这项工作|该工作)", s)
    )
    rep_rate = (rep_hits + author_starts * 1.5) / n * 1000.0
    reader_hooks = ("你", "我们", "其实", "换句话说", "说到底", "也就是", "先看", "再看", "为什么", "怎么")
    hook_hits = sum(t.count(w) for w in reader_hooks) + t.count("？")
    return {
        "M2_overlong_sent_ratio": round(overlong_cnt / len(sents), 3),
        "M2_mean_sent_len": round(statistics.mean(sent_lens), 2),
        "M1_reportese_rate": round(rep_rate, 3),
        "M1_reader_hook_rate": round(hook_hits / n * 1000.0, 3),
    }


def score_four_dims(
    feats: Dict[str, float],
    band: HumanBand,
    compensatory_lock: bool = False,
) -> Dict[str, float]:
    # D1 微观：气口缓冲双边带(零气口赤字与 >6.0/千字过载均重罚) + 标点/冒号提示体拐杖 + 名词堆叠 + 表演性口癖越界
    z_buf = (
        max(0.0, 0.85 - feats["M1_speech_buffer_rate"]) * 1.15
        + max(0.0, feats["M1_speech_buffer_rate"] - 6.0) * 1.5
    )
    z_crutch = (
        band.two_sided_z("M1_punct_crutch_total", feats["M1_punct_crutch_total"])
        + band.two_sided_z("M1_colon_prompt_rate", feats.get("M1_colon_prompt_rate", 0.0)) * 0.9
        + feats["M1_em_dash_rate"] * 0.8
    )
    z_noun = band.two_sided_z("M1_noun_stack_rate", feats["M1_noun_stack_rate"])
    z_persona = band.two_sided_z("M1_persona_hacking_rate", feats.get("M1_persona_hacking_rate", 0.0))
    d1 = (z_buf + z_crutch * 1.3 + z_noun * 0.7 + z_persona * 0.8) / 2.8

    # D2 中观：句首惊奇冲击 + 句内信息波动赤字 + 跨句搭扣 + 以旧带新违规 + 剁碎短句/短段耦合锁
    z_onset = band.two_sided_z("UID_onset_minus_end_surp", feats["UID_onset_minus_end_surp"]) + max(0.0, feats["UID_onset_minus_end_surp"] - 1.75) * 1.1
    z_wave = max(0.0, 0.072 - feats["UID_sentence_wave_cv"]) * 14.0 + band.two_sided_z("UID_sentence_wave_cv", feats["UID_sentence_wave_cv"]) * 0.5
    z_coh = band.two_sided_z("M2_cohesion_break_rate", feats["M2_cohesion_break_rate"])
    z_o2n = band.two_sided_z("M2_old_to_new_violation", feats["M2_old_to_new_violation"])
    z_chop = band.two_sided_z("M2_short_para_ratio", feats.get("M2_short_para_ratio", 0.0))
    coupling = 0.6 * max(0.0, z_coh - 0.3) * max(0.0, z_onset - 0.5)
    d2 = statistics.mean([z_onset, z_wave, z_coh, z_o2n, z_chop]) + coupling

    # 长文生成代偿防刷锁（Lock 1 气口赤字与汇报腔越界 + Lock 2b 右半边「一逗到底粘连长句锁」）
    if compensatory_lock or "M2_overlong_sent_ratio" in feats:
        z_buf_def = max(0.0, (feats.get("M1_buffer_deficit", 1.0) - 2.0) / 1.2)
        z_rep = max(0.0, (feats.get("M1_reportese_rate", 0.0) - 0.90) / 0.65)
        d1 = d1 + (z_buf_def * 0.45 + z_rep * 0.40) / 2.8
        z_overlong = max(0.0, (feats.get("M2_overlong_sent_ratio", 0.0) - 0.18) / 0.07)
        z_mslen = max(0.0, (feats.get("M2_mean_sent_len", 38.0) - 48.5) / 5.5)
        d2 = d2 + 0.35 * z_overlong + 0.25 * z_mslen

    # D3 句式骨架：滑动窗口段首起手同质化 + 段落字数均匀度 + 词汇多样性双边带
    d3 = statistics.mean([
        band.two_sided_z("M3_opener_homogeneity", feats["M3_opener_homogeneity"]) + feats["M3_opener_homogeneity"] * 2.0,
        band.two_sided_z("M2_para_flatness", feats["M2_para_flatness"]),
        band.two_sided_z("M3_lexical_diversity_MATTR", feats["M3_lexical_diversity_MATTR"]),
    ])

    # D4 宏观：跨段远距回环复述 + 远距回声均值
    d4 = statistics.mean([
        band.two_sided_z("M3_macro_redundancy", feats["M3_macro_redundancy"]),
        band.two_sided_z("M3_mean_distant_echo", feats["M3_mean_distant_echo"]),
    ])

    comp = statistics.mean([d1, d2, d3, d4])
    return {"D1": round(d1, 3), "D2": round(d2, 3), "D3": round(d3, 3), "D4": round(d4, 3),
            "composite_z": round(comp, 3),
            "humanpass_100": round(max(0.0, 100.0 * math.exp(-0.35 * comp)), 1)}


# ──────────────────── 中文版 DetectorScope 双极稀疏探针（Lock 4） ────────────────────

def normalize_for_sparse(raw: str) -> str:
    """净化：英文数字串合并掩码为单个 E，去空白，统一引号 —— 屏蔽专名与数字泄漏。"""
    t = clean_markdown_header(raw)
    t = re.sub(r"[A-Za-z0-9_\-\.%]+", "E", t)
    t = re.sub(r"E{2,}", "E", t)
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"[\"“”『』*#`]", "「", t)
    return t


def fit_sparse_probe(human_norms: List[str], ai_norms: List[str]):
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=4, max_df=0.85,
                          max_features=3000, sublinear_tf=True)
    X = vec.fit_transform(human_norms + ai_norms)
    y = np.array([0] * len(human_norms) + [1] * len(ai_norms))
    clf = LogisticRegression(penalty="elasticnet", l1_ratio=0.35, C=2.0, solver="saga",
                             class_weight="balanced", max_iter=2000, random_state=42)
    clf.fit(X, y)
    return vec, clf


def sparse_cross_val(human_norms: List[str], ai_norms: List[str]) -> float:
    """5-Fold CV AUROC，供 Bench 报告披露。"""
    X_all = human_norms + ai_norms
    y = np.array([0] * len(human_norms) + [1] * len(ai_norms))
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=4, max_df=0.85,
                          max_features=3000, sublinear_tf=True)
    X = vec.fit_transform(X_all)
    clf = LogisticRegression(penalty="elasticnet", l1_ratio=0.35, C=2.0, solver="saga",
                             class_weight="balanced", max_iter=2000, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ypred = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return round(float(roc_auc_score(y, ypred)), 4)


# ──────────────────── 自动处方单生成（Prescription Engine） ────────────────────

PRONOUN_HOOKS = ("这", "那", "他", "她", "它", "其", "该", "此", "但", "可", "而", "所以",
                 "结果", "毕竟", "也", "而且", "再", "于是", "反过来", "说白了", "其实")


def is_structural_or_meta_para(p: str) -> bool:
    """判断段落是否为 Markdown 标题、图注 (<center>)、公式块 ($$)、代码块、表格或论文元信息行。"""
    s = p.strip()
    if not s or len(s) < 18:
        return True
    if s.startswith(("#", "![", "<center", "<img", "<div", "$$", "```", "|", ">", "---")):
        return True
    if re.match(
        r"^[·•\-\*\s]*(?:一句话讲清楚|论文标题|论文题目|论文链接|项目链接|开源链接|代码链接|作者|机构|来源|arXiv)",
        s,
        re.I,
    ):
        return True
    if re.match(r"^(?:Figure|Fig\.|Table|图|表)\s*\d+", s, re.I):
        return True
    return False


def generate_prescriptions(raw: str, include_compensatory: bool = True) -> List[Dict[str, Any]]:
    text = clean_markdown_header(raw)
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    rx, seen = [], set()
    no_buf_run = 0
    for p_idx, p in enumerate(paras):
        if is_structural_or_meta_para(p):
            continue
        sents = split_sentences(p)
        issues = []
        if "——" in p:
            for si, s in enumerate(sents):
                if "——" in s:
                    issues.append({"type": "EM_DASH_CRUTCH", "sent_idx": si,
                                   "evidence": s.replace("——", "[[——]]"),
                                   "instruction": "去破折号，用逗号分句或自然动词（「说白了就是」「相当于」）把解释顺下来；严禁换成括号或冒号。"})
        parens = re.findall(r"（[A-Za-z0-9_\-\s]{2,30}）|\([A-Za-z0-9_\-\s]{2,30}\)", p)
        if len(parens) >= 2:
            issues.append({"type": "PARENTHETICAL_ENUM_TIC", "sent_idx": 0,
                           "evidence": re.sub(r"(（[A-Za-z0-9_\-\s]{2,30}）|\([A-Za-z0-9_\-\s]{2,30}\))", r"[[\1]]", p),
                           "instruction": f"删除句内多余的英文缩写/编号括号 {parens[:5]}，首次出现保留即可，后续直接用中文简称指代。"})
        if re.search(r"(?:^|[。！？\n])[^，。！？\n：:《》]{2,14}：", p) and len(p) >= 35:
            issues.append({"type": "COLON_PROMPT_TIC", "sent_idx": 0,
                           "evidence": p.replace("：", "[[：]]", 1),
                           "instruction": "消除『总结词：展开』冒号提示体，改为自然主谓宾陈述句（用『核心在于』『说的是』『做法是』衔接）。"})
        # 抓句号后直接用英文模型名/术语硬起手（稀疏探针高权重的 `。E` 病灶）
        abrupt_eng = re.findall(r"[。！？]\s*([A-Za-z][A-Za-z0-9_\-]{2,18})\s*[\u4e00-\u9fff]", p)
        if len(abrupt_eng) >= 1:
            issues.append({"type": "ABRUPT_ENGLISH_SUBJECT_ONSET", "sent_idx": 0,
                           "evidence": f"[[。{abrupt_eng[0]}...]]",
                           "instruction": f"消除句号后直接拿英文名『{abrupt_eng[0]}』硬顶句首的机器感：在句首垫一个承接上句的指代或顺承短语（如『在此基础上，{abrupt_eng[0]}』『为了做到这一点，它』）。"})
        if sents and p_idx >= 2:
            s0 = sents[0]
            cw0 = [w for w, pos in pseg.cut(s0) if pos.startswith("n") and len(w) >= 2]
            has_old = any(w in seen for w in cw0[:2])
            has_hook = s0.startswith(PRONOUN_HOOKS)
            if not has_old and not has_hook and len(s0) <= 35:
                issues.append({"type": "ABRUPT_ONSET_BREAK", "sent_idx": 0,
                               "evidence": f"[[{s0}]]",
                               "instruction": "段首硬抛新概念（违反以旧带新律）。句首补承接上一段末尾概念的过渡词或代词搭扣，遵循『句首接旧知、句尾抛新知』。"})
        if int(len(paras) * 0.30) <= p_idx <= int(len(paras) * 0.70):
            has_num = bool(re.search(r"\d+|[「」“”]", p))
            has_buf = any(b in p for b in SPEECH_BUFFERS) or any(mp in p for mp in MODAL_PARTICLES)
            if not has_num and not has_buf and len(p) >= 70:
                issues.append({"type": "MID_SECTION_ABSTRACT_DRIFT", "sent_idx": 0,
                               "evidence": f"[[{p[:35]}...]]",
                               "instruction": "中段全抽象机制名词堆叠且无气口。拆开连续名词定语，补一句直白的『人话降维解释』或指代承接。"})

        if include_compensatory:
            # 代偿病灶 1：用逗号把多句独立意思粘成 >=92 字超长流水句（或 >=78 字且逗号 >=4）以逃避句号/冒号检测
            for si, s in enumerate(sents):
                if len(s) >= 92 or (len(s) >= 78 and s.count("，") >= 4):
                    issues.append({
                        "type": "COMMA_SPLICE_OVERLONG",
                        "sent_idx": si,
                        "evidence": f"[[({len(s)}字/{s.count('，')}逗) {s[:42]}...]]",
                        "instruction": f"消除『一逗到底』粘连长句（本句长达 {len(s)} 字，含 {s.count('，')} 个逗号）：在语义转折或新分句处改打句号，拆成 2–3 个 30–50 字的短句；新句句首用自然指代或动作承接，禁止靠不打句号来逃避 `。E` 检测。",
                    })
                    break
            # 代偿病灶 2：第三人称论文汇报腔（句首硬顶「作者/论文/研究者」或单段密集出现）
            rep_hits = re.findall(r"(?:作者|论文中|原论文|研究者|研究团队|实验表明|结果表明)", p)
            author_starts = [s for s in sents if re.match(r"^(?:作者|论文|研究者|研究团队|本文)", s)]
            if len(rep_hits) >= 2 or author_starts:
                ev = author_starts[0][:36] if author_starts else f"段内汇报词 {rep_hits}"
                issues.append({
                    "type": "REPORTESE_AUTHOR_SUBJECT",
                    "sent_idx": 0,
                    "evidence": f"[[{ev}]]",
                    "instruction": "消除『作者做了…/论文指出…』的审稿汇报腔：直接把机制、任务或实验现象本身作为主语（如把『作者在两个任务上做了 14 组对照』改为『拿两个任务上的 14 组预设对照来看』）。",
                })
            # 代偿病灶 3：连续多段零气口赤字（每满 4 段触发一次定位，避免连续刷屏）
            has_any_buf = (
                any(w in p for w in ("其实", "换句话说", "说到底", "也就是", "先看", "再看", "说白了", "你看"))
                or ("？" in p)
                or ("你" in p)
            )
            if not has_any_buf:
                no_buf_run += 1
                if no_buf_run >= 4 and len(p) >= 90:
                    issues.append({
                        "type": "PARAGRAPH_BUFFER_DEFICIT",
                        "sent_idx": 0,
                        "evidence": f"[[连续 {no_buf_run} 段零气口: {p[:32]}...]]",
                        "instruction": "连续多段零讲课气口，读感干瘪机械：在句中解释或转折处自然补入 1 处气口词（如『其实』『换句话说』『说到底』『也就是』），放在句中不要放段首。",
                    })
                    no_buf_run = 0
            else:
                no_buf_run = 0

        for s in sents:
            for w, pos in pseg.cut(s):
                if pos.startswith("n") and len(w) >= 2:
                    seen.add(w)
        if issues:
            rx.append({"para_idx": p_idx, "issues": issues,
                       "prev_para_tail": paras[p_idx - 1][-40:] if p_idx else ""})
    return rx


# ──────────────────────────── 主评测入口 ────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/c_longslop_bench_v1.jsonl")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--output", default="results/heldout_scores.json")
    ap.add_argument("--with-prescriptions", action="store_true")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent.parent
    rows = [json.loads(l) for l in open(base / args.input, encoding="utf-8")]

    # 1. 用 dev 集人类样本拟合人类分布带（heldout 永不参与，兼容含完整 text 或仅含预计算特征的合规公开版 JSONL）
    dev_human_rows = [r for r in rows if r["split"] == "dev" and r["track"] == "human"]
    dev_ai_rows = [r for r in rows if r["split"] == "dev" and r["track"] == "machine"]

    lm_global = ChineseReferenceLM()
    lm_global.fit_hc3_human()
    globals()["lm_global"] = lm_global

    print("[1/4] 拟合人类分布带 (dev human) ...")
    band = HumanBand(dev_human_rows, lm_global)

    print("[2/4] 拟合双极稀疏探针 (dev human vs dev machine, 5-Fold CV) ...")
    hn = [r.get("masked_sparse_text") or normalize_for_sparse(r["text"]) for r in dev_human_rows]
    an = [r.get("masked_sparse_text") or normalize_for_sparse(r["text"]) for r in dev_ai_rows]
    cv_auc = sparse_cross_val(hn, an)
    vec, clf = fit_sparse_probe(hn, an)

    print(f"[3/4] 扫描 {args.split} 集 ({sum(1 for r in rows if r['split'] == args.split)} 篇) ...")
    per_sample = []
    for r in rows:
        if r["split"] != args.split:
            continue
        feats_all = r.get("precomputed_4dim_features") or (band._extract(r["text"]) if r.get("text") else None)
        if not feats_all:
            continue
        dims = score_four_dims(feats_all, band)
        masked_s = r.get("masked_sparse_text") or normalize_for_sparse(r["text"])
        x = vec.transform([masked_s])
        ai_prob = round(float(clf.predict_proba(x)[0, 1]), 3)
        rx = generate_prescriptions(r["text"]) if (args.with_prescriptions and r.get("text")) else None
        verdict = "HUMANLIKE" if (dims["composite_z"] < 0.6 and ai_prob < 0.5) else (
            "AI_FLAVOR" if (dims["composite_z"] > 1.2 or ai_prob > 0.6) else "GRAY_ZONE")
        per_sample.append({
            "sample_id": r["sample_id"], "domain": r["domain"], "track": r["track"],
            **dims, "sparse_ai_prob": ai_prob, "verdict": verdict,
            "char_len": r["char_len"],
            **({"prescriptions_count": len(rx), "prescriptions": rx} if rx else {}),
        })

    # 4. 聚合
    def agg(key, sub):
        vals = [s[key] for s in sub]
        return round(statistics.mean(vals), 3) if vals else None

    summary = {}
    for track in ["human", "machine"]:
        sub = [s for s in per_sample if s["track"] == track]
        if sub:
            summary[track] = {
                "n": len(sub),
                "mean_composite_z": agg("composite_z", sub),
                "mean_humanpass": agg("humanpass_100", sub),
                "mean_sparse_ai_prob": agg("sparse_ai_prob", sub),
                "D1": agg("D1", sub), "D2": agg("D2", sub), "D3": agg("D3", sub), "D4": agg("D4", sub),
                "humanlike_rate": round(sum(1 for s in sub if s["verdict"] == "HUMANLIKE") / len(sub), 3),
                "aiflavor_rate": round(sum(1 for s in sub if s["verdict"] == "AI_FLAVOR") / len(sub), 3),
            }

    # 判别 AUROC（composite_z 与 sparse_prob 各自作为单判据）
    ys, z_scores, probs = [], [], []
    for s in per_sample:
        ys.append(1 if s["track"] == "machine" else 0)
        z_scores.append(s["composite_z"])
        probs.append(s["sparse_ai_prob"])
    auroc_z = round(float(roc_auc_score(ys, z_scores)), 4) if len(set(ys)) > 1 else None
    auroc_p = round(float(roc_auc_score(ys, probs)), 4) if len(set(ys)) > 1 else None

    out = {
        "bench_version": "C-LongSlopBench v1.0",
        "eval_split": args.split,
        "sparse_probe_5fold_cv_auc_dev": cv_auc,
        "discrimination_auroc_composite_z": auroc_z,
        "discrimination_auroc_sparse_prob": auroc_p,
        "summary": summary,
        "per_sample_count": len(per_sample),
        "per_sample": per_sample,
        "band_definition": band.band,
    }
    outp = base / args.output
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"\n===== C-LongSlopBench {args.split} 评测摘要 =====")
    print(f"稀疏探针 dev 5-Fold CV AUROC: {cv_auc}")
    print(f"heldout 判别 AUROC (composite_z): {auroc_z} | (sparse_prob): {auroc_p}")
    for track, st in summary.items():
        print(f"  [{track:7s}] n={st['n']:3d} | Z={st['mean_composite_z']:.3f} | "
              f"HumanPass={st['mean_humanpass']:.1f} | AI_prob={st['mean_sparse_ai_prob']:.3f} | "
              f"HUMANLIKE率={st['humanlike_rate']:.1%}")
    print(f"Saved -> {outp}")


if __name__ == "__main__":
    main()
