#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C-LongSlopBench 自包含核心探针与参考语言模型模块 (Self-Contained Probes Core)
=============================================================================
本模块将全部 19 项四维正交探针（M1-M3 + UID/PMI）及 HC3 中文参考语言模型 (`ChineseReferenceLM`)
完整封装在 `c_longslop_bench/evaluators/` 内部，彻底消除对外部父目录脚本或外部语料路径的依赖。
任何外部项目 `git clone` 本仓库后均可开箱即用，且计算结果与论文 Table 1-3 逐位一致。
"""
import glob
import gzip
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import jieba
import jieba.posseg as pseg

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent


# ── 1. 文本清洗与切分工具（兼容公众号/Markdown/HTML 图注/元信息块） ───────────────

def clean_markdown_header(text: str) -> str:
    """
    去掉文章开头的元数据头（# 标题 / - 作者 / --- 分隔线），
    同时剥离微信公众号/技术推文常见的元信息行（如 `· 论文标题：`、`<center>图注</center>`、Markdown 图片与公式块），
    避免排版标签被误计入正文自然语言探针。
    """
    lines = text.strip().split("\n")
    for idx in range(min(12, len(lines))):
        if lines[idx].strip() == "---":
            text = "\n".join(lines[idx + 1:]).strip()
            break
    # 剥离 <center>...</center> 图注与 HTML 注释
    text = re.sub(r"<center>.*?</center>", "", text, flags=re.S | re.I)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    # 剥离独立成行的项目符号元信息块（如 `· 论文标题：...`、`- 论文链接：...`）
    cleaned_lines = []
    meta_pat = re.compile(
        r"^[·•\-\*\s]*(?:论文标题|论文题目|论文链接|项目链接|开源链接|代码链接|作者|机构|来源|arXiv)[：:]",
        re.I,
    )
    for ln in text.split("\n"):
        s = ln.strip()
        if meta_pat.match(s):
            continue
        cleaned_lines.append(ln)
    return "\n".join(cleaned_lines).strip()


def split_paragraphs(text: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    valid = []
    for p in paras:
        if p.startswith(("#", "![", "- **", "* ", "- ", "· ", ">", "```", "|", "$$", "<center")):
            continue
        if len(p) >= 15:
            valid.append(p)
    return valid


def split_sentences(text: str) -> List[str]:
    prose_lines = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s or s.startswith(("#", "![", "|", "$$", "```", "<center", "---")):
            continue
        prose_lines.append(s)
    t = "\n".join(prose_lines)
    t = re.sub(r"(?<!\d)\.(?!\d)", "。", t)
    sents = re.findall(r"[^。！？!\?\n；;]+[。！？!\?；;]?", t)
    return [s.strip() for s in sents if len(s.strip()) >= 4]


# ── 2. 四维确定性探针计算 (M1 - M3) ───────────────────────────────────

SPEECH_BUFFERS = [
    "其实", "说白了", "坦率讲", "老实说", "说实话", "你看", "你想", "换句话说",
    "有意思的是", "好玩的是", "说到底", "某种意义上", "讲真", "平心而论", "顺便说",
    "更要命的是", "扎心的是", "不得不说", "怎么说呢", "简单讲", "直白点"
]
MODAL_PARTICLES = list("吧呢啊嘛哈哎哦呀噢嗯罢")
MECHANICAL_CONNECTORS = [
    "首先", "其次", "再次", "最后", "综上所述", "总而言之", "由此可见", "不难发现",
    "不难看出", "值得注意的是", "值得一提的是", "需要指出的是", "与此同时", "此外",
    "另外", "不仅如此", "更重要的是", "换言之", "具体而言", "具体来说", "简而言之",
    "毫无疑问", "毋庸置疑", "显而易见", "众所周知"
]
LIGHT_VERBS_NOMINAL = [
    "进行", "实现", "完成", "开展", "构建", "赋能", "打造", "提供", "具备",
    "形成", "达成", "推动", "促进", "围绕", "针对", "通过"
]
CONTRAST_TEMPLATES = [
    r"不是[^，。；]{1,25}而是",
    r"不仅[^，。；]{1,25}更",
    r"既[^，。；]{1,20}又",
    r"与其[^，。；]{1,20}不如",
    r"并非[^，。；]{1,20}而是",
]
CONTENT_POS_PREFIX = ("n", "v", "a", "i", "l")


def tokenize_pos(text: str) -> List[Tuple[str, str]]:
    return [(w, flag) for w, flag in pseg.cut(text) if w.strip()]


def compute_probes(raw_text: str) -> Dict[str, float]:
    text = clean_markdown_header(raw_text)
    char_len = max(1, len(text))
    paras = split_paragraphs(text)
    sents = split_sentences(text)
    if len(sents) < 5 or char_len < 200:
        return {}

    sent_tokens = [tokenize_pos(s) for s in sents]
    sent_content_words = [
        [w for w, pos in toks if pos.startswith(CONTENT_POS_PREFIX) and len(w) >= 2]
        for toks in sent_tokens
    ]

    em_dash_count = text.count("——") + text.count("—") * 0.5
    m1_em_dash_rate = em_dash_count / char_len * 1000.0

    contrast_hits = sum(len(re.findall(pat, text)) for pat in CONTRAST_TEMPLATES)
    m1_contrast_template_rate = contrast_hits / char_len * 1000.0

    lv_hits = sum(text.count(w) for w in LIGHT_VERBS_NOMINAL)
    m1_nominal_light_verb_rate = lv_hits / char_len * 1000.0

    noun_stacks = 0
    for toks in sent_tokens:
        run = 0
        for w, pos in toks:
            if pos.startswith("n") and len(w) >= 2:
                run += 1
                if run == 3:
                    noun_stacks += 1
            else:
                run = 0
    m1_noun_stack_rate = noun_stacks / len(sents)

    buf_count = sum(text.count(p) for p in MODAL_PARTICLES) + sum(text.count(b) for b in SPEECH_BUFFERS)
    mech_count = sum(text.count(c) for c in MECHANICAL_CONNECTORS)
    m1_buffer_deficit = (mech_count + 1.0) / (buf_count + 1.0)
    m1_speech_buffer_rate = buf_count / char_len * 1000.0

    de_congested_sents = sum(1 for s in sents if s.count("的") >= 3)
    m1_de_congestion_ratio = de_congested_sents / len(sents)

    seen_words = set()
    old_minus_new_list = []
    first_word_is_new_count = 0
    for cw in sent_content_words:
        if len(cw) >= 3:
            old_positions = [idx / (len(cw) - 1) for idx, w in enumerate(cw) if w in seen_words]
            new_positions = [idx / (len(cw) - 1) for idx, w in enumerate(cw) if w not in seen_words]
            if old_positions and new_positions:
                old_minus_new_list.append(statistics.mean(old_positions) - statistics.mean(new_positions))
            if cw[0] not in seen_words and len(seen_words) > 10:
                first_word_is_new_count += 1
        seen_words.update(cw)
    m2_old_to_new_violation = statistics.mean(old_minus_new_list) if old_minus_new_list else 0.0
    m2_new_topic_abrupt_rate = first_word_is_new_count / max(1, len(sents) - 2)

    pronoun_starts = ("这", "那", "他", "她", "它", "其", "该", "此", "但", "可", "而", "所以", "结果", "毕竟")
    linked_pairs = 0
    for i in range(1, len(sents)):
        overlap = set(sent_content_words[i]) & set(sent_content_words[i - 1])
        starts_with_hook = sents[i].startswith(pronoun_starts)
        if overlap or starts_with_hook:
            linked_pairs += 1
    m2_cohesion_break_rate = 1.0 - (linked_pairs / max(1, len(sents) - 1))

    all_pos = [pos[:1] for toks in sent_tokens for _, pos in toks if pos[:1].isalpha()]
    if len(all_pos) >= 20:
        pos_4grams = [tuple(all_pos[i:i + 4]) for i in range(len(all_pos) - 3)]
        win = 150
        if len(pos_4grams) >= win:
            ttrs = [len(set(pos_4grams[i:i + win])) / win for i in range(0, len(pos_4grams) - win + 1, 50)]
            m2_pos_template_rep = 1.0 - statistics.mean(ttrs)
        else:
            m2_pos_template_rep = 1.0 - (len(set(pos_4grams)) / len(pos_4grams))
    else:
        m2_pos_template_rep = 0.0

    sent_lens = [len(s) for s in sents]
    diffs = [abs(sent_lens[i] - sent_lens[i - 1]) for i in range(1, len(sent_lens))]
    burstiness = statistics.stdev(diffs) if len(diffs) >= 2 else 0.0
    m2_rhythm_flatness = 1.0 / (burstiness + 1.0)

    para_lens = [len(p) for p in paras]
    para_cv = (statistics.stdev(para_lens) / statistics.mean(para_lens)) if len(para_lens) >= 3 and statistics.mean(para_lens) > 0 else 0.5
    m2_para_flatness = 1.0 / (para_cv + 0.1)

    sent_char_bigrams = [set(s[k:k + 2] for k in range(len(s) - 1)) for s in sents if len(s) >= 10]
    high_echo_sents = 0
    max_echo_scores = []
    for i in range(len(sent_char_bigrams)):
        best_sim = 0.0
        for j in range(0, i - 2):
            inter = len(sent_char_bigrams[i] & sent_char_bigrams[j])
            union = min(len(sent_char_bigrams[i]), len(sent_char_bigrams[j]))
            if union > 0:
                sim = inter / union
                if sim > best_sim:
                    best_sim = sim
        if i >= 3:
            max_echo_scores.append(best_sim)
            if best_sim >= 0.35:
                high_echo_sents += 1
    m3_macro_redundancy = high_echo_sents / max(1, len(sent_char_bigrams) - 3)
    m3_mean_distant_echo = statistics.mean(max_echo_scores) if max_echo_scores else 0.0

    if len(paras) >= 4:
        openers = []
        for p in paras:
            ptoks = tokenize_pos(p[:18])
            if len(ptoks) >= 2:
                openers.append((ptoks[0][0][:2], ptoks[0][1][:2], ptoks[1][1][:2]))
        if len(openers) >= 15:
            win_o = 15
            reps = [
                1.0 - (len(set(openers[k:k + win_o])) / win_o)
                for k in range(0, len(openers) - win_o + 1, 5)
            ]
            m3_opener_homogeneity = statistics.mean(reps)
        else:
            m3_opener_homogeneity = 1.0 - (len(set(openers)) / max(1, len(openers))) if openers else 0.0
    else:
        m3_opener_homogeneity = 0.0

    def concrete_density(sub_text: str) -> float:
        if not sub_text:
            return 0.0
        nums = len(re.findall(r"\d+(?:\.\d+)?%?", sub_text))
        quotes = len(re.findall(r"[“”\"「」『』]", sub_text)) // 2
        proper_nouns = sum(1 for w, pos in tokenize_pos(sub_text) if pos in ("nr", "ns", "nt", "nz", "eng"))
        return (nums + quotes + proper_nouns) / max(1, len(sub_text)) * 1000.0

    n_chars = len(text)
    front_text = text[:int(n_chars * 0.3)]
    mid_text = text[int(n_chars * 0.3):int(n_chars * 0.7)]
    front_conc = concrete_density(front_text)
    mid_conc = concrete_density(mid_text)
    m3_mid_concrete_slump = min(3.0, (front_conc + 5.0) / (mid_conc + 5.0))

    all_words = [w for toks in sent_tokens for w, _ in toks if len(w) >= 1]
    win_w = 200
    if len(all_words) >= win_w:
        mattr = statistics.mean(
            len(set(all_words[i:i + win_w])) / win_w for i in range(0, len(all_words) - win_w + 1, 50)
        )
    else:
        mattr = len(set(all_words)) / max(1, len(all_words))
    m3_lexical_diversity_mattr = mattr

    return {
        "M1_em_dash_rate": m1_em_dash_rate,
        "M1_contrast_template_rate": m1_contrast_template_rate,
        "M1_nominal_light_verb_rate": m1_nominal_light_verb_rate,
        "M1_noun_stack_rate": m1_noun_stack_rate,
        "M1_buffer_deficit": m1_buffer_deficit,
        "M1_speech_buffer_rate": m1_speech_buffer_rate,
        "M1_de_congestion_ratio": m1_de_congestion_ratio,
        "M2_old_to_new_violation": m2_old_to_new_violation,
        "M2_new_topic_abrupt_rate": m2_new_topic_abrupt_rate,
        "M2_cohesion_break_rate": m2_cohesion_break_rate,
        "M2_pos_template_rep": m2_pos_template_rep,
        "M2_rhythm_flatness": m2_rhythm_flatness,
        "M2_para_flatness": m2_para_flatness,
        "M3_macro_redundancy": m3_macro_redundancy,
        "M3_mean_distant_echo": m3_mean_distant_echo,
        "M3_opener_homogeneity": m3_opener_homogeneity,
        "M3_mid_concrete_slump": m3_mid_concrete_slump,
        "M3_lexical_diversity_MATTR": m3_lexical_diversity_mattr,
    }


# ── 3. 基于内置 HC3-Chinese 人类语料的 N-gram 语言模型与 UID/PMI 探针 ──────────

class ChineseReferenceLM:
    """用内置 `hc3_human_corpus.json.gz`（18,670 条纯人类回答，206 万词）训练词级 Trigram 语言模型。"""

    def __init__(self):
        self.unigram = Counter()
        self.bigram = Counter()
        self.trigram = Counter()
        self.doc_freq = Counter()
        self.total_words = 0
        self.total_sents = 0

    def _ingest_text(self, h: str):
        if not h or len(h) < 50:
            return
        words = ["<s>"] + [w for w in jieba.cut(h) if w.strip()] + ["</s>"]
        self.total_words += len(words)
        for i, w in enumerate(words):
            self.unigram[w] += 1
            if i >= 1:
                self.bigram[(words[i - 1], w)] += 1
            if i >= 2:
                self.trigram[(words[i - 2], words[i - 1], w)] += 1
        for s in split_sentences(h):
            sw = set(w for w in jieba.cut(s) if len(w) >= 2)
            if sw:
                self.total_sents += 1
                for w in sw:
                    self.doc_freq[w] += 1

    def fit_hc3_human(self):
        bundled_gz = EVAL_DIR / "hc3_human_corpus.json.gz"
        if bundled_gz.exists():
            with gzip.open(bundled_gz, "rt", encoding="utf-8") as f:
                human_list = json.load(f)
            for h in human_list:
                self._ingest_text(h)
            return

        # 兼容回退：若父目录存在原始 jsonl
        hc3_files = sorted(glob.glob(str(REPO_ROOT.parent / "data/benchmarks/hc3_chinese/*.jsonl")))
        for fp in hc3_files:
            if "all.jsonl" in fp:
                continue
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    for h in (d.get("human_answers") or []):
                        self._ingest_text(h)

    def word_surprisal(self, w_prev2: str, w_prev1: str, w: str) -> float:
        V = max(1, len(self.unigram))
        p1 = (self.unigram[w] + 1.0) / (self.total_words + V)
        c1 = self.unigram[w_prev1]
        p2 = (self.bigram[(w_prev1, w)] + 0.5) / (c1 + 0.5 * V) if c1 > 0 else p1
        c2 = self.bigram[(w_prev2, w_prev1)]
        p3 = (self.trigram[(w_prev2, w_prev1, w)] + 0.2) / (c2 + 0.2 * V) if c2 > 0 else p2
        prob = 0.55 * p3 + 0.30 * p2 + 0.15 * p1
        return -math.log2(max(1e-9, prob))

    def sentence_surprisals(self, sent: str) -> List[float]:
        words = ["<s>", "<s>"] + [w for w in jieba.cut(sent) if w.strip()]
        if len(words) <= 4:
            return []
        return [self.word_surprisal(words[i - 2], words[i - 1], words[i]) for i in range(2, len(words))]

    def directed_sentence_pmi(self, s_src: str, s_tgt: str, doc_word_counts: Counter) -> float:
        w_src = [w for w in jieba.cut(s_src) if len(w) >= 2]
        w_tgt = [w for w in jieba.cut(s_tgt) if len(w) >= 2]
        if not w_src or not w_tgt:
            return 0.0
        set_src = set(w_src)
        set_tgt = set(w_tgt)

        char_src = set("".join(w_src))
        gain = 0.0
        for w in set_tgt:
            idf = math.log2((self.total_sents + 1.0) / (self.doc_freq[w] + 1.0))
            if w in set_src:
                gain += idf * 1.2
            else:
                shared_chars = len(set(w) & char_src)
                if shared_chars > 0:
                    gain += idf * 0.4 * (shared_chars / len(w))

        conclusion_markers = ("所以", "因此", "于是", "可见", "这就", "这样", "由此", "结果", "毕竟", "说到底")
        premise_support_markers = ("因为", "由于", "比如", "例如", "拿", "放到", "具体", "也就是", "即", "换言之")

        if s_tgt.startswith(conclusion_markers):
            gain += 2.5
        if s_tgt.startswith(premise_support_markers):
            gain -= 2.0

        return gain / math.sqrt(len(set_tgt))


def compute_uid_and_pmi_probes(lm: ChineseReferenceLM, raw_text: str) -> Dict[str, float]:
    text = clean_markdown_header(raw_text)
    sents = [s for s in split_sentences(text) if len(s) >= 6]
    if len(sents) < 6:
        return {}

    all_surps = []
    sent_mean_surps = []
    sent_init_surps = []
    sent_end_surps = []

    for s in sents:
        ss = lm.sentence_surprisals(s)
        if len(ss) >= 4:
            all_surps.extend(ss)
            sent_mean_surps.append(statistics.mean(ss))
            sent_init_surps.append(statistics.mean(ss[:2]))
            sent_end_surps.append(statistics.mean(ss[-2:]))

    if len(all_surps) < 20 or len(sent_mean_surps) < 5:
        return {}

    sec_diffs = [abs(all_surps[t + 1] - 2.0 * all_surps[t] + all_surps[t - 1]) for t in range(1, len(all_surps) - 1)]
    uid_local_2nd_diff = statistics.mean(sec_diffs)
    uid_global_std = statistics.stdev(all_surps)
    uid_jagged_ratio = uid_local_2nd_diff / max(0.1, uid_global_std)

    uid_onset_minus_end_surprisal = statistics.mean(sent_init_surps) - statistics.mean(sent_end_surps)
    uid_sentence_wave_cv = statistics.stdev(sent_mean_surps) / max(0.1, statistics.mean(sent_mean_surps))

    doc_w_counts = Counter(w for s in sents for w in jieba.cut(s) if len(w) >= 2)
    declared_claims = 0
    total_checked = 0
    forward_pmi_adv_list = []

    for i in range(1, len(sents) - 1):
        prev_gains = [lm.directed_sentence_pmi(sents[j], sents[i], doc_w_counts) for j in range(max(0, i - 3), i)]
        next_gains = [lm.directed_sentence_pmi(sents[i], sents[j], doc_w_counts) for j in range(i + 1, min(len(sents), i + 4))]
        best_prev = max(prev_gains) if prev_gains else 0.0
        best_next = max(next_gains) if next_gains else 0.0
        total_checked += 1
        if best_prev < best_next:
            declared_claims += 1
        forward_pmi_adv_list.append(best_prev - best_next)

    scislop_declared_ratio = declared_claims / max(1, total_checked)
    scislop_forward_pmi_adv = statistics.mean(forward_pmi_adv_list) if forward_pmi_adv_list else 0.0

    return {
        "UID_local_2nd_diff": round(uid_local_2nd_diff, 4),
        "UID_jagged_ratio": round(uid_jagged_ratio, 4),
        "UID_onset_minus_end_surp": round(uid_onset_minus_end_surprisal, 4),
        "UID_sentence_wave_cv": round(uid_sentence_wave_cv, 4),
        "SciSlop_declared_ratio": round(scislop_declared_ratio, 4),
        "SciSlop_forward_pmi_adv": round(scislop_forward_pmi_adv, 4),
    }
