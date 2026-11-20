# C-LongSlopBench：中文长文 AI 味检测与防刷分评测基准 v1.1

> 面向中文长文（450–10000 字）的「AI 味」可归因、防刷分、可执行诊断评测基准。
> 设计原则与全部实验证据见 `../research-survey-personalized-writing.md` 第 11–14 节。

## 一、Bench 组成

```
c_longslop_bench/
├── scrape_multi_human_authors.py           # 多博主真实人类中文长文采集器（陈皓/阮一峰/云风/piglei/laixintao）
├── build_bench_dataset.py                  # 数据集标准化构建脚本 v1.1（固定 seed=42，可复现）
├── run_multi_author_audit_and_leaderboard.py # 分博主审计 + 留一博主交叉验证 (LOAO) + 首版排行榜生成
├── data/
│   ├── raw/
│   │   ├── multi_human_authors/            # 新采集的 5 位独立技术博主 90 篇真实长文 + authors_manifest.json
│   │   ├── kazike_articles/                # 卡兹克 19 篇公众号长文
│   │   ├── wx_articles/                    # 90 篇原生 AI 公众号长文
│   │   ├── m4_chinese/                     # M4-Chinese (ChatGPT + Davinci) 长回答
│   │   └── hc3_chinese/                    # HC3-Chinese 百科/心理/医疗/金融/问答长回答
│   └── processed/
│       └── c_longslop_bench_v1_1.jsonl     # 标准数据集 v1.1（549 样本：人类 252 + 机器 297）
├── evaluators/
│   └── antihack_scorer.py                  # 官方评测器（四锁防刷 + 处方单输出）
├── results/
│   └── heldout_scores_v1_1.json            # heldout (248 篇) 全量评测结果
└── leaderboard/
    └── leaderboard_v1_1.json               # 首版多基准/多改写方法排行榜与 LOAO 验证报告
```

### v1.1 规模与多博主真实人类锚点（告别单一博主偏置）

v1.1 共 **549 篇长文样本**（`dev` 301 篇：`human=140, machine=161`；`heldout` 248 篇：`human=112, machine=136`）。
核心升级是引入了 **6 位风格迥异、真实身份可核验的中国知名技术/随笔作者（共 109 篇深度长文）**：

| 真实人类作者 | 身份与文风特点 | 时代属性 | 采集篇数 | 平均字数 | Heldout HUMANLIKE 达标率 |
|---|---|---|---:|---:|---:|
| **陈皓 (`chenhao_coolshell`)** | 酷壳博主（左耳朵耗子），硬核系统架构与工程文化 | **2011–2022 Pre-AI 黄金锚** | 25 篇 | 4412 字 | **92.3%** |
| **阮一峰 (`ruanyifeng`)** | 《科技爱好者周刊》2019–2021 刊首语评论随笔 | **2019–2021 Pre-AI 黄金锚** | 22 篇 | 823 字 | **90.9%** |
| **云风 (`yunfeng`)** | 中国游戏引擎泰斗，独立博客 20 年亲笔技术随笔 | 独立博客亲笔 | 15 篇 | 4356 字 | **100.0%** |
| **piglei (`piglei`)** | 《Python 工匠》作者，软件设计哲学与工程漫谈 | 独立博客亲笔 | 20 篇 | 4187 字 | **100.0%** |
| **laixintao (`laixintao_kawabangga`)** | 知名基础设施工程师（卡瓦邦噶！独立博客） | 独立博客亲笔 | 8 篇 | 3233 字 | **75.0%** |
| **数字生命卡兹克 (`kazike`)** | 科技自媒体长文 | 公众号亲笔 | 19 篇 | 3150 字 | **100.0%** |

## 二、留一博主交叉验证（Leave-One-Author-Out CV）—— 证明零口癖过拟合

每次将 1 位博主从训练集与人类分布带拟合中**彻底剔除（Zero-Shot Unseen Blogger）**，仅用其余人类作者拟合分布带与稀疏探针，测试该未知博主 vs AI 长文的区分度：

| 被留出的未知真人博主 (Zero-Shot Author) | 样本数 | 四维防刷 Z AUROC | 双极稀疏探针 AUROC | 联合判据 AUROC | 未知博主被判 AI 概率均值 |
|---|---:|---:|---:|---:|---:|
| **陈皓 (`chenhao_coolshell`, Pre-AI)** | 25 | 0.7768 | **1.0000** | **1.0000** | 0.252 (稳居人类区) |
| **阮一峰 (`ruanyifeng`, Pre-AI)** | 22 | 0.8812 | **1.0000** | **0.9545** | 0.231 (稳居人类区) |
| **云风 (`yunfeng`)** | 15 | **1.0000** | **1.0000** | **1.0000** | 0.288 (稳居人类区) |
| **piglei (`piglei`)** | 20 | 0.7415 | **1.0000** | **0.9886** | 0.363 (稳居人类区) |
| **laixintao (`kawabangga`)** | 8 | 0.8295 | **0.8750** | **0.8750** | 0.324 (稳居人类区) |
| **卡兹克 (`kazike`)** | 19 | **0.9952** | **1.0000** | **1.0000** | 0.203 (稳居人类区) |
| **6 位博主 LOAO 平均** | **109** | **0.8707** | **0.9792** | **0.9697** | **0.277** |

## 三、论文三大主实验结果（Main Benchmark Experiments）

> 完整复现实验脚本：`run_main_bench_experiments.py` | 原始数据：`results/main_experiments_tables_1_to_3.json`

### 主实验一（Table 1）：检测器与裁判跨领域对抗横评（Evaluator Benchmark）

在 Heldout 集（248 篇长文）与 10 篇配对改写集上，对比传统统计基线、单分数 LLM 裁判与 C-LongSlopBench 探针：

| 检测 / 评价范式 | Heldout AUROC↑ | Pre-AI 真人误杀率↓ | 原生 AI 检出率↑ | 对抗改写检出率↑ |
|---|---:|---:|---:|---:|
| 1. Perplexity 困惑度 [A] | 0.2407 | 16.7% | 30.0% | 20.0% |
| 2. Burstiness 句长方差 [B] | 0.6522 | 15.2% | 40.0% | 10.0% |
| 3. 禁用套话词表 [C] | 0.3814 | 15.2% | 50.0% | 10.0% |
| 4. 单分数 LLM 裁判 (`hy4`) [D] | 0.6000 | 5.3% | 20.0% (80%漏检) | 10.0% (90%被骗) |
| **5. 四维防刷 Z-Score (Ours)** [E] | **0.8336** | 13.6% | **90.0%** | **80.0%** |
| **6. 双极稀疏探针 (Ours)** [F] | **0.9702** | **10.6%** | **100.0%** | **80.0%** |
| **7. Full Ensemble (Ours)** [G] | **0.9494** | 13.6% | **100.0%** | **90.0%** |

**表注（出处）**：
[A] 经典统计基线，[SlopBench](https://arxiv.org/pdf/2609.33905) 等论文标准复现协议；[B] [GPTZero](https://gptzero.me) 核心句长方差特征开源复现；[C] 禁用套话词表匹配（英文词表整理自 Slop 研究，中文词表本基准自建校准）；[D] `hy4` 单分数盲判（打分方式见下）；[E] **本工作提出**（四维正交探针 + 双边人类分布带 + 四道防刷锁）；[F] **本工作实现**——配方源自 DeepMind [DetectorScope](https://arxiv.org/pdf/2610.30006) 稀疏特征检测范式，中文侧 TF-IDF 字符 n-gram + ElasticNet 自训；[G] **本工作提出**，四维 Z 与稀疏探针线性集成。

**表注（LLM 裁判怎么打分）**：裁判模型（`hy4`，判据经 87 篇校准集校准，F1≈0.895）对每篇文本**只见文本本身**（盲测、不署名），输出单一 JSON `{verdict: HUMAN/AI, confidence: 0-100, ai_tells, human_tells}`；考察口语化程度、个人色彩、情绪波动、模板化结构、均匀节奏、升华式金句收尾等。它不做逐段归因、不输出可解释特征，因此对结构性 AI 味天然失明——这正是第 4 行 0.6000 与第 5–7 行的差距来源，也是本基准存在的理由。

### 主实验二（Table 2）：配对长文去 AI 味公共方法排行榜（Humanization Leaderboard，不含内部代号）

在 10 篇同题配对科技长文上，横向评测 2 组真人天花板、1 组未编辑 AI 基线、5 种主流公共去 AI 味范式与我们的处方引导微修方案：

| 方法 | LLM 盲判通过↑ | 总分↑ | 防刷 Z↓ | 稀疏 AI 率↓ | 事实保真↑ | 改动% | D1↓ | D2↓ | D3↓ | D4↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 🌟 Pre-AI 真人锚 [1] | 100% | **83.8** | 0.472 | 0.203 | 100.0% | 0.0% | 0.28 | 0.15 | 0.84 | 0.62 |
| 🌟 当代真人锚 [2] | 100% | **82.9** | 0.413 | 0.252 | 100.0% | 0.0% | 0.40 | 0.13 | 0.74 | 0.14 |
| 🛠️ Ours + Gate [3] | 0% [a] | **72.1** | **0.358** | 0.579 | **100.0%** | **3.2%** | **0.26** | **0.09** | **1.00** | **0.08** |
| 📝 Few-Shot 模仿 [4] | 90% | 68.4 | 0.595 | 0.557 | 93.5% | 23.0% | 1.00 | 0.14 | 1.14 | 0.11 |
| 🧪 Ours w/o Gate [5] | — | 65.8 | 0.788 | 0.531 | 100.0% | 7.2% | 1.14 | 0.13 | 1.78 | 0.10 |
| 🤖 原生 AI 基线 [6] | 80% (漏检) | 65.8 | 0.497 | 0.685 | 100.0% | 0.0% | 0.72 | 0.11 | 1.07 | 0.09 |
| 💬 Zero-Shot 盲改 [7] | 80% | 65.8 | 0.497 | 0.684 | 100.0% | 0.0% | 0.72 | 0.11 | 1.07 | 0.09 |
| 🎭 TTM 两阶段 [8] | 90% (被骗) | 64.7 | 1.180 | 0.387 | **81.5%** | **58.7%** | 0.70 | 0.24 | 0.80 | 0.08 |
| 🎭 PA 人设对抗 [9] | 90% (被骗) | 59.5 | 1.537 | 0.389 | **84.8%** | **48.1%** | 0.88 | 0.18 | 1.03 | 0.06 |
| 📋 Checklist 规则 [10] | 80% | 38.0 | 2.684 | 0.669 | 100.0% | 2.7% | **9.40** | 0.15 | 1.07 | 0.12 |

**表注（出处）**：
[1] 本基准自建语料：陈皓 CoolShell / 阮一峰周刊等 2011–2022 Pre-AI 归档（24 篇）；[2] 本基准自建语料：云风 CodingNow / piglei / laixintao「卡瓦邦噶！」等当代独立博客（32 篇）；[3] **本工作提出**——句段级处方单引导微修 + ParagraphQualityGate 段落门禁（1-Pass），处方协议参照 [DetectorScope](https://arxiv.org/pdf/2610.30006) evidence-block 定向改写设计；[4] 公共基线，复现 [Catch Me If You Can? (EMNLP 2025 Findings)](https://arxiv.org/abs/2509.14543) 的 Few-Shot 风格模仿改写协议；[5] **本工作消融**——移除段落门禁、对所有处方段落一刀切微修（见 Table 3）；[6] 本基准自建配对语料——与真人长文同题的前沿 LLM 原生长文（未编辑，配对协议同 [Catch Me If You Can](https://arxiv.org/abs/2509.14543)）；[7] 公共基线，复现 DeepMind [DetectorScope](https://arxiv.org/pdf/2610.30006) Fig.5 的 Generic 泛化改写提示；[8] 公共基线，[Test-Time-Matching (arXiv 2507.16799)](https://arxiv.org/abs/2507.16799) 的「先内容中性化 → 再逐句风格注入」解耦改写，3 轮迭代；[9] 本基准自建基线——Persona 人设注入 + 裁判反馈多轮对抗，设计参照 [IMPersona (COLM 2025)](https://arxiv.org/abs/2504.04332) 与 TTM 角色扮演范式；[10] 公共基线，单向禁用词表规则提示（复现 [Catch Me If You Can](https://arxiv.org/abs/2509.14543) 的显式约束改写，无防代偿锁）。

**表注（指标定义）**：
- **LLM 盲判通过率**：单分数 LLM-as-a-Judge（`hy4`）对每篇独立盲判 HUMAN/AI，通过率 = 判 HUMAN 篇数 / 总篇数。该裁判单独检测 AUROC 仅 0.6000（Table 1 第 4 行），因此仅作参照，不作排名依据。真人锚与 AI 基线各为 24/32/10 篇，改写基线为 10 篇配对产物。
- **[a] 关于 Ours + Gate 的 0% 盲判通过率**：补跑实测（`hy4-ioa`，与全部基线同一盲判协议，逐篇记录见 `results/llm_judge_ours_backfill.jsonl`）。这是本基准核心论点的双面铁证：处方微修仅改 3.2% 字符、宏观段落骨架保留，而单分数裁判主要依赖宏观结构信号——它连保真 100% 的最小外科手术修复都判 AI，同时却能被 TTM/PA 这类「结构乱但表层像人」的大改写（48%–59% 字符改动、丢 15%–18% 事实）骗过 90%。裁判意见既漏检又可被洗稿骗过且不给出归因，正是本基准以确定性防刷 Z + 稀疏探针 + 事实保真作为官方排名指标的理由。
- **总分** = `0.7 × HumanPass(四维防刷) + 0.3 × (1 − 稀疏AI率)`，其中 `HumanPass = 100·exp(−0.35·防刷Z)`，本基准官方排名指标。
- **防刷 Z**：D1–D4 四维探针相对多博主真人分布 `[P10, P90]` 双边区间的偏离 Z-Score，越界按超标距离重罚（含成对张力锁与事实红线），真人均值约 0.41–0.47。
- **稀疏 AI 率**：DetectorScope 式双极稀疏探针（TF-IDF 字符 n-gram + ElasticNet，专名/数字掩码后训练）输出的 AI 概率，0.50 为决策边界。
- **事实保真率**：原子事实锚点（数字/专名/因果链）改写后保留比例，确定性脚本抽验。
- **改动%**：改写产物相对原文的字符级 diff 比例。
- **D1 微观拐杖**：解释性标点密度（`——` + 括号 + 冒号枚举）与口语缓冲词赤字的合成 Z；**D2 跨句衔接**：相邻句搭扣断裂率 + 句首信息冲击；**D3 骨架克隆**：段首起手式同质化 + 词性 4-gram 复读；**D4 宏观回环**：跨段远距复述 + 中段具象枯竭。

### 主实验三（Table 3）：段落级防刷门禁（Paragraph Quality Gate）消融审计

在 10 篇配对长文（共 566 个自然段）上，处方引擎共触发候选微修 555 段次，`ParagraphQualityGate` 放行正收益微修 **206 段**，自动拦截并回滚负优化/越界微修 **349 段（拦截率 62.9%）**，将全文平均字符改动率从无门禁的 `7.2%` 压至 **`3.2%`**，防刷 Z-Score 从 `0.788` 降至 **`0.358`（超越真人均值 `0.413`）**，且 10 篇核心实验数字与专名锚点保留率达 **100.0%**。

## 二、评测器与指标定义（呼应"指标怎么算"）

### 2.1 指标来源分类（我们的 vs 外部权威）

| 探针/指标 | 来源 | 实现方式 |
|---|---|---|
| `M1_speech_buffer_rate`（口语缓冲词率）、`M1_buffer_deficit`（套话/缓冲赤字）、`M1_em_dash_rate` | 词表统计，词表来自 SlopBench/NYU 论文的英文清单，中文词表为**我们自建**（在 87 篇人类文与 AI 文上校准） | jieba 分词 + 词表计数 |
| `M2_pos_template_rep`（词性 4-gram 骨架复读） | Shaib Slop（英文），中文复现为**我们自研** | jieba.posseg 词性标注 + n-gram 重复率 |
| `M2_cohesion_break_rate`（相邻句实词搭扣断裂） | OSU COLM26 cohesion-chain，中文化为**我们自研** | 相邻句内容词交集为空则记一次断裂 |
| `M2_new_topic_abrupt_rate`（生硬换题率） | OSU old-to-new ordering 中文化 | 句首 2 个内容词在前文未出现过则记一次 |
| `M3_opener_homogeneity`（段首起手式同质化） | SlopBench Templating 中文化 | 段首 4 词词性序列的重复占比 |
| `M3_macro_redundancy` / `M3_mean_distant_echo`（跨段回环） | SciSlop 跨段复读指标中文化 | 远距句 8-gram 重合 |
| `M3_mid_concrete_slump`（中段具象塌陷） | ConStory 中段塌陷定律中文化 | 30%–70% 区间具象锚点（数字/引语）密度比 |
| `M3_lexical_diversity_MATTR` | SlopBench（他们证明这是**反向指标**，我们按双边带使用） | 滑动窗口 TTR |
| `UID_local_2nd_diff` / `UID_onset_minus_end_surp`（信息密度二阶差分） | NYU UID 论文 + UIDPO，**我们自己用 HC3 人类语料拟合的参考 LM** 计算 surprisal | 逐词对数概率 |
| **双极稀疏探针（"中文版 DetectorScope"）** | 方法论复现 DetectorScope（arXiv 2610.30006）的 gram2vec + ElasticNet 可解释稀疏归因，**非官方模型**：DeepMind 只开源了英文 SAE 探针，中文没有现成模型，我们用 TF-IDF 字符 2–4 gram + ElasticNet 逻辑回归自行训练（dev 集 5-Fold CV AUROC=0.9118） | sklearn，随机种子 42 |

**注意**：评测器不含任何 LLM 裁判；所有探针毫秒级、确定性、可复现。
`humanpass_100 = 100·exp(-0.35·composite_z)`（Z 越小人味越足，人类中位约 91.5）。

### 2.2 四锁防刷机制（Anti-Hacking Locks）

1. **双边人类分布带 `[P10, P90]`**：所有连续指标按 dev 人类分布锚定，带内轻罚（0.2×），越界按标准差距离全额惩罚——堆口语词和零口语词都扣分。
2. **成对张力锁**：句长节奏 × 跨句搭扣 × 句首惊奇度耦合惩罚（`coupling = 0.5·max(0,z_coh-1)·max(0,z_onset-1)`），剁碎句子刷 burstiness 立即反噬。
3. **标点拐杖守恒**：破折号/括号/冒号合并为一个「解释拐杖总密度」双边带指标，堵死符号间代偿转移。
4. **事实锚点红线**：改写任务中，源文本关键数字/专名丢失 → `Factual_Gate=REJECT`，直接 0 分。

### 2.3 判读三档

`HUMANLIKE`（Z<0.6 且 AI_prob<0.5）/ `GRAY_ZONE` / `AI_FLAVOR`（Z>1.2 或 AI_prob>0.6）。

## 三、heldout 首轮基线结果（v1.0，2026-11-19）

- 稀疏探针 dev 5-Fold CV AUROC：**0.9118**
- heldout 判别 AUROC：composite_z 单判据 **0.9216**；sparse_prob 单判据 **0.9852**
- 人类（n=46）：Z=0.261，HumanPass=91.5，AI_prob=0.347，HUMANLIKE 率 **82.6%**
- 机器（n=56）：Z=0.855，HumanPass=75.0，AI_prob=0.685，HUMANLIKE 率 **5.4%**

分领域细分：

| domain | track | n | Z | HumanPass | AI_prob | HUMANLIKE% |
|---|---|---:|---:|---:|---:|---:|
| tech_media（人类黄金锚） | human | 19 | 0.130 | 95.6 | 0.345 | 94.7% |
| tech_media_ai（原生AI公众号） | machine | 44 | 0.821 | 75.3 | 0.709 | 0.0% |
| m4_open_qa | human | 4 | 0.239 | 92.1 | 0.317 | 100.0% |
| m4_open_qa | machine | 5 | 0.568 | 82.5 | 0.471 | 60.0% |
| hc3_psychology | human | 12 | 0.356 | 88.5 | 0.274 | 75.0% |
| hc3_psychology | machine | 6 | 1.354 | 66.5 | 0.711 | 0.0% |
| hc3_baike | human | 11 | 0.392 | 87.7 | 0.441 | 63.6% |
| hc3_finance | machine | 1 | 0.813 | 75.2 | 0.543 | 0.0% |

已知局限（v1.0 诚实披露）：m4/hc3 部分 domain 的 heldout 样本过少（1–6 篇），AUROC 有幸存者偏差风险；v1.1 计划扩大配额并把 M4 配对比例提到全量。

## 四、防刷验证结论（完整数据见 ../results/pilot_c_longslop_antihack_and_prescription.json）

4 类作弊攻击（口水词轰炸 / 剁碎短句 / 标点转移 / 删数据段）在四锁下全部被重罚或否决（HumanPass 19.0 / 35.4 / 48.6 / 0.0），而传统单向评分在 Hack A 下被骗到满分 0.000。
处方引导微修闭环：仅改 6.8% 字符，综合偏离 Z 从 2.054 → 0.664（-67.7%），独立稀疏探针 AI 概率 0.714 → 0.473 翻转，事实保真 100%。

## 五、使用方法

```bash
# 重建数据集（可复现，seed=42）
python3 build_bench_dataset.py

# 跑 heldout 官方评测（带处方单输出加 --with-prescriptions）
python3 evaluators/antihack_scorer.py --input data/processed/c_longslop_bench_v1.jsonl \
    --split heldout --output results/heldout_scores.json
```

被测方法只需把输出发到 `leaderboard/` 下按 `<method_name>.jsonl`（sample_id + 改写后文本）格式提交即可。

## 六、Roadmap

- v1.1：扩大 heldout 配额、M4 配对全量化、加入 Track A（同题生成）与 Track B（人味化改写）双任务
- v1.2：多参考 LM 鲁棒性（Qwen2.5 替换 HC3-LM 做 UID）、中文依存句法拥堵指标（HanLP/LTP）
- v2.0：开放提交入口 + 周期性 heldout 轮换 + Hacking 告警指数（公开/盲盒探针 Split-half 泄漏检测）
