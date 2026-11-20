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

| 检测 / 评价范式 | 出处 / Provenance | Heldout AUROC↑ | Pre-AI 真人误杀率 (FPR)↓ | 原生 AI 检出率↑ | 对抗改写检出率↑ |
|---|---|---:|---:|---:|---:|
| 1. Perplexity / Surprisal Baseline | 经典统计基线（GPTZero/Binoculars 时代通用做法，参考 [SlopBench](https://arxiv.org/pdf/2609.33905) 复现协议） | 0.2407 | 16.7% | 30.0% | 20.0% |
| 2. Burstiness / Sent-Length CV (GPTZero-style) | 经典统计基线，[GPTZero](https://gptzero.me) 核心句长方差特征的开源复现 | 0.6522 | 15.2% | 40.0% | 10.0% |
| 3. Classic Cliche Lexicon (Forbidden Words) | 传统禁用套话词表匹配（词表整理自 Slop 风格研究，中文清单为本基准自建校准） | 0.3814 | 15.2% | 50.0% | 10.0% |
| 4. Single-Score LLM-as-a-Judge (`hy3-ioa`) | 单分数 LLM 盲判裁判（内部模型 hy3-ioa，判 HUMAN/AI 单标签） | 0.6000 | 5.3% | 20.0% (80%漏检) | 10.0% (90%被骗) |
| **5. C-LongSlop 4-Dim Anti-Hack Z-Score (Ours)** | **本工作提出**（四维正交探针 + 双边人类分布带 + 四道防刷锁） | **0.8336** | 13.6% | **90.0%** | **80.0%** |
| **6. C-LongSlop Bipolar Sparse Probe (Ours)** | **本工作实现**（方法配方源自 DeepMind [DetectorScope](https://arxiv.org/pdf/2610.30006) 稀疏特征检测范式，中文 TF-IDF 字符 n-gram + ElasticNet 自训） | **0.9702** | **10.6%** | **100.0%** | **80.0%** |
| **7. C-LongSlop Full Ensemble (Ours: 4-Dim Z + Sparse)** | **本工作提出**（四维防刷 Z 与稀疏探针的联合集成判据） | **0.9494** | 13.6% | **100.0%** | **90.0%** |

### 主实验二（Table 2）：配对长文去 AI 味公共方法排行榜（Humanization Leaderboard，不含内部代号）

在 10 篇同题配对科技长文上，横向评测 2 组真人天花板、1 组未编辑 AI 基线、5 种主流公共去 AI 味范式与我们的处方引导微修方案：

| 梯队与公共方法名称 | 出处 / Provenance | LLM 盲判通过率 | D1 微观↓ | D2 衔接↓ | D3 骨架↓ | D4 宏观↓ | 防刷 Z↓ | 稀疏 AI 率↓ | 字符改动% | 事实保真率↑ | **综合总分↑** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 🌟 **1. Human Ceiling (Pre-AI Blogs 2011–2022)** | 本基准自建语料（陈皓 CoolShell / 阮一峰周刊等 Pre-AI 归档，LOAO §留一博主验证） | 100% | 0.28 | 0.15 | 0.84 | 0.62 | 0.472 | 0.203 | 0.0% | 100.0% | **83.8** |
| 🌟 **2. Human Ceiling (Contemporary Tech Blogs)** | 本基准自建语料（云风 CodingNow / piglei / laixintao 等当代独立博客，LOAO §留一博主验证） | 100% | 0.40 | 0.13 | 0.74 | 0.14 | 0.413 | 0.252 | 0.0% | 100.0% | **82.9** |
| 🛠️ **3. C-LongSlop Guided Repair + Quality Gate (Ours)** | **本工作提出**（句段级处方单参照 DetectorScope evidence-block 定向改写协议 [arXiv 2610.30006](https://arxiv.org/pdf/2610.30006)） | — | **0.26** | **0.09** | **1.00** | **0.08** | **0.358** | 0.579 | **3.2%** | **100.0%** | **72.1** |
| 📝 **4. Few-Shot Exemplar Imitation (5-Shot)** | 公共基线，复现 [Catch Me If You Can? (EMNLP 2025 Findings)](https://arxiv.org/abs/2509.14543) 的 Few-Shot 风格模仿改写协议 | 90% | 1.00 | 0.14 | 1.14 | 0.11 | 0.595 | 0.557 | 23.0% | 93.5% | **68.4** |
| 🧪 **5. C-LongSlop Guided Repair (w/o Gate Ablation)** | **本工作消融**（移除 ParagraphQualityGate，证明门禁必要性，见 Table 3） | — | 1.14 | 0.13 | 1.78 | 0.10 | 0.788 | 0.531 | 7.2% | 100.0% | **65.8** |
| 🤖 **6. Raw LLM Long-Form (Unedited Paired-10)** | 本基准自建配对语料（与真人长文同题的前沿 LLM 原生长文，未编辑；配对协议同 [Catch Me If You Can](https://arxiv.org/abs/2509.14543)） | 80% (漏检) | 0.72 | 0.11 | 1.07 | 0.09 | 0.497 | 0.685 | 0.0% | 100.0% | **65.8** |
| 💬 **7. Zero-Shot Generic Prompting (Blind Rewrite)** | 公共基线，复现 DeepMind [DetectorScope (arXiv 2610.30006)](https://arxiv.org/pdf/2610.30006) Fig.5 的 Generic 泛化改写提示 | 80% | 0.72 | 0.11 | 1.07 | 0.09 | 0.497 | 0.684 | 0.0% | 100.0% | **65.8** |
| 🎭 **8. Two-Stage Style Transfer (TTM 3-Round)** | 公共基线，[Test-Time-Matching (arXiv 2507.16799)](https://arxiv.org/abs/2507.16799) 的「先内容中性化 → 再逐句风格注入」两阶段解耦改写，3 轮迭代 | 90% (被骗) | 0.70 | 0.24 | 0.80 | 0.08 | 1.180 | 0.387 | **58.7%** | **81.5%** | **64.7** |
| 🎭 **9. Persona-Adversarial Loop (PA 3-Round)** | 本基准自建基线（Persona 人设注入 + 裁判反馈多轮对抗，设计参照 [IMPersona, COLM 2025](https://arxiv.org/abs/2504.04332) / [TTM](https://arxiv.org/abs/2507.16799) 角色扮演范式） | 90% (被骗) | 0.88 | 0.18 | 1.03 | 0.06 | 1.537 | 0.389 | **48.1%** | **84.8%** | **59.5** |
| 📋 **10. Naive Rule-Checklist Rewriting (Single-Sided)** | 公共基线，单向禁用词表规则提示（复现 [Catch Me If You Can](https://arxiv.org/abs/2509.14543) 的显式 checklist 约束改写，无防代偿锁） | 80% | **9.40** | 0.15 | 1.07 | 0.12 | **2.684** | 0.669 | 2.7% | 100.0% | **38.0** |

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
