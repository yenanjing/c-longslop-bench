# QRRanker：中科院联合腾讯微信开源长文本重排序新SOTA

- **来源**: Hyman的杂货铺
- **链接**: https://mp.weixin.qq.com/s?src=11&timestamp=1785309751&ver=6871&signature=w8HCAiG9aIED*s968nBKVSQ4xgfCLeQ-WcVHaDdWiEoJLw5kob5VMWOvXbw7WFGNFmfEbSLn5CJY3fWAzbqFPx4de063ixStNaQpiW9D6jyIXbiIasM7kGwAnQouv79Y&new=1
- **发布时间**: 2026-03-01 07:30
- **字数**: 5395

---

一句话讲清楚👉🏻 中科院信息工程研究所联合腾讯微信AI提出了一种基于Query-focused Retrieval Heads的轻量级重排序框架QRRanker，仅用4B参数就在多个长文本检索基准上超越了32B模型，并在LoCoMo对话记忆理解基准上刷新SOTA。

为什么我们需要更好的Reranker？

在RAG（检索增强生成）系统中，检索质量直接决定了最终答案的准确性。传统的向量检索虽然高效，但存在一个核心问题：

语义相似度 ≠ 查询-文档相关性

。

一个文档可能与查询在语义上很接近，但并未直接回答查询的问题。为了解决这个问题，业界普遍采用

两阶段检索策略

：

第一阶段

：使用双编码器（Bi-Encoder）快速召回大量候选文档（如top-50或top-100）

第二阶段

：使用Reranker对候选文档进行精细化重排序

然而，现有的Reranker方法存在两大局限：

Pointwise方法的局限

：逐个对"查询-文档"对打分，忽略了文档之间的相互关系。一个文档是否相关，往往需要与其他文档比较才能判断。

Listwise方法的局限

：虽然能利用全局信息，但通常需要离散的Likert-scale标签（如1-5分）进行监督，这限制了训练数据的来源。

中科院信息工程研究所、中科院大学网络空间安全学院联合腾讯微信AI的这篇论文提出了一个巧妙的解决方案：

直接利用LLM内部注意力头的打分能力

，实现了一种天然的Listwise重排序方法。

核心创新：Query-focused Retrieval Heads

什么是QR Heads？

研究发现，LLM中存在一些特殊的注意力头（Attention Heads），它们的注意力模式天然聚焦于与查询相关的文档段落。这些被称为

Query-focused Retrieval Heads（QR Heads）

。

图1：QR Score基于QR注意力头的注意力分数计算。图中Doc2是金标准文档。

如图所示，QR Score的计算基于QR注意力头对各个文档的注意力权重分布。

注意力权重天然就是一个连续的相关性分数

，这为训练提供了丰富的监督信号。

QRRanker的技术架构

图2：QRRanker整体框架。高亮的注意力头是用于文档打分的QR Heads。

QRRanker的核心流程如下：

(1) QR-Head识别

首先在种子数据上计算QR分数，识别出那些天然关注相关文档的注意力头。计算公式为：

其中 

 表示注意力头 

 从token 

 到 

 的注意力权重，

 是查询token集合，

 是文档token集合。

(2) Listwise训练

构造训练实例时，将top-50召回的候选文档全部输入模型，使用

组内对比损失（Group-wise Contrastive Loss）

进行优化：

其中 

 是基于QR注意力头的相关性分数，

 是正样本文档，

 是候选文档数量。

(3) 推理阶段的注意力打分

在推理时，直接聚合QR注意力头的注意力权重，输出连续相关性分数，

无需额外的生成步骤

，效率极高。

(4) 记忆增强（可选）

对于长篇叙事和对话场景，可以预先在文档前添加上下文摘要，帮助模型捕捉更广泛的语境信息。

记忆增强策略

QRRanker支持两种互补的摘要构建策略：

Block-based Summary（块级摘要）

：针对长篇小说，将书籍分割成块（每块约20个连续chunk），为每个块生成一个摘要。这种方式尊重故事的时序性，保留角色名称、关键对话和场景变化等细节。

Event-centric Summary（事件中心摘要）

：针对对话数据，从对话中提取结构化事件，每个事件用简短描述表示，并链接到原始对话，实现可追溯性。

检索分数计算细节

QR注意力头对文档 

 的检索分数计算如下：

其中 

 是注意力头 

 对文档 

 的分数。为提高稳定性，使用最大-最小归一化：

其中 scale 因子用于将分数范围缩放到 

。

组内对比损失（Group-wise Contrastive Loss）

传统对比损失每次只采样一个正文档，但top-50候选文档可能包含多个正文档。为充分利用所有正样本，论文提出了组内对比损失：

其中 

 是正文档集合，

 是负文档集合，

 是指数函数。这个目标函数将每个正文档视为独立子样本，并在样本内部平均损失。

实验结果：全面超越现有方法

检索性能对比

研究团队在多个领域的基准数据集上进行了评测，包括：

Wikipedia多跳QA

：MuSiQue、HotpotQA

长篇故事QA

：NarrativeQA、DetectiveQA

对话记忆

：LoCoMo

方法

MuSiQue R@5

MuSiQue R@10

HotpotQA R@5

HotpotQA R@10

NarrativeQA R@5

NarrativeQA R@10

DetectiveQA R@5

DetectiveQA R@10

Qwen3-Embedding-8B

62.55

72.47

89.05

95.15

20.92

32.39

20.00

31.17

Qwen-Reranker-4B

66.37

74.26

94.15

96.75

28.25

41.98

30.50

42.09

GroupRank-32B

65.08

73.07

90.60

94.50

33.76

48.83

39.21

51.38

QRHeads-4B (开箱即用)

71.22

78.99

94.80

96.90

33.44

48.89

32.89

45.58

QRRanker-4B

77.37

82.13

96.90

97.70

38.89

54.93

41.32

53.76

关键发现

：

QRRanker在所有数据集上均取得最佳性能

在MuSiQue上，Recall@5达到

77.37%

，比Qwen-Reranker-4B提升

+11个百分点

仅用4B参数就超越了32B的GroupRank

，证明了方法的效率优势

在故事理解任务（NarrativeQA、DetectiveQA）上提升尤为明显

LoCoMo对话记忆基准：新SOTA

LoCoMo是一个评估长对话理解和记忆使用能力的基准，QRRanker在这里取得了

57.03 F1

的最佳成绩：

方法

Tokens

Single-hop

Multi-hop

Temporal

Open-domain

Overall F1

Qwen3-Embedding-8B

846

47.95

35.24

41.36

24.79

42.81

Mem0

1,764

47.65

38.72

48.93

28.64

45.09

Membox

2,166

60.09

39.88

58.03

27.96

53.10

CompassMem

20,000

57.36

38.84

57.96

26.61

52.18

QRRanker

854

62.95

43.06

61.90

29.79

57.03

令人印象深刻的是

：QRRanker仅使用

854个平均token

就达到了最佳性能，而CompassMem使用了

20,000个token

。这意味着QRRanker在

效率上提升了20倍以上

！

检索性能可视化

下图展示了QRRanker在DetectiveQA和MuSiQue数据集上的检索召回性能对比：

图3：QRRanker在DetectiveQA（中英文数据集平均）和MuSiQue上的检索召回性能。QRRanker在不同检索阶段都持续提升召回率。

记忆增强的效果

下表展示了添加摘要前缀后的性能变化：

数据集

QRRanker (仅Chunk)

QRRanker (+摘要)

提升

LoCoMo

86.64

87.34

+0.70

NarrativeQA

28.09

29.11

+1.02

DetectiveQA

29.55

32.22

+2.67

HotpotQA

95.05

94.75

-0.30

MuSiQue

70.19

70.16

-0.03

关键发现

：摘要前缀对长对话和长故事场景有显著提升，但对Wikipedia多跳QA几乎无效果甚至略有下降。这表明抽象的全局摘要对证据高度局部化的Wikipedia段落帮助有限。

中间层注意力头选择实验

论文发现，使用中间层（17-24层）的注意力头进行训练可以获得与完整模型相当甚至更好的性能：

方法

R@3

R@5

R@10

QRRanker（完整）

28.87

39.16

54.44

10-17层（底层）

24.51

34.52

49.91

17-24层（中层）

28.15

39.07

54.28

28-35层（顶层）

28.48

38.88

54.65

重要发现

：

底层（10-17层）性能显著下降

中层和顶层几乎保持相同性能

可以

截断高层

以获得更小更快的模型

推理效率对比

方法

P50延迟(ms)

P95延迟(ms)

TFLOPs/query

峰值内存(GB)

Qwen3-Reranker (batch=50)

1221.59

1256.29

115.69

13.88

Qwen3-Reranker (batch=1)

1895.26

1929.09

113.65

7.78

QRRanker

1095.42

1133.38

82.74

11.18

QRRanker (中间层)

910.42

928.1

69.83

8.71

效率优势

：

QRRanker比Qwen3-Reranker延迟降低约10%

计算量（TFLOPs）减少约30%

使用中间层截断版本，延迟降低约25%，内存减少约37%

方法优势总结

1. 轻量高效

仅需4B参数即可超越32B模型

推理时无需生成步骤，直接输出连续分数

平均token使用量极低

2. 灵活的训练方式

利用注意力分数作为连续监督信号

不依赖Likert-scale标注

，可在任意检索数据集上训练

支持Listwise训练，充分利用全局信息

3. 强泛化能力

跨领域表现优异（Wikipedia、故事、对话）

支持记忆增强扩展

可选用中间层注意力头以提升效率

开源资源

QRRanker已在HuggingFace开源，模型权重可直接使用：

模型地址

：https://huggingface.co/MindscapeRAG/QRRanker

论文地址

：https://arxiv.org/abs/2602.12192

项目主页

：https://qdcassie-li.github.io/QRRanker/

QR Heads 位置信息

对于Qwen3-4B-Instruct-2507模型（36层×32头），论文选取的16个QR Heads位于中间层：

17层：27、25、17、4号头

18层：15、19号头

19层：12号头

20层：15号头

21层：11、10、8、18、31号头

22层：4号头

23层：10号头

24层：13号头

这些QR Heads主要集中在

17-24层

的中层位置，与消融实验发现的最佳层范围一致。

相关工作

QR Heads的发现背景

QR Heads的概念源自两篇重要工作：

Retrieval Head

（Wu et al., 2024）：发现LLM中存在专门负责检索的注意力头

Query-focused Retrieval Heads

（Zhang et al., 2025）：进一步发现这些头的注意力模式天然聚焦于查询相关段落

QRRanker的创新在于：

不仅识别这些头，还通过训练优化它们

，从而获得更好的迁移性和稳定性。

与传统Reranker的对比

方法类型

代表模型

优势

局限

Pointwise

Qwen-Reranker、BGE-m3

高效、稳定

忽略文档间关系

Listwise（生成式）

RankVicuna、GroupRank

利用全局信息

需要离散标签、生成不稳定

QRRanker

本文方法

连续分数+全局信息

需要识别QR Heads

论文链接：https://arxiv.org/abs/2602.12192

⭐️关注我，实时跟进AI最新进展⭐️