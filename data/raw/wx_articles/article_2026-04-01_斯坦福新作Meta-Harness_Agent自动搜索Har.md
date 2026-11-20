# 斯坦福新作Meta-Harness：Agent自动搜索Harness代码，三大任务全面超越手工设计最优方案

- **来源**: Hyman的杂货铺
- **链接**: https://mp.weixin.qq.com/s?src=11&timestamp=1785309749&ver=6871&signature=w8HCAiG9aIED*s968nBKVSQ4xgfCLeQ-WcVHaDdWiEqsUFTn0*Xt4Yk5xWG27vC1KUTVkBwk-z5U-DLVsX6hDeqHPEh1w2mx-QuQWWdaTKflQS3m0ZXx2zEtHgr4lR2u&new=1
- **发布时间**: 2026-04-01 07:30
- **字数**: 5617

---

一句话讲清楚👉🏻

 Stanford团队发布Meta-Harness，让coding agent通过文件系统访问完整历史经验来自动优化LLM系统的harness代码，在文本分类、数学推理、TerminalBench-2代码生成三大任务上全面超越手工设计方案。

2026年，AI领域在悄然转向。从Prompt Engineering到Context Engineering，再到如今的Harness Engineering，关注点从"怎么问"转到了"怎么搭环境"。Stanford AI Lab刚发的Meta-Harness论文，踩在了这个节点上。

Harness是什么？

 简单来说，harness是包裹在LLM外面的那层代码——它决定了存储什么信息、检索什么内容、以及向模型呈现什么上下文。根据论文数据，

改变同一个LLM的harness，可以在同一基准上产生6倍的性能差距

。换句话说，模型本身的权重可能只决定了一半的表现，而另一半取决于harness的设计。

然而，当前harness的设计仍然高度依赖人工——工程师们反复检查失败案例、调整启发式规则、在少量设计方案间迭代。Meta-Harness的核心问题是：

这个过程本身能否自动化？

为什么现有方法不够用？

在Meta-Harness之前，已经有不少工作尝试自动化文本优化，比如OPRO、TextGrad、AlphaEvolve、GEPA等。但这些方法与harness优化的场景存在根本性的不匹配。

问题出在哪？

反馈压缩得太厉害

。

传统的文本优化方法要么只看当前候选方案，要么只依赖标量分数，要么把反馈压缩成简短的摘要。但在harness优化中，一个关于"存储什么"、"何时检索"、"如何呈现"的决策，可能在很多步推理之后才显现出效果。

压缩反馈往往会丢失追溯下游失败原因所需的关键信息

。

论文Table 1给出了一个直观的对比：传统文本优化方法每次迭代的上下文量级在0.002到0.026 MToken之间，而Meta-Harness的单次评估可以产生高达10M Token的诊断信息——相差了将近三个数量级。

Meta-Harness：一个优化Harness的Harness

Meta-Harness的核心思路很简单：

给proposer完整的历史访问权限，让它自己决定检查什么、诊断什么、修改什么

。

Meta-Harness搜索循环。Agent读取包含所有先前候选方案源代码、执行轨迹和分数的文件系统，然后提出新的harness。

具体来说，Meta-Harness的工作流程分为三步：

Step 1：Agent读取文件系统。

 每个被评估过的harness都对应一个目录，里面存放着源代码、评估分数和执行轨迹（包括prompt、工具调用、模型输出、状态更新等）。文件系统通常远超proposer的上下文窗口，所以proposer需要通过grep、cat等终端工具主动查询，而不是一次性全部读入。

Step 2：Proposer提出新harness。

 与传统方法不同，Meta-Harness的proposer本身就是一个coding agent（论文中使用Claude Code + Opus-4.6）。它可以自己决定检查哪些先前的工件、诊断哪些失败模式、是做局部修改还是全面重写。

Step 3：评估并记录。

 新harness通过接口验证后，在评估任务上运行，所有日志存入文件系统，循环继续。

这个设计的关键在于：

Meta-Harness本身就是一个harness

（这也是名字的由来）——它决定了proposer在搜索过程中能看到什么信息。但与传统方法相比，它对proposer几乎不加限制，让agent自己做诊断和决策。

形式化目标

harness优化的目标形式化如下：

其中 

 是固定的语言模型，

 是任务分布，

 是harness，

 是rollout轨迹，

 是任务特定的奖励函数。当存在多个目标（如准确率和上下文成本）时，使用Pareto占优来评估候选方案。

实验一：在线文本分类

论文首先在在线文本分类任务上验证Meta-Harness。任务设定是：LLM逐个接收带标签的示例，更新记忆，然后在测试集上评估。

数据集选择

选择了三个难度高、领域差异大的数据集：

LawBench

：根据案件描述预测罪名（215类）

Symptom2Disease

：根据症状描述预测疾病（22类）

USPTO-50k

：根据产物分子预测前体反应物（180类）

与手工设计Harness的对比

（左）在文本分类任务上，Meta-Harness超越所有先前的手工设计和文本优化方法。（右）在TerminalBench-2上，Meta-Harness在所有Claude Haiku 4.5方案中排名第一。

实验结果显示，Meta-Harness发现的harness达到48.6%的准确率，比当前最优的手工设计方案ACE高出7.7个点，比MCE高出8.6个点。更关键的是，

这些提升不是通过堆上下文换来的

：Meta-Harness只使用了11.4K上下文token，而ACE使用了50.8K，MCE使用了28.5K。

方案

准确率

上下文token(K)

Zero-Shot

27.4

0

Few-Shot (all)

40.8

12.3

MCE

40.0

28.5

ACE

40.9

50.8

Meta-Harness

48.6

11.4

与文本优化方法的对比

准确率-上下文token的Pareto前沿。Meta-Harness在所有对比方法中取得了更强的Pareto前沿。

与Best-of-N、OpenEvolve、TTT-Discover等文本优化方法相比，Meta-Harness仅用1/10的评估次数就匹配了对方的最终性能，最终准确率则高出10个点以上。

泛化能力

有人会担心：搜索出来的harness会不会过拟合特定的数据集？论文在9个完全没碰过的OOD数据集上测了一下，Meta-Harness达到73.1%的平均准确率，超过ACE的70.2%，在9个数据集中的6个上拿了最高分。

实验二：检索增强的数学推理

第二个实验设置更有挑战性：用检索增强来解决IMO级别的数学问题。

为什么检索对数学推理有用但很难？

数学问题的求解往往复用可迁移的证明模式，因此之前的推理轨迹理论上应该能帮助当前问题。但实践中，

朴素检索很少能以正确形式找到相关轨迹

。成功的关键不在于"是否添加检索"，而在于"发现正确的检索策略"。

实验设定

检索语料

：来自8个开源数据集的50万+已解问题，经过严格去重和去污染

搜索集

：250道奥林匹克难度的数学问题

评估集

：200道之前未见的IMO级问题（来自IMO-AnswerBench、IMO-ProofBench、ArXivMath）

评估模型

：5个搜索期间未见的模型（GPT-5.4-nano、GPT-5.4-mini、Gemini-3.1-Flash-Lite、Gemini-3-Flash、GPT-OSS-20B）

实验结果

方法

GPT-5.4n

GPT-5.4m

Gem-3.1FL

Gem-3F

GPT-20B

Avg.

无检索

23.0

28.8

28.6

42.6

47.6

34.1

Dense Retrieval (k=1)

27.1

24.5

31.3

42.3

46.9

34.4

Dense Retrieval (k=5)

31.1

28.3

37.1

47.2

46.7

38.1

BM25 Retrieval

30.2

29.2

32.8

46.6

48.9

37.5

Meta-Harness

31.7

30.4

34.9

46.3

50.6

38.8

Meta-Harness发现的检索策略在5个模型上的平均提升达到4.7个点，超越BM25检索1.3个点，并且避免了Dense Retrieval和Random Few-shot在某些模型上的性能回退。

实验三：TerminalBench-2上的Agentic Coding

第三个实验是最具挑战性的：在TerminalBench-2上评估agentic coding harness。这个基准包含89个需要长时程完全自主执行的复杂任务。

实验结果

Harness

Opus 4.6 通过率

Claude Code

58.0%

Terminus 2

62.9%

Terminus-KIRA

74.7%

Meta-Harness

76.4%

在Claude Opus 4.6上，Meta-Harness发现的harness达到76.4%的通过率，超越手工设计的Terminus-KIRA（74.7%），在TerminalBench-2排行榜上位列所有Opus 4.6方案的第二名。

在较弱的Claude Haiku 4.5上，改进更为显著：Meta-Harness达到37.6%，超过第二名Goose（35.5%）2.1个点。

消融实验：什么才是关键？

论文进行了一个重要的消融实验，对比了不同信息供给条件下的效果：

信息类型

准确率(中位)

准确率(最佳)

仅分数

34.6

41.3

分数+摘要

34.9

38.7

完整Meta-Harness

50.0

56.7

结果表明：

完整访问执行轨迹是接口中最重要的组成部分

。摘要并没有弥补缺失的信号，甚至可能因为压缩了诊断性有用的细节而起到反效果。仅看分数的方案中位数准确率34.6，加入摘要后反而可能降低到34.9；而完整访问执行轨迹后，中位数直接跳到50.0。

Proposer的定性行为分析

论文附录A记录了proposer在搜索过程中的具体行为，能帮我们理解Meta-Harness为什么有效。

在早期迭代中，proposer同时尝试了结构性修复和prompt模板修改，但发现两个候选方案都出现了性能回退。然后，

它主动假设回退是由共享的prompt干预引起的混杂效应

，将结构性修改从prompt重写中隔离出来，最终转向一个更安全的增量修改方案。

这个行为模式非常像人类工程师的调试思路：读取大量代码和日志，形成因果假设，隔离变量，迭代改进。Meta-Harness的proposer每轮迭代中位数会读取82个文件，引用超过20个先前的候选方案。

Harness Engineering：从Prompt到系统的演进

Meta-Harness刚好踩在2026年AI领域的一个转向节点上：从Prompt Engineering到Harness Engineering。

Prompt Engineering (2022-2024)

：关注如何精心设计单次指令。核心问题是"怎么问"。

Context Engineering (2025)

：认识到单次prompt永远不够，需要动态构建包含文档、历史、工具定义的上下文窗口。核心问题是"给什么信息"。

Harness Engineering (2026)

：进一步认识到整个系统的架构——存储、检索、验证、重试、护栏——比任何单个prompt都重要。核心问题是"如何构建运行环境"。

Meta-Harness是Harness Engineering范式下的代表性工作。它不再试图优化单个prompt，而是搜索整个harness代码——包括prompt构建、检索策略、状态更新逻辑等所有方面。

总结与展望

Meta-Harness做了三件事：

1. 给予proposer对历史经验的完整访问权限。

 不压缩、不摘要，让agent自己通过文件系统工具（grep、cat）按需读取源代码、分数和执行轨迹。这是与传统文本优化方法的根本区别。

2. 将搜索任务委托给coding agent。

 不是用固定的搜索循环+LLM作为算子，而是让coding agent自己决定检查什么、诊断什么、修改什么。这让系统能够随着coding agent能力的提升而自动变得更强。

3. 在三个不同领域验证了通用性。

 文本分类、数学推理、代码生成，每个领域都展现出显著的提升，说明这种方法的普适性。

Rich Sutton在《The Bitter Lesson》里说过：搜索空间一旦变得可访问，更强的通用Agent就能超过手工设计的方案。Meta-Harness就是这个道理在harness优化领域的具体落地。

下一步自然的方向是

联合优化harness和模型权重

——让策略塑造模型的学习内容，反之亦然。同时，论文也坦诚目前只使用了一个特别强的coding agent（Claude Code），更广泛的proposer agent对比研究还有待未来工作。

资源链接

📄 论文链接

https://arxiv.org/abs/2603.28052

🏠 项目主页

https://yoonholee.com/meta-harness/

💻 GitHub代码

https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact

⭐️关注我，实时跟进AI最新进展⭐️