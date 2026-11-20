# Kimi重大突破：重新定义深度残差连接，训练效率提升1.25倍

- **来源**: Hyman的杂货铺
- **链接**: https://mp.weixin.qq.com/s?src=11&timestamp=1785309751&ver=6871&signature=w8HCAiG9aIED*s968nBKVSQ4xgfCLeQ-WcVHaDdWiEqlzGy*m34pQRuzV5CsQL1TcUj2IgXhqPyj-TERB2PZRYWCBAkoc-ErtkSlh1Qv6e6eMzT4YITnLSIlQX21Zq2H&new=1
- **发布时间**: 2026-03-17 07:30
- **字数**: 3934

---

一句话讲清楚👉🏻
 Kimi团队提出Attention Residuals技术，用可学习的注意力机制替代Transformer中十年不变的固定残差连接，让模型每层能智能选择性地聚合之前各层的输出，在48B参数模型上实现训练效率提升1.25倍。

研究背景：残差连接，Transformer成功的基石却从未被重新审视

自2017年Transformer架构诞生以来，其核心组件——自注意力机制、多头注意力、位置编码等——经历了无数次的改进和优化。但有一个基础组件几乎从未被重新审视，那就是
残差连接（Residual Connection）
。

在标准的Transformer架构中，每一层的输出都会通过残差连接直接加到输入上：

这种设计意味着每一层的输出都以固定权重1直接累加到最终输出中。随着模型层数越来越深（从最初的12层到如今的数百层），这种均匀聚合带来了一个严重问题：
前面层的信息被不断稀释
。

打个比方：这就像你跟朋友传话，第一遍、第二遍还能听清，传到第十遍的时候，前面说的内容早就被"稀释"得差不多了。Transformer的深层同样面临这个问题——前面层学到的关键信息，经过层层残差传递后，最终被"淹没"在大量后续层的输出中。

更糟糕的是，这种固定权重的累加方式无法适应不同输入的实际需求。对于某些需要"回头看"早期信息的复杂推理任务，模型只能通过增加注意力头数或扩大上下文窗口来弥补。

核心创新：用注意力替代固定累加

Kimi团队祭出了
Attention Residuals（注意力残差，简称AttnRes）
这一创新方案。其核心思想：
不再让每层"全都要"，而是让每层能够根据输入内容动态选择哪些之前的层输出更重要
。

图1：三种残差结构对比。(a)标准残差：均匀累加；(b)Full AttnRes：每层都能选择性聚合所有之前的输出；(c)Block AttnRes：分块内标准残差，块间注意力。

具体来说，AttnRes用可学习的softmax注意力机制替代了固定的残差累加。每一层不再简单地将上一层输出"抖"进去，而是有一个专门的"调度员"（伪查询向量）来决定：到底该从之前哪一层获取信息？

核心公式如下：

其中 

 是通过注意力机制计算得到的权重：

这里 

 是第 

 层的伪查询向量（可学习参数），

 是第 

 层的键值向量。当 

 时，

（即初始嵌入）；当 

 时，

。

这样一来，模型可以智能地决定：从第3层获取60%的信息、从第1层获取30%、从第10层获取10%......而不是像传统残差那样每层都"平均主义"。

Block AttnRes：实用的"即插即用"方案

Full AttnRes效果虽好，但计算开销随层数平方增长（

），在超大规模模型上不太现实。Kimi团队进一步提出了
Block AttnRes（分块注意力残差）
，在效果和效率之间取得完美平衡。

Block AttnRes的思路：把层划分为N个分块，在每个分块内部仍然用标准残差累积，只在块与块之间应用注意力机制。这样内存开销从

降到

，而效果几乎不打折扣。

论文数据显示，设置约8个分块就能保留Full AttnRes的大部分收益，同时延迟增加不到2%。

块大小消融实验结果：

块大小 S

验证损失

32

1.757

16

1.753

8

1.748

4

1.746

2

1.746

S≤4时性能趋于饱和，这与"设置约8个分块"的建议一致。

与DeepSeek mHC的对比

提到残差连接的改进，就不得不提DeepSeek在2025年底发布的
mHC (Manifold-Constrained Hyper-Connections)
 架构。两者虽然目标相似（都是改进残差连接），但思路截然不同：

特性

mHC

Attention Residuals

核心思想

将残差连接投影到Birkhoff多面体上，使用Sinkhorn-Knopp迭代

用注意力机制动态学习层间权重

计算方式

需要迭代求解（计算密集）

前向传播时直接计算（更高效）

参数共享

跨层共享变换矩阵

每层独立可学习参数

I/O开销

34d（m=4时）

5.5d

从论文中的消融实验数据来看，AttnRes (input-dependent query) 的Loss为
1.731
，而mHC的Loss为
1.747
——AttnRes表现更好。

mHC的主要问题是
训练不稳定
：27B参数模型训练时会出现损失剧烈波动和梯度范数异常。这是因为mHC在追求性能的过程中，破坏了残差连接最核心的性质——恒等映射（Identity Mapping）。

相比之下，AttnRes保留了恒等映射的稳定性，同时通过注意力机制实现了更智能的信息聚合。

实验结果：48B模型实测

Kimi团队在48B参数的Kimi Linear模型（激活3B，训练1.4T tokens）上进行了大规模实验。

缩放定律：白得1.25倍算力

图2：缩放定律对比。Block AttnRes达到与Baseline相同的Loss只需1.25倍的计算量。

如上图所示，AttnRes在所有计算预算下都一致优于Baseline。关键数据：
Block AttnRes达到了相当于Baseline训练1.25倍计算量才能达到的Loss水平
。

这意味着什么？按照当前大模型训练的算力成本估算，
训练成本直接降低20%
。

下游任务：全面提升，推理和代码提升最显著

类别

基准

Baseline

AttnRes

提升

通用能力

MMLU

73.5

74.6

+1.1

GPQA-Diamond

36.9

44.4

+7.5

BBH

76.3

78.0

+1.7

TriviaQA

69.9

71.8

+1.9

数学与代码

Math

53.5

57.1

+3.6

HumanEval

59.1

62.2

+3.1

MBPP

72.0

73.9

+1.9

中文能力

CMMLU

82.0

82.9

+0.9

C-Eval

79.6

82.5

+2.9

值得关注的结果出现在
多步推理（GPQA-Diamond）
和
代码生成（HumanEval）
上，分别提升了7.5%和3.1%。

训练动态：解决PreNorm稀释问题

图3：训练动态对比。(a)验证损失曲线；(b)输出幅度随层数变化；(c)梯度幅度分布。

看图3(b)：随着层数加深，Baseline的输出幅度呈指数级增长（从接近0飙升到12+），这正是PreNorm稀释问题的表现！而Block AttnRes始终维持在低位（0-3之间），输出幅度保持稳定。

图3(c)更有意思：Baseline的梯度在浅层很大、深层很小（"头重脚轻"），而Block AttnRes的梯度分布更加均匀。这意味着
每层都能有效学习，不会出现梯度消失或爆炸
。

与其他方法的消融对比

变体

Loss

Baseline (PreNorm)

1.766

DenseFormer

1.767

mHC

1.747

AttnRes (input-dependent query)

1.731

AttnRes (input-independent mixing)

1.749

AttnRes (sigmoid)

1.741

Block AttnRes (S=4)

1.746

结论很明确：
使用输入依赖的查询向量（即标准注意力机制）效果最好
，比传统固定残差连接的Baseline降低了0.035的Loss。

与传统方法的结构对比

论文还提供了与Highway、mHC等方法的结构对比：

Highway
：使用可学习的门控标量 

 来控制残差流动

(m)HC
：使用可学习的向量 

 和变换矩阵 

 来计算跨层连接

Full AttnRes
：使用完整的注意力机制，每层可以访问所有前序层的输出

Block AttnRes
：在Full AttnRes基础上引入分块，减少计算开销

技术细节：I/O复杂度分析

对于实际部署，论文分析了不同方案的计算效率：

操作

总I/O (典型值)

Standard Residuals

3d

mHC (m=4)

34d

Block AttnRes

5.5d

Block AttnRes的I/O开销仅为标准残差的约2倍，但带来的性能提升远超这个代价。

总结与展望

Attention Residuals是Kimi团队的一次漂亮的架构创新。
即使是Transformer中最"古老"、最基础的组件，也存在巨大的优化空间。

这项工作让我想到：以前我们只关注层与层之间的信息传递（注意力机制），现在终于有人关注层与层之间的"记忆整合"问题了。

随着大模型层数越来越深（从几十层到几百层），AttnRes这类技术会变得越来越重要。它让模型能够更智能地利用深层信息，而不是被固定权重"绑架"。

资源链接

论文链接：https://github.com/MoonshotAI/Attention-Residuals

GitHub：https://github.com/MoonshotAI/Attention-Residuals

⭐关注我，实时跟进AI最新进展⭐