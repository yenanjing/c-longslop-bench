# DeepSeek V4蓄势待发，发布最新论文DualPath：用空闲网卡打破Agent推理的存储带宽瓶颈

- **来源**: Hyman的杂货铺
- **链接**: https://mp.weixin.qq.com/s?src=11&timestamp=1785309748&ver=6871&signature=w8HCAiG9aIED*s968nBKVSQ4xgfCLeQ-WcVHaDdWiEp1wbBxPc-k2Alf5eyK9brafeqZCxV1ACGVGx1zApV9hYg7R-k4SlA3qUnJmXsGboK4JXw-eh4ojX0Ovx1gH0Jq&new=1
- **发布时间**: 2026-02-28 07:30
- **字数**: 3734

---

一句话讲清楚👉🏻 DeepSeek联合北大清华提出DualPath，通过双路径KV-Cache加载，将Agent推理吞吐量提升

1.87倍

，关键洞察是"KV-Cache加载不必以Prefill为中心"。

当全球AI界都在关注DeepSeek的V4何时发布时，它悄悄在arXiv上发表了一篇与北大、清华合作的论文——DualPath。这篇论文直击Agent推理的核心痛点：

存储带宽瓶颈

。

更妙的是，这个思路与前几天DeepSeek V4计算力话题密切相关——

计算免费，数据移动昂贵

。

Agent推理的I/O之痛

从计算瓶颈到I/O瓶颈

现代LLM推理架构采用

Prefill-Decode分离（PD分离）

设计：Prefill引擎负责处理新输入，Decode引擎负责生成输出。这种架构在单轮对话中表现良好。

但Agent场景完全不同：

Agent执行多轮推理、多步规划、工具调用，呈现出"

长上下文、短追加、多轮次

"的特征。DeepSeek的生产数据显示，32K-64K token上下文的KV-Cache命中率高达

98.7%

。

高命中率是好事？不完全是。

高命中率意味着每轮对话都要从存储加载

大量KV-Cache

，系统从计算受限变成了

I/O受限

。

不对称的资源浪费

在PD分离架构中，KV-Cache只能从存储加载到Prefill引擎。这导致：

Prefill引擎的存储网卡

：带宽瞬间饱和

Decode引擎的存储网卡

：完全空闲

这就是DualPath要解决的核心矛盾。

图：传统架构（左）vs DualPath（右）。传统方式下Prefill侧存储网卡饱和，Decode侧闲置；DualPath通过双路径均衡负载

DualPath的核心创新

双路径KV-Cache加载

DualPath的核心洞察是：

KV-Cache加载不必以Prefill为中心

。

传统系统只支持一条路径：

Storage → Prefill

DualPath引入第二条路径：

Storage → Decode → Prefill

（通过RDMA计算网络传输）

这意味着：

Decode引擎的存储网卡不再闲置

全局存储带宽被充分利用

Prefill引擎的带宽压力被分摊

三大技术支柱

DualPath的实现依赖三个关键组件：

(1) 双路径加载（DPL）

动态选择Storage-to-Prefill或Storage-to-Decode路径

CNIC辅助的KV-Cache拷贝，避免本地PCIe瓶颈

严格流量隔离，防止干扰延迟敏感的模型执行

(2) 全局调度器

引擎间调度

：选择最优Prefill引擎处理请求

引擎内调度

：基于计算配额的批处理选择

(3) 分层预填充

Layerwise Prefill技术

逐层加载KV-Cache，优化TTFT

图：PE读取路径——传统方式从存储直接加载到Prefill引擎

图：DE读取路径——DualPath新增的路径，先加载到Decode引擎，再通过RDMA传输到Prefill

性能表现：近Oracle水平

实验基于三个模型：DeepSeek 27B、DeepSeek 660B（V3）、Qwen 32B，使用真实Agent工作负载评估。

离线推理吞吐量

模型

上下文长度

Agent数量

相比Baseline提升

DS 27B

64K

1024

1.69×

DS 660B

64K

1024

1.87×

Qwen 32B

64K

1024

1.76×

DualPath在660B模型上达到接近Oracle（零I/O开销的理论上限）的性能。

图：不同模型和上下文长度下的离线推理性能对比，DualPath显著优于Baseline

在线服务吞吐量

在线服务场景下，设置SLO为：TTFT ≤ 4秒，TPOT ≤ 50ms。

模型

基线吞吐(APS)

DualPath吞吐(APS)

提升倍数

DS 27B

0.4

0.8

2.0×

DS 660B

0.4

0.7

1.75×

平均提升1.96倍

，且TTFT显著降低，TPOT保持稳定。

图：在线服务性能——TTFT、TTST、TPOT随Agent到达率的变化

大规模扩展性

在1152个GPU（48P96D配置，48000个Agent）上的测试表明：

离线推理：近似线性加速

在线服务：吞吐量达8.8 APS，是基线的

22倍

调度器CPU使用率低于10核，不构成瓶颈

图：48P96D大规模离线推理指标，展示近似线性扩展性

技术深度解析

为什么是存储瓶颈？

Agent推理的关键特征是

高KV-Cache复用

。每轮对话时：

加

载

量

上

下

文

长

度

层

数

头

数

头

维

度

对于660B模型，64K上下文的KV-Cache约

数十GB

。而GPU计算能力的增长远快于网络带宽和HBM容量的增长，这使得I/O成为瓶颈。

论文给出了硬件趋势数据：

H100比A100：计算提升3.1×，但存储带宽仅提升1.5×

这个差距在Agent场景下被放大

双路径如何工作？

DualPath的调度逻辑是工作负载感知的：

(1) 路径选择

当请求到达时，全局调度器评估：

Prefill引擎存储带宽利用率

Decode引擎存储带宽可用性

计算网络拥塞状态

选择最优路径加载KV-Cache。

(2) CNIC辅助拷贝

传统方式：

Storage → CPU内存 → GPU HBM（经PCIe）

DualPath的Storage-to-Decode路径：

Storage → Decode引擎HBM → RDMA → Prefill引擎HBM

关键优化

：使用GPU Direct RDMA（GDRDMA），绕过CPU和PCIe瓶颈。

(3) 流量隔离

计算网络承载两类流量：

KV-Cache传输

：大带宽，可容忍延迟

模型执行通信

：延迟敏感

DualPath通过QoS机制严格隔离这两类流量，确保KV-Cache传输不干扰模型推理。

调度器设计

引擎间调度（Inter-Engine）

图：引擎间调度示例——8个GPU在同一PE组内，调度器选择最优引擎

调度器维护全局视图：

各Prefill引擎的负载状态

KV-Cache分布位置

存储带宽利用率

目标是：

最小化TTFT

 + 

最大化带宽利用率

。

引擎内调度（Intra-Engine）

图：引擎内调度——基于计算配额的批处理选择

核心思想是

计算配额

：

每个请求分配计算配额

配额基于请求的KV-Cache大小和生成长度

调度器选择配额总和最优的批次

这避免了传统批处理中"长请求阻塞短请求"的问题。

TTFT分解

论文给出了TTFT的详细分解：

其中：

：调度延迟

：内存分配

：KV-Cache读取

：预填充计算

DualPath主要优化

，通过双路径将读取带宽翻倍。

图：TTFT分解——DualPath显著减少KV-Cache读取时间

消融实验

论文通过消融实验验证了各组件的贡献：

组件

性能提升

Layerwise Prefill

+0.23×

Dual-Path Loading

+0.89×

全局调度器

+0.35×

完整系统

+1.87×

双路径加载是最大的贡献者，验证了核心假设的正确性。

负载均衡效果

DualPath的另一个关键效果是

存储网卡流量均衡

。

图：存储网卡流量负载均衡——DualPath让所有网卡的带宽利用率趋于一致

传统架构下，Prefill引擎网卡负载高，Decode引擎网卡闲置。DualPath实现了全局均衡，充分利用所有可用带宽。

意义与展望

DualPath的核心贡献不仅是性能提升，更在于

重新定义了问题

：

I/O瓶颈不是根本性的

，而是资源利用不均的结果

KV-Cache加载不必以Prefill为中心

，这个洞察可以推广到更多场景

计算免费，数据移动昂贵

——这个原则将在AI基础设施设计中愈发重要

与V4的关系

论文开头提到的"计算力"话题，正是DualPath的设计哲学：在硬件受限条件下，通过系统优化榨取性能。这很可能是DeepSeek V4架构设计的核心理念之一。

未来方向

论文提到几个扩展方向：

支持RDMA over Converged Ethernet（RoCE）

适配UnifiedBus、Ultra Ethernet等新兴技术

将调度逻辑推广到更多异构计算场景

论文地址：https://arxiv.org/abs/2602.21548

⭐️关注我，实时跟进AI最新进展⭐️