# UC Berkeley等团队推出Flash-KMeans：GPU加速K-Means极致优化！比FAISS快200倍！

- **来源**: Hyman的杂货铺
- **链接**: https://mp.weixin.qq.com/s?src=11&timestamp=1785309751&ver=6871&signature=w8HCAiG9aIED*s968nBKVSQ4xgfCLeQ-WcVHaDdWiErg9USyUygZtN1rNr-FufgIeVnHgZQsLb2ioLfuGG3Yj*SmG*mOqQOtFfalHpEFV9oSaYU9DnS4Mz7aSjrJDXXk&new=1
- **发布时间**: 2026-03-13 07:30
- **字数**: 10113

---

一句话讲清楚👉🏻

 UC Berkeley 和 UT Austin 的研究团队提出了 Flash-KMeans，通过消除 GPU 内存瓶颈和原子写入竞争，实现了比业界标准 FAISS 快 

200倍

、比 cuML 快 

33倍

 的性能提升，让 K-Means 聚类首次成为可在线部署的实时原语。

研究背景：K-Means 的性能瓶颈在哪里？

K-Means 聚类是机器学习中最经典的无监督学习算法。从数据预处理、特征量化到向量检索、推荐系统，K-Means 几乎无处不在。但随着数据规模从百万级增长到十亿级，传统实现跑不动了。

现有 GPU 实现的两大核心问题

研究团队分析了现有 GPU K-Means 实现（NVIDIA cuML、Facebook FAISS、PyTorch 实现），发现两个关键的性能瓶颈：

1. 分配阶段的 IO 瓶颈

在每次迭代中，K-Means 需要计算所有 N 个数据点与 K 个聚类中心的距离，并找到最近的中心。标准实现会显式构建一个 

N×K 的距离矩阵

：

数据点数量 N = 100万，聚类数 K = 1000 时，距离矩阵需要 

100万个 × 1000 = 10亿

 个浮点数

以 FP16 存储，需要 

2GB

 显存

以 FP32 存储，需要 

4GB

 显存

当 N = 10亿，K = 32768 时，距离矩阵需要 

128TB

 显存——远超任何 GPU 的容量

这导致

显存带宽被大量无意义的中间数据占用

，实际计算被 IO 等待拖慢。

2. 质心更新阶段的原子竞争

找到最近中心后，需要将每个数据点"归并"到对应的聚类中心。标准实现使用

原子加法（atomic add）

：

多个线程同时更新同一个聚类中心时，硬件会串行化这些写入

写入冲突导致线程等待，GPU 计算单元大量空闲

当聚类数 K 较大时，竞争更加激烈

为什么这些问题被忽视了？

虽然 K-Means 的

算法复杂度

早已被充分研究（每次迭代 O(NKD)），但

系统层面的瓶颈

长期被忽视：

学术研究关注理论复杂度，而非硬件约束

工业界使用近似算法（如 Mini-Batch K-Means）绕过问题

GPU 架构的复杂性（内存层次、并行度、原子操作代价）未被充分考虑

研究团队说得很直接：

"瓶颈不在于算法，而在于系统实现"

。所以他们从硬件架构出发重新设计了 K-Means 的执行流程。

核心贡献：三大技术创新

Flash-KMeans 没有改变 K-Means 的数学定义（保持精确性），只是通过

系统级优化

突破性能瓶颈。三个核心创新如下：

1. FlashAssign：融合距离计算与在线最小值搜索

传统方法的问题

：

先计算完整距离矩阵（N×K），再找最小值

距离矩阵占用大量显存，写入和读取耗时

FlashAssign 的解决方案

：

受 FlashAttention 启发，采用

分块流式处理

将数据分块加载到高速 SRAM

在 SRAM 内完成距离计算，

立即计算最小值

，无需存储完整矩阵

完全消除 N×K 距离矩阵的显存占用

类比理解

：就像吃自助餐，传统方法是先把所有菜都拿到桌上（距离矩阵），再慢慢挑选最喜欢的（找最小值）。FlashAssign 则是每拿一道菜就立即判断是否喜欢，不留冗余。

2. Sort-Inverse Update：消除原子写入竞争

传统方法的问题

：

数据点随机分布，更新聚类中心时产生大量原子操作竞争

GPU 硬件需要串行处理冲突的写入请求

Sort-Inverse Update 的解决方案

：

首先按聚类 ID 对数据点

排序

构建

逆映射

：记录每个聚类包含哪些数据点

将随机写入转化为

连续段级归约

利用 GPU 的高带宽顺序访问，避免竞争

类比理解

：传统方法像一群人同时涌向几个收银台结账（原子竞争），容易堵塞。Sort-Inverse Update 则是先让大家排队分组（排序），然后每组按顺序结账（连续访问），高效无堵塞。

3. 算法-系统协同设计：实用化部署

除了核心算法优化，研究团队还解决了实际部署的两大挑战：

挑战一：大规模数据的内存管理

当数据量超过 GPU 显存时，传统方法需要手动分块，效率低下。Flash-KMeans 引入：

异步外核数据流水线

：数据传输与计算重叠

分块流式处理

：智能管理 CPU-GPU 数据流

挑战二：动态形状的编译开销

不同数据规模需要不同的 CUDA kernel 配置，传统方法需要：

自动调优找到最优配置，耗时数小时

或使用通用配置，性能损失 30-50%

Flash-KMeans 引入：

缓存感知编译启发式

：根据数据形状自动选择接近最优配置

调优时间从 

数小时降至数秒

，性能损失 <1%

技术方法详解

FlashAssign：如何避免距离矩阵实体化？

FlashAssign 的核心思想是

计算即消费

：距离计算后立即用于最小值判断，不存储中间结果。

传统分配流程

标准 K-Means 分配阶段的伪代码：

for i in range(N):  # 对每个数据点

    for j in range(K):  # 对每个聚类中心

        dist[i,j] = ||x[i] - c[j]||²  # 计算距离

    assignment[i] = argmin(dist[i,:])  # 找最近中心

问题

：

dist

 矩阵需要 

N×K×4 字节

（FP32）或 

N×K×2 字节

（FP16）的显存。

FlashAssign 的流式处理

FlashAssign 采用分块策略：

for block_i in blocks(N):  # 分块处理数据点

    load x_block to SRAM  # 加载数据块到 SRAM

    min_dist = [∞] * block_size

    min_idx = [0] * block_size

    

    for j in range(K):  # 对每个聚类中心

        load c[j] to SRAM

        compute distances in SRAM  # 在 SRAM 内计算

        update min_dist and min_idx  # 立即更新最小值

        # 不存储距离矩阵！

    

    write assignments to HBM

关键优势

：

SRAM 容量有限，但带宽极高（H200 上约 30TB/s）

距离计算后立即比较，无需写入 HBM

显存占用从 

O(NK)

 降至 

O(N)

（只存储分配结果）

实现细节

FlashAssign 使用 

Triton GPU Kernel

 实现，具有以下特点：

分块策略

：根据 SRAM 大小和计算能力动态调整块大小

在线归约

：在 SRAM 内维护当前最小距离和索引

内存合并访问

：优化数据加载模式，提高带宽利用率

图1：FlashAssign 的流水线执行流程。数据从 HBM 流式加载到 SRAM，距离计算与最小值判断融合，完全避免距离矩阵的实体化。

Sort-Inverse Update：如何消除原子竞争？

质心更新阶段需要聚合所有属于同一聚类的数据点，传统方法使用

原子加法

，竞争严重。

传统更新的竞争问题

标准质心更新伪代码：

for i in range(N):

    cluster_id = assignment[i]

    atomic_add(centroid[cluster_id], x[i])  # 原子操作

问题

：

多个线程同时更新 

centroid[j]

 时，硬件串行处理

GPU 的原子操作延迟约 

100-200 个时钟周期

当聚类数 K 较大时，竞争概率增加

Sort-Inverse Update 的三步流程

步骤 1：按聚类 ID 排序

sorted_indices = argsort(assignment)  # O(N log N)

排序后，属于同一聚类的数据点连续存储。

步骤 2：构建逆映射

inverse_map = {}  # 聚类ID -> 数据点索引列表

for i in sorted_indices:

    cluster_id = assignment[i]

    inverse_map[cluster_id].append(i)

步骤 3：段级归约

for cluster_id in range(K):

    points = inverse_map[cluster_id]

    centroid[cluster_id] = sum(x[points]) / len(points)

关键优势

：

排序后，每个聚类的数据点连续存储

归约操作访问连续内存，GPU 带宽利用率高

完全避免原子操作竞争

图2：Sort-Inverse Update 的执行机制。(a) 传统方法：随机写入导致原子竞争。(b) 新方法：排序后连续访问，无竞争。(c) 执行时间线对比：新方法显著减少等待时间。

性能对比

在 NVIDIA H200 GPU 上的测试结果：

方法

N=1M, K=1000

N=100M, K=10000

N=1B, K=32768

传统原子更新

12ms

1.2s

26s

Sort-Inverse Update

2ms

0.2s

4.1s

加速比

6×

6×

6.3×

实验设计与结果

研究团队在 NVIDIA H200 GPU（显存 141GB，带宽 3.35TB/s）上进行了全面的性能评估，对比了四个主流基线：

fast_pytorch_kmeans

：PyTorch 实现的 K-Means

fastkmeans

：Triton 实现的 K-Means

NVIDIA cuML

：NVIDIA 官方 GPU 加速库

Facebook FAISS

：业界标准的向量检索库

端到端性能对比

图3：Flash-KMeans 与主流基线的端到端延迟对比。y轴为对数尺度，红色标记表示内存溢出（OOM）失败。

关键结果

：

相比最佳基线 fast_pytorch_kmeans

：

平均加速 

17.9×

在大 K 场景（K=32000）加速高达 

21.2×

相比 NVIDIA cuML

：

平均加速 

33×

cuML 在大 K 场景出现 OOM，Flash-KMeans 稳定运行

相比 Facebook FAISS

：

平均加速 

超过 200×

FAISS 在大 N 场景性能急剧下降

内存效率

：

Flash-KMeans 在所有测试中未出现 OOM

传统方法在大 K 或大 N 场景频繁 OOM

核函数级性能分解

图4：FlashAssign 和 Sort-Inverse Update 的核函数级加速效果。左：分配阶段加速；右：更新阶段加速。

FlashAssign 性能

：

相比标准实现加速 

高达 21.2×

完全消除距离矩阵的内存 IO

在大 K 场景优势最明显

Sort-Inverse Update 性能

：

相比标准原子更新加速 

高达 6.3×

消除原子竞争带来的等待时间

性能稳定，受聚类分布影响小

大规模数据测试：十亿级数据点

Flash-KMeans 的核心优势在于

可扩展性

。研究团队测试了极端大规模场景：

数据规模

聚类数 K

维度 D

Flash-KMeans

基线

加速比

100M

16384

128

8.4s

88.4s

10.5×

400M

16384

128

15.2s

OOM

-

1B

32768

128

41.4s

261.8s

6.3×

关键发现

：

Flash-KMeans 成功处理 

10亿数据点

 的聚类任务

基线方法在大规模场景出现 OOM 或性能崩溃

异步流水线设计保证了外核数据的高效处理

编译启发式的效果

图5：缓存感知编译启发式的效果。左：编译时间对比；右：运行性能对比。

传统自动调优的问题

：

需要测试数十种配置

编译时间长达 

数小时

每次数据形状变化需重新调优

Flash-KMeans 的启发式方法

：

根据数据形状自动推断最优配置

编译时间降至 

数秒

调优开销减少 

175×

性能损失 <1%

消融实验分析

为了验证各个组件的贡献，研究团队进行了详细的消融实验：

组件

移除后性能损失

FlashAssign

-85%（恢复到标准实现）

Sort-Inverse Update

-40%

异步流水线

-15%（大规模数据）

编译启发式

+0.8%（调优时间增加）

结论

：FlashAssign 是最关键的组件，Sort-Inverse Update 提供了重要的额外加速。

与业界标准的详细对比

相比 NVIDIA cuML

cuML 是 NVIDIA 官方的 GPU 加速机器学习库，采用了多项优化：

使用 CUDA 实现的 K-Means

支持多 GPU 分布式计算

与 Scikit-Learn API 兼容

Flash-KMeans 的优势

：

33× 加速

：主要来自避免距离矩阵实体化

更好的显存效率：cuML 在大 K 场景下容易 OOM

更简单的 API：支持批量处理，无需手动分块

相比 Facebook FAISS

FAISS 是业界标准的向量检索库，其 K-Means 实现经过高度优化：

支持 GPU 加速

广泛应用于向量数据库（如 Milvus、Pinecone）

支持近似最近邻搜索

Flash-KMeans 的优势

：

200× 加速

：来自系统级优化

精确算法：不引入近似误差

更适合实时应用：低延迟，高吞吐

为什么 FAISS 这么慢？

研究团队分析发现，FAISS 的主要瓶颈在于：

分配阶段：显式构建距离矩阵

更新阶段：频繁的 GPU-CPU 数据交换

内存管理：缺乏细粒度的内存复用

这些瓶颈在 Flash-KMeans 中被系统性解决。

实际应用案例

研究团队展示了 Flash-KMeans 在实际场景中的应用：

案例一：视频生成中的语义聚类

Sparse VideoGen2（SVG2）是一个加速视频生成的项目，需要：

对视频 token 进行语义聚类

根据聚类结果重排注意力模式

每个视频帧需要处理数万个 token

使用 Flash-KMeans 后：

聚类时间从 

数秒降至数十毫秒

视频生成整体加速 

15-20×

显存占用减少 

50%

案例二：向量数据库的实时更新

向量数据库需要：

对新向量进行聚类

更新索引结构

保持查询延迟稳定

使用 Flash-KMeans 后：

索引更新延迟从 

分钟级降至秒级

支持更大规模的向量（十亿级）

硬件成本降低 

70%

局限性与未来工作

尽管 Flash-KMeans 取得了显著成果，研究团队诚实地讨论了局限性：

当前局限性

1. 仅支持欧氏距离

Flash-KMeans 目前只实现了欧氏距离（L2 距离）。对于其他距离度量（如余弦相似度、曼哈顿距离），需要额外开发。

扩展方案

：

余弦相似度：可以先对向量归一化，然后使用欧氏距离

其他距离：需要重新设计 FlashAssign 的融合策略

2. Triton 编译器的限制

Flash-KMeans 使用 Triton 编写 GPU kernel，但：

Triton 对某些 GPU 架构支持不完善

编译器优化可能不是最优的

调试和性能分析工具有限

未来改进

：

提供 CUDA 原生实现作为备选

与 NVIDIA 合作优化 Triton 编译器

3. 多 GPU 扩展

当前版本主要针对单 GPU 优化，多 GPU 场景下：

需要额外的数据分片和通信

负载均衡策略需要重新设计

通信开销可能抵消加速效果

未来工作

：

开发多 GPU 版本

优化通信与计算的重叠

未来研究方向

1. 扩展到其他聚类算法

Flash-KMeans 的优化思想可以推广到：

Mini-Batch K-Means

：进一步降低内存需求

K-Means++ 初始化

：优化初始中心选择

层次聚类

：递归应用 FlashAssign

DBSCAN

：密度聚类的不同挑战

2. 与深度学习框架集成

将 Flash-KMeans 集成到主流框架：

PyTorch

：作为原生算子，支持自动微分

TensorFlow

：提供 TF 原生接口

JAX

：支持 JIT 编译和自动向量化

3. 在线学习场景

探索 K-Means 在在线学习中的应用：

流式数据聚类

增量更新聚类中心

自适应调整聚类数

4. 硬件适配

将 Flash-KMeans 移植到其他硬件平台：

AMD GPU

：使用 ROCm 和 HIP

Intel GPU

：使用 OneAPI

Apple Silicon

：使用 Metal

TPU

：使用 JAX/XLA

对 AI 系统设计的启示

Flash-KMeans 的成功为 AI 系统设计提供了重要启示：

启示一：系统优化与算法优化同样重要

传统研究往往关注算法复杂度（Big-O），但忽视：

内存层次结构的利用

并行度与同步开销的权衡

硬件特性的适配

Flash-KMeans 证明：

系统级优化可以带来数量级的性能提升

，甚至超过算法改进。

启示二：IO 感知设计是关键

现代 AI 系统往往受限于内存带宽，而非计算能力：

Transformer 的注意力机制：FlashAttention 通过融合计算避免中间矩阵

K-Means 的距离计算：FlashAssign 通过在线归约避免距离矩阵

向量检索的索引构建：通过流水线处理隐藏 IO 延迟

设计原则

：

尽量减少中间结果的实体化，让数据流动起来

。

启示三：原子操作是并行计算的隐形成本

GPU 的并行能力强大，但：

原子操作引入串行化

竞争导致线程等待

性能难以预测

设计原则

：

通过算法重排（如排序）减少竞争，或使用无锁数据结构

。

启示四：实用化需要算法-系统协同优化

研究不仅要追求峰值性能，还要考虑：

易用性：开箱即用，无需复杂配置

鲁棒性：处理各种边界情况

可维护性：代码清晰，易于调试

Flash-KMeans 的编译启发式就是一个好例子：牺牲 1% 的性能，换取 175× 的调优时间减少。

实际部署建议

研究团队提供了实际部署 Flash-KMeans 的建议：

安装与使用

# 使用 pip 安装

pip install flash-kmeans

# 或从源码安装

git 

clone

 https://github.com/svg-project/flash-kmeans.git

cd

 flash-kmeans

pip install -e .

基本用法

import

 torch

from

 flash_kmeans 

import

 batch_kmeans_Euclid

# 准备数据：32 个批次，每批次 75600 个点，128 维

x = torch.randn(

32

, 

75600

, 

128

, device=

"cuda"

, dtype=torch.float16)

# 运行 K-Means

cluster_ids, centers, _ = batch_kmeans_Euclid(

    x, 

    n_clusters=

1000

, 

    tol=

1e-4

, 

    verbose=

True

)

性能调优建议

1. 数据类型选择

FP16：速度最快，精度略低（适合大规模数据）

FP32：精度高，速度略慢（适合精度敏感场景）

2. 批处理大小

大批处理：提高 GPU 利用率，但显存占用高

小批处理：显存占用低，但可能降低性能

建议：根据显存大小动态调整

3. 聚类数 K 的选择

K < 1000：FlashAssign 加速最显著

K = 1000-10000：整体性能提升明显

K > 10000：显存优势突出

与现有系统集成

与 PyTorch 集成

：

from

 flash_kmeans 

import

 KMeans  

# 类似 sklearn API

kmeans = KMeans(n_clusters=

100

, device=

"cuda"

)

labels = kmeans.fit_predict(data)

与向量数据库集成

：

# 在索引构建中使用

from

 flash_kmeans 

import

 batch_kmeans_Euclid

def

 

build_index

(

vectors, n_clusters

):

    _, centroids, _ = batch_kmeans_Euclid(

        vectors.unsqueeze(

0

),  

# 添加批次维度

        n_clusters=n_clusters

    )

    

return

 centroids.squeeze(

0

)

总结与展望

Flash-KMeans 通过系统级优化，让 K-Means 从"离线预处理工具"蜕变为"在线实时原语"，为大规模 AI 系统开辟了新的可能性。

核心成果总结

性能突破

：

比最佳基线快 

17.9×

比 cuML 快 

33×

比 FAISS 快 

200×

支持十亿级数据处理

技术创新

：

FlashAssign：融合距离计算与最小值搜索，消除距离矩阵

Sort-Inverse Update：通过排序消除原子竞争

编译启发式：将调优时间从数小时降至数秒

实用价值

：

开源代码，开箱即用

支持批量处理，API 简单

已集成到 Sparse VideoGen2 项目

对研究社区的影响

Flash-KMeans 的成功将启发更多

系统导向的算法优化

研究：

重新审视经典算法

：许多"已解决"的算法可能仍有巨大的优化空间

跨领域灵感迁移

：FlashAttention 的思想成功迁移到 K-Means

系统-算法协同设计

：不再是算法先行、系统适配，而是协同优化

对工业界的意义

对于依赖大规模聚类的应用场景，Flash-KMeans 带来显著价值：

向量数据库

：实时索引更新，降低硬件成本

推荐系统

：快速用户/物品聚类，提升响应速度

视频生成

：加速 token 聚类，提升生成效率

科学计算

：处理更大规模的数据，加速发现周期

下一步行动

如果你从事以下工作，强烈建议尝试 Flash-KMeans：

向量检索与数据库

：索引构建、实时更新

推荐与广告系统

：用户分群、物品聚类

视频与图像处理

：特征量化、压缩

科学数据分析

：大规模数据探索

资源链接

📄 论文链接：https://arxiv.org/abs/2603.09229

💻 GitHub：https://github.com/svg-project/flash-kmeans

⭐️关注我，实时跟进AI最新进展⭐️