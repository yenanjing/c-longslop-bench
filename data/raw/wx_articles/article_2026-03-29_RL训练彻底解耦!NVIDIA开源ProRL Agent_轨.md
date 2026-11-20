# RL训练彻底解耦！NVIDIA开源ProRL Agent：轨迹生成即服务，SWE-Bench 8B模型性能提升近2倍

- **来源**: Hyman的杂货铺
- **链接**: https://mp.weixin.qq.com/s?src=11&timestamp=1785309752&ver=6871&signature=w8HCAiG9aIED*s968nBKVSQ4xgfCLeQ-WcVHaDdWiEql0gOCPSaWN87MSNIFT2gV*MbT9iix3*rUTuV8Jt91nK9B4i2UIEm*vrbho7-oyqjq10FpSmKqeY3zImbdYq2b&new=1
- **发布时间**: 2026-03-29 07:30
- **字数**: 7701

---

一句话讲清楚👉🏻

 NVIDIA提出ProRL Agent，把多轮LLM Agent的RL训练中「轨迹生成（Rollout）」这一步从训练框架中彻底剥离出来，变成一个独立的HTTP服务，训练侧只需发HTTP请求就能拿到轨迹和奖励信号，不再需要自己管理沙箱、工具调用和评测——SWE-Bench Verified上8B模型性能相比SkyRL-v0提升近2倍。

为什么多轮Agent的RL训练是个麻烦事

强化学习（RL）在LLM推理能力提升上已经证明了其价值，从DeepSeek-R1到ProRL，单轮任务上的RL训练效果有目共睹。但当RL从"做一道数学题"扩展到"修一个GitHub issue"时，情况完全不同了。

多轮Agent任务的典型特征是：Agent需要和外部环境反复交互——读代码文件、写代码、运行测试、查看报错、再修改……一个软件工程任务的单次轨迹（rollout）可能涉及几十轮工具调用，每一轮的延迟高度不确定，取决于代码执行结果和环境响应。

这里有一个很实际的工程矛盾：

轨迹生成是I/O密集型的

（创建沙箱、管理容器、异步协调数百个并发实例），而

策略训练是GPU密集型的

（前向传播、反向传播、梯度同步）。把这两种工作负载绑在同一个进程里，互相拖累，谁都干不好。

现有的Agent RL训练框架——SkyRL-Agent、VeRL-Tool、Agent Lightning、rLLM、GEM——虽然各有所长，但有一个共同的设计缺陷：

轨迹生成的整个生命周期被嵌入在训练循环内部

。这意味着：

换一个训练后端，就得重新实现整个Agent执行流水线

改善轨迹基础设施（比如支持新的运行环境或任务类型），改动会波及训练代码

轨迹节点和训练节点的资源无法独立优化

NVIDIA团队借鉴了推理即服务（inference-as-a-service）的思路，提出了

Rollout-as-a-Service

的核心设计理念，将多轮Agent的完整轨迹生命周期从训练框架中彻底解耦。

ProRL Agent：轨迹生成变成独立的HTTP服务

ProRL Agent是一个开源的、可扩展的多轮Agent轨迹基础设施，核心思路很直接：

轨迹生成是一个独立的HTTP服务，训练框架通过API交互即可获取轨迹和奖励

。

训练框架只需要：

通过 

POST /process

 提交任务实例

通过 

POST /add_llm_server

 管理推理后端

通过 

POST /cancel

 取消过时的任务

然后就能收到完整的轨迹和奖励信号。训练侧完全不需要关心沙箱怎么创建、工具怎么调用、奖励怎么计算。

耦合式 vs. 解耦式设计对比。左图：现有框架将Agent的完整轨迹生命周期嵌入RL训练栈中。右图：ProRL Agent将轨迹生成作为独立的HTTP服务，训练侧只需提交请求和接收轨迹。

这种解耦设计带来了三个实际好处：

独立开发和部署

：轨迹节点和训练节点可以分别优化，各自追求更大的吞吐量

新任务零侵入添加

：只需在轨迹服务端实现一个handler插件，训练代码完全不用改

Agent脚手架自由替换

：轨迹服务和Agent实现完全解耦，脚手架可以随时替换而不影响训练基础设施

系统架构全景

ProRL Agent的架构由三个核心组件构成：可扩展沙箱环境、ProRL Agent Server、以及RL训练后端。

ProRL Agent架构概览。系统由三部分组成：(1) 沙箱环境——每个轨迹在SingularityRuntime容器中执行，通过AgentHandler编排三个生命周期方法；(2) ProRL Agent Server——通过三阶段异步流水线管理轨迹，维护支持动态注册和checkpoint热切换的LLM后端池；(3) RL训练器——任何训练框架通过HTTP与服务交互。

组件一：可扩展沙箱环境

多轮Agent训练需要一个沙箱层，能够兼容异构任务环境，同时在HPC集群上无特权运行。ProRL Agent通过两个子系统实现了这一点。

可插拔任务抽象——AgentHandler

不同类型的Agent任务（软件工程、数学推理、计算机使用等）需要各自的环境设置、Agent行为和奖励计算。ProRL Agent把这些差异封装在一个抽象接口 

AgentHandler

 中，定义了三个核心生命周期方法：

init()

：初始化沙箱环境，配置Agent的工具集

run()

：在沙箱中驱动多轮Agent循环，收集行动-观测轨迹

eval()

：对Agent输出进行评分，返回标量奖励信号

每个handler还暴露了各阶段的异常回调（

init_exception

、

run_exception

、

eval_exception

）和最终结果序列化方法 

final_result()

，确保即使轨迹中途失败，服务也能返回结构化的输出。

当服务收到一个任务时，它读取任务实例，在注册表中查找对应的handler，按顺序调用其生命周期方法。

新增一个任务域，只需实现一个新的handler并注册

，服务核心不需要任何改动。

HPC兼容的容器运行时——SingularityRuntime

大多数沙箱环境假设Docker可用，但HPC集群通常禁止Docker守护进程，要求所有用户进程在无root权限下运行。ProRL Agent选择基于Singularity构建容器系统：

无守护进程

：容器作为非特权用户进程的子进程启动，不需要root权限

独立loopback IP

：每个容器实例通过线程安全的分配器获得127.x.x.x范围内的唯一loopback IP，避免端口冲突

Fakeroot模式

：通过 

--fakeroot

 在容器内模拟root权限安装软件包，不需要宿主机特权

可选网络隔离

：

--network none

 可禁用外部网络访问，防止轨迹间干扰

SIF格式

：容器镜像打包为单个便携的 

.sif

 文件，特别适合Slurm共享文件系统

高效的工具后端

Agent主要通过工具与环境交互：读写文件、执行shell命令、运行Python代码等。每次工具调用都是同步阻塞操作，一个典型轨迹涉及几十次调用，单次延迟累积起来会成为主要瓶颈。ProRL Agent针对三个关键工具后端做了深度优化：

高效Bash

：用ptyprocess直连伪终端替代tmux会话，消除终端复用器的开销，shell命令往返延迟显著降低

持久IPython

：通过进程内API直连IPython内核，跳过Jupyter kernel gateway的网络往返，变量和import在多步间保持

UDS通信

：用Unix Domain Socket替代TCP loopback，通过OS内核直接传递消息，没有网络协议栈开销

这三项优化确保工具执行不会在数百个并行Agent扩展时成为吞吐瓶颈。

组件二：ProRL Agent Server

Server的核心职责是在RL训练期间编排数百个并发轨迹，同时给训练框架提供对轨迹基础设施的实时控制。

Server通过两个机制满足两个基本需求：

三阶段异步流水线

轨迹生成过程就像一条装配线。朴素实现会让一个worker负责整个任务：启动容器、执行Agent、评分结果，然后再处理下一个任务。问题是三个阶段的耗时和资源需求差异巨大：容器启动受I/O限制，Agent执行受LLM推理限制，评测可能几毫秒也可能几分钟。一个串行处理的worker大部分时间在空闲等待最慢的那个阶段。

ProRL Agent的解决方案是

把三个阶段解耦

，就像工厂把装配工位拆开一样。AgentHandler的三个生命周期方法映射到三个独立的worker池，各自有独立的队列：

初始化worker持续拉取新任务，启动容器，交给轨迹队列

轨迹worker驱动Agent循环，将完成的轨迹交给评测队列

评测worker评分结果，返回给调用方

在任何时刻，三个池子同时处理不同的任务：一个在评测，一个在执行轨迹，一个在启动容器。而且三个池子可以独立调整大小——比如初始化慢就多配一些init worker，测试套件长就多配一些eval worker。

LLM后端管理

每一步Agent循环都需要一次LLM完成调用。当数百个轨迹并行运行时，推理请求高频率到达，单个LLM服务器（如vLLM）很快成为瓶颈。

ProRL Agent Server直接管理推理后端池，提供：

动态注册

：训练框架通过 

POST /add_llm_server

 随时注册新的推理服务器

Checkpoint热切换

：策略checkpoint更新后，训练框架调用 

POST /clear_llm_server

 清除所有旧后端，重新注册加载了新权重的服务器端点。后续轨迹自动使用新模型，流水线中的已有任务不受影响

最小堆负载均衡

：每个LLM后端关联一个分配计数器，存放在最小堆中。每次需要推理调用时，选择计数器最小的后端，并**按任务粒度（而非单次调用粒度）**递增计数器。这确保同一任务的所有调用都路由到同一后端，最大化prefix cache复用

关键技术细节

Token-in/Token-out：消除重新分词漂移

如果轨迹以纯文本形式在训练流水线中传输，训练侧的重新分词可能是有损的——生成的token序列可能与原始rollout时不一致，导致非预期的off-policy偏差。

ProRL Agent通过

全程使用token ID作为规范表示

来消除这个问题。轨迹worker将 

prompt_ids

 直接发送给LLM后端，收到带逐token对数概率的 

response_ids

；每条消息额外携带 

input_ids

、

output_ids

 和 

logprobs

 字段，在生成时填充并在整个过程中不变传递。多轮轨迹中，先前的assistant轮次保留原始token ID，直接拼接到输入缓冲区；只有新消息（如环境观测）才需要分词追加。

高效的DAPO实现

ProRL Agent采用DAPO（Dynamic Sampling Policy Optimization）作为核心RL算法。DAPO通过过滤零方差prompt（所有rollout奖励一致的prompt）来提升训练稳定性和数据效率，但将其应用到Agent RL面临挑战：Agent轨迹通常长时间运行、异步且计算昂贵。

朴素的批处理实现——请求n个prompt，过滤掉无信息的，反复触发新批次直到收集到n个有信息prompt——效率极低，会导致worker空闲和冗余轨迹。

ProRL Agent实现了

异步补充机制

：

持续吞吐

：队列一空就补充任务，维持最大轨迹吞吐量

提前终止

：一旦收集到目标数量的有信息prompt，立即终止剩余活跃任务

跨迭代持久化

：未完成的任务带到下一个迭代，保留部分进度

这一优化显著减少了worker空闲时间，提升了整体硬件利用率。

任务生命周期与取消机制

每个任务关联一个 

PausableTimer

，只在活跃流水线阶段（init、run、eval）累积耗时，排除了在阶段间队列等待的时间，确保超时预算反映实际执行时间而非瞬态服务端延迟。

训练框架可通过 

POST /cancel

 随时中止任何进行中的任务。收到取消请求后，Server会：标记任务为丢弃、取消当前执行的异步任务、关闭关联的容器释放资源、发出完成信号使等待的HTTP处理器立即返回。

此外，每个流水线阶段注册了专用的异常回调，单个轨迹失败不会阻塞共享的worker池。收到 

POST /stop

 后，Server取消所有进行中的任务，通过进程组扫描终止Singularity进程，清空worker池后干净退出。

实验验证

软件工程任务：SWE-Bench Verified

ProRL Agent主要在软件工程任务上进行评估。研究团队在SWE-Gym的293个实例子集上训练了Qwen3-4B、Qwen3-8B和Qwen3-14B三个规模的模型。

结果如下：

4B模型

：从基线的14.8提升到21.2（+43%）

8B模型

：从基线的9.6提升到18.0，相比SkyRL-Agent-8B-v0报告的9.4提升近2倍

14B模型

：从基线的15.4提升到23.6，超越SkyRL-Agent-14B-v0报告的21.6

在所有模型规模上，ProRL Agent都带来了一致的性能提升，8B模型上的提升尤为突出。

多领域泛化验证

ProRL Agent的通用性不止于软件工程。研究团队还在STEM、数学和编程三个领域进行了验证。

STEM Agent

：装备了web搜索（Tavily）、Bash和IPython工具，用于跨科学、技术、工程和数学的复杂问答任务。使用SCP-116K数据集训练，60个训练步后平均奖励从约0.2稳步提升到约0.65，曲线尚未饱和，暗示更多训练可能带来进一步提升。

STEM Agent的RL训练曲线。平均奖励从约0.2稳步上升到约0.65，60步后仍未饱和。

Math Agent

：以IPython执行为主要工具，预装了NumPy、SciPy和SymPy等科学计算库，还配备了think工具用于显式规划。使用DeepScaleR数据训练，AMC上Pass@1从0.4提升到约0.9。基础模型最初不太擅长通过工具使用解决数学问题，但RL训练让Agent学会了有效利用外部工具进行数学推理。

Math Agent的RL训练曲线。AMC上Pass@1从0.4提升到约0.9，Agent学会了通过IPython工具进行数学推理和验证。

Code Agent

：以 

str_replace_editor

 文件编辑为主要工具，配合Bash执行测试脚本和IPython快速原型开发。采用测试驱动的训练设置，使用Eurus-2-RL-Data训练，Codeforces上Pass@1从0.23提升到约0.42。RL训练显著改善了Agent使用文件编辑工具和基于测试验证的能力。

Code Agent的RL训练曲线。Codeforces上Pass@1从0.23提升到约0.42，Agent学会了通过文件编辑和测试验证进行代码生成。

三个领域的训练曲线都稳步上升，说明ProRL Agent不只适用于软件工程这一类Agent任务。

系统分析

线性扩展能力

ProRL Agent的吞吐量随计算节点数量近似线性增长，这意味着可以有效地利用额外的计算资源，扩展开销很小。对于RL训练来说这一点尤为关键——高效的轨迹生成通常是整个训练系统的主要瓶颈，直接影响总体训练效率。

软件工程任务的轨迹吞吐量（实例/秒）随计算节点数的变化。近似线性的增长表明ProRL Agent能高效利用额外的计算资源。

组件消融实验

研究团队对三个关键组件进行了消融分析：负载均衡（LB）、高效Bash（EB）和过时任务清理（SC）。

移除负载均衡后，GPU利用率从78%骤降到42%，吞吐量从0.37降到0.25实例/秒

移除高效Bash后，动作执行时间从0.42秒增加到0.78秒，GPU利用率降到68%，吞吐量降到0.29

移除过时任务清理后，GPU利用率降到65%，吞吐量降到0.30

每个组件都对更高的轨迹吞吐量有贡献：负载均衡和过时任务清理通过提升GPU利用率改善吞吐量，高效Bash通过减少动作执行时间改善吞吐量。

与现有框架的对比

ProRL Agent与现有Agent RL框架的关键区别在于三个维度：

框架

训练-轨迹解耦

无root沙箱

脚手架无关

SkyRL-Agent

❌

❌

✅

VeRL-Tool

❌

❌

✅

Agent Lightning

❌

❌

❌

rLLM

❌

❌

✅

GEM

❌

❌

✅

ProRL Agent

✅

✅

✅

在附录中，研究团队详细分析了每个现有框架的架构局限：

SkyRL-Agent

：训练驱动器运行在单个CPU进程上，轨迹控制仍留在训练驱动器内部

Agent Lightning

：训练循环、LightningStoreServer和所有轨迹worker在同一进程树中，训练进程终止时store也停止，轨迹worker被中断

VeRL-Tool

：扩展标准veRL训练器支持多轮Agent轨迹，但轨迹控制仍在训练器内部

rLLM

：基于veRL的深度修改分支，Agent循环、环境管理和轨迹编排全在单个驱动器进程中，没有独立的轨迹服务

GEM

：环境执行留在训练进程内，通过ThreadPoolExecutor提供并行性，环境和轨迹生命周期完全嵌入训练栈

这些框架的共同特点是轨迹逻辑以进程内库的形式实现在训练循环中，导致换训练后端时需要重新实现或移植整个轨迹栈。

总结与展望

ProRL Agent的核心贡献可以总结为三点：

识别了现有Agent RL框架的关键瓶颈

：多轮Agent轨迹生成通常与RL训练栈紧密耦合，尽管轨迹和训练有截然不同的资源和执行特性。ProRL Agent通过统一的HTTP接口将完整轨迹生命周期从训练器中解耦。

实用的工程设计

：token-in/token-out轨迹通信消除重新分词漂移、可扩展沙箱环境支持异构工具和任务、无root部署适配共享HPC集群。

全面的实验验证

：在软件工程（SWE-Bench Verified）、STEM、数学和编程任务上进行了端到端RL训练，多规模模型均获得显著提升。

ProRL Agent已开源，作为NVIDIA NeMo Gym的一部分发布。研究者和工程师可以独立迭代Agent任务环境和RL训练算法，不用在两套不同的工程需求之间来回妥协。

资源链接

📄 论文链接

https://arxiv.org/abs/2603.18815

💻 代码仓库

https://github.com/NVIDIA-NeMo/ProRL-Agent-Server

🌐 NeMo Gym

https://github.com/NVIDIA-NeMo/Gym

⭐️关注我，实时跟进AI最新进展⭐️