# NVIDIA开源：30B MoE模型只激活3B参数，却拿下IMO/IOI/ICPC三项世界金牌，参数量仅为DeepSeek的1/20

- **来源**: Hyman的杂货铺
- **链接**: https://mp.weixin.qq.com/s?src=11&timestamp=1785309750&ver=6871&signature=w8HCAiG9aIED*s968nBKVSQ4xgfCLeQ-WcVHaDdWiErqqpXa6vDIkLL8Td8V7inLtBHNnPnVxn4Lr2svPvG8TAxHD5kKtwQ9R3ZOmgFb5pTBWdvIL674WgJm4ZIjjiRX&new=1
- **发布时间**: 2026-03-22 07:30
- **字数**: 9105

---

一句话讲清楚👉🏻

 NVIDIA发布开源模型Nemotron-Cascade 2，一个30B MoE架构但只有3B激活参数的"小模型"，通过Cascade RL和Multi-Domain On-Policy Distillation两项核心技术，在2025年IMO数学奥林匹克、IOI信息学奥赛和ICPC程序设计世界总决赛中全部获得金牌级成绩，比DeepSeek-V3.2-Speciale-671B-A37B少用20倍参数，却实现了相当的推理能力。

一、背景：当"小模型"遇上"大智慧"

在当前大模型军备竞赛中，一个明显的趋势是：模型参数量不断膨胀，从几十亿到几千亿甚至万亿参数。但问题是，更大的模型就意味着更高的推理成本、更多的GPU显存需求、更长的推理延迟。

NVIDIA提出了一个关键概念——

智能密度（Intelligence Density）

：用尽可能少的参数实现尽可能强的推理能力。Nemotron-Cascade 2正是这一理念的典型代表。

具体来说，Nemotron-Cascade 2采用了Mixture-of-Experts（MoE）架构，总参数量30B，但每次推理只激活3B参数。这意味着在保持模型容量的同时，推理效率大幅提升。

更令人瞩目的是，这个"小模型"在多项顶级竞赛中取得了令人惊讶的成绩：

2025年IMO（国际数学奥林匹克）

：35/42分，金牌

2025年IOI（国际信息学奥林匹克）

：439.28/600分，金牌

2025年ICPC世界总决赛

：10/12题，金牌

这些成绩此前被认为只有前沿闭源模型（如Gemini Deep Think）或超大规模开源模型（如DeepSeek-V3.2-Speciale-671B-A37B）才能达到。Nemotron-Cascade 2用20倍少的参数实现了相当的性能，证明了"小模型"同样可以拥有"大智慧"。

二、Nemotron-Cascade 2：模型概览与性能亮点

2.1 模型基本信息

Nemotron-Cascade 2是NVIDIA Nemotron系列的最新成员，建立在预训练模型Nemotron-Nano-V3之上。以下是关键参数：

架构

：Mixture-of-Experts (MoE)

总参数量

：30B

激活参数量

：3B（每次推理）

基础模型

：Nemotron-3-Nano-30B-A3B-Base

上下文窗口

：支持1M tokens

推理模式

：支持thinking模式和instruct模式

Nemotron-Cascade 2在数学、代码推理、对齐、指令遵循等多个基准测试中的表现

2.2 核心性能数据

根据论文中的主要结果表，Nemotron-Cascade-2-30B-A3B在多个基准测试中表现优异：

数学推理

：

AIME 2025：92.4（使用TIR后达98.6）

HMMT Feb25：94.6

IMO ProofBench：72.9

代码推理

：

LiveCodeBench v6：87.2（使用TIR后达88.4）

LiveCodeBenchPro 25Q2 (Medium)：27.6（使用TIR后达36.8）

对齐与指令遵循

：

ArenaHard v2 (Avg.)：83.5

IFBench (prompt)：82.9

长上下文理解

：

NIAH@1M (RULER Subset)：99.0

值得注意的是，Nemotron-Cascade 2在知识密集型任务（如MMLU-Pro、GPQA-Diamond）和某些Agent任务上表现相对较弱，这体现了模型在推理能力和其他能力之间的战略性权衡。

三、核心技术：Cascade RL（级联强化学习）

Cascade RL是Nemotron-Cascade 2最核心的训练框架，它将强化学习按照领域顺序依次进行，而非同时训练所有任务。

3.1 为什么需要Cascade RL？

传统的多领域RL训练通常采用"混合训练"的方式，即把所有领域的数据混合在一起同时训练。这种方法存在几个问题：

灾难性遗忘

：不同领域的RL信号可能相互冲突，导致某些领域的能力退化

超参数冲突

：不同任务可能需要不同的学习率、温度、惩罚结构

计算效率低

：混合训练时，不同任务的响应长度和验证时间差异很大，造成GPU利用率低下

Cascade RL通过"顺序式、领域特定"的训练方式解决了这些问题。

3.2 Cascade RL的训练顺序

Nemotron-Cascade 2的Cascade RL包含以下阶段：

IF-RL（指令遵循强化学习）

：建立基础的指令遵循能力

Multi-domain RL（多领域强化学习）

：增强工具调用、STEM推理、格式遵循能力

MOPD（多领域在线策略蒸馏）

：统一各领域的专业能力

RLHF（人类反馈强化学习）

：对齐人类偏好

Long-context RL（长上下文强化学习）

：增强长文本推理能力

Code RL（代码强化学习）

：提升竞赛编程能力

SWE RL（软件工程强化学习）

：掌握Agent软件工程交互

从基础模型到Nemotron-Cascade 2的完整训练流程

3.3 Cascade RL的三大优势

优势一：抵抗灾难性遗忘

领域特定的RL阶段对早期领域获得的性能几乎不会退化，甚至可能有所提升。这是因为每个阶段专注于特定领域，避免了不同领域信号的冲突。

优势二：超参数定制化

每个RL阶段可以针对特定领域调整学习率、温度、惩罚结构等超参数，实现最优的学习动态。例如，复杂的多步数学证明需要完全不同的学习率和温度设置，与严格的JSON格式化任务截然不同。

优势三：计算效率提升

同一领域内的任务通常具有相似的响应长度和验证时间，这减少了因等待较长生成或较慢环境验证而造成的训练效率损失。

3.4 确定训练顺序的原则

论文中提到了一个实用原则：

减轻领域间干扰

。

具体来说，Cascade RL的排序主要由"减轻灾难性遗忘"的需求驱动。核心设计原则是识别一个能够最小化领域间负面干扰的排序，同时充分优化最高优先级的领域。

例如，IF-RL被放在最前面的原因是：IF-RL可能会对人类对齐能力产生负面影响，而后续的RLHF对指令遵循分数的影响可以忽略不计。通过优先处理指令遵循，可以最大化指令遵循性能，然后利用后续阶段恢复和优化人类偏好对齐。

四、核心技术：Multi-Domain On-Policy Distillation（MOPD）

MOPD是Nemotron-Cascade 2的另一项关键创新，它在Cascade RL过程中引入了知识蒸馏机制。

4.1 为什么需要MOPD？

虽然精心设计的Cascade RL大大减少了灾难性遗忘，但并不能完全消除能力漂移。随着训练环境数量的增加，不同基准测试类别在整个训练过程中会出现明显的波动。

例如，某些RLVR训练可能会降低模型熵并缩短推理链，从而对数学推理性能产生负面影响；而RLHF优化可能会部分牺牲指令遵循行为。

MOPD作为Cascade RL的补充训练阶段，用于重新平衡各领域能力。

4.2 MOPD的工作原理

MOPD的核心思想是：在Cascade RL的每个阶段，从该领域表现最好的中间检查点中选择"教师模型"，然后通过知识蒸馏将这些教师模型的能力传递给学生模型。

具体来说，MOPD包含以下步骤：

选择教师模型

：对于每个训练样本，选择与该样本领域对应的最佳中间教师模型

计算蒸馏优势

：使用反向KL散度计算token级别的蒸馏优势

重要性加权

：使用截断重要性加权来处理训练-推理不匹配问题

优化目标

：最小化学生模型与教师模型之间的分布差异

MOPD的优势在于：

教师检查点直接从Cascade RL流程中选择，无需引入外部模型家族

教师和学生共享相同的tokenizer和词汇表，减少分布偏移

提供密集的token级别训练优势，比稀疏的结果奖励更高效

4.3 MOPD的训练效率优势

论文中的实验表明，MOPD比传统的GRPO算法具有显著的训练效率优势：

在AIME25基准测试上：

GRPO从89.9提升到91.0需要25步

MOPD在30步内达到92.0，恢复到教师模型水平

在ArenaHard v2上：

MOPD经过52步训练，Hard Prompt从71.5提升到85.5

RLHF训练需要160步才能达到80.7

这种效率优势源于MOPD提供了密集的token级别蒸馏优势，而GRPO依赖于在所有生成token间共享的稀疏序列级别结果奖励。

五、SFT数据工程：精心策划的训练数据

Nemotron-Cascade 2的成功很大程度上归功于其精心策划的SFT数据集。

5.1 数据集规模与组成

SFT数据集涵盖了广泛的领域，包括数学、编码、科学、工具使用、Agent任务、软件工程、多轮对话、知识密集型问答、创意写作、角色扮演、安全性和指令遵循等。

关键数据组成：

数学

：1.8M工具调用样本 + 2.6M非工具样本

代码推理

：1.9M Python推理轨迹 + 1.0M C++14推理轨迹 + 1.3M Python工具调用推理轨迹

数学证明

：816K数学自然语言证明样本

科学

：2.7M科学SFT样本

长上下文

：234K长上下文SFT样本

通用对话

：4.9M推理开启 + 372K推理关闭样本

软件工程

：125K Agent样本 + 389K Agentless样本

终端Agent

：490K样本

5.2 数据质量保证

NVIDIA采用了多种策略确保数据质量：

去重

：对编码提示应用严格的去重，使用样本I/O指纹和n-gram文本分析，去除约24.2%的重复编码提示

正确性过滤

：对于有可验证测试用例的编码提示，只保留生成正确代码的推理轨迹

多教师模型

：使用GPT-OSS-120B、DeepSeek-V3.2、DeepSeek-V3.2-Speciale等多个强教师模型生成响应

人工验证

：论文的共同作者之一是2015年IMO金牌得主，他审查并评分了模型生成的IMO解决方案

5.3 训练配置

SFT训练采用以下关键配置：

所有SFT样本被打包成最多256K token的序列

模型在单个阶段训练约1.5个epoch达到最佳性能

使用AdamW优化器

六、GRPO算法与训练细节

6.1 GRPO算法

Nemotron-Cascade 2在整个Cascade RL过程中使用Group Relative Policy Optimization（GRPO）算法，采用严格的on-policy训练方式。

GRPO的关键特点：

每次迭代从当前策略生成一组rollout，然后执行单次梯度更新

完全移除KL散度项，简化为标准的REINFORCE目标

使用组归一化奖励和token级别损失

GRPO目标函数为：

其中 $\hat{A}

{i,t} = \frac{r_i - \text{mean}({r_i}

{i=1}^{G})}{\text{std}({r_i}_{i=1}^{G})}

，

r_i

是

响

应

o_i$ 的奖励。

6.2 关键训练配置

批量大小

：128

每提示采样数

：16

温度

：1.0

top-p

：1.0

学习率

：3e-6（IF-RL），2e-6（MOPD）

熵损失系数

：0

KL损失系数

：0

七、IMO/IOI/ICPC竞赛表现详解

7.1 IMO 2025成绩

Nemotron-Cascade 2在2025年IMO中获得35/42分，达到金牌标准：

题目

P1

P2

P3

P4

P5

P6

总分

奖牌

得分

7

7†

7

7

7

0

35/42

金牌

† 对于P2，由于模型使用了大量的分析几何方法，人工专家难以验证所有中间推导步骤，因此使用了LLM评分器和ProofBench的参考解决方案和评分标准。

这是第二个在2025年IMO中达到金牌水平的开源权重LLM，前一个是DeepSeek-V3.2-Speciale-671B-A37B。

7.2 IOI 2025成绩

Nemotron-Cascade 2在2025年IOI中获得439.28/600分，同样达到金牌标准：

题目

P1

P2

P3

P4

P5

P6

总分

奖牌

得分

39

88.53

100

100

28.75

83

439.28/600

金牌

IOI的解决方案通过OnlineJudge使用官方测试用例进行验证。

7.3 ICPC World Finals 2025成绩

Nemotron-Cascade 2在2025年ICPC世界总决赛中解决了12道题目中的10道，获得金牌：

题目

A

B

C

D

E

F

G

H

I

J

K

L

总分

奖牌

结果

+

-

+

+

+

+

-

+

+

+

+

+

10/12

金牌

7.4 Codeforces ELO评分

论文还对模型进行了Codeforces ELO评分分析：

Thinking模式：2320分

Thinking + Tool Calling (Python)：2345分

这个评分表明Nemotron-Cascade 2的编程能力达到了顶级人类竞赛程序员的水平。

八、技术架构与推理特性

8.1 Chat Template设计

Nemotron-Cascade 2采用ChatML模板，支持两种推理模式：

Thinking模式

：使用 

<think>\n

 开始推理过程

Instruct（非推理）模式

：使用 

<think></think>

 跳过推理

对于工具调用任务，可用工具在系统提示中的 

<think>

 和 

</think>

 标签内指定，模型被指示在 `` 和 

</think>

 标签内调用工具。

8.2 Tool-Integrated Reasoning（TIR）

TIR是Nemotron-Cascade 2的一个重要特性，允许模型在推理过程中使用Python执行器来验证中间步骤。这在复杂算术或逻辑证明等手动推理容易出错的领域显著提升了性能。

例如，在AIME 2025上：

Thinking模式：92.4

Thinking + TIR：98.6（提升6.2分）

在LiveCodeBench v6上：

Thinking模式：87.2

Thinking + TIR：88.4（提升1.2分）

8.3 推荐采样参数

温度：1.0

top-p：0.95

九、与同类模型的对比

Nemotron-Cascade 2在多个基准测试中超越了同类模型：

9.1 对比Qwen3.5-35B-A3B（2026-02-24发布）

在以下基准测试中，Nemotron-Cascade 2表现更优：

AIME 2025：92.4 vs 91.9

IOI 2025：439.28 vs 348.6

LiveCodeBench v6：87.2 vs 74.6

ArenaHard v2：83.5 vs 65.4

IFBench：82.9 vs 70.2

但在知识密集型和Agent任务上，Qwen3.5-35B-A3B表现更优：

MMLU-Pro：79.8 vs 85.3

GPQA-Diamond：76.1 vs 84.2

SWE Verified：50.2 vs 69.2

9.2 对比Nemotron-3-Super-120B-A12B（2026-03-11发布）

Nemotron-Cascade 2在多个基准测试中超越了更大的Nemotron-3-Super-120B-A12B：

AIME 2025：92.4 vs 90.2

HMMT Feb25：94.6 vs 93.7

LiveCodeBench v6：87.2 vs 78.7

ArenaHard v2：83.5 vs 73.9

这再次证明了Nemotron-Cascade 2的高智能密度——用更少的参数实现更强的性能。

十、开源与应用

10.1 开源资源

NVIDIA完全开源了Nemotron-Cascade 2的模型权重、训练数据和方法细节：

模型检查点

：

Nemotron-Cascade-2-30B-A3B（

https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B

）：支持thinking和instruct两种模式

训练数据

：

Nemotron-Cascade-2-SFT-Data（

https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data

）：SFT阶段的训练数据

Nemotron-Cascade-2-RL-Data（

https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-RL-data

）：RL阶段的训练数据

10.2 应用场景

Nemotron-Cascade 2特别适合以下应用场景：

数学推理

：竞赛级数学问题求解、数学证明生成

代码生成

：竞赛编程、代码调试、算法设计

Agent系统

：工具调用、多步骤任务执行

教育辅助

：数学题目讲解、编程教学

10.3 使用示例

使用Hugging Face Transformers加载模型：

from

 transformers 

import

 AutoTokenizer, AutoModelForCausalLM

model_name = 

'nvidia/Nemotron-Cascade-2-30B-A3B'

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

messages = [

    {

"role"

: 

"system"

, 

"content"

: 

"You are a helpful assistant."

},

    {

"role"

: 

"user"

, 

"content"

: 

"Solve the equation x^2 + 2x + 1 = 0"

}

]

prompt = tokenizer.apply_chat_template(messages, tokenize=

False

, add_generation_prompt=

True

, enable_thinking=

True

)

inputs = tokenizer(prompt, return_tensors=

"pt"

)

outputs = model.generate(**inputs, max_new_tokens=

1024

, temperature=

1.0

, top_p=

0.95

)

print

(tokenizer.decode(outputs[

0

], skip_special_tokens=

True

))

十一、总结与展望

Nemotron-Cascade 2代表了LLM后训练技术的重要进步，通过Cascade RL和Multi-Domain On-Policy Distillation两项核心技术，实现了：

高智能密度

：30B MoE模型、3B激活参数，达到前沿模型水平

金牌级推理

：在IMO、IOI、ICPC世界总决赛中获得金牌

全面的能力覆盖

：数学、代码、对齐、指令遵循、长上下文、Agent

完全开源

：模型权重、训练数据、方法细节全部公开

这项工作证明了一个重要观点：通过精心设计的后训练流程，"小模型"同样可以实现"大智慧"。未来的研究方向可能包括：

更强的知识密集型能力

：通过更高质量的预训练数据提升知识任务表现

更广泛的Agent能力

：扩展Agent RL训练覆盖更多真实世界任务

多模态推理

：将Cascade RL扩展到视觉、语音等多模态领域

更高效的训练

：进一步优化MOPD的训练效率

资源链接

📄 论文链接：https://arxiv.org/abs/2603.19220

🤗 模型下载：https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B

📊 SFT数据集：https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data

📊 RL数据集：https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-RL-data

🔬 NVIDIA研究页面：https://research.nvidia.com/labs/nemotron/nemotron-cascade-2/

🏠 Nemotron系列：https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/