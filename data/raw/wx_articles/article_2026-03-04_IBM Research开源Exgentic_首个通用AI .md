# IBM Research开源Exgentic：首个通用AI Agent评测榜单

- **来源**: Hyman的杂货铺
- **链接**: https://mp.weixin.qq.com/s?src=11&timestamp=1785309751&ver=6871&signature=w8HCAiG9aIED*s968nBKVSQ4xgfCLeQ-WcVHaDdWiEr-7KrHoXQcw*olV-qCzOKgwfCj*nVzhyony8kjemWXn4rST8IlUtAxU5bzLWBZVJxCIwHeETJqKsjbLcoTc3cA&new=1
- **发布时间**: 2026-03-04 07:30
- **字数**: 3579

---

一句话讲清楚👉🏻 IBM Research发布Exgentic框架和统一协议，首次实现通用AI Agent在跨领域环境下的公平评测，实验耗资$22K揭示：模型质量决定性能上限，通用Agent已媲美专用系统。

研究背景：Agent评测的三大痛点

AI Agent正在从实验室走向实际应用——从软件工程任务到网页导航，从客服对话到日常数字任务处理。然而，

通用Agent的评测一直是个难题

。

痛点一：Benchmark孤岛

现有的Agent评测基准（如SWE-Bench、τ²-Bench、AppWorld等）各自独立，使用不同的通信协议。一个Agent要想在多个Benchmark上评测，就需要为每个Benchmark单独适配，工作量巨大。

痛点二：专用Agent难比较

很多Benchmark隐含了特定领域的知识和语义，Agent需要"知道"Benchmark的特殊规则才能表现好。这就导致评测的是"专用Agent"而非"通用Agent"的能力。

痛点三：评测成本高昂

整合多个Benchmark的尝试（如BrowserGym、Harbor）虽然向前迈了一步，但它们强制使用单一协议（网页或CLI），实际上评测的是一个"削弱版"的Agent。

核心贡献：统一协议+评测框架+开放榜单

IBM Research团队在这篇论文中做出了三个核心贡献：

1. 统一协议（Unified Protocol）

统一协议是一个Agent-Benchmark中介层，作为"窄腰"连接任意Agent和任意Benchmark。它定义了三个核心字段：

Task（任务）

：Agent应该做什么的文字描述

Context（上下文）

：完成任务所需的背景信息

Actions（动作）

：环境提供的可执行操作集合

图：Agent评测的三种模式演进。(A) 每个Benchmark需要定制Agent适配；(B) 多个Benchmark通过单一协议整合；(C) Exgentic的统一协议适配任意Agent和Benchmark。

2. Exgentic评测框架

基于统一协议，团队开发了Exgentic评测框架，支持：

模块化洞察

：比较架构、分析语言模型影响、优化Agent-模型配对

大规模并行评测

：支持缓存和并行执行

标准化输出

：统一的轨迹记录、成本报告、结果格式

图：Exgentic架构。统一协议定义Agent与Benchmark之间的接口，Orchestrator协调双方通信，Adaptor完成协议转换。

3. Open General Agent Leaderboard

团队发布了首个开放通用Agent榜单，评测了：

5种Agent架构

：ReAct、ReAct Short、Smolagent、OpenAI Solo、Claude Code

3种前沿模型

：GPT 5.2、Claude Opus 4.5、Gemini 3 Pro

6个Benchmark环境

：SWE-Bench Verified、BrowseComp+、τ²-Bench（3个子域）、AppWorld

图：Open General Agent Leaderboard是首个在多样化环境中一致测试通用Agent架构关键能力的基准。

总评测成本：

$22,000

实验发现：四大核心洞察

发现一：通用Agent已媲美专用系统

通用Agent无需领域特定调优即可跨环境泛化

，在大多数Benchmark上达到或超过专用Agent的性能。

Benchmark

最佳配置

得分

专用Agent最高分

SWE-Bench Verified

OpenAI Solo + Claude Opus 4.5

0.81

0.79

BrowseComp+

Smolagent + Claude Opus 4.5

0.61

0.80

τ²-Bench-Airline

OpenAI Solo + Claude Opus 4.5

0.74

0.73

τ²-Bench-Retail

OpenAI Solo + Claude Opus 4.5

0.85

0.86

τ²-Bench-Telecom

OpenAI Solo + Gemini 3

0.89

0.98

AppWorld

Smolagent + Claude Opus 4.5

0.70

0.73

表：各Benchmark最佳配置与专用Agent对比。通用Agent在多数场景接近专用水平。

发现二：模型质量是性能决定因素

通过方差分解分析，

模型选择贡献了28.2%的性能方差

，而Agent架构仅贡献0.6%。模型质量的影响是架构选择的

85倍

。

模型

平均成功率

稳定性(STD)

Claude Opus 4.5

0.66

0.06

Gemini 3

0.60

0.09

GPT 5.2

0.40

0.07

表：三大模型性能对比。Claude Opus 4.5在性能和稳定性上均领先。

发现三：没有万能Agent

没有任何单一Agent在所有领域都占优

：

OpenAI Solo

 在4个Benchmark上获胜，擅长结构化API交互

Smolagent

 在1个Benchmark上获胜，擅长网页导航和多应用环境

Agent排名因模型而异：在Claude上OpenAI Solo领先(0.73)，在GPT 5.2上ReAct Short领先(0.46)。

发现四：成本-效率权衡显著

图：成本-性能权衡。Pareto前沿（红色虚线）显示最优权衡点：某些配置提供最佳成本效率，而其他配置以更高成本实现最高性能。

配置

得分

成本/任务

效率

ReAct + GPT 5.2

0.41

$0.17

2.41

ReAct + Gemini 3

0.62

$0.66

0.93

Claude Code + Claude Opus 4.5

0.67

$8.03

0.08

表：成本效率排行。GPT 5.2配置效率最高，Claude配置性能最佳但成本高。

达到顶级性能需要付出30倍于最高效率配置的成本

，这是性能与成本之间的根本权衡。

技术细节：统一协议如何工作

协议适配方法论

适配Benchmark

：

检查参考Agent实现，了解其如何与环境交互

构建任务描述，只包含必要信息

省略实现特定细节和冗余信号

适配Agent

：

将任务和上下文映射到Agent的API

将动作映射为Python函数、OpenAI工具或MCP工具

处理特殊动作（如用户交互、任务提交）

关键设计原则

协议保持

：Agent和Banchmark各自使用原生协议运行

进程隔离

：各组件独立进程，互不影响

行为保真

：框架不改变Agent或Benchmark的原始行为

局限性与未来方向

当前局限

文本交互限制

：目前仅支持文本交互，不支持视觉或网页交互

协议覆盖

：统一协议基于现有系统API设计，可能需要扩展以支持新协议

环境隔离

：每个任务独立环境，无法评测跨任务学习能力

未来方向

多模态支持

：扩展到视觉、网页交互

成本优化

：在评测中纳入计算和工具使用成本

主动评测

：测试Agent识别潜在问题的能力

社区扩展

：邀请研究者贡献新Benchmark和评测协议

行业影响

对Agent开发者

统一评测标准

：无需为每个Benchmark单独适配

模块化洞察

：清晰了解架构组件的贡献

成本优化

：在性能和成本间做出明智选择

对企业用户

科学选型

：基于标准化评测选择Agent方案

风险控制

：了解Agent的真实能力和局限

投资决策

：量化评估AI项目的投入产出

对研究领域

公平比较

：首次实现跨Agent、跨Benchmark的公平比较

研究基础

：为通用Agent研究提供基础设施

开放生态

：开源框架和榜单推动社区发展

资源链接

📄 论文链接：https://arxiv.org/abs/2602.22953

🌐 项目主页：https://www.exgentic.ai/

💻 GitHub：https://github.com/Exgentic/exgentic

🏆 榜单数据：https://www.exgentic.ai/results.csv

⭐️关注我，实时跟进AI最新进展⭐️