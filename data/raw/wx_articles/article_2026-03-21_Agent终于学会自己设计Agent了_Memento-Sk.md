# Agent终于学会自己设计Agent了：Memento-Skills实现116%性能飞跃

- **来源**: Hyman的杂货铺
- **链接**: https://mp.weixin.qq.com/s?src=11&timestamp=1785309752&ver=6871&signature=w8HCAiG9aIED*s968nBKVSQ4xgfCLeQ-WcVHaDdWiEqjuaaFVFrFyLSde1pmaSFwifd4AIfr0DAmFMe6ILrGQZSJTHAM9oItEJNR09t1szGmuYUWCn2RsLsOmaIqMMtp&new=1
- **发布时间**: 2026-03-21 07:30
- **字数**: 4020

---

一句话讲清楚👉🏻

 Memento-Skills 让 AI Agent 能从失败中学习、自己改技能库，甚至自己设计新的 Agent。在 GAIA 和 HLE 两个基准上分别涨了 26.2% 和 116.2%，全程不动 LLM 参数。

Agent 的"失忆症"

你有个很聪明的助手，能写代码、查资料、做分析。某天它在一个任务上犯了错，你指出了问题。第二天，你给它类似任务——它又犯了同样的错。

这就是当前 LLM Agent 的老毛病：

没有真正的学习能力

。每次交互都是"新的一天"，所有经验教训都会被遗忘。想让 Agent 变聪明？传统做法是重新训练或微调底层 LLM，但成本高、耗时长，还容易灾难性遗忘。

Memento-Team 的研究团队搞了个新东西：

Memento-Skills

。一个让 Agent"自己设计自己"的框架——能在真实任务中学习，自主改进、重写甚至创造新技能。

Read → Execute → Reflect → Write 闭环

Memento-Skills 的核心是个持续学习的闭环机制，思路来自人类的学习过程。

1. Read：智能检索技能

当 Agent 面对新任务时，它不会把所有技能都塞进上下文，而是通过一个

行为可训练的技能路由器

，根据当前的"状态化提示"从技能库中检索最相关的技能。

当技能库变大时，怎么精准找到合适的技能是个问题。Memento-Skills 用了混合检索策略：

BM25 关键词检索，快速匹配描述中的关键词

向量语义检索，理解技能的深层语义

云端技能目录，支持从远程下载验证过的技能

2. Execute：本地沙盒运行

检索到技能后，Agent 会在隔离的本地沙盒环境中执行。这样做有几个好处：技能执行不影响系统环境，可以精确追踪执行结果，失败时能准确定位是哪个技能出了问题。

Agent 可以操作文件、运行脚本、访问网页、调用外部系统，真正和真实世界交互。

3. Reflect：从失败中学习

这是 Memento-Skills 和其他系统不一样的地方。执行失败或质量下降时，系统会记录状态、更新每个技能的"有用程度"评分、尽可能精确定位问题技能。

这个过程类似人类的"复盘"——不是简单重试，而是深入想"为什么失败"和"怎么改进"。

4. Write：技能自主进化

反思之后，系统会做真正的"学习"：优化表现不佳的技能、重构反复失败的技能、当现有技能都不够用时自主设计新技能。

所有改进都会写回技能库，成为 Agent 永久性的能力提升。而且这一切都发生在

外部技能文件

中，完全不需要修改 LLM 参数。

Memento-Skills 整体框架。Read → Execute → Reflect → Write 闭环，实现持续学习和自我进化。

技术架构

Memento-Skills 不只是概念，而是一个完全自主研发、可部署的 Agent 框架。

技能作为一等公民

在 Memento-Skills 中，

技能（Skill）是核心抽象单元

，有四个特性：可检索（通过语义相似度和关键词混合检索）、可执行（在沙盒环境中安全运行）、可持久化（存储为结构化 Markdown 文件）、可演化（可以被修改、重写或重新创建）。

技能存为 Markdown 文件，包含技能描述、执行代码或提示词模板、使用示例、效用评分和历史记录。这种设计让技能成为"活"的能力单元，不是静态的工具函数。

Memento-Skills 技术架构。技能作为一等公民被组织和管理。

自我进化架构

Memento-Skills 的自我进化能力来自几个组件：

技能路由器

：根据当前状态选择最相关的技能，使用行为可训练的策略而非简单相似度匹配，支持从本地库和云端目录检索。

技能验证器

：下载新技能时进行静态审查，在沙盒环境中执行验证测试，确保安全性和有效性。

技能创造器

：没有合适技能时自主创建，基于任务需求设计提示词和代码，自动生成测试用例验证。

自我进化组件。路由器、验证器和创造器协同工作，实现技能库持续优化。

检索管道

技能库规模扩大时，检索准确性是个挑战。Memento-Skills 实现了三层检索管道：

检索管道设计。结合关键词检索、向量检索和云端目录。

第一层是本地 BM25 检索，基于关键词快速匹配；第二层是向量语义检索，用嵌入向量理解深层语义；第三层是云端技能目录，访问社区贡献的验证技能。

技能路由器工作原理。根据当前状态和任务需求，智能选择最相关的技能。

内置技能

Memento-Skills 初始提供 9 个内置技能：

技能

功能

filesystem

文件读写、搜索、目录操作

web-search

基于 Tavily 的网页搜索

image-analysis

图像理解、OCR

pdf

PDF 读取、表单填充、OCR

docx

Word 文档创建编辑

xlsx

电子表格处理

pptx

PowerPoint 创建编辑

skill-creator

新技能创建优化评估

uv-pip-install

Python 依赖安装

但这些只是起点。系统目标是维护一个持续增长的技能库——能不断被检索、修复和改进。

实验结果

研究团队在两个基准上验证了效果。

GAIA 基准

GAIA（General AI Assistants）评估通用 AI 助手能力，包含需要多步推理、工具使用和信息整合的复杂任务。

GAIA 基准表现。学习轮次增加，任务完成率持续提升。

多轮反思学习后，Memento-Skills 在 GAIA 上实现了 

26.2% 相对提升

。这种提升来自技能库优化，不是模型参数更新。

HLE 基准

HLE（Humanity's Last Exam）更有挑战性，包含需要深度推理和专业知识的难题。

HLE 基准训练和测试表现。性能提升泛化到未见过的测试任务。

这个基准上取得了 

116.2% 相对提升

。通过技能演化获得的能力有良好泛化性。

技能库演化

自我演化结果。左图：性能随学习轮次提升；右图：技能库从少量原子技能成长为丰富集群。

能看到两个信号：性能持续改进，技能库有机增长。

技能效用分布。系统持续调整技能的效用评分。

和 OpenClaw 对比

Memento-Skills 和 OpenClaw 有很多共同基因，但关注点不同。

两者都把技能作为能力单元，都是可部署可工程化的系统，都有工具使用和本地执行、持久化记忆。

但差异也明显：

维度

Memento-Skills

OpenClaw

焦点

让 Agent 学习

让助手部署

演化

失败触发反思闭环，修改或创建新技能

能力增长依赖外部插件和人工集成

路由

检索路由是核心问题

优化真实世界集成

创建

无合适技能时自主创建

缺失技能由人工提供

一句话：

OpenClaw 是让助手跑起来，Memento-Skills 是让 Agent 学会

。

部署能力

Memento-Skills 不是纸上谈兵，有完整部署能力：

桌面 GUI 界面。会话列表、聊天界面、斜杠命令。

部署方式

命令

说明

命令行

memento agent

交互或单消息模式

桌面 GUI

memento-gui

会话列表、聊天界面

飞书桥接

memento feishu

WebSocket IM 桥接

技能验证

memento verify

下载、审查、验证

本地沙盒

uv

隔离执行、依赖安装

适合基准测试 Agent，也适合个人生产力、长期运行的本地助手。

中文 LLM 支持

Memento-Skills 对中文 LLM 生态做了特别优化，配置后可接入 Kimi/Moonshot、MiniMax、GLM/智谱等。只需在配置文件设置 

base_url

 和模型名称。

技术栈

层次

技术

界面

Flet、Typer、Rich

LLM

litellm

检索

BM25、嵌入向量、sqlite-vec

执行

本地子进程 + uv 沙盒

存储

SQLite、SQLAlchemy

配置

JSON + Pydantic

测试

pytest、异步验证

快速上手

git 

clone

 https://github.com/Memento-Teams/Memento-Skills.git

cd

 Memento-Skills

python -m venv .venv

source

 .venv/bin/activate

pip install -e .

memento doctor

memento agent

首次启动会在 

~/memento_s/config.json

 创建配置文件，填入模型配置即可。

意义

Memento-Skills 证明了件事：Agent 可以完全通过外部技能演化来提升能力，不需要动底层 LLM 参数。系统不仅能改进现有技能，还能自主创造新技能应对未知任务。

而且这不是概念，从 CLI 到 GUI，从沙盒执行到 IM 桥接，都能真实部署。两个权威基准上的数据也说明了持续学习的价值。

资源链接

📄 论文链接：https://arxiv.org/abs/2603.18743

💻 GitHub：https://github.com/Memento-Teams/Memento-Skills

⭐️关注我，实时跟进AI最新进展⭐️