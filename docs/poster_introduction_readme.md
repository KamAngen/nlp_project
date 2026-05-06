# Poster Introduction README

## English Introduction

This project implements a unified Chinese legal study agent that is designed to improve three limitations of ordinary LLM tutoring systems: weak long-horizon context retention, shallow task planning, and insufficiently grounded retrieval. Rather than treating legal study as isolated question answering, the repository turns it into a persistent workflow that connects user profiling, context accumulation, targeted practice generation, answer-sheet scoring, and report writing inside one agent architecture.

The first contribution is context modeling. The codebase includes a disk-backed context engine that maintains user profiles, session summaries, working memory, long-term memory, and system memories, then retrieves and organizes them before reasoning. This design allows the agent to preserve stable facts about the learner, track unfinished issues across turns, and reuse prior mistakes or weak points when generating new study actions. In practice, this makes the agent more suitable for iterative exam preparation than a stateless chatbot.

The second contribution is planning-oriented execution. At runtime, the unified agent first analyzes the user turn, prepares study context, and then enters an LLM-centered ReAct loop with access to tools such as profile update, context preparation, legal retrieval, mock-exam generation, answer scoring, and report generation. When the model does not form a reliable tool chain, the system falls back to deterministic routes for exam generation, exam scoring, report requests, and direct legal QA. This hybrid design improves controllability: the agent can still perform multi-step study tasks even when free-form generation alone would be unreliable.

The third contribution is retrieval grounding. The repository builds two retrieval surfaces directly from code: a study knowledge service over question banks, case banks, and common knowledge, and a statute retriever built from parsed Chinese legal documents. The retrieval stack supports sparse matching, dense indices, reranking, and graph-related artifacts, enabling both concept-level study support and citation-oriented legal grounding. Together with the QLoRA fine-tuning pipeline over generated trajectories, these components move the system beyond a generic baseline chatbot toward a personalized, context-aware, planning-capable, and retrieval-grounded legal learning agent.

## 中文对照翻译

本项目实现了一个统一的中文法律学习 Agent，目标是针对普通大模型学习助手的三个核心短板进行改进：长期上下文保持能力弱、任务规划深度不足，以及检索依据不够扎实。项目并不把法律学习视为一次性的问答任务，而是通过同一套 Agent 架构，把用户画像维护、上下文积累、定向出题、答题卡评分和学习报告生成串联成一个持续运行的学习工作流。

第一项改进是上下文建模。代码中实现了一个基于磁盘持久化的上下文引擎，用于维护用户画像、会话摘要、工作记忆、长期记忆和系统记忆，并在推理前完成检索与组织。这样的设计使 Agent 能够保存学习者的稳定信息，跨轮次追踪未完成问题，并在生成后续学习动作时复用历史错题和薄弱点。相较于无状态聊天机器人，这种机制更适合法考这类持续性的备考场景。

第二项改进是面向规划的执行机制。在运行时，统一 Agent 会先分析用户当前输入，准备学习上下文，然后进入以 LLM 为中心的 ReAct 推理循环，可调用的工具包括画像更新、上下文准备、法律检索、模拟测试生成、答卷评分和报告生成等。当模型没有形成可靠的工具链时，系统还会使用确定性的 fallback 路径处理出题、评分、报告和直接法律问答等高频任务。这样的混合设计提升了可控性，使 Agent 即使在自由生成不稳定的情况下，也能完成多步骤学习任务。

第三项改进是检索增强与依据落地。仓库中直接实现了两条检索表面：一条是面向学习任务的知识服务，覆盖题库、案例库和通用知识；另一条是基于解析后的中国法律文档构建的法规检索器。整个检索栈支持稀疏匹配、稠密索引、重排序以及图相关工件，从而同时提供概念层面的学习支持和法条层面的依据支撑。再结合基于生成轨迹的 QLoRA 微调流程，这些组件共同将系统从一个通用聊天基线，推进为一个个性化、上下文感知、具备规划能力且检索有据可依的法律学习 Agent。