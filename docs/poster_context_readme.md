# Poster Context README

## Scope

This document focuses only on the context-related implementation in the current codebase. It is written for speech and poster use, so the first part is a spoken introduction and the second part is a compact poster paragraph. Both are based on the real Context Engine, MemoryManager, DiskMemoryStore, QwenMemoryReasoner, and related retrieval logic.

## English Speech Script

Let me focus on the context module, because this is the part that turns the agent from a one-turn chatbot into a persistent legal study tutor. In this project, context is not just a prompt window or a short conversation buffer. It is a disk-backed memory system managed by MemoryManager and stored under memory/agent_memory. For each user, the system keeps a profile, session states, raw turns, reusable memory items, and memory edges. The memory items are organized into seven layers: profile, system, working, long-term, summary, episodic, and semantic. After every interaction, the agent analyzes the turn, extracts stable profile updates and reusable memory drafts, and writes them back to disk.

When a new question arrives, the context engine does not simply append the latest messages. It retrieves relevant memories with a mixed score that combines lexical overlap, vector similarity, importance decay, freshness, layer priority, historical hit bonus, and graph-based relation bonus. The selected results are then rendered into planning context, summary blocks, and related hits for the reasoning engine. To keep the system efficient, older turns are compressed into summary memories once the session becomes long enough, while recent turns remain intact for immediate reasoning. This design lets the agent remember study goals, weak points, unfinished issues, and recent exam feedback, so later answers can stay personalized, continuous, and computationally manageable.

## 中文对照翻译

我想重点介绍一下 context 模块，因为正是这一部分，把系统从一个一次性聊天机器人，变成了一个能够持续陪伴用户学习的法律学习导师。在这个项目里，context 不是简单的 prompt 窗口，也不是只保留最近几轮消息的对话缓存。它是一个由 MemoryManager 管理、并且真实落盘到 memory/agent_memory 的持久化记忆系统。对于每个用户，系统都会保存用户画像、会话状态、原始对话轮次、可复用记忆条目，以及记忆之间的关系边。记忆条目被划分成七个层次：profile、system、working、long-term、summary、episodic 和 semantic。每次交互结束后，Agent 都会分析这一轮对话，提取稳定的画像更新和可复用的记忆草稿，然后把它们写回磁盘。

当新的问题到来时，context engine 并不是简单地把最近消息拼接起来，而是会按照混合分数去检索相关记忆。这个分数综合了词汇重叠、向量相似度、重要度衰减、新鲜度、层级优先级、历史命中加成，以及基于 memory graph 的关系加成。检索出的结果会被重新组织成 planning context、summary blocks 和 related hits，再交给后续的推理引擎使用。为了保持系统效率，较早的对话在会话长度达到阈值后会被压缩成 summary memories，而最近轮次则仍然完整保留，供即时推理使用。正因为有这一套设计，Agent 才能记住用户的备考目标、薄弱点、未完成的问题以及最近的测试反馈，让后续回答既连续、又个性化，同时还能控制计算成本。

## Poster Short Version

The context module turns the agent from a one-shot legal chatbot into a persistent study assistant. It stores user profiles, raw turns, summary memories, semantic memories, and memory relations on disk, then retrieves the most relevant pieces for each new question. Ranking is hybrid: lexical overlap, vector similarity, importance, freshness, layer priority, hit history, and graph bonuses all contribute. Older conversations are compressed into summary memories while recent turns remain available. As a result, the agent can remember goals, weak points, unfinished follow-up issues, and exam feedback, and use them to produce more stable and personalized legal study support.

## 中文对照翻译

Context 模块把系统从一次性的法律聊天机器人，变成了一个可以持续陪伴学习的智能助教。它会把用户画像、原始对话轮次、摘要记忆、语义记忆以及记忆关系持久化到磁盘中，并在每次新问题到来时检索最相关的部分。排序采用混合机制：词汇重叠、向量相似度、重要度、新鲜度、层级优先级、历史命中情况以及图关系加成都会共同参与。较早的对话会被压缩成摘要记忆，而最近轮次仍然保留可用。这样一来，Agent 就能记住用户目标、薄弱点、未完成追问和测试反馈，从而给出更稳定、更个性化的法律学习支持。