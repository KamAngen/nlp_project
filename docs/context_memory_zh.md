# 上下文与记忆模块完整说明

这份文档只回答一个问题：现在这套 context memory 到底是怎么工作的。

它不是改动日志，也不是某一步补丁说明，而是当前实现的整体设计、落盘结构、写入时机、检索流程和观察方法。

## 1. 模块目标

当前上下文模块同时负责四类工作：

1. 保留完整原始对话轨迹，包括用户消息、助手回复、工具轨迹和 reasoning trace。
2. 把模型理解后认为真正有长期价值的信息，整理成结构化记忆。
3. 在会话变长时自动压缩旧历史，但不删除原始 turns。
4. 给 planning、tool 和 unified agent 暴露一个统一上下文入口 prepare_context。

核心原则有两个：

1. 记忆写入以模型理解为主，而不是依赖固定句式匹配。
2. 磁盘结构尽量收口，保证你能直接打开 JSON/JSONL 观察状态。

## 2. 逻辑上有哪些记忆层

逻辑层仍然保留 7 层，但磁盘上不会拆成 7 个文件夹。

- profile
  用户画像的结构化稳定字段，例如姓名、备考目标、薄弱点、强项、回答偏好、目标分数等。

- system
  代理级长期约束。包括启动时加载的系统种子记忆，也包括后续由 LLM 判断确实应上升为系统规则的内容。

- working
  当前会话的短时工作记忆，例如最近几轮摘要、当前激活试卷、待确认事项。

- long_term
  用户跨会话稳定事实，例如持续复习方向、固定偏好、稳定薄弱点。

- summary
  长会话压缩后得到的摘要节点，用来减少上下文长度。

- episodic
  单轮事件记忆，描述这一轮发生了什么。

- semantic
  解释型记忆，描述这一轮沉淀了什么结论、经验、反馈或理解。

你可以把它理解成：

1. profile 是结构化档案。
2. system 是全局规则。
3. long_term / semantic / episodic / summary 是模型理解后沉淀的记忆节点。
4. working 是为了当前回合临时拼出来的工作上下文。

## 3. 磁盘上到底有哪些文件

当前稳态目录结构如下：

```text
memory/agent_memory/
  system/
    system_memories.jsonl
  users/
    <user_id>/
      profile.json
      memories.jsonl
      memory_edges.jsonl
      sessions/
        <session_id>.json
```

每个文件的职责是固定的。

### 3.1 system/system_memories.jsonl

这里放所有 system 层记忆。

包括两类内容：

1. 启动时从 data/legal_study_agent/system_seed_memories.json 加载的系统种子记忆。
2. 后续 record_turn 时由 reasoner 判断应上升为系统规则的动态 system memories。

每一行是一个 MemoryItem，常见字段包括：

1. id
2. category
3. text
4. importance
5. tags
6. hit_count
7. last_accessed_at

### 3.2 users/<user_id>/profile.json

这是用户画像的主文件。

它保存的不是自由文本，而是结构化字段，例如：

1. name
2. study_goals
3. weak_points
4. strong_points
5. preferences
6. attributes
7. notes
8. updated_at

profile.json 是“当前用户长期档案”的权威来源。

### 3.3 users/<user_id>/memories.jsonl

这里放用户自己的结构化记忆节点。

通常包含：

1. long_term 记忆
2. semantic 记忆
3. episodic 记忆
4. summary 记忆
5. 由考试评分写回的反馈记忆

它不包含 profile.json 本身，但 prepare_context 检索时会把 profile.json 临时渲染成 profile 层候选记忆一起参与排序。

### 3.4 users/<user_id>/memory_edges.jsonl

这里放用户记忆图的派生边索引。

它不是主存储，也不是事实权威来源，而是为了提升召回质量额外生成的结构化关系文件。

每一行是一个 MemoryEdge，常见字段包括：

1. id
2. source_id
3. target_id
4. relation
5. weight
6. payload
7. created_at

当前这份图索引是从 users/<user_id>/memories.jsonl 派生出来的，典型建边信号包括：

1. shared_reference
2. tag_overlap
3. keyword_overlap
4. same_session
5. same_category

也就是说，这里保存的是“哪些记忆节点彼此关联、关联强度多大”，而不是替代 memories.jsonl 本身。

### 3.5 users/<user_id>/sessions/<session_id>.json

这是单会话文件。

它同时保存：

1. turns
2. summary
3. active_exam_session_id
4. last_report_path
5. turn_count
6. compression_cursor
7. compression_count
8. metadata
9. updated_at

也就是说，旧版那种 state.json + turns.jsonl 的双文件会话结构已经被收口成一个单文件 session。

如果磁盘上还有旧结构，运行时会自动迁移到新的 sessions/<session_id>.json。

## 4. 为什么你有时看不到 profile.json

这个问题的答案通常不是“没实现”，而是当前磁盘状态还没触发用户目录的创建。

目前有三种常见情况：

1. 你清空过 memory/agent_memory/users 之后，还没有重新创建用户或进入任何用户会话。
2. 你只启动了程序，但还没有执行 create_user、首次聊天、首次按钮动作或任何会写 session/profile 的操作。
3. 你观察的是一个还没被 touch 过的新 user_id。

只要发生以下任一动作，对应文件就会出现：

1. create_user：会创建 users/<user_id>/profile.json。
2. ensure_session 或首次会话交互：会创建 users/<user_id>/sessions/<session_id>.json。
3. update_profile / record_turn / score_exam 等写记忆操作：会创建或更新 memories.jsonl。

如果你只看到 system/system_memories.jsonl，而看不到某个用户的 profile.json，通常说明系统种子记忆已经启动，但这个用户空间还没重新生成。

## 5. 一轮对话是怎么写进记忆的

主入口是 MemoryManager.record_turn。

每次 record_turn 的流程是：

1. 构造 ConversationTurn。
2. 把 turn 追加进 sessions/<session_id>.json。
3. 调用 reasoner.analyze_turn 产出 TurnAnalysis。
4. 用 TurnAnalysis.profile_updates 更新 profile.json。
5. 把 TurnAnalysis 里的 episodic / semantic / long_term / system memory drafts 转成 MemoryItem 后落盘。
6. 更新 session summary、open_loops、turn_count、compression metadata。

这里最关键的一点是：

profile、long_term 和 system memory 的写入，当前已经以 reasoner 的语义判断为主，不再由 manager 自己用“我叫 / 我在备考 / 我的薄弱点”这类固定句式二次抽取。

## 6. reasoner 如何工作

当前有两个 reasoner：

1. HeuristicMemoryReasoner
2. QwenMemoryReasoner

### 6.1 默认运行路径

在项目默认配置里，configs/study_agent.yaml 现在设置为：

```yaml
turn_analysis_mode: llm
```

这意味着默认 runtime 会优先绑定 QwenMemoryReasoner。

UnifiedLegalAgent 不只在聊天链路里绑定它，按钮路径例如：

1. generate_exam
2. generate_report_response
3. direct fallback scoring / profile update

也都会先确保 LLM reasoner 已就绪，然后再写 turn 和记忆。

### 6.2 QwenMemoryReasoner 会产出什么

它当前会尝试输出一个结构化 JSON，包括：

1. summary
2. reasoning_digest
3. episodic_memories
4. semantic_memories
5. long_term_memories
6. system_memories
7. open_loops
8. tags
9. profile_updates
10. importance

其中 system_memories 的约束是：

1. 只有真正应该变成代理级长期规则的内容，才允许写到 system。
2. 用户个人偏好和个人档案必须写到 profile_updates 或 long_term_memories。
3. 普通一轮问答不应该污染 system memory。

### 6.3 HeuristicMemoryReasoner 现在退到什么角色

它现在更像保底回退：

1. 负责生成 summary、reasoning_digest、episodic/semantic 这类基础结构。
2. 如果工具链里已经显式出现 profile_upsert，会从 tool trace 结果回收 profile_updates。
3. 不再依赖那组严格的固定句式去硬抽画像。

所以语义画像写入的主路径已经转到 LLM reasoner。

## 7. prepare_context 是怎么整理上下文的

现在统一上下文接口是：

1. MemoryManager.prepare_turn_context
2. MemoryManager.prepare_turn_context_payload
3. StudyToolExecutor 对外暴露的 prepare_context

prepare_context 会自动完成三件事：

1. maintain_context
2. search
3. 组装 summary_blocks 和 planning_context

### 7.1 maintain_context

内部先做：

1. decay_memories
2. _compress_session_if_needed

也就是：新回合进来时，不需要调用者手工先压缩再检索。

### 7.2 search 的候选来自哪里

search 当前会综合四类候选：

1. profile.json 临时渲染出来的 profile 层候选。
2. system_memories.jsonl 中的 system 层记忆。
3. SessionState 临时渲染出来的 working 层候选。
4. memories.jsonl 中的 long_term / summary / episodic / semantic / working 记忆。

也就是说，主存储仍然是可直接观察的 JSON / JSONL 文件，图结构只是派生索引，不单独承载事实。

### 7.3 当前召回算法

当前上下文召回不是单纯 lexical，也不是只靠共享 reference。

它是一个混合检索流程：

1. 先对所有候选计算 base score。
2. 再根据记忆图关系做 graph bonus。
3. 最后强制补齐 profile / system / working / long_term 这几类锚点。

其中 base score 由以下部分组成：

1. lexical overlap
2. vector similarity
3. importance
4. freshness
5. layer prior
6. hit_count bonus

如果 `configs/study_agent.yaml` 里是下面这个默认配置：

```yaml
memory:
  vectorizer: embedding
  embedding_model_path: models/embeddings/bge-small-zh
  embedding_device: cpu
```

那么这里的 vector similarity 就是通过本地 BGE 模型 `models/embeddings/bge-small-zh` 计算的。

所以，对“现在的上下文召回是不是通过 BGE 检索”这个问题，当前默认 study agent 路径的答案是：

1. 是，会用本地 BGE embedding 参与记忆召回。
2. 但不是只靠 BGE，而是 BGE + lexical + importance + freshness + layer prior + graph bonus 的混合打分。

graph bonus 的来源有两部分：

1. 磁盘上的派生图索引 users/<user_id>/memory_edges.jsonl。
2. 当前候选集合里即时推导出来的关系边。

当前建边与扩散会考虑这些信号：

1. shared_reference
2. tag_overlap
3. keyword_overlap
4. same_session
5. same_category
6. same_layer
7. anchor_bridge

也就是说，即使某条记忆和 query 的字面重合不强，只要它和高分锚点在图上强相关，也仍然有机会被补召回。

### 7.4 为什么保留 JSON 主存储，而不是改成纯图数据库

当前实现刻意没有把主存储改成“只有图，没有原始文件”的形式，原因很直接：

1. profile.json、memories.jsonl、sessions/<session_id>.json 对调试和人工观察最友好。
2. 这些文件本身就是事实权威来源，便于回放、测试和导出报告。
3. memory_edges.jsonl 可以随时重建，不需要承担事实一致性的责任。
4. 图结构最适合做召回增强，而不是取代原始记忆落盘。

所以当前是“主存储可观察 + 派生图索引可重建”的混合方案。

### 7.5 prepare_context 返回哪些块

prepare_turn_context_payload 里最值得看的字段是：

1. user_profile
2. session_state
3. summary_blocks
4. profile_hits
5. system_hits
6. working_hits
7. long_term_hits
8. session_hits
9. guaranteed_hits
10. related_hits
11. planning_context
12. retrieval_meta
13. maintenance

summary_blocks 现在固定包含：

1. profile
2. system
3. long_term
4. session
5. memory
6. relevant
7. maintenance

其中 guaranteed_hits 的设计目标是固定保底返回：

1. profile 锚点
2. system 锚点
3. working 锚点
4. long_term 锚点

related_hits 则是在这些保底锚点之上，再补一层动态高相关历史命中。

retrieval_meta 用来回答“这次到底怎么召回出来的”，当前会暴露：

1. retrieval_strategy
2. vectorizer
3. vector_model
4. graph_edge_count
5. selected_layer_counts
6. guaranteed_hit_count
7. related_hit_count
8. guaranteed_limits

planning_engine、direct fallback 和统一 agent 当前都把它当作正式上下文接口使用。

## 8. 考试、评分和报告如何回写记忆

这一部分现在和记忆层是打通的。

### 8.1 生成测试

generate_exam 会：

1. 读取 profile.json 里的 weak_points 和 study_goals 作为选题偏好。
2. 记录 active_exam_session_id。
3. 在 session metadata 中保存 exam_sessions。
4. 写一条 working 层记忆，表示当前有待作答试卷。

### 8.2 评分

score_exam 会生成结构化评分结果，包括：

1. score_percent
2. earned_score / total_score
3. details
4. weak_tags
5. strong_tags
6. wrong_questions
7. corrected_question_ids

其中 wrong_questions 现在包含：

1. index
2. question
3. options
4. user_answer
5. correct_answer
6. analysis
7. tags

所以用户提交答案后，agent 现在可以直接根据 score_exam 的结构化结果，把错题的正确答案和解释返回给用户，而不需要再临时猜。

### 8.3 store_exam_result 会做什么

它会同时更新三处：

1. session metadata 中对应 exam_session 的 scored 状态。
2. profile.json 里的 weak_points、strong_points、wrong_question_bank、recent_exam_scores。
3. memories.jsonl 中的考试反馈记忆节点。

### 8.4 生成报告

generate_report 会把报告写到 reports/user_reports/<user_id>/，同时把 last_report_path 写回 session 文件。

## 9. 历史压缩如何做

压缩不会删除原始 turns。

原始 turns 永远还在 sessions/<session_id>.json 里。

压缩做的是：

1. 从较早历史切出一段 turn slice。
2. 调用 reasoner.compress_history 得到 CompressionDraft。
3. 把压缩结果写成一条 summary 层 MemoryItem。
4. 更新 session 的 compression_cursor 和 compression_count。

当前默认参数来自 configs/study_agent.yaml：

```yaml
memory:
  recent_turn_window: 8
  compression_after_turns: 10
  compression_chunk_size: 8
  retain_recent_turns: 6
  vectorizer: embedding
  embedding_model_path: models/embeddings/bge-small-zh
  embedding_device: cpu
```

含义分别是：

1. recent_turn_window：SessionState 常驻保留的最近窗口。
2. compression_after_turns：到多少轮后才考虑压缩。
3. compression_chunk_size：每次至少压多少轮。
4. retain_recent_turns：最新多少轮永远不压。

## 10. 你应该如何观察这套系统

如果你想直接观察效果，重点看五个位置：

1. memory/agent_memory/system/system_memories.jsonl
2. memory/agent_memory/users/<user_id>/profile.json
3. memory/agent_memory/users/<user_id>/memories.jsonl
4. memory/agent_memory/users/<user_id>/memory_edges.jsonl
5. memory/agent_memory/users/<user_id>/sessions/<session_id>.json

建议的观察顺序是：

1. 创建一个新用户和新会话，看 profile.json 和 session.json 是否生成。
2. 发一条带稳定偏好的自然表达，看 profile.json、memories.jsonl 和 memory_edges.jsonl 是否同步更新。
3. 调一次 prepare_context，直接观察返回里的 summary_blocks、guaranteed_hits 和 retrieval_meta。
4. 做一次模拟测试并提交错题答案，看 wrong_question_bank、weak_points 和报告文件是否变化。
5. 连续聊多轮，再看 summary 层记忆和 compression_count 是否增长。

## 11. 当前实现的关键结论

如果只记住最重要的几件事，当前版本可以概括成下面这些事实：

1. 记忆维护默认走 LLM reasoner，不再依赖固定句式硬匹配。
2. 默认 study agent 路径的记忆召回会使用本地 BGE embedding，但它只是混合召回中的一部分。
3. system memory、profile、session、user memories 仍然是可直接打开观察的 JSON/JSONL 主存储。
4. memory_edges.jsonl 是派生图索引，用来增强召回，不是事实主文件。
5. prepare_context 是正式上下文接口，memory_search 只是兼容包装。
6. prepare_context 现在会保底返回 profile / system / working / long_term 锚点，再补动态图命中。
7. 会话压缩不会删 turns，只会额外生成 summary 记忆。
8. 评分回复现在会直接基于 score_exam 结果输出错题正确答案和解释。
9. 你看不到 profile.json 时，通常只是对应用户目录尚未重新生成，而不是这套机制不存在。