# 法律学习 Agent 中文说明

本 README 描述当前仓库里“法律学习 Agent”这条正式运行链路，重点覆盖四件事：

1. Agent 收到用户输入后，如何决定走问答、追问、出题、评分还是报告。
2. planning 和工具触发标准是什么，哪些工具只在特定条件下触发。
3. 题库现在如何从源头生成，为什么不再依赖 runtime 修题。
4. 模拟测试、主观题评分、错题库、记忆、报告和 Web UI 是怎样串起来的。

当前实现的核心原则只有一条：

- 题库质量问题必须在生成阶段解决，运行时只消费结构化题库，不再对单道题做临时补丁式修复。

## 1. 项目定位

这个 Agent 不是一个泛法律咨询机器人，而是一个面向法考学习场景的统一学习代理。它的职责包括：

- 处理法考知识问答。
- 记录用户画像，例如目标科目、薄弱点、表达偏好、会话记忆。
- 生成模拟测试，支持单选题、简答题、案例分析题。
- 对用户答题进行评分，其中主观题支持 LLM 理解式评分。
- 自动沉淀错题、强弱项、报告和会话记忆。

当前默认配置见 configs/study_agent.yaml，关键路径如下：

- 题库：data/legal_study_agent/question_bank.jsonl
- 案例库：data/legal_study_agent/case_bank.jsonl
- 通用知识：data/legal_study_agent/common_knowledge.jsonl
- 系统记忆：data/legal_study_agent/system_seed_memories.json
- 学习记忆：memory/agent_memory
- 学习报告：reports/user_reports
- 题库 manifest：artifacts/study_knowledge_manifest.json

## 2. 整体行为链路

### 2.1 入口层

统一入口是 UnifiedLegalAgent。默认 study 配置里：

- planner_backend: llm_react
- turn_analysis_mode: llm

实际处理顺序不是“盲目先问 LLM 再说”，而是“先做局部规则路由，再让 engine 和工具链协作”。

高层流程如下：

1. 载入用户 profile、session state、active exam、最近会话记忆。
2. 识别是否是按钮专用动作或直接工具型请求。
3. 如果当前存在 active exam，优先判断用户输入是不是答题卡。
4. 如果不是答题卡，再判断是否是报告请求、模拟测试请求、画像更新或普通问答。
5. 默认路径下由 llm_react engine 结合 planning_context 决定是否调用工具。
6. 如果模型没有正确触发目标工具，统一入口会使用 direct fallback 补做一次确定性路由。

这意味着系统同时具备两层保障：

- 第一层：LLM 驱动的工具规划与自然语言处理。
- 第二层：基于输入形态的直接兜底，不让核心功能因为模型漏调工具而失效。

### 2.2 追问逻辑

只有在信息缺失到足以影响结果可靠性时，系统才会触发 ask_followup。当前实现专门修过两类问题：

- 用户已经回答了关键追问后，不再继续进行无关的第二轮追问。
- 检索证据已经足够时，优先进入最终综合答复，而不是继续追问。

因此，现在追问应当只服务于“缺少关键事实”这一件事，而不是泛化成任何不确定都追问。

## 3. Planning 与路由规则

StudyPlanner 负责非按钮路径下的轻量 planning。默认识别以下 intent：

- profile_lookup
- profile_update
- mock_exam_generate
- mock_exam_score
- report_generation
- legal_calculation
- legal_qa

### 3.1 题型识别规则

模拟测试请求除了主题和题量，还会识别 question_types：

- 命中“简答 / 主观 / 问答”时：short_answer
- 命中“案例 / 案例分析”时：case_analysis
- 命中“混合 / 综合题型”时：single_choice + short_answer + case_analysis
- 其余默认：single_choice

### 3.2 答题卡识别规则

只要当前 session 有 active exam，以下两类输入都会优先进入评分路径：

- 选择题格式：1.A 2.B 3.C
- 编号文本格式：

```text
1. 第一题答案
2. 第二题答案
```

这个优先级高于画像更新抽取，避免用户提交答案时被错误当成“我想更新偏好”。

### 3.3 direct fallback 规则

当模型没有主动把请求路由到正确工具时，统一入口会按以下优先级兜底：

1. active exam + 答题卡：score_exam
2. 报告请求：generate_report
3. 模拟测试请求：generate_exam
4. 其余问题：rag_search，必要时补 legacy statute retrieval

UI 按钮路径不依赖聊天模型，直接进入对应 direct tool helper。

## 4. 工具触发标准

| 工具 | 典型触发条件 | 主要作用 | 是否直接面向用户 |
| --- | --- | --- | --- |
| prepare_context | 生成测试、生成报告、需要 planning_context 时 | 汇总长期画像、会话摘要、召回命中和上下文块 | 否 |
| profile_upsert | 用户主动透露备考目标、薄弱点、偏好、习惯 | 更新长期画像 | 是 |
| profile_view | 查看画像、出题前读取偏好、报告前读取状态 | 返回当前用户画像 | 是 |
| rag_search | 普通法考问答、法律分析、学习型检索 | 检索题库、案例库、常识知识 | 是 |
| calculator | 用户问题含明确数值计算 | 执行安全计算 | 是 |
| generate_exam | 用户要刷题、模拟测试、章节练习、真题模拟 | 组卷并写入 active exam | 是 |
| score_exam | 用户提交答题卡且当前有 active exam | 评分、更新错题和强弱项 | 是 |
| generate_report | 用户主动要报告，或评分后顺带生成反馈报告 | 输出 markdown 报告和 JSON snapshot | 是 |
| ask_followup | 关键事实缺失且无法稳妥回答 | 向用户追问一个明确问题 | 是 |

## 5. 每个工具的实际行为

### 5.1 prepare_context

prepare_context 返回的是 planning payload，不是最终面向用户的回答。里面包含：

- planning_context
- summary_blocks
- profile_hits
- system_hits
- working_hits
- long_term_hits
- session_hits
- guaranteed_hits
- related_hits
- retrieval_meta

它的职责是把 MemoryManager 当前知道的用户长期信息、近期会话和高相关记忆压缩成一段可直接供 planner 或 engine 消费的上下文。

### 5.2 profile_upsert / profile_view

profile_upsert 会把用户自然语言中可稳定保留的信息写入长期画像，常见字段包括：

- study_goals
- weak_points
- strong_points
- preferences
- attributes

profile_view 则负责回读画像，给后续选题、报告和答复语气提供依据。

### 5.3 rag_search

rag_search 当前主要检索三类学习资源：

- question_bank
- case_bank
- common_knowledge

如果 study config 开启 use_legacy_statute_rag，还会补走法规知识库检索，用于法条层兜底。

### 5.4 generate_exam

generate_exam 的输入结构已经升级，不再只有 topic 和 question_count，还包括：

- topic
- question_count
- exam_type
- question_types

当前支持的 question_type：

- single_choice
- short_answer
- case_analysis

它会做三件事：

1. 从 profile 中读取 weak_points、study_goals、strong_points、wrong_question_bank。
2. 调用 KnowledgeService.sample_questions 进行结构化选题。
3. 将生成好的题目写入 active exam session，等待后续 score_exam 使用。

### 5.5 score_exam

score_exam 会从当前 active exam 里读取题目，然后按题型分流：

- single_choice：按选项标签或选项文本精确比对。
- short_answer / case_analysis：走主观评分链。

主观评分链当前设计为：

1. 优先调用 UnifiedLegalAgent 注入的 subjective_exam_grader。
2. 这个 grader 使用本地 Qwen 模型输出严格 JSON 评分结果。
3. 如果模型不可用或 JSON 解析失败，退回 heuristic fallback。

主观评分输出字段包括：

- score
- feedback
- matched_points
- missing_points

评分完成后，系统会同步更新：

- exam result
- wrong_question_bank
- weak_points
- strong_points
- corrected_question_ids

### 5.6 generate_report

generate_report 会构建 report snapshot，并写出两份文件：

- markdown 报告
- JSON snapshot

报告内容来自 MemoryManager 汇总的会话结果，不是临时拼接的一段聊天文本。评分完成后通常会顺带生成一份 exam_feedback 或 mock_exam_review。

## 6. 模拟测试的完整生命周期

### 6.1 出题

出题时的选择逻辑现在以 source-level metadata 为准：

- topic 直接读取题库记录的 metadata.topic
- question_type 直接读取 metadata.question_type
- reference_answer / references / source_metadata 也直接来自题库记录

运行时不再做以下事情：

- 不再把自由问答硬改造成单选题
- 不再对某道题临时修选项
- 不再靠 question/analysis/options 里的零散关键词给题目临时改科目

如果题目本身脏，正确做法是重新 build study knowledge，而不是在 sample_questions 阶段补丁修。

### 6.2 用户作答

当前支持三种答题方式：

```text
1.A 2.B 3.C
```

```text
1. 第一题的文字答案
2. 第二题的文字答案
```

```text
1.A
2. 这道简答题我认为……
3. 案例中应先判断……
```

### 6.3 评分与复盘

评分结果会返回：

- 总分和百分比
- 每题得分
- 主观题 grading_feedback
- matched_points / missing_points
- wrong_questions
- 强项与薄弱标签

统一入口渲染给用户时，会根据题型分别展示：

- 选择题：你的答案、正确答案、正确选项内容、解析
- 主观题：得分、你的作答、参考答案、评价、缺失要点、解析

### 6.4 错题回放

wrong_question_bank 会在“薄弱点强化”这类考试里被优先抽回。答对后，会自动把对应题从错题库中移除。

## 7. 题库生成逻辑

### 7.1 当前生成入口

题库生成入口有两条：

```bash
bash scripts/build_study_knowledge.sh
```

```bash
python -m legal_agent.cli build-study-kb \
  --config configs/study_agent.yaml \
  --app-config configs/defaults.yaml \
  --force-rebuild
```

prepare-study-data 也可以单独只生成学习数据，不重建 manifest。

### 7.2 0 = 全部可用题

当前 question_count 和 case_count 的新语义是：

- 0：生成全部可用题 / 全部可用案例
- 正整数：按目标数量裁剪

默认脚本 scripts/build_study_knowledge.sh 已经切到：

- QUESTION_COUNT=0
- CASE_COUNT=0

也就是说，默认行为已经从“抽固定 180 / 96”改成“全量可用题重建”。

### 7.3 builder 做了什么

prepare_study_knowledge_assets 现在在生成阶段完成这些工作：

1. 从 DISC-Law normalized 数据中提取 candidate。
2. 识别显式单选题，保留原始选项和正确答案。
3. 对非显式选择题，按 task_family 和文本特征判定为 short_answer 或 case_analysis。
4. 通过 question + answer + references 的加权打分推断 topic。
5. 输出结构化 question_bank row，而不是只写一段文本。

### 7.4 当前 question_bank schema

自动生成题目的关键字段包括：

- question_id
- topic
- difficulty
- question
- options
- answer
- analysis
- tags
- score
- question_type
- evaluation_mode
- reference_answer
- references
- source_metadata

其中最重要的三项是：

- question_type：运行时题型分流依据
- evaluation_mode：objective_choice 或 llm_subjective
- source_metadata：保留 subset、task_family、topic_confidence 等源信息

### 7.5 为什么不再做 runtime 修题

旧路径的问题是：

- 自由问答数据会被硬改成伪单选题。
- topic 会在运行时因为弱关键词被误判。
- 某些题的 question 和 options 本来就不是一套，runtime 越修越脏。

现在的原则是：

- builder 负责干净。
- runtime 负责消费。

如果要修题库，应该回到 study_knowledge.py，而不是在 service.py 里对单题打补丁。

## 8. 记忆与 RAG

### 8.1 MemoryManager 维护的核心状态

- 用户长期画像
- session turns
- active exam
- exam results
- wrong_question_bank
- last_report_path

### 8.2 记忆层用途

- 长期画像用于后续选题偏好和答复风格。
- 会话摘要用于 planning_context。
- exam session 用于评分时关联当前试卷。
- 错题库用于强化练习优先回放。

### 8.3 检索层用途

RAG 不是单独为了回答“法条问答”，也服务于：

- 学习型解释
- 题目解析补充
- 报告生成引用
- 追问后的最终综合答复

## 9. Web UI 与按钮行为

Web UI 入口：

```bash
bash scripts/launch_web_ui.sh
```

聊天区和按钮区的行为不完全相同：

- 聊天区优先走 llm_react + direct fallback。
- 按钮区直接走统一入口的 direct tool helper。

这保证了“生成模拟测试”“生成学习报告”这类高确定性动作不会因为模型自由发挥而偏航。

## 10. 常用命令

### 10.1 强制重建题库

```bash
FORCE_REBUILD=1 bash scripts/build_study_knowledge.sh
```

### 10.2 单次提问

```bash
python -m legal_agent.cli study-ask \
  --config configs/study_agent.yaml \
  --question "给我两道民诉简答题"
```

### 10.3 命令行聊天

```bash
python -m legal_agent.cli study-chat \
  --config configs/study_agent.yaml
```

### 10.4 生成报告

```bash
python -m legal_agent.cli study-report \
  --config configs/study_agent.yaml \
  --report-type study_progress
```

## 11. 建议测试顺序

当前最有价值的回归不是只跑一个文件，而是按下面顺序：

```bash
python -m pytest tests/test_study_data_builder.py tests/test_study_planner.py tests/test_study_tools.py tests/test_study_agent.py -q
```

如果要做更宽一层的联调回归，再追加：

```bash
python -m pytest \
  tests/test_context_engine.py \
  tests/test_study_data_builder.py \
  tests/test_study_tools.py \
  tests/test_study_planner.py \
  tests/test_study_agent.py \
  tests/test_web_app.py \
  tests/test_web_study_workspace.py -q
```

## 12. 当前实现的边界

最后强调几条当前实现边界，避免误解：

- runtime 不负责修脏题；脏题应回源头重建。
- 主观题默认支持 LLM 评分，但仍保留 heuristic fallback 兜底。
- mixed exam 已支持生成和评分，但默认自然语言出题请求仍优先单选，除非用户明确说“简答”“案例”“混合题型”。
- 如果法规 RAG 未准备完整，普通学习问答仍可运行，但法条层依据会变弱。

如果你要继续扩展系统，优先级建议如下：

1. 继续提升 builder 的 topic purity 和 question_type 覆盖。
2. 扩充主观题 rubric 和评分 prompt。
3. 补更完整的 e2e UI 回归，而不是继续在 runtime 做单题补丁。
