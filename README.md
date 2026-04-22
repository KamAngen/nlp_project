# 法律学习 Agent

这份 README 服务两个目标：

1. 让组员能够从环境、模型、数据、测试到 UI，按顺序直接跑通整个项目。
2. 让三位组员清楚自己分别改哪个目录、负责什么模块、后面还能往哪一块继续做。

## 先记住这四件事

1. 这个实验统一在 `agent_env` 环境里进行。
2. 所有脚本默认都已经切到 `agent_env`，不再默认使用别的环境。
3. Agent 的训练输出目录是 `ckpt/unified_agent_qlora`。
4. Agent 的记忆目录是 `memory/agent_memory`，报告目录是 `reports/user_reports`。

## 一次性环境准备

在仓库根目录执行：

```bash
cd path/to/nlp_project
export AGENT_ENV_NAME=agent_env
bash scripts/create_agent_env.sh
conda activate "$AGENT_ENV_NAME"
```

说明：

1. `scripts/create_agent_env.sh` 会根据 `environment.agent_env.yml` 创建或更新 `agent_env`。
2. 这个环境文件已经包含 `requirements.txt`、`pytest` 和项目自身的 editable install。
3. 如果以后有人补了新依赖，必须同时更新 `requirements.txt`，必要时同步更新 `environment.agent_env.yml`。
4. 大多数脚本都会通过 `scripts/common.sh` 自动解析 `agent_env` 里的 Python 和相关可执行文件；只要 Conda 能找到这个环境，即使你当前 shell 还没 `conda activate`，脚本也可以直接运行。

如果你只想在已有 `agent_env` 上补装依赖，可以运行：

```bash
bash scripts/install_requirements.sh
```

## 最短可运行路径

如果你的目标只是让系统跑起来，按下面顺序执行即可。

### 1. 下载基础模型

```bash
cd path/to/nlp_project
HF_ENDPOINT=https://hf-mirror.com bash scripts/download_models.sh
```

默认会准备：

1. `models/embeddings/bge-small-zh`
2. `models/qwen/Qwen3_4B`

### 2. 下载 DISC-Law-SFT

如果你的 `data/` 目录是空的，先执行：

```bash
bash scripts/download_disc_law.sh
```

这一步会把原始 jsonl 下载到 `data/disc_law/raw/`，并生成规范化文件 `data/disc_law/disc_law_normalized.jsonl`。

如果你已经从组内共享数据中拿到了 `data/disc_law/`，可以跳过这一步。

### 3. 生成学习场景数据并构建学习知识清单

```bash
bash scripts/build_study_knowledge.sh
```

这一步会先检查并补齐 `data/legal_study_agent/` 下的四个运行时文件：

1. `question_bank.jsonl`：模拟测试题库。
2. `case_bank.jsonl`：案例讲解题库。
3. `common_knowledge.jsonl`：学习方法与通用知识点。
4. `system_seed_memories.json`：系统级学习记忆与回答策略。

默认情况下：

1. 如果这四个文件已经存在且数量足够，脚本会跳过重建。
2. 如果它们缺失或只有仓库里的小样例，脚本会优先利用 `data/disc_law/disc_law_normalized.jsonl` 自动生成足量数据。
3. 如果 `data/law_files/catalogs/law_catalog_master.csv` 存在，脚本还会用法规目录补充部分检索提示型常识条目。

如果你需要强制重建这四个文件，可以执行：

```bash
FORCE_REBUILD=1 bash scripts/build_study_knowledge.sh
```

### 4. 准备法规检索资源

```bash
bash scripts/build_knowledge_base.sh
```

这条脚本有两种工作模式：

1. 如果本地已经有 `data/law_files` 下的法规文件，它会直接构建语料和索引。
2. 如果本地没有法规文件、但你配置了 Kaggle，它会尝试下载预构建的 RAG artifacts。

也就是说，从“空的 data/artifacts/memory/models/reports 目录”起步时，完整可运行路径分成两种：

1. 你有共享的 `data.zip` 或其他方式拿到 `data/law_files`：直接执行这一步即可本地构建法规检索资源。
2. 你没有 `data/law_files`：请先配置 `~/.kaggle/kaggle.json`，再执行这一步，让脚本下载预构建 artifacts。

如果要走 Kaggle 预构建 artifacts 路径，请确认两件事：

1. `agent_env` 已经按上面的环境步骤创建完成。
2. `~/.kaggle/kaggle.json` 已配置好可用凭证。

如果你只想单独下载预构建的 RAG artifacts，也可以直接运行：

```bash
bash scripts/download_rag_artifacts.sh
```

### 5. 运行核心测试

```bash
cd path/to/nlp_project
PYTHONPATH=src python -m pytest \
  tests/test_study_data_builder.py \
  tests/test_context_engine.py \
  tests/test_study_tools.py \
  tests/test_study_agent.py \
  tests/test_web_study_workspace.py \
  tests/test_web_app.py
```

这组核心回归应当在 `agent_env` 中全部通过；以当前 `pytest` 实际输出为准，不再在 README 里写死通过数量。

说明：

1. `memory/agent_memory` 和 `reports/user_reports` 不需要手工创建，运行时会自动生成。
2. 如果 `data/legal_study_agent/` 没生成好，UI 可能能打开，但模拟测试、案例解释和学习报告不会完整可用。
3. 如果 `artifacts/` 没有通过本地构建或 Kaggle 下载补齐，法规检索链路不会完整可用。

### 6. 启动 Web UI

```bash
bash scripts/launch_web_ui.sh
```

脚本会优先使用 `127.0.0.1:7860`。如果 7860 已被占用，它会自动顺延到后续空闲端口，并在终端里打印实际使用的端口。

默认地址：

```text
http://127.0.0.1:7860
```

如果 7860 已占用，请以终端里打印出来的实际地址为准。

如果你要指定端口：

```bash
HOST=127.0.0.1 PORT=7864 bash scripts/launch_web_ui.sh
```

## 完整实验链路

如果你的目标不是只看 UI，而是完整跑训练与评测，请按下面顺序执行。

### 1. 下载并规范化 DISC-Law-SFT

```bash
bash scripts/download_disc_law.sh
```

### 2. 构建法规知识库

```bash
bash scripts/build_knowledge_base.sh
```

### 3. 构建 Agent 训练数据

```bash
TRAIN_COUNT=1600 EVAL_COUNT=200 bash scripts/generate_agent_data.sh
```

这一步会把两类样本一起写进训练集：

1. DISC-Law-SFT 派生的旧法律 Agent 轨迹。
2. 本地学习场景轨迹，包括画像更新、法考问答、案例解释、模拟测试、评分和报告。

### 4. 训练 LoRA 适配器

```bash
bash scripts/train_agent.sh
```

输出目录固定为：

```text
ckpt/unified_agent_qlora/
```

### 5. 评测基座模型和适配器模型

```bash
bash scripts/evaluate_agent.sh
```

评测结果默认写到：

1. `outputs/eval_base`
2. `outputs/eval_adapter`
3. `outputs/eval_case_studies.md`

### 6. 一键跑完整正式实验

```bash
bash scripts/run_formal_experiment.sh
```

## 手动命令入口

除脚本外，以下 CLI 入口也可以直接调用。

### 1. 聊天

```bash
cd path/to/nlp_project
PYTHONPATH=src python -m legal_agent.cli chat \
  --config configs/defaults.yaml \
  --study-config configs/study_agent.yaml \
  --model-path models/qwen/Qwen3_4B \
  --retrieval-device cpu \
  --model-device auto
```

### 2. 启动 Web UI

```bash
cd path/to/nlp_project
PYTHONPATH=src python -m legal_agent.cli web-ui \
  --config configs/web_ui.yaml \
  --study-config configs/study_agent.yaml \
  --host 127.0.0.1 \
  --port 7860 \
  --retrieval-device cpu
```

如果 7860 已占用，这条命令也会自动切换到后续空闲端口。

## 当前目录结构

推荐把仓库理解成四层：

1. context_engine：第一位同学负责的上下文与记忆层。
2. planning_engine：第二位同学负责的规划层。
3. rag_engine：第三位同学负责的知识库与检索层。
4. legal_agent：集成层，由负责人维护，负责 CLI、UI 和三模块接线。

```text
nlp_project/
├── configs/
│   ├── defaults.yaml
│   ├── study_agent.yaml
│   └── web_ui.yaml
├── data/
│   ├── disc_law/
│   ├── generated/
│   ├── law_files/
│   └── legal_study_agent/
├── artifacts/
├── ckpt/
│   └── unified_agent_qlora/
├── memory/
│   └── agent_memory/
├── reports/
│   └── user_reports/
├── scripts/
├── src/
│   ├── context_engine/
│   ├── planning_engine/
│   ├── rag_engine/
│   └── legal_agent/
├── tests/
└── README_zh.md
```

## 分工边界

### 组员 1：记忆、画像与会话状态

负责目录：

1. `src/context_engine/`
2. `memory/agent_memory/`

重点文件：

1. `src/context_engine/manager.py`
2. `src/context_engine/store.py`
3. `src/context_engine/schemas.py`

你主要改什么：

1. 用户画像字段抽取和更新策略。
2. 会话摘要、长期记忆、工作记忆的写回逻辑。
3. 考试状态、报告路径、最近会话摘要的持久化。

当前已实现：

1. 分层 memory 数据结构。
2. 文件型存储层。
3. 用户画像抽取与更新。
4. 会话状态持久化。
5. 考试结果反哺用户画像。
6. 报告快照生成。
7. 用户与会话 CRUD。

你接下来要继续做的任务：

1. 为 memory item 增加冲突消解和版本控制。
2. 增加 embedding 检索，替代当前 lexical + heuristic 命中。
3. 增加长期语义记忆 consolidation。
4. 对同一用户跨会话共享的稳定事实做去重。
5. 设计“用户删除”后的安全清理策略。
6. 为画像更新添加可信度与来源标记。
7. 提高画像抽取准确率，减少错误自动写回。
8. 优化多层记忆检索打分，减少无关历史命中。
9. 为报告生成补更多结构化学习快照。

你必须遵守的接口约束：

1. UserProfile 结构字段名不能随意改。
2. SessionState 的 turns、summary、metadata 字段必须向后兼容。
3. profile.json 和 session.json 必须保持 JSON 可直接读取，不要换成二进制格式。
4. 所有持久化操作都要通过 store.py 和 manager.py，不要在 UI 层直接写文件。

### 组员 2：Agent 主链、训练与评测

负责目录：

1. `src/legal_agent/agent/`
2. `src/legal_agent/training/`
3. `src/legal_agent/unified_agent.py`
4. `src/legal_agent/unified_tools.py`
5. `configs/defaults.yaml`
6. `configs/study_agent.yaml`

重点文件：

1. `src/legal_agent/agent/engine.py`
2. `src/legal_agent/unified_agent.py`
3. `src/legal_agent/training/dataset_builder.py`
4. `src/legal_agent/training/trajectory_builder.py`

你主要改什么：

1. ReAct 主链、工具调用与 direct fallback。
2. 训练数据构造、LoRA 训练与评测闭环。
3. `ckpt/unified_agent_qlora` 的训练参数和输出管理。

后续改进方向：

1. 继续提升基础模型不调工具时的恢复策略。
2. 扩大本地学习样本和轨迹覆盖面。
3. 优化首轮分析与追问策略，减少无意义澄清。

### 组员 3：RAG 与知识库

负责目录：src/rag_engine

当前已实现：

1. 题库、案例库、常识知识库加载。
2. 本地 lexical retrieval。
3. 法规 RAG 兼容适配。
4. 面向模拟测试的题目采样。

你接下来要继续做的任务：

1. 扩大知识库，增加常识库以及更多的法律案例、法考题目以及解答，并以合适的方式进行存储，提供检索工具接口。
1. 给题库和案例库增加 embedding 检索。
2. 增加 reranker。
3. 为题目增加更细的 topic、difficulty、knowledge point 标注。
4. 设计题目 curriculum sampling。
5. 为案例与法条建立引用关系图。
6. 提供离线构建索引与在线热更新的统一接口。

### 集成负责人：UI 与交互产品层

负责目录：

1. `src/legal_agent/web/`
2. `scripts/launch_web_ui.sh`
3. `tests/test_web_study_workspace.py`
4. `tests/test_web_app.py`

重点文件：

1. `src/legal_agent/web/unified_workspace.py`
2. `src/legal_agent/web/app.py`

你主要改什么：

1. 单页工作台布局。
2. 用户/会话管理控件。
3. 聊天区、阶段提示、Trace、报告展示和复制按钮。
4. 页面在不同窗口尺寸下的稳定性。

后续改进方向：

1. 继续优化移动端和小屏适配。
2. 把阶段提示做得更细，但不要打断最终答案替换。
3. 强化报告面板和聊天区之间的联动体验。

## 当前最重要的文件阅读顺序

如果是第一次接手这个仓库，建议按下面顺序阅读：

1. `src/legal_agent/unified_agent.py`
2. `src/legal_agent/unified_tools.py`
3. `src/legal_agent/agent/engine.py`
4. `src/legal_agent/web/unified_workspace.py`
5. `src/context_engine/manager.py`
6. `src/legal_agent/training/dataset_builder.py`
7. `configs/defaults.yaml`
8. `configs/study_agent.yaml`
9. `configs/web_ui.yaml`
