# Legal Study Agent 数据初始化指南

本指南介绍如何清空现有数据并重新初始化法律学习助手的核心数据库、向量索引及知识图谱。

## 1. 环境准备

确保你已经激活了 Conda 环境，并且设置了正确的 `PYTHONPATH`。

```bash
conda activate agent_env
export PYTHONPATH=$(pwd)/src
```

## 2. 彻底清理现有数据 (可选)

如果你需要从零开始，请运行以下命令删除所有索引、图谱和缓存的原始数据。

```bash
# 杀掉可能正在运行的服务器
lsof -ti:2266 | xargs kill -9 2>/dev/null

# 删除索引和图谱
rm -rf data/indices/*

# 删除原始 JSONL 存证 (慎用，如果你有自定义数据请先备份)
rm -f data/legal_study_agent/*.jsonl
```

## 3. 重新生成基础数据

使用内置脚本根据配置生成模拟的题库、案例库和常识知识。

```bash
# 设置 Python 路径并执行生成脚本
export PYTHON_BIN=$(which python)
bash scripts/build_study_knowledge.sh
```

生成后的文件位于 `data/legal_study_agent/` 目录下。

## 4. 构建向量索引 (Embedding)

这一步会加载本地的 Qwen-4B 模型（或配置的 Embedding 模型），并为所有记录生成向量。在 Mac 上会自动使用 MPS 加速。

```bash
python scripts/build_vector_indices.py
```

执行完成后，`data/indices/` 目录下会生成 `case_bank`, `question_bank` 等子文件夹及其对应的 `embeddings.pt`。

## 5. 构建知识图谱

基于案例和题库中的标签（Tags/Statutes）建立实体间的关联关系。

```bash
python scripts/build_knowledge_graph.py
```

生成的文件位于 `data/indices/legal_graph.json`。

## 6. 启动/重启服务器

一切就绪后，启动 RAG 引擎服务器：

```bash
python -u src/rag_engine/server.py
```

## 常用脚本说明

| 脚本名称 | 作用 | 备注 |
| :--- | :--- | :--- |
| `scripts/build_study_knowledge.sh` | 生成原始 JSONL 数据 | 可通过环境变量修改生成数量 |
| `scripts/build_vector_indices.py` | 生成向量索引 | 耗时较长，依赖 GPU/MPS |
| `scripts/build_knowledge_graph.py` | 生成图谱 JSON | 速度快，依赖 JSONL 数据 |
| `src/rag_engine/server.py` | 启动后端服务 | 端口默认为 2266 |

## 环境变量配置 (在 scripts/common.sh 中定义)
- `QUESTION_COUNT`: 生成题目的数量 (默认 180)
- `CASE_COUNT`: 生成案例的数量 (默认 96)
- `COMMON_COUNT`: 生成常识的数量 (默认 24)
