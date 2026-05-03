# Legal RAG Pro 运行指南

这份指南将帮助你从零开始在本地运行 Legal RAG 项目。本项目已经预先构建好了庞大的向量索引和知识图谱，你**不需要**重新执行耗时极长的数据向量化过程，只需补齐核心模型即可瞬间启动。

## 📦 环境准备

### 1. 解压项目
将收到的 `nlp_project_share.zip` 解压到本地。项目中已经包含了完整的代码结构和 `data/indices` 索引文件。

### 2. 创建 Conda 环境
推荐使用 Python 3.11 构建独立的运行环境：
```bash
conda create -n legal_rag python=3.11 -y
conda activate legal_rag
```

### 3. 安装依赖
进入项目根目录，安装所需的依赖包：
```bash
pip install -r requirements.txt
```

---

## 🤖 补齐 Qwen 核心模型

为了减小打包体积，压缩包中排除了体积巨大的 Qwen-4B 模型。你需要手动下载它。

1. 在项目根目录下，确认或创建以下文件夹路径：
   `models/qwen/Qwen3_4B`
2. 前往 HuggingFace 或 ModelScope 下载 **Qwen/Qwen3-4B** 的完整权重文件。
3. 将下载好的所有文件（包括 `config.json`, `safetensors` 文件等）放入 `models/qwen/Qwen3_4B` 文件夹中。

*(注：项目已经自带了 `bge-reranker-base` 和默认的 embedding 模型，仅缺失 Qwen)*

---

## 🚀 启动服务

当模型补齐后，你可以直接启动后端服务器，它会自动加载现成的索引，提供极速响应。

在项目根目录下运行：

```bash
export PYTHONPATH=src
python src/rag_engine/server.py
```

当你在控制台看到如下提示时，说明服务已成功启动：
> `INFO: Uvicorn running on http://0.0.0.0:2266 (Press CTRL+C to quit)`

---

## 🌐 访问前端面板

服务启动后，直接在浏览器中打开：
[http://127.0.0.1:2266](http://127.0.0.1:2266)

你将看到完整的可视化面板，支持：
- 混合检索（语义 + 关键词）
- 知识图谱动态可视化
- 智能课程选题
- 数据的热更新导入
