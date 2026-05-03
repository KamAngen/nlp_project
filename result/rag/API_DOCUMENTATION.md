# Legal RAG Pro - API 接口文档

**Base URL**: `http://localhost:2266`

---

## 1. 混合检索 API

执行包含法条、案例、常识的混合检索，支持自动重排（Rerank）。

**Endpoint**: `/search`
**Method**: `POST`

**请求 Body (JSON)**:
```json
{
  "query": "非法集资的量刑标准是什么？",
  "sources": ["case_bank", "question_bank", "common_knowledge"], 
  "mode": "hybrid",
  "top_k": 10
}
```
*参数说明*:
- `sources`: 数据源过滤。可选值：`case_bank` (案例), `question_bank` (题目), `common_knowledge` (常识)。若传空数组 `[]` 则代表搜索全部。
- `mode`: 检索模式。可选值：`hybrid` (词法+语义融合), `lexical` (纯关键词), `embedding` (纯语义向量)。
- `top_k`: 返回的最大结果数量。

*(注：系统为了向下兼容，依然保留了 `/search/embedding` 和 `/search/lexical` 这两个独立的历史接口。但目前推荐统一使用上述的 `/search` 接口，通过改变 `mode` 参数来切换底层引擎，这样代码维护更简洁。)*

---

## 2. 数据热加载 (Hot Update)

系统支持不停机动态更新数据，更新后会立即生效并反映在检索和图谱中。

### 2.1 单条数据上传

**Endpoint**: `/update`
**Method**: `POST`

**请求 Body (JSON)**:
```json
{
  "source_type": "case_bank",
  "title": "测试案例标题",
  "content": "案例的详细事实内容...",
  "tags": ["刑法第二百六十六条"],
  "metadata": {}
}
```
*参数说明*:
- `source_type`: 目标知识库，与搜索 `sources` 保持一致。

### 2.2 批量导入 (支持进度轮询)

**Endpoint**: `/update_batch`
**Method**: `POST`

**请求 Body (JSON Array)**:
```json
[
  {
    "source_type": "question_bank",
    "title": "测试题目1",
    "content": "这是一道关于非法集资的单选题...",
    "tags": ["刑法", "非法集资"],
    "metadata": {
      "difficulty": "medium",
      "options": ["A. 正确", "B. 错误"],
      "answer": "A"
    }
  },
  {
    "source_type": "case_bank",
    "title": "王某非法吸收公众存款案",
    "content": "王某在未取得金融许可证的情况下，向社会不特定对象公开宣传...",
    "tags": ["非法吸收公众存款罪", "刑法第一百七十六条"],
    "metadata": {
      "court": "北京市朝阳区人民法院",
      "case_number": "(2021)京0105刑初123号",
      "statutes": ["刑法第一百七十六条", "最高法关于审理非法集资刑事案件具体应用法律若干问题的解释"]
    }
  },
  {
    "source_type": "common_knowledge",
    "title": "非法集资的定义与特征",
    "content": "非法集资是指未经国务院金融管理部门依法许可或者违反国家金融管理规定...",
    "tags": ["金融犯罪", "概念解析"],
    "metadata": {
      "category": "概念定义",
      "importance": "high"
    }
  }
]
```
**响应 (JSON)**:
```json
{
  "message": "Batch update started in background",
  "task_id": "8a7b6c5d-4e3f-2g1h-0i9j"
}
```
*说明*: 批量导入为异步任务，后端会自动切分为 Batch=32 进行向量化并跳过重复 ID。返回 `task_id` 供前端轮询进度。

### 2.3 查询后台任务进度

**Endpoint**: `/task/{task_id}`
**Method**: `GET`

**响应 (JSON)**:
```json
{
  "status": "processing",
  "total": 1000,
  "processed": 320
}
```
*状态值*: `processing` (处理中), `completed` (已完成), `not_found` (任务不存在)。

---

## 3. 智能课程采样

根据用户指定的掌握程度和关键主题，智能抽取定制化题目池。采用 RRF 混合检索构建候选池并基于掌握度权重进行采样。

**Endpoint**: `/sample`
**Method**: `GET`

**Query 参数**:
- `mastery`: (Float) 用户当前对该领域的掌握度，范围 `0.0 ~ 1.0`。数值越低越容易抽中难度标记为 `easy` 的题目。
- `topic`: (String, 可选) 指定的主题词（如："非法集资"）。如果为空则为全局随机采样。

**请求示例**:
`GET /sample?mastery=0.3&topic=非法集资`

---

## 4. 知识图谱 API

### 4.1 获取节点的关联图谱数据

用于 Cytoscape 前端可视化的数据结构，包含核心节点、相关法条、二级关联案例及语义相似节点。

**Endpoint**: `/graph/{node_id}`
**Method**: `GET`

**响应 (JSON)**:
```json
{
  "center": {
    "id": "cail2018_443306",
    "title": "张某某集资诈骗案"
  },
  "statutes": ["刑法第一百九十二条", "刑法第二十五条第一款"],
  "related_cases": [
    {"id": "case_123", "title": "王某非法吸收公众存款案"}
  ],
  "similar_cases": [
    {"id": "case_456", "title": "李某某集资诈骗案二审"}
  ],
  "similar_questions": []
}
```

### 4.2 获取单一节点/记录详情

用于在用户点击图谱节点或搜索结果时，拉取完整的文本内容。

**Endpoint**: `/record/{record_id}`
**Method**: `GET`

**响应 (JSON)**:
```json
{
  "record_id": "cail2018_443306",
  "source_type": "case_bank",
  "title": "张某某集资诈骗案",
  "content": "详细的案情描述...",
  "tags": ["刑法第一百九十二条"],
  "metadata": {}
}
```

---

## 5. 系统状态 API

获取当前各个知识库的容量统计。

**Endpoint**: `/stats`
**Method**: `GET`

**响应 (JSON)**:
```json
{
  "question_bank_count": 1947,
  "case_bank_count": 2096,
  "common_knowledge_count": 25,
  "legacy_statute_enabled": false,
  "vector_indices_loaded": ["question_bank", "case_bank", "common_knowledge"]
}
```
