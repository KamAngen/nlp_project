import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from rag_engine.service import KnowledgeService

def main():
    print("初始化服务...")
    service = KnowledgeService(
        question_bank_path=PROJECT_ROOT / "data/legal_study_agent/question_bank.jsonl",
        case_bank_path=PROJECT_ROOT / "data/legal_study_agent/case_bank.jsonl",
        common_knowledge_path=PROJECT_ROOT / "data/legal_study_agent/common_knowledge.jsonl"
    )
    
    index_path = PROJECT_ROOT / "data/indices"
    emb_model_path = PROJECT_ROOT / "models/qwen/Qwen3_4B"
    reranker_path = PROJECT_ROOT / "models/reranker/bge-reranker-base"
    
    print("加载 Qwen3-4B Embeddings...")
    service.load_indices(index_path, emb_model_path)
    print("加载 BGE Reranker...")
    service.load_reranker(reranker_path)
    
    # 评测集：Query -> 期望包含的关键字或特定 ID (简化版评测)
    # 由于没有标注的 Ground Truth，我们用"相关性平均得分"或"命中期望关键词的文档比例"来打分
    test_queries = [
        "非法集资", # 短文本
        "房东不退押金怎么办？", # 生活长句
        "故意伤害罪的量刑标准是什么，如果取得谅解能判缓刑吗？", # 复杂法律提问
        "死刑", # 短文本
        "离婚财产分割"
    ]
    
    # 记录三种模式的平均相关性分数或某种自定指标
    modes = ['Lexical (BM25)', 'Embedding (Qwen)', 'Hybrid (RRF+Rerank)']
    scores_matrix = {m: [] for m in modes}
    
    print("开始评测...")
    for query in test_queries:
        print(f"  查询: {query}")
        
        # 1. Lexical
        l_hits = service.search(query, mode="lexical", top_k=5)
        # BM25 分数不是 0-1，简单用命中数或归一化，这里为了演示，我们将第一个结果的分数缩放
        l_score = sum([min(h.score / 10.0, 1.0) for h in l_hits]) / 5.0 if l_hits else 0
        scores_matrix['Lexical (BM25)'].append(l_score)
        
        # 2. Embedding
        e_hits = service.search(query, mode="embedding", rerank=False, top_k=5)
        e_score = sum([h.score for h in e_hits]) / 5.0 if e_hits else 0
        scores_matrix['Embedding (Qwen)'].append(e_score)
        
        # 3. Hybrid
        h_hits = service.search(query, mode="hybrid", rerank=True, top_k=5)
        # sigmoid 后的重排分数
        h_score = sum([h.score for h in h_hits]) / 5.0 if h_hits else 0
        scores_matrix['Hybrid (RRF+Rerank)'].append(h_score)

    # 画图
    print("正在生成图表...")
    
    # 设置中文字体 (Mac 常见中文字体)
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    x = np.arange(len(test_queries))
    width = 0.25  # 柱子宽度
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rects1 = ax.bar(x - width, scores_matrix['Lexical (BM25)'], width, label='Lexical (BM25)', color='#60a5fa')
    rects2 = ax.bar(x, scores_matrix['Embedding (Qwen)'], width, label='Embedding (Qwen)', color='#fbbf24')
    rects3 = ax.bar(x + width, scores_matrix['Hybrid (RRF+Rerank)'], width, label='Hybrid (RRF+Rerank)', color='#34d399')
    
    ax.set_ylabel('Top-5 平均相关性得分 (归一化估计)')
    ax.set_title('三种 RAG 检索策略在不同类型 Query 下的表现对比')
    ax.set_xticks(x)
    
    # 截断长标题
    short_labels = [q[:8] + '...' if len(q) > 8 else q for q in test_queries]
    ax.set_xticklabels(short_labels)
    
    ax.legend()
    
    ax.bar_label(rects1, fmt='%.2f', padding=3)
    ax.bar_label(rects2, fmt='%.2f', padding=3)
    ax.bar_label(rects3, fmt='%.2f', padding=3)
    
    fig.tight_layout()
    
    output_file = PROJECT_ROOT / "retrieval_comparison_chart.png"
    plt.savefig(output_file, dpi=300)
    print(f"图表已保存至: {output_file}")

if __name__ == "__main__":
    main()
