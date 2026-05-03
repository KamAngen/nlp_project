import sys
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from rag_engine.service import KnowledgeService
from rich.console import Console
from rich.table import Table

def main():
    console = Console()
    
    # Initialize service
    service = KnowledgeService(
        question_bank_path=PROJECT_ROOT / "data/legal_study_agent/question_bank.jsonl",
        case_bank_path=PROJECT_ROOT / "data/legal_study_agent/case_bank.jsonl",
        common_knowledge_path=PROJECT_ROOT / "data/legal_study_agent/common_knowledge.jsonl"
    )
    
    # Load indices and reranker
    index_path = PROJECT_ROOT / "data/indices"
    emb_model_path = PROJECT_ROOT / "models/qwen/Qwen3_4B"
    reranker_path = PROJECT_ROOT / "models/reranker/bge-reranker-base"
    
    print(f"Loading Qwen3-4B embeddings...")
    service.load_indices(index_path, emb_model_path)
    print(f"Loading BGE reranker...")
    service.load_reranker(reranker_path)
    
    test_queries = [
        "关于非法集资的判例",
        "租赁合同纠纷怎么处理",
        "故意伤害罪的量刑标准"
    ]
    
    for query in test_queries:
        table = Table(title=f"RAG Evolution Comparison: {query}")
        table.add_column("Rank", justify="center")
        table.add_column("Lexical (Old)", style="cyan")
        table.add_column("Embedding (Qwen)", style="yellow")
        table.add_column("Reranked (BGE)", style="green bold")
        table.add_column("Rerank Score", justify="right")
        
        lexical_hits = service.search(query, mode="lexical", top_k=3)
        embedding_hits = service.search(query, mode="embedding", top_k=3)
        reranked_hits = service.search(query, mode="embedding", rerank=True, top_k=3)
        
        for i in range(3):
            l_hit = lexical_hits[i] if i < len(lexical_hits) else None
            e_hit = embedding_hits[i] if i < len(embedding_hits) else None
            r_hit = reranked_hits[i] if i < len(reranked_hits) else None
            
            table.add_row(
                str(i+1),
                l_hit.title if l_hit else "-",
                e_hit.title if e_hit else "-",
                r_hit.title if r_hit else "-",
                f"{r_hit.score:.4f}" if r_hit else "-"
            )
            
        console.print(table)

if __name__ == "__main__":
    main()
