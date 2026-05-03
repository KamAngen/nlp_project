import sys
from pathlib import Path
import pytest
import torch

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from rag_engine.service import KnowledgeService

@pytest.fixture(scope="module")
def service():
    s = KnowledgeService(
        question_bank_path=PROJECT_ROOT / "data/legal_study_agent/question_bank.jsonl",
        case_bank_path=PROJECT_ROOT / "data/legal_study_agent/case_bank.jsonl",
        common_knowledge_path=PROJECT_ROOT / "data/legal_study_agent/common_knowledge.jsonl"
    )
    # Load the indices and reranker
    index_path = PROJECT_ROOT / "data/indices"
    emb_model_path = PROJECT_ROOT / "models/qwen/Qwen3_4B"
    reranker_path = PROJECT_ROOT / "models/reranker/bge-reranker-base"
    
    s.load_indices(index_path, emb_model_path)
    s.load_reranker(reranker_path)
    return s

def test_task1_embedding_retrieval(service):
    query = "非法集资的定义"
    hits = service.search(query, mode="embedding", top_k=5)
    assert len(hits) > 0
    assert hits[0].score > 0.5
    assert any("集资" in h.title for h in hits)

def test_task2_reranking(service):
    query = "租赁合同解除的条件"
    # Search without reranking
    base_hits = service.search(query, mode="embedding", rerank=False, top_k=5)
    # Search with reranking
    reranked_hits = service.search(query, mode="embedding", rerank=True, top_k=5)
    
    assert len(reranked_hits) == 5
    # The top hit might change after reranking
    # We at least check that scores are updated (Reranker scores are logits, usually different scale)
    assert base_hits[0].score != reranked_hits[0].score
    
    # Check that reranked hits have their scores from the reranker (which can be > 1 or < 0)
    # Most embedding scores are between 0 and 1 (cosine similarity)
    # We verify the logic doesn't crash
    print(f"Top 1 before rerank: {base_hits[0].title} ({base_hits[0].score:.4f})")
    print(f"Top 1 after rerank: {reranked_hits[0].title} ({reranked_hits[0].score:.4f})")

if __name__ == "__main__":
    pytest.main([__file__])
