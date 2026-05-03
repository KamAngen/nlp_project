import sys
from pathlib import Path
import pytest
import torch

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from rag_engine.service import KnowledgeService
from rag_engine.schema import KnowledgeRecord

@pytest.fixture
def service():
    s = KnowledgeService(
        question_bank_path=PROJECT_ROOT / "data/legal_study_agent/question_bank.jsonl",
        case_bank_path=PROJECT_ROOT / "data/legal_study_agent/case_bank.jsonl",
        common_knowledge_path=PROJECT_ROOT / "data/legal_study_agent/common_knowledge.jsonl"
    )
    # Load the indices we built earlier
    index_path = PROJECT_ROOT / "data/indices"
    model_path = PROJECT_ROOT / "models/embeddings/bge-small-zh"
    s.load_indices(index_path, model_path)
    return s

def test_embedding_retrieval_logic(service):
    # Test 1: Search for a specific case title using embedding
    query = "租赁合同纠纷"
    hits = service.search(query, mode="embedding", top_k=5)
    
    assert len(hits) > 0
    # The score should be relatively high for a semantic match
    assert hits[0].score > 0.5
    
    # Verify that the results contain something related to 租赁 (Leasing)
    titles = [h.title for h in hits]
    assert any("租赁" in title for title in titles)

def test_persistence(service):
    # Verify that records were loaded correctly from the index
    assert "question_bank" in service.indices
    assert "case_bank" in service.indices
    assert len(service.indices["question_bank"].records) > 0

def test_mode_switching(service):
    query = "非法集资"
    lexical_hits = service.search(query, mode="lexical", top_k=5)
    embedding_hits = service.search(query, mode="embedding", top_k=5)
    
    # They should return different scores and potentially different orders
    assert lexical_hits[0].score != embedding_hits[0].score
    
if __name__ == "__main__":
    pytest.main([__file__])
