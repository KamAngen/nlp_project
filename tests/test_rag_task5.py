import sys
from pathlib import Path
import pytest

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
    graph_path = PROJECT_ROOT / "data/indices/legal_graph.json"
    s.load_graph(graph_path)
    return s

def test_graph_relationships(service):
    assert service.graph is not None
    
    # Test case: Find statutes for a known case (auto-case-06cfc70db015bb48)
    case_id = "auto-case-06cfc70db015bb48"
    statutes = service.graph.get_related_statutes(case_id)
    print(f"Statutes for {case_id}: {statutes}")
    assert "民事诉讼法" in statutes
    
    # Test related cases via common statutes
    related_cases = service.graph.get_related_cases(case_id)
    print(f"Related cases for {case_id}: {related_cases}")
    assert len(related_cases) > 0

if __name__ == "__main__":
    pytest.main([__file__])
