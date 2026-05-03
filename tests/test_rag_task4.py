import sys
from pathlib import Path
import pytest
from collections import Counter

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
    return s

def test_curriculum_sampling_beginner(service):
    # Low mastery should favor easy/medium questions
    questions = service.sample_questions(question_count=20, user_mastery_level=0.1, random_seed=42)
    difficulties = Counter([q.difficulty for q in questions])
    print(f"Beginner difficulties: {difficulties}")
    # If easy is missing, it should favor medium over hard
    if "easy" in difficulties or "medium" in difficulties:
        # P(medium) should be significantly higher than its natural distribution if it's the easiest available
        assert difficulties.get("easy", 0) + difficulties.get("medium", 0) >= difficulties.get("hard", 0)

def test_curriculum_sampling_advanced(service):
    # High mastery should favor hard questions
    questions = service.sample_questions(question_count=20, user_mastery_level=0.9, random_seed=42)
    difficulties = Counter([q.difficulty for q in questions])
    print(f"Advanced difficulties: {difficulties}")
    # It should favor hard over easy
    assert difficulties.get("hard", 0) > difficulties.get("easy", 0)

if __name__ == "__main__":
    pytest.main([__file__])
