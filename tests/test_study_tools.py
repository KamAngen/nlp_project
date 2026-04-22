from pathlib import Path

from context_engine.manager import MemoryManager
from context_engine.store import DiskMemoryStore
from legal_agent.study_tools import StudyToolExecutor
from rag_engine.service import KnowledgeService


def _data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "legal_study_agent"


def test_study_tools_generate_score_exam_and_report(tmp_path: Path):
    data_root = _data_root()
    memory_manager = MemoryManager(DiskMemoryStore(tmp_path / "memory_store"), system_seed_path=data_root / "system_seed_memories.json")
    knowledge_service = KnowledgeService(
        question_bank_path=data_root / "question_bank.jsonl",
        case_bank_path=data_root / "case_bank.jsonl",
        common_knowledge_path=data_root / "common_knowledge.jsonl",
        use_legacy_statute_rag=False,
    )
    tools = StudyToolExecutor(memory_manager, knowledge_service, report_root=tmp_path / "reports")

    tools.execute("profile_upsert", {"updates": {"weak_points": ["民法"]}}, user_id="u1", session_id="s1")
    exam_payload = tools.execute(
        "generate_exam",
        {"topic": "民法", "question_count": 2},
        user_id="u1",
        session_id="s1",
    )
    answer_sheet = " ".join(
        f"{question['index']}.{question['answer']}"
        for question in exam_payload["questions"]
    )
    score_payload = tools.execute(
        "score_exam",
        {"answers_text": answer_sheet},
        user_id="u1",
        session_id="s1",
    )
    report_payload = tools.execute(
        "generate_report",
        {"report_type": "exam_feedback"},
        user_id="u1",
        session_id="s1",
    )

    assert exam_payload["question_count"] == 2
    assert score_payload["total_score"] == 40
    assert score_payload["score_percent"] == 100
    assert Path(report_payload["report_path"]).exists()


def test_study_tools_rag_search_hits_case_bank(tmp_path: Path):
    data_root = _data_root()
    memory_manager = MemoryManager(DiskMemoryStore(tmp_path / "memory_store"), system_seed_path=data_root / "system_seed_memories.json")
    knowledge_service = KnowledgeService(
        question_bank_path=data_root / "question_bank.jsonl",
        case_bank_path=data_root / "case_bank.jsonl",
        common_knowledge_path=data_root / "common_knowledge.jsonl",
        use_legacy_statute_rag=False,
    )
    tools = StudyToolExecutor(memory_manager, knowledge_service, report_root=tmp_path / "reports")

    payload = tools.execute(
        "rag_search",
        {"query": "租赁押金返还如何处理", "sources": ["case_bank"], "top_k": 3},
        user_id="u1",
        session_id="s1",
    )

    assert payload["results"]
    assert payload["results"][0]["source_type"] == "case_bank"


def test_study_tools_replay_wrong_questions_and_clear_bank_on_correct(tmp_path: Path):
    data_root = _data_root()
    memory_manager = MemoryManager(DiskMemoryStore(tmp_path / "memory_store"), system_seed_path=data_root / "system_seed_memories.json")
    knowledge_service = KnowledgeService(
        question_bank_path=data_root / "question_bank.jsonl",
        case_bank_path=data_root / "case_bank.jsonl",
        common_knowledge_path=data_root / "common_knowledge.jsonl",
        use_legacy_statute_rag=False,
    )
    tools = StudyToolExecutor(memory_manager, knowledge_service, report_root=tmp_path / "reports")

    first_exam = tools.execute(
        "generate_exam",
        {"topic": "行政法", "question_count": 1, "exam_type": "章节练习"},
        user_id="u1",
        session_id="s1",
    )
    first_question = first_exam["questions"][0]
    first_record_id = first_question["record_id"]
    first_correct_answer = str(first_question["answer"] or "").upper()
    wrong_answer = next(choice for choice in ("A", "B", "C", "D") if choice != first_correct_answer)
    wrong_score = tools.execute(
        "score_exam",
        {"answers_text": f"1.{wrong_answer}", "exam_session_id": "hallucinated-exam-id"},
        user_id="u1",
        session_id="s1",
    )

    profile_after_wrong = memory_manager.get_user_profile("u1")
    wrong_bank_after_wrong = dict(profile_after_wrong.attributes.get("wrong_question_bank") or {})

    assert wrong_score["exam_session_id"] == first_exam["exam_session_id"]
    assert first_record_id in wrong_bank_after_wrong
    assert "行政法" in profile_after_wrong.weak_points

    replay_exam = tools.execute(
        "generate_exam",
        {"topic": "行政法", "question_count": 1, "exam_type": "薄弱点强化"},
        user_id="u1",
        session_id="s1",
    )

    assert replay_exam["reused_wrong_question_count"] == 1
    assert replay_exam["questions"][0]["record_id"] == first_record_id

    tools.execute(
        "score_exam",
        {"answers_text": f"1.{replay_exam['questions'][0]['answer']}"},
        user_id="u1",
        session_id="s1",
    )
    report_payload = tools.execute(
        "generate_report",
        {"report_type": "mock_exam_review"},
        user_id="u1",
        session_id="s1",
    )

    profile_after_fix = memory_manager.get_user_profile("u1")
    wrong_bank_after_fix = dict(profile_after_fix.attributes.get("wrong_question_bank") or {})

    assert first_record_id not in wrong_bank_after_fix
    assert "行政法" in profile_after_fix.strong_points
    assert "错题库待复盘" in report_payload["report_markdown"]