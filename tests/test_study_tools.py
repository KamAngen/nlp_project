import json
from pathlib import Path

import pytest

from context_engine.manager import MemoryManager
from context_engine.store import DiskMemoryStore
from legal_agent.study_tools import StudyToolExecutor
from rag_engine.service import KnowledgeService


def _data_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "legal_study_agent"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _build_custom_tools(
    tmp_path: Path,
    question_rows: list[dict],
    *,
    subjective_exam_grader=None,
) -> StudyToolExecutor:
    data_root = _data_root()
    question_bank_path = tmp_path / "question_bank.jsonl"
    case_bank_path = tmp_path / "case_bank.jsonl"
    common_knowledge_path = tmp_path / "common_knowledge.jsonl"
    _write_jsonl(question_bank_path, question_rows)
    _write_jsonl(case_bank_path, [])
    _write_jsonl(common_knowledge_path, [])

    memory_manager = MemoryManager(DiskMemoryStore(tmp_path / "memory_store"), system_seed_path=data_root / "system_seed_memories.json")
    knowledge_service = KnowledgeService(
        question_bank_path=question_bank_path,
        case_bank_path=case_bank_path,
        common_knowledge_path=common_knowledge_path,
        use_legacy_statute_rag=False,
    )
    return StudyToolExecutor(
        memory_manager,
        knowledge_service,
        report_root=tmp_path / "reports",
        subjective_exam_grader=subjective_exam_grader,
    )


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


def test_study_tools_prepare_context_returns_planning_payload(tmp_path: Path):
    data_root = _data_root()
    memory_manager = MemoryManager(DiskMemoryStore(tmp_path / "memory_store"), system_seed_path=data_root / "system_seed_memories.json")
    knowledge_service = KnowledgeService(
        question_bank_path=data_root / "question_bank.jsonl",
        case_bank_path=data_root / "case_bank.jsonl",
        common_knowledge_path=data_root / "common_knowledge.jsonl",
        use_legacy_statute_rag=False,
    )
    tools = StudyToolExecutor(memory_manager, knowledge_service, report_root=tmp_path / "reports")

    tools.execute(
        "profile_upsert",
        {"updates": {"study_goals": ["行政法"], "weak_points": ["听证程序"]}},
        user_id="u1",
        session_id="s1",
    )
    memory_manager.record_turn("u1", "s1", "继续复习行政法听证程序。", "已继续整理重点。")

    payload = tools.execute("prepare_context", {"query": "继续给我出一道行政法题"}, user_id="u1", session_id="s1")

    assert "长期用户画像" in payload["planning_context"]
    assert payload["profile_hits"]
    assert payload["system_hits"]
    assert payload["working_hits"]
    assert payload["summary_blocks"]["long_term"]
    assert payload["summary_blocks"]["system"]
    assert payload["guaranteed_hits"]
    assert payload["related_hits"]
    assert payload["retrieval_meta"]["retrieval_strategy"].startswith("hybrid_")


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


def test_generate_exam_uses_source_level_choice_rows_and_scores_sheet(tmp_path: Path):
    tools = _build_custom_tools(
        tmp_path,
        [
            {
                "question_id": "m1",
                "topic": "民诉",
                "question": "申请诉前财产保全通常应满足什么条件？",
                "options": {
                    "A": "申请人应证明情况紧急，并依法提供担保。",
                    "B": "只能在终审判决生效后才能申请。",
                    "C": "无需说明紧急情况，也不需要担保。",
                    "D": "申请人应证明情况紧急，并依法提供担保。",
                },
                "answer": "D",
                "analysis": "诉前保全通常要求情况紧急，不立即保全会使合法权益受到难以弥补的损害，并且申请人依法提供担保。",
                "tags": ["民诉", "保全"],
                "score": 20,
            },
            {
                "question_id": "m2",
                "topic": "民诉",
                "question": "对一审民事判决不服时，当事人通常应如何救济？",
                "options": {
                    "A": "当然直接申请再审，不需要先上诉。",
                    "B": "应在法定期限内向上一级人民法院提起上诉。",
                    "C": "只能向公安机关报案处理。",
                    "D": "案件一经宣判就绝对不能救济。",
                },
                "answer": "B",
                "analysis": "对一审判决不服的，当事人通常应在法定期限内向上一级人民法院提起上诉。",
                "tags": ["民诉", "上诉"],
                "score": 20,
            },
            {
                "question_id": "m3",
                "topic": "民诉",
                "question": "人民法院采取证据保全措施的前提通常是什么？",
                "options": {
                    "A": "证据可能灭失或者以后难以取得。",
                    "B": "必须先由检察机关出具书面许可。",
                    "C": "只有进入执行阶段后才能申请。",
                    "D": "一旦立案就当然保全全部证据。",
                },
                "answer": "A",
                "analysis": "证据保全通常适用于证据可能灭失或者以后难以取得的情形。",
                "tags": ["民诉", "证据"],
                "score": 20,
            },
            {
                "question_id": "m4",
                "topic": "民诉",
                "question": "进入执行程序后，申请执行人首先需要关注什么？",
                "options": {
                    "A": "执行依据是否已经生效并具备执行内容。",
                    "B": "只要提交申请书，就不再审查执行依据。",
                    "C": "被执行人是否有财产与执行程序无关。",
                    "D": "申请执行前无需确认文书是否生效。",
                },
                "answer": "A",
                "analysis": "申请执行前应先确认执行依据已经生效且具备明确的给付内容。",
                "tags": ["民诉", "执行"],
                "score": 20,
            },
            {
                "question_id": "c1",
                "topic": "刑诉",
                "question": "侦查阶段讯问犯罪嫌疑人时应注意什么？",
                "options": {
                    "A": "可以当然排除辩护和证据规则的约束。",
                    "B": "应保障讯问程序合法并依法保障辩护权。",
                    "C": "只要涉嫌犯罪，程序阶段保障就不再重要。",
                    "D": "程序违法通常不影响案件处理，因此无需继续审查。",
                },
                "answer": "B",
                "analysis": "侦查阶段仍应遵守法定程序，并依法保障辩护权与证据规则。",
                "tags": ["刑诉", "侦查"],
                "score": 20,
            },
        ],
    )

    exam_payload = tools.execute(
        "generate_exam",
        {"topic": "民诉", "question_count": 3, "exam_type": "章节练习"},
        user_id="u1",
        session_id="s1",
    )

    assert exam_payload["question_count"] == 3
    assert all("一致性修复" not in note for note in exam_payload["selection_notes"])
    for question in exam_payload["questions"]:
        assert question["topic"] == "民诉"
        assert question["answer"] in question["options"]
        assert question["question_type"] == "single_choice"

    answer_sheet = " ".join(f"{question['index']}.{question['answer']}" for question in exam_payload["questions"])
    score_payload = tools.execute(
        "score_exam",
        {"answers_text": answer_sheet, "exam_session_id": exam_payload["exam_session_id"]},
        user_id="u1",
        session_id="s1",
    )

    assert score_payload["score_percent"] == 100
    assert score_payload["total_score"] == 60


def test_generate_exam_returns_fewer_questions_instead_of_cross_topic_padding(tmp_path: Path):
    tools = _build_custom_tools(
        tmp_path,
        [
            {
                "question_id": "m1",
                "topic": "民诉",
                "question": "申请诉前财产保全通常应满足什么条件？",
                "options": {
                    "A": "申请人应证明情况紧急，并依法提供担保。",
                    "B": "只能在败诉后申请。",
                    "C": "必须先经过行政机关批准。",
                    "D": "无需证明任何紧急性。",
                },
                "answer": "A",
                "analysis": "诉前保全通常要求情况紧急，并依法提供担保。",
                "tags": ["民诉", "保全"],
                "score": 20,
            },
            {
                "question_id": "c1",
                "topic": "刑诉",
                "question": "侦查阶段讯问犯罪嫌疑人时应注意什么？",
                "options": {
                    "A": "可以当然排除辩护和证据规则的约束。",
                    "B": "应保障讯问程序合法并依法保障辩护权。",
                    "C": "只要涉嫌犯罪，程序阶段保障就不再重要。",
                    "D": "程序违法通常不影响案件处理，因此无需继续审查。",
                },
                "answer": "B",
                "analysis": "侦查阶段仍应遵守法定程序，并依法保障辩护权与证据规则。",
                "tags": ["刑诉", "侦查"],
                "score": 20,
            },
        ],
    )

    exam_payload = tools.execute(
        "generate_exam",
        {"topic": "民诉", "question_count": 2, "exam_type": "章节练习"},
        user_id="u1",
        session_id="s1",
    )

    assert exam_payload["question_count"] == 1
    assert exam_payload["questions"][0]["topic"] == "民诉"
    assert any("未使用跨主题题目凑数" in note for note in exam_payload["selection_notes"])


def test_generate_exam_uses_source_topic_metadata_instead_of_option_and_analysis_keywords(tmp_path: Path):
    tools = _build_custom_tools(
        tmp_path,
        [
            {
                "question_id": "m1",
                "topic": "民法",
                "question": "申请执行的期限通常如何计算？",
                "options": {
                    "A": "根据《民事诉讼法》第二百四十六条规定，申请执行的期限通常为二年。",
                    "B": "一律不受期限限制。",
                    "C": "必须在判决当日提出。",
                    "D": "只能等法院通知后提出。",
                },
                "answer": "A",
                "analysis": "该题实际对应执行程序，常见依据是《民事诉讼法》关于申请执行期限的规定。",
                "tags": ["民法", "民事诉讼法", "执行"],
                "score": 20,
            },
            {
                "question_id": "a1",
                "topic": "行政法",
                "question": "某慈善组织未按要求备案即发布公开募捐信息，是否合法？",
                "options": {
                    "A": "根据《慈善法》相关规定，该行为不合法。",
                    "B": "只要作出行政决定，当事人原则上不能再主张任何程序权利。",
                    "C": "行政机关作出处分前无需再审查程序性保障是否到位。",
                    "D": "程序瑕疵不影响行政行为评价，因此无需讨论听证或告知。",
                },
                "answer": "A",
                "analysis": "该题属于行政法题，虽然解析里有“再审查程序性保障”字样，但不应被当作民诉题。",
                "tags": ["行政法", "慈善法", "政府"],
                "score": 20,
            },
        ],
    )

    with pytest.raises(ValueError, match="题库为空"):
        tools.execute(
            "generate_exam",
            {"topic": "民诉", "question_count": 2, "exam_type": "章节练习"},
            user_id="u1",
            session_id="s1",
        )


def test_generate_exam_supports_subjective_rows_and_partial_credit_scoring(tmp_path: Path):
    def fake_grader(question: dict[str, object], user_answer: str) -> dict[str, object]:
        assert str(question.get("question_type")) == "short_answer"
        assert "情况紧急" in user_answer
        return {
            "score": 12,
            "feedback": "抓住了紧急性，但遗漏了担保要件。",
            "matched_points": ["说明了情况紧急"],
            "missing_points": ["依法提供担保"],
        }

    tools = _build_custom_tools(
        tmp_path,
        [
            {
                "question_id": "sa-1",
                "topic": "民诉",
                "question": "申请诉前财产保全时，申请人通常需要说明哪些核心条件？",
                "question_type": "short_answer",
                "evaluation_mode": "llm_subjective",
                "reference_answer": "应说明情况紧急，不立即保全将导致合法权益受到难以弥补的损害，并依法提供担保。",
                "answer": "应说明情况紧急，不立即保全将导致合法权益受到难以弥补的损害，并依法提供担保。",
                "analysis": "诉前保全通常要求紧急性和担保两项核心条件。",
                "tags": ["民诉", "保全"],
                "score": 20,
            }
        ],
        subjective_exam_grader=fake_grader,
    )

    exam_payload = tools.execute(
        "generate_exam",
        {"topic": "民诉", "question_count": 1, "exam_type": "章节练习", "question_types": ["short_answer"]},
        user_id="u1",
        session_id="s1",
    )

    assert exam_payload["question_count"] == 1
    assert exam_payload["questions"][0]["question_type"] == "short_answer"
    assert exam_payload["questions"][0]["reference_answer"].startswith("应说明情况紧急")
    assert any("short_answer 1 题" in note for note in exam_payload["selection_notes"])

    score_payload = tools.execute(
        "score_exam",
        {"answers_text": "第1题：应当先说明情况紧急，否则损害会扩大。", "exam_session_id": exam_payload["exam_session_id"]},
        user_id="u1",
        session_id="s1",
    )

    assert score_payload["score_percent"] == 60.0
    assert score_payload["details"][0]["score"] == 12
    assert score_payload["details"][0]["user_answer"] == "应当先说明情况紧急，否则损害会扩大。"
    assert score_payload["details"][0]["grading_feedback"] == "抓住了紧急性，但遗漏了担保要件。"
    assert score_payload["details"][0]["classification"] == "review"
    assert score_payload["review_count"] == 1
    assert score_payload["incorrect_count"] == 0
    assert score_payload["wrong_questions"] == []
    assert score_payload["review_questions"][0]["missing_points"] == ["依法提供担保"]

    profile = tools.memory_manager.get_user_profile("u1")
    wrong_bank = dict(profile.attributes.get("wrong_question_bank") or {})

    assert "sa-1" not in wrong_bank
    assert "民诉" in profile.weak_points


def test_score_exam_accepts_single_subjective_answer_without_numbering(tmp_path: Path):
    def fake_grader(question: dict[str, object], user_answer: str) -> dict[str, object]:
        assert user_answer.startswith("应先说明情况紧急")
        return {
            "score": 16,
            "feedback": "主要结论到位，但论证展开还可以更完整。",
            "matched_points": ["提到情况紧急"],
            "missing_points": ["担保要求展开不足"],
        }

    tools = _build_custom_tools(
        tmp_path,
        [
            {
                "question_id": "sa-2",
                "topic": "民诉",
                "question": "申请诉前财产保全时，申请人通常需要说明哪些核心条件？",
                "question_type": "short_answer",
                "evaluation_mode": "llm_subjective",
                "reference_answer": "应说明情况紧急，不立即保全将导致合法权益受到难以弥补的损害，并依法提供担保。",
                "answer": "应说明情况紧急，不立即保全将导致合法权益受到难以弥补的损害，并依法提供担保。",
                "analysis": "诉前保全通常要求紧急性和担保两项核心条件。",
                "tags": ["民诉", "保全"],
                "score": 20,
            }
        ],
        subjective_exam_grader=fake_grader,
    )

    exam_payload = tools.execute(
        "generate_exam",
        {"topic": "民诉", "question_count": 1, "exam_type": "章节练习", "question_types": ["short_answer"]},
        user_id="u1",
        session_id="s1",
    )

    score_payload = tools.execute(
        "score_exam",
        {"answers_text": "应先说明情况紧急，否则可能来不及保全。", "exam_session_id": exam_payload["exam_session_id"]},
        user_id="u1",
        session_id="s1",
    )

    assert score_payload["unanswered_count"] == 0
    assert score_payload["details"][0]["user_answer"] == "应先说明情况紧急，否则可能来不及保全。"
    assert score_payload["details"][0]["classification"] == "mastered"


def test_score_exam_splits_inline_subjective_answers_across_multiple_questions(tmp_path: Path):
    seen_answers: list[tuple[int, str]] = []

    def fake_grader(question: dict[str, object], user_answer: str) -> dict[str, object]:
        seen_answers.append((int(question.get("index") or 0), user_answer))
        return {
            "score": 20,
            "feedback": "要点完整。",
            "matched_points": [user_answer],
            "missing_points": [],
            "quality_level": "mastered",
        }

    tools = _build_custom_tools(
        tmp_path,
        [
            {
                "question_id": "sa-inline-1",
                "topic": "民诉",
                "question": "申请诉前财产保全时，申请人通常需要说明哪些核心条件？",
                "question_type": "short_answer",
                "evaluation_mode": "llm_subjective",
                "reference_answer": "应说明情况紧急，并依法提供担保。",
                "answer": "应说明情况紧急，并依法提供担保。",
                "analysis": "诉前保全通常要求紧急性和担保。",
                "tags": ["民诉", "保全"],
                "score": 20,
            },
            {
                "question_id": "sa-inline-2",
                "topic": "民诉",
                "question": "人民法院采取证据保全措施的前提通常是什么？",
                "question_type": "short_answer",
                "evaluation_mode": "llm_subjective",
                "reference_answer": "证据可能灭失或者以后难以取得。",
                "answer": "证据可能灭失或者以后难以取得。",
                "analysis": "证据保全通常适用于证据可能灭失或者以后难以取得的情形。",
                "tags": ["民诉", "证据"],
                "score": 20,
            },
        ],
        subjective_exam_grader=fake_grader,
    )

    exam_payload = tools.execute(
        "generate_exam",
        {"topic": "民诉", "question_count": 2, "exam_type": "章节练习", "question_types": ["short_answer"]},
        user_id="u1",
        session_id="s1",
    )

    score_payload = tools.execute(
        "score_exam",
        {
            "answers_text": "1. 应当说明情况紧急，并依法提供担保。 2. 证据可能灭失或者以后难以取得。",
            "exam_session_id": exam_payload["exam_session_id"],
        },
        user_id="u1",
        session_id="s1",
    )

    assert score_payload["unanswered_count"] == 0
    assert seen_answers == [
        (1, "应当说明情况紧急，并依法提供担保。"),
        (2, "证据可能灭失或者以后难以取得。"),
    ]
    assert [detail["classification"] for detail in score_payload["details"]] == ["mastered", "mastered"]


def test_score_exam_treats_blank_subjective_answer_as_unanswered_without_calling_grader(tmp_path: Path):
    called = False

    def fake_grader(question: dict[str, object], user_answer: str) -> dict[str, object]:
        nonlocal called
        called = True
        return {
            "score": 20,
            "feedback": "不应调用。",
            "matched_points": [],
            "missing_points": [],
            "quality_level": "mastered",
        }

    tools = _build_custom_tools(
        tmp_path,
        [
            {
                "question_id": "sa-blank-1",
                "topic": "民诉",
                "question": "申请诉前财产保全时，申请人通常需要说明哪些核心条件？",
                "question_type": "short_answer",
                "evaluation_mode": "llm_subjective",
                "reference_answer": "应说明情况紧急，并依法提供担保。",
                "answer": "应说明情况紧急，并依法提供担保。",
                "analysis": "诉前保全通常要求紧急性和担保。",
                "tags": ["民诉", "保全"],
                "score": 20,
            }
        ],
        subjective_exam_grader=fake_grader,
    )

    exam_payload = tools.execute(
        "generate_exam",
        {"topic": "民诉", "question_count": 1, "exam_type": "章节练习", "question_types": ["short_answer"]},
        user_id="u1",
        session_id="s1",
    )

    score_payload = tools.execute(
        "score_exam",
        {"answers_text": "", "exam_session_id": exam_payload["exam_session_id"]},
        user_id="u1",
        session_id="s1",
    )

    assert called is False
    assert score_payload["unanswered_count"] == 1
    assert score_payload["details"][0]["classification"] == "unanswered"
    assert score_payload["details"][0]["score"] == 0


def test_generate_exam_filters_misclassified_case_rows_and_normalizes_fact_only_case_prompt(tmp_path: Path):
    tools = _build_custom_tools(
        tmp_path,
        [
            {
                "question_id": "bad-case-1",
                "topic": "行政法",
                "question": "什么是国家赔偿？",
                "question_type": "case_analysis",
                "evaluation_mode": "llm_subjective",
                "reference_answer": "国家赔偿是国家机关违法行使职权造成损害时依法承担的赔偿责任。",
                "answer": "国家赔偿是国家机关违法行使职权造成损害时依法承担的赔偿责任。",
                "analysis": "应先说明国家赔偿的定义，再说明赔偿义务机关。",
                "tags": ["行政法", "国家赔偿法"],
                "source_metadata": {"task_family": "legal_question_answering"},
                "score": 20,
            },
            {
                "question_id": "good-case-1",
                "topic": "刑法",
                "question": "二、2016年4月23日早上，被告人邹1某和伙同他人再次到果园砍伐果树，共毁坏果树100株。",
                "question_type": "case_analysis",
                "evaluation_mode": "llm_subjective",
                "reference_answer": "应结合毁坏数量、主观故意和共同犯罪情节分析是否构成故意毁坏财物罪。",
                "answer": "应结合毁坏数量、主观故意和共同犯罪情节分析是否构成故意毁坏财物罪。",
                "analysis": "需要围绕毁坏财物数额、行为方式和共同故意展开论证。",
                "tags": ["刑法", "案例分析题"],
                "source_metadata": {"task_family": "jud_read_compre"},
                "score": 20,
            },
        ],
    )

    exam_payload = tools.execute(
        "generate_exam",
        {"topic": "综合", "question_count": 2, "exam_type": "综合练习", "question_types": ["case_analysis"]},
        user_id="u1",
        session_id="s1",
    )

    assert exam_payload["question_count"] == 1
    assert exam_payload["questions"][0]["record_id"] == "good-case-1"
    assert exam_payload["questions"][0]["question"].startswith("请阅读以下案情，结合法律规定进行案例分析并作答：")
    assert "什么是国家赔偿？" not in exam_payload["questions"][0]["question"]