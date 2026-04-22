from pathlib import Path

from context_engine.manager import MemoryManager
from context_engine.store import DiskMemoryStore
from legal_agent.utils.io import write_json


def test_memory_manager_layered_context_and_profile_updates(tmp_path: Path):
    seed_path = tmp_path / "system_seed.json"
    write_json(
        seed_path,
        [
            {
                "id": "system-1",
                "category": "policy",
                "text": "法考学习场景下优先返回结论和依据。",
                "importance": 0.95,
                "tags": ["reply_style"],
            }
        ],
    )
    store = DiskMemoryStore(tmp_path / "memory_store")
    manager = MemoryManager(store, system_seed_path=seed_path)

    manager.update_profile(
        "user_a",
        {
            "study_goals": ["民法"],
            "weak_points": ["行政法"],
            "preferences": {"daily_hours": 2},
        },
    )
    manager.record_turn(
        "user_a",
        "session_a",
        "我在备考民法，我的薄弱点是行政法。",
        "已记录你的画像，后续会优先围绕行政法复盘。",
    )

    bundle = manager.assemble_context("请根据行政法给我出题", "user_a", "session_a")

    assert bundle.user_profile.study_goals == ["民法"]
    assert bundle.user_profile.weak_points == ["行政法"]
    assert bundle.layer_hits["profile"]
    assert bundle.layer_hits["system"]
    assert bundle.layer_hits["working"]
    assert "行政法" in bundle.summary_blocks["profile"]


def test_memory_manager_exam_result_updates_weak_points(tmp_path: Path):
    store = DiskMemoryStore(tmp_path / "memory_store")
    manager = MemoryManager(store)
    manager.record_exam_session(
        "user_b",
        "session_b",
        {
            "exam_session_id": "exam-1",
            "topic": "行政法",
            "questions": [{"index": 1, "question": "示例题"}],
        },
    )
    manager.store_exam_result(
        "user_b",
        "session_b",
        {
            "exam_session_id": "exam-1",
            "topic": "行政法",
            "score_percent": 40,
            "weak_tags": ["听证程序", "行政处罚"],
        },
    )

    profile = manager.get_user_profile("user_b")

    assert "听证程序" in profile.weak_points
    assert "行政处罚" in profile.weak_points


def test_memory_manager_exam_result_updates_wrong_bank_and_strengths(tmp_path: Path):
    store = DiskMemoryStore(tmp_path / "memory_store")
    manager = MemoryManager(store)
    manager.record_exam_session(
        "user_c",
        "session_c",
        {
            "exam_session_id": "exam-1",
            "topic": "行政法",
            "exam_type": "薄弱点强化",
            "questions": [
                {"index": 1, "record_id": "q-a-001", "question": "示例题"},
            ],
        },
    )
    manager.store_exam_result(
        "user_c",
        "session_c",
        {
            "exam_session_id": "exam-1",
            "topic": "行政法",
            "exam_type": "薄弱点强化",
            "score_percent": 75,
            "weak_tags": ["听证程序"],
            "strong_tags": ["行政法"],
            "wrong_questions": [
                {
                    "record_id": "q-a-001",
                    "topic": "行政法",
                    "question": "示例题",
                    "tags": ["听证程序", "行政法"],
                }
            ],
            "corrected_question_ids": [],
        },
    )

    profile = manager.get_user_profile("user_c")
    snapshot = manager.build_report_snapshot("user_c", "session_c")

    assert "听证程序" in profile.weak_points
    assert "行政法" in profile.strong_points
    assert snapshot["wrong_question_bank_count"] == 1
    assert snapshot["wrong_question_bank_preview"][0]["record_id"] == "q-a-001"


def test_memory_manager_user_session_crud(tmp_path: Path):
    store = DiskMemoryStore(tmp_path / "memory_store")
    manager = MemoryManager(store)

    manager.create_user("alice", display_name="Alice")
    manager.ensure_session("alice", "session_1")
    manager.ensure_session("alice", "session_2")

    assert manager.list_users() == ["alice"]
    sessions = manager.list_sessions("alice")
    assert [item["session_id"] for item in sessions] == ["session_2", "session_1"]

    assert manager.delete_session("alice", "session_1") is True
    assert [item["session_id"] for item in manager.list_sessions("alice")] == ["session_2"]
    assert manager.delete_user("alice") is True
    assert manager.list_users() == []


def test_memory_manager_recovers_empty_session_state_file(tmp_path: Path):
    store = DiskMemoryStore(tmp_path / "memory_store")
    manager = MemoryManager(store)

    manager.create_user("alice", display_name="Alice")
    broken_path = store.session_path("alice", "broken_session")
    broken_path.parent.mkdir(parents=True, exist_ok=True)
    broken_path.write_text("", encoding="utf-8")

    sessions = manager.list_sessions("alice")

    assert [item["session_id"] for item in sessions] == ["broken_session"]
    recovered = store.load_session_state("alice", "broken_session")
    assert recovered.session_id == "broken_session"
    assert recovered.user_id == "alice"
    assert broken_path.read_text(encoding="utf-8").strip().startswith("{")
    backup_files = list(broken_path.parent.glob("broken_session.corrupt-*.json"))
    assert backup_files