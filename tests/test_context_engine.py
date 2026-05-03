from pathlib import Path
from types import SimpleNamespace

from context_engine.manager import MemoryManager
from context_engine.reasoner import CompressionDraft, MemoryDraft, QwenMemoryReasoner, TurnAnalysis
from context_engine.schemas import ConversationTurn, SessionState, UserProfile
from context_engine.store import DiskMemoryStore
from legal_agent.utils.io import read_json, read_jsonl, write_json


class _SemanticReasonerStub:
    def analyze_turn(self, turn, *, user_profile, session_state):
        return TurnAnalysis(
            summary="用户补充了稳定学习目标和回答偏好，并要求系统长期按该策略工作。",
            reasoning_digest="需要把稳定画像写入 profile，把代理级约束写入 system memory。",
            long_term_memories=[
                MemoryDraft(layer="long_term", category="user_goal", text="用户当前备考目标：行政法", importance=0.96, tags=["行政法"], decay_enabled=False),
                MemoryDraft(layer="long_term", category="user_preference", text="用户偏好 answer_style：concise", importance=0.9, tags=["answer_style", "concise"], decay_enabled=False),
            ],
            system_memories=[
                MemoryDraft(layer="system", category="memory_policy", text="当用户用自然表达透露稳定目标或偏好时，应按语义理解写入记忆，而不是等待固定句式。", importance=0.97, tags=["memory", "llm_reasoning"], decay_enabled=False),
            ],
            profile_updates={"study_goals": ["行政法"], "preferences": {"answer_style": "concise"}},
            tags=["行政法", "偏好"],
            importance=0.9,
        )

    def compress_history(self, *, turns, prior_summaries=None):
        return CompressionDraft(summary="压缩摘要", salient_points=["保持上下文"], importance=0.84)


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


def test_memory_manager_persists_trace_and_exposes_prepare_payload(tmp_path: Path):
    store = DiskMemoryStore(tmp_path / "memory_store")
    manager = MemoryManager(store, reasoner=_SemanticReasonerStub())

    manager.record_turn(
        "u1",
        "s1",
        "最近主要在啃行政法，之后答题时尽量短一点，结论先行。",
        "已记录你的长期偏好，后续我会尽量简洁回答。",
        reasoning_trace="Thought: 识别出稳定学习目标与回答风格偏好。",
    )

    turns = store.load_session_turns("u1", "s1")
    memories = store.load_user_memories("u1")
    memory_edges = read_jsonl(store.memory_edges_path("u1"))
    payload = manager.prepare_turn_context_payload("继续给我出一道行政法题", "u1", "s1")

    assert len(turns) == 1
    assert "Thought:" in turns[0].reasoning_trace
    assert turns[0].reasoning_summary
    assert any(item.layer == "long_term" and "行政法" in item.text for item in memories)
    assert any(item.layer == "long_term" and "answer_style" in item.text for item in memories)
    assert store.profile_path("u1").exists()
    assert store.session_path("u1", "s1").exists()
    assert store.memory_edges_path("u1").exists()
    assert store.system_memories_path().exists()
    assert memory_edges
    assert payload["profile_hits"]
    assert payload["system_hits"]
    assert payload["working_hits"]
    assert payload["long_term_hits"]
    assert payload["guaranteed_hits"]
    assert "长期用户画像" in payload["planning_context"]
    assert payload["summary_blocks"]["long_term"]
    assert payload["summary_blocks"]["system"]
    assert payload["retrieval_meta"]["graph_edge_count"] >= 1
    assert payload["retrieval_meta"]["related_hit_count"] >= len(payload["guaranteed_hits"])


def test_memory_manager_persists_profile_session_and_system_files(tmp_path: Path):
    store = DiskMemoryStore(tmp_path / "memory_store")
    manager = MemoryManager(store, reasoner=_SemanticReasonerStub())

    manager.record_turn(
        "observer",
        "memory_demo",
        "这阶段我主要复习行政法，后面答复尽量精简一些。",
        "收到，我会按更精简的方式继续辅导。",
    )

    profile_payload = read_json(store.profile_path("observer"))
    session_payload = read_json(store.session_path("observer", "memory_demo"))
    system_payload = read_jsonl(store.system_memories_path())

    assert profile_payload["study_goals"] == ["行政法"]
    assert profile_payload["preferences"]["answer_style"] == "concise"
    assert session_payload["turns"][0]["user_message"].startswith("这阶段我主要复习行政法")
    assert any("按语义理解写入记忆" in row["text"] for row in system_payload)


def test_memory_manager_compresses_history_without_losing_raw_turns(tmp_path: Path):
    store = DiskMemoryStore(tmp_path / "memory_store")
    manager = MemoryManager(
        store,
        compression_after_turns=6,
        compression_chunk_size=4,
        retain_recent_turns=2,
    )

    for index in range(7):
        manager.record_turn(
            "u2",
            "s2",
            f"第 {index + 1} 轮：继续围绕行政法听证程序复盘。",
            f"第 {index + 1} 轮已完成总结。",
            reasoning_trace=f"Thought: 第 {index + 1} 轮聚焦听证程序。",
        )

    payload = manager.prepare_turn_context_payload("继续行政法复盘", "u2", "s2")
    session_state = manager.get_session_state("u2", "s2")
    turns = store.load_session_turns("u2", "s2")
    memories = store.load_user_memories("u2")

    assert payload["maintenance"]["compressed"] is True
    assert session_state.compression_count >= 1
    assert len(turns) == 7
    assert any(item.layer == "summary" and item.category == "session_compression" for item in memories)


def test_qwen_reasoner_drops_profile_updates_that_only_echo_existing_profile():
    class _StubModel:
        def generate(self, messages, **kwargs):
            return SimpleNamespace(
                raw_text=(
                    '{'
                    '"summary": "用户只提交了答题卡，没有新增长期画像。",'
                    '"reasoning_digest": "不应把旧画像回写为新增画像。",'
                    '"profile_updates": {'
                    '"study_goals": ["行政法"],'
                    '"preferences": {"response_length": "短"}'
                    '},'
                    '"importance": 0.4'
                    '}'
                ),
                content="",
                reasoning="",
            )

    reasoner = QwenMemoryReasoner(_StubModel())
    analysis = reasoner.analyze_turn(
        ConversationTurn(user_message="1.A", assistant_message=""),
        user_profile=UserProfile(user_id="u1", study_goals=["行政法"], preferences={"response_length": "短"}),
        session_state=SessionState(session_id="s1", user_id="u1"),
    )

    assert analysis.profile_updates == {}