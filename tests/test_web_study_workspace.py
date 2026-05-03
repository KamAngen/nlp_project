import inspect
from pathlib import Path

from legal_agent.study_agent import StudyAgentResponse
from legal_agent.web import unified_workspace


class _DummyAgent:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.users: dict[str, dict[str, object]] = {}
        self.last_generate_exam_kwargs: dict[str, object] | None = None

    def list_users(self) -> list[str]:
        return sorted(self.users)

    def create_user(self, user_id: str, *, display_name: str | None = None) -> dict[str, object]:
        self.users.setdefault(
            user_id,
            {
                "profile": {
                    "name": display_name or user_id,
                    "study_goals": [],
                    "weak_points": [],
                    "strong_points": [],
                },
                "sessions": {},
            },
        )
        return self.users[user_id]["profile"]

    def delete_user(self, user_id: str) -> bool:
        self.users.pop(user_id, None)
        return True

    def list_sessions(self, user_id: str) -> list[dict[str, object]]:
        sessions = self.users[user_id]["sessions"]
        return [{"session_id": session_id} for session_id in sorted(sessions)]

    def create_session(self, user_id: str, session_id: str) -> dict[str, object]:
        sessions = self.users[user_id]["sessions"]
        sessions.setdefault(session_id, {"turns": [], "summary": "", "last_report_path": None, "active_exam_session_id": None})
        return sessions[session_id]

    def delete_session(self, user_id: str, session_id: str) -> bool:
        self.users[user_id]["sessions"].pop(session_id, None)
        return True

    def get_profile(self, user_id: str) -> dict[str, object]:
        return self.users[user_id]["profile"]

    def get_session_state(self, user_id: str, session_id: str) -> dict[str, object]:
        session = dict(self.users[user_id]["sessions"][session_id])
        session.setdefault("turns", [])
        session.setdefault("summary", "")
        session.setdefault("last_report_path", None)
        session.setdefault("active_exam_session_id", None)
        return session

    def get_session_history(self, user_id: str, session_id: str):
        return list(self.users[user_id]["sessions"][session_id]["turns"])

    def handle_message(self, question: str, *, user_id: str, session_id: str, **kwargs) -> StudyAgentResponse:
        self.users[user_id]["sessions"][session_id]["turns"].append((kwargs.get("display_question") or question, "统一回答"))
        self.users[user_id]["sessions"][session_id]["summary"] = "最近一次是统一回答"
        return StudyAgentResponse(
            intent="legal_qa",
            answer="统一回答",
            plan={"planner_backend": "llm_react"},
            tool_results=[],
            trace="Thought: 已完成回答。",
        )

    def stream_message(self, question: str, *, user_id: str, session_id: str, **kwargs):
        yield {"event": "status", "message": "正在规划下一步。", "trace": "Thought: 正在规划。"}
        yield {"event": "final", "response": self.handle_message(question, user_id=user_id, session_id=session_id, **kwargs)}

    def generate_exam(self, *, user_id: str, session_id: str, topic: str | None = None, question_count: int | None = None, **kwargs) -> StudyAgentResponse:
        self.last_generate_exam_kwargs = dict(kwargs)
        self.users[user_id]["sessions"][session_id]["turns"].append((f"[UI操作] 生成模拟测试 topic={topic} question_count={question_count}", "已生成模拟测试"))
        self.users[user_id]["sessions"][session_id]["active_exam_session_id"] = "exam-1"
        return StudyAgentResponse(
            intent="mock_exam_generate",
            answer="已生成模拟测试",
            plan={},
            tool_results=[],
            trace="Thought: 已生成模拟测试。",
        )

    def generate_report_response(self, *, user_id: str, session_id: str, report_type: str = "study_progress", **kwargs) -> StudyAgentResponse:
        report_path = self.tmp_path / f"{user_id}_{session_id}_{report_type}.md"
        report_path.write_text("# 学习报告\n\n- 当前进度稳定。", encoding="utf-8")
        self.users[user_id]["sessions"][session_id]["turns"].append((f"[UI操作] 生成学习报告 report_type={report_type}", "已生成学习报告"))
        self.users[user_id]["sessions"][session_id]["last_report_path"] = str(report_path)
        return StudyAgentResponse(
            intent="report_generation",
            answer="已生成学习报告",
            plan={},
            tool_results=[],
            report_path=str(report_path),
            report_markdown="# 学习报告\n\n- 当前进度稳定。",
            trace="Thought: 已生成学习报告。",
        )


class _DeferredHistoryAgent(_DummyAgent):
    def handle_message(self, question: str, *, user_id: str, session_id: str, **kwargs) -> StudyAgentResponse:
        return StudyAgentResponse(
            intent="legal_qa",
            answer="这是延迟持久化时也应立即显示的最终答复。",
            plan={"planner_backend": "llm_react"},
            tool_results=[],
            trace="Final Answer: 这是延迟持久化时也应立即显示的最终答复。",
        )

    def get_session_history(self, user_id: str, session_id: str):
        return []


def test_workspace_user_and_session_crud(tmp_path: Path, monkeypatch):
    dummy_agent = _DummyAgent(tmp_path)
    monkeypatch.setattr(unified_workspace, "_get_agent", lambda *args, **kwargs: dummy_agent)

    payload = unified_workspace._create_user_action("张三", None, "configs/defaults.yaml", "configs/study_agent.yaml", "auto", "cpu")
    state = payload[2]
    assert state["user_id"] == "张三"
    assert state["session_id"] == "default_session"
    assert payload[5].value == "张三"

    session_payload = unified_workspace._create_session_action("民法冲刺", state, "configs/defaults.yaml", "configs/study_agent.yaml", "auto", "cpu")
    assert session_payload[2]["session_id"] == "民法冲刺"
    assert session_payload[6].value == "民法冲刺"

    delete_payload = unified_workspace._delete_session_action(session_payload[2], "configs/defaults.yaml", "configs/study_agent.yaml", "auto", "cpu")
    assert delete_payload[2]["session_id"] == "default_session"


def test_workspace_chat_exam_and_report_actions(tmp_path: Path, monkeypatch):
    dummy_agent = _DummyAgent(tmp_path)
    dummy_agent.create_user("李四", display_name="李四")
    dummy_agent.create_session("李四", "default_session")
    monkeypatch.setattr(unified_workspace, "_get_agent", lambda *args, **kwargs: dummy_agent)
    monkeypatch.setattr(unified_workspace, "_resolve_model_choice", lambda *args, **kwargs: ("models/qwen/Qwen3_4B", None))

    state = {"user_id": "李四", "session_id": "default_session", "trace": "", "report_markdown": "", "report_path": None, "pending_root_question": None, "pending_question": None, "clarification_answers": []}

    chat_updates = list(
        unified_workspace._submit_chat(
        "押金到期不退怎么办？",
        state,
        "models/qwen/Qwen3_4B",
        "configs/defaults.yaml",
        "configs/study_agent.yaml",
        "auto",
        "cpu",
    )
    )
    assert chat_updates[0][0][-1] == {"role": "assistant", "content": "已接收问题，正在规划下一步。"}

    chat_payload = chat_updates[-1]
    assert chat_payload[0][-1] == {"role": "assistant", "content": "统一回答"}
    assert "Thought" in chat_payload[1]

    exam_payload = unified_workspace._run_exam_action(
        "章节练习",
        "行政法",
        2,
        chat_payload[2],
        "models/qwen/Qwen3_4B",
        "configs/defaults.yaml",
        "configs/study_agent.yaml",
        "auto",
        "cpu",
    )
    assert any(msg["content"] == "已生成模拟测试" for msg in exam_payload[0] if msg["role"] == "assistant")
    assert dummy_agent.last_generate_exam_kwargs is not None
    assert dummy_agent.last_generate_exam_kwargs["question_types"] == ["single_choice", "short_answer", "case_analysis"]

    subjective_exam_payload = unified_workspace._run_exam_action(
        "章节练习",
        "行政法",
        2,
        chat_payload[2],
        "models/qwen/Qwen3_4B",
        "configs/defaults.yaml",
        "configs/study_agent.yaml",
        "auto",
        "cpu",
        "简答题",
    )
    assert any(msg["content"] == "已生成模拟测试" for msg in subjective_exam_payload[0] if msg["role"] == "assistant")
    assert dummy_agent.last_generate_exam_kwargs is not None
    assert dummy_agent.last_generate_exam_kwargs["question_types"] == ["short_answer"]

    report_payload = unified_workspace._run_report_action(
        "学习进度报告",
        exam_payload[2],
        "models/qwen/Qwen3_4B",
        "configs/defaults.yaml",
        "configs/study_agent.yaml",
        "auto",
        "cpu",
    )
    assert report_payload[3].startswith("# 学习报告")
    assert report_payload[4] is not None


def test_workspace_submit_chat_renders_final_message_without_waiting_for_history_refresh(tmp_path: Path, monkeypatch):
    dummy_agent = _DeferredHistoryAgent(tmp_path)
    dummy_agent.create_user("赵六", display_name="赵六")
    dummy_agent.create_session("赵六", "default_session")
    monkeypatch.setattr(unified_workspace, "_get_agent", lambda *args, **kwargs: dummy_agent)
    monkeypatch.setattr(unified_workspace, "_resolve_model_choice", lambda *args, **kwargs: ("models/qwen/Qwen3_4B", None))

    state = {"user_id": "赵六", "session_id": "default_session", "trace": "", "report_markdown": "", "report_path": None, "pending_root_question": None, "pending_question": None, "clarification_answers": []}

    updates = list(
        unified_workspace._submit_chat(
            "请解释听证程序适用情形",
            state,
            "models/qwen/Qwen3_4B",
            "configs/defaults.yaml",
            "configs/study_agent.yaml",
            "auto",
            "cpu",
        )
    )

    final_payload = updates[-1]
    assert final_payload[0][-1] == {"role": "assistant", "content": "这是延迟持久化时也应立即显示的最终答复。"}
    assert final_payload[2]["last_assistant_message"] == "这是延迟持久化时也应立即显示的最终答复。"


def test_workspace_ignores_missing_report_file(tmp_path: Path, monkeypatch):
    dummy_agent = _DummyAgent(tmp_path)
    dummy_agent.create_user("王五", display_name="王五")
    dummy_agent.create_session("王五", "default_session")
    dummy_agent.users["王五"]["sessions"]["default_session"]["last_report_path"] = str(tmp_path / "missing-report.md")
    monkeypatch.setattr(unified_workspace, "_get_agent", lambda *args, **kwargs: dummy_agent)

    payload = unified_workspace._refresh_workspace_data(
        {"user_id": "王五", "session_id": "default_session"},
        config_path="configs/defaults.yaml",
        study_config_path="configs/study_agent.yaml",
        runtime_device="auto",
        default_retrieval_device="cpu",
    )

    assert payload["report_path"] is None
    assert payload["report_display_markdown"].startswith("### 报告面板")
    assert "薄弱点" not in payload["status_markdown"]


def test_runtime_device_choices_hide_unavailable_cuda(monkeypatch):
    class _CudaProbe:
        @staticmethod
        def is_available() -> bool:
            return False

    class _TorchProbe:
        cuda = _CudaProbe()

    monkeypatch.setattr(unified_workspace, "torch", _TorchProbe())

    choices, default_value = unified_workspace._build_runtime_device_choices("configs/defaults.yaml", "auto")

    assert choices == ["auto", "cpu"]
    assert default_value == "auto"


def test_workspace_columns_keep_ratio_layout_hooks():
    source = inspect.getsource(unified_workspace.build_unified_workspace)

    assert source.count("min_width=1") >= 2
    assert 'elem_id="config-device-row"' in source
    assert 'elem_id="user-session-row"' in source
    assert 'elem_id="new-entity-row"' in source
