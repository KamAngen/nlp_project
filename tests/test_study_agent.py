from pathlib import Path
from types import SimpleNamespace

from legal_agent.study_agent import LegalStudyAgent, StudyAgentResponse
from legal_agent.study_config import StudyAgentConfig


def _build_config(tmp_path: Path) -> StudyAgentConfig:
    data_root = Path(__file__).resolve().parents[1] / "data" / "legal_study_agent"
    return StudyAgentConfig(
        project_root=Path(__file__).resolve().parents[1],
        memory_root=tmp_path / "memory_store",
        report_root=tmp_path / "reports",
        question_bank_path=data_root / "question_bank.jsonl",
        case_bank_path=data_root / "case_bank.jsonl",
        common_knowledge_path=data_root / "common_knowledge.jsonl",
        system_memory_path=data_root / "system_seed_memories.json",
        study_manifest_path=tmp_path / "study_manifest.json",
        use_legacy_statute_rag=False,
        legacy_config_path=None,
        retrieval_top_k=6,
        default_exam_question_count=2,
        planner_backend="llm_react",
        turn_analysis_mode="heuristic",
    )


class _DummyEngine:
    def __init__(self, agent: LegalStudyAgent, *, user_id: str, session_id: str) -> None:
        self.agent = agent
        self.user_id = user_id
        self.session_id = session_id

    def run(self, question: str, history=None):
        if "记住" in question or "备考" in question:
            upsert = self.agent.tool_executor.execute(
                "profile_upsert",
                {"raw_text": question, "updates": {}},
                user_id=self.user_id,
                session_id=self.session_id,
            )
            profile = self.agent.tool_executor.execute(
                "profile_view",
                {},
                user_id=self.user_id,
                session_id=self.session_id,
            )
            return SimpleNamespace(
                final_answer="已更新你的学习档案。",
                trace="Thought: 已写回画像。",
                tool_history=[
                    {"tool_name": "profile_upsert", "reason": "更新画像", "arguments": {"raw_text": question}, "result": upsert},
                    {"tool_name": "profile_view", "reason": "确认画像", "arguments": {}, "result": profile},
                ],
                errors=[],
                needs_user_input=False,
                clarification_question=None,
            )

        if "答题卡" in question or "score_exam" in question:
            score = self.agent.tool_executor.execute(
                "score_exam",
                {"answers_text": "1.C"},
                user_id=self.user_id,
                session_id=self.session_id,
            )
            report = self.agent.tool_executor.execute(
                "generate_report",
                {"report_type": "mock_exam_review"},
                user_id=self.user_id,
                session_id=self.session_id,
            )
            return SimpleNamespace(
                final_answer="本次测试得分为 100 分，错题复盘已同步到学习报告。",
                trace="Thought: 已完成评分与报告。",
                tool_history=[
                    {"tool_name": "score_exam", "reason": "评分", "arguments": {"answers_text": "1.C"}, "result": score},
                    {"tool_name": "generate_report", "reason": "生成报告", "arguments": {"report_type": "mock_exam_review"}, "result": report},
                ],
                errors=[],
                needs_user_input=False,
                clarification_question=None,
            )

        if "生成一套" in question or "调用 generate_exam" in question or "模拟测试" in question:
            profile = self.agent.tool_executor.execute("profile_view", {}, user_id=self.user_id, session_id=self.session_id)
            memory = self.agent.tool_executor.execute(
                "memory_search",
                {"query": "行政法", "top_k": 6},
                user_id=self.user_id,
                session_id=self.session_id,
            )
            exam = self.agent.tool_executor.execute(
                "generate_exam",
                {"topic": "行政法", "question_count": 1},
                user_id=self.user_id,
                session_id=self.session_id,
            )
            tool_history = [
                {"tool_name": "profile_view", "reason": "读取画像", "arguments": {}, "result": profile},
                {"tool_name": "memory_search", "reason": "读取记忆", "arguments": {"query": "行政法", "top_k": 6}, "result": memory},
                {"tool_name": "generate_exam", "reason": "生成模拟题", "arguments": {"topic": "行政法", "question_count": 1}, "result": exam},
            ]
            return SimpleNamespace(
                final_answer=self.agent._render_exam_answer(tool_history),
                trace="Thought: 已生成模拟测试。",
                tool_history=tool_history,
                errors=[],
                needs_user_input=False,
                clarification_question=None,
            )

        if "学习报告" in question:
            report = self.agent.tool_executor.execute(
                "generate_report",
                {"report_type": "study_progress"},
                user_id=self.user_id,
                session_id=self.session_id,
            )
            return SimpleNamespace(
                final_answer="已生成学习报告，内容已同步到右侧面板。",
                trace="Thought: 已生成学习报告。",
                tool_history=[
                    {"tool_name": "generate_report", "reason": "生成报告", "arguments": {"report_type": "study_progress"}, "result": report},
                ],
                errors=[],
                needs_user_input=False,
                clarification_question=None,
            )

        rag = self.agent.tool_executor.execute(
            "rag_search",
            {"query": question, "top_k": 2},
            user_id=self.user_id,
            session_id=self.session_id,
        )
        return SimpleNamespace(
            final_answer="以下回答用于法考学习与知识梳理。",
            trace="Thought: 已检索学习资料。",
            tool_history=[
                {"tool_name": "rag_search", "reason": "检索学习知识", "arguments": {"query": question, "top_k": 2}, "result": rag},
            ],
            errors=[],
            needs_user_input=False,
            clarification_question=None,
        )


def test_study_agent_unified_workflow(tmp_path: Path, monkeypatch):
    agent = LegalStudyAgent(_build_config(tmp_path))
    monkeypatch.setattr(
        agent,
        "_build_engine",
        lambda **kwargs: _DummyEngine(agent, user_id=kwargs["user_id"], session_id=kwargs["session_id"]),
    )

    update_result = agent.handle_message("记住，我在备考民法，我的薄弱点是行政法。", user_id="demo_user", session_id="demo_session")
    exam_result = agent.handle_message("给我来一套行政法 1 题模拟测试", user_id="demo_user", session_id="demo_session")
    score_result = agent.handle_message("我的答案是 1.C", user_id="demo_user", session_id="demo_session")
    qa_result = agent.handle_message("押金到期不退怎么办？", user_id="demo_user", session_id="demo_session")
    report_result = agent.handle_message("请生成我的学习报告", user_id="demo_user", session_id="demo_session")

    assert update_result.intent == "profile_update"
    assert "已更新" in update_result.answer
    assert exam_result.intent == "mock_exam_generate"
    assert "请按“1.A 2.B 3.C”这样的格式" in exam_result.answer
    assert score_result.intent == "mock_exam_score"
    assert "本次测试得分为" in score_result.answer
    assert score_result.report_path is not None
    assert qa_result.intent == "legal_qa"
    assert "法考学习" in qa_result.answer
    assert report_result.intent == "report_generation"
    assert report_result.report_path is not None
    assert Path(report_result.report_path).exists()


def test_study_agent_button_only_actions_are_blocked_in_chat(tmp_path: Path):
    agent = LegalStudyAgent(_build_config(tmp_path))
    response = agent.handle_message(
        "请生成我的学习报告",
        user_id="demo_user",
        session_id="demo_session",
        allow_button_only_intents=False,
    )

    assert response.intent == "ui_button_only"
    assert "按钮专用入口" in response.answer


def test_study_agent_button_actions_fallback_to_direct_tools(tmp_path: Path, monkeypatch):
    agent = LegalStudyAgent(_build_config(tmp_path))

    monkeypatch.setattr(
        agent,
        "handle_message",
        lambda *args, **kwargs: StudyAgentResponse(
            intent="legal_qa",
            answer="未触发目标工具。",
            plan={},
            tool_results=[],
        ),
    )

    agent.memory_manager.create_user("demo_user", display_name="演示用户")
    agent.memory_manager.ensure_session("demo_user", "demo_session")
    agent.memory_manager.update_profile("demo_user", {"study_goals": ["行政法"], "weak_points": ["听证程序"]})

    exam_response = agent.generate_exam(user_id="demo_user", session_id="demo_session", topic="行政法", question_count=2)
    report_response = agent.generate_report_response(user_id="demo_user", session_id="demo_session", report_type="study_progress")

    session_state = agent.memory_manager.get_session_state("demo_user", "demo_session")

    assert exam_response.intent == "mock_exam_generate"
    assert "本次选题优先参考的画像标签" in exam_response.answer
    assert "答案：" not in exam_response.answer
    assert "分析：" not in exam_response.answer
    assert "Action: generate_exam" in exam_response.trace
    assert '"answer"' in exam_response.trace
    assert '"analysis"' in exam_response.trace
    assert report_response.intent == "report_generation"
    assert report_response.report_path is not None
    assert Path(report_response.report_path).exists()
    assert any(turn.user_message.startswith("[UI操作] 生成模拟测试") for turn in session_state.turns)


def test_generate_exam_button_uses_direct_tool_trace(tmp_path: Path):
    agent = LegalStudyAgent(_build_config(tmp_path))

    response = agent.generate_exam(user_id="demo_user", session_id="demo_session", topic="行政法", question_count=2)

    assert response.intent == "mock_exam_generate"
    assert response.tool_results[-1]["tool_name"] == "generate_exam"
    assert "以下是为当前用户生成的法考模拟测试题目：" in response.answer
    assert "答案：" not in response.answer
    assert "分析：" not in response.answer
    assert response.trace.count("Final Answer:") == 1
    assert "答案：C" not in response.trace
    assert "分析：这是一段不该展示给用户的解析。" not in response.trace
    assert '"answer"' in response.trace
    assert '"analysis"' in response.trace


def test_study_agent_falls_back_to_direct_rag_when_model_skips_tools(tmp_path: Path, monkeypatch):
    agent = LegalStudyAgent(_build_config(tmp_path))

    class _NoToolEngine:
        def run(self, question: str, history=None):
            return SimpleNamespace(
                final_answer="直接裸答，没有证据。",
                trace="Final Answer: 直接裸答，没有证据。",
                tool_history=[],
                errors=[],
                needs_user_input=False,
                clarification_question=None,
            )

    monkeypatch.setattr(agent, "_build_engine", lambda **kwargs: _NoToolEngine())

    response = agent.handle_message("押金到期不退怎么办？", user_id="demo_user", session_id="demo_session")

    assert response.intent == "legal_qa"
    assert response.tool_results
    assert response.tool_results[0]["tool_name"] == "memory_search"
    assert response.tool_results[1]["tool_name"] == "rag_search"
    assert "[statute]" not in response.answer
    assert "[question_bank]" not in response.answer
    assert response.trace.count("Final Answer:") == 1
    assert response.trace.endswith(response.answer)


def test_study_agent_preserves_smalltalk_when_direct_rag_has_no_strong_hits(tmp_path: Path, monkeypatch):
    agent = LegalStudyAgent(_build_config(tmp_path))

    class _NoToolEngine:
        def run(self, question: str, history=None):
            return SimpleNamespace(
                final_answer="你好，我是你的法考学习助手。",
                trace="Final Answer: 你好，我是你的法考学习助手。",
                tool_history=[],
                errors=[],
                needs_user_input=False,
                clarification_question=None,
            )

    monkeypatch.setattr(agent, "_build_engine", lambda **kwargs: _NoToolEngine())

    response = agent.handle_message("你好", user_id="demo_user", session_id="demo_session")

    assert response.intent == "legal_qa"
    assert [entry["tool_name"] for entry in response.tool_results[:2]] == ["memory_search", "rag_search"]
    assert response.answer == "你好，我是你的法考学习助手。"
    assert response.trace.endswith(response.answer)


def test_direct_qa_filters_irrelevant_statute_results(tmp_path: Path):
    agent = LegalStudyAgent(_build_config(tmp_path))

    filtered = agent._select_relevant_statute_results(
        "押金到期不退怎么办？",
        {
            "results": [
                {
                    "document_title": "中华人民共和国刑法",
                    "article_heading": "第二百六十三条",
                    "text": "以暴力、胁迫或者其他方法抢劫公私财物的，处三年以上十年以下有期徒刑。",
                    "score": 0.021,
                }
            ]
        },
    )

    assert filtered == []


def test_direct_qa_filters_irrelevant_knowledge_hits(tmp_path: Path):
    agent = LegalStudyAgent(_build_config(tmp_path))

    filtered = agent._select_relevant_knowledge_hits(
        "押金到期不退怎么办？",
        [
            {
                "source_type": "question_bank",
                "record_id": "q-m-001",
                "title": "甲将房屋出租给乙，租赁期满后乙已经腾退，但甲无故不退押金。下列哪一项最符合民法上的处理思路？",
                "excerpt": "若不存在应抵扣的租金、违约金或损失，出租人应返还押金。",
                "score": 0.7667,
                "metadata": {"answer": "B", "analysis": "若不存在可扣减项目，应返还押金。"},
            },
            {
                "source_type": "question_bank",
                "record_id": "q-c-001",
                "title": "驾驶人交通肇事后为逃避法律追究而逃逸，致被害人因得不到救助而死亡，通常应重点考虑哪一类罪责评价？",
                "excerpt": "重点考察交通肇事罪及逃逸致人死亡情节。",
                "score": 0.4333,
                "metadata": {"answer": "B", "analysis": "法考语境下应识别逃逸加重情节。"},
            },
        ],
    )

    assert [item["record_id"] for item in filtered] == ["q-m-001"]


def test_direct_qa_answer_uses_readable_source_labels(tmp_path: Path):
    agent = LegalStudyAgent(_build_config(tmp_path))

    answer = agent._render_direct_qa_answer(
        "押金到期不退怎么办？",
        [
            {
                "source_type": "question_bank",
                "record_id": "q-1",
                "title": "租赁合同中押金返还",
                "excerpt": "承租人返还房屋后，出租人应在扣除合理费用后返还押金。",
                "score": 0.72,
                "metadata": {"answer": "C", "analysis": "不存在应抵扣事项时，出租人应及时返还押金。"},
            }
        ],
        statute_results=[
            {
                "document_title": "中华人民共和国民法典",
                "article_heading": "第五百七十七条",
                "text": "当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。",
                "score": 0.26,
            }
        ],
    )

    assert "题库解析" in answer
    assert "法规依据" in answer
    assert "[statute]" not in answer
    assert "[question_bank]" not in answer
