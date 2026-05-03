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
            context = self.agent.tool_executor.execute(
                "prepare_context",
                {"query": "行政法"},
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
                {"tool_name": "prepare_context", "reason": "整理上下文", "arguments": {"query": "行政法"}, "result": context},
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


class _StreamingEngine:
    def run_with_updates(self, question: str, history=None):
        yield {"event": "status", "message": "正在整理最终答复。", "trace": "Final Answer: 草稿答案"}
        yield {
            "event": "final",
            "result": SimpleNamespace(
                final_answer="草稿答案",
                trace="Final Answer: 草稿答案",
                tool_history=[],
                errors=[],
                needs_user_input=False,
                clarification_question=None,
            ),
        }


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
    assert "本次测试得分" in score_result.answer
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


def test_score_response_explains_wrong_answers(tmp_path: Path, monkeypatch):
    agent = LegalStudyAgent(_build_config(tmp_path))

    class _NoToolEngine:
        def run(self, question: str, history=None):
            return SimpleNamespace(
                final_answer="模型未主动调用工具。",
                trace="Final Answer: 模型未主动调用工具。",
                tool_history=[],
                errors=[],
                needs_user_input=False,
                clarification_question=None,
            )

    monkeypatch.setattr(agent, "_build_engine", lambda **kwargs: _NoToolEngine())

    agent.memory_manager.record_exam_session(
        "demo_user",
        "demo_session",
        {
            "exam_session_id": "exam-1",
            "topic": "行政法",
            "exam_type": "综合练习",
            "questions": [
                {
                    "index": 1,
                    "record_id": "q-1",
                    "topic": "行政法",
                    "question": "收到信息更正请求后应如何处理？",
                    "options": {
                        "A": "可以忽略",
                        "B": "核实身份后及时提供查询、更正或删除",
                        "C": "统一拖延处理",
                        "D": "只保留纸面登记",
                    },
                    "answer": "B",
                    "analysis": "电子商务经营者在收到用户的信息更正、删除请求后，应当先核实身份，再及时处理。",
                    "tags": ["行政法", "电子商务法"],
                    "score": 20,
                }
            ],
        },
    )

    response = agent.handle_message("1.A", user_id="demo_user", session_id="demo_session")

    assert response.intent == "mock_exam_score"
    assert "正确答案是 B" in response.answer
    assert "解释：" in response.answer
    assert "学习反馈报告已同步到右侧面板" in response.answer


def test_render_exam_answer_surfaces_selection_notes_and_non_misleading_profile_copy(tmp_path: Path):
    agent = LegalStudyAgent(_build_config(tmp_path))

    answer = agent._render_exam_answer(
        [
            {"tool_name": "profile_view", "result": {"profile": {"weak_points": [], "study_goals": []}}},
            {
                "tool_name": "generate_exam",
                "result": {
                    "topic": "民诉",
                    "exam_type": "章节练习",
                    "question_count": 1,
                    "reused_wrong_question_count": 0,
                    "selection_notes": ["当前题库中与民诉严格匹配且通过校验的题目不足 2 题，本次返回 1 题，未使用跨主题题目凑数。"],
                    "questions": [
                        {
                            "index": 1,
                            "question": "申请诉前财产保全通常应满足什么条件？",
                            "options": {
                                "A": "申请人应证明情况紧急，并依法提供担保。",
                                "B": "只能在败诉后申请。",
                                "C": "必须先经过行政机关批准。",
                                "D": "无需证明任何紧急性。",
                            },
                        }
                    ],
                },
            },
        ]
    )

    assert "当前画像暂无额外选题标签" in answer
    assert "未使用跨主题题目凑数" in answer


def test_direct_fallback_scores_answer_sheet_before_profile_updates(tmp_path: Path, monkeypatch):
    agent = LegalStudyAgent(_build_config(tmp_path))

    class _NoToolEngine:
        def run(self, question: str, history=None):
            return SimpleNamespace(
                final_answer="模型未主动调用工具。",
                trace="Final Answer: 模型未主动调用工具。",
                tool_history=[],
                errors=[],
                needs_user_input=False,
                clarification_question=None,
            )

    monkeypatch.setattr(agent, "_build_engine", lambda **kwargs: _NoToolEngine())
    monkeypatch.setattr(
        agent.memory_manager,
        "extract_profile_updates_for_user",
        lambda *args, **kwargs: {"study_goals": ["行政法"], "preferences": {"response_length": "短"}},
    )

    agent.memory_manager.record_exam_session(
        "demo_user",
        "demo_session",
        {
            "exam_session_id": "exam-1",
            "topic": "行政法",
            "exam_type": "综合练习",
            "questions": [
                {
                    "index": 1,
                    "record_id": "q-1",
                    "topic": "行政法",
                    "question": "收到信息更正请求后应如何处理？",
                    "options": {"A": "可以忽略", "B": "核实身份后及时处理", "C": "统一拖延", "D": "只做登记"},
                    "answer": "B",
                    "analysis": "应核实身份后及时处理。",
                    "tags": ["行政法"],
                    "score": 20,
                }
            ],
        },
    )

    response = agent.handle_message("1.A", user_id="demo_user", session_id="demo_session")

    assert response.intent == "mock_exam_score"
    assert response.tool_results[0]["tool_name"] == "score_exam"
    assert all(entry["tool_name"] != "profile_upsert" for entry in response.tool_results)


def test_direct_profile_update_fallback_uses_synthesized_answer(tmp_path: Path, monkeypatch):
    agent = LegalStudyAgent(_build_config(tmp_path))

    class _NoToolEngine:
        def run(self, question: str, history=None):
            return SimpleNamespace(
                final_answer="模型未主动调用工具。",
                trace="Final Answer: 模型未主动调用工具。",
                tool_history=[],
                errors=[],
                needs_user_input=False,
                clarification_question=None,
            )

    monkeypatch.setattr(agent, "_build_engine", lambda **kwargs: _NoToolEngine())
    monkeypatch.setattr(
        agent.memory_manager,
        "extract_profile_updates_for_user",
        lambda *args, **kwargs: {"study_goals": ["民法"], "preferences": {"response_length": "简洁"}},
    )
    monkeypatch.setattr(
        agent,
        "_synthesize_tool_based_answer",
        lambda question, tool_results, **kwargs: "记住了，你当前主要复习民法，后续我会尽量简洁回答，并继续围绕这个方向辅导。",
    )

    response = agent.handle_message("最近主要复习民法，后续答复尽量简洁一点。", user_id="demo_user", session_id="demo_session")

    assert response.intent == "profile_update"
    assert response.answer.startswith("记住了，你当前主要复习民法")
    assert response.tool_results[0]["tool_name"] == "profile_upsert"
    assert response.tool_results[1]["tool_name"] == "profile_view"
    assert response.tool_results[2]["tool_name"] == "prepare_context"


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
    monkeypatch.setattr(
        agent,
        "_synthesize_tool_based_answer",
        lambda question, tool_results, **kwargs: "根据当前检索到的题库解析和学习材料，押金到期无正当抵扣事由时原则上应当返还；如果你愿意，我可以继续按法条依据展开。",
    )

    response = agent.handle_message("押金到期不退怎么办？", user_id="demo_user", session_id="demo_session")

    assert response.intent == "legal_qa"
    assert response.tool_results
    assert response.tool_results[0]["tool_name"] == "prepare_context"
    assert response.tool_results[1]["tool_name"] == "rag_search"
    assert "[statute]" not in response.answer
    assert "[question_bank]" not in response.answer
    assert response.trace.count("Final Answer:") == 1
    assert response.trace.endswith(response.answer)


def test_study_agent_prepare_context_returns_planning_payload(tmp_path: Path, monkeypatch):
    agent = LegalStudyAgent(_build_config(tmp_path))
    monkeypatch.setattr(
        agent,
        "_build_engine",
        lambda **kwargs: _DummyEngine(agent, user_id=kwargs["user_id"], session_id=kwargs["session_id"]),
    )

    agent.handle_message("记住，我在备考民法，我的薄弱点是行政法。", user_id="demo_user", session_id="demo_session")
    payload = agent.prepare_context("请继续围绕行政法给我出题", user_id="demo_user", session_id="demo_session")

    assert payload["long_term_hits"]
    assert payload["related_hits"]
    assert "长期用户画像" in payload["planning_context"]


def test_study_agent_stream_message_yields_final_before_background_persistence(tmp_path: Path, monkeypatch):
    agent = LegalStudyAgent(_build_config(tmp_path))
    monkeypatch.setattr(agent, "_ensure_memory_reasoner", lambda **kwargs: None)
    monkeypatch.setattr(agent, "_normalize_runtime_question", lambda question, **kwargs: question)
    monkeypatch.setattr(agent, "_session_history", lambda *args, **kwargs: [])
    monkeypatch.setattr(agent, "_build_engine", lambda **kwargs: _StreamingEngine())
    monkeypatch.setattr(
        agent,
        "_finalize_response",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("stream_message should not call _finalize_response directly")),
    )

    persisted_answers: list[str] = []
    monkeypatch.setattr(
        agent,
        "_persist_response_turn_async",
        lambda **kwargs: persisted_answers.append(kwargs["response"].answer),
    )
    monkeypatch.setattr(
        agent,
        "_compose_response",
        lambda *args, **kwargs: StudyAgentResponse(
            intent="legal_qa",
            answer="这是最终答复",
            plan={"planner_backend": "llm_react"},
            tool_results=[],
            trace="Final Answer: 这是最终答复",
        ),
    )

    updates = list(agent.stream_message("什么是听证程序", user_id="demo_user", session_id="demo_session"))

    assert updates[0]["event"] == "status"
    assert updates[-1]["event"] == "final"
    assert updates[-1]["response"].answer == "这是最终答复"
    assert persisted_answers == ["这是最终答复"]


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
    assert [entry["tool_name"] for entry in response.tool_results[:2]] == ["prepare_context", "rag_search"]
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
