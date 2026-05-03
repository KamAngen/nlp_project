from context_engine.schemas import ContextBundle, SessionState, UserProfile
from planning_engine.planner import StudyPlanner
from planning_engine.schema import ActionPlan, ToolPlanStep


def _context(*, active_exam: bool = False) -> ContextBundle:
    return ContextBundle(
        user_profile=UserProfile(user_id="u1", study_goals=["民法"], weak_points=["行政法"]),
        session_state=SessionState(
            session_id="s1",
            user_id="u1",
            active_exam_session_id="exam-1" if active_exam else None,
        ),
        layer_hits={},
        summary_blocks={},
    )


# ==================== 原有测试 ====================

def test_planner_detects_profile_update():
    planner = StudyPlanner()
    plan = planner.plan("记住我是张三，我叫张三，我在备考法考。", _context())

    assert plan.intent == "profile_update"
    assert [step.tool_name for step in plan.steps] == ["profile_upsert", "profile_view"]


def test_planner_detects_exam_generation():
    planner = StudyPlanner(default_exam_question_count=3)
    plan = planner.plan("给我来一套行政法 3 题模拟测试", _context())

    assert plan.intent == "mock_exam_generate"
    assert "profile_view" in [step.tool_name for step in plan.steps]
    assert "memory_search" in [step.tool_name for step in plan.steps]
    assert "generate_exam" in [step.tool_name for step in plan.steps]
    assert plan.steps[-1].arguments["topic"] == "行政法"
    assert plan.steps[-1].arguments["question_count"] == 3
    assert plan.steps[-1].arguments["exam_type"] == "综合练习"


def test_planner_extracts_specific_exam_type():
    planner = StudyPlanner(default_exam_question_count=3)
    plan = planner.plan("给我来一套行政法真题 3 题模拟测试", _context())

    assert plan.intent == "mock_exam_generate"
    assert plan.steps[-1].arguments["exam_type"] == "真题模拟"


def test_planner_extracts_subjective_question_type():
    planner = StudyPlanner(default_exam_question_count=2)
    plan = planner.plan("给我来一套民诉 2 题简答题", _context())

    assert plan.intent == "mock_exam_generate"
    assert plan.steps[-1].arguments["topic"] == "民诉"
    assert plan.steps[-1].arguments["question_count"] == 2


def test_planner_detects_exam_scoring_when_active_exam_exists():
    planner = StudyPlanner()
    plan = planner.plan("我的答案是 1.A 2.B 3.C", _context(active_exam=True))

    assert plan.intent == "mock_exam_score"
    assert [step.tool_name for step in plan.steps] == ["score_exam", "generate_report"]


def test_planner_detects_numbered_text_answer_sheet_when_active_exam_exists():
    planner = StudyPlanner()
    plan = planner.plan("1. 应先说明情况紧急\n2. 应依法提供担保", _context(active_exam=True))

    assert plan.intent == "mock_exam_score"
    assert [step.tool_name for step in plan.steps] == ["score_exam", "generate_report"]


def test_planner_detects_legal_calculation():
    planner = StudyPlanner()
    plan = planner.plan("请帮我计算 20000 * 0.1 - 210", _context())

    assert plan.intent == "legal_calculation"
    assert [step.tool_name for step in plan.steps] == ["memory_search", "rag_search", "calculator"]


def test_planner_uses_planning_context_when_query_has_no_subject():
    planner = StudyPlanner(default_exam_question_count=2)
    context = ContextBundle(
        user_profile=UserProfile(user_id="u1"),
        session_state=SessionState(session_id="s1", user_id="u1"),
        layer_hits={},
        summary_blocks={"session": "最近一直在复习刑法总则。"},
        planning_context="【当前会话关键上下文】\n最近一直在复习刑法总则。",
    )

    plan = planner.plan("给我来一套 2 题模拟测试", context)

    assert plan.intent == "mock_exam_generate"
    assert plan.steps[-1].arguments["topic"] == "刑法"


# ==================== Schema 新增字段测试 ====================

def test_action_plan_has_original_query_field():
    plan = ActionPlan(intent="legal_qa", objective="测试", original_query="原始问题")
    assert plan.original_query == "原始问题"
    assert plan.turn_state == {}


def test_action_plan_has_turn_state_field():
    plan = ActionPlan(intent="legal_qa", objective="测试", turn_state={"step": 1, "missing": ["topic"]})
    assert plan.turn_state["step"] == 1
    assert plan.turn_state["missing"] == ["topic"]


def test_tool_plan_step_has_condition_field():
    step = ToolPlanStep(tool_name="test", reason="测试", condition="topic is not None")
    assert step.condition == "topic is not None"
    assert step.depends_on == []
    assert step.is_optional is False


def test_tool_plan_step_has_optional_flag():
    step = ToolPlanStep(tool_name="memory_search", reason="可选检索", is_optional=True)
    assert step.is_optional is True


# ==================== 意图识别增强测试 ====================

def test_detect_intent_with_confidence_returns_tuple():
    planner = StudyPlanner()
    result = planner._detect_intent_with_confidence("查看我的档案", _context())
    assert isinstance(result, tuple)
    assert len(result) == 2
    intent, confidence = result
    assert intent == "profile_lookup"
    assert 0.0 <= confidence <= 1.0


def test_planner_detects_stop_intent():
    planner = StudyPlanner()
    stop_queries = ["我不想做了", "算了，不学了", "停止", "取消任务", "放弃", "别问了", "就这样吧"]
    for query in stop_queries:
        intent, confidence = planner._detect_intent_with_confidence(query, _context())
        assert intent == "stop", f"查询 '{query}' 应该被识别为 stop 意图"
        assert confidence >= 0.9, f"查询 '{query}' 的置信度应该 >= 0.9"


def test_planner_detects_general_qa_for_weather():
    planner = StudyPlanner()
    general_queries = [
        "今天天气怎么样？",
        "现在几点了？",
        "讲个笑话听听",
        "你好，早上好",
        "明天星期几？",
    ]
    for query in general_queries:
        intent, confidence = planner._detect_intent_with_confidence(query, _context())
        assert intent == "general_qa", f"查询 '{query}' 应该被识别为 general_qa 意图"


def test_planner_detects_legal_qa_via_keywords():
    planner = StudyPlanner()
    legal_queries = [
        "合同违约怎么赔偿？",
        "盗窃罪如何量刑？",
        "什么是正当防卫？",
        "起诉流程是什么？",
        "法律规定怎么说的？",
    ]
    for query in legal_queries:
        intent, confidence = planner._detect_intent_with_confidence(query, _context())
        assert intent == "legal_qa", f"查询 '{query}' 应该被识别为 legal_qa 意图"


# ==================== 追问功能测试 ====================

def test_generate_clarification_question_for_missing_topic():
    planner = StudyPlanner()
    missing_info = ["未明确指定测试主题"]
    question = planner._generate_clarification_question(missing_info, "mock_exam_generate")
    assert question is not None
    assert "科目" in question or "民法" in question


def test_generate_clarification_question_for_missing_value():
    planner = StudyPlanner()
    missing_info = ["未提供具体数值"]
    question = planner._generate_clarification_question(missing_info, "legal_calculation")
    assert question is not None
    assert "数值" in question


def test_generate_clarification_question_includes_original_query():
    planner = StudyPlanner()
    missing_info = ["未明确指定测试主题"]
    original = "我想做几道题"
    question = planner._generate_clarification_question(missing_info, "mock_exam_generate", original)
    assert question is not None
    assert "原问题" in question
    assert original in question


def test_generate_clarification_question_returns_none_when_no_missing():
    planner = StudyPlanner()
    question = planner._generate_clarification_question([], "legal_qa")
    assert question is None


# ==================== 领域切换检测测试 ====================

def test_analyze_turn_detects_domain_switch_to_general():
    planner = StudyPlanner()
    history = [
        ("user", "合同违约怎么赔偿？"),
        ("assistant", "根据民法典相关规定..."),
    ]
    analysis = planner.analyze_turn(
        "今天天气怎么样？",
        _context(),
        history=history,
    )
    assert analysis["is_domain_switch"] is True
    assert analysis["domain_switch_from"] == "legal"
    assert analysis["domain_switch_to"] == "general"


def test_analyze_turn_detects_domain_switch_to_legal():
    planner = StudyPlanner()
    history = [
        ("user", "今天天气怎么样？"),
        ("assistant", "今天晴天..."),
    ]
    analysis = planner.analyze_turn(
        "合同违约怎么赔偿？",
        _context(),
        history=history,
    )
    assert analysis["is_domain_switch"] is True
    assert analysis["domain_switch_from"] == "general"
    assert analysis["domain_switch_to"] == "legal"


def test_analyze_turn_no_domain_switch_on_first_turn():
    planner = StudyPlanner()
    analysis = planner.analyze_turn("今天天气怎么样？", _context(), history=None)
    assert analysis["is_domain_switch"] is False


# ==================== plan() 返回测试 ====================

def test_plan_for_stop_returns_empty_steps():
    planner = StudyPlanner()
    plan = planner.plan("我不想做了，停止吧", _context())
    assert plan.intent == "stop"
    assert plan.steps == []
    assert plan.response_style == "acknowledge"
    assert "停止" in plan.objective


def test_plan_for_general_qa_returns_empty_steps():
    planner = StudyPlanner()
    plan = planner.plan("今天天气怎么样？", _context())
    assert plan.intent == "general_qa"
    assert plan.steps == []
    assert plan.response_style == "direct_answer"
    assert "通用问题" in plan.notes[0]


# ==================== analyze_turn 新增字段测试 ====================

def test_analyze_turn_returns_clarification_question():
    planner = StudyPlanner()
    ctx_no_profile = ContextBundle(
        user_profile=UserProfile(user_id="u1", study_goals=[], weak_points=[]),
        session_state=SessionState(session_id="s1", user_id="u1"),
        layer_hits={},
        summary_blocks={},
    )
    analysis = planner.analyze_turn("我想做题，帮我出题", ctx_no_profile)
    assert "clarification_question" in analysis
    assert analysis["clarification_question"] is not None


def test_analyze_turn_returns_stop_flag():
    planner = StudyPlanner()
    analysis = planner.analyze_turn("算了，不做了", _context())
    assert analysis["should_stop_current_task"] is True
    assert analysis["stop_reason"] == "user_requested"


def test_analyze_turn_returns_domain_switch_fields():
    planner = StudyPlanner()
    analysis = planner.analyze_turn("你好", _context())
    assert "is_domain_switch" in analysis
    assert "domain_switch_from" in analysis
    assert "domain_switch_to" in analysis


# ==================== 置信度评分测试 ====================

def test_confidence_score_for_clear_intent():
    planner = StudyPlanner()
    _, confidence = planner._detect_intent_with_confidence("查看我的档案", _context())
    assert confidence >= 0.8


def test_confidence_score_for_ambiguous_query():
    planner = StudyPlanner()
    _, confidence = planner._detect_intent_with_confidence("帮我看看", _context())
    assert 0.0 <= confidence <= 1.0


# ==================== 考试类型提取测试 ====================

def test_extract_exam_type_weak_point():
    planner = StudyPlanner()
    plan = planner.plan("帮我复习薄弱点", _context())
    assert plan.intent == "mock_exam_generate"
    assert plan.steps[-1].arguments["exam_type"] == "薄弱点强化"


def test_extract_exam_type_chapter():
    planner = StudyPlanner()
    plan = planner.plan("给我出第三章的练习题", _context())
    assert plan.intent == "mock_exam_generate"
    assert plan.steps[-1].arguments["exam_type"] == "章节练习"


def test_extract_exam_type_sprint():
    planner = StudyPlanner()
    plan = planner.plan("来一套冲刺押题卷", _context())
    assert plan.intent == "mock_exam_generate"
    assert plan.steps[-1].arguments["exam_type"] == "冲刺练习"
