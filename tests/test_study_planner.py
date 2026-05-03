from context_engine.schemas import ContextBundle, SessionState, UserProfile
from planning_engine.planner import StudyPlanner


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


def test_planner_detects_profile_update():
    planner = StudyPlanner()
    plan = planner.plan("最近主要复习民法，行政法还是我的短板，之后回答尽量短一点。", _context())

    assert plan.intent == "profile_update"
    assert [step.tool_name for step in plan.steps] == ["profile_upsert", "profile_view"]


def test_planner_detects_exam_generation():
    planner = StudyPlanner(default_exam_question_count=3)
    plan = planner.plan("给我来一套行政法 3 题模拟测试", _context())

    assert plan.intent == "mock_exam_generate"
    assert [step.tool_name for step in plan.steps] == ["generate_exam"]
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
    assert plan.steps[-1].arguments["question_types"] == ["short_answer"]


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
    assert [step.tool_name for step in plan.steps] == ["rag_search", "calculator"]


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