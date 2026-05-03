from __future__ import annotations

import re

from context_engine.schemas import ContextBundle
from planning_engine.schema import ActionPlan, ToolPlanStep


ANSWER_SHEET_RE = re.compile(r"(?m)^\s*\d+\s*[\.、:：-]")
QUESTION_COUNT_RE = re.compile(r"(?P<count>\d{1,2})\s*题")
CALC_MARKERS = ("计算", "算", "赔偿", "补偿", "罚款", "税额", "利息", "金额", "数额")
PROFILE_VIEW_MARKERS = ("我的档案", "我的画像", "我的信息", "查看档案", "查看画像")
REPORT_MARKERS = ("报告", "总结", "学习画像", "学习总结", "学习报告", "周报")
EXAM_MARKERS = ("法考测试", "模拟测试", "出题", "练习题", "刷题", "来一套", "模拟卷", "测试题")
SUBJECT_MARKERS = ("民法", "刑法", "行政法", "民诉", "刑诉", "商经", "理论法", "宪法")
PROFILE_DISCLOSURE_HINTS = ("备考", "复习", "薄弱", "短板", "强项", "目标", "分数", "偏好", "习惯", "每天", "小时", "名字", "称呼")


class StudyPlanner:
    def __init__(self, *, default_exam_question_count: int = 5) -> None:
        self.default_exam_question_count = default_exam_question_count

    def plan(self, query: str, context: ContextBundle, tool_definitions: list[dict[str, object]] | None = None) -> ActionPlan:
        intent = self._detect_intent(query, context)
        planning_hint = self._planning_hint(context)

        if intent == "profile_lookup":
            return ActionPlan(
                intent=intent,
                objective="读取用户画像与近期学习状态。",
                steps=[ToolPlanStep("profile_view", "先读取用户档案，避免重复提问。")],
                response_style="summary",
            )

        if intent == "profile_update":
            return ActionPlan(
                intent=intent,
                objective="更新并确认用户画像。",
                steps=[
                    ToolPlanStep("profile_upsert", "将用户刚刚提供的档案信息写入长期画像。", {"raw_text": query}),
                    ToolPlanStep("profile_view", "写入后回读一次，确保返回给用户的是最新档案。"),
                ],
                response_style="confirm",
            )

        if intent == "mock_exam_generate":
            return ActionPlan(
                intent=intent,
                objective=f"结合当前 planning_context 生成一套与当前学习目标相匹配的法考模拟题。{planning_hint}",
                steps=[
                    ToolPlanStep(
                        "generate_exam",
                        "根据 planning_context 已整理出的薄弱点和当前主题生成模拟卷。",
                        {
                            "topic": self._extract_topic(query, context),
                            "question_count": self._extract_question_count(query),
                            "exam_type": self._extract_exam_type(query),
                            "question_types": self._extract_question_types(query),
                        },
                    ),
                ],
                response_style="exam_sheet",
            )

        if intent == "mock_exam_score":
            return ActionPlan(
                intent=intent,
                objective="对用户已提交的模拟测试答案评分并生成反馈。",
                steps=[
                    ToolPlanStep("score_exam", "先对当前激活的模拟测试进行评分。", {"answers_text": query}),
                    ToolPlanStep("generate_report", "评分后顺手产出一份学习反馈报告。", {"report_type": "exam_feedback"}),
                ],
                response_style="grading",
            )

        if intent == "report_generation":
            return ActionPlan(
                intent=intent,
                objective=f"基于 summary_blocks 和 planning_context 整理用户近期学习报告。{planning_hint}",
                steps=[
                    ToolPlanStep("generate_report", "生成结构化学习报告。", {"report_type": "study_progress"}),
                ],
                response_style="report",
            )

        if intent == "legal_calculation":
            return ActionPlan(
                intent=intent,
                objective=f"结合 planning_context、法律知识和数字计算给出学习型分析。{planning_hint}",
                steps=[
                    ToolPlanStep("rag_search", "检索法条、题库、案例与常识知识。", {"query": query, "top_k": 6}),
                    ToolPlanStep("calculator", "对用户显式提出的数字问题做安全计算。", {"expression": self._extract_expression(query)}),
                ],
                response_style="legal_analysis",
            )

        return ActionPlan(
            intent="legal_qa",
            objective=f"结合 planning_context 回答法律学习与法考相关问题，并给出知识依据。{planning_hint}",
            steps=[
                ToolPlanStep("rag_search", "综合检索法条、题库、案例和常识知识。", {"query": query, "top_k": 6}),
            ],
            response_style="legal_analysis",
        )

    def _detect_intent(self, query: str, context: ContextBundle) -> str:
        normalized = "".join(str(query or "").split())
        if any(marker in normalized for marker in PROFILE_VIEW_MARKERS):
            return "profile_lookup"
        if self._looks_like_answer_sheet(normalized) and context.session_state.active_exam_session_id:
            return "mock_exam_score"
        if any(marker in normalized for marker in REPORT_MARKERS):
            return "report_generation"
        if any(marker in normalized for marker in EXAM_MARKERS):
            return "mock_exam_generate"
        if self._looks_like_profile_update(normalized):
            return "profile_update"
        if self._looks_like_calculation(normalized):
            return "legal_calculation"
        return "legal_qa"

    def _looks_like_profile_update(self, query: str) -> bool:
        normalized = "".join(str(query or "").split())
        if not normalized or "?" in normalized or "？" in normalized:
            return False
        if not any(hint in normalized for hint in PROFILE_DISCLOSURE_HINTS):
            return False
        return any(token in normalized for token in ("我", "我的", "以后", "之后"))

    def _looks_like_answer_sheet(self, query: str) -> bool:
        return bool(ANSWER_SHEET_RE.search(query)) or "我的答案" in query or "提交答案" in query

    def _looks_like_calculation(self, query: str) -> bool:
        has_number = bool(re.search(r"\d", query))
        has_operator = bool(re.search(r"[+\-*/%()]", query))
        if has_number and has_operator:
            return True
        return has_number and any(marker in query for marker in CALC_MARKERS)

    def _extract_topic(self, query: str, context: ContextBundle) -> str:
        for subject in SUBJECT_MARKERS:
            if subject in query:
                return subject
        for subject in SUBJECT_MARKERS:
            if subject in context.planning_context:
                return subject
        if context.user_profile.weak_points:
            return context.user_profile.weak_points[0]
        if context.user_profile.study_goals:
            return context.user_profile.study_goals[0]
        return "综合"

    def _planning_hint(self, context: ContextBundle) -> str:
        session_summary = str(context.summary_blocks.get("session") or "").strip()
        if not session_summary:
            return ""
        compact = session_summary.replace("\n", " ").strip()
        if len(compact) > 48:
            compact = compact[:47].rstrip(" ，,；;") + "…"
        return f"当前会话摘要：{compact}"

    def _extract_question_count(self, query: str) -> int:
        match = QUESTION_COUNT_RE.search(query)
        if match is None:
            return self.default_exam_question_count
        return max(1, min(int(match.group("count")), 20))

    def _extract_exam_type(self, query: str) -> str:
        normalized = "".join(str(query or "").split())
        if "薄弱" in normalized or "错题" in normalized:
            return "薄弱点强化"
        if "章节" in normalized:
            return "章节练习"
        if "真题" in normalized:
            return "真题模拟"
        return "综合练习"

    def _extract_question_types(self, query: str) -> list[str]:
        normalized = "".join(str(query or "").split())
        if any(marker in normalized for marker in ("简答", "主观", "问答")):
            return ["short_answer"]
        if any(marker in normalized for marker in ("案例", "案例分析")):
            return ["case_analysis"]
        if any(marker in normalized for marker in ("混合", "综合题型")):
            return ["single_choice", "short_answer", "case_analysis"]
        return ["single_choice"]

    def _extract_expression(self, query: str) -> str:
        candidate = query.replace("请", "").replace("帮我", "").strip()
        return candidate or "0"