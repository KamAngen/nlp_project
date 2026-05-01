from __future__ import annotations

import logging
import re
from typing import Any

from context_engine.schemas import ContextBundle
from planning_engine.schema import ActionPlan, ToolPlanStep

logger = logging.getLogger(__name__)

ANSWER_SHEET_RE = re.compile(r"(?:^|[\s,，；;])\d+\s*[\.、:：-]?\s*[A-Da-d]")
QUESTION_COUNT_RE = re.compile(r"(?P<count>\d{1,2})\s*题")
CHINESE_NUMBERS = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "两": 2,
}
CHINESE_NUMBER_RE = re.compile(r"[一二三四五六七八九十两]")

CALC_MARKERS = ("计算", "算", "赔偿", "补偿", "罚款", "税额", "利息", "金额", "数额")
PROFILE_VIEW_MARKERS = ("我的档案", "我的画像", "我的信息", "查看档案", "查看画像", "学习情况", "学习状态", "我的学习")
REPORT_MARKERS = ("报告", "总结", "学习画像", "学习总结", "学习报告", "周报", "复盘")
EXAM_MARKERS = ("法考测试", "模拟测试", "出题", "练习题", "刷题", "来一套", "模拟卷", "测试题", "做套题", "考考我", "测验", "练习练习", "给我几道题")
SUBJECT_MARKERS = ("民法", "刑法", "行政法", "民诉", "刑诉", "商经", "理论法", "宪法", "国际法", "环境法", "经济法", "知识产权法")
PROFILE_UPDATE_MARKERS = ("记住", "我是", "我叫", "我在备考", "我的薄弱点", "我的强项", "目标分数")
FOLLOWUP_ANSWER_MARKERS = ("回答", "答案是", "选", "我的选择", "我选")

EXAM_TYPE_PATTERNS = {
    "薄弱点强化": ("薄弱", "错题", "弱点", "容易错"),
    "章节练习": ("章节", "第.*章", "专项"),
    "真题模拟": ("真题", "历年", "往年", "真实考题"),
    "冲刺练习": ("冲刺", "押题", "预测", "密卷"),
}


class StudyPlanner:
    def __init__(
        self,
        *,
        default_exam_question_count: int = 5,
        enable_logging: bool = True,
    ) -> None:
        self.default_exam_question_count = default_exam_question_count
        self.enable_logging = enable_logging

    def plan(
        self,
        query: str,
        context: ContextBundle,
        tool_definitions: list[dict[str, object]] | None = None,
    ) -> ActionPlan:
        intent, confidence = self._detect_intent_with_confidence(query, context)

        if self.enable_logging:
            logger.info(
                "Planning: query='%s', intent='%s', confidence=%.2f",
                query[:50],
                intent,
                confidence,
            )

        if intent == "profile_lookup":
            return ActionPlan(
                intent=intent,
                objective="读取用户画像与近期学习状态。",
                steps=[ToolPlanStep("profile_view", "先读取用户档案，避免重复提问。")],
                response_style="summary",
                metadata={"confidence": confidence},
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
                metadata={"confidence": confidence},
            )

        if intent == "mock_exam_generate":
            topic = self._extract_topic(query, context)
            question_count = self._extract_question_count(query)
            exam_type = self._extract_exam_type(query)

            steps = [
                ToolPlanStep("profile_view", "先读取用户画像，确定选题偏好与难点。"),
            ]

            # TODO: 组员1负责实现 memory_search 的完整功能，当前仅作为计划步骤预留
            steps.append(
                ToolPlanStep(
                    "memory_search",
                    "优先命中用户画像里的薄弱点与近期学习主题。",
                    {"query": query, "top_k": 6},
                )
            )

            steps.append(
                ToolPlanStep(
                    "generate_exam",
                    "根据题目需求和用户画像生成模拟卷。",
                    {
                        "topic": topic,
                        "question_count": question_count,
                        "exam_type": exam_type,
                    },
                )
            )

            return ActionPlan(
                intent=intent,
                objective=f"生成一套{topic}主题的{exam_type}，共{question_count}题。",
                steps=steps,
                response_style="exam_sheet",
                metadata={
                    "confidence": confidence,
                    "extracted_params": {
                        "topic": topic,
                        "question_count": question_count,
                        "exam_type": exam_type,
                    },
                },
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
                metadata={"confidence": confidence},
            )

        if intent == "report_generation":
            return ActionPlan(
                intent=intent,
                objective="整理用户近期学习报告。",
                steps=[
                    # TODO: 组员1负责实现 memory_search 的完整功能，当前仅作为计划步骤预留
                    ToolPlanStep("memory_search", "先检索近期会话、薄弱点和测试记录。", {"query": query, "top_k": 8}),
                    ToolPlanStep("generate_report", "生成结构化学习报告。", {"report_type": "study_progress"}),
                ],
                response_style="report",
                metadata={"confidence": confidence},
            )

        if intent == "legal_calculation":
            return ActionPlan(
                intent=intent,
                objective="结合法律知识和数字计算给出学习型分析。",
                steps=[
                    # TODO: 组员1负责实现 memory_search 的完整功能，当前仅作为计划步骤预留
                    ToolPlanStep("memory_search", "先读取与当前问题相关的历史上下文。", {"query": query, "top_k": 6}),
                    # TODO: 组员3负责实现 rag_search 的完整功能，当前仅作为计划步骤预留
                    ToolPlanStep("rag_search", "检索法条、题库、案例与常识知识。", {"query": query, "top_k": 6}),
                    ToolPlanStep("calculator", "对用户显式提出的数字问题做安全计算。", {"expression": self._extract_expression(query)}),
                ],
                response_style="legal_analysis",
                metadata={"confidence": confidence},
            )

        if intent == "followup_answer":
            return ActionPlan(
                intent=intent,
                objective="处理用户对追问的回答，继续原任务分析。",
                steps=[
                    # TODO: 组员1负责实现 memory_search 的完整功能，当前仅作为计划步骤预留
                    ToolPlanStep("memory_search", "读取上一轮追问上下文。", {"query": query, "top_k": 4}),
                    # TODO: 组员3负责实现 rag_search 的完整功能，当前仅作为计划步骤预留
                    ToolPlanStep("rag_search", "基于补充事实继续检索相关依据。", {"query": query, "top_k": 4}),
                ],
                response_style="continuation",
                metadata={"confidence": confidence},
            )

        # 默认：legal_qa
        return ActionPlan(
            intent="legal_qa",
            objective="回答法律学习与法考相关问题，并给出知识依据。",
            steps=[
                # TODO: 组员1负责实现 memory_search 的完整功能，当前仅作为计划步骤预留
                ToolPlanStep("memory_search", "先取回用户画像、最近对话和关键系统记忆。", {"query": query, "top_k": 6}),
                # TODO: 组员3负责实现 rag_search 的完整功能，当前仅作为计划步骤预留
                ToolPlanStep("rag_search", "综合检索法条、题库、案例和常识知识。", {"query": query, "top_k": 6}),
            ],
            response_style="legal_analysis",
            metadata={"confidence": confidence},
        )

    def _detect_intent_with_confidence(self, query: str, context: ContextBundle) -> tuple[str, float]:
        normalized = "".join(str(query or "").split())
        if not normalized:
            return "legal_qa", 0.0

        scores: dict[str, float] = {}

        if any(marker in normalized for marker in PROFILE_VIEW_MARKERS):
            scores["profile_lookup"] = 0.9

        if self._looks_like_answer_sheet(normalized) and context.session_state.active_exam_session_id:
            scores["mock_exam_score"] = 0.95
        elif self._looks_like_followup_answer(normalized) and context.session_state.active_exam_session_id:
            scores["mock_exam_score"] = 0.7

        if any(marker in normalized for marker in REPORT_MARKERS):
            scores["report_generation"] = 0.85

        if any(marker in normalized for marker in EXAM_MARKERS):
            scores["mock_exam_generate"] = 0.9

        if any(marker in normalized for marker in PROFILE_UPDATE_MARKERS):
            scores["profile_update"] = 0.9

        if self._looks_like_calculation(normalized):
            scores["legal_calculation"] = 0.8

        if self._looks_like_followup_answer(normalized) and not context.session_state.active_exam_session_id:
            scores["followup_answer"] = 0.7

        if not scores:
            return "legal_qa", 0.5

        best_intent = max(scores, key=scores.get)
        confidence = scores[best_intent]

        if self.enable_logging:
            logger.debug("Intent scores: %s", scores)

        return best_intent, confidence

    def _detect_intent(self, query: str, context: ContextBundle) -> str:
        intent, _ = self._detect_intent_with_confidence(query, context)
        return intent

    def _looks_like_answer_sheet(self, query: str) -> bool:
        return bool(ANSWER_SHEET_RE.search(query)) or "我的答案" in query or "提交答案" in query

    def _looks_like_followup_answer(self, query: str) -> bool:
        normalized = "".join(str(query or "").split())
        if len(normalized) > 80:
            return False
        if any(marker in normalized for marker in FOLLOWUP_ANSWER_MARKERS):
            return True
        if ANSWER_SHEET_RE.search(query):
            return True
        if normalized in {"是", "否", "对", "错", "有", "没有", "好的", "ok", "yes", "no"}:
            return True
        return False

    def _looks_like_calculation(self, query: str) -> bool:
        has_number = bool(re.search(r"\d", query))
        has_operator = bool(re.search(r"[+\-*/%()]", query))
        if has_number and has_operator:
            return True
        return has_number and any(marker in query for marker in CALC_MARKERS)

    def _extract_topic(self, query: str, context: ContextBundle) -> str:
        normalized = str(query or "").strip()
        for subject in SUBJECT_MARKERS:
            if subject in normalized:
                return subject

        topic_keywords = {
            "合同": "民法",
            "物权": "民法",
            "侵权": "民法",
            "婚姻": "民法",
            "继承": "民法",
            "犯罪": "刑法",
            "刑罚": "刑法",
            "盗窃": "刑法",
            "诈骗": "刑法",
            "行政诉讼": "行政法",
            "行政处罚": "行政法",
            "民事诉讼": "民诉",
            "刑事诉讼": "刑诉",
            "公司": "商经",
            "破产": "商经",
            "票据": "商经",
        }
        for keyword, subject in topic_keywords.items():
            if keyword in normalized:
                return subject

        if context.user_profile.weak_points:
            return context.user_profile.weak_points[0]
        if context.user_profile.study_goals:
            return context.user_profile.study_goals[0]
        return "综合"

    def _extract_question_count(self, query: str) -> int:
        match = QUESTION_COUNT_RE.search(query)
        if match:
            return max(1, min(int(match.group("count")), 20))

        for chinese_char, value in CHINESE_NUMBERS.items():
            if chinese_char in query and "题" in query:
                return max(1, min(value, 20))

        count_patterns = [
            r"(\d+)\s*道\s*题",
            r"(\d+)\s*个?\s*题",
            r"来\s*(\d+)\s*题",
            r"出\s*(\d+)\s*题",
        ]
        for pattern in count_patterns:
            match = re.search(pattern, query)
            if match:
                return max(1, min(int(match.group(1)), 20))

        return self.default_exam_question_count

    def _extract_exam_type(self, query: str) -> str:
        normalized = str(query or "").strip()
        for exam_type, markers in EXAM_TYPE_PATTERNS.items():
            for marker in markers:
                if re.search(marker, normalized):
                    return exam_type
        return "综合练习"

    def _extract_expression(self, query: str) -> str:
        candidate = str(query or "").strip()
        candidate = re.sub(r"请(帮我|帮我)?(计算|算一下|算算|算)?", "", candidate)
        candidate = re.sub(r"结果是?多少[？?]?", "", candidate)
        candidate = candidate.strip(" ，,;；。.")

        math_expr = re.findall(r"[\d+\-*/%().\s]+", candidate)
        if math_expr:
            longest = max(math_expr, key=len).strip()
            if len(longest) >= 3 and any(c.isdigit() for c in longest):
                return longest

        return candidate or "0"

    def analyze_turn(
        self,
        query: str,
        context: ContextBundle,
        *,
        history: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any]:
        normalized = str(query or "").strip()
        if not normalized:
            return self._empty_turn_analysis()

        intent, confidence = self._detect_intent_with_confidence(query, context)
        input_role = self._classify_input_role(query, context, history)
        missing_info = self._identify_missing_info(query, context, intent)

        analysis = {
            "current_input_role": input_role,
            "user_goal": self._summarize_user_goal(query, intent),
            "needs_history": self._should_use_history(input_role, history),
            "history_usage": self._describe_history_usage(input_role),
            "requires_precise_result": self._requires_precise_result(query),
            "preferred_answer_style": self._suggest_answer_style(query, intent),
            "likely_missing_info": missing_info,
            "recommended_next_step": self._recommend_next_step(intent, missing_info),
            "intent": intent,
            "intent_confidence": confidence,
            "should_ask_user": self._should_ask_user(query, context, intent, missing_info),
            "clarification_priority": self._get_clarification_priority(missing_info),
        }

        if self.enable_logging:
            logger.info(
                "Turn analysis: input_role='%s', intent='%s', should_ask_user=%s, missing_info=%s",
                input_role,
                intent,
                analysis["should_ask_user"],
                missing_info[:2] if missing_info else [],
            )

        return analysis

    def _empty_turn_analysis(self) -> dict[str, Any]:
        return {
            "current_input_role": "empty",
            "user_goal": "",
            "needs_history": False,
            "history_usage": "无输入，无需处理。",
            "requires_precise_result": False,
            "preferred_answer_style": "brief_direct",
            "likely_missing_info": [],
            "recommended_next_step": "wait_for_input",
            "intent": "legal_qa",
            "intent_confidence": 0.0,
            "should_ask_user": False,
            "clarification_priority": "none",
        }

    def _classify_input_role(
        self,
        query: str,
        context: ContextBundle,
        history: list[tuple[str, str]] | None = None,
    ) -> str:
        normalized = "".join(str(query or "").split())

        if self._looks_like_answer_sheet(normalized):
            return "answer_sheet"

        if self._looks_like_followup_answer(normalized):
            if context.session_state.active_exam_session_id:
                return "exam_answer"
            return "supplement"

        if not history:
            return "new_question"

        last_assistant = history[-1][1] if history else ""
        if self._assistant_asked_question(last_assistant):
            return "answer_to_followup"

        if any(marker in normalized for marker in ("对", "不对", "是的", "不是", "纠正", "补充")):
            return "correction"

        return "new_question"

    def _assistant_asked_question(self, assistant_message: str) -> bool:
        question_markers = (
            "请补充",
            "请说明",
            "请确认",
            "是否",
            "有无",
            "能否",
            "请问",
            "？",
            "?",
            "请回答",
        )
        return any(marker in assistant_message for marker in question_markers)

    def _should_use_history(self, input_role: str, history: list[tuple[str, str]] | None) -> bool:
        if not history:
            return False
        if input_role in {"supplement", "answer_to_followup", "correction"}:
            return True
        if input_role == "exam_answer":
            return True
        return len(history) > 0

    def _describe_history_usage(self, input_role: str) -> str:
        usage_map = {
            "supplement": "将当前补充的事实并回上一轮问题继续分析。",
            "answer_to_followup": "将用户回答与上一轮追问结合，继续原任务。",
            "correction": "根据用户纠正调整之前的分析方向。",
            "exam_answer": "结合激活的试卷上下文进行评分。",
            "new_question": "以当前问题为主，历史仅作参考。",
        }
        return usage_map.get(input_role, "以当前输入为主。")

    def _summarize_user_goal(self, query: str, intent: str) -> str:
        goal_map = {
            "profile_lookup": "查看个人学习画像",
            "profile_update": "更新个人学习画像",
            "mock_exam_generate": "生成模拟测试",
            "mock_exam_score": "提交测试答案并评分",
            "report_generation": "生成学习报告",
            "legal_calculation": "进行法律相关计算",
            "followup_answer": "回答追问补充事实",
            "legal_qa": "咨询法律问题",
        }
        return goal_map.get(intent, self._compact_text(query, max_chars=60))

    def _compact_text(self, text: str, *, max_chars: int = 60) -> str:
        cleaned = " ".join(str(text or "").split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max_chars - 1].rstrip() + "…"

    def _requires_precise_result(self, query: str) -> bool:
        normalized = "".join(str(query or "").split())
        markers = ("精确", "多少", "几", "几年", "金额", "税额", "罚款", "赔偿", "具体怎么算", "准确", "具体数额")
        return any(marker in normalized for marker in markers)

    def _suggest_answer_style(self, query: str, intent: str) -> str:
        if intent == "mock_exam_generate":
            return "exam_sheet"
        if intent == "mock_exam_score":
            return "grading"
        if intent == "report_generation":
            return "report"
        if intent == "profile_lookup":
            return "summary"
        if self._requires_precise_result(query):
            return "structured_explanation"
        if len(str(query or "").strip()) <= 40:
            return "brief_direct"
        return "structured_explanation"

    def _identify_missing_info(
        self,
        query: str,
        context: ContextBundle,
        intent: str,
    ) -> list[str]:
        missing: list[str] = []
        normalized = str(query or "").strip()

        if intent == "mock_exam_generate":
            topic = self._extract_topic(query, context)
            if topic == "综合":
                missing.append("未明确指定测试主题")
            if context.user_profile.weak_points:
                missing.append(f"用户薄弱点：{', '.join(context.user_profile.weak_points[:2])}")

        if intent == "legal_qa":
            if not any(marker in normalized for marker in ("?", "？", "怎么", "如何", "什么", "哪些")):
                missing.append("输入可能不是明确的问句")

            location_markers = ("所在地", "居住地", "省", "市", "区县", "地址", "位置")
            if any(marker in normalized for marker in location_markers):
                missing.append("可能需要更具体的地理位置信息")

        if intent == "legal_calculation":
            if not re.search(r"\d", normalized):
                missing.append("未提供具体数值")

        return missing

    def _recommend_next_step(self, intent: str, missing_info: list[str]) -> str:
        step_map = {
            "profile_lookup": "profile_view",
            "profile_update": "profile_upsert",
            "mock_exam_generate": "generate_exam",
            "mock_exam_score": "score_exam",
            "report_generation": "generate_report",
            "legal_calculation": "calculator",
            "followup_answer": "rag_search",
            "legal_qa": "memory_search",
        }
        return step_map.get(intent, "unknown")

    def _should_ask_user(
        self,
        query: str,
        context: ContextBundle,
        intent: str,
        missing_info: list[str],
    ) -> bool:
        if intent in {"profile_lookup", "profile_update", "mock_exam_score", "report_generation"}:
            return False

        if intent == "mock_exam_generate":
            return False

        if intent == "legal_calculation":
            if not re.search(r"\d", str(query or "")):
                return True
            return False

        if not missing_info:
            return False

        critical_missing = [
            m for m in missing_info
            if any(keyword in m for keyword in ("关键事实", "具体数值", "地理位置", "时间"))
        ]
        return len(critical_missing) > 0

    def _get_clarification_priority(self, missing_info: list[str]) -> str:
        if not missing_info:
            return "none"

        critical_keywords = ("关键事实", "具体数值", "时间", "金额")
        for info in missing_info:
            if any(kw in info for kw in critical_keywords):
                return "high"

        if len(missing_info) >= 3:
            return "medium"

        return "low"