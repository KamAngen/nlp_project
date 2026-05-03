from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterator

from legal_agent.agent.parser import ParsedStep, parse_react_output
from legal_agent.agent.prompting import build_system_prompt, continue_instruction
from legal_agent.agent.tools import ToolRegistry, observation_to_text
from legal_agent.models.qwen_local import LocalQwenChatModel
from legal_agent.utils.text import simple_tokenize

from planning_engine.planner import StudyPlanner


QUERY_STOPWORDS = {
    "怎么",
    "如何",
    "一下",
    "这个",
    "那个",
    "什么",
    "哪些",
    "还有",
    "然后",
    "就是",
    "已经",
    "现在",
    "刚刚",
    "好像",
    "不知道",
    "是否",
    "可以",
    "需要",
    "应该",
    "一下子",
    "请问",
    "帮我",
    "一下吧",
    "情况",
    "问题",
    "补充",
    "事实",
    "原始",
    "法律",
    "继续",
    "分析",
    "这家",
    "大概",
    "具体",
    "精确",
}

UNCERTAINTY_MARKERS = (
    "无法精确",
    "无法确定",
    "不能确定",
    "难以确定",
    "难以判断",
    "无法直接判断",
    "仍需确认",
    "需进一步确认",
    "仍需补充",
    "信息不足",
    "证据不足",
)

@dataclass(slots=True)
class AgentRunResult:
    final_answer: str
    trace: str
    tool_history: list[dict[str, Any]]
    errors: list[str]
    needs_user_input: bool = False
    clarification_question: str | None = None


class LegalAgentEngine:
    def __init__(
        self,
        model: LocalQwenChatModel,
        registry: ToolRegistry,
        *,
        max_steps: int = 6,
        max_new_tokens: int = 768,
        temperature: float = 0.2,
        top_p: float = 0.9,
        top_k: int = 20,
        presence_penalty: float = 1.0,
        enable_thinking: bool = False,
        prompt_mode: str = "pure",
        turn_analysis_mode: str = "heuristic",
        use_planning_engine: bool = False,
    ) -> None:
        self.model = model
        self.registry = registry
        self.max_steps = max_steps
        self.max_llm_retries = 2
        self.max_followup_questions = 2
        self.max_new_tokens = max_new_tokens
        self.step_max_new_tokens = min(max_new_tokens, 320)
        self.final_max_new_tokens = min(max_new_tokens, 448)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.enable_thinking = enable_thinking
        self.prompt_mode = prompt_mode
        self.turn_analysis_mode = turn_analysis_mode
        self.use_planning_engine = use_planning_engine
        self.planner = StudyPlanner(enable_logging=True) if use_planning_engine else None
        self.system_prompt = build_system_prompt(
            self.registry.tool_definitions(),
            stepwise=True,
            prompt_mode=prompt_mode,
        )

    def run(self, question: str, *, history: list[tuple[str, str]] | None = None) -> AgentRunResult:
        state = self._run_state_machine(self._initial_state(question, history))
        return self._result_from_state(state)

    def run_with_updates(self, question: str, *, history: list[tuple[str, str]] | None = None) -> Iterator[dict[str, Any]]:
        state = self._initial_state(question, history)
        yield {
            "event": "status",
            "trace": "",
            "message": "已接收问题，正在通读对话历史、已有思考记录和当前输入，规划下一步。",
        }

        while True:
            if state.get("scratchpad"):
                yield {
                    "event": "status",
                    "trace": state.get("scratchpad", ""),
                    "message": "正在结合已有证据规划下一步。",
                }
            last_draft = ""
            for event in self._iter_llm_step(state):
                if event["event"] == "llm_partial":
                    draft = str(event.get("draft", "")).strip()
                    if not draft:
                        continue
                    if len(draft) - len(last_draft) >= 24 or draft.endswith(("\n", "。", "；", "}", ")")):
                        last_draft = draft
                        yield {
                            "event": "llm_partial",
                            "trace": self._compose_live_trace(state.get("scratchpad", ""), draft),
                            "message": self._live_message_from_draft(draft),
                        }
                    continue
                state = event["state"]

            route = self._route_after_llm(state)
            if route == "retry":
                yield {
                    "event": "status",
                    "trace": state.get("scratchpad", ""),
                    "message": "模型输出格式不合规，正在自动重试。",
                }
                continue
            if route == "final":
                result = self._result_from_state(state)
                yield {
                    "event": "final",
                    "trace": result.trace,
                    "message": "分析完成。",
                    "result": result,
                }
                return

            tool_name = str(state.get("parsed_payload", {}).get("tool_name") or "unknown")
            yield {
                "event": "status",
                "trace": state.get("scratchpad", ""),
                "message": f"正在执行工具：{tool_name}",
            }
            state = self._tool_node(state)
            yield {
                "event": "tool",
                "trace": state.get("scratchpad", ""),
                "message": f"工具 {tool_name} 已返回结果。",
            }
            route = self._route_after_tool(state)
            if route == "final":
                result = self._result_from_state(state)
                yield {
                    "event": "final",
                    "trace": result.trace,
                    "message": "分析完成。",
                    "result": result,
                }
                return

    def _initial_state(self, question: str, history: list[tuple[str, str]] | None = None) -> dict[str, Any]:
        normalized_history = history or []
        turn_analysis = self._resolve_turn_analysis(question, normalized_history)

        if turn_analysis.get("should_stop_current_task"):
            return {
                "question": question,
                "history": normalized_history,
                "turn_analysis": turn_analysis,
                "scratchpad": "用户主动停止当前任务。",
                "step_count": 0,
                "tool_history": [],
                "errors": [],
                "final_answer": "好的，已停止当前任务。有什么其他问题需要帮助吗？",
            }

        if turn_analysis.get("intent") == "general_qa":
            return {
                "question": question,
                "history": normalized_history,
                "turn_analysis": turn_analysis,
                "scratchpad": "通用问题，由 LLM 直接回答。",
                "step_count": 0,
                "tool_history": [],
                "errors": [],
                "skip_tools": True,
            }

        return {
            "question": question,
            "history": normalized_history,
            "turn_analysis": turn_analysis,
            "scratchpad": "",
            "step_count": 0,
            "llm_retry_count": 0,
            "max_steps": self.max_steps,
            "tool_history": [],
            "errors": [],
        }

    def _resolve_turn_analysis(self, question: str, history: list[tuple[str, str]]) -> dict[str, Any]:
        if self.use_planning_engine and self.planner:
            return self._resolve_turn_analysis_with_planner(question, history)
        if self.turn_analysis_mode != "llm":
            return self._default_turn_analysis(question, history)
        try:
            return self._analyze_user_turn(question, history)
        except Exception:
            return self._default_turn_analysis(question, history)

    def _resolve_turn_analysis_with_planner(self, question: str, history: list[tuple[str, str]]) -> dict[str, Any]:
        from context_engine.schemas import ContextBundle, SessionState, UserProfile

        context = ContextBundle(
            user_profile=UserProfile(user_id="default", study_goals=[], weak_points=[]),
            session_state=SessionState(session_id="default", user_id="default"),
            layer_hits={},
            summary_blocks={},
        )

        analysis = self.planner.analyze_turn(question, context, history=history)

        return {
            "current_input_role": analysis.get("current_input_role", "new_question"),
            "user_goal": analysis.get("user_goal", ""),
            "needs_history": analysis.get("needs_history", bool(history)),
            "history_usage": analysis.get("history_usage", ""),
            "requires_precise_result": analysis.get("requires_precise_result", False),
            "preferred_answer_style": analysis.get("preferred_answer_style", "brief_direct"),
            "likely_missing_info": analysis.get("likely_missing_info", []),
            "recommended_next_step": analysis.get("recommended_next_step", "unknown"),
            "intent": analysis.get("intent", "legal_qa"),
            "intent_confidence": analysis.get("intent_confidence", 0.0),
            "should_stop_current_task": analysis.get("should_stop_current_task", False),
            "clarification_question": analysis.get("clarification_question"),
            "is_domain_switch": analysis.get("is_domain_switch", False),
            "domain_switch_from": analysis.get("domain_switch_from"),
            "domain_switch_to": analysis.get("domain_switch_to"),
        }

    def _run_state_machine(self, state: dict[str, Any]) -> dict[str, Any]:
        while True:
            state = self._llm_node(state)
            route = self._route_after_llm(state)
            if route == "retry":
                continue
            if route == "final":
                return state
            state = self._tool_node(state)
            route = self._route_after_tool(state)
            if route == "final":
                return state

    def _result_from_state(self, state: dict[str, Any]) -> AgentRunResult:
        final_answer = state.get("final_answer") or self._synthesize_final_answer(state, "主流程未显式产出最终答案")
        return AgentRunResult(
            final_answer=final_answer,
            trace=state.get("scratchpad", ""),
            tool_history=state.get("tool_history", []),
            errors=state.get("errors", []),
            needs_user_input=bool(state.get("needs_user_input", False)),
            clarification_question=state.get("clarification_question"),
        )

    def _normalize_text(self, text: str) -> str:
        return "".join(str(text or "").split())

    def _contains_any(self, text: str, markers: tuple[str, ...]) -> bool:
        return any(marker in text for marker in markers)

    def _compact_message(self, text: str, *, max_chars: int = 220) -> str:
        flattened = " ".join(str(text or "").split())
        if len(flattened) <= max_chars:
            return flattened
        return flattened[: max_chars - 1].rstrip(" ，,；;：:") + "…"

    def _history_summary(self, history: list[tuple[str, str]], *, max_turns: int = 4, max_chars: int = 900) -> str:
        if not history:
            return "无"
        lines: list[str] = []
        for index, (user_text, assistant_text) in enumerate(history[-max_turns:], start=max(1, len(history) - max_turns + 1)):
            lines.append(f"第{index}轮用户：{self._compact_message(user_text, max_chars=160)}")
            lines.append(f"第{index}轮助手：{self._compact_message(assistant_text, max_chars=220)}")
        summary = "\n".join(lines)
        if len(summary) <= max_chars:
            return summary
        return summary[: max_chars - 1].rstrip() + "…"

    def _trim_scratchpad(self, scratchpad: str, *, max_chars: int = 6800) -> str:
        compact = str(scratchpad or "").strip()
        if len(compact) <= max_chars:
            return compact
        marker = "Observation: [已压缩较早的中间记录，仅保留最近步骤以控制上下文长度]"
        tail_budget = max(max_chars - len(marker) - 1, 200)
        tail = compact[-tail_budget:]
        thought_anchor = tail.find("Thought:")
        if thought_anchor > 0:
            tail = tail[thought_anchor:]
        return marker + "\n" + tail.lstrip()

    def _parse_json_object(self, text: str) -> dict[str, Any] | None:
        cleaned = str(text or "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _default_turn_analysis(self, question: str, history: list[tuple[str, str]]) -> dict[str, Any]:
        root_question, facts = self._extract_question_context(question)
        return {
            "current_input_role": "supplement" if facts else "new_question",
            "user_goal": self._compact_message(root_question, max_chars=120),
            "needs_history": bool(history),
            "history_usage": "若当前输入明显是在补充上一轮问题，则结合历史；否则以当前问题为主。",
            "requires_precise_result": self._question_requests_precise_result(question),
            "preferred_answer_style": "brief_direct" if len(root_question) <= 80 else "structured_explanation",
            "likely_missing_info": [],
            "recommended_next_step": "unknown",
        }

    def _analyze_user_turn(self, question: str, history: list[tuple[str, str]]) -> dict[str, Any]:
        default_analysis = self._default_turn_analysis(question, history)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是法律 Agent 的回合分析器。你的任务不是直接回答法律问题，而是先理解最新用户输入。"
                    "你必须先分析最新输入本身，再判断是否需要结合历史内容。"
                    "只有当最新输入明显是在补充、回答、纠正或追问上一轮问题时，才把历史作为强相关上下文。"
                    "请只输出 JSON，不要输出任何解释。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"最新用户输入：\n{question}\n\n"
                    f"最近对话历史摘要：\n{self._history_summary(history)}\n\n"
                    "请输出 JSON，字段固定为："
                    "current_input_role, user_goal, needs_history, history_usage, requires_precise_result, preferred_answer_style, likely_missing_info, recommended_next_step。"
                    "其中 current_input_role 取值建议为 new_question/supplement/answer_to_followup/correction/unclear；"
                    "recommended_next_step 取值建议为 direct_answer/retrieve_from_kb/ask_user/calculator/lookup_statute/unclear。"
                ),
            },
        ]
        output = self.model.generate(
            messages,
            max_new_tokens=256,
            temperature=0.0,
            top_p=1.0,
            top_k=self.top_k,
            presence_penalty=1.0,
            enable_thinking=False,
        )
        payload = self._parse_json_object(output.content or output.raw_text)
        if payload is None:
            return default_analysis

        analysis = dict(default_analysis)
        analysis.update(payload)
        analysis["user_goal"] = self._compact_message(str(analysis.get("user_goal") or default_analysis["user_goal"]), max_chars=120)
        likely_missing_info = analysis.get("likely_missing_info")
        if isinstance(likely_missing_info, str):
            analysis["likely_missing_info"] = [item.strip() for item in likely_missing_info.split("|") if item.strip()]
        elif not isinstance(likely_missing_info, list):
            analysis["likely_missing_info"] = []
        analysis["needs_history"] = bool(analysis.get("needs_history"))
        analysis["requires_precise_result"] = bool(analysis.get("requires_precise_result"))
        analysis["history_usage"] = self._compact_message(str(analysis.get("history_usage") or default_analysis["history_usage"]), max_chars=180)
        analysis["recommended_next_step"] = str(analysis.get("recommended_next_step") or "unknown")
        analysis["current_input_role"] = str(analysis.get("current_input_role") or default_analysis["current_input_role"])
        analysis["preferred_answer_style"] = str(analysis.get("preferred_answer_style") or default_analysis["preferred_answer_style"])
        return analysis

    def _render_turn_analysis(self, state: dict[str, Any]) -> str:
        analysis = dict(state.get("turn_analysis") or {})
        likely_missing_info = analysis.get("likely_missing_info") or []
        missing_text = "、".join(str(item) for item in likely_missing_info[:4]) if likely_missing_info else "无"
        return (
            "对最新用户输入的首轮理解如下：\n"
            f"- 输入性质：{analysis.get('current_input_role', 'unknown')}\n"
            f"- 用户当前目标：{analysis.get('user_goal', '')}\n"
            f"- 是否需要结合历史：{'是' if analysis.get('needs_history') else '否'}\n"
            f"- 历史使用方式：{analysis.get('history_usage', '')}\n"
            f"- 是否要求精确结果：{'是' if analysis.get('requires_precise_result') else '否'}\n"
            f"- 建议回答风格：{analysis.get('preferred_answer_style', 'brief_direct')}\n"
            f"- 可能仍缺的信息：{missing_text}\n"
            f"- 当前建议优先动作：{analysis.get('recommended_next_step', 'unknown')}"
        )

    def _compact_text_for_retrieval(self, text: str) -> str:
        raw = str(text or "").strip()
        if len(raw) <= 220 and raw.count("\n") <= 4:
            return raw

        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        kept: list[str] = []
        for line in lines[:6]:
            tentative = " ".join(kept + [line]).strip()
            if tentative and len(tentative) > 220:
                break
            kept.append(line)

        compact = " ".join(kept).strip() or raw[:220].strip()
        return compact[:220].rstrip(" ，,;；")

    def _sanitize_statute_title(self, title: str) -> str:
        cleaned = str(title or "").strip()
        if "::" in cleaned:
            cleaned = cleaned.split("::", maxsplit=1)[0].strip()
        return cleaned

    def _extract_question_context(self, question: str) -> tuple[str, str]:
        question_text = str(question or "").strip()
        root_question = question_text
        facts = ""

        if "原始法律问题：" in question_text:
            match = re.search(r"原始法律问题：([^\n]+)", question_text)
            if match:
                root_question = match.group(1).strip() or root_question

        fact_pairs = self._supplemental_fact_pairs(question_text)
        if fact_pairs:
            facts = " ".join(answer for _, answer in fact_pairs if answer)

        return root_question, facts

    def _looks_like_clarification_question(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        if not normalized:
            return False
        markers = (
            "请",
            "补充",
            "说明",
            "确认",
            "是否",
            "有无",
            "何时",
            "多少",
            "哪",
            "谁",
            "责任",
            "比例",
            "标准",
            "税率",
            "税种",
            "税则",
            "完税价格",
            "汇率",
            "金额",
            "数额",
        )
        return len(normalized) <= 120 and any(marker in normalized for marker in markers)

    def _supplemental_fact_pairs(self, question: str) -> list[tuple[str, str]]:
        question_text = str(question or "")
        if "已补充事实：" not in question_text:
            return []

        _, facts_block = question_text.split("已补充事实：", maxsplit=1)
        fact_pairs: list[tuple[str, str]] = []
        for line in facts_block.splitlines():
            cleaned = line.strip().lstrip("- ").strip()
            if not cleaned:
                continue

            prompt = ""
            answer = cleaned
            for separator in ("：", ":"):
                if separator not in cleaned:
                    continue
                maybe_prompt, maybe_answer = cleaned.split(separator, maxsplit=1)
                maybe_prompt = maybe_prompt.strip()
                maybe_answer = maybe_answer.strip()
                if maybe_prompt and maybe_answer and self._looks_like_clarification_question(maybe_prompt):
                    prompt = maybe_prompt
                    answer = maybe_answer
                    break

            fact_pairs.append((prompt, answer))
        return fact_pairs

    def _significant_tokens(self, text: str) -> list[str]:
        tokens: list[str] = []
        for token in simple_tokenize(text):
            cleaned = self._normalize_text(token)
            if not cleaned:
                continue
            if cleaned in QUERY_STOPWORDS:
                continue
            if cleaned.isdigit() and len(cleaned) < 2:
                continue
            if len(cleaned) == 1 and not cleaned.isdigit():
                continue
            tokens.append(cleaned)
        return tokens

    def _extract_search_keywords(self, question: str) -> list[str]:
        candidates: list[str] = []
        try:
            import jieba.analyse

            candidates.extend(jieba.analyse.extract_tags(question, topK=8, withWeight=False))
        except Exception:
            pass
        if not candidates:
            candidates.extend(self._significant_tokens(question))
        candidates.extend(re.findall(r"\d+(?:\.\d+)?[%％]?", question))

        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            cleaned = self._normalize_text(candidate)
            if not cleaned or cleaned in seen or cleaned in QUERY_STOPWORDS:
                continue
            seen.add(cleaned)
            deduped.append(candidate.strip())
        return deduped[:10]

    def _tool_signature(self, tool_name: str | None, tool_args: dict[str, Any] | None) -> str:
        return json.dumps(
            {
                "tool_name": str(tool_name or ""),
                "tool_args": tool_args or {},
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _identical_tool_call_count(self, state: dict[str, Any], tool_name: str | None, tool_args: dict[str, Any] | None) -> int:
        signature = self._tool_signature(tool_name, tool_args)
        count = 0
        for item in state.get("tool_history", []):
            if self._tool_signature(item.get("tool_name"), item.get("tool_args")) == signature:
                count += 1
        return count

    def _question_requests_precise_result(self, question: str) -> bool:
        normalized = self._normalize_text(question)
        markers = (
            "精确",
            "多少",
            "几",
            "几年",
            "金额",
            "税额",
            "罚款",
            "赔偿",
            "补交",
            "具体怎么算",
            "具体计算",
            "准确",
            "具体数额",
        )
        return any(marker in normalized for marker in markers)

    def _question_is_computation_request(self, question: str) -> bool:
        normalized = self._normalize_text(question)
        markers = (
            "算",
            "计算",
            "税额",
            "罚款",
            "赔偿",
            "补偿",
            "利息",
            "金额",
            "数额",
            "补交",
            "多少税",
        )
        return any(marker in normalized for marker in markers)

    def _numeric_fact_count(self, question: str) -> int:
        _, facts = self._extract_question_context(question)
        target = facts or question
        return len(re.findall(r"\d+(?:\.\d+)?[%％]?", target))

    def _normalize_clarification_question(self, question: str) -> str:
        normalized = "\n".join(line.strip() for line in str(question or "").splitlines() if line.strip())
        normalized = normalized.strip()
        if not normalized:
            return "请补充当前最影响结论的一项关键信息。"
        if normalized.endswith(("。", "？", "?", "！", "!")):
            return normalized
        if normalized.startswith(("请", "是否", "有无", "能否", "哪一", "哪种", "何时", "多少", "几")):
            return normalized + "？"
        return normalized + "。"

    def _answer_looks_like_statute_dump(self, answer: str) -> bool:
        text = str(answer or "").strip()
        article_hits = len(re.findall(r"第[一二三四五六七八九十百千万\d]+条", text))
        return len(text) >= 260 and article_hits >= 4

    def _has_meaningful_followup_answer(self, state: dict[str, Any]) -> bool:
        analysis = dict(state.get("turn_analysis") or {})
        current_input_role = str(analysis.get("current_input_role") or "")
        if current_input_role not in {"supplement", "answer_to_followup", "correction"}:
            return False

        question = str(state.get("question", "") or "").strip()
        if not question:
            return False
        if self._current_input_has_supplemental_facts(question):
            return True
        if re.search(r"\d+(?:\.\d+)?", question):
            return True
        return len(self._significant_tokens(question)) >= 3

    def _has_contextual_retrieval_evidence(self, state: dict[str, Any]) -> bool:
        tool_names = {str(item.get("tool_name") or "") for item in state.get("tool_history", [])}
        return bool(tool_names & {"retrieve_from_kb", "lookup_statute", "calculator"})

    def _should_skip_precise_result_review(self, state: dict[str, Any], answer: str) -> bool:
        if not self._clarification_questions(state) or not self._has_meaningful_followup_answer(state):
            return False
        if self._contains_any(answer, UNCERTAINTY_MARKERS) or self._answer_looks_like_statute_dump(answer):
            return False
        return True

    def _should_review_draft_answer(self, state: dict[str, Any], answer: str) -> tuple[bool, str]:
        analysis = dict(state.get("turn_analysis") or {})
        if self._contains_any(answer, UNCERTAINTY_MARKERS):
            return True, "答案中出现了明显不确定表述，需要判断是否应继续规划。"
        if bool(analysis.get("requires_precise_result")):
            if self._should_skip_precise_result_review(state, answer):
                return False, ""
            return True, "用户要求精确结果，需要核查当前答案是否真正满足了精确回答需求。"
        if analysis.get("preferred_answer_style") == "brief_direct" and self._answer_looks_like_statute_dump(answer):
            return True, "当前答案疑似只是罗列法条，尚未直接回应用户最关心的问题。"
        return False, ""

    def _plan_recovery_step_with_llm(self, state: dict[str, Any], draft_answer: str, reason: str) -> ParsedStep:
        root_question, facts = self._extract_question_context(str(state.get("question", "")))
        prior_questions = self._clarification_questions(state)
        scratchpad = str(state.get("scratchpad", "")).strip()
        analysis_block = self._render_turn_analysis(state)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是中国法律智能体。你已经看到了当前窗口中的完整对话历史、最新用户输入，以及你自己已有的 Thought/Observation 记录。"
                    "你刚刚给出的暂定答案可能还没有真正满足用户需求。"
                    "现在请你重新判断下一步最合理的动作，可以是：直接输出更好的 Final Answer，或者输出一个新的 Thought + Action。"
                    "如果你选择 ask_user，问题必须紧扣当前案情和已有 Observation，自主生成，禁止套用通用模板。"
                    "如果你选择 retrieve_from_kb、lookup_statute 或 calculator，动作必须直接服务于当前用户问题，禁止复制历史示例中的查询词。"
                    "禁止输出 Observation。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"原始问题：{root_question}\n"
                    f"补充事实：{facts or '无'}\n"
                    f"本轮输入理解：\n{analysis_block}\n"
                    f"已有 Thought/Observation：\n{scratchpad or '无'}\n"
                    f"你刚刚写出的暂定最终答案：{draft_answer}\n"
                    f"之前已经追问过的问题：{prior_questions or '无'}\n"
                    f"需要重新规划的原因：{reason}\n"
                    "请你自己决定：如果当前答案已经足够好，就直接输出 Final Answer；否则输出 Thought 和一个最合适的 Action。"
                ),
            },
        ]
        output = self.model.generate(
            messages,
            max_new_tokens=self.step_max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            presence_penalty=self.presence_penalty,
            enable_thinking=False,
        )
        parsed = parse_react_output(output.content or output.raw_text)
        if parsed.kind == "tool":
            tool_args = dict(parsed.tool_args or {})
            if parsed.tool_name == "ask_user":
                tool_args["question"] = self._normalize_clarification_question(str(tool_args.get("question") or ""))
                if not str(tool_args.get("field_name") or "").strip():
                    tool_args["field_name"] = f"followup_facts_{len(prior_questions) + 1}"
            parsed.tool_args = tool_args
            return parsed
        if parsed.kind == "final" and parsed.final_answer:
            return parsed
        return ParsedStep(kind="final", thought="现有信息下不再额外扩展动作，直接给出条件式分析。", final_answer=draft_answer)

    def _final_to_followup_or_final(self, state: dict[str, Any], thought: str, answer: str) -> ParsedStep:
        prior_questions = self._clarification_questions(state)
        if len(prior_questions) >= self.max_followup_questions:
            return ParsedStep(kind="final", thought=thought, final_answer=answer)

        should_review, reason = self._should_review_draft_answer(state, answer)
        if not should_review:
            return ParsedStep(kind="final", thought=thought, final_answer=answer)

        next_step = self._plan_recovery_step_with_llm(state, answer, reason)
        if next_step.kind != "tool":
            final_answer = next_step.final_answer or answer
            return ParsedStep(kind="final", thought=thought, final_answer=final_answer)

        if next_step.tool_name != "ask_user":
            return next_step

        followup_question = self._normalize_clarification_question(
            str((next_step.tool_args or {}).get("question") or "")
        )
        if any(self._question_overlap_ratio(followup_question, previous) >= 0.55 for previous in prior_questions):
            return ParsedStep(kind="final", thought=thought, final_answer=answer)

        next_step.tool_args = dict(next_step.tool_args or {})
        next_step.tool_args["question"] = followup_question
        return next_step

    def _current_input_has_supplemental_facts(self, question: str) -> bool:
        _, facts = self._extract_question_context(question)
        if not facts:
            return False
        if re.search(r"\d+(?:\.\d+)?", facts):
            return True
        return len(self._significant_tokens(facts)) >= 3

    def _embedded_clarification_questions(self, question: str) -> list[str]:
        return [question_text for question_text, _ in self._supplemental_fact_pairs(question) if question_text]

    def _has_location_clarification(self, state: dict[str, Any]) -> bool:
        markers = ("所在地", "居住地", "省、市", "区县", "街道", "乡镇", "哪个省", "哪个市", "行政区")
        for question in self._clarification_questions(state):
            if any(marker in question for marker in markers):
                return True
        return False

    def _clarification_questions(self, state: dict[str, Any]) -> list[str]:
        questions: list[str] = []
        for item in state.get("tool_history", []):
            if item.get("tool_name") != "ask_user":
                continue
            tool_args = item.get("tool_args") or {}
            question = str(tool_args.get("question") or "").strip()
            if question:
                questions.append(question)
        questions.extend(self._embedded_clarification_questions(str(state.get("question", ""))))
        return questions

    def _question_overlap_ratio(self, left: str, right: str) -> float:
        left_tokens = set(self._significant_tokens(left))
        right_tokens = set(self._significant_tokens(right))
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = left_tokens & right_tokens
        universe = left_tokens | right_tokens
        return len(overlap) / max(len(universe), 1)

    def _build_retrieval_query(self, question: str) -> str:
        root_question, facts = self._extract_question_context(question)
        compact_root_question = self._compact_text_for_retrieval(root_question)
        keyword_source = root_question[:1200] if len(root_question) > 1200 else root_question
        combined = " ".join(part for part in [keyword_source, facts] if part)
        query_parts = [compact_root_question]
        if facts:
            query_parts.append(facts)
        query_parts.extend(self._extract_search_keywords(combined))

        seen: set[str] = set()
        deduped_parts: list[str] = []
        for part in query_parts:
            cleaned = part.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            deduped_parts.append(cleaned)
        return " ".join(deduped_parts)

    def _build_retrieval_query_from_state(self, state: dict[str, Any]) -> str:
        question = str(state.get("question", ""))
        analysis = dict(state.get("turn_analysis") or {})
        current_input_role = str(analysis.get("current_input_role") or "")
        if current_input_role not in {"supplement", "answer_to_followup", "correction"}:
            return self._build_retrieval_query(question)
        history = list(state.get("history", []))
        if not history:
            return self._build_retrieval_query(question)

        last_user_question = str(history[-1][0] or "").strip()
        if not last_user_question:
            return self._build_retrieval_query(question)

        combined_question = (
            f"原始法律问题：{last_user_question}\n"
            "用户刚刚补充了以下事实，请在同一问题上继续分析；若这些事实已经足够支撑条件式分析，请直接给出结论，不要机械重复 ask_user。\n"
            "已补充事实：\n"
            f"- {question}"
        )
        return self._build_retrieval_query(combined_question)

    def _is_location_followup_question(self, question: str) -> bool:
        normalized = self._normalize_text(question)
        if not normalized:
            return False
        markers = (
            "所在地",
            "居住地",
            "住址",
            "地址",
            "地点",
            "省",
            "市",
            "区县",
            "区",
            "县",
            "街道",
            "乡镇",
            "位置",
        )
        return any(marker in normalized for marker in markers)

    def _current_question_has_precise_location(self, state: dict[str, Any]) -> bool:
        retriever = getattr(self.registry, "retriever", None)
        if retriever is None or not hasattr(retriever, "inspect_query"):
            return False
        try:
            query_context = retriever.inspect_query(str(state.get("question", "")))
        except Exception:
            return False

        explicit_level = getattr(query_context, "explicit_region_level", None)
        if explicit_level in {"county", "town", "village"}:
            return True

        resolution = getattr(query_context, "location_resolution", None)
        if resolution is None:
            return False
        return bool(
            getattr(resolution, "county_name", None)
            or getattr(resolution, "town_name", None)
            or getattr(resolution, "village_name", None)
        )

    def _build_messages(self, state: dict[str, Any]) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        for user_text, assistant_text in state.get("history", []):
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
        analysis_block = self._render_turn_analysis(state)
        current_question = (
            "你已经看到了当前窗口中的完整对话历史。以下输入可能是对之前问题的补充、追问或省略表达。"
            "如果它是在补充你上一轮主动追问的事实，必须把它并回同一法律问题继续分析。"
            "你还会看到自己已有的 Thought/Observation 记录；你必须先统一理解这些上下文，再决定下一步是直接回答、检索、追问还是计算。"
            "若现有事实已足够支撑条件式分析，应直接给出结论，不要重复 ask_user。\n"
            f"当前用户输入：{state['question']}\n\n"
            f"{analysis_block}"
        )
        messages.append({"role": "user", "content": current_question})
        scratchpad = state.get("scratchpad", "").strip()
        if scratchpad:
            messages.append({"role": "assistant", "content": scratchpad})
            messages.append({"role": "user", "content": continue_instruction()})
        return messages

    def _postprocess_parsed_step(self, state: dict[str, Any], parsed: ParsedStep) -> ParsedStep:
        if parsed.kind == "final" and parsed.final_answer:
            return self._final_to_followup_or_final(state, parsed.thought, parsed.final_answer)

        if parsed.kind != "tool":
            return parsed

        if parsed.tool_name == "lookup_statute":
            tool_args = dict(parsed.tool_args or {})
            tool_args["title"] = self._sanitize_statute_title(str(tool_args.get("title") or ""))
            parsed.tool_args = tool_args

        if self._identical_tool_call_count(state, parsed.tool_name, parsed.tool_args) >= 1:
            return self._final_to_followup_or_final(
                state,
                "同一工具调用已重复，改为基于现有证据收束，并在必要时只追问一次关键事实。",
                self._synthesize_final_answer(state, f"工具 {parsed.tool_name} 重复执行未带来新信息"),
            )

        if parsed.tool_name == "retrieve_from_kb":
            tool_args = dict(parsed.tool_args or {})
            tool_args["query"] = str(tool_args.get("query") or self._build_retrieval_query_from_state(state)).strip()
            tool_args["top_k"] = int(tool_args.get("top_k", 6))
            parsed.tool_args = tool_args
            return parsed

        if parsed.tool_name != "ask_user":
            return parsed

        tool_args = dict(parsed.tool_args or {})
        question = self._normalize_clarification_question(str(tool_args.get("question") or ""))
        if self._is_location_followup_question(question) and self._current_question_has_precise_location(state):
            return ParsedStep(
                kind="tool",
                thought="用户已经补充到足够细的地点信息，直接检索并继续分析。",
                tool_name="retrieve_from_kb",
                tool_args={"query": self._build_retrieval_query_from_state(state), "top_k": 6},
            )
        tool_args["question"] = question
        if not str(tool_args.get("field_name") or "").strip():
            tool_args["field_name"] = f"followup_facts_{len(self._clarification_questions(state)) + 1}"
        parsed.tool_args = tool_args

        prior_questions = self._clarification_questions(state)
        repeated = any(self._question_overlap_ratio(question, previous) >= 0.55 for previous in prior_questions)
        if repeated or len(prior_questions) >= self.max_followup_questions:
            return ParsedStep(
                kind="final",
                thought="关键事实已追问过，转为基于现有信息给出条件式分析并明确不确定项。",
                final_answer=self._synthesize_final_answer(state, "已达到通用追问上限或出现重复追问"),
            )
        if (
            prior_questions
            and self._has_meaningful_followup_answer(state)
            and self._has_contextual_retrieval_evidence(state)
        ):
            return ParsedStep(
                kind="final",
                thought="用户已补充关键事实，且已经完成针对性检索，转为基于现有证据给出条件式分析。",
                final_answer=self._synthesize_final_answer(state, "补充事实后已完成检索，不再继续追加新的澄清问题"),
            )

        return parsed

    def _synthesize_final_answer(self, state: dict[str, Any], reason: str) -> str:
        scratchpad = str(state.get("scratchpad", "")).strip()
        if not scratchpad:
            return "当前信息仍不足以形成可靠法律结论。请补充更具体的事实，尤其是时间、金额、行为方式或涉及主体。"

        root_question, facts = self._extract_question_context(str(state.get("question", "")))
        messages = [
            {
                "role": "system",
                "content": (
                    "你是中国法律智能体。现在禁止再调用任何工具。"
                    "你只能依据已有事实和已有 Observation，直接写出最终中文答复。"
                    "如果事实仍不完整，请给条件式分析，不要重复追问相同问题。"
                    "答复必须包含：已知事实、适用依据、结论、仍需确认事项。"
                    "先给一句最直接的结论，再补充最关键的 1 到 3 条依据或限制条件。"
                    "除非用户明确要求详细展开，否则不要连续大段抄写法条原文。"
                    "不要输出 Thought、Action、Observation 或 Final Answer 标签。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"原始问题：{root_question}\n"
                    f"补充事实：{facts or '无'}\n"
                    f"已有推理与工具结果：\n{scratchpad}\n"
                    f"当前需要收束的原因：{reason}\n"
                    "请直接给出最终答复。"
                ),
            },
        ]
        output = self.model.generate(
            messages,
            max_new_tokens=self.final_max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=self.top_k,
            presence_penalty=1.0,
            enable_thinking=False,
        )
        answer_text = (output.content or output.raw_text).strip()
        parsed = parse_react_output(answer_text)
        if parsed.kind == "final" and parsed.final_answer:
            return parsed.final_answer.strip()
        cleaned = answer_text.replace("Final Answer:", "").strip()
        return cleaned or "基于当前已检索到的法规依据，仍需补充更具体的事实后才能给出更精确的法律结论。"

    def _format_tool_step(self, parsed: ParsedStep) -> str:
        lines = []
        if parsed.thought:
            lines.append(f"Thought: {parsed.thought}")
        payload = json.dumps(parsed.tool_args or {}, ensure_ascii=False, sort_keys=True)
        lines.append(f"Action: {parsed.tool_name}({payload})")
        return "\n".join(lines)

    def _format_final_step(self, parsed: ParsedStep) -> str:
        lines = []
        if parsed.thought:
            lines.append(f"Thought: {parsed.thought}")
        lines.append(f"Final Answer: {parsed.final_answer}")
        return "\n".join(lines)

    def _apply_parsed_step(self, state: dict[str, Any], parsed: ParsedStep, raw_output: str) -> dict[str, Any]:
        parsed = self._postprocess_parsed_step(state, parsed)
        scratchpad = state.get("scratchpad", "")
        new_state = dict(state)
        appended_text = ""
        if parsed.kind == "tool":
            appended_text = self._format_tool_step(parsed)
        elif parsed.kind == "final" and parsed.final_answer:
            appended_text = self._format_final_step(parsed)

        if appended_text:
            if scratchpad:
                scratchpad = scratchpad.rstrip() + "\n" + appended_text.strip()
            else:
                scratchpad = appended_text.strip()
        scratchpad = self._trim_scratchpad(scratchpad)
        new_state.update(
            {
                "scratchpad": scratchpad,
                "raw_output": raw_output,
                "llm_retry_count": 0 if parsed.kind in {"tool", "final"} else int(state.get("llm_retry_count", 0)),
                "parsed_kind": parsed.kind,
                "parsed_payload": {
                    "thought": parsed.thought,
                    "tool_name": parsed.tool_name,
                    "tool_args": parsed.tool_args,
                    "final_answer": parsed.final_answer,
                    "error": parsed.error,
                },
            }
        )
        if parsed.kind == "final" and parsed.final_answer:
            new_state["final_answer"] = parsed.final_answer
        return new_state

    def _iter_llm_step(self, state: dict[str, Any]) -> Iterator[dict[str, Any]]:
        messages = self._build_messages(state)
        last_output = None
        for partial in self.model.stream_generate(
            messages,
            max_new_tokens=self.step_max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            presence_penalty=self.presence_penalty,
            enable_thinking=self.enable_thinking,
        ):
            last_output = partial
            yield {"event": "llm_partial", "draft": partial.content or partial.raw_text}

        if last_output is None:
            output = self.model.generate(
                messages,
                max_new_tokens=self.step_max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                presence_penalty=self.presence_penalty,
                enable_thinking=self.enable_thinking,
            )
            last_output = output

        raw_output = last_output.raw_text
        parsed = parse_react_output(last_output.content or last_output.raw_text)
        yield {"event": "state", "state": self._apply_parsed_step(state, parsed, raw_output)}

    def _compose_live_trace(self, scratchpad: str, draft: str) -> str:
        preview = "[Live Draft]\n" + draft.strip()
        if scratchpad.strip():
            return scratchpad.rstrip() + "\n\n" + preview
        return preview

    def _live_message_from_draft(self, draft: str) -> str:
        action_match = re.search(r"Action:\s*([a-zA-Z_][\w]*)", draft)
        if action_match:
            return f"正在准备调用工具：{action_match.group(1)}"
        if "Final Answer:" in draft:
            return "正在整理最终答复。"
        if "Thought:" in draft:
            return "正在思考并规划下一步。"
        return "正在分析问题。"

    def _llm_node(self, state: dict[str, Any]) -> dict[str, Any]:
        for event in self._iter_llm_step(state):
            if event["event"] == "state":
                return event["state"]
        return state

    def _route_after_llm(self, state: dict[str, Any]) -> str:
        if state.get("skip_tools"):
            return "final"
        if state.get("parsed_kind") == "final":
            return "final"
        if state.get("parsed_kind") == "tool":
            return "tool"

        retries = int(state.get("llm_retry_count", 0)) + 1
        state["llm_retry_count"] = retries
        error_text = state.get("parsed_payload", {}).get("error") or "模型输出未遵守格式。"
        errors = list(state.get("errors", []))
        errors.append(error_text)
        state["errors"] = errors
        state["scratchpad"] = self._trim_scratchpad(
            state.get("scratchpad", "").rstrip() + "\nObservation: FORMAT_ERROR: 请严格输出 Thought/Action 或 Final Answer。"
        )
        if retries >= self.max_llm_retries or state.get("step_count", 0) >= state.get("max_steps", self.max_steps):
            state["final_answer"] = self._synthesize_final_answer(state, "模型多次未按 ReAct 格式输出")
            return "final"
        return "retry"

    def _tool_node(self, state: dict[str, Any]) -> dict[str, Any]:
        payload = state.get("parsed_payload", {})
        tool_name = payload.get("tool_name")
        tool_args = payload.get("tool_args") or {}
        new_state = dict(state)

        try:
            result = self.registry.execute(str(tool_name), tool_args)
        except Exception as exc:
            result = {"error": str(exc), "tool_name": tool_name, "tool_args": tool_args}
            new_state["errors"] = list(state.get("errors", [])) + [str(exc)]

        new_state["scratchpad"] = self._trim_scratchpad(
            state.get("scratchpad", "").rstrip() + "\nObservation: " + observation_to_text(result)
        )
        new_state["tool_history"] = list(state.get("tool_history", [])) + [
            {"tool_name": tool_name, "tool_args": tool_args, "result": result}
        ]
        new_state["step_count"] = int(state.get("step_count", 0)) + 1
        if (
            tool_name == "retrieve_from_kb"
            and result.get("needs_location_clarification")
            and not self._has_location_clarification(state)
            and len(self._clarification_questions(state)) < self.max_followup_questions
        ):
            clarification_question = str(
                result.get("location_clarification_question")
                or "请补充你所在的省、市、区县；如果知道更具体位置，也可以直接补充到街道，以便判断地方性法规是否适用。"
            )
            new_state["needs_user_input"] = True
            new_state["clarification_question"] = clarification_question
            new_state["final_answer"] = f"为继续分析，请先补充：{clarification_question}"
            return new_state
        if result.get("status") == "pending_user_input":
            clarification_question = str(result.get("question") or tool_args.get("question") or "请补充关键事实。")
            new_state["needs_user_input"] = True
            new_state["clarification_question"] = clarification_question
            new_state["final_answer"] = f"为继续分析，请先补充：{clarification_question}"
        return new_state

    def _route_after_tool(self, state: dict[str, Any]) -> str:
        if state.get("needs_user_input"):
            return "final"
        if state.get("step_count", 0) >= state.get("max_steps", self.max_steps):
            state["final_answer"] = self._synthesize_final_answer(state, "已达到最大工具步数")
            return "final"
        return "llm"
