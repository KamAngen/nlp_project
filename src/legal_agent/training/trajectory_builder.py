from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any

from legal_agent.agent.prompting import build_system_prompt
from legal_agent.agent.tools import ToolRegistry, observation_to_text
from legal_agent.rag.retriever import HybridLegalRetriever
from legal_agent.study_tools import StudyToolExecutor
from legal_agent.unified_tools import UnifiedToolRegistry


REFERENCE_TITLE_RE = re.compile(r"《([^》]+)》")


@dataclass(slots=True)
class TrajectorySeed:
    seed_id: str
    question: str
    expected_answer: str
    source: str
    sampling_bucket: str = ""
    references: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    scripted_answers: dict[str, str] = field(default_factory=dict)
    clarification_questions: dict[str, str] = field(default_factory=dict)
    calculator_expression: str | None = None
    query_for_retrieval: str | None = None
    force_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def extract_reference_titles(references: list[str]) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for ref in references:
        match = REFERENCE_TITLE_RE.search(ref)
        if match is None:
            continue
        title = match.group(1).strip()
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


class TrajectoryBuilder:
    def __init__(
        self,
        retriever: HybridLegalRetriever,
        *,
        study_tool_executor: StudyToolExecutor | None = None,
        user_id: str = "trajectory_builder",
        session_id: str = "builder_session",
    ) -> None:
        self.retriever = retriever
        self.study_tool_executor = study_tool_executor
        self.user_id = user_id
        self.session_id = session_id

    def build_example(self, seed: TrajectorySeed) -> dict[str, Any]:
        registry = self._build_registry(seed)
        steps = self._plan(seed)
        executed_steps = self._execute_steps(steps, registry)
        trace = self._render_trace(seed, executed_steps, registry)
        expected_tools = [step["tool_name"] for step in executed_steps]
        return {
            "sample_id": seed.seed_id,
            "source": seed.source,
            "question": seed.question,
            "expected_answer": seed.expected_answer,
            "references": seed.references,
            "expected_tools": expected_tools,
            "force_error": seed.force_error,
            "scripted_answers": seed.scripted_answers,
            "metadata": seed.metadata,
            "messages": [
                {"role": "system", "content": build_system_prompt(registry.tool_definitions(), stepwise=False)},
                {"role": "user", "content": seed.question},
                {"role": "assistant", "content": trace},
            ],
            "trace": trace,
            "tool_trace": executed_steps,
        }

    def _build_registry(self, seed: TrajectorySeed) -> ToolRegistry:
        if self.study_tool_executor is None:
            return ToolRegistry(self.retriever, scripted_answers=seed.scripted_answers, interactive=False)
        token = re.sub(r"[^a-zA-Z0-9]+", "-", seed.seed_id).strip("-")[:40] or "seed"
        return UnifiedToolRegistry(
            self.retriever,
            study_tool_executor=self.study_tool_executor,
            user_id=f"{self.user_id}-{token}",
            session_id=f"{self.session_id}-{token}",
            scripted_answers=seed.scripted_answers,
            interactive=False,
        )

    def _plan(self, seed: TrajectorySeed) -> list[dict[str, Any]]:
        scenario = str(seed.metadata.get("scenario") or "")
        if scenario:
            return self._plan_study_scenario(seed)

        reference_titles = extract_reference_titles(seed.references)
        steps: list[dict[str, Any]] = []
        requires_lookup = bool(seed.metadata.get("requires_lookup", bool(reference_titles)))
        requires_retrieval = bool(seed.metadata.get("requires_retrieval", True))

        if seed.force_error and requires_lookup:
            broken_title = (reference_titles[0] + "（错误标题）") if reference_titles else "不存在的法规标题"
            steps.append(
                {
                    "thought": "先尝试按标题精确定位法规。",
                    "tool_name": "lookup_statute",
                    "tool_args": {"title": broken_title},
                }
            )

        for field_name, question in seed.clarification_questions.items():
            steps.append(
                {
                    "thought": "当前缺少关键事实，需要先补齐。",
                    "tool_name": "ask_user",
                    "tool_args": {"question": question, "field_name": field_name},
                }
            )

        if requires_lookup and reference_titles:
            steps.append(
                {
                    "thought": "需要确认核心法规的元数据与名称。",
                    "tool_name": "lookup_statute",
                    "tool_args": {"title": reference_titles[0]},
                }
            )

        if requires_retrieval:
            steps.append(
                {
                    "thought": "需要从知识库检索可直接引用的法条。",
                    "tool_name": "retrieve_from_kb",
                    "tool_args": {"query": seed.query_for_retrieval or seed.question, "top_k": 5},
                }
            )

        if seed.calculator_expression:
            steps.append(
                {
                    "thought": "还需要做算术计算以形成最终结论。",
                    "tool_name": "calculator",
                    "tool_args": {"expression": seed.calculator_expression},
                }
            )

        return steps

    def _plan_study_scenario(self, seed: TrajectorySeed) -> list[dict[str, Any]]:
        scenario = str(seed.metadata.get("scenario") or "")
        profile_updates = dict(seed.metadata.get("profile_updates") or {})
        topic = str(seed.metadata.get("topic") or seed.query_for_retrieval or seed.question)
        report_type = str(seed.metadata.get("report_type") or "study_progress")
        question_count = int(seed.metadata.get("question_count") or 3)
        steps: list[dict[str, Any]] = []

        if profile_updates:
            steps.append(
                {
                    "thought": "用户在当前轮次透露了新的学习目标或偏好，需要先写回画像。",
                    "tool_name": "profile_upsert",
                    "tool_args": {"raw_text": seed.question, "updates": profile_updates},
                }
            )

        if scenario == "profile_update":
            steps.append(
                {
                    "thought": "更新后需要读取画像，确认当前学习档案。",
                    "tool_name": "profile_view",
                    "tool_args": {},
                }
            )
            return steps

        if scenario in {"study_qa", "study_case_analysis", "study_method_qa", "study_statute_qa"}:
            steps.append(
                {
                    "thought": "先整理当前用户画像、会话摘要和相关历史命中。",
                    "tool_name": "prepare_context",
                    "tool_args": {"query": seed.question},
                }
            )
            steps.append(
                {
                    "thought": "需要综合检索题库、案例库和学习常识，提取可用于讲解的材料。",
                    "tool_name": "rag_search",
                    "tool_args": {"query": seed.query_for_retrieval or seed.question, "top_k": 6},
                }
            )
            if scenario == "study_statute_qa":
                steps.append(
                    {
                        "thought": "这个问题还需要现行法规依据，补充一次法条检索。",
                        "tool_name": "retrieve_from_kb",
                        "tool_args": {"query": seed.query_for_retrieval or seed.question, "top_k": 5},
                    }
                )
            if seed.calculator_expression:
                steps.append(
                    {
                        "thought": "还需要计算分值或时间，补充一个安全计算步骤。",
                        "tool_name": "calculator",
                        "tool_args": {"expression": seed.calculator_expression},
                    }
                )
            return steps

        if scenario == "mock_exam_generate":
            steps.extend(
                [
                    {
                        "thought": "需要先读取画像，了解当前薄弱点和目标科目。",
                        "tool_name": "profile_view",
                        "tool_args": {},
                    },
                    {
                        "thought": "先整理当前画像和会话摘要，决定题目主题和难点。",
                        "tool_name": "prepare_context",
                        "tool_args": {"query": topic},
                    },
                    {
                        "thought": "现在可以为用户生成一套模拟测试。",
                        "tool_name": "generate_exam",
                        "tool_args": {"topic": topic, "question_count": question_count},
                    },
                ]
            )
            return steps

        if scenario == "report_generation":
            steps.extend(
                [
                    {
                        "thought": "先整理当前记忆、画像和会话摘要，为报告收集材料。",
                        "tool_name": "prepare_context",
                        "tool_args": {"query": report_type},
                    },
                    {
                        "thought": "基于当前快照生成结构化学习报告。",
                        "tool_name": "generate_report",
                        "tool_args": {"report_type": report_type},
                    },
                ]
            )
            return steps

        return steps

    def _execute_steps(self, steps: list[dict[str, Any]], registry: ToolRegistry) -> list[dict[str, Any]]:
        executed: list[dict[str, Any]] = []
        for step in steps:
            try:
                result = registry.execute(step["tool_name"], step["tool_args"])
                error = False
            except Exception as exc:
                result = {"error": str(exc), "tool_name": step["tool_name"], "tool_args": step["tool_args"]}
                error = True
            executed.append({**step, "result": result, "error": error})
        return executed

    def _render_trace(self, seed: TrajectorySeed, executed_steps: list[dict[str, Any]], registry: ToolRegistry) -> str:
        return self._build_fallback_trace(seed, executed_steps)

    def _render_action(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        payload = json.dumps(tool_args, ensure_ascii=False, sort_keys=True)
        return f"Action: {tool_name}({payload})"

    def _build_target_final_answer(self, seed: TrajectorySeed, *, max_chars: int = 420) -> str:
        answer = " ".join(seed.expected_answer.replace("\r", "").split())
        if len(answer) <= max_chars:
            return answer

        parts: list[str] = []
        total = 0
        for sentence in re.split(r"(?<=[。！？；])", answer):
            sentence = sentence.strip()
            if not sentence:
                continue
            if total + len(sentence) > max_chars and parts:
                break
            if total + len(sentence) > max_chars:
                return answer[: max_chars - 1].rstrip("，,；;：: ") + "。"
            parts.append(sentence)
            total += len(sentence)
            if total >= int(max_chars * 0.75) and len(parts) >= 2:
                break
        compressed = "".join(parts).strip()
        return compressed or answer[: max_chars - 1].rstrip("，,；;：: ") + "。"

    def _build_llm_guided_trace(
        self,
        seed: TrajectorySeed,
        executed_steps: list[dict[str, Any]],
    ) -> str:
        lines: list[str] = []
        if not executed_steps and not seed.metadata.get("requires_retrieval", True):
            lines.append("Thought: 用户已经在输入中提供了完成任务所需的文本、候选项或标签约束，可以直接基于原文完成任务。")
        for step in executed_steps:
            thought = step["thought"]
            lines.append(f"Thought: {thought}")
            lines.append(self._render_action(step["tool_name"], step["tool_args"]))
            lines.append(f"Observation: {observation_to_text(step['result'])}")
        lines.append(f"Final Answer: {self._build_target_final_answer(seed)}")
        return "\n".join(lines)

    def _build_fallback_trace(self, seed: TrajectorySeed, executed_steps: list[dict[str, Any]]) -> str:
        return self._build_llm_guided_trace(seed, executed_steps)
