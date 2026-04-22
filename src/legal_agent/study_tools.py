from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from uuid import uuid4

from context_engine.manager import MemoryManager
from legal_agent.agent.tools import SafeCalculator
from legal_agent.utils.io import ensure_dir, write_json
from legal_agent.utils.text import truncate_text
from rag_engine.service import KnowledgeService


ANSWER_LINE_RE = re.compile(r"(?P<index>\d+)\s*[\.、:：-]?\s*(?P<answer>[A-Da-d])")


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, str]
    output_schema: dict[str, str]


class StudyToolExecutor:
    def __init__(
        self,
        memory_manager: MemoryManager,
        knowledge_service: KnowledgeService,
        *,
        report_root: str | Path,
    ) -> None:
        self.memory_manager = memory_manager
        self.knowledge_service = knowledge_service
        self.report_root = ensure_dir(report_root)
        self.calculator = SafeCalculator()
        self.specs = {
            "memory_search": ToolSpec(
                "memory_search",
                "检索用户画像、会话记忆和系统全局记忆。",
                {"query": "str", "top_k": "int"},
                {"results": "list", "summary": "str"},
            ),
            "profile_upsert": ToolSpec(
                "profile_upsert",
                "更新用户画像，包括备考目标、薄弱点、偏好等。",
                {"raw_text": "str | null", "updates": "dict | null"},
                {"updates": "dict", "profile": "dict"},
            ),
            "profile_view": ToolSpec(
                "profile_view",
                "查看当前用户画像。",
                {},
                {"profile": "dict"},
            ),
            "rag_search": ToolSpec(
                "rag_search",
                "综合检索法条、题库、案例和常识知识。",
                {"query": "str", "sources": "list | null", "top_k": "int"},
                {"results": "list"},
            ),
            "calculator": ToolSpec(
                "calculator",
                "执行安全数值计算。",
                {"expression": "str"},
                {"result": "float"},
            ),
            "generate_exam": ToolSpec(
                "generate_exam",
                "从题库中抽取题目，生成一套模拟测试。",
                {"topic": "str | null", "question_count": "int", "exam_type": "str | null"},
                {"exam_session_id": "str", "exam_type": "str", "questions": "list"},
            ),
            "score_exam": ToolSpec(
                "score_exam",
                "对用户提交的答案进行评分。",
                {"answers_text": "str", "exam_session_id": "str | null"},
                {"score_percent": "float", "details": "list"},
            ),
            "generate_report": ToolSpec(
                "generate_report",
                "生成用户学习报告。",
                {"report_type": "str"},
                {"report_path": "str", "report_markdown": "str"},
            ),
            "ask_followup": ToolSpec(
                "ask_followup",
                "向用户追问关键信息。",
                {"question": "str", "slot": "str"},
                {"status": "str", "question": "str"},
            ),
        }

    def definitions(self) -> list[dict[str, object]]:
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.input_schema,
                "output_schema": spec.output_schema,
            }
            for spec in self.specs.values()
        ]

    def execute(self, tool_name: str, tool_args: dict[str, object], *, user_id: str, session_id: str) -> dict[str, object]:
        dispatch = {
            "memory_search": lambda: self._memory_search(user_id, session_id, **tool_args),
            "profile_upsert": lambda: self._profile_upsert(user_id, **tool_args),
            "profile_view": lambda: self._profile_view(user_id),
            "rag_search": lambda: self._rag_search(**tool_args),
            "calculator": lambda: self._calculator(**tool_args),
            "generate_exam": lambda: self._generate_exam(user_id, session_id, **tool_args),
            "score_exam": lambda: self._score_exam(user_id, session_id, **tool_args),
            "generate_report": lambda: self._generate_report(user_id, session_id, **tool_args),
            "ask_followup": lambda: self._ask_followup(**tool_args),
        }
        if tool_name not in dispatch:
            raise KeyError(f"Unsupported study tool: {tool_name}")
        return dispatch[tool_name]()

    def _memory_search(self, user_id: str, session_id: str, query: str, top_k: int = 6) -> dict[str, object]:
        bundle = self.memory_manager.assemble_context(query, user_id, session_id)
        results = []
        for layer in ("profile", "system", "working", "episodic", "semantic"):
            for hit in bundle.layer_hits.get(layer, [])[:top_k]:
                results.append(
                    {
                        "layer": layer,
                        "text": hit.item.text,
                        "score": round(hit.score, 4),
                        "reasons": list(hit.reasons),
                    }
                )
        return {"results": results[:top_k], "summary": bundle.summary_blocks["memory"]}

    def _profile_upsert(
        self,
        user_id: str,
        raw_text: str | None = None,
        updates: dict[str, object] | None = None,
    ) -> dict[str, object]:
        normalized_updates = dict(updates or {})
        if raw_text:
            extracted = self.memory_manager.extract_profile_updates(raw_text)
            for key, value in extracted.items():
                if key in normalized_updates and isinstance(normalized_updates[key], list) and isinstance(value, list):
                    normalized_updates[key] = normalized_updates[key] + value
                elif key in normalized_updates and isinstance(normalized_updates[key], dict) and isinstance(value, dict):
                    normalized_updates[key].update(value)
                else:
                    normalized_updates[key] = value
        if not normalized_updates and raw_text:
            normalized_updates = {"notes": [raw_text.strip()]}
        profile = self.memory_manager.update_profile(user_id, normalized_updates, source="study_tool")
        return {"updates": normalized_updates, "profile": profile.to_dict()}

    def _profile_view(self, user_id: str) -> dict[str, object]:
        return {"profile": self.memory_manager.get_user_profile(user_id).to_dict()}

    def _rag_search(
        self,
        query: str,
        sources: list[str] | None = None,
        top_k: int = 6,
    ) -> dict[str, object]:
        hits = self.knowledge_service.search(query, sources=sources, top_k=top_k)
        return {
            "query": query,
            "results": [hit.to_dict() for hit in hits],
        }

    def _calculator(self, expression: str) -> dict[str, object]:
        return {"expression": expression, "result": self.calculator.evaluate(expression)}

    def _generate_exam(
        self,
        user_id: str,
        session_id: str,
        topic: str | None = None,
        question_count: int = 5,
        exam_type: str | None = None,
    ) -> dict[str, object]:
        profile = self.memory_manager.get_user_profile(user_id)
        preferred_tags = list(dict.fromkeys([*profile.weak_points, *profile.study_goals]))
        wrong_question_bank = {
            str(record_id): dict(item)
            for record_id, item in dict(profile.attributes.get("wrong_question_bank") or {}).items()
            if str(record_id).strip() and isinstance(item, dict)
        }
        recent_question_ids = [
            str(record_id)
            for record_id in profile.attributes.get("recent_question_ids", [])
            if str(record_id).strip()
        ]
        effective_exam_type = str(exam_type or "综合练习").strip() or "综合练习"
        questions = self.knowledge_service.sample_questions(
            topic=topic,
            question_count=question_count,
            preferred_tags=preferred_tags,
            exam_type=effective_exam_type,
            avoid_question_ids=recent_question_ids,
            prioritized_question_ids=list(wrong_question_bank),
            strong_tags=list(profile.strong_points),
        )
        if not questions:
            raise ValueError("题库为空，无法生成模拟测试。")

        reused_wrong_question_ids = [question.record_id for question in questions if question.record_id in wrong_question_bank]
        exam_payload = {
            "exam_session_id": f"exam-{uuid4().hex[:10]}",
            "topic": topic or "综合",
            "exam_type": effective_exam_type,
            "question_count": len(questions),
            "preferred_tags": preferred_tags[:8],
            "reused_wrong_question_count": len(reused_wrong_question_ids),
            "questions": [
                {
                    "index": index,
                    "record_id": question.record_id,
                    "topic": str(question.metadata.get("topic") or topic or "综合"),
                    "question": question.title,
                    "options": question.metadata.get("options", {}),
                    "answer": question.metadata.get("answer"),
                    "analysis": question.metadata.get("analysis"),
                    "tags": list(question.tags),
                    "score": int(question.metadata.get("score", 20)),
                    "from_wrong_question_bank": question.record_id in wrong_question_bank,
                }
                for index, question in enumerate(questions, start=1)
            ],
        }
        self.memory_manager.record_exam_session(user_id, session_id, exam_payload)
        return exam_payload

    def _score_exam(
        self,
        user_id: str,
        session_id: str,
        answers_text: str,
        exam_session_id: str | None = None,
    ) -> dict[str, object]:
        exam_payload = self.memory_manager.load_exam_session(user_id, session_id, exam_session_id)
        if exam_payload is None:
            raise ValueError("当前没有待评分的模拟测试。")

        answers = self._parse_answer_sheet(answers_text)
        profile = self.memory_manager.get_user_profile(user_id)
        wrong_question_bank = {
            str(record_id): dict(item)
            for record_id, item in dict(profile.attributes.get("wrong_question_bank") or {}).items()
            if str(record_id).strip() and isinstance(item, dict)
        }
        total_score = 0
        earned_score = 0
        weak_tags: list[str] = []
        strong_tags: list[str] = []
        wrong_questions = []
        corrected_question_ids: list[str] = []
        details = []
        for question in exam_payload.get("questions", []):
            index = int(question["index"])
            score = int(question.get("score", 20))
            total_score += score
            user_answer = answers.get(str(index))
            correct_answer = str(question.get("answer") or "").upper()
            is_correct = user_answer == correct_answer
            record_id = str(question.get("record_id") or "").strip()
            question_tags = [str(tag) for tag in question.get("tags", []) if str(tag).strip()]
            if is_correct:
                earned_score += score
                strong_tags.extend(question_tags)
                if record_id and record_id in wrong_question_bank:
                    corrected_question_ids.append(record_id)
            else:
                weak_tags.extend(question_tags)
                wrong_questions.append(
                    {
                        "record_id": record_id,
                        "topic": str(question.get("topic") or exam_payload.get("topic") or "综合"),
                        "question": str(question.get("question") or ""),
                        "tags": question_tags,
                        "analysis": str(question.get("analysis") or ""),
                        "correct_answer": correct_answer,
                        "user_answer": user_answer,
                        "exam_session_id": str(exam_payload.get("exam_session_id") or ""),
                    }
                )
            details.append(
                {
                    "index": index,
                    "user_answer": user_answer,
                    "correct_answer": correct_answer,
                    "is_correct": is_correct,
                    "score": score if is_correct else 0,
                    "analysis": question.get("analysis"),
                    "question": question.get("question"),
                }
            )

        payload = {
            "exam_session_id": str(exam_payload["exam_session_id"]),
            "topic": exam_payload.get("topic", "综合"),
            "exam_type": exam_payload.get("exam_type", "综合练习"),
            "score_percent": round((earned_score / max(total_score, 1)) * 100, 2),
            "earned_score": earned_score,
            "total_score": total_score,
            "details": details,
            "weak_tags": list(dict.fromkeys(weak_tags))[:8],
            "strong_tags": [
                tag for tag in list(dict.fromkeys(strong_tags))[:8] if tag not in set(dict.fromkeys(weak_tags))
            ],
            "wrong_questions": wrong_questions,
            "corrected_question_ids": list(dict.fromkeys(corrected_question_ids)),
            "unanswered_count": sum(1 for item in details if not item.get("user_answer")),
        }
        self.memory_manager.store_exam_result(user_id, session_id, payload)
        return payload

    def _generate_report(self, user_id: str, session_id: str, report_type: str = "study_progress") -> dict[str, object]:
        snapshot = self.memory_manager.build_report_snapshot(user_id, session_id)
        report_markdown = self._render_report(snapshot, report_type=report_type)
        target_dir = ensure_dir(self.report_root / user_id)
        file_stem = f"{session_id}_{report_type}_{uuid4().hex[:8]}"
        report_path = target_dir / f"{file_stem}.md"
        report_path.write_text(report_markdown, encoding="utf-8")
        write_json(target_dir / f"{file_stem}.json", snapshot)
        self.memory_manager.set_last_report_path(user_id, session_id, str(report_path))
        return {
            "report_type": report_type,
            "report_path": str(report_path),
            "report_markdown": report_markdown,
            "snapshot": snapshot,
        }

    def _ask_followup(self, question: str, slot: str = "") -> dict[str, object]:
        return {"status": "pending_user_input", "question": question, "slot": slot}

    def _parse_answer_sheet(self, text: str) -> dict[str, str]:
        answers: dict[str, str] = {}
        for match in ANSWER_LINE_RE.finditer(text):
            answers[str(int(match.group("index")))] = match.group("answer").upper()
        return answers

    def _render_report(self, snapshot: dict[str, object], *, report_type: str) -> str:
        profile = dict(snapshot.get("profile") or {})
        exam_history = list(snapshot.get("exam_history") or [])
        latest_exam = exam_history[-1] if exam_history else None
        wrong_question_preview = list(snapshot.get("wrong_question_bank_preview") or [])
        lines = [
            f"# 学习报告：{report_type}",
            "",
            "## 用户画像",
            f"- 姓名：{profile.get('name') or '未设置'}",
            f"- 备考目标：{'、'.join(profile.get('study_goals') or []) or '未设置'}",
            f"- 薄弱点：{'、'.join(profile.get('weak_points') or []) or '暂无'}",
            f"- 强项：{'、'.join(profile.get('strong_points') or []) or '暂无'}",
            f"- 错题库待复盘：{snapshot.get('wrong_question_bank_count', 0)} 题",
            "",
            "## 会话进展",
            f"- 最近会话轮数：{snapshot.get('recent_turn_count', 0)}",
            f"- 会话摘要：{snapshot.get('session_summary') or '暂无'}",
            f"- 近期主题：{'、'.join(snapshot.get('recent_topics') or []) or '暂无'}",
            "",
            "## 最近一次测试",
        ]
        if latest_exam is None:
            lines.append("- 暂无测试记录")
        else:
            lines.extend(
                [
                    f"- 测试主题：{latest_exam.get('topic', '综合')}",
                    f"- 测试类型：{latest_exam.get('exam_type', '综合练习')}",
                    f"- 百分制成绩：{latest_exam.get('score_percent', 'N/A')}",
                    f"- 暴露薄弱点：{'、'.join(latest_exam.get('weak_tags') or []) or '暂无'}",
                    f"- 稳定掌握点：{'、'.join(latest_exam.get('strong_tags') or []) or '暂无'}",
                ]
            )
        lines.extend(
            [
                "",
                "## 错题库关注",
            ]
        )
        if not wrong_question_preview:
            lines.append("- 当前错题库为空，说明近期错题已基本消化。")
        else:
            for item in wrong_question_preview:
                lines.append(
                    f"- {item.get('topic') or '综合'}：{truncate_text(str(item.get('question') or ''), 60)}"
                    f"（累计错 {int(item.get('fail_count') or 1)} 次）"
                )
        lines.extend(
            [
                "",
                "## 建议动作",
                "1. 先围绕薄弱点复盘最近错题，整理一页自己的法条卡片。",
                "2. 再做同主题的小规模测验，观察错误是否收敛。",
                "3. 把新增的学习偏好和目标及时写回画像，方便下一轮规划。",
            ]
        )
        return "\n".join(lines)