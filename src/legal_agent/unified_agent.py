from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import logging
from pathlib import Path
import re
import threading
from typing import Any

from context_engine.manager import MemoryManager
from context_engine.reasoner import QwenMemoryReasoner
from context_engine.schemas import utcnow_iso
from context_engine.store import DiskMemoryStore
from context_engine.vectorizer import HashingVectorizer, TransformerVectorizer
from legal_agent.agent.engine import AgentRunResult, LegalAgentEngine
from legal_agent.agent.parser import parse_react_output
from legal_agent.agent.tools import observation_to_text
from legal_agent.config import AppConfig, load_app_config
from legal_agent.models.qwen_local import LocalQwenChatModel
from legal_agent.rag.retriever import HybridLegalRetriever
from legal_agent.study_config import StudyAgentConfig
from legal_agent.study_tools import INLINE_CHOICE_ANSWER_RE, NUMBERED_ANSWER_BLOCK_RE, StudyToolExecutor
from legal_agent.unified_tools import UnifiedToolRegistry
from legal_agent.utils.text import simple_tokenize, truncate_text
from rag_engine.service import KnowledgeService


BUTTON_ONLY_INTENTS = {"profile_lookup", "mock_exam_generate", "report_generation"}
BUTTON_ONLY_LABELS = {
    "profile_lookup": "学习报告",
    "mock_exam_generate": "生成模拟测试",
    "report_generation": "生成学习报告",
}
BUTTON_ONLY_HINTS = {
    "profile_lookup": ("查看画像", "用户画像", "学习画像", "profile"),
    "mock_exam_generate": ("模拟测试", "模拟题", "出题", "组卷", "练习题"),
    "report_generation": ("学习报告", "报告", "复盘报告"),
}
DIRECT_QA_SOURCE_LABELS = {
    "question_bank": "题库解析",
    "case_bank": "案例要点",
    "common_knowledge": "知识点",
    "statute": "法规依据",
}
DIRECT_QA_MIN_HIT_SCORE = 0.18
DIRECT_QA_MIN_STATUTE_SCORE = 0.12
DIRECT_QA_MIN_TOKEN_OVERLAP = 2
DIRECT_QA_MIN_DISPLAY_OVERLAP = 1
DEFAULT_EXAM_QUESTION_TYPES = ["single_choice", "short_answer", "case_analysis"]
SMALLTALK_MARKERS = (
    "你好",
    "您好",
    "嗨",
    "hi",
    "hello",
    "早上好",
    "下午好",
    "晚上好",
    "在吗",
    "谢谢",
    "多谢",
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class StudyAgentResponse:
    intent: str
    answer: str
    plan: dict[str, Any]
    tool_results: list[dict[str, Any]]
    report_path: str | None = None
    report_markdown: str | None = None
    trace: str = ""
    needs_user_input: bool = False
    clarification_question: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=4)
def _load_legal_config_cached(config_path: str) -> AppConfig:
    return load_app_config(config_path)


@lru_cache(maxsize=8)
def _get_retriever_cached(config_path: str, retrieval_device: str) -> HybridLegalRetriever:
    return HybridLegalRetriever(_load_legal_config_cached(config_path), device=retrieval_device)


@lru_cache(maxsize=8)
def _get_model_cached(
    model_path: str,
    adapter_path: str,
    model_device: str,
    load_in_4bit: bool,
    compute_dtype: str,
) -> LocalQwenChatModel:
    device_map: str | dict[str, int]
    if model_device == "cpu":
        device_map = "cpu"
    elif model_device.startswith("cuda:"):
        try:
            device_map = {"": int(model_device.split(":", maxsplit=1)[1])}
        except Exception:
            device_map = "auto"
    else:
        device_map = "auto"
    return LocalQwenChatModel(
        model_path,
        adapter_path=adapter_path or None,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        compute_dtype=compute_dtype,
    )


class UnifiedLegalAgent:
    def __init__(self, config: StudyAgentConfig, *, retrieval_device: str = "cpu") -> None:
        self.config = config
        self.default_retrieval_device = retrieval_device
        self.memory_store = DiskMemoryStore(config.memory_root)
        self.memory_manager = MemoryManager(
            self.memory_store,
            system_seed_path=config.system_memory_path,
            vectorizer=self._build_memory_vectorizer(retrieval_device),
            recent_turn_window=config.memory.recent_turn_window,
            compression_after_turns=config.memory.compression_after_turns,
            compression_chunk_size=config.memory.compression_chunk_size,
            retain_recent_turns=config.memory.retain_recent_turns,
        )
        self.knowledge_service = KnowledgeService(
            question_bank_path=config.question_bank_path,
            case_bank_path=config.case_bank_path,
            common_knowledge_path=config.common_knowledge_path,
            use_legacy_statute_rag=config.use_legacy_statute_rag,
            legacy_config_path=config.legacy_config_path,
            legacy_device=retrieval_device,
        )
        self.tool_executor = StudyToolExecutor(
            self.memory_manager,
            self.knowledge_service,
            report_root=config.report_root,
            subjective_exam_grader=self._grade_subjective_exam,
        )
        self.legal_config_path = str((config.legacy_config_path or (config.project_root / "configs" / "defaults.yaml")).resolve())
        self.legal_config = _load_legal_config_cached(self.legal_config_path)

    def _build_memory_vectorizer(self, retrieval_device: str):
        backend = str(self.config.memory.vectorizer or "hashing").strip().lower()
        if backend != "embedding" or self.config.memory.embedding_model_path is None:
            return HashingVectorizer()
        embedding_device = str(self.config.memory.embedding_device or "cpu").strip().lower()
        if embedding_device == "auto":
            embedding_device = retrieval_device if retrieval_device and retrieval_device != "auto" else "cpu"
        return TransformerVectorizer(self.config.memory.embedding_model_path, device=embedding_device)

    def handle_message(
        self,
        question: str,
        *,
        user_id: str = "default_user",
        session_id: str = "default_session",
        model_path: str | None = None,
        adapter_path: str | None = None,
        prompt_mode: str = "pure",
        retrieval_device: str | None = None,
        model_device: str = "auto",
        allow_button_only_intents: bool = True,
        display_question: str | None = None,
    ) -> StudyAgentResponse:
        self._ensure_user_session(user_id, session_id)
        if not allow_button_only_intents:
            button_only_intent = self._detect_button_only_intent(question)
            if button_only_intent is not None:
                return self._build_button_only_response(
                    question,
                    button_only_intent,
                    user_id=user_id,
                    session_id=session_id,
                )

        retrieval_device = retrieval_device or self.default_retrieval_device
        self._ensure_memory_reasoner(model_path=model_path, adapter_path=adapter_path, model_device=model_device)
        effective_question = self._normalize_runtime_question(question, user_id=user_id, session_id=session_id)
        history = self._session_history(user_id, session_id)
        engine = self._build_engine(
            user_id=user_id,
            session_id=session_id,
            model_path=model_path,
            adapter_path=adapter_path,
            prompt_mode=prompt_mode,
            retrieval_device=retrieval_device,
            model_device=model_device,
        )
        result = engine.run(effective_question, history=history)
        return self._finalize_response(
            question,
            effective_question=effective_question,
            result=result,
            user_id=user_id,
            session_id=session_id,
            retrieval_device=retrieval_device,
            prompt_mode=prompt_mode,
            display_question=display_question,
            model_path=model_path,
            adapter_path=adapter_path,
            model_device=model_device,
        )

    def stream_message(
        self,
        question: str,
        *,
        user_id: str = "default_user",
        session_id: str = "default_session",
        model_path: str | None = None,
        adapter_path: str | None = None,
        prompt_mode: str = "pure",
        retrieval_device: str | None = None,
        model_device: str = "auto",
        allow_button_only_intents: bool = True,
        display_question: str | None = None,
    ):
        self._ensure_user_session(user_id, session_id)
        if not allow_button_only_intents:
            button_only_intent = self._detect_button_only_intent(question)
            if button_only_intent is not None:
                response = self._build_button_only_response(
                    question,
                    button_only_intent,
                    user_id=user_id,
                    session_id=session_id,
                )
                yield {"event": "final", "message": response.answer, "trace": response.trace, "response": response}
                return

        retrieval_device = retrieval_device or self.default_retrieval_device
        yield {
            "event": "status",
            "message": "正在初始化模型与记忆模块。",
            "trace": "[Init]\n正在初始化模型与记忆模块。",
        }
        self._ensure_memory_reasoner(model_path=model_path, adapter_path=adapter_path, model_device=model_device)
        yield {
            "event": "status",
            "message": "正在读取会话上下文并构建推理引擎。",
            "trace": "[Init]\n正在初始化模型与记忆模块。\n[Init]\n正在读取会话上下文并构建推理引擎。",
        }
        effective_question = self._normalize_runtime_question(question, user_id=user_id, session_id=session_id)
        history = self._session_history(user_id, session_id)
        engine = self._build_engine(
            user_id=user_id,
            session_id=session_id,
            model_path=model_path,
            adapter_path=adapter_path,
            prompt_mode=prompt_mode,
            retrieval_device=retrieval_device,
            model_device=model_device,
        )
        yield {
            "event": "status",
            "message": "运行时初始化完成，开始规划下一步。",
            "trace": "[Init]\n正在初始化模型与记忆模块。\n[Init]\n正在读取会话上下文并构建推理引擎。\n[Init]\n运行时初始化完成，开始规划下一步。",
        }
        final_result: AgentRunResult | None = None
        for update in engine.run_with_updates(effective_question, history=history):
            if update.get("event") == "final":
                final_result = update.get("result")
                break
            yield {
                "event": str(update.get("event") or "status"),
                "message": str(update.get("message") or "正在处理中。"),
                "trace": str(update.get("trace") or ""),
            }

        if final_result is None:
            raise RuntimeError("统一 Agent 流式执行未返回最终结果。")

        response = self._compose_response(
            question,
            effective_question=effective_question,
            result=final_result,
            user_id=user_id,
            session_id=session_id,
            retrieval_device=retrieval_device,
            prompt_mode=prompt_mode,
            display_question=display_question,
            model_path=model_path,
            adapter_path=adapter_path,
            model_device=model_device,
        )
        yield {"event": "final", "message": response.answer, "trace": response.trace, "response": response}
        self._persist_response_turn_async(
            user_id=user_id,
            session_id=session_id,
            user_message=display_question or question,
            response=response,
        )

    def generate_exam(
        self,
        *,
        user_id: str = "default_user",
        session_id: str = "default_session",
        topic: str | None = None,
        question_count: int | None = None,
        exam_type: str | None = None,
        question_types: list[str] | None = None,
        model_path: str | None = None,
        adapter_path: str | None = None,
        prompt_mode: str = "pure",
        retrieval_device: str | None = None,
        model_device: str = "auto",
    ) -> StudyAgentResponse:
        effective_topic = (topic or "综合").strip() or "综合"
        effective_count = int(question_count or self.config.default_exam_question_count)
        effective_exam_type = (exam_type or "综合练习").strip() or "综合练习"
        self._ensure_user_session(user_id, session_id)
        self._ensure_memory_reasoner(model_path=model_path, adapter_path=adapter_path, model_device=model_device)
        return self._direct_generate_exam_response(
            user_id=user_id,
            session_id=session_id,
            topic=effective_topic,
            question_count=effective_count,
            exam_type=effective_exam_type,
            question_types=question_types,
        )

    def generate_report(self, *, user_id: str = "default_user", session_id: str = "default_session", report_type: str = "study_progress") -> dict[str, Any]:
        return self.tool_executor.execute(
            "generate_report",
            {"report_type": report_type},
            user_id=user_id,
            session_id=session_id,
        )

    def generate_report_response(
        self,
        *,
        user_id: str = "default_user",
        session_id: str = "default_session",
        report_type: str = "study_progress",
        model_path: str | None = None,
        adapter_path: str | None = None,
        prompt_mode: str = "pure",
        retrieval_device: str | None = None,
        model_device: str = "auto",
    ) -> StudyAgentResponse:
        self._ensure_user_session(user_id, session_id)
        self._ensure_memory_reasoner(model_path=model_path, adapter_path=adapter_path, model_device=model_device)
        return self._direct_generate_report_response(
            user_id=user_id,
            session_id=session_id,
            report_type=report_type,
        )

    def view_profile(self, *, user_id: str = "default_user", session_id: str = "default_session") -> StudyAgentResponse:
        self._ensure_user_session(user_id, session_id)
        tool_results = [
            {
                "tool_name": "profile_view",
                "reason": "读取当前用户画像，供兼容接口使用。",
                "arguments": {},
                "result": self.tool_executor.execute("profile_view", {}, user_id=user_id, session_id=session_id),
            }
        ]
        answer = self._render_profile_answer(tool_results)
        self._record_turn(user_id, session_id, "[兼容接口] 查看用户画像", answer, tool_results)
        return StudyAgentResponse(
            intent="profile_lookup",
            answer=answer,
            plan={"planner_backend": "direct_tool", "tool_names": ["profile_view"]},
            tool_results=tool_results,
            trace="",
        )

    def list_users(self) -> list[str]:
        return self.memory_manager.list_users()

    def create_user(self, user_id: str, *, display_name: str | None = None) -> dict[str, Any]:
        profile = self.memory_manager.create_user(user_id, display_name=display_name)
        return profile.to_dict()

    def delete_user(self, user_id: str) -> bool:
        return self.memory_manager.delete_user(user_id)

    def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        self._ensure_user_exists(user_id)
        return self.memory_manager.list_sessions(user_id)

    def create_session(self, user_id: str, session_id: str) -> dict[str, Any]:
        self._ensure_user_exists(user_id)
        state = self.memory_manager.ensure_session(user_id, session_id)
        return state.to_dict()

    def delete_session(self, user_id: str, session_id: str) -> bool:
        return self.memory_manager.delete_session(user_id, session_id)

    def get_profile(self, user_id: str) -> dict[str, Any]:
        self._ensure_user_exists(user_id)
        return self.memory_manager.get_user_profile(user_id).to_dict()

    def get_session_state(self, user_id: str, session_id: str) -> dict[str, Any]:
        self._ensure_user_session(user_id, session_id)
        return self.memory_manager.get_session_state(user_id, session_id).to_dict()

    def get_session_history(self, user_id: str, session_id: str) -> list[tuple[str, str]]:
        self._ensure_user_session(user_id, session_id)
        return self._session_history(user_id, session_id, max_turns=64)

    def _ensure_user_exists(self, user_id: str) -> None:
        if user_id not in self.memory_manager.list_users():
            self.memory_manager.create_user(user_id, display_name=user_id)

    def _ensure_user_session(self, user_id: str, session_id: str) -> None:
        self._ensure_user_exists(user_id)
        self.memory_manager.ensure_session(user_id, session_id)

    def _build_engine(
        self,
        *,
        user_id: str,
        session_id: str,
        model_path: str | None,
        adapter_path: str | None,
        prompt_mode: str,
        retrieval_device: str,
        model_device: str,
    ) -> LegalAgentEngine:
        retriever = _get_retriever_cached(self.legal_config_path, retrieval_device)
        registry = UnifiedToolRegistry(
            retriever,
            study_tool_executor=self.tool_executor,
            user_id=user_id,
            session_id=session_id,
            interactive=False,
            ask_user_handler=lambda _question, _field: None,
        )
        model = self._load_runtime_model(model_path=model_path, adapter_path=adapter_path, model_device=model_device)
        if self.config.turn_analysis_mode == "llm":
            self.memory_manager.bind_reasoner(QwenMemoryReasoner(model))
        return LegalAgentEngine(
            model,
            registry,
            max_steps=self.legal_config.inference.max_steps,
            max_new_tokens=self.legal_config.inference.max_new_tokens,
            temperature=self.legal_config.inference.temperature,
            top_p=self.legal_config.inference.top_p,
            top_k=self.legal_config.inference.top_k,
            presence_penalty=self.legal_config.inference.presence_penalty,
            enable_thinking=self.legal_config.inference.enable_thinking,
            prompt_mode=prompt_mode,
            turn_analysis_mode=self.config.turn_analysis_mode,
            use_planning_engine=True,
        )

    def _session_history(self, user_id: str, session_id: str, *, max_turns: int = 8) -> list[tuple[str, str]]:
        history: list[tuple[str, str]] = []
        turns = self.memory_manager.store.load_session_turns(user_id, session_id, limit=max_turns)
        for turn in turns:
            history.append((turn.user_message, turn.assistant_message))
        return history

    def _ensure_memory_reasoner(
        self,
        *,
        model_path: str | None,
        adapter_path: str | None,
        model_device: str,
    ) -> None:
        if self.config.turn_analysis_mode != "llm":
            return
        signature = self._runtime_model_signature(model_path=model_path, adapter_path=adapter_path, model_device=model_device)
        if getattr(self, "_memory_reasoner_signature", None) == signature:
            return
        model = _get_model_cached(*signature)
        self.memory_manager.bind_reasoner(QwenMemoryReasoner(model))
        self._memory_reasoner_signature = signature

    def _runtime_model_signature(
        self,
        *,
        model_path: str | None,
        adapter_path: str | None,
        model_device: str,
    ) -> tuple[str, str, str, bool, str]:
        resolved_model_path = self.legal_config.resolve_project_path(model_path) if model_path else self.legal_config.models.agent_base
        resolved_adapter_path = self.legal_config.resolve_project_path(adapter_path) if adapter_path else None
        return (
            str(resolved_model_path.resolve()),
            str(resolved_adapter_path.resolve()) if resolved_adapter_path else "",
            model_device,
            self.legal_config.inference.load_in_4bit,
            self.legal_config.inference.compute_dtype,
        )

    def _load_runtime_model(
        self,
        *,
        model_path: str | None,
        adapter_path: str | None,
        model_device: str,
    ) -> LocalQwenChatModel:
        return _get_model_cached(*self._runtime_model_signature(model_path=model_path, adapter_path=adapter_path, model_device=model_device))

    def _grade_subjective_exam(self, question: dict[str, object], user_answer: str) -> dict[str, object]:
        reference_answer = str(question.get("reference_answer") or question.get("answer") or "").strip()
        analysis = str(question.get("analysis") or "").strip()
        max_score = int(question.get("score") or 20)
        if not reference_answer and not analysis:
            return {
                "score": 0,
                "feedback": "当前题目缺少参考答案，无法完成主观评分。",
                "matched_points": [],
                "missing_points": [],
            }

        model = self._load_runtime_model(model_path=None, adapter_path=None, model_device="auto")
        prompt = (
            "你是中国法考阅卷器。请依据参考答案与解析，对用户的作答进行主观评分。"
            "只输出 JSON，不要输出任何额外说明。"
            "JSON 格式固定为："
            '{"score": 0, "feedback": "...", "matched_points": ["..."], "missing_points": ["..."]}。\n\n'
            f"题目类型：{question.get('question_type') or 'short_answer'}\n"
            f"题目：{question.get('question') or ''}\n"
            f"满分：{max_score}\n"
            f"参考答案：{reference_answer}\n"
            f"解析：{analysis}\n"
            f"参考法条：{question.get('references') or []}\n"
            f"用户答案：{user_answer.strip()}"
        )
        output = model.generate(
            [
                {"role": "system", "content": "你是严格的 JSON 阅卷器。输出必须是合法 JSON。"},
                {"role": "user", "content": prompt},
            ],
            max_new_tokens=320,
            temperature=0.0,
            top_p=0.9,
            top_k=20,
            presence_penalty=1.0,
            enable_thinking=False,
        )
        text = (output.content or output.raw_text).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                payload = json.loads(text[start : end + 1])
                payload["score"] = max(0, min(max_score, int(round(float(payload.get("score") or 0)))))
                payload["feedback"] = str(payload.get("feedback") or "").strip()
                payload["matched_points"] = [str(item) for item in payload.get("matched_points", []) if str(item).strip()][:5]
                payload["missing_points"] = [str(item) for item in payload.get("missing_points", []) if str(item).strip()][:5]
                return payload
            except Exception:
                pass
        return {
            "score": 0,
            "feedback": "主观评分结果解析失败，请回看参考答案与解析。",
            "matched_points": [],
            "missing_points": [truncate_text(reference_answer or analysis, 160)] if (reference_answer or analysis) else [],
        }

    def prepare_context(
        self,
        question: str,
        *,
        user_id: str = "default_user",
        session_id: str = "default_session",
    ) -> dict[str, Any]:
        self._ensure_user_session(user_id, session_id)
        return self.memory_manager.prepare_turn_context_payload(question, user_id, session_id)

    def _normalize_runtime_question(self, question: str, *, user_id: str, session_id: str) -> str:
        raw_question = str(question or "").strip()
        if not raw_question:
            return raw_question
        active_exam = self.memory_manager.load_active_exam(user_id, session_id)
        if self._should_treat_as_exam_submission(raw_question, active_exam):
            return (
                "用户刚提交了当前激活模拟测试的答案，请继续同一学习任务。\n"
                f"答题卡：\n{raw_question}\n"
                "请先调用 score_exam 对当前激活试卷评分；如果评分完成，再调用 generate_report 输出学习反馈摘要。"
            )
        return raw_question

    def _looks_like_answer_sheet(self, text: str) -> bool:
        matches = list(NUMBERED_ANSWER_BLOCK_RE.finditer(text))
        compact = " ".join(str(text or "").split())
        if len(matches) >= 1:
            return True
        inline_matches = INLINE_CHOICE_ANSWER_RE.findall(text)
        return len(inline_matches) >= 2 or bool(inline_matches and len(compact) <= 48)

    def _should_treat_as_exam_submission(self, text: str, active_exam: dict[str, Any] | None) -> bool:
        raw_text = str(text or "").strip()
        if not raw_text or active_exam is None:
            return False
        if self._looks_like_answer_sheet(raw_text):
            return True

        exam_questions = [dict(item) for item in active_exam.get("questions", []) if isinstance(item, dict)]
        if len(exam_questions) != 1:
            return False

        question_type = str(exam_questions[0].get("question_type") or "single_choice")
        if question_type == "single_choice":
            return False

        if self._looks_like_smalltalk(raw_text):
            return False
        if raw_text.endswith(("?", "？")):
            return False
        if self._detect_button_only_intent(raw_text) is not None:
            return False
        return True

    def _looks_like_smalltalk(self, text: str) -> bool:
        normalized = str(text or "").strip().lower()
        if not normalized:
            return False
        compact = "".join(normalized.split())
        if compact in {marker.lower() for marker in SMALLTALK_MARKERS}:
            return True
        return any(compact.startswith(marker.lower()) for marker in SMALLTALK_MARKERS)

    def _detect_button_only_intent(self, question: str) -> str | None:
        lowered = str(question or "").lower()
        for intent, markers in BUTTON_ONLY_HINTS.items():
            if any(marker.lower() in lowered for marker in markers):
                return intent
        return None

    def _build_button_only_response(
        self,
        question: str,
        intent: str,
        *,
        user_id: str,
        session_id: str,
    ) -> StudyAgentResponse:
        answer = f"该功能在界面里是按钮专用入口，请点击“{BUTTON_ONLY_LABELS.get(intent, '功能按钮')}”触发，不通过聊天消息直接调用。"
        self._record_turn(user_id, session_id, question, answer, [])
        return StudyAgentResponse(
            intent="ui_button_only",
            answer=answer,
            plan={"planner_backend": "ui_guard", "blocked_intent": intent},
            tool_results=[],
        )

    def _infer_intent(self, tool_results: list[dict[str, Any]], needs_user_input: bool) -> str:
        tool_names = [str(entry.get("tool_name") or "") for entry in tool_results]
        if needs_user_input:
            return "clarification"
        if "score_exam" in tool_names:
            return "mock_exam_score"
        if "generate_exam" in tool_names:
            return "mock_exam_generate"
        if "generate_report" in tool_names:
            return "report_generation"
        if any(name in tool_names for name in ["rag_search", "retrieve_from_kb", "lookup_statute", "resolve_hierarchy"]):
            return "legal_qa"
        if "profile_upsert" in tool_names:
            return "profile_update"
        if "profile_view" in tool_names:
            return "profile_lookup"
        if "calculator" in tool_names:
            return "legal_calculation"
        return "legal_qa"

    def _extract_report_artifacts(self, tool_results: list[dict[str, Any]]) -> tuple[str | None, str | None]:
        report_path = None
        report_markdown = None
        for entry in tool_results:
            if entry.get("tool_name") != "generate_report":
                continue
            result = dict(entry.get("result") or {})
            report_path = str(result.get("report_path") or "") or report_path
            report_markdown = str(result.get("report_markdown") or "") or report_markdown
        return report_path, report_markdown

    def _ensure_score_report(
        self,
        tool_results: list[dict[str, Any]],
        *,
        user_id: str,
        session_id: str,
        needs_user_input: bool,
    ) -> tuple[list[dict[str, Any]], bool]:
        items = list(tool_results)
        if needs_user_input or not self._has_tool(items, "score_exam") or self._has_tool(items, "generate_report"):
            return items, False

        items.append(
            {
                "tool_name": "generate_report",
                "reason": "补全本次评分对应的学习反馈报告。",
                "arguments": {"report_type": "mock_exam_review"},
                "result": self.tool_executor.execute(
                    "generate_report",
                    {"report_type": "mock_exam_review"},
                    user_id=user_id,
                    session_id=session_id,
                ),
            }
        )
        return items, True

    def _has_tool(self, tool_results: list[dict[str, Any]], tool_name: str) -> bool:
        return any(str(entry.get("tool_name") or "") == tool_name for entry in tool_results)

    def _postprocess_answer(self, answer: str, tool_results: list[dict[str, Any]]) -> str:
        cleaned_lines: list[str] = []
        for line in str(answer or "").splitlines():
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append("")
                continue
            if stripped.startswith("报告路径"):
                continue
            if "/" in stripped and stripped.endswith(".md"):
                continue
            cleaned_lines.append(line)
        cleaned = "\n".join(cleaned_lines).strip()
        if cleaned:
            return cleaned
        if self._has_tool(tool_results, "generate_report"):
            return "已生成学习报告，内容和下载入口已同步到右侧面板。"
        return "已完成本轮处理。"

    def _finalize_response(
        self,
        question: str,
        *,
        effective_question: str,
        result: AgentRunResult,
        user_id: str,
        session_id: str,
        retrieval_device: str,
        prompt_mode: str,
        display_question: str | None,
        model_path: str | None,
        adapter_path: str | None,
        model_device: str,
    ) -> StudyAgentResponse:
        response = self._compose_response(
            question,
            effective_question=effective_question,
            result=result,
            user_id=user_id,
            session_id=session_id,
            retrieval_device=retrieval_device,
            prompt_mode=prompt_mode,
            display_question=display_question,
            model_path=model_path,
            adapter_path=adapter_path,
            model_device=model_device,
        )
        self._persist_response_turn(
            user_id=user_id,
            session_id=session_id,
            user_message=display_question or question,
            response=response,
        )
        return response

    def _compose_response(
        self,
        question: str,
        *,
        effective_question: str,
        result: AgentRunResult,
        user_id: str,
        session_id: str,
        retrieval_device: str,
        prompt_mode: str,
        display_question: str | None,
        model_path: str | None,
        adapter_path: str | None,
        model_device: str,
    ) -> StudyAgentResponse:
        tool_results = list(result.tool_history)
        used_direct_fallback = False
        if not tool_results and not result.needs_user_input:
            fallback_payload = self._direct_fallback_payload(
                question,
                effective_question=effective_question,
                retrieval_device=retrieval_device,
                user_id=user_id,
                session_id=session_id,
                model_path=model_path,
                adapter_path=adapter_path,
                model_device=model_device,
            )
            if fallback_payload is not None:
                used_direct_fallback = True
                tool_results = list(fallback_payload["tool_results"])
                result.trace = (result.trace.rstrip() + "\n\n[Direct Fallback]\n" + str(fallback_payload.get("trace") or "")).strip()
                if fallback_payload.get("answer") is not None:
                    result.final_answer = str(fallback_payload.get("answer") or result.final_answer)

        tool_results, auto_generated_score_report = self._ensure_score_report(
            tool_results,
            user_id=user_id,
            session_id=session_id,
            needs_user_input=result.needs_user_input,
        )
        report_path, report_markdown = self._extract_report_artifacts(tool_results)
        intent = self._infer_intent(tool_results, result.needs_user_input)
        answer = self._postprocess_answer(result.final_answer, tool_results)
        if intent == "mock_exam_generate" and self._has_tool(tool_results, "generate_exam"):
            answer = self._render_exam_answer(tool_results)
        if intent == "mock_exam_score" and self._has_tool(tool_results, "score_exam"):
            answer = self._render_score_answer(tool_results)
        if report_markdown and intent == "report_generation":
            answer = answer or "已生成学习报告，内容和下载入口已同步到右侧面板。"
        if result.needs_user_input and result.clarification_question:
            answer = f"为继续分析，请先补充：{result.clarification_question}"
        elif intent != "mock_exam_generate" and (used_direct_fallback or "Final Answer:" in str(result.trace or "")):
            result.trace = self._sync_trace_final_answer(result.trace, answer)
        if auto_generated_score_report:
            result.trace = (result.trace.rstrip() + "\n\n[Auto Report]\nscore_exam -> generate_report").strip()

        return StudyAgentResponse(
            intent=intent,
            answer=answer,
            plan={
                "planner_backend": self.config.planner_backend,
                "turn_analysis_mode": self.config.turn_analysis_mode,
                "prompt_mode": prompt_mode,
                "tool_names": [entry.get("tool_name") for entry in tool_results],
            },
            tool_results=tool_results,
            report_path=report_path,
            report_markdown=report_markdown,
            trace=result.trace,
            needs_user_input=result.needs_user_input,
            clarification_question=result.clarification_question,
        )

    def _persist_response_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        user_message: str,
        response: StudyAgentResponse,
    ) -> None:
        self._record_turn(
            user_id,
            session_id,
            user_message,
            response.answer,
            response.tool_results,
            reasoning_trace=response.trace,
        )

    def _persist_response_turn_async(
        self,
        *,
        user_id: str,
        session_id: str,
        user_message: str,
        response: StudyAgentResponse,
    ) -> None:
        def _runner() -> None:
            try:
                self._persist_response_turn(
                    user_id=user_id,
                    session_id=session_id,
                    user_message=user_message,
                    response=response,
                )
            except Exception:
                LOGGER.exception("流式答复的后台记忆写回失败")

        threading.Thread(target=_runner, daemon=True).start()

    def _report_chat_answer(self, report_markdown: str | None) -> str:
        content = str(report_markdown or "").strip()
        if content:
            return content
        return "已生成学习报告，内容和下载入口已同步到右侧面板。"

    def _replace_last_turn_answer(self, user_id: str, session_id: str, answer: str) -> None:
        self.memory_manager.replace_last_turn_answer(user_id, session_id, answer)

    def _record_turn(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        answer: str,
        tool_results: list[dict[str, Any]],
        reasoning_trace: str | None = None,
    ) -> None:
        self.memory_manager.record_turn(
            user_id,
            session_id,
            user_message,
            answer,
            tool_trace=tool_results,
            reasoning_trace=reasoning_trace,
        )
        self.memory_manager.decay_memories(user_id)

    def _direct_fallback_payload(
        self,
        question: str,
        *,
        effective_question: str,
        retrieval_device: str,
        user_id: str,
        session_id: str,
        model_path: str | None,
        adapter_path: str | None,
        model_device: str,
    ) -> dict[str, Any] | None:
        active_exam = self.memory_manager.load_active_exam(user_id, session_id)
        if self._should_treat_as_exam_submission(question, active_exam):
            return self._build_direct_score_payload(question, user_id=user_id, session_id=session_id)
        profile_updates = self.memory_manager.extract_profile_updates_for_user(question, user_id=user_id, session_id=session_id)
        profile_update_steps = self._profile_update_steps(question, profile_updates, user_id=user_id, session_id=session_id)
        if self._looks_like_report_request(question):
            return self._build_direct_report_payload(
                user_id=user_id,
                session_id=session_id,
                report_type=self._infer_report_type(question),
                profile_update_steps=profile_update_steps,
            )
        if self._looks_like_exam_request(question):
            topic, question_count, exam_type, question_types = self._infer_exam_request(question)
            return self._build_direct_exam_payload(
                user_id=user_id,
                session_id=session_id,
                topic=topic,
                question_count=question_count,
                exam_type=exam_type,
                question_types=question_types,
                profile_update_steps=profile_update_steps,
            )
        return self._build_direct_qa_payload(
            effective_question,
            retrieval_device=retrieval_device,
            user_id=user_id,
            session_id=session_id,
            profile_update_steps=profile_update_steps,
            model_path=model_path,
            adapter_path=adapter_path,
            model_device=model_device,
        )

    def _sync_trace_final_answer(self, trace: str | None, answer: str) -> str:
        normalized = str(trace or "").strip()
        marker = "Final Answer:"
        marker_index = normalized.rfind(marker)
        if marker_index >= 0:
            body = normalized[:marker_index].rstrip()
        else:
            body = normalized
        final_line = f"Final Answer: {answer}".strip()
        if body:
            return f"{body}\n\n{final_line}".strip()
        return final_line

    def _question_tokens(self, text: str) -> set[str]:
        return {token for token in simple_tokenize(text) if len(token.strip()) >= 2}

    def _token_overlap(self, query_tokens: set[str], *texts: str) -> int:
        candidate_tokens: set[str] = set()
        for text in texts:
            candidate_tokens.update(self._question_tokens(str(text or "")))
        return len(query_tokens & candidate_tokens)

    def _numeric_score(self, item: dict[str, Any]) -> float:
        try:
            return float(item.get("score") or 0.0)
        except Exception:
            return 0.0

    def _select_relevant_knowledge_hits(self, question: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not hits:
            return []
        query_tokens = self._question_tokens(question)
        top_score = max(self._numeric_score(item) for item in hits)
        relevant: list[dict[str, Any]] = []
        for item in hits:
            score = self._numeric_score(item)
            overlap = self._token_overlap(
                query_tokens,
                str(item.get("title") or ""),
                str(item.get("excerpt") or ""),
                str((item.get("metadata") or {}).get("analysis") or ""),
                str((item.get("metadata") or {}).get("holding") or ""),
            )
            allow_by_overlap = score >= DIRECT_QA_MIN_HIT_SCORE and overlap >= DIRECT_QA_MIN_DISPLAY_OVERLAP
            allow_by_score_only = score >= max(0.78, top_score * 0.92)
            if allow_by_overlap or allow_by_score_only:
                relevant.append(item)
        return relevant[:3]

    def _select_relevant_statute_results(self, question: str, statute_payload: dict[str, Any] | None) -> list[dict[str, Any]]:
        results = list((statute_payload or {}).get("results") or [])
        if not results:
            return []
        query_tokens = self._question_tokens(question)
        relevant: list[dict[str, Any]] = []
        for item in results:
            score = self._numeric_score(item)
            overlap = self._token_overlap(
                query_tokens,
                str(item.get("document_title") or item.get("title") or ""),
                str(item.get("article_heading") or ""),
                str(item.get("text") or item.get("excerpt") or ""),
            )
            if score >= DIRECT_QA_MIN_STATUTE_SCORE or overlap >= DIRECT_QA_MIN_TOKEN_OVERLAP:
                relevant.append(item)
        return relevant[:2]

    def _summarize_knowledge_hit(self, item: dict[str, Any]) -> str:
        source_type = str(item.get("source_type") or "")
        metadata = dict(item.get("metadata") or {})
        if source_type == "question_bank":
            answer = str(metadata.get("answer") or "").strip()
            analysis = str(metadata.get("analysis") or item.get("excerpt") or "").strip()
            if answer and analysis:
                return truncate_text(f"更稳妥的处理思路可以参考题库结论 {answer}：{analysis}", 170)
            return truncate_text(analysis or str(item.get("excerpt") or item.get("title") or ""), 170)
        if source_type == "case_bank":
            holding = str(metadata.get("holding") or item.get("excerpt") or "").strip()
            return truncate_text(holding or str(item.get("title") or ""), 170)
        if source_type == "statute":
            return truncate_text(str(item.get("excerpt") or item.get("title") or ""), 170)
        return truncate_text(str(item.get("excerpt") or item.get("title") or ""), 170)

    def _summarize_statute_result(self, item: dict[str, Any]) -> str:
        title = str(item.get("document_title") or item.get("title") or "法规材料").strip()
        article = str(item.get("article_heading") or "").strip()
        excerpt = truncate_text(str(item.get("text") or item.get("excerpt") or ""), 140)
        name = f"{title}{(' ' + article) if article else ''}".strip()
        return f"可继续结合 {name}：{excerpt}".strip()

    def _render_reference_line(self, item: dict[str, Any]) -> str:
        title = str(item.get("title") or "").strip()
        source_type = str(item.get("source_type") or "")
        label = DIRECT_QA_SOURCE_LABELS.get(source_type, "参考材料")
        summary = self._summarize_knowledge_hit(item)
        if title and title not in summary:
            return f"- {label}：{title}。{summary}"
        return f"- {label}：{summary}"

    def _render_statute_reference_line(self, item: dict[str, Any]) -> str:
        title = str(item.get("document_title") or item.get("title") or "法规材料").strip()
        article = str(item.get("article_heading") or "").strip()
        summary = truncate_text(str(item.get("text") or item.get("excerpt") or ""), 120)
        name = f"{title}{(' ' + article) if article else ''}".strip()
        return f"- 法规依据：{name}。{summary}"

    def _looks_like_report_request(self, question: str) -> bool:
        lowered = str(question or "").lower()
        return any(marker in lowered for marker in ["学习报告", "报告", "复盘", "诊断报告"])

    def _looks_like_exam_request(self, question: str) -> bool:
        lowered = str(question or "").lower()
        return any(marker in lowered for marker in ["模拟测试", "模拟题", "出题", "组卷", "练习题"])

    def _infer_report_type(self, question: str) -> str:
        lowered = str(question or "").lower()
        if "诊断" in lowered or "薄弱" in lowered:
            return "weakness_diagnosis"
        if "测试" in lowered or "exam" in lowered or "mock" in lowered:
            return "mock_exam_review"
        return "study_progress"

    def _infer_exam_request(self, question: str) -> tuple[str, int, str, list[str]]:
        compact = str(question or "").strip()
        topic = "综合"
        for candidate in ["民法", "刑法", "行政法", "刑诉", "民诉", "理论法", "商经法"]:
            if candidate in compact:
                topic = candidate
                break
        match = re.search(r"(\d+)\s*题", compact)
        question_count = int(match.group(1)) if match else self.config.default_exam_question_count
        exam_type = "综合练习"
        if "薄弱" in compact or "错题" in compact:
            exam_type = "薄弱点强化"
        elif "章节" in compact:
            exam_type = "章节练习"
        elif "真题" in compact:
            exam_type = "真题模拟"
        question_types = list(DEFAULT_EXAM_QUESTION_TYPES)
        if any(marker in compact for marker in ["简答", "主观", "问答"]):
            question_types = ["short_answer"]
        elif any(marker in compact for marker in ["案例", "案例分析"]):
            question_types = ["case_analysis"]
        elif any(marker in compact for marker in ["混合", "综合题型"]):
            question_types = list(DEFAULT_EXAM_QUESTION_TYPES)
        return topic, question_count, exam_type, question_types

    def _build_direct_profile_update_payload(
        self,
        question: str,
        profile_updates: dict[str, Any],
        *,
        user_id: str,
        session_id: str,
    ) -> dict[str, Any]:
        tool_results = [
            {
                "tool_name": "profile_upsert",
                "reason": "从当前用户输入中提取学习画像更新。",
                "arguments": {"raw_text": question, "updates": profile_updates},
                "result": self.tool_executor.execute(
                    "profile_upsert",
                    {"raw_text": question, "updates": profile_updates},
                    user_id=user_id,
                    session_id=session_id,
                ),
            },
            {
                "tool_name": "profile_view",
                "reason": "读取更新后的画像。",
                "arguments": {},
                "result": self.tool_executor.execute("profile_view", {}, user_id=user_id, session_id=session_id),
            },
        ]
        profile = dict(tool_results[-1]["result"].get("profile") or {})
        answer = (
            "已更新你的学习档案。"
            f" 当前目标：{'、'.join(profile.get('study_goals') or []) or '未设置'}；"
            f"薄弱点：{'、'.join(profile.get('weak_points') or []) or '暂无'}。"
        )
        return {"tool_results": tool_results, "answer": answer, "trace": "profile_upsert -> profile_view"}

    def _profile_update_steps(
        self,
        question: str,
        profile_updates: dict[str, Any],
        *,
        user_id: str,
        session_id: str,
    ) -> list[dict[str, Any]]:
        if not profile_updates:
            return []
        payload = self._build_direct_profile_update_payload(question, profile_updates, user_id=user_id, session_id=session_id)
        return list(payload.get("tool_results") or [])

    def _build_direct_score_payload(self, question: str, *, user_id: str, session_id: str) -> dict[str, Any]:
        tool_results = [
            {
                "tool_name": "score_exam",
                "reason": "对当前激活试卷进行评分。",
                "arguments": {"answers_text": question},
                "result": self.tool_executor.execute(
                    "score_exam",
                    {"answers_text": question},
                    user_id=user_id,
                    session_id=session_id,
                ),
            },
            {
                "tool_name": "generate_report",
                "reason": "生成本次测试的学习反馈报告。",
                "arguments": {"report_type": "mock_exam_review"},
                "result": self.tool_executor.execute(
                    "generate_report",
                    {"report_type": "mock_exam_review"},
                    user_id=user_id,
                    session_id=session_id,
                ),
            },
        ]
        answer = self._render_score_answer(tool_results)
        return {"tool_results": tool_results, "answer": answer, "trace": "score_exam -> generate_report"}

    def _prepare_context_step(self, query: str, *, user_id: str, session_id: str, reason: str) -> dict[str, Any]:
        return {
            "tool_name": "prepare_context",
            "reason": reason,
            "arguments": {"query": query},
            "result": self.tool_executor.execute(
                "prepare_context",
                {"query": query},
                user_id=user_id,
                session_id=session_id,
            ),
        }

    def _needs_rag_search(self, question: str, *, has_profile_update: bool) -> bool:
        normalized = str(question or "").strip()
        if not normalized:
            return False
        if normalized.endswith(("?", "？")):
            return True
        query_markers = (
            "怎么",
            "如何",
            "为什么",
            "怎么办",
            "能否",
            "是否",
            "解释",
            "分析",
            "区别",
            "法条",
            "依据",
            "案例",
            "知识点",
        )
        if any(marker in normalized for marker in query_markers):
            return True
        if has_profile_update:
            return False
        return True

    def _render_profile_update_ack(self, tool_results: list[dict[str, Any]]) -> str:
        result_by_name = {entry["tool_name"]: entry["result"] for entry in tool_results}
        profile = dict(result_by_name.get("profile_view", {}).get("profile") or {})
        parts = ["记住了。"]
        goals = list(profile.get("study_goals") or [])
        weak_points = list(profile.get("weak_points") or [])
        preferences = dict(profile.get("preferences") or {})
        if goals:
            parts.append(f"当前我会按你主要复习 {'、'.join(goals[:4])} 来组织后续辅导。")
        if weak_points:
            parts.append(f"我也会把 {'、'.join(weak_points[:4])} 视为优先复盘的薄弱点。")
        response_length = str(preferences.get("response_length") or "").strip()
        if response_length:
            parts.append(f"后续答复我会尽量控制在{response_length}一点的风格。")
        if len(parts) == 1:
            parts.append("后续我会按这些长期信息调整回答、出题和复盘重点。")
        return "".join(parts)

    def _compact_tool_results_for_prompt(self, tool_results: list[dict[str, Any]]) -> str:
        sections: list[str] = []
        for index, entry in enumerate(tool_results[-6:], start=1):
            tool_name = str(entry.get("tool_name") or "unknown_tool")
            reason = str(entry.get("reason") or "")
            arguments = json.dumps(entry.get("arguments") or {}, ensure_ascii=False, sort_keys=True)
            result_text = observation_to_text(dict(entry.get("result") or {}))
            sections.append(
                f"步骤{index} 工具：{tool_name}\n原因：{reason}\n参数：{arguments}\n结果：\n{result_text}"
            )
        return truncate_text("\n\n".join(sections), 3600)

    def _fallback_answer_from_tools(
        self,
        question: str,
        tool_results: list[dict[str, Any]],
        *,
        hits: list[dict[str, Any]],
        statute_results: list[dict[str, Any]] | None = None,
    ) -> str | None:
        statute_results = list(statute_results or [])
        tool_names = {str(entry.get("tool_name") or "") for entry in tool_results}
        if "rag_search" in tool_names or "retrieve_from_kb" in tool_names:
            if hits or statute_results:
                return self._render_direct_qa_answer(question, hits, statute_results=statute_results)
            if self._looks_like_smalltalk(question):
                return None
            return "我已经结合当前上下文补做了检索，但这次还没有命中足够强相关的材料。你可以补充更具体的争点、法条名称、题型或事实背景，我再继续分析。"
        if "profile_upsert" in tool_names:
            return self._render_profile_update_ack(tool_results)
        return None

    def _synthesize_tool_based_answer(
        self,
        question: str,
        tool_results: list[dict[str, Any]],
        *,
        model_path: str | None,
        adapter_path: str | None,
        model_device: str,
        fallback_answer: str | None,
    ) -> str | None:
        if not tool_results:
            return fallback_answer
        try:
            model = self._load_runtime_model(model_path=model_path, adapter_path=adapter_path, model_device=model_device)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是法律学习 Agent 的最终答复整合器。现在不能再调用任何工具。"
                        "你必须只基于最新用户输入与已经执行完毕的工具结果来写最终中文答复。"
                        "先正面回应用户这一轮真正的需求，不要把回复固定成“已更新档案”之类模板。"
                        "如果本轮顺便更新了画像，要自然说明你记住了哪些长期信息，并继续回应用户当前问题。"
                        "如果检索结果不足，就明确说明不足，并指出还需要什么信息或下一步可以做什么。"
                        "不要编造工具结果中没有出现的法条、事实、分数或结论。"
                        "不要输出 Thought、Action、Observation 或 Final Answer 标签。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"最新用户输入：\n{question}\n\n"
                        f"已执行的工具结果：\n{self._compact_tool_results_for_prompt(tool_results)}\n\n"
                        f"如果工具结果不足时的保底答复：\n{fallback_answer or '无'}\n\n"
                        "请直接输出最终答复。"
                    ),
                },
            ]
            output = model.generate(
                messages,
                max_new_tokens=min(self.legal_config.inference.max_new_tokens, 320),
                temperature=0.0,
                top_p=1.0,
                top_k=self.legal_config.inference.top_k,
                presence_penalty=1.0,
                enable_thinking=False,
            )
            answer_text = str(output.content or output.raw_text or "").strip()
            parsed = parse_react_output(answer_text)
            if parsed.kind == "final" and parsed.final_answer:
                answer_text = parsed.final_answer.strip()
            else:
                answer_text = answer_text.replace("Final Answer:", "").strip()
            return answer_text or fallback_answer
        except Exception:
            return fallback_answer

    def _build_direct_exam_payload(
        self,
        *,
        user_id: str,
        session_id: str,
        topic: str,
        question_count: int,
        exam_type: str,
        question_types: list[str] | None = None,
        profile_update_steps: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        tool_results = list(profile_update_steps or [])
        if not any(str(entry.get("tool_name") or "") == "profile_view" for entry in tool_results):
            tool_results.append(
                {
                    "tool_name": "profile_view",
                    "reason": "先读取当前画像，用于确定选题偏好。",
                    "arguments": {},
                    "result": self.tool_executor.execute("profile_view", {}, user_id=user_id, session_id=session_id),
                }
            )
        tool_results.append(self._prepare_context_step(topic, user_id=user_id, session_id=session_id, reason="读取长期画像与当前会话摘要，辅助出题。"))
        tool_results.append(
            {
                "tool_name": "generate_exam",
                "reason": "依据画像与题型偏好生成模拟题。",
                "arguments": {"topic": topic, "question_count": question_count, "exam_type": exam_type, "question_types": question_types or DEFAULT_EXAM_QUESTION_TYPES},
                "result": self.tool_executor.execute(
                    "generate_exam",
                    {"topic": topic, "question_count": question_count, "exam_type": exam_type, "question_types": question_types or DEFAULT_EXAM_QUESTION_TYPES},
                    user_id=user_id,
                    session_id=session_id,
                ),
            }
        )
        trace = " -> ".join(str(entry.get("tool_name") or "") for entry in tool_results)
        return {"tool_results": tool_results, "answer": self._render_exam_answer(tool_results), "trace": trace}

    def _build_direct_report_payload(
        self,
        *,
        user_id: str,
        session_id: str,
        report_type: str,
        profile_update_steps: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        tool_results = list(profile_update_steps or [])
        tool_results.append(self._prepare_context_step(report_type, user_id=user_id, session_id=session_id, reason="先读取近期会话、画像与测试信息。"))
        tool_results.append(
            {
                "tool_name": "generate_report",
                "reason": "生成会话级学习报告。",
                "arguments": {"report_type": report_type},
                "result": self.tool_executor.execute("generate_report", {"report_type": report_type}, user_id=user_id, session_id=session_id),
            }
        )
        trace = " -> ".join(str(entry.get("tool_name") or "") for entry in tool_results)
        return {"tool_results": tool_results, "answer": "已生成学习报告，内容和下载入口已同步到右侧面板。", "trace": trace}

    def _build_direct_qa_payload(
        self,
        question: str,
        *,
        retrieval_device: str,
        user_id: str,
        session_id: str,
        profile_update_steps: list[dict[str, Any]] | None,
        model_path: str | None,
        adapter_path: str | None,
        model_device: str,
    ) -> dict[str, Any] | None:
        tool_results = list(profile_update_steps or [])
        context_result = self.tool_executor.execute(
            "prepare_context",
            {"query": question},
            user_id=user_id,
            session_id=session_id,
        )
        tool_results.append(
            {
                "tool_name": "prepare_context",
                "reason": "先读取用户画像、长期记忆与近期会话上下文。",
                "arguments": {"query": question},
                "result": context_result,
            }
        )
        run_rag = self._needs_rag_search(question, has_profile_update=bool(profile_update_steps))
        rag_result: dict[str, Any] = {"results": []}
        if run_rag:
            rag_result = self.tool_executor.execute(
                "rag_search",
                {"query": question, "top_k": self.config.retrieval_top_k},
                user_id=user_id,
                session_id=session_id,
            )
            tool_results.append(
                {
                    "tool_name": "rag_search",
                    "reason": "综合检索法规、题库、案例和学习常识。",
                    "arguments": {"query": question, "top_k": self.config.retrieval_top_k},
                    "result": rag_result,
                }
            )
        statute_payload = None
        if run_rag and self.config.use_legacy_statute_rag and self.config.legacy_config_path is not None:
            retriever = _get_retriever_cached(self.legal_config_path, retrieval_device)
            statute_payload = UnifiedToolRegistry(
                retriever,
                study_tool_executor=self.tool_executor,
                user_id=user_id,
                session_id=session_id,
                interactive=False,
            ).execute("retrieve_from_kb", {"query": question, "top_k": self.config.retrieval_top_k})
            tool_results.append(
                {
                    "tool_name": "retrieve_from_kb",
                    "reason": "补充一次法规知识库检索，确保兜底答复仍带有法条层依据。",
                    "arguments": {"query": question, "top_k": self.config.retrieval_top_k},
                    "result": statute_payload,
                }
            )
        hits = self._select_relevant_knowledge_hits(question, list(rag_result.get("results") or [])) if run_rag else []
        statute_results = self._select_relevant_statute_results(question, statute_payload)
        fallback_answer = self._fallback_answer_from_tools(question, tool_results, hits=hits, statute_results=statute_results)
        answer = fallback_answer
        should_synthesize = bool(profile_update_steps) or (
            run_rag and not (self._looks_like_smalltalk(question) and not hits and not statute_results)
        )
        if should_synthesize:
            answer = self._synthesize_tool_based_answer(
                question,
                tool_results,
                model_path=model_path,
                adapter_path=adapter_path,
                model_device=model_device,
                fallback_answer=fallback_answer,
            )
        trace = " -> ".join(str(entry.get("tool_name") or "") for entry in tool_results)
        return {"tool_results": tool_results, "answer": answer, "trace": trace}

    def _render_direct_qa_answer(
        self,
        question: str,
        hits: list[dict[str, Any]],
        *,
        statute_results: list[dict[str, Any]] | None = None,
    ) -> str:
        statute_results = list(statute_results or [])
        lead_parts: list[str] = []
        if hits:
            lead_parts.append(self._summarize_knowledge_hit(hits[0]))
        if statute_results:
            lead_parts.append(self._summarize_statute_result(statute_results[0]))
        lead = " ".join(part for part in lead_parts if part).strip()
        if not lead:
            lead = f"我已围绕“{question}”补做检索，但当前只能确认需要结合更具体事实继续判断。"
        lines = [lead]
        reference_lines: list[str] = []
        seen_keys: set[str] = set()
        for item in hits[:3]:
            key = f"hit::{item.get('source_type')}::{item.get('record_id')}::{item.get('title')}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            reference_lines.append(self._render_reference_line(item))
        for item in statute_results[:2]:
            key = f"statute::{item.get('document_title')}::{item.get('article_heading')}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            reference_lines.append(self._render_statute_reference_line(item))
        if reference_lines:
            lines.append("")
            lines.append("更相关的参考材料：")
            lines.extend(reference_lines)
        lines.append("")
        lines.append("如果你愿意，我可以继续展开成法条依据、案例对比，或者直接生成同主题模拟测试。")
        return "\n".join(lines)

    def _direct_generate_exam_response(
        self,
        *,
        user_id: str,
        session_id: str,
        topic: str,
        question_count: int,
        exam_type: str,
        question_types: list[str] | None = None,
    ) -> StudyAgentResponse:
        payload = self._build_direct_exam_payload(
            user_id=user_id,
            session_id=session_id,
            topic=topic,
            question_count=question_count,
            exam_type=exam_type,
            question_types=question_types,
        )
        tool_results = payload["tool_results"]
        answer = payload["answer"]
        trace = self._render_exam_trace(tool_results, answer)
        self._record_turn(
            user_id,
            session_id,
            f"[UI操作] 生成模拟测试 exam_type={exam_type} topic={topic} question_count={question_count} question_types={','.join(question_types or DEFAULT_EXAM_QUESTION_TYPES)}",
            answer,
            tool_results,
            reasoning_trace=trace,
        )
        return StudyAgentResponse(
            intent="mock_exam_generate",
            answer=answer,
            plan={"planner_backend": "direct_tool", "tool_names": [entry["tool_name"] for entry in tool_results]},
            tool_results=tool_results,
            trace=trace,
        )

    def _direct_generate_report_response(
        self,
        *,
        user_id: str,
        session_id: str,
        report_type: str,
    ) -> StudyAgentResponse:
        tool_results = [
            self._prepare_context_step(report_type, user_id=user_id, session_id=session_id, reason="先读取近期会话、画像与测试信息。"),
            {
                "tool_name": "generate_report",
                "reason": "生成会话级学习报告。",
                "arguments": {"report_type": report_type},
                "result": self.tool_executor.execute("generate_report", {"report_type": report_type}, user_id=user_id, session_id=session_id),
            },
        ]
        report_path, report_markdown = self._extract_report_artifacts(tool_results)
        answer = self._report_chat_answer(report_markdown)
        trace = " -> ".join(str(entry.get("tool_name") or "") for entry in tool_results)
        self._record_turn(
            user_id,
            session_id,
            f"[UI操作] 生成学习报告 report_type={report_type}",
            answer,
            tool_results,
            reasoning_trace=trace,
        )
        return StudyAgentResponse(
            intent="report_generation",
            answer=answer,
            plan={"planner_backend": "direct_tool", "tool_names": [entry["tool_name"] for entry in tool_results]},
            tool_results=tool_results,
            report_path=report_path,
            report_markdown=report_markdown,
            trace=trace,
        )

    def _render_profile_answer(self, tool_results: list[dict[str, Any]]) -> str:
        profile = dict(tool_results[0].get("result", {}).get("profile") or {})
        return (
            "当前用户画像如下：\n"
            f"- 姓名：{profile.get('name') or '未设置'}\n"
            f"- 备考目标：{'、'.join(profile.get('study_goals') or []) or '未设置'}\n"
            f"- 薄弱点：{'、'.join(profile.get('weak_points') or []) or '暂无'}\n"
            f"- 强项：{'、'.join(profile.get('strong_points') or []) or '暂无'}"
        )

    def _render_exam_answer(self, tool_results: list[dict[str, Any]]) -> str:
        result_by_name = {entry["tool_name"]: entry["result"] for entry in tool_results}
        exam_payload = dict(result_by_name.get("generate_exam") or {})
        profile = dict(result_by_name.get("profile_view", {}).get("profile") or {})
        profile_tags = list(dict.fromkeys([*(profile.get("weak_points") or []), *(profile.get("study_goals") or [])]))
        selection_notes = [str(note).strip() for note in exam_payload.get("selection_notes", []) if str(note).strip()]
        question_types = [str(item) for item in exam_payload.get("question_types", []) if str(item).strip()]
        if question_types == ["single_choice"]:
            answer_instruction = "请按“1.A 2.B 3.C”这样的格式直接回复答案，我会统一评分、记录并生成复盘。"
        else:
            answer_instruction = "请按编号逐题作答。选择题可写成“1.A”，简答或案例题请写成“2. 你的答案”。我会统一评分、记录并生成复盘。"
        lines = [
            "以下是为当前用户生成的法考模拟测试题目：",
            f"主题：{exam_payload.get('topic', '综合')}，题型：{exam_payload.get('exam_type', '综合练习')}，共 {exam_payload.get('question_count', 0)} 题。",
            f"本次选题优先参考的画像标签：{'、'.join(profile_tags[:5]) or '当前画像暂无额外选题标签，已按当前题型与主题要求出题'}。",
            *selection_notes,
            answer_instruction,
            "",
        ]
        if exam_payload.get("exam_type") == "薄弱点强化":
            lines.insert(3, f"本轮按薄弱点强化回放错题库 {exam_payload.get('reused_wrong_question_count', 0)} 题。")
        else:
            lines.insert(3, "当前为常规练习模式，不会主动混入错题库题目。")
        for question in exam_payload.get("questions", []):
            question_type = str(question.get("question_type") or "single_choice")
            type_label = {
                "single_choice": "单选题",
                "short_answer": "简答题",
                "case_analysis": "案例分析题",
            }.get(question_type, question_type)
            lines.append(f"{question['index']}. [{type_label}] {question['question']}")
            for option_key, option_value in (question.get("options") or {}).items():
                lines.append(f"   {option_key}. {option_value}")
            lines.append("")
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines)

    def _render_score_answer(self, tool_results: list[dict[str, Any]]) -> str:
        result_by_name = {entry["tool_name"]: entry["result"] for entry in tool_results}
        score_payload = dict(result_by_name.get("score_exam") or {})
        report_payload = dict(result_by_name.get("generate_report") or {})
        score_percent = score_payload.get("score_percent", 0)
        earned_score = score_payload.get("earned_score", 0)
        total_score = score_payload.get("total_score", 0)
        wrong_questions = list(score_payload.get("wrong_questions") or [])
        details = list(score_payload.get("details") or [])
        weak_tags = list(score_payload.get("weak_tags") or [])
        strong_tags = list(score_payload.get("strong_tags") or [])
        mastered_count = int(score_payload.get("mastered_count") or 0)
        review_count = int(score_payload.get("review_count") or 0)
        incorrect_count = int(score_payload.get("incorrect_count") or 0)
        unanswered_count = int(score_payload.get("unanswered_count") or 0)

        lines = [
            f"本次测试得分：{score_percent} 分（{earned_score}/{total_score}）。",
            f"评分概览：掌握较稳 {mastered_count} 题，待巩固 {review_count} 题，错题 {incorrect_count} 题，未作答 {unanswered_count} 题。",
        ]
        if weak_tags:
            lines.append(f"本轮主要薄弱点：{'、'.join(weak_tags[:6])}。")
        if strong_tags:
            lines.append(f"本轮表现较稳的知识点：{'、'.join(strong_tags[:6])}。")
        if wrong_questions:
            wrong_indexes = [str(item.get("index")) for item in wrong_questions if item.get("index")]
            if wrong_indexes:
                lines.append(f"已记入错题库复盘的题目：第 {'、'.join(wrong_indexes)} 题。")
        elif incorrect_count == 0:
            lines.append("本轮没有需要记入错题库的题目。")

        if details:
            lines.extend(["", "逐题反馈："])
            for detail in details:
                index = detail.get("index")
                question_type = str(detail.get("question_type") or "single_choice")
                type_label = {
                    "single_choice": "单选题",
                    "short_answer": "简答题",
                    "case_analysis": "案例分析题",
                }.get(question_type, question_type)
                classification = str(detail.get("classification") or ("mastered" if detail.get("is_correct") else "incorrect"))
                status_label = {
                    "mastered": "掌握较稳",
                    "review": "待巩固",
                    "incorrect": "错题",
                    "unanswered": "未作答",
                }.get(classification, classification)
                prompt = str(detail.get("question") or "").strip()
                options = dict(detail.get("options") or {})
                user_answer = str(detail.get("display_user_answer") or detail.get("user_answer") or "").strip()
                correct_answer = str(detail.get("correct_answer") or "").strip()

                lines.append(f"第{index}题 [{type_label}] {status_label}")
                if prompt:
                    lines.append(
                        f"题干：{self._format_exam_feedback_text(prompt, question_type=question_type, field='question')}"
                    )
                lines.append(f"得分：{detail.get('score', 0)}/{detail.get('max_score', 20)}")
                if question_type == "single_choice":
                    lines.append(f"你的作答：{self._format_choice_answer(user_answer, options, unanswered_label='未作答')}")
                    lines.append(f"正确答案：{self._format_choice_answer(correct_answer, options, unanswered_label='未提供')}")
                else:
                    lines.append(
                        f"你的作答：{self._format_exam_feedback_text(user_answer, question_type=question_type, field='answer') if user_answer else '未作答'}"
                    )
                    feedback = str(detail.get("grading_feedback") or "").strip()
                    if feedback:
                        lines.append(f"评语：{feedback}")
                    matched_points = [str(item) for item in detail.get("matched_points", []) if str(item).strip()]
                    if matched_points:
                        lines.append(f"答到的要点：{'；'.join(matched_points[:3])}")
                    missing_points = [str(item) for item in detail.get("missing_points", []) if str(item).strip()]
                    if missing_points:
                        lines.append(f"待补要点：{'；'.join(missing_points[:3])}")
                    if correct_answer:
                        lines.append(
                            f"参考要点：{self._format_exam_feedback_text(correct_answer, question_type=question_type, field='reference')}"
                        )
                analysis = str(detail.get("analysis") or "").strip()
                if analysis:
                    lines.append(
                        f"解析：{self._format_exam_feedback_text(analysis, question_type=question_type, field='analysis')}"
                    )
                lines.append("")
            while lines and not lines[-1].strip():
                lines.pop()
        if str(report_payload.get("report_path") or "").strip():
            lines.append("学习反馈报告已同步到右侧面板，并写入报告目录。")
        return "\n".join(lines)

    def _format_exam_feedback_text(
        self,
        text: str | None,
        *,
        question_type: str,
        field: str,
    ) -> str:
        normalized = str(text or "").strip()
        if not normalized:
            return ""
        limits = {
            ("case_analysis", "question"): 1600,
            ("case_analysis", "answer"): 1800,
            ("case_analysis", "reference"): 1400,
            ("case_analysis", "analysis"): 1400,
            ("short_answer", "question"): 900,
            ("short_answer", "answer"): 1200,
            ("short_answer", "reference"): 1000,
            ("short_answer", "analysis"): 1000,
        }
        limit = limits.get((question_type, field), 360)
        if len(normalized) <= limit:
            return normalized
        return truncate_text(normalized, limit)

    def _format_choice_answer(
        self,
        answer: str | None,
        options: dict[str, Any],
        *,
        unanswered_label: str,
    ) -> str:
        normalized = str(answer or "").strip()
        if not normalized:
            return unanswered_label
        label = normalized[:1].upper()
        option_text = str(options.get(label) or "").strip()
        if option_text:
            return f"{label}. {option_text}"
        return normalized

    def _render_exam_trace(self, tool_results: list[dict[str, Any]], answer: str) -> str:
        trace_lines: list[str] = []
        for entry in tool_results:
            tool_name = entry.get("tool_name") or "unknown_tool"
            arguments = entry.get("arguments") or {}
            result = entry.get("result")
            trace_lines.append(
                f"Action: {tool_name}({json.dumps(arguments, ensure_ascii=False, sort_keys=True)})"
            )
            trace_lines.append(
                f"Observation: {json.dumps(result, ensure_ascii=False, indent=2, default=str)}"
            )
        return self._sync_trace_final_answer("\n".join(trace_lines), answer)


LegalStudyAgent = UnifiedLegalAgent
