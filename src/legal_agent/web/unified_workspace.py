from __future__ import annotations

import inspect
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr

try:
    import torch
except Exception:  # pragma: no cover - torch is an install-time dependency in production
    torch = None

from legal_agent.config import load_app_config
from legal_agent.study_agent import LegalStudyAgent
from legal_agent.study_config import load_study_agent_config
from legal_agent.web.model_registry import build_choice_map


DEFAULT_PROMPT_MODE = "pure"
DEFAULT_LIVE_STATUS = "等待输入。"
EXAM_TYPE_CHOICES = ["综合练习", "章节练习", "薄弱点强化", "真题模拟"]
QUESTION_TYPE_CHOICES = ["混合题型", "单选题", "简答题", "案例分析题"]
EXAM_TOPIC_CHOICES = ["综合", "民法", "刑法", "行政法", "商经", "民诉", "刑诉", "理论法"]
REPORT_TYPE_LABELS = {
    "学习进度报告": "study_progress",
    "用户画像与薄弱点报告": "weakness_diagnosis",
    "模拟测试复盘报告": "mock_exam_review",
}
SELECT_CLASSES = ["ui-select"]
FILTERABLE_SELECT_CLASSES = ["ui-select", "ui-select-filterable"]


def _chatbot_runtime_kwargs() -> dict[str, object]:
    chatbot_params = inspect.signature(gr.Chatbot.__init__).parameters
    if "type" in chatbot_params:
        return {"type": "messages"}
    return {}


def _new_ui_state() -> dict[str, Any]:
    return {
        "user_id": None,
        "session_id": None,
        "pending_root_question": None,
        "pending_question": None,
        "clarification_answers": [],
        "trace": "",
        "report_markdown": "",
        "report_path": None,
        "last_assistant_message": "",
        "live_status": DEFAULT_LIVE_STATUS,
    }


@lru_cache(maxsize=6)
def _get_agent(config_path: str, retrieval_device: str) -> LegalStudyAgent:
    config = load_study_agent_config(config_path)
    return LegalStudyAgent(config, retrieval_device=retrieval_device)


def _chat_messages(history: list[tuple[str, str]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for user_text, assistant_text in history:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
    return messages


def _latest_assistant_message(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def _existing_report_path(report_path: str | None) -> str | None:
    if not report_path:
        return None
    path = Path(report_path)
    if not path.exists() or not path.is_file():
        return None
    return str(path)


def _read_report(report_path: str | None) -> str:
    existing_path = _existing_report_path(report_path)
    if not existing_path:
        return ""
    try:
        return Path(existing_path).read_text(encoding="utf-8")
    except Exception:
        return ""


def _render_live_status(message: str | None) -> str:
    return f"### 当前阶段\n{str(message or DEFAULT_LIVE_STATUS).strip() or DEFAULT_LIVE_STATUS}"


def _render_report_panel(report_markdown: str) -> str:
    text = str(report_markdown or "").strip()
    if text:
        return text
    return "### 报告面板\n尚未生成报告。请先选择报告类型并点击“生成报告”。"


def _build_model_choices(config_path: str) -> tuple[list[str], str | None]:
    config = load_app_config(config_path)
    choice_map = build_choice_map(config)
    labels = list(choice_map.keys())
    if not labels:
        return [], None

    default_label = labels[0]
    configured_model_path = str(config.models.agent_base.resolve())
    for label, choice in choice_map.items():
        if choice.kind == "base" and str(choice.model_path.resolve()) == configured_model_path and choice.adapter_path is None:
            default_label = label
            break
    return labels, default_label


def _build_runtime_device_choices(config_path: str, current_value: str | None) -> tuple[list[str], str]:
    config = load_app_config(config_path)
    labels = ["auto", "cpu"]
    if torch is not None and torch.cuda.is_available():
        visible_gpu_count = torch.cuda.device_count()
        labels.extend(f"cuda:{gpu_id}" for gpu_id in config.available_gpu_ids if gpu_id < visible_gpu_count)
    if current_value and current_value not in labels:
        labels.append(current_value)
    default_value = current_value if current_value in labels else "auto"
    return labels, default_value


def _resolve_runtime_devices(runtime_device: str, default_retrieval_device: str) -> tuple[str, str]:
    chosen = (runtime_device or "auto").strip() or "auto"
    if chosen == "auto":
        return default_retrieval_device, "auto"
    return chosen, chosen


def _resolve_report_type(report_label: str) -> str:
    return REPORT_TYPE_LABELS.get(report_label, "study_progress")


def _resolve_question_types(question_type_label: str | None) -> list[str]:
    mapping = {
        "混合题型": ["single_choice", "short_answer", "case_analysis"],
        "单选题": ["single_choice"],
        "简答题": ["short_answer"],
        "案例分析题": ["case_analysis"],
    }
    label = str(question_type_label or "").strip()
    return list(mapping.get(label, mapping["混合题型"]))


def _ensure_selection(agent: LegalStudyAgent, state: dict[str, Any] | None) -> tuple[str, str, dict[str, Any]]:
    state = dict(state or _new_ui_state())
    users = agent.list_users()
    if not users:
        agent.create_user("demo_user", display_name="demo_user")
        users = agent.list_users()
    user_id = str(state.get("user_id") or users[0])
    if user_id not in users:
        user_id = users[0]

    sessions = agent.list_sessions(user_id)
    if not sessions:
        agent.create_session(user_id, "default_session")
        sessions = agent.list_sessions(user_id)
    session_ids = [str(item.get("session_id") or "") for item in sessions]
    session_id = str(state.get("session_id") or session_ids[0])
    if session_id not in session_ids:
        session_id = session_ids[0]

    state["user_id"] = user_id
    state["session_id"] = session_id
    return user_id, session_id, state


def _load_report_payload(agent: LegalStudyAgent, state: dict[str, Any], user_id: str, session_id: str) -> tuple[str, str | None]:
    report_markdown = str(state.get("report_markdown") or "")
    report_path = _existing_report_path(str(state.get("report_path") or "") or None)
    if report_markdown:
        return report_markdown, report_path

    if report_path:
        return _read_report(report_path), report_path

    session_state = agent.get_session_state(user_id, session_id)
    latest_report_path = _existing_report_path(str(session_state.get("last_report_path") or "") or None)
    if latest_report_path:
        return _read_report(latest_report_path), latest_report_path
    return "", None


def _build_status_markdown(agent: LegalStudyAgent, user_id: str, session_id: str, runtime_device: str, state: dict[str, Any]) -> str:
    session_state = agent.get_session_state(user_id, session_id)
    has_report = bool(_existing_report_path(str(session_state.get("last_report_path") or "") or None) or str(state.get("report_markdown") or "").strip())
    lines = [
        "### 当前工作区",
        f"- 用户：{user_id}",
        f"- 会话：{session_id}",
        f"- 运行设备：{runtime_device}",
        f"- 最近摘要：{session_state.get('summary') or '暂无'}",
        f"- 对话轮数：{len(session_state.get('turns') or [])}",
        f"- 报告状态：{'可查看 / 可下载' if has_report else '未生成'}",
    ]
    active_exam = session_state.get("active_exam_session_id")
    if active_exam:
        lines.append(f"- 当前激活测试：{active_exam}")
    pending_question = str(state.get("pending_question") or "").strip()
    if pending_question:
        lines.append(f"- 待补充事实：{pending_question}")
    return "\n".join(lines)


def _refresh_workspace_data(
    state: dict[str, Any] | None,
    *,
    config_path: str,
    study_config_path: str,
    runtime_device: str,
    default_retrieval_device: str,
) -> dict[str, Any]:
    retrieval_device, _ = _resolve_runtime_devices(runtime_device, default_retrieval_device)
    agent = _get_agent(study_config_path, retrieval_device)
    user_id, session_id, state = _ensure_selection(agent, state)
    history = agent.get_session_history(user_id, session_id)
    messages = _chat_messages(history)
    report_markdown, report_path = _load_report_payload(agent, state, user_id, session_id)
    state["report_markdown"] = report_markdown
    state["report_path"] = report_path
    state["last_assistant_message"] = _latest_assistant_message(messages)
    state["live_status"] = str(state.get("live_status") or DEFAULT_LIVE_STATUS)
    users = agent.list_users()
    sessions = agent.list_sessions(user_id)
    session_ids = [str(item.get("session_id") or "") for item in sessions]
    return {
        "chat_messages": messages,
        "trace": str(state.get("trace") or ""),
        "state": state,
        "status_markdown": _build_status_markdown(agent, user_id, session_id, runtime_device, state),
        "report_display_markdown": _render_report_panel(report_markdown),
        "report_path": report_path,
        "user_choices": users,
        "user_value": user_id,
        "session_choices": session_ids,
        "session_value": session_id,
        "live_status": str(state.get("live_status") or DEFAULT_LIVE_STATUS),
        "reply_copy_text": str(state.get("last_assistant_message") or ""),
        "report_copy_text": report_markdown,
    }


def _workspace_outputs(payload: dict[str, Any], *, live_status: str | None = None):
    return (
        list(payload["chat_messages"]),
        str(payload["trace"]),
        dict(payload["state"]),
        str(payload["report_display_markdown"]),
        payload["report_path"],
        gr.Dropdown(choices=list(payload["user_choices"]), value=payload["user_value"]),
        gr.Dropdown(choices=list(payload["session_choices"]), value=payload["session_value"]),
    )


def _workspace_outputs_with_input(payload: dict[str, Any], *, live_status: str | None = None, question_value: str = ""):
    return (*_workspace_outputs(payload, live_status=live_status), question_value)


def _resolve_model_choice(config_path: str, model_label: str) -> tuple[str | None, str | None]:
    config = load_app_config(config_path)
    choice_map = build_choice_map(config)
    choice = choice_map.get(model_label)
    if choice is None:
        return None, None
    return str(choice.model_path), str(choice.adapter_path) if choice.adapter_path else None


def _compose_followup_question(state: dict[str, Any], latest_answer: str) -> tuple[str, list[dict[str, str]]]:
    clarification_answers = list(state.get("clarification_answers", []))
    pending_question = str(state.get("pending_question") or "请补充关键事实")
    clarification_answers.append({"question": pending_question, "answer": latest_answer.strip()})
    answer_lines = [
        f"- {item['question']}：{item['answer']}"
        for item in clarification_answers
        if item.get("question") and item.get("answer")
    ]
    joined_answers = "\n".join(answer_lines)
    prompt = (
        f"原始法律/学习问题：{state.get('pending_root_question') or ''}\n"
        "用户刚刚补充了以下事实，请继续同一任务分析；若这些信息已经足够支撑结论，请直接给出答案，不要机械重复追问。\n"
        f"已补充事实：\n{joined_answers}"
    )
    return prompt, clarification_answers


def _submit_chat(
    question: str,
    state: dict[str, Any] | None,
    model_label: str,
    config_path: str,
    study_config_path: str,
    runtime_device: str,
    default_retrieval_device: str,
):
    state = dict(state or _new_ui_state())
    runtime_device = runtime_device or "auto"
    base_payload = _refresh_workspace_data(
        state,
        config_path=config_path,
        study_config_path=study_config_path,
        runtime_device=runtime_device,
        default_retrieval_device=default_retrieval_device,
    )
    if not question.strip():
        state["live_status"] = "请输入问题后再发送。"
        base_payload["state"] = state
        base_payload["live_status"] = state["live_status"]
        yield _workspace_outputs_with_input(base_payload, live_status=state["live_status"], question_value=question)
        return

    model_path, adapter_path = _resolve_model_choice(config_path, model_label)
    if model_path is None:
        state["trace"] = "未找到对应模型，请先刷新模型列表。"
        state["live_status"] = "模型选择无效。"
        error_payload = _refresh_workspace_data(
            state,
            config_path=config_path,
            study_config_path=study_config_path,
            runtime_device=runtime_device,
            default_retrieval_device=default_retrieval_device,
        )
        yield _workspace_outputs_with_input(error_payload, live_status=state["live_status"], question_value="")
        return

    retrieval_device, model_device = _resolve_runtime_devices(runtime_device, default_retrieval_device)
    agent = _get_agent(study_config_path, retrieval_device)
    user_id, session_id, state = _ensure_selection(agent, state)
    effective_question = question
    clarification_answers = list(state.get("clarification_answers", []))
    if state.get("pending_question"):
        effective_question, clarification_answers = _compose_followup_question(state, question)
    elif not state.get("pending_root_question"):
        state["pending_root_question"] = question

    live_chat = _chat_messages(agent.get_session_history(user_id, session_id))
    live_chat.append({"role": "user", "content": question})
    live_chat.append({"role": "assistant", "content": "已接收问题，正在规划下一步。"})
    state["live_status"] = "已接收问题，正在规划下一步。"
    interim_payload = dict(base_payload)
    interim_payload["chat_messages"] = live_chat
    interim_payload["trace"] = str(state.get("trace") or "")
    interim_payload["state"] = state
    interim_payload["reply_copy_text"] = "已接收问题，正在规划下一步。"
    interim_payload["live_status"] = state["live_status"]
    yield _workspace_outputs_with_input(interim_payload, live_status=state["live_status"], question_value=question)

    try:
        for update in agent.stream_message(
            effective_question,
            user_id=user_id,
            session_id=session_id,
            model_path=model_path,
            adapter_path=adapter_path,
            prompt_mode=DEFAULT_PROMPT_MODE,
            retrieval_device=retrieval_device,
            model_device=model_device,
            allow_button_only_intents=False,
            display_question=question,
        ):
            event = str(update.get("event") or "status")
            if event == "final":
                response = update.get("response")
                if response is None:
                    raise RuntimeError("统一 Agent 流式执行未返回最终响应。")
                final_answer = str(getattr(response, "answer", "") or update.get("message") or "已完成本轮处理。")
                state["trace"] = str(getattr(response, "trace", "") or update.get("trace") or state.get("trace") or "")
                state["report_markdown"] = getattr(response, "report_markdown", None) or state.get("report_markdown") or ""
                state["report_path"] = getattr(response, "report_path", None) or state.get("report_path")
                state["last_assistant_message"] = final_answer
                state["live_status"] = "已完成分析。"
                final_chat = list(live_chat)
                final_chat[-1] = {"role": "assistant", "content": final_answer}
                if getattr(response, "needs_user_input", False) and getattr(response, "clarification_question", None):
                    state["pending_question"] = response.clarification_question
                    state["clarification_answers"] = clarification_answers
                    state["pending_root_question"] = str(state.get("pending_root_question") or question)
                else:
                    state["pending_question"] = None
                    state["clarification_answers"] = []
                    state["pending_root_question"] = None

                final_payload = dict(base_payload)
                final_payload["chat_messages"] = final_chat
                final_payload["trace"] = state["trace"]
                final_payload["state"] = state
                final_payload["report_display_markdown"] = _render_report_panel(str(state.get("report_markdown") or ""))
                final_payload["report_path"] = state.get("report_path")
                final_payload["user_choices"] = list(base_payload["user_choices"])
                final_payload["user_value"] = user_id
                final_payload["session_choices"] = list(base_payload["session_choices"])
                final_payload["session_value"] = session_id
                final_payload["reply_copy_text"] = final_answer
                final_payload["report_copy_text"] = str(state.get("report_markdown") or "")
                final_payload["live_status"] = state["live_status"]
                yield _workspace_outputs_with_input(final_payload, live_status=state["live_status"], question_value="")
                return

            state["trace"] = str(update.get("trace") or state.get("trace") or "")
            state["live_status"] = str(update.get("message") or "正在处理中。")
            live_payload = dict(base_payload)
            live_payload["chat_messages"] = list(live_chat[:-1]) + [{"role": "assistant", "content": state["live_status"]}]
            live_payload["trace"] = state["trace"]
            live_payload["state"] = state
            live_payload["reply_copy_text"] = state["live_status"]
            live_payload["live_status"] = state["live_status"]
            yield _workspace_outputs_with_input(live_payload, live_status=state["live_status"], question_value=question)

        raise RuntimeError("统一 Agent 流式执行未返回最终结果。")
    except Exception as exc:
        error_message = f"处理失败：{exc}"
        live_chat[-1] = {"role": "assistant", "content": error_message}
        state["trace"] = (str(state.get("trace") or "") + f"\n\n[UI Error]\n{exc}").strip()
        state["last_assistant_message"] = error_message
        state["live_status"] = "本轮处理失败。"
        error_payload = dict(base_payload)
        error_payload["chat_messages"] = live_chat
        error_payload["trace"] = state["trace"]
        error_payload["state"] = state
        error_payload["reply_copy_text"] = error_message
        error_payload["live_status"] = state["live_status"]
        yield _workspace_outputs_with_input(error_payload, live_status=state["live_status"], question_value=question)


def _run_exam_action(
    exam_type: str,
    topic: str,
    question_count: int,
    state: dict[str, Any] | None,
    model_label: str,
    config_path: str,
    study_config_path: str,
    runtime_device: str,
    default_retrieval_device: str,
    question_type_label: str = "混合题型",
):
    state = dict(state or _new_ui_state())
    model_path, adapter_path = _resolve_model_choice(config_path, model_label)
    retrieval_device, model_device = _resolve_runtime_devices(runtime_device, default_retrieval_device)
    agent = _get_agent(study_config_path, retrieval_device)
    user_id, session_id, state = _ensure_selection(agent, state)
    effective_topic = (topic or "综合").strip() or "综合"
    response = agent.generate_exam(
        user_id=user_id,
        session_id=session_id,
        topic=effective_topic,
        question_count=question_count,
        exam_type=exam_type,
        question_types=_resolve_question_types(question_type_label),
        model_path=model_path,
        adapter_path=adapter_path,
        prompt_mode=DEFAULT_PROMPT_MODE,
        retrieval_device=retrieval_device,
        model_device=model_device,
    )
    state["trace"] = response.trace
    state["pending_root_question"] = None
    state["pending_question"] = None
    state["clarification_answers"] = []
    state["live_status"] = f"已生成{exam_type}：{effective_topic}，题型为{question_type_label}。"
    refreshed = _refresh_workspace_data(
        state,
        config_path=config_path,
        study_config_path=study_config_path,
        runtime_device=runtime_device,
        default_retrieval_device=default_retrieval_device,
    )
    return _workspace_outputs(refreshed, live_status=state["live_status"])


def _run_report_action(
    report_label: str,
    state: dict[str, Any] | None,
    model_label: str,
    config_path: str,
    study_config_path: str,
    runtime_device: str,
    default_retrieval_device: str,
):
    state = dict(state or _new_ui_state())
    model_path, adapter_path = _resolve_model_choice(config_path, model_label)
    retrieval_device, model_device = _resolve_runtime_devices(runtime_device, default_retrieval_device)
    agent = _get_agent(study_config_path, retrieval_device)
    user_id, session_id, state = _ensure_selection(agent, state)
    response = agent.generate_report_response(
        user_id=user_id,
        session_id=session_id,
        report_type=_resolve_report_type(report_label),
        model_path=model_path,
        adapter_path=adapter_path,
        prompt_mode=DEFAULT_PROMPT_MODE,
        retrieval_device=retrieval_device,
        model_device=model_device,
    )
    state["trace"] = response.trace
    state["report_markdown"] = response.report_markdown or ""
    state["report_path"] = response.report_path
    state["last_assistant_message"] = response.answer
    state["live_status"] = "已生成报告并同步到聊天区与报告面板。"
    refreshed = _refresh_workspace_data(
        state,
        config_path=config_path,
        study_config_path=study_config_path,
        runtime_device=runtime_device,
        default_retrieval_device=default_retrieval_device,
    )
    return _workspace_outputs(refreshed, live_status=state["live_status"])


def _switch_user(
    user_id: str,
    state: dict[str, Any] | None,
    config_path: str,
    study_config_path: str,
    runtime_device: str,
    default_retrieval_device: str,
):
    state = dict(state or _new_ui_state())
    state["user_id"] = user_id
    state["session_id"] = None
    state["pending_root_question"] = None
    state["pending_question"] = None
    state["clarification_answers"] = []
    state["trace"] = ""
    state["report_markdown"] = ""
    state["report_path"] = None
    state["last_assistant_message"] = ""
    state["live_status"] = f"已切换到用户 {user_id}。"
    payload = _refresh_workspace_data(
        state,
        config_path=config_path,
        study_config_path=study_config_path,
        runtime_device=runtime_device,
        default_retrieval_device=default_retrieval_device,
    )
    return _workspace_outputs(payload, live_status=state["live_status"])


def _switch_session(
    session_id: str,
    state: dict[str, Any] | None,
    config_path: str,
    study_config_path: str,
    runtime_device: str,
    default_retrieval_device: str,
):
    state = dict(state or _new_ui_state())
    state["session_id"] = session_id
    state["pending_root_question"] = None
    state["pending_question"] = None
    state["clarification_answers"] = []
    state["trace"] = ""
    state["report_markdown"] = ""
    state["report_path"] = None
    state["last_assistant_message"] = ""
    state["live_status"] = f"已切换到会话 {session_id}。"
    payload = _refresh_workspace_data(
        state,
        config_path=config_path,
        study_config_path=study_config_path,
        runtime_device=runtime_device,
        default_retrieval_device=default_retrieval_device,
    )
    return _workspace_outputs(payload, live_status=state["live_status"])


def _create_user_action(
    new_user_id: str,
    state: dict[str, Any] | None,
    config_path: str,
    study_config_path: str,
    runtime_device: str,
    default_retrieval_device: str,
):
    retrieval_device, _ = _resolve_runtime_devices(runtime_device, default_retrieval_device)
    agent = _get_agent(study_config_path, retrieval_device)
    candidate = (new_user_id or "").strip() or f"user_{len(agent.list_users()) + 1}"
    agent.create_user(candidate, display_name=candidate)
    state = dict(state or _new_ui_state())
    state["user_id"] = candidate
    state["session_id"] = None
    state["trace"] = ""
    state["report_markdown"] = ""
    state["report_path"] = None
    state["last_assistant_message"] = ""
    state["live_status"] = f"已创建用户 {candidate}。"
    payload = _refresh_workspace_data(
        state,
        config_path=config_path,
        study_config_path=study_config_path,
        runtime_device=runtime_device,
        default_retrieval_device=default_retrieval_device,
    )
    return (*_workspace_outputs(payload, live_status=state["live_status"]), "")


def _delete_user_action(
    state: dict[str, Any] | None,
    config_path: str,
    study_config_path: str,
    runtime_device: str,
    default_retrieval_device: str,
):
    retrieval_device, _ = _resolve_runtime_devices(runtime_device, default_retrieval_device)
    agent = _get_agent(study_config_path, retrieval_device)
    user_id, _, state = _ensure_selection(agent, state)
    if len(agent.list_users()) > 1:
        agent.delete_user(user_id)
        state["user_id"] = None
        state["session_id"] = None
        state["live_status"] = f"已删除用户 {user_id}。"
    else:
        state["live_status"] = "至少保留一个用户，未执行删除。"
    payload = _refresh_workspace_data(
        state,
        config_path=config_path,
        study_config_path=study_config_path,
        runtime_device=runtime_device,
        default_retrieval_device=default_retrieval_device,
    )
    return _workspace_outputs(payload, live_status=state["live_status"])


def _create_session_action(
    new_session_id: str,
    state: dict[str, Any] | None,
    config_path: str,
    study_config_path: str,
    runtime_device: str,
    default_retrieval_device: str,
):
    retrieval_device, _ = _resolve_runtime_devices(runtime_device, default_retrieval_device)
    agent = _get_agent(study_config_path, retrieval_device)
    user_id, _, state = _ensure_selection(agent, state)
    candidate = (new_session_id or "").strip() or f"session_{len(agent.list_sessions(user_id)) + 1}"
    agent.create_session(user_id, candidate)
    state["session_id"] = candidate
    state["pending_root_question"] = None
    state["pending_question"] = None
    state["clarification_answers"] = []
    state["trace"] = ""
    state["report_markdown"] = ""
    state["report_path"] = None
    state["last_assistant_message"] = ""
    state["live_status"] = f"已创建会话 {candidate}。"
    payload = _refresh_workspace_data(
        state,
        config_path=config_path,
        study_config_path=study_config_path,
        runtime_device=runtime_device,
        default_retrieval_device=default_retrieval_device,
    )
    return (*_workspace_outputs(payload, live_status=state["live_status"]), "")


def _delete_session_action(
    state: dict[str, Any] | None,
    config_path: str,
    study_config_path: str,
    runtime_device: str,
    default_retrieval_device: str,
):
    retrieval_device, _ = _resolve_runtime_devices(runtime_device, default_retrieval_device)
    agent = _get_agent(study_config_path, retrieval_device)
    user_id, session_id, state = _ensure_selection(agent, state)
    sessions = agent.list_sessions(user_id)
    if len(sessions) > 1:
        agent.delete_session(user_id, session_id)
        state["session_id"] = None
        state["live_status"] = f"已删除会话 {session_id}。"
    else:
        state["live_status"] = "至少保留一个会话，未执行删除。"
    state["pending_root_question"] = None
    state["pending_question"] = None
    state["clarification_answers"] = []
    state["trace"] = ""
    state["report_markdown"] = ""
    state["report_path"] = None
    state["last_assistant_message"] = ""
    payload = _refresh_workspace_data(
        state,
        config_path=config_path,
        study_config_path=study_config_path,
        runtime_device=runtime_device,
        default_retrieval_device=default_retrieval_device,
    )
    return _workspace_outputs(payload, live_status=state["live_status"])


def build_unified_workspace(
    *,
    config_path: str,
    study_config_path: str,
    retrieval_device: str,
) -> None:
    model_labels, default_model = _build_model_choices(config_path)
    runtime_choices, default_runtime_device = _build_runtime_device_choices(config_path, "auto")
    initial = _refresh_workspace_data(
        _new_ui_state(),
        config_path=config_path,
        study_config_path=study_config_path,
        runtime_device=default_runtime_device,
        default_retrieval_device=retrieval_device,
    )

    workspace_state = gr.State(initial["state"])
    config_state = gr.State(config_path)
    study_config_state = gr.State(study_config_path)
    default_retrieval_state = gr.State(retrieval_device)

    with gr.Row(equal_height=True, elem_id="workspace-row"):
        with gr.Column(scale=5, min_width=1, elem_classes=["panel-card"], elem_id="left-panel"):
            gr.Markdown("## 运行配置")
            model_dropdown = gr.Dropdown(
                choices=model_labels,
                value=default_model,
                label="模型",
                interactive=True,
                filterable=False,
                elem_id="model-select",
                elem_classes=SELECT_CLASSES,
            )
            with gr.Row(elem_id="config-device-row"):
                runtime_device_dropdown = gr.Dropdown(
                    choices=runtime_choices,
                    value=default_runtime_device,
                    label="运行设备",
                    scale=4,
                    interactive=True,
                    filterable=False,
                    elem_id="runtime-device-select",
                    elem_classes=SELECT_CLASSES,
                )
                refresh_models = gr.Button("刷新模型", variant="secondary", scale=1, min_width=92, elem_id="refresh-model-button")

            gr.Markdown("## 用户与会话")
            with gr.Row(elem_id="user-session-row"):
                user_dropdown = gr.Dropdown(
                    choices=initial["user_choices"],
                    value=initial["user_value"],
                    label="用户",
                    scale=1,
                    interactive=True,
                    filterable=False,
                    elem_id="user-select",
                    elem_classes=SELECT_CLASSES,
                )
                session_dropdown = gr.Dropdown(
                    choices=initial["session_choices"],
                    value=initial["session_value"],
                    label="会话",
                    scale=1,
                    interactive=True,
                    filterable=False,
                    elem_id="session-select",
                    elem_classes=SELECT_CLASSES,
                )
            with gr.Row(elem_id="new-entity-row"):
                new_user = gr.Textbox(label="新用户", lines=1, scale=1, placeholder="例如：student_a")
                new_session = gr.Textbox(label="新会话", lines=1, scale=1, placeholder="例如：民法冲刺")
            with gr.Row(elem_id="crud-button-row"):
                create_user_btn = gr.Button("创建用户", variant="secondary")
                delete_user_btn = gr.Button("删除用户", variant="secondary")
                create_session_btn = gr.Button("创建会话", variant="secondary")
                delete_session_btn = gr.Button("删除会话", variant="secondary")

            gr.Markdown("## 学习动作")
            with gr.Group(elem_id="action-section"):
                exam_type = gr.Dropdown(
                    choices=EXAM_TYPE_CHOICES,
                    value=EXAM_TYPE_CHOICES[0],
                    label="练习模式",
                    interactive=True,
                    filterable=False,
                    elem_id="exam-type-select",
                    elem_classes=SELECT_CLASSES,
                )
                exam_question_type = gr.Dropdown(
                    choices=QUESTION_TYPE_CHOICES,
                    value=QUESTION_TYPE_CHOICES[0],
                    label="题型方案",
                    interactive=True,
                    filterable=False,
                    elem_id="exam-question-type-select",
                    elem_classes=SELECT_CLASSES,
                )
                exam_topic = gr.Dropdown(
                    choices=EXAM_TOPIC_CHOICES,
                    value="综合",
                    label="测试主题",
                    allow_custom_value=True,
                    interactive=True,
                    elem_id="exam-topic-select",
                    elem_classes=FILTERABLE_SELECT_CLASSES,
                )
                exam_count = gr.Slider(label="测试题量", minimum=1, maximum=10, step=1, value=5)
                report_type = gr.Dropdown(
                    choices=list(REPORT_TYPE_LABELS.keys()),
                    value="学习进度报告",
                    label="报告类型",
                    interactive=True,
                    filterable=False,
                    elem_id="report-type-select",
                    elem_classes=SELECT_CLASSES,
                )
                with gr.Row(elem_id="action-button-row"):
                    exam_button = gr.Button("生成测试", variant="secondary")
                    report_button = gr.Button("生成报告", variant="secondary")

            gr.Markdown("## Agent Trace")
            trace_box = gr.Textbox(label="思考轨迹 / Tool Trace", value=initial["trace"], lines=18, max_lines=18, elem_id="trace-component")

        with gr.Column(scale=7, min_width=1, elem_classes=["panel-card"], elem_id="right-panel"):
            gr.Markdown("## 对话窗口")
            chatbot = gr.Chatbot(
                value=initial["chat_messages"],
                show_label=False,
                elem_id="history-component",
                layout="bubble",
                height=520,
                min_height=360,
                max_height=720,
                **_chatbot_runtime_kwargs(),
            )
            with gr.Column(elem_id="input-stack"):
                user_input = gr.Textbox(
                    label="输入问题或答案",
                    lines=2,
                    max_lines=2,
                    placeholder="例如：请解释为什么这题选 C；或者直接回复 1.A 2.B 3.C 进行评分。",
                    elem_id="question-input",
                )
                submit_button = gr.Button("发送", variant="primary", elem_id="send-button")

            with gr.Group(elem_id="report-section"):
                gr.Markdown("## 报告内容")
                report_markdown = gr.Markdown(initial["report_display_markdown"], elem_id="report-component")
                report_file = gr.File(value=initial["report_path"], label="报告下载", interactive=False)

    refresh_models.click(
        fn=lambda path: gr.Dropdown(choices=_build_model_choices(path)[0], value=_build_model_choices(path)[1]),
        inputs=[config_state],
        outputs=[model_dropdown],
    )

    runtime_device_dropdown.change(
        fn=lambda current_state, device, default_device: _workspace_outputs(
            _refresh_workspace_data(
                current_state,
                config_path=config_path,
                study_config_path=study_config_path,
                runtime_device=device,
                default_retrieval_device=default_device,
            ),
            live_status=f"已切换运行设备为 {device}。",
        ),
        inputs=[workspace_state, runtime_device_dropdown, default_retrieval_state],
        outputs=[chatbot, trace_box, workspace_state, report_markdown, report_file, user_dropdown, session_dropdown],
    )

    user_dropdown.change(
        fn=_switch_user,
        inputs=[user_dropdown, workspace_state, config_state, study_config_state, runtime_device_dropdown, default_retrieval_state],
        outputs=[chatbot, trace_box, workspace_state, report_markdown, report_file, user_dropdown, session_dropdown],
    )
    session_dropdown.change(
        fn=_switch_session,
        inputs=[session_dropdown, workspace_state, config_state, study_config_state, runtime_device_dropdown, default_retrieval_state],
        outputs=[chatbot, trace_box, workspace_state, report_markdown, report_file, user_dropdown, session_dropdown],
    )
    create_user_btn.click(
        fn=_create_user_action,
        inputs=[new_user, workspace_state, config_state, study_config_state, runtime_device_dropdown, default_retrieval_state],
        outputs=[chatbot, trace_box, workspace_state, report_markdown, report_file, user_dropdown, session_dropdown, new_user],
    )
    delete_user_btn.click(
        fn=_delete_user_action,
        inputs=[workspace_state, config_state, study_config_state, runtime_device_dropdown, default_retrieval_state],
        outputs=[chatbot, trace_box, workspace_state, report_markdown, report_file, user_dropdown, session_dropdown],
    )
    create_session_btn.click(
        fn=_create_session_action,
        inputs=[new_session, workspace_state, config_state, study_config_state, runtime_device_dropdown, default_retrieval_state],
        outputs=[chatbot, trace_box, workspace_state, report_markdown, report_file, user_dropdown, session_dropdown, new_session],
    )
    delete_session_btn.click(
        fn=_delete_session_action,
        inputs=[workspace_state, config_state, study_config_state, runtime_device_dropdown, default_retrieval_state],
        outputs=[chatbot, trace_box, workspace_state, report_markdown, report_file, user_dropdown, session_dropdown],
    )
    submit_button.click(
        fn=_submit_chat,
        inputs=[user_input, workspace_state, model_dropdown, config_state, study_config_state, runtime_device_dropdown, default_retrieval_state],
        outputs=[chatbot, trace_box, workspace_state, report_markdown, report_file, user_dropdown, session_dropdown, user_input],
    )
    user_input.submit(
        fn=_submit_chat,
        inputs=[user_input, workspace_state, model_dropdown, config_state, study_config_state, runtime_device_dropdown, default_retrieval_state],
        outputs=[chatbot, trace_box, workspace_state, report_markdown, report_file, user_dropdown, session_dropdown, user_input],
    )
    exam_button.click(
        fn=_run_exam_action,
        inputs=[exam_type, exam_topic, exam_count, workspace_state, model_dropdown, config_state, study_config_state, runtime_device_dropdown, default_retrieval_state, exam_question_type],
        outputs=[chatbot, trace_box, workspace_state, report_markdown, report_file, user_dropdown, session_dropdown],
    )
    report_button.click(
        fn=_run_report_action,
        inputs=[report_type, workspace_state, model_dropdown, config_state, study_config_state, runtime_device_dropdown, default_retrieval_state],
        outputs=[chatbot, trace_box, workspace_state, report_markdown, report_file, user_dropdown, session_dropdown],
    )