from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass


THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=\n(?:Thought|Action|Observation|Final Answer):|\Z)", re.S)
ACTION_RE = re.compile(r"^Action:\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*$", re.M)
FINAL_RE = re.compile(r"^Final Answer:\s*(.*)", re.M | re.S)

TOOL_ARG_ORDER: dict[str, list[str]] = {
    "prepare_context": ["query"],
    "memory_search": ["query", "top_k"],
    "profile_upsert": ["raw_text", "updates"],
    "profile_view": [],
    "rag_search": ["query", "sources", "top_k"],
    "retrieve_from_kb": ["query", "top_k", "effect_level"],
    "lookup_statute": ["title"],
    "resolve_hierarchy": ["title_or_category"],
    "calculator": ["expression"],
    "generate_exam": ["topic", "question_count", "exam_type", "question_types"],
    "score_exam": ["answers_text", "exam_session_id"],
    "generate_report": ["report_type"],
    "ask_followup": ["question", "slot"],
    "ask_user": ["question", "field_name"],
}


@dataclass(slots=True)
class ParsedStep:
    kind: str
    thought: str = ""
    tool_name: str | None = None
    tool_args: dict[str, object] | None = None
    final_answer: str | None = None
    error: str | None = None


def _single_argument_name(tool_name: str) -> str:
    return TOOL_ARG_ORDER.get(tool_name, ["query"])[0]


def _extract_latest_thought(text: str, cutoff: int | None = None) -> str:
    scope = text if cutoff is None else text[:cutoff]
    thought_match = None
    for match in THOUGHT_RE.finditer(scope):
        thought_match = match
    return thought_match.group(1).strip() if thought_match else ""


def _coerce_jsonish(text: str, tool_name: str) -> dict[str, object]:
    text = text.strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return ast.literal_eval(text)
    if text.startswith("\"") or text.startswith("'"):
        return {_single_argument_name(tool_name): ast.literal_eval(text)}
    if "=" in text:
        result: dict[str, object] = {}
        for piece in text.split(","):
            name, value = piece.split("=", maxsplit=1)
            name = name.strip()
            value = value.strip()
            try:
                result[name] = ast.literal_eval(value)
            except Exception:
                result[name] = value
        return result
    return {_single_argument_name(tool_name): text}


def _parse_tool_args(tool_name: str, args_text: str) -> dict[str, object]:
    text = args_text.strip()
    if not text:
        return {}
    if text.startswith("{"):
        return _coerce_jsonish(text, tool_name)

    try:
        parsed = ast.parse(f"tool({text})", mode="eval").body
        if isinstance(parsed, ast.Call):
            result: dict[str, object] = {}
            positional_names = TOOL_ARG_ORDER.get(tool_name, [])
            for index, arg in enumerate(parsed.args):
                field_name = positional_names[index] if index < len(positional_names) else f"arg_{index}"
                result[field_name] = ast.literal_eval(arg)
            for keyword in parsed.keywords:
                if keyword.arg is None:
                    raise ValueError("不支持的可变参数工具调用")
                result[keyword.arg] = ast.literal_eval(keyword.value)
            if result:
                return result
    except Exception:
        pass

    return _coerce_jsonish(text, tool_name)


def parse_react_output(raw_text: str) -> ParsedStep:
    cleaned = raw_text.strip().strip("`")
    final_match = FINAL_RE.search(cleaned)
    action_match = None
    for match in ACTION_RE.finditer(cleaned):
        if final_match is None or match.start() < final_match.start():
            action_match = match
            break

    if action_match is not None:
        thought = _extract_latest_thought(cleaned, action_match.start())
        tool_name = action_match.group(1)
        args_text = action_match.group(2)
        try:
            tool_args = _parse_tool_args(tool_name, args_text)
        except Exception as exc:
            return ParsedStep(kind="invalid", thought=thought, error=f"工具参数无法解析: {exc}")
        return ParsedStep(kind="tool", thought=thought, tool_name=tool_name, tool_args=tool_args)

    if final_match is not None:
        thought = _extract_latest_thought(cleaned, final_match.start())
        return ParsedStep(kind="final", thought=thought, final_answer=final_match.group(1).strip())

    thought = _extract_latest_thought(cleaned)
    if "Action:" not in cleaned:
        return ParsedStep(kind="invalid", thought=thought, error="模型输出中缺少 Action 或 Final Answer。")
    return ParsedStep(kind="invalid", thought=thought, error="Action 行无法解析。")
