from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import re
from typing import Any, Protocol, Sequence

from context_engine.schemas import ConversationTurn, SessionState, UserProfile
from legal_agent.models.qwen_local import LocalQwenChatModel
from legal_agent.utils.text import simple_tokenize, truncate_text


FOLLOWUP_MARKERS = ("请补充", "需要确认", "请说明", "是否", "要不要", "还需要")
STOPWORDS = {
    "用户",
    "助手",
    "这个",
    "那个",
    "一下",
    "已经",
    "需要",
    "问题",
    "可以",
    "继续",
    "当前",
    "本次",
    "本轮",
    "还有",
    "然后",
}


@dataclass(slots=True)
class MemoryDraft:
    layer: str
    category: str
    text: str
    importance: float = 0.6
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.75
    decay_enabled: bool = True


@dataclass(slots=True)
class TurnAnalysis:
    summary: str
    reasoning_digest: str = ""
    episodic_memories: list[MemoryDraft] = field(default_factory=list)
    semantic_memories: list[MemoryDraft] = field(default_factory=list)
    long_term_memories: list[MemoryDraft] = field(default_factory=list)
    system_memories: list[MemoryDraft] = field(default_factory=list)
    open_loops: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    profile_updates: dict[str, object] = field(default_factory=dict)
    importance: float = 0.65


@dataclass(slots=True)
class CompressionDraft:
    summary: str
    salient_points: list[str] = field(default_factory=list)
    open_loops: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    importance: float = 0.8


class MemoryReasoner(Protocol):
    def analyze_turn(self, turn: ConversationTurn, *, user_profile: UserProfile, session_state: SessionState) -> TurnAnalysis:
        ...

    def compress_history(
        self,
        *,
        turns: Sequence[ConversationTurn],
        prior_summaries: Sequence[str] | None = None,
    ) -> CompressionDraft:
        ...


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _split_cn_list(text: str) -> list[str]:
    return _dedupe_strings(re.split(r"[、，,；;\s]+", str(text or "")))


def _normalize_profile_updates(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}

    normalized: dict[str, object] = {}
    name = str(payload.get("name") or "").strip()
    if name:
        normalized["name"] = name

    for field_name in ("study_goals", "weak_points", "strong_points", "notes"):
        values = payload.get(field_name)
        if isinstance(values, str):
            compact_values = _split_cn_list(values)
        elif isinstance(values, list):
            compact_values = _dedupe_strings(str(item) for item in values if str(item or "").strip())
        else:
            compact_values = []
        if compact_values:
            normalized[field_name] = compact_values

    preferences = payload.get("preferences")
    if isinstance(preferences, dict):
        compact_preferences = {
            str(key): value
            for key, value in preferences.items()
            if str(key or "").strip() and value not in (None, "", [], {})
        }
        if compact_preferences:
            normalized["preferences"] = compact_preferences

    target_score = payload.get("target_score")
    if target_score not in (None, ""):
        try:
            normalized["target_score"] = int(target_score)
        except Exception:
            pass

    daily_hours = payload.get("daily_hours")
    if daily_hours not in (None, ""):
        try:
            normalized.setdefault("preferences", {})["daily_hours"] = float(daily_hours)
        except Exception:
            pass
    return normalized


def _merge_profile_updates(*payloads: dict[str, object]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for payload in payloads:
        normalized = _normalize_profile_updates(payload)
        for key, value in normalized.items():
            if key in {"study_goals", "weak_points", "strong_points", "notes"}:
                merged[key] = _dedupe_strings([*(merged.get(key) or []), *value])
            elif key == "preferences" and isinstance(value, dict):
                current = dict(merged.get("preferences") or {})
                current.update(value)
                merged["preferences"] = current
            else:
                merged[key] = value
    return merged


def _profile_update_delta(payload: object, user_profile: UserProfile) -> dict[str, object]:
    normalized = _normalize_profile_updates(payload)
    if not normalized:
        return {}

    delta: dict[str, object] = {}
    name = str(normalized.get("name") or "").strip()
    if name and name != str(user_profile.name or "").strip():
        delta["name"] = name

    list_fields = {
        "study_goals": list(user_profile.study_goals or []),
        "weak_points": list(user_profile.weak_points or []),
        "strong_points": list(user_profile.strong_points or []),
        "notes": list(user_profile.notes or []),
    }
    for field_name, existing_values in list_fields.items():
        additions = [item for item in normalized.get(field_name) or [] if item not in existing_values]
        if additions:
            delta[field_name] = additions

    preference_updates: dict[str, object] = {}
    for key, value in dict(normalized.get("preferences") or {}).items():
        if user_profile.preferences.get(key) != value:
            preference_updates[str(key)] = value
    if preference_updates:
        delta["preferences"] = preference_updates

    if "target_score" in normalized and user_profile.attributes.get("target_score") != normalized["target_score"]:
        delta["target_score"] = normalized["target_score"]

    return delta


def _top_keywords(texts: Sequence[str], *, limit: int = 6) -> list[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        for token in simple_tokenize(text):
            normalized = str(token or "").strip().lower()
            if not normalized or normalized in STOPWORDS:
                continue
            if len(normalized) == 1 and not ("\u4e00" <= normalized <= "\u9fff"):
                continue
            counter[normalized] += 1
    return [token for token, _count in counter.most_common(limit)]


def _profile_updates_from_tool_trace(tool_trace: Sequence[dict[str, Any]]) -> dict[str, object]:
    merged: dict[str, object] = {}
    for step in tool_trace:
        if str(step.get("tool_name") or "") != "profile_upsert":
            continue
        merged = _merge_profile_updates(
            merged,
            dict(step.get("arguments") or {}).get("updates") or {},
            dict(step.get("result") or {}).get("updates") or {},
        )
    return merged


def _long_term_memories_from_profile_updates(profile_updates: dict[str, object]) -> list[MemoryDraft]:
    normalized = _normalize_profile_updates(profile_updates)
    long_term_memories: list[MemoryDraft] = []
    name = str(normalized.get("name") or "").strip()
    if name:
        long_term_memories.append(
            MemoryDraft(
                layer="long_term",
                category="user_name",
                text=f"用户姓名：{name}",
                importance=0.98,
                tags=[name],
                keywords=[name],
                decay_enabled=False,
            )
        )
    for item in normalized.get("study_goals") or []:
        long_term_memories.append(
            MemoryDraft(
                layer="long_term",
                category="user_goal",
                text=f"用户当前备考目标：{item}",
                importance=0.96,
                tags=[str(item)],
                keywords=[str(item)],
                decay_enabled=False,
            )
        )
    for item in normalized.get("weak_points") or []:
        long_term_memories.append(
            MemoryDraft(
                layer="long_term",
                category="weak_point",
                text=f"用户薄弱点：{item}",
                importance=0.94,
                tags=[str(item)],
                keywords=[str(item)],
                decay_enabled=False,
            )
        )
    for item in normalized.get("strong_points") or []:
        long_term_memories.append(
            MemoryDraft(
                layer="long_term",
                category="strong_point",
                text=f"用户强项：{item}",
                importance=0.88,
                tags=[str(item)],
                keywords=[str(item)],
                decay_enabled=False,
            )
        )
    preferences = dict(normalized.get("preferences") or {})
    for key, preference in preferences.items():
        long_term_memories.append(
            MemoryDraft(
                layer="long_term",
                category="user_preference",
                text=f"用户偏好 {key}：{preference}",
                importance=0.9,
                tags=[str(key), str(preference)],
                keywords=[str(key), str(preference)],
                decay_enabled=False,
            )
        )
    return long_term_memories


def _infer_open_loops(answer: str) -> list[str]:
    normalized = str(answer or "").strip()
    if not normalized:
        return []
    if normalized.endswith(("?", "？")):
        return [truncate_text(normalized, 120)]
    for marker in FOLLOWUP_MARKERS:
        if marker in normalized:
            return [truncate_text(normalized, 120)]
    return []


def _reasoning_digest(trace: str, tool_names: Sequence[str], answer: str) -> str:
    normalized = str(trace or "").strip()
    if normalized:
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        filtered = [line for line in lines if not line.startswith("Final Answer:")]
        if filtered:
            return truncate_text("；".join(filtered[-3:]), 220)
    if tool_names:
        return f"本轮主要依赖工具 {'、'.join(_dedupe_strings(tool_names)[:4])} 完成任务。"
    return truncate_text(answer, 160)


class HeuristicMemoryReasoner:
    def analyze_turn(self, turn: ConversationTurn, *, user_profile: UserProfile, session_state: SessionState) -> TurnAnalysis:
        tool_names = [str(step.get("tool_name") or "") for step in turn.tool_trace if str(step.get("tool_name") or "").strip()]
        profile_updates = _profile_updates_from_tool_trace(turn.tool_trace)
        tags = _top_keywords([turn.user_message, turn.assistant_message, turn.reasoning_trace], limit=8)
        summary_parts = [f"用户请求：{truncate_text(turn.user_message, 96)}"]
        if tool_names:
            summary_parts.append(f"执行工具：{'、'.join(_dedupe_strings(tool_names)[:4])}")
        summary_parts.append(f"结果：{truncate_text(turn.assistant_message, 140)}")
        summary = "；".join(summary_parts)
        reasoning_digest = _reasoning_digest(turn.reasoning_trace, tool_names, turn.assistant_message)

        episodic_memories = [
            MemoryDraft(
                layer="episodic",
                category="turn_episode",
                text=summary,
                importance=0.75 if tool_names else 0.64,
                tags=tags[:6],
                keywords=tags[:6],
                payload={"tool_names": tool_names},
            )
        ]
        semantic_memories: list[MemoryDraft] = []
        if reasoning_digest:
            semantic_memories.append(
                MemoryDraft(
                    layer="semantic",
                    category="reasoning_digest",
                    text=reasoning_digest,
                    importance=0.72 if tool_names else 0.6,
                    tags=tags[:6],
                    keywords=tags[:6],
                    payload={"tool_names": tool_names},
                )
            )
        if tool_names:
            semantic_memories.append(
                MemoryDraft(
                    layer="semantic",
                    category="tool_experience",
                    text=f"该轮问题通过工具 {'、'.join(_dedupe_strings(tool_names)[:4])} 处理。",
                    importance=0.66,
                    tags=_dedupe_strings(tool_names)[:4],
                    keywords=tags[:4],
                    payload={"tool_names": tool_names},
                )
            )

        long_term_memories = _long_term_memories_from_profile_updates(profile_updates)

        open_loops = _infer_open_loops(turn.assistant_message)
        importance = 0.82 if tool_names else 0.68
        return TurnAnalysis(
            summary=summary,
            reasoning_digest=reasoning_digest,
            episodic_memories=episodic_memories,
            semantic_memories=semantic_memories,
            long_term_memories=long_term_memories,
            system_memories=[],
            open_loops=open_loops,
            tags=tags[:8],
            profile_updates=profile_updates,
            importance=importance,
        )

    def compress_history(
        self,
        *,
        turns: Sequence[ConversationTurn],
        prior_summaries: Sequence[str] | None = None,
    ) -> CompressionDraft:
        summaries = [truncate_text(summary, 140) for summary in (prior_summaries or []) if str(summary or "").strip()]
        snippets: list[str] = []
        keywords = _top_keywords(
            [
                *(turn.user_message for turn in turns),
                *(turn.assistant_message for turn in turns),
                *summaries,
            ],
            limit=8,
        )
        for turn in list(turns)[:2] + list(turns)[-2:]:
            snippets.append(
                f"用户：{truncate_text(turn.user_message, 56)}；助手：{truncate_text(turn.assistant_message, 72)}"
            )
        open_loops = _dedupe_strings(
            loop
            for turn in turns
            for loop in _infer_open_loops(turn.assistant_message)
        )
        salient_points = []
        if keywords:
            salient_points.append(f"高频主题：{'、'.join(keywords[:6])}")
        if summaries:
            salient_points.append(f"继承摘要：{' | '.join(summaries[:2])}")
        salient_points.extend(snippets[:3])
        summary = "历史压缩摘要："
        if keywords:
            summary += f"围绕 {'、'.join(keywords[:5])} 持续展开。"
        if snippets:
            summary += " " + " | ".join(snippets[:3])
        return CompressionDraft(
            summary=truncate_text(summary, 360),
            salient_points=salient_points[:5],
            open_loops=open_loops[:4],
            tags=keywords[:6],
            importance=0.84,
        )


def _load_json_object(text: str) -> dict[str, Any]:
    normalized = str(text or "").strip()
    if not normalized:
        return {}
    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", normalized, flags=re.DOTALL)
        if match is None:
            raise
        return json.loads(match.group(0))


class QwenMemoryReasoner:
    def __init__(self, model: LocalQwenChatModel) -> None:
        self.model = model
        self.fallback = HeuristicMemoryReasoner()

    def _memory_drafts(self, rows: object, default_layer: str) -> list[MemoryDraft]:
        drafts: list[MemoryDraft] = []
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, str):
                drafts.append(MemoryDraft(layer=default_layer, category="auto", text=row))
                continue
            if not isinstance(row, dict):
                continue
            drafts.append(
                MemoryDraft(
                    layer=str(row.get("layer") or default_layer),
                    category=str(row.get("category") or "auto"),
                    text=str(row.get("text") or "").strip(),
                    importance=float(row.get("importance", 0.65)),
                    tags=_dedupe_strings(row.get("tags") or []),
                    keywords=_dedupe_strings(row.get("keywords") or []),
                    references=_dedupe_strings(row.get("references") or []),
                    payload=dict(row.get("payload") or {}),
                    confidence=float(row.get("confidence", 0.75)),
                    decay_enabled=bool(row.get("decay_enabled", True)),
                )
            )
        return [draft for draft in drafts if draft.text]

    def analyze_turn(self, turn: ConversationTurn, *, user_profile: UserProfile, session_state: SessionState) -> TurnAnalysis:
        fallback = self.fallback.analyze_turn(turn, user_profile=user_profile, session_state=session_state)
        prompt = {
            "turn": turn.to_dict(),
            "profile": user_profile.to_dict(),
            "session": {
                "session_id": session_state.session_id,
                "summary": session_state.summary,
                "turn_count": session_state.turn_count,
            },
            "task": {
                "summary": "概括本轮对话并提取会话记忆、长期记忆、系统记忆候选与结构化画像更新。",
                "output_schema": {
                    "summary": "str",
                    "reasoning_digest": "str",
                    "episodic_memories": [{"layer": "episodic", "category": "str", "text": "str"}],
                    "semantic_memories": [{"layer": "semantic", "category": "str", "text": "str"}],
                    "long_term_memories": [{"layer": "long_term", "category": "str", "text": "str"}],
                    "system_memories": [{"layer": "system", "category": "str", "text": "str"}],
                    "open_loops": ["str"],
                    "tags": ["str"],
                    "profile_updates": {"name": "str", "study_goals": ["str"], "weak_points": ["str"], "strong_points": ["str"], "preferences": {"k": "v"}, "notes": ["str"]},
                    "importance": "float",
                },
            },
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "你是上下文记忆整理器。输出必须是合法 JSON，不要包含解释文字。"
                    "只能保留会影响后续回答、规划或长期个性化的信息。"
                    "不要依赖固定句式，而要根据整轮对话语义判断用户是否透露了稳定目标、偏好、薄弱点、强项、约束或长期习惯。"
                    "profile_updates 只能填写当前这一轮新暴露、被更正或被明确强化的画像字段；如果本轮没有新增稳定画像信息，就返回空对象。"
                    "严禁把已有 profile 中已经存在的值原样抄回 profile_updates。"
                    "只有当用户提出的是代理级、跨用户或跨任务都应遵守的长期规则时，才写入 system_memories；"
                    "用户个人偏好和个人档案必须写入 profile_updates 或 long_term_memories，而不是 system_memories。"
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        try:
            output = self.model.generate(messages, max_new_tokens=512, temperature=0.0, enable_thinking=False)
            payload = _load_json_object(output.content or output.raw_text)
            profile_updates = _profile_update_delta(payload.get("profile_updates") or {}, user_profile)
            return TurnAnalysis(
                summary=str(payload.get("summary") or fallback.summary),
                reasoning_digest=str(payload.get("reasoning_digest") or fallback.reasoning_digest),
                episodic_memories=self._memory_drafts(payload.get("episodic_memories"), "episodic") or fallback.episodic_memories,
                semantic_memories=self._memory_drafts(payload.get("semantic_memories"), "semantic") or fallback.semantic_memories,
                long_term_memories=self._memory_drafts(payload.get("long_term_memories"), "long_term") or fallback.long_term_memories,
                system_memories=self._memory_drafts(payload.get("system_memories"), "system") or fallback.system_memories,
                open_loops=_dedupe_strings(payload.get("open_loops") or fallback.open_loops),
                tags=_dedupe_strings(payload.get("tags") or fallback.tags),
                profile_updates=profile_updates or fallback.profile_updates,
                importance=float(payload.get("importance", fallback.importance)),
            )
        except Exception:
            return fallback

    def compress_history(
        self,
        *,
        turns: Sequence[ConversationTurn],
        prior_summaries: Sequence[str] | None = None,
    ) -> CompressionDraft:
        fallback = self.fallback.compress_history(turns=turns, prior_summaries=prior_summaries)
        prompt = {
            "turns": [turn.to_dict() for turn in turns],
            "prior_summaries": list(prior_summaries or []),
            "task": {
                "summary": "压缩历史上下文，保留核心事实、决策、用户偏好、未完成事项。",
                "output_schema": {
                    "summary": "str",
                    "salient_points": ["str"],
                    "open_loops": ["str"],
                    "tags": ["str"],
                    "importance": "float",
                },
            },
        }
        messages = [
            {
                "role": "system",
                "content": "你是上下文压缩器。输出必须是合法 JSON，不要附加解释。压缩后仍应支撑后续规划与检索。",
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        try:
            output = self.model.generate(messages, max_new_tokens=384, temperature=0.0, enable_thinking=False)
            payload = _load_json_object(output.content or output.raw_text)
            return CompressionDraft(
                summary=str(payload.get("summary") or fallback.summary),
                salient_points=_dedupe_strings(payload.get("salient_points") or fallback.salient_points),
                open_loops=_dedupe_strings(payload.get("open_loops") or fallback.open_loops),
                tags=_dedupe_strings(payload.get("tags") or fallback.tags),
                importance=float(payload.get("importance", fallback.importance)),
            )
        except Exception:
            return fallback