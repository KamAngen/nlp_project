from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import re
from uuid import uuid4

from context_engine.scorer import decay_importance, memory_score
from context_engine.schemas import ContextBundle, ConversationTurn, MemoryHit, MemoryItem, SessionState, UserProfile, utcnow_iso
from context_engine.store import DiskMemoryStore
from legal_agent.utils.io import read_json
from legal_agent.utils.text import truncate_text


PROFILE_UPDATE_PATTERNS = [
    (re.compile(r"我叫(?P<value>[\u4e00-\u9fffA-Za-z0-9]{2,20})"), "name"),
    (re.compile(r"我在备考(?P<value>[^，,。；;\n]{1,40})"), "study_goal"),
    (re.compile(r"我想重点学(?P<value>[^，,。；;\n]{1,40})"), "study_goal"),
    (re.compile(r"我的薄弱(?:点|项|科目)(?:是|为)?(?P<value>[^，,。；;\n]{1,40})"), "weak_point"),
    (re.compile(r"我的强项(?:是|为)?(?P<value>[^，,。；;\n]{1,40})"), "strong_point"),
    (re.compile(r"我每天能学(?P<value>\d+(?:\.\d+)?)小时"), "daily_hours"),
    (re.compile(r"我的目标分数[是为](?P<value>\d{2,3})"), "target_score"),
]

DEFAULT_LAYER_QUOTAS = {
    "profile": 3,
    "system": 2,
    "working": 3,
    "episodic": 4,
    "semantic": 4,
}


class MemoryManager:
    def __init__(
        self,
        store: DiskMemoryStore,
        *,
        system_seed_path: str | Path | None = None,
        layer_quotas: dict[str, int] | None = None,
    ) -> None:
        self.store = store
        self.system_seed_path = Path(system_seed_path) if system_seed_path else None
        self.layer_quotas = dict(DEFAULT_LAYER_QUOTAS)
        if layer_quotas:
            self.layer_quotas.update(layer_quotas)
        self.bootstrap_system_memories()

    def bootstrap_system_memories(self) -> None:
        if self.system_seed_path is None or not self.system_seed_path.exists():
            return
        if self.store.load_system_memories():
            return
        rows = read_json(self.system_seed_path)
        memories: list[MemoryItem] = []
        for index, row in enumerate(rows, start=1):
            memories.append(
                MemoryItem(
                    id=str(row.get("id") or f"system-{index}"),
                    layer="system",
                    category=str(row.get("category") or "system_policy"),
                    text=str(row.get("text") or "").strip(),
                    importance=float(row.get("importance", 0.9)),
                    tags=[str(tag) for tag in row.get("tags", [])],
                    source="bootstrap",
                    decay_enabled=False,
                    payload=dict(row.get("payload") or {}),
                )
            )
        self.store.save_system_memories(memories)

    def get_user_profile(self, user_id: str) -> UserProfile:
        return self.store.load_profile(user_id)

    def list_users(self) -> list[str]:
        return self.store.list_users()

    def create_user(self, user_id: str, *, display_name: str | None = None) -> UserProfile:
        profile = self.store.create_user(user_id, default_name=display_name)
        self.remember(
            user_id,
            layer="profile",
            category="user_created",
            text=f"已创建用户 {user_id} 的个人空间。",
            importance=0.98,
            tags=["user_created"],
            source="ui",
            decay_enabled=False,
        )
        return profile

    def delete_user(self, user_id: str) -> bool:
        return self.store.delete_user(user_id)

    def list_sessions(self, user_id: str) -> list[dict[str, object]]:
        session_ids = self.store.list_sessions(user_id)
        sessions: list[dict[str, object]] = []
        for session_id in session_ids:
            state = self.store.load_session_state(user_id, session_id)
            sessions.append(
                {
                    "session_id": session_id,
                    "updated_at": state.updated_at,
                    "turn_count": len(state.turns),
                    "summary": state.summary,
                    "active_exam_session_id": state.active_exam_session_id,
                    "last_report_path": state.last_report_path,
                }
            )
        return sorted(
            sessions,
            key=lambda item: (str(item.get("updated_at") or ""), str(item.get("session_id") or "")),
            reverse=True,
        )

    def ensure_session(self, user_id: str, session_id: str) -> SessionState:
        return self.store.create_session(user_id, session_id)

    def get_session_state(self, user_id: str, session_id: str) -> SessionState:
        return self.store.load_session_state(user_id, session_id)

    def delete_session(self, user_id: str, session_id: str) -> bool:
        return self.store.delete_session(user_id, session_id)

    def update_profile(self, user_id: str, updates: dict[str, object], *, source: str = "tool") -> UserProfile:
        profile = self.store.load_profile(user_id)
        for key, value in updates.items():
            if value is None or value == "":
                continue
            if key == "name":
                profile.name = str(value)
                continue
            if key == "study_goals":
                for item in _coerce_list(value):
                    if item not in profile.study_goals:
                        profile.study_goals.append(item)
                continue
            if key == "weak_points":
                for item in _coerce_list(value):
                    if item not in profile.weak_points:
                        profile.weak_points.append(item)
                continue
            if key == "strong_points":
                for item in _coerce_list(value):
                    if item not in profile.strong_points:
                        profile.strong_points.append(item)
                continue
            if key == "notes":
                for item in _coerce_list(value):
                    if item not in profile.notes:
                        profile.notes.append(item)
                continue
            if key == "preferences" and isinstance(value, dict):
                profile.preferences.update(value)
                continue
            profile.attributes[key] = value

        profile.updated_at = utcnow_iso()
        self.store.save_profile(profile)
        self.remember(
            user_id,
            layer="profile",
            category="profile_update",
            text=f"用户档案已更新：{', '.join(sorted(updates))}",
            importance=0.96,
            tags=[source],
            source=source,
            decay_enabled=False,
        )
        return profile

    def extract_profile_updates(self, text: str) -> dict[str, object]:
        extracted: dict[str, object] = {}
        for pattern, field_name in PROFILE_UPDATE_PATTERNS:
            match = pattern.search(text)
            if match is None:
                continue
            value = match.group("value").strip(" ，,。；;")
            if not value:
                continue
            if field_name == "study_goal":
                extracted.setdefault("study_goals", []).extend(_split_cn_list(value))
            elif field_name == "weak_point":
                extracted.setdefault("weak_points", []).extend(_split_cn_list(value))
            elif field_name == "strong_point":
                extracted.setdefault("strong_points", []).extend(_split_cn_list(value))
            elif field_name == "daily_hours":
                extracted.setdefault("preferences", {})["daily_hours"] = float(value)
            elif field_name == "target_score":
                extracted["target_score"] = int(value)
            else:
                extracted[field_name] = value
        return extracted

    def remember(
        self,
        user_id: str,
        *,
        layer: str,
        category: str,
        text: str,
        importance: float,
        tags: list[str] | None = None,
        source: str = "agent",
        session_id: str | None = None,
        payload: dict[str, object] | None = None,
        decay_enabled: bool = True,
    ) -> MemoryItem:
        memories = self.store.load_user_memories(user_id)
        item = MemoryItem(
            id=f"{layer}-{uuid4().hex[:12]}",
            layer=layer,
            category=category,
            text=truncate_text(text, 600),
            importance=max(0.05, min(float(importance), 1.0)),
            tags=list(tags or []),
            source=source,
            user_id=user_id,
            session_id=session_id,
            payload=dict(payload or {}),
            decay_enabled=decay_enabled,
        )
        memories.append(item)
        self.store.save_user_memories(user_id, memories)
        return item

    def record_turn(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        *,
        tool_trace: list[dict[str, object]] | None = None,
    ) -> SessionState:
        session = self.store.load_session_state(user_id, session_id)
        session.turns.append(
            ConversationTurn(
                user_message=user_message,
                assistant_message=assistant_message,
                tool_trace=list(tool_trace or []),
            )
        )
        session.summary = self._summarize_session(session)
        session.updated_at = utcnow_iso()
        self.store.save_session_state(session)

        auto_updates = self.extract_profile_updates(user_message)
        if auto_updates:
            self.update_profile(user_id, auto_updates, source="auto_extract")

        memory_layer = "semantic" if tool_trace else "episodic"
        importance = 0.72 if tool_trace else 0.58
        self.remember(
            user_id,
            layer=memory_layer,
            category="conversation_turn",
            text=f"用户提问：{truncate_text(user_message, 240)}\n助手回复：{truncate_text(assistant_message, 240)}",
            importance=importance,
            tags=_split_cn_list(user_message)[:6],
            source="dialogue",
            session_id=session_id,
            payload={"tool_names": [step.get("tool_name") for step in tool_trace or []]},
        )
        return session

    def record_exam_session(self, user_id: str, session_id: str, exam_payload: dict[str, object]) -> None:
        session = self.store.load_session_state(user_id, session_id)
        exam_sessions = dict(session.metadata.get("exam_sessions") or {})
        stamped_payload = dict(exam_payload)
        stamped_payload.setdefault("created_at", utcnow_iso())
        stamped_payload.setdefault("status", "pending")
        exam_id = str(stamped_payload["exam_session_id"])
        exam_sessions[exam_id] = stamped_payload
        session.metadata["exam_sessions"] = exam_sessions
        session.active_exam_session_id = exam_id
        session.updated_at = utcnow_iso()
        self.store.save_session_state(session)

        profile = self.store.load_profile(user_id)
        question_ids = [
            str(question.get("record_id"))
            for question in stamped_payload.get("questions", [])
            if str(question.get("record_id") or "").strip()
        ]
        recent_question_ids = [
            str(record_id)
            for record_id in profile.attributes.get("recent_question_ids", [])
            if str(record_id).strip() and str(record_id) not in question_ids
        ]
        profile.attributes["recent_question_ids"] = (recent_question_ids + question_ids)[-24:]
        profile.attributes["last_exam_preferences"] = {
            "topic": str(stamped_payload.get("topic") or "综合"),
            "exam_type": str(stamped_payload.get("exam_type") or "综合练习"),
            "question_count": int(stamped_payload.get("question_count") or len(question_ids)),
        }
        profile.updated_at = utcnow_iso()
        self.store.save_profile(profile)

        self.remember(
            user_id,
            layer="working",
            category="exam_session",
            text=f"当前存在一场待作答的模拟测试：{stamped_payload.get('topic', '综合')}，共 {len(stamped_payload.get('questions', []))} 题。",
            importance=0.88,
            tags=[str(stamped_payload.get("topic") or "综合")],
            source="exam_tool",
            session_id=session_id,
            payload={"exam_session_id": exam_id},
            decay_enabled=False,
        )

    def load_active_exam(self, user_id: str, session_id: str) -> dict[str, object] | None:
        session = self.store.load_session_state(user_id, session_id)
        if not session.active_exam_session_id:
            return None
        exam_sessions = dict(session.metadata.get("exam_sessions") or {})
        payload = exam_sessions.get(session.active_exam_session_id)
        return dict(payload) if payload else None

    def load_exam_session(
        self,
        user_id: str,
        session_id: str,
        exam_session_id: str | None = None,
    ) -> dict[str, object] | None:
        session = self.store.load_session_state(user_id, session_id)
        exam_sessions = dict(session.metadata.get("exam_sessions") or {})
        if exam_session_id:
            payload = exam_sessions.get(exam_session_id)
            if payload:
                return dict(payload)
        if session.active_exam_session_id:
            payload = exam_sessions.get(session.active_exam_session_id)
            if payload:
                return dict(payload)
        pending = [
            dict(payload)
            for payload in exam_sessions.values()
            if str(payload.get("status") or "pending") != "scored"
        ]
        if pending:
            pending.sort(
                key=lambda payload: (
                    str(payload.get("created_at") or ""),
                    str(payload.get("exam_session_id") or ""),
                ),
                reverse=True,
            )
            return pending[0]
        if exam_session_id:
            return None
        history = [dict(payload) for payload in exam_sessions.values()]
        if not history:
            return None
        history.sort(
            key=lambda payload: (
                str(payload.get("scored_at") or payload.get("created_at") or ""),
                str(payload.get("exam_session_id") or ""),
            ),
            reverse=True,
        )
        return history[0]

    def store_exam_result(self, user_id: str, session_id: str, scoring_payload: dict[str, object]) -> None:
        session = self.store.load_session_state(user_id, session_id)
        exam_sessions = dict(session.metadata.get("exam_sessions") or {})
        exam_id = str(scoring_payload["exam_session_id"])
        merged_payload = {
            **exam_sessions.get(exam_id, {}),
            **scoring_payload,
            "status": "scored",
            "scored_at": utcnow_iso(),
        }
        exam_sessions[exam_id] = merged_payload
        session.metadata["exam_sessions"] = exam_sessions
        if session.active_exam_session_id == exam_id:
            session.active_exam_session_id = None
        session.updated_at = utcnow_iso()
        self.store.save_session_state(session)

        weak_tags = [str(tag) for tag in scoring_payload.get("weak_tags", [])]
        strong_tags = [str(tag) for tag in scoring_payload.get("strong_tags", [])]
        wrong_questions = [dict(item) for item in scoring_payload.get("wrong_questions", []) if isinstance(item, dict)]
        corrected_question_ids = {
            str(record_id)
            for record_id in scoring_payload.get("corrected_question_ids", [])
            if str(record_id).strip()
        }

        profile = self.store.load_profile(user_id)
        wrong_bank = {
            str(record_id): dict(item)
            for record_id, item in dict(profile.attributes.get("wrong_question_bank") or {}).items()
            if str(record_id).strip() and isinstance(item, dict)
        }
        for question in wrong_questions:
            record_id = str(question.get("record_id") or "").strip()
            if not record_id:
                continue
            previous = dict(wrong_bank.get(record_id) or {})
            wrong_bank[record_id] = {
                **previous,
                **question,
                "record_id": record_id,
                "fail_count": int(previous.get("fail_count") or 0) + 1,
                "last_incorrect_at": utcnow_iso(),
            }
        for record_id in corrected_question_ids:
            wrong_bank.pop(record_id, None)

        profile.attributes["wrong_question_bank"] = wrong_bank
        recent_scores = [item for item in profile.attributes.get("recent_exam_scores", []) if isinstance(item, dict)]
        recent_scores.append(
            {
                "exam_session_id": exam_id,
                "topic": str(merged_payload.get("topic") or "综合"),
                "exam_type": str(merged_payload.get("exam_type") or "综合练习"),
                "score_percent": float(merged_payload.get("score_percent") or 0.0),
                "weak_tags": weak_tags[:8],
                "strong_tags": strong_tags[:8],
                "scored_at": str(merged_payload.get("scored_at") or utcnow_iso()),
            }
        )
        profile.attributes["recent_exam_scores"] = recent_scores[-10:]

        for tag in weak_tags:
            if tag not in profile.weak_points:
                profile.weak_points.append(tag)
            if tag in profile.strong_points:
                profile.strong_points.remove(tag)
        for tag in strong_tags:
            if tag in weak_tags:
                continue
            if tag not in profile.strong_points:
                profile.strong_points.append(tag)
            if float(merged_payload.get("score_percent") or 0.0) >= 60 and tag in profile.weak_points:
                profile.weak_points.remove(tag)
        profile.updated_at = utcnow_iso()
        self.store.save_profile(profile)

        if weak_tags:
            self.remember(
                user_id,
                layer="semantic",
                category="exam_feedback",
                text=f"最近一次模拟测试暴露出的薄弱点：{'、'.join(weak_tags[:6])}",
                importance=0.84,
                tags=weak_tags[:6],
                source="exam_scoring",
                session_id=session_id,
                payload={"score": scoring_payload.get("score_percent")},
            )
        if strong_tags:
            self.remember(
                user_id,
                layer="semantic",
                category="exam_strength",
                text=f"最近一次模拟测试表现较稳的知识点：{'、'.join(strong_tags[:6])}",
                importance=0.76,
                tags=strong_tags[:6],
                source="exam_scoring",
                session_id=session_id,
                payload={"score": scoring_payload.get("score_percent")},
            )
        if wrong_questions:
            self.remember(
                user_id,
                layer="semantic",
                category="wrong_question_bank",
                text=f"错题库新增 {len(wrong_questions)} 题，当前累计 {len(wrong_bank)} 题待复盘。",
                importance=0.81,
                tags=weak_tags[:4],
                source="exam_scoring",
                session_id=session_id,
                payload={"wrong_question_ids": list(wrong_bank)[:10]},
            )

    def set_last_report_path(self, user_id: str, session_id: str, report_path: str) -> None:
        session = self.store.load_session_state(user_id, session_id)
        session.last_report_path = report_path
        session.updated_at = utcnow_iso()
        self.store.save_session_state(session)

    def search(
        self,
        query: str,
        user_id: str,
        session_id: str,
        *,
        layer_quotas: dict[str, int] | None = None,
    ) -> dict[str, list[MemoryHit]]:
        profile = self.store.load_profile(user_id)
        session = self.store.load_session_state(user_id, session_id)
        user_memories = self.store.load_user_memories(user_id)
        system_memories = self.store.load_system_memories()
        quotas = dict(self.layer_quotas)
        if layer_quotas:
            quotas.update(layer_quotas)

        all_items: list[MemoryItem] = []
        all_items.extend(self._profile_memory_items(profile))
        all_items.extend(system_memories)
        all_items.extend(self._working_memory_items(session))
        all_items.extend(user_memories)

        grouped: dict[str, list[MemoryHit]] = defaultdict(list)
        now = datetime.now(timezone.utc)
        for item in all_items:
            if item.status != "active":
                continue
            adjusted_importance = decay_importance(item, now=now)
            candidate = MemoryItem.from_dict({**item.to_dict(), "importance": adjusted_importance})
            score, reasons = memory_score(query, candidate, now=now)
            if score <= 0.0 and item.layer not in {"profile", "system", "working"}:
                continue
            grouped[item.layer].append(MemoryHit(item=item, score=score, reasons=reasons))

        selected: dict[str, list[MemoryHit]] = {}
        for layer, hits in grouped.items():
            ordered = sorted(hits, key=lambda hit: hit.score, reverse=True)
            selected[layer] = ordered[: quotas.get(layer, 3)]

        self._touch_selected_hits(user_id, selected, user_memories, system_memories)
        return selected

    def assemble_context(self, query: str, user_id: str, session_id: str) -> ContextBundle:
        profile = self.store.load_profile(user_id)
        session = self.store.load_session_state(user_id, session_id)
        layer_hits = self.search(query, user_id, session_id)
        summary_blocks = {
            "profile": self._render_profile(profile),
            "session": session.summary or "当前会话还没有形成摘要。",
            "memory": self._render_hits(layer_hits),
        }
        return ContextBundle(
            user_profile=profile,
            session_state=session,
            layer_hits=layer_hits,
            summary_blocks=summary_blocks,
        )

    def build_report_snapshot(self, user_id: str, session_id: str) -> dict[str, object]:
        profile = self.store.load_profile(user_id)
        session = self.store.load_session_state(user_id, session_id)
        memories = self.store.load_user_memories(user_id)
        exam_sessions = [dict(item) for item in dict(session.metadata.get("exam_sessions") or {}).values()]
        exam_sessions.sort(
            key=lambda item: (
                str(item.get("scored_at") or item.get("created_at") or ""),
                str(item.get("exam_session_id") or ""),
            )
        )
        wrong_question_bank = {
            str(record_id): dict(item)
            for record_id, item in dict(profile.attributes.get("wrong_question_bank") or {}).items()
            if str(record_id).strip() and isinstance(item, dict)
        }
        wrong_question_preview = sorted(
            wrong_question_bank.values(),
            key=lambda item: str(item.get("last_incorrect_at") or ""),
            reverse=True,
        )[:5]
        recent_topics = [tag for item in memories[-8:] for tag in item.tags[:2]]
        return {
            "profile": profile.to_dict(),
            "session_summary": session.summary,
            "recent_turn_count": len(session.turns),
            "exam_history": exam_sessions,
            "recent_topics": list(dict.fromkeys(recent_topics))[:8],
            "wrong_question_bank_count": len(wrong_question_bank),
            "wrong_question_bank_preview": wrong_question_preview,
            "memory_count": len(memories),
            "last_report_path": session.last_report_path,
        }

    def decay_memories(self, user_id: str) -> None:
        memories = self.store.load_user_memories(user_id)
        changed = False
        now = datetime.now(timezone.utc)
        for item in memories:
            new_importance = decay_importance(item, now=now)
            if abs(new_importance - item.importance) >= 1e-4:
                item.importance = round(new_importance, 4)
                item.updated_at = utcnow_iso()
                changed = True
        if changed:
            self.store.save_user_memories(user_id, memories)

    def _touch_selected_hits(
        self,
        user_id: str,
        selected: dict[str, list[MemoryHit]],
        user_memories: list[MemoryItem],
        system_memories: list[MemoryItem],
    ) -> None:
        user_by_id = {item.id: item for item in user_memories}
        system_by_id = {item.id: item for item in system_memories}
        touched_user = False
        touched_system = False
        now = utcnow_iso()

        for hits in selected.values():
            for hit in hits:
                item_id = hit.item.id
                if item_id in user_by_id:
                    user_by_id[item_id].hit_count += 1
                    user_by_id[item_id].last_accessed_at = now
                    touched_user = True
                elif item_id in system_by_id:
                    system_by_id[item_id].hit_count += 1
                    system_by_id[item_id].last_accessed_at = now
                    touched_system = True

        if touched_user:
            self.store.save_user_memories(user_id, list(user_by_id.values()))
        if touched_system:
            self.store.save_system_memories(list(system_by_id.values()))

    def _profile_memory_items(self, profile: UserProfile) -> list[MemoryItem]:
        items: list[MemoryItem] = []
        if profile.name:
            items.append(self._synthetic_memory("profile", "name", f"用户姓名：{profile.name}", importance=0.95))
        for goal in profile.study_goals:
            items.append(self._synthetic_memory("profile", "study_goal", f"用户当前备考目标：{goal}", importance=0.96, tags=[goal]))
        for weak_point in profile.weak_points:
            items.append(self._synthetic_memory("profile", "weak_point", f"用户薄弱点：{weak_point}", importance=0.94, tags=[weak_point]))
        for strong_point in profile.strong_points:
            items.append(self._synthetic_memory("profile", "strong_point", f"用户强项：{strong_point}", importance=0.82, tags=[strong_point]))
        for key, value in profile.preferences.items():
            items.append(self._synthetic_memory("profile", "preference", f"用户偏好 {key}：{value}", importance=0.86, tags=[key]))
        for key, value in profile.attributes.items():
            items.append(self._synthetic_memory("profile", "attribute", f"用户属性 {key}：{value}", importance=0.84, tags=[key]))
        return items

    def _working_memory_items(self, session: SessionState) -> list[MemoryItem]:
        items: list[MemoryItem] = []
        if session.summary:
            items.append(self._synthetic_memory("working", "session_summary", session.summary, importance=0.9))
        for turn in session.turns[-3:]:
            text = f"最近对话：用户说“{truncate_text(turn.user_message, 120)}”；助手答“{truncate_text(turn.assistant_message, 120)}”。"
            items.append(self._synthetic_memory("working", "recent_turn", text, importance=0.82))
        if session.active_exam_session_id:
            items.append(
                self._synthetic_memory(
                    "working",
                    "active_exam",
                    f"当前会话存在待评分的模拟测试，考试编号：{session.active_exam_session_id}",
                    importance=0.92,
                )
            )
        return items

    def _synthetic_memory(
        self,
        layer: str,
        category: str,
        text: str,
        *,
        importance: float,
        tags: list[str] | None = None,
    ) -> MemoryItem:
        return MemoryItem(
            id=f"synthetic::{layer}::{uuid4().hex[:8]}",
            layer=layer,
            category=category,
            text=text,
            importance=importance,
            tags=list(tags or []),
            source="synthetic",
            decay_enabled=False,
        )

    def _summarize_session(self, session: SessionState) -> str:
        if not session.turns:
            return ""
        snippets = []
        for turn in session.turns[-4:]:
            snippets.append(
                f"用户：{truncate_text(turn.user_message, 60)}；助手：{truncate_text(turn.assistant_message, 90)}"
            )
        return "最近会话摘要：" + " | ".join(snippets)

    def _render_profile(self, profile: UserProfile) -> str:
        parts = []
        if profile.name:
            parts.append(f"姓名：{profile.name}")
        if profile.study_goals:
            parts.append(f"目标：{'、'.join(profile.study_goals[:4])}")
        if profile.weak_points:
            parts.append(f"薄弱点：{'、'.join(profile.weak_points[:5])}")
        if profile.strong_points:
            parts.append(f"强项：{'、'.join(profile.strong_points[:5])}")
        if profile.preferences:
            parts.append(
                "偏好：" + "；".join(f"{key}={value}" for key, value in list(profile.preferences.items())[:4])
            )
        return "；".join(parts) if parts else "用户档案尚未形成稳定画像。"

    def _render_hits(self, layer_hits: dict[str, list[MemoryHit]]) -> str:
        sections = []
        for layer in ("profile", "system", "working", "episodic", "semantic"):
            hits = layer_hits.get(layer, [])
            if not hits:
                continue
            preview = "；".join(truncate_text(hit.item.text, 80) for hit in hits[:3])
            sections.append(f"{layer}: {preview}")
        return "\n".join(sections) if sections else "没有命中显著历史记忆。"


def _coerce_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _split_cn_list(text: str) -> list[str]:
    items = [item.strip() for item in re.split(r"[、，,；;\s]+", text) if item.strip()]
    return list(dict.fromkeys(items))