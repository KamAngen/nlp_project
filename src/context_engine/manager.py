from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from uuid import uuid4

from context_engine.reasoner import HeuristicMemoryReasoner, MemoryDraft, MemoryReasoner
from context_engine.scorer import decay_importance, memory_score
from context_engine.schemas import ContextBundle, ConversationTurn, MemoryEdge, MemoryHit, MemoryItem, SessionState, UserProfile, utcnow_iso
from context_engine.store import DiskMemoryStore
from context_engine.vectorizer import HashingVectorizer, MemoryVectorizer, TransformerVectorizer
from legal_agent.utils.io import read_json
from legal_agent.utils.text import simple_tokenize, truncate_text

DEFAULT_LAYER_QUOTAS = {
    "profile": 3,
    "system": 2,
    "working": 4,
    "long_term": 4,
    "summary": 4,
    "episodic": 4,
    "semantic": 4,
}
DEFAULT_GUARANTEED_LIMITS = {
    "profile": 2,
    "system": 2,
    "working": 2,
    "long_term": 3,
}
DEFAULT_DYNAMIC_RELATED_LIMIT = 8
DEFAULT_TOTAL_RELATED_LIMIT = 14
GRAPH_EDGE_MIN_WEIGHT = 0.18
MAX_GRAPH_NEIGHBORS = 8


class MemoryManager:
    def __init__(
        self,
        store: DiskMemoryStore,
        *,
        system_seed_path: str | Path | None = None,
        layer_quotas: dict[str, int] | None = None,
        reasoner: MemoryReasoner | None = None,
        vectorizer: MemoryVectorizer | None = None,
        recent_turn_window: int = 8,
        compression_after_turns: int = 10,
        compression_chunk_size: int = 8,
        retain_recent_turns: int = 6,
    ) -> None:
        self.store = store
        self.system_seed_path = Path(system_seed_path) if system_seed_path else None
        self.layer_quotas = dict(DEFAULT_LAYER_QUOTAS)
        if layer_quotas:
            self.layer_quotas.update(layer_quotas)
        self.reasoner = reasoner or HeuristicMemoryReasoner()
        self.vectorizer = vectorizer or HashingVectorizer()
        self.recent_turn_window = max(4, int(recent_turn_window))
        self.compression_after_turns = max(4, int(compression_after_turns))
        self.compression_chunk_size = max(4, int(compression_chunk_size))
        self.retain_recent_turns = max(2, int(retain_recent_turns))
        self.bootstrap_system_memories()

    def bind_reasoner(self, reasoner: MemoryReasoner) -> None:
        self.reasoner = reasoner

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
                    keywords=[str(tag) for tag in row.get("tags", [])],
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
            layer="long_term",
            category="user_created",
            text=f"已创建用户 {user_id} 的个人空间。",
            importance=0.98,
            tags=["user_created", user_id],
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
            if not state.turn_count:
                state.turn_count = len(self.store.load_session_turns(user_id, session_id))
            sessions.append(
                {
                    "session_id": session_id,
                    "updated_at": state.updated_at,
                    "turn_count": state.turn_count,
                    "summary": state.summary,
                    "active_exam_session_id": state.active_exam_session_id,
                    "last_report_path": state.last_report_path,
                    "compression_count": state.compression_count,
                }
            )
        return sorted(
            sessions,
            key=lambda item: (str(item.get("updated_at") or ""), str(item.get("session_id") or "")),
            reverse=True,
        )

    def ensure_session(self, user_id: str, session_id: str) -> SessionState:
        state = self.store.create_session(user_id, session_id)
        state.turns = self.store.load_session_turns(user_id, session_id, limit=self.recent_turn_window)
        state.turn_count = max(state.turn_count, len(self.store.load_session_turns(user_id, session_id)))
        self.store.save_session_state(state)
        return state

    def get_session_state(self, user_id: str, session_id: str) -> SessionState:
        state = self.store.load_session_state(user_id, session_id)
        state.turns = self.store.load_session_turns(user_id, session_id, limit=self.recent_turn_window)
        state.turn_count = max(state.turn_count, len(self.store.load_session_turns(user_id, session_id)))
        return state

    def delete_session(self, user_id: str, session_id: str) -> bool:
        return self.store.delete_session(user_id, session_id)

    def update_profile(self, user_id: str, updates: dict[str, object], *, source: str = "tool") -> UserProfile:
        profile = self.store.load_profile(user_id)
        changed_fields: list[str] = []
        for key, value in updates.items():
            if value is None or value == "":
                continue
            if key == "name":
                profile.name = str(value)
                changed_fields.append(key)
                continue
            if key == "study_goals":
                for item in _coerce_list(value):
                    if item not in profile.study_goals:
                        profile.study_goals.append(item)
                        changed_fields.append(key)
                continue
            if key == "weak_points":
                for item in _coerce_list(value):
                    if item not in profile.weak_points:
                        profile.weak_points.append(item)
                        changed_fields.append(key)
                continue
            if key == "strong_points":
                for item in _coerce_list(value):
                    if item not in profile.strong_points:
                        profile.strong_points.append(item)
                        changed_fields.append(key)
                continue
            if key == "notes":
                for item in _coerce_list(value):
                    if item not in profile.notes:
                        profile.notes.append(item)
                        changed_fields.append(key)
                continue
            if key == "preferences" and isinstance(value, dict):
                before = dict(profile.preferences)
                profile.preferences.update(value)
                if profile.preferences != before:
                    changed_fields.append(key)
                continue
            if profile.attributes.get(key) != value:
                profile.attributes[key] = value
                changed_fields.append(key)

        profile.updated_at = utcnow_iso()
        self.store.save_profile(profile)
        if changed_fields:
            memory_items = self._profile_update_memory_items(user_id, profile, updates, source=source)
            self._store_memory_items(user_id, memory_items)
        return profile

    def extract_profile_updates(self, text: str) -> dict[str, object]:
        return self.extract_profile_updates_for_user(text)

    def extract_profile_updates_for_user(
        self,
        text: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, object]:
        normalized = str(text or "").strip()
        if not normalized:
            return {}

        probe_user_id = user_id or "__memory_probe_user__"
        probe_session_id = session_id or "__memory_probe_session__"
        if user_id and self.store.user_exists(user_id):
            profile = self.store.load_profile(user_id)
        else:
            profile = UserProfile(user_id=probe_user_id)
        if user_id and session_id and self.store.session_exists(user_id, session_id):
            session = self.store.load_session_state(user_id, session_id)
        else:
            session = SessionState(session_id=probe_session_id, user_id=probe_user_id)

        analysis = self.reasoner.analyze_turn(
            ConversationTurn(user_message=normalized, assistant_message=""),
            user_profile=profile,
            session_state=session,
        )
        return dict(getattr(analysis, "profile_updates", {}) or {})

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
        references: list[str] | None = None,
        confidence: float = 0.75,
    ) -> MemoryItem:
        item = self._make_memory_item(
            layer=layer,
            category=category,
            text=text,
            importance=importance,
            tags=tags,
            source=source,
            user_id=user_id,
            session_id=session_id,
            payload=payload,
            decay_enabled=decay_enabled,
            references=references,
            confidence=confidence,
        )
        stored = self._store_memory_items(user_id, [item])
        return stored[0]

    def record_turn(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        *,
        tool_trace: list[dict[str, object]] | None = None,
        reasoning_trace: str | None = None,
    ) -> SessionState:
        session = self.store.load_session_state(user_id, session_id)
        profile = self.store.load_profile(user_id)
        turn = ConversationTurn(
            user_message=user_message,
            assistant_message=assistant_message,
            tool_trace=list(tool_trace or []),
            reasoning_trace=str(reasoning_trace or ""),
            tags=self._collect_tags(user_message, assistant_message),
        )
        analysis = self.reasoner.analyze_turn(turn, user_profile=profile, session_state=session)
        turn.reasoning_summary = analysis.reasoning_digest
        self.store.append_session_turn(user_id, session_id, turn)

        if analysis.profile_updates:
            self.update_profile(user_id, dict(analysis.profile_updates), source="auto_extract")

        memory_items = self._turn_analysis_items(user_id, session_id, turn, analysis)
        stored_items = self._store_memory_items(user_id, memory_items)

        all_turns = self.store.load_session_turns(user_id, session_id)
        session.turns = all_turns[-self.recent_turn_window :]
        session.turn_count = len(all_turns)
        session.last_turn_id = turn.turn_id
        session.metadata["open_loops"] = list(analysis.open_loops[:6])
        session.metadata["last_turn_memory_ids"] = [item.id for item in stored_items[:4]]
        session.summary = self._summarize_session(user_id, session_id, session)
        session.updated_at = utcnow_iso()
        self.store.save_session_state(session)
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
            importance=0.9,
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
                layer="long_term",
                category="exam_feedback",
                text=f"最近一次模拟测试暴露出的薄弱点：{'、'.join(weak_tags[:6])}",
                importance=0.88,
                tags=weak_tags[:6],
                source="exam_scoring",
                session_id=session_id,
                payload={"score": scoring_payload.get("score_percent")},
                decay_enabled=False,
            )
        if strong_tags:
            self.remember(
                user_id,
                layer="semantic",
                category="exam_strength",
                text=f"最近一次模拟测试表现较稳的知识点：{'、'.join(strong_tags[:6])}",
                importance=0.78,
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
                importance=0.82,
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

    def maintain_context(self, user_id: str, session_id: str) -> dict[str, object]:
        self.decay_memories(user_id)
        compression = self._compress_session_if_needed(user_id, session_id)
        return {"decayed": True, **compression}

    def search(
        self,
        query: str,
        user_id: str,
        session_id: str,
        *,
        layer_quotas: dict[str, int] | None = None,
    ) -> dict[str, list[MemoryHit]]:
        profile = self.store.load_profile(user_id)
        session = self.get_session_state(user_id, session_id)
        user_memories = self.store.load_user_memories(user_id)
        memory_edges = self.store.load_memory_edges(user_id)
        system_memories = self.store.load_system_memories()
        quotas = dict(self.layer_quotas)
        if layer_quotas:
            quotas.update(layer_quotas)

        all_items: list[MemoryItem] = []
        all_items.extend(self._profile_memory_items(profile))
        all_items.extend(system_memories)
        all_items.extend(self._working_memory_items(user_id, session))
        all_items.extend(user_memories)

        now = datetime.now(timezone.utc)
        base_scores: dict[str, float] = {}
        reasons_by_id: dict[str, list[str]] = {}
        breakdown_by_id: dict[str, dict[str, float]] = {}
        item_by_id: dict[str, MemoryItem] = {item.id: item for item in all_items if item.status == "active"}

        for item in item_by_id.values():
            score, reasons, breakdown = memory_score(query, item, vectorizer=self.vectorizer, now=now)
            base_scores[item.id] = score
            reasons_by_id[item.id] = reasons
            breakdown_by_id[item.id] = breakdown

        graph_bonus = self._graph_relation_bonus(item_by_id, base_scores, memory_edges)

        grouped: dict[str, list[MemoryHit]] = defaultdict(list)
        for item_id, item in item_by_id.items():
            breakdown = dict(breakdown_by_id.get(item_id) or {})
            lexical = breakdown.get("lexical", 0.0)
            vector = breakdown.get("vector", 0.0)
            relation = graph_bonus.get(item_id, 0.0)
            final_score = base_scores.get(item_id, 0.0) + relation
            always_include = item.layer in {"profile", "system", "working"}
            if not always_include and lexical < 0.01 and vector < 0.08 and relation < 0.05:
                continue
            if not always_include and final_score < 0.18:
                continue
            breakdown["graph"] = round(relation, 4)
            reasons = list(reasons_by_id.get(item_id) or [])
            if relation >= 0.05:
                reasons.append("命中记忆图关联")
            grouped[item.layer].append(
                MemoryHit(item=item, score=round(final_score, 4), reasons=reasons, breakdown=breakdown)
            )

        selected: dict[str, list[MemoryHit]] = {}
        for layer, hits in grouped.items():
            ordered = sorted(
                hits,
                key=lambda hit: (hit.score, hit.item.importance, hit.item.created_at),
                reverse=True,
            )
            selected[layer] = ordered[: quotas.get(layer, 3)]

        self._touch_selected_hits(user_id, selected, user_memories, system_memories)
        return selected

    def prepare_turn_context(
        self,
        query: str,
        user_id: str,
        session_id: str,
        *,
        layer_quotas: dict[str, int] | None = None,
    ) -> ContextBundle:
        self.ensure_session(user_id, session_id)
        maintenance = self.maintain_context(user_id, session_id)
        profile = self.store.load_profile(user_id)
        session = self.get_session_state(user_id, session_id)
        layer_hits = self.search(query, user_id, session_id, layer_quotas=layer_quotas)
        profile_hits = self._anchored_layer_hits(layer_hits, layer="profile", limit=DEFAULT_GUARANTEED_LIMITS["profile"])
        system_hits = self._anchored_layer_hits(layer_hits, layer="system", limit=DEFAULT_GUARANTEED_LIMITS["system"])
        working_hits = self._anchored_layer_hits(layer_hits, layer="working", limit=DEFAULT_GUARANTEED_LIMITS["working"])
        long_term_hits = self._anchored_long_term_hits(user_id, layer_hits)
        guaranteed_hits = self._merge_hit_groups(
            profile_hits,
            system_hits,
            long_term_hits[: DEFAULT_GUARANTEED_LIMITS["long_term"]],
            working_hits,
            limit=DEFAULT_TOTAL_RELATED_LIMIT,
        )
        session_hits = self._flatten_hits(layer_hits, layers=("working", "summary", "episodic", "semantic"))[:10]
        dynamic_hits = self._dynamic_related_hits(layer_hits, exclude_ids={hit.item.id for hit in guaranteed_hits}, limit=DEFAULT_DYNAMIC_RELATED_LIMIT)
        related_hits = self._merge_hit_groups(guaranteed_hits, dynamic_hits, limit=DEFAULT_TOTAL_RELATED_LIMIT)
        retrieval_meta = self._build_retrieval_meta(user_id, layer_hits, guaranteed_hits, related_hits)
        summary_blocks = {
            "profile": self._render_profile(profile),
            "system": self._render_hit_group(system_hits, empty_text="暂无激活的系统策略记忆。"),
            "long_term": self._render_hit_group(long_term_hits, empty_text="暂无稳定长期记忆。"),
            "session": self._render_session_block(session, session_hits),
            "memory": self._render_hits(layer_hits),
            "relevant": self._render_hit_group(related_hits, empty_text="没有命中显著相关记忆。"),
            "maintenance": self._render_maintenance(maintenance),
        }
        planning_context = self._render_planning_context(query, summary_blocks)
        return ContextBundle(
            user_profile=profile,
            session_state=session,
            layer_hits=layer_hits,
            summary_blocks=summary_blocks,
            maintenance=maintenance,
            profile_hits=profile_hits,
            system_hits=system_hits,
            working_hits=working_hits,
            long_term_hits=long_term_hits,
            session_hits=session_hits,
            guaranteed_hits=guaranteed_hits,
            related_hits=related_hits,
            planning_context=planning_context,
            retrieval_meta=retrieval_meta,
        )

    def prepare_turn_context_payload(
        self,
        query: str,
        user_id: str,
        session_id: str,
        *,
        layer_quotas: dict[str, int] | None = None,
    ) -> dict[str, object]:
        bundle = self.prepare_turn_context(query, user_id, session_id, layer_quotas=layer_quotas)
        return {
            "user_profile": bundle.user_profile.to_dict(),
            "session_state": bundle.session_state.to_dict(),
            "summary_blocks": dict(bundle.summary_blocks),
            "maintenance": dict(bundle.maintenance),
            "layer_hits": {layer: [hit.to_dict() for hit in hits] for layer, hits in bundle.layer_hits.items()},
            "profile_hits": [hit.to_dict() for hit in bundle.profile_hits],
            "system_hits": [hit.to_dict() for hit in bundle.system_hits],
            "working_hits": [hit.to_dict() for hit in bundle.working_hits],
            "long_term_hits": [hit.to_dict() for hit in bundle.long_term_hits],
            "session_hits": [hit.to_dict() for hit in bundle.session_hits],
            "guaranteed_hits": [hit.to_dict() for hit in bundle.guaranteed_hits],
            "related_hits": [hit.to_dict() for hit in bundle.related_hits],
            "planning_context": bundle.planning_context,
            "retrieval_meta": dict(bundle.retrieval_meta),
        }

    def assemble_context(self, query: str, user_id: str, session_id: str) -> ContextBundle:
        return self.prepare_turn_context(query, user_id, session_id)

    def build_report_snapshot(self, user_id: str, session_id: str) -> dict[str, object]:
        profile = self.store.load_profile(user_id)
        session = self.get_session_state(user_id, session_id)
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
        recent_topics = [tag for item in memories[-12:] for tag in item.tags[:2]]
        return {
            "profile": profile.to_dict(),
            "session_summary": session.summary,
            "recent_turn_count": session.turn_count,
            "exam_history": exam_sessions,
            "recent_topics": list(dict.fromkeys(recent_topics))[:8],
            "wrong_question_bank_count": len(wrong_question_bank),
            "wrong_question_bank_preview": wrong_question_preview,
            "memory_count": len(memories),
            "long_term_memory_count": sum(1 for item in memories if item.layer == "long_term"),
            "compression_count": session.compression_count,
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

    def _profile_update_memory_items(
        self,
        user_id: str,
        profile: UserProfile,
        updates: dict[str, object],
        *,
        source: str,
    ) -> list[MemoryItem]:
        items: list[MemoryItem] = [
            self._make_memory_item(
                layer="semantic",
                category="profile_update",
                text=f"用户档案已更新：{', '.join(sorted(updates))}",
                importance=0.92,
                tags=[source, *sorted(updates)],
                source=source,
                user_id=user_id,
                decay_enabled=False,
            )
        ]
        if profile.name:
            items.append(
                self._make_memory_item(
                    layer="long_term",
                    category="user_name",
                    text=f"用户姓名：{profile.name}",
                    importance=0.98,
                    tags=[profile.name],
                    source=source,
                    user_id=user_id,
                    decay_enabled=False,
                )
            )
        for goal in profile.study_goals:
            items.append(
                self._make_memory_item(
                    layer="long_term",
                    category="user_goal",
                    text=f"用户当前备考目标：{goal}",
                    importance=0.96,
                    tags=[goal],
                    source=source,
                    user_id=user_id,
                    decay_enabled=False,
                )
            )
        for weak_point in profile.weak_points:
            items.append(
                self._make_memory_item(
                    layer="long_term",
                    category="weak_point",
                    text=f"用户薄弱点：{weak_point}",
                    importance=0.94,
                    tags=[weak_point],
                    source=source,
                    user_id=user_id,
                    decay_enabled=False,
                )
            )
        for strong_point in profile.strong_points:
            items.append(
                self._make_memory_item(
                    layer="long_term",
                    category="strong_point",
                    text=f"用户强项：{strong_point}",
                    importance=0.88,
                    tags=[strong_point],
                    source=source,
                    user_id=user_id,
                    decay_enabled=False,
                )
            )
        for key, value in profile.preferences.items():
            items.append(
                self._make_memory_item(
                    layer="long_term",
                    category="user_preference",
                    text=f"用户偏好 {key}：{value}",
                    importance=0.9,
                    tags=[key, str(value)],
                    source=source,
                    user_id=user_id,
                    decay_enabled=False,
                )
            )
        return items

    def _turn_analysis_items(
        self,
        user_id: str,
        session_id: str,
        turn: ConversationTurn,
        analysis: object,
    ) -> list[MemoryItem]:
        tool_names = [str(step.get("tool_name") or "") for step in turn.tool_trace if str(step.get("tool_name") or "").strip()]
        primary = self._make_memory_item(
            layer="episodic",
            category="turn_summary",
            text=getattr(analysis, "summary", "") or f"用户：{truncate_text(turn.user_message, 90)}；助手：{truncate_text(turn.assistant_message, 120)}",
            importance=float(getattr(analysis, "importance", 0.68) or 0.68),
            tags=list(getattr(analysis, "tags", []) or turn.tags)[:8],
            source="turn_analysis",
            user_id=user_id,
            session_id=session_id,
            references=[turn.turn_id],
            payload={
                "tool_names": tool_names,
                "open_loops": list(getattr(analysis, "open_loops", []) or []),
                "turn_id": turn.turn_id,
            },
        )
        items = [primary]
        for draft in (
            list(getattr(analysis, "semantic_memories", []) or [])
            + list(getattr(analysis, "long_term_memories", []) or [])
            + list(getattr(analysis, "system_memories", []) or [])
        ):
            if not isinstance(draft, MemoryDraft) or not draft.text:
                continue
            item = self._draft_to_memory_item(user_id, session_id, draft, turn_id=turn.turn_id)
            items.append(item)
        return items

    def _draft_to_memory_item(
        self,
        user_id: str,
        session_id: str,
        draft: MemoryDraft,
        *,
        turn_id: str,
    ) -> MemoryItem:
        references = list(dict.fromkeys([turn_id, *draft.references]))
        target_user_id = None if draft.layer == "system" else user_id
        target_session_id = None if draft.layer == "system" else session_id
        return self._make_memory_item(
            layer=draft.layer,
            category=draft.category,
            text=draft.text,
            importance=draft.importance,
            tags=draft.tags,
            source="reasoner",
            user_id=target_user_id,
            session_id=target_session_id,
            payload=draft.payload,
            decay_enabled=draft.decay_enabled,
            references=references,
            confidence=draft.confidence,
            keywords=draft.keywords,
        )

    def _store_memory_items(self, user_id: str, items: list[MemoryItem]) -> list[MemoryItem]:
        memories = self.store.load_user_memories(user_id)
        system_memories = self.store.load_system_memories()
        stored_items: list[MemoryItem] = []
        changed_user = False
        changed_system = False
        for item in items:
            if not item.text.strip():
                continue
            if item.layer == "system":
                merged = self._merge_memory_item(system_memories, item)
                changed_system = True
            else:
                merged = self._merge_memory_item(memories, item)
                changed_user = True
            stored_items.append(merged)
        if changed_user:
            self.store.save_user_memories(user_id, memories)
            self.store.save_memory_edges(user_id, self._rebuild_memory_graph(user_id, memories))
        if changed_system:
            self.store.save_system_memories(system_memories)
        return stored_items

    def _merge_memory_item(self, memories: list[MemoryItem], incoming: MemoryItem) -> MemoryItem:
        now = utcnow_iso()
        for existing in reversed(memories):
            if not self._memory_equivalent(existing, incoming):
                continue
            existing.text = existing.text if len(existing.text) >= len(incoming.text) else incoming.text
            existing.importance = max(existing.importance, incoming.importance)
            existing.confidence = max(existing.confidence, incoming.confidence)
            existing.tags = _dedupe_strings([*existing.tags, *incoming.tags])
            existing.keywords = _dedupe_strings([*existing.keywords, *incoming.keywords])
            existing.references = _dedupe_strings([*existing.references, *incoming.references])
            existing.payload.update(incoming.payload)
            existing.updated_at = now
            existing.decay_enabled = existing.decay_enabled and incoming.decay_enabled
            return existing
        incoming.updated_at = now
        memories.append(incoming)
        return incoming

    def _memory_equivalent(self, existing: MemoryItem, incoming: MemoryItem) -> bool:
        if existing.layer != incoming.layer or existing.category != incoming.category:
            return False
        if existing.layer in {"episodic", "working"}:
            return False
        if existing.session_id != incoming.session_id and incoming.layer in {"summary", "semantic"}:
            return False
        normalized_existing = re.sub(r"\s+", "", existing.text)
        normalized_incoming = re.sub(r"\s+", "", incoming.text)
        if normalized_existing == normalized_incoming:
            return True
        existing_tokens = set(simple_tokenize(existing.text))
        incoming_tokens = set(simple_tokenize(incoming.text))
        if not existing_tokens or not incoming_tokens:
            return False
        overlap = len(existing_tokens & incoming_tokens) / max(len(existing_tokens | incoming_tokens), 1)
        return overlap >= 0.82

    def _compress_session_if_needed(self, user_id: str, session_id: str) -> dict[str, object]:
        session = self.store.load_session_state(user_id, session_id)
        turns = self.store.load_session_turns(user_id, session_id)
        if len(turns) < self.compression_after_turns:
            return {"compressed": False}
        compressible_end = len(turns) - self.retain_recent_turns
        if compressible_end - session.compression_cursor < self.compression_chunk_size:
            return {"compressed": False}

        start_index = session.compression_cursor
        end_index = min(start_index + self.compression_chunk_size, compressible_end)
        turn_slice = turns[start_index:end_index]
        if len(turn_slice) < self.compression_chunk_size:
            return {"compressed": False}

        user_memories = self.store.load_user_memories(user_id)
        prior_summaries = [
            item.text
            for item in user_memories
            if item.layer == "summary" and item.session_id == session_id and item.status == "active"
        ][-2:]
        draft = self.reasoner.compress_history(turns=turn_slice, prior_summaries=prior_summaries)
        summary_item = self._make_memory_item(
            layer="summary",
            category="session_compression",
            text=draft.summary,
            importance=max(float(draft.importance or 0.8), 0.8),
            tags=list(draft.tags[:6]),
            source="compression",
            user_id=user_id,
            session_id=session_id,
            references=[turn.turn_id for turn in turn_slice],
            payload={
                "from_turn_index": start_index,
                "to_turn_index": end_index - 1,
                "salient_points": draft.salient_points[:6],
                "open_loops": draft.open_loops[:4],
            },
            decay_enabled=False,
        )
        related_memory_ids = [
            item.id
            for item in user_memories
            if item.session_id == session_id and any(reference in summary_item.references for reference in item.references)
        ]
        summary_item.payload["covered_memory_count"] = len(related_memory_ids)
        stored = self._store_memory_items(user_id, [summary_item])
        session.compression_cursor = end_index
        session.compression_count += 1
        session.summary_node_ids = [*session.summary_node_ids, stored[0].id][-8:]
        session.turns = turns[-self.recent_turn_window :]
        session.turn_count = len(turns)
        session.summary = self._summarize_session(user_id, session_id, session)
        session.updated_at = utcnow_iso()
        self.store.save_session_state(session)
        return {"compressed": True, "created_summary_id": stored[0].id, "covered_turns": len(turn_slice)}

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
            items.append(self._synthetic_memory("profile", "name", f"用户姓名：{profile.name}", importance=0.98, tags=[profile.name]))
        for goal in profile.study_goals:
            items.append(self._synthetic_memory("profile", "study_goal", f"用户当前备考目标：{goal}", importance=0.96, tags=[goal]))
        for weak_point in profile.weak_points:
            items.append(self._synthetic_memory("profile", "weak_point", f"用户薄弱点：{weak_point}", importance=0.94, tags=[weak_point]))
        for strong_point in profile.strong_points:
            items.append(self._synthetic_memory("profile", "strong_point", f"用户强项：{strong_point}", importance=0.84, tags=[strong_point]))
        for key, value in profile.preferences.items():
            items.append(self._synthetic_memory("profile", "preference", f"用户偏好 {key}：{value}", importance=0.9, tags=[key, str(value)]))
        for key, value in profile.attributes.items():
            items.append(self._synthetic_memory("profile", "attribute", f"用户属性 {key}：{value}", importance=0.82, tags=[key]))
        for note in profile.notes[-3:]:
            items.append(self._synthetic_memory("profile", "note", f"用户备注：{note}", importance=0.8))
        return items

    def _working_memory_items(self, user_id: str, session: SessionState) -> list[MemoryItem]:
        items: list[MemoryItem] = []
        if session.summary:
            items.append(self._synthetic_memory("working", "session_summary", session.summary, importance=0.92))
        recent_turns = self.store.load_session_turns(user_id, session.session_id, limit=3)
        for turn in recent_turns:
            snippets = [f"最近对话：用户说“{truncate_text(turn.user_message, 90)}”；助手答“{truncate_text(turn.assistant_message, 90)}”。"]
            if turn.reasoning_summary:
                snippets.append(f"思路摘要：{truncate_text(turn.reasoning_summary, 80)}")
            items.append(
                self._synthetic_memory(
                    "working",
                    "recent_turn",
                    " ".join(snippets),
                    importance=0.84,
                    tags=turn.tags[:6],
                    references=[turn.turn_id],
                )
            )
        if session.active_exam_session_id:
            items.append(
                self._synthetic_memory(
                    "working",
                    "active_exam",
                    f"当前会话存在待评分的模拟测试，考试编号：{session.active_exam_session_id}",
                    importance=0.94,
                )
            )
        for open_loop in list(session.metadata.get("open_loops") or [])[:3]:
            items.append(self._synthetic_memory("working", "open_loop", f"当前待确认事项：{open_loop}", importance=0.88))
        return items

    def _make_memory_item(
        self,
        *,
        layer: str,
        category: str,
        text: str,
        importance: float,
        tags: list[str] | None = None,
        source: str,
        user_id: str | None = None,
        session_id: str | None = None,
        payload: dict[str, object] | None = None,
        decay_enabled: bool = True,
        references: list[str] | None = None,
        confidence: float = 0.75,
        keywords: list[str] | None = None,
    ) -> MemoryItem:
        normalized_text = truncate_text(text, 600)
        normalized_tags = _dedupe_strings(tags or [])[:10]
        normalized_keywords = _dedupe_strings(keywords or simple_tokenize(normalized_text))[:12]
        return MemoryItem(
            id=f"{layer}-{uuid4().hex[:12]}",
            layer=layer,
            category=category,
            text=normalized_text,
            importance=max(0.05, min(float(importance), 1.0)),
            confidence=max(0.05, min(float(confidence), 1.0)),
            tags=normalized_tags,
            keywords=normalized_keywords,
            source=source,
            user_id=user_id,
            session_id=session_id,
            references=_dedupe_strings(references or []),
            payload=dict(payload or {}),
            decay_enabled=decay_enabled,
        )

    def _synthetic_memory(
        self,
        layer: str,
        category: str,
        text: str,
        *,
        importance: float,
        tags: list[str] | None = None,
        references: list[str] | None = None,
    ) -> MemoryItem:
        digest = hashlib.md5(f"{layer}:{category}:{text}".encode("utf-8")).hexdigest()[:12]
        return MemoryItem(
            id=f"synthetic-{digest}",
            layer=layer,
            category=category,
            text=text,
            importance=importance,
            tags=_dedupe_strings(tags or []),
            keywords=_dedupe_strings(simple_tokenize(text))[:10],
            source="synthetic",
            decay_enabled=False,
            references=_dedupe_strings(references or []),
        )

    def _summarize_session(self, user_id: str, session_id: str, session: SessionState) -> str:
        memories = self.store.load_user_memories(user_id)
        summary_nodes = [item for item in memories if item.layer == "summary" and item.session_id == session_id and item.status == "active"]
        summary_nodes.sort(key=lambda item: item.updated_at, reverse=True)
        recent_turns = self.store.load_session_turns(user_id, session_id, limit=3)
        parts: list[str] = []
        if summary_nodes:
            parts.append("历史压缩：" + " | ".join(truncate_text(item.text, 110) for item in summary_nodes[:2]))
        if recent_turns:
            parts.append(
                "最近对话：" + " | ".join(
                    f"用户：{truncate_text(turn.user_message, 48)}；助手：{truncate_text(turn.assistant_message, 72)}"
                    for turn in recent_turns
                )
            )
        open_loops = list(session.metadata.get("open_loops") or [])[:3]
        if open_loops:
            parts.append("待确认：" + "；".join(truncate_text(loop, 80) for loop in open_loops))
        return "；".join(parts) if parts else "当前会话还没有形成摘要。"

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
            parts.append("偏好：" + "；".join(f"{key}={value}" for key, value in list(profile.preferences.items())[:4]))
        return "；".join(parts) if parts else "用户档案尚未形成稳定画像。"

    def _memory_edge_id(self, source_id: str, target_id: str, relation: str) -> str:
        digest = hashlib.md5(f"{source_id}:{target_id}:{relation}".encode("utf-8")).hexdigest()[:16]
        return f"edge-{digest}"

    def _candidate_edge_weight(
        self,
        source: MemoryItem,
        target: MemoryItem,
    ) -> tuple[float, str, dict[str, object]]:
        source_refs = {reference for reference in source.references if str(reference or "").strip()}
        target_refs = {reference for reference in target.references if str(reference or "").strip()}
        source_tags = {tag for tag in source.tags if str(tag or "").strip()}
        target_tags = {tag for tag in target.tags if str(tag or "").strip()}
        source_keywords = {keyword for keyword in source.keywords if str(keyword or "").strip()}
        target_keywords = {keyword for keyword in target.keywords if str(keyword or "").strip()}

        shared_refs = source_refs & target_refs
        tag_overlap = _overlap_ratio(source_tags, target_tags)
        keyword_overlap = _overlap_ratio(source_keywords, target_keywords)

        signals: dict[str, float] = {}
        payload: dict[str, object] = {}
        if shared_refs:
            signals["shared_reference"] = min(0.5, 0.28 + len(shared_refs) * 0.08)
            payload["shared_references"] = sorted(shared_refs)[:6]
        if tag_overlap > 0.0:
            signals["tag_overlap"] = min(0.28, 0.08 + tag_overlap * 0.22)
            payload["shared_tags"] = sorted(source_tags & target_tags)[:6]
        if keyword_overlap > 0.0:
            signals["keyword_overlap"] = min(0.18, 0.05 + keyword_overlap * 0.16)
            payload["shared_keywords"] = sorted(source_keywords & target_keywords)[:8]
        if source.session_id and source.session_id == target.session_id:
            signals["same_session"] = 0.06
        if source.category == target.category:
            signals["same_category"] = 0.04
        if source.layer == target.layer and source.layer in {"long_term", "summary", "semantic"}:
            signals["same_layer"] = 0.04
        if {source.layer, target.layer} & {"profile", "system", "working"} and (shared_refs or tag_overlap > 0.0 or keyword_overlap > 0.0):
            signals["anchor_bridge"] = 0.06

        weight = min(sum(signals.values()), 0.95)
        if weight < GRAPH_EDGE_MIN_WEIGHT:
            return 0.0, "weak_link", {}
        relation = max(signals.items(), key=lambda item: item[1])[0]
        payload["signals"] = {name: round(value, 4) for name, value in signals.items()}
        return weight, relation, payload

    def _rebuild_memory_graph(self, user_id: str, memories: list[MemoryItem] | None = None) -> list[MemoryEdge]:
        active_memories = [
            item
            for item in (memories or self.store.load_user_memories(user_id))
            if item.status == "active" and item.text.strip()
        ]
        outgoing: dict[str, list[tuple[str, float, str, dict[str, object]]]] = defaultdict(list)
        for index, source in enumerate(active_memories):
            for target in active_memories[index + 1 :]:
                weight, relation, payload = self._candidate_edge_weight(source, target)
                if weight < GRAPH_EDGE_MIN_WEIGHT:
                    continue
                outgoing[source.id].append((target.id, weight, relation, payload))
                outgoing[target.id].append((source.id, weight, relation, payload))

        edges: list[MemoryEdge] = []
        for source_id, neighbors in outgoing.items():
            for target_id, weight, relation, payload in sorted(neighbors, key=lambda item: (item[1], item[0]), reverse=True)[:MAX_GRAPH_NEIGHBORS]:
                edges.append(
                    MemoryEdge(
                        id=self._memory_edge_id(source_id, target_id, relation),
                        source_id=source_id,
                        target_id=target_id,
                        relation=relation,
                        weight=round(weight, 4),
                        payload=dict(payload),
                    )
                )
        return edges

    def _memory_graph_adjacency(
        self,
        item_by_id: dict[str, MemoryItem],
        stored_edges: list[MemoryEdge],
    ) -> dict[str, dict[str, float]]:
        adjacency: dict[str, dict[str, float]] = defaultdict(dict)
        for edge in stored_edges:
            if edge.source_id not in item_by_id or edge.target_id not in item_by_id:
                continue
            adjacency[edge.source_id][edge.target_id] = max(adjacency[edge.source_id].get(edge.target_id, 0.0), float(edge.weight))

        items = list(item_by_id.values())
        for index, source in enumerate(items):
            for target in items[index + 1 :]:
                weight, _relation, _payload = self._candidate_edge_weight(source, target)
                if weight < GRAPH_EDGE_MIN_WEIGHT:
                    continue
                adjacency[source.id][target.id] = max(adjacency[source.id].get(target.id, 0.0), weight)
                adjacency[target.id][source.id] = max(adjacency[target.id].get(source.id, 0.0), weight)
        return adjacency

    def _graph_relation_bonus(
        self,
        item_by_id: dict[str, MemoryItem],
        base_scores: dict[str, float],
        stored_edges: list[MemoryEdge],
    ) -> dict[str, float]:
        adjacency = self._memory_graph_adjacency(item_by_id, stored_edges)
        bonuses: dict[str, float] = defaultdict(float)
        if not adjacency:
            return bonuses

        anchor_ids: list[str] = []
        for item_id, item in sorted(
            item_by_id.items(),
            key=lambda entry: (base_scores.get(entry[0], 0.0), entry[1].importance, entry[1].updated_at),
            reverse=True,
        ):
            if base_scores.get(item_id, 0.0) >= 0.18 or item.layer in {"profile", "system"}:
                anchor_ids.append(item_id)
            if len(anchor_ids) >= 10:
                break

        for anchor_id in anchor_ids:
            anchor_item = item_by_id[anchor_id]
            anchor_score = base_scores.get(anchor_id, 0.0)
            if anchor_item.layer in {"profile", "system"}:
                anchor_score = max(anchor_score, min(anchor_item.importance, 0.72))
            if anchor_score <= 0.0:
                continue

            neighbors = sorted(
                adjacency.get(anchor_id, {}).items(),
                key=lambda item: (item[1], item_by_id[item[0]].importance),
                reverse=True,
            )[:MAX_GRAPH_NEIGHBORS]
            for target_id, weight in neighbors:
                if target_id == anchor_id:
                    continue
                direct_bonus = anchor_score * min(0.24, 0.05 + weight * 0.22)
                bonuses[target_id] = max(bonuses[target_id], direct_bonus)

                hop_neighbors = sorted(adjacency.get(target_id, {}).items(), key=lambda item: item[1], reverse=True)[:3]
                for hop_id, hop_weight in hop_neighbors:
                    if hop_id in {anchor_id, target_id}:
                        continue
                    propagated = anchor_score * weight * hop_weight * 0.12
                    bonuses[hop_id] = max(bonuses[hop_id], min(0.12, propagated))
        return bonuses

    def _render_hits(self, layer_hits: dict[str, list[MemoryHit]]) -> str:
        sections = []
        for layer in ("profile", "system", "working", "long_term", "summary", "episodic", "semantic"):
            hits = layer_hits.get(layer, [])
            if not hits:
                continue
            preview = "；".join(truncate_text(hit.item.text, 90) for hit in hits[:3])
            sections.append(f"{layer}: {preview}")
        return "\n".join(sections) if sections else "没有命中显著历史记忆。"

    def _render_hit_group(self, hits: list[MemoryHit], *, empty_text: str) -> str:
        if not hits:
            return empty_text
        return "\n".join(f"- [{hit.item.layer}] {truncate_text(hit.item.text, 120)}" for hit in hits[:8])

    def _render_session_block(self, session: SessionState, session_hits: list[MemoryHit]) -> str:
        parts = [session.summary or "当前会话还没有形成摘要。"]
        if session_hits:
            parts.append("关键片段：\n" + self._render_hit_group(session_hits[:6], empty_text=""))
        return "\n".join(part for part in parts if part).strip()

    def _render_maintenance(self, maintenance: dict[str, object]) -> str:
        if not maintenance:
            return "本轮未执行额外维护。"
        if maintenance.get("compressed"):
            return f"已执行历史压缩，新增摘要 {maintenance.get('created_summary_id')}，覆盖 {maintenance.get('covered_turns')} 轮。"
        return "本轮已完成记忆衰减检查，无需额外压缩。"

    def _render_planning_context(self, query: str, summary_blocks: dict[str, str]) -> str:
        sections = [
            f"【当前问题】\n{query}",
            f"【长期用户画像】\n{summary_blocks.get('profile', '')}",
            f"【系统策略记忆】\n{summary_blocks.get('system', '')}",
            f"【长期稳定记忆】\n{summary_blocks.get('long_term', '')}",
            f"【当前会话关键上下文】\n{summary_blocks.get('session', '')}",
            f"【与当前问题相关的历史命中】\n{summary_blocks.get('relevant', '')}",
        ]
        return "\n\n".join(section.strip() for section in sections if section.strip())

    def _anchored_layer_hits(self, layer_hits: dict[str, list[MemoryHit]], *, layer: str, limit: int) -> list[MemoryHit]:
        hits = list(layer_hits.get(layer, []))
        hits.sort(key=lambda hit: (hit.score, hit.item.importance, hit.item.updated_at), reverse=True)
        return hits[:limit]

    def _anchored_long_term_hits(self, user_id: str, layer_hits: dict[str, list[MemoryHit]]) -> list[MemoryHit]:
        hits = list(layer_hits.get("profile", [])) + list(layer_hits.get("long_term", []))
        if len(hits) >= 4:
            return sorted(hits, key=lambda hit: hit.score, reverse=True)[:6]
        memories = self.store.load_user_memories(user_id)
        anchored = [item for item in memories if item.layer == "long_term" and item.status == "active" and item.importance >= 0.86]
        anchored.sort(key=lambda item: (item.importance, item.hit_count, item.updated_at), reverse=True)
        known_ids = {hit.item.id for hit in hits}
        for item in anchored:
            if item.id in known_ids:
                continue
            hits.append(MemoryHit(item=item, score=round(item.importance, 4), reasons=["长期锚点"], breakdown={"importance": round(item.importance, 4)}))
            known_ids.add(item.id)
            if len(hits) >= 6:
                break
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:6]

    def _merge_hit_groups(self, *groups: list[MemoryHit], limit: int) -> list[MemoryHit]:
        merged: list[MemoryHit] = []
        seen_ids: set[str] = set()
        for group in groups:
            for hit in group:
                if hit.item.id in seen_ids:
                    continue
                seen_ids.add(hit.item.id)
                merged.append(hit)
                if len(merged) >= limit:
                    return merged
        return merged

    def _dynamic_related_hits(self, layer_hits: dict[str, list[MemoryHit]], *, exclude_ids: set[str], limit: int) -> list[MemoryHit]:
        dynamic_hits: list[MemoryHit] = []
        for hit in self._flatten_hits(layer_hits):
            if hit.item.id in exclude_ids:
                continue
            dynamic_hits.append(hit)
            if len(dynamic_hits) >= limit:
                break
        return dynamic_hits

    def _build_retrieval_meta(
        self,
        user_id: str,
        layer_hits: dict[str, list[MemoryHit]],
        guaranteed_hits: list[MemoryHit],
        related_hits: list[MemoryHit],
    ) -> dict[str, object]:
        vectorizer_name = self.vectorizer.__class__.__name__
        retrieval_strategy = "hybrid_hash_graph"
        vector_model = "hashing"
        if isinstance(self.vectorizer, TransformerVectorizer):
            retrieval_strategy = "hybrid_embedding_graph"
            vector_model = str(self.vectorizer.model_path)
        return {
            "retrieval_strategy": retrieval_strategy,
            "vectorizer": vectorizer_name,
            "vector_model": vector_model,
            "graph_edge_count": len(self.store.load_memory_edges(user_id)),
            "selected_layer_counts": {layer: len(hits) for layer, hits in layer_hits.items() if hits},
            "guaranteed_hit_count": len(guaranteed_hits),
            "related_hit_count": len(related_hits),
            "guaranteed_limits": dict(DEFAULT_GUARANTEED_LIMITS),
        }

    def _flatten_hits(self, layer_hits: dict[str, list[MemoryHit]], *, layers: tuple[str, ...] | None = None) -> list[MemoryHit]:
        ordered_layers = layers or ("profile", "system", "working", "long_term", "summary", "episodic", "semantic")
        hits: list[MemoryHit] = []
        for layer in ordered_layers:
            hits.extend(layer_hits.get(layer, []))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits

    def replace_last_turn_answer(self, user_id: str, session_id: str, answer: str) -> None:
        turns = self.store.load_session_turns(user_id, session_id)
        if not turns:
            return
        turns[-1].assistant_message = answer
        self.store.replace_last_session_turn(user_id, session_id, turns[-1])
        session = self.store.load_session_state(user_id, session_id)
        session.turns = turns[-self.recent_turn_window :]
        session.summary = self._summarize_session(user_id, session_id, session)
        session.updated_at = utcnow_iso()
        self.store.save_session_state(session)

    def _collect_tags(self, *texts: str) -> list[str]:
        tokens = []
        for text in texts:
            tokens.extend(simple_tokenize(text))
        return _dedupe_strings(tokens)[:8]


def _coerce_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _split_cn_list(text: str) -> list[str]:
    items = [item.strip() for item in re.split(r"[、，,；;\s]+", text) if item.strip()]
    return list(dict.fromkeys(items))


def _dedupe_strings(values: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _merge_update_dicts(primary: dict[str, object], secondary: dict[str, object]) -> dict[str, object]:
    merged = dict(primary or {})
    for key, value in (secondary or {}).items():
        if key not in merged:
            merged[key] = value
            continue
        if isinstance(merged[key], list) and isinstance(value, list):
            merged[key] = _dedupe_strings([*merged[key], *value])
            continue
        if isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key].update(value)
            continue
        merged[key] = value
    return merged