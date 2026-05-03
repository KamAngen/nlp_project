from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


MemoryLayer = Literal["profile", "system", "working", "episodic", "semantic", "long_term", "summary"]


def estimate_token_count(text: str) -> int:
    normalized = str(text or "").strip()
    if not normalized:
        return 0
    return max(1, len(normalized) // 2)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(slots=True)
class MemoryItem:
    id: str
    layer: MemoryLayer
    category: str
    text: str
    importance: float = 0.5
    confidence: float = 0.75
    abstraction: str = "fact"
    hit_count: int = 0
    tags: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    source: str = "agent"
    user_id: str | None = None
    session_id: str | None = None
    references: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)
    last_accessed_at: str | None = None
    status: str = "active"
    decay_enabled: bool = True
    token_count: int = 0

    def __post_init__(self) -> None:
        if self.token_count <= 0:
            self.token_count = estimate_token_count(self.text)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryItem":
        copied = dict(payload)
        copied.setdefault("confidence", 0.75)
        copied.setdefault("abstraction", "fact")
        copied.setdefault("keywords", [])
        copied.setdefault("references", [])
        copied.setdefault("token_count", estimate_token_count(copied.get("text", "")))
        return cls(**copied)


@dataclass(slots=True)
class MemoryHit:
    item: MemoryItem
    score: float
    reasons: list[str] = field(default_factory=list)
    breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item.to_dict(),
            "score": self.score,
            "reasons": list(self.reasons),
            "breakdown": dict(self.breakdown),
        }


@dataclass(slots=True)
class ConversationTurn:
    user_message: str
    assistant_message: str
    turn_id: str = field(default_factory=lambda: f"turn-{uuid4().hex[:12]}")
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    reasoning_trace: str = ""
    reasoning_summary: str = ""
    tags: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConversationTurn":
        copied = dict(payload)
        copied.setdefault("turn_id", f"turn-{uuid4().hex[:12]}")
        copied.setdefault("reasoning_trace", "")
        copied.setdefault("reasoning_summary", "")
        copied.setdefault("tags", [])
        copied.setdefault("payload", {})
        return cls(**copied)


@dataclass(slots=True)
class MemoryEdge:
    id: str
    source_id: str
    target_id: str
    relation: str
    weight: float = 0.5
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryEdge":
        return cls(**payload)


@dataclass(slots=True)
class SessionCompression:
    id: str
    user_id: str
    session_id: str
    from_turn_index: int
    to_turn_index: int
    summary: str
    salient_points: list[str] = field(default_factory=list)
    open_loops: list[str] = field(default_factory=list)
    memory_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utcnow_iso)
    token_count: int = 0

    def __post_init__(self) -> None:
        if self.token_count <= 0:
            self.token_count = estimate_token_count(self.summary)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionCompression":
        copied = dict(payload)
        copied.setdefault("salient_points", [])
        copied.setdefault("open_loops", [])
        copied.setdefault("memory_ids", [])
        copied.setdefault("token_count", estimate_token_count(copied.get("summary", "")))
        return cls(**copied)


@dataclass(slots=True)
class UserProfile:
    user_id: str
    name: str | None = None
    study_goals: list[str] = field(default_factory=list)
    weak_points: list[str] = field(default_factory=list)
    strong_points: list[str] = field(default_factory=list)
    preferences: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UserProfile":
        return cls(**payload)


@dataclass(slots=True)
class SessionState:
    session_id: str
    user_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    summary: str = ""
    active_exam_session_id: str | None = None
    last_report_path: str | None = None
    turn_count: int = 0
    compression_cursor: int = 0
    compression_count: int = 0
    summary_node_ids: list[str] = field(default_factory=list)
    last_turn_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["turns"] = [turn.to_dict() for turn in self.turns]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SessionState":
        turns = [ConversationTurn.from_dict(turn) for turn in payload.get("turns", [])]
        copied = dict(payload)
        copied["turns"] = turns
        copied.setdefault("turn_count", len(turns))
        copied.setdefault("compression_cursor", 0)
        copied.setdefault("compression_count", 0)
        copied.setdefault("summary_node_ids", [])
        copied.setdefault("last_turn_id", turns[-1].turn_id if turns else None)
        return cls(**copied)


@dataclass(slots=True)
class ContextBundle:
    user_profile: UserProfile
    session_state: SessionState
    layer_hits: dict[str, list[MemoryHit]]
    summary_blocks: dict[str, str]
    maintenance: dict[str, Any] = field(default_factory=dict)
    profile_hits: list[MemoryHit] = field(default_factory=list)
    system_hits: list[MemoryHit] = field(default_factory=list)
    working_hits: list[MemoryHit] = field(default_factory=list)
    long_term_hits: list[MemoryHit] = field(default_factory=list)
    session_hits: list[MemoryHit] = field(default_factory=list)
    guaranteed_hits: list[MemoryHit] = field(default_factory=list)
    related_hits: list[MemoryHit] = field(default_factory=list)
    planning_context: str = ""
    retrieval_meta: dict[str, Any] = field(default_factory=dict)

    def all_hits(self) -> list[MemoryHit]:
        hits: list[MemoryHit] = []
        for layer in ("profile", "system", "working", "long_term", "summary", "episodic", "semantic"):
            hits.extend(self.layer_hits.get(layer, []))
        return hits