from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


MemoryLayer = Literal["profile", "system", "working", "episodic", "semantic"]


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(slots=True)
class MemoryItem:
    id: str
    layer: MemoryLayer
    category: str
    text: str
    importance: float = 0.5
    hit_count: int = 0
    tags: list[str] = field(default_factory=list)
    source: str = "agent"
    user_id: str | None = None
    session_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)
    last_accessed_at: str | None = None
    status: str = "active"
    decay_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryItem":
        return cls(**payload)


@dataclass(slots=True)
class MemoryHit:
    item: MemoryItem
    score: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item.to_dict(),
            "score": self.score,
            "reasons": list(self.reasons),
        }


@dataclass(slots=True)
class ConversationTurn:
    user_message: str
    assistant_message: str
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConversationTurn":
        return cls(**payload)


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
        return cls(**copied)


@dataclass(slots=True)
class ContextBundle:
    user_profile: UserProfile
    session_state: SessionState
    layer_hits: dict[str, list[MemoryHit]]
    summary_blocks: dict[str, str]

    def all_hits(self) -> list[MemoryHit]:
        hits: list[MemoryHit] = []
        for layer in ("profile", "system", "working", "episodic", "semantic"):
            hits.extend(self.layer_hits.get(layer, []))
        return hits