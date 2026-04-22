from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class KnowledgeRecord:
    record_id: str
    source_type: str
    title: str
    content: str
    tags: list[str] = field(default_factory=list)
    difficulty: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class KnowledgeHit:
    source_type: str
    record_id: str
    title: str
    excerpt: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)