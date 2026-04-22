from __future__ import annotations

from datetime import datetime, timezone
from math import log1p

from context_engine.schemas import MemoryItem
from legal_agent.utils.text import simple_tokenize


LAYER_BASE_SCORE = {
    "profile": 1.0,
    "system": 0.94,
    "working": 0.86,
    "episodic": 0.74,
    "semantic": 0.68,
}


def _parse_iso(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def lexical_overlap_score(query: str, item: MemoryItem) -> tuple[float, list[str]]:
    query_tokens = set(simple_tokenize(query))
    if not query_tokens:
        return 0.0, []

    candidate_tokens = set(simple_tokenize(item.text))
    for tag in item.tags:
        candidate_tokens.update(simple_tokenize(tag))
    for key, value in item.payload.items():
        candidate_tokens.update(simple_tokenize(str(key)))
        candidate_tokens.update(simple_tokenize(str(value)))

    overlap = query_tokens & candidate_tokens
    if not overlap:
        return 0.0, []
    score = len(overlap) / max(len(query_tokens), 1)
    reasons = [f"关键词命中：{'、'.join(sorted(overlap)[:4])}"]
    return score, reasons


def recency_score(item: MemoryItem, *, now: datetime | None = None) -> float:
    if item.layer in {"profile", "system"}:
        return 1.0
    now = now or datetime.now(timezone.utc)
    last_touch = _parse_iso(item.last_accessed_at or item.updated_at or item.created_at)
    inactivity_days = max((now - last_touch).days, 0)
    return max(0.1, 1.0 - min(0.75, inactivity_days * 0.025))


def decay_importance(item: MemoryItem, *, now: datetime | None = None) -> float:
    if item.layer in {"profile", "system"} or not item.decay_enabled:
        return item.importance

    now = now or datetime.now(timezone.utc)
    created_at = _parse_iso(item.created_at)
    last_touch = _parse_iso(item.last_accessed_at or item.updated_at or item.created_at)
    age_days = max((now - created_at).days, 0)
    inactivity_days = max((now - last_touch).days, 0)
    decay = min(0.35, age_days * 0.003 + inactivity_days * 0.01)
    recovery = min(0.18, log1p(max(item.hit_count, 0)) * 0.05)
    return max(0.05, min(1.0, float(item.importance) - decay + recovery))


def memory_score(query: str, item: MemoryItem, *, now: datetime | None = None) -> tuple[float, list[str]]:
    now = now or datetime.now(timezone.utc)
    lexical, reasons = lexical_overlap_score(query, item)
    freshness = recency_score(item, now=now)
    importance = decay_importance(item, now=now)
    layer_score = LAYER_BASE_SCORE.get(item.layer, 0.5)
    hit_bonus = min(0.25, log1p(max(item.hit_count, 0)) * 0.08)
    score = 0.38 * lexical + 0.27 * importance + 0.15 * freshness + 0.12 * layer_score + 0.08 * hit_bonus
    if importance >= 0.85:
        reasons.append("高重要度")
    if item.hit_count >= 3:
        reasons.append("历史高命中")
    if item.layer in {"profile", "system"}:
        reasons.append("高优先层")
    return score, reasons