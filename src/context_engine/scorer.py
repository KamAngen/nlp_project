from __future__ import annotations

from datetime import datetime, timezone
from math import log1p

from context_engine.schemas import MemoryItem
from context_engine.vectorizer import MemoryVectorizer
from legal_agent.utils.text import simple_tokenize


LAYER_BASE_SCORE = {
    "profile": 1.0,
    "system": 0.94,
    "working": 0.9,
    "long_term": 0.88,
    "summary": 0.84,
    "episodic": 0.74,
    "semantic": 0.72,
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
    for keyword in item.keywords:
        candidate_tokens.update(simple_tokenize(keyword))
    for key, value in item.payload.items():
        candidate_tokens.update(simple_tokenize(str(key)))
        candidate_tokens.update(simple_tokenize(str(value)))

    overlap = query_tokens & candidate_tokens
    if not overlap:
        return 0.0, []
    score = len(overlap) / max(len(query_tokens), 1)
    reasons = [f"关键词命中：{'、'.join(sorted(overlap)[:4])}"]
    return score, reasons


def vector_similarity_score(query: str, item: MemoryItem, *, vectorizer: MemoryVectorizer | None = None) -> float:
    if vectorizer is None:
        return 0.0
    candidate_parts = [item.text, *item.tags[:4], *item.keywords[:6]]
    candidate_text = "\n".join(part for part in candidate_parts if str(part or "").strip())
    try:
        return vectorizer.similarity(query, candidate_text)
    except Exception:
        return 0.0


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
    if item.layer == "long_term":
        decay = min(0.22, age_days * 0.0015 + inactivity_days * 0.004)
    elif item.layer == "summary":
        decay = min(0.28, age_days * 0.0025 + inactivity_days * 0.007)
    else:
        decay = min(0.35, age_days * 0.003 + inactivity_days * 0.01)
    recovery = min(0.18, log1p(max(item.hit_count, 0)) * 0.05)
    return max(0.05, min(1.0, float(item.importance) - decay + recovery))


def memory_score(
    query: str,
    item: MemoryItem,
    *,
    vectorizer: MemoryVectorizer | None = None,
    now: datetime | None = None,
) -> tuple[float, list[str], dict[str, float]]:
    now = now or datetime.now(timezone.utc)
    lexical, reasons = lexical_overlap_score(query, item)
    vector = vector_similarity_score(query, item, vectorizer=vectorizer)
    freshness = recency_score(item, now=now)
    importance = decay_importance(item, now=now)
    layer_score = LAYER_BASE_SCORE.get(item.layer, 0.5)
    hit_bonus = min(0.25, log1p(max(item.hit_count, 0)) * 0.08)
    score = 0.34 * lexical + 0.24 * vector + 0.18 * importance + 0.1 * freshness + 0.08 * layer_score + 0.06 * hit_bonus
    if vector >= 0.2:
        reasons.append("语义相近")
    if importance >= 0.85:
        reasons.append("高重要度")
    if item.hit_count >= 3:
        reasons.append("历史高命中")
    if item.layer in {"profile", "system"}:
        reasons.append("高优先层")
    return score, reasons, {
        "lexical": round(lexical, 4),
        "vector": round(vector, 4),
        "freshness": round(freshness, 4),
        "importance": round(importance, 4),
        "layer": round(layer_score, 4),
        "hit_bonus": round(hit_bonus, 4),
    }