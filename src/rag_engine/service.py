from __future__ import annotations

from pathlib import Path
import random
from typing import Iterable

from legal_agent.utils.text import simple_tokenize, truncate_text
from rag_engine.loaders import load_case_bank, load_common_knowledge, load_question_bank
from rag_engine.schema import KnowledgeHit, KnowledgeRecord


class LegacyStatuteRetrieverAdapter:
    def __init__(self, legacy_config_path: str | Path, *, device: str = "cpu") -> None:
        self.legacy_config_path = Path(legacy_config_path)
        self.device = device
        self._retriever = None
        self._error: str | None = None

    def available(self) -> bool:
        return self.legacy_config_path.exists()

    def _ensure_retriever(self):
        if self._retriever is not None:
            return self._retriever
        from legal_agent.config import load_app_config
        from legal_agent.rag.retriever import HybridLegalRetriever

        config = load_app_config(self.legacy_config_path)
        self._retriever = HybridLegalRetriever(config, device=self.device)
        return self._retriever

    def search(self, query: str, *, top_k: int = 5) -> list[KnowledgeHit]:
        if not self.available():
            return []
        try:
            retriever = self._ensure_retriever()
            hits = retriever.search(query, top_k=top_k)
        except Exception as exc:
            self._error = str(exc)
            return []

        converted: list[KnowledgeHit] = []
        for hit in hits:
            converted.append(
                KnowledgeHit(
                    source_type="statute",
                    record_id=hit.chunk_id,
                    title=f"{hit.document_title} {hit.article_heading or ''}".strip(),
                    excerpt=truncate_text(hit.text, 220),
                    score=float(hit.score),
                    metadata={
                        "effect_level": hit.effect_level,
                        "region": hit.region_name,
                        "source_path": hit.source_path,
                    },
                )
            )
        return converted


class KnowledgeService:
    def __init__(
        self,
        *,
        question_bank_path: str | Path,
        case_bank_path: str | Path,
        common_knowledge_path: str | Path,
        use_legacy_statute_rag: bool = False,
        legacy_config_path: str | Path | None = None,
        legacy_device: str = "cpu",
    ) -> None:
        self.question_bank = load_question_bank(question_bank_path)
        self.case_bank = load_case_bank(case_bank_path)
        self.common_knowledge = load_common_knowledge(common_knowledge_path)
        self.legacy_statute = None
        if use_legacy_statute_rag and legacy_config_path is not None:
            self.legacy_statute = LegacyStatuteRetrieverAdapter(legacy_config_path, device=legacy_device)

    def summary(self) -> dict[str, object]:
        return {
            "question_bank_count": len(self.question_bank),
            "case_bank_count": len(self.case_bank),
            "common_knowledge_count": len(self.common_knowledge),
            "legacy_statute_enabled": bool(self.legacy_statute is not None),
        }

    def search(self, query: str, *, sources: Iterable[str] | None = None, top_k: int = 5) -> list[KnowledgeHit]:
        selected_sources = set(sources or ["statute", "question_bank", "case_bank", "common_knowledge"])
        hits: list[KnowledgeHit] = []
        if "question_bank" in selected_sources:
            hits.extend(self._search_local_records(query, self.question_bank, top_k=top_k))
        if "case_bank" in selected_sources:
            hits.extend(self._search_local_records(query, self.case_bank, top_k=top_k))
        if "common_knowledge" in selected_sources:
            hits.extend(self._search_local_records(query, self.common_knowledge, top_k=top_k))
        if "statute" in selected_sources and self.legacy_statute is not None:
            hits.extend(self.legacy_statute.search(query, top_k=top_k))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]

    def sample_questions(
        self,
        *,
        topic: str | None = None,
        question_count: int = 5,
        preferred_tags: list[str] | None = None,
        exam_type: str | None = None,
        avoid_question_ids: list[str] | None = None,
        prioritized_question_ids: list[str] | None = None,
        strong_tags: list[str] | None = None,
        random_seed: int | None = None,
    ) -> list[KnowledgeRecord]:
        preferred_tags = [str(tag) for tag in preferred_tags or [] if str(tag).strip()]
        avoid_ids = {str(record_id) for record_id in avoid_question_ids or [] if str(record_id).strip()}
        prioritized_ids = {str(record_id) for record_id in prioritized_question_ids or [] if str(record_id).strip()}
        mastered_tags = [str(tag) for tag in strong_tags or [] if str(tag).strip()]
        effective_exam_type = str(exam_type or "综合练习").strip() or "综合练习"
        rng = random.Random(random_seed)

        pool = list(self.question_bank)
        topic_hits = []
        if topic and topic != "综合":
            topic_hits = [record for record in pool if self._topic_matches(record, topic)]
            if topic_hits and effective_exam_type == "章节练习":
                pool = topic_hits
        if not pool:
            return []

        priority_pool = [
            record
            for record in pool
            if record.record_id in prioritized_ids and (not topic_hits or self._topic_matches(record, topic or ""))
        ]
        if not priority_pool and prioritized_ids:
            priority_pool = [record for record in self.question_bank if record.record_id in prioritized_ids]

        selected: list[KnowledgeRecord] = []
        priority_target = self._priority_question_target(effective_exam_type, question_count, len(priority_pool))
        if priority_target > 0:
            selected.extend(
                self._weighted_sample_without_replacement(
                    priority_pool,
                    priority_target,
                    rng,
                    weight_fn=lambda record: self._question_weight(
                        record,
                        topic=topic,
                        preferred_tags=preferred_tags,
                        exam_type=effective_exam_type,
                        avoid_ids=set(),
                        prioritized_ids=prioritized_ids,
                        strong_tags=mastered_tags,
                    )
                    + 2.4,
                )
            )

        selected_ids = {record.record_id for record in selected}
        candidate_pool = [record for record in pool if record.record_id not in selected_ids]
        if topic_hits and effective_exam_type != "章节练习":
            candidate_pool = self._dedupe_records([*topic_hits, *candidate_pool])

        remaining = max(question_count - len(selected), 0)
        if remaining > 0:
            fresh_pool = [record for record in candidate_pool if record.record_id not in avoid_ids]
            if len(fresh_pool) >= remaining:
                candidate_pool = fresh_pool
            selected.extend(
                self._weighted_sample_without_replacement(
                    candidate_pool,
                    remaining,
                    rng,
                    weight_fn=lambda record: self._question_weight(
                        record,
                        topic=topic,
                        preferred_tags=preferred_tags,
                        exam_type=effective_exam_type,
                        avoid_ids=avoid_ids,
                        prioritized_ids=prioritized_ids,
                        strong_tags=mastered_tags,
                    ),
                )
            )

        selected_ids = {record.record_id for record in selected}
        remaining = max(question_count - len(selected), 0)
        if remaining > 0:
            fallback_pool = [record for record in self.question_bank if record.record_id not in selected_ids]
            selected.extend(
                self._weighted_sample_without_replacement(
                    fallback_pool,
                    remaining,
                    rng,
                    weight_fn=lambda record: self._question_weight(
                        record,
                        topic=topic,
                        preferred_tags=preferred_tags,
                        exam_type=effective_exam_type,
                        avoid_ids=set(),
                        prioritized_ids=prioritized_ids,
                        strong_tags=mastered_tags,
                    ),
                )
            )

        rng.shuffle(selected)
        return selected[:question_count]

    def _priority_question_target(self, exam_type: str, question_count: int, priority_count: int) -> int:
        if priority_count <= 0 or question_count <= 0:
            return 0
        ratio = {
            "薄弱点强化": 0.7,
            "章节练习": 0.45,
            "真题模拟": 0.25,
            "综合练习": 0.35,
        }.get(exam_type, 0.3)
        target = max(1, round(question_count * ratio))
        return min(priority_count, question_count, target)

    def _question_weight(
        self,
        record: KnowledgeRecord,
        *,
        topic: str | None,
        preferred_tags: list[str],
        exam_type: str,
        avoid_ids: set[str],
        prioritized_ids: set[str],
        strong_tags: list[str],
    ) -> float:
        weight = 1.0
        tag_bonus = sum(1 for tag in preferred_tags if self._topic_matches(record, tag))
        topic_match = bool(topic and topic != "综合" and self._topic_matches(record, topic))
        difficulty = str(record.difficulty or "medium")
        difficulty_bonus = {"easy": 0.15, "medium": 0.35, "hard": 0.55}.get(difficulty, 0.3)

        if topic_match:
            weight += 2.4 if exam_type == "章节练习" else 1.3
        elif topic and topic != "综合" and exam_type == "章节练习":
            weight *= 0.2

        weight += tag_bonus * (1.4 if exam_type == "薄弱点强化" else 0.9)
        weight += difficulty_bonus

        if record.record_id in prioritized_ids:
            weight += 2.0 if exam_type == "薄弱点强化" else 1.0

        mastered_hits = sum(1 for tag in strong_tags if self._topic_matches(record, tag))
        if mastered_hits:
            weight *= max(0.2, 1 - 0.18 * mastered_hits)

        if record.record_id in avoid_ids:
            weight *= 0.22 if exam_type != "薄弱点强化" else 0.55

        if exam_type == "真题模拟":
            weight += {"easy": 0.0, "medium": 0.35, "hard": 0.85}.get(difficulty, 0.3)
        elif exam_type == "综合练习":
            weight += {"easy": 0.2, "medium": 0.45, "hard": 0.35}.get(difficulty, 0.25)

        return max(weight, 0.01)

    def _weighted_sample_without_replacement(
        self,
        records: list[KnowledgeRecord],
        count: int,
        rng: random.Random,
        *,
        weight_fn,
    ) -> list[KnowledgeRecord]:
        if count <= 0 or not records:
            return []
        available = list(records)
        selected: list[KnowledgeRecord] = []
        target_count = min(count, len(available))
        while available and len(selected) < target_count:
            weights = [max(float(weight_fn(record)), 0.01) for record in available]
            choice = rng.choices(available, weights=weights, k=1)[0]
            selected.append(choice)
            available = [record for record in available if record.record_id != choice.record_id]
        return selected

    def _dedupe_records(self, records: list[KnowledgeRecord]) -> list[KnowledgeRecord]:
        deduped: list[KnowledgeRecord] = []
        seen_ids: set[str] = set()
        for record in records:
            if record.record_id in seen_ids:
                continue
            seen_ids.add(record.record_id)
            deduped.append(record)
        return deduped

    def _search_local_records(self, query: str, records: list[KnowledgeRecord], *, top_k: int) -> list[KnowledgeHit]:
        scored: list[KnowledgeHit] = []
        query_tokens = set(simple_tokenize(query))
        for record in records:
            record_tokens = set(simple_tokenize(record.title)) | set(simple_tokenize(record.content))
            for tag in record.tags:
                record_tokens.update(simple_tokenize(tag))
            overlap = query_tokens & record_tokens
            if not overlap:
                continue
            score = len(overlap) / max(len(query_tokens), 1)
            if any(token in record.title for token in overlap):
                score += 0.1
            scored.append(
                KnowledgeHit(
                    source_type=record.source_type,
                    record_id=record.record_id,
                    title=record.title,
                    excerpt=truncate_text(record.content, 220),
                    score=round(score, 4),
                    metadata=record.metadata,
                )
            )
        return sorted(scored, key=lambda hit: hit.score, reverse=True)[:top_k]

    def _topic_matches(self, record: KnowledgeRecord, topic: str) -> bool:
        target = str(topic or "").strip()
        if not target:
            return False
        if any(target == str(tag).strip() for tag in record.tags):
            return True
        token_sources = [record.title, record.content, *record.tags]
        for source in token_sources:
            if target in simple_tokenize(str(source or "")):
                return True
        return False