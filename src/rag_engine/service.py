from __future__ import annotations

from collections import defaultdict
from hashlib import blake2b
from pathlib import Path
import random
import re
from typing import Iterable

from legal_agent.utils.text import clean_text, simple_tokenize, truncate_text
from rag_engine.loaders import load_case_bank, load_common_knowledge, load_question_bank
from rag_engine.schema import KnowledgeHit, KnowledgeRecord


EXAM_OPTION_LABELS = ("A", "B", "C", "D")
CANONICAL_TOPICS = ("民法", "刑法", "行政法", "民诉", "刑诉", "商经", "理论法", "综合")
TOPIC_QUERY_ALIASES = {
    "民法": ("民法", "民法典"),
    "刑法": ("刑法",),
    "行政法": ("行政法", "行政诉讼", "行诉"),
    "民诉": ("民诉", "民事诉讼", "民事诉讼法"),
    "刑诉": ("刑诉", "刑事诉讼", "刑事诉讼法"),
    "商经": ("商经", "商经法", "经济法"),
    "理论法": ("理论法", "法理", "宪法", "立法法"),
    "综合": ("综合",),
}
QUESTION_TYPE_ALIASES = {
    "single_choice": ("single_choice", "单选", "选择题", "单选题"),
    "short_answer": ("short_answer", "简答", "简答题", "主观题"),
    "case_analysis": ("case_analysis", "案例", "案例分析", "案例分析题"),
}
CONTEXT_DEPENDENT_QUESTION_SNIPPETS = (
    "根据以上案例",
    "根据以上背景",
    "阅读以上背景",
    "阅读以上材料",
    "根据上文",
    "根据前文",
    "根据下列材料",
    "以上案例",
    "以上背景",
    "以上材料",
)
CASE_ANALYSIS_ACTION_HINTS = (
    "请阅读以下案情",
    "请根据以下案情",
    "请结合以下案情",
    "请结合案情",
    "请分析",
    "请说明理由",
    "结合法律规定",
    "推理出以下案件中的判决",
    "推断案件判决",
    "判决结果",
    "裁判结果",
    "如何处理",
    "如何认定",
    "如何定性",
    "是否构成",
    "是否违法",
    "应承担",
    "争议焦点",
    "法律后果",
    "应如何",
    "何罪",
)
CASE_ANALYSIS_FACT_HINTS = (
    "案情",
    "本案",
    "被告人",
    "原告",
    "被告",
    "上诉人",
    "被上诉人",
    "申请人",
    "被申请人",
    "人民检察院指控",
    "法院经审理查明",
    "甲公司",
    "乙公司",
    "某公司",
    "某企业",
    "某厂",
    "某村",
    "某县",
    "某市",
)
CASE_ANALYSIS_FACT_RE = re.compile(r"(?:19|20)\d{2}年|甲[乙丙丁]|乙[方人]|丙[方人]|丁[方人]")
CASE_ANALYSIS_LEADING_ENUM_RE = re.compile(r"^(?:第?[一二三四五六七八九十百]+[、.．)]|\d+[、.．)])\s*")
CASE_ANALYSIS_PROMPT_BY_TASK_FAMILY = {
    "judgement_predit": "请阅读以下案情，推断案件判决结果并说明理由：",
    "jud_doc_sum": "请阅读以下案情，概括案件要点并结合法律规定作答：",
    "leg_case_cls": "请阅读以下案情，判断其所属法律问题并说明理由：",
    "sim_case_match": "请阅读以下案情，分析其法律争点并作答：",
    "jud_read_compre": "请阅读以下案情，结合法律规定进行案例分析并作答：",
}
FAST_ALNUM_TOKEN_RE = re.compile(r"[0-9A-Za-z_]{2,}")
FAST_CHINESE_BLOCK_RE = re.compile(r"[\u4e00-\u9fff]+")
COMPARISON_NORMALIZE_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")


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
        self._usable_question_bank: list[KnowledgeRecord] = []
        self._usable_question_type_index: dict[str, list[KnowledgeRecord]] = {}
        self._usable_question_loose_topic_index: dict[str, list[KnowledgeRecord]] = {}
        self._usable_question_strict_topic_index: dict[str, list[KnowledgeRecord]] = {}
        self._usable_question_loose_topic_type_index: dict[tuple[str, str], list[KnowledgeRecord]] = {}
        self._usable_question_strict_topic_type_index: dict[tuple[str, str], list[KnowledgeRecord]] = {}
        self._local_search_indexes: dict[str, dict[str, object]] = {}
        self.legacy_statute = None
        if use_legacy_statute_rag and legacy_config_path is not None:
            self.legacy_statute = LegacyStatuteRetrieverAdapter(legacy_config_path, device=legacy_device)
        self._build_question_indexes()
        self._build_local_search_indexes()

    def summary(self) -> dict[str, object]:
        topic_distribution: dict[str, int] = {}
        question_type_distribution: dict[str, int] = {}
        for record in self.question_bank:
            topic = self._primary_topic(record) or "综合"
            topic_distribution[topic] = topic_distribution.get(topic, 0) + 1
            question_type = self._question_type(record)
            question_type_distribution[question_type] = question_type_distribution.get(question_type, 0) + 1
        return {
            "question_bank_count": len(self.question_bank),
            "case_bank_count": len(self.case_bank),
            "common_knowledge_count": len(self.common_knowledge),
            "legacy_statute_enabled": bool(self.legacy_statute is not None),
            "question_topic_distribution": topic_distribution,
            "question_type_distribution": question_type_distribution,
        }

    def search(self, query: str, *, sources: Iterable[str] | None = None, top_k: int = 5) -> list[KnowledgeHit]:
        selected_sources = set(sources or ["statute", "question_bank", "case_bank", "common_knowledge"])
        hits: list[KnowledgeHit] = []
        if "question_bank" in selected_sources:
            hits.extend(self._search_local_records(query, self.question_bank, top_k=top_k, source_name="question_bank"))
        if "case_bank" in selected_sources:
            hits.extend(self._search_local_records(query, self.case_bank, top_k=top_k, source_name="case_bank"))
        if "common_knowledge" in selected_sources:
            hits.extend(self._search_local_records(query, self.common_knowledge, top_k=top_k, source_name="common_knowledge"))
        if "statute" in selected_sources and self.legacy_statute is not None:
            hits.extend(self.legacy_statute.search(query, top_k=top_k))
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]

    def sample_questions(
        self,
        *,
        topic: str | None = None,
        question_count: int = 5,
        question_types: list[str] | None = None,
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
        normalized_topic = self._normalize_topic_name(topic)
        normalized_question_types = self._normalize_question_types(question_types)
        strict_topic = bool(normalized_topic and normalized_topic != "综合" and effective_exam_type in {"章节练习", "薄弱点强化", "真题模拟"})

        question_pool = self._question_pool_from_index(normalized_question_types)
        pool = list(question_pool)
        topic_hits = []
        if normalized_topic and normalized_topic != "综合":
            topic_hits = self._question_pool_from_index(normalized_question_types, topic=normalized_topic, strict=strict_topic)
            if strict_topic:
                pool = topic_hits
            elif topic_hits:
                pool = topic_hits
        if strict_topic and not pool:
            return []
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
            fallback_source = pool if strict_topic else question_pool
            fallback_pool = [record for record in fallback_source if record.record_id not in selected_ids]
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

    def _build_question_indexes(self) -> None:
        usable_question_bank: list[KnowledgeRecord] = []
        by_type: dict[str, list[KnowledgeRecord]] = defaultdict(list)
        loose_topic_index: dict[str, list[KnowledgeRecord]] = defaultdict(list)
        strict_topic_index: dict[str, list[KnowledgeRecord]] = defaultdict(list)
        loose_topic_type_index: dict[tuple[str, str], list[KnowledgeRecord]] = defaultdict(list)
        strict_topic_type_index: dict[tuple[str, str], list[KnowledgeRecord]] = defaultdict(list)

        for record in self.question_bank:
            if not self._is_usable_exam_record(record):
                continue
            usable_question_bank.append(record)
            question_type = self._question_type(record)
            by_type[question_type].append(record)

            primary_topic = self._primary_topic(record)
            tag_topics = {
                normalized
                for tag in record.tags
                for normalized in [self._normalize_topic_name(str(tag or "").strip())]
                if normalized in CANONICAL_TOPICS and normalized != "综合"
            }
            loose_topics = set(tag_topics)
            strict_topics: set[str] = set()
            if primary_topic and primary_topic != "综合":
                loose_topics.add(primary_topic)
                strict_topics.add(primary_topic)

            for topic in loose_topics:
                loose_topic_index[topic].append(record)
                loose_topic_type_index[(topic, question_type)].append(record)
            for topic in strict_topics:
                strict_topic_index[topic].append(record)
                strict_topic_type_index[(topic, question_type)].append(record)

        self._usable_question_bank = usable_question_bank
        self._usable_question_type_index = dict(by_type)
        self._usable_question_loose_topic_index = dict(loose_topic_index)
        self._usable_question_strict_topic_index = dict(strict_topic_index)
        self._usable_question_loose_topic_type_index = dict(loose_topic_type_index)
        self._usable_question_strict_topic_type_index = dict(strict_topic_type_index)

    def _question_pool_from_index(
        self,
        question_types: list[str],
        *,
        topic: str | None = None,
        strict: bool = False,
    ) -> list[KnowledgeRecord]:
        normalized_topic = self._normalize_topic_name(topic)
        if normalized_topic and normalized_topic != "综合":
            if question_types:
                index = self._usable_question_strict_topic_type_index if strict else self._usable_question_loose_topic_type_index
                pooled: list[KnowledgeRecord] = []
                for question_type in question_types:
                    pooled.extend(index.get((normalized_topic, question_type), []))
                return self._dedupe_records(pooled)
            topic_index = self._usable_question_strict_topic_index if strict else self._usable_question_loose_topic_index
            return list(topic_index.get(normalized_topic, []))

        if question_types:
            pooled = []
            for question_type in question_types:
                pooled.extend(self._usable_question_type_index.get(question_type, []))
            return self._dedupe_records(pooled)
        return list(self._usable_question_bank)

    def _build_local_search_indexes(self) -> None:
        self._local_search_indexes = {
            "question_bank": self._build_local_search_index(self.question_bank),
            "case_bank": self._build_local_search_index(self.case_bank),
            "common_knowledge": self._build_local_search_index(self.common_knowledge),
        }

    def _build_local_search_index(self, records: list[KnowledgeRecord]) -> dict[str, object]:
        postings: dict[str, list[int]] = defaultdict(list)
        record_tokens: list[frozenset[str]] = []
        for index, record in enumerate(records):
            tokens = frozenset(self._search_index_tokens(record))
            record_tokens.append(tokens)
            for token in tokens:
                postings[token].append(index)
        return {
            "postings": {token: tuple(indexes) for token, indexes in postings.items()},
            "record_tokens": tuple(record_tokens),
        }

    def _search_tokens(self, text: str) -> set[str]:
        normalized = clean_text(text)
        if not normalized:
            return set()

        tokens = set(FAST_ALNUM_TOKEN_RE.findall(normalized.lower()))
        for chunk in FAST_CHINESE_BLOCK_RE.findall(normalized):
            chunk = chunk.strip()
            if len(chunk) < 2:
                continue
            if len(chunk) <= 8:
                tokens.add(chunk)
            for window in (2, 3):
                if len(chunk) < window:
                    continue
                for index in range(len(chunk) - window + 1):
                    tokens.add(chunk[index : index + window])
        return tokens

    def _search_index_tokens(self, record: KnowledgeRecord) -> set[str]:
        preview = truncate_text(record.content, 260)
        parts = [record.title, preview, *record.tags]
        return self._search_tokens("\n".join(part for part in parts if str(part or "").strip()))

    def _record_search_tokens(self, source_name: str | None, record_index: int, record: KnowledgeRecord) -> set[str]:
        if source_name:
            search_index = self._local_search_indexes.get(source_name)
            if search_index is not None:
                return set(search_index["record_tokens"][record_index])
        return self._search_index_tokens(record)

    def build_exam_questions(
        self,
        records: list[KnowledgeRecord],
        *,
        requested_topic: str | None = None,
    ) -> tuple[list[dict[str, object]], list[str]]:
        questions: list[dict[str, object]] = []
        skipped_count = 0
        for index, record in enumerate(records, start=1):
            built = self._build_exam_question(record, index=index, requested_topic=requested_topic)
            if built is None:
                skipped_count += 1
                continue
            questions.append(built)

        notes: list[str] = []
        if skipped_count:
            notes.append(f"本次跳过了 {skipped_count} 题上下文缺失或结构不完整的题目，避免把残缺题塞进试卷。")
        type_counts: dict[str, int] = {}
        for question in questions:
            question_type = str(question.get("question_type") or "single_choice")
            type_counts[question_type] = type_counts.get(question_type, 0) + 1
        if type_counts:
            fragments = [f"{question_type} {count} 题" for question_type, count in sorted(type_counts.items())]
            notes.append(f"本次试卷题型构成：{'，'.join(fragments)}。")
        return questions, notes

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
        strict_topic_quality = self._strict_topic_quality(record, topic or "") if topic and topic != "综合" else 0
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

        if exam_type in {"章节练习", "薄弱点强化", "真题模拟"} and strict_topic_quality:
            weight += 1.1 * strict_topic_quality

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

    def _search_local_records(
        self,
        query: str,
        records: list[KnowledgeRecord],
        *,
        top_k: int,
        source_name: str | None = None,
    ) -> list[KnowledgeHit]:
        scored: list[KnowledgeHit] = []
        query_tokens = self._search_tokens(query)
        if not query_tokens:
            return []

        candidate_indexes: Iterable[int]
        search_index = self._local_search_indexes.get(source_name or "") if source_name else None
        if search_index is not None:
            candidate_set: set[int] = set()
            postings = search_index["postings"]
            for token in query_tokens:
                candidate_set.update(postings.get(token, ()))
            candidate_indexes = candidate_set or range(len(records))
        else:
            candidate_indexes = range(len(records))

        for index in candidate_indexes:
            record = records[index]
            record_tokens = self._record_search_tokens(source_name, index, record)
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
        return self._record_matches_topic(record, topic, strict=False)

    def _record_matches_topic(self, record: KnowledgeRecord, topic: str, *, strict: bool) -> bool:
        target = self._normalize_topic_name(topic)
        if not target:
            return False
        if target == "综合":
            return True
        if strict:
            return self._strict_topic_quality(record, target) >= 2
        return self._strict_topic_quality(record, target) > 0

    def _normalize_topic_name(self, topic: str | None) -> str:
        value = str(topic or "").strip()
        if not value:
            return ""
        for canonical, aliases in TOPIC_QUERY_ALIASES.items():
            if value == canonical or value in aliases:
                return canonical
        return value

    def _normalize_question_types(self, question_types: list[str] | None) -> list[str]:
        normalized: list[str] = []
        for raw in question_types or []:
            value = str(raw or "").strip()
            if not value:
                continue
            for canonical, aliases in QUESTION_TYPE_ALIASES.items():
                if value == canonical or value in aliases:
                    if canonical not in normalized:
                        normalized.append(canonical)
                    break
            else:
                if value not in normalized:
                    normalized.append(value)
        return normalized

    def _matches_question_types(self, record: KnowledgeRecord, question_types: list[str]) -> bool:
        if not question_types:
            return True
        return self._question_type(record) in question_types

    def _match_aliases(self, aliases: tuple[str, ...], sources: Iterable[str]) -> int:
        hit_count = 0
        for source in sources:
            text = str(source or "").strip()
            if not text:
                continue
            tokens = set(simple_tokenize(text))
            for alias in aliases:
                if alias in text or alias in tokens:
                    hit_count += 1
        return hit_count

    def _primary_topic(self, record: KnowledgeRecord) -> str | None:
        metadata_topic = self._normalize_topic_name(str(record.metadata.get("topic") or "").strip())
        if metadata_topic in CANONICAL_TOPICS:
            return metadata_topic
        for tag in record.tags:
            value = self._normalize_topic_name(str(tag or "").strip())
            if value in CANONICAL_TOPICS:
                return value
        return None

    def _question_type(self, record: KnowledgeRecord) -> str:
        value = str(record.metadata.get("question_type") or "").strip()
        if value:
            return value
        return "single_choice" if dict(record.metadata.get("options") or {}) else "short_answer"

    def _strict_topic_quality(self, record: KnowledgeRecord, topic: str) -> int:
        target = self._normalize_topic_name(topic)
        if not target:
            return 0

        primary_topic = self._primary_topic(record)
        if primary_topic == target:
            return 2
        tag_topics = {self._normalize_topic_name(tag) for tag in record.tags if self._normalize_topic_name(tag)}
        if target in tag_topics:
            return 1
        return 0

    def _is_usable_exam_record(self, record: KnowledgeRecord) -> bool:
        question_text = self._normalize_exam_question_text(record)
        if not question_text or self._is_context_dependent_question(question_text):
            return False
        question_type = self._question_type(record)
        if question_type == "single_choice":
            options = self._normalize_options(record.metadata.get("options") or {})
            answer = str(record.metadata.get("answer") or "").upper().strip()
            return self._options_are_consistent(options, answer)
        reference_answer = str(record.metadata.get("reference_answer") or record.metadata.get("answer") or "").strip()
        analysis = str(record.metadata.get("analysis") or "").strip()
        return bool(reference_answer or analysis)

    def _is_context_dependent_question(self, text: str) -> bool:
        normalized = str(text or "").strip()
        if not normalized:
            return True
        return any(snippet in normalized for snippet in CONTEXT_DEPENDENT_QUESTION_SNIPPETS)

    def _normalize_exam_question_text(self, record: KnowledgeRecord) -> str:
        question_text = clean_text(str(record.title or ""))
        if not question_text:
            return ""
        if self._question_type(record) != "case_analysis":
            return question_text
        if self._has_case_analysis_action(question_text):
            return question_text
        if not self._looks_like_case_fact_text(question_text):
            return ""
        task_family = self._source_task_family(record)
        prompt = CASE_ANALYSIS_PROMPT_BY_TASK_FAMILY.get(task_family, "请阅读以下案情，结合法律规定进行案例分析并作答：")
        facts = clean_text(CASE_ANALYSIS_LEADING_ENUM_RE.sub("", question_text))
        if not facts:
            return ""
        return f"{prompt}\n{facts}"

    def _source_task_family(self, record: KnowledgeRecord) -> str:
        source_metadata = dict(record.metadata.get("source_metadata") or {})
        return str(source_metadata.get("task_family") or "").strip()

    def _has_case_analysis_action(self, text: str) -> bool:
        cleaned = clean_text(text)
        if not cleaned:
            return False
        return any(hint in cleaned for hint in CASE_ANALYSIS_ACTION_HINTS)

    def _looks_like_case_fact_text(self, text: str) -> bool:
        cleaned = clean_text(text)
        if not cleaned:
            return False
        if any(hint in cleaned for hint in CASE_ANALYSIS_FACT_HINTS):
            return True
        return bool(CASE_ANALYSIS_FACT_RE.search(cleaned))

    def _build_exam_question(
        self,
        record: KnowledgeRecord,
        *,
        index: int,
        requested_topic: str | None,
    ) -> dict[str, object] | None:
        question_text = self._normalize_exam_question_text(record)
        if not question_text or self._is_context_dependent_question(question_text):
            return None

        question_type = self._question_type(record)
        base_payload = {
            "index": index,
            "record_id": record.record_id,
            "topic": self._resolve_exam_topic(record, requested_topic=requested_topic),
            "question_type": question_type,
            "evaluation_mode": str(record.metadata.get("evaluation_mode") or ("objective_choice" if question_type == "single_choice" else "llm_subjective")),
            "question": question_text,
            "analysis": str(record.metadata.get("analysis") or "").strip(),
            "tags": self._normalize_exam_tags(record, requested_topic=requested_topic),
            "references": [str(item) for item in record.metadata.get("references", []) if str(item).strip()],
            "score": int(record.metadata.get("score", 20)),
        }
        if question_type == "single_choice":
            answer = str(record.metadata.get("answer") or "").upper().strip()
            options = self._normalize_options(record.metadata.get("options") or {})
            if not self._options_are_consistent(options, answer):
                return None
            return {
                **base_payload,
                "options": options,
                "answer": answer,
                "reference_answer": str(record.metadata.get("reference_answer") or options.get(answer) or "").strip(),
            }

        reference_answer = str(record.metadata.get("reference_answer") or record.metadata.get("answer") or "").strip()
        if not reference_answer and not base_payload["analysis"]:
            return None
        return {
            **base_payload,
            "options": {},
            "answer": reference_answer or base_payload["analysis"],
            "reference_answer": reference_answer or base_payload["analysis"],
        }

    def _resolve_exam_topic(self, record: KnowledgeRecord, *, requested_topic: str | None) -> str:
        target = self._normalize_topic_name(requested_topic)
        primary_topic = self._primary_topic(record)
        if target and target != "综合" and primary_topic == target:
            return target
        return primary_topic or target or "综合"

    def _normalize_exam_tags(self, record: KnowledgeRecord, *, requested_topic: str | None) -> list[str]:
        resolved_topic = self._resolve_exam_topic(record, requested_topic=requested_topic)
        tags: list[str] = []
        if resolved_topic:
            tags.append(resolved_topic)
        question_type = self._question_type(record)
        if question_type not in tags:
            tags.append(question_type)
        for tag in record.tags:
            value = str(tag or "").strip()
            if not value:
                continue
            normalized_topic = self._normalize_topic_name(value)
            if normalized_topic in CANONICAL_TOPICS and normalized_topic != resolved_topic:
                continue
            if value not in tags:
                tags.append(value)
        return tags

    def _normalize_options(self, raw_options: object) -> dict[str, str]:
        if not isinstance(raw_options, dict):
            return {}
        normalized: dict[str, str] = {}
        for label in EXAM_OPTION_LABELS:
            value = str(raw_options.get(label) or "").strip()
            if value:
                normalized[label] = value
        return normalized

    def _options_are_consistent(self, options: dict[str, str], answer: str) -> bool:
        if set(options) != set(EXAM_OPTION_LABELS):
            return False
        if answer not in options:
            return False
        normalized_values = {self._comparison_key(text) for text in options.values() if self._comparison_key(text)}
        return len(normalized_values) == len(EXAM_OPTION_LABELS)

    def _extract_correct_option_text(self, record: KnowledgeRecord) -> str:
        options = self._normalize_options(record.metadata.get("options") or {})
        answer = str(record.metadata.get("answer") or "").upper().strip()
        if answer in options:
            return self._clean_option_text(options[answer])
        return ""

    def _clean_option_text(self, text: str) -> str:
        normalized = str(text or "").strip()
        if not normalized:
            return ""
        if len(normalized) <= 160:
            return normalized
        for marker in ("。", "；", ";"):
            if marker in normalized:
                candidate = normalized.split(marker, maxsplit=1)[0].strip(" ，,；;：:")
                if 12 <= len(candidate) <= 160:
                    return candidate
        return truncate_text(normalized, 160)

    def _comparison_key(self, text: str) -> str:
        normalized = clean_text(str(text or "")).lower()
        return COMPARISON_NORMALIZE_RE.sub("", normalized)

    def _stable_seed(self, seed_text: str) -> int:
        digest = blake2b(str(seed_text).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)