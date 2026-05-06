from __future__ import annotations

from collections import defaultdict
from hashlib import blake2b
from pathlib import Path
import random
import json
import re
from typing import Iterable, List, Dict, Any, Set

from legal_agent.utils.text import clean_text, simple_tokenize, truncate_text
from rag_engine.loaders import load_case_bank, load_common_knowledge, load_question_bank
from rag_engine.schema import KnowledgeHit, KnowledgeRecord
from rag_engine.embedding import EmbeddingEngine
from rag_engine.indexer import VectorIndex
from rag_engine.reranker import RerankerEngine
from rag_engine.graph import LegalKnowledgeGraph


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
        self.legacy_statute = None
        if use_legacy_statute_rag and legacy_config_path is not None:
            self.legacy_statute = LegacyStatuteRetrieverAdapter(legacy_config_path, device=legacy_device)
            
        self.indices: dict[str, VectorIndex] = {}
        self.embedding_engine: EmbeddingEngine | None = None
        self.reranker_engine: RerankerEngine | None = None
        self.graph: LegalKnowledgeGraph | None = None
        self._build_question_indexes()

    def load_indices(self, index_root_path: str | Path, model_path: str | Path, device: str | None = None):
        self.embedding_engine = EmbeddingEngine(model_path, device=device)
        index_root_path = Path(index_root_path)
        
        for source in ["question_bank", "case_bank", "common_knowledge"]:
            index_path = index_root_path / source
            if index_path.exists():
                self.indices[source] = VectorIndex.load(index_path, self.embedding_engine)
            else:
                print(f"Index for {source} missing, rebuilding from memory...")
                self.indices[source] = VectorIndex(source, self.embedding_engine)
                records = []
                if source == "question_bank": records = list(self.question_bank.values()) if isinstance(self.question_bank, dict) else self.question_bank
                elif source == "case_bank": records = list(self.case_bank.values()) if isinstance(self.case_bank, dict) else self.case_bank
                elif source == "common_knowledge": records = list(self.common_knowledge.values()) if isinstance(self.common_knowledge, dict) else self.common_knowledge
                
                if records:
                    self.indices[source].add_records(records)
                    self.indices[source].save(index_path)

    def load_reranker(self, model_path: str | Path, device: str | None = None):
        self.reranker_engine = RerankerEngine(model_path, device=device)

    def load_graph(self, graph_path: str | Path):
        path = Path(graph_path)
        from rag_engine.graph import LegalKnowledgeGraph
        if path.exists():
            self.graph = LegalKnowledgeGraph.load(path)
        else:
            print(f"Graph file {path} missing, rebuilding from memory...")
            self.graph = LegalKnowledgeGraph()
            # Rebuild from existing cases
            cases_to_add = []
            records = list(self.case_bank.values()) if isinstance(self.case_bank, dict) else self.case_bank
            for rec in records:
                cases_to_add.append({
                    "case_id": rec.record_id,
                    "title": rec.title,
                    "statutes": rec.metadata.get("statutes") or rec.tags or []
                })
            if cases_to_add:
                self.graph.build_from_cases(cases_to_add)
                self.graph.save(path)

    def get_record_by_id(self, record_id: str) -> KnowledgeRecord | None:
        """
        Find a record across all banks by its ID.
        """
        for bank in [self.case_bank, self.question_bank, self.common_knowledge]:
            if isinstance(bank, dict):
                if record_id in bank:
                    return bank[record_id]
            else:
                # Linear search for lists
                for rec in bank:
                    if rec.record_id == record_id:
                        return rec
        
        # Robust fallback: check if it's a short ID suffix
        short_id = record_id.split("-")[-1]
        for bank in [self.case_bank, self.question_bank, self.common_knowledge]:
            items = bank.values() if isinstance(bank, dict) else bank
            for rec in items:
                if short_id in rec.record_id:
                    return rec
        return None

    def generate_id(self, source_type: str, manual_id: str | None = None) -> str:
        if manual_id: return manual_id
        import time, random
        prefix = source_type.split("_")[0]
        return f"auto-{prefix}-{int(time.time())}-{random.getrandbits(24):06x}"

    def update_record(self, record: KnowledgeRecord, persist: bool = True):
        """
        Hot update a record in memory and optionally on disk.
        """
        # 1. Update in-memory collections (Dict or List support)
        if record.source_type == "question_bank":
            if isinstance(self.question_bank, dict): self.question_bank[record.record_id] = record
            else: self.question_bank.append(record)
        elif record.source_type == "case_bank":
            if isinstance(self.case_bank, dict): self.case_bank[record.record_id] = record
            else: self.case_bank.append(record)
        elif record.source_type == "common_knowledge":
            if isinstance(self.common_knowledge, dict): self.common_knowledge[record.record_id] = record
            else: self.common_knowledge.append(record)
            
        # 2. Update Vector Indices
        if record.source_type in self.indices:
            self.indices[record.source_type].update_record(record)
            if persist:
                index_path = Path(f"data/indices/{record.source_type}")
                self.indices[record.source_type].save(index_path)
                
        # 3. Update Graph if it's a case
        if record.source_type == "case_bank" and self.graph:
            # We need the raw dict for the graph builder
            raw_record = {
                "case_id": record.record_id,
                "title": record.title,
                "statutes": record.metadata.get("statutes", [])
            }
            self.graph.build_from_cases([raw_record])
            if persist:
                self.graph.save("data/indices/legal_graph.json")

        if record.source_type == "question_bank":
            self._build_question_indexes()

        # 4. Persist to JSONL
        if persist:
            bank_path = {
                "question_bank": "data/legal_study_agent/question_bank.jsonl",
                "case_bank": "data/legal_study_agent/case_bank.jsonl",
                "common_knowledge": "data/legal_study_agent/common_knowledge.jsonl"
            }.get(record.source_type)
            
            if bank_path:
                # Map back to legacy field names for persistence consistency
                legacy_dict = {}
                if record.source_type == "case_bank":
                    legacy_dict = {
                        "case_id": record.record_id,
                        "title": record.title,
                        "facts": record.content,
                        "statutes": record.tags,
                        "tags": record.metadata.get("tags", []),
                        **{k: v for k, v in record.metadata.items() if k != "statutes"}
                    }
                elif record.source_type == "question_bank":
                    legacy_dict = {
                        "question_id": record.record_id,
                        "title": record.title,
                        "content": record.content,
                        "difficulty": record.metadata.get("difficulty", "medium"),
                        "tags": record.tags,
                        **{k: v for k, v in record.metadata.items() if k != "difficulty"}
                    }
                else:
                    legacy_dict = {
                        "id": record.record_id,
                        "title": record.title,
                        "content": record.content,
                        **record.metadata
                    }
                
                with open(bank_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(legacy_dict, ensure_ascii=False) + "\n")

    def update_records_batch(self, records: List[KnowledgeRecord], persist: bool = True):
        grouped = defaultdict(list)
        for r in records:
            grouped[r.source_type].append(r)
            
        for source_type, recs in grouped.items():
            # 0. Deduplicate: Skip records that already exist
            bank = getattr(self, source_type, None)
            if bank is not None:
                new_recs = []
                for r in recs:
                    exists = False
                    if isinstance(bank, dict) and r.record_id in bank:
                        exists = True
                    elif isinstance(bank, list) and any(existing.record_id == r.record_id for existing in bank):
                        exists = True
                    if not exists:
                        new_recs.append(r)
                recs = new_recs
                
            if not recs:
                continue # Skip if all records were duplicates
                
            # 1. Update In-Memory Bank
            if bank is not None:
                for r in recs:
                    if isinstance(bank, dict): bank[r.record_id] = r
                    else: bank.append(r)
                    
            # 2. Update Vector Indices with TRUE Batching
            if source_type in self.indices:
                self.indices[source_type].add_records(recs)
                if persist:
                    self.indices[source_type].save(Path(f"data/indices/{source_type}"))
                    
            # 3. Update Graph
            if source_type == "case_bank" and self.graph:
                raw_cases = [{"case_id": r.record_id, "title": r.title, "statutes": r.metadata.get("statutes", [])} for r in recs]
                self.graph.build_from_cases(raw_cases)
                if persist:
                    self.graph.save("data/indices/legal_graph.json")
                    
            # 4. Persist to JSONL (Full rewrite to ensure no duplicates)
            if persist:
                bank_path = {"question_bank": "data/legal_study_agent/question_bank.jsonl", "case_bank": "data/legal_study_agent/case_bank.jsonl", "common_knowledge": "data/legal_study_agent/common_knowledge.jsonl"}.get(source_type)
                if bank_path:
                    # Get the current clean bank from memory
                    full_bank = getattr(self, source_type, [])
                    with open(bank_path, "w", encoding="utf-8") as f: # Use "w" for full rewrite
                        for record in full_bank:
                            legacy_dict = {}
                            if source_type == "case_bank":
                                legacy_dict = {"case_id": record.record_id, "title": record.title, "facts": record.content, "statutes": record.tags, "tags": record.metadata.get("tags", []), **{k: v for k, v in record.metadata.items() if k != "statutes"}}
                            elif source_type == "question_bank":
                                legacy_dict = {"question_id": record.record_id, "title": record.title, "content": record.content, "difficulty": record.metadata.get("difficulty", "medium"), "tags": record.tags, **{k: v for k, v in record.metadata.items() if k != "difficulty"}}
                            else:
                                legacy_dict = {"id": record.record_id, "title": record.title, "content": record.content, **record.metadata}
                            f.write(json.dumps(legacy_dict, ensure_ascii=False) + "\n")

            if source_type == "question_bank":
                self._build_question_indexes()

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
            "vector_indices_loaded": list(self.indices.keys()),
            "reranker_loaded": bool(self.reranker_engine is not None),
            "graph_loaded": bool(self.graph is not None),
        }

    def search(self, query: str, *, sources: Iterable[str] | None = None, top_k: int = 5, mode: str = "hybrid", rerank: bool = False) -> list[KnowledgeHit]:
        selected_sources = set(sources or ["statute", "question_bank", "case_bank", "common_knowledge"])
        recall_k = top_k * 4 if rerank else top_k * 2
        
        all_hits: list[KnowledgeHit] = []
        
        # Phase 1: Retrieval
        if mode in ["lexical", "hybrid"]:
            lexical_hits = []
            if "question_bank" in selected_sources:
                lexical_hits.extend(self._search_lexical_records(query, self.question_bank, top_k=recall_k))
            if "case_bank" in selected_sources:
                lexical_hits.extend(self._search_lexical_records(query, self.case_bank, top_k=recall_k))
            if "common_knowledge" in selected_sources:
                lexical_hits.extend(self._search_lexical_records(query, self.common_knowledge, top_k=recall_k))
            
            if mode == "lexical":
                all_hits = lexical_hits
            else:
                # Store for hybrid merging
                all_hits.extend(lexical_hits)

        if mode in ["embedding", "hybrid"]:
            embedding_hits = []
            if "question_bank" in selected_sources:
                embedding_hits.extend(self._search_embedding_records(query, self.question_bank, source_type="question_bank", top_k=recall_k))
            if "case_bank" in selected_sources:
                embedding_hits.extend(self._search_embedding_records(query, self.case_bank, source_type="case_bank", top_k=recall_k))
            
            if mode == "embedding":
                all_hits = embedding_hits
            else:
                # Merge logic for Hybrid
                # We use a simple score-based merge or RRF-like logic
                # For now, let's just combine and deduplicate, giving lexical a boost
                seen_ids = {h.record_id: h for h in all_hits}
                for eh in embedding_hits:
                    if eh.record_id in seen_ids:
                        # If both hit, boost the score
                        seen_ids[eh.record_id].score = max(seen_ids[eh.record_id].score, eh.score) * 1.2
                    else:
                        all_hits.append(eh)
        
        if "statute" in selected_sources and self.legacy_statute is not None:
            all_hits.extend(self.legacy_statute.search(query, top_k=recall_k))
            
        # Initial sort
        all_hits = sorted(all_hits, key=lambda hit: hit.score, reverse=True)
        
        # Phase 2: Reranking (Cross-Encoder)
        if rerank and self.reranker_engine and all_hits:
            candidate_docs = [h.excerpt for h in all_hits[:recall_k]]
            rerank_scores = self.reranker_engine.rerank(query, candidate_docs)
            
            for i, score in enumerate(rerank_scores):
                all_hits[i].score = score
            
            all_hits = sorted(all_hits, key=lambda hit: hit.score, reverse=True)
            
        return all_hits[:top_k]

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
        user_mastery_level: float = 0.5,
    ) -> list[KnowledgeRecord]:
        preferred_tags = [str(tag) for tag in preferred_tags or [] if str(tag).strip()]
        avoid_ids = {str(record_id) for record_id in avoid_question_ids or [] if str(record_id).strip()}
        prioritized_ids = {str(record_id) for record_id in prioritized_question_ids or [] if str(record_id).strip()}
        mastered_tags = [str(tag) for tag in strong_tags or [] if str(tag).strip()]
        effective_exam_type = str(exam_type or "综合练习").strip() or "综合练习"
        rng = random.Random(random_seed)
        normalized_topic = self._normalize_topic_name(topic)
        normalized_question_types = self._normalize_question_types(question_types)
        strict_topic = bool(
            normalized_topic
            and normalized_topic != "综合"
            and effective_exam_type in {"章节练习", "薄弱点强化", "真题模拟"}
        )

        question_pool = self._question_pool_from_index(normalized_question_types)
        if normalized_topic and normalized_topic != "综合":
            topic_hits = self._question_pool_from_index(
                normalized_question_types,
                topic=normalized_topic,
                strict=strict_topic,
            )
            if strict_topic:
                pool = topic_hits
            else:
                pool = topic_hits or question_pool
        else:
            pool = question_pool

        if strict_topic and not pool:
            return []
        if not pool:
            return []

        candidate_pool = list(pool)
        if normalized_topic and normalized_topic != "综合":
            rrf_scores: dict[str, float] = {}
            lexical_hits = self._search_lexical_records(normalized_topic, self.question_bank, top_k=80)
            semantic_hits = self._search_embedding_records(normalized_topic, self.question_bank, source_type="question_bank", top_k=80)
            allowed_ids = {record.record_id for record in pool}
            fusion_k = 60
            for rank, hit in enumerate(lexical_hits):
                if hit.record_id not in allowed_ids:
                    continue
                rrf_scores[hit.record_id] = rrf_scores.get(hit.record_id, 0.0) + 1.0 / (fusion_k + rank + 1)
            for rank, hit in enumerate(semantic_hits):
                if hit.record_id not in allowed_ids:
                    continue
                rrf_scores[hit.record_id] = rrf_scores.get(hit.record_id, 0.0) + 1.0 / (fusion_k + rank + 1)
            if rrf_scores:
                ranked_ids = sorted(rrf_scores, key=lambda item: rrf_scores[item], reverse=True)
                candidate_pool = []
                seen_ids: set[str] = set()
                for record_id in ranked_ids:
                    record = self.get_record_by_id(record_id)
                    if record is None or record.record_id not in allowed_ids or record.record_id in seen_ids:
                        continue
                    record.metadata["_semantic_score"] = max(float(record.metadata.get("_semantic_score") or 0.0), rrf_scores[record_id] * 30)
                    candidate_pool.append(record)
                    seen_ids.add(record.record_id)
                candidate_pool.extend(record for record in pool if record.record_id not in seen_ids)

        priority_pool = [
            record
            for record in candidate_pool
            if record.record_id in prioritized_ids
            and (not normalized_topic or normalized_topic == "综合" or self._topic_matches(record, normalized_topic))
        ]
        if not priority_pool and prioritized_ids:
            priority_pool = [record for record in question_pool if record.record_id in prioritized_ids]

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
                        topic=normalized_topic or topic,
                        preferred_tags=preferred_tags,
                        exam_type=effective_exam_type,
                        avoid_ids=set(),
                        prioritized_ids=prioritized_ids,
                        strong_tags=mastered_tags,
                        user_mastery_level=user_mastery_level,
                    ) + 2.4,
                )
            )

        selected_ids = {record.record_id for record in selected}
        candidate_pool = [record for record in candidate_pool if record.record_id not in selected_ids]

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
                        topic=normalized_topic or topic,
                        preferred_tags=preferred_tags,
                        exam_type=effective_exam_type,
                        avoid_ids=avoid_ids,
                        prioritized_ids=prioritized_ids,
                        strong_tags=mastered_tags,
                        user_mastery_level=user_mastery_level,
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
                        topic=normalized_topic or topic,
                        preferred_tags=preferred_tags,
                        exam_type=effective_exam_type,
                        avoid_ids=set(),
                        prioritized_ids=prioritized_ids,
                        strong_tags=mastered_tags,
                        user_mastery_level=user_mastery_level,
                    ),
                )
            )

        rng.shuffle(selected)
        return selected[:question_count]

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

            for topic_name in loose_topics:
                loose_topic_index[topic_name].append(record)
                loose_topic_type_index[(topic_name, question_type)].append(record)
            for topic_name in strict_topics:
                strict_topic_index[topic_name].append(record)
                strict_topic_type_index[(topic_name, question_type)].append(record)

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
        user_mastery_level: float = 0.5,
    ) -> float:
        weight = 1.0
        tag_bonus = sum(1 for tag in preferred_tags if self._topic_matches(record, tag))
        strict_topic_quality = self._strict_topic_quality(record, topic or "") if topic and topic != "综合" else 0
        difficulty = str(record.difficulty or "medium")

        multiplier = 1.0
        if user_mastery_level < 0.35:
            multiplier = {"easy": 100.0, "medium": 30.0, "hard": 1.0}.get(difficulty, 1.0)
        elif user_mastery_level < 0.75:
            multiplier = {"easy": 5.0, "medium": 20.0, "hard": 5.0}.get(difficulty, 1.0)
        else:
            multiplier = {"easy": 0.1, "medium": 2.0, "hard": 50.0}.get(difficulty, 1.0)
        weight *= multiplier

        semantic_score = float(record.metadata.get("_semantic_score") or 0.0)
        if semantic_score > 0.5:
            weight *= 1.0 + (semantic_score - 0.5) * 5.0

        if topic and topic != "综合" and self._topic_matches(record, topic):
            weight *= 1.5

        weight += tag_bonus * (1.4 if exam_type == "薄弱点强化" else 0.9)

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

    def _search_lexical_records(self, query: str, records: list[KnowledgeRecord], *, top_k: int, source_type: str | None = None) -> list[KnowledgeHit]:
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

    def _search_embedding_records(self, query: str, records: list[KnowledgeRecord], *, source_type: str, top_k: int) -> list[KnowledgeHit]:
        index = self.indices.get(source_type)
        if not index:
            return self._search_lexical_records(query, records, top_k=top_k)

        results = index.search(query, top_k=top_k)
        hits = []
        for res in results:
            record = res["record"]
            hits.append(
                KnowledgeHit(
                    source_type=record.source_type,
                    record_id=record.record_id,
                    title=record.title,
                    excerpt=truncate_text(record.content, 220),
                    score=float(res["score"]),
                    metadata=record.metadata,
                )
            )
        return hits

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
        tag_topics = {
            self._normalize_topic_name(tag)
            for tag in record.tags
            if self._normalize_topic_name(tag)
        }
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

    def _comparison_key(self, text: str) -> str:
        normalized = clean_text(str(text or "")).lower()
        return COMPARISON_NORMALIZE_RE.sub("", normalized)
