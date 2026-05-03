from __future__ import annotations

from pathlib import Path
import random
import json
from typing import Iterable, List, Dict, Any, Set

from legal_agent.utils.text import simple_tokenize, truncate_text
from rag_engine.loaders import load_case_bank, load_common_knowledge, load_question_bank
from rag_engine.schema import KnowledgeHit, KnowledgeRecord
from rag_engine.embedding import EmbeddingEngine
from rag_engine.indexer import VectorIndex
from rag_engine.reranker import RerankerEngine
from rag_engine.graph import LegalKnowledgeGraph


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
            
        self.indices: dict[str, VectorIndex] = {}
        self.embedding_engine: EmbeddingEngine | None = None
        self.reranker_engine: RerankerEngine | None = None
        self.graph: LegalKnowledgeGraph | None = None

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
        from collections import defaultdict
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

    def summary(self) -> dict[str, object]:
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
        preferred_tags: list[str] | None = None,
        exam_type: str | None = None,
        avoid_question_ids: list[str] | None = None,
        user_mastery_level: float = 0.5,
    ) -> list[KnowledgeRecord]:
        rng = random.Random()
        effective_exam_type = exam_type or "综合练习"
        avoid_ids = set(avoid_question_ids or [])
        mastered_tags = [] 
        prioritized_ids = set() 

        # 1. Hybrid Retrieval for Candidate Pool (Lexical + Semantic RRF Fusion)
        if topic and topic != "综合":
            K = 60  # RRF constant

            # 1a. Lexical search
            lexical_hits = self._search_lexical_records(
                topic, self.question_bank, top_k=50
            )
            # 1b. Semantic search
            semantic_hits = self._search_embedding_records(
                topic, self.question_bank, source_type="question_bank", top_k=50
            )

            # 1c. RRF fusion: merge by record_id
            rrf_scores: dict[str, float] = {}
            for rank, h in enumerate(lexical_hits):
                rrf_scores[h.record_id] = rrf_scores.get(h.record_id, 0.0) + 1.0 / (K + rank + 1)
            for rank, h in enumerate(semantic_hits):
                rrf_scores[h.record_id] = rrf_scores.get(h.record_id, 0.0) + 1.0 / (K + rank + 1)

            # 1d. Build candidate pool sorted by RRF score (top 60)
            sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:60]
            candidate_pool = []
            for rid in sorted_ids:
                rec = self.get_record_by_id(rid)
                if rec:
                    rec.metadata["_semantic_score"] = rrf_scores[rid] * 30  # scale to ~0-1
                    candidate_pool.append(rec)

            # Fallback only if the specific topic has zero results
            if not candidate_pool:
                candidate_pool = list(self.question_bank)
        else:
            candidate_pool = list(self.question_bank)

        # 2. Filter out avoid_ids
        candidate_pool = [r for r in candidate_pool if r.record_id not in avoid_ids]

        # 3. Weighted Sampling from the pool
        # This will respect both the topic (via the pool) and the difficulty (via weighting)
        selected = self._weighted_sample_without_replacement(
            candidate_pool,
            min(len(candidate_pool), question_count),
            rng,
            weight_fn=lambda record: self._question_weight(
                record,
                topic=topic,
                preferred_tags=preferred_tags or [],
                exam_type=effective_exam_type,
                avoid_ids=avoid_ids,
                prioritized_ids=prioritized_ids,
                strong_tags=mastered_tags,
                user_mastery_level=user_mastery_level,
            ),
        )

        rng.shuffle(selected)
        return selected

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
        topic_match = bool(topic and topic != "综合" and self._topic_matches(record, topic))
        difficulty = str(record.difficulty or "medium")
        
        # Curriculum Sampling Logic
        multiplier = 1.0
        if user_mastery_level < 0.35:
            # Beginner: Strongly favor easy > medium > hard
            multiplier = {"easy": 100.0, "medium": 30.0, "hard": 1.0}.get(difficulty, 1.0)
        elif user_mastery_level < 0.75:
            # Intermediate: favor medium
            multiplier = {"easy": 5.0, "medium": 20.0, "hard": 5.0}.get(difficulty, 1.0)
        else:
            # Advanced: Strongly favor hard
            multiplier = {"easy": 0.1, "medium": 2.0, "hard": 50.0}.get(difficulty, 1.0)

        weight *= multiplier

        # Semantic Score Bonus (Priority over Keyword match)
        semantic_score = record.metadata.get("_semantic_score", 0.0)
        if semantic_score > 0.5: # Only apply significant boost for decent matches
            weight *= (1.0 + (semantic_score - 0.5) * 5.0)
        
        # Legacy Keyword match (Still useful as a secondary signal)
        topic_match = bool(topic and topic != "综合" and self._topic_matches(record, topic))
        if topic_match:
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
            # Fallback to lexical if index missing
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
        target = str(topic or "").strip().lower()
        if not target:
            return False
            
        # 1. Check Tags (Exact match)
        if any(target in str(tag).lower() for tag in record.tags):
            return True
            
        # 2. Check Title and Content (Substring match for robustness in Chinese)
        if target in record.title.lower() or target in record.content.lower():
            return True
            
        return False

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