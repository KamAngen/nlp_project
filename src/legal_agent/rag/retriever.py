from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from legal_agent.config import AppConfig
from legal_agent.data.admin_divisions import ResolvedLocation, compare_region_relation, extract_query_regions, query_likely_needs_local_law, resolve_location_text
from legal_agent.rag.embeddings import SentenceTransformerEmbedder
from legal_agent.utils.io import read_json, read_jsonl
from legal_agent.utils.text import effect_rank, generate_title_aliases, normalize_title, simple_tokenize, truncate_text


AUTHORITY_HINT_TOKENS = ("依据", "是否有效", "冲突", "上位法", "适用", "是否合法")


@dataclass(slots=True)
class RetrievalHit:
    chunk_id: str
    score: float
    document_title: str
    article_heading: str | None
    effect_level: str
    jurisdiction_type: str
    jurisdiction_scope: str
    region_name: str | None
    region_path_codes: list[str]
    region_path_names: list[str]
    text: str
    source_path: str


@dataclass(slots=True)
class QueryContext:
    regions: list[Any]
    has_explicit_region: bool
    likely_local_question: bool
    authority_boost: bool
    explicit_region_level: str | None
    location_resolution: ResolvedLocation | None


@dataclass(slots=True)
class DocumentHit:
    document_id: str
    score: float
    normalized_title: str
    effect_level: str
    jurisdiction_type: str
    jurisdiction_scope: str
    jurisdiction_rank: int
    region_path_codes: list[str]
    chunk_positions: list[int]


class HybridLegalRetriever:
    def __init__(self, config: AppConfig, *, device: str = "cpu") -> None:
        self.config = config
        self.device = device
        self.embedder = SentenceTransformerEmbedder(config.models.embedding_model, device=device, normalize=True)
        self.metadata = read_json(config.rag_dir / "metadata.json")
        self.dense_backend = self.metadata.get("dense_backend", "numpy")
        self.dense_embeddings = np.load(config.rag_dir / "dense_embeddings.npy", mmap_mode="r")
        with (config.rag_dir / "bm25.pkl").open("rb") as handle:
            self.bm25 = pickle.load(handle)
        self.document_shortlist_enabled = bool(
            self.config.retrieval.use_document_shortlist
            and self.metadata.get("document_shortlist_enabled", False)
            and (config.rag_dir / "document_records.json").exists()
            and (config.rag_dir / "document_embeddings.npy").exists()
            and (config.rag_dir / "document_bm25.pkl").exists()
        )
        self.doc_records = read_json(config.rag_dir / "document_records.json") if self.document_shortlist_enabled else []
        self.doc_dense_embeddings = (
            np.load(config.rag_dir / "document_embeddings.npy", mmap_mode="r")
            if self.document_shortlist_enabled else None
        )
        if self.document_shortlist_enabled:
            with (config.rag_dir / "document_bm25.pkl").open("rb") as handle:
                self.doc_bm25 = pickle.load(handle)
        self.chunks = read_jsonl(config.rag_dir / "chunks.jsonl")
        self.doc_to_chunks = read_json(config.rag_dir / "doc_to_chunks.json")
        self.chunk_to_doc = read_json(config.rag_dir / "chunk_to_doc.json")
        self.manifest = read_jsonl(config.manifest_path)
        self.manifest_by_title: dict[str, dict[str, Any]] = {}
        self.manifest_alias_to_title: dict[str, str] = {}
        for item in self.manifest:
            normalized_title = normalize_title(item["title"])
            self.manifest_by_title[normalized_title] = item
            for alias in generate_title_aliases(item["title"]):
                self.manifest_alias_to_title.setdefault(alias, normalized_title)
        self.chunk_index = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        self.chunk_positions = {chunk["chunk_id"]: index for index, chunk in enumerate(self.chunks)}
        self.doc_first_chunk: dict[str, str] = {}
        for normalized_title, chunk_ids in self.doc_to_chunks.items():
            if chunk_ids:
                self.doc_first_chunk[normalized_title] = chunk_ids[0]

    def inspect_query(self, query: str) -> QueryContext:
        regions = extract_query_regions(query)
        location_resolution = resolve_location_text(query)
        explicit_region_level = None
        if location_resolution is not None and location_resolution.explicit_level:
            explicit_region_level = location_resolution.explicit_level
        elif regions:
            explicit_region_level = max(regions, key=lambda item: len(item.path_codes)).level
        return QueryContext(
            regions=regions,
            has_explicit_region=bool(regions),
            likely_local_question=query_likely_needs_local_law(query),
            authority_boost=self._query_requires_authority_boost(query),
            explicit_region_level=explicit_region_level,
            location_resolution=location_resolution,
        )

    def _query_requires_authority_boost(self, query: str) -> bool:
        return any(token in query for token in AUTHORITY_HINT_TOKENS)

    def _encode_query(self, query: str) -> np.ndarray:
        return self.embedder.encode([query], batch_size=1)[0]

    def _dense_search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        *,
        candidate_indices: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if candidate_indices is None:
            scores = self.dense_embeddings @ query_embedding
            indices = np.argsort(scores)[::-1][:top_k]
            return scores[indices], indices

        if not candidate_indices:
            return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
        dense_scores = np.asarray(self.dense_embeddings[candidate_indices] @ query_embedding, dtype=np.float32)
        order = np.argsort(dense_scores)[::-1][:top_k]
        selected = np.asarray([candidate_indices[position] for position in order], dtype=np.int64)
        return dense_scores[order], selected

    def _score_document_hit(self, record: dict[str, Any], score: float, query: str, query_context: QueryContext) -> float:
        adjusted = float(score)
        if query_context.authority_boost:
            adjusted += self.config.retrieval.hierarchy_boost * float(record.get("effect_rank") or 0)
        if query_context.has_explicit_region:
            relation = compare_region_relation(query_context.regions, list(record.get("region_path_codes") or []))
            if relation == "exact":
                adjusted += self.config.retrieval.explicit_region_boost
            elif relation == "ancestor":
                adjusted += self.config.retrieval.ancestor_region_boost
            elif relation == "descendant":
                adjusted += self.config.retrieval.descendant_region_boost
            elif record.get("jurisdiction_type") == "local":
                adjusted *= self.config.retrieval.unrelated_local_penalty
        elif record.get("jurisdiction_type") == "local":
            local_penalty = self.config.retrieval.local_without_region_penalty
            if query_context.likely_local_question:
                local_penalty = min(0.82, local_penalty + 0.18)
            adjusted *= local_penalty

        normalized_query = normalize_title(query)
        if record.get("normalized_title") and record["normalized_title"] in normalized_query:
            adjusted += 0.18
        return adjusted

    def _shortlist_documents(
        self,
        query: str,
        query_embedding: np.ndarray,
        query_context: QueryContext,
        *,
        effect_level: str | None = None,
    ) -> list[DocumentHit]:
        if not self.document_shortlist_enabled or self.doc_dense_embeddings is None:
            return []

        dense_scores = self.doc_dense_embeddings @ query_embedding
        dense_indices = np.argsort(dense_scores)[::-1][: self.config.retrieval.document_dense_top_k]
        bm25_scores = self.doc_bm25.get_scores(simple_tokenize(query))
        bm25_indices = np.argsort(bm25_scores)[::-1][: self.config.retrieval.document_bm25_top_k]

        fused_scores: dict[int, float] = defaultdict(float)
        for rank, index in enumerate(dense_indices, start=1):
            fused_scores[int(index)] += 1.0 / (self.config.retrieval.rrf_k + rank)
        for rank, index in enumerate(bm25_indices, start=1):
            fused_scores[int(index)] += 1.0 / (self.config.retrieval.rrf_k + rank)

        hits: list[DocumentHit] = []
        for index, score in fused_scores.items():
            record = self.doc_records[index]
            adjusted = self._score_document_hit(record, score, query, query_context)
            if effect_level and record["effect_level"] != effect_level:
                adjusted *= 0.5
            hits.append(
                DocumentHit(
                    document_id=record["document_id"],
                    score=adjusted,
                    normalized_title=record["normalized_title"],
                    effect_level=record["effect_level"],
                    jurisdiction_type=str(record.get("jurisdiction_type") or "national"),
                    jurisdiction_scope=str(record.get("jurisdiction_scope") or "national"),
                    jurisdiction_rank=int(record.get("jurisdiction_rank") or 0),
                    region_path_codes=list(record.get("region_path_codes") or []),
                    chunk_positions=[int(position) for position in record.get("chunk_positions", [])],
                )
            )

        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[: self.config.retrieval.document_shortlist_k]

    def _candidate_chunk_indices(
        self,
        query: str,
        query_embedding: np.ndarray,
        query_context: QueryContext,
        *,
        effect_level: str | None = None,
    ) -> tuple[list[int] | None, dict[int, float]]:
        doc_hits = self._shortlist_documents(query, query_embedding, query_context, effect_level=effect_level)
        if not doc_hits:
            return None, {}

        candidate_indices: list[int] = []
        doc_score_by_chunk: dict[int, float] = {}
        for doc_hit in doc_hits:
            for position in doc_hit.chunk_positions:
                if position in doc_score_by_chunk:
                    doc_score_by_chunk[position] = max(doc_score_by_chunk[position], doc_hit.score)
                    continue
                candidate_indices.append(position)
                doc_score_by_chunk[position] = doc_hit.score
                if len(candidate_indices) >= self.config.retrieval.max_candidate_chunks:
                    break
            if len(candidate_indices) >= self.config.retrieval.max_candidate_chunks:
                break
        return candidate_indices, doc_score_by_chunk

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        effect_level: str | None = None,
        query_context: QueryContext | None = None,
    ) -> list[RetrievalHit]:
        top_k = top_k or self.config.retrieval.final_top_k
        query_context = query_context or self.inspect_query(query)
        query_embedding = self._encode_query(query)
        candidate_indices, doc_score_by_chunk = self._candidate_chunk_indices(
            query,
            query_embedding,
            query_context,
            effect_level=effect_level,
        )
        dense_scores, dense_indices = self._dense_search(
            query_embedding,
            self.config.retrieval.dense_top_k,
            candidate_indices=candidate_indices,
        )
        query_tokens = simple_tokenize(query)
        if candidate_indices is None:
            bm25_scores = self.bm25.get_scores(query_tokens)
            bm25_indices = np.argsort(bm25_scores)[::-1][: self.config.retrieval.bm25_top_k]
        else:
            batch_scores = self.bm25.get_batch_scores(query_tokens, candidate_indices)
            batch_order = np.argsort(batch_scores)[::-1][: self.config.retrieval.bm25_top_k]
            bm25_indices = np.asarray([candidate_indices[position] for position in batch_order], dtype=np.int64)

        fused_scores: dict[int, float] = defaultdict(float)
        for rank, index in enumerate(dense_indices, start=1):
            if index < 0:
                continue
            fused_scores[int(index)] += 1.0 / (self.config.retrieval.rrf_k + rank)
        for rank, index in enumerate(bm25_indices, start=1):
            fused_scores[int(index)] += 1.0 / (self.config.retrieval.rrf_k + rank)
        for index, doc_score in doc_score_by_chunk.items():
            fused_scores[int(index)] += doc_score * self.config.retrieval.document_score_boost

        normalized_query = normalize_title(query)
        authority_boost = query_context.authority_boost
        local_penalty = self.config.retrieval.local_without_region_penalty
        if query_context.likely_local_question and not query_context.has_explicit_region:
            local_penalty = min(0.82, local_penalty + 0.18)
        for index, score in list(fused_scores.items()):
            chunk = self.chunks[index]
            if effect_level and chunk["effect_level"] != effect_level:
                fused_scores[index] = score * 0.5
                continue
            if chunk["normalized_title"] in normalized_query or chunk["document_title"] in query:
                fused_scores[index] += 0.15
            if authority_boost:
                fused_scores[index] += self.config.retrieval.hierarchy_boost * float(chunk["effect_rank"])
            if query_context.has_explicit_region:
                relation = compare_region_relation(query_context.regions, list(chunk.get("region_path_codes") or []))
                if relation == "exact":
                    fused_scores[index] += self.config.retrieval.explicit_region_boost
                elif relation == "ancestor":
                    fused_scores[index] += self.config.retrieval.ancestor_region_boost
                    if authority_boost:
                        fused_scores[index] += 0.03 * float(chunk.get("jurisdiction_rank") or 0)
                elif relation == "descendant":
                    fused_scores[index] += self.config.retrieval.descendant_region_boost
                elif chunk.get("jurisdiction_type") == "local":
                    fused_scores[index] *= self.config.retrieval.unrelated_local_penalty
            elif chunk.get("jurisdiction_type") == "local":
                fused_scores[index] *= local_penalty
            if self.config.retrieval.use_graph_expansion:
                for ref_title in chunk.get("cross_references", [])[:2]:
                    ref_chunk_id = self.doc_first_chunk.get(normalize_title(ref_title))
                    if ref_chunk_id is None:
                        continue
                    ref_index = self.chunk_positions.get(ref_chunk_id)
                    if ref_index is None:
                        continue
                    fused_scores[ref_index] += score * self.config.retrieval.graph_boost

        selected = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        hits: list[RetrievalHit] = []
        for index, score in selected:
            chunk = self.chunks[index]
            hits.append(
                RetrievalHit(
                    chunk_id=chunk["chunk_id"],
                    score=float(score),
                    document_title=chunk["document_title"],
                    article_heading=chunk.get("article_heading"),
                    effect_level=chunk["effect_level"],
                    jurisdiction_type=str(chunk.get("jurisdiction_type") or "national"),
                    jurisdiction_scope=str(chunk.get("jurisdiction_scope") or "national"),
                    region_name=chunk.get("region_name"),
                    region_path_codes=list(chunk.get("region_path_codes") or []),
                    region_path_names=list(chunk.get("region_path_names") or []),
                    text=chunk["text"],
                    source_path=chunk["source_path"],
                )
            )
        return hits

    def lookup_statute(self, title: str) -> dict[str, Any] | None:
        normalized = normalize_title(title)
        canonical_title = self.manifest_alias_to_title.get(normalized)
        if canonical_title is not None:
            record = self.manifest_by_title[canonical_title]
            chunk_ids = self.doc_to_chunks.get(canonical_title, [])
            preview = [self.chunk_index[chunk_id] for chunk_id in chunk_ids[:3] if chunk_id in self.chunk_index]
            return {
                "title": record["title"],
                "effect_level": record["effect_level"],
                "promulgation_date": record["promulgation_date"],
                "effective_date": record["effective_date"],
                "status": record["status"],
                "jurisdiction_type": record.get("jurisdiction_type", "national"),
                "jurisdiction_scope": record.get("jurisdiction_scope", "national"),
                "region_name": record.get("region_name"),
                "region_path_names": record.get("region_path_names", []),
                "preview_articles": [chunk.get("article_heading") for chunk in preview],
                "preview_texts": [truncate_text(chunk.get("text", ""), self.config.retrieval.observation_max_chars) for chunk in preview],
            }
        return None

    def resolve_hierarchy(self, title_or_category: str) -> dict[str, Any]:
        normalized = normalize_title(title_or_category)
        record = self.manifest_by_title.get(normalized)
        if record is not None:
            level = record["effect_level"]
        else:
            level = title_or_category
        return {
            "input": title_or_category,
            "effect_level": level,
            "effect_rank": effect_rank(str(level)),
            "hierarchy": ["宪法", "法律", "法规", "司法解释"],
            "jurisdiction_type": record.get("jurisdiction_type", "national") if record is not None else "national",
            "jurisdiction_scope": record.get("jurisdiction_scope", "national") if record is not None else "national",
            "region_path_names": record.get("region_path_names", []) if record is not None else [],
        }
