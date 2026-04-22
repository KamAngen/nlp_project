from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from legal_agent.config import AppConfig
from legal_agent.data.corpus_builder import build_law_corpus
from legal_agent.rag.embeddings import SentenceTransformerEmbedder
from legal_agent.rag.graph_builder import build_citation_graph, graph_to_json
from legal_agent.utils.io import ensure_dir, read_jsonl, write_json, write_jsonl
from legal_agent.utils.text import join_non_empty, simple_tokenize


def _build_dense_embeddings(
    embedder: SentenceTransformerEmbedder,
    texts: list[str],
    output_path: Path,
    *,
    write_batch_size: int = 2048,
    encode_batch_size: int = 64,
) -> tuple[int, int]:
    if not texts:
        raise RuntimeError("No retrieval texts found for dense index building.")

    memmap: np.memmap | None = None
    embedding_dim = 0
    for start in range(0, len(texts), write_batch_size):
        end = min(len(texts), start + write_batch_size)
        batch_embeddings = embedder.encode(texts[start:end], batch_size=encode_batch_size)
        if memmap is None:
            embedding_dim = int(batch_embeddings.shape[1])
            memmap = np.lib.format.open_memmap(
                output_path,
                mode="w+",
                dtype="float32",
                shape=(len(texts), embedding_dim),
            )
        memmap[start:end] = batch_embeddings

    if memmap is None:
        raise RuntimeError("Dense embedding construction failed to produce any vectors.")
    memmap.flush()
    return len(texts), embedding_dim


def _build_document_records(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for index, chunk in enumerate(chunks):
        document_id = str(chunk["document_id"])
        record = documents.setdefault(
            document_id,
            {
                "document_id": document_id,
                "normalized_title": chunk["normalized_title"],
                "title": chunk["document_title"],
                "effect_level": chunk["effect_level"],
                "effect_rank": chunk["effect_rank"],
                "jurisdiction_type": chunk.get("jurisdiction_type", "national"),
                "jurisdiction_scope": chunk.get("jurisdiction_scope", "national"),
                "jurisdiction_rank": chunk.get("jurisdiction_rank", 0),
                "region_name": chunk.get("region_name"),
                "region_path_codes": chunk.get("region_path_codes", []),
                "region_path_names": chunk.get("region_path_names", []),
                "chunk_positions": [],
                "sample_texts": [],
            },
        )
        record["chunk_positions"].append(index)
        if len(record["sample_texts"]) < 4 and chunk.get("text"):
            record["sample_texts"].append(chunk["text"])

    doc_records: list[dict[str, Any]] = []
    for record in documents.values():
        record["retrieval_text"] = join_non_empty(
            [
                record["title"],
                record["effect_level"],
                " > ".join(record.get("region_path_names", [])) if record.get("region_path_names") else "全国适用",
                *record["sample_texts"],
            ]
        )
        record.pop("sample_texts", None)
        doc_records.append(record)

    doc_records.sort(key=lambda item: item["document_id"])
    return doc_records


def build_rag_index(config: AppConfig, *, device: str = "cpu") -> dict[str, Any]:
    ensure_dir(config.rag_dir)
    if not config.corpus_path.exists():
        build_law_corpus(config)

    chunks = read_jsonl(config.corpus_path)
    if not chunks:
        raise RuntimeError("No chunks found. Build corpus before building the index.")

    embedder = SentenceTransformerEmbedder(config.models.embedding_model, device=device, normalize=True)
    texts = [chunk["retrieval_text"] for chunk in chunks]
    chunk_count, embedding_dim = _build_dense_embeddings(
        embedder,
        texts,
        config.rag_dir / "dense_embeddings.npy",
    )

    tokenized = [simple_tokenize(text) for text in texts]
    bm25 = BM25Okapi(tokenized)
    with (config.rag_dir / "bm25.pkl").open("wb") as handle:
        pickle.dump(bm25, handle)

    doc_to_chunks, chunk_to_doc, graph = build_citation_graph(chunks)
    write_json(config.rag_dir / "graph.json", graph_to_json(graph))
    write_json(config.rag_dir / "doc_to_chunks.json", doc_to_chunks)
    write_json(config.rag_dir / "chunk_to_doc.json", chunk_to_doc)
    write_jsonl(config.rag_dir / "chunks.jsonl", chunks)

    doc_records = _build_document_records(chunks)
    doc_texts = [record["retrieval_text"] for record in doc_records]
    doc_count, doc_embedding_dim = _build_dense_embeddings(
        embedder,
        doc_texts,
        config.rag_dir / "document_embeddings.npy",
    )
    doc_bm25 = BM25Okapi([simple_tokenize(text) for text in doc_texts])
    with (config.rag_dir / "document_bm25.pkl").open("wb") as handle:
        pickle.dump(doc_bm25, handle)
    write_json(config.rag_dir / "document_records.json", doc_records)

    metadata = {
        "chunk_count": chunk_count,
        "embedding_dim": embedding_dim,
        "document_count": doc_count,
        "document_embedding_dim": doc_embedding_dim,
        "embedding_model": str(config.models.embedding_model),
        "dense_backend": "numpy",
        "document_shortlist_enabled": True,
    }
    write_json(config.rag_dir / "metadata.json", metadata)
    return metadata
