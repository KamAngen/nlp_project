import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from legal_agent.rag import index_builder


class FakeEmbedder:
    def __init__(self, model_name: str | Path, device: str = "cpu", normalize: bool = True):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return np.ones((len(texts), 3), dtype=np.float32)


def test_build_rag_index_serializes_embedding_model_path(tmp_path: Path, monkeypatch):
    corpus_path = tmp_path / "law_chunks.jsonl"
    rag_dir = tmp_path / "rag"
    chunk = {
        "document_id": "doc-1",
        "normalized_title": "中华人民共和国劳动法",
        "document_title": "中华人民共和国劳动法",
        "effect_level": "法律",
        "effect_rank": 2,
        "retrieval_text": "中华人民共和国劳动法 劳动合同 工资",
        "text": "劳动者依法享有劳动报酬。",
        "jurisdiction_scope": "national",
    }
    corpus_path.write_text(json.dumps(chunk, ensure_ascii=False) + "\n", encoding="utf-8")

    config = SimpleNamespace(
        rag_dir=rag_dir,
        corpus_path=corpus_path,
        models=SimpleNamespace(embedding_model=tmp_path / "models" / "bge-small-zh"),
    )

    monkeypatch.setattr(index_builder, "SentenceTransformerEmbedder", FakeEmbedder)
    monkeypatch.setattr(index_builder, "build_citation_graph", lambda chunks: ({}, {}, {}))
    monkeypatch.setattr(index_builder, "graph_to_json", lambda graph: {})

    metadata = index_builder.build_rag_index(config, device="cpu")
    stored = json.loads((rag_dir / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["embedding_model"] == str(config.models.embedding_model)
    assert stored["embedding_model"] == str(config.models.embedding_model)
    assert stored["document_shortlist_enabled"] is True
    assert stored["chunk_count"] == 1
    assert stored["document_count"] == 1