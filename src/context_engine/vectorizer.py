from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

from legal_agent.rag.embeddings import SentenceTransformerEmbedder
from legal_agent.utils.text import simple_tokenize


class MemoryVectorizer(Protocol):
    def similarity(self, query: str, candidate: str) -> float:
        ...


@dataclass(slots=True)
class HashingVectorizer:
    dim: int = 256
    _cache: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def _vectorize(self, text: str) -> np.ndarray:
        normalized = str(text or "").strip()
        cached = self._cache.get(normalized)
        if cached is not None:
            return cached

        vector = np.zeros(self.dim, dtype=np.float32)
        for token in simple_tokenize(normalized):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        self._cache[normalized] = vector
        return vector

    def similarity(self, query: str, candidate: str) -> float:
        if not str(query or "").strip() or not str(candidate or "").strip():
            return 0.0
        query_vector = self._vectorize(query)
        candidate_vector = self._vectorize(candidate)
        if not query_vector.any() or not candidate_vector.any():
            return 0.0
        return float(np.clip(float(np.dot(query_vector, candidate_vector)), 0.0, 1.0))


@dataclass(slots=True)
class TransformerVectorizer:
    model_path: str | Path
    device: str = "cpu"
    _embedder: SentenceTransformerEmbedder | None = field(default=None, init=False, repr=False)
    _cache: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def _ensure_embedder(self) -> SentenceTransformerEmbedder:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(self.model_path, device=self.device)
        return self._embedder

    def _vectorize(self, text: str) -> np.ndarray:
        normalized = str(text or "").strip()
        cached = self._cache.get(normalized)
        if cached is not None:
            return cached
        embedder = self._ensure_embedder()
        vector = embedder.encode([normalized], batch_size=1)[0]
        self._cache[normalized] = vector
        return vector

    def similarity(self, query: str, candidate: str) -> float:
        if not str(query or "").strip() or not str(candidate or "").strip():
            return 0.0
        query_vector = self._vectorize(query)
        candidate_vector = self._vectorize(candidate)
        numerator = float(np.dot(query_vector, candidate_vector))
        query_norm = float(np.linalg.norm(query_vector))
        candidate_norm = float(np.linalg.norm(candidate_vector))
        if query_norm <= 0.0 or candidate_norm <= 0.0:
            return 0.0
        cosine = numerator / (query_norm * candidate_norm)
        return float(np.clip(cosine, 0.0, 1.0))