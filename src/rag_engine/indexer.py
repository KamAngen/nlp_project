from __future__ import annotations

import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from rag_engine.schema import KnowledgeRecord
from rag_engine.embedding import EmbeddingEngine

class VectorIndex:
    def __init__(self, name: str, embedding_engine: EmbeddingEngine):
        self.name = name
        self.embedding_engine = embedding_engine
        self.records: List[KnowledgeRecord] = []
        self.embeddings: Optional[torch.Tensor] = None

    def add_records(self, records: List[KnowledgeRecord]):
        self.records.extend(records)
        # We index the title (doubled weight) and content combined
        # This makes the title more influential in the semantic space
        texts = [f"{r.title}\n{r.title}\n{r.content}" for r in records]
        new_embeddings = self.embedding_engine.embed_texts(texts)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = torch.cat([self.embeddings, new_embeddings], dim=0)

    def update_record(self, record: KnowledgeRecord):
        # 1. Find existing record index
        found_idx = -1
        for i, r in enumerate(self.records):
            if r.record_id == record.record_id:
                found_idx = i
                break
        
        if found_idx != -1:
            # 2. Update existing
            self.records[found_idx] = record
            text = f"{record.title}\n{record.title}\n{record.content}"
            new_emb = self.embedding_engine.embed_texts([text])
            if self.embeddings is not None:
                self.embeddings[found_idx] = new_emb[0]
        else:
            # 3. Add as new
            self.add_records([record])

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.embeddings is None:
            return []
            
        query_embedding = self.embedding_engine.embed_query(query)
        # Cosine similarity (already normalized)
        scores = torch.mm(query_embedding, self.embeddings.t())[0]
        
        top_scores, top_indices = torch.topk(scores, min(top_k, len(self.records)))
        
        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append({
                "record": self.records[idx],
                "score": score
            })
        return results

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata and records
        with open(path / "records.jsonl", "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
                
        # Save embeddings
        if self.embeddings is not None:
            torch.save(self.embeddings, path / "embeddings.pt")

    @classmethod
    def load(cls, path: Union[str, Path], embedding_engine: EmbeddingEngine) -> VectorIndex:
        path = Path(path)
        name = path.name
        index = cls(name, embedding_engine)
        
        if (path / "records.jsonl").exists():
            with open(path / "records.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    index.records.append(KnowledgeRecord(**data))
                    
        if (path / "embeddings.pt").exists():
            index.embeddings = torch.load(path / "embeddings.pt", map_location=embedding_engine.device)
            
        return index
