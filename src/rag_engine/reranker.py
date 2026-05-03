from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path
from typing import List, Tuple

class RerankerEngine:
    def __init__(self, model_path: str | Path, device: str | None = None):
        # Auto-detect best device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Initializing RerankerEngine on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Use fp16/bf16 for speedup
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def rerank(self, query: str, documents: List[str]) -> List[float]:
        if not documents:
            return []
            
        # Prepare pairs: (query, doc1), (query, doc2), ...
        pairs = [[query, doc] for doc in documents]
        inputs = self.tokenizer(
            pairs, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=512
        ).to(self.device)
        
        outputs = self.model(**inputs, return_dict=True)
        # BGE Reranker output is in logits[0]
        scores = outputs.logits.view(-1).float().tolist()
        return scores
