from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from typing import List, Union

class EmbeddingEngine:
    def __init__(self, model_path: Union[str, Path], device: str | None = None):
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
            
        print(f"Initializing EmbeddingEngine on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use bfloat16 on MPS/CUDA for massive speedup
        dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        
        self.model = AutoModel.from_pretrained(
            model_path, 
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        
        self.is_qwen = "qwen" in str(model_path).lower()

    @torch.no_grad()
    def embed_texts(self, texts: List[str], batch_size: int | None = None) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, self.model.config.hidden_size), device=self.device)
            
        if batch_size is None:
            batch_size = self._calculate_dynamic_batch_size()
            
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            # Ensure no empty strings
            batch_texts = [t if t.strip() else "无内容" for t in batch_texts]
            
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=512
            ).to(self.device)
            
            outputs = self.model(**inputs)
            mask = inputs["attention_mask"]
            
            if self.is_qwen:
                # Optimized Masked Mean Pooling for decoder-only models
                # We use the hidden states and apply the mask to ignore padding tokens
                last_hidden = outputs.last_hidden_state
                weighted_sum = (last_hidden * mask.unsqueeze(-1)).sum(dim=1)
                sum_mask = mask.sum(dim=1).unsqueeze(-1)
                batch_embeddings = weighted_sum / sum_mask
            else:
                # Default to CLS token for encoder models (like BERT/BGE)
                batch_embeddings = outputs.last_hidden_state[:, 0]
                
            all_embeddings.append(F.normalize(batch_embeddings, p=2, dim=1))
            
        return torch.cat(all_embeddings, dim=0)

    def _calculate_dynamic_batch_size(self) -> int:
        """
        Smartly estimate batch size based on hardware.
        """
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # Base logic
        if self.device == "cpu":
            return 4
            
        # For GPU/MPS
        if "mps" in self.device:
            # Apple Silicon: unified memory
            if total_ram_gb >= 32:
                return 32 # Stable sweet spot
            else:
                return 16
        else:
            # CUDA
            try:
                # Real-time free VRAM check
                free_vram = torch.cuda.mem_get_info()[0] / (1024**3)
                if free_vram > 12: return 128
                if free_vram > 6: return 64
                return 32
            except:
                return 16
        
        return 8 # Safe fallback

    def embed_query(self, query: str) -> torch.Tensor:
        # BGE sometimes needs a prefix for queries
        if not self.is_qwen and "bge" in self.tokenizer.name_or_path.lower():
            query = f"查询: {query}"
        return self.embed_texts([query])
