from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


@dataclass(slots=True)
class SentenceTransformerEmbedder:
    model_name: str | Path
    device: str = "cpu"
    normalize: bool = True
    max_length: int = 512
    tokenizer: Any = None
    model: Any = None

    def _load_config_with_fallback(self, model_path: str) -> Any:
        try:
            return AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        except ValueError:
            config_path = Path(model_path) / "config.json"
            raw_config = json.loads(config_path.read_text(encoding="utf-8"))
            inferred_model_type = str(raw_config.get("model_type") or "bert")
            sanitized = dict(raw_config)
            sanitized.pop("model_type", None)
            return AutoConfig.for_model(inferred_model_type, **sanitized)

    def __post_init__(self) -> None:
        model_path = str(self.model_name)
        config = self._load_config_with_fallback(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            config=config,
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            config=config,
        )
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            inputs = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = self.model(**inputs)
                hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                embeddings = summed / counts
                if self.normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
            batches.append(embeddings.detach().cpu().numpy().astype("float32"))
        return np.concatenate(batches, axis=0)
