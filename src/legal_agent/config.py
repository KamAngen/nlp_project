from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _expand_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def _resolve_path(value: str | Path, base_dir: Path | None = None) -> Path:
    path = _expand_path(value)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return Path(os.path.abspath(path))


@dataclass(slots=True)
class ModelsConfig:
    agent_base: Path
    embedding_model: Path


@dataclass(slots=True)
class RetrievalConfig:
    max_chunk_chars: int = 900
    chunk_overlap_chars: int = 120
    dense_top_k: int = 12
    bm25_top_k: int = 12
    final_top_k: int = 6
    rrf_k: int = 60
    hierarchy_boost: float = 0.12
    graph_boost: float = 0.08
    use_graph_expansion: bool = True
    use_document_shortlist: bool = True
    document_dense_top_k: int = 48
    document_bm25_top_k: int = 48
    document_shortlist_k: int = 96
    max_candidate_chunks: int = 8000
    document_score_boost: float = 0.2
    local_without_region_penalty: float = 0.58
    explicit_region_boost: float = 0.22
    ancestor_region_boost: float = 0.14
    descendant_region_boost: float = 0.08
    unrelated_local_penalty: float = 0.45
    observation_max_results: int = 4
    observation_max_chars: int = 360


@dataclass(slots=True)
class GenerationConfig:
    law_seed_count: int = 0
    disc_seed_count: int = 1000
    train_trajectory_count: int = 1600
    eval_trajectory_count: int = 200
    max_retries: int = 2


@dataclass(slots=True)
class TrainingConfig:
    output_dir: Path
    adapter_name: str
    max_seq_length: int = 4096
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    num_train_epochs: float = 2.0
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 200
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ])


@dataclass(slots=True)
class InferenceConfig:
    max_steps: int = 12
    max_new_tokens: int = 768
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 20
    presence_penalty: float = 1.0
    enable_thinking: bool = False
    load_in_4bit: bool = True
    compute_dtype: str = "bfloat16"


@dataclass(slots=True)
class AppConfig:
    project_root: Path
    law_dir: Path
    law_doc_dir: Path
    law_catalog_glob: str
    artifact_root: Path
    disc_law_dir: Path
    generated_data_dir: Path
    output_root: Path
    available_gpu_ids: list[int]
    models: ModelsConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    training: TrainingConfig
    inference: InferenceConfig

    @property
    def manifest_path(self) -> Path:
        return self.artifact_root / "law_manifest.jsonl"

    @property
    def corpus_path(self) -> Path:
        return self.artifact_root / "law_chunks.jsonl"

    @property
    def corpus_summary_path(self) -> Path:
        return self.artifact_root / "law_corpus_summary.json"

    @property
    def rag_dir(self) -> Path:
        return self.artifact_root / "rag"

    @property
    def disc_law_raw_dir(self) -> Path:
        return self.disc_law_dir / "raw"

    @property
    def disc_law_normalized_path(self) -> Path:
        return self.disc_law_dir / "disc_law_normalized.jsonl"

    @property
    def generated_train_path(self) -> Path:
        return self.generated_data_dir / "agent_train.jsonl"

    @property
    def generated_eval_path(self) -> Path:
        return self.generated_data_dir / "agent_eval.jsonl"

    @property
    def seed_manifest_path(self) -> Path:
        return self.generated_data_dir / "seed_manifest.jsonl"

    @property
    def generation_progress_path(self) -> Path:
        return self.generated_data_dir / "generation_progress.json"

    def resolve_project_path(self, value: str | Path) -> Path:
        path = _expand_path(value)
        if path.is_absolute():
            return Path(os.path.abspath(path))
        return Path(os.path.abspath(self.project_root / path))

    def project_relative_path(self, value: str | Path | None) -> str | None:
        if value is None:
            return None
        path = _expand_path(value)
        if not path.is_absolute():
            return path.as_posix()
        try:
            return Path(os.path.abspath(path)).relative_to(self.project_root).as_posix()
        except ValueError:
            return str(Path(os.path.abspath(path)))


def _build_models(raw: dict[str, Any], project_root: Path) -> ModelsConfig:
    return ModelsConfig(
        agent_base=_resolve_path(raw["agent_base"], project_root),
        embedding_model=_resolve_path(raw["embedding_model"], project_root),
    )


def _build_training(raw: dict[str, Any], project_root: Path) -> TrainingConfig:
    return TrainingConfig(
        output_dir=_resolve_path(raw["output_dir"], project_root),
        adapter_name=str(raw["adapter_name"]),
        max_seq_length=int(raw.get("max_seq_length", 4096)),
        per_device_train_batch_size=int(raw.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(raw.get("gradient_accumulation_steps", 16)),
        learning_rate=float(raw.get("learning_rate", 1e-4)),
        warmup_ratio=float(raw.get("warmup_ratio", 0.03)),
        num_train_epochs=float(raw.get("num_train_epochs", 2.0)),
        logging_steps=int(raw.get("logging_steps", 10)),
        eval_steps=int(raw.get("eval_steps", 100)),
        save_steps=int(raw.get("save_steps", 200)),
        lora_r=int(raw.get("lora_r", 32)),
        lora_alpha=int(raw.get("lora_alpha", 64)),
        lora_dropout=float(raw.get("lora_dropout", 0.05)),
        target_modules=[str(item) for item in raw.get("target_modules", [])],
    )


def load_app_config(config_path: str | Path) -> AppConfig:
    config_path = _resolve_path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    project_root = _resolve_path(raw.get("project_root", ".."), config_path.parent)

    return AppConfig(
        project_root=project_root,
        law_dir=_resolve_path(raw["law_dir"], project_root),
        law_doc_dir=_resolve_path(raw["law_doc_dir"], project_root),
        law_catalog_glob=str(raw.get("law_catalog_glob", "catalogs/law_catalog_master.csv")),
        artifact_root=_resolve_path(raw["artifact_root"], project_root),
        disc_law_dir=_resolve_path(raw["disc_law_dir"], project_root),
        generated_data_dir=_resolve_path(raw["generated_data_dir"], project_root),
        output_root=_resolve_path(raw["output_root"], project_root),
        available_gpu_ids=[int(item) for item in raw.get("available_gpu_ids", [1, 2, 4, 6])],
        models=_build_models(raw["models"], project_root),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        generation=GenerationConfig(**raw.get("generation", {})),
        training=_build_training(raw.get("training", {}), project_root),
        inference=InferenceConfig(**raw.get("inference", {})),
    )


def configured_cuda_visible_devices(config: AppConfig) -> str:
    return ",".join(str(gpu_id) for gpu_id in config.available_gpu_ids)


def apply_configured_cuda_visible_devices(config: AppConfig, *, overwrite: bool = False) -> str | None:
    desired = configured_cuda_visible_devices(config)
    if not desired:
        return None

    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current and not overwrite:
        return current

    os.environ["CUDA_VISIBLE_DEVICES"] = desired
    return desired
