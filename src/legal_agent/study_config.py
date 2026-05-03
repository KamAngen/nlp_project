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
class MemoryConfig:
    recent_turn_window: int = 8
    compression_after_turns: int = 10
    compression_chunk_size: int = 8
    retain_recent_turns: int = 6
    vectorizer: str = "hashing"
    embedding_model_path: Path | None = None
    embedding_device: str = "cpu"


@dataclass(slots=True)
class StudyAgentConfig:
    project_root: Path
    memory_root: Path
    report_root: Path
    question_bank_path: Path
    case_bank_path: Path
    common_knowledge_path: Path
    system_memory_path: Path
    study_manifest_path: Path
    use_legacy_statute_rag: bool = True
    legacy_config_path: Path | None = None
    retrieval_top_k: int = 6
    default_exam_question_count: int = 5
    planner_backend: str = "llm_react"
    turn_analysis_mode: str = "llm"
    memory: MemoryConfig = field(default_factory=MemoryConfig)


def load_study_agent_config(config_path: str | Path) -> StudyAgentConfig:
    config_path = _resolve_path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    project_root = _resolve_path(raw.get("project_root", ".."), config_path.parent)
    legacy_path = raw.get("legacy_config_path")
    raw_memory = dict(raw.get("memory") or {})
    embedding_model_path = raw_memory.get("embedding_model_path")
    return StudyAgentConfig(
        project_root=project_root,
        memory_root=_resolve_path(raw["memory_root"], project_root),
        report_root=_resolve_path(raw["report_root"], project_root),
        question_bank_path=_resolve_path(raw["question_bank_path"], project_root),
        case_bank_path=_resolve_path(raw["case_bank_path"], project_root),
        common_knowledge_path=_resolve_path(raw["common_knowledge_path"], project_root),
        system_memory_path=_resolve_path(raw["system_memory_path"], project_root),
        study_manifest_path=_resolve_path(raw["study_manifest_path"], project_root),
        use_legacy_statute_rag=bool(raw.get("use_legacy_statute_rag", True)),
        legacy_config_path=_resolve_path(legacy_path, project_root) if legacy_path else None,
        retrieval_top_k=int(raw.get("retrieval_top_k", 6)),
        default_exam_question_count=int(raw.get("default_exam_question_count", 5)),
        planner_backend=str(raw.get("planner_backend", "llm_react")),
        turn_analysis_mode=str(raw.get("turn_analysis_mode", "llm")),
        memory=MemoryConfig(
            recent_turn_window=int(raw_memory.get("recent_turn_window", 8)),
            compression_after_turns=int(raw_memory.get("compression_after_turns", 10)),
            compression_chunk_size=int(raw_memory.get("compression_chunk_size", 8)),
            retain_recent_turns=int(raw_memory.get("retain_recent_turns", 6)),
            vectorizer=str(raw_memory.get("vectorizer", "embedding")),
            embedding_model_path=_resolve_path(embedding_model_path, project_root) if embedding_model_path else None,
            embedding_device=str(raw_memory.get("embedding_device", "cpu")),
        ),
    )