from __future__ import annotations

from pathlib import Path

from legal_agent.utils.io import write_json
from rag_engine.service import KnowledgeService


def build_study_knowledge_assets(
    *,
    question_bank_path: str | Path,
    case_bank_path: str | Path,
    common_knowledge_path: str | Path,
    manifest_path: str | Path,
    use_legacy_statute_rag: bool = False,
    legacy_config_path: str | Path | None = None,
    legacy_device: str = "cpu",
) -> dict[str, object]:
    service = KnowledgeService(
        question_bank_path=question_bank_path,
        case_bank_path=case_bank_path,
        common_knowledge_path=common_knowledge_path,
        use_legacy_statute_rag=use_legacy_statute_rag,
        legacy_config_path=legacy_config_path,
        legacy_device=legacy_device,
    )
    summary = service.summary()
    payload = {
        **summary,
        "question_bank_path": str(Path(question_bank_path)),
        "case_bank_path": str(Path(case_bank_path)),
        "common_knowledge_path": str(Path(common_knowledge_path)),
    }
    write_json(manifest_path, payload)
    return payload