from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from legal_agent.utils.io import ensure_dir, write_json
from legal_agent.utils.text import normalize_title


SUPPORTED_WORD_SUFFIXES = {".docx", ".docm"}
LEGACY_WORD_SUFFIXES = {".doc"}
CATALOG_COLUMNS = ("标题", "公布日期", "施行日期", "法律法规分类")


@dataclass(slots=True)
class LawDatasetPreparationSummary:
    source_root: str
    target_root: str
    linked_documents: int
    archived_legacy_docs: int
    catalog_rows: int
    unique_catalog_titles: int


def _normalize_date(value: Any) -> str | None:
    if pd.isna(value):
        return None
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    return text[:10]


def _iter_source_catalog_paths(source_root: Path) -> list[Path]:
    nested_catalogs = sorted((source_root / "law_files").glob("*.xlsx"))
    if nested_catalogs:
        return nested_catalogs
    legacy_catalog = source_root / "catalogs" / "law_catalog_master.csv"
    return [legacy_catalog] if legacy_catalog.exists() else []


def _iter_source_word_paths(source_root: Path) -> list[Path]:
    candidates = [source_root / "law_files" / "files", source_root / "files"]
    return [path for path in candidates if path.exists()]


def _read_catalog(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_excel(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = [column for column in CATALOG_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Catalog {path} 缺少必要列: {missing}")
    normalized = pd.DataFrame(
        {
            "标题": frame["标题"].map(lambda value: str(value).strip() if not pd.isna(value) else ""),
            "公布日期": frame["公布日期"].map(_normalize_date),
            "施行日期": frame["施行日期"].map(_normalize_date),
            "法律法规分类": frame["法律法规分类"].map(lambda value: str(value).strip() if not pd.isna(value) else ""),
        }
    )
    normalized = normalized[normalized["标题"] != ""]
    normalized["source_catalog"] = path.name
    normalized["normalized_title"] = normalized["标题"].map(normalize_title)
    return normalized


def _build_master_catalog(source_root: Path, target_root: Path) -> pd.DataFrame:
    catalog_paths = _iter_source_catalog_paths(source_root)
    if not catalog_paths:
        raise FileNotFoundError(f"在 {source_root} 下未找到可用的法规目录表。")

    frames = [_read_catalog(path) for path in catalog_paths]
    master = pd.concat(frames, ignore_index=True)
    master = master.drop_duplicates(subset=["标题", "公布日期", "施行日期", "法律法规分类"], keep="first")
    master = master.sort_values(by=["标题", "公布日期", "施行日期", "法律法规分类"], kind="stable")

    catalog_dir = ensure_dir(target_root / "catalogs")
    master.drop(columns=["normalized_title"]).to_csv(catalog_dir / "law_catalog_master.csv", index=False)
    master.drop(columns=["normalized_title"]).to_excel(catalog_dir / "law_catalog_master.xlsx", index=False)
    return master


def _link_or_copy(source: Path, target: Path) -> None:
    ensure_dir(target.parent)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _build_document_store(source_root: Path, target_root: Path) -> tuple[int, int]:
    target_dir = ensure_dir(target_root / "files")
    legacy_dir = ensure_dir(target_root / "legacy_docs")
    linked_documents = 0
    archived_legacy_docs = 0

    seen_names: set[str] = set()
    seen_legacy_names: set[str] = set()
    for directory in _iter_source_word_paths(source_root):
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in LEGACY_WORD_SUFFIXES:
                if path.name in seen_legacy_names:
                    continue
                _link_or_copy(path, legacy_dir / path.name)
                archived_legacy_docs += 1
                seen_legacy_names.add(path.name)
                continue
            if suffix not in SUPPORTED_WORD_SUFFIXES or path.name in seen_names:
                continue
            _link_or_copy(path, target_dir / path.name)
            linked_documents += 1
            seen_names.add(path.name)
    return linked_documents, archived_legacy_docs


def prepare_law_dataset(
    source_root: str | Path,
    target_root: str | Path,
    *,
    cleanup_source: bool = False,
) -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    target_root = Path(target_root).resolve()
    ensure_dir(target_root)

    master_catalog = _build_master_catalog(source_root, target_root)
    linked_documents, archived_legacy_docs = _build_document_store(source_root, target_root)

    summary = LawDatasetPreparationSummary(
        source_root=str(source_root),
        target_root=str(target_root),
        linked_documents=linked_documents,
        archived_legacy_docs=archived_legacy_docs,
        catalog_rows=len(master_catalog),
        unique_catalog_titles=int(master_catalog["标题"].nunique()),
    )
    summary_payload = {
        "source_root": summary.source_root,
        "target_root": summary.target_root,
        "linked_documents": summary.linked_documents,
        "archived_legacy_docs": summary.archived_legacy_docs,
        "catalog_rows": summary.catalog_rows,
        "unique_catalog_titles": summary.unique_catalog_titles,
    }
    write_json(target_root / "catalogs" / "dataset_summary.json", summary_payload)

    if cleanup_source and source_root != target_root and source_root.exists():
        shutil.rmtree(source_root)

    return summary_payload