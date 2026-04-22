from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import difflib
import re

import pandas as pd

from legal_agent.data.admin_divisions import infer_jurisdiction
from legal_agent.utils.text import effect_rank, normalize_category, normalize_title, safe_slug


DATE_SUFFIX_RE = re.compile(r"_(\d{8})$")
SUPPORTED_WORD_SUFFIXES = {".docx", ".docm"}


@dataclass(slots=True)
class ManifestSummary:
    total_catalog_rows: int
    total_docx_files: int
    matched_documents: int
    missing_docx: int
    missing_catalog: int


def load_catalog_tables(law_dir: str | Path, catalog_glob: str) -> pd.DataFrame:
    law_dir = Path(law_dir)
    frames: list[pd.DataFrame] = []
    for path in sorted(law_dir.glob(catalog_glob)):
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        else:
            frame = pd.read_excel(path)
        frame["source_catalog"] = path.relative_to(law_dir).as_posix()
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No catalog files matched {catalog_glob} under {law_dir}")

    catalog = pd.concat(frames, ignore_index=True)
    catalog["标题"] = catalog["标题"].map(lambda value: str(value).strip() if not pd.isna(value) else "")
    catalog["公布日期"] = catalog["公布日期"].map(lambda value: None if pd.isna(value) else str(value))
    catalog["施行日期"] = catalog["施行日期"].map(lambda value: None if pd.isna(value) else str(value))
    catalog["法律法规分类"] = catalog["法律法规分类"].map(lambda value: "未知" if pd.isna(value) else str(value).strip())
    catalog["normalized_title"] = catalog["标题"].map(normalize_title)
    catalog["effect_level"] = catalog["法律法规分类"].map(normalize_category)
    catalog["effect_rank"] = catalog["effect_level"].map(effect_rank)
    return catalog


def scan_docx_files(law_doc_dir: str | Path, project_root: str | Path) -> pd.DataFrame:
    law_doc_dir = Path(law_doc_dir)
    project_root = Path(project_root).resolve()
    rows: list[dict[str, object]] = []
    for path in sorted(law_doc_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_WORD_SUFFIXES:
            continue
        stem = path.stem
        version_date = None
        match = DATE_SUFFIX_RE.search(stem)
        if match:
            version_date = match.group(1)
            stem = DATE_SUFFIX_RE.sub("", stem)
        normalized = normalize_title(stem)
        resolved_path = path.resolve()
        try:
            source_path = resolved_path.relative_to(project_root).as_posix()
        except ValueError:
            source_path = str(resolved_path)
        rows.append(
            {
                "source_path": source_path,
                "file_name": path.name,
                "doc_title": stem,
                "normalized_title": normalized,
                "version_date": version_date,
                "document_id": safe_slug(normalized),
            }
        )
    return pd.DataFrame(rows)


def _pick_doc_record(row: pd.Series, docs_by_norm: dict[str, list[dict[str, object]]], all_doc_titles: list[str]) -> dict[str, object] | None:
    normalized_title = row["normalized_title"]
    if normalized_title in docs_by_norm:
        candidates = docs_by_norm[normalized_title]
        return sorted(candidates, key=lambda item: str(item.get("version_date") or ""), reverse=True)[0]

    close_matches = difflib.get_close_matches(normalized_title, all_doc_titles, n=1, cutoff=0.92)
    if not close_matches:
        return None
    return docs_by_norm[close_matches[0]][0]


def build_document_manifest(
    law_dir: str | Path,
    law_doc_dir: str | Path,
    catalog_glob: str,
    project_root: str | Path,
) -> tuple[list[dict[str, object]], ManifestSummary]:
    catalog = load_catalog_tables(law_dir, catalog_glob)
    docs = scan_docx_files(law_doc_dir, project_root)
    docs_by_norm: dict[str, list[dict[str, object]]] = {}
    for record in docs.to_dict(orient="records"):
        docs_by_norm.setdefault(str(record["normalized_title"]), []).append(record)

    all_doc_titles = list(docs_by_norm.keys())
    manifest: list[dict[str, object]] = []
    matched_titles: set[str] = set()

    for row in catalog.to_dict(orient="records"):
        doc_record = _pick_doc_record(pd.Series(row), docs_by_norm, all_doc_titles)
        jurisdiction = infer_jurisdiction(str(row["标题"]), str(row["法律法规分类"]))
        entry = {
            "document_id": safe_slug(str(row["normalized_title"])),
            "title": row["标题"],
            "normalized_title": row["normalized_title"],
            "promulgation_date": str(row["公布日期"]),
            "effective_date": str(row["施行日期"]),
            "category_raw": row["法律法规分类"],
            "effect_level": row["effect_level"],
            "effect_rank": int(row["effect_rank"]),
            "source_catalog": row["source_catalog"],
            **jurisdiction,
        }
        if doc_record is None:
            entry.update(
                {
                    "status": "missing_docx",
                    "source_path": None,
                    "file_name": None,
                    "version_date": None,
                }
            )
        else:
            matched_titles.add(str(doc_record["normalized_title"]))
            entry.update(
                {
                    "status": "matched",
                    "source_path": doc_record["source_path"],
                    "file_name": doc_record["file_name"],
                    "version_date": doc_record["version_date"],
                    "document_id": doc_record["document_id"],
                }
            )
        manifest.append(entry)

    catalog_titles = {str(item["normalized_title"]) for item in manifest}
    for doc_record in docs.to_dict(orient="records"):
        if str(doc_record["normalized_title"]) in catalog_titles:
            continue
        manifest.append(
            {
                "document_id": doc_record["document_id"],
                "title": doc_record["doc_title"],
                "normalized_title": doc_record["normalized_title"],
                "promulgation_date": None,
                "effective_date": None,
                "category_raw": "未知",
                "effect_level": "未知",
                "effect_rank": 0,
                "source_catalog": None,
                **infer_jurisdiction(str(doc_record["doc_title"]), None),
                "status": "missing_catalog",
                "source_path": doc_record["source_path"],
                "file_name": doc_record["file_name"],
                "version_date": doc_record["version_date"],
            }
        )

    summary = ManifestSummary(
        total_catalog_rows=len(catalog),
        total_docx_files=len(docs),
        matched_documents=sum(item["status"] == "matched" for item in manifest),
        missing_docx=sum(item["status"] == "missing_docx" for item in manifest),
        missing_catalog=sum(item["status"] == "missing_catalog" for item in manifest),
    )
    return manifest, summary
