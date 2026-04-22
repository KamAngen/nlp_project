from __future__ import annotations

from pathlib import Path
from typing import Any
import zipfile
from xml.etree import ElementTree as ET

from legal_agent.config import AppConfig
from legal_agent.data.docx_parser import parse_docx_blocks
from legal_agent.data.law_metadata import build_document_manifest
from legal_agent.utils.io import ensure_dir, write_json, write_jsonl
from legal_agent.utils.text import (
    clean_text,
    extract_article_heading,
    extract_cross_references,
    extract_section_heading,
    join_non_empty,
    split_into_char_windows,
)


def _build_units(blocks: list[Any]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    pending_section: list[str] = []
    current: dict[str, Any] | None = None
    preamble: list[str] = []

    for block in blocks:
        text = clean_text(block.text)
        if not text:
            continue

        section_heading = extract_section_heading(text)
        article_heading = extract_article_heading(text)

        if section_heading:
            pending_section = [text]
            if current is not None:
                current.setdefault("section_context", []).append(text)
            else:
                preamble.append(text)
            continue

        if article_heading:
            if current is not None:
                units.append(current)
            current = {
                "article_heading": article_heading,
                "section_context": list(pending_section),
                "texts": [text],
            }
            continue

        if current is None:
            preamble.append(text)
        else:
            current["texts"].append(text)

    if current is not None:
        units.append(current)

    if not units and preamble:
        units.append(
            {
                "article_heading": None,
                "section_context": [],
                "texts": preamble,
            }
        )
    elif preamble:
        units.insert(
            0,
            {
                "article_heading": None,
                "section_context": [],
                "texts": preamble,
            },
        )
    return units


def build_law_corpus(config: AppConfig) -> dict[str, Any]:
    ensure_dir(config.artifact_root)
    manifest, summary = build_document_manifest(
        config.law_dir,
        config.law_doc_dir,
        config.law_catalog_glob,
        config.project_root,
    )
    write_jsonl(config.manifest_path, manifest)

    chunks: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    for record in manifest:
        if record["status"] != "matched" or not record.get("source_path"):
            continue

        try:
            blocks = parse_docx_blocks(config.resolve_project_path(record["source_path"]))
        except (FileNotFoundError, KeyError, zipfile.BadZipFile, ET.ParseError, OSError) as exc:
            parse_errors.append(
                {
                    "title": str(record.get("title") or ""),
                    "source_path": str(record.get("source_path") or ""),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        units = _build_units(blocks)
        if not units:
            continue

        for unit_index, unit in enumerate(units, start=1):
            unit_text = join_non_empty([record["title"], *unit["section_context"], *unit["texts"]])
            windows = split_into_char_windows(
                unit_text,
                max_chars=config.retrieval.max_chunk_chars,
                overlap=config.retrieval.chunk_overlap_chars,
            )
            for window_index, window in enumerate(windows, start=1):
                chunk_id = f"{record['document_id']}::{unit_index:04d}::{window_index:02d}"
                chunk = {
                    "chunk_id": chunk_id,
                    "document_id": record["document_id"],
                    "document_title": record["title"],
                    "normalized_title": record["normalized_title"],
                    "article_heading": unit["article_heading"],
                    "section_context": unit["section_context"],
                    "category_raw": record["category_raw"],
                    "effect_level": record["effect_level"],
                    "effect_rank": record["effect_rank"],
                    "jurisdiction_type": record.get("jurisdiction_type", "national"),
                    "jurisdiction_scope": record.get("jurisdiction_scope", "national"),
                    "jurisdiction_rank": record.get("jurisdiction_rank", 0),
                    "region_code": record.get("region_code"),
                    "region_name": record.get("region_name"),
                    "region_path_codes": record.get("region_path_codes", []),
                    "region_path_names": record.get("region_path_names", []),
                    "promulgation_date": record["promulgation_date"],
                    "effective_date": record["effective_date"],
                    "version_date": record["version_date"],
                    "source_path": record["source_path"],
                    "cross_references": extract_cross_references(window),
                    "text": window,
                    "retrieval_text": join_non_empty([
                        record["title"],
                        record["effect_level"],
                        " > ".join(record.get("region_path_names", [])) if record.get("region_path_names") else "全国适用",
                        unit["article_heading"] or "",
                        window,
                    ]),
                }
                chunks.append(chunk)

    category_counts: dict[str, int] = {}
    jurisdiction_counts: dict[str, int] = {}
    for chunk in chunks:
        category_counts[chunk["effect_level"]] = category_counts.get(chunk["effect_level"], 0) + 1
        scope = str(chunk.get("jurisdiction_scope") or "unknown")
        jurisdiction_counts[scope] = jurisdiction_counts.get(scope, 0) + 1

    summary_payload = {
        "catalog_rows": summary.total_catalog_rows,
        "docx_files": summary.total_docx_files,
        "matched_documents": summary.matched_documents,
        "missing_docx": summary.missing_docx,
        "missing_catalog": summary.missing_catalog,
        "chunk_count": len(chunks),
        "parse_error_count": len(parse_errors),
        "effect_level_chunk_counts": category_counts,
        "jurisdiction_scope_chunk_counts": jurisdiction_counts,
        "local_chunk_count": sum(1 for chunk in chunks if chunk.get("jurisdiction_type") == "local"),
    }
    write_jsonl(config.corpus_path, chunks)
    if parse_errors:
        write_json(config.artifact_root / "law_parse_errors.json", parse_errors)
    write_json(config.corpus_summary_path, summary_payload)
    return summary_payload
