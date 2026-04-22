from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import zipfile

from legal_agent.utils.io import ensure_dir, write_json


OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
SUPPORTED_BINARY_WORD_SUFFIXES = {".doc", ".docx", ".docm"}


@dataclass(slots=True)
class WordRepairSummary:
    repaired_legacy_docs: int = 0
    repaired_mislabeled_word_docs: int = 0
    skipped_existing_valid_docx: int = 0
    failed_repairs: int = 0


def _is_valid_wordprocessingml(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return "word/document.xml" in archive.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def _is_ole_word_document(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        with path.open("rb") as handle:
            return handle.read(len(OLE_MAGIC)) == OLE_MAGIC
    except OSError:
        return False


def _save_as_docx(source_path: Path, target_path: Path) -> None:
    try:
        import aspose.words as aw
    except Exception as exc:  # pragma: no cover - import failure is environment-specific
        raise RuntimeError(
            "缺少 aspose-words，无法转换旧版 Word 文档；请先安装 aspose-words>=26.3.0。"
        ) from exc

    ensure_dir(target_path.parent)
    with tempfile.TemporaryDirectory(prefix="law_doc_convert_", dir=target_path.parent) as temp_dir:
        temp_path = Path(temp_dir) / target_path.name
        document = aw.Document(str(source_path))
        if getattr(document, "has_macros", False):
            document.remove_macros()
        document.save(str(temp_path), aw.SaveFormat.DOCX)
        temp_path.replace(target_path)


def repair_law_documents(law_dir: str | Path) -> dict[str, object]:
    law_dir = Path(law_dir).resolve()
    files_dir = ensure_dir(law_dir / "files")
    legacy_dir = ensure_dir(law_dir / "legacy_docs")

    summary = WordRepairSummary()
    failures: list[dict[str, str]] = []

    for source_path in sorted(legacy_dir.glob("*.doc")):
        target_path = files_dir / f"{source_path.stem}.docx"
        if _is_valid_wordprocessingml(target_path):
            summary.skipped_existing_valid_docx += 1
            continue
        try:
            _save_as_docx(source_path, target_path)
            summary.repaired_legacy_docs += 1
        except Exception as exc:  # pragma: no cover - conversion failure depends on file content
            summary.failed_repairs += 1
            failures.append({
                "source_path": str(source_path),
                "target_path": str(target_path),
                "error": f"{type(exc).__name__}: {exc}",
            })

    for source_path in sorted(files_dir.iterdir()):
        if not source_path.is_file() or source_path.suffix.lower() not in SUPPORTED_BINARY_WORD_SUFFIXES:
            continue
        if _is_valid_wordprocessingml(source_path):
            continue
        if not _is_ole_word_document(source_path):
            continue
        try:
            _save_as_docx(source_path, source_path.with_suffix(".docx"))
            summary.repaired_mislabeled_word_docs += 1
        except Exception as exc:  # pragma: no cover - conversion failure depends on file content
            summary.failed_repairs += 1
            failures.append({
                "source_path": str(source_path),
                "target_path": str(source_path.with_suffix('.docx')),
                "error": f"{type(exc).__name__}: {exc}",
            })

    payload = {
        "law_dir": str(law_dir),
        "repaired_legacy_docs": summary.repaired_legacy_docs,
        "repaired_mislabeled_word_docs": summary.repaired_mislabeled_word_docs,
        "skipped_existing_valid_docx": summary.skipped_existing_valid_docx,
        "failed_repairs": summary.failed_repairs,
        "failures": failures,
    }
    write_json(law_dir / "catalogs" / "doc_repair_summary.json", payload)
    return payload