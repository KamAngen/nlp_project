from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import re
from typing import Any, Iterable

from huggingface_hub import hf_hub_download, list_repo_files

from legal_agent.utils.io import ensure_dir, write_jsonl


DISC_LAW_REPO = "ShengbinYue/DISC-Law-SFT"
TASK_SUFFIX_RE = re.compile(r"[-_](\d+)$")


def download_disc_law_dataset(raw_dir: str | Path, *, repo_id: str = DISC_LAW_REPO) -> list[Path]:
    raw_dir = ensure_dir(raw_dir)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    downloaded: list[Path] = []
    for filename in list_repo_files(repo_id=repo_id, repo_type="dataset"):
        if not filename.endswith(".jsonl"):
            continue
        local_cache = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename)
        target = raw_dir / Path(filename).name
        shutil.copy2(local_cache, target)
        downloaded.append(target)
    return downloaded


def infer_task_family(sample_id: str) -> str:
    sample_id = TASK_SUFFIX_RE.sub("", sample_id)
    return sample_id.replace("-", "_")


def iter_disc_law_records(raw_dir: str | Path) -> Iterable[dict[str, Any]]:
    raw_dir = Path(raw_dir)
    for path in sorted(raw_dir.glob("*.jsonl")):
        subset_name = path.stem
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                sample_id = str(raw["id"])
                references = raw.get("reference") or []
                if isinstance(references, str):
                    references = [references]
                yield {
                    "sample_id": sample_id,
                    "task_family": infer_task_family(sample_id),
                    "subset_name": subset_name,
                    "instruction": raw.get("input", ""),
                    "answer": raw.get("output", ""),
                    "references": references,
                    "has_references": bool(references),
                }


def normalize_disc_law_dataset(raw_dir: str | Path, output_path: str | Path) -> list[dict[str, Any]]:
    records = list(iter_disc_law_records(raw_dir))
    write_jsonl(output_path, records)
    return records
