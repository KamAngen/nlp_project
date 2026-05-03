from __future__ import annotations

from pathlib import Path

from legal_agent.utils.io import read_jsonl
from rag_engine.schema import KnowledgeRecord


def _safe_read_jsonl(path: str | Path) -> list[dict]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    return read_jsonl(file_path)


def load_question_bank(path: str | Path) -> list[KnowledgeRecord]:
    records: list[KnowledgeRecord] = []
    for row in _safe_read_jsonl(path):
        record_id = str(row.get("question_id") or row.get("id") or row.get("record_id") or "")
        title = str(row.get("title") or row.get("stem") or row.get("question") or "").strip()
        stem = str(row.get("stem") or row.get("question") or row.get("title") or "").strip()
        options = dict(row.get("options") or {})
        option_text = " ".join(f"{key}. {value}" for key, value in options.items())
        analysis = str(row.get("analysis") or "").strip()
        tags = [str(tag) for tag in row.get("tags", [])]
        topic = str(row.get("topic") or "综合")
        content = "\n".join(part for part in [stem, option_text, analysis] if part)
        records.append(
            KnowledgeRecord(
                record_id=record_id,
                source_type="question_bank",
                title=title or "未命名题目",
                content=content,
                tags=list(dict.fromkeys([topic, *tags])),
                difficulty=str(row.get("difficulty") or "medium"),
                metadata={
                    "answer": row.get("answer"),
                    "analysis": analysis,
                    "options": options,
                    "topic": topic,
                    "score": int(row.get("score", 20)),
                },
            )
        )
    return records


def load_case_bank(path: str | Path) -> list[KnowledgeRecord]:
    records: list[KnowledgeRecord] = []
    for row in _safe_read_jsonl(path):
        title = str(row.get("title") or row.get("case_title") or "未命名案例").strip()
        facts = str(row.get("facts") or row.get("summary") or "").strip()
        issue = str(row.get("issue") or "").strip()
        holding = str(row.get("holding") or row.get("analysis") or "").strip()
        statutes = [str(item) for item in row.get("statutes", [])]
        tags = [str(tag) for tag in row.get("tags", [])]
        records.append(
            KnowledgeRecord(
                record_id=str(row.get("case_id") or row.get("id") or title),
                source_type="case_bank",
                title=title,
                content="\n".join(part for part in [facts, issue, holding] if part),
                tags=list(dict.fromkeys([*tags, *statutes])),
                metadata={"issue": issue, "holding": holding, "statutes": statutes},
            )
        )
    return records


def load_common_knowledge(path: str | Path) -> list[KnowledgeRecord]:
    records: list[KnowledgeRecord] = []
    for row in _safe_read_jsonl(path):
        title = str(row.get("title") or row.get("name") or "常识条目").strip()
        content = str(row.get("content") or row.get("text") or "").strip()
        tags = [str(tag) for tag in row.get("tags", [])]
        records.append(
            KnowledgeRecord(
                record_id=str(row.get("entry_id") or row.get("id") or title),
                source_type="common_knowledge",
                title=title,
                content=content,
                tags=tags,
                metadata={"source": row.get("source", "local_demo")},
            )
        )
    return records