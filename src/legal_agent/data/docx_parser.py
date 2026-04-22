from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET
import zipfile

from legal_agent.utils.text import clean_text


WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
ASPOSE_EVAL_PREFIX = "Created with an evaluation copy of Aspose.Words."


@dataclass(slots=True)
class DocBlock:
    block_type: str
    text: str
    order: int


def _iter_child_text(node: ET.Element) -> str:
    pieces: list[str] = []
    for child in node.iter():
        if child.tag == f"{WORD_NS}t" and child.text:
            pieces.append(child.text)
        elif child.tag == f"{WORD_NS}tab":
            pieces.append(" ")
        elif child.tag == f"{WORD_NS}br":
            pieces.append("\n")
    return clean_text("".join(pieces))


def _table_to_text(table: ET.Element) -> str:
    rows: list[str] = []
    for row in table.findall(f"{WORD_NS}tr"):
        cells: list[str] = []
        for cell in row.findall(f"{WORD_NS}tc"):
            cell_text = _iter_child_text(cell)
            cells.append(cell_text)
        joined = " | ".join(cell for cell in cells if cell)
        if joined:
            rows.append(joined)
    if not rows:
        return ""
    return clean_text("表格：\n" + "\n".join(rows))


def parse_docx_blocks(docx_path: str | Path) -> list[DocBlock]:
    docx_path = Path(docx_path)
    blocks: list[DocBlock] = []
    with zipfile.ZipFile(docx_path) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    body = root.find(f"{WORD_NS}body")
    if body is None:
        return blocks

    order = 0
    for child in body:
        if child.tag == f"{WORD_NS}p":
            text = _iter_child_text(child)
            if text:
                if text.startswith(ASPOSE_EVAL_PREFIX):
                    continue
                blocks.append(DocBlock(block_type="paragraph", text=text, order=order))
                order += 1
        elif child.tag == f"{WORD_NS}tbl":
            text = _table_to_text(child)
            if text:
                blocks.append(DocBlock(block_type="table", text=text, order=order))
                order += 1
    return blocks
