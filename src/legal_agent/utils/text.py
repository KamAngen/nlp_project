from __future__ import annotations

import re
from collections.abc import Iterable

TITLE_DATE_SUFFIX_RE = re.compile(r"_(\d{8})$")
WHITESPACE_RE = re.compile(r"\s+")
CROSS_REF_RE = re.compile(r"《([^》\n]{2,120})》")
ARTICLE_RE = re.compile(r"^第([一二三四五六七八九十百千万零〇两\d]+)条")
SECTION_RE = re.compile(r"^第([一二三四五六七八九十百千万零〇两\d]+)(编|章|节)")

RAW_TO_EFFECT_LEVEL = {
    "宪法": "宪法",
    "法律": "法律",
    "法律解释": "法律",
    "修正案": "法律",
    "有关法律问题和重大问题的决定（部分）": "法律",
    "行政法规": "法规",
    "监察法规": "法规",
    "地方性法规": "法规",
    "地方法规": "法规",
    "法规性决定": "法规",
    "司法解释": "司法解释",
}

LOCAL_CATEGORY_VALUES = {
    "地方性法规",
    "地方法规",
    "法规性决定",
}

EFFECT_LEVEL_RANK = {
    "宪法": 4,
    "法律": 3,
    "法规": 2,
    "司法解释": 1,
    "未知": 0,
}


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.replace("\u3000", " ").replace("\xa0", " ")).strip()


def clean_text(text: str) -> str:
    text = text.replace("+", " ")
    text = text.replace("（ ", "（").replace(" ）", "）")
    text = text.replace("《 ", "《").replace(" 》", "》")
    return normalize_whitespace(text)


def normalize_title(title: str) -> str:
    title = TITLE_DATE_SUFFIX_RE.sub("", title)
    title = clean_text(title.replace("_", " "))
    if title.startswith("《") and title.endswith("》"):
        title = title[1:-1].strip()
    return title


def generate_title_aliases(title: str) -> list[str]:
    normalized = normalize_title(title)
    if not normalized:
        return []

    aliases = [normalized]
    if normalized.startswith("中华人民共和国"):
        short_title = normalized.removeprefix("中华人民共和国").strip()
        if short_title and short_title not in aliases:
            aliases.append(short_title)
    return aliases


def normalize_category(raw_category: str) -> str:
    return RAW_TO_EFFECT_LEVEL.get(raw_category, "未知")


def is_local_category(raw_category: str) -> bool:
    return str(raw_category or "").strip() in LOCAL_CATEGORY_VALUES


def effect_rank(raw_category_or_level: str) -> int:
    level = RAW_TO_EFFECT_LEVEL.get(raw_category_or_level, raw_category_or_level)
    return EFFECT_LEVEL_RANK.get(level, 0)


def extract_cross_references(text: str) -> list[str]:
    refs = []
    seen: set[str] = set()
    for match in CROSS_REF_RE.finditer(text):
        title = normalize_title(match.group(1))
        if not title or title in seen:
            continue
        seen.add(title)
        refs.append(title)
    return refs


def split_into_char_windows(text: str, max_chars: int, overlap: int) -> list[str]:
    text = clean_text(text)
    if len(text) <= max_chars:
        return [text] if text else []

    windows: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        window = text[start:end].strip()
        if window:
            windows.append(window)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return windows


def extract_article_heading(text: str) -> str | None:
    match = ARTICLE_RE.match(clean_text(text))
    if match:
        return match.group(0)
    return None


def extract_section_heading(text: str) -> str | None:
    match = SECTION_RE.match(clean_text(text))
    if match:
        return match.group(0)
    return None


def safe_slug(text: str) -> str:
    text = normalize_title(text)
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", text)
    return text.strip("_") or "untitled"


def simple_tokenize(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    try:
        import jieba

        tokens = [token.strip() for token in jieba.lcut(text, cut_all=False) if token.strip()]
        if tokens:
            return tokens
    except Exception:
        pass

    alnum_tokens = re.findall(r"[0-9A-Za-z_]+", text.lower())
    chinese_chars = [char for char in text if "\u4e00" <= char <= "\u9fff"]
    bigrams = ["".join(chinese_chars[index : index + 2]) for index in range(max(0, len(chinese_chars) - 1))]
    tokens = alnum_tokens + chinese_chars + bigrams
    return [token for token in tokens if token]


def join_non_empty(parts: Iterable[str]) -> str:
    return "\n".join(clean_text(part) for part in parts if clean_text(part))


def truncate_text(text: str, max_chars: int = 320) -> str:
    normalized = clean_text(text)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip(" ，,；;：:") + "…"
