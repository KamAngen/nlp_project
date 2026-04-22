from __future__ import annotations

import contextlib
from dataclasses import dataclass
from functools import lru_cache
import io
import json
from pathlib import Path
import re
from typing import Any

from legal_agent.utils.text import is_local_category, normalize_title


RESOURCE_PATH = Path(__file__).with_name("china_admin_divisions_county.json")

LEVEL_BY_CODE_LENGTH = {
    2: "province",
    4: "prefecture",
    6: "county",
}

JURISDICTION_RANK = {
    "national": 4,
    "province": 3,
    "prefecture": 2,
    "county": 1,
    "local_unknown": 0,
    "unknown": 0,
}

PLACEHOLDER_REGION_NAMES = {
    "市辖区",
    "县",
    "自治区直辖县级行政区划",
    "省直辖县级行政区划",
}

SUFFIX_PATTERNS = (
    "特别行政区",
    "维吾尔自治区",
    "壮族自治区",
    "回族自治区",
    "自治区",
    "自治州",
    "自治县",
    "林区",
    "地区",
    "盟",
    "省",
    "市",
    "县",
    "区",
)

TOWN_SUFFIX_PATTERNS = (
    "街道",
    "镇",
    "乡",
    "苏木",
    "民族乡",
    "地区办事处",
    "办事处",
)

LOCALITY_HINT_TOKENS = (
    "地方性法规",
    "地方法规",
    "地方条例",
    "地方规定",
    "本地",
    "当地",
    "本省",
    "本市",
    "本县",
    "本区",
    "所在省",
    "所在市",
    "所在区",
    "所在县",
    "所在地",
    "辖区",
)


@dataclass(frozen=True, slots=True)
class AdminDivision:
    code: str
    name: str
    level: str
    parent_code: str | None
    path_codes: tuple[str, ...]
    path_names: tuple[str, ...]
    aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResolvedLocation:
    raw_input: str
    province_name: str | None
    city_name: str | None
    county_name: str | None
    town_name: str | None
    village_name: str | None
    detail: str | None
    full_location: str | None
    matched_division: AdminDivision | None
    explicit_level: str | None
    path_codes: tuple[str, ...]
    path_names: tuple[str, ...]


def _load_tree() -> list[dict[str, Any]]:
    return json.loads(RESOURCE_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _get_jionlp() -> Any | None:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import jionlp
    except Exception:
        return None
    return jionlp


def _compact_path_names(path_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in path_names if name not in PLACEHOLDER_REGION_NAMES)


def _strip_region_suffix(name: str) -> str | None:
    for suffix in SUFFIX_PATTERNS:
        if not name.endswith(suffix):
            continue
        stripped = name[: -len(suffix)].strip()
        if len(stripped) >= 2:
            return stripped
    return None


def _extract_town_name(detail: str | None) -> str | None:
    cleaned = str(detail or "").strip()
    if not cleaned:
        return None
    for suffix in TOWN_SUFFIX_PATTERNS:
        pattern = rf"^(.+?{re.escape(suffix)})"
        match = re.match(pattern, cleaned)
        if match is None:
            continue
        town_name = match.group(1).strip()
        if len(town_name) >= len(suffix) + 1:
            return town_name
    return None


def _dedupe_regions(regions: list[AdminDivision], *, limit: int = 3) -> list[AdminDivision]:
    if not regions:
        return []
    deduped: list[AdminDivision] = []
    for division in sorted(regions, key=lambda item: len(item.path_codes), reverse=True):
        if any(existing.code == division.code for existing in deduped):
            continue
        if any(division.code in existing.path_codes[:-1] for existing in deduped):
            continue
        deduped.append(division)
        if len(deduped) >= limit:
            break
    return deduped


def _iter_divisions(
    nodes: list[dict[str, Any]],
    *,
    parent_code: str | None = None,
    path_codes: tuple[str, ...] = (),
    path_names: tuple[str, ...] = (),
) -> list[tuple[str, str, str | None, tuple[str, ...], tuple[str, ...]]]:
    records: list[tuple[str, str, str | None, tuple[str, ...], tuple[str, ...]]] = []
    for node in nodes:
        code = str(node.get("code") or "").strip()
        name = str(node.get("name") or "").strip()
        if not code or not name or len(code) not in LEVEL_BY_CODE_LENGTH:
            continue
        next_path_codes = path_codes + (code,)
        next_path_names = path_names + (name,)
        records.append((code, name, parent_code, next_path_codes, next_path_names))
        children = node.get("children") or []
        if children:
            records.extend(
                _iter_divisions(
                    children,
                    parent_code=code,
                    path_codes=next_path_codes,
                    path_names=next_path_names,
                )
            )
    return records


class ChinaAdminHierarchy:
    def __init__(self) -> None:
        raw_records = _iter_divisions(_load_tree())
        short_alias_counts: dict[str, int] = {}
        for _, name, _, _, _ in raw_records:
            short_alias = _strip_region_suffix(name)
            if short_alias:
                short_alias_counts[short_alias] = short_alias_counts.get(short_alias, 0) + 1

        self.divisions_by_code: dict[str, AdminDivision] = {}
        for code, name, parent_code, path_codes, path_names in raw_records:
            aliases = [name]
            short_alias = _strip_region_suffix(name)
            if short_alias and short_alias_counts.get(short_alias) == 1:
                aliases.append(short_alias)
            compact_path_names = _compact_path_names(path_names)
            self.divisions_by_code[code] = AdminDivision(
                code=code,
                name=name,
                level=LEVEL_BY_CODE_LENGTH[len(code)],
                parent_code=parent_code,
                path_codes=path_codes,
                path_names=compact_path_names,
                aliases=tuple(dict.fromkeys(alias for alias in aliases if alias)),
            )

        prefix_aliases: dict[str, list[tuple[str, AdminDivision]]] = {}
        search_aliases: list[tuple[str, AdminDivision]] = []
        alias_to_divisions: dict[str, list[AdminDivision]] = {}
        for division in self.divisions_by_code.values():
            for alias in division.aliases:
                first_char = alias[:1]
                prefix_aliases.setdefault(first_char, []).append((alias, division))
                search_aliases.append((alias, division))
                alias_to_divisions.setdefault(alias, []).append(division)

        self.prefix_aliases = {
            key: sorted(values, key=lambda item: len(item[0]), reverse=True)
            for key, values in prefix_aliases.items()
        }
        self.search_aliases = sorted(search_aliases, key=lambda item: len(item[0]), reverse=True)
        self.alias_to_divisions = {
            key: sorted(values, key=lambda item: len(item.path_codes), reverse=True)
            for key, values in alias_to_divisions.items()
        }

    def match_title_region(self, title: str) -> AdminDivision | None:
        normalized_title = normalize_title(title)
        if not normalized_title:
            return None
        for alias, division in self.prefix_aliases.get(normalized_title[:1], []):
            if normalized_title.startswith(alias):
                return division
        return None

    def divisions_for_name(self, name: str, *, level: str | None = None) -> list[AdminDivision]:
        candidates = list(self.alias_to_divisions.get(str(name or "").strip(), []))
        if level is None:
            return candidates
        return [division for division in candidates if division.level == level]

    def resolve_location_parts(
        self,
        *,
        province_name: str | None = None,
        city_name: str | None = None,
        county_name: str | None = None,
    ) -> AdminDivision | None:
        if county_name:
            candidates = self.divisions_for_name(county_name, level="county")
            filtered = [
                division
                for division in candidates
                if (not province_name or province_name in division.path_names)
                and (not city_name or city_name in division.path_names)
            ]
            if len(filtered) == 1:
                return filtered[0]
            if filtered:
                return filtered[0]

        if city_name:
            candidates = self.divisions_for_name(city_name, level="prefecture")
            filtered = [
                division
                for division in candidates
                if not province_name or province_name in division.path_names
            ]
            if len(filtered) == 1:
                return filtered[0]
            if filtered:
                return filtered[0]

        if province_name:
            provinces = self.divisions_for_name(province_name, level="province")
            if provinces:
                return provinces[0]
        return None

    def find_regions_in_text(self, text: str, *, limit: int = 3) -> list[AdminDivision]:
        normalized_text = normalize_title(text)
        if not normalized_text:
            return []

        matched: list[AdminDivision] = []
        for alias, division in self.search_aliases:
            if alias not in normalized_text:
                continue
            matched.append(division)

        if not matched:
            return []

        return _dedupe_regions(matched, limit=limit)

    def is_ancestor(self, ancestor_code: str, target_code: str) -> bool:
        target = self.divisions_by_code.get(target_code)
        if target is None:
            return False
        return ancestor_code in target.path_codes[:-1]


@lru_cache(maxsize=1)
def get_admin_hierarchy() -> ChinaAdminHierarchy:
    return ChinaAdminHierarchy()


def query_likely_needs_local_law(question: str) -> bool:
    normalized_question = normalize_title(question)
    return any(token in normalized_question for token in LOCALITY_HINT_TOKENS) or bool(
        extract_query_regions(normalized_question)
    )


def resolve_location_text(text: str) -> ResolvedLocation | None:
    raw_text = str(text or "").strip()
    if not raw_text:
        return None

    province_name = None
    city_name = None
    county_name = None
    town_name = None
    village_name = None
    detail = None
    full_location = None

    parser = _get_jionlp()
    if parser is not None:
        try:
            parsed = parser.parse_location(raw_text, town_village=True)
        except TypeError:
            parsed = parser.parse_location(raw_text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            province_name = parsed.get("province") or None
            city_name = parsed.get("city") or None
            county_name = parsed.get("county") or None
            town_name = parsed.get("town") or None
            village_name = parsed.get("village") or None
            detail = parsed.get("detail") or None
            full_location = parsed.get("full_location") or None

    if not town_name:
        town_name = _extract_town_name(detail)

    hierarchy = get_admin_hierarchy()
    matched_division = hierarchy.resolve_location_parts(
        province_name=province_name,
        city_name=city_name,
        county_name=county_name,
    )

    if matched_division is None:
        matched_regions = hierarchy.find_regions_in_text(raw_text, limit=1)
        if matched_regions:
            matched_division = matched_regions[0]

    explicit_level = None
    if matched_division is not None:
        explicit_level = matched_division.level
    elif county_name:
        explicit_level = "county"
    elif city_name:
        explicit_level = "prefecture"
    elif province_name:
        explicit_level = "province"

    if not any([province_name, city_name, county_name, town_name, village_name, matched_division]):
        return None

    if matched_division is not None:
        path_codes = matched_division.path_codes
        path_names = matched_division.path_names
    else:
        path_codes = ()
        path_names = tuple(name for name in (province_name, city_name, county_name) if name)

    return ResolvedLocation(
        raw_input=raw_text,
        province_name=province_name,
        city_name=city_name,
        county_name=county_name,
        town_name=town_name,
        village_name=village_name,
        detail=detail,
        full_location=full_location,
        matched_division=matched_division,
        explicit_level=explicit_level,
        path_codes=path_codes,
        path_names=path_names,
    )


def extract_query_regions(question: str) -> list[AdminDivision]:
    hierarchy = get_admin_hierarchy()
    matched: list[AdminDivision] = []
    resolved = resolve_location_text(question)
    if resolved is not None and resolved.matched_division is not None:
        matched.append(resolved.matched_division)
    matched.extend(hierarchy.find_regions_in_text(question))
    return _dedupe_regions(matched)


def infer_jurisdiction(title: str, raw_category: str | None) -> dict[str, Any]:
    hierarchy = get_admin_hierarchy()
    region = hierarchy.match_title_region(title)
    normalized_category = str(raw_category or "").strip()
    local_category = is_local_category(normalized_category) or (
        region is not None and normalized_category.lower() in {"", "nan", "none"}
    )

    if not local_category:
        return {
            "jurisdiction_type": "national",
            "jurisdiction_scope": "national",
            "jurisdiction_rank": JURISDICTION_RANK["national"],
            "region_code": None,
            "region_name": None,
            "region_path_codes": [],
            "region_path_names": [],
        }

    if region is None:
        return {
            "jurisdiction_type": "local",
            "jurisdiction_scope": "local_unknown",
            "jurisdiction_rank": JURISDICTION_RANK["local_unknown"],
            "region_code": None,
            "region_name": None,
            "region_path_codes": [],
            "region_path_names": [],
        }

    return {
        "jurisdiction_type": "local",
        "jurisdiction_scope": region.level,
        "jurisdiction_rank": JURISDICTION_RANK[region.level],
        "region_code": region.code,
        "region_name": region.name,
        "region_path_codes": list(region.path_codes),
        "region_path_names": list(region.path_names),
    }


def compare_region_relation(query_regions: list[AdminDivision], document_region_codes: list[str]) -> str:
    if not query_regions or not document_region_codes:
        return "none"

    document_code = document_region_codes[-1]
    document_region_set = set(document_region_codes)
    best_relation = "none"
    relation_order = {
        "none": 0,
        "descendant": 1,
        "ancestor": 2,
        "exact": 3,
    }
    for region in query_regions:
        if document_code == region.code:
            relation = "exact"
        elif document_code in region.path_codes[:-1]:
            relation = "ancestor"
        elif region.code in document_region_set:
            relation = "descendant"
        else:
            relation = "none"
        if relation_order[relation] > relation_order[best_relation]:
            best_relation = relation
    return best_relation