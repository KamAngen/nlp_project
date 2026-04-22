from legal_agent.agent.tools import build_location_clarification
from legal_agent.data.admin_divisions import extract_query_regions, resolve_location_text
from legal_agent.rag.retriever import QueryContext, RetrievalHit


def _make_query_context(question: str) -> QueryContext:
    regions = extract_query_regions(question)
    resolution = resolve_location_text(question)
    explicit_level = resolution.explicit_level if resolution is not None else None
    return QueryContext(
        regions=regions,
        has_explicit_region=bool(regions),
        likely_local_question=True,
        authority_boost=False,
        explicit_region_level=explicit_level,
        location_resolution=resolution,
    )


def _make_hit(title: str, path_codes: list[str], path_names: list[str]) -> RetrievalHit:
    return RetrievalHit(
        chunk_id=f"{title}::0001::01",
        score=1.0,
        document_title=title,
        article_heading=None,
        effect_level="法规",
        jurisdiction_type="local",
        jurisdiction_scope="prefecture",
        region_name=path_names[-1],
        region_path_codes=path_codes,
        region_path_names=path_names,
        text="示例正文",
        source_path="data/law_files/files/example.docx",
    )


def test_location_clarification_requests_city_under_province_query():
    query_context = _make_query_context("我在浙江省，想了解港口船舶污染物管理规定")
    hits = [
        _make_hit("舟山市港口船舶污染物管理条例", ["33", "3309"], ["浙江省", "舟山市"]),
    ]

    payload = build_location_clarification(query_context, hits)

    assert payload["needs_location_clarification"] is True
    assert "所在市" in payload["location_clarification_question"]
    assert "街道" in payload["location_clarification_question"]


def test_location_clarification_skips_when_county_already_known_from_street():
    query_context = _make_query_context("我住在徐州市云龙区彭城街道，养犬管理有什么规定")
    hits = [
        _make_hit("徐州市养犬管理条例", ["32", "3203"], ["江苏省", "徐州市"]),
    ]

    payload = build_location_clarification(query_context, hits)

    assert payload["needs_location_clarification"] is False