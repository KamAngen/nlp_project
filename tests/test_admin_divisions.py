from legal_agent.data.admin_divisions import compare_region_relation, extract_query_regions, infer_jurisdiction, query_likely_needs_local_law, resolve_location_text


def test_infer_jurisdiction_for_province_level_local_law():
    payload = infer_jurisdiction("黑龙江省城市供热条例", "地方性法规")

    assert payload["jurisdiction_type"] == "local"
    assert payload["jurisdiction_scope"] == "province"
    assert payload["region_name"] == "黑龙江省"
    assert payload["region_path_names"] == ["黑龙江省"]


def test_compare_region_relation_prefers_province_law_for_city_query():
    query_regions = extract_query_regions("哈尔滨市小区里能不能放烟花")
    payload = infer_jurisdiction("黑龙江省烟花爆竹安全管理条例", "地方性法规")

    assert query_likely_needs_local_law("哈尔滨市小区里能不能放烟花") is True
    assert compare_region_relation(query_regions, payload["region_path_codes"]) == "ancestor"


def test_resolve_location_text_promotes_street_to_county_hierarchy():
    resolution = resolve_location_text("徐州市云龙区彭城街道")

    assert resolution is not None
    assert resolution.province_name == "江苏省"
    assert resolution.city_name == "徐州市"
    assert resolution.county_name == "云龙区"
    assert resolution.town_name == "彭城街道"
    assert list(resolution.path_names) == ["江苏省", "徐州市", "云龙区"]


def test_extract_query_regions_understands_street_level_address():
    regions = extract_query_regions("我住在徐州市云龙区彭城街道，想了解养犬管理规定")

    assert regions
    assert regions[0].name == "云龙区"