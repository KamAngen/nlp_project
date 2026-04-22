from legal_agent.utils.text import extract_cross_references, generate_title_aliases, is_local_category, normalize_category, normalize_title


def test_normalize_title_removes_date_suffix_and_plus_signs():
    raw = "最高人民法院+最高人民检察院关于适用《中华人民共和国刑法》第三百四十四条有关问题的批复_20200101"
    normalized = normalize_title(raw)
    assert normalized == "最高人民法院 最高人民检察院关于适用《中华人民共和国刑法》第三百四十四条有关问题的批复"


def test_normalize_title_strips_surrounding_book_quotes():
    assert normalize_title("《中华人民共和国民法典》") == "中华人民共和国民法典"


def test_generate_title_aliases_include_short_prc_title():
    aliases = generate_title_aliases("中华人民共和国民事诉讼法")

    assert aliases == ["中华人民共和国民事诉讼法", "民事诉讼法"]


def test_extract_cross_references_returns_unique_titles():
    text = "依据《中华人民共和国个人所得税法》和《中华人民共和国个人所得税法实施条例》处理；同时参照《中华人民共和国个人所得税法》。"
    refs = extract_cross_references(text)
    assert refs == ["中华人民共和国个人所得税法", "中华人民共和国个人所得税法实施条例"]


def test_local_categories_map_to_regulation_level():
    assert normalize_category("地方性法规") == "法规"
    assert normalize_category("地方法规") == "法规"
    assert is_local_category("法规性决定") is True


def test_normalize_title_replaces_underscores_for_local_law_titles():
    normalized = normalize_title("《湖南省实施_中华人民共和国种子法_办法》")

    assert normalized == "湖南省实施 中华人民共和国种子法 办法"
