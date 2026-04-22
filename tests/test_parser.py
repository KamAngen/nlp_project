from legal_agent.agent.parser import parse_react_output


def test_parse_prefers_action_before_final_answer():
    raw_text = (
        "Thought: 需要先定位相关法规。\n"
        'Action: lookup_statute("中华人民共和国劳动合同法")\n'
        'Observation: {"status": "mock"}\n'
        "Final Answer: 这是一个占位答案。"
    )

    parsed = parse_react_output(raw_text)

    assert parsed.kind == "tool"
    assert parsed.tool_name == "lookup_statute"
    assert parsed.tool_args == {"title": "中华人民共和国劳动合同法"}


def test_parse_mixed_positional_and_keyword_args():
    raw_text = (
        "Thought: 需要检索相关法条。\n"
        'Action: retrieve_from_kb("解除劳动合同 经济补偿", top_k=3, effect_level="法律")'
    )

    parsed = parse_react_output(raw_text)

    assert parsed.kind == "tool"
    assert parsed.tool_name == "retrieve_from_kb"
    assert parsed.tool_args == {
        "query": "解除劳动合同 经济补偿",
        "top_k": 3,
        "effect_level": "法律",
    }
