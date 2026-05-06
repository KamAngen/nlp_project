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


def test_parse_prepare_context_positional_arg_for_study_tool():
    raw_text = (
        "Thought: 先整理用户上下文。\n"
        'Action: prepare_context("押金到期不退怎么办？")'
    )

    parsed = parse_react_output(raw_text)

    assert parsed.kind == "tool"
    assert parsed.tool_name == "prepare_context"
    assert parsed.tool_args == {"query": "押金到期不退怎么办？"}


def test_parse_ask_followup_positional_args_for_study_tool():
    raw_text = (
        "Thought: 需要补充押金性质。\n"
        'Action: ask_followup("押金的性质是什么？", "押金性质")'
    )

    parsed = parse_react_output(raw_text)

    assert parsed.kind == "tool"
    assert parsed.tool_name == "ask_followup"
    assert parsed.tool_args == {
        "question": "押金的性质是什么？",
        "slot": "押金性质",
    }
