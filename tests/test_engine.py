from types import SimpleNamespace

from legal_agent.agent.engine import LegalAgentEngine
from legal_agent.agent.parser import ParsedStep


class _DummyModel:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or ["Final Answer: 占位答案"]
        self.index = 0

    def _is_turn_analysis_request(self, messages) -> bool:
        if not messages:
            return False
        system_text = str(messages[0].get("content") or "")
        return "回合分析器" in system_text

    def _next_response(self) -> str:
        response = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return response

    def generate(self, messages, **kwargs):
        if self._is_turn_analysis_request(messages):
            text = (
                '{"current_input_role":"new_question","user_goal":"占位目标","needs_history":false,'
                '"history_usage":"以当前问题为主。","requires_precise_result":false,'
                '"preferred_answer_style":"brief_direct","likely_missing_info":[],"recommended_next_step":"unclear"}'
            )
        else:
            text = self._next_response()

        class _Output:
            def __init__(self, text: str) -> None:
                self.raw_text = text
                self.content = text
                self.reasoning = ""

        return _Output(text)

    def stream_generate(self, messages, **kwargs):
        text = self._next_response()
        parts = [text[: max(1, len(text) // 2)], text]
        seen = ""
        for part in parts:
            if part == seen:
                continue
            seen = part

            class _Output:
                def __init__(self, text: str) -> None:
                    self.raw_text = text
                    self.content = text
                    self.reasoning = ""

            yield _Output(part)


class _DummyRegistry:
    def tool_definitions(self):
        return []

    def execute(self, tool_name, tool_args):
        return {
            "query": tool_args.get("query"),
            "results": [
                {
                    "document_title": "中华人民共和国道路交通安全法",
                    "article_heading": "第七十条",
                    "effect_level": "法律",
                    "text": "在道路上发生交通事故，车辆驾驶人应当立即停车，保护现场。",
                    "source_path": "law.docx",
                    "score": 1.0,
                }
            ],
        }


class _LocalityRegistry(_DummyRegistry):
    def execute(self, tool_name, tool_args):
        return {
            "query": tool_args.get("query"),
            "needs_location_clarification": True,
            "location_clarification_question": "请补充你所在的省、市或区县，以便判断地方性法规是否适用。",
            "results": [
                {
                    "document_title": "黑龙江省烟花爆竹安全管理条例",
                    "article_heading": None,
                    "effect_level": "法规",
                    "text": "县级以上人民政府可以划定禁放区域。",
                    "source_path": "law.docx",
                    "score": 0.92,
                }
            ],
        }


class _PreciseLocationRegistry(_DummyRegistry):
    def __init__(self) -> None:
        self.retriever = SimpleNamespace(
            inspect_query=lambda query: SimpleNamespace(
                has_explicit_region=True,
                explicit_region_level="town",
                location_resolution=SimpleNamespace(
                    county_name="云龙区",
                    town_name="彭城街道",
                    village_name=None,
                ),
            )
        )


def _build_engine() -> LegalAgentEngine:
    return LegalAgentEngine(_DummyModel(), _DummyRegistry(), max_steps=4)


def test_build_messages_include_history_and_scratchpad():
    engine = _build_engine()
    state = {
        "question": "补充：事故里有人死亡。",
        "history": [("原始问题", "上一轮回答")],
        "scratchpad": "Thought: 先检索。\nObservation: 已找到交通肇事相关法条。",
    }

    messages = engine._build_messages(state)

    assert messages[1] == {"role": "user", "content": "原始问题"}
    assert messages[2] == {"role": "assistant", "content": "上一轮回答"}
    assert messages[-2] == {"role": "assistant", "content": state["scratchpad"]}
    assert "当前窗口中的完整对话历史" in messages[-3]["content"]


def test_first_step_comes_from_model_output_not_forced_rule():
    model = _DummyModel(
        [
            'Thought: 我还缺少一个会直接影响结论的事实。\nAction: ask_user({"question": "本次解除劳动关系的原因、你的月工资和工作年限分别是什么？", "field_name": "termination_compensation_facts"})'
        ]
    )
    engine = LegalAgentEngine(model, _DummyRegistry(), max_steps=4)

    state = engine._llm_node(engine._initial_state("请精确算一下公司辞退我大概要给多少补偿，我目前只知道被辞退了。"))

    assert state["parsed_kind"] == "tool"
    assert state["parsed_payload"]["tool_name"] == "ask_user"
    assert state["parsed_payload"]["tool_args"]["question"] == "本次解除劳动关系的原因、你的月工资和工作年限分别是什么？"


def test_explicit_statute_title_is_not_forced_to_lookup_before_model_planning():
    model = _DummyModel(
        [
            'Thought: 我先检索与经济补偿直接相关的规范内容。\nAction: retrieve_from_kb({"query": "劳动合同法 经济补偿", "top_k": 6})'
        ]
    )
    engine = LegalAgentEngine(model, _DummyRegistry(), max_steps=4)

    state = engine._llm_node(engine._initial_state("《中华人民共和国劳动合同法》里关于经济补偿是怎么规定的？"))

    assert state["parsed_kind"] == "tool"
    assert state["parsed_payload"]["tool_name"] == "retrieve_from_kb"


def test_followup_input_keeps_model_generated_clarification_when_not_repeated():
    engine = _build_engine()
    parsed = ParsedStep(
        kind="tool",
        thought="仍需追问。",
        tool_name="ask_user",
        tool_args={"question": "请补充事故发生时间、地点和结果。", "field_name": "accident_facts"},
    )
    state = {
        "question": (
            "原始法律问题：小明刚刚在路上撞了人，要判几年？\n"
            "用户刚刚补充了以下事实，请在同一问题上继续分析；若这些事实已经足够支撑条件式分析，请直接给出结论，不要机械重复 ask_user。\n"
            "已补充事实：\n- 不知道，他跑了，好像有个孕妇死了"
        ),
        "tool_history": [
            {
                "tool_name": "ask_user",
                "tool_args": {"question": "请先补充事故是否造成人员死亡、重伤，司机是否逃逸。"},
                "result": {"status": "pending_user_input"},
            }
        ],
        "scratchpad": "Thought: 先检索。",
    }

    updated = engine._postprocess_parsed_step(state, parsed)

    assert updated.kind == "tool"
    assert updated.tool_name == "ask_user"
    assert updated.tool_args["question"] == "请补充事故发生时间、地点和结果。"


def test_repeated_clarification_forces_final_answer():
    engine = LegalAgentEngine(_DummyModel(["Final Answer: 条件式分析结果"]), _DummyRegistry(), max_steps=4)
    parsed = ParsedStep(
        kind="tool",
        thought="继续追问。",
        tool_name="ask_user",
        tool_args={"question": "请补充事故是否造成人员死亡、重伤，司机是否逃逸。", "field_name": "accident_facts"},
    )
    state = {
        "question": "小明刚刚在路上撞了人，要判几年？",
        "tool_history": [
            {
                "tool_name": "ask_user",
                "tool_args": {"question": "请补充事故是否造成人员死亡、重伤，司机是否逃逸。"},
                "result": {"status": "pending_user_input"},
            },
            {
                "tool_name": "ask_user",
                "tool_args": {"question": "请补充事故是否造成人员死亡、重伤，司机是否逃逸。"},
                "result": {"status": "pending_user_input"},
            },
        ],
        "scratchpad": "Observation: 已多次追问。",
        "history": [],
        "errors": [],
        "step_count": 1,
        "llm_retry_count": 0,
    }

    updated = engine._postprocess_parsed_step(state, parsed)

    assert updated.kind == "final"
    assert "条件式分析结果" in updated.final_answer


def test_run_with_updates_emits_partial_trace_and_final_result():
    model = _DummyModel(
        [
            'Thought: 先检索事故定性。\nAction: retrieve_from_kb({"query": "交通事故 逃逸 死亡 量刑", "top_k": 6})',
            "Final Answer: 可能涉及交通肇事后逃逸并致人死亡，应结合死亡结果、逃逸事实和责任划分综合判断量刑。",
        ]
    )
    engine = LegalAgentEngine(model, _DummyRegistry(), max_steps=4)

    updates = list(engine.run_with_updates("小明刚刚在路上撞了人，要判几年？"))

    assert any(update["event"] == "llm_partial" for update in updates)
    assert updates[-1]["event"] == "final"
    assert "交通肇事后逃逸" in updates[-1]["result"].final_answer


def test_uncertain_final_answer_is_converted_to_followup_question():
    engine = LegalAgentEngine(
        _DummyModel(
            [
                'Thought: 还缺少一个会直接改变结论的事实。\nAction: ask_user({"question": "请补充解除原因、你的月工资和工作年限。", "field_name": "termination_compensation_facts"})'
            ]
        ),
        _DummyRegistry(),
        max_steps=4,
    )
    parsed = ParsedStep(
        kind="final",
        thought="现有信息不足。",
        final_answer="当前仍缺少会直接影响补偿结论的关键信息，因此暂时无法精确估算补偿或赔偿金额。",
    )
    state = {
        "question": "公司突然辞退我，请你为我精确地估一下补偿或赔偿金额。",
        "history": [],
        "tool_history": [],
        "scratchpad": "Observation: 已检索相关法条。",
    }

    updated = engine._postprocess_parsed_step(state, parsed)

    assert updated.kind == "tool"
    assert updated.tool_name == "ask_user"
    assert updated.tool_args["question"] == "请补充解除原因、你的月工资和工作年限。"


def test_retrieval_query_uses_only_supplemental_answers_not_prior_questions():
    engine = _build_engine()
    question = (
        "原始法律问题：你能给估一下具体能赔多少钱吗？\n"
        "用户刚刚补充了以下事实，请在同一问题上继续分析；若这些事实已经足够支撑条件式分析，请直接给出结论，不要机械重复 ask_user。\n"
        "已补充事实：\n"
        "- 请补充事故责任比例、医疗费、误工期等信息：交警认定对方全责，我已经花了2万元医疗费，误工30天"
    )

    query = engine._build_retrieval_query(question)

    assert "交警认定对方全责" in query
    assert "请补充事故责任比例" not in query


def test_second_identical_tool_call_is_cut_off_before_execution():
    engine = _build_engine()
    parsed = ParsedStep(
        kind="tool",
        thought="继续检索。",
        tool_name="retrieve_from_kb",
        tool_args={"query": "交通事故 全责 医疗费 2万 误工30天 赔偿", "top_k": 6},
    )
    state = {
        "question": "交通事故对方全责，我花了2万元医疗费，误工30天，你来估一下赔偿。",
        "tool_history": [
            {
                "tool_name": "retrieve_from_kb",
                "tool_args": {"query": "交通事故 全责 医疗费 2万 误工30天 赔偿", "top_k": 6},
                "result": {"results": []},
            }
        ],
        "scratchpad": "Observation: 已检索一次。",
        "history": [],
        "errors": [],
        "step_count": 1,
        "llm_retry_count": 0,
    }

    updated = engine._postprocess_parsed_step(state, parsed)

    assert updated.kind == "final"


def test_run_with_updates_emits_informative_live_message():
    model = _DummyModel(
        [
            'Thought: 先检索事故定性。\nAction: retrieve_from_kb({"query": "交通事故 逃逸 死亡 量刑", "top_k": 6})',
            "Final Answer: 可能涉及交通肇事后逃逸并致人死亡。",
        ]
    )
    engine = LegalAgentEngine(model, _DummyRegistry(), max_steps=4)

    updates = list(engine.run_with_updates("小明刚刚在路上撞了人，要判几年？"))

    live_messages = [update.get("message") for update in updates if update["event"] == "llm_partial"]
    assert any(message == "正在准备调用工具：retrieve_from_kb" for message in live_messages)


def test_location_ambiguity_after_retrieval_triggers_followup():
    engine = LegalAgentEngine(_DummyModel(), _LocalityRegistry(), max_steps=4)
    state = {
        "question": "小区里能不能放烟花？",
        "history": [],
        "scratchpad": 'Thought: 先检索相关法规。\nAction: retrieve_from_kb({"query": "小区里能不能放烟花", "top_k": 6})',
        "parsed_payload": {"tool_name": "retrieve_from_kb", "tool_args": {"query": "小区里能不能放烟花", "top_k": 6}},
        "tool_history": [],
        "step_count": 0,
        "errors": [],
    }

    updated = engine._tool_node(state)

    assert updated["needs_user_input"] is True
    assert "省、市或区县" in updated["clarification_question"]


def test_document_summary_request_can_be_answered_directly_by_model():
    model = _DummyModel(
        [
            "Final Answer: 这是一份租赁合同纠纷的一审民事判决书。法院认定被告拖欠租金构成违约，支持原告关于租金和逾期付款违约责任的主要请求，并驳回其余请求。",
        ]
    )
    engine = LegalAgentEngine(model, _DummyRegistry(), max_steps=4)
    question = (
        "请大致描述这篇文书的内容：\n"
        "余姚市之江外贸有限公司与章超租赁合同纠纷一审民事判决书\n"
        "原告：余姚市之江外贸有限公司。\n"
        "被告：章超。\n"
        "本院认为：依法成立的合同，对当事人具有法律约束力。\n"
        "判决如下：一、被告章超支付原告租金49512元并承担逾期付款违约责任；二、驳回其他诉讼请求。\n"
        "审判员 朱章程\n"
        "书记员 史钟林\n"
        "2017年5月16日。"
    )

    result = engine.run(question)

    assert "租赁合同纠纷" in result.final_answer


def test_lookup_statute_title_suffix_is_sanitized():
    engine = _build_engine()
    parsed = ParsedStep(
        kind="tool",
        thought="需要确认法规元数据。",
        tool_name="lookup_statute",
        tool_args={"title": "中华人民共和国合同法::0001::01"},
    )
    state = {
        "question": "这条法条的标题是什么？",
        "tool_history": [],
        "scratchpad": "",
        "history": [],
        "errors": [],
        "step_count": 0,
        "llm_retry_count": 0,
    }

    updated = engine._postprocess_parsed_step(state, parsed)

    assert updated.kind == "tool"
    assert updated.tool_args["title"] == "中华人民共和国合同法"


def test_precise_location_followup_is_converted_to_retrieval():
    engine = LegalAgentEngine(_DummyModel(), _PreciseLocationRegistry(), max_steps=4)
    parsed = ParsedStep(
        kind="tool",
        thought="还需要确认地点。",
        tool_name="ask_user",
        tool_args={"question": "请补充你所在的省、市或区县，以便提供具体的地方性养犬管理规定。", "field_name": "user_location"},
    )
    state = {
        "question": "徐州市云龙区彭城街道",
        "history": [("养犬管理有哪些地方性规定？", "请补充地点。")],
        "turn_analysis": {"current_input_role": "answer_to_followup"},
        "tool_history": [
            {
                "tool_name": "ask_user",
                "tool_args": {"question": "请补充你所在的省、市或区县，以便提供具体的地方性养犬管理规定。"},
                "result": {"status": "pending_user_input"},
            }
        ],
        "scratchpad": "Observation: 用户已补充地点。",
        "errors": [],
        "step_count": 1,
        "llm_retry_count": 0,
    }

    updated = engine._postprocess_parsed_step(state, parsed)

    assert updated.kind == "tool"
    assert updated.tool_name == "retrieve_from_kb"
    assert "养犬管理有哪些地方性规定" in updated.tool_args["query"]
    assert "徐州市云龙区彭城街道" in updated.tool_args["query"]