from __future__ import annotations

from typing import TypedDict, Any

from langgraph.graph import END, StateGraph


class AgentState(TypedDict, total=False):
    question: str
    history: list[tuple[str, str]]
    scratchpad: str
    step_count: int
    llm_retry_count: int
    max_steps: int
    parsed_kind: str
    parsed_payload: dict[str, Any]
    raw_output: str
    final_answer: str
    tool_history: list[dict[str, Any]]
    errors: list[str]
    needs_user_input: bool
    clarification_question: str


def build_agent_graph(engine: Any):
    graph = StateGraph(AgentState)
    graph.add_node("llm", engine._llm_node)
    graph.add_node("tool", engine._tool_node)
    graph.add_conditional_edges(
        "llm",
        engine._route_after_llm,
        {
            "tool": "tool",
            "retry": "llm",
            "final": END,
        },
    )
    graph.add_conditional_edges(
        "tool",
        engine._route_after_tool,
        {
            "llm": "llm",
            "final": END,
        },
    )
    graph.set_entry_point("llm")
    return graph.compile()
