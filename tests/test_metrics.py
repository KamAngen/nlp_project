from legal_agent.evaluation.judge import HeuristicJudge
from legal_agent.evaluation.metrics import citation_hit_rate, error_recovery_score, overall_score, tool_use_accuracy


def test_tool_use_accuracy_penalizes_unnecessary_tools_for_direct_task():
    score = tool_use_accuracy(
        [{"tool_name": "retrieve_from_kb", "tool_args": {}, "result": {}}],
        [],
    )

    assert score == 0.0


def test_non_applicable_metrics_are_excluded_from_overall_score():
    row = {
        "answer_quality": 0.8,
        "citation_hit_rate": None,
        "tool_use_accuracy": 0.7,
        "format_compliance": 1.0,
        "task_completion": 1.0,
        "error_recovery": None,
    }

    score = overall_score(row)

    assert 0.83 < score < 0.84


def test_citation_and_error_recovery_return_none_when_not_applicable():
    assert citation_hit_rate("任意答案", []) is None
    assert error_recovery_score("Final Answer: 直接回答", False) is None


def test_heuristic_judge_handles_samples_without_references():
    payload = HeuristicJudge().judge("这是一个直接答案", "这是一个直接答案", [])

    assert payload["is_correct"] is True
    assert payload["citation_score"] is None