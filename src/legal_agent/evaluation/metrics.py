from __future__ import annotations

import re
from collections import Counter
from typing import Any

from legal_agent.utils.text import simple_tokenize


ACTION_COUNT_RE = re.compile(r"^Action:", re.M)
OBS_COUNT_RE = re.compile(r"^Observation:", re.M)
NORMALIZE_RE = re.compile(r"[\s，。、“”‘’：:；;,！!？?（）()【】\[\]《》<>]+")

OVERALL_SCORE_WEIGHTS = {
    "answer_quality": 0.40,
    "citation_hit_rate": 0.15,
    "tool_use_accuracy": 0.15,
    "format_compliance": 0.10,
    "task_completion": 0.10,
    "error_recovery": 0.10,
}

TASK_COMPLETION_THRESHOLDS = {
    "answer_quality": 0.68,
    "semantic_similarity_with_citation": 0.82,
}


def normalize_answer_text(text: str) -> str:
    return NORMALIZE_RE.sub("", text or "").lower()


def exact_match_score(prediction: str, reference: str) -> float:
    normalized_prediction = normalize_answer_text(prediction)
    normalized_reference = normalize_answer_text(reference)
    if not normalized_prediction or not normalized_reference:
        return 0.0
    return 1.0 if normalized_prediction == normalized_reference else 0.0


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = simple_tokenize(prediction)
    ref_tokens = simple_tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    common = pred_counter & ref_counter
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def format_compliance(trace: str) -> float:
    action_count = len(ACTION_COUNT_RE.findall(trace))
    observation_count = len(OBS_COUNT_RE.findall(trace))
    if "Final Answer:" not in trace:
        return 0.0
    if action_count == 0:
        return 1.0
    return 1.0 if observation_count >= action_count else 0.0


def tool_use_accuracy(tool_history: list[dict[str, Any]], expected_tools: list[str]) -> float:
    predicted = [item["tool_name"] for item in tool_history]
    if not expected_tools:
        return 1.0 if not predicted else 0.0
    matches = 0
    for pred, gold in zip(predicted, expected_tools):
        if pred == gold:
            matches += 1
    prefix_score = matches / len(expected_tools)

    predicted_counter = Counter(predicted)
    gold_counter = Counter(expected_tools)
    overlap = sum((predicted_counter & gold_counter).values())
    if overlap == 0:
        bag_f1 = 0.0
    else:
        precision = overlap / max(1, len(predicted))
        recall = overlap / max(1, len(expected_tools))
        bag_f1 = 2 * precision * recall / (precision + recall)
    return 0.6 * prefix_score + 0.4 * bag_f1


def citation_hit_rate(answer: str, references: list[str]) -> float | None:
    if not references:
        return None
    hits = 0
    for ref in references:
        title_start = ref.find("《")
        title_end = ref.find("》")
        if title_start == -1 or title_end == -1:
            continue
        title = ref[title_start : title_end + 1]
        if title in answer:
            hits += 1
    return hits / max(1, len(references))


def error_recovery_score(trace: str, force_error: bool) -> float | None:
    if not force_error:
        return None
    if "error" not in trace.lower() and "失败" not in trace:
        return 0.0
    last_error_pos = max(trace.lower().rfind("error"), trace.rfind("失败"))
    tail = trace[last_error_pos:]
    return 1.0 if "Action:" in tail and "Final Answer:" in tail else 0.0


def answer_quality_score(exact_match: float, answer_f1: float, semantic_similarity: float) -> float:
    return 0.25 * exact_match + 0.35 * answer_f1 + 0.40 * semantic_similarity


def overall_score(row: dict[str, Any]) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for metric_name, weight in OVERALL_SCORE_WEIGHTS.items():
        value = row.get(metric_name)
        if value is None:
            continue
        weighted_sum += weight * float(value)
        total_weight += weight
    if total_weight <= 0:
        return 0.0
    return weighted_sum / total_weight


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "format_compliance": 0.0,
            "tool_use_accuracy": 0.0,
            "task_completion": 0.0,
            "citation_hit_rate": 0.0,
            "error_recovery": 0.0,
            "answer_exact_match": 0.0,
            "answer_f1": 0.0,
            "semantic_similarity": 0.0,
            "answer_quality": 0.0,
            "overall_score": 0.0,
        }
    metric_names = [
        "format_compliance",
        "tool_use_accuracy",
        "task_completion",
        "citation_hit_rate",
        "error_recovery",
        "answer_exact_match",
        "answer_f1",
        "semantic_similarity",
        "answer_quality",
        "overall_score",
    ]
    return {
        metric_name: (
            sum(float(row[metric_name]) for row in rows if row.get(metric_name) is not None)
            / max(1, sum(1 for row in rows if row.get(metric_name) is not None))
        )
        for metric_name in metric_names
    }
