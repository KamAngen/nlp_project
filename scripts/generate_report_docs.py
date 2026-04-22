from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any

from legal_agent.config import AppConfig, load_app_config
from legal_agent.utils.io import read_json, read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = PROJECT_ROOT / "docs"

TASK_FAMILY_LABELS_EN = {
    "exam": "exam QA",
    "jud_doc_sum": "judgment summarization",
    "jud_read_compre": "judgment reading comprehension",
    "judgement_predit": "judgment prediction",
    "leg_case_cls": "legal case classification",
    "leg_eve_detec": "legal event detection",
    "legal_question_answering": "legal question answering",
    "op_sum": "document/news summarization",
    "sent_pred": "sentiment prediction",
    "sim_case_match": "similar case matching",
}

TASK_FAMILY_LABELS_ZH = {
    "exam": "法考题",
    "jud_doc_sum": "判决书摘要",
    "jud_read_compre": "判决阅读理解",
    "judgement_predit": "判决推断",
    "leg_case_cls": "法律问题分类",
    "leg_eve_detec": "法律事件检测",
    "legal_question_answering": "法律问答",
    "op_sum": "文书/资讯摘要",
    "sent_pred": "情感判断",
    "sim_case_match": "相似案例匹配",
}

STRATEGY_LABELS_EN = {
    "direct_answer": "direct-answer",
    "retrieve_then_answer": "retrieve-then-answer",
    "lookup_then_retrieve": "lookup-then-retrieve",
}

STRATEGY_LABELS_ZH = {
    "direct_answer": "直接回答",
    "retrieve_then_answer": "先检索再回答",
    "lookup_then_retrieve": "先精确查法再检索",
}

METRIC_SPECS = {
    "format_compliance": {
        "label_en": "Format compliance",
        "label_zh": "格式遵从率",
        "note_en": "Checks whether the trace preserves the required Thought/Action/Observation/Final Answer shell and whether every Action is paired with an Observation.",
        "note_zh": "检查轨迹是否保留 Thought/Action/Observation/Final Answer 的 ReAct 外壳，以及每个 Action 是否都有对应的 Observation。",
    },
    "tool_use_accuracy": {
        "label_en": "Tool-use accuracy",
        "label_zh": "工具调用准确率",
        "note_en": "Compares the executed tool sequence against the expected tool plan using prefix matching and bag-level overlap.",
        "note_zh": "将实际执行的工具序列与期望工具轨迹进行比较，综合前缀匹配和工具集合重叠度计算。",
    },
    "task_completion": {
        "label_en": "Task completion",
        "label_zh": "任务完成率",
        "note_en": "A binary success metric triggered when answer quality crosses the threshold, or when citation-aware similarity is high enough on reference-bearing samples.",
        "note_zh": "二值成功指标。当答案质量超过阈值，或在带参考法条的样本上语义相似度和引用命中达到阈值时记为完成。",
    },
    "error_recovery": {
        "label_en": "Error recovery",
        "label_zh": "错误恢复能力",
        "note_en": "Measures whether the agent re-plans after an injected tool/runtime failure instead of terminating immediately.",
        "note_zh": "衡量在注入工具或运行时错误后，Agent 是否会重新规划并继续完成任务，而不是直接失败退出。",
    },
    "citation_hit_rate": {
        "label_en": "Citation hit rate",
        "label_zh": "法条引用命中率",
        "note_en": "Checks whether the final answer explicitly mentions the statute titles contained in the reference set.",
        "note_zh": "检查最终答案是否显式提到参考答案中给出的法规标题。",
    },
    "answer_exact_match": {
        "label_en": "Exact match",
        "label_zh": "精确匹配率",
        "note_en": "A strict normalized exact-match score on the final answer text.",
        "note_zh": "对最终答案文本做归一化后的严格精确匹配。",
    },
    "answer_f1": {
        "label_en": "Token F1",
        "label_zh": "Token F1",
        "note_en": "Measures lexical overlap between the generated answer and the reference answer after tokenization.",
        "note_zh": "对生成答案和参考答案分词后计算词级重合度。",
    },
    "semantic_similarity": {
        "label_en": "Semantic similarity",
        "label_zh": "语义相似度",
        "note_en": "Dense-embedding similarity between the generated answer and the reference answer, used to distinguish paraphrases from genuine failures.",
        "note_zh": "利用稠密向量比较生成答案与参考答案的语义相近程度，用于区分近义改写和真正答错。",
    },
    "answer_quality": {
        "label_en": "Answer quality",
        "label_zh": "答案质量",
        "note_en": "A weighted blend of exact match, token F1, and semantic similarity. This is the main textual-quality score used by the completion heuristic.",
        "note_zh": "由精确匹配、Token F1 和语义相似度加权得到，是任务完成判断最核心的文本质量分。",
    },
    "overall_score": {
        "label_en": "Overall score",
        "label_zh": "综合得分",
        "note_en": "The weighted aggregate required by the assignment. Metrics with zero support are removed from the denominator instead of being forced to zero.",
        "note_zh": "作业要求的加权总分。对当前切分中没有支持样本的指标，会从分母中剔除，而不是机械记为 0。",
    },
}

RUNTIME_VALIDATIONS = [
    {
        "title_en": "Street-to-hierarchy backfill for unseen local mentions",
        "title_zh": "陌生街道名称的层级回填",
        "query": "浙江省舟山市定海区环城南街道港务码头附近，港口船舶污染物管理有哪些规定？",
        "behavior_en": "The address parser can still recover the locality path 浙江省 > 舟山市 > 定海区 even when the street name is not explicitly emitted as a structured field, because the detail span is post-processed and backfilled into the county/city/province hierarchy.",
        "behavior_zh": "即使街道名没有被地址解析器直接放入结构化字段，系统也会从 detail 文本中抽取街道前缀，并回填到 浙江省 > 舟山市 > 定海区 的层级路径。",
        "why_en": "This is the key mechanism that lets the agent search the right local regulations even when the user mentions a street name that is not already enumerated in the county-level law metadata.",
        "why_zh": "这使得 Agent 在用户给出一个知识库元数据中未显式枚举的街道名时，仍然能够退回到对应区、市、省的法规范围继续检索。",
    },
    {
        "title_en": "Province-only ambiguity now triggers finer clarification",
        "title_zh": "仅给省份时会主动追问更细地点",
        "query": "我在浙江省，想了解港口船舶污染物管理规定。",
        "behavior_en": "When the query only provides a province but the retriever sees city-level local regulations underneath that province, the tool layer now asks for city/county/street before committing to a local-law answer.",
        "behavior_zh": "当用户只给出省份，但检索器发现该省下面存在更细粒度的市级地方性法规时，工具层会先追问市、区县或街道，再决定适用哪部地方性法规。",
        "why_en": "This prevents the agent from prematurely answering with the wrong city-specific rule and directly matches the assignment requirement on proactive clarification.",
        "why_zh": "这样可以避免 Agent 过早套用错误的市级法规，也直接满足作业对“模糊地点时要主动追问”的要求。",
    },
    {
        "title_en": "Multi-turn continuation after location clarification",
        "title_zh": "地点追问后的多轮延续分析",
        "query": "养犬管理有哪些地方性规定？\n用户补充：徐州市云龙区彭城街道。",
        "behavior_en": "After the clarification turn, the runtime continues the same legal analysis instead of resetting the conversation. The added street-level location is treated as supplemental evidence and the retriever can pivot to 徐州市养犬管理条例 without repeating the same ask-user turn.",
        "behavior_zh": "在追问地点后的下一轮，运行时会把用户补充的街道信息视为同一法律问题的补充事实，而不是新的问题，从而直接切换到徐州市养犬管理条例的分析。",
        "why_en": "This is important for agent realism: a legal assistant should not lose track of the user’s original question after asking for missing facts.",
        "why_zh": "这体现了多轮 Agent 的真实性：追问缺失事实后，法律分析必须沿着同一个问题继续，而不是丢失上下文。",
    },
    {
        "title_en": "Legacy DOC repair surfaces previously unreachable local regulations",
        "title_zh": "修复 legacy DOC 后可命中新入库地方性法规",
        "query": "舟山市港口船舶污染物管理条例主要规定了哪些内容？",
        "behavior_en": "After repairing legacy Word files and rebuilding the corpus/index, the retriever can directly surface local regulations such as 舟山市港口船舶污染物管理条例 and 舟山市居家养老服务促进条例 that were previously unreachable during parsing failures.",
        "behavior_zh": "在修复旧版 Word 文档并重建语料与索引后，检索器已经可以直接命中舟山市港口船舶污染物管理条例、舟山市居家养老服务促进条例等此前因解析失败而不可用的地方性法规。",
        "why_en": "This qualitative check connects the data-engineering fixes to user-visible behavior and shows that corpus repair is not merely an offline preprocessing detail.",
        "why_zh": "这说明知识库修复不是“只在预处理阶段看起来漂亮”，而是真正改变了最终用户可见的检索与回答质量。",
    },
]


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def clean_excerpt(text: str, limit: int = 280) -> str:
    flattened = " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split())
    if len(flattened) <= limit:
        return flattened
    return flattened[: limit - 1].rstrip(" ，,；;：:") + "…"


def tt(text: str) -> str:
    return f"\\texttt{{{latex_escape(text)}}}"


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def load_json_list_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = read_json(path)
    return payload if isinstance(payload, list) else []


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_metric_value(value: Any) -> str:
    parsed = safe_float(value)
    if parsed is None:
        return "N/A"
    return f"{parsed:.3f}"


def format_delta(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.3f}"


def join_en(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def join_zh(items: list[str]) -> str:
    return "、".join(item for item in items if item)


def latex_itemize(items: list[str]) -> str:
    if not items:
        return ""
    lines = ["\\begin{itemize}[leftmargin=*]"]
    lines.extend(f"\\item {item}" for item in items)
    lines.append("\\end{itemize}")
    return "\n".join(lines)


def latex_quote(text: str, limit: int = 320) -> str:
    excerpt = clean_excerpt(text, limit)
    return "\\begin{quote}\\small " + latex_escape(excerpt) + "\\end{quote}"


def latex_multiline(text: str, limit: int = 900) -> str:
    raw = str(text or "").replace("\r", "").strip()
    if len(raw) > limit:
        raw = raw[: limit - 1].rstrip(" ，,；;：:\n") + "…"
    lines = raw.splitlines() or [raw]
    rendered = [latex_escape(line) if line else r"\," for line in lines]
    return " \\\n".join(rendered)


def counter_text_en(counter: Counter[str], label_map: dict[str, str]) -> str:
    parts = [f"{count} {label_map.get(key, key)}" for key, count in counter.items()]
    return join_en(parts)


def counter_text_zh(counter: Counter[str], label_map: dict[str, str]) -> str:
    parts = [f"{count} 个{label_map.get(key, key)}" for key, count in counter.items()]
    return join_zh(parts)


def tool_sequence_from_history(row: dict[str, Any]) -> str:
    names = [str(item.get("tool_name", "")) for item in row.get("tool_history", []) if item.get("tool_name")]
    if not names:
        return "no tool calls"
    return " -> ".join(names)


def expected_tool_sequence(sample: dict[str, Any]) -> str:
    names = [str(name) for name in sample.get("expected_tools", []) if name]
    if not names:
        return "no tool calls expected"
    return " -> ".join(names)


def tool_sequence_from_history_zh(row: dict[str, Any]) -> str:
    names = [str(item.get("tool_name", "")) for item in row.get("tool_history", []) if item.get("tool_name")]
    if not names:
        return "未调用工具"
    return " -> ".join(names)


def expected_tool_sequence_zh(sample: dict[str, Any]) -> str:
    names = [str(name) for name in sample.get("expected_tools", []) if name]
    if not names:
        return "无需调用工具"
    return " -> ".join(names)


@dataclass(slots=True)
class ExperimentArtifacts:
    config: AppConfig
    training_metrics: dict[str, Any]
    generation_summary: dict[str, Any]
    dataset_summary: dict[str, Any]
    corpus_summary: dict[str, Any]
    rag_metadata: dict[str, Any]
    base_summary: dict[str, Any]
    adapted_summary: dict[str, Any]
    base_rows: list[dict[str, Any]]
    adapted_rows: list[dict[str, Any]]
    sample_map: dict[str, dict[str, Any]]
    tool_behavior_cases: list[dict[str, Any]]
    base_eval_source_en: str
    adapted_eval_source_en: str
    base_eval_source_zh: str
    adapted_eval_source_zh: str
    eval_splits_match: bool


def load_eval_bundle(
    primary_dir: Path,
    fallback_dir: Path,
    sample_map: dict[str, dict[str, Any]],
    *,
    primary_name_en: str,
    fallback_name_en: str,
    primary_name_zh: str,
    fallback_name_zh: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], str, str, str]:
    primary_summary = load_json_if_exists(primary_dir / "eval_summary.json")
    primary_rows = enrich_eval_rows(load_jsonl_if_exists(primary_dir / "eval_details.jsonl"), sample_map)
    if primary_summary or primary_rows:
        return primary_summary, primary_rows, "formal", primary_name_en, primary_name_zh

    fallback_summary = load_json_if_exists(fallback_dir / "eval_summary.json")
    fallback_rows = enrich_eval_rows(load_jsonl_if_exists(fallback_dir / "eval_details.jsonl"), sample_map)
    return fallback_summary, fallback_rows, "smoke", fallback_name_en, fallback_name_zh


def load_experiment_artifacts(config_path: Path) -> ExperimentArtifacts:
    config = load_app_config(config_path)
    sample_rows = load_jsonl_if_exists(config.generated_eval_path)
    sample_map = {row["sample_id"]: row for row in sample_rows if row.get("sample_id")}
    base_summary, base_rows, base_scope, base_source_en, base_source_zh = load_eval_bundle(
        config.output_root / "eval_base",
        config.output_root / "smoke" / "eval_base",
        sample_map,
        primary_name_en="formal held-out evaluation",
        fallback_name_en="latest completed smoke evaluation",
        primary_name_zh="正式 held-out 评测",
        fallback_name_zh="最近完成的 smoke 评测",
    )
    adapted_summary, adapted_rows, adapted_scope, adapted_source_en, adapted_source_zh = load_eval_bundle(
        config.output_root / "eval_adapter",
        config.output_root / "smoke" / "eval_adapter",
        sample_map,
        primary_name_en="formal held-out evaluation",
        fallback_name_en="latest completed smoke/checkpoint evaluation",
        primary_name_zh="正式 held-out 评测",
        fallback_name_zh="最近完成的 smoke/checkpoint 评测",
    )
    return ExperimentArtifacts(
        config=config,
        training_metrics=load_json_if_exists(config.training.output_dir / "training_metrics.json"),
        generation_summary=load_json_if_exists(config.generated_data_dir / "dataset_summary.json"),
        dataset_summary=load_json_if_exists(config.law_dir / "catalogs" / "dataset_summary.json"),
        corpus_summary=load_json_if_exists(config.corpus_summary_path),
        rag_metadata=load_json_if_exists(config.rag_dir / "metadata.json"),
        base_summary=base_summary,
        adapted_summary=adapted_summary,
        base_rows=base_rows,
        adapted_rows=adapted_rows,
        sample_map=sample_map,
        tool_behavior_cases=load_json_list_if_exists(config.output_root / "tool_behavior_cases.json"),
        base_eval_source_en=base_source_en,
        adapted_eval_source_en=adapted_source_en,
        base_eval_source_zh=base_source_zh,
        adapted_eval_source_zh=adapted_source_zh,
        eval_splits_match=base_scope == adapted_scope,
    )


def enrich_eval_rows(rows: list[dict[str, Any]], sample_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        sample = sample_map.get(row.get("sample_id"), {})
        metadata = sample.get("metadata", {})
        merged = dict(row)
        merged["task_family"] = metadata.get("task_family", "unknown")
        merged["strategy"] = metadata.get("strategy", "unknown")
        merged["expected_tools"] = list(sample.get("expected_tools", []))
        merged["has_references"] = bool(sample.get("references"))
        merged["force_error"] = bool(sample.get("force_error"))
        enriched.append(merged)
    return enriched


def build_eval_dataset_stats(samples: list[dict[str, Any]]) -> dict[str, Any]:
    strategy_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    expected_tool_bins: Counter[str] = Counter()
    reference_count = 0
    force_error_count = 0
    scripted_answer_count = 0
    for sample in samples:
        metadata = sample.get("metadata", {})
        strategy_counts[str(metadata.get("strategy", "unknown"))] += 1
        task_counts[str(metadata.get("task_family", "unknown"))] += 1
        tool_count = len(sample.get("expected_tools", []))
        if tool_count == 0:
            expected_tool_bins["0"] += 1
        elif tool_count == 1:
            expected_tool_bins["1"] += 1
        else:
            expected_tool_bins["2+"] += 1
        if sample.get("references"):
            reference_count += 1
        if sample.get("force_error"):
            force_error_count += 1
        if sample.get("scripted_answers"):
            scripted_answer_count += 1
    return {
        "total": len(samples),
        "strategy_counts": strategy_counts,
        "task_counts": task_counts,
        "expected_tool_bins": expected_tool_bins,
        "reference_count": reference_count,
        "force_error_count": force_error_count,
        "scripted_answer_count": scripted_answer_count,
        "tool_required_count": expected_tool_bins.get("1", 0) + expected_tool_bins.get("2+", 0),
    }


def build_tool_usage_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tool_counter: Counter[str] = Counter()
    samples_with_tools = 0
    multi_tool_samples = 0
    ask_user_samples = 0
    for row in rows:
        names = [str(item.get("tool_name", "")) for item in row.get("tool_history", []) if item.get("tool_name")]
        if names:
            samples_with_tools += 1
        if len(names) > 1:
            multi_tool_samples += 1
        if "ask_user" in names:
            ask_user_samples += 1
        tool_counter.update(names)
    return {
        "tool_counter": tool_counter,
        "samples_with_tools": samples_with_tools,
        "multi_tool_samples": multi_tool_samples,
        "ask_user_samples": ask_user_samples,
    }


def paired_rows(base_rows: list[dict[str, Any]], adapted_rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    base_map = {row["sample_id"]: row for row in base_rows if row.get("sample_id")}
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for adapted in adapted_rows:
        base = base_map.get(adapted.get("sample_id"))
        if base is not None:
            pairs.append((base, adapted))
    return pairs


def improvement_score(base: dict[str, Any], adapted: dict[str, Any]) -> float:
    base_overall = safe_float(base.get("overall_score")) or 0.0
    adapted_overall = safe_float(adapted.get("overall_score")) or 0.0
    base_quality = safe_float(base.get("answer_quality")) or 0.0
    adapted_quality = safe_float(adapted.get("answer_quality")) or 0.0
    base_tool = safe_float(base.get("tool_use_accuracy")) or 0.0
    adapted_tool = safe_float(adapted.get("tool_use_accuracy")) or 0.0
    base_completion = safe_float(base.get("task_completion")) or 0.0
    adapted_completion = safe_float(adapted.get("task_completion")) or 0.0
    return (
        1.5 * (adapted_overall - base_overall)
        + 1.0 * (adapted_completion - base_completion)
        + 0.5 * (adapted_quality - base_quality)
        + 0.25 * (adapted_tool - base_tool)
    )


def select_diverse_pairs(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    max_examples: int,
    sort_key,
    reverse: bool,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    ordered = sorted(pairs, key=sort_key, reverse=reverse)
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen_families: set[str] = set()
    for pair in ordered:
        family = str(pair[1].get("task_family", "unknown"))
        if family in seen_families:
            continue
        selected.append(pair)
        seen_families.add(family)
        if len(selected) >= max_examples:
            return selected
    for pair in ordered:
        if pair in selected:
            continue
        selected.append(pair)
        if len(selected) >= max_examples:
            break
    return selected


def select_case_studies(base_rows: list[dict[str, Any]], adapted_rows: list[dict[str, Any]], *, max_examples: int = 3) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs = paired_rows(base_rows, adapted_rows)
    return select_diverse_pairs(
        pairs,
        max_examples=max_examples,
        sort_key=lambda pair: improvement_score(pair[0], pair[1]),
        reverse=True,
    )


def select_failure_cases(base_rows: list[dict[str, Any]], adapted_rows: list[dict[str, Any]], *, max_examples: int = 2) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    pairs = paired_rows(base_rows, adapted_rows)
    return select_diverse_pairs(
        pairs,
        max_examples=max_examples,
        sort_key=lambda pair: (
            safe_float(pair[1].get("task_completion")) or 0.0,
            safe_float(pair[1].get("overall_score")) or 0.0,
            safe_float(pair[1].get("answer_quality")) or 0.0,
        ),
        reverse=False,
    )


def support_value(summary: dict[str, Any], key: str) -> int:
    metric_support = summary.get("metric_support") or {}
    parsed = safe_float(metric_support.get(key))
    if parsed is None:
        return 0
    return int(parsed)


def metric_rows(
    base_summary: dict[str, Any],
    adapted_summary: dict[str, Any],
    *,
    comparable: bool,
) -> list[tuple[str, float | None, int, float | None, int, float | None]]:
    base_metrics = base_summary.get("metrics", {})
    adapted_metrics = adapted_summary.get("metrics", {})
    ordered_keys = list(METRIC_SPECS.keys())
    rows: list[tuple[str, float | None, int, float | None, int, float | None]] = []
    for key in ordered_keys:
        base_value = safe_float(base_metrics.get(key))
        adapted_value = safe_float(adapted_metrics.get(key))
        delta = None if (not comparable or base_value is None or adapted_value is None) else adapted_value - base_value
        rows.append((key, base_value, support_value(base_summary, key), adapted_value, support_value(adapted_summary, key), delta))
    return rows


def infer_total_eval(
    base_summary: dict[str, Any],
    adapted_summary: dict[str, Any],
    base_rows: list[dict[str, Any]],
    adapted_rows: list[dict[str, Any]],
) -> int:
    candidates = [1, len(base_rows), len(adapted_rows)]
    for summary in (base_summary, adapted_summary):
        metric_support = summary.get("metric_support") or {}
        for value in metric_support.values():
            parsed = safe_float(value)
            if parsed is not None:
                candidates.append(int(parsed))
    return max(candidates)


def metric_table_rows_en(base_summary: dict[str, Any], adapted_summary: dict[str, Any], *, comparable: bool) -> str:
    lines: list[str] = []
    for key, base_value, base_support, adapted_value, adapted_support, delta in metric_rows(
        base_summary,
        adapted_summary,
        comparable=comparable,
    ):
        label = METRIC_SPECS[key]["label_en"]
        lines.append(
            f"{latex_escape(label)} & {base_support} & {format_metric_value(base_value)} & {adapted_support} & {format_metric_value(adapted_value)} & {format_delta(delta)} \\\\"
        )
    return "\n".join(lines)


def metric_table_rows_zh(base_summary: dict[str, Any], adapted_summary: dict[str, Any], *, comparable: bool) -> str:
    lines: list[str] = []
    for key, base_value, base_support, adapted_value, adapted_support, delta in metric_rows(
        base_summary,
        adapted_summary,
        comparable=comparable,
    ):
        label = METRIC_SPECS[key]["label_zh"]
        lines.append(
            f"{latex_escape(label)} & {base_support} & {format_metric_value(base_value)} & {adapted_support} & {format_metric_value(adapted_value)} & {format_delta(delta)} \\\\"
        )
    return "\n".join(lines)


def metric_note_items_en(base_summary: dict[str, Any], adapted_summary: dict[str, Any], total_eval: int) -> list[str]:
    items: list[str] = []
    for key, base_value, base_support, adapted_value, adapted_support, delta in metric_rows(
        base_summary,
        adapted_summary,
        comparable=True,
    ):
        note = METRIC_SPECS[key]["note_en"]
        label = METRIC_SPECS[key]["label_en"]
        if base_support == 0 and adapted_support == 0:
            items.append(
                latex_escape(f"{label} (base {base_support}/{total_eval}, adapter {adapted_support}/{total_eval}): {note} This metric is inactive on the current split and is therefore excluded from the effective denominator of the overall score.")
            )
        else:
            items.append(
                latex_escape(f"{label} (base {base_support}/{total_eval}, adapter {adapted_support}/{total_eval}): {note} Base={format_metric_value(base_value)}, adapter={format_metric_value(adapted_value)}, delta={format_delta(delta)}.")
            )
    return items


def metric_note_items_zh(base_summary: dict[str, Any], adapted_summary: dict[str, Any], total_eval: int) -> list[str]:
    items: list[str] = []
    for key, base_value, base_support, adapted_value, adapted_support, delta in metric_rows(
        base_summary,
        adapted_summary,
        comparable=True,
    ):
        note = METRIC_SPECS[key]["note_zh"]
        label = METRIC_SPECS[key]["label_zh"]
        if base_support == 0 and adapted_support == 0:
            items.append(
                latex_escape(f"{label}（基座 {base_support}/{total_eval}，后训练 {adapted_support}/{total_eval}）：{note} 该指标在当前切分上没有支持样本，因此不会被强行记为 0，而是从综合分的有效分母中剔除。")
            )
        else:
            items.append(
                latex_escape(f"{label}（基座 {base_support}/{total_eval}，后训练 {adapted_support}/{total_eval}）：{note} 基座={format_metric_value(base_value)}，后训练={format_metric_value(adapted_value)}，差值={format_delta(delta)}。")
            )
    return items


def build_failure_reason_summary(rows: list[dict[str, Any]]) -> Counter[str]:
    reasons: Counter[str] = Counter()
    for row in rows:
        if (safe_float(row.get("task_completion")) or 0.0) >= 1.0:
            continue
        expected_tools = row.get("expected_tools", [])
        tool_accuracy = safe_float(row.get("tool_use_accuracy")) or 0.0
        citation = row.get("citation_hit_rate")
        semantic = safe_float(row.get("semantic_similarity")) or 0.0
        answer_quality = safe_float(row.get("answer_quality")) or 0.0
        if expected_tools and tool_accuracy < 0.5:
            reasons["tool mismatch"] += 1
        if citation is not None and safe_float(citation) == 0.0:
            reasons["citation miss"] += 1
        if semantic < 0.75:
            reasons["retrieval or reasoning drift"] += 1
        elif answer_quality < 0.68:
            reasons["partial but not complete answer"] += 1
        if row.get("errors"):
            reasons["runtime error"] += 1
        if not row.get("errors") and not expected_tools and semantic >= 0.75 and answer_quality < 0.68:
            reasons["threshold miss on direct-answer task"] += 1
    return reasons


def build_failure_reason_summary_zh(rows: list[dict[str, Any]]) -> Counter[str]:
    translated: Counter[str] = Counter()
    mapping = {
        "tool mismatch": "工具轨迹偏离期望",
        "citation miss": "未显式命中参考法条标题",
        "retrieval or reasoning drift": "检索或推理发生漂移",
        "partial but not complete answer": "答案部分正确但未达到完成阈值",
        "runtime error": "存在运行时错误",
        "threshold miss on direct-answer task": "直接回答任务卡在阈值边缘",
    }
    for key, count in build_failure_reason_summary(rows).items():
        translated[mapping.get(key, key)] += count
    return translated


def case_analysis_en(base: dict[str, Any], adapted: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if (safe_float(adapted.get("overall_score")) or 0.0) > (safe_float(base.get("overall_score")) or 0.0):
        notes.append("The adapter improves the aggregate score on this sample.")
    else:
        notes.append("This sample remains difficult even after post-training.")
    if (safe_float(adapted.get("tool_use_accuracy")) or 0.0) > (safe_float(base.get("tool_use_accuracy")) or 0.0):
        notes.append("Tool selection is closer to the scripted reference trajectory.")
    if (safe_float(adapted.get("citation_hit_rate")) or 0.0) > (safe_float(base.get("citation_hit_rate")) or 0.0):
        notes.append("The adapted answer cites the expected statute titles more reliably.")
    if (safe_float(adapted.get("answer_quality")) or 0.0) > (safe_float(base.get("answer_quality")) or 0.0):
        notes.append("Textual answer quality is higher after training.")
    if (safe_float(adapted.get("task_completion")) or 0.0) > (safe_float(base.get("task_completion")) or 0.0):
        notes.append("The sample crosses the binary completion threshold only after adaptation.")
    return notes[:3]


def case_analysis_zh(base: dict[str, Any], adapted: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if (safe_float(adapted.get("overall_score")) or 0.0) > (safe_float(base.get("overall_score")) or 0.0):
        notes.append("后训练模型在该样本上的综合分更高。")
    else:
        notes.append("该样本在后训练后仍然较难，是后续误差分析的重点。")
    if (safe_float(adapted.get("tool_use_accuracy")) or 0.0) > (safe_float(base.get("tool_use_accuracy")) or 0.0):
        notes.append("后训练后的工具调用更接近期望轨迹。")
    if (safe_float(adapted.get("citation_hit_rate")) or 0.0) > (safe_float(base.get("citation_hit_rate")) or 0.0):
        notes.append("后训练答案对法规标题的显式命中更稳定。")
    if (safe_float(adapted.get("answer_quality")) or 0.0) > (safe_float(base.get("answer_quality")) or 0.0):
        notes.append("后训练后的文本答案质量更高。")
    if (safe_float(adapted.get("task_completion")) or 0.0) > (safe_float(base.get("task_completion")) or 0.0):
        notes.append("该样本只有在后训练后才跨过任务完成阈值。")
    return notes[:3]


def failure_analysis_en(row: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if row.get("expected_tools") and (safe_float(row.get("tool_use_accuracy")) or 0.0) < 0.5:
        notes.append("The executed tool sequence diverges from the scripted reference.")
    citation = row.get("citation_hit_rate")
    if citation is not None and safe_float(citation) == 0.0:
        notes.append("The final answer does not explicitly cite the expected statute titles.")
    if (safe_float(row.get("semantic_similarity")) or 0.0) < 0.75:
        notes.append("The answer drifts semantically, which usually points to retrieval mismatch or reasoning drift.")
    elif (safe_float(row.get("answer_quality")) or 0.0) < 0.68:
        notes.append("The answer is partially correct but still below the completion threshold.")
    if row.get("errors"):
        notes.append("The row records runtime errors that the model did not fully recover from.")
    if not notes:
        notes.append("The sample is near-miss quality rather than a catastrophic failure.")
    return notes[:3]


def failure_analysis_zh(row: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if row.get("expected_tools") and (safe_float(row.get("tool_use_accuracy")) or 0.0) < 0.5:
        notes.append("实际工具调用序列偏离了脚本化期望轨迹。")
    citation = row.get("citation_hit_rate")
    if citation is not None and safe_float(citation) == 0.0:
        notes.append("最终答案没有显式命中期望的法规标题。")
    if (safe_float(row.get("semantic_similarity")) or 0.0) < 0.75:
        notes.append("答案语义发生了明显漂移，通常对应检索错位或推理偏航。")
    elif (safe_float(row.get("answer_quality")) or 0.0) < 0.68:
        notes.append("答案部分正确，但仍低于任务完成阈值。")
    if row.get("errors"):
        notes.append("该样本记录了未完全恢复的运行时错误。")
    if not notes:
        notes.append("该样本更像是临界阈值附近的近失误，而不是完全失败。")
    return notes[:3]


def render_case_studies_en(cases: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    if not cases:
        return "No paired evaluation details are available yet."
    blocks: list[str] = []
    for index, (base, adapted) in enumerate(cases, start=1):
        family = TASK_FAMILY_LABELS_EN.get(str(adapted.get("task_family", "unknown")), str(adapted.get("task_family", "unknown")))
        strategy = STRATEGY_LABELS_EN.get(str(adapted.get("strategy", "unknown")), str(adapted.get("strategy", "unknown")))
        blocks.append(f"\\subsubsection{{Case Study {index}: {latex_escape(family)} ({latex_escape(strategy)})}}")
        blocks.append(f"\\textbf{{Question.}} {latex_escape(clean_excerpt(adapted.get('question', ''), 220))}")
        blocks.append("\\par")
        blocks.append(f"\\textbf{{Reference answer.}} {latex_escape(clean_excerpt(adapted.get('expected_answer', ''), 220))}")
        blocks.append("\\par")
        blocks.append(f"\\textbf{{Expected tools.}} {latex_escape(expected_tool_sequence(adapted))}")
        blocks.append("\\par")
        blocks.append(f"\\textbf{{Base tool trace.}} {latex_escape(tool_sequence_from_history(base))}")
        blocks.append("\\par")
        blocks.append(f"\\textbf{{Adapter tool trace.}} {latex_escape(tool_sequence_from_history(adapted))}")
        blocks.append("\\par")
        blocks.append(
            f"\\textbf{{Base metrics.}} overall={format_metric_value(base.get('overall_score'))}, quality={format_metric_value(base.get('answer_quality'))}, tool={format_metric_value(base.get('tool_use_accuracy'))}, completion={format_metric_value(base.get('task_completion'))}."
        )
        blocks.append(latex_quote(base.get("final_answer", ""), 300))
        blocks.append(
            f"\\textbf{{Adapter metrics.}} overall={format_metric_value(adapted.get('overall_score'))}, quality={format_metric_value(adapted.get('answer_quality'))}, tool={format_metric_value(adapted.get('tool_use_accuracy'))}, completion={format_metric_value(adapted.get('task_completion'))}."
        )
        blocks.append(latex_quote(adapted.get("final_answer", ""), 300))
        blocks.append(latex_itemize([latex_escape(note) for note in case_analysis_en(base, adapted)]))
    return "\n".join(blocks)


def render_case_studies_zh(cases: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    if not cases:
        return "当前还没有可配对的评测明细，暂时无法自动生成案例。"
    blocks: list[str] = []
    for index, (base, adapted) in enumerate(cases, start=1):
        family = TASK_FAMILY_LABELS_ZH.get(str(adapted.get("task_family", "unknown")), str(adapted.get("task_family", "unknown")))
        strategy = STRATEGY_LABELS_ZH.get(str(adapted.get("strategy", "unknown")), str(adapted.get("strategy", "unknown")))
        blocks.append(f"\\subsubsection{{案例 {index}：{latex_escape(family)}（{latex_escape(strategy)}）}}")
        blocks.append(f"\\textbf{{问题。}} {latex_escape(clean_excerpt(adapted.get('question', ''), 220))}")
        blocks.append("\\par")
        blocks.append(f"\\textbf{{参考答案。}} {latex_escape(clean_excerpt(adapted.get('expected_answer', ''), 220))}")
        blocks.append("\\par")
        blocks.append(f"\\textbf{{期望工具。}} {latex_escape(expected_tool_sequence_zh(adapted))}")
        blocks.append("\\par")
        blocks.append(f"\\textbf{{基座工具轨迹。}} {latex_escape(tool_sequence_from_history_zh(base))}")
        blocks.append("\\par")
        blocks.append(f"\\textbf{{后训练工具轨迹。}} {latex_escape(tool_sequence_from_history_zh(adapted))}")
        blocks.append("\\par")
        blocks.append(
            f"\\textbf{{基座指标。}} overall={format_metric_value(base.get('overall_score'))}，quality={format_metric_value(base.get('answer_quality'))}，tool={format_metric_value(base.get('tool_use_accuracy'))}，completion={format_metric_value(base.get('task_completion'))}。"
        )
        blocks.append(latex_quote(base.get("final_answer", ""), 300))
        blocks.append(
            f"\\textbf{{后训练指标。}} overall={format_metric_value(adapted.get('overall_score'))}，quality={format_metric_value(adapted.get('answer_quality'))}，tool={format_metric_value(adapted.get('tool_use_accuracy'))}，completion={format_metric_value(adapted.get('task_completion'))}。"
        )
        blocks.append(latex_quote(adapted.get("final_answer", ""), 300))
        blocks.append(latex_itemize([latex_escape(note) for note in case_analysis_zh(base, adapted)]))
    return "\n".join(blocks)


def render_failure_cases_en(cases: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    if not cases:
        return "No failure cases are available yet."
    blocks: list[str] = []
    for index, (_, adapted) in enumerate(cases, start=1):
        family = TASK_FAMILY_LABELS_EN.get(str(adapted.get("task_family", "unknown")), str(adapted.get("task_family", "unknown")))
        blocks.append(f"\\subsubsection{{Failure Case {index}: {latex_escape(family)}}}")
        blocks.append(f"\\textbf{{Question.}} {latex_escape(clean_excerpt(adapted.get('question', ''), 220))}")
        blocks.append("\\par")
        blocks.append(f"\\textbf{{Expected tools.}} {latex_escape(expected_tool_sequence(adapted))}")
        blocks.append("\\par")
        blocks.append(f"\\textbf{{Observed tools.}} {latex_escape(tool_sequence_from_history(adapted))}")
        blocks.append("\\par")
        blocks.append(
            f"\\textbf{{Adapter metrics.}} overall={format_metric_value(adapted.get('overall_score'))}, quality={format_metric_value(adapted.get('answer_quality'))}, tool={format_metric_value(adapted.get('tool_use_accuracy'))}, citation={format_metric_value(adapted.get('citation_hit_rate'))}."
        )
        blocks.append(latex_quote(adapted.get("final_answer", ""), 300))
        blocks.append(latex_itemize([latex_escape(note) for note in failure_analysis_en(adapted)]))
    return "\n".join(blocks)


def render_failure_cases_zh(cases: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    if not cases:
        return "当前没有可自动生成的失败案例。"
    blocks: list[str] = []
    for index, (_, adapted) in enumerate(cases, start=1):
        family = TASK_FAMILY_LABELS_ZH.get(str(adapted.get("task_family", "unknown")), str(adapted.get("task_family", "unknown")))
        blocks.append(f"\\subsubsection{{失败案例 {index}：{latex_escape(family)}}}")
        blocks.append(f"\\textbf{{问题。}} {latex_escape(clean_excerpt(adapted.get('question', ''), 220))}")
        blocks.append("\\par")
        blocks.append(f"\\textbf{{期望工具。}} {latex_escape(expected_tool_sequence_zh(adapted))}")
        blocks.append("\\par")
        blocks.append(f"\\textbf{{实际工具。}} {latex_escape(tool_sequence_from_history_zh(adapted))}")
        blocks.append("\\par")
        blocks.append(
            f"\\textbf{{后训练指标。}} overall={format_metric_value(adapted.get('overall_score'))}，quality={format_metric_value(adapted.get('answer_quality'))}，tool={format_metric_value(adapted.get('tool_use_accuracy'))}，citation={format_metric_value(adapted.get('citation_hit_rate'))}。"
        )
        blocks.append(latex_quote(adapted.get("final_answer", ""), 300))
        blocks.append(latex_itemize([latex_escape(note) for note in failure_analysis_zh(adapted)]))
    return "\n".join(blocks)


def render_runtime_validations_en() -> str:
    blocks: list[str] = []
    for index, item in enumerate(RUNTIME_VALIDATIONS, start=1):
        blocks.append(f"\\subsubsection{{Runtime Validation {index}: {latex_escape(item['title_en'])}}}")
        blocks.append(f"\\textbf{{Prompt.}} {latex_escape(item['query'])}")
        blocks.append("\\par")
        blocks.append(f"\\textbf{{Observed behavior.}} {latex_escape(item['behavior_en'])}")
        blocks.append("\\par")
        blocks.append(f"\\textbf{{Why it matters.}} {latex_escape(item['why_en'])}")
    return "\n".join(blocks)


def render_runtime_validations_zh() -> str:
    blocks: list[str] = []
    for index, item in enumerate(RUNTIME_VALIDATIONS, start=1):
        blocks.append(f"\\subsubsection{{运行时验证 {index}：{latex_escape(item['title_zh'])}}}")
        blocks.append(f"\\textbf{{提示词。}} {latex_escape(item['query'])}")
        blocks.append("\\par")
        blocks.append(f"\\textbf{{观察到的行为。}} {latex_escape(item['behavior_zh'])}")
        blocks.append("\\par")
        blocks.append(f"\\textbf{{意义。}} {latex_escape(item['why_zh'])}")
    return "\n".join(blocks)


def render_tool_behavior_cases_en(cases: list[dict[str, Any]]) -> str:
    if not cases:
        return ""
    blocks: list[str] = []
    for index, item in enumerate(cases, start=1):
        title = str(item.get("title_en") or item.get("tool_name") or f"Tool Behavior {index}")
        blocks.append(
            "\\begin{tcolorbox}[colback=blue!2!white,colframe=blue!55!black,title={"
            + latex_escape(f"Tool Behavior {index}: {title}")
            + "}]"
        )
        blocks.append(f"\\textbf{{Tool focus.}} {latex_escape(str(item.get('tool_name') or 'unknown'))}")
        dialogue = str(item.get("dialogue") or item.get("query") or "").strip()
        if dialogue:
            blocks.append("\\par\\textbf{Dialogue.}\\par")
            blocks.append("{\\small " + latex_multiline(dialogue, 900) + "}")
        trace = str(item.get("trace") or "").strip()
        if trace:
            blocks.append("\\par\\textbf{Trace.}\\par")
            blocks.append("{\\ttfamily\\footnotesize " + latex_multiline(trace, 1200) + "}")
        behavior = str(item.get("behavior_en") or "").strip()
        if behavior:
            blocks.append("\\par\\textbf{Observed behavior.} " + latex_escape(behavior))
        why = str(item.get("why_en") or "").strip()
        if why:
            blocks.append("\\par\\textbf{Why representative.} " + latex_escape(why))
        blocks.append("\\end{tcolorbox}")
    return "\n".join(blocks)


def render_tool_behavior_cases_zh(cases: list[dict[str, Any]]) -> str:
    if not cases:
        return ""
    blocks: list[str] = []
    for index, item in enumerate(cases, start=1):
        title = str(item.get("title_zh") or item.get("tool_name") or f"工具行为 {index}")
        blocks.append(
            "\\begin{tcolorbox}[colback=green!2!white,colframe=green!45!black,title={"
            + latex_escape(f"代表性案例 {index}：{title}")
            + "}]"
        )
        blocks.append(f"\\textbf{{工具焦点。}} {latex_escape(str(item.get('tool_name') or 'unknown'))}")
        dialogue = str(item.get("dialogue") or item.get("query") or "").strip()
        if dialogue:
            blocks.append("\\par\\textbf{对话。}\\par")
            blocks.append("{\\small " + latex_multiline(dialogue, 900) + "}")
        trace = str(item.get("trace") or "").strip()
        if trace:
            blocks.append("\\par\\textbf{Trace。}\\par")
            blocks.append("{\\ttfamily\\footnotesize " + latex_multiline(trace, 1200) + "}")
        behavior = str(item.get("behavior_zh") or "").strip()
        if behavior:
            blocks.append("\\par\\textbf{观察到的行为。} " + latex_escape(behavior))
        why = str(item.get("why_zh") or "").strip()
        if why:
            blocks.append("\\par\\textbf{代表性意义。} " + latex_escape(why))
        blocks.append("\\end{tcolorbox}")
    return "\n".join(blocks)


def dataset_overview_rows_en(artifacts: ExperimentArtifacts) -> str:
    dataset = artifacts.dataset_summary
    corpus = artifacts.corpus_summary
    rag = artifacts.rag_metadata
    rows = [
        ("Catalog rows", dataset.get("catalog_rows", "N/A")),
        ("Direct DOCX/DOCM files", corpus.get("docx_files", dataset.get("linked_documents", "N/A"))),
        ("Archived legacy DOC files", dataset.get("archived_legacy_docs", "N/A")),
        ("Matched documents", corpus.get("matched_documents", "N/A")),
        ("Missing DOCX links", corpus.get("missing_docx", "N/A")),
        ("Missing catalog rows", corpus.get("missing_catalog", "N/A")),
        ("Chunk count", corpus.get("chunk_count", "N/A")),
        ("Locality-tagged chunks", corpus.get("local_chunk_count", "N/A")),
        ("Parse errors", corpus.get("parse_error_count", "N/A")),
        ("Embedding dim", rag.get("embedding_dim", "N/A")),
        ("Document shortlist size", rag.get("document_count", "N/A")),
    ]
    return "\n".join(f"{latex_escape(label)} & {latex_escape(value)} \\\\" for label, value in rows)


def dataset_overview_rows_zh(artifacts: ExperimentArtifacts) -> str:
    dataset = artifacts.dataset_summary
    corpus = artifacts.corpus_summary
    rag = artifacts.rag_metadata
    rows = [
        ("合并目录记录数", dataset.get("catalog_rows", "N/A")),
        ("可直接检索 DOCX/DOCM 数", corpus.get("docx_files", dataset.get("linked_documents", "N/A"))),
        ("归档 legacy DOC 数", dataset.get("archived_legacy_docs", "N/A")),
        ("成功匹配文档数", corpus.get("matched_documents", "N/A")),
        ("缺失 DOCX 链接数", corpus.get("missing_docx", "N/A")),
        ("缺失目录记录数", corpus.get("missing_catalog", "N/A")),
        ("Chunk 总数", corpus.get("chunk_count", "N/A")),
        ("带地域标签 chunk 数", corpus.get("local_chunk_count", "N/A")),
        ("解析错误数", corpus.get("parse_error_count", "N/A")),
        ("向量维度", rag.get("embedding_dim", "N/A")),
        ("法规级 shortlist 规模", rag.get("document_count", "N/A")),
    ]
    return "\n".join(f"{latex_escape(label)} & {latex_escape(value)} \\\\" for label, value in rows)


def training_config_rows_en(config: AppConfig) -> str:
    rows = [
        ("Base model", config.project_relative_path(config.models.agent_base) or str(config.models.agent_base)),
        ("Embedding model", config.project_relative_path(config.models.embedding_model) or str(config.models.embedding_model)),
        ("Epochs", config.training.num_train_epochs),
        ("Learning rate", config.training.learning_rate),
        ("LoRA rank", config.training.lora_r),
        ("LoRA alpha", config.training.lora_alpha),
        ("LoRA dropout", config.training.lora_dropout),
        ("Batch size x grad accumulation", f"{config.training.per_device_train_batch_size} x {config.training.gradient_accumulation_steps}"),
        ("Max sequence length", config.training.max_seq_length),
        ("Inference max steps", config.inference.max_steps),
    ]
    return "\n".join(f"{latex_escape(label)} & {latex_escape(value)} \\\\" for label, value in rows)


def training_config_rows_zh(config: AppConfig) -> str:
    rows = [
        ("基座模型", config.project_relative_path(config.models.agent_base) or str(config.models.agent_base)),
        ("向量模型", config.project_relative_path(config.models.embedding_model) or str(config.models.embedding_model)),
        ("训练轮数", config.training.num_train_epochs),
        ("学习率", config.training.learning_rate),
        ("LoRA rank", config.training.lora_r),
        ("LoRA alpha", config.training.lora_alpha),
        ("LoRA dropout", config.training.lora_dropout),
        ("批大小 x 梯度累积", f"{config.training.per_device_train_batch_size} x {config.training.gradient_accumulation_steps}"),
        ("最大序列长度", config.training.max_seq_length),
        ("推理最大步数", config.inference.max_steps),
    ]
    return "\n".join(f"{latex_escape(label)} & {latex_escape(value)} \\\\" for label, value in rows)


def quantitative_insights_en(artifacts: ExperimentArtifacts, eval_stats: dict[str, Any], tool_stats: dict[str, Any]) -> list[str]:
    rows = metric_rows(
        artifacts.base_summary,
        artifacts.adapted_summary,
        comparable=artifacts.eval_splits_match,
    )
    positive = [(key, delta) for key, _, _, _, _, delta in rows if delta is not None and delta > 0]
    negative = [(key, delta) for key, _, _, _, _, delta in rows if delta is not None and delta < 0]
    positive.sort(key=lambda item: item[1], reverse=True)
    negative.sort(key=lambda item: item[1])
    items: list[str] = []
    if not artifacts.eval_splits_match:
        items.append(
            latex_escape(
                f"The full adapter formal evaluation is still running, so the table combines the completed base result from {artifacts.base_eval_source_en} with the latest available adapter result from {artifacts.adapted_eval_source_en}. Because the splits differ, delta values are intentionally omitted."
            )
        )
        items.append(
            latex_escape(
                f"The currently available adapter summary covers {len(artifacts.adapted_rows)} sample(s), while the completed base evaluation covers {len(artifacts.base_rows)} sample(s)."
            )
        )
        items.append(
            latex_escape(
                f"The best training checkpoint is {artifacts.training_metrics.get('best_checkpoint', 'N/A')} with eval_loss={artifacts.training_metrics.get('eval_loss', 'N/A')}; this checkpoint summary is reported to provide a latest completed checkpoint signal before the full formal agent evaluation finishes."
            )
        )
        return items
    if positive:
        metric, delta = positive[0]
        items.append(latex_escape(f"The largest metric gain is on {METRIC_SPECS[metric]['label_en']} ({delta:+.3f})."))
    else:
        items.append(latex_escape("On the currently available split, the adapter does not yet produce a clear aggregate metric gain; this makes the detailed failure analysis especially important."))
    if negative:
        metric, delta = negative[0]
        items.append(latex_escape(f"The largest degradation is on {METRIC_SPECS[metric]['label_en']} ({delta:+.3f}), which should be read together with the qualitative cases below."))
    if eval_stats["scripted_answer_count"] == 0:
        items.append(latex_escape("The held-out evaluation split contains no scripted multi-turn clarification examples, so the locality-aware ask-user behavior is documented separately in the runtime validation section instead of being over-claimed from aggregate metrics."))
    items.append(latex_escape(f"The adapted model executed at least one tool on {tool_stats['samples_with_tools']}/{len(artifacts.adapted_rows)} evaluated samples, and used multi-step tool chains on {tool_stats['multi_tool_samples']} samples."))
    return items


def quantitative_insights_zh(artifacts: ExperimentArtifacts, eval_stats: dict[str, Any], tool_stats: dict[str, Any]) -> list[str]:
    rows = metric_rows(
        artifacts.base_summary,
        artifacts.adapted_summary,
        comparable=artifacts.eval_splits_match,
    )
    positive = [(key, delta) for key, _, _, _, _, delta in rows if delta is not None and delta > 0]
    negative = [(key, delta) for key, _, _, _, _, delta in rows if delta is not None and delta < 0]
    positive.sort(key=lambda item: item[1], reverse=True)
    negative.sort(key=lambda item: item[1])
    items: list[str] = []
    if not artifacts.eval_splits_match:
        items.append(
            latex_escape(
                f"完整的后训练 formal 评测仍在运行，因此下表把已经完成的基座 {artifacts.base_eval_source_zh} 结果，与最近可用的后训练 {artifacts.adapted_eval_source_zh} 结果并列展示。由于不是同一切分，差值列刻意留空，不做误导性比较。"
            )
        )
        items.append(
            latex_escape(
                f"当前可用的后训练结果只覆盖 {len(artifacts.adapted_rows)} 个样本，而已完成的基座正式评测覆盖 {len(artifacts.base_rows)} 个样本。"
            )
        )
        items.append(
            latex_escape(
                f"训练阶段自动选出的最佳 checkpoint 为 {artifacts.training_metrics.get('best_checkpoint', 'N/A')}，其 eval_loss={artifacts.training_metrics.get('eval_loss', 'N/A')}；在 formal agent 评测尚未跑完之前，这个 checkpoint 指标可作为最近一次已完成训练验证信号。"
            )
        )
        return items
    if positive:
        metric, delta = positive[0]
        items.append(latex_escape(f"提升最大的指标是 {METRIC_SPECS[metric]['label_zh']}（{delta:+.3f}）。"))
    else:
        items.append(latex_escape("在当前可用切分上，后训练模型暂未形成明显的总体指标提升，因此更需要依赖后面的失败案例和细粒度分析来解释现象。"))
    if negative:
        metric, delta = negative[0]
        items.append(latex_escape(f"下降最大的指标是 {METRIC_SPECS[metric]['label_zh']}（{delta:+.3f}），需要结合下面的定性案例一起解读。"))
    if eval_stats["scripted_answer_count"] == 0:
        items.append(latex_escape("当前 held-out 评测切分不包含脚本化多轮追问样本，因此地点澄清和多轮延续能力被单列到运行时验证部分，而不是被不恰当地夸大到量化指标中。"))
    items.append(latex_escape(f"在 {len(artifacts.adapted_rows)} 个评测样本中，后训练模型在 {tool_stats['samples_with_tools']} 个样本上至少调用了一次工具，在 {tool_stats['multi_tool_samples']} 个样本上执行了多步工具链。"))
    return items


def build_en_report(artifacts: ExperimentArtifacts) -> str:
    config = artifacts.config
    samples = list(artifacts.sample_map.values())
    eval_stats = build_eval_dataset_stats(samples)
    tool_stats = build_tool_usage_stats(artifacts.adapted_rows)
    case_studies = select_case_studies(artifacts.base_rows, artifacts.adapted_rows, max_examples=3)
    failure_cases = select_failure_cases(artifacts.base_rows, artifacts.adapted_rows, max_examples=2)
    generation_summary = artifacts.generation_summary
    strategy_counts = generation_summary.get("selected_strategies", {})
    task_counts = generation_summary.get("selected_task_families", {})
    failure_summary = build_failure_reason_summary(artifacts.adapted_rows)
    tool_behavior_cases = artifacts.tool_behavior_cases

    prompt_sample = next((sample for sample in samples if sample.get("messages")), None)
    direct_sample = next((sample for sample in samples if not sample.get("expected_tools")), None)
    retrieve_sample = next((sample for sample in samples if sample.get("expected_tools")), None)

    prompt_preview = (
        latex_quote(prompt_sample.get("messages", [{}])[0].get("content", ""), 760)
        if prompt_sample and prompt_sample.get("messages")
        else latex_escape("Prompt preview will be available after dataset construction.")
    )
    direct_example_preview = (
        latex_quote(
            (
                f"sample_id={direct_sample.get('sample_id')} | "
                f"task_family={direct_sample.get('metadata', {}).get('task_family', 'unknown')} | "
                f"strategy={direct_sample.get('metadata', {}).get('strategy', 'unknown')} | "
                f"expected_tools={expected_tool_sequence(direct_sample)} | "
                f"trace={clean_excerpt(direct_sample.get('trace', ''), 260)}"
            ),
            760,
        )
        if direct_sample else ""
    )
    retrieve_example_preview = (
        latex_quote(
            (
                f"sample_id={retrieve_sample.get('sample_id')} | "
                f"task_family={retrieve_sample.get('metadata', {}).get('task_family', 'unknown')} | "
                f"strategy={retrieve_sample.get('metadata', {}).get('strategy', 'unknown')} | "
                f"expected_tools={expected_tool_sequence(retrieve_sample)} | "
                f"trace={clean_excerpt(retrieve_sample.get('trace', ''), 320)}"
            ),
            760,
        )
        if retrieve_sample else ""
    )
    runtime_trace_preview = (
        latex_quote(retrieve_sample.get("trace", ""), 720)
        if retrieve_sample else (latex_quote(direct_sample.get("trace", ""), 720) if direct_sample else "")
    )

    abstract = (
        "This report presents an offline Chinese legal agent for Assignment 3. The system combines a repaired local law corpus, a hierarchical hybrid RAG stack, structured local tools, agent-style training trajectories, QLoRA post-training on Qwen3-4B, and assignment-aligned evaluation that compares the base model with the post-trained agent on the same runtime."
    )

    research_topic = "\n".join([
        "\\paragraph{Research Topic} " + latex_escape(
            "The project studies agent-oriented post-training for a Chinese legal assistant. Instead of treating legal QA as plain instruction following, the work turns the model into a local agent that must decide when to retrieve statutes, when to compare authority hierarchy, when to ask for missing facts, and when to compute an exact quantity before answering."
        ),
        "\\paragraph{Problem Setting} " + latex_escape(
            "Chinese legal questions are difficult because national law and local regulations coexist, user inputs are often incomplete, and many practical questions require both statute retrieval and structured follow-up. The target system therefore needs grounded retrieval, location awareness, multi-step tool use, and robust fallback behavior rather than fluent text generation alone."
        ),
        "\\paragraph{System Scope} " + latex_escape(
            "The final system is fully local: the backbone model, embedding model, corpus, tools, retrieval index, training data construction, and evaluation loop all run offline. This makes the project a concrete study of end-to-end legal agent engineering under realistic local-compute constraints."
        ),
    ])

    experiment_design = "\n".join([
        "\\paragraph{Task Formulation and Backbone Choice} " + latex_escape(
            f"The downstream task is agent post-training. The formal setting uses a local Qwen3-4B backbone and optimizes it to emit valid ReAct-style decisions over local tools. QLoRA is adopted as the main training technique because it keeps the experiment feasible on local GPUs while still allowing non-trivial adaptation of the 4B backbone."
        ),
        "\\paragraph{Data Engineering Plan} " + latex_escape(
            f"The project uses two data sources with different roles. The law repository under data/law_files is turned into the legal knowledge base; DISC-Law-SFT is used as the supervised seed source for trajectory construction. The selected formal training mix contains {counter_text_en(Counter({str(k): int(v) for k, v in strategy_counts.items()}), STRATEGY_LABELS_EN)} trajectories across {counter_text_en(Counter({str(k): int(v) for k, v in task_counts.items()}), TASK_FAMILY_LABELS_EN)}. The formal split stays inside DISC-derived seeds rather than hand-crafted showcase questions, which keeps the train/eval protocol clean."
        ),
        "\\paragraph{Agent Data Construction Methodology} " + latex_escape(
            "The agent dataset is not a collection of raw QA pairs. Each seed question is converted into an executable trajectory with an expected tool plan, optional scripted answers for ask-user turns, optional forced errors for recovery training, and a final assistant trace that contains Thought, Action, Observation, and Final Answer segments. Direct-answer tasks stay tool-free; retrieval tasks generate retrieve-then-answer traces; reference-bearing tasks can generate lookup-then-retrieve traces."
        ),
        "\\paragraph{Agent Prompt Design} " + latex_escape(
            "The prompt is built from the four components recommended in the assignment: a system role, explicit tool definitions, an output protocol, and the user query. Training traces use the full ReAct shell with Observation included inside the assistant demonstration, while the online runtime uses a stepwise variant with the same role/tool/rule blocks but restricts each model turn to either Thought+Action or Final Answer so that observations are injected only by the runtime. The codebase also keeps pure, one-shot, and few-shot prompt modes for qualitative probing; the formal evaluation uses the same strict pure-mode agent runtime for both the base model and the adapter."
        ),
        "\\paragraph{Tool Definitions Used in the Experiments} " + latex_escape(
            "All tools are locally executable. retrieve_from_kb performs hybrid retrieval over the law corpus, lookup_statute resolves title-centric queries, resolve_hierarchy compares legal authority levels, calculator evaluates exact arithmetic expressions, and ask_user collects missing facts when a reliable legal conclusion would otherwise be impossible."
        ),
        latex_itemize([
            latex_escape("retrieve_from_kb: hybrid chunk retrieval over the local corpus, with locality-aware scoring and clarification signals."),
            latex_escape("lookup_statute: exact or alias-based regulation lookup for title verification and metadata grounding."),
            latex_escape("resolve_hierarchy: interprets legal authority ordering when the question is about precedence or conflict."),
            latex_escape("calculator: executes safe local arithmetic for compensation, tax, interest, or penalty calculations."),
            latex_escape("ask_user: requests the single most important missing fact, including location when local-law applicability is unclear."),
        ]),
        "\\paragraph{Evaluation Protocol} " + latex_escape(
            "The assignment asks for an agent-vs-agent comparison. Accordingly, both the base model and the post-trained adapter are evaluated through the same runtime rather than by raw text generation alone. The result section reports the required metrics: format compliance, tool-use accuracy, task completion, error recovery, citation hit rate, textual overlap metrics, and representative case studies."
        ),
        "\\begin{table}[t]\n\\centering\n\\small\n\\begin{tabular}{lr}\n\\toprule\nArtifact & Value \\\\ \n\\midrule\n"
        + dataset_overview_rows_en(artifacts)
        + "\n\\bottomrule\n\\end{tabular}\n\\caption{Knowledge-base scale after repair, parsing, chunking, and index construction.}\n\\end{table}",
        "\\begin{table}[t]\n\\centering\n\\small\n\\begin{tabular}{ll}\n\\toprule\nSetting & Value \\\\ \n\\midrule\n"
        + training_config_rows_en(config)
        + "\n\\bottomrule\n\\end{tabular}\n\\caption{Core model, training, and runtime settings used by the formal pipeline.}\n\\end{table}",
    ])

    code_parts_en = [
        "\\paragraph{1. Overall Software Architecture} " + latex_escape(
            "The codebase is organized as a full experimental stack rather than a single notebook. The data layer repairs Word files and normalizes metadata, the RAG layer builds indices and retrieval-time scoring structures, the agent layer defines prompts/tools/parsing/runtime transitions, the training layer constructs agent trajectories and runs QLoRA, and the evaluation layer replays the same agent runtime to produce metrics and side-by-side traces."
        ),
        latex_itemize([
            latex_escape("data/: Word repair, document parsing, metadata normalization, jurisdiction inference, and chunk construction."),
            latex_escape("rag/: dense embeddings, BM25, document shortlist, citation graph, and hybrid retrieval logic."),
            latex_escape("agent/: system prompt, ReAct parser, tool registry, and the runtime state machine."),
            latex_escape("training/: DISC seed selection, trajectory planning/execution, and QLoRA fine-tuning."),
            latex_escape("evaluation/: heuristic judging, quantitative metrics, and detailed per-sample comparison files."),
        ]),
        "\\paragraph{2. Knowledge-Base Construction and Data Schema} " + latex_escape(
            "The legal knowledge base starts from a mixed repository of catalog spreadsheets and Word files. The preprocessing path first repairs legacy documents, then builds a manifest in which each matched regulation carries title metadata, effect level, jurisdiction type/scope, region hierarchy, dates, and source paths. The corpus builder parses each document into section/article units and finally slices those units into overlapping character windows so that the retriever operates on semantically coherent but GPU-manageable chunks."
        ),
        latex_itemize([
            latex_escape("Manifest record schema: title, normalized_title, category_raw, effect_level, effect_rank, jurisdiction_type, jurisdiction_scope, region_name, region_path_codes, region_path_names, promulgation_date, effective_date, version_date, status, and source_path."),
            latex_escape("Chunk schema: chunk_id, document_id, document_title, article_heading, section_context, effect metadata, jurisdiction metadata, cross_references, text, and retrieval_text."),
            latex_escape("The retrieval_text field concatenates title, effect level, region path, article heading, and the chunk text window so that both dense retrieval and BM25 see the same normalized representation."),
            latex_escape("The corpus is stored as JSONL, which keeps preprocessing transparent and makes later inspection/debugging straightforward."),
        ]),
        "\\paragraph{3. Hierarchical Indexing and Retrieval} " + latex_escape(
            "The retriever is a two-level hybrid RAG system. At the chunk level it keeps a dense index and a BM25 index; at the document level it maintains a second dense/BM25 pair for shortlist generation. Dense vectors are written incrementally into numpy memmap files to avoid holding the full matrix in RAM during index construction. Retrieval first builds a document shortlist, then re-ranks candidate chunks with reciprocal rank fusion, and finally adjusts scores using authority signals, locality relation, and cross-reference expansion."
        ),
        latex_itemize([
            latex_escape("Index artifacts: dense_embeddings.npy, bm25.pkl, chunks.jsonl, document_embeddings.npy, document_bm25.pkl, document_records.json, doc_to_chunks.json, chunk_to_doc.json, graph.json, and metadata.json."),
            latex_escape("Document records store document_id, normalized_title, effect metadata, jurisdiction metadata, region paths, chunk_positions, and a retrieval_text built from title plus representative sample texts."),
            latex_escape("During scoring, exact region matches get an explicit boost, ancestor/descendant regions receive weaker boosts, unrelated local laws are penalized, and citation-linked regulations can be pulled in via graph expansion."),
            latex_escape("This design is important for legal QA because the correct answer is often located in a city-level regulation that would be buried inside a flat global top-k search."),
        ]),
        "\\paragraph{4. Query Understanding and Locality Modeling} " + latex_escape(
            "Before retrieval, the system builds a QueryContext that stores explicit regions, whether the question is likely local-law-sensitive, whether authority comparison should be emphasized, the explicit region level, and a resolved location object. The location resolver combines rule-based region extraction with address parsing and ancestor backfill, so a street mention can still map to the correct county, prefecture, and province even if that street name is not explicitly present in the knowledge-base metadata."
        ),
        latex_itemize([
            latex_escape("Exact region relations are computed against region_path_codes, not only against raw surface strings, which makes locality scoring more stable."),
            latex_escape("The retrieval result payload includes both explicit_query_regions and a resolved_query_location object so that downstream logic can decide whether further clarification is needed."),
            latex_escape("Province-only or city-only inputs can trigger a finer clarification request when the top local hits imply that more specific municipal or district regulations may change the answer."),
        ]),
        "\\paragraph{5. Agent-Style Dataset Construction} " + latex_escape(
            "The training data is synthesized programmatically from DISC-Law seeds. Each TrajectorySeed carries the source question, gold answer, references, expected tools, optional scripted answers, optional clarification questions, optional calculator expressions, an optional retrieval query, and a force_error flag. The TrajectoryBuilder converts these seeds into executable plans, actually runs the local tools, captures observations, and writes a finished JSONL example with messages, trace, and tool_trace fields."
        ),
        latex_itemize([
            latex_escape("Direct-answer tasks such as judgment summarization deliberately produce zero-tool traces so that the model learns not to retrieve unnecessarily."),
            latex_escape("Retrieve-then-answer tasks produce a retrieval action followed by an observation-grounded final answer."),
            latex_escape("Lookup-then-retrieve tasks first verify statute identity by title and then retrieve supporting passages."),
            latex_escape("Error-recovery examples are injected by first issuing a deliberately broken lookup title on selected reference-bearing samples, forcing the trajectory to continue after a failure rather than stopping."),
            latex_escape("The final JSONL example stores sample_id, question, expected_answer, references, expected_tools, force_error, scripted_answers, metadata, messages, trace, and tool_trace."),
        ]),
    ]
    if direct_example_preview:
        code_parts_en.extend([
            "\\paragraph{Direct-Answer Example} " + latex_escape(
                "A representative direct-answer training sample contains a legal document or self-contained text and teaches the model to answer without calling tools when the evidence is already inside the user input."
            ),
            direct_example_preview,
        ])
    if retrieve_example_preview:
        code_parts_en.extend([
            "\\paragraph{Tool-Using Example} " + latex_escape(
                "A representative retrieval sample teaches the model to issue a concrete retrieval query, consume the returned observation, and then synthesize the final answer instead of guessing statutes from memory."
            ),
            retrieve_example_preview,
        ])
    code_parts_en.extend([
        "\\paragraph{6. Prompt Template and Parsing Contract} " + latex_escape(
            "The prompt implementation mirrors the assignment recommendations. It concatenates a legal-agent system role, typed tool definitions with return formats, an explicit output protocol, and detailed behavioral rules. The code keeps the same tool block and behavioral constraints across training and inference, while the runtime swaps in a stricter stepwise output-format line so that the model cannot fabricate observations before tool execution. The parser then extracts Thought, Action, and Final Answer using regexes and accepts JSON-style, quoted single-argument, or keyword-style argument payloads."
        ),
        prompt_preview,
        "\\paragraph{7. Tool Registry and Tool Semantics} " + latex_escape(
            "Each tool is implemented as a local function behind a ToolRegistry. The registry also standardizes tool schemas for prompting and compacts observations before they are appended back into the scratchpad. This keeps tool outputs machine-readable enough for the runtime while still exposing human-auditable evidence in the trace."
        ),
        latex_itemize([
            latex_escape("retrieve_from_kb returns the query string, explicit regions, explicit region level, the resolved location payload, a clarification flag/question, and a list of scored retrieval hits with title, article heading, jurisdiction, region path, source path, and text snippets."),
            latex_escape("lookup_statute performs exact and alias-based title matching over the manifest and returns regulation metadata plus preview articles/texts."),
            latex_escape("resolve_hierarchy maps a title or legal category onto its authority level so that the agent can discuss precedence conflicts explicitly."),
            latex_escape("calculator uses a restricted AST evaluator with explicit operator/function whitelists, which makes arithmetic deterministic and safe."),
            latex_escape("ask_user can run in scripted mode for dataset generation, callback mode for the web UI, or interactive terminal mode; in every case the returned payload is normalized into a common question/answer structure."),
        ]),
        "\\paragraph{8. Runtime Workflow, State Transitions, and Fallback} " + latex_escape(
            "The online agent is a real state machine rather than a single generate-once call. It begins with a turn-analysis pass that decides whether the latest user input is a new question, a supplement, or an answer to a previous follow-up. It then builds the current messages from history, user input, and accumulated scratchpad, streams the next model step, parses it, executes the tool if needed, and routes to the next state. The runtime explicitly handles duplicate tool calls, format errors, repeated clarification questions, tool failures, step limits, and partially satisfactory draft answers. When the agent can no longer take more useful actions, it synthesizes a final answer from the existing scratchpad instead of returning a generic failure message."
        ),
        latex_itemize([
            latex_escape("Format violations trigger an automatic retry with a FORMAT_ERROR observation inserted into the scratchpad."),
            latex_escape("Repeated tool calls are blocked and converted into answer synthesis so that the agent does not loop indefinitely."),
            latex_escape("Clarification questions are normalized, deduplicated, and capped; after the cap is reached the runtime switches to conditional analysis instead of asking forever."),
            latex_escape("retrieve_from_kb can directly terminate the current turn with a clarification request when locality-sensitive questions still need a more precise location."),
            latex_escape("The final-answer synthesizer runs with tool use disabled and forces the model to organize the answer into known facts, legal basis, conclusion, and remaining uncertainty."),
        ]),
    ])
    if runtime_trace_preview:
        code_parts_en.extend([
            "\\paragraph{Runtime Trace Example} " + latex_escape(
                "The following excerpt illustrates the deployed ReAct trace format: a thought, a concrete action, an injected observation, and a grounded final answer."
            ),
            runtime_trace_preview,
        ])
    code_parts_en.extend([
        "\\paragraph{9. Training and Evaluation Implementation} " + latex_escape(
            f"The QLoRA trainer performs completion-only supervision over the assistant trace, with 4-bit loading, rank {config.training.lora_r}, alpha {config.training.lora_alpha}, dropout {config.training.lora_dropout}, maximum sequence length {config.training.max_seq_length}, and best-checkpoint selection on evaluation loss. The evaluation pipeline then reruns the base model and the adapter through the same runtime, computes all assignment metrics, and writes both aggregate summaries and per-sample JSONL diagnostics so that qualitative analysis is grounded in actual traces rather than anecdotal screenshots."
        ),
        latex_itemize([
            latex_escape("Training artifacts include the adapter checkpoints, tokenizer files, and training_metrics.json with runtime, losses, and best checkpoint."),
            latex_escape("Evaluation artifacts include eval_summary.json with metric supports/weights and eval_details.jsonl with question, final answer, trace, tool history, and all per-sample metrics."),
            latex_escape("This file layout makes the experiment reproducible: the same outputs are later consumed for tables, case studies, and failure analysis."),
        ]),
    ])
    code_implementation = "\n".join(part for part in code_parts_en if part)

    evaluation_setup = "\n".join([
        "\\paragraph{Held-out Evaluation Split} " + latex_escape(
            f"The formal held-out set contains {eval_stats['total']} trajectories. {eval_stats['reference_count']} samples include gold statutory references, {eval_stats['tool_required_count']} samples expect at least one tool call, and {eval_stats['force_error_count']} samples explicitly test recovery behavior. The evaluation strategy mix is {counter_text_en(eval_stats['strategy_counts'], STRATEGY_LABELS_EN)}."
        ),
        "\\paragraph{Metric Design} " + latex_escape(
            "The quantitative protocol follows the assignment closely. Format compliance measures whether the ReAct shell is respected; tool-use accuracy compares executed tool sequences against the expected plan; task completion is a binary success metric derived from textual quality and citation-aware similarity; error recovery checks whether the agent re-plans after forced failures; and exact match, token F1, semantic similarity, answer quality, and citation hit rate measure textual grounding and legal citation quality. Metrics that have zero support on the current split are excluded from the denominator of the overall score rather than forced to zero."
        ),
        "\\paragraph{What the Aggregate Table Can and Cannot Show} " + latex_escape(
            "The formal held-out data comes from DISC-Law-SFT, so it is strong on general legal QA but comparatively light on open-ended locality-clarification dialogues. For that reason, the quantitative table is complemented by explicit runtime validations for location-sensitive behavior rather than over-interpreting one aggregate number."
        ),
    ])

    quantitative_results = "\n".join([
        "\\begin{table}[t]\n\\centering\n\\scriptsize\n\\begin{tabular}{lrrrrr}\n\\toprule\nMetric & Base sup. & Base & Adapter sup. & Adapter & Delta \\\\ \n\\midrule\n"
        + metric_table_rows_en(
            artifacts.base_summary,
            artifacts.adapted_summary,
            comparable=artifacts.eval_splits_match,
        )
        + "\n\\bottomrule\n\\end{tabular}\n\\caption{Base metrics are taken from "
        + latex_escape(artifacts.base_eval_source_en)
        + "; adapter metrics are taken from "
        + latex_escape(artifacts.adapted_eval_source_en)
        + ".}\n\\end{table}",
        "\\paragraph{Training Summary} " + latex_escape(
            f"Training loss={artifacts.training_metrics.get('train_loss', 'N/A')}, validation loss={artifacts.training_metrics.get('eval_loss', 'N/A')}, runtime={artifacts.training_metrics.get('train_runtime', 'N/A')} seconds, best checkpoint={artifacts.training_metrics.get('best_checkpoint', 'N/A')}."
        ),
        latex_itemize(quantitative_insights_en(artifacts, eval_stats, tool_stats)),
        "\\paragraph{Interpretation} " + latex_escape(
            f"Across the currently available adapter rows ({artifacts.adapted_eval_source_en}), the adapted model used at least one tool on {tool_stats['samples_with_tools']} samples, executed multi-tool chains on {tool_stats['multi_tool_samples']} samples, and invoked ask_user on {tool_stats['ask_user_samples']} samples. These counts matter because agent post-training is valuable only if the model actually changes its behavior inside the runtime rather than merely paraphrasing the final answer."
        ),
    ])

    failure_items = [latex_escape(f"{count} samples show {reason}.") for reason, count in failure_summary.most_common(5)]
    if not failure_items:
        failure_items = [latex_escape("No failure patterns are available yet because the evaluation details are missing.")]

    result_analysis = "\n".join([
        "\\subsection{Evaluation Setup}\n" + evaluation_setup,
        "\\subsection{Quantitative Results}\n" + quantitative_results,
        "\\subsection{Side-by-Side Case Studies}\n" + render_case_studies_en(case_studies),
        (
            "\\subsection{Representative Tool Behaviors}\n"
            + latex_escape(
                "The held-out DISC split does not naturally cover every tool in balanced proportions. The following targeted runtime examples therefore complement the formal case studies and illustrate how retrieval, statute lookup, hierarchy resolution, follow-up questions, and calculation are used in the deployed agent."
            )
            + "\n"
            + render_tool_behavior_cases_en(tool_behavior_cases)
        ) if tool_behavior_cases else "",
        "\\subsection{Failure Cases and Discussion}\n"
        + latex_itemize(failure_items)
        + "\n"
        + render_failure_cases_en(failure_cases),
        (
            "\\subsection{Locality and Multi-turn Validation}\n"
            + latex_escape(
                "The formal split still under-covers locality-sensitive follow-up. One concise supplementary note is therefore kept to document the verified location-clarification behavior of the deployed runtime."
            )
            + "\n"
            + render_runtime_validations_en()
        ) if not tool_behavior_cases else "",
    ])

    conclusion = "\n".join([
        "\\paragraph{Conclusion} " + latex_escape(
            "The project demonstrates that legal agent quality is a systems problem rather than a pure fine-tuning problem. The final performance depends jointly on corpus repair, metadata normalization, hierarchical retrieval, prompt discipline, tool schemas, runtime control, and QLoRA adaptation of the backbone model."
        ),
        "\\paragraph{Key Takeaways} " + latex_escape(
            "Three takeaways are especially important. First, agent-style trajectories are more informative than plain QA pairs because they teach the model when not to retrieve and when to recover from failure. Second, locality handling is not a cosmetic feature in legal QA; region hierarchy directly changes which statutes are applicable. Third, detailed per-sample traces are indispensable for understanding why a post-trained agent succeeds or fails."
        ),
    ])

    latex_slash = "\\"

    return dedent(
        rf"""
        \documentclass{{article}}
        \usepackage{{STY/iclr2024_conference,times}}
        \input{{STY/math_commands.tex}}
        \usepackage{{hyperref}}
        \usepackage{{url}}
        \usepackage{{CJK}}
        \usepackage{{booktabs}}
        \usepackage{{multirow}}
        \usepackage{{graphicx}}
        \usepackage{{array}}
        \usepackage{{tabularx}}
        \usepackage{{longtable}}
        \usepackage{{enumitem}}
        \usepackage{{float}}
        \usepackage[most]{{tcolorbox}}
        {latex_slash}title{{Agent Post-Training for a Local Chinese Legal Assistant with Hybrid Legal RAG}}
        \author{{Yijin Zhao \\
        225010231 \\
        The Chinese University of Hong Kong, Shenzhen \\
        {latex_slash}texttt{{225010231@link.cuhk.edu.cn}}}}
        \iclrfinalcopy
        \begin{{document}}
        \begin{{CJK}}{{UTF8}}{{gbsn}}
        \maketitle

        \begin{{abstract}}
        {latex_escape(abstract)}
        \end{{abstract}}

        \section{{Research Topic}}
        {research_topic}

        \section{{Experiment Design}}
        {experiment_design}

        \section{{Code Implementation}}
        {code_implementation}

        \section{{Result Analysis}}
        {result_analysis}

        \section{{Conclusion}}
        {conclusion}

        \newpage
        \section*{{Acknowledgment}}
        This report is prepared for CSC6052 / MDS5110 / CSC5051 Assignment 3 and follows the provided ICLR-style formatting template.

        \end{{CJK}}
        \end{{document}}
        """
    ).strip() + "\n"


def build_zh_report(artifacts: ExperimentArtifacts) -> str:
    config = artifacts.config
    samples = list(artifacts.sample_map.values())
    eval_stats = build_eval_dataset_stats(samples)
    tool_stats = build_tool_usage_stats(artifacts.adapted_rows)
    case_studies = select_case_studies(artifacts.base_rows, artifacts.adapted_rows, max_examples=3)
    failure_cases = select_failure_cases(artifacts.base_rows, artifacts.adapted_rows, max_examples=2)
    generation_summary = artifacts.generation_summary
    strategy_counts = generation_summary.get("selected_strategies", {})
    task_counts = generation_summary.get("selected_task_families", {})
    failure_summary = build_failure_reason_summary_zh(artifacts.adapted_rows)
    tool_behavior_cases = artifacts.tool_behavior_cases

    prompt_sample = next((sample for sample in samples if sample.get("messages")), None)
    direct_sample = next((sample for sample in samples if not sample.get("expected_tools")), None)
    retrieve_sample = next((sample for sample in samples if sample.get("expected_tools")), None)

    prompt_preview = (
        latex_quote(prompt_sample.get("messages", [{}])[0].get("content", ""), 760)
        if prompt_sample and prompt_sample.get("messages")
        else latex_escape("待数据集构建完成后，这里会展示系统提示词样例。")
    )
    direct_example_preview = (
        latex_quote(
            (
                f"sample_id={direct_sample.get('sample_id')} | "
                f"task_family={direct_sample.get('metadata', {}).get('task_family', 'unknown')} | "
                f"strategy={direct_sample.get('metadata', {}).get('strategy', 'unknown')} | "
                f"expected_tools={expected_tool_sequence_zh(direct_sample)} | "
                f"trace={clean_excerpt(direct_sample.get('trace', ''), 260)}"
            ),
            760,
        )
        if direct_sample else ""
    )
    retrieve_example_preview = (
        latex_quote(
            (
                f"sample_id={retrieve_sample.get('sample_id')} | "
                f"task_family={retrieve_sample.get('metadata', {}).get('task_family', 'unknown')} | "
                f"strategy={retrieve_sample.get('metadata', {}).get('strategy', 'unknown')} | "
                f"expected_tools={expected_tool_sequence_zh(retrieve_sample)} | "
                f"trace={clean_excerpt(retrieve_sample.get('trace', ''), 320)}"
            ),
            760,
        )
        if retrieve_sample else ""
    )
    runtime_trace_preview = (
        latex_quote(retrieve_sample.get("trace", ""), 720)
        if retrieve_sample else (latex_quote(direct_sample.get("trace", ""), 720) if direct_sample else "")
    )

    abstract = (
        "本文报告一个可完全离线运行的中文法律 Agent。系统将地方法规知识库构建、分层混合检索、本地工具调用、Agent 风格轨迹数据、Qwen3-4B 的 QLoRA 后训练，以及按作业要求组织的量化评测与案例分析整合为一条完整实验链路。"
    )

    research_topic = "\n".join([
        "\\section{研究主题}",
        "\\paragraph{研究对象} " + latex_escape(
            "本项目研究中文法律场景下的 Agent-oriented post-training。目标不是做一个泛化闲聊模型，而是构建一个能够在本地法规环境中稳定检索、调用工具、处理追问并给出可核验法律答复的智能体。"
        ),
        "\\paragraph{问题背景} " + latex_escape(
            "中文法律问答的难点在于：全国性法律与地方性法规并存，用户输入常缺少地点、金额、时间或身份等关键事实，许多问题还需要先检索再计算或先追问再判断。因此，真正可用的系统必须同时具备检索、规划、工具调用、多轮延续和错误恢复能力。"
        ),
        "\\paragraph{系统范围} " + latex_escape(
            "本系统从知识库构建、Agent 数据集生成、模型后训练到运行时评测全部离线完成。因而它既是一次法律 Agent 的后训练实验，也是一次在本地算力约束下完成端到端 LLM Agent 工程实现的案例。"
        ),
    ])

    experiment_design = "\n".join([
        "\\section{实验设计}",
        "\\paragraph{任务定义与骨干模型选择} " + latex_escape(
            "下游任务是 Agent post-training：模型必须在 ReAct 框架下判断是否需要调用工具、何时追问、何时直接回答，并在工具返回 Observation 后继续规划。骨干模型选择本地 Qwen3-4B，训练方法采用 QLoRA，是因为它能够在本地 GPU 条件下完成可复现的后训练，同时保留足够的表达与推理能力。"
        ),
        "\\paragraph{数据工程设计} " + latex_escape(
            f"项目使用两类数据源。法规侧数据来自 data/law_files，用于构建本地知识库；监督侧种子数据来自 DISC-Law-SFT，用于构造 Agent 风格轨迹。正式训练样本的策略分布为 {counter_text_zh(Counter({str(k): int(v) for k, v in strategy_counts.items()}), STRATEGY_LABELS_ZH)}；任务族分布为 {counter_text_zh(Counter({str(k): int(v) for k, v in task_counts.items()}), TASK_FAMILY_LABELS_ZH)}。正式 train/eval 保持在 DISC 派生轨迹范围内，不把人工 showcase 题混入评测。"
        ),
        "\\paragraph{Agent 数据构造方法} " + latex_escape(
            "本项目并不直接把 QA 对喂给模型，而是把种子问题转换成可执行轨迹。每条轨迹都包含预期工具、可选的脚本化追问答案、可选的计算表达式、可选的错误注入和最终完整 ReAct trace。这样训练目标就从“学会回答”升级为“学会在何时调用哪种工具并基于 Observation 继续决策”。"
        ),
        "\\paragraph{Prompt Engineering 与 Agent Prompt 设计} " + latex_escape(
            "提示词严格围绕作业要求的四部分设计：系统角色、工具定义、输出协议和用户查询。训练样本中的 assistant 轨迹使用完整 Thought/Action/Observation/Final Answer 外壳；在线运行时使用同一套系统角色、工具块和规则块，但把输出格式收紧成“每轮只能输出 Thought+Action 或 Final Answer”，以防模型在工具执行前伪造 Observation。代码中同时保留 pure、one-shot 和 few-shot 三种模式用于定性测试；正式评测对基座模型和后训练模型都使用同一套 pure-mode runtime。"
        ),
        "\\paragraph{工具定义} " + latex_escape(
            "所有工具均为本地可执行工具。retrieve_from_kb 负责法规检索，lookup_statute 负责标题核验，resolve_hierarchy 负责效力层级解释，calculator 负责精确算术，ask_user 负责补齐关键事实。"
        ),
        latex_itemize([
            latex_escape("retrieve_from_kb：对本地法规知识库执行混合检索，并返回法规片段、层级与地点信息。"),
            latex_escape("lookup_statute：按法规标题或别名精确定位法规元数据和预览条文。"),
            latex_escape("resolve_hierarchy：解释法规效力顺序，支持上位法/下位法判断。"),
            latex_escape("calculator：对赔偿、利息、税额、罚款等表达式做安全本地计算。"),
            latex_escape("ask_user：在缺失事实会改变结论时，仅追问最关键的一组信息。"),
        ]),
        "\\paragraph{评测协议} " + latex_escape(
            "作业要求的是 agent-level comparison，因此本项目使用同一套运行时分别评测基座模型与后训练模型，而不是只比对裸模型文本输出。量化部分报告格式遵从率、工具调用准确率、任务完成率、错误恢复能力以及文本质量指标，并配套给出并排案例分析。"
        ),
        "\\begin{table}[H]\n\\centering\n\\small\n\\begin{tabular}{lr}\n\\toprule\n语料项 & 数值 \\\\ \n\\midrule\n"
        + dataset_overview_rows_zh(artifacts)
        + "\n\\bottomrule\n\\end{tabular}\n\\caption{修复、解析、分块和索引构建后的知识库规模。}\n\\end{table}",
        "\\begin{table}[H]\n\\centering\n\\small\n\\begin{tabular}{ll}\n\\toprule\n设置项 & 数值 \\\\ \n\\midrule\n"
        + training_config_rows_zh(config)
        + "\n\\bottomrule\n\\end{tabular}\n\\caption{正式实验中的核心模型、训练与运行时配置。}\n\\end{table}",
    ])

    code_parts_zh = [
        "\\section{代码实现}",
        "\\paragraph{1. 整体软件架构} " + latex_escape(
            "整个项目被实现为一套完整实验系统，而不是一个临时脚本。数据层负责 Word 修复、目录归一化和法规分块；RAG 层负责索引构建与检索打分；Agent 层负责提示词、工具、解析器和状态机；训练层负责轨迹构造与 QLoRA；评测层负责运行时回放、逐样本指标与案例导出。"
        ),
        latex_itemize([
            latex_escape("data/：旧版 Word 修复、文档解析、法规元数据归一化、地域层级推断与 chunk 构造。"),
            latex_escape("rag/：向量编码、BM25、法规级 shortlist、引用图与混合检索逻辑。"),
            latex_escape("agent/：系统提示词、ReAct 解析器、工具注册表和运行时状态机。"),
            latex_escape("training/：DISC 种子筛选、Agent 轨迹规划/执行与 QLoRA 训练。"),
            latex_escape("evaluation/：启发式 judge、逐项指标和详细样本级评测文件。"),
        ]),
        "\\paragraph{2. 知识库构建方式与数据结构} " + latex_escape(
            "知识库来自混合的目录表和 Word 文档仓库。预处理会先修复旧版文档，再构建法规 manifest，其中每条法规都带有标题、归一化标题、效力层级、地域层级、日期、匹配状态和源文件路径。随后，文档被解析为“篇章标题/条文标题/正文块”结构，再切成带重叠的字符窗口，从而兼顾语义完整性和索引粒度。"
        ),
        latex_itemize([
            latex_escape("manifest 记录包含：title、normalized_title、category_raw、effect_level、effect_rank、jurisdiction_type、jurisdiction_scope、region_name、region_path_codes、region_path_names、promulgation_date、effective_date、version_date、status、source_path。"),
            latex_escape("chunk 记录包含：chunk_id、document_id、document_title、article_heading、section_context、法规层级字段、地域层级字段、cross_references、text、retrieval_text。"),
            latex_escape("retrieval_text 会把标题、效力层级、地域路径、条文标题和正文窗口拼接到同一字段中，使 dense 检索和 BM25 看到统一的归一化表示。"),
            latex_escape("语料最终写成 JSONL，便于抽查、调试和后续索引阶段复用。"),
        ]),
        "\\paragraph{3. 分层索引与检索架构} " + latex_escape(
            "检索器并不是简单的一层 top-k，而是一个两级混合 RAG。Chunk 级别同时维护 dense index 和 BM25；法规级别再维护一套 dense/BM25 用于 shortlist。构建 dense 向量时采用 numpy memmap 逐批写入，避免 60 余万 chunk 的全量矩阵常驻内存。在线检索时先做法规级 shortlist，再在候选 chunk 内做 reciprocal rank fusion，最后叠加效力层级、地域关系和引用图扩展信号。"
        ),
        latex_itemize([
            latex_escape("索引产物包括：dense_embeddings.npy、bm25.pkl、chunks.jsonl、document_embeddings.npy、document_bm25.pkl、document_records.json、doc_to_chunks.json、chunk_to_doc.json、graph.json、metadata.json。"),
            latex_escape("document_records 记录 document_id、normalized_title、法规层级、地域层级、chunk_positions 和法规级 retrieval_text，用于先筛法规再筛条文。"),
            latex_escape("地域打分不是只看字符串命中，而是比较 region_path_codes 的 exact、ancestor、descendant 关系；不相关的地方性法规会被降权。"),
            latex_escape("引用图扩展允许从当前高分法规扩展到其交叉引用法规，这对法律解释链尤其重要。"),
        ]),
        "\\paragraph{4. Query 理解与地点建模} " + latex_escape(
            "在真正检索之前，系统会构造 QueryContext，其中保存显式地点、是否疑似地方性法规问题、是否需要上位法加权、显式地点层级和解析后的完整地点对象。地点解析同时结合规则化行政区匹配与地址解析器输出，并在必要时做祖先层级回填，因此用户即便提到一个知识库元数据里没有直接列出的街道名，系统也能退回到对应区、市、省继续检索。"
        ),
        latex_itemize([
            latex_escape("系统在检索结果中同时返回 explicit_query_regions 和 resolved_query_location，方便下游判断是否还需要继续追问。"),
            latex_escape("当用户只给出省份或市级信息，而高分结果暗示更细粒度地方法规时，工具层会直接触发地点细化追问。"),
            latex_escape("这一层实现保证了地点不是只参与字符串过滤，而是真正参与打分与运行时控制。"),
        ]),
        "\\paragraph{5. Agent Style 数据集生成细节} " + latex_escape(
            "训练数据由 DISC-Law-SFT 种子程序化生成。每个 TrajectorySeed 都包含 question、expected_answer、references、expected_tools、scripted_answers、clarification_questions、calculator_expression、query_for_retrieval 和 force_error 等字段。TrajectoryBuilder 会先按策略规划工具序列，再真正执行本地工具，把 Observation 写回轨迹，最后生成可直接用于训练的 JSONL 样本。"
        ),
        latex_itemize([
            latex_escape("direct_answer 任务会生成零工具轨迹，用来教会模型在用户已经提供全文时不要多余检索。"),
            latex_escape("retrieve_then_answer 任务会先检索再回答，是最常见的 Agent 轨迹。"),
            latex_escape("lookup_then_retrieve 任务会先核对法规标题，再检索相关条文。"),
            latex_escape("error_recovery 样本通过故意注入错误标题或错误步骤，迫使轨迹展示失败后的重新规划能力。"),
            latex_escape("最终样本统一写入 sample_id、question、expected_answer、references、expected_tools、force_error、scripted_answers、metadata、messages、trace、tool_trace 等字段。"),
        ]),
    ]
    if direct_example_preview:
        code_parts_zh.extend([
            "\\paragraph{直接回答样例} " + latex_escape(
                "下面的样例展示了文书摘要类任务的数据格式：由于用户输入本身已包含完整正文，因此轨迹会直接给出 Thought + Final Answer，不调用任何工具。"
            ),
            direct_example_preview,
        ])
    if retrieve_example_preview:
        code_parts_zh.extend([
            "\\paragraph{工具调用样例} " + latex_escape(
                "下面的样例展示了检索型任务的数据格式：轨迹先发出具体检索查询，读入 Observation，再生成最终答案。"
            ),
            retrieve_example_preview,
        ])
    code_parts_zh.extend([
        "\\paragraph{6. Prompt 模板与解析契约} " + latex_escape(
            "提示词实现完全围绕 Agent 使用场景设计。系统提示词由法律角色定义、工具签名与返回格式、输出协议和行为规则组成。训练阶段保留完整 ReAct 轨迹；运行时则使用逐步输出版本，以防模型在工具执行前伪造 Observation。解析器 parse_react_output 负责从模型输出中抽取 Thought、Action 和 Final Answer，并支持 JSON 风格参数、单字符串参数和关键字参数三种写法，从而提高运行时鲁棒性。"
        ),
        prompt_preview,
        "\\paragraph{7. 工具注册表与工具语义} " + latex_escape(
            "所有工具都通过 ToolRegistry 对外暴露。注册表既负责把工具规格写入提示词，也负责把工具输出压缩成可回填到 scratchpad 的 Observation 文本，因此运行时既保留了机器可解析结构，也保留了人工可核验的证据内容。"
        ),
        latex_itemize([
            latex_escape("retrieve_from_kb 返回 query、显式地点、显式地点层级、解析后的地点对象、是否需要继续追问地点，以及带标题/条款/地域/分数/正文片段的检索结果列表。"),
            latex_escape("lookup_statute 在 manifest 上做精确或别名匹配，返回法规元数据和条文预览。"),
            latex_escape("resolve_hierarchy 把法规标题或类型映射到效力层级，用于冲突与适用顺序分析。"),
            latex_escape("calculator 通过受限 AST 解释器执行白名单算术运算，避免任意代码执行。"),
            latex_escape("ask_user 既可在数据生成阶段读取脚本化答案，也可在 Web UI 或命令行中进入真实交互模式，但输出统一归一化为 question/answer 结构。"),
        ]),
        "\\paragraph{8. Agent 工作流、状态转移与错误回退} " + latex_escape(
            "在线 Agent 不是一次性生成，而是一个显式状态机。它先做 turn analysis，判断当前输入是新问题、补充事实还是对上一轮追问的回答；随后根据 history、当前用户输入和已有 scratchpad 构造消息；模型输出后由解析器判定是 tool step 还是 final step；若是工具调用则进入 tool node 执行并写回 Observation；若输出不合规则自动重试。运行时还专门处理重复工具调用、重复追问、格式错误、工具失败、最大步数上限以及“暂定答案仍不够好”的情况。"
        ),
        latex_itemize([
            latex_escape("格式错误会在 scratchpad 中插入 FORMAT_ERROR Observation，并触发有限次自动重试。"),
            latex_escape("如果模型重复发出相同工具调用，系统会阻断循环并转为基于现有证据综合最终答案。"),
            latex_escape("澄清问题会被标准化、去重并设上限；达到上限后系统转为条件式分析，而不是无限 ask_user。"),
            latex_escape("当 retrieve_from_kb 判断地点仍不足时，当前轮会直接结束为“请先补充地点”的追问结果。"),
            latex_escape("最终收束阶段会禁用工具，只允许模型根据已有 scratchpad 组织“已知事实、适用依据、结论、仍需确认事项”。"),
        ]),
    ])
    if runtime_trace_preview:
        code_parts_zh.extend([
            "\\paragraph{运行时轨迹示例} " + latex_escape(
                "下面的片段展示了部署时真实使用的 ReAct 轨迹形式：Thought、Action、系统注入的 Observation，以及最终答案。"
            ),
            runtime_trace_preview,
        ])
    code_parts_zh.extend([
        "\\paragraph{9. 训练与评测代码实现} " + latex_escape(
            f"QLoRA 训练采用 completion-only 监督，4-bit 加载，LoRA rank={config.training.lora_r}，alpha={config.training.lora_alpha}，dropout={config.training.lora_dropout}，最大序列长度={config.training.max_seq_length}，并按验证损失选择最佳 checkpoint。评测阶段则把基座模型和后训练模型都放入同一套 Agent runtime 中运行，再统一计算作业要求的各项指标，并将摘要与逐样本明细分别写入 JSON/JSONL 文件。"
        ),
        latex_itemize([
            latex_escape("训练产物包括 adapter checkpoint、tokenizer 文件以及 training_metrics.json。"),
            latex_escape("评测产物包括 eval_summary.json（含 metric support 与权重）和 eval_details.jsonl（含问题、答案、trace、tool_history 和逐样本指标）。"),
            latex_escape("这种文件布局保证了后续案例分析、失败分析和最终结果汇总都能直接回溯到原始运行痕迹。"),
        ]),
    ])
    code_implementation = "\n".join(part for part in code_parts_zh if part)

    result_analysis = "\n".join([
        "\\section{结果分析}",
        "\\paragraph{评测设置} " + latex_escape(
            f"正式 held-out 集共有 {eval_stats['total']} 条轨迹，其中 {eval_stats['reference_count']} 条带参考法条，{eval_stats['tool_required_count']} 条期望至少调用一次工具，{eval_stats['force_error_count']} 条包含显式错误恢复压力。评测策略分布为 {counter_text_zh(eval_stats['strategy_counts'], STRATEGY_LABELS_ZH)}。"
        ),
        "\\paragraph{指标设计} " + latex_escape(
            "量化协议与作业要求严格对应。格式遵从率衡量 ReAct 外壳是否正确；工具调用准确率比较真实工具轨迹与期望工具轨迹；任务完成率依据答案质量和引用感知阈值决定；错误恢复能力衡量失败后是否继续规划；精确匹配、Token F1、语义相似度、答案质量和法条引用命中率共同衡量最终答案的文本与依据质量。对于当前切分中没有支持样本的指标，综合分分母会自动剔除该项，而不是机械记 0。"
        ),
        "\\paragraph{为什么还需要定性验证} " + latex_escape(
            "正式 held-out 集来自 DISC-Law-SFT，因此在一般法律问答上覆盖较好，但对地方性法规追问、多轮地点补充这类交互行为覆盖不足。因而除了量化表格之外，报告还需要单独列出运行时验证案例，避免对单个综合分做过度解释。"
        ),
        "\\paragraph{训练摘要} " + latex_escape(
            f"训练损失={artifacts.training_metrics.get('train_loss', 'N/A')}，验证损失={artifacts.training_metrics.get('eval_loss', 'N/A')}，训练耗时={artifacts.training_metrics.get('train_runtime', 'N/A')} 秒，最佳 checkpoint={artifacts.training_metrics.get('best_checkpoint', 'N/A')}。"
        ),
        "\\begin{table}[H]\n\\centering\n\\scriptsize\n\\begin{tabular}{lrrrrr}\n\\toprule\n指标 & 基座支持数 & 基座 & 后训练支持数 & 后训练 & 差值 \\\\ \n\\midrule\n"
        + metric_table_rows_zh(
            artifacts.base_summary,
            artifacts.adapted_summary,
            comparable=artifacts.eval_splits_match,
        )
        + "\n\\bottomrule\n\\end{tabular}\n\\caption{基座结果来自"
        + latex_escape(artifacts.base_eval_source_zh)
        + "；后训练结果来自"
        + latex_escape(artifacts.adapted_eval_source_zh)
        + "。}\n\\end{table}",
        latex_itemize(quantitative_insights_zh(artifacts, eval_stats, tool_stats)),
        "\\paragraph{指标解读} " + latex_escape(
            f"在当前可用的后训练结果（{artifacts.adapted_eval_source_zh}）中，模型在 {tool_stats['samples_with_tools']} 个样本上至少调用过一次工具，在 {tool_stats['multi_tool_samples']} 个样本上执行了多步工具链，在 {tool_stats['ask_user_samples']} 个样本上调用了 ask_user。Agent 后训练是否有效，不仅要看文本答案，还要看模型是否真的在运行时改变了工具使用行为。"
        ),
        "\\subsection{并排案例分析}\n" + render_case_studies_zh(case_studies),
        (
            "\\subsection{代表性工具行为分析}\n"
            + latex_escape(
                "由于 DISC held-out 切分并不会均衡覆盖所有工具，下面补充若干定向运行时案例，用来展示检索、法规核验、层级判断、追问与计算等工具在实际 Agent 中分别如何发挥作用。"
            )
            + "\n"
            + render_tool_behavior_cases_zh(tool_behavior_cases)
        ) if tool_behavior_cases else "",
        "\\subsection{失败案例与讨论}\n"
        + latex_itemize([latex_escape(f"{count} 个样本表现为{reason}。") for reason, count in failure_summary.most_common(5)] or [latex_escape("当前没有可统计的失败模式。")])
        + "\n"
        + render_failure_cases_zh(failure_cases),
        (
            "\\subsection{地点理解与多轮对话补充说明}\n"
            + latex_escape(
                "如果代表性工具案例中尚未覆盖地点追问与多轮延续，则保留一条简短补充说明，用于交代部署时验证过的地点相关行为。"
            )
            + "\n"
            + render_runtime_validations_zh()
        ) if not tool_behavior_cases else "",
    ])

    conclusion = "\n".join([
        "\\section{结论}",
        "\\paragraph{结论} " + latex_escape(
            "本项目说明，法律 Agent 的能力提升不是单靠一次微调就能完成的。最终效果同时取决于知识库是否可解析、元数据是否规范、检索是否分层、Prompt 和工具协议是否清晰、运行时控制是否严谨，以及后训练是否真正把模型行为往 Agent 方向推。"
        ),
        "\\paragraph{关键收获} " + latex_escape(
            "最重要的三点收获是：第一，Agent 风格轨迹比普通 QA 更能教会模型何时检索、何时停止、何时恢复；第二，地点层级在法律问答中不是附属细节，而是直接决定适用法规范围的核心变量；第三，逐样本 trace 和失败案例分析比单一综合分更能解释后训练到底改变了什么。"
        ),
    ])

    latex_slash = "\\"

    return dedent(
        rf"""
        \documentclass[12pt]{{ctexart}}
        \usepackage[a4paper,margin=1in]{{geometry}}
        \usepackage{{booktabs}}
        \usepackage{{array}}
        \usepackage{{longtable}}
        \usepackage{{enumitem}}
        \usepackage{{hyperref}}
        \usepackage{{graphicx}}
        \usepackage{{float}}
        \usepackage[most]{{tcolorbox}}
        \hypersetup{{colorlinks=true,linkcolor=blue,urlcolor=blue}}
        {latex_slash}title{{中文法律 Agent 后训练实验报告}}
        \author{{赵一锦\\225010231\\\texttt{{225010231@link.cuhk.edu.cn}}}}
        \date{{2026 年 4 月}}
        \begin{{document}}
        \maketitle

        \begin{{abstract}}
        {latex_escape(abstract)}
        \end{{abstract}}

        {research_topic}

        {experiment_design}

        {code_implementation}

        {result_analysis}

        {conclusion}

        \end{{document}}
        """
    ).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Assignment 3 English and Chinese LaTeX reports.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "defaults.yaml"), help="Path to the experiment config file.")
    parser.add_argument("--en-output", default=str(DOCS_DIR / "report_en.tex"), help="Output path for the English report.")
    parser.add_argument("--zh-output", default=str(DOCS_DIR / "report_zh.tex"), help="Output path for the Chinese report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    en_output = Path(args.en_output).resolve()
    zh_output = Path(args.zh_output).resolve()

    artifacts = load_experiment_artifacts(config_path)
    en_output.parent.mkdir(parents=True, exist_ok=True)
    zh_output.parent.mkdir(parents=True, exist_ok=True)
    en_output.write_text(build_en_report(artifacts), encoding="utf-8")
    zh_output.write_text(build_zh_report(artifacts), encoding="utf-8")
    print(
        json.dumps(
            {
                "report_en": str(en_output),
                "report_zh": str(zh_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()