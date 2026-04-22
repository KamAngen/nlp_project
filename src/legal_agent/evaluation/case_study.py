from __future__ import annotations

from pathlib import Path

from legal_agent.utils.io import read_jsonl


def _metric_line(label: str, row: dict) -> str:
    return (
        f"- {label}: overall={row['overall_score']:.3f}, quality={row['answer_quality']:.3f}, "
        f"tool={row['tool_use_accuracy']:.3f}, citation={row['citation_hit_rate']:.3f}, completion={row['task_completion']:.3f}"
    )


def _analysis(base: dict, adapted: dict) -> list[str]:
    notes: list[str] = []
    if adapted["overall_score"] > base["overall_score"]:
        notes.append("后训练模型总体分更高。")
    else:
        notes.append("该案例中后训练模型未超过基座模型，需要结合误差来源分析。")
    if adapted["tool_use_accuracy"] > base["tool_use_accuracy"]:
        notes.append("后训练模型的工具调用更接近期望轨迹。")
    if adapted["citation_hit_rate"] > base["citation_hit_rate"]:
        notes.append("后训练模型的法条引用命中率更高。")
    if adapted["format_compliance"] > base["format_compliance"]:
        notes.append("后训练模型在 ReAct 格式遵从上更稳定。")
    if adapted["answer_quality"] > base["answer_quality"]:
        notes.append("后训练模型的最终答案更接近参考答案。")
    if not notes:
        notes.append("两者表现接近，适合作为误差分析案例。")
    return notes


def export_case_studies(
    base_results_path: str | Path,
    adapted_results_path: str | Path,
    output_markdown_path: str | Path,
    *,
    max_examples: int = 3,
) -> Path:
    base_results = {row["sample_id"]: row for row in read_jsonl(base_results_path)}
    adapted_results = {row["sample_id"]: row for row in read_jsonl(adapted_results_path)}

    selected = []
    for sample_id, adapted in adapted_results.items():
        base = base_results.get(sample_id)
        if base is None:
            continue
        if adapted["task_completion"] > base["task_completion"]:
            selected.append((sample_id, base, adapted))
    if len(selected) < max_examples:
        for sample_id, adapted in adapted_results.items():
            base = base_results.get(sample_id)
            if base is None:
                continue
            triple = (sample_id, base, adapted)
            if triple not in selected:
                selected.append(triple)
            if len(selected) >= max_examples:
                break

    lines = ["# Agent Case Studies", ""]
    for sample_id, base, adapted in selected[:max_examples]:
        lines.append(f"## {sample_id}")
        lines.append("")
        lines.append(f"**Question**: {adapted['question']}")
        lines.append("")
        lines.append(f"**Expected Answer**: {adapted['expected_answer']}")
        lines.append("")
        lines.append("**Metrics**")
        lines.append("")
        lines.append(_metric_line("Base", base))
        lines.append(_metric_line("Post-trained", adapted))
        lines.append("")
        lines.append("**Base Model**")
        lines.append("")
        lines.append(base["final_answer"])
        lines.append("")
        lines.append("```text")
        lines.append(base["trace"])
        lines.append("```")
        lines.append("")
        lines.append("**Post-trained Model**")
        lines.append("")
        lines.append(adapted["final_answer"])
        lines.append("")
        lines.append("```text")
        lines.append(adapted["trace"])
        lines.append("```")
        lines.append("")
        lines.append("**Analysis**")
        lines.append("")
        for note in _analysis(base, adapted):
            lines.append(f"- {note}")
        lines.append("")

    output_markdown_path = Path(output_markdown_path)
    output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    output_markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return output_markdown_path
