from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from legal_agent.agent.engine import LegalAgentEngine
from legal_agent.agent.tools import ToolRegistry
from legal_agent.config import AppConfig, load_app_config
from legal_agent.evaluation.judge import HeuristicJudge
from legal_agent.evaluation.metrics import (
    OVERALL_SCORE_WEIGHTS,
    TASK_COMPLETION_THRESHOLDS,
    answer_quality_score,
    citation_hit_rate,
    error_recovery_score,
    format_compliance,
    overall_score,
    summarize_metrics,
    tool_use_accuracy,
)
from legal_agent.models.qwen_local import LocalQwenChatModel
from legal_agent.rag.retriever import HybridLegalRetriever
from legal_agent.utils.io import read_jsonl, write_json, write_jsonl


def _metric_support_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
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
        metric_name: sum(1 for row in rows if row.get(metric_name) is not None)
        for metric_name in metric_names
    }


def _semantic_similarity(retriever: HybridLegalRetriever, answer: str, expected_answer: str) -> float:
    if not answer.strip() or not expected_answer.strip():
        return 0.0
    embeddings = retriever.embedder.encode([answer, expected_answer], batch_size=2)
    return float(np.dot(embeddings[0], embeddings[1]))


def evaluate_model(
    config: AppConfig,
    *,
    dataset_path: str | Path,
    model_path: str | Path | None = None,
    adapter_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    retrieval_device: str = "cpu",
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = config.resolve_project_path(dataset_path)
    output_dir = Path(output_dir or (config.output_root / "eval"))
    if not output_dir.is_absolute():
        output_dir = config.resolve_project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retriever = HybridLegalRetriever(config, device=retrieval_device)
    model = LocalQwenChatModel(
        model_path or config.models.agent_base,
        adapter_path=adapter_path,
        device_map="auto",
        load_in_4bit=config.inference.load_in_4bit,
        compute_dtype=config.inference.compute_dtype,
    )
    judge = HeuristicJudge()
    samples = read_jsonl(dataset_path)
    results: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc="Evaluate agent"):
        registry = ToolRegistry(retriever, scripted_answers=sample.get("scripted_answers", {}), interactive=False)
        engine = LegalAgentEngine(
            model,
            registry,
            max_steps=config.inference.max_steps,
            max_new_tokens=config.inference.max_new_tokens,
            temperature=config.inference.temperature,
            top_p=config.inference.top_p,
            top_k=config.inference.top_k,
            presence_penalty=config.inference.presence_penalty,
            enable_thinking=config.inference.enable_thinking,
        )
        result = engine.run(sample["question"])
        heuristic = judge.judge(result.final_answer, sample["expected_answer"], sample.get("references", []))
        citation_score = citation_hit_rate(result.final_answer, sample.get("references", []))
        tool_score = tool_use_accuracy(result.tool_history, sample.get("expected_tools", []))
        semantic_score = _semantic_similarity(retriever, result.final_answer, sample["expected_answer"])
        answer_quality = answer_quality_score(heuristic["exact_match"], heuristic["answer_f1"], semantic_score)
        task_completion = 1.0 if (
            answer_quality >= TASK_COMPLETION_THRESHOLDS["answer_quality"]
            or (
                citation_score is not None
                and semantic_score >= TASK_COMPLETION_THRESHOLDS["semantic_similarity_with_citation"]
                and citation_score > 0
            )
        ) else 0.0
        row = {
            "sample_id": sample["sample_id"],
            "question": sample["question"],
            "expected_answer": sample["expected_answer"],
            "final_answer": result.final_answer,
            "trace": result.trace,
            "tool_history": result.tool_history,
            "format_compliance": format_compliance(result.trace),
            "tool_use_accuracy": tool_score,
            "task_completion": task_completion,
            "citation_hit_rate": citation_score,
            "error_recovery": error_recovery_score(result.trace, sample.get("force_error", False)),
            "answer_exact_match": heuristic["exact_match"],
            "answer_f1": heuristic["answer_f1"],
            "semantic_similarity": semantic_score,
            "answer_quality": answer_quality,
            "errors": result.errors,
        }
        row["overall_score"] = overall_score(row)
        results.append(row)

    summary = summarize_metrics(results)
    metric_support = _metric_support_counts(results)
    write_json(
        output_dir / "eval_summary.json",
        {
            "metrics": summary,
            "metric_support": metric_support,
            "weights": OVERALL_SCORE_WEIGHTS,
            "task_completion_thresholds": TASK_COMPLETION_THRESHOLDS,
        },
    )
    write_jsonl(output_dir / "eval_details.jsonl", results)
    return {
        "summary": summary,
        "metric_support": metric_support,
        "details_path": config.project_relative_path(output_dir / "eval_details.jsonl"),
        "weights": OVERALL_SCORE_WEIGHTS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a legal agent model on the held-out dataset.")
    parser.add_argument("--config", default="configs/defaults.yaml")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--retrieval-device", default="cpu")
    args = parser.parse_args()

    config = load_app_config(args.config)
    dataset_path = args.dataset_path or config.generated_eval_path
    payload = evaluate_model(
        config,
        dataset_path=dataset_path,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        retrieval_device=args.retrieval_device,
    )
    print(payload)


if __name__ == "__main__":
    main()
