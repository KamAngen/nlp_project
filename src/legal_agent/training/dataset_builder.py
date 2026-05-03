from __future__ import annotations

import json
import random
import shutil
from typing import Any

from tqdm import tqdm

from legal_agent.config import AppConfig
from legal_agent.study_config import load_study_agent_config
from legal_agent.study_tools import StudyToolExecutor
from legal_agent.data.corpus_builder import build_law_corpus
from legal_agent.data.disc_law import download_disc_law_dataset, normalize_disc_law_dataset
from legal_agent.rag.index_builder import build_rag_index
from legal_agent.rag.retriever import HybridLegalRetriever
from legal_agent.training.trajectory_builder import TrajectoryBuilder, TrajectorySeed, extract_reference_titles
from legal_agent.utils.io import append_jsonl, ensure_dir, write_json, write_jsonl
from context_engine.manager import MemoryManager
from context_engine.store import DiskMemoryStore
from rag_engine.service import KnowledgeService


DISC_SUBSET_NAMES = (
    "DISC-Law-SFT-Pair-QA-released",
    "DISC-Law-SFT-Pair",
    "DISC-Law-SFT-Triplet-QA-released",
    "DISC-Law-SFT-Triplet-released",
)

DIRECT_ANSWER_TASK_FAMILIES = {
    "jud_doc_sum",
    "jud_read_compre",
    "leg_case_cls",
    "leg_eve_detec",
    "op_sum",
    "sim_case_match",
}

RETRIEVAL_TASK_FAMILIES = {
    "legal_question_answering",
    "exam",
    "sent_pred",
    "judgement_predit",
}

SUPPORTED_TASK_FAMILIES = DIRECT_ANSWER_TASK_FAMILIES | RETRIEVAL_TASK_FAMILIES


def _extract_question(instruction: str, task_family: str) -> str:
    if task_family == "legal_question_answering":
        for marker in ("<问题>：", "<问题>:", "问题：", "问题:"):
            if marker in instruction:
                return instruction.split(marker)[-1].strip()
    return instruction.strip()


def _compact_text(text: str, *, max_chars: int = 280) -> str:
    flattened = " ".join(str(text or "").split())
    if len(flattened) <= max_chars:
        return flattened
    return flattened[: max_chars - 1].rstrip(" ，,；;：:") + "…"


def _compress_self_contained_prompt(text: str, *, max_chars: int = 2800) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= max_chars:
        return cleaned

    head = cleaned[:1600].rstrip()
    tail_start = max_chars - 900
    markers = ("本院认为", "判决如下", "裁定如下", "判决结果", "法院认为", "结论")
    marker_pos = [cleaned.find(marker) for marker in markers if cleaned.find(marker) != -1]
    if marker_pos:
        start = min(marker_pos)
        tail = cleaned[max(start - 280, 1600) : start + 900].strip()
    else:
        tail = cleaned[-900:].strip()
    return head + "\n\n[中间内容截断]\n\n" + tail


def _prepare_question_text(record: dict[str, Any]) -> str:
    task_family = str(record["task_family"])
    question = _extract_question(record["instruction"], task_family)
    if task_family in DIRECT_ANSWER_TASK_FAMILIES:
        return _compress_self_contained_prompt(question)
    return question.strip()


def _build_retrieval_query(task_family: str, question: str, reference_titles: list[str]) -> str | None:
    if task_family in DIRECT_ANSWER_TASK_FAMILIES:
        return None

    query_parts = [_compact_text(question, max_chars=260)]
    if task_family == "exam":
        query_parts.append("法条 选项 解析")
    if task_family in {"sent_pred", "judgement_predit"}:
        query_parts.append("定罪 量刑 判决")
    if reference_titles:
        query_parts.extend(f"《{title}》" for title in reference_titles[:2])

    seen: set[str] = set()
    deduped: list[str] = []
    for part in query_parts:
        cleaned = str(part or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return " ".join(deduped)


def _disc_seed_strategy(task_family: str, references: list[str]) -> tuple[bool, bool, str]:
    if task_family in DIRECT_ANSWER_TASK_FAMILIES:
        return False, False, "direct_answer"
    if references:
        return True, True, "lookup_then_retrieve"
    return False, True, "retrieve_then_answer"


def _build_disc_seed(record: dict[str, Any]) -> TrajectorySeed | None:
    subset_name = str(record["subset_name"])
    task_family = str(record["task_family"])
    if subset_name not in DISC_SUBSET_NAMES or task_family not in SUPPORTED_TASK_FAMILIES:
        return None

    question = _prepare_question_text(record)
    if len(question) < 6:
        return None
    if len(question) > 3600:
        return None

    references = list(record["references"])
    reference_titles = extract_reference_titles(references)
    requires_lookup, requires_retrieval, strategy = _disc_seed_strategy(task_family, references)
    retrieval_query = _build_retrieval_query(task_family, question, reference_titles)

    expected_tools: list[str] = []
    if requires_lookup and reference_titles:
        expected_tools.append("lookup_statute")
    if requires_retrieval:
        expected_tools.append("retrieve_from_kb")

    sampling_bucket = f"{subset_name}::{task_family}"
    return TrajectorySeed(
        seed_id=f"disc::{subset_name}::{record['sample_id']}",
        question=question,
        expected_answer=record["answer"],
        source=subset_name,
        sampling_bucket=sampling_bucket,
        references=references,
        expected_tools=expected_tools,
        query_for_retrieval=retrieval_query,
        force_error=bool(requires_lookup and references and str(record["sample_id"]).endswith("0")),
        metadata={
            "task_family": task_family,
            "subset_name": subset_name,
            "has_references": bool(references),
            "requires_lookup": requires_lookup,
            "requires_retrieval": requires_retrieval,
            "strategy": strategy,
            "sampling_bucket": sampling_bucket,
            "answer_source": "disc",
        },
    )


def _build_disc_seed_buckets(records: list[dict[str, Any]]) -> dict[str, list[TrajectorySeed]]:
    buckets: dict[str, list[TrajectorySeed]] = {}
    for record in records:
        seed = _build_disc_seed(record)
        if seed is None:
            continue
        bucket_name = seed.sampling_bucket or seed.source
        buckets.setdefault(bucket_name, []).append(seed)

    for index, bucket_name in enumerate(sorted(buckets)):
        random.Random(2026 + index).shuffle(buckets[bucket_name])
    return buckets


def _draw_balanced_seeds(
    seed_buckets: dict[str, list[TrajectorySeed]],
    limit: int,
) -> tuple[list[TrajectorySeed], dict[str, list[TrajectorySeed]]]:
    if limit <= 0:
        return [], {bucket_name: list(seeds) for bucket_name, seeds in seed_buckets.items()}

    buckets = {bucket_name: list(seeds) for bucket_name, seeds in seed_buckets.items()}
    active = [bucket_name for bucket_name, seeds in sorted(buckets.items()) if seeds]
    selected: list[TrajectorySeed] = []
    turn = 0

    while len(selected) < limit and active:
        bucket_name = active[turn % len(active)]
        selected.append(buckets[bucket_name].pop())
        if not buckets[bucket_name]:
            active.remove(bucket_name)
            if active:
                turn %= len(active)
        else:
            turn += 1

    if len(selected) < limit:
        raise RuntimeError(f"可用 DISC seeds 数量不足：需要 {limit} 条，当前只有 {len(selected)} 条。")
    return selected, buckets


def _count_seeds_by_source(seeds: list[TrajectorySeed]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for seed in seeds:
        counts[seed.source] = counts.get(seed.source, 0) + 1
    return {source: count for source, count in sorted(counts.items()) if count > 0}


def _count_seeds_by_task_family(seeds: list[TrajectorySeed]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for seed in seeds:
        task_family = str(seed.metadata.get("task_family") or seed.metadata.get("scenario") or "unknown")
        counts[task_family] = counts.get(task_family, 0) + 1
    return {task_family: count for task_family, count in sorted(counts.items()) if count > 0}


def _count_seeds_by_strategy(seeds: list[TrajectorySeed]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for seed in seeds:
        strategy = str(seed.metadata.get("strategy") or "unknown")
        counts[strategy] = counts.get(strategy, 0) + 1
    return {strategy: count for strategy, count in sorted(counts.items()) if count > 0}


def _shuffle_stably(seeds: list[TrajectorySeed], seed: int) -> list[TrajectorySeed]:
    copied = list(seeds)
    random.Random(seed).shuffle(copied)
    return copied


def _read_jsonl_records(path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            records.append(json.loads(payload))
    return records


def _build_local_study_seeds(study_config) -> list[TrajectorySeed]:
    question_rows = _read_jsonl_records(study_config.question_bank_path)
    case_rows = _read_jsonl_records(study_config.case_bank_path)
    common_rows = _read_jsonl_records(study_config.common_knowledge_path)
    seeds: list[TrajectorySeed] = []

    seeds.extend(
        [
            TrajectorySeed(
                seed_id="study::profile::civil",
                question="我在备考民法，我的薄弱点是租赁合同和押金返还，每天能学2小时，请记住。",
                expected_answer="已更新学习画像，并确认当前重点是民法中的租赁合同与押金返还。",
                source="study_profile",
                expected_tools=["profile_upsert", "profile_view"],
                metadata={
                    "scenario": "profile_update",
                    "profile_updates": {"study_goals": ["民法"], "weak_points": ["租赁合同", "押金返还"], "preferences": {"daily_hours": 2.0}},
                },
            ),
            TrajectorySeed(
                seed_id="study::profile::admin",
                question="我最近主攻行政法，我的薄弱点是听证程序，目标分数是120分。",
                expected_answer="已更新画像，当前学习目标为行政法，薄弱点集中在听证程序。",
                source="study_profile",
                expected_tools=["profile_upsert", "profile_view"],
                metadata={
                    "scenario": "profile_update",
                    "profile_updates": {"study_goals": ["行政法"], "weak_points": ["听证程序"], "target_score": 120},
                },
            ),
        ]
    )

    for row in common_rows[:3]:
        title = str(row.get("title") or "法考学习方法")
        content = str(row.get("content") or "")
        tags = [str(tag) for tag in row.get("tags", [])]
        seeds.append(
            TrajectorySeed(
                seed_id=f"study::method::{row.get('entry_id')}",
                question=f"我在准备法考，请结合“{title}”给我一个简洁的复习建议。",
                expected_answer=content,
                source="study_common",
                expected_tools=["prepare_context", "rag_search"],
                query_for_retrieval=f"{title} {' '.join(tags)} {content}",
                metadata={
                    "scenario": "study_method_qa",
                    "profile_updates": {"study_goals": ["法考"]},
                },
            )
        )

    for row in case_rows[:3]:
        issue = str(row.get("issue") or "")
        holding = str(row.get("holding") or "")
        tags = [str(tag) for tag in row.get("tags", [])]
        seeds.append(
            TrajectorySeed(
                seed_id=f"study::case::{row.get('case_id')}",
                question=f"请结合案例“{row.get('title')}”讲清楚：{issue}",
                expected_answer=holding,
                source="study_case",
                expected_tools=["prepare_context", "rag_search"],
                query_for_retrieval=f"{row.get('title')} {issue} {' '.join(tags)}",
                references=[f"《{title}》" for title in row.get("statutes", [])],
                metadata={
                    "scenario": "study_case_analysis",
                    "profile_updates": {"study_goals": tags[:1], "weak_points": tags[1:2]},
                },
            )
        )

    seen_topics: set[str] = set()
    for row in question_rows:
        topic = str(row.get("topic") or "综合").strip() or "综合"
        if topic in seen_topics:
            continue
        seen_topics.add(topic)
        analysis = str(row.get("analysis") or "")
        tags = [str(tag) for tag in row.get("tags", [])]
        seeds.append(
            TrajectorySeed(
                seed_id=f"study::statute::{row.get('question_id')}",
                question=f"我在复习{topic}，请解释这道题为什么选{row.get('answer')}：{row.get('question')}",
                expected_answer=analysis,
                source="study_question",
                expected_tools=["prepare_context", "rag_search", "retrieve_from_kb"],
                query_for_retrieval=f"{row.get('question')} {' '.join(tags)}",
                metadata={
                    "scenario": "study_statute_qa",
                    "profile_updates": {"study_goals": [topic], "weak_points": tags[:1]},
                },
            )
        )
        seeds.append(
            TrajectorySeed(
                seed_id=f"study::exam::{row.get('question_id')}",
                question=f"请按照我当前的薄弱点，给我出一套 {topic} 的两题模拟测试。",
                expected_answer=f"已生成一套围绕 {topic} 的模拟测试。",
                source="study_exam",
                expected_tools=["profile_view", "prepare_context", "generate_exam"],
                query_for_retrieval=topic,
                metadata={
                    "scenario": "mock_exam_generate",
                    "topic": topic,
                    "question_count": 2,
                    "profile_updates": {"study_goals": [topic], "weak_points": tags[:1]},
                },
            )
        )
        if len(seen_topics) >= 3:
            break

    seeds.extend(
        [
            TrajectorySeed(
                seed_id="study::report::progress",
                question="请根据我最近的学习情况生成一份学习进度报告。",
                expected_answer="已生成一份学习进度报告，并概括当前画像、会话进展和最近一次测试。",
                source="study_report",
                expected_tools=["prepare_context", "generate_report"],
                metadata={
                    "scenario": "report_generation",
                    "report_type": "study_progress",
                    "profile_updates": {"study_goals": ["法考"], "weak_points": ["听证程序"]},
                },
            ),
            TrajectorySeed(
                seed_id="study::report::diagnosis",
                question="请给我出一份偏重薄弱点诊断的学习报告。",
                expected_answer="已生成一份薄弱点诊断报告，强调最近会话和易错主题。",
                source="study_report",
                expected_tools=["prepare_context", "generate_report"],
                metadata={
                    "scenario": "report_generation",
                    "report_type": "weakness_diagnosis",
                    "profile_updates": {"study_goals": ["民法"], "weak_points": ["押金返还", "合同解除"]},
                },
            ),
        ]
    )
    return seeds


def _split_disc_seed_pool(
    disc_seed_buckets: dict[str, list[TrajectorySeed]],
    *,
    train_target: int,
    eval_target: int,
) -> tuple[list[TrajectorySeed], list[TrajectorySeed]]:
    total_target = train_target + eval_target
    if total_target <= 0:
        return [], []

    selected_pool, _ = _draw_balanced_seeds(disc_seed_buckets, total_target)
    selected_buckets: dict[str, list[TrajectorySeed]] = {}
    for seed in selected_pool:
        selected_buckets.setdefault(seed.sampling_bucket or seed.source, []).append(seed)

    eval_seeds, remaining_disc = _draw_balanced_seeds(selected_buckets, eval_target)
    train_seeds, _ = _draw_balanced_seeds(remaining_disc, train_target)
    return _shuffle_stably(train_seeds, 5050), _shuffle_stably(eval_seeds, 4040)


def build_agent_datasets(
    config: AppConfig,
    *,
    study_config_path: str = "configs/study_agent.yaml",
    train_count: int | None = None,
    eval_count: int | None = None,
    retrieval_device: str = "cpu",
) -> dict[str, Any]:
    ensure_dir(config.disc_law_dir)
    ensure_dir(config.generated_data_dir)

    if not config.disc_law_raw_dir.exists() or not list(config.disc_law_raw_dir.glob("*.jsonl")):
        download_disc_law_dataset(config.disc_law_raw_dir)
    disc_records = normalize_disc_law_dataset(config.disc_law_raw_dir, config.disc_law_normalized_path)

    if not config.corpus_path.exists():
        build_law_corpus(config)
    if not (config.rag_dir / "dense_embeddings.npy").exists():
        build_rag_index(config, device=retrieval_device)

    retriever = HybridLegalRetriever(config, device=retrieval_device)
    train_target = train_count or config.generation.train_trajectory_count
    eval_target = eval_count or config.generation.eval_trajectory_count

    study_config = load_study_agent_config(study_config_path)
    training_memory_root = config.generated_data_dir / "training_memory"
    training_report_root = config.generated_data_dir / "training_reports"
    if training_memory_root.exists():
        shutil.rmtree(training_memory_root)
    if training_report_root.exists():
        shutil.rmtree(training_report_root)
    memory_manager = MemoryManager(
        DiskMemoryStore(training_memory_root),
        system_seed_path=study_config.system_memory_path,
    )
    knowledge_service = KnowledgeService(
        question_bank_path=study_config.question_bank_path,
        case_bank_path=study_config.case_bank_path,
        common_knowledge_path=study_config.common_knowledge_path,
        use_legacy_statute_rag=study_config.use_legacy_statute_rag,
        legacy_config_path=study_config.legacy_config_path,
        legacy_device=retrieval_device,
    )
    study_tool_executor = StudyToolExecutor(
        memory_manager,
        knowledge_service,
        report_root=training_report_root,
    )
    local_seed_pool = _shuffle_stably(_build_local_study_seeds(study_config), 3030)
    split_index = max(1, int(len(local_seed_pool) * 0.75)) if local_seed_pool else 0
    local_train_seeds = local_seed_pool[: min(split_index, train_target)]
    local_eval_seeds = local_seed_pool[split_index : split_index + eval_target]
    if not local_eval_seeds and len(local_train_seeds) > 1 and eval_target > 0:
        local_eval_seeds = [local_train_seeds.pop()]

    disc_seed_buckets = _build_disc_seed_buckets(disc_records)
    disc_limit = min(
        max(config.generation.disc_seed_count, (train_target - len(local_train_seeds)) + (eval_target - len(local_eval_seeds))),
        sum(len(seeds) for seeds in disc_seed_buckets.values()),
    )
    disc_seed_pool, _ = _draw_balanced_seeds(disc_seed_buckets, disc_limit)
    disc_pool_buckets: dict[str, list[TrajectorySeed]] = {}
    for seed in disc_seed_pool:
        disc_pool_buckets.setdefault(seed.sampling_bucket or seed.source, []).append(seed)

    train_seeds, eval_seeds = _split_disc_seed_pool(
        disc_pool_buckets,
        train_target=max(train_target - len(local_train_seeds), 0),
        eval_target=max(eval_target - len(local_eval_seeds), 0),
    )

    train_seeds = _shuffle_stably(local_train_seeds + train_seeds, 7070)
    eval_seeds = _shuffle_stably(local_eval_seeds + eval_seeds, 8080)

    builder = TrajectoryBuilder(retriever, study_tool_executor=study_tool_executor)

    for path in (config.generated_train_path, config.generated_eval_path, config.seed_manifest_path):
        if path.exists():
            path.unlink()

    write_jsonl(
        config.seed_manifest_path,
        [{**seed.to_record(), "split": "train"} for seed in train_seeds]
        + [{**seed.to_record(), "split": "eval"} for seed in eval_seeds],
    )

    progress = {
        "status": "running",
        "train_target": len(train_seeds),
        "eval_target": len(eval_seeds),
        "train_completed": 0,
        "eval_completed": 0,
    }
    write_json(config.generation_progress_path, progress)

    train_examples = 0
    for index, seed in enumerate(tqdm(train_seeds, desc="Build train trajectories"), start=1):
        example = builder.build_example(seed)
        append_jsonl(config.generated_train_path, [example])
        train_examples += 1
        progress["train_completed"] = index
        if index == len(train_seeds) or index % 10 == 0:
            write_json(config.generation_progress_path, progress)

    eval_examples = 0
    for index, seed in enumerate(tqdm(eval_seeds, desc="Build eval trajectories"), start=1):
        example = builder.build_example(seed)
        append_jsonl(config.generated_eval_path, [example])
        eval_examples += 1
        progress["eval_completed"] = index
        if index == len(eval_seeds) or index % 10 == 0:
            write_json(config.generation_progress_path, progress)

    progress["status"] = "completed"
    write_json(config.generation_progress_path, progress)

    selected_seeds = train_seeds + eval_seeds
    disc_selected = len(selected_seeds)
    local_selected = len(local_train_seeds) + len(local_eval_seeds)

    summary = {
        "train_examples": train_examples,
        "eval_examples": eval_examples,
        "disc_seed_pool": len(disc_seed_pool),
        "local_seed_pool": len(local_seed_pool),
        "disc_selected": disc_selected - local_selected,
        "local_selected": local_selected,
        "disc_source_ratio": round((disc_selected - local_selected) / max(len(selected_seeds), 1), 4),
        "disc_pool_by_subset": _count_seeds_by_source(disc_seed_pool),
        "disc_selected_by_subset": _count_seeds_by_source([seed for seed in selected_seeds if seed.source.startswith("DISC")]),
        "selected_task_families": _count_seeds_by_task_family(selected_seeds),
        "selected_strategies": _count_seeds_by_strategy(selected_seeds),
        "selected_disc_task_families": _count_seeds_by_task_family(selected_seeds),
        "selected_local_scenarios": _count_seeds_by_task_family([seed for seed in selected_seeds if str(seed.metadata.get("scenario") or "")]),
        "excluded_task_families": ["leg_ele_extra"],
        "train_path": config.project_relative_path(config.generated_train_path),
        "eval_path": config.project_relative_path(config.generated_eval_path),
        "seed_manifest_path": config.project_relative_path(config.seed_manifest_path),
        "progress_path": config.project_relative_path(config.generation_progress_path),
    }
    write_json(config.generated_data_dir / "dataset_summary.json", summary)
    return summary
