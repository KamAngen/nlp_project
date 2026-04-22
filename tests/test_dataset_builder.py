from legal_agent.training.dataset_builder import _build_disc_seed, _build_disc_seed_buckets, _count_seeds_by_source, _draw_balanced_seeds


def _record(subset_name: str, index: int, *, with_reference: bool) -> dict:
    return {
        "sample_id": f"sample_{index}",
        "task_family": "legal_question_answering",
        "subset_name": subset_name,
        "instruction": f"问题：这是第 {index} 条测试问题，应该如何处理？",
        "answer": f"这是第 {index} 条测试答案。",
        "references": ["《中华人民共和国民法典》第一条"] if with_reference else [],
        "has_references": with_reference,
    }


def test_build_disc_seed_buckets_uses_all_four_raw_subsets():
    records = [
        _record("DISC-Law-SFT-Pair-QA-released", 1, with_reference=False),
        _record("DISC-Law-SFT-Pair", 2, with_reference=False),
        _record("DISC-Law-SFT-Triplet-QA-released", 3, with_reference=True),
        _record("DISC-Law-SFT-Triplet-released", 4, with_reference=True),
    ]

    buckets = _build_disc_seed_buckets(records)

    assert set(buckets) == {
        "DISC-Law-SFT-Pair-QA-released::legal_question_answering",
        "DISC-Law-SFT-Pair::legal_question_answering",
        "DISC-Law-SFT-Triplet-QA-released::legal_question_answering",
        "DISC-Law-SFT-Triplet-released::legal_question_answering",
    }
    assert buckets["DISC-Law-SFT-Pair::legal_question_answering"][0].expected_tools == ["retrieve_from_kb"]
    assert buckets["DISC-Law-SFT-Triplet-released::legal_question_answering"][0].expected_tools == ["lookup_statute", "retrieve_from_kb"]


def test_draw_balanced_seeds_is_uniform_across_subsets():
    records = []
    subset_specs = [
        ("DISC-Law-SFT-Pair-QA-released", False),
        ("DISC-Law-SFT-Pair", False),
        ("DISC-Law-SFT-Triplet-QA-released", True),
        ("DISC-Law-SFT-Triplet-released", True),
    ]
    for subset_name, with_reference in subset_specs:
        for index in range(6):
            records.append(_record(subset_name, index, with_reference=with_reference))

    buckets = _build_disc_seed_buckets(records)
    selected, _ = _draw_balanced_seeds(buckets, 10)
    counts = _count_seeds_by_source(selected)

    assert sum(counts.values()) == 10
    assert max(counts.values()) - min(counts.values()) <= 1
    assert len(counts) == 4


def test_direct_answer_task_family_has_no_expected_tools():
    record = {
        "sample_id": "jud_doc_sum-1",
        "task_family": "jud_doc_sum",
        "subset_name": "DISC-Law-SFT-Pair",
        "instruction": "请大致描述这篇文书的内容。\n\n某民事判决书……",
        "answer": "总结：这是一起民事纠纷案件。",
        "references": [],
        "has_references": False,
    }

    seed = _build_disc_seed(record)

    assert seed is not None
    assert seed.expected_tools == []
    assert seed.metadata["requires_retrieval"] is False