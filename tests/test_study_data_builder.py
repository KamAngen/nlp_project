from __future__ import annotations

import csv
from pathlib import Path

from legal_agent.config import AppConfig, GenerationConfig, InferenceConfig, ModelsConfig, RetrievalConfig, TrainingConfig
from legal_agent.data.study_knowledge import prepare_study_knowledge_assets
from legal_agent.study_config import StudyAgentConfig
from legal_agent.utils.io import read_json, read_jsonl, write_json, write_jsonl
from rag_engine.loaders import load_question_bank


def _app_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        project_root=tmp_path,
        law_dir=tmp_path / "data" / "law_files",
        law_doc_dir=tmp_path / "data" / "law_files" / "files",
        law_catalog_glob="catalogs/law_catalog_master.csv",
        artifact_root=tmp_path / "artifacts",
        disc_law_dir=tmp_path / "data" / "disc_law",
        generated_data_dir=tmp_path / "data" / "generated",
        output_root=tmp_path / "outputs",
        available_gpu_ids=[0],
        models=ModelsConfig(
            agent_base=tmp_path / "models" / "qwen" / "Qwen3_4B",
            embedding_model=tmp_path / "models" / "embeddings" / "bge-small-zh",
        ),
        retrieval=RetrievalConfig(),
        generation=GenerationConfig(),
        training=TrainingConfig(output_dir=tmp_path / "ckpt", adapter_name="test-adapter"),
        inference=InferenceConfig(),
    )


def _study_config(tmp_path: Path) -> StudyAgentConfig:
    return StudyAgentConfig(
        project_root=tmp_path,
        memory_root=tmp_path / "memory",
        report_root=tmp_path / "reports",
        question_bank_path=tmp_path / "data" / "legal_study_agent" / "question_bank.jsonl",
        case_bank_path=tmp_path / "data" / "legal_study_agent" / "case_bank.jsonl",
        common_knowledge_path=tmp_path / "data" / "legal_study_agent" / "common_knowledge.jsonl",
        system_memory_path=tmp_path / "data" / "legal_study_agent" / "system_seed_memories.json",
        study_manifest_path=tmp_path / "artifacts" / "study_knowledge_manifest.json",
        use_legacy_statute_rag=False,
        legacy_config_path=None,
    )


def _seed_disc_law(tmp_path: Path) -> None:
    disc_path = tmp_path / "data" / "disc_law" / "disc_law_normalized.jsonl"
    rows = [
        {
            "sample_id": "civil-001",
            "task_family": "legal_question_answering",
            "subset_name": "DISC-Law-SFT-Pair-QA-released",
            "instruction": "租赁期满后房东无故不退押金，该如何处理？",
            "answer": "根据《中华人民共和国民法典》，若不存在租金、违约金或损失抵扣项目，出租人应返还押金。",
            "references": ["《中华人民共和国民法典》"],
        },
        {
            "sample_id": "criminal-001",
            "task_family": "exam",
            "subset_name": "DISC-Law-SFT-Pair",
            "instruction": "诈骗罪量刑标准通常如何判断？",
            "answer": "应先判断是否符合《中华人民共和国刑法》关于诈骗罪的构成，再结合数额和情节判断量刑档次。",
            "references": ["《中华人民共和国刑法》"],
        },
        {
            "sample_id": "admin-001",
            "task_family": "legal_question_answering",
            "subset_name": "DISC-Law-SFT-Pair",
            "instruction": "行政机关作出较大数额罚款前是否应保障听证权？",
            "answer": "根据《中华人民共和国行政处罚法》，符合法定情形时应告知并保障当事人申请听证的权利。",
            "references": ["《中华人民共和国行政处罚法》"],
        },
        {
            "sample_id": "civilproc-001",
            "task_family": "jud_read_compre",
            "subset_name": "DISC-Law-SFT-Pair",
            "instruction": "民事诉讼中证据保全申请通常应审查哪些要点？",
            "answer": "应结合《中华人民共和国民事诉讼法》审查证据可能灭失、以后难以取得以及申请必要性。",
            "references": ["《中华人民共和国民事诉讼法》"],
        },
        {
            "sample_id": "criminalproc-001",
            "task_family": "jud_read_compre",
            "subset_name": "DISC-Law-SFT-Pair",
            "instruction": "刑事诉讼中犯罪嫌疑人的辩护权何时得到保障？",
            "answer": "依据《中华人民共和国刑事诉讼法》，犯罪嫌疑人自被侦查机关第一次讯问或者采取强制措施之日起有权获得辩护。",
            "references": ["《中华人民共和国刑事诉讼法》"],
        },
        {
            "sample_id": "business-001",
            "task_family": "legal_question_answering",
            "subset_name": "DISC-Law-SFT-Pair",
            "instruction": "公司法中股东会决议效力争议应先看什么？",
            "answer": "通常先依据《中华人民共和国公司法》审查召集程序、表决方式和决议内容是否合法。",
            "references": ["《中华人民共和国公司法》"],
        },
        {
            "sample_id": "theory-001",
            "task_family": "exam",
            "subset_name": "DISC-Law-SFT-Pair",
            "instruction": "法律效力层级发生冲突时应优先适用什么规则？",
            "answer": "应坚持上位法优于下位法，同时结合特别法优于一般法等规则判断规范适用顺序。",
            "references": ["《中华人民共和国立法法》"],
        },
        {
            "sample_id": "admin-noisy-001",
            "task_family": "exam",
            "subset_name": "DISC-Law-SFT-Pair",
            "instruction": "这是一道单选题。 行政机关作出较大数额罚款决定前，是否应依法保障当事人听证权？请给出答案和解析。",
            "answer": "A. 应当依法告知并保障听证权 解析：根据《中华人民共和国行政处罚法》，符合法定情形时行政机关应告知并保障听证申请权。",
            "references": ["《中华人民共和国行政处罚法》"],
        },
        {
            "sample_id": "criminal-multi-001",
            "task_family": "exam",
            "subset_name": "DISC-Law-SFT-Pair",
            "instruction": "请给出这道多选题的答案和推理过程。 刑法中下列哪些属于法定主刑？ A. 管制 B. 拘役 C. 没收财产 D. 有期徒刑",
            "answer": "A. 管制 B. 拘役 D. 有期徒刑 解析：根据《中华人民共和国刑法》，主刑包括管制、拘役、有期徒刑、无期徒刑和死刑。",
            "references": ["《中华人民共和国刑法》"],
        },
    ]
    write_jsonl(disc_path, rows)


def _seed_explicit_choice_disc_law(tmp_path: Path) -> None:
    disc_path = tmp_path / "data" / "disc_law" / "disc_law_normalized.jsonl"
    rows = [
        {
            "sample_id": "admin-choice-001",
            "task_family": "exam",
            "subset_name": "DISC-Law-SFT-Pair",
            "instruction": "关于行政处罚听证，下列哪一选项正确？ A. 行政机关当然无需告知听证权 B. 只有法院才能组织听证 C. 符合法定情形时应告知当事人申请听证的权利 D. 听证结束后必须撤销处罚决定",
            "answer": "C. 符合法定情形时应告知当事人申请听证的权利 解析：根据《中华人民共和国行政处罚法》，较大数额罚款等法定情形下，应告知当事人申请听证的权利。",
            "references": ["《中华人民共和国行政处罚法》"],
        },
        {
            "sample_id": "civil-case-001",
            "task_family": "jud_read_compre",
            "subset_name": "DISC-Law-SFT-Pair",
            "instruction": "租赁期满后房东无故不退押金，承租人通常可以主张什么？",
            "answer": "若不存在租金、违约金或损失抵扣项目，出租人应返还押金。",
            "references": ["《中华人民共和国民法典》"],
        },
    ]
    write_jsonl(disc_path, rows)


def _seed_catalog(tmp_path: Path) -> None:
    catalog_path = tmp_path / "data" / "law_files" / "catalogs" / "law_catalog_master.csv"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with catalog_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["标题", "公布日期", "施行日期", "法律法规分类", "source_catalog"])
        writer.writeheader()
        writer.writerow(
            {
                "标题": "中华人民共和国行政处罚法",
                "公布日期": "2021-01-22",
                "施行日期": "2021-07-15",
                "法律法规分类": "法律",
                "source_catalog": "sample.xlsx",
            }
        )
        writer.writerow(
            {
                "标题": "某市城市管理条例",
                "公布日期": "2024-01-01",
                "施行日期": "2024-03-01",
                "法律法规分类": "地方性法规",
                "source_catalog": "sample.xlsx",
            }
        )


def _seed_existing_manual_rows(study_config: StudyAgentConfig) -> None:
    study_config.question_bank_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        study_config.question_bank_path,
        [
            {
                "question_id": "q-m-001",
                "topic": "民法",
                "difficulty": "medium",
                "question": "示例手工题目",
                "options": {"A": "错", "B": "对", "C": "错", "D": "错"},
                "answer": "B",
                "analysis": "示例解析。",
                "tags": ["民法"],
                "score": 20,
            }
        ],
    )
    write_jsonl(
        study_config.case_bank_path,
        [
            {
                "case_id": "case-001",
                "title": "示例案例",
                "facts": "示例事实",
                "issue": "示例争点",
                "holding": "示例结论",
                "statutes": ["中华人民共和国民法典"],
                "tags": ["民法"],
            }
        ],
    )
    write_jsonl(
        study_config.common_knowledge_path,
        [
            {
                "entry_id": "common-001",
                "title": "示例常识",
                "content": "示例内容",
                "tags": ["学习方法"],
                "source": "manual",
            }
        ],
    )
    write_json(
        study_config.system_memory_path,
        [
            {
                "id": "system-style-01",
                "category": "response_policy",
                "text": "示例系统记忆。",
                "importance": 0.9,
                "tags": ["reply_style"],
            }
        ],
    )


def test_prepare_study_knowledge_assets_generates_and_preserves_manual_rows(tmp_path: Path):
    app_config = _app_config(tmp_path)
    study_config = _study_config(tmp_path)
    _seed_disc_law(tmp_path)
    _seed_catalog(tmp_path)
    _seed_existing_manual_rows(study_config)

    summary = prepare_study_knowledge_assets(
        app_config,
        study_config,
        question_count=6,
        case_count=4,
        common_count=8,
    )

    question_rows = list(read_jsonl(study_config.question_bank_path))
    case_rows = list(read_jsonl(study_config.case_bank_path))
    common_rows = list(read_jsonl(study_config.common_knowledge_path))
    system_rows = list(read_json(study_config.system_memory_path))
    loaded_questions = load_question_bank(study_config.question_bank_path)

    assert summary["generated"] is True
    assert len(question_rows) >= 6
    assert len(case_rows) >= 4
    assert len(common_rows) >= 8
    assert len(system_rows) >= 6
    assert any(row["question_id"] == "q-m-001" for row in question_rows)
    assert any(row["case_id"] == "case-001" for row in case_rows)
    assert any(row["entry_id"] == "common-001" for row in common_rows)
    assert any(row["id"] == "system-style-01" for row in system_rows)
    assert any(str(row.get("question_id") or "").startswith("auto-q-") for row in question_rows)
    assert any(str(row.get("case_id") or "").startswith("auto-case-") for row in case_rows)
    assert any(row.get("source") == "auto_builder" for row in common_rows)
    auto_question_rows = [row for row in question_rows if str(row.get("question_id") or "").startswith("auto-q-")]
    assert auto_question_rows
    for row in auto_question_rows:
        rendered = row["question"] + "\n" + "\n".join((row.get("options") or {}).values()) + "\n" + row["analysis"]
        assert "解析：" not in rendered
        assert "答案：" not in rendered
        assert "请给出这道多选题" not in rendered
        assert "…" not in rendered
    auto_case_rows = [row for row in case_rows if str(row.get("case_id") or "").startswith("auto-case-")]
    assert auto_case_rows
    for row in auto_case_rows:
        rendered = row["title"] + "\n" + row["facts"] + "\n" + row["issue"] + "\n" + row["holding"]
        assert "…" not in rendered
    assert all("七台河" not in row["title"] + row["content"] for row in common_rows)
    assert all(not str(row["entry_id"]).startswith("auto-common-catalog-") for row in common_rows)
    first_auto_question = auto_question_rows[0]
    loaded_auto_question = next(record for record in loaded_questions if record.record_id == first_auto_question["question_id"])
    assert loaded_auto_question.title == first_auto_question["question"]


def test_prepare_study_knowledge_assets_keeps_explicit_single_choice_options(tmp_path: Path):
    app_config = _app_config(tmp_path)
    study_config = _study_config(tmp_path)
    _seed_explicit_choice_disc_law(tmp_path)
    _seed_catalog(tmp_path)
    _seed_existing_manual_rows(study_config)

    prepare_study_knowledge_assets(
        app_config,
        study_config,
        question_count=3,
        case_count=1,
        common_count=4,
    )

    question_rows = list(read_jsonl(study_config.question_bank_path))
    explicit_row = next(row for row in question_rows if row.get("question") == "关于行政处罚听证，下列哪一选项正确？")

    assert explicit_row["question"] == "关于行政处罚听证，下列哪一选项正确？"
    assert explicit_row["options"] == {
        "A": "行政机关当然无需告知听证权",
        "B": "只有法院才能组织听证",
        "C": "符合法定情形时应告知当事人申请听证的权利",
        "D": "听证结束后必须撤销处罚决定",
    }
    assert explicit_row["answer"] == "C"
    rendered = explicit_row["question"] + "\n" + "\n".join(explicit_row["options"].values()) + "\n" + explicit_row["analysis"]
    assert "…" not in rendered
    assert "解析：" not in rendered


def test_prepare_study_knowledge_assets_skips_when_existing_counts_are_sufficient(tmp_path: Path):
    app_config = _app_config(tmp_path)
    study_config = _study_config(tmp_path)
    _seed_disc_law(tmp_path)
    _seed_catalog(tmp_path)
    _seed_existing_manual_rows(study_config)

    prepare_study_knowledge_assets(
        app_config,
        study_config,
        question_count=6,
        case_count=4,
        common_count=8,
    )
    summary = prepare_study_knowledge_assets(
        app_config,
        study_config,
        question_count=6,
        case_count=4,
        common_count=8,
    )

    assert summary["generated"] is False
    assert summary["question_count"] >= 6
    assert summary["case_count"] >= 4
    assert summary["common_count"] >= 8
    assert summary["system_count"] >= 6