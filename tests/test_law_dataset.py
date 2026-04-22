from pathlib import Path

import pandas as pd

from legal_agent.data.law_dataset import prepare_law_dataset


def test_prepare_law_dataset_archives_legacy_docs(tmp_path: Path):
    source_root = tmp_path / "law_files"
    target_root = tmp_path / "data" / "law_files"
    (source_root / "catalogs").mkdir(parents=True)
    (source_root / "files").mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "标题": "黑龙江省城市供热条例",
                "公布日期": "2024-01-01",
                "施行日期": "2024-03-01",
                "法律法规分类": "地方性法规",
            }
        ]
    ).to_csv(source_root / "catalogs" / "law_catalog_master.csv", index=False)

    (source_root / "files" / "黑龙江省城市供热条例.docx").write_text("docx content", encoding="utf-8")
    (source_root / "files" / "黑龙江省城市供热条例.doc").write_text("legacy content", encoding="utf-8")

    summary = prepare_law_dataset(source_root, target_root)

    assert summary["linked_documents"] == 1
    assert summary["archived_legacy_docs"] == 1
    assert summary["catalog_rows"] == 1
    assert (target_root / "files" / "黑龙江省城市供热条例.docx").exists()
    assert (target_root / "legacy_docs" / "黑龙江省城市供热条例.doc").exists()
    assert (target_root / "catalogs" / "law_catalog_master.csv").exists()
    assert (target_root / "catalogs" / "law_catalog_master.xlsx").exists()