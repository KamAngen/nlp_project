from pathlib import Path
import shutil
import types

from legal_agent.data import word_conversion


def test_save_as_docx_uses_target_directory_for_temp_files(tmp_path: Path, monkeypatch):
    source_path = tmp_path / "legacy.doc"
    target_path = tmp_path / "nested" / "converted.docx"
    source_path.write_text("legacy", encoding="utf-8")

    calls: dict[str, Path | None] = {"dir": None}

    class FakeTemporaryDirectory:
        def __init__(self, prefix: str, dir: Path | str | None = None):
            calls["dir"] = Path(dir) if dir is not None else None
            self.path = Path(dir) / f"{prefix}unit"

        def __enter__(self) -> str:
            self.path.mkdir(parents=True, exist_ok=True)
            return str(self.path)

        def __exit__(self, exc_type, exc, tb) -> None:
            shutil.rmtree(self.path, ignore_errors=True)

    class FakeDocument:
        def __init__(self, source: str):
            self.source = source

        def save(self, destination: str, save_format: str) -> None:
            assert self.source == str(source_path)
            assert save_format == "DOCX"
            Path(destination).write_text("converted", encoding="utf-8")

    fake_words = types.ModuleType("aspose.words")
    fake_words.Document = FakeDocument
    fake_words.SaveFormat = types.SimpleNamespace(DOCX="DOCX")
    fake_aspose = types.ModuleType("aspose")
    fake_aspose.words = fake_words

    monkeypatch.setattr(word_conversion.tempfile, "TemporaryDirectory", FakeTemporaryDirectory)
    monkeypatch.setitem(__import__("sys").modules, "aspose", fake_aspose)
    monkeypatch.setitem(__import__("sys").modules, "aspose.words", fake_words)

    word_conversion._save_as_docx(source_path, target_path)

    assert calls["dir"] == target_path.parent
    assert target_path.read_text(encoding="utf-8") == "converted"


def test_save_as_docx_removes_macros_before_saving(tmp_path: Path, monkeypatch):
    source_path = tmp_path / "macro.doc"
    target_path = tmp_path / "macro.docx"
    source_path.write_text("legacy", encoding="utf-8")

    class FakeTemporaryDirectory:
        def __init__(self, prefix: str, dir: Path | str | None = None):
            self.path = Path(dir) / f"{prefix}macro"

        def __enter__(self) -> str:
            self.path.mkdir(parents=True, exist_ok=True)
            return str(self.path)

        def __exit__(self, exc_type, exc, tb) -> None:
            shutil.rmtree(self.path, ignore_errors=True)

    state = {"removed": False}

    class FakeDocument:
        has_macros = True

        def __init__(self, source: str):
            self.source = source

        def remove_macros(self) -> None:
            state["removed"] = True

        def save(self, destination: str, save_format: str) -> None:
            assert self.source == str(source_path)
            assert state["removed"] is True
            assert save_format == "DOCX"
            Path(destination).write_text("converted", encoding="utf-8")

    fake_words = types.ModuleType("aspose.words")
    fake_words.Document = FakeDocument
    fake_words.SaveFormat = types.SimpleNamespace(DOCX="DOCX")
    fake_aspose = types.ModuleType("aspose")
    fake_aspose.words = fake_words

    monkeypatch.setattr(word_conversion.tempfile, "TemporaryDirectory", FakeTemporaryDirectory)
    monkeypatch.setitem(__import__("sys").modules, "aspose", fake_aspose)
    monkeypatch.setitem(__import__("sys").modules, "aspose.words", fake_words)

    word_conversion._save_as_docx(source_path, target_path)

    assert state["removed"] is True
    assert target_path.read_text(encoding="utf-8") == "converted"