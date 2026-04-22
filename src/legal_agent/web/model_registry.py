from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from legal_agent.config import AppConfig


@dataclass(slots=True)
class ModelChoice:
    label: str
    kind: str
    model_path: Path
    adapter_path: Path | None
    description: str


def _read_model_metadata(path: Path) -> dict[str, object]:
    config_path = path / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _is_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").exists() or not (
        (path / "tokenizer_config.json").exists() or (path / "tokenizer.json").exists()
    ):
        return False
    metadata = _read_model_metadata(path)
    architectures = metadata.get("architectures") or []
    if architectures and not any(str(item).endswith("ForCausalLM") for item in architectures):
        return False
    return True


def _choice_label(relative_path: str, *, kind: str) -> str:
    if kind == "base":
        return relative_path
    return f"{relative_path} [adapter]"


def discover_base_models(config: AppConfig) -> list[ModelChoice]:
    qwen_root = config.project_root / "models" / "qwen"
    discovered: dict[str, Path] = {}
    if qwen_root.exists():
        for config_file in sorted(qwen_root.rglob("config.json")):
            model_dir = config_file.parent
            if not _is_model_dir(model_dir) or (model_dir / "adapter_config.json").exists():
                continue
            relative_path = config.project_relative_path(model_dir)
            if relative_path is None:
                continue
            discovered.setdefault(relative_path, model_dir.resolve())

    choices: list[ModelChoice] = []
    for relative_path, model_path in sorted(discovered.items()):
        description = f"当前项目内的基础 Qwen 模型目录：{relative_path}。"
        choices.append(
            ModelChoice(
                label=_choice_label(relative_path, kind="base"),
                kind="base",
                model_path=model_path,
                adapter_path=None,
                description=description,
            )
        )
    return choices


def discover_adapter_models(config: AppConfig) -> list[ModelChoice]:
    search_roots: list[Path] = []
    for candidate in (config.output_root, config.training.output_dir, config.training.output_dir.parent, config.project_root / "ckpt"):
        resolved = candidate.resolve()
        if resolved.exists() and resolved not in search_roots:
            search_roots.append(resolved)
    if not search_roots:
        return []

    choices: list[ModelChoice] = []
    seen_adapters: set[Path] = set()
    for search_root in search_roots:
        for adapter_config_path in sorted(search_root.rglob("adapter_config.json")):
            adapter_dir = adapter_config_path.parent.resolve()
            if adapter_dir in seen_adapters:
                continue
            seen_adapters.add(adapter_dir)
            relative_path = config.project_relative_path(adapter_dir)
            assert relative_path is not None

            base_model_path = config.models.agent_base
            try:
                payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
                raw_base_model = payload.get("base_model_name_or_path")
                if raw_base_model:
                    candidate = Path(raw_base_model)
                    if not candidate.is_absolute():
                        candidate = config.resolve_project_path(candidate)
                    if candidate.exists():
                        base_model_path = candidate.resolve()
            except Exception:
                pass

            description = f"LoRA 后训练适配器目录：{relative_path}。默认挂载到 {config.project_relative_path(base_model_path)}。"
            choices.append(
                ModelChoice(
                    label=_choice_label(relative_path, kind="post_train"),
                    kind="post_train",
                    model_path=base_model_path,
                    adapter_path=adapter_dir,
                    description=description,
                )
            )
    return choices


def build_choice_map(config: AppConfig) -> dict[str, ModelChoice]:
    choices = discover_base_models(config) + discover_adapter_models(config)
    return {choice.label: choice for choice in choices}
