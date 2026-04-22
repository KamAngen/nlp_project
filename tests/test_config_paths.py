import os
from pathlib import Path

from legal_agent.config import apply_configured_cuda_visible_devices, configured_cuda_visible_devices, load_app_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_config_uses_project_relative_paths():
    config = load_app_config(PROJECT_ROOT / "configs" / "smoke.yaml")

    assert config.project_relative_path(config.project_root) == "."
    assert config.project_relative_path(config.generated_data_dir) == "data/generated/smoke"
    assert config.project_relative_path(config.output_root) == "outputs/smoke"
    assert config.project_relative_path(config.models.agent_base) == "models/qwen/Qwen3_4B"


def test_apply_configured_cuda_visible_devices_uses_config_when_env_absent(monkeypatch):
    config = load_app_config(PROJECT_ROOT / "configs" / "smoke.yaml")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    visible = apply_configured_cuda_visible_devices(config)

    assert visible == configured_cuda_visible_devices(config)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == configured_cuda_visible_devices(config)


def test_apply_configured_cuda_visible_devices_preserves_existing_env(monkeypatch):
    config = load_app_config(PROJECT_ROOT / "configs" / "smoke.yaml")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "6")

    visible = apply_configured_cuda_visible_devices(config)

    assert visible == "6"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "6"
