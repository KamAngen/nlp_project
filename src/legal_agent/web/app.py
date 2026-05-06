from __future__ import annotations

import inspect
import socket

import gradio as gr

from legal_agent.config import load_app_config
from legal_agent.web.unified_workspace import build_unified_workspace


def _blocks_runtime_kwargs(*, theme: object, css: str) -> tuple[dict[str, object], dict[str, object]]:
    blocks_params = inspect.signature(gr.Blocks.__init__).parameters
    launch_params = inspect.signature(gr.Blocks.launch).parameters

    blocks_kwargs: dict[str, object] = {}
    launch_kwargs: dict[str, object] = {}

    if "theme" in blocks_params:
        blocks_kwargs["theme"] = theme
    elif "theme" in launch_params:
        launch_kwargs["theme"] = theme

    if "css" in blocks_params:
        blocks_kwargs["css"] = css
    elif "css" in launch_params:
        launch_kwargs["css"] = css

    return blocks_kwargs, launch_kwargs


def _resolve_launch_port(host: str, requested_port: int, *, fallback_window: int = 50) -> int:
    if requested_port < 1:
        raise ValueError("requested_port must be a positive integer")

    for candidate in range(requested_port, requested_port + fallback_window):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, candidate))
            except OSError:
                continue
        return candidate

    raise OSError(
        f"Cannot find empty port in range: {requested_port}-{requested_port + fallback_window - 1}."
    )


CSS = """
:root {
    --bg: #f3f4f6;
    --panel: #ffffff;
    --panel-muted: #f8fafc;
    --ink: #111827;
    --subtle: #6b7280;
    --border: #d0d7de;
    --accent: #2563eb;
    --accent-strong: #1d4ed8;
    --trace-height: 320px;
    --report-height: 280px;
    --input-height: 84px;
    --control-height: 50px;
    --workspace-gap: 14px;
    --left-panel-width: 41.5%;
    --right-panel-width: 58.5%;
}
.dark,
body.dark {
    --bg: #111827;
    --panel: #1f2937;
    --panel-muted: #111827;
    --ink: #f9fafb;
    --subtle: #cbd5e1;
    --border: #374151;
    --accent: #60a5fa;
    --accent-strong: #3b82f6;
}
html, body {
    height: 100%;
    margin: 0;
    background: var(--bg);
    color: var(--ink);
}
.gradio-container {
    max-width: none !important;
    width: 100% !important;
    margin: 0 auto;
    padding: 12px 16px 16px 16px !important;
    box-sizing: border-box;
    background: transparent !important;
    min-height: 100dvh !important;
    height: 100dvh !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
}
.gradio-container > .main.fillable,
.gradio-container > .main.fillable > .wrap,
.gradio-container > .main.fillable > .wrap > .contain,
.gradio-container > .main.fillable > .wrap > .contain > .column {
    min-height: 0 !important;
}
.gradio-container > .main.fillable,
.gradio-container > .main.fillable > .wrap,
.gradio-container > .main.fillable > .wrap > .contain,
.gradio-container > .main.fillable > .wrap > .contain > .column {
    display: flex !important;
    flex: 1 1 auto !important;
    flex-direction: column !important;
}
.gradio-container > .main.fillable > .wrap,
.gradio-container > .main.fillable > .wrap > .contain,
.gradio-container > .main.fillable > .wrap > .contain > .column {
    overflow: hidden !important;
}
.panel-card {
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--panel);
    box-shadow: none;
    padding: 12px;
    min-height: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 10px;
}
.panel-card > .prose {
    margin: 0 !important;
}
.panel-card > .prose h2 {
    margin: 0 0 2px 0 !important;
    font-size: 1.02rem !important;
}
#workspace-row {
    display: grid !important;
    grid-template-columns:
        minmax(0, calc(var(--left-panel-width) - (var(--workspace-gap) / 2)))
        minmax(0, calc(var(--right-panel-width) - (var(--workspace-gap) / 2)));
    flex: 1 1 auto !important;
    min-height: 0 !important;
    height: auto !important;
    max-height: 100% !important;
    width: 100% !important;
    max-width: 100% !important;
    gap: var(--workspace-gap);
    align-items: stretch !important;
    overflow: hidden !important;
}
#workspace-row > * {
    min-width: 0 !important;
}
#left-panel,
#right-panel {
    position: relative;
    display: flex !important;
    flex-direction: column !important;
    flex-wrap: nowrap !important;
    align-content: flex-start !important;
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    min-height: 0 !important;
    height: 100% !important;
    max-height: 100% !important;
    overflow-y: auto !important;
    overflow-x: auto !important;
    scrollbar-gutter: stable;
    scrollbar-width: thin;
    scrollbar-color: #b8c1cc transparent;
}
#left-panel > *,
#right-panel > * {
    flex: 0 0 auto !important;
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
}
#left-panel::-webkit-scrollbar,
#right-panel::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}
#left-panel::-webkit-scrollbar-thumb,
#right-panel::-webkit-scrollbar-thumb {
    background: #b8c1cc;
    border-radius: 999px;
    border: 2px solid transparent;
    background-clip: padding-box;
}
#left-panel::-webkit-scrollbar-track,
#right-panel::-webkit-scrollbar-track {
    background: transparent;
}
#left-panel .block,
#right-panel .block,
#left-panel .gr-form,
#right-panel .gr-form,
#left-panel .gr-group,
#right-panel .gr-group,
#left-panel .gr-column,
#right-panel .gr-column {
    position: relative;
    min-width: 0 !important;
}
#report-section,
#action-section {
    border: 1px solid var(--border);
    border-radius: 12px;
    background: var(--panel-muted);
    padding: 10px 12px;
    overflow: visible !important;
}
#config-device-row,
#user-session-row,
#new-entity-row,
#crud-button-row,
#action-button-row {
    flex-wrap: wrap !important;
    align-items: stretch !important;
    gap: 10px;
}
#config-device-row > *,
#user-session-row > *,
#new-entity-row > *,
#crud-button-row > *,
#action-button-row > * {
    min-width: 0 !important;
}
#crud-button-row button,
#action-button-row button {
    white-space: nowrap !important;
}
#trace-component textarea {
    min-height: var(--trace-height) !important;
    max-height: var(--trace-height) !important;
    resize: none !important;
    font-size: 0.94rem !important;
    line-height: 1.45 !important;
}
#history-component {
    flex: 1 1 auto;
    min-height: 320px !important;
    height: clamp(360px, 52vh, 640px) !important;
    overflow: auto !important;
    margin-top: 0 !important;
}
#input-stack {
    gap: 8px;
    flex: 0 0 auto;
}
#question-input {
    width: 100%;
}
#question-input textarea {
    height: var(--input-height) !important;
    min-height: var(--input-height) !important;
    max-height: var(--input-height) !important;
    resize: none !important;
}
#send-button,
#refresh-model-button {
    width: 100%;
}
#send-button button,
#refresh-model-button button {
    width: 100% !important;
    height: var(--control-height) !important;
    min-height: var(--control-height) !important;
    max-height: var(--control-height) !important;
    padding-left: 14px !important;
    padding-right: 14px !important;
}
#report-section {
    flex: 0 0 auto;
}
#report-component {
    min-height: 180px;
    max-height: var(--report-height);
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--panel);
    padding: 12px 14px;
}
.gradio-container [data-testid="dropdown"],
.gradio-container [data-testid="dropdown"] > div,
.ui-select input[role="listbox"],
.ui-select-filterable input[role="listbox"],
.ui-select [role="combobox"],
.ui-select-filterable [role="combobox"],
.gradio-container [role="option"] {
    color: var(--ink) !important;
}
.ui-select,
.ui-select-filterable,
.ui-select .container,
.ui-select-filterable .container,
.ui-select .wrap,
.ui-select-filterable .wrap,
.ui-select .wrap-inner,
.ui-select-filterable .wrap-inner,
.ui-select .secondary-wrap,
.ui-select-filterable .secondary-wrap,
.ui-select [data-testid="dropdown"],
.ui-select-filterable [data-testid="dropdown"] {
    position: relative;
    overflow: visible !important;
    min-width: 0 !important;
}
.ui-select [data-testid="dropdown"] button,
.ui-select input[role="listbox"],
.ui-select-filterable input[role="listbox"],
.ui-select [role="combobox"],
.ui-select-filterable [role="combobox"] {
    min-height: 52px !important;
    border-radius: 10px !important;
    padding: 0 42px 0 16px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    font-size: 0.98rem !important;
    line-height: 1.2 !important;
    background: var(--panel) !important;
    box-shadow: none !important;
    color: var(--ink) !important;
    border-color: var(--border) !important;
    text-align: center !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
.ui-select [data-testid="dropdown"] button,
.ui-select input[role="listbox"],
.ui-select-filterable input[role="listbox"],
.ui-select [role="combobox"],
.ui-select-filterable [role="combobox"],
.gradio-container input:not([type="checkbox"]):not([type="radio"]),
.gradio-container textarea {
    color: var(--ink) !important;
    border-color: var(--border) !important;
}
.ui-select .icon-wrap,
.ui-select-filterable .icon-wrap,
.ui-select [data-testid="dropdown"] .dropdown-arrow,
.ui-select [data-testid="dropdown"] button svg,
.ui-select [role="combobox"] svg,
.ui-select-filterable [role="combobox"] svg {
    position: absolute !important;
    right: 14px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    pointer-events: none !important;
    opacity: 0.88;
}
.ui-select input[role="listbox"],
.ui-select-filterable input[role="listbox"],
.ui-select [data-testid="dropdown"] button span,
.ui-select-filterable [role="combobox"] input,
.ui-select [role="combobox"] input {
    width: 100% !important;
    text-align: center !important;
}
#model-select,
#runtime-device-select,
#user-select,
#session-select,
#exam-type-select,
#exam-topic-select,
#report-type-select {
    position: relative !important;
    min-width: 0 !important;
}
.ui-select:focus-within,
.ui-select-filterable:focus-within,
#model-select:focus-within,
#runtime-device-select:focus-within,
#user-select:focus-within,
#session-select:focus-within,
#exam-type-select:focus-within,
#exam-topic-select:focus-within,
#report-type-select:focus-within {
    z-index: 2147482000 !important;
}
#headlessui-portal-root,
[data-headlessui-portal],
.gradio-container [data-testid="dropdown-options"],
.gradio-container .dropdown-options,
.gradio-container .options,
.gradio-container ul[role="listbox"] {
    position: fixed !important;
    z-index: 2147483647 !important;
}
.gradio-container [data-testid="dropdown-options"],
.gradio-container .dropdown-options,
.gradio-container .options,
.gradio-container ul[role="listbox"] {
    border: 1px solid var(--border) !important;
    background: var(--panel) !important;
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.18) !important;
    text-align: center !important;
    max-height: min(320px, 42vh) !important;
    overflow-y: auto !important;
}
.gradio-container .options .item,
.gradio-container [role="option"] {
    background: transparent !important;
    justify-content: center !important;
    text-align: center !important;
}
.gradio-container .options .item.selected,
.gradio-container .options .item:hover,
.gradio-container [role="option"][aria-selected="true"],
.gradio-container [role="option"]:hover {
    background: rgba(37, 99, 235, 0.10) !important;
}
.dark .ui-select [data-testid="dropdown"] button,
.dark .ui-select input[role="listbox"],
.dark .ui-select-filterable input[role="listbox"],
.dark .ui-select [role="combobox"],
.dark .ui-select-filterable [role="combobox"] {
    background: var(--panel) !important;
    color: var(--ink) !important;
    border-color: var(--border) !important;
    box-shadow: none !important;
}
.dark .gradio-container input:not([type="checkbox"]):not([type="radio"]),
.dark .gradio-container textarea {
    background: var(--panel) !important;
    color: var(--ink) !important;
    border-color: var(--border) !important;
}
.dark .gradio-container .options,
.dark .gradio-container ul[role="listbox"],
.dark .gradio-container [data-testid="dropdown-options"],
.dark .gradio-container .dropdown-options {
    background: var(--panel) !important;
}
.dark .gradio-container .options .item.selected,
.dark .gradio-container .options .item:hover,
.dark .gradio-container [role="option"][aria-selected="true"],
.dark .gradio-container [role="option"]:hover {
    background: rgba(96, 165, 250, 0.14) !important;
}
.gr-button-primary {
    background: var(--accent) !important;
    border: 1px solid var(--accent-strong) !important;
    color: #ffffff !important;
    box-shadow: none !important;
}
.gr-button-secondary {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    color: var(--ink) !important;
    box-shadow: none !important;
}
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container label,
.gradio-container p,
.gradio-container li {
    color: var(--ink) !important;
}
.gradio-container .prose {
    color: var(--ink) !important;
}
@media (max-width: 560px) {
    :root {
        --trace-height: 240px;
        --report-height: 220px;
    }
    .gradio-container {
        min-height: 0 !important;
        height: auto !important;
        overflow: visible !important;
    }
    #workspace-row {
        flex: 0 0 auto !important;
        height: auto !important;
        min-height: 0 !important;
        grid-template-columns: minmax(0, 1fr) !important;
        overflow: visible !important;
    }
    #left-panel,
    #right-panel {
        height: auto !important;
        max-height: none !important;
    }
    #history-component {
        min-height: 380px !important;
        height: auto !important;
    }
}
"""


def launch_web_ui(
    *,
    config_path: str = "configs/defaults.yaml",
    study_config_path: str = "configs/study_agent.yaml",
    host: str = "127.0.0.1",
    port: int = 7860,
    retrieval_device: str = "cpu",
    public: bool | None = None,
) -> None:
    app_config = load_app_config(config_path)
    share = app_config.public if public is None else bool(public)
    theme = gr.themes.Soft(primary_hue="amber", neutral_hue="stone")
    blocks_kwargs, launch_kwargs = _blocks_runtime_kwargs(theme=theme, css=CSS)
    with gr.Blocks(title="统一法律学习 Agent", **blocks_kwargs) as demo:
        gr.Markdown(
            "# 统一法律学习 Agent\n"
            "单一模型、单一会话界面，集成法规问答、法考学习、模拟测试、评分反馈、学习报告与持久化记忆。"
        )
        build_unified_workspace(
            config_path=config_path,
            study_config_path=study_config_path,
            retrieval_device=retrieval_device,
        )

    demo.queue(default_concurrency_limit=4)
    launch_port = _resolve_launch_port(host, port)
    if launch_port != port:
        print(f"[web-ui] 端口 {port} 已被占用，自动切换到 {launch_port}。")
    demo.launch(server_name=host, server_port=launch_port, share=share, **launch_kwargs)
