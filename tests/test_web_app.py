from types import SimpleNamespace

from legal_agent.web import app as web_app


def test_launch_web_ui_uses_unified_workspace(monkeypatch):
    called: dict[str, object] = {}

    monkeypatch.setattr(web_app, "build_unified_workspace", lambda **kwargs: called.update(kwargs))
    monkeypatch.setattr(web_app.gr.Blocks, "queue", lambda self, **kwargs: self)
    monkeypatch.setattr(web_app, "_resolve_launch_port", lambda host, port, fallback_window=50: port)
    monkeypatch.setattr(web_app.gr.Blocks, "launch", lambda self, **kwargs: called.update({"launch": kwargs}))
    monkeypatch.setattr(web_app, "load_app_config", lambda path: SimpleNamespace(public=True))

    web_app.launch_web_ui(
        config_path="configs/defaults.yaml",
        study_config_path="configs/study_agent.yaml",
        host="127.0.0.1",
        port=7860,
        retrieval_device="cpu",
    )

    assert called["config_path"] == "configs/defaults.yaml"
    assert called["study_config_path"] == "configs/study_agent.yaml"
    assert called["retrieval_device"] == "cpu"
    assert called["launch"]["server_name"] == "127.0.0.1"
    assert called["launch"]["server_port"] == 7860
    assert called["launch"]["share"] is True


def test_launch_web_ui_falls_back_when_default_port_is_busy(monkeypatch, capsys):
    called: dict[str, object] = {}

    monkeypatch.setattr(web_app, "build_unified_workspace", lambda **kwargs: called.update(kwargs))
    monkeypatch.setattr(web_app.gr.Blocks, "queue", lambda self, **kwargs: self)
    monkeypatch.setattr(web_app, "_resolve_launch_port", lambda host, port, fallback_window=50: 7861)
    monkeypatch.setattr(web_app.gr.Blocks, "launch", lambda self, **kwargs: called.update({"launch": kwargs}))
    monkeypatch.setattr(web_app, "load_app_config", lambda path: SimpleNamespace(public=False))

    web_app.launch_web_ui(
        config_path="configs/defaults.yaml",
        study_config_path="configs/study_agent.yaml",
        host="127.0.0.1",
        port=7860,
        retrieval_device="cpu",
    )

    assert called["launch"]["server_port"] == 7861
    assert called["launch"]["share"] is False
    output = capsys.readouterr().out
    assert "7860" in output
    assert "7861" in output


def test_css_handles_zoom_layout_and_dropdown_overlay():
    css = web_app.CSS

    assert "--left-panel-width: 41.5%;" in css
    assert "--right-panel-width: 58.5%;" in css
    assert ".gradio-container > .main.fillable > .wrap > .contain > .column" in css
    assert "grid-template-columns:" in css
    assert "overflow: hidden !important;" in css
    assert "overflow-y: auto !important;" in css
    assert "flex-wrap: nowrap !important;" in css
    assert "overflow-x: auto !important;" in css
    assert ".ui-select:focus-within," in css
    assert ".gradio-container .options," in css
    assert "position: fixed !important;" in css
    assert "@media (max-width: 560px)" in css
