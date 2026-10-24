import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from utils import gpt_workflow_settings as settings


def test_default_is_dedicated_direct_output():
    state = settings.load_settings({})
    assert state["dedicated_key_enabled"] and not state["same_as_gemini"]
    assert not state["postprocess_enabled"]
    assert state["timeout_seconds"] == 600


def test_timeout_budget_matches_http_retries_without_endpoint_fallback(monkeypatch):
    monkeypatch.setenv("IMAGE_MAKER_IMAGE_MAX_RETRIES", "0")
    cfg = {"timeout": 600, "upstream_error_retries": -1}
    assert settings.generation_timeout_budget(cfg, mode="edit", has_images=True) == 630
    assert settings.generation_timeout_budget(cfg, mode="generate", has_images=True) == 630
    cfg["upstream_error_retries"] = 2
    assert settings.generation_timeout_budget(cfg, mode="edit", has_images=False) == 1830


def test_timeout_setting_is_independent_even_with_shared_credentials(monkeypatch):
    from modules.others import api_backend
    config = {settings.NODE: {"same_as_gemini": True, "timeout_seconds": 600},
        "apis": {"aigc2d": {"api_key": "gemini", "timeout": 60}, "aigc-2d-gpt": {"model": "gpt-image-2"}}}
    monkeypatch.setattr(api_backend, "load_config", lambda _: config)
    assert api_backend.get_api_config(api_type="aigc-2d-gpt")["timeout"] == 600
    assert api_backend.get_api_config(api_type="aigc2d")["timeout"] == 60


def test_settings_merge_without_secrets(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"other": {"keep": 1}}))
    settings.save_settings({"same_as_gemini": True, "base_url": "https://example.test/v1"}, path)
    data = json.loads(path.read_text())
    assert data["other"] == {"keep": 1}
    assert settings.load_settings(data)["same_as_gemini"]
    assert "api_key" not in path.read_text()


def test_secret_merge_preserves_other_env(tmp_path, monkeypatch):
    monkeypatch.delenv(settings.KEY_ENV, raising=False)
    path = tmp_path / ".env"
    path.write_text("# keep\nOTHER_KEY=unchanged\n" + settings.KEY_ENV + "=old\n")
    settings.save_dedicated_key("test-dedicated-secret", path)
    assert "OTHER_KEY=unchanged" in path.read_text()
    assert path.read_text().count(settings.KEY_ENV + "=") == 1
    assert settings.dedicated_key() == "test-dedicated-secret"


def test_runtime_keys_separate_and_shared_is_dynamic(monkeypatch):
    from modules.others import api_backend as backend
    monkeypatch.setenv(settings.KEY_ENV, "dedicated")
    config = {settings.NODE: {}, "apis": {"aigc2d": {"api_key": "gemini", "base_url": "https://example.test/v1beta/models"},
                        "aigc-2d-gpt": {"api_key": "old", "model": "gpt-image-2"}}}
    monkeypatch.setattr(backend, "load_config", lambda _: config)
    monkeypatch.setattr(backend, "resolve_api_key", lambda cfg, node: cfg.get("api_key", ""))
    monkeypatch.setattr(backend, "api_key_source", lambda node, cfg: "config")
    assert backend.get_api_config(api_type="aigc-2d-gpt")["api_key"] == "dedicated"
    assert backend.get_api_config(api_type="aigc2d")["api_key"] == "gemini"
    config[settings.NODE] = {"same_as_gemini": True}
    cfg = backend.get_api_config(api_type="aigc-2d-gpt")
    assert cfg["api_key"] == "gemini" and cfg["base_url"] == "https://example.test/v1"
    assert cfg["model"] == "gpt-image-2"
    config["apis"]["aigc2d"]["api_key"] = "new-gemini"
    assert backend.get_api_config(api_type="aigc-2d-gpt")["api_key"] == "new-gemini"


def test_missing_dedicated_does_not_fallback(monkeypatch):
    monkeypatch.setattr(settings, "dedicated_key", lambda: "")
    assert settings.apply_dedicated_key({"api_key": "old"}, settings.load_settings({}))["api_key"] == ""


def test_legacy_config_keeps_existing_key_until_workflow_is_configured(monkeypatch):
    from modules.others import api_backend
    monkeypatch.setattr(api_backend, "load_config", lambda _: {"apis": {"aigc-2d-gpt": {"api_key": "legacy"}}})
    monkeypatch.setattr(api_backend, "resolve_api_key", lambda cfg, node: cfg.get("api_key", ""))
    assert api_backend.get_api_config(api_type="aigc-2d-gpt")["api_key"] == "legacy"


def test_gpt_widget_defaults_to_no_postprocessing(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from modules.image_generation import gpt_image2_tab as tab
    app = QApplication.instance() or QApplication([])
    config = tmp_path / "config.json"
    config.write_text(json.dumps({settings.NODE: {"postprocess_enabled": False}}))
    monkeypatch.setattr(settings, "CONFIG_PATH", config)
    monkeypatch.setattr(tab, "get_api_config", lambda **k: {"api_key": "test"})
    for name in ("load_defaults", "save_defaults", "save_repaint_defaults", "_refresh_budget", "_start_pricing_refresh"):
        monkeypatch.setattr(tab.GptImage2Widget, name, lambda self: None)
    widget = tab.GptImage2Widget()
    try:
        widget.repaint_check.setChecked(True)
        widget.post_ink_check.setChecked(True)
        assert not widget.workflow_controls.post_check.isChecked()
        assert widget.post_pipeline_steps() == {} and not widget.automatic_refine_enabled()
        assert not widget.repaint_check.isEnabled()
    finally:
        widget.refine_timer.stop()
        widget.deleteLater()
        app.processEvents()


def test_blocked_direct_generation_has_no_review(monkeypatch):
    from utils import first_image_review as review
    monkeypatch.setattr(review, "review_request", lambda *a: (_ for _ in ()).throw(AssertionError("review called")))
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        return {"saved_files": [], "server_response_raw": {"error": {"code": "moderation_blocked"}}}
    assert review.generate_first_image(generate, prompt="example", postprocess_enabled=False) == ([], "example")
    assert len(calls) == 1 and "postprocess_enabled" not in calls[0]


def test_analysis_worker_direct_has_no_pipeline_or_audit(tmp_path, monkeypatch):
    from PIL import Image
    from PyQt6.QtWidgets import QApplication
    from modules.image_analysis.single_analyzer import GptImageGenWorkerThread
    from modules.others import api_backend
    from utils import analysis_gen
    app = QApplication.instance() or QApplication([])
    image = tmp_path / "result.png"
    Image.new("RGB", (16, 16)).save(image)
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        return {"saved_files": [str(image)]}
    monkeypatch.setattr(api_backend, "generate_image_aigc2d_gpt", generate)
    monkeypatch.setattr(analysis_gen, "run_gpt_image_pipeline", lambda *a, **k: (_ for _ in ()).throw(AssertionError("pipeline called")))
    worker = GptImageGenWorkerThread({"prompt": "example", "postprocess_enabled": False},
        steps={"repaint": {"enabled": True}}, analysis_result={"task_hash": "12345678"},
        checkpoint_path=str(tmp_path / "checkpoint.json"))
    results = []
    worker.finish_signal.connect(results.append)
    worker.run()
    assert worker.last_status == "success" and results == [[str(image)]]
    assert len(calls) == 1
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert checkpoint["postprocess_enabled"] is False
    assert all(stage not in checkpoint["stages"] for stage in ("identity", "hands", "final_review"))


def test_dialog_shared_disables_fields_and_restores_independent(monkeypatch):
    from PyQt6.QtWidgets import QApplication, QLineEdit
    from utils import gpt_workflow_widget as widget
    from modules.others import api_backend
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(widget, "load_settings", lambda: {"base_url": "https://own.test/v1", "same_as_gemini": False})
    monkeypatch.setattr(widget, "dedicated_key", lambda: "saved-secret")
    monkeypatch.setattr(api_backend, "get_api_config", lambda **k: {"base_url": "https://shared.test/v1", "api_key": "shared-secret"})
    dialog = widget.GptConfigDialog()
    dialog.key_edit.setText("pending-secret")
    dialog.same_check.setChecked(True)
    assert not dialog.url_edit.isEnabled() and not dialog.key_edit.isEnabled()
    assert dialog.url_edit.text() == "https://shared.test/v1"
    dialog.same_check.setChecked(False)
    assert dialog.url_edit.isEnabled() and dialog.key_edit.isEnabled()
    assert dialog.url_edit.text() == "https://own.test/v1"
    assert dialog.key_edit.text() == "pending-secret"
    assert dialog.key_edit.echoMode() == QLineEdit.EchoMode.Password
