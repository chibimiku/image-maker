import importlib
import importlib.util
import os
from pathlib import Path

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QMimeData, QUrl
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication, QWidget

import modules.image_generation.char_design as char_design_module
import modules.image_generation.image_edit as image_edit_module
import modules.image_analysis.single_analyzer as single_analyzer_module
import modules.image_generation.prompt_generator as prompt_generator_module
import modules.image_generation.sd_workflow_tab as sd_workflow_module
import modules.image_generation.single_gen_debug_tab as single_gen_debug_module
import modules.image_analysis.json_dataset_tab as json_dataset_module
import utils.task_runtime as task_runtime_module
from modules.image_analysis.json_dataset_tab import JsonDatasetWidget
from modules.image_analysis.single_analyzer import SingleAnalyzerWidget
from modules.image_generation.char_design import CharDesignWidget
from modules.image_generation.flux2_client_tab import Flux2ClientWidget
from modules.image_generation.image_edit import ImageEditWidget
from modules.image_generation.prompt_generator import PromptGeneratorWidget
from modules.image_generation.sd_workflow_tab import SdWorkflowWidget
from modules.image_generation.single_gen_debug_tab import SingleGenDebugWidget
from modules.image_generation.upscaler_tab import UpscalerTabWidget
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def load_module(module_name: str, relative_path: str, block_onnxruntime: bool = False):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    if not block_onnxruntime:
        spec.loader.exec_module(module)
        return module

    original_import_module = importlib.import_module

    def safe_import_module(name, package=None):
        if name == "onnxruntime":
            raise ImportError("skip onnxruntime preload in smoke test")
        return original_import_module(name, package)

    importlib.import_module = safe_import_module
    try:
        spec.loader.exec_module(module)
    finally:
        importlib.import_module = original_import_module
    return module


def text_config():
    return ("https://example.invalid/v1", "test-key", "test-model")


def image_config():
    return ("https://example.invalid/v1", "image-key", "image-model", "openai")


def styles_config():
    return {"默认风格": "masterpiece", "维多利亚": "victorian dress"}


def ar_policy():
    return {
        "default_aspect_ratio": "1:1",
        "override_first": "不覆盖(沿用原逻辑)",
        "override_second": "3:4",
    }


def create_png(path: Path):
    image = QImage(12, 12, QImage.Format.Format_RGB32)
    image.fill(0x00FF00)
    assert image.save(str(path))
    return str(path)


class DropEventStub:
    def __init__(self, mime_data):
        self._mime_data = mime_data
        self.accepted = False
        self.ignored = False

    def mimeData(self):
        return self._mime_data

    def acceptProposedAction(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


def test_main_windows_construct_under_pyqt6(qapp):
    app_module = load_module("app_module_smoke", "app.py", block_onnxruntime=True)
    make_pic_module = load_module("make_pic_module_smoke", "make-pic.py")
    doujin_module = load_module("doujin_module_smoke", "doujin_translator.py")

    windows = [
        app_module.AppWindow(),
        make_pic_module.ImageGeneratorGUI(),
        doujin_module.AppWindow(),
    ]

    for window in windows:
        window.show()
        qapp.processEvents()
        window.close()


def test_standalone_pyqt6_widgets_construct(qapp):
    widgets = [
        Flux2ClientWidget(),
        UpscalerTabWidget(),
        SdWorkflowWidget(),
    ]

    for widget in widgets:
        widget.show()
        qapp.processEvents()
        widget.close()


def test_app_window_contains_sd_workflow_tab(qapp):
    app_module = load_module("app_module_sd_tab_smoke", "app.py", block_onnxruntime=True)
    window = app_module.AppWindow()
    window.show()
    qapp.processEvents()

    labels = [window.generation_tabs.tabText(i) for i in range(window.generation_tabs.count())]
    assert "SD 批量工作流" in labels
    assert isinstance(window.sd_workflow_tab, sd_workflow_module.SdWorkflowWidget)
    setting_labels = [window.config_tabs.tabText(i) for i in range(window.config_tabs.count())]
    assert "SD-WebUI接口配置" in setting_labels
    assert hasattr(window.sd_workflow_tab, "use_nsfw_text_api_cb")
    assert not hasattr(window.sd_workflow_tab, "use_cohere_cb")
    assert window.sd_workflow_tab.style_combo.currentText() == window.last_used_style
    window.close()


def test_sd_workflow_style_combo_shares_main_style_state(qapp):
    app_module = load_module("app_module_sd_style_sync", "app.py", block_onnxruntime=True)
    window = app_module.AppWindow()
    window.show()
    qapp.processEvents()

    window.styles_data["测试共享风格"] = "shared style prompt"
    window.update_style_combos()
    window.sync_selected_style("测试共享风格")
    qapp.processEvents()

    assert window.last_used_style == "测试共享风格"
    assert window.sd_workflow_tab.style_combo.currentText() == "测试共享风格"
    window.close()


def test_app_window_tab_switching_preserves_single_analyzer_interactions(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(single_analyzer_module, "list_esrgan_models", lambda: ["realesrgan-x4plus"])
    monkeypatch.setattr(single_analyzer_module.ImageGrab, "grabclipboard", lambda: Image.new("RGB", (18, 18), "blue"))

    app_module = load_module("app_module_tab_smoke", "app.py", block_onnxruntime=True)
    window = app_module.AppWindow()
    window.show()
    qapp.processEvents()

    for tabs in (window.main_tabs, window.analysis_tabs, window.generation_tabs, window.others_tabs):
        for index in range(tabs.count()):
            tabs.setCurrentIndex(index)
            qapp.processEvents()

    image_path = create_png(tmp_path / "app-window-drop.png")
    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile(image_path)])

    window.main_tabs.setCurrentWidget(window.analysis_root_tab)
    window.analysis_tabs.setCurrentWidget(window.single_analyzer_tab)
    qapp.processEvents()
    window.single_analyzer_tab.dropEvent(DropEventStub(mime_data))

    assert os.path.normpath(window.single_analyzer_tab.image_source) == os.path.normpath(image_path)
    assert window.single_analyzer_tab.send_btn.isEnabled() is True

    class KeyEventStub:
        def modifiers(self):
            return single_analyzer_module.Qt.KeyboardModifier.ControlModifier

        def key(self):
            return single_analyzer_module.Qt.Key.Key_V

    window.main_tabs.setCurrentWidget(window.generation_root_tab)
    qapp.processEvents()
    window.main_tabs.setCurrentWidget(window.analysis_root_tab)
    window.analysis_tabs.setCurrentWidget(window.single_analyzer_tab)
    qapp.processEvents()
    window.single_analyzer_tab.keyPressEvent(KeyEventStub())

    assert isinstance(window.single_analyzer_tab.image_source, Image.Image)
    assert "已从剪贴板加载图片" in window.single_analyzer_tab.log_text.toPlainText()
    window.close()


def test_system_notifier_notify_is_safe_with_and_without_system_tray(monkeypatch):
    host = QWidget()
    notifier = task_runtime_module.SystemNotifier(host)
    calls = []

    class FakeTray:
        def showMessage(self, title, message, icon, timeout_ms):
            calls.append((title, message, icon, timeout_ms))

    notifier._tray = FakeTray()
    monkeypatch.setattr(task_runtime_module.QSystemTrayIcon, "isSystemTrayAvailable", lambda: True)
    notifier.notify("完成", "任务已结束", 1234)

    monkeypatch.setattr(task_runtime_module.QSystemTrayIcon, "isSystemTrayAvailable", lambda: False)
    notifier.notify("忽略", "无托盘时不应报错", 4321)

    assert calls == [("完成", "任务已结束", task_runtime_module.MESSAGE_ICON, 1234)]
    host.close()


def test_single_analyzer_widget_notification_bridge_forwards_to_notifier(qapp, monkeypatch):
    monkeypatch.setattr(single_analyzer_module, "list_esrgan_models", lambda: ["realesrgan-x4plus"])

    forwarded = []

    class FakeNotifier:
        def notify(self, title, message, timeout_ms=5000):
            forwarded.append((title, message, timeout_ms))

    widget = SingleAnalyzerWidget(
        config_getter_func=text_config,
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
        save_img_cfg_callback=lambda *args, **kwargs: None,
        ar_policy_getter_func=ar_policy,
        nsfw_default_getter_func=lambda: False,
        upscale_options_getter_func=lambda: {},
        outfit_style_history_getter_func=lambda: [],
        outfit_style_default_getter_func=lambda: "",
    )
    widget._notifier = FakeNotifier()

    widget._send_system_notification("单图分析完成", "任务已完成并生成结果文件。")
    widget._send_system_notification("生图任务已终止", "当前生图任务已手动取消。")

    assert forwarded == [
        ("单图分析完成", "任务已完成并生成结果文件。", 5000),
        ("生图任务已终止", "当前生图任务已手动取消。", 5000),
    ]
    widget.close()


def test_single_analyzer_widget_outfit_history_interaction(qapp, monkeypatch):
    monkeypatch.setattr(single_analyzer_module, "list_esrgan_models", lambda: ["realesrgan-x4plus"])

    changes = []
    deletes = []
    widget = SingleAnalyzerWidget(
        config_getter_func=text_config,
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
        save_img_cfg_callback=lambda *args, **kwargs: None,
        ar_policy_getter_func=ar_policy,
        nsfw_default_getter_func=lambda: True,
        nsfw_changed_callback=lambda checked: changes.append(("nsfw", checked)),
        booru_tag_limit_getter_func=lambda: 42,
        timeout_getter_func=lambda: 60,
        upscale_options_getter_func=lambda: {"enabled": True, "model_name": "realesrgan-x4plus"},
        upscale_options_changed_callback=lambda opts: changes.append(("upscale", opts)),
        outfit_check_default_getter_func=lambda: True,
        outfit_check_changed_callback=lambda checked: changes.append(("outfit_check", checked)),
        outfit_style_history_getter_func=lambda: ["维多利亚风格", "校服", "哥特"],
        outfit_style_default_getter_func=lambda: "维多利亚风格",
        outfit_style_changed_callback=lambda text, add_to_history=False: changes.append(("style", text, add_to_history)),
        outfit_style_delete_callback=lambda text: deletes.append(text),
    )

    widget.update_styles(["默认风格", "维多利亚"])
    widget._filter_outfit_style_history_live("维")
    qapp.processEvents()
    assert widget.outfit_style_combo.count() == 1
    assert widget.outfit_style_combo.itemText(0) == "维多利亚风格"

    widget._commit_outfit_style_text("新风格", add_to_history=True)
    widget.outfit_style_combo.setCurrentText("校服")
    widget._delete_current_outfit_style_history()

    assert ("style", "新风格", True) in changes
    assert deletes == ["校服"]
    widget.close()


def test_single_analyzer_widget_drop_event_loads_image(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(single_analyzer_module, "list_esrgan_models", lambda: ["realesrgan-x4plus"])

    widget = SingleAnalyzerWidget(
        config_getter_func=text_config,
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
        save_img_cfg_callback=lambda *args, **kwargs: None,
        ar_policy_getter_func=ar_policy,
        nsfw_default_getter_func=lambda: False,
        upscale_options_getter_func=lambda: {},
        outfit_style_history_getter_func=lambda: [],
        outfit_style_default_getter_func=lambda: "",
    )

    image_path = create_png(tmp_path / "drop.png")
    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile(image_path)])

    widget.dropEvent(DropEventStub(mime_data))

    assert os.path.normpath(widget.image_source) == os.path.normpath(image_path)
    assert widget.send_btn.isEnabled() is True
    widget.close()


def test_single_analyzer_widget_ctrl_v_loads_clipboard_image(qapp, monkeypatch):
    monkeypatch.setattr(single_analyzer_module, "list_esrgan_models", lambda: ["realesrgan-x4plus"])
    monkeypatch.setattr(single_analyzer_module.ImageGrab, "grabclipboard", lambda: Image.new("RGB", (16, 16), "red"))

    widget = SingleAnalyzerWidget(
        config_getter_func=text_config,
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
        save_img_cfg_callback=lambda *args, **kwargs: None,
        ar_policy_getter_func=ar_policy,
        nsfw_default_getter_func=lambda: False,
        upscale_options_getter_func=lambda: {},
        outfit_style_history_getter_func=lambda: [],
        outfit_style_default_getter_func=lambda: "",
    )

    class KeyEventStub:
        def modifiers(self):
            return single_analyzer_module.Qt.KeyboardModifier.ControlModifier

        def key(self):
            return single_analyzer_module.Qt.Key.Key_V

    widget.keyPressEvent(KeyEventStub())

    assert isinstance(widget.image_source, Image.Image)
    assert widget.send_btn.isEnabled() is True
    assert "已从剪贴板加载图片" in widget.log_text.toPlainText()
    widget.close()


def test_prompt_generator_widget_collects_upscale_options(qapp, monkeypatch):
    monkeypatch.setattr(prompt_generator_module, "list_esrgan_models", lambda: ["realesrgan-x4plus", "omnisr"])

    captured = []
    widget = PromptGeneratorWidget(
        config_getter_func=text_config,
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
        save_img_cfg_callback=lambda *args, **kwargs: None,
        ar_policy_getter_func=ar_policy,
        upscale_options_getter_func=lambda: {"enabled": False, "model_name": "omnisr", "upscale_by": 1.5, "webp_target_mb": 8.0},
        upscale_options_changed_callback=lambda opts: captured.append(opts),
    )

    widget.update_styles(["默认风格", "维多利亚"])
    widget.enable_jpg_upscale_cb.setChecked(True)
    widget.upscale_model_combo.setCurrentText("realesrgan-x4plus")
    widget.upscale_by_spin.setValue(2.5)
    widget.webp_target_mb_spin.setValue(12.0)

    options = widget._collect_upscale_options()
    assert widget.main_style_combo.count() == 2
    assert options["enabled"] is True
    assert options["model_name"] == "realesrgan-x4plus"
    assert options["upscale_by"] == 2.5
    assert options["webp_target_mb"] == 12.0
    assert captured
    widget.close()


def test_single_gen_debug_widget_loads_json_into_prompt(qapp, tmp_path):
    widget = SingleGenDebugWidget(
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
        save_img_cfg_callback=lambda *args, **kwargs: None,
        ar_policy_getter_func=ar_policy,
    )

    widget.update_styles(["默认风格", "维多利亚"])
    json_path = tmp_path / "sample.json"
    json_path.write_text(
        '{"english_description": "girl in victorian dress", "original_english_description": "girl in school uniform"}',
        encoding="utf-8",
    )

    assert widget._load_json_data(str(json_path)) is True
    widget.json_field_combo.setCurrentIndex(0)
    widget.apply_json_field()

    assert "girl in victorian dress" in widget.prompt_edit.toPlainText()
    assert widget.json_apply_btn.isEnabled()
    widget.close()


def test_single_gen_debug_widget_drop_json_updates_prompt_immediately(qapp, tmp_path):
    widget = SingleGenDebugWidget(
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
        save_img_cfg_callback=lambda *args, **kwargs: None,
        ar_policy_getter_func=ar_policy,
    )

    json_path = tmp_path / "dropped.json"
    json_path.write_text(
        '{"english_description": "drop refined prompt", "original_english_description": "drop original prompt"}',
        encoding="utf-8",
    )
    widget.json_field_combo.setCurrentIndex(0)

    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile(str(json_path))])
    widget.json_drop_label.dropEvent(DropEventStub(mime_data))

    assert widget.json_apply_btn.isEnabled() is True
    assert widget.prompt_edit.toPlainText() == "drop refined prompt"
    assert "已将「精修英文描述」内容同步到 Prompt" in widget.log_text.toPlainText()
    widget.close()


def test_json_file_drop_list_widget_drop_event_adds_unique_json_paths(qapp, tmp_path):
    file_list = json_dataset_module.JsonFileDropListWidget()

    first_json = tmp_path / "first.json"
    second_json = tmp_path / "second.json"
    ignored_txt = tmp_path / "ignored.txt"
    first_json.write_text("{}", encoding="utf-8")
    second_json.write_text("{}", encoding="utf-8")
    ignored_txt.write_text("skip", encoding="utf-8")

    file_list.add_json_files([str(first_json)])

    mime_data = QMimeData()
    mime_data.setUrls(
        [
            QUrl.fromLocalFile(str(first_json)),
            QUrl.fromLocalFile(str(second_json)),
            QUrl.fromLocalFile(str(ignored_txt)),
        ]
    )
    event = DropEventStub(mime_data)
    file_list.dropEvent(event)

    assert event.accepted is True
    assert file_list.count() == 2
    assert file_list.get_all_paths() == [str(first_json), str(second_json)]
    file_list.close()


def test_image_edit_widget_drop_collects_directory_images_and_updates_list(qapp, monkeypatch, tmp_path):
    prompt_dir = tmp_path / "image-edit-prompts-drop"
    prompt_dir.mkdir()
    (prompt_dir / "template_a.md").write_text("template-a", encoding="utf-8")
    state_file = tmp_path / "image_edit_drop_state.json"

    monkeypatch.setattr(image_edit_module, "PROMPT_DIR", str(prompt_dir))
    monkeypatch.setattr(image_edit_module, "IMAGE_EDIT_UI_STATE_FILE", str(state_file))

    image_dir = tmp_path / "images"
    nested_dir = image_dir / "nested"
    nested_dir.mkdir(parents=True)
    image_one = create_png(image_dir / "one.png")
    image_two = create_png(nested_dir / "two.png")
    (image_dir / "ignored.txt").write_text("skip", encoding="utf-8")

    widget = ImageEditWidget(
        config_getter_func=text_config,
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
    )

    mime_data = QMimeData()
    mime_data.setUrls(
        [
            QUrl.fromLocalFile(str(image_dir)),
            QUrl.fromLocalFile(image_one),
        ]
    )
    collected = widget._collect_image_paths_from_drop(mime_data)
    assert set(collected) == {os.path.normpath(image_one), os.path.normpath(image_two)}
    assert len(collected) == 2

    event = DropEventStub(mime_data)
    widget.dropEvent(event)

    assert event.accepted is True
    assert widget.image_list.count() == 2
    assert widget.image_list.item(0).data(image_edit_module.Qt.ItemDataRole.UserRole) in collected
    assert widget.image_list.item(1).data(image_edit_module.Qt.ItemDataRole.UserRole) in collected
    assert "新增 2 张" in widget.log_text.toPlainText()
    widget.close()


def test_json_dataset_widget_running_state_and_quick_jump(qapp, tmp_path):
    json_dataset_module.QMessageBox.information = lambda *args, **kwargs: None
    json_dataset_module.QMessageBox.warning = lambda *args, **kwargs: None

    widget = JsonDatasetWidget()
    captured = []
    widget.quick_split_requested.connect(captured.append)

    output_dir = tmp_path / "output"
    json_path = tmp_path / "result.json"
    json_path.write_text("{}", encoding="utf-8")

    widget.prefill_for_batch([str(json_path)], str(output_dir))
    assert widget.file_list.count() == 1

    widget.set_running_state(True)
    assert widget.start_btn.isEnabled() is False
    assert widget.cancel_btn.isEnabled() is True

    output_dir.mkdir()
    widget.output_dir_input.setText(str(output_dir))
    widget.on_processing_finished("success", "done")
    widget.quick_jump_to_pic_cate()

    assert captured == [str(output_dir)]
    widget.close()


def test_single_gen_debug_widget_ui_state_roundtrip(qapp, monkeypatch, tmp_path):
    state_file = tmp_path / "single_debug_state.json"
    monkeypatch.setattr(single_gen_debug_module, "SINGLE_DEBUG_UI_STATE_FILE", str(state_file))

    attachment = create_png(tmp_path / "attach.png")

    widget = SingleGenDebugWidget(
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
        save_img_cfg_callback=lambda *args, **kwargs: None,
        ar_policy_getter_func=ar_policy,
    )
    widget.prompt_edit.setPlainText("state prompt")
    widget.aspect_ratio_combo.setCurrentText("3:4")
    widget.resolution_combo.setCurrentText("1024x1536")
    widget.attach_image_paths = [attachment]
    widget.save_ui_state()
    widget.close()

    restored = SingleGenDebugWidget(
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
        save_img_cfg_callback=lambda *args, **kwargs: None,
        ar_policy_getter_func=ar_policy,
    )
    assert restored.prompt_edit.toPlainText() == "state prompt"
    assert restored.aspect_ratio_combo.currentText() == "3:4"
    assert restored.resolution_combo.currentText() == "1024x1536"
    assert restored.attach_image_paths == [attachment]
    restored.close()


def test_single_gen_debug_widget_switches_json_fields_and_appends_prompt(qapp, tmp_path):
    widget = SingleGenDebugWidget(
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
        save_img_cfg_callback=lambda *args, **kwargs: None,
        ar_policy_getter_func=ar_policy,
    )

    json_path = tmp_path / "fields.json"
    json_path.write_text(
        (
            '{"english_description": "refined prompt", '
            '"original_english_description": "original prompt", '
            '"short_description": "short prompt"}'
        ),
        encoding="utf-8",
    )

    assert widget.json_apply_btn.isEnabled() is False
    assert widget._load_json_data(str(json_path)) is True
    assert widget.json_apply_btn.isEnabled() is True

    widget.json_field_combo.setCurrentIndex(1)
    assert widget._get_selected_field_text() == "original prompt"
    assert widget.prompt_edit.toPlainText() == "original prompt"

    widget.prompt_edit.setPlainText("base prompt")
    widget.apply_json_field()
    assert widget.prompt_edit.toPlainText() == "base prompt\n\noriginal prompt"

    widget.json_field_combo.setCurrentIndex(2)
    assert widget._get_selected_field_text() == "short prompt"
    assert widget.prompt_edit.toPlainText() == "short prompt"

    widget.prompt_edit.setPlainText("base prompt")
    widget.apply_json_field()
    assert widget.prompt_edit.toPlainText() == "base prompt\n\nshort prompt"
    assert "已切换到「简短描述」" in widget.log_text.toPlainText()
    widget.close()


def test_single_analyzer_widget_apply_history_record_enables_generate_buttons(qapp, monkeypatch):
    monkeypatch.setattr(single_analyzer_module, "list_esrgan_models", lambda: ["realesrgan-x4plus"])

    widget = SingleAnalyzerWidget(
        config_getter_func=text_config,
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
        save_img_cfg_callback=lambda *args, **kwargs: None,
        ar_policy_getter_func=ar_policy,
        nsfw_default_getter_func=lambda: False,
        upscale_options_getter_func=lambda: {},
        outfit_style_history_getter_func=lambda: [],
        outfit_style_default_getter_func=lambda: "",
    )

    assert widget.gen_orig_btn.isEnabled() is False
    assert widget.gen_ref_btn.isEnabled() is False

    applied = widget._apply_history_record_to_current_state(
        {
            "status": "success",
            "task_hash": "abc123",
            "aspect_ratio": "3:4",
            "result_json": {
                "english_description": "refined history prompt",
                "original_english_description": "original history prompt",
            },
        }
    )

    assert applied is True
    assert widget.current_task_hash == "abc123"
    assert widget.current_aspect_ratio == "3:4"
    assert widget.current_refine_desc == "refined history prompt"
    assert widget.current_orig_desc == "original history prompt"
    assert widget.gen_orig_btn.isEnabled() is True
    assert widget.gen_ref_btn.isEnabled() is True
    widget.close()


def test_char_design_widget_ui_state_roundtrip(qapp, monkeypatch, tmp_path):
    prompts_dir = tmp_path / "char-prompts"
    prompts_dir.mkdir()
    (prompts_dir / "hero.json").write_text(
        '{"common_prompt": {"prompt": "base"}, "specialized_prompts": [{"id": "p1", "description": "front", "aspect_ratio": "1:1"}]}',
        encoding="utf-8",
    )
    state_file = tmp_path / "char_state.json"

    monkeypatch.setattr(char_design_module, "CHAR_PROMPT_DIR", str(prompts_dir))
    monkeypatch.setattr(char_design_module, "CHAR_DESIGN_UI_STATE_FILE", str(state_file))

    widget = CharDesignWidget(
        config_getter_func=text_config,
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
        upscale_options_getter_func=lambda: {"enabled": False},
        upscale_options_changed_callback=lambda *args, **kwargs: None,
    )
    widget.update_styles(["默认风格", "维多利亚"])
    widget.json_combo.setCurrentText("hero.json")
    widget.main_style_combo.setCurrentText("维多利亚")
    widget.thread_spin.setValue(5)
    widget.resolution_combo.setCurrentText("2K")
    widget.custom_prefix_prompt.setPlainText("prefix")
    widget.concat_requirement_position_combo.setCurrentText("拼接在正文前")
    widget.save_ui_state()
    widget.close()

    restored = CharDesignWidget(
        config_getter_func=text_config,
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
        upscale_options_getter_func=lambda: {"enabled": False},
        upscale_options_changed_callback=lambda *args, **kwargs: None,
    )
    restored.update_styles(["默认风格", "维多利亚"])
    assert restored.json_combo.currentText() == "hero.json"
    assert restored.main_style_combo.currentText() == "维多利亚"
    assert restored.thread_spin.value() == 5
    assert restored.resolution_combo.currentText() == "2K"
    assert restored.custom_prefix_prompt.toPlainText() == "prefix"
    assert restored.concat_requirement_position_combo.currentText() == "拼接在正文前"
    restored.close()


def test_char_design_widget_select_and_deselect_tasks(qapp, monkeypatch, tmp_path):
    prompts_dir = tmp_path / "char-prompts-select"
    prompts_dir.mkdir()
    (prompts_dir / "hero.json").write_text(
        '{"common_prompt": {"prompt": "base"}, "specialized_prompts": [{"id": "p1", "description": "front", "aspect_ratio": "1:1"}, {"id": "p2", "description": "back", "aspect_ratio": "3:4"}]}',
        encoding="utf-8",
    )
    state_file = tmp_path / "char_state_select.json"

    monkeypatch.setattr(char_design_module, "CHAR_PROMPT_DIR", str(prompts_dir))
    monkeypatch.setattr(char_design_module, "CHAR_DESIGN_UI_STATE_FILE", str(state_file))

    widget = CharDesignWidget(
        config_getter_func=text_config,
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
        upscale_options_getter_func=lambda: {"enabled": False},
        upscale_options_changed_callback=lambda *args, **kwargs: None,
    )

    assert widget.task_list_widget.count() == 2
    widget.deselect_all_tasks()
    assert all(
        widget.task_list_widget.item(i).checkState() == char_design_module.Qt.CheckState.Unchecked
        for i in range(widget.task_list_widget.count())
    )

    widget.select_all_tasks()
    assert all(
        widget.task_list_widget.item(i).checkState() == char_design_module.Qt.CheckState.Checked
        for i in range(widget.task_list_widget.count())
    )
    widget.close()


def test_image_edit_widget_ui_state_roundtrip(qapp, monkeypatch, tmp_path):
    prompt_dir = tmp_path / "image-edit-prompts"
    prompt_dir.mkdir()
    (prompt_dir / "template_a.md").write_text("template-a", encoding="utf-8")
    state_file = tmp_path / "image_edit_state.json"

    monkeypatch.setattr(image_edit_module, "PROMPT_DIR", str(prompt_dir))
    monkeypatch.setattr(image_edit_module, "IMAGE_EDIT_UI_STATE_FILE", str(state_file))

    widget = ImageEditWidget(
        config_getter_func=text_config,
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
    )
    widget.update_styles(["默认风格", "维多利亚"])
    widget.template_combo.setCurrentIndex(widget.template_combo.findText("template_a"))
    widget.main_style_combo.setCurrentText("维多利亚")
    widget.thread_spin.setValue(6)
    widget.save_ui_state()
    widget.close()

    restored = ImageEditWidget(
        config_getter_func=text_config,
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
    )
    restored.update_styles(["默认风格", "维多利亚"])
    assert restored.template_combo.currentText() == "template_a"
    assert restored.main_style_combo.currentText() == "维多利亚"
    assert restored.thread_spin.value() == 6
    restored.close()


def test_image_edit_widget_template_switch_and_add_images(qapp, monkeypatch, tmp_path):
    prompt_dir = tmp_path / "image-edit-prompts-live"
    prompt_dir.mkdir()
    (prompt_dir / "template_a.md").write_text("template-a", encoding="utf-8")
    (prompt_dir / "template_b.md").write_text("template-b", encoding="utf-8")
    state_file = tmp_path / "image_edit_live_state.json"

    monkeypatch.setattr(image_edit_module, "PROMPT_DIR", str(prompt_dir))
    monkeypatch.setattr(image_edit_module, "IMAGE_EDIT_UI_STATE_FILE", str(state_file))

    image_one = create_png(tmp_path / "one.png")
    image_two = create_png(tmp_path / "two.png")
    unsupported = tmp_path / "note.txt"
    unsupported.write_text("skip", encoding="utf-8")

    widget = ImageEditWidget(
        config_getter_func=text_config,
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
    )
    widget.update_styles(["默认风格", "维多利亚"])

    widget.on_template_changed(widget.template_combo.findText("template_b"))
    widget.template_combo.setCurrentIndex(widget.template_combo.findText("template_b"))
    assert widget.prompt_edit.toPlainText() == "template-b"

    added_count, skipped_count = widget._add_images_from_paths([image_one, image_two, image_one, str(unsupported)])
    assert added_count == 2
    assert skipped_count == 2
    assert widget.image_list.count() == 2
    assert widget.image_list.item(0).data(image_edit_module.Qt.ItemDataRole.UserRole) == os.path.normpath(image_one)
    assert widget.image_list.item(1).data(image_edit_module.Qt.ItemDataRole.UserRole) == os.path.normpath(image_two)
    widget.close()
