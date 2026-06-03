import importlib
import importlib.util
import json
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
from modules.fashion_collection.collector_tab import FashionCollectorWidget
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
        FashionCollectorWidget(project_root=str(REPO_ROOT)),
    ]

    for widget in widgets:
        widget.show()
        qapp.processEvents()
        widget.close()


def test_fashion_collector_widget_has_generation_controls(qapp):
    widget = FashionCollectorWidget(project_root=str(REPO_ROOT))
    widget.show()
    qapp.processEvents()

    assert widget.site_combo.count() >= 3
    assert hasattr(widget, "theme_input")
    assert hasattr(widget, "style_combo")
    assert hasattr(widget, "character_count_combo")
    assert hasattr(widget, "generate_btn")
    assert hasattr(widget, "hair_accessory_cb")
    assert hasattr(widget, "bag_cb")
    assert widget.generate_btn.text() == "生成少女图"
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
    other_labels = [window.others_tabs.tabText(i) for i in range(window.others_tabs.count())]
    assert "SD-WebUI接口配置" in setting_labels
    assert "服饰素材采集" in other_labels
    assert hasattr(window.sd_workflow_tab, "use_nsfw_text_api_cb")
    assert hasattr(window.sd_workflow_tab, "story_page_count_input")
    assert hasattr(window.sd_workflow_tab, "story_prompt_preset_combo")
    assert hasattr(window.sd_workflow_tab, "story_prompt_min_words_input")
    assert hasattr(window.sd_workflow_tab, "story_prompt_keyword_count_input")
    assert hasattr(window.sd_workflow_tab, "story_no_appearance_description_cb")
    assert hasattr(window.sd_workflow_tab, "story_no_outfit_description_cb")
    assert hasattr(window.sd_workflow_tab, "open_payload_editor_btn")
    assert hasattr(window.sd_workflow_tab, "webui_extra_payload_summary_label")
    assert hasattr(window.sd_workflow_tab, "load_story_file_btn")
    assert hasattr(window.sd_workflow_tab, "render_story_btn")
    assert hasattr(window.sd_workflow_tab, "open_story_preview_btn")
    assert not hasattr(window.sd_workflow_tab, "use_cohere_cb")
    assert not hasattr(window.sd_webui_settings_tab, "extra_payload_input")
    assert window.sd_workflow_tab.style_combo.currentText() == window.last_used_style
    window.close()


def test_story_sequence_editor_table_and_preview(qapp, tmp_path):
    story_path = tmp_path / "story-sequence.json"
    story_payload = {
        "theme": "测试主题",
        "title_zh": "中文标题",
        "title_en": "English Title",
        "pages": [
            {
                "page": 1,
                "title_zh": "第一页",
                "title_en": "Page One",
                "prompt_zh": "中文提示词一",
                "prompt_en": "english prompt one",
                "width": 768,
                "height": 1024,
            },
            {
                "page": 2,
                "title_zh": "第二页",
                "title_en": "Page Two",
                "prompt_zh": "中文提示词二",
                "prompt_en": "english prompt two",
                "width": 832,
                "height": 1216,
            },
        ],
    }
    story_path.write_text(json.dumps(story_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    dialog = sd_workflow_module.StorySequenceEditorDialog(str(story_path))
    dialog.show()
    qapp.processEvents()

    assert dialog.table.rowCount() == 2
    assert dialog.theme_input.text() == "测试主题"
    assert dialog.preview_zh.toPlainText() == "中文提示词一"

    dialog.table.selectRow(1)
    qapp.processEvents()
    assert dialog.preview_en.toPlainText() == "english prompt two"

    dialog.table.item(1, 4).setText("updated english prompt")
    normalized = dialog.get_normalized_data()
    assert normalized["pages"][1]["prompt_en"] == "updated english prompt"
    dialog.close()


def test_story_sequence_preview_dialog_loads_pages(qapp, tmp_path):
    story_path = tmp_path / "story-preview.json"
    story_payload = {
        "theme": "预览主题",
        "title_zh": "预览标题",
        "title_en": "Preview Title",
        "pages": [
            {
                "page": 1,
                "title_zh": "预览第一页",
                "title_en": "Preview One",
                "prompt_zh": "预览中文提示词",
                "prompt_en": "preview english prompt",
                "width": 768,
                "height": 1024,
            }
        ],
    }
    story_path.write_text(json.dumps(story_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    dialog = sd_workflow_module.StorySequencePreviewDialog(str(story_path))
    dialog.show()
    qapp.processEvents()

    assert dialog.page_list.count() == 1
    assert "预览第一页" in dialog.page_list.item(0).text()
    assert dialog.zh_preview.toPlainText() == "预览中文提示词"
    assert dialog.en_preview.toPlainText() == "preview english prompt"
    dialog.close()


def test_fetch_llm_reply_with_continuation_concatenates_length_responses(monkeypatch):
    responses = [
        {
            "choices": [
                {
                    "message": {"content": '{"page":1,"title_en":"Page 1","title_zh":"第一页","prompt_en":"first half '},
                    "finish_reason": "length",
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {"content": 'second half","prompt_zh":"中文","width":768,"height":1024}'},
                    "finish_reason": "stop",
                }
            ]
        },
    ]

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_post(*args, **kwargs):
        return FakeResponse(responses.pop(0))

    monkeypatch.setattr(sd_workflow_module.requests, "post", fake_post)
    reply_text = sd_workflow_module.fetch_llm_reply_with_continuation(
        base_url="http://example.com/v1",
        api_key="test-key",
        model="test-model",
        system_prompt="system",
        user_content="user",
        force_json=False,
        merge_system_prompt=False,
    )
    parsed = sd_workflow_module._parse_json_reply(reply_text)
    assert parsed["prompt_en"] == "first half second half"
    assert parsed["prompt_zh"] == "中文"


def test_webui_extra_payload_editor_dialog_parses_nested_values(qapp):
    payload_text = json.dumps(
        {
            "override_settings": {"sd_model_checkpoint": "model-a", "forge_additional_modules": ["vae-a"]},
            "cfg_scale": 7.5,
            "enable_hr": True,
            "note": "plain text",
        },
        ensure_ascii=False,
        indent=2,
    )
    dialog = sd_workflow_module.WebuiExtraPayloadEditorDialog(payload_text)
    dialog.show()
    qapp.processEvents()

    assert dialog.table.rowCount() == 4
    assert "sd_model_checkpoint" in dialog._get_row_value_text(0)

    dialog.table.selectRow(0)
    qapp.processEvents()
    dialog.value_editor.setPlainText('{"nested": {"value": 1}}')
    dialog.apply_value_to_selected_row()
    dialog.table.selectRow(3)
    qapp.processEvents()
    dialog.value_editor.setPlainText("updated note")
    dialog.apply_value_to_selected_row()

    payload = dialog.get_payload_dict()
    assert payload["override_settings"] == {"nested": {"value": 1}}
    assert payload["cfg_scale"] == 7.5
    assert payload["enable_hr"] is True
    assert payload["note"] == "updated note"
    dialog.close()


def test_sd_workflow_existing_story_enables_story_actions(qapp, tmp_path):
    story_path = tmp_path / "existing-story.json"
    story_payload = {
        "theme": "已有故事",
        "title_zh": "已有标题",
        "title_en": "Existing Title",
        "pages": [
            {
                "page": 1,
                "title_zh": "第一页",
                "title_en": "Page One",
                "prompt_zh": "中文内容",
                "prompt_en": "english content",
                "width": 768,
                "height": 1024,
            }
        ],
    }
    story_path.write_text(json.dumps(story_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    widget = sd_workflow_module.SdWorkflowWidget()
    widget._set_story_file_path(str(story_path))

    assert widget.generate_story_btn.isEnabled()
    assert widget.load_story_file_btn.isEnabled()
    assert widget.open_story_preview_btn.isEnabled()
    assert widget.open_story_editor_btn.isEnabled()
    assert widget.render_story_btn.isEnabled()
    widget.close()


def test_sd_workflow_story_prompt_settings_persist_in_state(qapp):
    widget = sd_workflow_module.SdWorkflowWidget()
    widget.story_prompt_min_words_input.setValue(320)
    widget.story_prompt_keyword_count_input.setValue(24)
    widget.story_no_outfit_description_cb.setChecked(True)
    widget.update_config_from_ui()

    assert widget.config["story_prompt_preset"] == "自定义"
    assert widget.config["story_prompt_min_words"] == 320
    assert widget.config["story_prompt_keyword_count"] == 24
    assert widget.config["story_no_appearance_description"] is True
    assert widget.config["story_no_outfit_description"] is True
    assert widget.config["story_no_character_description"] is True
    widget.close()


def test_sd_workflow_story_prompt_preset_updates_values(qapp):
    widget = sd_workflow_module.SdWorkflowWidget()
    widget.story_prompt_preset_combo.setCurrentText("保守")

    assert widget.story_prompt_min_words_input.value() == 180
    assert widget.story_prompt_keyword_count_input.value() == 16

    widget.story_prompt_min_words_input.setValue(333)
    assert widget.story_prompt_preset_combo.currentText() == "自定义"
    widget.close()


def test_sd_workflow_story_description_options_are_hierarchical(qapp):
    widget = sd_workflow_module.SdWorkflowWidget()
    widget.story_no_appearance_description_cb.setChecked(False)
    widget.story_no_outfit_description_cb.setChecked(False)
    widget.sync_story_description_options()

    assert widget.story_no_appearance_description_cb.isChecked() is False
    assert widget.story_no_outfit_description_cb.isEnabled() is False

    widget.story_no_outfit_description_cb.setChecked(True)
    assert widget.story_no_appearance_description_cb.isChecked() is True
    assert widget.story_no_outfit_description_cb.isChecked() is True
    assert widget.story_no_outfit_description_cb.isEnabled() is True

    widget.story_no_appearance_description_cb.setChecked(False)
    assert widget.story_no_appearance_description_cb.isChecked() is False
    assert widget.story_no_outfit_description_cb.isChecked() is False
    assert widget.story_no_outfit_description_cb.isEnabled() is False
    widget.close()


def test_save_story_sequence_uses_short_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(sd_workflow_module, "STORY_SEQUENCE_DIR", str(tmp_path))
    story_payload = {
        "theme": "这是一个非常非常长的故事主题文件名测试",
        "title_zh": "标题",
        "title_en": "Title",
        "pages": [
            {
                "page": 1,
                "title_zh": "第一页",
                "title_en": "Page One",
                "prompt_zh": "中文内容",
                "prompt_en": "english content",
                "width": 768,
                "height": 1024,
            }
        ],
    }

    saved_path, _normalized = sd_workflow_module.save_story_sequence(story_payload)
    basename = Path(saved_path).name

    assert basename.endswith(".json")
    assert len(basename) <= 32


def test_normalize_story_sequence_maps_old_resolution_to_recommended():
    normalized = sd_workflow_module.normalize_story_sequence(
        {
            "theme": "测试故事",
            "pages": [
                {
                    "page": 1,
                    "title_zh": "第一页",
                    "title_en": "Page One",
                    "prompt_zh": "中文内容",
                    "prompt_en": "english content",
                    "width": 1024,
                    "height": 768,
                }
            ],
        }
    )

    assert normalized["pages"][0]["width"] == 1344
    assert normalized["pages"][0]["height"] == 1024


def test_sd_workflow_can_load_history_story_file(qapp, monkeypatch, tmp_path):
    story_path = tmp_path / "picked-story.json"
    story_payload = {
        "theme": "载入故事",
        "title_zh": "载入标题",
        "title_en": "Loaded Title",
        "pages": [
            {
                "page": 1,
                "title_zh": "第一页",
                "title_en": "Page One",
                "prompt_zh": "中文内容",
                "prompt_en": "english content",
                "width": 768,
                "height": 1024,
            }
        ],
    }
    story_path.write_text(json.dumps(story_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    widget = sd_workflow_module.SdWorkflowWidget()
    monkeypatch.setattr(
        sd_workflow_module.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(story_path), "JSON Files (*.json)"),
    )

    widget.choose_story_sequence_file()

    assert widget.config["last_story_json_path"] == str(story_path)
    assert widget.open_story_preview_btn.isEnabled()
    assert widget.open_story_editor_btn.isEnabled()
    assert widget.render_story_btn.isEnabled()
    widget.close()


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
