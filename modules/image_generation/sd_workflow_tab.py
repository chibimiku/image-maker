import os
import json
import time

import requests
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from modules.others.tag_completer import TagAutocompleteManager
from utils.prompt_loader import get_prompt_path

# 门面再导出：从 core 导入所有公开名字，保证 sd_workflow_module.X 测试访问兼容
from modules.image_generation import sd_workflow_core  # 供测试 monkeypatch sd_workflow_module.sd_workflow_core.STORY_SEQUENCE_DIR
from modules.image_generation.sd_workflow_core import (
    BASE_DIR,
    CONFIG_FILE,
    PROMPTS_DIR,
    NEG_PROMPTS_DIR,
    OUTPUT_DIR,
    CACHE_DIR,
    STORY_SEQUENCE_DIR,
    SYSTEM_PROMPT_FILE,
    STORY_SYSTEM_PROMPT_FILE,
    STORY_PAGE_SYSTEM_PROMPT_FILE,
    DEFAULT_SD_WORKFLOW_STATE,
    DEFAULT_SD_WEBUI_SETTINGS,
    STORY_PROMPT_PRESETS,
    STORY_RESOLUTION_PRESETS,
    SD_WORKFLOW_STATE_KEYS,
    SD_WEBUI_SETTING_KEYS,
    load_sd_workflow_state,
    save_sd_workflow_state,
    load_sd_webui_settings,
    save_sd_webui_settings,
    load_text_api_config_from_file,
    load_webui_extra_payload,
    dump_webui_extra_payload,
    _strip_json_fence,
    _parse_json_reply,
    _coerce_positive_int,
    _normalize_story_resolution,
    _sanitize_filename_part,
    normalize_story_sequence,
    save_story_sequence,
    load_story_sequence,
    normalize_story_outline,
    _extract_chat_message_and_reason,
    _request_chat_completion,
    fetch_llm_reply_with_continuation,
    _format_payload_value_text,
    _payload_value_preview,
    _parse_payload_value_text,
    GuiLogHandler,
    SdWorkflowThread,
    SdStorySequenceThread,
    SdStoryRenderThread,
    StorySequenceEditorDialog,
    StorySequencePreviewDialog,
    WebuiExtraPayloadEditorDialog,
)
from modules.image_generation.sd_workflow_common_panel import SdCommonConfigPanel
from modules.image_generation.sd_theme_batch_tab import SdThemeBatchTab
from modules.image_generation.sd_storyline_tab import SdStorylineTab


class SdWebuiSettingsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = load_sd_webui_settings()
        self.init_ui()
        self.load_settings_to_ui(self.settings.get("current_sd_group", "Default"))

    def init_ui(self):
        layout = QVBoxLayout(self)

        top_sd_layout = QHBoxLayout()
        top_sd_layout.addWidget(QLabel("SD API URL:"))
        self.sd_url_input = QLineEdit(self.settings.get("sd_url", ""))
        top_sd_layout.addWidget(self.sd_url_input)
        top_sd_layout.addWidget(QLabel("配置组:"))
        self.sd_group_combo = QComboBox()
        self.sd_group_combo.setMinimumWidth(140)
        self.sd_group_combo.currentTextChanged.connect(self.on_sd_group_changed)
        top_sd_layout.addWidget(self.sd_group_combo)
        self.save_sd_group_btn = QPushButton("保存为新配置组")
        self.save_sd_group_btn.clicked.connect(self.save_as_sd_group)
        top_sd_layout.addWidget(self.save_sd_group_btn)
        self.del_sd_group_btn = QPushButton("删除当前组")
        self.del_sd_group_btn.clicked.connect(self.delete_sd_group)
        top_sd_layout.addWidget(self.del_sd_group_btn)
        layout.addLayout(top_sd_layout)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Checkpoint:"))
        self.sd_model_input = QLineEdit()
        self.sd_model_input.setPlaceholderText("留空则使用 WebUI 当前模型")
        model_layout.addWidget(self.sd_model_input)
        layout.addLayout(model_layout)

        vae_main_layout = QVBoxLayout()
        vae_header_layout = QHBoxLayout()
        vae_header_layout.addWidget(QLabel("VAE (支持多个拼接):"))
        self.add_vae_btn = QPushButton("+ 添加 VAE")
        self.add_vae_btn.setFixedWidth(100)
        self.add_vae_btn.clicked.connect(lambda: self.add_vae_input_field(""))
        vae_header_layout.addWidget(self.add_vae_btn)
        vae_header_layout.addStretch()
        vae_main_layout.addLayout(vae_header_layout)
        self.vae_inputs_container = QVBoxLayout()
        self.vae_inputs_list = []
        vae_main_layout.addLayout(self.vae_inputs_container)
        layout.addLayout(vae_main_layout)

        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Sampler:"))
        self.sampler_input = QLineEdit()
        self.sampler_input.setFixedWidth(100)
        param_layout.addWidget(self.sampler_input)
        param_layout.addWidget(QLabel("Scheduler:"))
        self.scheduler_input = QLineEdit()
        self.scheduler_input.setFixedWidth(100)
        self.scheduler_input.setPlaceholderText("Automatic")
        param_layout.addWidget(self.scheduler_input)
        param_layout.addWidget(QLabel("Steps:"))
        self.steps_input = QSpinBox()
        self.steps_input.setRange(1, 150)
        param_layout.addWidget(self.steps_input)
        param_layout.addWidget(QLabel("CFG:"))
        self.cfg_input = QDoubleSpinBox()
        self.cfg_input.setRange(1.0, 30.0)
        self.cfg_input.setSingleStep(0.5)
        param_layout.addWidget(self.cfg_input)
        layout.addLayout(param_layout)

        save_layout = QHBoxLayout()
        save_layout.addStretch()
        self.save_btn = QPushButton("保存 SD-WebUI 接口配置")
        self.save_btn.clicked.connect(lambda: self.save_settings(silent=False))
        save_layout.addWidget(self.save_btn)
        layout.addLayout(save_layout)

        self.refresh_sd_groups()

    def add_vae_input_field(self, text=""):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        input_field = QLineEdit(text)
        input_field.setPlaceholderText("例如: qwen_image_vae.safetensors")
        row_layout.addWidget(input_field)
        del_btn = QPushButton("-")
        del_btn.setFixedWidth(30)
        del_btn.clicked.connect(lambda: self.remove_vae_field(row_widget, input_field))
        row_layout.addWidget(del_btn)
        self.vae_inputs_container.addWidget(row_widget)
        self.vae_inputs_list.append(input_field)

    def remove_vae_field(self, widget, input_field):
        self.vae_inputs_container.removeWidget(widget)
        widget.deleteLater()
        if input_field in self.vae_inputs_list:
            self.vae_inputs_list.remove(input_field)

    def refresh_sd_groups(self):
        self.sd_group_combo.blockSignals(True)
        self.sd_group_combo.clear()
        groups = list(self.settings.get("sd_config_groups", {}).keys())
        self.sd_group_combo.addItems(groups)
        current = self.settings.get("current_sd_group")
        if current in groups:
            self.sd_group_combo.setCurrentText(current)
        elif groups:
            self.sd_group_combo.setCurrentText(groups[0])
        self.sd_group_combo.blockSignals(False)

    def load_settings_to_ui(self, group_name):
        settings = self.settings.get("sd_config_groups", {}).get(group_name, {})
        self.sd_url_input.setText(self.settings.get("sd_url", ""))
        self.sd_model_input.setText(settings.get("sd_model", ""))
        self.sampler_input.setText(settings.get("sampler", "Euler a"))
        self.scheduler_input.setText(settings.get("scheduler", "Automatic"))
        self.steps_input.setValue(settings.get("steps", 20))
        self.cfg_input.setValue(settings.get("cfg_scale", 7.0))

        for i in reversed(range(self.vae_inputs_container.count())):
            layout_item = self.vae_inputs_container.itemAt(i)
            if layout_item is None:
                continue
            widget = layout_item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.vae_inputs_list.clear()

        sd_vaes = settings.get("sd_vae", [])
        if isinstance(sd_vaes, str):
            sd_vaes = [sd_vaes] if sd_vaes and sd_vaes.lower() != "automatic" else []
        if not sd_vaes:
            self.add_vae_input_field("")
        else:
            for vae in sd_vaes:
                self.add_vae_input_field(vae)

    def on_sd_group_changed(self, group_name):
        if group_name:
            self.settings["current_sd_group"] = group_name
            self.load_settings_to_ui(group_name)

    def update_current_group_from_ui(self):
        group_name = self.sd_group_combo.currentText() or "Default"
        if group_name not in self.settings["sd_config_groups"]:
            self.settings["sd_config_groups"][group_name] = {}
        valid_vaes = [field.text().strip() for field in self.vae_inputs_list if field.text().strip()]
        self.settings["sd_url"] = self.sd_url_input.text().strip()
        self.settings["current_sd_group"] = group_name
        self.settings["sd_config_groups"][group_name] = {
            "sd_model": self.sd_model_input.text().strip(),
            "sd_vae": valid_vaes,
            "sampler": self.sampler_input.text().strip(),
            "scheduler": self.scheduler_input.text().strip(),
            "steps": self.steps_input.value(),
            "cfg_scale": self.cfg_input.value(),
        }

    def get_settings(self):
        self.update_current_group_from_ui()
        current = dict(self.settings)
        current["sd_config_groups"] = dict(self.settings.get("sd_config_groups", {}))
        return current

    def save_settings(self, silent=False):
        self.update_current_group_from_ui()
        save_sd_webui_settings(self.settings)
        if not silent:
            QMessageBox.information(self, "成功", "SD-WebUI 接口配置已保存。")
        return True

    def save_as_sd_group(self):
        self.update_current_group_from_ui()
        new_name, ok = QInputDialog.getText(self, "保存配置组", "请输入新配置组名称:")
        if ok and new_name.strip():
            group_name = new_name.strip()
            valid_vaes = [field.text().strip() for field in self.vae_inputs_list if field.text().strip()]
            self.settings["sd_config_groups"][group_name] = {
                "sd_model": self.sd_model_input.text().strip(),
                "sd_vae": valid_vaes,
                "sampler": self.sampler_input.text().strip(),
                "scheduler": self.scheduler_input.text().strip(),
                "steps": self.steps_input.value(),
                "cfg_scale": self.cfg_input.value(),
            }
            self.settings["current_sd_group"] = group_name
            self.refresh_sd_groups()
            self.sd_group_combo.setCurrentText(group_name)
            QMessageBox.information(self, "成功", f"配置组 '{group_name}' 已保存！")

    def delete_sd_group(self):
        group_name = self.sd_group_combo.currentText()
        if len(self.settings.get("sd_config_groups", {})) <= 1:
            QMessageBox.warning(self, "警告", "必须保留至少一个配置组！")
            return
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除配置组 '{group_name}' 吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            del self.settings["sd_config_groups"][group_name]
            self.settings["current_sd_group"] = list(self.settings["sd_config_groups"].keys())[0]
            self.refresh_sd_groups()
            self.load_settings_to_ui(self.settings["current_sd_group"])


class SdWorkflowWidget(QWidget):
    """SD 批量工作流容器：公共配置面板 + QTabWidget(主题批量/故事线) + 共享日志区。"""

    def __init__(
        self,
        text_config_getter_func=None,
        sd_webui_settings_getter_func=None,
        styles_getter_func=None,
        current_style_name_getter_func=None,
        style_changed_callback=None,
    ):
        super().__init__()
        self.setWindowTitle("AI 自动化绘画工作流")
        self.resize(1000, 900)

        os.makedirs(PROMPTS_DIR, exist_ok=True)
        os.makedirs(NEG_PROMPTS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(STORY_SEQUENCE_DIR, exist_ok=True)

        self.text_config_getter = text_config_getter_func or load_text_api_config_from_file
        self.sd_webui_settings_getter = sd_webui_settings_getter_func or load_sd_webui_settings
        self.styles_getter = styles_getter_func
        self.current_style_name_getter = current_style_name_getter_func
        self.style_changed_callback = style_changed_callback
        self.config = dict(DEFAULT_SD_WORKFLOW_STATE)
        self.worker = None
        self.tag_manager = TagAutocompleteManager()
        self._syncing = False

        self.load_config()
        self.init_ui()
        self._bind_attribute_references()
        self._connect_signals()
        self._load_config_to_sub_widgets()

    # ---------------------------------------------------------- 配置 IO
    def load_config(self):
        self.config.update(load_sd_workflow_state())
        if self.config.get("story_no_character_description", False):
            self.config["story_no_appearance_description"] = True
            self.config["story_no_outfit_description"] = True

    def save_config(self):
        save_sd_workflow_state(self.config)

    # ---------------------------------------------------------- UI 构建
    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # 公共配置面板（可折叠）
        self.common_panel = SdCommonConfigPanel(
            self.text_config_getter,
            self.sd_webui_settings_getter,
            parent=self,
        )
        main_layout.addWidget(self.common_panel)

        # 子 Tab 容器
        self.sub_tabs = QTabWidget()
        self.theme_batch_tab = SdThemeBatchTab(
            self.styles_getter,
            self.current_style_name_getter,
            self.tag_manager,
            parent=self,
        )
        self.storyline_tab = SdStorylineTab(
            self.styles_getter,
            self.current_style_name_getter,
            self._build_llm_runtime_context,
            self._is_render_running,
            parent=self,
        )
        self.sub_tabs.addTab(self.theme_batch_tab, "主题批量生成")
        self.sub_tabs.addTab(self.storyline_tab, "故事线生成")
        main_layout.addWidget(self.sub_tabs, stretch=1)

        # 共享日志区
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(160)
        main_layout.addWidget(self.log_area)

    def _bind_attribute_references(self):
        """把子组件控件引用直接挂到容器上，保证 widget.xxx 测试访问兼容。"""
        # 公共面板控件
        self.use_nsfw_text_api_cb = self.common_panel.use_nsfw_text_api_cb
        self.merge_prompt_cb = self.common_panel.merge_prompt_cb
        self.text_api_summary_label = self.common_panel.text_api_summary_label
        self.fixed_prompt_input = self.common_panel.fixed_prompt_input
        self.fixed_neg_prompt_input = self.common_panel.fixed_neg_prompt_input
        self.open_payload_editor_btn = self.common_panel.open_payload_editor_btn
        self.webui_extra_payload_summary_label = self.common_panel.webui_extra_payload_summary_label
        self.sd_webui_summary_label = self.common_panel.sd_webui_summary_label
        self.webui_extra_payload_text = self.common_panel.webui_extra_payload_text
        self.payload_editor_dialog = None  # 由 common_panel 管理
        # 主题批量 Tab 控件
        self.theme_input = self.theme_batch_tab.theme_input
        self.style_combo = self.theme_batch_tab.style_combo
        self.template_combo = self.theme_batch_tab.template_combo
        self.template_editor = self.theme_batch_tab.template_editor
        self.neg_template_combo = self.theme_batch_tab.neg_template_combo
        self.neg_template_editor = self.theme_batch_tab.neg_template_editor
        self.loop_count_input = self.theme_batch_tab.loop_count_input
        self.count_input = self.theme_batch_tab.count_input
        self.start_btn = self.theme_batch_tab.start_btn
        self.stop_btn = self.theme_batch_tab.stop_btn
        self.save_template_btn = self.theme_batch_tab.save_template_btn
        self.save_as_template_btn = self.theme_batch_tab.save_as_template_btn
        self.save_neg_template_btn = self.theme_batch_tab.save_neg_template_btn
        self.save_as_neg_template_btn = self.theme_batch_tab.save_as_neg_template_btn
        # 故事线 Tab 控件
        self.story_page_count_input = self.storyline_tab.story_page_count_input
        self.story_prompt_preset_combo = self.storyline_tab.story_prompt_preset_combo
        self.story_prompt_min_words_input = self.storyline_tab.story_prompt_min_words_input
        self.story_prompt_keyword_count_input = self.storyline_tab.story_prompt_keyword_count_input
        self.story_no_appearance_description_cb = self.storyline_tab.story_no_appearance_description_cb
        self.story_no_outfit_description_cb = self.storyline_tab.story_no_outfit_description_cb
        self.generate_story_btn = self.storyline_tab.generate_story_btn
        self.load_story_file_btn = self.storyline_tab.load_story_file_btn
        self.open_story_preview_btn = self.storyline_tab.open_story_preview_btn
        self.open_story_editor_btn = self.storyline_tab.open_story_editor_btn
        self.render_story_btn = self.storyline_tab.render_story_btn
        self.story_file_label = self.storyline_tab.story_file_label
        # 故事相关对话框引用（委托给 storyline_tab）
        self.story_generation_worker = None
        self.story_editor_dialog = None
        self.story_preview_dialog = None
        # autocomplete
        self.theme_batch_tab.setup_autocomplete()
        if self.tag_manager is not None:
            self.tag_manager.setup_line_edit(self.fixed_prompt_input)

    def _connect_signals(self):
        # 主题批量 Tab → 容器
        self.theme_batch_tab.start_requested.connect(self.start_workflow)
        self.theme_batch_tab.stop_requested.connect(self.stop_workflow)
        self.theme_batch_tab.style_changed.connect(self._on_style_changed)
        self.theme_batch_tab.theme_edited.connect(self._on_theme_edited)
        self.theme_batch_tab.config_changed.connect(self._on_sub_config_changed)
        # 故事线 Tab → 容器
        self.storyline_tab.start_render_requested.connect(self._on_start_render_requested)
        self.storyline_tab.style_changed.connect(self._on_style_changed)
        self.storyline_tab.theme_edited.connect(self._on_theme_edited)
        self.storyline_tab.config_changed.connect(self._on_sub_config_changed)
        self.storyline_tab.log_requested.connect(self.append_log)
        self.storyline_tab.clear_log_requested.connect(self.log_area.clear)
        self.storyline_tab.refresh_text_api_summary_requested.connect(self.refresh_text_api_summary)
        self.storyline_tab.story_file_changed.connect(self._on_story_file_changed)
        # 公共面板 → 容器
        self.common_panel.config_changed.connect(self._on_sub_config_changed)

    def _load_config_to_sub_widgets(self):
        self.common_panel.load_config_to_ui(self.config)
        self.theme_batch_tab.load_config_to_ui(self.config)
        self.storyline_tab.load_config_to_ui(self.config)
        self.refresh_text_api_summary()
        self.refresh_sd_webui_summary()

    # ---------------------------------------------------------- 配置聚合
    def update_config_from_ui(self):
        self.common_panel.update_config_from_ui(self.config)
        self.theme_batch_tab.update_config_from_ui(self.config)
        self.storyline_tab.update_config_from_ui(self.config)
        self.save_config()

    def _on_sub_config_changed(self):
        if self._syncing:
            return
        self.update_config_from_ui()

    # ---------------------------------------------------------- 跨 Tab 同步
    def _on_theme_edited(self, text):
        if self._syncing:
            return
        self._syncing = True
        try:
            # 把编辑方的 theme 同步到另一方
            if self.theme_batch_tab.theme_input.text() != text:
                self.theme_batch_tab.set_theme_value(text)
            if self.storyline_tab.theme_input.text() != text:
                self.storyline_tab.set_theme_value(text)
        finally:
            self._syncing = False
        self.update_config_from_ui()

    def _on_style_changed(self, name):
        if self._syncing:
            return
        if callable(self.style_changed_callback):
            self.style_changed_callback(name)
        self._syncing = True
        try:
            self.theme_batch_tab.set_style_value(name)
            self.storyline_tab.set_style_value(name)
        finally:
            self._syncing = False

    def sync_style_from_external(self, style_name):
        """app.py 用 blockSignals + setCurrentText 设置 style_combo 后，
        调用此方法把同一 style 同步到 storyline_tab 的 style_combo。"""
        self._syncing = True
        try:
            self.storyline_tab.set_style_value(style_name)
        finally:
            self._syncing = False

    def _on_story_file_changed(self, story_path):
        self.save_config()

    # ---------------------------------------------------------- 运行时上下文
    def _get_selected_style_prompt(self):
        return self.theme_batch_tab._get_selected_style_prompt()

    def _build_llm_runtime_context(self, require_template=True):
        theme = self.theme_input.text().strip()
        if not theme:
            raise ValueError("请填写绘画主题！")

        template_text = self.template_editor.toPlainText().strip()
        if require_template and not template_text:
            raise ValueError("正向模板内容不能为空！")

        use_nsfw = bool(self.use_nsfw_text_api_cb.isChecked())
        base_url, api_key, model = self.text_config_getter(use_nsfw)
        if not base_url or not api_key or not model:
            api_name = "文本分析（NSFW）" if use_nsfw else "文本分析 API"
            raise ValueError(f"请先在设置中补全 {api_name} 的 Base URL / API Key / Model。")

        self.update_config_from_ui()
        runtime_config = dict(self.config)
        runtime_config["base_url"] = base_url
        runtime_config["api_key"] = api_key
        runtime_config["model"] = model
        runtime_config["last_used_style_key"] = self.style_combo.currentText().strip()
        runtime_config["last_used_style"] = self._get_selected_style_prompt()
        return runtime_config, theme, template_text

    def _build_sd_runtime_context(self):
        extra_json = self.common_panel.validate_extra_payload()
        sd_webui_settings = self.sd_webui_settings_getter()
        self.update_config_from_ui()

        runtime_config = dict(self.config)
        runtime_config["last_used_style_key"] = self.style_combo.currentText().strip()
        runtime_config["last_used_style"] = self._get_selected_style_prompt()
        runtime_config["webui_extra_payload"] = extra_json
        runtime_config.update(sd_webui_settings)
        runtime_config["webui_extra_payload"] = extra_json

        neg_template_text = self.neg_template_editor.toPlainText().strip()
        return runtime_config, neg_template_text

    # ---------------------------------------------------------- worker 管理
    def _is_render_running(self):
        return bool(self.worker and self.worker.isRunning())

    def _set_render_controls(self, running):
        self.common_panel.set_running(running)
        self.theme_batch_tab.set_running(running)
        self.storyline_tab.set_running(running)

    def start_workflow(self):
        if self._is_render_running():
            QMessageBox.warning(self, "提示", "当前有生成任务正在运行，请先等待完成或手动停止。")
            return
        try:
            runtime_config_llm, theme, template_text = self._build_llm_runtime_context(require_template=True)
            runtime_config_sd, neg_template_text = self._build_sd_runtime_context()
        except ValueError as e:
            QMessageBox.warning(self, "警告", str(e))
            return

        runtime_config = dict(runtime_config_llm)
        runtime_config.update(runtime_config_sd)
        self.log_area.clear()
        self._set_render_controls(True)
        self.refresh_text_api_summary()
        self.refresh_sd_webui_summary()
        self.worker = SdWorkflowThread(runtime_config, theme, template_text, neg_template_text)
        self.worker.log_signal.connect(self.append_log)
        self.worker.finished_signal.connect(self.on_workflow_finished)
        self.worker.start()

    def stop_workflow(self):
        if self._is_render_running():
            self.append_log("收到停止指令，正在等待当前网络请求完成并安全退出...")
            self.worker.stop()
            self.stop_btn.setEnabled(False)

    def _on_start_render_requested(self, story_sequence, story_path):
        if self._is_render_running():
            QMessageBox.warning(self, "提示", "当前有生成任务正在运行，请先等待完成或手动停止。")
            return
        try:
            runtime_config, neg_template_text = self._build_sd_runtime_context()
        except ValueError as e:
            QMessageBox.warning(self, "JSON 格式错误", str(e))
            return

        self.log_area.clear()
        self._set_render_controls(True)
        self.refresh_text_api_summary()
        self.refresh_sd_webui_summary()
        self.worker = SdStoryRenderThread(
            runtime_config, story_sequence, neg_template_text, story_path=story_path
        )
        self.worker.log_signal.connect(self.append_log)
        self.worker.finished_signal.connect(self.on_workflow_finished)
        self.worker.start()

    def on_workflow_finished(self):
        self.worker = None
        self._set_render_controls(False)
        # 标记对应 Tab 的步骤为已完成
        if self.sub_tabs.currentWidget() is self.theme_batch_tab:
            self.theme_batch_tab.mark_workflow_done()
        else:
            self.storyline_tab.mark_render_done()

    # ---------------------------------------------------------- 委托方法
    def _set_story_file_path(self, story_path):
        self.storyline_tab._set_story_file_path(story_path)
        # 同步容器侧的 config 引用（storyline_tab 已写入 self.config）
        self.save_config()

    def sync_story_description_options(self):
        self.storyline_tab.sync_story_description_options()

    def choose_story_sequence_file(self):
        self.storyline_tab.choose_story_sequence_file()

    def update_styles(self, style_names=None, current_style_name=None):
        self.theme_batch_tab.update_styles(style_names, current_style_name)
        self.storyline_tab.update_styles(style_names, current_style_name)

    def refresh_text_api_summary(self):
        self.common_panel.refresh_text_api_summary()

    def refresh_sd_webui_summary(self):
        self.common_panel.refresh_sd_webui_summary()

    def append_log(self, text):
        timestamp = time.strftime("%H:%M:%S")
        self.log_area.append(f"[{timestamp}] {text}")
        scrollbar = self.log_area.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event):
        self.update_config_from_ui()
        super().closeEvent(event)
