"""SD 批量工作流 - 故事线生成 Tab

按主题生成故事分镜 JSON，预览/编辑后顺序渲染的工作流，步骤从上到下：
1. 填写主题与风格
2. 配置故事参数
3. 生成/载入故事 JSON
4. 预览/编辑故事
5. 顺序渲染
"""

import os

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from modules.image_generation.sd_workflow_core import (
    BASE_DIR,
    STORY_SEQUENCE_DIR,
    STORY_PROMPT_PRESETS,
    SdStorySequenceThread,
    StorySequenceEditorDialog,
    StorySequencePreviewDialog,
    load_story_sequence,
)


# 步骤定义：(标题, 描述)
STORYLINE_STEPS = [
    ("步骤 1：填写主题与风格", "选择绘画主题与 Prompt 风格预设"),
    ("步骤 2：配置故事参数", "设置页数、Prompt 预设、关键词个数等"),
    ("步骤 3：生成/载入故事 JSON", "生成故事分镜或载入历史故事文件"),
    ("步骤 4：预览/编辑故事", "打开预览/编辑窗口校对每页提示词"),
    ("步骤 5：顺序渲染", "按当前故事顺序交给 SD-WebUI 出图"),
]


class SdStorylineTab(QWidget):
    """故事线生成 Tab。"""

    start_render_requested = pyqtSignal(object, str)  # (story_sequence, story_path)
    style_changed = pyqtSignal(str)
    theme_edited = pyqtSignal(str)
    config_changed = pyqtSignal()
    story_file_changed = pyqtSignal(str)
    log_requested = pyqtSignal(str)
    clear_log_requested = pyqtSignal()
    refresh_text_api_summary_requested = pyqtSignal()

    def __init__(
        self,
        styles_getter,
        current_style_name_getter,
        build_llm_runtime_context_func,
        is_render_running_func,
        parent=None,
    ):
        super().__init__(parent)
        self.styles_getter = styles_getter
        self.current_style_name_getter = current_style_name_getter
        self._build_llm_runtime_context_func = build_llm_runtime_context_func
        self._is_render_running_func = is_render_running_func
        self.style_options = {"默认(无附加)": ""}

        self.story_generation_worker = None
        self.story_editor_dialog = None
        self.story_preview_dialog = None

        self._story_viewed = False
        self._render_done = False
        self._render_running = False
        self._building_ui = False
        self._syncing = False
        self._syncing_story_prompt_preset = False
        self._syncing_story_description_options = False
        self._config_ref = None
        self.init_ui()

    # ------------------------------------------------------------------ UI
    def init_ui(self):
        layout = QVBoxLayout(self)

        # 步骤指示器
        self._step_group, self._step_labels = self._build_step_indicator()
        layout.addWidget(self._step_group)

        # 步骤 1：主题与风格
        layout.addWidget(self._build_theme_style_group())

        # 步骤 2：故事参数
        layout.addWidget(self._build_story_params_group())

        # 步骤 3：生成/载入故事
        layout.addWidget(self._build_story_source_group())

        # 步骤 4：预览/编辑
        layout.addWidget(self._build_story_view_group())

        # 步骤 5：渲染
        layout.addWidget(self._build_story_render_group())

        layout.addStretch()

    def _build_step_indicator(self):
        group = QGroupBox("工作流进度")
        v = QVBoxLayout()
        labels = []
        for title, desc in STORYLINE_STEPS:
            lbl = QLabel()
            lbl.setWordWrap(True)
            v.addWidget(lbl)
            labels.append(lbl)
        group.setLayout(v)
        return group, labels

    def _build_theme_style_group(self):
        group = QGroupBox("步骤 1：填写主题与风格")
        v = QVBoxLayout()
        hint = QLabel("主题与风格会与「主题批量生成」Tab 自动同步。")
        hint.setWordWrap(True)
        v.addWidget(hint)

        form = QHBoxLayout()
        form.addWidget(QLabel("绘画主题 (必填):"))
        self.theme_input = QLineEdit()
        self.theme_input.setPlaceholderText("例如：中秋主题少女、赛博朋克城市...")
        self.theme_input.textEdited.connect(self._on_theme_edited)
        form.addWidget(self.theme_input)
        form.addWidget(QLabel("风格预设:"))
        self.style_combo = QComboBox()
        self.style_combo.setMinimumWidth(200)
        self.style_combo.currentTextChanged.connect(self._on_style_changed)
        form.addWidget(self.style_combo)
        v.addLayout(form)

        group.setLayout(v)
        return group

    def _build_story_params_group(self):
        group = QGroupBox("步骤 2：配置故事参数")
        v = QVBoxLayout()

        param = QHBoxLayout()
        param.addWidget(QLabel("故事页数:"))
        self.story_page_count_input = QSpinBox()
        self.story_page_count_input.setRange(1, 999)
        param.addWidget(self.story_page_count_input)
        param.addWidget(QLabel("Prompt 预设:"))
        self.story_prompt_preset_combo = QComboBox()
        self.story_prompt_preset_combo.addItems(list(STORY_PROMPT_PRESETS.keys()) + ["自定义"])
        self.story_prompt_preset_combo.currentTextChanged.connect(self.on_story_prompt_preset_changed)
        param.addWidget(self.story_prompt_preset_combo)
        param.addWidget(QLabel("英文最少词数:"))
        self.story_prompt_min_words_input = QSpinBox()
        self.story_prompt_min_words_input.setRange(50, 5000)
        self.story_prompt_min_words_input.valueChanged.connect(self.on_story_prompt_values_changed)
        param.addWidget(self.story_prompt_min_words_input)
        param.addWidget(QLabel("关键词个数:"))
        self.story_prompt_keyword_count_input = QSpinBox()
        self.story_prompt_keyword_count_input.setRange(1, 200)
        self.story_prompt_keyword_count_input.valueChanged.connect(self.on_story_prompt_values_changed)
        param.addWidget(self.story_prompt_keyword_count_input)
        param.addStretch()
        v.addLayout(param)

        option = QHBoxLayout()
        self.story_no_appearance_description_cb = QCheckBox("不生成外观设定")
        self.story_no_appearance_description_cb.toggled.connect(self.on_story_no_appearance_toggled)
        option.addWidget(self.story_no_appearance_description_cb)
        self.story_no_outfit_description_cb = QCheckBox("不生成服装设定")
        self.story_no_outfit_description_cb.toggled.connect(self.on_story_no_outfit_toggled)
        option.addWidget(self.story_no_outfit_description_cb)
        option.addStretch()
        v.addLayout(option)

        group.setLayout(v)
        return group

    def _build_story_source_group(self):
        group = QGroupBox("步骤 3：生成/载入故事 JSON")
        v = QVBoxLayout()
        hint = QLabel("根据主题和页数直接生成故事分镜 JSON，不使用正向模板；支持载入历史故事文件。")
        hint.setWordWrap(True)
        v.addWidget(hint)

        btn = QHBoxLayout()
        self.generate_story_btn = QPushButton("生成故事序列 JSON")
        self.generate_story_btn.clicked.connect(self.generate_story_sequence)
        btn.addWidget(self.generate_story_btn)
        self.load_story_file_btn = QPushButton("载入历史故事 JSON")
        self.load_story_file_btn.clicked.connect(self.choose_story_sequence_file)
        btn.addWidget(self.load_story_file_btn)
        btn.addStretch()
        v.addLayout(btn)

        self.story_file_label = QLabel()
        self.story_file_label.setWordWrap(True)
        v.addWidget(self.story_file_label)

        group.setLayout(v)
        return group

    def _build_story_view_group(self):
        group = QGroupBox("步骤 4：预览/编辑故事")
        v = QVBoxLayout()
        btn = QHBoxLayout()
        self.open_story_preview_btn = QPushButton("打开故事预览窗口")
        self.open_story_preview_btn.clicked.connect(self.open_story_preview_dialog)
        btn.addWidget(self.open_story_preview_btn)
        self.open_story_editor_btn = QPushButton("打开故事编辑窗口")
        self.open_story_editor_btn.clicked.connect(self.open_story_sequence_editor)
        btn.addWidget(self.open_story_editor_btn)
        btn.addStretch()
        v.addLayout(btn)
        group.setLayout(v)
        return group

    def _build_story_render_group(self):
        group = QGroupBox("步骤 5：顺序渲染")
        v = QVBoxLayout()
        btn = QHBoxLayout()
        self.render_story_btn = QPushButton("按当前故事顺序生成")
        self.render_story_btn.setMinimumHeight(36)
        self.render_story_btn.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        self.render_story_btn.clicked.connect(self.start_story_render_from_current_file)
        btn.addWidget(self.render_story_btn)
        btn.addStretch()
        v.addLayout(btn)
        group.setLayout(v)
        return group

    # ---------------------------------------------------------- 配置同步
    def load_config_to_ui(self, config):
        self._config_ref = config
        self._building_ui = True
        try:
            self.theme_input.setText(str(config.get("last_used_theme", "中秋主题少女")))
            self.story_page_count_input.setValue(int(config.get("story_page_count", 6)))
            self.story_prompt_min_words_input.setValue(int(config.get("story_prompt_min_words", 250)))
            self.story_prompt_keyword_count_input.setValue(int(config.get("story_prompt_keyword_count", 30)))
            self.story_no_appearance_description_cb.setChecked(
                bool(config.get("story_no_appearance_description", False))
            )
            self.story_no_outfit_description_cb.setChecked(
                bool(config.get("story_no_outfit_description", False))
            )
        finally:
            self._building_ui = False
        self.apply_story_prompt_preset_from_config()
        self.sync_story_description_options()
        self.update_styles()
        self._refresh_story_file_label()
        self._sync_story_buttons_state()
        self._update_step_indicator()

    def update_config_from_ui(self, config):
        config["last_used_theme"] = self.theme_input.text().strip()
        config["story_page_count"] = self.story_page_count_input.value()
        config["story_prompt_preset"] = self.story_prompt_preset_combo.currentText()
        config["story_prompt_min_words"] = self.story_prompt_min_words_input.value()
        config["story_prompt_keyword_count"] = self.story_prompt_keyword_count_input.value()
        config["story_no_appearance_description"] = self.story_no_appearance_description_cb.isChecked()
        config["story_no_outfit_description"] = self.story_no_outfit_description_cb.isChecked()
        config["story_no_character_description"] = (
            self.story_no_appearance_description_cb.isChecked()
            and self.story_no_outfit_description_cb.isChecked()
        )
        selected_style_key = self.style_combo.currentText()
        config["last_used_style_key"] = selected_style_key
        config["last_used_style"] = self.style_options.get(selected_style_key, "")

    def _on_theme_edited(self, text):
        if self._syncing:
            return
        self._update_step_indicator()
        self.theme_edited.emit(text)
        self.config_changed.emit()

    def _on_style_changed(self, name):
        if self._syncing:
            return
        self.style_changed.emit(str(name or "").strip())

    # ---------------------------------------------------------- 跨 Tab 同步入口
    def set_theme_value(self, text):
        self._syncing = True
        try:
            self.theme_input.setText(text)
        finally:
            self._syncing = False
        self._update_step_indicator()

    def set_style_value(self, name):
        self._syncing = True
        try:
            self.style_combo.blockSignals(True)
            self.style_combo.setCurrentText(name)
            self.style_combo.blockSignals(False)
        finally:
            self._syncing = False

    # ---------------------------------------------------------- 风格下拉
    def _get_selected_style_prompt(self):
        selected_style_key = self.style_combo.currentText().strip()
        return str(self.style_options.get(selected_style_key, "") or "")

    def update_styles(self, style_names=None, current_style_name=None):
        # 复用 theme_batch_tab 的风格选项获取逻辑（读取同一份配置）
        if callable(self.styles_getter):
            try:
                loaded = self.styles_getter() or {}
                if isinstance(loaded, dict) and loaded:
                    self.style_options = dict(loaded)
            except Exception:
                pass
        if not self.style_options:
            self.style_options = {"默认(无附加)": ""}

        ordered = list(self.style_options.keys()) if not style_names else list(style_names)
        if "默认(无附加)" not in ordered:
            ordered.insert(0, "默认(无附加)")
            self.style_options.setdefault("默认(无附加)", "")

        selected_style_name = current_style_name
        if not selected_style_name and callable(self.current_style_name_getter):
            try:
                selected_style_name = str(self.current_style_name_getter() or "").strip()
            except Exception:
                selected_style_name = ""
        if not selected_style_name:
            selected_style_name = "默认(无附加)"
        if selected_style_name not in ordered:
            selected_style_name = "默认(无附加)"

        self._syncing = True
        try:
            self.style_combo.blockSignals(True)
            self.style_combo.clear()
            self.style_combo.addItems(ordered)
            self.style_combo.setCurrentText(selected_style_name)
            self.style_combo.blockSignals(False)
        finally:
            self._syncing = False

    # ---------------------------------------------------------- 故事 Prompt 预设
    def _match_story_prompt_preset(self, min_words, keyword_count):
        for preset_name, preset_config in STORY_PROMPT_PRESETS.items():
            if (
                int(preset_config["min_words"]) == int(min_words)
                and int(preset_config["keyword_count"]) == int(keyword_count)
            ):
                return preset_name
        return "自定义"

    def apply_story_prompt_preset_from_config(self):
        preset_name = str(self._config_ref.get("story_prompt_preset", "平衡") or "平衡") if self._config_ref else "平衡"
        if preset_name in STORY_PROMPT_PRESETS:
            preset_values = STORY_PROMPT_PRESETS[preset_name]
            self._syncing_story_prompt_preset = True
            self.story_prompt_min_words_input.setValue(int(preset_values["min_words"]))
            self.story_prompt_keyword_count_input.setValue(int(preset_values["keyword_count"]))
            self.story_prompt_preset_combo.setCurrentText(preset_name)
            self._syncing_story_prompt_preset = False
            return
        self.on_story_prompt_values_changed()

    def on_story_prompt_preset_changed(self, preset_name):
        if self._syncing_story_prompt_preset:
            return
        preset_name = str(preset_name or "").strip()
        if preset_name not in STORY_PROMPT_PRESETS:
            return
        preset_values = STORY_PROMPT_PRESETS[preset_name]
        self._syncing_story_prompt_preset = True
        self.story_prompt_min_words_input.setValue(int(preset_values["min_words"]))
        self.story_prompt_keyword_count_input.setValue(int(preset_values["keyword_count"]))
        self._syncing_story_prompt_preset = False
        self.config_changed.emit()

    def on_story_prompt_values_changed(self):
        if self._syncing_story_prompt_preset:
            return
        matched = self._match_story_prompt_preset(
            self.story_prompt_min_words_input.value(),
            self.story_prompt_keyword_count_input.value(),
        )
        self._syncing_story_prompt_preset = True
        self.story_prompt_preset_combo.setCurrentText(matched)
        self._syncing_story_prompt_preset = False
        self.config_changed.emit()

    def sync_story_description_options(self):
        if self._syncing_story_description_options:
            return
        self._syncing_story_description_options = True
        appearance_locked = self.story_no_appearance_description_cb.isChecked()
        if not appearance_locked and self.story_no_outfit_description_cb.isChecked():
            self.story_no_outfit_description_cb.setChecked(False)
        self.story_no_outfit_description_cb.setEnabled(appearance_locked)
        self._syncing_story_description_options = False

    def on_story_no_appearance_toggled(self, checked):
        if self._syncing_story_description_options:
            return
        if not checked and self.story_no_outfit_description_cb.isChecked():
            self._syncing_story_description_options = True
            self.story_no_outfit_description_cb.setChecked(False)
            self._syncing_story_description_options = False
        self.sync_story_description_options()
        self._update_step_indicator()
        self.config_changed.emit()

    def on_story_no_outfit_toggled(self, checked):
        if self._syncing_story_description_options:
            return
        if checked and not self.story_no_appearance_description_cb.isChecked():
            self._syncing_story_description_options = True
            self.story_no_appearance_description_cb.setChecked(True)
            self._syncing_story_description_options = False
        self.sync_story_description_options()
        self._update_step_indicator()
        self.config_changed.emit()

    # ---------------------------------------------------------- 故事文件管理
    def _refresh_story_file_label(self):
        story_path = str(self._config_ref.get("last_story_json_path", "") or "").strip() if self._config_ref else ""
        if story_path:
            exists_text = "存在" if os.path.isfile(story_path) else "不存在"
            self.story_file_label.setText(f"当前故事文件: {story_path} ({exists_text})")
        else:
            self.story_file_label.setText("当前故事文件: 未生成")

    def _get_current_story_path(self):
        if not self._config_ref:
            return ""
        return str(self._config_ref.get("last_story_json_path", "") or "").strip()

    def _has_current_story_file(self):
        story_path = self._get_current_story_path()
        return bool(story_path and os.path.isfile(story_path))

    def _sync_story_buttons_state(self):
        can_use_story = self._has_current_story_file()
        story_busy = self.story_generation_worker is not None
        render_busy = bool(self._is_render_running_func and self._is_render_running_func())
        allow_open = not render_busy
        self.generate_story_btn.setEnabled(not render_busy and not story_busy)
        self.load_story_file_btn.setEnabled(not render_busy and not story_busy)
        self.open_story_preview_btn.setEnabled(allow_open and can_use_story)
        self.open_story_editor_btn.setEnabled(allow_open and can_use_story)
        self.render_story_btn.setEnabled(allow_open and can_use_story)
        self._update_step_indicator()

    def _set_story_file_path(self, story_path):
        if self._config_ref is None:
            return
        self._config_ref["last_story_json_path"] = os.path.abspath(story_path) if story_path else ""
        self._refresh_story_file_label()
        self._sync_story_buttons_state()
        if story_path:
            self.story_file_changed.emit(story_path)
        if self.story_preview_dialog is not None and self.story_preview_dialog.isVisible() and story_path:
            self.story_preview_dialog.load_story(story_path)

    def choose_story_sequence_file(self):
        initial_dir = STORY_SEQUENCE_DIR if os.path.isdir(STORY_SEQUENCE_DIR) else BASE_DIR
        file_path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "选择故事序列 JSON",
            initial_dir,
            "JSON Files (*.json);;All Files (*.*)",
        )
        if not file_path:
            return
        try:
            load_story_sequence(file_path)
        except Exception as e:
            QMessageBox.warning(self, "文件错误", f"所选故事文件无法读取:\n{e}")
            return
        self._set_story_file_path(file_path)
        self.log_requested.emit(f"已载入历史故事文件: {file_path}")

    # ---------------------------------------------------------- 故事生成
    def generate_story_sequence(self):
        if self._is_render_running_func and self._is_render_running_func():
            QMessageBox.warning(self, "提示", "当前有生成任务正在运行，请先等待完成或手动停止。")
            return
        if self.story_generation_worker is not None:
            QMessageBox.information(self, "提示", "故事序列正在生成中，请稍候。")
            return
        try:
            runtime_config, theme, _template_text = self._build_llm_runtime_context_func(require_template=False)
        except ValueError as e:
            QMessageBox.warning(self, "警告", str(e))
            return

        self.clear_log_requested.emit()
        self.refresh_text_api_summary_requested.emit()
        self.story_generation_worker = SdStorySequenceThread(
            runtime_config,
            theme,
            self.story_page_count_input.value(),
            self._get_selected_style_prompt(),
            self.story_prompt_min_words_input.value(),
            self.story_prompt_keyword_count_input.value(),
            self.story_no_appearance_description_cb.isChecked(),
            self.story_no_outfit_description_cb.isChecked(),
        )
        self.story_generation_worker.log_signal.connect(self.log_requested.emit)
        self.story_generation_worker.success_signal.connect(self.on_story_sequence_generated)
        self.story_generation_worker.finished_signal.connect(self.on_story_sequence_generation_finished)
        self._sync_story_buttons_state()
        self.story_generation_worker.start()

    def on_story_sequence_generated(self, story_path, _story_data):
        self._set_story_file_path(story_path)
        self.open_story_sequence_editor(story_path)

    def on_story_sequence_saved(self, story_path, _story_data):
        self._set_story_file_path(story_path)

    def on_story_sequence_generation_finished(self):
        self.story_generation_worker = None
        self._sync_story_buttons_state()

    # ---------------------------------------------------------- 故事预览/编辑
    def open_story_preview_dialog(self):
        target_path = self._get_current_story_path()
        if not target_path:
            QMessageBox.warning(self, "提示", "当前还没有可预览的故事序列文件，请先生成。")
            return
        target_path = os.path.abspath(target_path)
        if self.story_preview_dialog is not None and self.story_preview_dialog.isVisible():
            self.story_preview_dialog.load_story(target_path)
            self.story_preview_dialog.raise_()
            self.story_preview_dialog.activateWindow()
            self._story_viewed = True
            self._update_step_indicator()
            return
        dialog = StorySequencePreviewDialog(target_path, self)
        dialog.destroyed.connect(lambda _obj=None: setattr(self, "story_preview_dialog", None))
        self.story_preview_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self._story_viewed = True
        self._update_step_indicator()

    def open_story_sequence_editor(self, story_path=None):
        target_path = str(story_path or self._get_current_story_path() or "").strip()
        if not target_path:
            QMessageBox.warning(self, "提示", "当前还没有可编辑的故事序列文件，请先生成。")
            return
        target_path = os.path.abspath(target_path)
        if not os.path.isfile(target_path):
            QMessageBox.warning(self, "提示", f"故事序列文件不存在:\n{target_path}")
            return

        if self.story_editor_dialog is not None and self.story_editor_dialog.isVisible():
            if os.path.abspath(self.story_editor_dialog.story_path) == target_path:
                self.story_editor_dialog.raise_()
                self.story_editor_dialog.activateWindow()
                self._story_viewed = True
                self._update_step_indicator()
                return

        dialog = StorySequenceEditorDialog(target_path, self)
        dialog.start_btn.clicked.connect(lambda: self.start_story_render_from_editor(dialog))
        dialog.saved_signal.connect(self.on_story_sequence_saved)
        dialog.destroyed.connect(lambda _obj=None: setattr(self, "story_editor_dialog", None))
        self.story_editor_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        self._story_viewed = True
        self._update_step_indicator()

    def start_story_render_from_editor(self, dialog):
        try:
            story_sequence = dialog.save_to_disk(show_message=True)
        except Exception as e:
            QMessageBox.warning(dialog, "JSON 错误", str(e))
            return
        self._set_story_file_path(dialog.story_path)
        dialog.close()
        self.start_story_render(story_sequence, dialog.story_path)

    def start_story_render(self, story_sequence, story_path=""):
        if self._is_render_running_func and self._is_render_running_func():
            QMessageBox.warning(self, "提示", "当前有生成任务正在运行，请先等待完成或手动停止。")
            return
        self.start_render_requested.emit(story_sequence, story_path)

    def start_story_render_from_current_file(self):
        story_path = self._get_current_story_path()
        if not story_path:
            QMessageBox.warning(self, "提示", "当前还没有可用的故事文件，请先生成或载入。")
            return
        if not os.path.isfile(story_path):
            QMessageBox.warning(self, "提示", f"故事文件不存在:\n{story_path}")
            self._sync_story_buttons_state()
            return
        try:
            story_sequence = load_story_sequence(story_path)
        except Exception as e:
            QMessageBox.warning(self, "文件错误", f"故事文件读取失败:\n{e}")
            return
        self.start_story_render(story_sequence, story_path)

    # ---------------------------------------------------------- 运行态 / 步骤
    def set_running(self, running):
        self._render_running = running
        if running:
            self._render_done = False
        self._sync_story_buttons_state()

    def mark_render_done(self):
        self._render_done = True
        self._render_running = False
        self._update_step_indicator()

    def _update_step_indicator(self):
        theme_ok = bool(self.theme_input.text().strip())
        story_file_ok = self._has_current_story_file()
        step_done = [
            theme_ok,
            theme_ok,  # 步骤 2：参数默认值即合法
            story_file_ok,
            self._story_viewed,
            self._render_done,
        ]
        current_idx = -1
        for i, done in enumerate(step_done):
            if not done:
                current_idx = i
                break
        for i, lbl in enumerate(self._step_labels):
            if self._render_running and i == 4:
                status = "in_progress"
            elif self.story_generation_worker is not None and i == 2:
                status = "in_progress"
            elif i == current_idx:
                status = "in_progress"
            elif step_done[i]:
                status = "done"
            else:
                status = "pending"
            self._apply_step_label(lbl, i, status)

    def _apply_step_label(self, lbl, idx, status):
        title, desc = STORYLINE_STEPS[idx]
        if status == "done":
            icon = "[√]"
            tag = "已完成"
            color = "#2e7d32"
            weight = "normal"
        elif status == "in_progress":
            icon = "[→]"
            tag = "进行中"
            color = "#1565c0"
            weight = "bold"
        else:
            icon = "[ ]"
            tag = "未开始"
            color = "#9e9e9e"
            weight = "normal"
        lbl.setText(
            f"<span style='color:{color}; font-weight:{weight};'>"
            f"{icon} <b>{title}</b>　<span style='font-size:small;'>{desc}</span>　[{tag}]"
            f"</span>"
        )
