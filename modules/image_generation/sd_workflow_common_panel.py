"""SD 批量工作流 - 公共配置面板

把原本散落在 SdWorkflowWidget 顶部（大模型接口来源）和底部（固定正反词 / Payload / SD 摘要）
的公共生效配置收拢到一个可折叠的面板里。
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from modules.image_generation.sd_workflow_core import (
    load_webui_extra_payload,
    dump_webui_extra_payload,
    WebuiExtraPayloadEditorDialog,
)


class SdCommonConfigPanel(QWidget):
    """公共配置面板（可折叠，默认展开）。

    包含：
    - LLM 接口来源（复用文本分析 API 配置 + NSFW 开关 + System Prompt 合并开关）
    - SD 运行时附加（固定正反词、WebUI 附加 Payload、SD-WebUI 配置摘要）
    """

    config_changed = pyqtSignal()

    def __init__(self, text_config_getter, sd_webui_settings_getter, parent=None):
        super().__init__(parent)
        self.text_config_getter = text_config_getter
        self.sd_webui_settings_getter = sd_webui_settings_getter
        self.webui_extra_payload_text = "{\n  \n}"
        self.payload_editor_dialog = None
        self._building_ui = False
        self.init_ui()

    # ------------------------------------------------------------------ UI
    def init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # 折叠标题栏
        self.toggle_btn = QToolButton()
        self.toggle_btn.setText("公共生效配置（LLM 接口来源 / SD 运行时附加）")
        self.toggle_btn.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(True)
        self.toggle_btn.setArrowType(Qt.ArrowType.DownArrow)
        self.toggle_btn.toggled.connect(self._on_toggle)
        outer.addWidget(self.toggle_btn)

        # 内容容器（默认展开）
        self.content_widget = QWidget()
        content_layout = QVBoxLayout(self.content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)

        content_layout.addWidget(self._build_llm_group())
        content_layout.addWidget(self._build_runtime_group())
        outer.addWidget(self.content_widget)

        self.content_widget.setVisible(True)

    def _on_toggle(self, checked):
        self.toggle_btn.setChecked(checked)
        self.content_widget.setVisible(checked)
        self.toggle_btn.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def _build_llm_group(self):
        group = QGroupBox("大模型接口来源")
        layout = QGridLayout()
        layout.addWidget(QLabel("直接复用“设置 -> 文本分析 API”中的配置。"), 0, 0, 1, 4)

        self.use_nsfw_text_api_cb = QCheckBox("使用文本分析（NSFW）配置")
        self.use_nsfw_text_api_cb.toggled.connect(self._on_control_changed)
        self.use_nsfw_text_api_cb.toggled.connect(self.refresh_text_api_summary)
        layout.addWidget(self.use_nsfw_text_api_cb, 1, 0, 1, 2)

        self.merge_prompt_cb = QCheckBox("启用 System Prompt 兼容模式 (合并到 User)")
        self.merge_prompt_cb.toggled.connect(self._on_control_changed)
        layout.addWidget(self.merge_prompt_cb, 1, 2, 1, 2)

        self.text_api_summary_label = QLabel()
        self.text_api_summary_label.setWordWrap(True)
        layout.addWidget(self.text_api_summary_label, 2, 0, 1, 4)

        group.setLayout(layout)
        return group

    def _build_runtime_group(self):
        group = QGroupBox("SD 运行时附加配置")
        layout = QFormLayout()

        self.fixed_prompt_input = QLineEdit()
        self.fixed_prompt_input.setPlaceholderText("自动拼接到大模型结果后")
        self.fixed_prompt_input.textEdited.connect(self._on_control_changed)
        layout.addRow("附加固定正向提示词:", self.fixed_prompt_input)

        self.fixed_neg_prompt_input = QLineEdit()
        self.fixed_neg_prompt_input.setPlaceholderText("自动拼接到最终反向提示词后")
        self.fixed_neg_prompt_input.textEdited.connect(self._on_control_changed)
        layout.addRow("附加固定反向提示词:", self.fixed_neg_prompt_input)

        payload_layout = QVBoxLayout()
        payload_btn_layout = QHBoxLayout()
        self.open_payload_editor_btn = QPushButton("打开 Payload 编辑窗口")
        self.open_payload_editor_btn.clicked.connect(self.open_webui_payload_editor)
        payload_btn_layout.addWidget(self.open_payload_editor_btn)
        payload_btn_layout.addStretch()
        payload_layout.addLayout(payload_btn_layout)
        self.webui_extra_payload_summary_label = QLabel()
        self.webui_extra_payload_summary_label.setWordWrap(True)
        payload_layout.addWidget(self.webui_extra_payload_summary_label)
        layout.addRow("WebUI 附加 Payload:", payload_layout)

        self.sd_webui_summary_label = QLabel()
        self.sd_webui_summary_label.setWordWrap(True)
        layout.addRow("SD-WebUI 配置:", self.sd_webui_summary_label)

        group.setLayout(layout)
        return group

    # ---------------------------------------------------------- 配置同步
    def load_config_to_ui(self, config):
        self._building_ui = True
        try:
            self.use_nsfw_text_api_cb.setChecked(bool(config.get("use_nsfw_text_api", False)))
            self.merge_prompt_cb.setChecked(bool(config.get("merge_system_prompt", True)))
            self.fixed_prompt_input.setText(str(config.get("fixed_prompt", "") or ""))
            self.fixed_neg_prompt_input.setText(str(config.get("fixed_negative_prompt", "") or ""))
            self.webui_extra_payload_text = str(
                config.get("webui_extra_payload", "{\n  \n}") or "{\n  \n}"
            )
        finally:
            self._building_ui = False
        self.refresh_text_api_summary()
        self.refresh_sd_webui_summary()
        self.refresh_webui_extra_payload_summary()

    def update_config_from_ui(self, config):
        config["merge_system_prompt"] = self.merge_prompt_cb.isChecked()
        config["use_nsfw_text_api"] = self.use_nsfw_text_api_cb.isChecked()
        config["fixed_prompt"] = self.fixed_prompt_input.text().strip()
        config["fixed_negative_prompt"] = self.fixed_neg_prompt_input.text().strip()
        config["webui_extra_payload"] = self.webui_extra_payload_text

    def _on_control_changed(self, *args):
        if self._building_ui:
            return
        self.config_changed.emit()

    # ---------------------------------------------------------- 摘要刷新
    def refresh_text_api_summary(self):
        use_nsfw = bool(self.use_nsfw_text_api_cb.isChecked())
        base_url, _api_key, model = self.text_config_getter(use_nsfw)
        api_name = "文本分析（NSFW）" if use_nsfw else "文本分析 API"
        url_text = base_url or "未配置"
        model_text = model or "未配置"
        self.text_api_summary_label.setText(
            f"当前来源: {api_name} | Base URL: {url_text} | Model: {model_text}"
        )

    def refresh_sd_webui_summary(self):
        settings = self.sd_webui_settings_getter()
        group_name = str(settings.get("current_sd_group", "Default") or "Default")
        sd_url = str(settings.get("sd_url", "") or "").strip() or "未配置"
        group = settings.get("sd_config_groups", {}).get(group_name, {})
        checkpoint = str(group.get("sd_model", "") or "").strip() or "沿用 WebUI 当前模型"
        sampler = str(group.get("sampler", "Euler a") or "Euler a")
        steps = group.get("steps", 20)
        cfg_scale = group.get("cfg_scale", 7.0)
        self.sd_webui_summary_label.setText(
            f"当前 WebUI 配置组: {group_name} | URL: {sd_url} | Checkpoint: {checkpoint} | "
            f"Sampler: {sampler} | Steps: {steps} | CFG: {cfg_scale}"
        )

    def refresh_webui_extra_payload_summary(self):
        try:
            payload = load_webui_extra_payload(self.webui_extra_payload_text)
        except Exception as e:
            self.webui_extra_payload_summary_label.setText(f"当前 Payload 无法解析: {e}")
            return

        keys = list(payload.keys())
        if not keys:
            self.webui_extra_payload_summary_label.setText("当前 Payload: 空对象 {}")
            return
        preview_keys = ", ".join(keys[:5])
        if len(keys) > 5:
            preview_keys += " ..."
        self.webui_extra_payload_summary_label.setText(
            f"当前 Payload 共 {len(keys)} 项 | Key: {preview_keys}"
        )

    # ---------------------------------------------------------- Payload 编辑
    def open_webui_payload_editor(self):
        if self.payload_editor_dialog is not None and self.payload_editor_dialog.isVisible():
            self.payload_editor_dialog.raise_()
            self.payload_editor_dialog.activateWindow()
            return
        dialog = WebuiExtraPayloadEditorDialog(self.webui_extra_payload_text, self)
        dialog.saved_signal.connect(self.on_webui_payload_saved)
        dialog.destroyed.connect(lambda _obj=None: setattr(self, "payload_editor_dialog", None))
        self.payload_editor_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def on_webui_payload_saved(self, payload_text, _payload_dict):
        self.webui_extra_payload_text = payload_text
        self.refresh_webui_extra_payload_summary()
        self.config_changed.emit()

    def validate_extra_payload(self):
        """校验并归一化 Payload 文本，返回归一化后的文本。失败抛 ValueError。"""
        try:
            payload = load_webui_extra_payload(self.webui_extra_payload_text)
        except Exception as e:
            raise ValueError(f"WebUI 附加 Payload JSON 格式错误: {e}") from e
        normalized_text = dump_webui_extra_payload(payload)
        self.webui_extra_payload_text = normalized_text
        self.refresh_webui_extra_payload_summary()
        return normalized_text

    # ---------------------------------------------------------- 运行态
    def set_running(self, running):
        self.open_payload_editor_btn.setEnabled(not running)
