"""Compact synchronized controls shared by GUI generation entry points."""
import weakref
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QCheckBox,
    QPushButton, QLineEdit, QMessageBox, QDialog, QDialogButtonBox, QSpinBox)
from utils.gpt_workflow_settings import load_settings, save_settings, dedicated_key, save_dedicated_key


class GptConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        from modules.others.api_backend import get_api_config
        self.setWindowTitle("GPT 出图独立配置")
        self.setMinimumWidth(420)
        state = load_settings()
        self._shared = get_api_config(api_type="aigc2d")
        self._own_url = state["base_url"] or get_api_config(api_type="aigc-2d-gpt").get("base_url", "")
        self._own_key = ""
        layout = QVBoxLayout(self)
        self.same_check = QCheckBox("和 Gemini 一样")
        self.same_check.setToolTip("使用 Gemini 的统一 URL 和 key；GPT 模型仍由生图界面选择。")
        layout.addWidget(self.same_check)
        form = QFormLayout()
        self.url_edit = QLineEdit(self._own_url)
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("专用 key 已保存，留空保持" if dedicated_key() else "请输入 GPT 专用 key")
        form.addRow("接口 URL", self.url_edit)
        form.addRow("API Key", self.key_edit)
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(30, 3600)
        self.timeout_spin.setSuffix(" 秒")
        self.timeout_spin.setValue(state.get("timeout_seconds", 600))
        self.timeout_spin.setToolTip("GPT 单次接口请求超时；界面另计入允许的重试和本地收尾余量。与 Gemini 超时独立。")
        form.addRow("GPT 请求超时", self.timeout_spin)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.same_check.toggled.connect(self._toggle)
        self.same_check.setChecked(state["same_as_gemini"])
        self._toggle(state["same_as_gemini"])

    def _toggle(self, shared):
        if shared:
            if self.url_edit.isEnabled():
                self._own_url, self._own_key = self.url_edit.text(), self.key_edit.text()
            self.url_edit.setText(str(self._shared.get("base_url") or ""))
            self.key_edit.clear()
            self.key_edit.setPlaceholderText("使用统一配置的 key")
        else:
            self.url_edit.setText(self._own_url)
            self.key_edit.setText(self._own_key)
            self.key_edit.setPlaceholderText("专用 key 已保存，留空保持" if dedicated_key() else "请输入 GPT 专用 key")
        self.url_edit.setEnabled(not shared)
        self.key_edit.setEnabled(not shared)

    def _save(self):
        try:
            shared = self.same_check.isChecked()
            changes = {"same_as_gemini": shared, "timeout_seconds": self.timeout_spin.value()}
            if not shared:
                from urllib.parse import urlparse
                url = self.url_edit.text().strip()
                if urlparse(url).scheme not in ("http", "https") or not urlparse(url).netloc:
                    raise ValueError("请输入有效的接口 URL")
                if self.key_edit.text().strip():
                    save_dedicated_key(self.key_edit.text())
                elif not dedicated_key():
                    raise ValueError("请填写 GPT 专用 key")
                changes["base_url"] = url
            save_settings(changes)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "GPT 配置", str(exc))
            return
        GptWorkflowControls.broadcast()
        self.accept()


class GptWorkflowControls(QWidget):
    changed = pyqtSignal()
    _instances = weakref.WeakSet()

    def __init__(self, parent=None, show_config=False):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.config_button = QPushButton("GPT配置")
        self.config_button.setMaximumWidth(75)
        self.post_check = QCheckBox("后续工序")
        self.post_check.setToolTip("默认关闭：直接交付 GPT 首图，跳过 Gemini 重绘、诊断、身份/人体/衣装/最终审计及本地后处理。开启后按各页面勾选执行。")
        row.addWidget(self.config_button)
        self.config_button.setVisible(show_config)
        row.addWidget(self.post_check)
        self._instances.add(self)
        self.refresh()
        self.post_check.toggled.connect(self._save)
        self.config_button.clicked.connect(lambda: GptConfigDialog(self).exec())

    def refresh(self):
        state = load_settings()
        self.post_check.blockSignals(True)
        self.post_check.setChecked(state["postprocess_enabled"])
        self.post_check.blockSignals(False)
        self.config_button.setToolTip("和 Gemini 一样：使用统一配置" if state["same_as_gemini"] else "独立 GPT 配置；key " + ("已保存" if dedicated_key() else "未配置"))

    @classmethod
    def broadcast(cls):
        for widget in list(cls._instances):
            try:
                widget.refresh()
                widget.changed.emit()
            except RuntimeError:
                cls._instances.discard(widget)

    def _save(self):
        try:
            save_settings({"postprocess_enabled": self.post_check.isChecked()})
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "GPT 工作流设置", str(exc))
            self.refresh()
            return
        self.broadcast()
