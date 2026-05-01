import os

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QComboBox, QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

from single_analyzer import ImageGenWorkerThread


MANUAL_AR_OPTIONS = ["跟随全局策略", "1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2"]
MANUAL_RES_OPTIONS = ["默认(跟配置)", "1K", "2K", "4K"]


class SingleGenDebugWidget(QWidget):
    def __init__(self, img_config_getter_func, styles_getter_func, save_img_cfg_callback, ar_policy_getter_func=None):
        super().__init__()
        self.get_img_config = img_config_getter_func
        self.get_styles = styles_getter_func
        self.save_img_cfg = save_img_cfg_callback
        self.get_ar_policy = ar_policy_getter_func
        self.img_thread = None
        self._last_image_path = ""
        self.attach_image_paths = []
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("画风预设:"))
        self.main_style_combo = QComboBox()
        style_layout.addWidget(self.main_style_combo, stretch=1)
        style_layout.addStretch()
        layout.addLayout(style_layout)

        prompt_label = QLabel("调试提示词:")
        layout.addWidget(prompt_label)
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("输入用于调试的生图文本...")
        self.prompt_edit.setMinimumHeight(120)
        layout.addWidget(self.prompt_edit)

        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("长宽比覆盖:"))
        self.aspect_ratio_combo = QComboBox()
        self.aspect_ratio_combo.setEditable(True)
        self.aspect_ratio_combo.addItems(MANUAL_AR_OPTIONS)
        self.aspect_ratio_combo.setCurrentText("跟随全局策略")
        control_layout.addWidget(self.aspect_ratio_combo)

        control_layout.addWidget(QLabel("分辨率覆盖:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.setEditable(True)
        self.resolution_combo.addItems(MANUAL_RES_OPTIONS)
        self.resolution_combo.setCurrentText("默认(跟配置)")
        control_layout.addWidget(self.resolution_combo)
        control_layout.addStretch()
        layout.addLayout(control_layout)

        attach_layout = QHBoxLayout()
        self.select_attach_btn = QPushButton("添加图片附件")
        self.select_attach_btn.clicked.connect(self.select_attachments)
        attach_layout.addWidget(self.select_attach_btn)
        self.clear_attach_btn = QPushButton("清空附件")
        self.clear_attach_btn.clicked.connect(self.clear_attachments)
        attach_layout.addWidget(self.clear_attach_btn)
        self.attach_info_label = QLabel("附件: 无")
        self.attach_info_label.setStyleSheet("color: gray;")
        attach_layout.addWidget(self.attach_info_label, stretch=1)
        layout.addLayout(attach_layout)
        self.attach_list_text = QTextEdit()
        self.attach_list_text.setReadOnly(True)
        self.attach_list_text.setMinimumHeight(72)
        self.attach_list_text.setPlaceholderText("附件列表（每行一个文件路径）")
        layout.addWidget(self.attach_list_text)

        btn_layout = QHBoxLayout()
        self.generate_btn = QPushButton("生成单图")
        self.generate_btn.clicked.connect(self.generate_image)
        btn_layout.addWidget(self.generate_btn)
        self.cancel_btn = QPushButton("终止")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_generation)
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.status_label = QLabel("准备就绪")
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

        self.preview_label = QLabel("暂无图片")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(360)
        self.preview_label.setStyleSheet("QLabel { background-color: #eee; border: 1px dashed #aaa; }")
        layout.addWidget(self.preview_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(120)
        layout.addWidget(self.log_text)

        self.setLayout(layout)

    def update_styles(self, style_keys):
        current = self.main_style_combo.currentText()
        self.main_style_combo.blockSignals(True)
        self.main_style_combo.clear()
        self.main_style_combo.addItems(style_keys)
        if current in style_keys:
            self.main_style_combo.setCurrentText(current)
        self.main_style_combo.blockSignals(False)

    def _resolve_aspect_ratio(self):
        selected = self.aspect_ratio_combo.currentText().strip()
        if not selected or selected == "跟随全局策略":
            policy = self.get_ar_policy() if self.get_ar_policy else {}
            override_second = str((policy or {}).get("override_second", "")).strip()
            if override_second and not override_second.startswith("不覆盖"):
                return override_second
            return str((policy or {}).get("default_aspect_ratio", "1:1")).strip() or "1:1"
        return selected

    def _resolve_resolution(self):
        selected = self.resolution_combo.currentText().strip()
        if not selected or selected == "默认(跟配置)":
            return None
        return selected

    def _append_log(self, text):
        self.log_text.append(text)

    def _refresh_attach_info(self):
        count = len(self.attach_image_paths)
        if count == 0:
            self.attach_info_label.setText("附件: 无")
            self.attach_list_text.setPlainText("")
            return
        if count == 1:
            self.attach_info_label.setText(f"附件: 1 张 ({os.path.basename(self.attach_image_paths[0])})")
        else:
            self.attach_info_label.setText(f"附件: {count} 张")
        self.attach_list_text.setPlainText("\n".join(self.attach_image_paths))

    def select_attachments(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择图片附件",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.webp *.gif *.bmp)"
        )
        if not files:
            return
        merged = list(self.attach_image_paths)
        for f in files:
            if f not in merged:
                merged.append(f)
        self.attach_image_paths = merged
        self._refresh_attach_info()
        self._append_log(f"已添加附件 {len(files)} 张，当前共 {len(self.attach_image_paths)} 张")

    def clear_attachments(self):
        self.attach_image_paths = []
        self._refresh_attach_info()
        self._append_log("已清空附件")

    def _current_style_text(self):
        style_name = self.main_style_combo.currentText()
        return (self.get_styles() or {}).get(style_name, "")

    def generate_image(self):
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "提示", "请先输入提示词。")
            return

        self.save_img_cfg()
        _img_url, img_key, model_name, api_type = self.get_img_config()
        if not img_key:
            QMessageBox.warning(self, "缺少配置", "生图 API Key 不能为空。")
            return

        aspect_ratio = self._resolve_aspect_ratio()
        resolution = self._resolve_resolution()
        instructions = self._current_style_text()

        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status_label.setText("正在生成...")
        self.preview_label.setText("正在请求生图，请稍候...")
        self._append_log(
            f"开始生图: api={api_type}, model={model_name}, ar={aspect_ratio}, "
            f"resolution={resolution or '默认'}, attachments={len(self.attach_image_paths)}"
        )

        self.img_thread = ImageGenWorkerThread(
            prompt=prompt,
            model_name=model_name,
            aspect_ratio=aspect_ratio,
            instructions=instructions,
            api_type=api_type,
            resolution=resolution,
            image_paths=self.attach_image_paths,
            verbose_debug=True
        )
        self.img_thread.log_signal.connect(self._append_log)
        self.img_thread.finish_signal.connect(self.on_image_finished)
        self.img_thread.start()

    def cancel_generation(self):
        if self.img_thread and self.img_thread.isRunning():
            self.img_thread.request_cancel()
            self.status_label.setText("已请求终止，等待线程退出...")

    def on_image_finished(self, saved_files):
        self.generate_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if saved_files and len(saved_files) > 0:
            self._last_image_path = saved_files[0]
            pixmap = QPixmap(self._last_image_path)
            self.preview_label.setPixmap(
                pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            self.status_label.setText(f"生成成功: {self._last_image_path}")
            self._append_log(f"生成完成: {self._last_image_path}")
        else:
            self.preview_label.setText("生成失败或超时")
            self.status_label.setText("生成失败或超时")
            self._append_log("生成失败或未返回图片。")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_image_path:
            pixmap = QPixmap(self._last_image_path)
            if not pixmap.isNull():
                self.preview_label.setPixmap(
                    pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
