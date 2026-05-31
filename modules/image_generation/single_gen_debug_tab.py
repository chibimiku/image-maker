import json
import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QComboBox, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap

from modules.image_analysis.single_analyzer import ImageGenWorkerThread


MANUAL_AR_OPTIONS = ["跟随全局策略", "1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2"]
MANUAL_RES_OPTIONS = ["默认(跟配置)", "1K", "2K", "4K"]
SINGLE_DEBUG_UI_STATE_FILE = "data/single_gen_debug_ui_state.json"

JSON_FIELD_OPTIONS = [
    ("english_description", "精修英文描述"),
    ("original_english_description", "原始英文描述"),
    ("short_description", "简短描述"),
]


class JsonDropLabel(QLabel):
    file_loaded = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setText("拖放分析结果 JSON 文件到此处\n或点击「加载JSON」按钮选择文件")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(52)
        self.setStyleSheet("QLabel { background-color: #f7f7f7; border: 1px dashed #aaa; color: #888; }")
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        for url in urls:
            path = url.toLocalFile()
            if path.lower().endswith('.json'):
                self.file_loaded.emit(path)
                return
        QMessageBox.warning(self, "格式错误", "请拖放 JSON 文件。")


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
        self.json_data = {}
        self.json_file_path = ""
        self._is_restoring_state = False
        self._last_synced_json_field_key = ""
        self.init_ui()
        self.load_ui_state()

    def init_ui(self):
        layout = QVBoxLayout()

        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("画风预设:"))
        self.main_style_combo = QComboBox()
        style_layout.addWidget(self.main_style_combo, stretch=1)
        style_layout.addStretch()
        layout.addLayout(style_layout)

        json_group_layout = QVBoxLayout()
        json_label_layout = QHBoxLayout()
        json_label_layout.addWidget(QLabel("📋 加载分析JSON:"))
        self.json_info_label = QLabel("(未加载)")
        self.json_info_label.setStyleSheet("color: gray;")
        json_label_layout.addWidget(self.json_info_label, stretch=1)
        self.json_load_btn = QPushButton("加载JSON")
        self.json_load_btn.clicked.connect(self.load_json_file)
        json_label_layout.addWidget(self.json_load_btn)
        self.json_apply_btn = QPushButton("填入Prompt")
        self.json_apply_btn.setEnabled(False)
        self.json_apply_btn.clicked.connect(self.apply_json_field)
        json_label_layout.addWidget(self.json_apply_btn)
        json_group_layout.addLayout(json_label_layout)

        json_field_layout = QHBoxLayout()
        json_field_layout.addWidget(QLabel("选择字段:"))
        self.json_field_combo = QComboBox()
        for key, label in JSON_FIELD_OPTIONS:
            self.json_field_combo.addItem(label, key)
        self.json_field_combo.currentIndexChanged.connect(self._on_json_field_changed)
        json_field_layout.addWidget(self.json_field_combo, stretch=1)
        json_group_layout.addLayout(json_field_layout)

        self.json_drop_label = JsonDropLabel()
        self.json_drop_label.file_loaded.connect(self._on_json_file_loaded)
        json_group_layout.addWidget(self.json_drop_label)

        layout.addLayout(json_group_layout)

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
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(360)
        self.preview_label.setStyleSheet("QLabel { background-color: #eee; border: 1px dashed #aaa; }")
        layout.addWidget(self.preview_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(120)
        layout.addWidget(self.log_text)

        self.main_style_combo.currentTextChanged.connect(self.save_ui_state)
        self.prompt_edit.textChanged.connect(self._on_prompt_changed)
        self.aspect_ratio_combo.currentTextChanged.connect(self.save_ui_state)
        self.resolution_combo.currentTextChanged.connect(self.save_ui_state)

        self.setLayout(layout)

    def _load_json_data(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.json_data = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "加载失败", f"无法解析 JSON 文件:\n{e}")
            return False
        self.json_file_path = file_path
        basename = os.path.basename(file_path)
        self.json_info_label.setText(f"已加载: {basename}")
        self.json_info_label.setStyleSheet("color: #228B22;")
        self.json_apply_btn.setEnabled(True)
        self.json_drop_label.setText(f"✅ {basename}")

        field_preview = self._get_selected_field_text()
        if field_preview:
            preview = field_preview[:80].replace('\n', ' ') + ('…' if len(field_preview) > 80 else '')
            self._append_log(f"JSON 已加载: {basename} | 当前字段预览: {preview}")
        else:
            self._append_log(f"JSON 已加载: {basename} | 当前字段内容为空")
        return True

    def _replace_prompt_with_selected_json_field(self, reason="sync"):
        text = self._get_selected_field_text()
        if not text:
            return False
        self.prompt_edit.setPlainText(text)
        field_label = self.json_field_combo.currentText()
        field_key = str(self.json_field_combo.currentData() or "")
        self._last_synced_json_field_key = field_key
        if reason == "field_changed":
            self._append_log(f"已切换到「{field_label}」，并自动刷新 Prompt（{len(text)} 字符）")
        else:
            self._append_log(f"已将「{field_label}」内容同步到 Prompt（{len(text)} 字符）")
        return True

    def _get_selected_field_text(self):
        key = self.json_field_combo.currentData()
        if not self.json_data or not key:
            return ""
        return str(self.json_data.get(key, "")).strip()

    def load_json_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择分析结果 JSON 文件", "",
            "JSON 文件 (*.json)"
        )
        if file_path:
            if self._load_json_data(file_path):
                self._replace_prompt_with_selected_json_field()

    def _on_json_file_loaded(self, file_path):
        if self._load_json_data(file_path):
            self._replace_prompt_with_selected_json_field()

    def _on_json_field_changed(self, _index):
        if not self.json_data:
            return
        self._replace_prompt_with_selected_json_field(reason="field_changed")

    def apply_json_field(self):
        text = self._get_selected_field_text()
        if not text:
            QMessageBox.information(self, "提示", "所选字段内容为空。")
            return
        current = self.prompt_edit.toPlainText().rstrip()
        if current:
            new_text = current + "\n\n" + text
        else:
            new_text = text
        self.prompt_edit.setPlainText(new_text)
        field_label = self.json_field_combo.currentText()
        self._append_log(f"已将「{field_label}」内容填入 Prompt（{len(text)} 字符）")

    def _on_prompt_changed(self):
        from PyQt6.QtCore import QTimer
        if hasattr(self, '_save_timer'):
            self._save_timer.stop()
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(500)
        self._save_timer.timeout.connect(self.save_ui_state)
        self._save_timer.start()

    def load_ui_state_data(self):
        if not os.path.exists(SINGLE_DEBUG_UI_STATE_FILE):
            return {}
        try:
            with open(SINGLE_DEBUG_UI_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def load_ui_state(self):
        state = self.load_ui_state_data()
        if not state:
            return
        self._is_restoring_state = True
        try:
            prompt_text = state.get("prompt")
            if isinstance(prompt_text, str) and prompt_text:
                self.prompt_edit.setPlainText(prompt_text)

            style_name = state.get("main_style")
            if isinstance(style_name, str) and style_name:
                if self.main_style_combo.findText(style_name) >= 0:
                    self.main_style_combo.setCurrentText(style_name)

            ar_text = state.get("aspect_ratio")
            if isinstance(ar_text, str) and ar_text:
                idx = self.aspect_ratio_combo.findText(ar_text)
                if idx >= 0:
                    self.aspect_ratio_combo.setCurrentText(ar_text)
                else:
                    self.aspect_ratio_combo.setEditText(ar_text)

            res_text = state.get("resolution")
            if isinstance(res_text, str) and res_text:
                idx = self.resolution_combo.findText(res_text)
                if idx >= 0:
                    self.resolution_combo.setCurrentText(res_text)
                else:
                    self.resolution_combo.setEditText(res_text)

            saved_attachments = state.get("attachments")
            if isinstance(saved_attachments, list):
                existing = [p for p in saved_attachments if isinstance(p, str) and os.path.exists(p)]
                if existing:
                    self.attach_image_paths = existing
                    self._refresh_attach_info()
                    self._append_log(f"已恢复 {len(existing)} 个附件路径 ({len(saved_attachments) - len(existing)} 个已失效)")
        finally:
            self._is_restoring_state = False

    def save_ui_state(self, *args):
        if self._is_restoring_state:
            return
        state = {
            "prompt": self.prompt_edit.toPlainText(),
            "main_style": self.main_style_combo.currentText().strip(),
            "aspect_ratio": self.aspect_ratio_combo.currentText().strip(),
            "resolution": self.resolution_combo.currentText().strip(),
            "attachments": list(self.attach_image_paths)
        }
        try:
            os.makedirs(os.path.dirname(SINGLE_DEBUG_UI_STATE_FILE), exist_ok=True)
            with open(SINGLE_DEBUG_UI_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=4)
        except Exception:
            pass

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
        self.save_ui_state()

    def clear_attachments(self):
        self.attach_image_paths = []
        self._refresh_attach_info()
        self._append_log("已清空附件")
        self.save_ui_state()

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
                pixmap.scaled(
                    self.preview_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            self.status_label.setText(f"生成成功: {self._last_image_path}")
            self._append_log(f"生成完成: {self._last_image_path}")
        else:
            self.preview_label.setText("生成失败或超时")
            self.status_label.setText("生成失败或超时")
            self._append_log("❌ 生成失败或未返回图片。请查看上方的「服务器原始返回」了解原因。")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_image_path:
            pixmap = QPixmap(self._last_image_path)
            if not pixmap.isNull():
                self.preview_label.setPixmap(
                    pixmap.scaled(
                        self.preview_label.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
