import json
import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QComboBox, QMessageBox, QFileDialog,
    QDialog, QLineEdit, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QPixmap

from modules.image_analysis.single_analyzer import ImageGenWorkerThread
from modules.others.api_backend import fetch_llm_json, _extract_json_object, resolve_text_api_key
from utils.styles import (
    MODE_OFF,
    style_prompt, style_prompt_compressed, style_ref_image, ref_image_valid, build_style_entry,
    normalize_style_entry, assemble_style_instructions, save_styles_file, build_ref_gen_params,
)
from utils.style_gpt import is_gpt_image_api_type, style_prompt_gpt
from utils.style_ref_widget import StyleRefModeCombo


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_STYLES_FILE = os.path.join(BASE_DIR, "conf", "config-styles.json")
CONFIG_TEXT_FILE = os.path.join(BASE_DIR, "conf", "config.json")

COMPRESS_SYSTEM_PROMPT = (
    "You are an expert at condensing anime art-style specification texts for an image generation model, "
    "without losing any rule that affects the output.\n"
    "The user will paste the FULL style instruction. Produce a CONDENSED English style spec that:\n"
    "1. Keeps the overall artistic vibe, production modes, and every hard constraint "
    "(required framing, forbidden elements, fixed color/lighting rules).\n"
    "2. Keeps color & tonality rules, lighting/shadow treatment, line art and rendering conventions, detail density.\n"
    "3. Drops verbose prose, examples, repetitions and decorative phrasing.\n"
    "4. Target length: about 1/4 of the original, at most {max_chars} characters. "
    "If the original is already short, return it nearly unchanged.\n"
    "5. Respond ONLY with JSON: {{\"prompt_compressed\": \"<compressed text>\"}}"
)

class CompressPromptThread(QThread):
    """调用文本 LLM 接口把完整样式指令压缩为精简版（不阻塞 UI）。"""

    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, prompt_text, max_chars=700, parent=None):
        super().__init__(parent)
        self.prompt_text = prompt_text
        self.max_chars = max_chars

    def run(self):
        try:
            if not os.path.isfile(CONFIG_TEXT_FILE):
                self.failed.emit("未找到 conf/config.json，无法读取文本 API 配置")
                return
            with open(CONFIG_TEXT_FILE, "r", encoding="utf-8") as f:
                text_cfg = json.load(f)
            base_url = str(text_cfg.get("base_url") or "").strip()
            api_key = resolve_text_api_key(text_cfg)
            model = str(text_cfg.get("model") or "").strip()
            if not (base_url and api_key and model):
                self.failed.emit(
                    "缺少文本 API 配置：请在环境变量 IMAGE_MAKER_TEXT_API_KEY（.env）"
                    "或 conf/config.json 中配置 base_url / api_key / model"
                )
                return
            system_prompt = COMPRESS_SYSTEM_PROMPT.format(max_chars=self.max_chars)
            raw = fetch_llm_json(
                base_url, api_key, model,
                system_prompt, self.prompt_text,
                temperature=0.3, merge_system_prompt=False,
            )
            obj = _extract_json_object(raw)
            text = (obj.get("prompt_compressed") or obj.get("compressed") or "").strip()
            if not text:
                self.failed.emit("LLM 返回内容无法解析为压缩版指令，请重试")
                return
            self.finished_ok.emit(text)
        except Exception as e:
            self.failed.emit(f"压缩失败: {e}")

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


class GptDebugGenThread(QThread):
    """「单图调试生图」的 gpt-image-2 通道：首图（与图片分析 Tab 同一条组装链）+ 可选后处理。"""

    log_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(list)

    def __init__(self, generate, first_image, run_pipeline, request_kwargs, prompt, steps=None,
                 style_ref_path="", parent=None):
        super().__init__(parent)
        self._generate = generate
        self._first_image = first_image
        self._run_pipeline = run_pipeline
        self.request_kwargs = dict(request_kwargs or {})
        self.prompt = prompt
        self.steps = dict(steps or {})
        self.style_ref_path = style_ref_path or ""
        self._cancelled = False

    def request_cancel(self):
        self._cancelled = True

    def _log(self, message):
        self.log_signal.emit(str(message))

    def run(self):
        try:
            keywords = dict(self.request_kwargs)
            keywords.setdefault("return_metadata", False)
            files, effective_prompt = self._first_image(
                self._generate, prompt=self.prompt, log_callback=self._log, **keywords)
            if self._cancelled:
                self.finish_signal.emit([])
                return
            files = [path for path in (files or []) if os.path.isfile(path)]
            if not files:
                self._log("❌ gpt-image 首图没有产物")
                self.finish_signal.emit([])
                return
            current = files[0]
            self._log(f"✅ 首图完成: {os.path.basename(current)}")
            if self.steps and not self._cancelled:
                from utils.analysis_gen import run_gpt_image_pipeline
                self._log("▶ 后处理步骤: " + ", ".join(
                    key for key, value in self.steps.items() if (value or {}).get("enabled")))
                outputs = run_gpt_image_pipeline(
                    [current], self.steps, log_callback=self._log,
                    style_ref_path=self.style_ref_path or "", strict=False)
                outputs = [path for path in (outputs or []) if os.path.isfile(path)]
                if outputs:
                    current = outputs[-1]
                    self._log(f"✅ 后处理完成: {os.path.basename(current)}")
            self.finish_signal.emit([current])
        except Exception as exc:  # noqa: BLE001 —— 线程里必须兜住，否则 Qt 直接崩
            self._log(f"❌ gpt 通道失败: {exc}")
            self.finish_signal.emit([])


class SingleGenDebugWidget(QWidget):
    def __init__(self, img_config_getter_func, styles_getter_func, save_img_cfg_callback, ar_policy_getter_func=None, styles_reload_callback=None):
        super().__init__()
        self.get_img_config = img_config_getter_func
        self.get_styles = styles_getter_func
        self.save_img_cfg = save_img_cfg_callback
        self.get_ar_policy = ar_policy_getter_func
        self.styles_reload_callback = styles_reload_callback
        self.img_thread = None
        self.gpt_thread = None
        self._last_image_path = ""
        self.attach_image_paths = []
        self.style_ref_image_path = ""
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
        self.edit_style_btn = QPushButton("编辑样式...")
        self.edit_style_btn.setToolTip("打开样式编辑器：编辑当前画风预设的指令文本与参考图路径")
        self.edit_style_btn.clicked.connect(self.open_style_editor)
        style_layout.addWidget(self.edit_style_btn)
        layout.addLayout(style_layout)

        # 生图通道（2026-10-06）：与「图片分析」Tab 的两条链路对齐，方便横向对比
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("生图通道:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["Gemini", "gpt-image-2"])
        self.channel_combo.setToolTip(
            "Gemini：走 Gemini 直出（aspect_ratio + 画风说明 + 内容），与旧行为完全一致。\n"
            "gpt-image-2：与「图片分析」Tab 的 gpt 通道同一条组装链\n"
            "（build_first_pass_request：画风 prompt_gpt → 内容锚 → 参考图排除句 → 渲染条款），\n"
            "并可选跑重绘/色调/加墨后处理。提示词框作为内容锚使用。"
        )
        self.channel_combo.currentTextChanged.connect(self._on_channel_changed)
        channel_layout.addWidget(self.channel_combo)
        self.post_pipeline_check = QCheckBox("gpt 通道跑后处理（重绘+色调+加墨）")
        self.post_pipeline_check.setToolTip(
            "只对 gpt-image-2 通道生效：出首图后按「图片分析」Tab 的默认配方执行\n"
            "重绘(lines_only) → 色调校准(目标=画风参考图) → 线条加墨。\n全流程要 2~3 份联网调用。")
        self.post_pipeline_check.setChecked(True)
        channel_layout.addWidget(self.post_pipeline_check)
        channel_layout.addStretch(1)
        layout.addLayout(channel_layout)

        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("风格参考模式:"))
        self.style_mode_combo = StyleRefModeCombo()
        self.style_mode_combo.setToolTip(
            "关闭: 不参考样例图画风\n"
            "头部插入: 样式指令头部追加「艺术风格参考」指令，样例图作为附件\n"
            "参考优先: 样例图主导画风，样式指令压缩为精简版约束\n"
            "图文交错: 样式指令在前，样例图之后紧跟风格参考指令\n"
            "（当前无有效参考图时，参考类模式不可用）"
        )
        mode_layout.addWidget(self.style_mode_combo, stretch=1)
        layout.addLayout(mode_layout)

        style_ref_layout = QHBoxLayout()
        style_ref_layout.addWidget(QLabel("🎨 风格参考图(仅画风):"))
        self.select_style_ref_btn = QPushButton("手动指定")
        self.select_style_ref_btn.setToolTip(
            "手动指定一张样例图作为「艺术风格参考图」，优先级高于画风预设配置中的参考图。\n"
            "生成时只参考其画风（线条/上色/光影/配色/渲染惯例），不参考人物、服装、姿势、场景等内容。"
        )
        self.select_style_ref_btn.clicked.connect(self.select_style_ref_image)
        style_ref_layout.addWidget(self.select_style_ref_btn)
        self.clear_style_ref_btn = QPushButton("清除")
        self.clear_style_ref_btn.clicked.connect(self.clear_style_ref_image)
        style_ref_layout.addWidget(self.clear_style_ref_btn)
        self.style_ref_info_label = QLabel("未设置")
        self.style_ref_info_label.setStyleSheet("color: gray;")
        self.style_ref_info_label.setWordWrap(True)
        style_ref_layout.addWidget(self.style_ref_info_label, stretch=1)
        layout.addLayout(style_ref_layout)

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
        self.main_style_combo.currentTextChanged.connect(self._refresh_style_ref_info)
        self.prompt_edit.textChanged.connect(self._on_prompt_changed)
        self.aspect_ratio_combo.currentTextChanged.connect(self.save_ui_state)
        self.resolution_combo.currentTextChanged.connect(self.save_ui_state)
        self.style_mode_combo.currentIndexChanged.connect(self.save_ui_state)

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

            saved_style_ref = state.get("style_ref_image")
            if isinstance(saved_style_ref, str) and saved_style_ref:
                if os.path.exists(saved_style_ref):
                    self.style_ref_image_path = saved_style_ref
                    self._refresh_style_ref_info()
                    self._append_log(f"已恢复风格参考图: {saved_style_ref}")

            saved_channel = state.get("channel")
            if isinstance(saved_channel, str) and saved_channel:
                if self.channel_combo.findText(saved_channel) >= 0:
                    self.channel_combo.setCurrentText(saved_channel)
            self.post_pipeline_check.setChecked(bool(state.get("post_pipeline", True)))

            self._refresh_style_ref_info()
        finally:
            self._is_restoring_state = False
        self._on_channel_changed()

    def save_ui_state(self, *args):
        if self._is_restoring_state:
            return
        state = {
            "prompt": self.prompt_edit.toPlainText(),
            "main_style": self.main_style_combo.currentText().strip(),
            "aspect_ratio": self.aspect_ratio_combo.currentText().strip(),
            "resolution": self.resolution_combo.currentText().strip(),
            "attachments": list(self.attach_image_paths),
            "style_ref_image": self.style_ref_image_path,
            "channel": self.channel_combo.currentText().strip(),
            "post_pipeline": bool(self.post_pipeline_check.isChecked()),
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
        self._refresh_style_ref_info()

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
        return style_prompt(self.get_styles() or {}, style_name)

    def _effective_style_ref_image(self):
        """返回 (参考图路径, 来源)。手动指定优先，其次当前样式的配置参考图。"""
        if self.style_ref_image_path and ref_image_valid(self.style_ref_image_path):
            return self.style_ref_image_path, "手动"
        style_name = self.main_style_combo.currentText()
        cfg_ref = style_ref_image(self.get_styles() or {}, style_name)
        if ref_image_valid(cfg_ref):
            return cfg_ref, "样式配置"
        return "", ""

    def _update_mode_validity(self):
        """参考图无效时，禁用所有使用参考图的模式并回退到「关闭」。"""
        has_ref = ref_image_valid(self._effective_style_ref_image()[0])
        self.style_mode_combo.set_modes_available(has_ref)
        if not has_ref:
            self._append_log("⚠️ 当前没有有效参考图，已回退到「关闭」模式（参考类模式不可用）")

    def _refresh_style_ref_info(self):
        path, source = self._effective_style_ref_image()
        if path:
            self.style_ref_info_label.setText(f"{source}: {os.path.basename(path)}")
            self.style_ref_info_label.setStyleSheet("color: #228B22;")
        else:
            self.style_ref_info_label.setText("未设置（可在「编辑样式」中为该画风配置参考图）")
            self.style_ref_info_label.setStyleSheet("color: gray;")
        self._update_mode_validity()

    def select_style_ref_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "手动指定风格参考图（仅参考画风）",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.webp *.gif *.bmp)"
        )
        if not file_path:
            return
        self.style_ref_image_path = file_path
        self._refresh_style_ref_info()
        self._append_log(f"🎨 已手动指定风格参考图(仅画风): {file_path}")
        self.save_ui_state()

    def clear_style_ref_image(self):
        if not self.style_ref_image_path:
            return
        self.style_ref_image_path = ""
        self._refresh_style_ref_info()
        self._append_log("已清除手动指定参考图")
        self.save_ui_state()

    def open_style_editor(self):
        name = self.main_style_combo.currentText()
        if not name:
            return
        entry = normalize_style_entry((self.get_styles() or {}).get(name))
        dlg = StyleEditDialog(name, entry["prompt"], entry["ref_image"], entry["prompt_compressed"], parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        prompt, compressed, ref = dlg.get_values()
        if ref and not ref_image_valid(ref):
            QMessageBox.warning(self, "提示", f"参考图文件不存在：\n{ref}")
            return
        styles = dict(self.get_styles() or {})
        old = styles.get(name)
        merged = dict(old) if isinstance(old, dict) else {}
        merged.update(build_style_entry(
            prompt, ref, compressed, prompt_gpt=entry["prompt_gpt"], enabled=entry["enabled"]))
        if not ref:
            merged.pop("ref_image", None)
        if not compressed:
            merged.pop("prompt_compressed", None)
        styles[name] = merged
        try:
            save_styles_file(CONFIG_STYLES_FILE, styles)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存画风配置文件失败: {e}")
            return
        if self.styles_reload_callback:
            self.styles_reload_callback()
            if self.main_style_combo.findText(name) >= 0:
                self.main_style_combo.setCurrentText(name)
        self._append_log(f"💾 已保存样式 '{name}'（含参考图）")
        self._refresh_style_ref_info()

    def generate_image(self):
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "提示", "请先输入提示词。")
            return
        if self.channel_combo.currentText().strip().startswith("gpt"):
            self._generate_via_gpt_channel(prompt)
            return

        self.save_img_cfg()
        _img_url, img_key, model_name, api_type = self.get_img_config()
        if not img_key:
            QMessageBox.warning(self, "缺少配置", "生图 API Key 不能为空。")
            return

        aspect_ratio = self._resolve_aspect_ratio()
        resolution = self._resolve_resolution()
        style_name = self.main_style_combo.currentText()
        style_text = self._current_style_text()
        ref_path, ref_source = self._effective_style_ref_image()
        has_ref = ref_image_valid(ref_path)
        mode = self.style_mode_combo.effective_mode(has_ref)

        # 与「图片分析」Tab 的 Gemini 通道**同一套组装**（2026-10-06 对齐）：
        # build_ref_gen_params 负责画风说明/参考指令/参考图路径；参考优先模式再叠加
        # 「短内容锚 + 图后重申」，这两步以前只有分析 Tab 有，导致本页的图不可横向对比。
        from utils.analysis_gen import resolve_gemini_reference_content, build_gemini_content_postamble
        styles_data = self.get_styles() or {}
        instructions, post_instructions, ref_paths = build_ref_gen_params(
            styles_data, style_name, mode, api_type=api_type)
        style_entry = normalize_style_entry(styles_data.get(style_name, {}))
        # 内容锚（gemini_content_field）由 `resolve_gemini_reference_content` 从**分析产物 JSON 路径**读取，
        # 所以这里必须把已加载的 JSON 路径一起传进去（只传字段字典是解析不出锚的）。
        prompt_context = {"analysis_json_path": self.json_file_path or "",
                          "english_description": prompt, "short_description": prompt}
        if not self.json_file_path:
            self._append_log("未加载分析 JSON：画风若配置了内容锚，会退回用提示词框内容（可能触发不了锚）")
        content, anchor_field = resolve_gemini_reference_content(
            style_entry, prompt_context, prompt, mode)
        post_instructions = build_gemini_content_postamble(content, post_instructions)
        if anchor_field:
            self._append_log(f"内容锚：使用 {anchor_field}（{len(content)} 字符）")
        image_paths = list(self.attach_image_paths)
        for ref in ref_paths:
            if ref and ref not in image_paths:
                image_paths.insert(0, ref)
        if has_ref and mode != MODE_OFF and ref_path not in image_paths:
            image_paths.insert(0, ref_path)

        mode_label = self.style_mode_combo.itemText(self.style_mode_combo.findData(mode)) or "关闭"
        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status_label.setText("正在生成...")
        self.preview_label.setText("正在请求生图，请稍候...")
        self._append_log(
            f"开始生图: api={api_type}, model={model_name}, ar={aspect_ratio}, "
            f"resolution={resolution or '默认'}, attachments={len(image_paths)}, "
            f"模式={mode_label}, 参考图={'ON(' + ref_source + ')' if has_ref and mode != MODE_OFF else 'OFF'}"
        )

        self.img_thread = ImageGenWorkerThread(
            prompt=prompt,
            model_name=model_name,
            aspect_ratio=aspect_ratio,
            instructions=instructions,
            api_type=api_type,
            resolution=resolution,
            image_paths=image_paths,
            verbose_debug=True,
            post_instructions=post_instructions
        )
        self.img_thread.log_signal.connect(self._append_log)
        self.img_thread.finish_signal.connect(self.on_image_finished)
        self.img_thread.start()

    # ------------------------------------------------------------------ gpt-image-2 通道
    def _on_channel_changed(self, *args):
        is_gpt = self.channel_combo.currentText().strip().startswith("gpt")
        self.post_pipeline_check.setEnabled(is_gpt)
        self.prompt_edit.setPlaceholderText(
            "输入内容描述（作为内容锚）…" if is_gpt else "输入提示词…"
        )
        self.save_ui_state()

    def _generate_via_gpt_channel(self, prompt):
        """与「图片分析」Tab 的 gpt 通道同一条组装链（build_first_pass_request）。"""
        from modules.others.api_backend import generate_image_aigc2d_gpt
        from utils.analysis_gen import build_first_pass_request, pipeline_steps_from_flags
        from utils.first_image_review import generate_first_image
        from utils.post_process import run_pipeline

        self.save_img_cfg()
        _img_url, img_key, _model, api_type = self.get_img_config()
        if not img_key:
            QMessageBox.warning(self, "缺少配置", "生图 API Key 不能为空。")
            return
        gpt_api_type = api_type if api_type in ("aigc2d", "autodl") else "aigc2d"

        style_name = self.main_style_combo.currentText()
        ref_path, ref_source = self._effective_style_ref_image()
        has_ref = ref_image_valid(ref_path)
        mode = self.style_mode_combo.effective_mode(has_ref)
        # 与 gpt 通道一致：提示词框内容当作内容锚，tier=short（画风图要能说话）
        analysis_like = {"short_description": prompt, "english_description": prompt,
                         "gpt_image_prompt": "", "gpt_image_prompt_short": ""}
        size = ""
        ar = self._resolve_aspect_ratio()
        if ar and ":" in ar:
            w, h = ar.split(":", 1)
            try:
                size = "1536x1024" if float(w) > float(h) else ("1024x1536" if float(h) > float(w) else "1024x1024")
            except ValueError:
                size = ""
        payload = build_first_pass_request(
            self.get_styles() or {}, style_name, analysis_like,
            content_text=prompt, tier="short", size=size, api_type=gpt_api_type,
        )
        request_kwargs = dict(payload.get("request_kwargs") or {})
        style_entry = normalize_style_entry((self.get_styles() or {}).get(style_name, {}))
        style_ref_for_pipeline = style_entry.get("ref_image") or ""
        steps = {}
        if self.post_pipeline_check.isChecked():
            steps = pipeline_steps_from_flags(repaint=True, tone=True, ink=True,
                                              repaint_ref_mode="style",
                                              style_ref_path=style_ref_for_pipeline)
        enabled = [key for key, value in steps.items() if (value or {}).get("enabled")]
        self.generate_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status_label.setText("正在生成（gpt-image-2 通道）...")
        self.preview_label.setText("正在请求 gpt-image，请稍候...")
        self._append_log(
            f"开始生图: 通道=gpt-image-2, 模型={request_kwargs.get('model')}, size={size or '跟随比例'}, "
            f"模式={mode}, 参考图={'ON(' + ref_source + ')' if has_ref and mode != MODE_OFF else 'OFF'}, "
            f"后处理={enabled or '无'}"
        )

        self.gpt_thread = GptDebugGenThread(
            generate=lambda **kwargs: generate_image_aigc2d_gpt(**kwargs),
            first_image=generate_first_image,
            run_pipeline=run_pipeline,
            request_kwargs=request_kwargs,
            prompt=prompt,
            steps=steps,
            style_ref_path=style_ref_for_pipeline,
            parent=self,
        )
        self.gpt_thread.log_signal.connect(self._append_log)
        self.gpt_thread.finish_signal.connect(self.on_image_finished)
        self.gpt_thread.start()

    def cancel_generation(self):
        if self.img_thread and self.img_thread.isRunning():
            self.img_thread.request_cancel()
            self.status_label.setText("已请求终止，等待线程退出...")
        if getattr(self, "gpt_thread", None) and self.gpt_thread.isRunning():
            self.gpt_thread.request_cancel()
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


class DropLineEdit(QLineEdit):
    """支持直接拖拽图片文件到输入框，替换路径（用于参考图路径编辑）。"""

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(
            url.toLocalFile().lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"))
            for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path and path.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
                self.setText(path)
                event.acceptProposedAction()
                return
        event.ignore()


class StyleEditDialog(QDialog):
    """样式编辑器：编辑某个画风预设的指令文本、压缩版指令与参考图路径（config-styles.json 新格式）。"""

    def __init__(self, style_name, prompt_text, ref_image, prompt_compressed="", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"编辑样式: {style_name}")
        self.setMinimumSize(680, 560)
        layout = QVBoxLayout(self)

        name_label = QLabel(f"样式名: {style_name}")
        name_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(name_label)

        layout.addWidget(QLabel("指令文本:"))
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlainText(prompt_text)
        self.prompt_edit.setPlaceholderText("该画风的完整风格指令（生成时插入到请求文本头部）")
        layout.addWidget(self.prompt_edit)

        compressed_header = QHBoxLayout()
        compressed_header.addWidget(QLabel("压缩版指令(参考优先模式用):"))
        compressed_header.addStretch()
        self.compress_btn = QPushButton("请求 LLM 重新生成")
        self.compress_btn.setToolTip("调用文本 API（conf/config.json）把上面的完整指令压缩为精简版，填入本框")
        self.compress_btn.clicked.connect(self._regenerate_compressed)
        compressed_header.addWidget(self.compress_btn)
        layout.addLayout(compressed_header)

        self.compressed_edit = QTextEdit()
        self.compressed_edit.setPlainText(prompt_compressed)
        self.compressed_edit.setPlaceholderText(
            "参考优先模式下替代完整指令的精简版；可留空（留空时自动用本地启发式压缩）"
        )
        self.compressed_edit.setMaximumHeight(140)
        layout.addWidget(self.compressed_edit)

        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("参考图(仅画风):"))
        self.ref_edit = DropLineEdit(ref_image)
        self.ref_edit.setAcceptDrops(True)
        self.ref_edit.setPlaceholderText("可留空；样例图路径，生成时仅参考其画风（支持直接拖拽图片到此处）")
        ref_row.addWidget(self.ref_edit, stretch=1)
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse_ref)
        ref_row.addWidget(browse_btn)
        clear_btn = QPushButton("清除")
        clear_btn.clicked.connect(lambda: self.ref_edit.clear())
        ref_row.addWidget(clear_btn)
        layout.addLayout(ref_row)

        tip = QLabel(
            "提示：参考图仅用于提取画风（线条/上色/光影/配色/渲染惯例），不会改变主体与构图。\n"
            "留空或文件不存在时，「头部插入 / 参考优先 / 图文交错」模式将不可用。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: gray;")
        layout.addWidget(tip)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        save_btn = QPushButton("保存")
        save_btn.clicked.connect(self.accept)
        btn_row.addWidget(save_btn)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _browse_ref(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择参考图（仅参考画风）", "",
            "图片文件 (*.png *.jpg *.jpeg *.webp *.gif *.bmp)"
        )
        if file_path:
            self.ref_edit.setText(file_path)

    def _regenerate_compressed(self):
        prompt_text = self.prompt_edit.toPlainText().strip()
        if not prompt_text:
            QMessageBox.warning(self, "提示", "指令文本为空，无法生成压缩版。")
            return
        self.compress_btn.setEnabled(False)
        self.compress_btn.setText("生成中...")
        self._compress_thread = CompressPromptThread(prompt_text, parent=self)
        self._compress_thread.finished_ok.connect(self._on_compressed_ok)
        self._compress_thread.failed.connect(self._on_compressed_failed)
        self._compress_thread.finished.connect(self._on_compress_thread_done)
        self._compress_thread.start()

    def _on_compressed_ok(self, text):
        self.compressed_edit.setPlainText(text)
        QMessageBox.information(self, "完成", f"已生成压缩版指令（{len(text)} 字符）。")

    def _on_compressed_failed(self, err_msg):
        QMessageBox.warning(self, "生成失败", err_msg)

    def _on_compress_thread_done(self):
        self.compress_btn.setEnabled(True)
        self.compress_btn.setText("请求 LLM 重新生成")

    def get_values(self):
        return self.prompt_edit.toPlainText().strip(), self.compressed_edit.toPlainText().strip(), self.ref_edit.text().strip()
