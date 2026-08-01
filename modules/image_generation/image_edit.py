import os
import json
import hashlib
import re
import traceback
from functools import partial
from datetime import datetime
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QListWidget, QListWidgetItem, QFileDialog, QLabel, 
                             QTextEdit, QMessageBox, QComboBox, QSplitter, QProgressBar, QSpinBox, QApplication,
                             QAbstractItemView)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImageReader
from openai import OpenAI

# 复用已有的工具函数
from modules.image_analysis.single_analyzer import compress_and_encode_image, calculate_closest_aspect_ratio
from modules.others.api_backend import generate_image_whatai, generate_image_aigc2d
from utils.prompt_loader import PROMPTS_DIR
from utils.styles import style_ref_image, ref_image_valid, build_ref_gen_params
from utils.style_ref_widget import StyleRefModeCombo

PROMPT_DIR = os.path.join(PROMPTS_DIR, "image-edit")
IMAGE_EDIT_UI_STATE_FILE = "data/image_edit_ui_state.json"
MD5_TAIL_RE = re.compile(r"^[0-9a-fA-F]{32}$")
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def compute_file_md5(file_path, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)
    return md5.hexdigest()

class ImageEditWorker(QThread):
    result_ready = pyqtSignal(dict, str)  # result_json, image_path
    error = pyqtSignal(str, str)      # error_msg, image_path
    log = pyqtSignal(str)

    def __init__(self, image_path, prompt, img_config_snapshot, style_instructions="", style_ref_paths=None, post_instructions=""):
        super().__init__()
        self.image_path = image_path
        self.prompt = prompt
        self.img_config_snapshot = img_config_snapshot or ("", "", "", "")
        self.style_instructions = style_instructions
        self.style_ref_paths = list(style_ref_paths or [])
        self.post_instructions = post_instructions or ""

    def run(self):
        try:
            self.log.emit(f"开始处理图片: {os.path.basename(self.image_path)}")
            
            # 1. 获取配置
            _img_url, img_key, img_model, api_type = self.img_config_snapshot
            
            if not img_key:
                raise ValueError("请先配置生图 API Key")
            
            # 2. 调用生图接口
            self.log.emit("正在请求生成修改后的图片...")
            
            # 计算长宽比
            aspect_ratio = calculate_closest_aspect_ratio(self.image_path)
            self.log.emit(f"计算得到的长宽比: {aspect_ratio}")
            input_md5 = compute_file_md5(self.image_path)
            
            if api_type == "aigc2d":
                saved_files = generate_image_aigc2d(
                    prompt=self.prompt, 
                    image_paths=[self.image_path] + self.style_ref_paths,
                    model=img_model, 
                    aspect_ratio=aspect_ratio,
                    instructions=self.style_instructions,
                    api_type=api_type,
                    save_sub_dir="image-edit",
                    file_prefix=os.path.splitext(os.path.basename(self.image_path))[0],
                    cancel_check=lambda: self.isInterruptionRequested(),
                    post_instructions=self.post_instructions
                )
            else:
                saved_files = generate_image_whatai(
                    prompt=self.prompt, 
                    image_paths=[self.image_path] + self.style_ref_paths,
                    model=img_model, 
                    aspect_ratio=aspect_ratio,
                    instructions=self.style_instructions,
                    api_type=api_type,
                    save_sub_dir="image-edit",
                    file_prefix=os.path.splitext(os.path.basename(self.image_path))[0],
                    cancel_check=lambda: self.isInterruptionRequested(),
                    post_instructions=self.post_instructions
                )
            
            if self.isInterruptionRequested():
                self.log.emit("任务已被取消。")
                return

            if not saved_files:
                raise ValueError("生图接口未返回任何图片")
            
            # 组合最终提示词用于保存
            final_prompt = self.prompt
            if self.style_instructions:
                final_prompt = f"{self.style_instructions}, {self.prompt}"
                
            result_json = {
                "status": "success",
                "message": "图片编辑成功",
                "original_prompt": self.prompt,
                "final_prompt": final_prompt,
                "aspect_ratio": aspect_ratio,
                "input_image_md5": input_md5,
                "generated_images": saved_files
            }
            
            self.result_ready.emit(result_json, self.image_path)
            
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"{e}\n{tb}", self.image_path)

class ImageEditWidget(QWidget):
    def __init__(self, config_getter_func, img_config_getter_func, styles_getter_func):
        super().__init__()
        self.config_getter_func = config_getter_func
        self.img_config_getter_func = img_config_getter_func
        self.get_styles = styles_getter_func
        self.image_paths = []
        self.processing = False
        self.active_workers = {}
        self.results = {}  # 记录处理结果
        self._is_restoring_state = False
        self._is_initializing = True
        self._pending_main_style = ""
        self._pending_template = ""
        self._md5_cache = {}
        self._run_scope_paths = None
        
        # 确保提示词目录存在
        os.makedirs(PROMPT_DIR, exist_ok=True)
        
        self.initUI()
        self.load_prompt_templates()
        self.load_ui_state()
        self._is_initializing = False
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._on_app_about_to_quit)

    def initUI(self):
        layout = QVBoxLayout(self)
        self.setAcceptDrops(True)
        
        # 顶部：提示词模板管理和画风选择
        top_layout = QHBoxLayout()
        
        # 画风选择
        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("附加画风:"))
        self.main_style_combo = QComboBox()
        self.main_style_combo.setMaximumWidth(200)
        style_layout.addWidget(self.main_style_combo)
        style_layout.addWidget(QLabel("参考模式:"))
        self.style_ref_mode_combo = StyleRefModeCombo(self)
        self.style_ref_mode_combo.setMaximumWidth(130)
        style_layout.addWidget(self.style_ref_mode_combo)
        self.main_style_combo.currentTextChanged.connect(self._on_style_changed)
        style_layout.addWidget(QLabel("并发线程数:"))
        self.thread_spin = QSpinBox()
        self.thread_spin.setRange(1, 10)
        self.thread_spin.setValue(3)
        self.thread_spin.setMaximumWidth(80)
        style_layout.addWidget(self.thread_spin)
        top_layout.addLayout(style_layout)
        
        # 提示词模板
        template_layout = QHBoxLayout()
        template_layout.addWidget(QLabel("提示词模板:"))
        self.template_combo = QComboBox()
        self.template_combo.currentIndexChanged.connect(self.on_template_changed)
        template_layout.addWidget(self.template_combo)
        
        self.save_template_btn = QPushButton("保存为新模板")
        self.save_template_btn.clicked.connect(self.save_template)
        template_layout.addWidget(self.save_template_btn)
        self.save_current_template_btn = QPushButton("保存当前模板")
        self.save_current_template_btn.clicked.connect(self.save_current_template)
        template_layout.addWidget(self.save_current_template_btn)
        
        top_layout.addLayout(template_layout)
        layout.addLayout(top_layout)
        
        # 中间：分割器 (左侧图片列表，右侧提示词编辑和日志)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：图片列表
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("添加图片")
        self.add_btn.clicked.connect(self.add_images)
        self.clear_btn = QPushButton("清空列表")
        self.clear_btn.clicked.connect(self.clear_images)
        self.scan_missing_btn = QPushButton("扫描缺失(目录)")
        self.scan_missing_btn.clicked.connect(self.scan_missing_from_directory)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.scan_missing_btn)
        left_layout.addLayout(btn_layout)
        
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.image_list.itemChanged.connect(self._update_action_buttons)
        left_layout.addWidget(self.image_list)
        
        splitter.addWidget(left_widget)
        
        # 右侧：提示词编辑和日志
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        right_layout.addWidget(QLabel("编辑提示词 (Prompt):"))
        self.prompt_edit = QTextEdit()
        right_layout.addWidget(self.prompt_edit)
        
        right_layout.addWidget(QLabel("处理日志:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        right_layout.addWidget(self.log_text)
        
        splitter.addWidget(right_widget)
        
        # 设置分割器比例
        splitter.setSizes([300, 500])
        layout.addWidget(splitter)
        
        # 底部：进度条和控制按钮
        bottom_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        bottom_layout.addWidget(self.progress_bar)
        
        self.start_btn = QPushButton("开始任务")
        self.start_btn.clicked.connect(self.start_processing)
        bottom_layout.addWidget(self.start_btn)
        
        self.retry_btn = QPushButton("重试失败任务")
        self.retry_btn.clicked.connect(self.retry_failed)
        self.retry_btn.setEnabled(False)
        bottom_layout.addWidget(self.retry_btn)

        self.retry_selected_btn = QPushButton("重试勾选失败项")
        self.retry_selected_btn.clicked.connect(self.retry_selected_failed)
        self.retry_selected_btn.setEnabled(False)
        bottom_layout.addWidget(self.retry_selected_btn)
        
        layout.addLayout(bottom_layout)

        self.main_style_combo.currentTextChanged.connect(self.save_ui_state)
        self.template_combo.currentTextChanged.connect(self.save_ui_state)
        self.template_combo.currentTextChanged.connect(self._update_template_save_buttons)
        self.thread_spin.valueChanged.connect(self.save_ui_state)

    def load_prompt_templates(self):
        self.template_combo.clear()
        self.template_combo.addItem("自定义")
        
        if os.path.exists(PROMPT_DIR):
            for filename in os.listdir(PROMPT_DIR):
                if filename.endswith(".md"):
                    self.template_combo.addItem(filename[:-3])
        if self._pending_template:
            if self.template_combo.findText(self._pending_template) >= 0:
                self.template_combo.setCurrentText(self._pending_template)
            self._pending_template = ""
        self._update_template_save_buttons()

    def load_ui_state_data(self):
        if not os.path.exists(IMAGE_EDIT_UI_STATE_FILE):
            return {}
        try:
            with open(IMAGE_EDIT_UI_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            self.log_msg(f"读取批量编辑UI状态失败，已使用默认设置: {e}")
            return {}

    def load_ui_state(self):
        state = self.load_ui_state_data()
        if not state:
            return
        self._is_restoring_state = True
        try:
            thread_count = state.get("thread_count")
            if isinstance(thread_count, int):
                thread_count = max(self.thread_spin.minimum(), min(self.thread_spin.maximum(), thread_count))
                self.thread_spin.setValue(thread_count)

            template_name = state.get("template_name")
            if isinstance(template_name, str) and template_name:
                template_index = self.template_combo.findText(template_name)
                if template_index >= 0:
                    self.template_combo.setCurrentIndex(template_index)
                else:
                    self._pending_template = template_name

            style_name = state.get("main_style")
            if isinstance(style_name, str) and style_name:
                style_index = self.main_style_combo.findText(style_name)
                if style_index >= 0:
                    self.main_style_combo.setCurrentIndex(style_index)
                else:
                    self._pending_main_style = style_name
        finally:
            self._is_restoring_state = False

    def save_ui_state(self):
        if self._is_restoring_state or self._is_initializing:
            return
        state = {
            "thread_count": int(self.thread_spin.value()),
            "template_name": self.template_combo.currentText().strip(),
            "main_style": self.main_style_combo.currentText().strip()
        }
        try:
            os.makedirs(os.path.dirname(IMAGE_EDIT_UI_STATE_FILE), exist_ok=True)
            with open(IMAGE_EDIT_UI_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=4)
        except Exception as e:
            self.log_msg(f"保存批量编辑UI状态失败: {e}")

    def on_template_changed(self, index):
        if index <= 0:
            self._update_template_save_buttons()
            return
            
        template_name = self.template_combo.currentText()
        filepath = os.path.join(PROMPT_DIR, f"{template_name}.md")
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.prompt_edit.setPlainText(content)
            except Exception as e:
                self.log_msg(f"加载模板失败: {e}")
        self._update_template_save_buttons()

    def save_template(self):
        content = self.prompt_edit.toPlainText().strip()
        if not content:
            QMessageBox.warning(self, "警告", "提示词不能为空")
            return
            
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "保存模板", "请输入模板名称:")
        if ok and name:
            filepath = os.path.join(PROMPT_DIR, f"{name}.md")
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.load_prompt_templates()
                self.template_combo.setCurrentText(name)
                self.log_msg(f"模板已保存: {name}")
            except Exception as e:
                self.log_msg(f"保存模板失败: {e}")

    def _update_template_save_buttons(self):
        template_name = self.template_combo.currentText().strip()
        can_save_current = bool(template_name and template_name != "自定义")
        self.save_current_template_btn.setEnabled(can_save_current and not self.processing)
        self.save_template_btn.setEnabled(not self.processing)

    def save_current_template(self):
        template_name = self.template_combo.currentText().strip()
        if not template_name or template_name == "自定义":
            QMessageBox.information(self, "提示", "当前为“自定义”，请使用“保存为新模板”。")
            return

        content = self.prompt_edit.toPlainText().strip()
        if not content:
            QMessageBox.warning(self, "警告", "提示词不能为空")
            return

        filepath = os.path.join(PROMPT_DIR, f"{template_name}.md")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            self.log_msg(f"已保存当前模板: {template_name}")
        except Exception as e:
            self.log_msg(f"保存当前模板失败: {e}")

    def add_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if files:
            added_count, _ = self._add_images_from_paths(files)
            self.log_msg(f"已添加 {added_count} 张图片。")
            self._update_action_buttons()

    def _is_supported_image_file(self, file_path):
        _, ext = os.path.splitext(file_path)
        return ext.lower() in SUPPORTED_IMAGE_EXTENSIONS

    def _add_images_from_paths(self, paths):
        added_count = 0
        skipped_count = 0
        for file in paths:
            norm_path = os.path.normpath(str(file))
            if not os.path.isfile(norm_path):
                skipped_count += 1
                continue
            if not self._is_supported_image_file(norm_path):
                skipped_count += 1
                continue
            if norm_path in self.image_paths:
                skipped_count += 1
                continue
            self.image_paths.append(norm_path)
            item = QListWidgetItem(os.path.basename(norm_path))
            item.setData(Qt.ItemDataRole.UserRole, norm_path)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.image_list.addItem(item)
            self._cache_md5(norm_path)
            added_count += 1
        if added_count:
            self.update_progress()
        return added_count, skipped_count

    def _collect_image_paths_from_drop(self, mime_data):
        if not mime_data or not mime_data.hasUrls():
            return []
        collected = []
        for url in mime_data.urls():
            if not url.isLocalFile():
                continue
            local_path = os.path.normpath(url.toLocalFile())
            if os.path.isdir(local_path):
                for root, _dirs, files in os.walk(local_path):
                    for filename in files:
                        candidate = os.path.join(root, filename)
                        if self._is_supported_image_file(candidate):
                            collected.append(candidate)
            elif os.path.isfile(local_path) and self._is_supported_image_file(local_path):
                collected.append(local_path)
        # 去重并保持原顺序
        return list(dict.fromkeys(collected))

    def dragEnterEvent(self, event):
        if self._collect_image_paths_from_drop(event.mimeData()):
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event):
        if self._collect_image_paths_from_drop(event.mimeData()):
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event):
        dropped_paths = self._collect_image_paths_from_drop(event.mimeData())
        if not dropped_paths:
            event.ignore()
            return
        added_count, skipped_count = self._add_images_from_paths(dropped_paths)
        self.log_msg(f"拖拽添加完成：新增 {added_count} 张，跳过 {skipped_count} 项。")
        event.acceptProposedAction()

    def clear_images(self):
        if self.processing:
            QMessageBox.warning(self, "提示", "任务执行中，暂不支持清空列表")
            return
        self.image_paths.clear()
        self.image_list.clear()
        self.results.clear()
        self.active_workers.clear()
        self._run_scope_paths = None
        self._md5_cache.clear()
        self.update_progress()
        self.log_text.clear()
        self._update_action_buttons()

    def _cache_md5(self, image_path):
        if image_path in self._md5_cache:
            return self._md5_cache[image_path]
        try:
            value = compute_file_md5(image_path)
            self._md5_cache[image_path] = value
            return value
        except Exception as e:
            self.log_msg(f"计算MD5失败: {os.path.basename(image_path)} - {e}")
            return ""

    def scan_missing_from_directory(self):
        if not self.image_paths:
            QMessageBox.warning(self, "警告", "请先添加图片")
            return
        target_dir = QFileDialog.getExistingDirectory(self, "选择用于比对的目录", "")
        if not target_dir:
            return

        # 仅把图片文件视为“已生成结果”，避免 json/txt 造成误判
        image_extensions = set()
        for fmt in QImageReader.supportedImageFormats():
            try:
                ext = bytes(fmt).decode("ascii").lower()
            except Exception:
                ext = str(fmt).lower()
            if ext:
                image_extensions.add(f".{ext}")
        if not image_extensions:
            image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

        found_md5 = set()
        json_md5_seen = set()
        json_md5_has_existing_image = {}
        workspace_root = os.path.abspath(".")
        for root, _dirs, files in os.walk(target_dir):
            for filename in files:
                _, ext = os.path.splitext(filename)
                ext_lower = ext.lower()
                if ext_lower in image_extensions:
                    stem = os.path.splitext(filename)[0]
                    last_seg = stem.split("-")[-1].strip().lower()
                    if MD5_TAIL_RE.match(last_seg):
                        found_md5.add(last_seg)
                    continue

                if ext_lower != ".json":
                    continue

                json_path = os.path.join(root, filename)
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        result_obj = json.load(f)
                except Exception:
                    continue

                input_md5 = str(result_obj.get("input_image_md5", "")).strip().lower()
                if not MD5_TAIL_RE.match(input_md5):
                    continue

                json_md5_seen.add(input_md5)
                if input_md5 not in json_md5_has_existing_image:
                    json_md5_has_existing_image[input_md5] = False

                generated_images = result_obj.get("generated_images", []) or []
                for out_path in generated_images:
                    out_path = str(out_path).strip()
                    if not out_path:
                        continue
                    if os.path.isabs(out_path):
                        check_path = out_path
                    else:
                        check_path = os.path.normpath(os.path.join(workspace_root, out_path))
                    if os.path.exists(check_path):
                        json_md5_has_existing_image[input_md5] = True
                        break

        broken_json_md5 = {
            md5 for md5 in json_md5_seen
            if not json_md5_has_existing_image.get(md5, False)
        }

        missing_count = 0
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            image_path = item.data(Qt.ItemDataRole.UserRole)
            image_md5 = self._cache_md5(image_path).lower()
            is_missing_by_files = image_md5 and image_md5 not in found_md5
            is_missing_by_json = image_md5 and image_md5 in broken_json_md5
            if is_missing_by_files or is_missing_by_json:
                item.setCheckState(Qt.CheckState.Checked)
                missing_count += 1
            else:
                item.setCheckState(Qt.CheckState.Unchecked)

        total = self.image_list.count()
        self.log_msg(
            f"目录扫描完成：共发现 {len(found_md5)} 个输出图片md5，"
            f"发现 {len(broken_json_md5)} 个json缺图md5，待处理 {total} 项，缺失 {missing_count} 项（已自动勾选）。"
        )
        self._update_action_buttons()

    def log_msg(self, msg):
        self.log_text.append(msg)
        # 滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_progress(self):
        if self._run_scope_paths is None:
            scoped_paths = set(self.image_paths)
        else:
            scoped_paths = set(self._run_scope_paths)

        total = len(scoped_paths)
        if total == 0:
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("0/0")
            return

        processed = sum(1 for path in scoped_paths if path in self.results)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(processed)
        self.progress_bar.setFormat(f"{processed}/{total}")

    def _start_processing_with_scope(self, scope_paths=None):
        if not self.image_paths:
            QMessageBox.warning(self, "警告", "请先添加图片")
            return
            
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "警告", "请输入提示词")
            return

        if scope_paths is None:
            self._run_scope_paths = None
        else:
            normalized_scope = {p for p in scope_paths if p in self.image_paths}
            if not normalized_scope:
                self.log_msg("没有可执行的勾选任务")
                return
            self._run_scope_paths = normalized_scope
            
        self.processing = True
        self.update_progress()
        self.start_btn.setEnabled(False)
        self.add_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.retry_btn.setEnabled(False)
        self.retry_selected_btn.setEnabled(False)
        self.template_combo.setEnabled(False)
        self.save_template_btn.setEnabled(False)
        self.save_current_template_btn.setEnabled(False)
        self.prompt_edit.setEnabled(False)
        self.thread_spin.setEnabled(False)
        
        self.log_msg("开始批量处理任务...")
        self.launch_available_workers()

    def start_processing(self):
        self._start_processing_with_scope(None)

    def retry_failed(self):
        # 直接重试所有红色失败任务
        failed_paths = [path for path, status in self.results.items() if status == "error"]
        if not failed_paths:
            self.log_msg("没有失败的任务需要重试")
            return
            
        for path in failed_paths:
            if path in self.results:
                del self.results[path]
                
        self._start_processing_with_scope(set(failed_paths))

    def retry_selected_failed(self):
        checked_paths = []
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item.checkState() != Qt.CheckState.Checked:
                continue
            path = item.data(Qt.ItemDataRole.UserRole)
            checked_paths.append(path)

        if not checked_paths:
            self.log_msg("没有勾选任务可执行")
            return

        for path in checked_paths:
            if path in self.results:
                del self.results[path]
            self.update_item_status(path, Qt.GlobalColor.white)
        self.log_msg(f"将执行 {len(checked_paths)} 个勾选任务")
        self._start_processing_with_scope(set(checked_paths))

    def _get_next_unprocessed_path(self):
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            if self._run_scope_paths is not None and path not in self._run_scope_paths:
                continue
            if path in self.results or path in self.active_workers:
                continue
            return path
        return None

    def _update_action_buttons(self):
        if self.processing:
            return
        has_errors = any(status == "error" for status in self.results.values())
        has_checked = any(
            self.image_list.item(i).checkState() == Qt.CheckState.Checked
            for i in range(self.image_list.count())
        )
        self.retry_btn.setEnabled(has_errors)
        self.retry_selected_btn.setEnabled(has_checked)

    def launch_available_workers(self):
        if not self.processing:
            return

        try:
            max_workers = int(self.thread_spin.value())
            while len(self.active_workers) < max_workers:
                next_path = self._get_next_unprocessed_path()
                if not next_path:
                    break

                # 更新UI状态
                for i in range(self.image_list.count()):
                    item = self.image_list.item(i)
                    if item.data(Qt.ItemDataRole.UserRole) == next_path:
                        item.setBackground(Qt.GlobalColor.yellow)
                        self.image_list.scrollToItem(item)
                        break

                prompt = self.prompt_edit.toPlainText().strip()
                # 获取选中的画风指令（含参考图模式组装）
                selected_style_name = self.main_style_combo.currentText()
                styles_data = self.get_styles() or {}
                has_ref = ref_image_valid(style_ref_image(styles_data, selected_style_name))
                active_mode = self.style_ref_mode_combo.effective_mode(has_ref)
                active_instructions, post_instructions, style_ref_paths = build_ref_gen_params(
                    styles_data, selected_style_name, active_mode
                )
                # 关键修复：在主线程拍快照，避免工作线程读取 Qt 控件
                img_config_snapshot = self.img_config_getter_func()

                worker = ImageEditWorker(
                    next_path,
                    prompt,
                    img_config_snapshot=img_config_snapshot,
                    style_instructions=active_instructions,
                    style_ref_paths=style_ref_paths,
                    post_instructions=post_instructions
                )
                worker.result_ready.connect(self.on_worker_finished)
                worker.error.connect(self.on_worker_error)
                worker.log.connect(self.log_msg)
                worker.finished.connect(partial(self.on_worker_thread_finished, next_path))
                self.active_workers[next_path] = worker
                worker.start()
        except Exception as e:
            self.log_msg(f"启动线程失败: {e}")
            self.log_msg(traceback.format_exc())
            self.finish_processing()

        # 没有待处理且没有运行中的任务，说明当前批次结束
        if self.processing and not self.active_workers and self._get_next_unprocessed_path() is None:
            self.finish_processing()

    def update_styles(self, style_keys):
        curr_main = self.main_style_combo.currentText()
        self.main_style_combo.blockSignals(True)
        self.main_style_combo.clear()
        self.main_style_combo.addItems(style_keys)
        if curr_main in style_keys:
            self.main_style_combo.setCurrentText(curr_main)
        elif self._pending_main_style and self._pending_main_style in style_keys:
            self.main_style_combo.setCurrentText(self._pending_main_style)
            self._pending_main_style = ""
        self.main_style_combo.blockSignals(False)
        self._refresh_style_ref_availability()

    def _refresh_style_ref_availability(self):
        """加载样式列表后按当前样式的参考图是否存在刷新参考模式可用性。"""
        styles_data = self.get_styles() or {}
        name = self.main_style_combo.currentText()
        has_ref = ref_image_valid(style_ref_image(styles_data, name))
        self.style_ref_mode_combo.set_modes_available(has_ref)
        return has_ref

    def _on_style_changed(self, _name=None):
        self._refresh_style_ref_availability()

    def on_worker_finished(self, result_json, image_path):
        self.results[image_path] = "success"
        self.update_item_status(image_path, Qt.GlobalColor.green)
        self.log_msg(f"处理成功: {os.path.basename(image_path)}")
        
        # 保存结果
        self.save_result(result_json, image_path)
        
        self.update_progress()

    def on_worker_error(self, error_msg, image_path):
        self.results[image_path] = "error"
        self.update_item_status(image_path, Qt.GlobalColor.red)
        self.log_msg(f"处理失败: {os.path.basename(image_path)} - {error_msg}")
        
        self.update_progress()
 
    def on_worker_thread_finished(self, image_path):
        if image_path in self.active_workers:
            worker = self.active_workers.pop(image_path)
            worker.deleteLater()
        self.launch_available_workers()

    def update_item_status(self, image_path, color):
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == image_path:
                item.setBackground(color)
                break

    def save_result(self, result_json, image_path):
        try:
            # 创建保存目录
            now = datetime.now()
            now_str = now.strftime("%Y%m%d-%H%M%S")
            date_str = now.strftime("%Y%m%d")
            save_dir = os.path.join("data", date_str, "image-edit")
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成文件名
            original_name = os.path.splitext(os.path.basename(image_path))[0]
            input_md5 = str(result_json.get("input_image_md5", "")).strip().lower()
            if not input_md5:
                input_md5 = self._cache_md5(image_path).lower()
            if not MD5_TAIL_RE.match(input_md5):
                input_md5 = "unknownmd5"
            base_filename = f"{now_str}-{original_name}-edit"

            # 将生成图片重命名为以 md5 结尾，便于后续目录扫描比对
            renamed_images = []
            for gen_path in result_json.get("generated_images", []) or []:
                renamed_images.append(self._rename_output_with_md5_tail(gen_path, input_md5))
            result_json["generated_images"] = renamed_images
            
            # 保存 JSON 结果
            json_filename = self._build_filename_with_md5(save_dir, base_filename, ".json", input_md5)
            save_path = os.path.join(save_dir, json_filename)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(result_json, f, ensure_ascii=False, indent=4)
                
            self.log_msg(f"✅ 成功！JSON 结果已保存至: {json_filename}")
            
            # 保存提示词 txt 文件
            final_prompt = result_json.get("final_prompt", "")
            original_prompt = result_json.get("original_prompt", "")
            
            txt_filename = self._build_filename_with_md5(save_dir, f"{base_filename}-prompts", ".txt", input_md5)
            orig_txt_filename = self._build_filename_with_md5(save_dir, f"{base_filename}-original-prompts", ".txt", input_md5)
            
            with open(os.path.join(save_dir, txt_filename), "w", encoding="utf-8") as f: 
                f.write(final_prompt)
            with open(os.path.join(save_dir, orig_txt_filename), "w", encoding="utf-8") as f: 
                f.write(original_prompt)
                
            self.log_msg(f"✅ 成功！两份提示词文件已保存:\n - {txt_filename}\n - {orig_txt_filename}")
            
            # 将分析文件的修改时间同步到与生成的第一张图片一致
            if renamed_images:
                try:
                    first_img = renamed_images[0]
                    self.log_msg(f"🕐 开始同步分析文件 mtime，基准图片: {os.path.basename(first_img)}")
                    if not os.path.exists(first_img):
                        self.log_msg(f"⚠️ 同步 mtime 跳过：图片文件不存在 {first_img}")
                    else:
                        img_mtime = os.stat(first_img).st_mtime
                        img_mtime_str = datetime.fromtimestamp(img_mtime).strftime("%H:%M:%S")
                        self.log_msg(f"  基准图片 mtime: {img_mtime_str}")

                        synced = 0
                        for fpath, label in ((save_path, "JSON"),
                                              (os.path.join(save_dir, txt_filename), "优化提示词"),
                                              (os.path.join(save_dir, orig_txt_filename), "原始提示词")):
                            if os.path.exists(fpath):
                                old_mtime = os.stat(fpath).st_mtime
                                old_str = datetime.fromtimestamp(old_mtime).strftime("%H:%M:%S")
                                os.utime(fpath, (os.stat(fpath).st_atime, img_mtime))
                                synced += 1
                                self.log_msg(f"  ✓ {label} : {old_str} → {img_mtime_str}  ({os.path.basename(fpath)})")
                            else:
                                self.log_msg(f"  ✗ {label} 文件不存在，跳过 ({fpath})")

                        self.log_msg(f"🕐 同步完成: {synced}/3 个文件已同步 mtime")
                except Exception as e:
                    self.log_msg(f"⚠️ 同步分析文件 mtime 失败: {e}")
            else:
                self.log_msg("⏭ 同步 mtime 跳过：renamed_images 为空，无生成图片")
            
            # 打印生成的图片路径
            generated_images = result_json.get("generated_images", [])
            if generated_images:
                self.log_msg(f"\n🎉 成功生成了 {len(generated_images)} 张图片！")
                for img_path in generated_images:
                    self.log_msg(f" 📂 保存路径: {img_path}")
            else:
                self.log_msg("\n⚠️ 未能获取到图片，请检查上方日志，或查看日志文件夹（log）的记录。")
            
        except Exception as e:
            self.log_msg(f"❌ 保存结果失败: {e}")

    def _build_filename_with_md5(self, save_dir, base_root, ext, md5_text):
        candidate = f"{base_root}-{md5_text}{ext}"
        if not os.path.exists(os.path.join(save_dir, candidate)):
            return candidate
        idx = 1
        while True:
            candidate = f"{base_root}-{idx}-{md5_text}{ext}"
            if not os.path.exists(os.path.join(save_dir, candidate)):
                return candidate
            idx += 1

    def _rename_output_with_md5_tail(self, file_path, md5_text):
        if not file_path or not os.path.exists(file_path):
            return file_path
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        stem, ext = os.path.splitext(filename)
        last_seg = stem.split("-")[-1].strip().lower()
        if MD5_TAIL_RE.match(last_seg):
            return file_path
        new_name = self._build_filename_with_md5(directory, stem, ext, md5_text)
        new_path = os.path.join(directory, new_name)
        try:
            os.replace(file_path, new_path)
            return new_path
        except Exception as e:
            self.log_msg(f"重命名输出文件失败: {filename} -> {new_name}, 错误: {e}")
            return file_path

    def finish_processing(self):
        self.processing = False
        self._run_scope_paths = None
        self.update_progress()
        self.start_btn.setEnabled(True)
        self.add_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.template_combo.setEnabled(True)
        self._update_template_save_buttons()
        self.prompt_edit.setEnabled(True)
        self.thread_spin.setEnabled(True)

        # 更新按钮可用性（失败重试/勾选执行）
        self._update_action_buttons()
        has_errors = any(status == "error" for status in self.results.values())
        
        self.log_msg("批量处理任务完成！")
        if has_errors:
            self.log_msg("部分任务处理失败，可点击'重试失败任务'(重试全部红色)或'重试勾选失败项'。")

    def _shutdown_workers(self, wait_ms=8000):
        workers = list(self.active_workers.values())
        if not workers:
            return

        self.log_msg(f"检测到 {len(workers)} 个后台线程，正在安全停止...")
        for worker in workers:
            try:
                worker.requestInterruption()
            except Exception:
                pass

        for worker in workers:
            try:
                if worker.isRunning() and not worker.wait(wait_ms):
                    self.log_msg("线程等待超时，执行强制终止")
                    worker.terminate()
                    worker.wait(1000)
            except Exception as e:
                self.log_msg(f"停止线程失败: {e}")

        self.active_workers.clear()

    def closeEvent(self, event):
        try:
            self.processing = False
            self._shutdown_workers()
        except Exception:
            pass
        super().closeEvent(event)

    def _on_app_about_to_quit(self):
        self.processing = False
        self._shutdown_workers()
