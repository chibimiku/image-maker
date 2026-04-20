import os
import json
import uuid
from datetime import datetime
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QFileDialog, QLabel, QTextEdit, QMessageBox, QComboBox, 
                             QSplitter, QProgressBar, QSpinBox, QScrollArea, QGridLayout, QFrame,
                             QListWidget, QListWidgetItem, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QThreadPool, QRunnable, QObject
from PyQt5.QtGui import QPixmap, QImageReader

from api_backend import generate_image_whatai, generate_image_aigc2d
from utils.image_upscale_runtime import JpgAutoUpscaleThread, normalize_upscale_options

CHAR_PROMPT_DIR = "data/prompts/char"
CHAR_DESIGN_UI_STATE_FILE = "data/char_design_ui_state.json"

class DropImageLabel(QLabel):
    image_changed = pyqtSignal(str)

    def __init__(self, text="拖入/粘贴/点击选择图片", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("QLabel { border: 2px dashed #aaa; border-radius: 5px; background-color: #f9f9f9; color: #666; }")
        self.setMinimumSize(150, 150)
        self.setAcceptDrops(True)
        self.image_path = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
            if file_path:
                self.set_image(file_path)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                self.set_image(file_path)

    def set_image(self, file_path):
        self.image_path = file_path
        pixmap = QPixmap(file_path)
        if not pixmap.isNull():
            self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.image_changed.emit(file_path)

    def clear_image(self):
        self.image_path = None
        self.clear()
        self.setText("拖入/粘贴/点击选择图片")

class MultiImageDropArea(QWidget):
    images_changed = pyqtSignal(list)

    def __init__(self, max_images=10, parent=None):
        super().__init__(parent)
        self.max_images = max_images
        self.image_paths = []
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.drop_label = QLabel(f"拖入/粘贴/点击选择其他参考图 (最多{max_images}张)")
        self.drop_label.setAlignment(Qt.AlignCenter)
        self.drop_label.setStyleSheet("QLabel { border: 2px dashed #aaa; border-radius: 5px; background-color: #f9f9f9; color: #666; padding: 20px; }")
        self.drop_label.setAcceptDrops(True)
        self.layout.addWidget(self.drop_label)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMaximumHeight(150)
        self.scroll_widget = QWidget()
        self.scroll_layout = QHBoxLayout(self.scroll_widget)
        self.scroll_layout.setAlignment(Qt.AlignLeft)
        self.scroll_area.setWidget(self.scroll_widget)
        self.layout.addWidget(self.scroll_area)
        
        self.drop_label.mousePressEvent = self.on_mouse_press
        self.drop_label.dragEnterEvent = self.on_drag_enter
        self.drop_label.dropEvent = self.on_drop

    def on_mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            files, _ = QFileDialog.getOpenFileNames(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
            if files:
                self.add_images(files)

    def on_drag_enter(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def on_drop(self, event):
        urls = event.mimeData().urls()
        files = [url.toLocalFile() for url in urls if url.toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        if files:
            self.add_images(files)

    def add_images(self, files):
        for file in files:
            if len(self.image_paths) >= self.max_images:
                break
            if file not in self.image_paths:
                self.image_paths.append(file)
                self.add_thumbnail(file)
        self.images_changed.emit(self.image_paths)
        self.update_label_text()

    def add_thumbnail(self, file_path):
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)
        
        lbl = QLabel()
        pixmap = QPixmap(file_path)
        lbl.setPixmap(pixmap.scaled(80, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        vbox.addWidget(lbl)
        
        btn = QPushButton("删除")
        btn.clicked.connect(lambda: self.remove_image(file_path, container))
        vbox.addWidget(btn)
        
        self.scroll_layout.addWidget(container)

    def remove_image(self, file_path, container):
        if file_path in self.image_paths:
            self.image_paths.remove(file_path)
        container.deleteLater()
        self.images_changed.emit(self.image_paths)
        self.update_label_text()

    def clear_images(self):
        self.image_paths.clear()
        for i in reversed(range(self.scroll_layout.count())): 
            self.scroll_layout.itemAt(i).widget().setParent(None)
        self.images_changed.emit(self.image_paths)
        self.update_label_text()

    def update_label_text(self):
        self.drop_label.setText(f"拖入/粘贴/点击选择其他参考图 ({len(self.image_paths)}/{self.max_images}张)")

class WorkerSignals(QObject):
    finished = pyqtSignal(dict, dict)  # result_json, task_info
    error = pyqtSignal(str, dict)      # error_msg, task_info
    log = pyqtSignal(str)

class CharDesignWorker(QRunnable):
    def __init__(self, task_info, config_getter_func, img_config_getter_func, is_stopped_func):
        super().__init__()
        self.task_info = task_info
        self.config_getter_func = config_getter_func
        self.img_config_getter_func = img_config_getter_func
        self.is_stopped_func = is_stopped_func
        self.signals = WorkerSignals()

    def run(self):
        if self.is_stopped_func():
            self.signals.log.emit(f"任务已取消: {self.task_info['prompt_item'].get('id', 'unknown')}")
            self.task_info['status'] = 'cancelled'
            self.signals.error.emit("任务被用户停止", self.task_info)
            return
            
        try:
            prompt_item = self.task_info['prompt_item']
            self.signals.log.emit(f"开始处理任务: {prompt_item.get('id', 'unknown')} - {prompt_item.get('description', '')}")
            self.signals.log.emit(
                f"任务参数: style={self.task_info.get('style_name', '默认(无附加)')} | prompt预览={self.task_info.get('prompt_preview', '')}"
            )
            
            img_url, img_key, img_model, api_type = self.img_config_getter_func()
            
            if not img_key:
                raise ValueError("请先配置生图 API Key")
            
            aspect_ratio = prompt_item.get('aspect_ratio', '1:1')
            full_prompt = self.task_info['full_prompt']
            image_paths = self.task_info['image_paths']
            save_dir = self.task_info['save_dir']
            resolution = self.task_info.get('resolution', '默认')
            
            if api_type == "aigc2d":
                generation_result = generate_image_aigc2d(
                    prompt=full_prompt, 
                    image_paths=image_paths,
                    model=img_model, 
                    aspect_ratio=aspect_ratio,
                    instructions="", # Style is already in full_prompt
                    api_type=api_type,
                    save_sub_dir=save_dir,
                    file_prefix=prompt_item.get('id', ''),
                    resolution=resolution if resolution != "默认" else "",
                    return_metadata=True
                )
            else:
                generation_result = generate_image_whatai(
                    prompt=full_prompt, 
                    image_paths=image_paths,
                    model=img_model, 
                    aspect_ratio=aspect_ratio,
                    instructions="",
                    api_type=api_type,
                    save_sub_dir=save_dir,
                    file_prefix=prompt_item.get('id', ''),
                    resolution=resolution if resolution != "默认" else "",
                    return_metadata=True
                )
            if isinstance(generation_result, dict):
                saved_files = generation_result.get("saved_files", [])
                annotation_json = generation_result.get("annotation", {})
                raw_text_output = generation_result.get("raw_text", "")
            else:
                saved_files = generation_result
                annotation_json = {}
                raw_text_output = ""
            
            if not saved_files:
                raise ValueError("生图接口未返回任何图片")
                
            result_json = {
                "status": "success",
                "task_id": self.task_info['task_id'],
                "prompt_id": prompt_item.get('id', ''),
                "full_prompt": full_prompt,
                "aspect_ratio": aspect_ratio,
                "generated_images": saved_files,
                "annotation": annotation_json,
                "raw_text_output": raw_text_output
            }
            
            self.signals.finished.emit(result_json, self.task_info)
            
        except Exception as e:
            self.signals.error.emit(str(e), self.task_info)

class CharDesignWidget(QWidget):
    def __init__(self, config_getter_func, img_config_getter_func, styles_getter_func, upscale_options_getter_func=None, upscale_options_changed_callback=None):
        super().__init__()
        self.config_getter_func = config_getter_func
        self.img_config_getter_func = img_config_getter_func
        self.get_styles = styles_getter_func
        self.get_upscale_options = upscale_options_getter_func
        self.on_upscale_options_changed = upscale_options_changed_callback
        
        self.threadpool = QThreadPool()
        self.tasks = []
        self.results = {}
        self.current_batch_id = None
        self.current_save_dir = None
        self.is_stopped = False
        self.current_batch_total = 0
        self.current_batch_completed = 0
        self._is_restoring_state = False
        self._pending_main_style = ""
        self._post_threads = []
        
        self.initUI()
        self.load_prompt_jsons()
        self.load_ui_state()

    def initUI(self):
        layout = QVBoxLayout(self)
        
        # 顶部：配置选择
        top_layout = QHBoxLayout()
        
        top_layout.addWidget(QLabel("角色描述JSON:"))
        self.json_combo = QComboBox()
        top_layout.addWidget(self.json_combo)
        
        top_layout.addWidget(QLabel("附加画风:"))
        self.main_style_combo = QComboBox()
        top_layout.addWidget(self.main_style_combo)
        
        top_layout.addWidget(QLabel("并发线程数:"))
        self.thread_spin = QSpinBox()
        self.thread_spin.setRange(1, 10)
        self.thread_spin.setValue(3)
        top_layout.addWidget(self.thread_spin)
        
        top_layout.addWidget(QLabel("强制分辨率:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["默认", "1K", "2K", "4K"])
        top_layout.addWidget(self.resolution_combo)
        
        layout.addLayout(top_layout)

        upscale_layout = QHBoxLayout()
        self.enable_jpg_upscale_cb = QCheckBox("生图后自动处理 JPG")
        self.enable_jpg_upscale_cb.toggled.connect(self._persist_upscale_options)
        upscale_layout.addWidget(self.enable_jpg_upscale_cb)
        upscale_layout.addStretch()
        layout.addLayout(upscale_layout)
        self.set_upscale_options_defaults(self.get_upscale_options() if self.get_upscale_options else {})

        extra_prompt_layout = QVBoxLayout()
        extra_prompt_layout.addWidget(QLabel("自定义前置 Prompt（可选，拼接到正文前）:"))
        self.custom_prefix_prompt = QTextEdit()
        self.custom_prefix_prompt.setPlaceholderText("例如：masterpiece, best quality, ultra detailed")
        self.custom_prefix_prompt.setMaximumHeight(70)
        extra_prompt_layout.addWidget(self.custom_prefix_prompt)

        extra_prompt_layout.addWidget(QLabel("自定义后置 Prompt（可选，拼接到正文后）:"))
        self.custom_suffix_prompt = QTextEdit()
        self.custom_suffix_prompt.setPlaceholderText("例如：clean background, no watermark, no text")
        self.custom_suffix_prompt.setMaximumHeight(70)
        extra_prompt_layout.addWidget(self.custom_suffix_prompt)

        extra_prompt_layout.addWidget(QLabel("拼接额外要求（可选）:"))
        self.concat_requirement_prompt = QTextEdit()
        self.concat_requirement_prompt.setPlaceholderText("例如：保持角色服装与发色一致，避免改变人物年龄")
        self.concat_requirement_prompt.setMaximumHeight(70)
        extra_prompt_layout.addWidget(self.concat_requirement_prompt)

        concat_position_layout = QHBoxLayout()
        concat_position_layout.addWidget(QLabel("额外要求位置:"))
        self.concat_requirement_position_combo = QComboBox()
        self.concat_requirement_position_combo.addItems(["拼接在正文后", "拼接在正文前"])
        concat_position_layout.addWidget(self.concat_requirement_position_combo)
        concat_position_layout.addStretch()
        extra_prompt_layout.addLayout(concat_position_layout)
        layout.addLayout(extra_prompt_layout)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：图片输入区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        grid_layout = QGridLayout()
        
        self.img_front = DropImageLabel("正面视图\n(必填)")
        self.img_side = DropImageLabel("侧面视图\n(必填)")
        self.img_back = DropImageLabel("背面视图\n(必填)")
        self.img_head = DropImageLabel("头像\n(必填)")
        
        grid_layout.addWidget(QLabel("正面视图:"), 0, 0)
        grid_layout.addWidget(self.img_front, 1, 0)
        grid_layout.addWidget(QLabel("侧面视图:"), 0, 1)
        grid_layout.addWidget(self.img_side, 1, 1)
        grid_layout.addWidget(QLabel("背面视图:"), 2, 0)
        grid_layout.addWidget(self.img_back, 3, 0)
        grid_layout.addWidget(QLabel("头像:"), 2, 1)
        grid_layout.addWidget(self.img_head, 3, 1)
        
        left_layout.addLayout(grid_layout)
        
        self.img_others = MultiImageDropArea(max_images=10)
        left_layout.addWidget(self.img_others)
        
        btn_layout = QHBoxLayout()
        self.clear_imgs_btn = QPushButton("清空所有图片")
        self.clear_imgs_btn.clicked.connect(self.clear_all_images)
        btn_layout.addWidget(self.clear_imgs_btn)
        left_layout.addLayout(btn_layout)
        
        splitter.addWidget(left_widget)
        
        # 右侧：任务列表和日志
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # 添加全选和全不选按钮
        select_buttons_layout = QHBoxLayout()
        right_layout.addWidget(QLabel("任务列表:"))
        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.clicked.connect(self.select_all_tasks)
        select_buttons_layout.addWidget(self.select_all_btn)
        
        self.deselect_all_btn = QPushButton("全不选")
        self.deselect_all_btn.clicked.connect(self.deselect_all_tasks)
        select_buttons_layout.addWidget(self.deselect_all_btn)

        self.select_missing_btn = QPushButton("勾选缺失任务")
        self.select_missing_btn.clicked.connect(self.select_missing_tasks_from_output_dir)
        select_buttons_layout.addWidget(self.select_missing_btn)
        
        right_layout.addLayout(select_buttons_layout)
        
        self.task_list_widget = QListWidget()
        right_layout.addWidget(self.task_list_widget)
        
        right_layout.addWidget(QLabel("处理日志:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        right_layout.addWidget(self.log_text)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 400])
        layout.addWidget(splitter)
        
        # 底部：控制
        bottom_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        bottom_layout.addWidget(self.progress_bar)
        
        self.start_btn = QPushButton("开始批量生成")
        self.start_btn.clicked.connect(self.start_processing)
        bottom_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        bottom_layout.addWidget(self.stop_btn)
        
        self.retry_btn = QPushButton("重试勾选任务")
        self.retry_btn.clicked.connect(self.retry_selected)
        self.retry_btn.setEnabled(False)
        bottom_layout.addWidget(self.retry_btn)
        
        layout.addLayout(bottom_layout)
        
        # Connect signal after all UI components are initialized
        self.json_combo.currentIndexChanged.connect(self.on_json_changed)
        self.json_combo.currentTextChanged.connect(self.save_ui_state)
        self.main_style_combo.currentTextChanged.connect(self.save_ui_state)
        self.thread_spin.valueChanged.connect(self.save_ui_state)
        self.resolution_combo.currentTextChanged.connect(self.save_ui_state)
        self.custom_prefix_prompt.textChanged.connect(self.save_ui_state)
        self.custom_suffix_prompt.textChanged.connect(self.save_ui_state)
        self.concat_requirement_prompt.textChanged.connect(self.save_ui_state)
        self.concat_requirement_position_combo.currentTextChanged.connect(self.save_ui_state)

    def load_prompt_jsons(self):
        self.json_combo.blockSignals(True)
        self.json_combo.clear()
        if os.path.exists(CHAR_PROMPT_DIR):
            for filename in os.listdir(CHAR_PROMPT_DIR):
                if filename.endswith(".json"):
                    self.json_combo.addItem(filename)
        self.json_combo.blockSignals(False)
        
        # Manually trigger the first load if items exist
        if self.json_combo.count() > 0:
            self.on_json_changed()
    
    def load_ui_state_data(self):
        if not os.path.exists(CHAR_DESIGN_UI_STATE_FILE):
            return {}
        try:
            with open(CHAR_DESIGN_UI_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            self.log_msg(f"读取UI状态失败，已使用默认设置: {e}")
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

            resolution = state.get("resolution")
            if isinstance(resolution, str) and self.resolution_combo.findText(resolution) >= 0:
                self.resolution_combo.setCurrentText(resolution)
            else:
                self.resolution_combo.setCurrentIndex(0)

            json_filename = state.get("json_filename")
            if isinstance(json_filename, str) and self.json_combo.findText(json_filename) >= 0:
                self.json_combo.setCurrentText(json_filename)
            elif self.json_combo.count() > 0:
                self.json_combo.setCurrentIndex(0)

            style_name = state.get("main_style")
            if isinstance(style_name, str):
                self._pending_main_style = style_name
                if self.main_style_combo.findText(style_name) >= 0:
                    self.main_style_combo.setCurrentText(style_name)

            custom_prefix = state.get("custom_prefix_prompt")
            if isinstance(custom_prefix, str):
                self.custom_prefix_prompt.setPlainText(custom_prefix)

            custom_suffix = state.get("custom_suffix_prompt")
            if isinstance(custom_suffix, str):
                self.custom_suffix_prompt.setPlainText(custom_suffix)

            concat_requirement = state.get("concat_requirement_prompt")
            if isinstance(concat_requirement, str):
                self.concat_requirement_prompt.setPlainText(concat_requirement)

            concat_requirement_position = state.get("concat_requirement_position")
            if (
                isinstance(concat_requirement_position, str)
                and self.concat_requirement_position_combo.findText(concat_requirement_position) >= 0
            ):
                self.concat_requirement_position_combo.setCurrentText(concat_requirement_position)
        finally:
            self._is_restoring_state = False

        self.save_ui_state()

    def save_ui_state(self, *args):
        if self._is_restoring_state:
            return

        state = {
            "json_filename": self.json_combo.currentText(),
            "thread_count": self.thread_spin.value(),
            "main_style": self.main_style_combo.currentText(),
            "resolution": self.resolution_combo.currentText(),
            "custom_prefix_prompt": self.custom_prefix_prompt.toPlainText().strip(),
            "custom_suffix_prompt": self.custom_suffix_prompt.toPlainText().strip(),
            "concat_requirement_prompt": self.concat_requirement_prompt.toPlainText().strip(),
            "concat_requirement_position": self.concat_requirement_position_combo.currentText()
        }
        try:
            os.makedirs(os.path.dirname(CHAR_DESIGN_UI_STATE_FILE), exist_ok=True)
            with open(CHAR_DESIGN_UI_STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.log_msg(f"保存UI状态失败: {e}")

    def on_json_changed(self):
        filename = self.json_combo.currentText()
        if not filename: return
        filepath = os.path.join(CHAR_PROMPT_DIR, filename)
        self.log_msg(f"正在加载JSON文件: {filename}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                prompts = data.get("specialized_prompts", [])
                self.task_list_widget.clear()
                for p in prompts:
                    item_text = f"[{p.get('id', 'N/A')}] {p.get('description', '')} (比例: {p.get('aspect_ratio', '1:1')})"
                    item = QListWidgetItem(item_text)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked)
                    # Store prompt data in the item for later use
                    item.setData(Qt.UserRole, p)
                    self.task_list_widget.addItem(item)
        except Exception as e:
            self.log_msg(f"加载JSON失败: {e}")

    def update_styles(self, style_keys):
        curr_main = self.main_style_combo.currentText()
        self.main_style_combo.blockSignals(True)
        self.main_style_combo.clear()
        self.main_style_combo.addItems(style_keys)
        if self._pending_main_style and self._pending_main_style in style_keys:
            self.main_style_combo.setCurrentText(self._pending_main_style)
            self._pending_main_style = ""
        elif curr_main in style_keys:
            self.main_style_combo.setCurrentText(curr_main)
        self.main_style_combo.blockSignals(False)
        self.save_ui_state()

    def clear_all_images(self):
        self.img_front.clear_image()
        self.img_side.clear_image()
        self.img_back.clear_image()
        self.img_head.clear_image()
        self.img_others.clear_images()

    def log_msg(self, msg):
        self.log_text.append(msg)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def stop_processing(self):
        self.is_stopped = True
        self.log_msg("正在停止任务，等待当前正在执行的任务完成...")
        self.stop_btn.setEnabled(False)

    def start_processing(self):
        # 验证必填图片
        required_imgs = [self.img_head.image_path, self.img_front.image_path, self.img_side.image_path, self.img_back.image_path]
        if not all(required_imgs):
            QMessageBox.warning(self, "警告", "请提供所有必填的参考图片（头像、正面、侧面、背面）")
            return
            
        json_filename = self.json_combo.currentText()
        if not json_filename:
            QMessageBox.warning(self, "警告", "请选择角色描述JSON")
            return
            
        try:
            with open(os.path.join(CHAR_PROMPT_DIR, json_filename), 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"读取JSON失败: {e}")
            return
            
        common_prompt = prompt_data.get("common_prompt", {}).get("prompt", "")
        specialized_prompts = prompt_data.get("specialized_prompts", [])
        
        if not specialized_prompts:
            QMessageBox.warning(self, "警告", "JSON中没有 specialized_prompts 任务")
            return

        selected_style_name = self.main_style_combo.currentText()
        styles_data = self.get_styles()
        style_instructions = styles_data.get(selected_style_name, "")
        custom_prefix_prompt = self.custom_prefix_prompt.toPlainText().strip()
        custom_suffix_prompt = self.custom_suffix_prompt.toPlainText().strip()
        concat_requirement_prompt = self.concat_requirement_prompt.toPlainText().strip()
        concat_requirement_position = self.concat_requirement_position_combo.currentText()
        self.log_msg(f"本次批量使用画风: {selected_style_name}")

        all_image_paths = required_imgs + self.img_others.image_paths

        # Only generate new batch ID if we are starting fresh (not resuming)
        if not self.current_batch_id or self.is_stopped:
            self.current_batch_id = str(uuid.uuid4())[:8]
            self.current_save_dir = os.path.join("char-design", self.current_batch_id)
            self.tasks = []
            self.results = {}
            is_resume = False
        else:
            is_resume = True
            
        self.is_stopped = False
        
        # Reset list widget colors and get selected tasks
        selected_prompts = []
        for i in range(self.task_list_widget.count()):
            item = self.task_list_widget.item(i)
            # Only process checked items
            if item.checkState() == Qt.Checked:
                selected_prompts.append((i, item.data(Qt.UserRole)))
                item.setBackground(Qt.white) # Reset color for tasks we are about to run
                
        if not selected_prompts:
            QMessageBox.warning(self, "警告", "请至少勾选一个生成任务")
            self.start_btn.setEnabled(True)
            return
        
        # 处理选中的任务，包括已完成的任务
        tasks_to_run = []
        for idx, (list_idx, p_item) in enumerate(selected_prompts):
            # 生成新的任务ID，包含时间戳以避免覆盖
            timestamp = datetime.now().strftime("%H%M%S")
            task_id = f"{self.current_batch_id}_{list_idx}_{timestamp}"
            
            prompt_parts = [
                custom_prefix_prompt,
                style_instructions,
                common_prompt,
                p_item.get('prompt', ''),
                custom_suffix_prompt
            ]
            full_prompt = "\n".join([part for part in prompt_parts if str(part).strip()]).strip()
            if concat_requirement_prompt:
                concat_text = f"额外拼接要求：{concat_requirement_prompt}"
                if concat_requirement_position == "拼接在正文前":
                    full_prompt = f"{concat_text}\n\n{full_prompt}".strip()
                else:
                    full_prompt = f"{full_prompt}\n\n{concat_text}".strip()
            prompt_preview = full_prompt.replace("\n", " ")[:120]
            task_info = {
                'task_id': task_id,
                'prompt_item': p_item,
                'full_prompt': full_prompt,
                'style_name': selected_style_name,
                'prompt_preview': prompt_preview,
                'custom_prefix_prompt': custom_prefix_prompt,
                'custom_suffix_prompt': custom_suffix_prompt,
                'concat_requirement_prompt': concat_requirement_prompt,
                'concat_requirement_position': concat_requirement_position,
                'image_paths': all_image_paths,
                'save_dir': self.current_save_dir,
                'status': 'pending',
                'list_idx': list_idx, # Store list index to update color later
                'resolution': self.resolution_combo.currentText()
            }
            self.log_msg(
                f"排队任务: {p_item.get('id', 'unknown')} | style={selected_style_name} | prompt预览={prompt_preview}"
            )
            
            # 添加到全局 tasks/results
            self.tasks.append(task_info)
            self.results[task_id] = task_info
            tasks_to_run.append(task_info)

        if not tasks_to_run:
            self.log_msg("没有需要执行的任务。")
            self.start_btn.setEnabled(True)
            return

        self.threadpool.setMaxThreadCount(self.thread_spin.value())
        self.progress_bar.setMaximum(len(tasks_to_run))
        self.current_batch_total = len(tasks_to_run)
        self.current_batch_completed = 0
        self.update_progress()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.retry_btn.setEnabled(False)
        self.log_msg(f"开始批量生成任务，批次ID: {self.current_batch_id}，本次将执行 {len(tasks_to_run)} 个任务")
        
        for task in tasks_to_run:
            worker = CharDesignWorker(task, self.config_getter_func, self.img_config_getter_func, lambda: self.is_stopped)
            worker.signals.finished.connect(self.on_task_finished)
            worker.signals.error.connect(self.on_task_error)
            worker.signals.log.connect(self.log_msg)
            self.threadpool.start(worker)

    def retry_selected(self):
        failed_tasks = []
        
        self.is_stopped = False
        
        # Find checked items that have failed or haven't been run
        for i in range(self.task_list_widget.count()):
            item = self.task_list_widget.item(i)
            if item.checkState() == Qt.Checked:
                # Find corresponding task in results
                for task in self.results.values():
                    if task.get('list_idx') == i:
                        if task['status'] == 'error':
                            failed_tasks.append(task)
                            item.setBackground(Qt.white) # Reset color for retry
                        break
                
        if not failed_tasks:
            self.log_msg("没有勾选的失败任务需要重试")
            return
            
        self.threadpool.setMaxThreadCount(self.thread_spin.value())
        self.log_msg(f"重试 {len(failed_tasks)} 个勾选的失败任务...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.retry_btn.setEnabled(False)
        
        self.current_batch_total = len(failed_tasks)
        self.current_batch_completed = 0
        self.progress_bar.setMaximum(self.current_batch_total)
        self.update_progress()
        
        for task in failed_tasks:
            task['status'] = 'pending'
            task['resolution'] = self.resolution_combo.currentText()
            worker = CharDesignWorker(task, self.config_getter_func, self.img_config_getter_func, lambda: self.is_stopped)
            worker.signals.finished.connect(self.on_task_finished)
            worker.signals.error.connect(self.on_task_error)
            worker.signals.log.connect(self.log_msg)
            self.threadpool.start(worker)

    def on_task_finished(self, result_json, task_info):
        task_id = task_info['task_id']
        self.results[task_id]['status'] = 'success'
        self.log_msg(f"✅ 任务成功: {task_info['prompt_item'].get('id', '')}")
        
        # Update list item color to green and uncheck
        list_idx = task_info.get('list_idx')
        if list_idx is not None and list_idx < self.task_list_widget.count():
            item = self.task_list_widget.item(list_idx)
            item.setBackground(Qt.green)
            item.setCheckState(Qt.Unchecked)
        
        try:
            today_str = datetime.now().strftime("%Y%m%d")
            save_path_full = os.path.join("data", today_str, self.current_save_dir or "")
            os.makedirs(save_path_full, exist_ok=True)
            json_filename = f"{task_info['prompt_item'].get('id', 'task')}_{task_id}.json"
            with open(os.path.join(save_path_full, json_filename), 'w', encoding='utf-8') as f:
                json.dump(result_json, f, ensure_ascii=False, indent=4)
            annotation = result_json.get("annotation", {})
            generated_images = result_json.get("generated_images", [])
            self.save_annotation_texts(save_path_full, generated_images, annotation)
            self._start_jpg_postprocess(generated_images, task_info['prompt_item'].get('id', 'unknown'))
        except Exception as e:
            self.log_msg(f"保存任务JSON结果失败: {e}")
            
        self.current_batch_completed += 1
        self.update_progress()

    def on_task_error(self, error_msg, task_info):
        task_id = task_info['task_id']
        # If task was cancelled, we don't treat it as an error for retry purposes
        if task_info.get('status') != 'cancelled':
            self.results[task_id]['status'] = 'error'
            self.log_msg(f"❌ 任务失败: {task_info['prompt_item'].get('id', '')} - {error_msg}")
            
            # Update list item color to red
            list_idx = task_info.get('list_idx')
            if list_idx is not None and list_idx < self.task_list_widget.count():
                self.task_list_widget.item(list_idx).setBackground(Qt.red)
        
        self.current_batch_completed += 1
        self.update_progress()

    def update_progress(self):
        self.progress_bar.setValue(self.current_batch_completed)
        self.progress_bar.setFormat(f"{self.current_batch_completed}/{self.current_batch_total}")
        
        if self.current_batch_completed >= self.current_batch_total and self.current_batch_total > 0:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            has_errors = any(t['status'] == 'error' for t in self.results.values())
            self.retry_btn.setEnabled(has_errors)
            self.log_msg("当前批次任务处理完毕！" + ("有失败任务可重试。" if has_errors else ""))
    
    def select_all_tasks(self):
        """全选所有任务"""
        for i in range(self.task_list_widget.count()):
            item = self.task_list_widget.item(i)
            item.setCheckState(Qt.Checked)
    
    def deselect_all_tasks(self):
        """全不选所有任务"""
        for i in range(self.task_list_widget.count()):
            item = self.task_list_widget.item(i)
            item.setCheckState(Qt.Unchecked)

    def select_missing_tasks_from_output_dir(self):
        """根据输出目录中缺失的图片文件，自动勾选对应任务。"""
        output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录", "")
        if not output_dir:
            self.log_msg("未选择输出目录，已取消勾选缺失任务。")
            return

        self.deselect_all_tasks()

        image_extensions = set()
        for fmt in QImageReader.supportedImageFormats():
            try:
                ext = bytes(fmt).decode("ascii").lower()
            except Exception:
                ext = str(fmt).lower()
            if ext:
                image_extensions.add(f".{ext}")

        try:
            dir_entries = os.listdir(output_dir)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"读取目录失败: {e}")
            self.log_msg(f"读取目录失败: {e}")
            return

        missing_count = 0
        total_count = self.task_list_widget.count()

        for i in range(total_count):
            item = self.task_list_widget.item(i)
            prompt_item = item.data(Qt.UserRole) or {}
            prompt_id = str(prompt_item.get("id", "")).strip()
            if not prompt_id:
                continue

            prefix = f"{prompt_id}_"
            has_image = False
            for filename in dir_entries:
                file_path = os.path.join(output_dir, filename)
                if not os.path.isfile(file_path):
                    continue
                if not filename.startswith(prefix):
                    continue
                _, ext = os.path.splitext(filename)
                if ext.lower() in image_extensions:
                    has_image = True
                    break

            if not has_image:
                item.setCheckState(Qt.Checked)
                missing_count += 1

        self.log_msg(
            f"缺失任务勾选完成: 输出目录={output_dir}，共扫描 {total_count} 个任务，勾选缺失 {missing_count} 个。"
        )
        QMessageBox.information(
            self,
            "完成",
            f"已扫描 {total_count} 个任务，并勾选缺失项 {missing_count} 个。"
        )

    def _collect_upscale_options(self):
        base_options = normalize_upscale_options(self.get_upscale_options() if self.get_upscale_options else {})
        base_options["enabled"] = bool(self.enable_jpg_upscale_cb.isChecked())
        return normalize_upscale_options(base_options)

    def _persist_upscale_options(self):
        if self.on_upscale_options_changed:
            self.on_upscale_options_changed(self._collect_upscale_options())

    def set_upscale_options_defaults(self, options):
        opts = normalize_upscale_options(options)
        self.enable_jpg_upscale_cb.blockSignals(True)
        self.enable_jpg_upscale_cb.setChecked(bool(opts.get("enabled", False)))
        self.enable_jpg_upscale_cb.blockSignals(False)

    def _start_jpg_postprocess(self, saved_files, prompt_id):
        if not self.enable_jpg_upscale_cb.isChecked():
            return
        jpg_files = [str(path) for path in (saved_files or []) if str(path).lower().endswith((".jpg", ".jpeg"))]
        if not jpg_files:
            return
        options = self._collect_upscale_options()
        if not options.get("model_name"):
            self.log_msg("⚠️ 已启用 JPG 自动处理，但未选择 upscaler 模型，已跳过。")
            return
        thread = JpgAutoUpscaleThread(
            image_paths=jpg_files,
            options=options,
            task_name=f"角色任务后处理({prompt_id})",
        )
        self._post_threads.append(thread)
        thread.log_signal.connect(self.log_msg)
        thread.finish_signal.connect(self._on_postprocess_finished)
        thread.finished.connect(lambda t=thread: self._cleanup_post_thread(t))
        thread.start()

    def _on_postprocess_finished(self, results):
        success = sum(1 for item in results if item.get("success"))
        webp_count = sum(1 for item in results if item.get("webp_path"))
        self.log_msg(f"✅ JPG 自动处理完成，新增 fixed.png: {success} 张，WebP: {webp_count} 张")

    def _cleanup_post_thread(self, thread):
        if thread in self._post_threads:
            self._post_threads.remove(thread)

    def save_annotation_texts(self, save_path_full, generated_images, annotation):
        if not isinstance(annotation, dict):
            return
        booru_tags_value = annotation.get("booru-tags", "")
        if isinstance(booru_tags_value, list):
            booru_tags_value = ", ".join([str(tag).strip() for tag in booru_tags_value if str(tag).strip()])
        text_mapping = {
            "long_description": annotation.get("long_description", ""),
            "short_description": annotation.get("short_description", ""),
            "booru-tags": booru_tags_value
        }
        if not any(value.strip() for value in text_mapping.values() if isinstance(value, str)):
            return
        for key, value in text_mapping.items():
            if not isinstance(value, str) or not value.strip():
                continue
            target_dir = os.path.join(save_path_full, key)
            os.makedirs(target_dir, exist_ok=True)
            for img_path in generated_images:
                image_name = os.path.basename(img_path)
                target_path = os.path.join(target_dir, f"{image_name}.txt")
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(value.strip())
