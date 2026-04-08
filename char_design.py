import os
import json
import uuid
from datetime import datetime
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QFileDialog, QLabel, QTextEdit, QMessageBox, QComboBox, 
                             QSplitter, QProgressBar, QSpinBox, QScrollArea, QGridLayout, QFrame,
                             QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QThreadPool, QRunnable, QObject
from PyQt5.QtGui import QPixmap, QImageReader

from api_backend import generate_image_whatai, generate_image_aigc2d

CHAR_PROMPT_DIR = "data/prompts/char"

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
            
            img_url, img_key, img_model, api_type = self.img_config_getter_func()
            
            if not img_key:
                raise ValueError("请先配置生图 API Key")
            
            aspect_ratio = prompt_item.get('aspect_ratio', '1:1')
            full_prompt = self.task_info['full_prompt']
            image_paths = self.task_info['image_paths']
            save_dir = self.task_info['save_dir']
            
            if api_type == "aigc2d":
                saved_files = generate_image_aigc2d(
                    prompt=full_prompt, 
                    image_paths=image_paths,
                    model=img_model, 
                    aspect_ratio=aspect_ratio,
                    instructions="", # Style is already in full_prompt
                    api_type=api_type,
                    save_sub_dir=save_dir,
                    file_prefix=prompt_item.get('id', '')
                )
            else:
                saved_files = generate_image_whatai(
                    prompt=full_prompt, 
                    image_paths=image_paths,
                    model=img_model, 
                    aspect_ratio=aspect_ratio,
                    instructions="",
                    api_type=api_type,
                    save_sub_dir=save_dir,
                    file_prefix=prompt_item.get('id', '')
                )
            
            if not saved_files:
                raise ValueError("生图接口未返回任何图片")
                
            result_json = {
                "status": "success",
                "task_id": self.task_info['task_id'],
                "prompt_id": prompt_item.get('id', ''),
                "full_prompt": full_prompt,
                "aspect_ratio": aspect_ratio,
                "generated_images": saved_files
            }
            
            self.signals.finished.emit(result_json, self.task_info)
            
        except Exception as e:
            self.signals.error.emit(str(e), self.task_info)

class CharDesignWidget(QWidget):
    def __init__(self, config_getter_func, img_config_getter_func, styles_getter_func):
        super().__init__()
        self.config_getter_func = config_getter_func
        self.img_config_getter_func = img_config_getter_func
        self.get_styles = styles_getter_func
        
        self.threadpool = QThreadPool()
        self.tasks = []
        self.results = {}
        self.current_batch_id = None
        self.current_save_dir = None
        self.is_stopped = False
        self.current_batch_total = 0
        self.current_batch_completed = 0
        
        self.initUI()
        self.load_prompt_jsons()

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
        
        layout.addLayout(top_layout)
        
        splitter = QSplitter(Qt.Horizontal)
        
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
        
        right_layout.addWidget(QLabel("任务列表:"))
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
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
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
        if curr_main in style_keys:
            self.main_style_combo.setCurrentText(curr_main)
        self.main_style_combo.blockSignals(False)

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
        
        # If resuming, we only add tasks that are checked and not already successful
        tasks_to_run = []
        for idx, (list_idx, p_item) in enumerate(selected_prompts):
            task_id = f"{self.current_batch_id}_{list_idx}"
            
            # Skip if resuming and task was already successful
            if is_resume and task_id in self.results and self.results[task_id]['status'] == 'success':
                continue
                
            full_prompt = f"{style_instructions}\n{common_prompt}\n{p_item.get('prompt', '')}".strip()
            task_info = {
                'task_id': task_id,
                'prompt_item': p_item,
                'full_prompt': full_prompt,
                'image_paths': all_image_paths,
                'save_dir': self.current_save_dir,
                'status': 'pending',
                'list_idx': list_idx # Store list index to update color later
            }
            
            # Update or add to global tasks/results
            existing_task_idx = next((i for i, t in enumerate(self.tasks) if t['task_id'] == task_id), -1)
            if existing_task_idx >= 0:
                self.tasks[existing_task_idx] = task_info
            else:
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
            save_path_full = os.path.join("data", today_str, self.current_save_dir)
            os.makedirs(save_path_full, exist_ok=True)
            json_filename = f"{task_info['prompt_item'].get('id', 'task')}_{task_id}.json"
            with open(os.path.join(save_path_full, json_filename), 'w', encoding='utf-8') as f:
                json.dump(result_json, f, ensure_ascii=False, indent=4)
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
