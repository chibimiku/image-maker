import os
import json
import hashlib
import re
from datetime import datetime
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QListWidget, QListWidgetItem, QFileDialog, QLabel, 
                             QTextEdit, QMessageBox, QComboBox, QSplitter, QProgressBar, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from openai import OpenAI

# 复用已有的工具函数
from single_analyzer import compress_and_encode_image, calculate_closest_aspect_ratio
from api_backend import generate_image_whatai, generate_image_aigc2d

PROMPT_DIR = "data/prompts/image-edit"
IMAGE_EDIT_UI_STATE_FILE = "data/image_edit_ui_state.json"
MD5_TAIL_RE = re.compile(r"^[0-9a-fA-F]{32}$")


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
    finished = pyqtSignal(dict, str)  # result_json, image_path
    error = pyqtSignal(str, str)      # error_msg, image_path
    log = pyqtSignal(str)

    def __init__(self, image_path, prompt, config_getter_func, img_config_getter_func, style_instructions=""):
        super().__init__()
        self.image_path = image_path
        self.prompt = prompt
        self.config_getter_func = config_getter_func
        self.img_config_getter_func = img_config_getter_func
        self.style_instructions = style_instructions

    def run(self):
        try:
            self.log.emit(f"开始处理图片: {os.path.basename(self.image_path)}")
            
            # 1. 获取配置
            img_url, img_key, img_model, api_type = self.img_config_getter_func()
            
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
                    image_paths=[self.image_path],
                    model=img_model, 
                    aspect_ratio=aspect_ratio,
                    instructions=self.style_instructions,
                    api_type=api_type,
                    save_sub_dir="image-edit",
                    file_prefix=os.path.splitext(os.path.basename(self.image_path))[0]
                )
            else:
                saved_files = generate_image_whatai(
                    prompt=self.prompt, 
                    image_paths=[self.image_path],
                    model=img_model, 
                    aspect_ratio=aspect_ratio,
                    instructions=self.style_instructions,
                    api_type=api_type,
                    save_sub_dir="image-edit",
                    file_prefix=os.path.splitext(os.path.basename(self.image_path))[0]
                )
            
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
            
            self.finished.emit(result_json, self.image_path)
            
        except Exception as e:
            self.error.emit(str(e), self.image_path)

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
        self._pending_main_style = ""
        self._pending_template = ""
        self._md5_cache = {}
        
        # 确保提示词目录存在
        os.makedirs(PROMPT_DIR, exist_ok=True)
        
        self.initUI()
        self.load_prompt_templates()
        self.load_ui_state()

    def initUI(self):
        layout = QVBoxLayout(self)
        
        # 顶部：提示词模板管理和画风选择
        top_layout = QHBoxLayout()
        
        # 画风选择
        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("附加画风:"))
        self.main_style_combo = QComboBox()
        self.main_style_combo.setMaximumWidth(200)
        style_layout.addWidget(self.main_style_combo)
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
        
        top_layout.addLayout(template_layout)
        layout.addLayout(top_layout)
        
        # 中间：分割器 (左侧图片列表，右侧提示词编辑和日志)
        splitter = QSplitter(Qt.Horizontal)
        
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
        self.image_list.setSelectionMode(QListWidget.ExtendedSelection)
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
                if self.template_combo.findText(template_name) >= 0:
                    self.template_combo.setCurrentText(template_name)
                else:
                    self._pending_template = template_name

            style_name = state.get("main_style")
            if isinstance(style_name, str) and style_name:
                if self.main_style_combo.findText(style_name) >= 0:
                    self.main_style_combo.setCurrentText(style_name)
                else:
                    self._pending_main_style = style_name
        finally:
            self._is_restoring_state = False

    def save_ui_state(self):
        if self._is_restoring_state:
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

    def save_template(self):
        content = self.prompt_edit.toPlainText().strip()
        if not content:
            QMessageBox.warning(self, "警告", "提示词不能为空")
            return
            
        from PyQt5.QtWidgets import QInputDialog
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

    def add_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if files:
            for file in files:
                if file not in self.image_paths:
                    self.image_paths.append(file)
                    item = QListWidgetItem(os.path.basename(file))
                    item.setData(Qt.UserRole, file)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Unchecked)
                    self.image_list.addItem(item)
                    self._cache_md5(file)
            self.update_progress()

    def clear_images(self):
        self.image_paths.clear()
        self.image_list.clear()
        self.results.clear()
        self.active_workers.clear()
        self._md5_cache.clear()
        self.update_progress()
        self.log_text.clear()

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

        found_md5 = set()
        for root, _dirs, files in os.walk(target_dir):
            for filename in files:
                stem = os.path.splitext(filename)[0]
                last_seg = stem.split("-")[-1].strip().lower()
                if MD5_TAIL_RE.match(last_seg):
                    found_md5.add(last_seg)

        missing_count = 0
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            image_path = item.data(Qt.UserRole)
            image_md5 = self._cache_md5(image_path).lower()
            if image_md5 and image_md5 not in found_md5:
                item.setCheckState(Qt.Checked)
                missing_count += 1
            else:
                item.setCheckState(Qt.Unchecked)

        total = self.image_list.count()
        self.log_msg(f"目录扫描完成：共发现 {len(found_md5)} 个输出md5，待处理 {total} 项，缺失 {missing_count} 项（已自动勾选）。")

    def log_msg(self, msg):
        self.log_text.append(msg)
        # 滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_progress(self):
        total = len(self.image_paths)
        if total == 0:
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("0/0")
            return
            
        processed = len(self.results)
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(processed)
        self.progress_bar.setFormat(f"{processed}/{total}")

    def start_processing(self):
        if not self.image_paths:
            QMessageBox.warning(self, "警告", "请先添加图片")
            return
            
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "警告", "请输入提示词")
            return
            
        self.processing = True
        self.start_btn.setEnabled(False)
        self.add_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self.retry_btn.setEnabled(False)
        self.retry_selected_btn.setEnabled(False)
        self.template_combo.setEnabled(False)
        self.prompt_edit.setEnabled(False)
        self.thread_spin.setEnabled(False)
        
        self.log_msg("开始批量处理任务...")
        self.launch_available_workers()

    def retry_failed(self):
        # 直接重试所有红色失败任务
        failed_paths = [path for path, status in self.results.items() if status == "error"]
        if not failed_paths:
            self.log_msg("没有失败的任务需要重试")
            return
            
        for path in failed_paths:
            if path in self.results:
                del self.results[path]
                
        self.update_progress()
        self.start_processing()

    def retry_selected_failed(self):
        failed_paths = []
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item.checkState() != Qt.Checked:
                continue
            path = item.data(Qt.UserRole)
            if self.results.get(path) == "error":
                failed_paths.append(path)

        if not failed_paths:
            self.log_msg("没有勾选的失败任务需要重试")
            return

        for path in failed_paths:
            if path in self.results:
                del self.results[path]
            self.update_item_status(path, Qt.white)
        self.log_msg(f"将重试 {len(failed_paths)} 个勾选失败任务")
        self.update_progress()
        self.start_processing()

    def _get_next_unprocessed_path(self):
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            path = item.data(Qt.UserRole)
            if path in self.results or path in self.active_workers:
                continue
            return path
        return None

    def launch_available_workers(self):
        if not self.processing:
            return

        max_workers = int(self.thread_spin.value())
        while len(self.active_workers) < max_workers:
            next_path = self._get_next_unprocessed_path()
            if not next_path:
                break

            # 更新UI状态
            for i in range(self.image_list.count()):
                item = self.image_list.item(i)
                if item.data(Qt.UserRole) == next_path:
                    item.setBackground(Qt.yellow)
                    self.image_list.scrollToItem(item)
                    break

            prompt = self.prompt_edit.toPlainText().strip()
            # 获取选中的画风指令
            selected_style_name = self.main_style_combo.currentText()
            styles_data = self.get_styles()
            active_instructions = styles_data.get(selected_style_name, "")

            worker = ImageEditWorker(
                next_path, prompt, 
                self.config_getter_func, 
                self.img_config_getter_func,
                style_instructions=active_instructions
            )
            worker.finished.connect(self.on_worker_finished)
            worker.error.connect(self.on_worker_error)
            worker.log.connect(self.log_msg)
            self.active_workers[next_path] = worker
            worker.start()

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

    def on_worker_finished(self, result_json, image_path):
        if image_path in self.active_workers:
            del self.active_workers[image_path]
        self.results[image_path] = "success"
        self.update_item_status(image_path, Qt.green)
        self.log_msg(f"处理成功: {os.path.basename(image_path)}")
        
        # 保存结果
        self.save_result(result_json, image_path)
        
        self.update_progress()
        self.launch_available_workers()

    def on_worker_error(self, error_msg, image_path):
        if image_path in self.active_workers:
            del self.active_workers[image_path]
        self.results[image_path] = "error"
        self.update_item_status(image_path, Qt.red)
        self.log_msg(f"处理失败: {os.path.basename(image_path)} - {error_msg}")
        
        self.update_progress()
        self.launch_available_workers()

    def update_item_status(self, image_path, color):
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item.data(Qt.UserRole) == image_path:
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
        self.start_btn.setEnabled(True)
        self.add_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.template_combo.setEnabled(True)
        self.prompt_edit.setEnabled(True)
        self.thread_spin.setEnabled(True)
        
        # 检查是否有失败的任务
        has_errors = any(status == "error" for status in self.results.values())
        self.retry_btn.setEnabled(has_errors)
        self.retry_selected_btn.setEnabled(has_errors)
        
        self.log_msg("批量处理任务完成！")
        if has_errors:
            self.log_msg("部分任务处理失败，可点击'重试失败任务'(重试全部红色)或'重试勾选失败项'。")
