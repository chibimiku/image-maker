import os
import json
import datetime
import re
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QPushButton, QTextEdit, QComboBox, QMessageBox, QFileDialog, QListWidget, QListWidgetItem, QAbstractItemView, QProgressBar, QSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal

from single_analyzer import WorkerThread, ImageGenWorkerThread

class BatchAnalyzerWidget(QWidget):
    quick_export_requested = pyqtSignal(list)

    def __init__(self, config_getter_func, img_config_getter_func, styles_getter_func, save_img_cfg_callback, ar_policy_getter_func=None, nsfw_default_getter_func=None, nsfw_changed_callback=None, booru_tag_limit_getter_func=None):
        super().__init__()
        self.get_text_config = config_getter_func
        self.get_img_config = img_config_getter_func
        self.get_styles = styles_getter_func
        self.save_img_cfg = save_img_cfg_callback
        self.get_ar_policy = ar_policy_getter_func
        self.get_nsfw_default = nsfw_default_getter_func
        self.on_nsfw_changed = nsfw_changed_callback
        self.get_booru_tag_limit = booru_tag_limit_getter_func
        
        self.target_directory = ""
        self.image_files = []
        self._active_threads = []
        self.current_index = 0
        self.current_run_total = 0
        self.current_run_images = []
        self.pending_images = []
        self.active_workers = {}
        self.next_worker_id = 1
        self.current_run_json_paths = []
        self.last_finished_json_paths = []
        self.failed_image_files = []
        
        self.initUI()
    
    def initUI(self):
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        layout = QVBoxLayout()
        
        # 拖拽提示区域
        drag_label = QLabel("📁 请将图片拖拽至此，或点击下方按钮选择目录")
        drag_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drag_label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 2px dashed #aaa; padding: 20px; font-size: 14px; }")
        drag_label.setMinimumHeight(100)
        layout.addWidget(drag_label)
        
        # 图片列表区域
        list_layout = QVBoxLayout()
        list_layout.addWidget(QLabel("已添加的图片:"))
        
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.image_list.setMinimumHeight(150)
        list_layout.addWidget(self.image_list)
        
        # 图片管理按钮
        manage_layout = QHBoxLayout()
        self.select_dir_btn = QPushButton("选择目录")
        self.select_dir_btn.clicked.connect(self.select_directory)
        self.clear_all_btn = QPushButton("清空列表")
        self.clear_all_btn.clicked.connect(self.clear_all_images)
        self.remove_selected_btn = QPushButton("移除选中")
        self.remove_selected_btn.clicked.connect(self.remove_selected_images)
        manage_layout.addWidget(self.select_dir_btn)
        manage_layout.addWidget(self.clear_all_btn)
        manage_layout.addWidget(self.remove_selected_btn)
        list_layout.addLayout(manage_layout)
        
        layout.addLayout(list_layout)
        
        # 批量处理选项
        options_layout = QVBoxLayout()
        refine_layout = QHBoxLayout()
        self.enable_refine_cb = QCheckBox("启用 refine 二次优化")
        self.enable_refine_cb.setChecked(False)
        refine_layout.addWidget(self.enable_refine_cb)
        refine_layout.addStretch()
        options_layout.addLayout(refine_layout)

        extra_prompt_layout = QVBoxLayout()
        extra_prompt_layout.addWidget(QLabel("请求附加 prompts（可选，识别时重点关注）:"))
        self.extra_llm_prompt_edit = QTextEdit()
        self.extra_llm_prompt_edit.setPlaceholderText("例如：请特别关注人物服饰材质、镜头视角与场景光源方向")
        self.extra_llm_prompt_edit.setMinimumHeight(72)
        extra_prompt_layout.addWidget(self.extra_llm_prompt_edit)
        options_layout.addLayout(extra_prompt_layout)
        
        # 自动生成图片选项
        auto_gen_layout = QHBoxLayout()
        self.auto_gen_orig_cb = QCheckBox("分析完成后生成图片（基于原始提示词）")
        self.auto_gen_ref_cb = QCheckBox("分析完成后生成图片（基于优化提示词）")
        auto_gen_layout.addWidget(self.auto_gen_orig_cb)
        auto_gen_layout.addWidget(self.auto_gen_ref_cb)
        options_layout.addLayout(auto_gen_layout)
        
        # 画风选择
        style_select_layout = QHBoxLayout()
        style_select_layout.addWidget(QLabel("生成时使用的画风预设:"))
        self.main_style_combo = QComboBox()
        style_select_layout.addWidget(self.main_style_combo, stretch=1)
        options_layout.addLayout(style_select_layout)
        nsfw_layout = QHBoxLayout()
        self.use_nsfw_cb = QCheckBox("使用nsfw接口")
        self.use_nsfw_cb.setChecked(bool(self.get_nsfw_default()) if self.get_nsfw_default else False)
        self.use_nsfw_cb.toggled.connect(self.on_use_nsfw_toggled)
        nsfw_layout.addWidget(self.use_nsfw_cb)
        nsfw_layout.addStretch()
        options_layout.addLayout(nsfw_layout)
        thread_layout = QHBoxLayout()
        thread_layout.addWidget(QLabel("批量并发线程数:"))
        self.concurrent_threads_spin = QSpinBox()
        self.concurrent_threads_spin.setRange(1, 16)
        self.concurrent_threads_spin.setValue(3)
        thread_layout.addWidget(self.concurrent_threads_spin)
        thread_layout.addStretch()
        options_layout.addLayout(thread_layout)

        
        layout.addLayout(options_layout)
        
        # 开始按钮
        self.start_btn = QPushButton("开始批量处理")
        self.start_btn.setFixedHeight(40)
        self.start_btn.clicked.connect(self.start_batch_processing)
        self.start_btn.setEnabled(False)
        layout.addWidget(self.start_btn)

        self.quick_export_btn = QPushButton("快捷切换到 JSON数据集导出（本次结果）")
        self.quick_export_btn.clicked.connect(self.trigger_quick_export)
        self.quick_export_btn.setEnabled(False)
        layout.addWidget(self.quick_export_btn)

        self.retry_failed_btn = QPushButton("重试失败文件")
        self.retry_failed_btn.clicked.connect(self.retry_failed_images)
        self.retry_failed_btn.setEnabled(False)
        layout.addWidget(self.retry_failed_btn)

        self.failed_list = QListWidget()
        self.failed_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.failed_list.setMinimumHeight(80)
        layout.addWidget(QLabel("失败文件列表:"))
        layout.addWidget(self.failed_list)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("进度: 0/0")
        layout.addWidget(self.progress_bar)
        
        # 日志区域
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        self.setLayout(layout)
    
    def update_styles(self, style_keys):
        """由外部 app.py 调用以同步最新的画风列表"""
        curr_main = self.main_style_combo.currentText()
        self.main_style_combo.blockSignals(True)
        self.main_style_combo.clear()
        self.main_style_combo.addItems(style_keys)
        if curr_main in style_keys:
            self.main_style_combo.setCurrentText(curr_main)
        self.main_style_combo.blockSignals(False)

    def select_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if directory:
            self.target_directory = directory
            
            # 扫描目录中的图片文件
            supported_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
            new_images = []
            for filename in os.listdir(directory):
                if filename.lower().endswith(supported_extensions):
                    img_path = os.path.join(directory, filename)
                    if img_path not in self.image_files:
                        new_images.append(img_path)
            
            if new_images:
                self.add_images_to_list(new_images)
                self.log_msg(f"从目录中添加了 {len(new_images)} 个图片文件")
            else:
                self.log_msg("目录中没有找到新的图片文件")
    
    def log_msg(self, text):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {text}")
        scrollbar = self.log_text.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            image_paths = []
            for url in urls:
                file_path = url.toLocalFile()
                if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                    image_paths.append(file_path)
            
            if image_paths:
                self.add_images_to_list(image_paths)
                self.log_msg(f"从拖拽添加了 {len(image_paths)} 个图片文件")
            else:
                self.log_msg("没有找到有效的图片文件")
    
    def add_images_to_list(self, image_paths):
        for img_path in image_paths:
            if img_path not in self.image_files:
                self.image_files.append(img_path)
                item = QListWidgetItem(os.path.basename(img_path))
                item.setData(Qt.ItemDataRole.UserRole, img_path)
                self.image_list.addItem(item)
        
        self.start_btn.setEnabled(len(self.image_files) > 0)
    
    def clear_all_images(self):
        self.image_files = []
        self.image_list.clear()
        self.start_btn.setEnabled(False)
        self.failed_image_files = []
        self.failed_list.clear()
        self.retry_failed_btn.setEnabled(False)
        self.reset_progress()
        self.log_msg("已清空图片列表")
    
    def remove_selected_images(self):
        selected_items = self.image_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "提示", "请先选择要移除的图片")
            return
        
        removed_count = 0
        for item in selected_items:
            candidate = item.data(Qt.ItemDataRole.UserRole)
            if candidate and candidate in self.image_files:
                self.image_files.remove(candidate)
                self.remove_failed_image(candidate)
                removed_count += 1
            self.image_list.takeItem(self.image_list.row(item))
        
        self.start_btn.setEnabled(len(self.image_files) > 0)
        if len(self.image_files) == 0:
            self.reset_progress()
        self.log_msg(f"已移除 {removed_count} 个选中的图片文件")
    
    def start_batch_processing(self, checked=False, target_images=None, is_retry=False):
        if target_images is None and isinstance(checked, (list, tuple, set)):
            target_images = list(checked)
        elif target_images is None and isinstance(checked, str) and checked.strip():
            target_images = [checked]
        if self.active_workers:
            QMessageBox.warning(self, "提示", "已有批量任务在运行中")
            return
        if target_images is None:
            if len(self.image_files) == 0:
                QMessageBox.warning(self, "错误", "请先添加图片文件")
                return
            refreshed_image_files = []
            for i in range(self.image_list.count()):
                item = self.image_list.item(i)
                if item:
                    image_path = item.data(Qt.ItemDataRole.UserRole)
                    if image_path:
                        refreshed_image_files.append(image_path)
            self.image_files = refreshed_image_files
            if len(self.image_files) == 0:
                QMessageBox.warning(self, "错误", "请先添加图片文件")
                return
            run_images = list(self.image_files)
            self.failed_image_files = []
            self.failed_list.clear()
        else:
            run_images = [str(path) for path in target_images if str(path).strip()]
            if len(run_images) == 0:
                QMessageBox.warning(self, "提示", "没有可重试的失败文件")
                return
        base_url, api_key, model_name = self.get_text_config(self.use_nsfw_cb.isChecked())
        if not api_key or not model_name:
            QMessageBox.warning(self, "缺少配置", "文本分析 API Key 和 模型名称不能为空！")
            return
        self.current_index = 0
        self.current_run_images = list(run_images)
        self.pending_images = list(run_images)
        self.current_run_total = len(self.current_run_images)
        self.active_workers = {}
        self.next_worker_id = 1
        self.current_run_json_paths = []
        self.start_btn.setEnabled(False)
        self.quick_export_btn.setEnabled(False)
        self.retry_failed_btn.setEnabled(False)
        self.set_manage_buttons_enabled(False)
        self.log_text.clear()
        self.update_progress()
        mode_text = "失败重试" if is_retry else "批量图片分析"
        self.log_msg(f"开始{mode_text}，共 {self.current_run_total} 个文件，并发线程数: {int(self.concurrent_threads_spin.value())}")
        self.dispatch_next_workers()

    def dispatch_next_workers(self):
        max_workers = int(self.concurrent_threads_spin.value())
        while self.pending_images and len(self.active_workers) < max_workers:
            image_path = self.pending_images.pop(0)
            worker_id = self.next_worker_id
            self.next_worker_id += 1
            base_url, api_key, model_name = self.get_text_config(self.use_nsfw_cb.isChecked())
            booru_tag_limit = int(self.get_booru_tag_limit()) if self.get_booru_tag_limit else 30
            extra_llm_prompt = self.extra_llm_prompt_edit.toPlainText().strip()
            thread = WorkerThread(
                image_path,
                api_key,
                base_url,
                model_name,
                enable_refine=self.enable_refine_cb.isChecked(),
                booru_tag_limit=booru_tag_limit,
                extra_llm_prompt=extra_llm_prompt
            )
            thread.log_signal.connect(lambda text, wid=worker_id: self.log_msg(f"[线程-{wid}] {text}"))
            thread.finish_signal.connect(lambda result, t=thread, wid=worker_id, path=image_path: self.on_worker_finished(t, wid, path, result))
            thread.start()
            self._active_threads.append(thread)
            self.active_workers[thread] = {"id": worker_id, "image_path": image_path}
            self.log_msg(f"[线程-{worker_id}] 开始处理: {os.path.basename(image_path)}")

    def on_worker_finished(self, thread, worker_id, image_path, result_json):
        if thread in self.active_workers:
            self.active_workers.pop(thread)
        if not result_json:
            self.log_msg(f"[线程-{worker_id}] ❌ 处理失败: {os.path.basename(image_path)}")
            self.add_failed_image(image_path)
        else:
            self.log_msg(f"[线程-{worker_id}] ✅ 分析完成: {os.path.basename(image_path)}")
            output_json_path = self.save_result(result_json, image_path)
            if output_json_path:
                self.current_run_json_paths.append(output_json_path)
            if self.auto_gen_orig_cb.isChecked() or self.auto_gen_ref_cb.isChecked():
                self.generate_images(result_json)
            self.remove_failed_image(image_path)
        self.current_index += 1
        self.update_progress()
        if self.current_index >= self.current_run_total and not self.pending_images and not self.active_workers:
            self.finish_batch_processing()
            return
        self.dispatch_next_workers()

    def finish_batch_processing(self):
        self.update_progress()
        self.log_msg("🎉 批量分析完成！")
        self.last_finished_json_paths = list(self.current_run_json_paths)
        has_outputs = len(self.last_finished_json_paths) > 0
        self.quick_export_btn.setEnabled(has_outputs)
        if has_outputs:
            self.log_msg(f"🧭 已生成 {len(self.last_finished_json_paths)} 个 JSON，可点击快捷入口继续导出数据集")
        if self.failed_image_files:
            self.log_msg(f"⚠️ 本轮失败 {len(self.failed_image_files)} 个文件，可点击“重试失败文件”继续处理")
            for failed_path in self.failed_image_files:
                self.log_msg(f" 失败文件: {os.path.basename(failed_path)}")
        self.retry_failed_btn.setEnabled(len(self.failed_image_files) > 0)
        self.start_btn.setEnabled(True)
        self.set_manage_buttons_enabled(True)

    def retry_failed_images(self):
        self.start_batch_processing(target_images=list(self.failed_image_files), is_retry=True)

    def update_progress(self):
        total = self.current_run_total
        current = min(self.current_index, total) if total > 0 else 0
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(current if total > 0 else 0)
        self.progress_bar.setFormat(f"进度: {current}/{total}")

    def reset_progress(self):
        self.current_index = 0
        self.current_run_total = 0
        self.current_run_images = []
        self.pending_images = []
        self.active_workers = {}
        self.update_progress()

    def set_manage_buttons_enabled(self, enabled):
        self.clear_all_btn.setEnabled(bool(enabled))
        self.remove_selected_btn.setEnabled(bool(enabled))

    def add_failed_image(self, image_path):
        if image_path in self.failed_image_files:
            return
        self.failed_image_files.append(image_path)
        item = QListWidgetItem(os.path.basename(image_path))
        item.setData(Qt.ItemDataRole.UserRole, image_path)
        self.failed_list.addItem(item)
        self.retry_failed_btn.setEnabled(len(self.failed_image_files) > 0)

    def remove_failed_image(self, image_path):
        if image_path in self.failed_image_files:
            self.failed_image_files.remove(image_path)
        for i in range(self.failed_list.count() - 1, -1, -1):
            item = self.failed_list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == image_path:
                self.failed_list.takeItem(i)
        self.retry_failed_btn.setEnabled(len(self.failed_image_files) > 0)
    
    def save_result(self, result_json, image_path):
        try:
            result_json["source_image_path"] = os.path.abspath(image_path)
            jp_title = result_json.get("japanese_title", "未命名")
            safe_title = re.sub(r'[\\/*?:"<>|]', "", jp_title).strip() or "未命名"
            
            now = datetime.datetime.now()
            date_str = now.strftime("%Y%m%d")
            save_dir = os.path.join('data', date_str, 'batch-result')
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # 生成唯一的文件名
            base_filename = f"{now.strftime('%Y%m%d-%H%M%S')}-{safe_title}"
            json_filename = f"{base_filename}.json"
            
            output_json_path = os.path.join(save_dir, json_filename)
            # 保存 JSON
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=4)
            
            # 处理长宽比
            raw_ar = result_json.get("aspect_ratio", "2:3")
            current_aspect_ratio = self._resolve_ar_for_first_stage(raw_ar)
            
            # 保存提示词
            current_refine_desc = result_json.get("english_description", "")
            current_orig_desc = result_json.get("original_english_description", "")
            
            selected_style_name = self.main_style_combo.currentText()
            styles_data = self.get_styles()
            current_fixed_tags = styles_data.get(selected_style_name, "")
            
            # 在风格标签和描述之间添加两个回车
            style_part = f"--ar {current_aspect_ratio} {current_fixed_tags}".strip()
            final_prompt = f"{style_part}\n\n{current_refine_desc}".strip()
            orig_prompt = f"{style_part}\n\n{current_orig_desc}".strip()
            
            txt_filename = f"{base_filename}-prompts.txt"
            orig_txt_filename = f"{base_filename}-original-prompts.txt"
            
            with open(os.path.join(save_dir, txt_filename), "w", encoding="utf-8") as f:
                f.write(final_prompt)
            with open(os.path.join(save_dir, orig_txt_filename), "w", encoding="utf-8") as f:
                f.write(orig_prompt)
            
            self.log_msg(f"📁 结果已保存至: {json_filename}")
            return output_json_path
            
        except Exception as e:
            self.log_msg(f"❌ 保存结果时出错: {e}")
            return ""

    def generate_images(self, result_json):
        self.save_img_cfg()
        
        img_base_url, img_key, model_name, api_type = self.get_img_config()
        if not img_key:
            self.log_msg("⚠️ 生图 API Key 为空，跳过图片生成")
            return
        
        selected_style_name = self.main_style_combo.currentText()
        styles_data = self.get_styles()
        active_instructions = styles_data.get(selected_style_name, "")
        
        raw_ar = result_json.get("aspect_ratio", "2:3")
        current_aspect_ratio = self._resolve_ar_for_first_stage(raw_ar)
        final_gen_ar = self._resolve_ar_for_second_stage(current_aspect_ratio)
        
        # 生成原始提示词图片
        if self.auto_gen_orig_cb.isChecked():
            orig_desc = result_json.get("original_english_description", "")
            if orig_desc:
                self.log_msg("📸 正在生成基于原始提示词的图片...")
                img_thread = ImageGenWorkerThread(
                    prompt=orig_desc,
                    model_name=model_name,
                    aspect_ratio=final_gen_ar,
                    instructions=active_instructions,
                    api_type=api_type
                )
                img_thread.log_signal.connect(self.log_msg)
                img_thread.finish_signal.connect(lambda files: self.on_image_generation_finished(files, "original"))
                img_thread.start()
                self._active_threads.append(img_thread)
        
        # 生成优化提示词图片
        if self.auto_gen_ref_cb.isChecked():
            refine_desc = result_json.get("english_description", "")
            if refine_desc:
                self.log_msg("📸 正在生成基于优化提示词的图片...")
                img_thread = ImageGenWorkerThread(
                    prompt=refine_desc,
                    model_name=model_name,
                    aspect_ratio=final_gen_ar,
                    instructions=active_instructions,
                    api_type=api_type
                )
                img_thread.log_signal.connect(self.log_msg)
                img_thread.finish_signal.connect(lambda files: self.on_image_generation_finished(files, "refined"))
                img_thread.start()
                self._active_threads.append(img_thread)
    
    def on_image_generation_finished(self, saved_files, prompt_type):
        if saved_files:
            self.log_msg(f"🎉 成功生成了 {len(saved_files)} 张 {prompt_type} 图片！")
            for file_path in saved_files:
                self.log_msg(f" 📂 保存路径: {file_path}")
        else:
            self.log_msg(f"⚠️ 未能生成 {prompt_type} 图片")
    
    def _resolve_ar_for_first_stage(self, original_ar: str) -> str:
        """第一次：分析完成后用于保存 prompts 的长宽比"""
        if not self.get_ar_policy:
            return original_ar
        policy = self.get_ar_policy() or {}
        override_first = (policy.get("override_first") or "").strip()
        if override_first.startswith("不覆盖"):
            return original_ar
        return override_first
    
    def _resolve_ar_for_second_stage(self, original_ar: str) -> str:
        """第二次：真正调用生图接口时的长宽比"""
        if not self.get_ar_policy:
            return original_ar
        policy = self.get_ar_policy() or {}
        override_second = (policy.get("override_second") or "").strip()
        if override_second.startswith("不覆盖"):
            return original_ar
        return override_second

    def on_use_nsfw_toggled(self, checked):
        if self.on_nsfw_changed:
            self.on_nsfw_changed(bool(checked))

    def set_use_nsfw_default(self, checked):
        self.use_nsfw_cb.blockSignals(True)
        self.use_nsfw_cb.setChecked(bool(checked))
        self.use_nsfw_cb.blockSignals(False)

    def trigger_quick_export(self):
        if not self.last_finished_json_paths:
            QMessageBox.warning(self, "提示", "还没有可用于导出的批量分析 JSON")
            return
        self.quick_export_requested.emit(list(self.last_finished_json_paths))
