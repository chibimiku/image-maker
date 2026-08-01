import os
import json
import datetime
import re
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QPushButton, QTextEdit, QComboBox, QMessageBox, QFileDialog, QListWidget, QListWidgetItem, QAbstractItemView, QProgressBar, QSpinBox, QDoubleSpinBox, QCompleter)
from PyQt6.QtCore import Qt, pyqtSignal, QStringListModel

from modules.image_analysis.single_analyzer import WorkerThread, ImageGenWorkerThread, get_single_analyzer_missing_prompt_files
from utils.task_runtime import SystemNotifier, TaskCountdown
from utils.image_upscale_runtime import JpgAutoUpscaleThread, list_esrgan_models, normalize_upscale_options
from utils.styles import style_prompt, style_ref_image, ref_image_valid, build_ref_gen_params
from utils.style_ref_widget import StyleRefModeCombo

class BatchAnalyzerWidget(QWidget):
    quick_export_requested = pyqtSignal(list)

    def __init__(self, config_getter_func, img_config_getter_func, styles_getter_func, save_img_cfg_callback, ar_policy_getter_func=None, nsfw_default_getter_func=None, nsfw_changed_callback=None, booru_tag_limit_getter_func=None, timeout_getter_func=None, upscale_options_getter_func=None, upscale_options_changed_callback=None, outfit_check_default_getter_func=None, outfit_check_changed_callback=None, outfit_style_history_getter_func=None, outfit_style_default_getter_func=None, outfit_style_changed_callback=None, outfit_style_delete_callback=None):
        super().__init__()
        self.get_text_config = config_getter_func
        self.get_img_config = img_config_getter_func
        self.get_styles = styles_getter_func
        self.save_img_cfg = save_img_cfg_callback
        self.get_ar_policy = ar_policy_getter_func
        self.get_nsfw_default = nsfw_default_getter_func
        self.on_nsfw_changed = nsfw_changed_callback
        self.get_booru_tag_limit = booru_tag_limit_getter_func
        self.get_timeout_seconds = timeout_getter_func
        self.get_upscale_options = upscale_options_getter_func
        self.on_upscale_options_changed = upscale_options_changed_callback
        self.get_outfit_check_default = outfit_check_default_getter_func
        self.on_outfit_check_changed = outfit_check_changed_callback
        self.get_outfit_style_history = outfit_style_history_getter_func
        self.get_outfit_style_default = outfit_style_default_getter_func
        self.on_outfit_style_changed = outfit_style_changed_callback
        self.on_outfit_style_deleted = outfit_style_delete_callback
        
        self.target_directory = ""
        self.image_files = []
        self._active_threads = []
        self.current_index = 0
        self.current_run_total = 0
        self.current_run_images = []
        self.pending_images = []
        self.active_workers = {}
        self._active_image_threads = []
        self.next_worker_id = 1
        self.current_run_json_paths = []
        self.last_finished_json_paths = []
        self.failed_image_files = []
        self.cancel_soft_requested = False
        self.cancel_hard_requested = False
        self.current_run_timeout_count = 0
        self.batch_run_state = "idle"
        self._notifier = SystemNotifier(self)
        self._img_gen_running = False
        self._img_gen_timeout_seconds = 0
        self._img_gen_countdown = TaskCountdown(
            parent=self,
            on_tick=self._on_image_gen_countdown_tick,
            on_timeout=lambda: self.cancel_image_generation(reason="timeout")
        )
        self._auto_gen_expected = 0
        self._auto_gen_finished = 0
        self._auto_gen_cancelled = False
        self._pending_finish_reason = "completed"
        self._pending_completion_notice = False
        self._active_post_threads = []
        self._updating_outfit_style_combo = False
        self._outfit_style_history_cache = []
        
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
        self.image_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
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
        self.enable_outfit_check_cb = QCheckBox("服装搭配检查")
        self.enable_outfit_check_cb.setToolTip("分析出 prompts 后，再额外检查人物服装搭配是否协调；若修订，仅覆盖 prompts 相关字段并保留原值备用。")
        self.enable_outfit_check_cb.setChecked(bool(self.get_outfit_check_default()) if self.get_outfit_check_default else False)
        self.enable_outfit_check_cb.toggled.connect(self.on_enable_outfit_check_toggled)
        refine_layout.addWidget(self.enable_outfit_check_cb)
        refine_layout.addStretch()
        options_layout.addLayout(refine_layout)

        outfit_style_layout = QHBoxLayout()
        outfit_style_layout.addWidget(QLabel("服装风格覆盖:"))
        self.outfit_style_combo = QComboBox()
        self.outfit_style_combo.setEditable(True)
        self.outfit_style_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.outfit_style_completer_model = QStringListModel(self)
        self.outfit_style_completer = QCompleter(self.outfit_style_completer_model, self)
        self.outfit_style_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.outfit_style_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.outfit_style_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self.outfit_style_combo.setCompleter(self.outfit_style_completer)
        if self.outfit_style_combo.lineEdit() is not None:
            self.outfit_style_combo.lineEdit().setPlaceholderText("留空则不覆盖，例如：维多利亚风格")
            self.outfit_style_combo.lineEdit().setClearButtonEnabled(True)
            self.outfit_style_combo.lineEdit().editingFinished.connect(self._commit_outfit_style_from_editor)
            self.outfit_style_combo.lineEdit().textEdited.connect(self._filter_outfit_style_history_live)
        self.outfit_style_combo.currentTextChanged.connect(self._on_outfit_style_text_changed)
        self.outfit_style_combo.activated.connect(lambda _idx: self._commit_outfit_style_text(self.outfit_style_combo.currentText(), add_to_history=False))
        outfit_style_layout.addWidget(self.outfit_style_combo, stretch=1)
        self.delete_outfit_style_btn = QPushButton("x")
        self.delete_outfit_style_btn.setFixedWidth(28)
        self.delete_outfit_style_btn.setToolTip("删除当前历史项")
        self.delete_outfit_style_btn.clicked.connect(self._delete_current_outfit_style_history)
        outfit_style_layout.addWidget(self.delete_outfit_style_btn)
        options_layout.addLayout(outfit_style_layout)

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

        # 保存到原图同目录
        self.save_to_source_dir_cb = QCheckBox("分析结果保存到原图同目录（不生成图片）")
        self.save_to_source_dir_cb.setToolTip(
            "勾选后，每个图片的分析结果 JSON 和 TXT 将保存到各自原图所在目录，\n"
            "并同步文件修改时间。此时只分析不生成图片，JSON 可用于直接拖入投稿 Server。"
        )
        self.save_to_source_dir_cb.toggled.connect(self._on_save_to_source_dir_toggled)
        options_layout.addWidget(self.save_to_source_dir_cb)

        upscale_layout = QHBoxLayout()
        self.enable_jpg_upscale_cb = QCheckBox("生图后自动处理 JPG")
        self.enable_jpg_upscale_cb.toggled.connect(self._persist_upscale_options)
        upscale_layout.addWidget(self.enable_jpg_upscale_cb)
        upscale_layout.addWidget(QLabel("模型:"))
        self.upscale_model_combo = QComboBox()
        self.upscale_model_combo.currentTextChanged.connect(self._persist_upscale_options)
        upscale_layout.addWidget(self.upscale_model_combo)
        self.reload_upscale_models_btn = QPushButton("刷新模型")
        self.reload_upscale_models_btn.clicked.connect(self._reload_upscale_models)
        upscale_layout.addWidget(self.reload_upscale_models_btn)
        upscale_layout.addWidget(QLabel("倍率:"))
        self.upscale_by_spin = QDoubleSpinBox()
        self.upscale_by_spin.setRange(1.0, 8.0)
        self.upscale_by_spin.setSingleStep(0.1)
        self.upscale_by_spin.setValue(2.0)
        self.upscale_by_spin.valueChanged.connect(self._persist_upscale_options)
        upscale_layout.addWidget(self.upscale_by_spin)
        upscale_layout.addWidget(QLabel("WebP目标MB:"))
        self.webp_target_mb_spin = QDoubleSpinBox()
        self.webp_target_mb_spin.setRange(0.1, 100.0)
        self.webp_target_mb_spin.setDecimals(1)
        self.webp_target_mb_spin.setSingleStep(0.5)
        self.webp_target_mb_spin.setValue(10.0)
        self.webp_target_mb_spin.valueChanged.connect(self._persist_upscale_options)
        upscale_layout.addWidget(self.webp_target_mb_spin)
        upscale_layout.addStretch()
        options_layout.addLayout(upscale_layout)
        
        # 画风选择
        style_select_layout = QHBoxLayout()
        style_select_layout.addWidget(QLabel("生成时使用的画风预设:"))
        self.main_style_combo = QComboBox()
        style_select_layout.addWidget(self.main_style_combo, stretch=1)
        style_select_layout.addWidget(QLabel("参考模式:"))
        self.style_ref_mode_combo = StyleRefModeCombo(self)
        self.style_ref_mode_combo.setMaximumWidth(130)
        style_select_layout.addWidget(self.style_ref_mode_combo)
        self.main_style_combo.currentTextChanged.connect(self._on_style_changed)
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
        self.cancel_btn = QPushButton("取消当前任务")
        self.cancel_btn.setFixedHeight(36)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.request_cancel_batch)
        layout.addWidget(self.cancel_btn)
        gen_ctrl_layout = QHBoxLayout()
        self.gen_countdown_label = QLabel("生图超时倒计时: --")
        self.cancel_gen_btn = QPushButton("终止当前生图")
        self.cancel_gen_btn.setEnabled(False)
        self.cancel_gen_btn.clicked.connect(self.cancel_image_generation)
        gen_ctrl_layout.addWidget(self.gen_countdown_label)
        gen_ctrl_layout.addStretch()
        gen_ctrl_layout.addWidget(self.cancel_gen_btn)
        layout.addLayout(gen_ctrl_layout)

        self.quick_export_btn = QPushButton("快捷切换到 JSON数据集导出（本次结果）")
        self.quick_export_btn.clicked.connect(self.trigger_quick_export)
        self.quick_export_btn.setEnabled(False)
        layout.addWidget(self.quick_export_btn)

        self.retry_failed_btn = QPushButton("重试失败文件")
        self.retry_failed_btn.clicked.connect(self.retry_failed_images)
        self.retry_failed_btn.setEnabled(False)
        layout.addWidget(self.retry_failed_btn)

        self.failed_list = QListWidget()
        self.failed_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
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
        self._reload_upscale_models()
        self.set_upscale_options_defaults(self.get_upscale_options() if self.get_upscale_options else {})
        self.set_outfit_style_options(
            self.get_outfit_style_history() if self.get_outfit_style_history else [],
            self.get_outfit_style_default() if self.get_outfit_style_default else ""
        )

    def _send_system_notification(self, title, message):
        self._notifier.notify(title, message)
    
    def update_styles(self, style_keys):
        """由外部 app.py 调用以同步最新的画风列表"""
        curr_main = self.main_style_combo.currentText()
        self.main_style_combo.blockSignals(True)
        self.main_style_combo.clear()
        self.main_style_combo.addItems(style_keys)
        if curr_main in style_keys:
            self.main_style_combo.setCurrentText(curr_main)
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
        self._commit_outfit_style_text(self.outfit_style_combo.currentText(), add_to_history=True)
        enable_outfit_check = self.enable_outfit_check_cb.isChecked()
        # 批量分析未提供「去除照片风格」选项；当启用服装搭配检查时，Step 5 会重新计算 pixiv_tags
        enable_recompute_pixiv_tags = enable_outfit_check
        missing_prompt_files = get_single_analyzer_missing_prompt_files(
            enable_refine=self.enable_refine_cb.isChecked(),
            enable_outfit_check=enable_outfit_check,
            enable_recompute_pixiv_tags=enable_recompute_pixiv_tags,
        )
        if missing_prompt_files:
            missing_text = "\n".join(missing_prompt_files)
            QMessageBox.warning(self, "缺少 Prompt 文件", f"以下 Prompt 文件不存在，请补齐后再执行：\n{missing_text}")
            self.log_msg(f"❌ 缺少 Prompt 文件，已中止批量分析：\n{missing_text}")
            return
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
        self.current_run_timeout_count = 0
        self.cancel_soft_requested = False
        self.cancel_hard_requested = False
        self.batch_run_state = "running"
        self._auto_gen_expected = 0
        self._auto_gen_finished = 0
        self._auto_gen_cancelled = False
        self._pending_finish_reason = "completed"
        self._pending_completion_notice = False
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setText("取消当前任务")
        self.cancel_gen_btn.setEnabled(False)
        self.gen_countdown_label.setText("生图超时倒计时: --")
        self.quick_export_btn.setEnabled(False)
        self.retry_failed_btn.setEnabled(False)
        self.set_manage_buttons_enabled(False)
        self.log_text.clear()
        timeout_seconds = int(self.get_timeout_seconds()) if self.get_timeout_seconds else 120
        submit_time = datetime.datetime.now()
        ddl = submit_time + datetime.timedelta(seconds=max(1, timeout_seconds))
        self.log_msg(f"提交时间: {submit_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_msg(f"超时设置: {timeout_seconds} 秒（预计超时点: {ddl.strftime('%H:%M:%S')}）")
        self.update_progress()
        mode_text = "失败重试" if is_retry else "批量图片分析"
        self.log_msg(f"开始{mode_text}，共 {self.current_run_total} 个文件，并发线程数: {int(self.concurrent_threads_spin.value())}")
        self.dispatch_next_workers()

    def dispatch_next_workers(self):
        if self.cancel_soft_requested or self.cancel_hard_requested or self.batch_run_state != "running":
            return
        max_workers = int(self.concurrent_threads_spin.value())
        while self.pending_images and len(self.active_workers) < max_workers:
            image_path = self.pending_images.pop(0)
            worker_id = self.next_worker_id
            self.next_worker_id += 1
            base_url, api_key, model_name = self.get_text_config(self.use_nsfw_cb.isChecked())
            booru_tag_limit = int(self.get_booru_tag_limit()) if self.get_booru_tag_limit else 30
            timeout_seconds = int(self.get_timeout_seconds()) if self.get_timeout_seconds else 120
            extra_llm_prompt = self.extra_llm_prompt_edit.toPlainText().strip()
            thread = WorkerThread(
                image_path,
                api_key,
                base_url,
                model_name,
                enable_refine=self.enable_refine_cb.isChecked(),
                booru_tag_limit=booru_tag_limit,
                extra_llm_prompt=extra_llm_prompt,
                timeout_seconds=timeout_seconds,
                enable_outfit_check=self.enable_outfit_check_cb.isChecked(),
                outfit_style_override=self.outfit_style_combo.currentText().strip()
            )
            thread.log_signal.connect(lambda text, wid=worker_id: self.log_msg(f"[线程-{wid}] {text}"))
            thread.finish_signal.connect(lambda result, t=thread, wid=worker_id, path=image_path: self.on_worker_finished(t, wid, path, result))
            thread.finished.connect(lambda t=thread, wid=worker_id, path=image_path: self._on_worker_thread_finished(t, wid, path))
            thread.start()
            self._active_threads.append(thread)
            self.active_workers[thread] = {"id": worker_id, "image_path": image_path}
            self.log_msg(f"[线程-{worker_id}] 开始处理: {os.path.basename(image_path)}")

    def on_worker_finished(self, thread, worker_id, image_path, result_json):
        # 清理线程引用
        if thread in self._active_threads:
            self._active_threads.remove(thread)
        if thread in self.active_workers:
            self.active_workers.pop(thread)
        elif self.batch_run_state != "running":
            return
        worker_status = getattr(thread, "last_status", "unknown")
        if not result_json:
            if worker_status == "timeout":
                self.current_run_timeout_count += 1
                self.log_msg(f"[线程-{worker_id}] ⏰ 请求超时: {os.path.basename(image_path)}")
            elif worker_status == "cancelled":
                self.log_msg(f"[线程-{worker_id}] 🛑 已取消: {os.path.basename(image_path)}")
            else:
                self.log_msg(f"[线程-{worker_id}] ❌ 处理失败: {os.path.basename(image_path)}")
            self.add_failed_image(image_path)
        else:
            self.log_msg(f"[线程-{worker_id}] ✅ 分析完成: {os.path.basename(image_path)}")
            output_json_path = self.save_result(result_json, image_path)
            if output_json_path:
                self.current_run_json_paths.append(output_json_path)
            if (self.auto_gen_orig_cb.isChecked() or self.auto_gen_ref_cb.isChecked()) and not self.save_to_source_dir_cb.isChecked():
                self.generate_images(result_json, analysis_json_path=output_json_path)
            self.remove_failed_image(image_path)
        self.current_index += 1
        self.update_progress()
        if self.current_index >= self.current_run_total and not self.pending_images and not self.active_workers:
            final_reason = "cancelled" if (self.cancel_soft_requested or self.cancel_hard_requested) else "completed"
            self.finish_batch_processing(final_reason=final_reason)
            return
        if self.cancel_soft_requested and not self.pending_images and not self.active_workers:
            self.finish_batch_processing(final_reason="cancelled")
            return
        self.dispatch_next_workers()

    def _on_worker_thread_finished(self, thread, worker_id, image_path):
        """兜底清理：线程退出时确保线程引用被正确清除。
        正常情况下 on_worker_finished 已通过 finish_signal 完成处理，
        此方法作为 QThread.finished 的兜底，处理 finish_signal 丢失的极端情况。"""
        # 清理 _active_threads 引用
        if thread in self._active_threads:
            self._active_threads.remove(thread)

        # 如果 on_worker_finished 已经处理过（线程已不在 active_workers 中），无需额外操作
        if thread not in self.active_workers:
            return

        # finish_signal 丢失，需要兜底处理
        self.active_workers.pop(thread)
        image_name = os.path.basename(image_path)
        worker_status = getattr(thread, "last_status", "unknown")
        self.log_msg(f"[线程-{worker_id}] ⚠️ 兜底：finish_signal 丢失，线程已退出（状态: {worker_status}），正在清理: {image_name}")

        # 根据线程状态处理
        if worker_status == "success":
            self.remove_failed_image(image_path)
        else:
            if worker_status == "timeout":
                self.current_run_timeout_count += 1
            self.add_failed_image(image_path)

        self.current_index += 1
        self.update_progress()

        # 检查是否所有任务已完成
        if self.current_index >= self.current_run_total and not self.pending_images and not self.active_workers:
            final_reason = "cancelled" if (self.cancel_soft_requested or self.cancel_hard_requested) else "completed"
            self.finish_batch_processing(final_reason=final_reason)
        elif self.cancel_soft_requested and not self.pending_images and not self.active_workers:
            self.finish_batch_processing(final_reason="cancelled")
        elif self.batch_run_state == "running":
            self.dispatch_next_workers()

    def finish_batch_processing(self, final_reason="completed"):
        self.batch_run_state = "idle"
        self._pending_finish_reason = final_reason
        self.update_progress()
        if final_reason == "cancelled":
            self.log_msg("🛑 批量任务已取消。")
            if self._active_image_threads:
                self.cancel_image_generation(reason="manual")
            self._send_system_notification("批量任务已取消", "批量图片分析已结束（用户取消）。")
        else:
            self.log_msg("🎉 批量分析完成！")
            if self._auto_gen_expected > 0 and self._auto_gen_finished < self._auto_gen_expected:
                self._pending_completion_notice = True
                self.log_msg("🕒 分析阶段已完成，正在等待自动生图全部结束后发送完成通知。")
            else:
                self._send_system_notification("批量任务完成", "批量图片分析已全部结束。")
        self.last_finished_json_paths = list(self.current_run_json_paths)
        has_outputs = len(self.last_finished_json_paths) > 0
        self.quick_export_btn.setEnabled(has_outputs)
        if has_outputs:
            self.log_msg(f"🧭 已生成 {len(self.last_finished_json_paths)} 个 JSON，可点击快捷入口继续导出数据集")
        if self.failed_image_files:
            self.log_msg(f"⚠️ 本轮失败 {len(self.failed_image_files)} 个文件，可点击“重试失败文件”继续处理")
            for failed_path in self.failed_image_files:
                self.log_msg(f" 失败文件: {os.path.basename(failed_path)}")
        if self.current_run_timeout_count > 0:
            self.log_msg(f"⏰ 本轮超时 {self.current_run_timeout_count} 个任务，请检查超时设置是否生效。")
            self._send_system_notification("批量任务发生超时", f"本轮共有 {self.current_run_timeout_count} 个任务超时。")
        self.retry_failed_btn.setEnabled(len(self.failed_image_files) > 0)
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("取消当前任务")
        if not self._active_image_threads:
            self.cancel_gen_btn.setEnabled(False)
            self.gen_countdown_label.setText("生图超时倒计时: --")
        self.set_manage_buttons_enabled(True)

    def request_cancel_batch(self):
        if self.batch_run_state != "running":
            self.log_msg("当前没有正在运行的批量任务。")
            return
        if not self.cancel_soft_requested:
            self.cancel_soft_requested = True
            self.pending_images = []
            self.cancel_btn.setText("强制取消（再次点击）")
            self.log_msg("已请求取消：不再派发新任务，将等待当前运行中的任务完成。再次点击将强制取消当前任务。")
            if not self.active_workers:
                self.finish_batch_processing(final_reason="cancelled")
            return
        if self.cancel_hard_requested:
            return
        self.cancel_hard_requested = True
        self.log_msg("收到二次取消，正在请求停止当前运行中的任务，并等待线程自行退出...")
        running_threads = list(self.active_workers.keys())
        for thread in running_threads:
            try:
                if hasattr(thread, "request_cancel"):
                    thread.request_cancel(force=True)
                else:
                    thread.requestInterruption()
            except Exception:
                pass
        self.pending_images = []
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("等待线程退出...")
        if not self.active_workers:
            self.finish_batch_processing(final_reason="cancelled")

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

            # 确定保存目录和基础文件名（支持保存到原图同目录）
            save_to_source = self.save_to_source_dir_cb.isChecked()
            if save_to_source:
                save_dir = os.path.dirname(os.path.abspath(image_path))
                source_basename = os.path.basename(image_path)
                source_key = os.path.splitext(source_basename)[0].split("_")[0]
                base_filename = f"{now.strftime('%Y%m%d-%H%M%S')}-{source_key}-{safe_title}"
            else:
                save_dir = os.path.join('data', date_str, 'batch-result')
                base_filename = f"{now.strftime('%Y%m%d-%H%M%S')}-{safe_title}"
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # 生成唯一的文件名
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
            current_fixed_tags = style_prompt(styles_data, selected_style_name)
            
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
            
            # 保存到原图同目录时，同步文件修改时间
            if save_to_source:
                src_stat = os.stat(image_path)
                for fpath in (output_json_path,
                              os.path.join(save_dir, txt_filename),
                              os.path.join(save_dir, orig_txt_filename)):
                    if os.path.exists(fpath):
                        try:
                            os.utime(fpath, (src_stat.st_atime, src_stat.st_mtime))
                        except Exception:
                            pass
            
            self.log_msg(f"📁 结果已保存至: {json_filename}")
            return output_json_path
            
        except Exception as e:
            self.log_msg(f"❌ 保存结果时出错: {e}")
            return ""

    def generate_images(self, result_json, analysis_json_path=None):
        self.save_img_cfg()
        
        img_base_url, img_key, model_name, api_type = self.get_img_config()
        if not img_key:
            self.log_msg("⚠️ 生图 API Key 为空，跳过图片生成")
            return
        
        selected_style_name = self.main_style_combo.currentText()
        styles_data = self.get_styles() or {}
        has_ref = ref_image_valid(style_ref_image(styles_data, selected_style_name))
        active_mode = self.style_ref_mode_combo.effective_mode(has_ref)
        active_instructions, post_instructions, style_ref_paths = build_ref_gen_params(
            styles_data, selected_style_name, active_mode
        )
        
        raw_ar = result_json.get("aspect_ratio", "2:3")
        current_aspect_ratio = self._resolve_ar_for_first_stage(raw_ar)
        final_gen_ar = self._resolve_ar_for_second_stage(current_aspect_ratio)
        timeout_seconds = int(self.get_timeout_seconds()) if self.get_timeout_seconds else 120
        
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
                    api_type=api_type,
                    image_paths=style_ref_paths,
                    post_instructions=post_instructions
                )
                img_thread.meta_prompt_type = "original"
                img_thread.meta_analysis_json_path = analysis_json_path
                img_thread.log_signal.connect(self.log_msg)
                img_thread.finish_signal.connect(lambda files, t=img_thread: self.on_image_generation_finished(t, files))
                img_thread.finished.connect(lambda t=img_thread: self._on_image_thread_stopped(t))
                img_thread.start()
                self._active_threads.append(img_thread)
                self._active_image_threads.append(img_thread)
                self._auto_gen_expected += 1
                self._start_image_gen_runtime(timeout_seconds)
        
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
                    api_type=api_type,
                    image_paths=style_ref_paths,
                    post_instructions=post_instructions
                )
                img_thread.meta_prompt_type = "refined"
                img_thread.meta_analysis_json_path = analysis_json_path
                img_thread.log_signal.connect(self.log_msg)
                img_thread.finish_signal.connect(lambda files, t=img_thread: self.on_image_generation_finished(t, files))
                img_thread.finished.connect(lambda t=img_thread: self._on_image_thread_stopped(t))
                img_thread.start()
                self._active_threads.append(img_thread)
                self._active_image_threads.append(img_thread)
                self._auto_gen_expected += 1
                self._start_image_gen_runtime(timeout_seconds)
    
    def _start_image_gen_runtime(self, timeout_seconds):
        self._img_gen_running = True
        self._img_gen_timeout_seconds = max(1, int(timeout_seconds))
        self.cancel_gen_btn.setEnabled(True)
        self.log_msg(f"生图超时设置: {self._img_gen_timeout_seconds} 秒")
        self._img_gen_countdown.start(self._img_gen_timeout_seconds)

    def _stop_image_gen_runtime(self):
        self._img_gen_running = False
        self._img_gen_timeout_seconds = 0
        self._img_gen_countdown.stop()
        self.cancel_gen_btn.setEnabled(False)
        self.gen_countdown_label.setText("生图超时倒计时: --")

    def _on_image_gen_countdown_tick(self, remain_seconds):
        if not self._img_gen_running:
            self.gen_countdown_label.setText("生图超时倒计时: --")
            return
        remain = int(remain_seconds)
        if remain <= 0:
            self.gen_countdown_label.setText("生图超时倒计时: 0 秒")
            self.log_msg("⏰ 生图超时倒计时已到，正在终止当前生图任务...")
            return
        self.gen_countdown_label.setText(f"生图超时倒计时: {remain} 秒")

    def _on_image_thread_stopped(self, thread):
        if thread in self._active_threads:
            self._active_threads.remove(thread)
        if thread in self._active_image_threads:
            self._active_image_threads.remove(thread)
        if not self._active_image_threads:
            self._stop_image_gen_runtime()
            if self._pending_completion_notice and self._pending_finish_reason == "completed":
                self._pending_completion_notice = False
                self._send_system_notification("批量任务完成", "批量分析与自动生图均已完成。")

    def cancel_image_generation(self, reason="manual"):
        if not self._active_image_threads:
            self.log_msg("当前没有正在执行的生图任务。")
            return
        self._auto_gen_cancelled = True
        running_threads = list(self._active_image_threads)
        self.log_msg(f"正在请求终止 {len(running_threads)} 个生图任务，等待线程自行退出...")
        self._img_gen_running = False
        self._img_gen_timeout_seconds = 0
        self._img_gen_countdown.stop()
        self.cancel_gen_btn.setEnabled(False)
        self.gen_countdown_label.setText("生图超时倒计时: 正在退出...")
        for thread in running_threads:
            try:
                if hasattr(thread, "request_cancel"):
                    thread.request_cancel()
                else:
                    thread.requestInterruption()
            except Exception:
                pass
        if reason == "timeout":
            self.log_msg("⏰ 已发出生图超时终止请求，等待当前任务退出。")
            self._send_system_notification("批量生图超时", "自动生图任务已超时并终止。")
        else:
            self.log_msg("🛑 已发出生图终止请求，等待当前任务退出。")
            self._send_system_notification("批量生图已终止", "自动生图任务已手动取消。")

    def on_image_generation_finished(self, thread, saved_files):
        prompt_type = getattr(thread, "meta_prompt_type", "unknown")
        self._auto_gen_finished += 1
        if saved_files:
            self.log_msg(f"🎉 成功生成了 {len(saved_files)} 张 {prompt_type} 图片！")
            for file_path in saved_files:
                self.log_msg(f" 📂 保存路径: {file_path}")
            self._sync_analysis_mtimes(thread, saved_files)
            self._start_jpg_postprocess(saved_files, prompt_type)
        else:
            status = getattr(thread, "last_status", "unknown")
            if status == "cancelled":
                self.log_msg(f"🛑 已取消 {prompt_type} 图片生成")
            else:
                self.log_msg(f"⚠️ 未能生成 {prompt_type} 图片")

    def _sync_analysis_mtimes(self, thread, saved_files):
        """将分析文件（json、两个txt）的修改时间同步到与生成的第一张图片一致"""
        analysis_json_path = getattr(thread, "meta_analysis_json_path", None)
        if not analysis_json_path:
            self.log_msg("⏭ 同步 mtime 跳过：未关联分析 JSON 路径（meta_analysis_json_path 为空）")
            return
        if not saved_files:
            self.log_msg("⏭ 同步 mtime 跳过：saved_files 为空，无生成图片")
            return

        first_img = saved_files[0]
        self.log_msg(f"🕐 开始同步分析文件 mtime，基准图片: {os.path.basename(first_img)}")
        try:
            if not os.path.exists(first_img):
                self.log_msg(f"⚠️ 同步 mtime 跳过：图片文件不存在 {first_img}")
                return
            if not os.path.exists(analysis_json_path):
                self.log_msg(f"⚠️ 同步 mtime 跳过：分析 JSON 文件不存在 {analysis_json_path}")
                return

            img_mtime = os.stat(first_img).st_mtime
            img_mtime_str = datetime.datetime.fromtimestamp(img_mtime).strftime("%H:%M:%S")
            self.log_msg(f"  基准图片 mtime: {img_mtime_str}")

            synced = 0
            for fpath, label in ((analysis_json_path, "JSON"),
                                  (analysis_json_path.rsplit(".json", 1)[0] + "-prompts.txt", "优化提示词"),
                                  (analysis_json_path.rsplit(".json", 1)[0] + "-original-prompts.txt", "原始提示词")):
                if os.path.exists(fpath):
                    old_mtime = os.stat(fpath).st_mtime
                    old_str = datetime.datetime.fromtimestamp(old_mtime).strftime("%H:%M:%S")
                    os.utime(fpath, (os.stat(fpath).st_atime, img_mtime))
                    synced += 1
                    self.log_msg(f"  ✓ {label} : {old_str} → {img_mtime_str}  ({os.path.basename(fpath)})")
                else:
                    self.log_msg(f"  ✗ {label} 文件不存在，跳过 ({fpath})")

            self.log_msg(f"🕐 同步完成: {synced}/3 个文件已同步 mtime")
        except Exception as e:
            self.log_msg(f"⚠️ 同步分析文件 mtime 失败: {e}")
    
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

    def on_enable_outfit_check_toggled(self, checked):
        if self.on_outfit_check_changed:
            self.on_outfit_check_changed(bool(checked))

    def _on_save_to_source_dir_toggled(self, checked):
        """保存到原图同目录时，禁用自动生图选项"""
        if checked:
            self.auto_gen_orig_cb.setEnabled(False)
            self.auto_gen_ref_cb.setEnabled(False)
        else:
            self.auto_gen_orig_cb.setEnabled(True)
            self.auto_gen_ref_cb.setEnabled(True)

    def set_use_nsfw_default(self, checked):
        self.use_nsfw_cb.blockSignals(True)
        self.use_nsfw_cb.setChecked(bool(checked))
        self.use_nsfw_cb.blockSignals(False)

    def set_outfit_check_default(self, checked):
        self.enable_outfit_check_cb.blockSignals(True)
        self.enable_outfit_check_cb.setChecked(bool(checked))
        self.enable_outfit_check_cb.blockSignals(False)

    def set_outfit_style_options(self, history_items, current_text=""):
        items = []
        for item in (history_items or []):
            text = str(item or "").strip()
            if text and text not in items:
                items.append(text)
        current_text = str(current_text or "").strip()
        self._outfit_style_history_cache = list(items)
        self._apply_outfit_style_combo_items(items, current_text)

    def _apply_outfit_style_combo_items(self, items, current_text):
        self._updating_outfit_style_combo = True
        self.outfit_style_combo.blockSignals(True)
        self.outfit_style_combo.clear()
        self.outfit_style_combo.addItems(items)
        self.outfit_style_combo.setCurrentText(current_text)
        self.outfit_style_combo.blockSignals(False)
        self._updating_outfit_style_combo = False
        self.outfit_style_completer_model.setStringList(list(items))
        self._refresh_outfit_style_delete_btn()

    def _filter_outfit_style_history_live(self, text):
        if self._updating_outfit_style_combo:
            return
        keyword = str(text or "").strip().lower()
        if keyword:
            filtered_items = [item for item in self._outfit_style_history_cache if keyword in item.lower()]
        else:
            filtered_items = list(self._outfit_style_history_cache)
        self._apply_outfit_style_combo_items(filtered_items, text)
        if filtered_items:
            self.outfit_style_combo.showPopup()
        else:
            self.outfit_style_combo.hidePopup()

    def _refresh_outfit_style_delete_btn(self):
        current_text = str(self.outfit_style_combo.currentText() or "").strip()
        history_items = self.get_outfit_style_history() if self.get_outfit_style_history else []
        normalized_history = {str(item or "").strip() for item in (history_items or []) if str(item or "").strip()}
        self.delete_outfit_style_btn.setEnabled(bool(current_text and current_text in normalized_history))

    def _on_outfit_style_text_changed(self, _text):
        if self._updating_outfit_style_combo:
            return
        self._refresh_outfit_style_delete_btn()

    def _commit_outfit_style_from_editor(self):
        self._commit_outfit_style_text(self.outfit_style_combo.currentText(), add_to_history=True)

    def _commit_outfit_style_text(self, text, add_to_history):
        if self._updating_outfit_style_combo:
            return
        value = str(text or "").strip()
        if self.on_outfit_style_changed:
            self.on_outfit_style_changed(value, add_to_history=bool(add_to_history and value))
        self._refresh_outfit_style_delete_btn()

    def _delete_current_outfit_style_history(self):
        value = str(self.outfit_style_combo.currentText() or "").strip()
        if not value:
            return
        if self.on_outfit_style_deleted:
            self.on_outfit_style_deleted(value)

    def _reload_upscale_models(self):
        current = self.upscale_model_combo.currentText().strip()
        models = list_esrgan_models()
        self.upscale_model_combo.blockSignals(True)
        self.upscale_model_combo.clear()
        self.upscale_model_combo.addItems(models)
        if current:
            self.upscale_model_combo.setCurrentText(current)
        self.upscale_model_combo.blockSignals(False)
        if not models:
            self.log_msg("⚠️ 未找到 ESRGAN 模型，请确认 data/models/ESRGAN 或 models/ESRGAN 目录")
        self._persist_upscale_options()

    def _collect_upscale_options(self):
        raw = {
            "enabled": bool(self.enable_jpg_upscale_cb.isChecked()),
            "model_name": self.upscale_model_combo.currentText().strip(),
            "upscale_mode": 0,
            "upscale_by": float(self.upscale_by_spin.value()),
            "max_side_length": 0,
            "upscale_to_width": 1024,
            "upscale_to_height": 1024,
            "upscale_crop": False,
            "upscaler_2_name": "",
            "upscaler_2_visibility": 0.0,
            "cache_size": 4,
            "webp_target_mb": float(self.webp_target_mb_spin.value()),
        }
        return normalize_upscale_options(raw)

    def _persist_upscale_options(self):
        if self.on_upscale_options_changed:
            self.on_upscale_options_changed(self._collect_upscale_options())

    def set_upscale_options_defaults(self, options):
        opts = normalize_upscale_options(options)
        self.enable_jpg_upscale_cb.blockSignals(True)
        self.enable_jpg_upscale_cb.setChecked(bool(opts.get("enabled", False)))
        self.enable_jpg_upscale_cb.blockSignals(False)
        self.upscale_by_spin.blockSignals(True)
        self.upscale_by_spin.setValue(float(opts.get("upscale_by", 2.0)))
        self.upscale_by_spin.blockSignals(False)
        self.webp_target_mb_spin.blockSignals(True)
        self.webp_target_mb_spin.setValue(float(opts.get("webp_target_mb", 10.0)))
        self.webp_target_mb_spin.blockSignals(False)
        model_name = str(opts.get("model_name", "")).strip()
        if model_name:
            self.upscale_model_combo.setCurrentText(model_name)

    def _start_jpg_postprocess(self, saved_files, prompt_type):
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
            task_name=f"批量{prompt_type}后处理",
        )
        self._active_post_threads.append(thread)
        thread.log_signal.connect(self.log_msg)
        thread.finish_signal.connect(lambda results, t=thread: self._on_postprocess_finished(t, results))
        thread.finished.connect(lambda t=thread: self._cleanup_post_thread(t))
        thread.start()

    def _on_postprocess_finished(self, thread, results):
        success = 0
        webp_count = 0
        for item in results or []:
            if item.get("fixed_png_path") and not item.get("error"):
                success += 1
            if item.get("webp_path"):
                webp_count += 1
        if success > 0:
            self.log_msg(f"✅ JPG 自动处理完成，新增 fixed.png: {success} 张，WebP: {webp_count} 张")

    def _cleanup_post_thread(self, thread):
        if thread in self._active_post_threads:
            self._active_post_threads.remove(thread)

    def trigger_quick_export(self):
        if not self.last_finished_json_paths:
            QMessageBox.warning(self, "提示", "还没有可用于导出的批量分析 JSON")
            return
        self.quick_export_requested.emit(list(self.last_finished_json_paths))
