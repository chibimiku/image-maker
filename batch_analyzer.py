import os
import json
import datetime
import re
from openai import OpenAI
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QPushButton, QTextEdit, QComboBox, QMessageBox, QFileDialog, QListWidget, QListWidgetItem, QAbstractItemView)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from single_analyzer import step_1_analyze_image, step_2_refine_description, calculate_closest_aspect_ratio, WorkerThread, ImageGenWorkerThread, compress_and_encode_image
from api_backend import generate_image_whatai, generate_image_aigc2d, _extract_json_object

ANNOTATION_SYSTEM_PROMPT = "You are an expert image annotation assistant. You must return strict JSON only."
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "data", "prompts")
BATCH_ANNOTATION_PROMPT_PATH = os.path.join(PROMPT_DIR, "batch-annotation.md")

def load_batch_annotation_prompt():
    if not os.path.exists(BATCH_ANNOTATION_PROMPT_PATH):
        raise FileNotFoundError(f"未找到批量标注 Prompt 文件: {BATCH_ANNOTATION_PROMPT_PATH}")
    with open(BATCH_ANNOTATION_PROMPT_PATH, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    if not prompt:
        raise ValueError(f"批量标注 Prompt 文件为空: {BATCH_ANNOTATION_PROMPT_PATH}")
    return prompt

def _normalize_annotation_result(result_json):
    if not isinstance(result_json, dict):
        return {}
    long_description = result_json.get("long_description") or result_json.get("longDescription") or ""
    short_description = result_json.get("short_description") or result_json.get("shortDescription") or ""
    booru_tags = result_json.get("booru-tags")
    if booru_tags is None:
        booru_tags = result_json.get("booru_tags")
    if booru_tags is None:
        booru_tags = result_json.get("booruTags")
    if isinstance(booru_tags, list):
        booru_tags = ", ".join([str(tag).strip() for tag in booru_tags if str(tag).strip()])
    if booru_tags is None:
        booru_tags = ""
    return {
        "long_description": str(long_description).strip(),
        "short_description": str(short_description).strip(),
        "booru-tags": str(booru_tags).strip()
    }

class BatchAnnotationWorkerThread(QThread):
    log_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(dict)

    def __init__(self, image_path, api_key, base_url, model_name):
        super().__init__()
        self.image_path = image_path
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

    def run(self):
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            self.log_signal.emit(f"初始化 API 客户端失败: {e}")
            self.finish_signal.emit({})
            return
        try:
            annotation_prompt = load_batch_annotation_prompt()
        except Exception as e:
            self.log_signal.emit(f"读取标注 Prompt 文件失败: {e}")
            self.finish_signal.emit({})
            return
        mime_type, base64_image = compress_and_encode_image(self.image_path, log_callback=self.log_signal.emit)
        if not base64_image:
            self.log_signal.emit("图片压缩或编码失败")
            self.finish_signal.emit({})
            return
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": annotation_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{base64_image}", "detail": "high"}
                            }
                        ]
                    }
                ],
                temperature=0.6,
                max_completion_tokens=16384
            )
            raw_text = response.choices[0].message.content if response and response.choices else ""
            parsed = _extract_json_object(raw_text or "")
            normalized = _normalize_annotation_result(parsed)
            if not normalized:
                self.log_signal.emit("返回内容无法解析为有效 JSON")
                self.finish_signal.emit({})
                return
            self.finish_signal.emit(normalized)
        except Exception as e:
            self.log_signal.emit(f"标注请求失败: {e}")
            self.finish_signal.emit({})

class BatchAnalyzerWidget(QWidget):
    def __init__(self, config_getter_func, img_config_getter_func, styles_getter_func, save_img_cfg_callback, ar_policy_getter_func=None):
        super().__init__()
        self.get_text_config = config_getter_func
        self.get_img_config = img_config_getter_func
        self.get_styles = styles_getter_func
        self.save_img_cfg = save_img_cfg_callback
        self.get_ar_policy = ar_policy_getter_func
        
        self.target_directory = ""
        self.image_files = []
        self._active_threads = []
        self.current_index = 0
        
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
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("处理模式:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["批量图片分析", "批量图片标注(JSON三字段)"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo, stretch=1)
        options_layout.addLayout(mode_layout)
        
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
        

        
        layout.addLayout(options_layout)
        
        # 开始按钮
        self.start_btn = QPushButton("开始批量处理")
        self.start_btn.setFixedHeight(40)
        self.start_btn.clicked.connect(self.start_batch_processing)
        self.start_btn.setEnabled(False)
        layout.addWidget(self.start_btn)
        
        # 日志区域
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        
        self.setLayout(layout)
        self.on_mode_changed(self.mode_combo.currentText())
    
    def update_styles(self, style_keys):
        """由外部 app.py 调用以同步最新的画风列表"""
        curr_main = self.main_style_combo.currentText()
        self.main_style_combo.blockSignals(True)
        self.main_style_combo.clear()
        self.main_style_combo.addItems(style_keys)
        if curr_main in style_keys:
            self.main_style_combo.setCurrentText(curr_main)
        self.main_style_combo.blockSignals(False)

    def on_mode_changed(self, mode_name):
        is_annotation_mode = mode_name == "批量图片标注(JSON三字段)"
        self.auto_gen_orig_cb.setEnabled(not is_annotation_mode)
        self.auto_gen_ref_cb.setEnabled(not is_annotation_mode)
    
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
        self.log_msg("已清空图片列表")
    
    def remove_selected_images(self):
        selected_items = self.image_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "提示", "请先选择要移除的图片")
            return
        
        removed_count = 0
        for item in selected_items:
            # 通过 QListWidgetItem 的文本反查完整路径
            candidate = None
            for img_path in self.image_files:
                if os.path.basename(img_path) == item.text():
                    candidate = img_path
                    break
            if candidate and candidate in self.image_files:
                # 通过 QListWidgetItem 的文本反查完整路径
                candidate = None
                for img_path in self.image_files:
                    if os.path.basename(img_path) == item.text():
                        candidate = img_path
                        break
                if candidate:
                    self.image_files.remove(candidate)
                removed_count += 1
            self.image_list.takeItem(self.image_list.row(item))
        
        self.start_btn.setEnabled(len(self.image_files) > 0)
        self.log_msg(f"已移除 {removed_count} 个选中的图片文件")
    
    def start_batch_processing(self):
        if len(self.image_files) == 0:
            QMessageBox.warning(self, "错误", "请先添加图片文件")
            return
        
        # 从列表中更新图片文件列表
        self.image_files = []
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item and hasattr(item, 'image_path'):
                # 通过 item.text()（文件名）反查完整路径
                for img_path in self.image_files:
                    if os.path.basename(img_path) == item.text():
                        self.image_files.append(img_path)
                        break
        
        if len(self.image_files) == 0:
            QMessageBox.warning(self, "错误", "请先添加图片文件")
            return
        
        base_url, api_key, model_name = self.get_text_config()
        if not api_key or not model_name:
            QMessageBox.warning(self, "缺少配置", "文本分析 API Key 和 模型名称不能为空！")
            return
        
        self.current_index = 0
        self.start_btn.setEnabled(False)
        self.log_text.clear()
        mode_name = self.mode_combo.currentText()
        self.log_msg(f"开始{mode_name}，共 {len(self.image_files)} 个文件")
        
        # 开始处理第一个文件
        self.process_next_image()
    
    def process_next_image(self):
        if self.current_index >= len(self.image_files):
            if self.mode_combo.currentText() == "批量图片标注(JSON三字段)":
                self.log_msg("🎉 批量标注完成！")
            else:
                self.log_msg("🎉 批量分析完成！")
            self.start_btn.setEnabled(True)
            return
        
        image_path = self.image_files[self.current_index]
        self.log_msg(f"\n=== 处理第 {self.current_index + 1}/{len(self.image_files)} 个文件 ===")
        self.log_msg(f"文件: {os.path.basename(image_path)}")
        
        # 调用单图分析逻辑
        base_url, api_key, model_name = self.get_text_config()
        if self.mode_combo.currentText() == "批量图片标注(JSON三字段)":
            thread = BatchAnnotationWorkerThread(image_path, api_key, base_url, model_name)
        else:
            thread = WorkerThread(image_path, api_key, base_url, model_name)
        thread.log_signal.connect(self.log_msg)
        thread.finish_signal.connect(lambda result: self.on_image_processed(result, image_path))
        thread.start()
        self._active_threads.append(thread)
    
    def on_image_processed(self, result_json, image_path):
        if not result_json:
            self.log_msg("❌ 处理失败，跳过此文件")
        else:
            if self.mode_combo.currentText() == "批量图片标注(JSON三字段)":
                self.log_msg("✅ 标注完成")
                self.save_annotation_result(result_json, image_path)
            else:
                self.log_msg("✅ 分析完成")
                self.save_result(result_json, image_path)
                if self.auto_gen_orig_cb.isChecked() or self.auto_gen_ref_cb.isChecked():
                    self.generate_images(result_json)
        
        # 处理下一个文件
        self.current_index += 1
        self.process_next_image()
    
    def save_result(self, result_json, image_path):
        try:
            jp_title = result_json.get("japanese_title", "未命名")
            safe_title = re.sub(r'[\\/*?:"<>|]', "", jp_title).strip() or "未命名"
            
            now = datetime.datetime.now()
            date_str = now.strftime("%Y%m%d")
            save_dir = os.path.join('data', date_str)
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # 生成唯一的文件名
            base_filename = f"{now.strftime('%Y%m%d-%H%M%S')}-{safe_title}"
            json_filename = f"{base_filename}.json"
            
            # 保存 JSON
            with open(os.path.join(save_dir, json_filename), "w", encoding="utf-8") as f:
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
            
        except Exception as e:
            self.log_msg(f"❌ 保存结果时出错: {e}")

    def save_annotation_result(self, result_json, image_path):
        try:
            normalized = _normalize_annotation_result(result_json)
            if not normalized:
                self.log_msg("❌ 标注 JSON 结构无效，跳过保存")
                return
            now = datetime.datetime.now()
            date_str = now.strftime("%Y%m%d")
            save_dir = os.path.join('data', date_str, "annotations")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            image_stem = os.path.splitext(os.path.basename(image_path))[0]
            safe_stem = re.sub(r'[\\/*?:"<>|]', "", image_stem).strip() or "image"
            json_filename = f"{now.strftime('%Y%m%d-%H%M%S')}-{safe_stem}-annotation.json"
            output_path = os.path.join(save_dir, json_filename)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(normalized, f, ensure_ascii=False, indent=4)
            self.log_msg("🧾 标注结果:")
            self.log_msg(json.dumps(normalized, ensure_ascii=False, indent=2))
            self.log_msg(f"📁 标注 JSON 已保存: {output_path}")
        except Exception as e:
            self.log_msg(f"❌ 保存标注结果时出错: {e}")
    
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
