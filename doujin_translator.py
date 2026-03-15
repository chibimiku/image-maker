import os
import json
import base64
import io
import time
from openai import OpenAI
from PIL import Image

# --- PyQt5 界面相关库 ---
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTextEdit, QFileDialog, 
                             QListWidget, QListWidgetItem, QCheckBox, 
                             QSpinBox, QSplitter, QLineEdit, QMessageBox,
                             QScrollArea, QGroupBox, QFormLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap

# --- 读取配置文件 ---
def load_config():
    config_path = "config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"读取配置文件失败: {e}")
    # 如果读取失败或文件不存在，返回默认值
    return {
        "base_url": "https://api.openai.com/v1",
        "api_key": os.environ.get("OPENAI_API_KEY","YOUR_API_KEY"),
        "model": "grok-4"
    }

config = load_config()

# 初始化 OpenAI 客户端，直接使用配置文件中的值
client = OpenAI(
    api_key=config.get("api_key"),
    base_url=config.get("base_url")
)

system_prompt = """
You are an expert manga and doujinshi translator and image analyzer.
You must respond strictly in JSON format.
"""

def generate_user_prompt(source_lang, target_lang, img_width, img_height, project_context=""):
    # 动态注入上下文
    context_instruction = ""
    if project_context.strip():
        context_instruction = f"""
--- PREVIOUS PAGES CONTEXT (For Translation Consistency) ---
Here is the summary of previously translated pages in this project:
{project_context}

Please use the above context to maintain consistent character names, tone, relationships, and terminology.
----------------------------------------------------------
"""

    return f"""
Please analyze and translate the provided image.
Important Context: This is a doujinshi (fan comic), and the text is highly likely to be written vertically.

{context_instruction}

CRITICAL IMAGE DIMENSIONS:
The image provided has a resolution of {img_width} pixels in width and {img_height} pixels in height.

Your task is to identify all text regions, extract the original text in {source_lang}, and translate it into {target_lang}.
ALSO, you must generate a summary of the current page's plot, characters involved, and key events.

CRITICAL INSTRUCTION FOR REGION SEPARATION AND COORDINATES: 
1. You MUST treat every single speech bubble, text box, or visually separated text block as a completely independent region. 
2. The coordinates in "xyxy" and "lines" MUST BE ABSOLUTE PIXEL VALUES based on the {img_width}x{img_height} resolution. 
3. DO NOT output fake, sequential, or grid-like coordinates (e.g., [50,50, 300,300]). You must visually locate the actual text and estimate its precise bounding box on this {img_width}x{img_height} canvas. X values must be between 0 and {img_width}, Y values must be between 0 and {img_height}.

Return the result STRICTLY as a JSON object containing TWO main keys: "page_summary" and "regions".

1. "page_summary" MUST contain:
   - "characters": Array of strings (Names of characters appearing or mentioned on this page).
   - "plot": String (A brief summary of the events/dialogue on this page).
   - "key_terms": Array of strings (Important items, places, or specific terms introduced or used).

2. "regions" MUST be an array of objects. Each object MUST contain the following keys:
   - "xyxy": Array of 4 integers [x_min, y_min, x_max, y_max] representing the precise bounding box.
   - "lines": A list of text lines within the region. Each line is represented by a polygon of 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]. 
   - "text": Array of strings. The original extracted text.
   - "translation": String. The final translated text for the entire region combined.
   - "src_is_vertical": Boolean. true if the original text is written vertically, false otherwise.
   - "_detected_font_size": Integer. The estimated font size of the text.

Example structure:
{{
  "page_summary": {{
    "characters": ["Tachibana", "Yamada"],
    "plot": "Tachibana hands Yamada a glass of water and thanks him for his help.",
    "key_terms": ["Water glass"]
  }},
  "regions": [
    {{
      "xyxy": [331, 331, 370, 558],
      "lines": [
        [[331, 331], [370, 331], [370, 558], [331, 558]]
      ],
      "text": ["水だワン！！"],
      "translation": "是水汪！！",
      "src_is_vertical": true,
      "_detected_font_size": 39
    }}
  ]
}}
"""

def compress_and_encode_image(image_source, log_callback=None, max_dim=2048):
    try:
        file_size_mb = os.path.getsize(image_source) / (1024 * 1024)
        ext = os.path.splitext(image_source)[1].lower()
        filename = os.path.basename(image_source)

        img = Image.open(image_source)
        original_width, original_height = img.size

        is_large_file = file_size_mb >= 8.0
        is_not_jpg = ext not in ['.jpg', '.jpeg']
        
        if is_large_file:
            if is_not_jpg:
                if log_callback:
                    log_callback(f"[{filename}] 图片体积为 {file_size_mb:.2f}MB (>=8MB) 且格式为 {ext}，将转换为 JPG 并进行压缩以减小体积。")
            else:
                if log_callback:
                    log_callback(f"[{filename}] 图片体积为 {file_size_mb:.2f}MB (>=8MB)，将降低质量进行压缩。")

        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        if max(original_width, original_height) > max_dim:
            scaling_factor = max_dim / max(original_width, original_height)
            new_width = int(original_width * scaling_factor)
            new_height = int(original_height * scaling_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            if log_callback and not is_large_file:
                log_callback(f"[{filename}] 图片尺寸超出 {max_dim}，已缩放至 {new_width}x{new_height}。")
        else:
            new_width, new_height = original_width, original_height

        buffered = io.BytesIO()
        quality_setting = 75 if is_large_file else 95
        
        img.save(buffered, format="JPEG", quality=quality_setting)
        base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return "image/jpeg", base64_string, new_width, new_height

    except Exception as e:
        if log_callback:
            log_callback(f"处理图片 {os.path.basename(image_source)} 时发生错误: {e}")
        return None, None, None, None


# --- API 请求工作线程 ---
class TranslateWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int) # current, total
    finished_signal = pyqtSignal()
    item_done_signal = pyqtSignal(str, dict) # filepath, result_json

    def __init__(self, items_to_process, source_lang, target_lang, retries, current_dir):
        super().__init__()
        self.items_to_process = items_to_process # List of filepaths
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.retries = retries
        self.current_dir = current_dir # 传入当前工程目录以读取上下文
        self.is_running = True

    def build_project_context(self):
        """遍历目录，读取所有已生成的 _summary.json，合并为上下文提示词"""
        context_lines = []
        try:
            for filename in sorted(os.listdir(self.current_dir)):
                if filename.endswith("_summary.json"):
                    base_name = filename.replace("_summary.json", "")
                    file_path = os.path.join(self.current_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        chars = ", ".join(data.get("characters", []))
                        plot = data.get("plot", "")
                        terms = ", ".join(data.get("key_terms", []))
                        # 拼接简短的单页摘要
                        line = f"Page [{base_name}] -> Characters: {chars} | Terms: {terms} | Plot: {plot}"
                        context_lines.append(line)
        except Exception as e:
            self.log_signal.emit(f"读取工程上下文摘要时发生警告: {e}")
            
        if not context_lines:
            return ""
        return "\n".join(context_lines)

    def run(self):
        total = len(self.items_to_process)

        for i, filepath in enumerate(self.items_to_process):
            if not self.is_running:
                break
            
            filename = os.path.basename(filepath)
            self.log_signal.emit(f"=== 开始处理 ({i+1}/{total}): {filename} ===")
            
            mime_type, base64_image, img_width, img_height = compress_and_encode_image(
                filepath, 
                log_callback=self.log_signal.emit
            )
            
            if not base64_image:
                self.log_signal.emit(f"[{filename}] 图片读取或压缩失败。")
                continue

            # 动态生成项目上下文
            current_context = self.build_project_context()
            if current_context:
                self.log_signal.emit(f"[{filename}] 已成功注入当前工程的历史上下文。")

            prompt = generate_user_prompt(self.source_lang, self.target_lang, img_width, img_height, current_context)

            # 提前构造请求 Payload，方便后续调用以及失败时 Dump 出来
            request_payload = {
                "model": "grok-4", # 注：如需明确指定 Grok 模型，可在此处更改，例如 "grok-vision-beta"
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 3000
            }

            success = False
            result_data = {}
            for attempt in range(self.retries + 1):
                try:
                    self.log_signal.emit(f"[{filename}] 正在发起网络请求... (尝试 {attempt + 1}/{self.retries + 1})")
                    start_time = time.time()
                    
                    # 传入组装好的 payload
                    response = client.chat.completions.create(**request_payload)
                    
                    elapsed_time = time.time() - start_time
                    content = response.choices[0].message.content
                    
                    base_name = os.path.splitext(filepath)[0]
                    debug_raw_path = f"{base_name}_raw_debug.json"
                    try:
                        with open(debug_raw_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        self.log_signal.emit(f"[{filename}] 原始返回数据已输出至: {os.path.basename(debug_raw_path)}")
                    except Exception as e:
                        self.log_signal.emit(f"[{filename}] 保存 debug 原始数据失败: {e}")

                    result_data = json.loads(content)
                    self.log_signal.emit(f"[{filename}] 网络请求成功！耗时: {elapsed_time:.2f}秒。")
                    success = True
                    break 

                except Exception as e:
                    elapsed_time = time.time() - start_time
                    self.log_signal.emit(f"[{filename}] 请求异常 (耗时 {elapsed_time:.2f}秒): {e}")
                    
                    # ========== 新增：捕获异常并 Dump 原始请求数据 ==========
                    base_name = os.path.splitext(filepath)[0]
                    debug_req_path = f"{base_name}_raw_request_debug.json"
                    try:
                        # 引入 copy 防止修改原 payload 导致下一次重试失败
                        import copy
                        dump_payload = copy.deepcopy(request_payload)
                        # 将超长的 Base64 截断，方便人工用文本编辑器查看 json
                        dump_payload["messages"][1]["content"][1]["image_url"]["url"] = f"data:{mime_type};base64,<BASE64_STRING_OMITTED_FOR_DEBUG>"
                        
                        with open(debug_req_path, 'w', encoding='utf-8') as f:
                            json.dump(dump_payload, f, ensure_ascii=False, indent=4)
                        self.log_signal.emit(f"[{filename}] 原始请求参数已 Dump 至: {os.path.basename(debug_req_path)}")
                    except Exception as dump_e:
                        self.log_signal.emit(f"[{filename}] 保存请求 Debug 数据失败: {dump_e}")
                    # ========================================================

                    if attempt < self.retries:
                        self.log_signal.emit(f"[{filename}] 准备进行下一次重试...")
                        time.sleep(2) 
                    else:
                        self.log_signal.emit(f"[{filename}] 重试耗尽，该图片处理失败。")
            
            if success:
                base_name = os.path.splitext(filepath)[0]
                json_path = f"{base_name}_translation.json"
                summary_path = f"{base_name}_summary.json"
                
                try:
                    # 保存常规翻译数据
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(result_data, f, ensure_ascii=False, indent=4)
                    self.log_signal.emit(f"[{filename}] 识别与翻译数据已保存。")
                    
                    # 分离并保存梗概数据
                    if "page_summary" in result_data:
                        with open(summary_path, 'w', encoding='utf-8') as f:
                            json.dump(result_data["page_summary"], f, ensure_ascii=False, indent=4)
                        self.log_signal.emit(f"[{filename}] 本页剧情梗概已提取并单独保存。")

                    self.item_done_signal.emit(filepath, result_data)
                except Exception as e:
                    self.log_signal.emit(f"[{filename}] 保存 JSON 文件失败: {e}")

            self.progress_signal.emit(i + 1, total)
            
        self.finished_signal.emit()

    def stop(self):
        self.is_running = False


# --- PyQt5 主界面 ---
class AppWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.current_dir = ""
        self.image_files = [] 
        self.translation_cache = {} 
        self.worker = None
        self.edit_fields = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle("同人本批量翻译器 v1.4 (带剧情上下文记忆)")
        self.resize(1200, 800)

        main_layout = QVBoxLayout(self)

        top_layout = QHBoxLayout()
        self.btn_select_dir = QPushButton("选择目录")
        self.btn_select_dir.clicked.connect(self.select_directory)
        top_layout.addWidget(self.btn_select_dir)

        self.lbl_current_dir = QLabel("未选择目录")
        self.lbl_current_dir.setStyleSheet("color: gray;")
        top_layout.addWidget(self.lbl_current_dir)
        top_layout.addStretch()

        top_layout.addWidget(QLabel("输入语言:"))
        self.input_lang = QLineEdit("Japanese")
        self.input_lang.setFixedWidth(80)
        top_layout.addWidget(self.input_lang)

        top_layout.addWidget(QLabel("输出语言:"))
        self.output_lang = QLineEdit("Simplified Chinese")
        self.output_lang.setFixedWidth(120)
        top_layout.addWidget(self.output_lang)

        top_layout.addWidget(QLabel("网络重试次数:"))
        self.spin_retries = QSpinBox()
        self.spin_retries.setRange(0, 10)
        self.spin_retries.setValue(3)
        top_layout.addWidget(self.spin_retries)

        main_layout.addLayout(top_layout)

        splitter = QSplitter(Qt.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(100, 100))
        self.list_widget.setResizeMode(QListWidget.Adjust)
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        left_layout.addWidget(self.list_widget)

        left_btn_layout = QHBoxLayout()
        self.btn_select_all = QPushButton("全选")
        self.btn_select_all.clicked.connect(self.select_all_items)
        self.btn_deselect_all = QPushButton("全不选")
        self.btn_deselect_all.clicked.connect(self.deselect_all_items)
        left_btn_layout.addWidget(self.btn_select_all)
        left_btn_layout.addWidget(self.btn_deselect_all)
        left_layout.addLayout(left_btn_layout)

        self.btn_translate = QPushButton("一键翻译选中图片")
        self.btn_translate.setMinimumHeight(40)
        self.btn_translate.clicked.connect(self.start_translation)
        left_layout.addWidget(self.btn_translate)
        
        self.btn_export = QPushButton("导出 imgtrans 格式 JSON")
        self.btn_export.setMinimumHeight(40)
        self.btn_export.setStyleSheet("background-color: #2b5c8f; color: white;")
        self.btn_export.clicked.connect(self.export_imgtrans_json)
        left_layout.addWidget(self.btn_export)
        
        splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.check_show_translation = QCheckBox("显示翻译结果与编辑区")
        self.check_show_translation.setChecked(True)
        self.check_show_translation.stateChanged.connect(self.refresh_preview)
        right_layout.addWidget(self.check_show_translation)

        self.lbl_preview_image = QLabel("选择左侧图片进行预览")
        self.lbl_preview_image.setAlignment(Qt.AlignCenter)
        self.lbl_preview_image.setStyleSheet("background-color: #222; color: white;")
        self.lbl_preview_image.setMinimumSize(400, 300)
        right_layout.addWidget(self.lbl_preview_image, 5) 

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_area.setWidget(self.scroll_widget)
        right_layout.addWidget(self.scroll_area, 4) 

        splitter.addWidget(right_widget)
        splitter.setSizes([350, 850]) 

        main_layout.addWidget(splitter)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        main_layout.addWidget(self.log_text)

    def log_msg(self, text):
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        self.log_text.append(f"[{timestamp}] {text}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if dir_path:
            self.current_dir = dir_path
            self.lbl_current_dir.setText(dir_path)
            self.load_directory_contents()

    def load_directory_contents(self):
        self.list_widget.clear()
        self.image_files.clear()
        self.translation_cache.clear()

        supported_ext = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        
        for filename in os.listdir(self.current_dir):
            if filename.lower().endswith(supported_ext):
                filepath = os.path.join(self.current_dir, filename)
                self.image_files.append(filepath)
                
                item = QListWidgetItem(self.list_widget)
                item_widget = QWidget()
                h_layout = QHBoxLayout(item_widget)
                h_layout.setContentsMargins(5, 5, 5, 5)
                
                chk_box = QCheckBox()
                lbl_name = QLabel(filename)
                
                h_layout.addWidget(chk_box)
                h_layout.addWidget(lbl_name)
                h_layout.addStretch()
                
                item.setSizeHint(item_widget.sizeHint())
                self.list_widget.setItemWidget(item, item_widget)
                item.setData(Qt.UserRole, filepath) 

                base_name = os.path.splitext(filename)[0]
                json_path = os.path.join(self.current_dir, f"{base_name}_translation.json")
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            self.translation_cache[filepath] = json.load(f)
                            lbl_name.setText(filename + " [已翻译]")
                            lbl_name.setStyleSheet("color: green;")
                    except Exception:
                        pass
        
        self.log_msg(f"加载目录完成，共发现 {len(self.image_files)} 张图片。")

    def select_all_items(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            checkbox = widget.findChild(QCheckBox)
            if checkbox: checkbox.setChecked(True)

    def deselect_all_items(self):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            checkbox = widget.findChild(QCheckBox)
            if checkbox: checkbox.setChecked(False)

    def on_item_clicked(self, item):
        self.current_preview_path = item.data(Qt.UserRole)
        self.refresh_preview()

    def refresh_preview(self):
        if not hasattr(self, 'current_preview_path') or not self.current_preview_path:
            return

        filepath = self.current_preview_path
        
        pixmap = QPixmap(filepath)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(self.lbl_preview_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.lbl_preview_image.setPixmap(scaled_pixmap)

        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.edit_fields.clear()

        if self.check_show_translation.isChecked() and filepath in self.translation_cache:
            trans_data = self.translation_cache[filepath]
            regions = trans_data.get("regions", [])
            
            # 兼容展示梗概
            summary = trans_data.get("page_summary", {})
            if summary:
                gb_summary = QGroupBox("本页剧情梗概 (GPT提取)")
                form_summary = QFormLayout()
                lbl_chars = QLabel(", ".join(summary.get("characters", [])))
                lbl_chars.setWordWrap(True)
                lbl_plot = QLabel(summary.get("plot", ""))
                lbl_plot.setWordWrap(True)
                
                form_summary.addRow("出场人物:", lbl_chars)
                form_summary.addRow("剧情梗概:", lbl_plot)
                gb_summary.setLayout(form_summary)
                self.scroll_layout.addWidget(gb_summary)
            
            for idx, r in enumerate(regions):
                gb = QGroupBox(f"区域 {idx+1}")
                form = QFormLayout()
                
                raw_texts = r.get("text", [r.get("original_text", "")])
                orig_text_str = "\n".join(raw_texts) if isinstance(raw_texts, list) else raw_texts
                
                orig_edit = QTextEdit(orig_text_str)
                orig_edit.setMaximumHeight(60)
                trans_edit = QTextEdit(r.get("translation", ""))
                trans_edit.setMaximumHeight(60)
                
                form.addRow("原文:", orig_edit)
                form.addRow("译文:", trans_edit)
                gb.setLayout(form)
                self.scroll_layout.addWidget(gb)
                
                self.edit_fields.append((idx, orig_edit, trans_edit))
            
            if regions:
                btn_save = QPushButton("保存当前页修改")
                btn_save.setMinimumHeight(35)
                btn_save.clicked.connect(self.save_current_page_edits)
                self.scroll_layout.addWidget(btn_save)
                
            self.scroll_layout.addStretch()
        elif not self.check_show_translation.isChecked():
            lbl = QLabel("预览编辑区已被隐藏。")
            self.scroll_layout.addWidget(lbl)
            self.scroll_layout.addStretch()
        else:
            lbl = QLabel("此图片尚未进行翻译，没有可供编辑的记录。")
            self.scroll_layout.addWidget(lbl)
            self.scroll_layout.addStretch()

    def save_current_page_edits(self):
        filepath = self.current_preview_path
        if filepath not in self.translation_cache: return
        
        regions = self.translation_cache[filepath].get("regions", [])
        for idx, orig_edit, trans_edit in self.edit_fields:
            if idx < len(regions):
                regions[idx]["text"] = orig_edit.toPlainText().split('\n')
                regions[idx]["translation"] = trans_edit.toPlainText()
        
        base_name = os.path.splitext(filepath)[0]
        json_path = f"{base_name}_translation.json"
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.translation_cache[filepath], f, ensure_ascii=False, indent=4)
            self.log_msg(f"[{os.path.basename(filepath)}] 手工修改已成功保存。")
            QMessageBox.information(self, "成功", "修改已保存！")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存失败: {e}")

    def export_imgtrans_json(self):
        if not self.current_dir:
            QMessageBox.warning(self, "警告", "请先选择并加载目录！")
            return
            
        export_path, _ = QFileDialog.getSaveFileName(self, "导出合并的 JSON", self.current_dir, "JSON Files (*.json)")
        if not export_path: return
        
        out_json = {
            "directory": self.current_dir,
            "pages": {},
            "current_img": "",
            "image_info": {}
        }
        
        first_img = True
        for filepath in self.image_files:
            filename = os.path.basename(filepath)
            if first_img:
                out_json["current_img"] = filename
                first_img = False
                
            try:
                img = Image.open(filepath)
                width, height = img.size
            except:
                width, height = 1000, 1000
            
            out_json["image_info"][filename] = {
                "finish_code": 15,
                "width": width,
                "height": height
            }
            
            page_regions = []
            if filepath in self.translation_cache:
                regions = self.translation_cache[filepath].get("regions", [])
                for r in regions:
                    xyxy = r.get("xyxy", [0, 0, 100, 100])
                    w = max(1, xyxy[2] - xyxy[0])
                    h = max(1, xyxy[3] - xyxy[1])
                    font_size = float(r.get("_detected_font_size", 30.0))
                    translation = r.get("translation", "")
                    is_vert = r.get("src_is_vertical", True)
                    
                    rich_text = f"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\"><html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\np, li {{ white-space: pre-wrap; }}\nhr {{ height: 1px; border-width: 0; }}\nli.unchecked::marker {{ content: \"\\2610\"; }}\nli.checked::marker {{ content: \"\\2612\"; }}\n</style></head><body style=\" font-family:'Microsoft YaHei UI'; font-size:{font_size}pt; font-weight:400; font-style:normal;\"><p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" color:#000000;\">{translation}</span></p></body></html>"
                    
                    formatted_r = {
                        "xyxy": xyxy,
                        "lines": r.get("lines", []),
                        "language": "unknown",
                        "distance": None,
                        "angle": 0,
                        "vec": [0.0, float(h)],
                        "norm": float(h),
                        "merged": False,
                        "text": r.get("text", []),
                        "translation": translation,
                        "rich_text": rich_text,
                        "_bounding_rect": [xyxy[0], xyxy[1], w, h],
                        "src_is_vertical": is_vert,
                        "_detected_font_size": font_size,
                        "det_model": "ctd",
                        "label": None,
                        "region_mask": None,
                        "region_inpaint_dict": None,
                        "fontformat": {
                            "font_family": "Microsoft YaHei UI",
                            "font_size": font_size,
                            "stroke_width": 0.0,
                            "frgb": [0, 0, 0],
                            "srgb": [0, 0, 0],
                            "bold": False,
                            "underline": False,
                            "italic": False,
                            "alignment": 0,
                            "vertical": is_vert,
                            "font_weight": 400,
                            "line_spacing": 1.2,
                            "letter_spacing": 1.15,
                            "opacity": 1.0,
                            "shadow_radius": 0.0,
                            "shadow_strength": 1.0,
                            "shadow_color": [0, 0, 0],
                            "shadow_offset": [0.0, 0.0],
                            "gradient_enabled": False,
                            "gradient_start_color": [0, 0, 0],
                            "gradient_end_color": [255, 255, 255],
                            "gradient_angle": 0.0,
                            "gradient_size": 1.0,
                            "_style_name": "",
                            "line_spacing_type": 0,
                            "deprecated_attributes": {}
                        },
                        "_detected_font_name": "",
                        "_detected_font_confidence": 0.0
                    }
                    page_regions.append(formatted_r)
            
            out_json["pages"][filename] = page_regions
            
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(out_json, f, ensure_ascii=False, indent=4)
            self.log_msg(f"合集成功导出至: {export_path}")
            QMessageBox.information(self, "成功", "imgtrans 格式合集导出完毕！")
        except Exception as e:
            self.log_msg(f"导出失败: {e}")
            QMessageBox.warning(self, "错误", f"导出失败: {e}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_preview() 

    def start_translation(self):
        selected_files = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            checkbox = widget.findChild(QCheckBox)
            if checkbox and checkbox.isChecked():
                selected_files.append(item.data(Qt.UserRole))

        if not selected_files:
            QMessageBox.warning(self, "警告", "请至少选择一张需要翻译的图片！")
            return

        self.btn_translate.setEnabled(False)
        self.btn_select_dir.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.log_msg(f"=== 开始批量翻译，共 {len(selected_files)} 个任务 ===")

        source_lang = self.input_lang.text().strip() or "Japanese"
        target_lang = self.output_lang.text().strip() or "Simplified Chinese"
        retries = self.spin_retries.value()

        # 修改点：传入 self.current_dir
        self.worker = TranslateWorker(selected_files, source_lang, target_lang, retries, self.current_dir)
        self.worker.log_signal.connect(self.log_msg)
        self.worker.item_done_signal.connect(self.on_item_translated)
        self.worker.finished_signal.connect(self.on_translation_finished)
        self.worker.start()

    def on_item_translated(self, filepath, result_data):
        self.translation_cache[filepath] = result_data
        
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.UserRole) == filepath:
                widget = self.list_widget.itemWidget(item)
                lbl_name = widget.findChild(QLabel)
                filename = os.path.basename(filepath)
                lbl_name.setText(filename + " [已翻译]")
                lbl_name.setStyleSheet("color: green;")
                break

        if hasattr(self, 'current_preview_path') and self.current_preview_path == filepath:
            self.refresh_preview()

    def on_translation_finished(self):
        self.btn_translate.setEnabled(True)
        self.btn_select_dir.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.log_msg("=== 批处理队列执行结束 ===")

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec_())