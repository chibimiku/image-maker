import os
import json
import base64
import io
import datetime
import re
from openai import OpenAI
from PIL import Image, ImageGrab

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QCheckBox,
                             QLabel, QPushButton, QTextEdit, QComboBox, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

from api_backend import generate_image_whatai, generate_image_aigc2d 

system_prompt = """
You are an expert image analyzer and illustrator assistant. 
You must respond strictly in JSON format.
"""

# 新增读取 Prompt 的函数
def load_prompt_from_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"读取 Prompt 文件失败: {e}")
        # 如果读取失败，返回一个保底的简单提示词以防程序崩溃
        return "Please analyze the provided image and generate a detailed description in English. Return strict JSON."

# 动态构建文件路径并读取
PROMPT_DIR = os.path.join(os.path.dirname(__file__), 'data', 'prompts')
STYLE_ANALY_PATH = os.path.join(PROMPT_DIR, 'style-analy.md')

# 删除原本硬编码的长文本，改为调用函数
user_prompt_analyze = load_prompt_from_file(STYLE_ANALY_PATH)

def calculate_closest_aspect_ratio(image_source):
    """根据输入图片尺寸，从预设列表中计算最贴近的长宽比"""
    try:
        if isinstance(image_source, str):
            img = Image.open(image_source)
        else:
            img = image_source 
        w, h = img.size
        actual_ratio = w / h
        
        target_ratios = {
            "1:1": 1.0,
            "2:3": 2.0 / 3.0,
            "3:2": 3.0 / 2.0,
            "3:4": 3.0 / 4.0,
            "4:3": 4.0 / 3.0,
            "16:9": 16.0 / 9.0,
            "9:16": 9.0 / 16.0
        }
        
        # 寻找实际比例与目标比例差值绝对值最小的键名
        closest_ar = min(target_ratios.keys(), key=lambda k: abs(target_ratios[k] - actual_ratio))
        return closest_ar
    except Exception as e:
        print(f"计算长宽比失败: {e}")
        return "1:1" # 发生异常时默认返回 1:1

def compress_and_encode_image(image_source, max_dim=2048, log_callback=None):
    try:
        if isinstance(image_source, str):
            img = Image.open(image_source)
        else:
            img = image_source 

        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        original_width, original_height = img.size
        size_msg = f"原始图片尺寸: {original_width}x{original_height}"
        if log_callback:
            log_callback(size_msg)
        print(size_msg)

        if max(original_width, original_height) > max_dim:
            scaling_factor = max_dim / max(original_width, original_height)
            new_width = int(original_width * scaling_factor)
            new_height = int(original_height * scaling_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resize_msg = f"图片已成功压缩为: {new_width}x{new_height}"
            if log_callback:
                log_callback(resize_msg)
            print(resize_msg)

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=100)
        base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return "image/jpeg", base64_string

    except Exception as e:
        error_msg = f"处理图片时发生错误: {e}"
        if log_callback:
            log_callback(error_msg)
        print(error_msg)
        return None, None

def step_1_analyze_image(image_source, client, model_name, log_callback=None):
    mime_type, base64_image = compress_and_encode_image(image_source, log_callback=log_callback)
    if not base64_image:
        if log_callback:
            log_callback("Step 1 失败: 图片压缩或编码失败")
        return None
    try:
        response = client.chat.completions.create(
            model=model_name, 
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt_analyze},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_image}", "detail": "high"}
                        }
                    ]
                }
            ],
            temperature=0.7, max_completion_tokens=16384
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        error_msg = f"Step 1 请求错误: {e}"
        if log_callback:
            log_callback(error_msg)
        print(error_msg)
        return None

def step_2_refine_description(original_json_data, client, model_name):
    original_description = original_json_data.get("english_description", "")
    jp_title = original_json_data.get("japanese_title", "")
    cn_title = original_json_data.get("chinese_title", "")
    tags = original_json_data.get("pixiv_tags", [])
    
    tags_str = json.dumps(tags, ensure_ascii=False)
    
    # 构建模板文件路径并读取
    REFINE_DESC_PATH = os.path.join(os.path.dirname(__file__), 'data', 'prompts', 'refine-desc.md')
    template = load_prompt_from_file(REFINE_DESC_PATH)
    
    # 安全替换占位符（避免 f-string 遇到 JSON 大括号报错）
    refine_prompt = template.replace("{jp_title}", jp_title) \
                            .replace("{cn_title}", cn_title) \
                            .replace("{original_description}", original_description) \
                            .replace("{tags_str}", tags_str)
    
    try:
        response = client.chat.completions.create(
            model=model_name, 
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": refine_prompt}
            ],
            temperature=0.7, max_completion_tokens=16384
        )
        final_result_json = json.loads(response.choices[0].message.content)
        # 将原始描述也存入最终结果，方便后续对比或同时生成
        final_result_json["original_english_description"] = original_description
        return final_result_json
    except Exception as e:
        print(f"Step 2 二次加工时发生错误: {e}")
        return None

class WorkerThread(QThread):
    log_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(dict)

    def __init__(self, image_source, api_key, base_url, model_name):
        super().__init__()
        self.image_source = image_source
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

        self.log_signal.emit(f"正在使用模型 [{self.model_name}] 开始 Step 1: 读取并压缩图片，发送 Vision 请求...")
        initial_result = step_1_analyze_image(self.image_source, client, self.model_name, log_callback=self.log_signal.emit)
        
        if initial_result:
            self.log_signal.emit("Step 1 完成。初步结果已获取。")
            self.log_signal.emit("正在开始 Step 2: 根据中文指令对英文描述进行加工并推断长宽比...")
            final_result = step_2_refine_description(initial_result, client, self.model_name)

            # 【修改】强制将本地计算好的长宽比注入到大模型的返回结果中
            if final_result:
                final_result["aspect_ratio"] = calculate_closest_aspect_ratio(self.image_source)
                
                # 【新增】保存第一次和第二次的 pixiv tags，并计算交集
                initial_tags = initial_result.get("pixiv_tags", [])
                refined_tags = final_result.get("pixiv_tags", [])
                
                # 保存原始两次的 tags
                final_result["pixiv_tags_first"] = initial_tags
                final_result["pixiv_tags_second"] = refined_tags
                
                # 计算交集
                if initial_tags and refined_tags:
                    initial_tags_lower = [tag.lower() for tag in initial_tags]
                    intersection_tags = []
                    for tag in refined_tags:
                        if tag.lower() in initial_tags_lower:
                            intersection_tags.append(tag)
                    # 如果交集为空，则使用 refined_tags
                    final_result["pixiv_tags"] = intersection_tags if intersection_tags else refined_tags
            
            self.finish_signal.emit(final_result if final_result else {})
        else:
            self.log_signal.emit("Step 1 失败，流程终止。")
            self.finish_signal.emit({})

class ImageGenWorkerThread(QThread):
    log_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(list)

    def __init__(self, prompt, model_name, aspect_ratio, instructions, api_type=None):
        super().__init__()
        self.prompt = prompt
        self.model_name = model_name
        self.aspect_ratio = aspect_ratio
        self.instructions = instructions
        self.api_type = api_type

    def run(self):
        self.log_signal.emit(f"\n🚀 开始请求生图 API (模型: {self.model_name})...")
        self.log_signal.emit("请耐心等待，这可能需要几十秒的时间...")
        try:
            # 根据api_type调用相应的生成函数
            if self.api_type == "aigc2d":
                saved_files = generate_image_aigc2d(
                    prompt=self.prompt, 
                    image_paths=[],
                    model=self.model_name, 
                    aspect_ratio=self.aspect_ratio, 
                    instructions=self.instructions,
                    api_type=self.api_type
                )
            else:
                saved_files = generate_image_whatai(
                    prompt=self.prompt, 
                    image_paths=[],
                    model=self.model_name, 
                    aspect_ratio=self.aspect_ratio, 
                    instructions=self.instructions,
                    api_type=self.api_type
                )
            self.finish_signal.emit(saved_files)
        except Exception as e:
            self.log_signal.emit(f"❌ 生图请求发生异常: {e}")
            self.finish_signal.emit([])

# --- 单图分析核心界面 Widget ---
class SingleAnalyzerWidget(QWidget):
    def __init__(self, config_getter_func, img_config_getter_func, styles_getter_func, save_img_cfg_callback, ar_policy_getter_func=None):
        super().__init__()
        self.get_text_config = config_getter_func
        self.get_img_config = img_config_getter_func
        self.get_styles = styles_getter_func
        self.save_img_cfg = save_img_cfg_callback
        
        self.image_source = None
        self.current_aspect_ratio = "1:1"
        self.current_orig_desc = ""
        self.current_refine_desc = ""

        # 【新增】用来保存正在执行的生图线程池，防止被垃圾回收
        self._active_img_threads = []

        self.get_ar_policy = ar_policy_getter_func
        
        self.initUI()
        
    def initUI(self):
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.StrongFocus)
        layout = QVBoxLayout()

        self.image_label = QLabel("请将图片拖拽至此，\n或在窗口内按 Ctrl+V 粘贴")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color : #f0f0f0; border: 2px dashed #aaa; font-size: 16px; }")
        self.image_label.setMinimumHeight(220)
        layout.addWidget(self.image_label)

        self.send_btn = QPushButton("① 发送并生成分析描述")
        self.send_btn.setFixedHeight(40)
        self.send_btn.clicked.connect(self.process_image)
        self.send_btn.setEnabled(False) 
        layout.addWidget(self.send_btn)

        # 【新增】两项自动生成图片的勾选框
        auto_gen_layout = QHBoxLayout()
        self.auto_gen_orig_cb = QCheckBox("分析完成后立即生成图片（基于原始提示词）")
        self.auto_gen_ref_cb = QCheckBox("分析完成后立即生成图片（基于优化提示词）")
        auto_gen_layout.addWidget(self.auto_gen_orig_cb)
        auto_gen_layout.addWidget(self.auto_gen_ref_cb)
        layout.addLayout(auto_gen_layout)

        # 画风选择
        style_select_layout = QHBoxLayout()
        style_select_layout.addWidget(QLabel("生成时使用的画风预设:"))
        self.main_style_combo = QComboBox()
        self.main_style_combo.setMaximumWidth(200)
        style_select_layout.addWidget(self.main_style_combo)
        
        self.reload_styles_btn = QPushButton("🔄 重新加载配置")
        self.reload_styles_btn.setFixedWidth(120)
        self.reload_styles_btn.clicked.connect(self.reload_styles)
        style_select_layout.addWidget(self.reload_styles_btn)
        
        style_select_layout.addStretch()
        layout.addLayout(style_select_layout)
        

        
        gen_img_layout = QHBoxLayout()
        self.gen_orig_btn = QPushButton("② 生成图片 (基于 原始 提示词)")
        self.gen_orig_btn.setFixedHeight(35)
        self.gen_orig_btn.clicked.connect(lambda: self.trigger_image_generation("original"))
        self.gen_orig_btn.setEnabled(False)
        
        self.gen_ref_btn = QPushButton("② 生成图片 (基于 优化 提示词)")
        self.gen_ref_btn.setFixedHeight(35)
        self.gen_ref_btn.clicked.connect(lambda: self.trigger_image_generation("refined"))
        self.gen_ref_btn.setEnabled(False)
        
        gen_img_layout.addWidget(self.gen_orig_btn)
        gen_img_layout.addWidget(self.gen_ref_btn)
        layout.addLayout(gen_img_layout)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        self.setLayout(layout)

    def _resolve_ar_for_first_stage(self, original_ar: str) -> str:
        """第一次：分析完成后用于保存 prompts 的长宽比"""
        if not self.get_ar_policy:
            return original_ar
        policy = self.get_ar_policy() or {}
        override_first = (policy.get("override_first") or "").strip()
        if override_first.startswith("不覆盖"):
            return original_ar
        # 【修改前】return default_ar
        # 【修改后】返回用户选择的覆盖比例
        return override_first

    def _resolve_ar_for_second_stage(self, original_ar: str) -> str:
        """第二次：真正调用生图接口时的长宽比"""
        if not self.get_ar_policy:
            return original_ar
        policy = self.get_ar_policy() or {}
        override_second = (policy.get("override_second") or "").strip()
        if override_second.startswith("不覆盖"):
            return original_ar
        # 【修改前】return default_ar
        # 【修改后】返回用户选择的覆盖比例
        return override_second


    def update_styles(self, style_keys):
        """由外部 app.py 调用以同步最新的画风列表"""
        curr_main = self.main_style_combo.currentText()
        self.main_style_combo.blockSignals(True)
        self.main_style_combo.clear()
        self.main_style_combo.addItems(style_keys)
        if curr_main in style_keys:
            self.main_style_combo.setCurrentText(curr_main)
        self.main_style_combo.blockSignals(False)

    def reload_styles(self):
        """重新加载 config-styles.json 配置文件"""
        try:
            import json
            config_path = os.path.join(os.path.dirname(__file__), 'config-styles.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                styles_data = json.load(f)
            style_keys = list(styles_data.keys())
            self.update_styles(style_keys)
            self.log_msg(f"✅ 已重新加载配置文件，共 {len(style_keys)} 个画风预设")
        except Exception as e:
            self.log_msg(f"❌ 重新加载配置文件失败: {e}")

    def mousePressEvent(self, event):
        self.setFocus()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp')):
                self.image_source = file_path
                self.show_preview(file_path)
                self.log_msg(f"已加载图片: {file_path}")
            else:
                self.log_msg("不支持的文件格式，请拖入图片。")

    def keyPressEvent(self, event):
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_V:
            clipboard_img = ImageGrab.grabclipboard()
            if isinstance(clipboard_img, Image.Image):
                self.image_source = clipboard_img
                self.log_msg("已从剪贴板加载图片。")
                self.show_clipboard_preview(clipboard_img)
            else:
                self.log_msg("剪贴板中没有有效的图片。")

    def show_preview(self, filepath):
        pixmap = QPixmap(filepath)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.send_btn.setEnabled(True)

    def show_clipboard_preview(self, pil_image):
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        image = QImage()
        image.loadFromData(img_byte_arr)
        pixmap = QPixmap.fromImage(image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.send_btn.setEnabled(True)

    def log_msg(self, text):
        self.log_text.append(text)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def process_image(self):
        if not self.image_source: return
            
        base_url, api_key, model_name = self.get_text_config()
        if not api_key or not model_name:
            QMessageBox.warning(self, "缺少配置", "文本分析 API Key 和 模型名称不能为空！")
            return
        
        self.send_btn.setEnabled(False)
        self.gen_orig_btn.setEnabled(False)
        self.gen_ref_btn.setEnabled(False)
        self.log_text.clear()
        self.log_msg("任务已启动...\n")
        
        self.thread = WorkerThread(self.image_source, api_key, base_url, model_name)
        self.thread.log_signal.connect(self.log_msg)
        self.thread.finish_signal.connect(self.on_process_finished)
        self.thread.start()

    def on_process_finished(self, result_json):
        self.send_btn.setEnabled(True)
        if not result_json:
            self.log_msg("\n处理失败，未能获取到有效的 JSON 数据。")
            return

        self.log_msg("\n========== 最终处理结果 ==========\n")
        self.log_msg(json.dumps(result_json, indent=4, ensure_ascii=False))
        
        jp_title = result_json.get("japanese_title", "未命名")
        safe_title = re.sub(r'[\\/*?:"<>|]', "", jp_title).strip() or "未命名"
        
        now = datetime.datetime.now()
        now_str = now.strftime("%Y%m%d-%H%M%S")
        date_str = now.strftime("%Y%m%d") 
        save_dir = os.path.join('data', date_str) 
        
        if not os.path.exists(save_dir): os.makedirs(save_dir) 
            
        base_filename = f"{now_str}-{safe_title}"
        json_filename = f"{base_filename}.json"
        
        try:
            with open(os.path.join(save_dir, json_filename), "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=4)
            self.log_msg(f"\n✅ 成功！JSON 结果已保存至: {json_filename}")
        except Exception as e:
            self.log_msg(f"\n❌ 保存 JSON 文件时出错: {e}")

        try:
            raw_ar = result_json.get("aspect_ratio", "2:3")
            self.current_aspect_ratio = self._resolve_ar_for_first_stage(raw_ar)
            self.log_msg(f"📌 确定的图片长宽比: {self.current_aspect_ratio}")

            self.current_refine_desc = result_json.get("english_description", "")
            self.current_orig_desc = result_json.get("original_english_description", "")
            
            
            selected_style_name = self.main_style_combo.currentText()
            styles_data = self.get_styles()
            current_fixed_tags = styles_data.get(selected_style_name, "")
            
            # 在风格标签和描述之间添加两个回车
            style_part = f"--ar {self.current_aspect_ratio} {current_fixed_tags}".strip()
            final_prompt = f"{style_part}\n\n{self.current_refine_desc}".strip()
            orig_prompt = f"{style_part}\n\n{self.current_orig_desc}".strip()
            
            txt_filename = f"{base_filename}-prompts.txt"
            orig_txt_filename = f"{base_filename}-original-prompts.txt"
            
            with open(os.path.join(save_dir, txt_filename), "w", encoding="utf-8") as f: f.write(final_prompt)
            with open(os.path.join(save_dir, orig_txt_filename), "w", encoding="utf-8") as f: f.write(orig_prompt)
                
            self.log_msg(f"✅ 成功！两份画幅与提示词文件已保存:\n - {txt_filename}\n - {orig_txt_filename}")
            
            self.gen_orig_btn.setEnabled(True)
            self.gen_ref_btn.setEnabled(True)
            self.log_msg("\n💡 提示: 现在可以点击下方的按钮，根据提取的描述直接生成图片了！")
        except Exception as e:
            self.log_msg(f"❌ 保存提示词 txt 文件时出错: {e}")

        # 【新增】自动执行发图逻辑
        if self.auto_gen_orig_cb.isChecked():
            self.trigger_image_generation("original")
        if self.auto_gen_ref_cb.isChecked():
            self.trigger_image_generation("refined")

    def trigger_image_generation(self, prompt_type):
        self.save_img_cfg()
        
        img_base_url, img_key, model_name, api_type = self.get_img_config()
        if not img_key:
            QMessageBox.warning(self, "缺少配置", "生图 API Key 不能为空，请检查【全局配置】。")
            return
            
        prompt_to_use = self.current_orig_desc if prompt_type == "original" else self.current_refine_desc
        
        selected_style_name = self.main_style_combo.currentText()
        styles_data = self.get_styles()
        active_instructions = styles_data.get(selected_style_name, "")
        
        self.gen_orig_btn.setEnabled(False)
        self.gen_ref_btn.setEnabled(False)
        
        # 【修改】动态实例化线程对象存放至列表，避免并发勾选导致线程互相覆盖报错
        final_gen_ar = self._resolve_ar_for_second_stage(self.current_aspect_ratio)

        img_thread = ImageGenWorkerThread(
            prompt=prompt_to_use,
            model_name=model_name,
            aspect_ratio=final_gen_ar,
            instructions=active_instructions,
            api_type=api_type
        )

        self._active_img_threads.append(img_thread)
        
        img_thread.log_signal.connect(self.log_msg)
        img_thread.finish_signal.connect(self.on_image_generation_finished)
        
        # 清除完成的线程
        img_thread.finished.connect(lambda t=img_thread: self._active_img_threads.remove(t) if t in self._active_img_threads else None)
        img_thread.start()

    def on_image_generation_finished(self, saved_files):
        self.gen_orig_btn.setEnabled(True)
        self.gen_ref_btn.setEnabled(True)
        
        if saved_files:
            self.log_msg(f"\n🎉 成功生成了 {len(saved_files)} 张图片！")
            for file_path in saved_files:
                self.log_msg(f" 📂 保存路径: {file_path}")
        else:
            self.log_msg("\n⚠️ 未能获取到图片，请检查上方日志，或查看日志文件夹（log）的记录。")