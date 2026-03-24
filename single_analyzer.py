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

from api_backend import generate_image_whatai 

system_prompt = """
You are an expert image analyzer and illustrator assistant. 
You must respond strictly in JSON format.
"""

user_prompt_analyze = """
Please analyze the provided image and generate a highly detailed description in English (approximately 500 words). 
Include the following elements: art style, composition, lighting, camera angle, hair color, and eye color. 

CRITICAL ACTION REQUIREMENT: Meticulously describe the precise character poses and dynamic actions. You must explicitly detail their body language, the positioning of their limbs, and exactly how their movements physically interact with the surrounding environment, objects, and other characters.

If present in the original image, meticulously describe the clothing (tops and bottoms, or dresses), painted patterns on the clothing, types and colors of shoes and socks, accessories, and the environment. 
Do not use ambiguous language. Do not describe any text that appears in the image. 

Important style constraints: 
If the original image's art style is a photograph (photo), describe it as an 'illustration' and adjust all other domain descriptions to fit an illustration style. 
Always use the word 'girl' to describe female characters. 
Output an English text description. Do not generate an image.
If the concept of 'lolita' applies, use 'rococo' instead. 
Strictly prohibit sexually explicit or NSFW words, including 'cleavage' and 'nude'.

Additionally, provide the following based on the image content:
1. A poetic Japanese title using complex kanji (maximum 20 characters).
2. The Chinese translation of this title.
3. Exactly 12 Japanese tags suitable for the Pixiv tagging system (e.g., 女の子) that accurately describe the visual content.

Return the result strictly as a JSON object with the following keys:
{
  "english_description": "...",
  "japanese_title": "...",
  "chinese_title": "...",
  "pixiv_tags": ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7", "tag8", "tag9", "tag10", "tag11", "tag12"]
}
"""

def compress_and_encode_image(image_source, max_dim=2048):
    try:
        if isinstance(image_source, str):
            img = Image.open(image_source)
        else:
            img = image_source 

        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        original_width, original_height = img.size
        print(f"原始图片尺寸: {original_width}x{original_height}")

        if max(original_width, original_height) > max_dim:
            scaling_factor = max_dim / max(original_width, original_height)
            new_width = int(original_width * scaling_factor)
            new_height = int(original_height * scaling_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"图片已成功压缩为: {new_width}x{new_height}")

        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=100)
        base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return "image/jpeg", base64_string

    except Exception as e:
        print(f"处理图片时发生错误: {e}")
        return None, None

def step_1_analyze_image(image_source, client, model_name):
    mime_type, base64_image = compress_and_encode_image(image_source)
    if not base64_image: return None
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
        print(f"Step 1 请求错误: {e}")
        return None

def step_2_refine_description(original_json_data, client, model_name):
    original_description = original_json_data.get("english_description", "")
    jp_title = original_json_data.get("japanese_title", "")
    cn_title = original_json_data.get("chinese_title", "")
    tags = original_json_data.get("pixiv_tags", [])
    
    tags_str = json.dumps(tags, ensure_ascii=False)
    
    refine_prompt = f"""
    请根据以下中文指示，修改并丰富下面提供的英文图片描述。
    
    修改要求：
    1. 增加背景描述。
    2. 强调人物动作与场景的互动，使人物经常处于“重心不稳定”的状态（例如踮脚、跳起、失去平衡等），以此构造出具有强烈动感的画面。同时像是在讲述一个故事，人物表情和场景相符，尽量避免让人物位于画面最中央（使用 rule of thirds 或 off-center 构图）。
    3. 给人物增加蕾丝半透明有宝石的手套。
    4. 人物穿着蕾丝半透明过膝吊带袜，有着丝袜质感，上面还有与衣服风格相匹配的刺绣。
    5. 增加人物衣服和鞋子上的交叉系带元素。
    6. 如果人物有穿着高跟鞋的描述，则把鞋跟高度的描述修改得更高，并带有小饰品。
    7. 增加人物姿势中大腿分开的描述。
    8. 根据当前描述的构图和场景内容，推断最合适的画幅长宽比（例如竖图推荐 9:16 或 2:3，横图推荐 16:9 或 3:2，正方形推荐 1:1）。
    
    约束条件：
    - 输出的图片描述必须全为英文，字数维持在约 600-750 词。
    - 维持设定的安全与风格限制（禁止使用'cleavage'、'nude'，若符合'lolita'概念请替换为'rococo'）。
    - 标签总数必须严格保持在 12 个，请根据新增的描述替换部分原有标签。
    - 必须输出严格的 JSON 格式，保留原有标题，并新增 "aspect_ratio" 字段。
    
    以下是 Step 1 已经生成好的基础数据，请在最终输出的 JSON 中直接保留这两个标题：
    原日文标题：{jp_title}
    原中文标题：{cn_title}
    原英文描述：\n{original_description}
    原标签：\n{tags_str}
    
    预期 JSON 结构参考：
    {{
      "english_description": "<修改后的英文描述>",
      "japanese_title": "{jp_title}",
      "chinese_title": "{cn_title}",
      "pixiv_tags": ["<更新后的12个标签>"],
      "aspect_ratio": "<例如 2:3>"
    }}
    """
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
        initial_result = step_1_analyze_image(self.image_source, client, self.model_name)
        
        if initial_result:
            self.log_signal.emit("Step 1 完成。初步结果已获取。")
            self.log_signal.emit("正在开始 Step 2: 根据中文指令对英文描述进行加工并推断长宽比...")
            final_result = step_2_refine_description(initial_result, client, self.model_name)
            self.finish_signal.emit(final_result if final_result else {})
        else:
            self.log_signal.emit("Step 1 失败，流程终止。")
            self.finish_signal.emit({})

class ImageGenWorkerThread(QThread):
    log_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(list)

    def __init__(self, prompt, model_name, aspect_ratio, instructions):
        super().__init__()
        self.prompt = prompt
        self.model_name = model_name
        self.aspect_ratio = aspect_ratio
        self.instructions = instructions

    def run(self):
        self.log_signal.emit(f"\n🚀 开始请求生图 API (模型: {self.model_name})...")
        self.log_signal.emit("请耐心等待，这可能需要几十秒的时间...")
        try:
            saved_files = generate_image_whatai(
                prompt=self.prompt, 
                image_paths=[],
                model=self.model_name, 
                aspect_ratio=self.aspect_ratio, 
                instructions=self.instructions
            )
            self.finish_signal.emit(saved_files)
        except Exception as e:
            self.log_signal.emit(f"❌ 生图请求发生异常: {e}")
            self.finish_signal.emit([])

# --- 单图分析核心界面 Widget ---
class SingleAnalyzerWidget(QWidget):
    def __init__(self, config_getter_func, img_config_getter_func, styles_getter_func, save_img_cfg_callback):
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

        style_select_layout = QHBoxLayout()
        style_select_layout.addWidget(QLabel("生成时使用的画风预设:"))
        self.main_style_combo = QComboBox()
        style_select_layout.addWidget(self.main_style_combo, stretch=1)
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

    def update_styles(self, style_keys):
        """由外部 app.py 调用以同步最新的画风列表"""
        curr_main = self.main_style_combo.currentText()
        self.main_style_combo.blockSignals(True)
        self.main_style_combo.clear()
        self.main_style_combo.addItems(style_keys)
        if curr_main in style_keys:
            self.main_style_combo.setCurrentText(curr_main)
        self.main_style_combo.blockSignals(False)

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
            self.current_aspect_ratio = result_json.get("aspect_ratio", "2:3")
            self.current_refine_desc = result_json.get("english_description", "")
            self.current_orig_desc = result_json.get("original_english_description", "")
            
            selected_style_name = self.main_style_combo.currentText()
            styles_data = self.get_styles()
            current_fixed_tags = styles_data.get(selected_style_name, "")
            
            final_prompt = f"--ar {self.current_aspect_ratio} {current_fixed_tags} {self.current_refine_desc}".strip()
            orig_prompt = f"--ar {self.current_aspect_ratio} {current_fixed_tags} {self.current_orig_desc}".strip()
            
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
        
        img_base_url, img_key, model_name = self.get_img_config()
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
        img_thread = ImageGenWorkerThread(
            prompt=prompt_to_use,
            model_name=model_name,
            aspect_ratio=self.current_aspect_ratio,
            instructions=active_instructions
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