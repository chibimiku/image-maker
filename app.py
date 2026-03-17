import os
import json
import base64
import io
import datetime
import re
from openai import OpenAI
from PIL import Image, ImageGrab

# --- PyQt5 界面相关库 ---
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTextEdit, QLineEdit, 
                             QComboBox, QGroupBox, QFormLayout, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

CONFIG_FILE = "config.json"

system_prompt = """
You are an expert image analyzer and illustrator assistant. 
You must respond strictly in JSON format.
"""

user_prompt_analyze = """
Please analyze the provided image and generate a highly detailed description in English (approximately 500 words). 
Include the following elements: art style, composition, lighting, camera angle, character poses and actions, relationships and interactions between characters, hair color, and eye color. 
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
    if not base64_image:
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
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            temperature=0.7,
			max_completion_tokens=16384
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
    
    原英文描述：
    {original_description}
    
    原标签（需根据新描述稍作替换，总数保持12个，保持语言为日语tag）：
    {tags_str}
    
    预期 JSON 结构参考：
    {{
      "english_description": "<这里填入修改后的英文描述>",
      "japanese_title": "{jp_title}",
      "chinese_title": "{cn_title}",
      "pixiv_tags": ["<填入更新后的12个标签>"],
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
            temperature=0.7,
			max_completion_tokens=16384
        )
        
        final_result_json = json.loads(response.choices[0].message.content)
        final_result_json["original_english_description"] = original_description
        
        return final_result_json
    except Exception as e:
        print(f"Step 2 二次加工时发生错误: {e}")
        return None

def get_fixed_content():
    fixed_file = "data/fixed_tags.txt"
    if not os.path.exists(fixed_file):
        os.makedirs("data", exist_ok=True)
        with open(fixed_file, "w", encoding="utf-8") as f:
            f.write("")
    
    try:
        with open(fixed_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""

# --- API 请求工作线程 ---
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

# --- PyQt5 主界面 ---
class AppWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.image_source = None
        self.initUI()
        self.load_config()

    def initUI(self):
        self.setWindowTitle("图片分析与描述生成器")
        self.resize(700, 750)
        self.setAcceptDrops(True)

        layout = QVBoxLayout()

        # --- 配置区域 ---
        config_group = QGroupBox("API 配置")
        config_layout = QFormLayout()

        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("例如: https://api.openai.com/v1")
        config_layout.addRow("Base URL:", self.url_input)

        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("API_KEY")
        self.key_input.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        config_layout.addRow("API Key:", self.key_input)

        # 模型选择与获取
        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True) # 允许用户自己输入模型名称
        self.model_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        model_layout.addWidget(self.model_combo, stretch=1)

        self.fetch_btn = QPushButton("获取模型列表")
        self.fetch_btn.clicked.connect(self.fetch_models)
        model_layout.addWidget(self.fetch_btn)
        
        config_layout.addRow("模型 (Model):", model_layout)

        # 保存配置按钮
        self.save_cfg_btn = QPushButton("保存配置")
        self.save_cfg_btn.clicked.connect(self.save_config)
        config_layout.addRow("", self.save_cfg_btn)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # --- 图片和操作区域 ---
        self.image_label = QLabel("请将图片拖拽至此，\n或在窗口内按 Ctrl+V 粘贴")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color : #f0f0f0; border: 2px dashed #aaa; font-size: 16px; }")
        self.image_label.setMinimumHeight(250)
        layout.addWidget(self.image_label)

        self.send_btn = QPushButton("发送并生成分析")
        self.send_btn.setFixedHeight(40)
        self.send_btn.clicked.connect(self.process_image)
        self.send_btn.setEnabled(False) 
        layout.addWidget(self.send_btn)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        self.setLayout(layout)

        # [新增代码] 设置主窗口的焦点策略为强焦点，并在启动时将焦点赋给主窗口
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self.url_input.setText(config.get("base_url", ""))
                    self.key_input.setText(config.get("api_key", ""))
                    
                    saved_model = config.get("model", "")
                    if saved_model:
                        self.model_combo.clear()
                        self.model_combo.addItem(saved_model)
                        self.model_combo.setCurrentText(saved_model)
            except Exception as e:
                self.log_msg(f"加载配置文件失败: {e}")

    # [新增方法] 放在 AppWindow 类里面的任意位置，比如 keyPressEvent 下方
    def mousePressEvent(self, event):
        # 当用户点击窗口空白处或图片 Label 时，强制主窗口获取焦点，从而取消输入框的焦点
        self.setFocus()
        super().mousePressEvent(event)

    def save_config(self):
        config = {
            "base_url": self.url_input.text().strip(),
            "api_key": self.key_input.text().strip(),
            "model": self.model_combo.currentText().strip()
        }
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            QMessageBox.information(self, "成功", "配置已保存至 config.json")
        except Exception as e:
            QMessageBox.warning(self, "失败", f"保存配置文件失败: {e}")

    def fetch_models(self):
        api_key = self.key_input.text().strip()
        base_url = self.url_input.text().strip()
        
        if not api_key:
            QMessageBox.warning(self, "错误", "请先输入 API Key")
            return
            
        self.fetch_btn.setEnabled(False)
        self.fetch_btn.setText("获取中...")
        QApplication.processEvents()

        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            models = client.models.list()
            # 提取模型名称并排序
            model_names = sorted([m.id for m in models.data])
            
            # 保留当前选中的文本
            current_text = self.model_combo.currentText()
            
            self.model_combo.clear()
            self.model_combo.addItems(model_names)
            
            # 如果之前的文本在新列表中，自动选中它，否则保留为自定义输入
            index = self.model_combo.findText(current_text)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
            elif current_text:
                self.model_combo.setCurrentText(current_text)
                
            QMessageBox.information(self, "成功", f"成功获取 {len(model_names)} 个可用模型！")
        except Exception as e:
            QMessageBox.warning(self, "获取失败", f"获取模型列表失败，请检查 URL 和 Key 是否正确。\n错误信息: {e}")
        finally:
            self.fetch_btn.setEnabled(True)
            self.fetch_btn.setText("获取模型列表")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

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
        if not self.image_source:
            return
            
        api_key = self.key_input.text().strip()
        base_url = self.url_input.text().strip()
        model_name = self.model_combo.currentText().strip()
        
        if not api_key or not model_name:
            QMessageBox.warning(self, "缺少配置", "API Key 和 模型名称不能为空！")
            return
        
        self.send_btn.setEnabled(False)
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
        safe_title = re.sub(r'[\\/*?:"<>|]', "", jp_title).strip()
        if not safe_title:
            safe_title = "未命名"
        
        now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = 'data' 
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        base_filename = f"{now_str}-{safe_title}"
        json_filename = f"{base_filename}.json"
        
        try:
            with open( os.path.join(save_dir, json_filename), "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=4)
            self.log_msg(f"\n✅ 成功！JSON 结果已保存至: {json_filename}")
        except Exception as e:
            self.log_msg(f"\n❌ 保存 JSON 文件时出错: {e}")

        try:
            aspect_ratio = result_json.get("aspect_ratio", "2:3")
            eng_desc = result_json.get("english_description", "")
            orig_eng_desc = result_json.get("original_english_description", "")
            fixed_content = get_fixed_content()
            
            final_prompt = f"--ar {aspect_ratio} {fixed_content} {eng_desc}".strip()
            orig_prompt = f"--ar {aspect_ratio} {fixed_content} {orig_eng_desc}".strip()
            
            txt_filename = f"{base_filename}-prompts.txt"
            orig_txt_filename = f"{base_filename}-original-prompts.txt"
            
            with open( os.path.join(save_dir, txt_filename), "w", encoding="utf-8") as f:
                f.write(final_prompt)
            
            with open( os.path.join(save_dir, orig_txt_filename), "w", encoding="utf-8") as f:
                f.write(orig_prompt)
                
            self.log_msg(f"✅ 成功！两份画幅与提示词文件已保存:\n - {txt_filename}\n - {orig_txt_filename}")
        except Exception as e:
            self.log_msg(f"❌ 保存提示词 txt 文件时出错: {e}")

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec_())