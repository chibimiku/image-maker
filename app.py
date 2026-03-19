import os
import json
import base64
import io
import datetime
import re
from openai import OpenAI
from PIL import Image, ImageGrab

# 引入后端的生图函数
from api_backend import generate_image_whatai 

# --- PyQt5 界面相关库 ---
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTextEdit, QLineEdit, QInputDialog,
                             QComboBox, QGroupBox, QFormLayout, QMessageBox,
                             QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage

# 引入我们刚刚新建的多图画风分析模块
from style_analyzer import StyleAnalyzerWidget

CONFIG_FILE = "config.json"
CONFIG_IMAGE_FILE = "config-image.json"
CONFIG_STYLES_FILE = "config-styles.json"

# 默认内置的一套空画风
DEFAULT_STYLES = {
    "默认(无附加)": ""
}

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

# --- 图片生成 API 请求工作线程 ---
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
            # api_backend 会自行读取 config-image.json，所以在启动线程前需要确保文件已保存
            saved_files = generate_image_whatai(
                prompt=self.prompt, 
                image_paths=[], # 这里使用纯文本生图
                model=self.model_name, 
                aspect_ratio=self.aspect_ratio, 
                instructions=self.instructions
            )
            self.finish_signal.emit(saved_files)
        except Exception as e:
            self.log_signal.emit(f"❌ 生图请求发生异常: {e}")
            self.finish_signal.emit([])

# --- PyQt5 主界面 ---
class AppWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.image_source = None
        
        # 保存当前解析出的数据供生图使用
        self.current_aspect_ratio = "1:1"
        self.current_orig_desc = ""
        self.current_refine_desc = ""
        self.styles_data = DEFAULT_STYLES.copy() # <--- 新增数据容器

        self.initUI()
        self.load_config()
        self.load_styles_config()

    def initUI(self):
        self.setWindowTitle("图片分析与描述生成器")
        self.resize(700, 750)
        self.setAcceptDrops(True)

        main_layout = QVBoxLayout()

        # --- 顶部配置选项卡 ---
        self.tabs = QTabWidget()
        
        # 标签页 1：文本分析配置
        tab_text = QWidget()
        text_layout = QFormLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("例如: https://api.openai.com/v1")
        text_layout.addRow("Base URL:", self.url_input)
        
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("API_KEY")
        self.key_input.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        text_layout.addRow("API Key:", self.key_input)
        
        model_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True) 
        model_layout.addWidget(self.model_combo, stretch=1)
        self.fetch_btn = QPushButton("获取模型列表")
        self.fetch_btn.clicked.connect(self.fetch_models)
        model_layout.addWidget(self.fetch_btn)
        text_layout.addRow("分析模型:", model_layout)
        
        self.save_text_cfg_btn = QPushButton("保存分析配置")
        self.save_text_cfg_btn.clicked.connect(self.save_text_config)
        text_layout.addRow("", self.save_text_cfg_btn)
        tab_text.setLayout(text_layout)
        
        # 标签页 2：图片生成配置
        tab_image = QWidget()
        image_layout = QFormLayout()
        self.img_url_input = QLineEdit()
        self.img_url_input.setPlaceholderText("例如: https://api.whatai.cc/v1")
        image_layout.addRow("Base URL:", self.img_url_input)
        
        self.img_key_input = QLineEdit()
        self.img_key_input.setPlaceholderText("API_KEY (config-image.json)")
        self.img_key_input.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        image_layout.addRow("API Key:", self.img_key_input)
        
        self.img_model_combo = QComboBox()
        self.img_model_combo.setEditable(True)
        # 默认填入一个后端的模型
        self.img_model_combo.addItem("nano-banana-2") 
        image_layout.addRow("生图模型:", self.img_model_combo)
        
        self.save_img_cfg_btn = QPushButton("保存生图配置")
        self.save_img_cfg_btn.clicked.connect(self.save_image_config)
        image_layout.addRow("", self.save_img_cfg_btn)
        tab_image.setLayout(image_layout)

        # --- 标签页 3：画风预设管理 ---
        tab_style = QWidget()
        style_layout = QVBoxLayout()
        
        style_top_layout = QHBoxLayout()
        style_top_layout.addWidget(QLabel("选择预设:"))
        self.style_manage_combo = QComboBox()
        self.style_manage_combo.currentTextChanged.connect(self.on_manage_style_changed)
        style_top_layout.addWidget(self.style_manage_combo, stretch=1)
        
        self.add_style_btn = QPushButton("新建预设")
        self.add_style_btn.clicked.connect(self.add_new_style)
        self.del_style_btn = QPushButton("删除预设")
        self.del_style_btn.clicked.connect(self.delete_current_style)
        style_top_layout.addWidget(self.add_style_btn)
        style_top_layout.addWidget(self.del_style_btn)
        
        style_layout.addLayout(style_top_layout)
        
        self.style_content_edit = QTextEdit()
        self.style_content_edit.setPlaceholderText("在此输入该预设固定的提示词 (例如: rococo style, masterpiece...)")
        style_layout.addWidget(self.style_content_edit)
        
        self.save_style_btn = QPushButton("保存当前预设")
        self.save_style_btn.clicked.connect(self.save_current_style)
        style_layout.addWidget(self.save_style_btn)
        
        tab_style.setLayout(style_layout)

        
        self.tabs.addTab(tab_text, "文本分析 API (config.json)")
        self.tabs.addTab(tab_image, "图片生成 API (config-image.json)")
        self.tabs.addTab(tab_style, "画风预设管理")

        # 初始化独立的多图画风分析 Widget
        # 传入一个 lambda 函数，让子组件能够实时获取当前的 URL、Key 和模型名称
        self.style_analyzer_tab = StyleAnalyzerWidget(
            config_getter_func=lambda: (
                self.url_input.text().strip(), 
                self.key_input.text().strip(), 
                self.model_combo.currentText().strip()
            )
        )
        self.tabs.addTab(self.style_analyzer_tab, "多图画风提取")

        main_layout.addWidget(self.tabs)

        # --- 图片和操作区域 ---
        self.image_label = QLabel("请将图片拖拽至此，\n或在窗口内按 Ctrl+V 粘贴")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color : #f0f0f0; border: 2px dashed #aaa; font-size: 16px; }")
        self.image_label.setMinimumHeight(220)
        main_layout.addWidget(self.image_label)

        # 分析按钮
        self.send_btn = QPushButton("① 发送并生成分析描述")
        self.send_btn.setFixedHeight(40)
        self.send_btn.clicked.connect(self.process_image)
        self.send_btn.setEnabled(False) 
        main_layout.addWidget(self.send_btn)

        style_select_layout = QHBoxLayout()
        style_select_layout.addWidget(QLabel("生成时使用的画风预设:"))
        self.main_style_combo = QComboBox()
        style_select_layout.addWidget(self.main_style_combo, stretch=1)
        main_layout.addLayout(style_select_layout)
        
        # 生图按钮组（初始禁用）
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
        main_layout.addLayout(gen_img_layout)

        # 日志输出区域
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text)

        self.setLayout(main_layout)

        # 设置主窗口的焦点策略
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

    def load_config(self):
        # 1. 加载文本配置
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
                self.log_msg(f"加载 {CONFIG_FILE} 失败: {e}")
                
        # 2. 加载图片配置
        if os.path.exists(CONFIG_IMAGE_FILE):
            try:
                with open(CONFIG_IMAGE_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self.img_url_input.setText(config.get("base_url", "https://api.whatai.cc/v1"))
                    self.img_key_input.setText(config.get("api_key", ""))
                    
                    saved_model = config.get("model", "")
                    if saved_model:
                        if self.img_model_combo.findText(saved_model) == -1:
                            self.img_model_combo.addItem(saved_model)
                        self.img_model_combo.setCurrentText(saved_model)
            except Exception as e:
                self.log_msg(f"加载 {CONFIG_IMAGE_FILE} 失败: {e}")

    def load_styles_config(self):
        if os.path.exists(CONFIG_STYLES_FILE):
            try:
                with open(CONFIG_STYLES_FILE, "r", encoding="utf-8") as f:
                    loaded_styles = json.load(f)
                    if loaded_styles:
                        self.styles_data = loaded_styles
            except Exception as e:
                self.log_msg(f"加载画风配置失败: {e}")
        self.update_style_combos()

    def update_style_combos(self):
        curr_manage = self.style_manage_combo.currentText()
        curr_main = self.main_style_combo.currentText()
        
        self.style_manage_combo.blockSignals(True)
        self.style_manage_combo.clear()
        self.main_style_combo.clear()
        
        keys = list(self.styles_data.keys())
        self.style_manage_combo.addItems(keys)
        self.main_style_combo.addItems(keys)
        
        if curr_manage in keys: self.style_manage_combo.setCurrentText(curr_manage)
        if curr_main in keys: self.main_style_combo.setCurrentText(curr_main)
            
        self.style_manage_combo.blockSignals(False)
        self.on_manage_style_changed(self.style_manage_combo.currentText())

    def on_manage_style_changed(self, style_name):
        if style_name in self.styles_data:
            self.style_content_edit.setPlainText(self.styles_data[style_name])

    def save_current_style(self):
        style_name = self.style_manage_combo.currentText()
        if not style_name: return
        self.styles_data[style_name] = self.style_content_edit.toPlainText().strip()
        self.save_styles_to_disk()
        QMessageBox.information(self, "成功", f"画风预设 '{style_name}' 已保存！")

    def save_styles_to_disk(self):
        try:
            with open(CONFIG_STYLES_FILE, "w", encoding="utf-8") as f:
                json.dump(self.styles_data, f, ensure_ascii=False, indent=4)
            self.update_style_combos()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存画风文件失败: {e}")

    def add_new_style(self):
        text, ok = QInputDialog.getText(self, '新建预设', '请输入新画风预设的名称:')
        if ok and text.strip():
            name = text.strip()
            if name in self.styles_data:
                QMessageBox.warning(self, "提示", "预设名称已存在！")
                return
            self.styles_data[name] = ""
            self.update_style_combos()
            self.style_manage_combo.setCurrentText(name)

    def delete_current_style(self):
        style_name = self.style_manage_combo.currentText()
        if not style_name or style_name == "默认(无附加)":
            QMessageBox.warning(self, "提示", "无法删除默认预设！")
            return
        reply = QMessageBox.question(self, '确认删除', f"确定要删除 '{style_name}' 吗？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            del self.styles_data[style_name]
            self.save_styles_to_disk()

    # [新增方法] 放在 AppWindow 类里面的任意位置，比如 keyPressEvent 下方
    def mousePressEvent(self, event):
        # 当用户点击窗口空白处或图片 Label 时，强制主窗口获取焦点，从而取消输入框的焦点
        self.setFocus()
        super().mousePressEvent(event)

    def save_text_config(self):
        config = {
            "base_url": self.url_input.text().strip(),
            "api_key": self.key_input.text().strip(),
            "model": self.model_combo.currentText().strip()
        }
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            QMessageBox.information(self, "成功", f"配置已保存至 {CONFIG_FILE}")
        except Exception as e:
            QMessageBox.warning(self, "失败", f"保存配置文件失败: {e}")
            
    def save_image_config(self, silent=False):
        config = {
            "base_url": self.img_url_input.text().strip() or "https://api.whatai.cc/v1",
            "api_key": self.img_key_input.text().strip(),
            "model": self.img_model_combo.currentText().strip()
        }
        try:
            with open(CONFIG_IMAGE_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            # 只有在非静默模式下（用户手动点击保存按钮），才弹出成功提示
            if not silent:
                QMessageBox.information(self, "成功", f"生图配置已保存至 {CONFIG_IMAGE_FILE}")
        except Exception as e:
            # 静默模式下出错只打日志，不弹窗
            if not silent:
                QMessageBox.warning(self, "失败", f"保存配置文件失败: {e}")
            else:
                self.log_msg(f"⚠️ 自动保存生图配置失败: {e}")

    def fetch_models(self):
        api_key = self.key_input.text().strip()
        base_url = self.url_input.text().strip()
        
        if not api_key:
            QMessageBox.warning(self, "错误", "请先输入文本分析的 API Key")
            return
            
        self.fetch_btn.setEnabled(False)
        self.fetch_btn.setText("获取中...")
        QApplication.processEvents()

        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            models = client.models.list()
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
            QMessageBox.warning(self, "缺少配置", "文本分析 API Key 和 模型名称不能为空！")
            return
        
        # 重置按钮状态
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
        safe_title = re.sub(r'[\\/*?:"<>|]', "", jp_title).strip()
        if not safe_title:
            safe_title = "未命名"
        
        now = datetime.datetime.now()
        now_str = now.strftime("%Y%m%d-%H%M%S")
        date_str = now.strftime("%Y%m%d") # 获取当天日期，格式如 20260319
        save_dir = os.path.join('data', date_str) 
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir) # 自动创建 data 及其下属的当天日期目录
            
        base_filename = f"{now_str}-{safe_title}"
        json_filename = f"{base_filename}.json"
        
        try:
            with open( os.path.join(save_dir, json_filename), "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=4)
            self.log_msg(f"\n✅ 成功！JSON 结果已保存至: {json_filename}")
        except Exception as e:
            self.log_msg(f"\n❌ 保存 JSON 文件时出错: {e}")

        try:
            # 更新上下文变量供生图使用
            self.current_aspect_ratio = result_json.get("aspect_ratio", "2:3")
            self.current_refine_desc = result_json.get("english_description", "")
            self.current_orig_desc = result_json.get("original_english_description", "")
            selected_style_name = self.main_style_combo.currentText()
            current_fixed_tags = self.styles_data.get(selected_style_name, "")
            
            final_prompt = f"--ar {self.current_aspect_ratio} {current_fixed_tags} {self.current_refine_desc}".strip()
            orig_prompt = f"--ar {self.current_aspect_ratio} {current_fixed_tags} {self.current_orig_desc}".strip()
            
            txt_filename = f"{base_filename}-prompts.txt"
            orig_txt_filename = f"{base_filename}-original-prompts.txt"
            
            with open( os.path.join(save_dir, txt_filename), "w", encoding="utf-8") as f:
                f.write(final_prompt)
            
            with open( os.path.join(save_dir, orig_txt_filename), "w", encoding="utf-8") as f:
                f.write(orig_prompt)
                
            self.log_msg(f"✅ 成功！两份画幅与提示词文件已保存:\n - {txt_filename}\n - {orig_txt_filename}")
            
            # 分析完成，开放生图按钮
            self.gen_orig_btn.setEnabled(True)
            self.gen_ref_btn.setEnabled(True)
            self.log_msg("\n💡 提示: 现在可以点击下方的按钮，根据提取的描述直接生成图片了！")
            
        except Exception as e:
            self.log_msg(f"❌ 保存提示词 txt 文件时出错: {e}")

    def trigger_image_generation(self, prompt_type):
        # 必须先保存图片 API 配置到 json 文件中，让 api_backend 能够读取最新配置
        # 新增 silent=True 实现静默保存，不弹窗打扰
        self.save_image_config(silent=True)
        
        model_name = self.img_model_combo.currentText().strip()
        if not self.img_key_input.text().strip():
            QMessageBox.warning(self, "缺少配置", "生图 API Key 不能为空，请检查配置卡片。")
            return
            
        # 根据用户选择，决定使用原始描述还是优化描述
        prompt_to_use = self.current_orig_desc if prompt_type == "original" else self.current_refine_desc
        
        # 动态获取当前选中的画风
        selected_style_name = self.main_style_combo.currentText()
        active_instructions = self.styles_data.get(selected_style_name, "")
        
        # 禁用生图按钮防止重复点击
        self.gen_orig_btn.setEnabled(False)
        self.gen_ref_btn.setEnabled(False)
        
        # 启动生图线程
        self.img_thread = ImageGenWorkerThread(
            prompt=prompt_to_use,
            model_name=model_name,
            aspect_ratio=self.current_aspect_ratio,
            instructions=active_instructions
        )
        self.img_thread.log_signal.connect(self.log_msg)
        self.img_thread.finish_signal.connect(self.on_image_generation_finished)
        self.img_thread.start()

    def on_image_generation_finished(self, saved_files):
        self.gen_orig_btn.setEnabled(True)
        self.gen_ref_btn.setEnabled(True)
        
        if saved_files:
            self.log_msg(f"\n🎉 成功生成了 {len(saved_files)} 张图片！")
            for file_path in saved_files:
                self.log_msg(f" 📂 保存路径: {file_path}")
        else:
            self.log_msg("\n⚠️ 未能获取到图片，请检查上方日志，或查看日志文件夹（log）的记录。")

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec_())