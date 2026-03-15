import sys
import json
import os
import time
import traceback
import threading
import shutil
from io import BytesIO

# 需要额外安装 Pillow: pip install Pillow
from PIL import Image

from google import genai
from google.genai import types

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QTextEdit, QLabel, QTabWidget, 
                             QScrollArea, QComboBox, QLineEdit, QCheckBox, 
                             QFileDialog, QSplitter, QInputDialog, QMessageBox)
from PyQt5.QtCore import pyqtSignal, QObject, Qt, QSize
from PyQt5.QtGui import QPixmap, QImage

# ----------------- 基础工具类 -----------------

class EmittingStream(QObject):
    """用于重定向控制台输出到 GUI 的类"""
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass

def compress_image_if_needed(image_path, max_dim=2048, max_size_mb=8):
    """如果图片超过尺寸或文件大小，进行等比例压缩并返回PIL Image对象，否则直接返回PIL Image"""
    img = Image.open(image_path)
    
    # 尺寸压缩
    w, h = img.size
    if w > max_dim or h > max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    
    # 转换并检查文件大小
    if img.mode != 'RGB' and img.mode != 'RGBA':
        img = img.convert('RGB')
        
    buffer = BytesIO()
    # 优先使用 PNG 格式来评估大小
    img.save(buffer, format="PNG")
    size_mb = buffer.tell() / (1024 * 1024)
    
    if size_mb > max_size_mb:
        # 如果PNG太大了，尝试用JPEG压缩以满足大小限制
        buffer = BytesIO()
        img.convert('RGB').save(buffer, format="JPEG", quality=85)
        img = Image.open(buffer)
        
    return img

# ----------------- UI 组件类 -----------------

class DropImageLabel(QLabel):
    """支持拖拽和点击选择图片的 Label"""
    imageChanged = pyqtSignal(str) # 发送图片路径

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px dashed #aaa; background-color: #2d2d2d; color: #aaa;")
        self.setMinimumSize(120, 120)
        self.setMaximumSize(120, 120)
        self.setAcceptDrops(True)
        self.setScaledContents(True)
        self.current_image_path = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("border: 2px dashed #4CAF50; background-color: #e8f5e9; color: #4CAF50;")
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._reset_style()

    def dropEvent(self, event):
        self._reset_style()
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                self.set_image(file_path)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            file_path, _ = QFileDialog.getOpenFileName(self, "选择参考图片", "", "Images (*.png *.jpg *.jpeg *.webp)")
            if file_path:
                self.set_image(file_path)

    def set_image(self, file_path):
        self.current_image_path = file_path
        pixmap = QPixmap(file_path)
        self.setPixmap(pixmap)
        self.imageChanged.emit(file_path)

    def clear_image(self):
        self.current_image_path = None
        self.clear()
        self.setText("点击或拖拽\n添加图片")
        self._reset_style()

    def _reset_style(self):
        if self.current_image_path:
            self.setStyleSheet("border: 2px solid #555;")
        else:
            self.setStyleSheet("border: 2px dashed #aaa; background-color: #2d2d2d; color: #aaa;")

class ImageSlotWidget(QWidget):
    """代表一个分类的参考图槽位 (例如: 衣服1)"""
    def __init__(self, category_name):
        super().__init__()
        self.category_name = category_name
        self.history_dir = os.path.join("data", "history", category_name)
        os.makedirs(self.history_dir, exist_ok=True)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 标题栏: 名称 + 历史按钮
        title_layout = QHBoxLayout()
        title_label = QLabel(f"<b>{category_name}</b>")
        history_btn = QPushButton("历史")
        history_btn.setFixedSize(40, 20)
        history_btn.clicked.connect(self.select_from_history)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(history_btn)
        layout.addLayout(title_layout)

        # 拖拽图片区
        self.image_label = DropImageLabel(f"点击/拖拽\n添加 {category_name}")
        self.image_label.imageChanged.connect(self.on_image_changed)
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # 描述输入框
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText(f"补充描述(可选)")
        layout.addWidget(self.prompt_input)

        # 保存复选框
        self.save_cb = QCheckBox("保存到历史")
        self.save_cb.setChecked(True)
        layout.addWidget(self.save_cb)
        
        # 清除按钮
        clear_btn = QPushButton("清除")
        clear_btn.clicked.connect(self.clear)
        layout.addWidget(clear_btn)

        self.setLayout(layout)
        self.setFixedWidth(150)
        self.setStyleSheet("QWidget { background-color: #f0f0f0; border-radius: 5px; }")

    def on_image_changed(self, file_path):
        pass # 可在此添加图片加载后的额外逻辑

    def select_from_history(self):
        file_path, _ = QFileDialog.getOpenFileName(self, f"从历史选择 {self.category_name}", self.history_dir, "Images (*.png *.jpg *.jpeg *.webp)")
        if file_path:
            self.image_label.set_image(file_path)
            self.save_cb.setChecked(False) # 选择已有的默认不再重复保存

    def clear(self):
        self.image_label.clear_image()
        self.prompt_input.clear()
        self.save_cb.setChecked(True)

    def get_data(self):
        path = self.image_label.current_image_path
        if not path:
            return None
            
        # 检查是否需要保存副本到历史
        if self.save_cb.isChecked():
            filename = os.path.basename(path)
            dest = os.path.join(self.history_dir, filename)
            if path != dest:
                try:
                    shutil.copy2(path, dest)
                except Exception as e:
                    print(f"保存 {self.category_name} 到历史失败: {e}")
            
        return {
            "category": self.category_name,
            "filename": os.path.basename(path),
            "filepath": path,
            "prompt": self.prompt_input.text()
        }

# ----------------- 标签页 1: 原始批量生成 -----------------
class JsonGeneratorTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.json_file_path = None
        self.initUI()

    def initUI(self):
        self.setAcceptDrops(True)
        layout = QVBoxLayout()

        self.drop_area = QLabel("请将 JSON 文件拖拽到此区域", self)
        self.drop_area.setAlignment(Qt.AlignCenter)
        self.reset_drop_area_style()
        layout.addWidget(self.drop_area)

        self.run_btn = QPushButton("执行生成", self)
        self.run_btn.setEnabled(False) 
        self.run_btn.setStyleSheet("padding: 10px; font-size: 14px; font-weight: bold;")
        self.run_btn.clicked.connect(self.start_generation)
        layout.addWidget(self.run_btn)

        self.setLayout(layout)

    def reset_drop_area_style(self):
        self.drop_area.setStyleSheet("border: 2px dashed #aaa; padding: 30px; font-size: 14px; color: #aaa; background-color: transparent;")

    def highlight_drop_area_style(self):
        self.drop_area.setStyleSheet("border: 2px dashed #4CAF50; padding: 30px; font-size: 14px; color: #4CAF50; background-color: #e8f5e9;")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.highlight_drop_area_style()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.reset_drop_area_style()

    def dropEvent(self, event):
        self.reset_drop_area_style()
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith('.json'):
                self.json_file_path = file_path
                self.drop_area.setText(f"已加载: {os.path.basename(file_path)}")
                self.run_btn.setEnabled(True)
                print(f"[基础生成] 成功加载文件: {file_path}")
            else:
                print("[基础生成] 格式错误：请拖拽 .json 格式的文件！")

    def start_generation(self):
        if not self.json_file_path: return
        self.run_btn.setEnabled(False)
        self.drop_area.setText("正在生成中，请耐心等待...")
        print("-" * 40)
        thread = threading.Thread(target=self.run_script, args=(self.json_file_path,))
        thread.daemon = True
        thread.start()

    def run_script(self, json_path):
        try:
            self.main_window.process_image_basic(json_path)
        except Exception as e:
            print(f"运行发生异常: {e}")
            traceback.print_exc()
        finally:
            self.run_btn.setEnabled(True)
            self.drop_area.setText(f"已加载: {os.path.basename(self.json_file_path)}")
            print("-" * 40)


# ----------------- 标签页 2: 赛博暖暖 -----------------
class CyberNikiTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.slots = []
        self.instructions_file = os.path.join("data", "history", "instructions.json")
        self.initUI()
        self.load_instructions_history()

    def initUI(self):
        main_layout = QVBoxLayout()
        
        # 1. 顶部：预览区与插槽
        top_layout = QHBoxLayout()
        
        # 左侧滚动区 (角色, 动作, 衣服1, 衣服2, 衣服3)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_content = QWidget()
        left_vbox = QVBoxLayout(left_content)
        left_cats = ["角色", "动作", "衣服1", "衣服2", "衣服3"]
        for cat in left_cats:
            slot = ImageSlotWidget(cat)
            self.slots.append(slot)
            left_vbox.addWidget(slot)
        left_vbox.addStretch()
        left_scroll.setWidget(left_content)
        left_scroll.setFixedWidth(180)
        top_layout.addWidget(left_scroll)

        # 中间预览区
        center_layout = QVBoxLayout()
        self.preview_label = QLabel("产出结果预览")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("border: 1px solid #ccc; background-color: #fff;")
        self.preview_label.setMinimumSize(300, 400)
        self.preview_label.setScaledContents(True)
        center_layout.addWidget(self.preview_label, 1)
        
        self.generate_btn = QPushButton("✨ 开始生成 (赛博暖暖) ✨")
        self.generate_btn.setStyleSheet("padding: 15px; font-size: 16px; font-weight: bold; background-color: #4CAF50; color: white; border-radius: 8px;")
        self.generate_btn.clicked.connect(self.start_generation)
        center_layout.addWidget(self.generate_btn)
        top_layout.addLayout(center_layout, 1)

        # 右侧滚动区 (背景, 鞋子, 袜子, 手套, 发饰, 手持物)
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_content = QWidget()
        right_vbox = QVBoxLayout(right_content)
        right_cats = ["背景", "鞋子", "袜子", "手套", "发饰", "手持物"]
        for cat in right_cats:
            slot = ImageSlotWidget(cat)
            self.slots.append(slot)
            right_vbox.addWidget(slot)
        right_vbox.addStretch()
        right_scroll.setWidget(right_content)
        right_scroll.setFixedWidth(180)
        top_layout.addWidget(right_scroll)

        main_layout.addLayout(top_layout, 2)

        # 2. 中间：配置区
        config_group = QWidget()
        config_layout = QVBoxLayout(config_group)
        config_layout.setContentsMargins(0, 10, 0, 10)
        
        # Instructions 行
        inst_layout = QHBoxLayout()
        inst_layout.addWidget(QLabel("Instructions (基础风格提示):"))
        self.inst_combo = QComboBox()
        self.inst_combo.setEditable(True)
        self.inst_combo.setSizePolicy(self.inst_combo.sizePolicy().Expanding, self.inst_combo.sizePolicy().Fixed)
        inst_layout.addWidget(self.inst_combo, 1)
        
        save_inst_btn = QPushButton("保存指令组合")
        save_inst_btn.clicked.connect(self.save_current_instruction)
        inst_layout.addWidget(save_inst_btn)
        config_layout.addLayout(inst_layout)

        # 额外 Prompts
        config_layout.addWidget(QLabel("额外补充 Prompts:"))
        self.extra_prompt_input = QTextEdit()
        self.extra_prompt_input.setMaximumHeight(60)
        self.extra_prompt_input.setPlaceholderText("在这里输入额外的画面要求、氛围描述等...")
        config_layout.addWidget(self.extra_prompt_input)

        # 比例与配置设置
        bottom_config_layout = QHBoxLayout()
        bottom_config_layout.addWidget(QLabel("输出比例:"))
        self.ratio_combo = QComboBox()
        self.ratio_combo.addItems(["2:3", "3:4", "1:1", "16:9", "9:16"])
        bottom_config_layout.addWidget(self.ratio_combo)
        bottom_config_layout.addStretch()
        
        edit_config_btn = QPushButton("编辑 API 配置")
        edit_config_btn.clicked.connect(self.main_window.edit_api_config)
        bottom_config_layout.addWidget(edit_config_btn)
        
        config_layout.addLayout(bottom_config_layout)

        main_layout.addWidget(config_group, 0)
        self.setLayout(main_layout)

    def load_instructions_history(self):
        os.makedirs(os.path.dirname(self.instructions_file), exist_ok=True)
        if os.path.exists(self.instructions_file):
            try:
                with open(self.instructions_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    self.inst_combo.addItems(history)
            except Exception as e:
                print(f"加载指令历史失败: {e}")

    def save_current_instruction(self):
        current_text = self.inst_combo.currentText().strip()
        if not current_text: return
        
        items = [self.inst_combo.itemText(i) for i in range(self.inst_combo.count())]
        if current_text not in items:
            self.inst_combo.addItem(current_text)
            items.append(current_text)
            
            try:
                with open(self.instructions_file, 'w', encoding='utf-8') as f:
                    json.dump(items, f, ensure_ascii=False, indent=2)
                print("指令已保存到历史。")
            except Exception as e:
                print(f"保存指令失败: {e}")

    def update_preview_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.preview_label.setPixmap(pixmap)

    def start_generation(self):
        # 收集数据
        active_items = []
        for slot in self.slots:
            data = slot.get_data()
            if data:
                active_items.append(data)
                
        if not active_items and not self.inst_combo.currentText().strip() and not self.extra_prompt_input.toPlainText().strip():
            print("请至少添加一张参考图或输入提示词！")
            return

        instruction = self.inst_combo.currentText()
        extra_prompt = self.extra_prompt_input.toPlainText()
        aspect_ratio = self.ratio_combo.currentText()

        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("⏳ 生成中...")
        print("=" * 40)
        print("开始[赛博暖暖]拼装生成任务...")

        thread = threading.Thread(target=self.run_generation_task, args=(active_items, instruction, extra_prompt, aspect_ratio))
        thread.daemon = True
        thread.start()

    def run_generation_task(self, active_items, instruction, extra_prompt, aspect_ratio):
        try:
            output_path = self.main_window.process_image_cyber_niki(active_items, instruction, extra_prompt, aspect_ratio)
            if output_path:
                self.update_preview_image(output_path)
        except Exception as e:
            print(f"生成发生异常: {e}")
            traceback.print_exc()
        finally:
            self.generate_btn.setEnabled(True)
            self.generate_btn.setText("✨ 开始生成 (赛博暖暖) ✨")
            print("=" * 40)


# ----------------- 主程序窗口 -----------------
class ImageGeneratorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.CONFIG_PATH = "config-image.json"
        self.initUI()
        self.ensure_config_exists()

    def initUI(self):
        self.setWindowTitle('Gemini 图片生成控制台')
        self.resize(1000, 800)
        
        main_layout = QVBoxLayout()
        
        # 顶部标签页
        self.tabs = QTabWidget()
        self.json_tab = JsonGeneratorTab(self)
        self.cyber_tab = CyberNikiTab(self)
        
        self.tabs.addTab(self.cyber_tab, "👗 赛博暖暖 (可视化拼图)")
        self.tabs.addTab(self.json_tab, "📄 JSON 批量生成")
        main_layout.addWidget(self.tabs, 3)

        # 底部日志区
        log_layout = QVBoxLayout()
        log_layout.addWidget(QLabel("执行日志:"))
        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas; font-size: 12px;")
        log_layout.addWidget(self.log_output)
        
        main_layout.addLayout(log_layout, 1)
        self.setLayout(main_layout)

        # 替换系统标准输出
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        sys.stderr = EmittingStream(textWritten=self.normalOutputWritten)
        
        print("界面初始化完成，欢迎使用。")

    def normalOutputWritten(self, text):
        cursor = self.log_output.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.log_output.setTextCursor(cursor)
        self.log_output.ensureCursorVisible()

    def ensure_config_exists(self):
        if not os.path.exists(self.CONFIG_PATH):
            default_config = {
                "api_key": "",
                "base_url": "",
                "model": "gemini-3.1-flash-image" # 自动更新为当前推荐模型
            }
            with open(self.CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4)

    def edit_api_config(self):
        try:
            with open(self.CONFIG_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            content = "{}"
            
        text, ok = QInputDialog.getMultiLineText(self, '编辑 config-image.json', '请编辑 JSON 格式的配置:', content)
        if ok and text:
            try:
                # 校验 JSON 格式
                json.loads(text)
                with open(self.CONFIG_PATH, 'w', encoding='utf-8') as f:
                    f.write(text)
                QMessageBox.information(self, "成功", "配置已更新。")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无效的 JSON 格式:\n{e}")

    def get_genai_client(self):
        try:
            with open(self.CONFIG_PATH, 'r', encoding='utf-8') as cf:
                config_data = json.load(cf)
                
            api_key = config_data.get("api_key")
            base_url = config_data.get("base_url")
            model_name = config_data.get("model", "gemini-3.1-flash-image")
            
            if not api_key:
                print("错误: config-image.json 中缺失 api_key 参数！请点击[编辑 API 配置]添加。")
                return None, None

            client_options = {"api_key": api_key}
            if base_url:
                client_options["http_options"] = {"base_url": base_url}
                
            client = genai.Client(**client_options)
            return client, model_name
        except Exception as e:
            print(f"读取配置或初始化客户端失败: {e}")
            return None, None

    # --- 赛博暖暖核心生成逻辑 ---
    def process_image_cyber_niki(self, active_items, instruction, extra_prompt, aspect_ratio):
        SAVE_DIR = os.path.join("output", "ootd")
        os.makedirs(SAVE_DIR, exist_ok=True)

        client, model_name = self.get_genai_client()
        if not client: return None

        # 代理设置
        os.environ["http_proxy"] = "http://127.0.0.1:7897"
        os.environ["https_proxy"] = "http://127.0.0.1:7897"

        # 构建 Prompt 内容数组
        contents = []
        
        # 组装基础文本 Prompt
        text_prompt = "【生成任务约束】\n"
        text_prompt += "1. 严格参考提供的图片元素。如果描述语言和图片内容发生冲突（例如图片是黑丝而描述是白丝），一律以参考图片内容为准。\n"
        text_prompt += "2. 请完美提取参考图片中的相应特征（如衣服款式、鞋子细节等），并将它们整合到一个人物/场景中。\n"
        text_prompt += "3. 图片整体生成风格必须是非照片的艺术风格，具体请参考下方的 Instructions。如果提供的参考图是真实照片，必须进行艺术化风格转换。\n"
        text_prompt += "4. 画面要求纯净无水印。\n\n"
        
        text_prompt += "【参考元素清单】\n"
        for item in active_items:
            text_prompt += f"- 分类: {item['category']}, 文件名: {item['filename']}, 附加描述: {item['prompt']}\n"
        
        if instruction.strip():
            text_prompt += f"\n【风格 Instructions】\n{instruction}\n"
        if extra_prompt.strip():
            text_prompt += f"\n【用户补充 Prompts】\n{extra_prompt}\n"

        contents.append(text_prompt)
        print("--- 组装的请求 Prompt ---")
        print(text_prompt)

        # 压缩并加入图片实体
        for item in active_items:
            try:
                img_pil = compress_image_if_needed(item['filepath'])
                contents.append(f"[{item['category']} 参考图: {item['filename']}]:")
                contents.append(img_pil)
                print(f"已加载图片: {item['filename']} (原图路径: {item['filepath']})")
            except Exception as e:
                print(f"处理图片 {item['filename']} 失败: {e}")

        print(f"请求模型: {model_name}, 比例: {aspect_ratio}")
        
        try:
            # 使用 ImageConfig，并指定 output_mime_type 为 image/png
            result = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    image_config=types.ImageConfig(
                        image_size='2K',
                        aspect_ratio=aspect_ratio,
                        output_mime_type="image/png"
                    )
                ),
            )

            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"Niki-{timestamp}"
            filepath_base = os.path.join(SAVE_DIR, filename)
            
            output_file = None
            try:
                for part in result.parts:
                    if part.text is not None:
                        print(f"模型返回文本说明: {part.text}")
                    elif part.inline_data is not None:
                        image = part.as_image()
                        
                        if str(image.mime_type) == 'image/png':
                            output_file = filepath_base + ".png"
                        else:
                            output_file = filepath_base + ".jpg"
                        
                        image.save(output_file)
                        print(f"✅ 图片生成成功，已保存至: {output_file}")
                return output_file
            except Exception as e:
                print("解析返回体时出错:")
                traceback.print_exc()

        except Exception as e:
            print("调用模型时出错:")
            traceback.print_exc()
        
        return None

    # --- 基础版的生成逻辑保持兼容 ---
    def process_image_basic(self, JSON_FILE_PATH):
        SAVE_DIR = "data"
        client, model_name = self.get_genai_client()
        if not client: return

        base_path = os.path.splitext(JSON_FILE_PATH)[0]
        prompt_file_path = f"{base_path}-prompts.txt"

        if not os.path.exists(prompt_file_path):
            print(f"找不到Prompt文件: {prompt_file_path}")
            return

        with open(prompt_file_path, 'r', encoding='utf-8') as pf:
            prompt = pf.read().strip()

        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        aspect_ratio = data.get("aspect_ratio", "2:3") 
        title = data.get("chinese_title", data.get("japanese_title", "untitled"))

        os.environ["http_proxy"] = "http://127.0.0.1:7897"
        os.environ["https_proxy"] = "http://127.0.0.1:7897"

        print(f"正在使用模型 [{model_name}] 发送生成请求...")
        try:
            result = client.models.generate_content(
                model=model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    image_config=types.ImageConfig(
                        image_size='2K',
                        aspect_ratio=aspect_ratio,
                    )
                ),
            )

            os.makedirs(SAVE_DIR, exist_ok=True)
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_', '-')).strip()
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{safe_title}-{timestamp}"
            filepath = os.path.join(SAVE_DIR, filename)
            
            for part in result.parts:
                if part.text is not None:
                    print(f"模型返回文本: {part.text}")
                elif part.inline_data is not None:
                    image = part.as_image()
                    s_filename = filepath + (".png" if str(image.mime_type) == 'image/png' else ".jpg")
                    image.save(s_filename)
                    print(f"✅ 图片生成成功，已保存至: {s_filename}")

        except Exception as e:
            print("调用模型时出错:")
            traceback.print_exc()

if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    gui = ImageGeneratorGUI()
    gui.show()
    sys.exit(app.exec_())