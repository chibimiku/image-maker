import sys
import json
import os
import time
import traceback
import threading
from google import genai
from google.genai import types

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel
from PyQt5.QtCore import pyqtSignal, QObject, Qt

# 用于重定向控制台输出到 GUI 的类
class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass

class ImageGeneratorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.json_file_path = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Gemini 图片生成器')
        self.resize(600, 450)
        
        # 将整个主窗口设置为允许接收拖拽
        self.setAcceptDrops(True)

        layout = QVBoxLayout()

        # 1. 拖拽提示区域 (仅作为视觉展示)
        self.drop_area = QLabel("请将 JSON 文件拖拽到窗口任意位置", self)
        self.drop_area.setAlignment(Qt.AlignCenter)
        self.reset_drop_area_style()
        layout.addWidget(self.drop_area)

        # 2. 运行按钮
        self.run_btn = QPushButton("执行生成", self)
        self.run_btn.setEnabled(False) 
        self.run_btn.setStyleSheet("padding: 10px; font-size: 14px; font-weight: bold;")
        self.run_btn.clicked.connect(self.start_generation)
        layout.addWidget(self.run_btn)

        # 3. 日志输出文本框
        self.log_output = QTextEdit(self)
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas; font-size: 12px;")
        layout.addWidget(self.log_output)

        self.setLayout(layout)

        # 替换系统标准输出和标准错误输出
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        sys.stderr = EmittingStream(textWritten=self.normalOutputWritten)
        
        print("界面初始化完成，等待拖入 JSON 文件...")

    # --- 拖拽事件处理 (绑定在主窗口上) ---
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
                print(f"成功加载文件路径: {file_path}")
            else:
                print("格式错误：请拖拽 .json 格式的文件！")

    # --- 日志输出处理 ---
    def normalOutputWritten(self, text):
        cursor = self.log_output.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.log_output.setTextCursor(cursor)
        self.log_output.ensureCursorVisible()

    # --- 触发生成逻辑 ---
    def start_generation(self):
        if not self.json_file_path:
            print("未选择 JSON 文件。")
            return
        
        self.run_btn.setEnabled(False)
        self.drop_area.setText("正在生成中，请耐心等待...")
        print("-" * 40)
        print(f"开始处理: {os.path.basename(self.json_file_path)}")
        
        # 启动独立线程执行任务
        thread = threading.Thread(target=self.run_script, args=(self.json_file_path,))
        thread.daemon = True
        thread.start()

    def run_script(self, json_path):
        try:
            self.process_image(json_path)
        except Exception as e:
            print(f"运行发生异常: {e}")
            traceback.print_exc()
        finally:
            self.run_btn.setEnabled(True)
            self.drop_area.setText(f"已加载: {os.path.basename(self.json_file_path)}")
            print("-" * 40)

    # --- 原有的生图核心逻辑 ---
    def process_image(self, JSON_FILE_PATH):
        SAVE_DIR = "data"
        CONFIG_PATH = "config-image.json"

        # 1. 读取 API 配置文件
        if not os.path.exists(CONFIG_PATH):
            print(f"当前目录下找不到配置文件: {CONFIG_PATH}，请确保文件存在。")
            return
            
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as cf:
                config_data = json.load(cf)
                
            api_key = config_data.get("api_key")
            base_url = config_data.get("base_url")
            model_name = config_data.get("model", "gemini-3.1-flash-image-preview")
            
            if not api_key:
                print("config.json 中缺失 api_key 参数！")
                return
        except Exception as e:
            print(f"读取 config.json 失败: {e}")
            return

        # 2. 读取目标处理文件
        if not os.path.exists(JSON_FILE_PATH):
            print(f"找不到JSON文件: {JSON_FILE_PATH}")
            return

        base_path = os.path.splitext(JSON_FILE_PATH)[0]
        prompt_file_path = f"{base_path}-prompts.txt"

        if not os.path.exists(prompt_file_path):
            print(f"找不到Prompt文件: {prompt_file_path}")
            return

        with open(prompt_file_path, 'r', encoding='utf-8') as pf:
            prompt = pf.read().strip()

        if not prompt:
            print("Prompt文件内容为空，请检查文件内容。")
            return

        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        aspect_ratio = data.get("aspect_ratio", "2:3") 
        title = data.get("chinese_title", data.get("japanese_title", "untitled"))

        # 保留原有的代理设置
        os.environ["http_proxy"] = "http://127.0.0.1:7897"
        os.environ["https_proxy"] = "http://127.0.0.1:7897"

        print(f"成功读取提示词文件: {os.path.basename(prompt_file_path)}")
        print(f"正在使用模型 [{model_name}] 发送生成请求...")
        if base_url:
            print(f"已应用自定义 API 接口: {base_url}")
        
        try:
            # 3. 动态配置 Client 参数
            client_options = {"api_key": api_key}
            if base_url:
                client_options["http_options"] = {"base_url": base_url}
                
            client = genai.Client(**client_options)

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
            
            try:
                for part in result.parts:
                    if part.text is not None:
                        print(f"模型返回文本: {part.text}")
                    elif part.inline_data is not None:
                        image = part.as_image()
                        print(f"检测到图像，格式: {image.mime_type}")
                        
                        if str(image.mime_type) == 'image/png':
                            s_filename = filepath + ".png"
                        else:
                            s_filename = filepath + ".jpg"
                        
                        image.save(s_filename)
                        print(f"图片生成成功，已保存至: {s_filename}")
            except Exception as e:
                print("解析返回体时出错:")
                traceback.print_exc()

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