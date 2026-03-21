import sys
import json
import os
import requests
import time
import base64
import re
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QGroupBox, QFormLayout, QLabel, QLineEdit,
                             QComboBox, QSpinBox, QDoubleSpinBox, QPushButton,
                             QTextEdit, QMessageBox, QInputDialog, QGridLayout)
from PyQt6.QtCore import QThread, pyqtSignal

CONFIG_FILE = "config-sd.json"
PROMPTS_DIR = "data/prompts"
OUTPUT_DIR = "outputs"

class WorkerThread(QThread):
    """后台工作线程，负责 LLM 请求和 SD 绘图"""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, config, theme_text, template_text):
        super().__init__()
        self.config = config
        self.theme_text = theme_text
        self.template_text = template_text
        self.is_running = True

    def run(self):
        try:
            generate_count = self.config.get("generate_count", 3)
            for i in range(generate_count):
                if not self.is_running:
                    self.log_signal.emit("任务已被用户中止。")
                    break
                    
                self.log_signal.emit(f"\n--- 开始执行第 {i+1}/{generate_count} 次任务 ---")
                
                # 1. 请求大模型获取变体 Prompts
                self.log_signal.emit("正在请求大模型基于主题和模板生成 SD 提示词...")
                llm_data = self.fetch_llm_prompt()
                if not llm_data and self.is_running:
                    self.log_signal.emit("大模型请求失败，跳过本次循环。")
                    time.sleep(2)
                    continue

                if not self.is_running: break

                self.log_signal.emit(f"LLM 生成成功! 计划尺寸: {llm_data.get('width')}x{llm_data.get('height')}")
                self.log_signal.emit(f"正向提示词: {llm_data.get('prompt')[:100]}...")

                # 2. 请求本地 SD WebUI 生成图片
                self.log_signal.emit("正在将参数发送至本地 Stable Diffusion...")
                self.generate_sd_image(llm_data)

                if self.is_running and i < generate_count - 1:
                    time.sleep(1) # 批次间短暂休息

            self.log_signal.emit("\n✅ 工作流执行完毕。")
        except Exception as e:
            self.log_signal.emit(f"发生未捕获的异常: {str(e)}")
        finally:
            self.finished_signal.emit()

    def fetch_llm_prompt(self):
        url = f"{self.config['base_url'].rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json"
        }
        
        system_prompt = """你是一个 Stable Diffusion 提示词编写专家。
请基于用户提供的【绘画主题】、【目标风格】和【基础模板】，发挥想象力进行扩写，添加丰富的细节（环境、光影、服装、动作等）。
【重要要求】：
1. 每次生成必须有随机性和差异化。
2. 必须返回一个合法的 JSON 对象，不要包含任何 Markdown 标记（如 ```json），直接输出 JSON 文本。
严格包含以下字段：
{
  "prompt": "正向提示词(英文，逗号分隔)",
  "negative_prompt": "反向提示词(英文，逗号分隔)",
  "width": 512或768或1024等64的倍数,
  "height": 512或768或1024等64的倍数
}"""

        user_content = f"绘画主题: {self.theme_text}\n目标风格: {self.config['last_used_style']}\n基础模板内容: {self.template_text}"

        payload = {
            "model": self.config['model'],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.8 
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            reply_text = response.json()['choices'][0]['message']['content'].strip()
            
            # 清理可能存在的 Markdown 代码块标记
            reply_text = re.sub(r'^```(json)?\s*', '', reply_text, flags=re.IGNORECASE)
            reply_text = re.sub(r'\s*```$', '', reply_text)
            
            return json.loads(reply_text)
        except Exception as e:
            self.log_signal.emit(f"LLM API 错误: {str(e)}")
            return None

    def generate_sd_image(self, llm_data):
        url = f"{self.config['sd_url'].rstrip('/')}/sdapi/v1/txt2img"
        
        payload = {
            "prompt": llm_data.get("prompt", ""),
            "negative_prompt": llm_data.get("negative_prompt", ""),
            "width": llm_data.get("width", 512),
            "height": llm_data.get("height", 512),
            "sampler_name": self.config["sampler"],
            "steps": self.config["steps"],
            "cfg_scale": self.config["cfg_scale"]
        }
        
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            images_base64 = response.json().get("images", [])
            
            # 保存图片
            for idx, img_b64 in enumerate(images_base64):
                img_data = base64.b64decode(img_b64)
                timestamp = int(time.time())
                filename = os.path.join(OUTPUT_DIR, f"gen_{timestamp}_{idx}.png")
                with open(filename, "wb") as img_file:
                    img_file.write(img_data)
                self.log_signal.emit(f"图片已保存至: {filename}")
        except Exception as e:
            self.log_signal.emit(f"SD WebUI API 错误: {str(e)}")

    def stop(self):
        self.is_running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 自动化绘画工作流 (PyQt6 增强版)")
        self.resize(900, 850)
        
        # 初始化目录
        os.makedirs(PROMPTS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        self.config = {
            "base_url": "[https://api.pumpkinaigc.online/v1](https://api.pumpkinaigc.online/v1)",
            "api_key": "",
            "model": "gpt-5",
            "last_used_style": "galgame CG style, rococo aesthetic, high quality",
            "sd_url": "[http://127.0.0.1:7860](http://127.0.0.1:7860)",
            "sampler": "Euler a",
            "steps": 20,
            "cfg_scale": 7.0,
            "generate_count": 3
        }
        self.worker = None
        
        self.load_config()
        self.init_ui()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self.config.update(loaded)
            except Exception as e:
                print(f"配置文件读取失败: {e}")

    def save_config(self):
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # === 1. 大模型配置 ===
        llm_group = QGroupBox("大模型 API 配置 (OpenAI 格式)")
        llm_layout = QGridLayout()
        
        llm_layout.addWidget(QLabel("Base URL:"), 0, 0)
        self.url_input = QLineEdit(self.config.get("base_url"))
        llm_layout.addWidget(self.url_input, 0, 1, 1, 3)
        
        llm_layout.addWidget(QLabel("API Key:"), 1, 0)
        self.key_input = QLineEdit(self.config.get("api_key"))
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        llm_layout.addWidget(self.key_input, 1, 1, 1, 3)
        
        llm_layout.addWidget(QLabel("Model:"), 2, 0)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True) # 允许手动输入
        self.model_combo.setCurrentText(self.config.get("model"))
        llm_layout.addWidget(self.model_combo, 2, 1, 1, 2)
        
        self.fetch_models_btn = QPushButton("获取/刷新模型列表")
        self.fetch_models_btn.clicked.connect(self.fetch_available_models)
        llm_layout.addWidget(self.fetch_models_btn, 2, 3)
        
        llm_group.setLayout(llm_layout)
        main_layout.addWidget(llm_group)

        # === 2. 任务与模板设置 ===
        task_group = QGroupBox("任务与模板设置")
        task_layout = QVBoxLayout()
        
        # 主题与风格设置
        theme_style_layout = QFormLayout()
        self.theme_input = QLineEdit("中秋主题少女")
        self.theme_input.setPlaceholderText("例如：中秋主题少女、赛博朋克城市...")
        theme_style_layout.addRow("绘画主题 (必填):", self.theme_input)
        
        self.style_input = QLineEdit(self.config.get("last_used_style"))
        theme_style_layout.addRow("Prompt 风格预设:", self.style_input)
        task_layout.addLayout(theme_style_layout)

        # 模板选择与编辑
        template_ctrl_layout = QHBoxLayout()
        template_ctrl_layout.addWidget(QLabel("选择模板 (.txt):"))
        
        self.template_combo = QComboBox()
        self.template_combo.setMinimumWidth(200)
        self.template_combo.currentTextChanged.connect(self.load_template_content)
        template_ctrl_layout.addWidget(self.template_combo)
        
        self.save_template_btn = QPushButton("保存当前模板")
        self.save_template_btn.clicked.connect(self.save_current_template)
        template_ctrl_layout.addWidget(self.save_template_btn)
        
        self.save_as_template_btn = QPushButton("模板另存为...")
        self.save_as_template_btn.clicked.connect(self.save_as_new_template)
        template_ctrl_layout.addWidget(self.save_as_template_btn)
        
        task_layout.addLayout(template_ctrl_layout)
        
        self.template_editor = QTextEdit()
        self.template_editor.setPlaceholderText("在这里编辑提示词基础模板的内容...")
        self.template_editor.setMaximumHeight(150)
        task_layout.addWidget(self.template_editor)
        
        task_group.setLayout(task_layout)
        main_layout.addWidget(task_group)
        self.refresh_templates()

        # === 3. SD 配置 ===
        sd_group = QGroupBox("Stable Diffusion WebUI 配置")
        sd_layout = QFormLayout()
        
        self.sd_url_input = QLineEdit(self.config.get("sd_url"))
        sd_layout.addRow("SD API URL:", self.sd_url_input)
        
        # 水平布局放采样器、步数、CFG 和 数量
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Sampler:"))
        self.sampler_input = QLineEdit(self.config.get("sampler"))
        self.sampler_input.setFixedWidth(100)
        param_layout.addWidget(self.sampler_input)
        
        param_layout.addWidget(QLabel("Steps:"))
        self.steps_input = QSpinBox()
        self.steps_input.setRange(1, 150)
        self.steps_input.setValue(self.config.get("steps", 20))
        param_layout.addWidget(self.steps_input)
        
        param_layout.addWidget(QLabel("CFG Scale:"))
        self.cfg_input = QDoubleSpinBox()
        self.cfg_input.setRange(1.0, 30.0)
        self.cfg_input.setSingleStep(0.5)
        self.cfg_input.setValue(self.config.get("cfg_scale", 7.0))
        param_layout.addWidget(self.cfg_input)

        param_layout.addWidget(QLabel("生成数量:"))
        self.count_input = QSpinBox()
        self.count_input.setRange(1, 9999)
        self.count_input.setValue(self.config.get("generate_count", 3))
        param_layout.addWidget(self.count_input)
        
        sd_layout.addRow(param_layout)
        sd_group.setLayout(sd_layout)
        main_layout.addWidget(sd_group)

        # === 4. 控制面板与日志 ===
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("保存配置并开始生成")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        self.start_btn.clicked.connect(self.start_workflow)
        
        self.stop_btn = QPushButton("停止任务")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("font-weight: bold; background-color: #f44336; color: white;")
        self.stop_btn.clicked.connect(self.stop_workflow)
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        main_layout.addLayout(btn_layout)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        main_layout.addWidget(self.log_area)

    def fetch_available_models(self):
        url = f"{self.url_input.text().strip().rstrip('/')}/models"
        api_key = self.key_input.text().strip()
        
        if not api_key:
            QMessageBox.warning(self, "警告", "请先输入 API Key！")
            return
            
        headers = {"Authorization": f"Bearer {api_key}"}
        self.fetch_models_btn.setEnabled(False)
        self.fetch_models_btn.setText("获取中...")
        QApplication.processEvents() # 强制刷新 UI 状态
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            
            if models:
                current_text = self.model_combo.currentText()
                self.model_combo.clear()
                self.model_combo.addItems(models)
                # 尝试恢复之前选中的模型
                if current_text in models:
                    self.model_combo.setCurrentText(current_text)
                else:
                    self.model_combo.setCurrentText(models[0])
                QMessageBox.information(self, "成功", f"成功获取 {len(models)} 个可用模型！")
            else:
                QMessageBox.warning(self, "提示", "API 返回成功，但模型列表为空。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"获取模型列表失败: {e}")
        finally:
            self.fetch_models_btn.setEnabled(True)
            self.fetch_models_btn.setText("获取/刷新模型列表")

    def refresh_templates(self):
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        templates = [f for f in os.listdir(PROMPTS_DIR) if f.endswith('.txt')]
        if templates:
            self.template_combo.addItems(templates)
            self.load_template_content(templates[0])
        else:
            self.template_combo.addItem("未找到 txt 文件")
            self.template_editor.clear()
        self.template_combo.blockSignals(False)

    def load_template_content(self, filename):
        if not filename or filename == "未找到 txt 文件":
            return
        filepath = os.path.join(PROMPTS_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.template_editor.setPlainText(f.read())

    def save_current_template(self):
        filename = self.template_combo.currentText()
        if not filename or filename == "未找到 txt 文件":
            QMessageBox.warning(self, "警告", "当前没有选中有效的模板文件。")
            return
            
        filepath = os.path.join(PROMPTS_DIR, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.template_editor.toPlainText())
            QMessageBox.information(self, "成功", f"模板 '{filename}' 已保存！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def save_as_new_template(self):
        new_name, ok = QInputDialog.getText(self, "模板另存为", "请输入新模板名称 (无需输入 .txt 后缀):")
        if ok and new_name.strip():
            filename = f"{new_name.strip()}.txt"
            filepath = os.path.join(PROMPTS_DIR, filename)
            
            if os.path.exists(filepath):
                reply = QMessageBox.question(self, "确认覆盖", f"文件 '{filename}' 已存在，是否覆盖？", 
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    return
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(self.template_editor.toPlainText())
                self.refresh_templates()
                self.template_combo.setCurrentText(filename)
                QMessageBox.information(self, "成功", f"新模板 '{filename}' 已保存！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def append_log(self, text):
        timestamp = time.strftime('%H:%M:%S')
        self.log_area.append(f"[{timestamp}] {text}")
        scrollbar = self.log_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_config_from_ui(self):
        self.config["base_url"] = self.url_input.text().strip()
        self.config["api_key"] = self.key_input.text().strip()
        self.config["model"] = self.model_combo.currentText().strip()
        self.config["last_used_style"] = self.style_input.text().strip()
        self.config["sd_url"] = self.sd_url_input.text().strip()
        self.config["sampler"] = self.sampler_input.text().strip()
        self.config["steps"] = self.steps_input.value()
        self.config["cfg_scale"] = self.cfg_input.value()
        self.config["generate_count"] = self.count_input.value()
        self.save_config()

    def start_workflow(self):
        theme = self.theme_input.text().strip()
        if not theme:
            QMessageBox.warning(self, "警告", "请填写绘画主题！")
            return
            
        template_text = self.template_editor.toPlainText().strip()
        if not template_text:
            QMessageBox.warning(self, "警告", "模板内容不能为空！")
            return

        self.update_config_from_ui()
        self.log_area.clear()
        
        # 切换按钮状态
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.fetch_models_btn.setEnabled(False)
        self.save_template_btn.setEnabled(False)
        self.save_as_template_btn.setEnabled(False)

        # 初始化并启动后台线程
        self.worker = WorkerThread(self.config, theme, template_text)
        self.worker.log_signal.connect(self.append_log)
        self.worker.finished_signal.connect(self.on_workflow_finished)
        self.worker.start()

    def stop_workflow(self):
        if self.worker and self.worker.isRunning():
            self.append_log("收到停止指令，正在等待当前网络请求完成并安全退出...")
            self.worker.stop()
            self.stop_btn.setEnabled(False)

    def on_workflow_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.fetch_models_btn.setEnabled(True)
        self.save_template_btn.setEnabled(True)
        self.save_as_template_btn.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())