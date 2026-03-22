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
CACHE_DIR = "cache/sd-req"  # <--- 新增：提示词请求缓存目录

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
                self.log_signal.emit(f"原始正向提示词: {llm_data.get('prompt')[:80]}...")

                # ========== 新增：保存 LLM 返回的 prompts 到缓存目录 ==========
                try:
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    cache_filename = os.path.join(CACHE_DIR, f"prompt_{timestamp_str}_{i}.json")
                    with open(cache_filename, "w", encoding="utf-8") as f:
                        json.dump(llm_data, f, ensure_ascii=False, indent=4)
                    self.log_signal.emit(f"提示词已缓存至: {cache_filename}")
                except Exception as e:
                    self.log_signal.emit(f"提示词缓存失败: {str(e)}")

                # 2. 请求本地 SD WebUI 生成图片
                self.log_signal.emit("正在拼装固定提示词并将参数发送至本地 Stable Diffusion...")
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
        
        # 获取当前配置组
        current_group_name = self.config.get("current_sd_group", "Default")
        sd_settings = self.config.get("sd_config_groups", {}).get(current_group_name, {})
        
        # 提取 LLM 生成的提示词和固定提示词
        llm_prompt = llm_data.get("prompt", "").strip()
        llm_neg_prompt = llm_data.get("negative_prompt", "").strip()
        
        fixed_prompt = self.config.get("fixed_prompt", "").strip()
        fixed_neg_prompt = self.config.get("fixed_negative_prompt", "").strip()
        
        # 拼接提示词 (LLM提示词在前，固定词在后，安全处理逗号)
        final_prompt = f"{llm_prompt}, {fixed_prompt}" if fixed_prompt else llm_prompt
        final_neg_prompt = f"{llm_neg_prompt}, {fixed_neg_prompt}" if fixed_neg_prompt else llm_neg_prompt
        
        final_prompt = final_prompt.strip(", ")
        final_neg_prompt = final_neg_prompt.strip(", ")

        # 构建基础 payload
        payload = {
            "prompt": final_prompt,
            "negative_prompt": final_neg_prompt,
            "width": llm_data.get("width", 512),
            "height": llm_data.get("height", 512),
            "sampler_name": sd_settings.get("sampler", "Euler a"),
            "scheduler": sd_settings.get("scheduler", "Automatic"),
            "steps": sd_settings.get("steps", 20),
            "cfg_scale": sd_settings.get("cfg_scale", 7.0),
            "override_settings": {}
        }
        
        # 处理模型和 VAE 覆写参数
        sd_model = sd_settings.get("sd_model", "").strip()
        sd_vae_list = sd_settings.get("sd_vae", [])
        
        if sd_model:
            payload["override_settings"]["sd_model_checkpoint"] = sd_model
            
        # 1. 严格清洗数据：去除首尾空格，过滤掉空字符串和 "automatic"
        final_modules = [v.strip() for v in sd_vae_list if v.strip() and v.strip().lower() != "automatic"]
        
        # 2. 只有在用户确实输入了有效模型时才处理
        if final_modules:
            # 给 Forge 新版 API 传数组
            payload["override_settings"]["forge_additional_modules"] = final_modules
            
            # 【容错处理】安全地取第一个元素赋给老版 sd_vae
            # 虽然外层 if 已经保证了列表不为空，但显式地检查长度是好习惯
            # c传入sd_vae 对 neo 会有问题 禁用
            #if len(final_modules) > 0:
            #    payload["override_settings"]["sd_vae"] = final_modules[0]
        else:
            # 【容错处理】如果用户什么都没填，或者填的都是无效字符
            # 可选：显式地将 VAE 设置为 Automatic，或者直接从 override_settings 里删掉这俩 key 避免干扰
            payload["override_settings"]["sd_vae"] = "Automatic"
            # 确保不会传个空数组过去导致后端报错
            payload["override_settings"].pop("forge_additional_modules", None)
        # 智能合并附加字段 (JSON)
        extra_payload_str = self.config.get("webui_extra_payload", "").strip()
        if extra_payload_str:
            try:
                extra_payload = json.loads(extra_payload_str)
                if isinstance(extra_payload, dict):
                    for k, v in extra_payload.items():
                        if k == "override_settings" and isinstance(v, dict):
                            payload["override_settings"].update(v)
                        else:
                            payload[k] = v
                self.log_signal.emit("成功载入自定义 WebUI 附加 JSON 字段。")
            except Exception as e:
                self.log_signal.emit(f"警告：WebUI 附加字段 JSON 解析失败，已忽略 ({e})")

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
        self.resize(950, 980)
        
        # 初始化目录
        os.makedirs(PROMPTS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)

        
        # 默认配置加入配置组支持、固定提示词和附加字段
        self.config = {
            "base_url": "[https://api.pumpkinaigc.online/v1](https://api.pumpkinaigc.online/v1)",
            "api_key": "",
            "model": "gpt-5",
            "last_used_style": "galgame CG style, rococo aesthetic, high quality",
            "sd_url": "[http://127.0.0.1:7860](http://127.0.0.1:7860)",
            "generate_count": 3,
            "current_sd_group": "Default",
            "fixed_prompt": "(masterpiece, best quality:1.2), ultra-detailed, highres",
            "fixed_negative_prompt": "(worst quality, low quality:1.4), bad anatomy, deformed, signature, watermark",
            "webui_extra_payload": "{\n  \n}",
            "sd_config_groups": {
                "Default": {
                    "sd_model": "",
                    "sd_vae": "Automatic",
                    "sampler": "Euler a",
                    "scheduler": "Automatic",
                    "steps": 20,
                    "cfg_scale": 7.0
                }
            }
        }
        self.worker = None
        
        self.load_config()
        # 确保基础配置结构存在
        if "sd_config_groups" not in self.config:
            self.config["sd_config_groups"] = {"Default": {"sampler": "Euler a", "steps": 20, "cfg_scale": 7.0}}
        if "current_sd_group" not in self.config:
            self.config["current_sd_group"] = "Default"
            
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
        llm_group = QGroupBox("大模型 API 配置")
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
        self.model_combo.setEditable(True)
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
        
        theme_style_layout = QFormLayout()
        self.theme_input = QLineEdit("中秋主题少女")
        self.theme_input.setPlaceholderText("例如：中秋主题少女、赛博朋克城市...")
        theme_style_layout.addRow("绘画主题 (必填):", self.theme_input)
        
        self.style_input = QLineEdit(self.config.get("last_used_style"))
        theme_style_layout.addRow("Prompt 风格预设:", self.style_input)
        task_layout.addLayout(theme_style_layout)

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
        self.template_editor.setMaximumHeight(80)
        task_layout.addWidget(self.template_editor)
        
        task_group.setLayout(task_layout)
        main_layout.addWidget(task_group)
        self.refresh_templates()

        # === 3. SD WebUI 高级配置组 ===
        sd_group = QGroupBox("Stable Diffusion WebUI 配置")
        sd_layout = QVBoxLayout()
        
        # 组管理与 URL
        top_sd_layout = QHBoxLayout()
        top_sd_layout.addWidget(QLabel("SD API URL:"))
        self.sd_url_input = QLineEdit(self.config.get("sd_url"))
        top_sd_layout.addWidget(self.sd_url_input)
        
        top_sd_layout.addWidget(QLabel("  配置组:"))
        self.sd_group_combo = QComboBox()
        self.sd_group_combo.setMinimumWidth(120)
        self.sd_group_combo.currentTextChanged.connect(self.on_sd_group_changed)
        top_sd_layout.addWidget(self.sd_group_combo)
        
        self.save_sd_group_btn = QPushButton("保存为新配置组")
        self.save_sd_group_btn.clicked.connect(self.save_as_sd_group)
        top_sd_layout.addWidget(self.save_sd_group_btn)
        
        self.del_sd_group_btn = QPushButton("删除当前组")
        self.del_sd_group_btn.clicked.connect(self.delete_sd_group)
        top_sd_layout.addWidget(self.del_sd_group_btn)
        sd_layout.addLayout(top_sd_layout)

        # 模型与 VAE
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("大模型 (Checkpoint):"))
        self.sd_model_input = QLineEdit()
        self.sd_model_input.setPlaceholderText("留空则使用 WebUI 当前模型")
        model_layout.addWidget(self.sd_model_input)
        sd_layout.addLayout(model_layout)

        # 动态 VAE 列表容器
        vae_main_layout = QVBoxLayout()
        vae_header_layout = QHBoxLayout()
        vae_header_layout.addWidget(QLabel("VAE (支持多个拼接):"))
        self.add_vae_btn = QPushButton("+ 添加 VAE")
        self.add_vae_btn.setFixedWidth(100)
        self.add_vae_btn.clicked.connect(lambda: self.add_vae_input_field(""))
        vae_header_layout.addWidget(self.add_vae_btn)
        vae_header_layout.addStretch()
        vae_main_layout.addLayout(vae_header_layout)
        
        self.vae_inputs_container = QVBoxLayout()
        self.vae_inputs_list = [] # 用于存储所有的 QLineEdit 对象
        vae_main_layout.addLayout(self.vae_inputs_container)
        sd_layout.addLayout(vae_main_layout)

        # 采样参数
        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Sampler:"))
        self.sampler_input = QLineEdit()
        self.sampler_input.setFixedWidth(100)
        param_layout.addWidget(self.sampler_input)
        
        param_layout.addWidget(QLabel("Scheduler:"))
        self.scheduler_input = QLineEdit()
        self.scheduler_input.setPlaceholderText("Automatic")
        self.scheduler_input.setFixedWidth(100)
        param_layout.addWidget(self.scheduler_input)
        
        param_layout.addWidget(QLabel("Steps:"))
        self.steps_input = QSpinBox()
        self.steps_input.setRange(1, 150)
        param_layout.addWidget(self.steps_input)
        
        param_layout.addWidget(QLabel("CFG:"))
        self.cfg_input = QDoubleSpinBox()
        self.cfg_input.setRange(1.0, 30.0)
        self.cfg_input.setSingleStep(0.5)
        param_layout.addWidget(self.cfg_input)

        param_layout.addWidget(QLabel("生成数量:"))
        self.count_input = QSpinBox()
        self.count_input.setRange(1, 9999)
        self.count_input.setValue(self.config.get("generate_count", 3))
        param_layout.addWidget(self.count_input)
        sd_layout.addLayout(param_layout)

        # 固定提示词区域
        fixed_prompt_layout = QFormLayout()
        
        self.fixed_prompt_input = QLineEdit(self.config.get("fixed_prompt", ""))
        self.fixed_prompt_input.setPlaceholderText("如: (masterpiece, best quality:1.2), highres (自动拼接到大模型结果后)")
        fixed_prompt_layout.addRow("附加固定正向提示词:", self.fixed_prompt_input)
        
        self.fixed_neg_prompt_input = QLineEdit(self.config.get("fixed_negative_prompt", ""))
        self.fixed_neg_prompt_input.setPlaceholderText("如: (worst quality, low quality:1.4), bad anatomy")
        fixed_prompt_layout.addRow("附加固定反向提示词:", self.fixed_neg_prompt_input)
        
        sd_layout.addLayout(fixed_prompt_layout)

        # 附加字段
        extra_payload_layout = QVBoxLayout()
        extra_payload_layout.addWidget(QLabel("WebUI 附加 Payload (JSON格式，会被合并到 API 请求中):"))
        self.extra_payload_input = QTextEdit()
        self.extra_payload_input.setPlaceholderText('例如：\n{\n  "clip_skip": 2,\n  "enable_hr": true,\n  "hr_scale": 2.0,\n  "override_settings": {"text_encoder_...": "..."}\n}')
        self.extra_payload_input.setMaximumHeight(70)
        self.extra_payload_input.setPlainText(self.config.get("webui_extra_payload", ""))
        extra_payload_layout.addWidget(self.extra_payload_input)
        sd_layout.addLayout(extra_payload_layout)

        sd_group.setLayout(sd_layout)
        main_layout.addWidget(sd_group)

        # 初始化加载组数据
        self.refresh_sd_groups()

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

    def add_vae_input_field(self, text=""):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        
        input_field = QLineEdit(text)
        input_field.setPlaceholderText("例如: qwen_image_vae.safetensors")
        row_layout.addWidget(input_field)
        
        del_btn = QPushButton("-")
        del_btn.setFixedWidth(30)
        del_btn.clicked.connect(lambda: self.remove_vae_field(row_widget, input_field))
        row_layout.addWidget(del_btn)
        
        self.vae_inputs_container.addWidget(row_widget)
        self.vae_inputs_list.append(input_field)

    def remove_vae_field(self, widget, input_field):
        self.vae_inputs_container.removeWidget(widget)
        widget.deleteLater()
        if input_field in self.vae_inputs_list:
            self.vae_inputs_list.remove(input_field)

    # ---------- SD 组管理逻辑 ----------
    def refresh_sd_groups(self):
        self.sd_group_combo.blockSignals(True)
        self.sd_group_combo.clear()
        groups = list(self.config["sd_config_groups"].keys())
        self.sd_group_combo.addItems(groups)
        
        current = self.config.get("current_sd_group")
        if current in groups:
            self.sd_group_combo.setCurrentText(current)
            self.load_sd_group_to_ui(current)
        elif groups:
            self.sd_group_combo.setCurrentText(groups[0])
            self.load_sd_group_to_ui(groups[0])
            
        self.sd_group_combo.blockSignals(False)

    def load_sd_group_to_ui(self, group_name):
        settings = self.config["sd_config_groups"].get(group_name, {})
        self.sd_model_input.setText(settings.get("sd_model", ""))
        self.sampler_input.setText(settings.get("sampler", "Euler a"))
        self.scheduler_input.setText(settings.get("scheduler", "Automatic"))
        self.steps_input.setValue(settings.get("steps", 20))
        self.cfg_input.setValue(settings.get("cfg_scale", 7.0))

        # 清理旧的 VAE 框
        for i in reversed(range(self.vae_inputs_container.count())):
            widget = self.vae_inputs_container.itemAt(i).widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()
        self.vae_inputs_list.clear()

        # 加载新的 VAE 列表
        sd_vaes = settings.get("sd_vae", [])
        if isinstance(sd_vaes, str): # 兼容旧版配置
            sd_vaes = [sd_vaes] if sd_vaes and sd_vaes.lower() != "automatic" else []
            
        if not sd_vaes:
            self.add_vae_input_field("") # 默认给一个空框
        else:
            for vae in sd_vaes:
                self.add_vae_input_field(vae)

    def on_sd_group_changed(self, group_name):
        if group_name:
            self.config["current_sd_group"] = group_name
            self.load_sd_group_to_ui(group_name)

    def update_current_sd_group_from_ui(self):
        group_name = self.sd_group_combo.currentText()
        if not group_name:
            group_name = "Default"
            self.config["sd_config_groups"][group_name] = {}

        # 提取用户填写的有效 VAE（自动容错未输入的空框）
        valid_vaes = [f.text().strip() for f in self.vae_inputs_list if f.text().strip()]
        
        self.config["sd_config_groups"][group_name] = {
            "sd_model": self.sd_model_input.text().strip(),
            "sd_vae": valid_vaes,
            "sampler": self.sampler_input.text().strip(),
            "scheduler": self.scheduler_input.text().strip(),
            "steps": self.steps_input.value(),
            "cfg_scale": self.cfg_input.value()
        }
        self.config["current_sd_group"] = group_name

    def save_as_sd_group(self):
        valid_vaes = [f.text().strip() for f in self.vae_inputs_list if f.text().strip()]
        new_name, ok = QInputDialog.getText(self, "保存配置组", "请输入新配置组名称:")
        if ok and new_name.strip():
            group_name = new_name.strip()
            self.config["sd_config_groups"][group_name] = {
                "sd_model": self.sd_model_input.text().strip(),
                "sd_vae": valid_vaes,
                "sampler": self.sampler_input.text().strip(),
                "scheduler": self.scheduler_input.text().strip(),
                "steps": self.steps_input.value(),
                "cfg_scale": self.cfg_input.value()
            }
            self.config["current_sd_group"] = group_name
            self.refresh_sd_groups()
            QMessageBox.information(self, "成功", f"配置组 '{group_name}' 已保存！")

    def delete_sd_group(self):
        group_name = self.sd_group_combo.currentText()
        if len(self.config["sd_config_groups"]) <= 1:
            QMessageBox.warning(self, "警告", "必须保留至少一个配置组！")
            return
            
        reply = QMessageBox.question(self, "确认删除", f"确定要删除配置组 '{group_name}' 吗？", 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            del self.config["sd_config_groups"][group_name]
            self.config["current_sd_group"] = list(self.config["sd_config_groups"].keys())[0]
            self.refresh_sd_groups()

    # ---------- 原有辅助逻辑保留 ----------
    def fetch_available_models(self):
        url = f"{self.url_input.text().strip().rstrip('/')}/models"
        api_key = self.key_input.text().strip()
        
        if not api_key:
            QMessageBox.warning(self, "警告", "请先输入 API Key！")
            return
            
        headers = {"Authorization": f"Bearer {api_key}"}
        self.fetch_models_btn.setEnabled(False)
        self.fetch_models_btn.setText("获取中...")
        QApplication.processEvents()
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            
            if models:
                current_text = self.model_combo.currentText()
                self.model_combo.clear()
                self.model_combo.addItems(models)
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
        # 基础参数
        self.config["base_url"] = self.url_input.text().strip()
        self.config["api_key"] = self.key_input.text().strip()
        self.config["model"] = self.model_combo.currentText().strip()
        self.config["last_used_style"] = self.style_input.text().strip()
        self.config["sd_url"] = self.sd_url_input.text().strip()
        self.config["generate_count"] = self.count_input.value()
        
        # 固定提示词
        self.config["fixed_prompt"] = self.fixed_prompt_input.text().strip()
        self.config["fixed_negative_prompt"] = self.fixed_neg_prompt_input.text().strip()
        
        # 附加字段
        self.config["webui_extra_payload"] = self.extra_payload_input.toPlainText()
        
        # 更新当前选中组的参数
        self.update_current_sd_group_from_ui()
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

        # 开始前校验自定义 JSON 格式
        extra_json = self.extra_payload_input.toPlainText().strip()
        if extra_json:
            try:
                json.loads(extra_json)
            except json.JSONDecodeError as e:
                QMessageBox.warning(self, "JSON 格式错误", f"附加 Payload 解析失败，请检查语法:\n{e}")
                return

        self.update_config_from_ui()
        self.log_area.clear()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.fetch_models_btn.setEnabled(False)
        self.save_template_btn.setEnabled(False)
        self.save_as_template_btn.setEnabled(False)

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