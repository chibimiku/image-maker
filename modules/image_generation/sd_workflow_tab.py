import base64
import json
import logging
import os
import time

import requests
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from modules.others.api_backend import fetch_llm_json
from modules.others.tag_completer import TagAutocompleteManager
from utils.prompt_loader import get_prompt_path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_FILE = os.path.join(BASE_DIR, "conf", "config-sd.json")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
NEG_PROMPTS_DIR = os.path.join(BASE_DIR, "data", "negative_prompts")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CACHE_DIR = os.path.join(BASE_DIR, "cache", "sd-req")
SYSTEM_PROMPT_FILE = get_prompt_path("sd-make-system_prompt.md")

DEFAULT_SD_WORKFLOW_STATE = {
    "merge_system_prompt": True,
    "use_nsfw_text_api": False,
    "last_used_theme": "中秋主题少女",
    "last_used_template": "",
    "last_used_negative_template": "",
    "last_used_style": "",
    "last_used_style_key": "默认(无附加)",
    "generate_count": 3,
    "loop_count": 1,
    "fixed_prompt": "(masterpiece, best quality:1.2), ultra-detailed, highres",
    "fixed_negative_prompt": "(worst quality, low quality:1.4), bad anatomy, deformed, signature, watermark",
}

DEFAULT_SD_WEBUI_SETTINGS = {
    "sd_url": "http://127.0.0.1:7860",
    "current_sd_group": "Default",
    "webui_extra_payload": "{\n  \n}",
    "sd_config_groups": {
        "Default": {
            "sd_model": "",
            "sd_vae": ["Automatic"],
            "sampler": "Euler a",
            "scheduler": "Automatic",
            "steps": 20,
            "cfg_scale": 7.0,
        }
    },
}

SD_WORKFLOW_STATE_KEYS = set(DEFAULT_SD_WORKFLOW_STATE.keys())
SD_WEBUI_SETTING_KEYS = set(DEFAULT_SD_WEBUI_SETTINGS.keys())


def _read_sd_config_file():
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_sd_config_file(data):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_sd_workflow_state():
    loaded = _read_sd_config_file()
    state = dict(DEFAULT_SD_WORKFLOW_STATE)
    for key in SD_WORKFLOW_STATE_KEYS:
        if key in loaded:
            state[key] = loaded[key]
    return state


def save_sd_workflow_state(state):
    existing = _read_sd_config_file()
    for key in SD_WORKFLOW_STATE_KEYS:
        if key in state:
            existing[key] = state[key]
    _write_sd_config_file(existing)


def load_sd_webui_settings():
    loaded = _read_sd_config_file()
    settings = dict(DEFAULT_SD_WEBUI_SETTINGS)
    current_group = loaded.get("current_sd_group", settings["current_sd_group"])
    sd_groups = loaded.get("sd_config_groups", settings["sd_config_groups"])
    if not isinstance(sd_groups, dict) or not sd_groups:
        sd_groups = dict(DEFAULT_SD_WEBUI_SETTINGS["sd_config_groups"])
    settings["sd_url"] = str(loaded.get("sd_url", settings["sd_url"]) or "").strip() or settings["sd_url"]
    settings["current_sd_group"] = str(current_group or "Default").strip() or "Default"
    settings["webui_extra_payload"] = str(loaded.get("webui_extra_payload", settings["webui_extra_payload"]) or "")
    settings["sd_config_groups"] = sd_groups
    if settings["current_sd_group"] not in settings["sd_config_groups"]:
        settings["current_sd_group"] = next(iter(settings["sd_config_groups"].keys()))
    return settings


def save_sd_webui_settings(settings):
    existing = _read_sd_config_file()
    for key in SD_WEBUI_SETTING_KEYS:
        if key in settings:
            existing[key] = settings[key]
    _write_sd_config_file(existing)


def load_text_api_config_from_file(use_nsfw=False):
    config_path = os.path.join(BASE_DIR, "conf", "config.json")
    if not os.path.exists(config_path):
        return "", "", ""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return "", "", ""
    if use_nsfw:
        return (
            str(data.get("nsfw_base_url", data.get("base_url", "")) or "").strip(),
            str(data.get("nsfw_api_key", "") or "").strip(),
            str(data.get("nsfw_model", data.get("model", "")) or "").strip(),
        )
    return (
        str(data.get("base_url", "") or "").strip(),
        str(data.get("api_key", "") or "").strip(),
        str(data.get("model", "") or "").strip(),
    )


class GuiLogHandler(logging.Handler):
    """Forward api_backend logs into the widget log panel."""

    def __init__(self, log_signal):
        super().__init__()
        self.log_signal = log_signal
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record):
        try:
            self.log_signal.emit(self.format(record))
        except Exception:
            pass


class SdWorkflowThread(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, config, theme_text, template_text, neg_template_text):
        super().__init__()
        self.config = dict(config or {})
        self.theme_text = str(theme_text or "")
        self.template_text = str(template_text or "")
        self.neg_template_text = str(neg_template_text or "")
        self.is_running = True

    def run(self):
        whatai_logger = logging.getLogger("whatai_logger")
        gui_handler = GuiLogHandler(self.log_signal)
        whatai_logger.addHandler(gui_handler)
        try:
            generate_count = int(self.config.get("generate_count", 3) or 3)
            loop_count = int(self.config.get("loop_count", 1) or 1)

            if not self.is_running:
                self.log_signal.emit("任务已被用户中止。")
                self.finished_signal.emit()
                return

            total_images = generate_count * loop_count
            self.log_signal.emit(
                f"=== 开始总计 {loop_count} 轮请求，每轮 {generate_count} 组，预计共生成 {total_images} 张图片 ==="
            )
            global_img_index = 1

            for loop_idx in range(loop_count):
                if not self.is_running:
                    self.log_signal.emit("任务已被用户中止。")
                    break

                self.log_signal.emit(f"\n>>> 正在执行第 {loop_idx + 1}/{loop_count} 轮大模型请求...")
                llm_response = self.fetch_llm_prompt(generate_count)

                if not llm_response or "results" not in llm_response or not self.is_running:
                    self.log_signal.emit(f"第 {loop_idx + 1} 轮大模型请求失败或返回格式错误，跳过本轮。")
                    continue

                llm_data_list = llm_response.get("results", [])
                actual_count = len(llm_data_list)
                self.log_signal.emit(f"第 {loop_idx + 1} 轮 LLM 成功返回了 {actual_count} 组差异化提示词！")

                try:
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    cache_filename = os.path.join(
                        CACHE_DIR,
                        f"prompts_batch_{timestamp_str}_loop{loop_idx + 1}.json",
                    )
                    with open(cache_filename, "w", encoding="utf-8") as f:
                        json.dump(llm_response, f, ensure_ascii=False, indent=4)
                    self.log_signal.emit(f"本轮提示词已统一缓存至: {cache_filename}")
                except Exception as e:
                    self.log_signal.emit(f"提示词批次缓存失败: {e}")

                for i, llm_data in enumerate(llm_data_list):
                    if not self.is_running:
                        self.log_signal.emit("任务已被用户中止。")
                        break

                    self.log_signal.emit(
                        f"\n--- [总进度 {global_img_index}/{total_images}] 开始执行第 {loop_idx + 1} 轮的第 {i + 1}/{actual_count} 次 SD 绘图 ---"
                    )
                    self.log_signal.emit(
                        f"计划尺寸: {llm_data.get('width', 512)}x{llm_data.get('height', 512)}"
                    )
                    self.log_signal.emit(f"原始正向提示词: {llm_data.get('prompt', '')[:80]}...")
                    self.log_signal.emit("正在拼装最终提示词并将参数发送至本地 Stable Diffusion...")
                    self.generate_sd_image(llm_data)
                    global_img_index += 1

                    if self.is_running:
                        time.sleep(1)

            if self.is_running:
                self.log_signal.emit("\n✅ 全部工作流循环执行完毕。")
        except Exception as e:
            self.log_signal.emit(f"发生未捕获的异常: {e}")
        finally:
            whatai_logger.removeHandler(gui_handler)
            self.finished_signal.emit()

    def fetch_llm_prompt(self, generate_count):
        if not os.path.isfile(SYSTEM_PROMPT_FILE):
            self.log_signal.emit(f"❌ 缺少 System Prompt 文件：{SYSTEM_PROMPT_FILE}")
            return None

        try:
            with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
                raw_system_prompt = f.read()
        except Exception as e:
            self.log_signal.emit(f"读取 System Prompt 文件失败: {e}，将中止请求。")
            return None

        system_prompt = raw_system_prompt.replace("{generate_count}", str(generate_count))
        user_content = f"绘画主题: {self.theme_text}\n基础模板内容: {self.template_text}"

        self.log_signal.emit("\n>>> 准备发送大模型网络请求...")
        self.log_signal.emit(f"已加载外部系统提示词: {SYSTEM_PROMPT_FILE}")

        self.log_signal.emit("开始使用文本分析 API 发送请求...")
        reply_text = fetch_llm_json(
            base_url=self.config["base_url"],
            api_key=self.config["api_key"],
            model=self.config["model"],
            system_prompt=system_prompt,
            user_content=user_content,
            temperature=0.7,
            merge_system_prompt=self.config.get("merge_system_prompt", True),
        )

        if not reply_text:
            self.log_signal.emit("❌ 【错误】大模型请求失败或返回为空，请查阅 log 目录下的最新日志排查问题。")
            return None

        clean_text = reply_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]

        start_idx = clean_text.find("{")
        end_idx = clean_text.rfind("}")
        if start_idx == -1 or end_idx == -1:
            self.log_signal.emit("❌ 【错误】模型返回值中未找到 JSON 的大括号结构。")
            self.log_signal.emit(f"模型原始输出: \n{reply_text}")
            return None

        clean_json_str = clean_text[start_idx : end_idx + 1]
        try:
            return json.loads(clean_json_str)
        except json.JSONDecodeError as e:
            self.log_signal.emit("❌ 【错误 - JSON 解析失败】模型返回了不规范的 JSON 格式。")
            self.log_signal.emit(f"尝试解析的文本: \n{clean_json_str}")
            self.log_signal.emit(f"具体异常: {e}")
            return None

    def generate_sd_image(self, llm_data):
        url = f"{self.config['sd_url'].rstrip('/')}/sdapi/v1/txt2img"
        current_group_name = self.config.get("current_sd_group", "Default")
        sd_settings = self.config.get("sd_config_groups", {}).get(current_group_name, {})

        llm_prompt = llm_data.get("prompt", "").strip()
        style_prompt = self.config.get("last_used_style", "").strip()
        fixed_prompt = self.config.get("fixed_prompt", "").strip()
        final_prompt = ", ".join([p for p in [fixed_prompt, llm_prompt, style_prompt] if p])

        base_neg_prompt = self.neg_template_text.strip()
        fixed_neg_prompt = self.config.get("fixed_negative_prompt", "").strip()
        final_neg_prompt = ", ".join([p for p in [base_neg_prompt, fixed_neg_prompt] if p])

        payload = {
            "prompt": final_prompt,
            "negative_prompt": final_neg_prompt,
            "width": llm_data.get("width", 512),
            "height": llm_data.get("height", 512),
            "sampler_name": sd_settings.get("sampler", "Euler a"),
            "scheduler": sd_settings.get("scheduler", "Automatic"),
            "steps": sd_settings.get("steps", 20),
            "cfg_scale": sd_settings.get("cfg_scale", 7.0),
            "override_settings": {},
        }

        sd_model = sd_settings.get("sd_model", "").strip()
        sd_vae_list = sd_settings.get("sd_vae", [])
        if sd_model:
            payload["override_settings"]["sd_model_checkpoint"] = sd_model

        final_modules = [v.strip() for v in sd_vae_list if v.strip() and v.strip().lower() != "automatic"]
        if final_modules:
            payload["override_settings"]["forge_additional_modules"] = final_modules
        else:
            payload["override_settings"]["sd_vae"] = "Automatic"
            payload["override_settings"].pop("forge_additional_modules", None)

        extra_payload_str = self.config.get("webui_extra_payload", "").strip()
        if extra_payload_str:
            try:
                extra_payload = json.loads(extra_payload_str)
                if isinstance(extra_payload, dict):
                    for key, value in extra_payload.items():
                        if key == "override_settings" and isinstance(value, dict):
                            payload["override_settings"].update(value)
                        else:
                            payload[key] = value
                self.log_signal.emit("成功载入自定义 WebUI 附加 JSON 字段。")
            except Exception as e:
                self.log_signal.emit(f"警告：WebUI 附加字段 JSON 解析失败，已忽略 ({e})")

        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            images_base64 = response.json().get("images", [])

            for idx, img_b64 in enumerate(images_base64):
                img_data = base64.b64decode(img_b64)
                timestamp = int(time.time())
                date_str = time.strftime("%Y%m%d")
                output_dir = os.path.join(BASE_DIR, "data", date_str, "sdmake")
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.join(output_dir, f"gen_{timestamp}_{idx}.png")
                with open(filename, "wb") as img_file:
                    img_file.write(img_data)
                self.log_signal.emit(f"图片已保存至: {filename}")
        except Exception as e:
            self.log_signal.emit(f"SD WebUI API 错误: {e}")

    def stop(self):
        self.is_running = False


class SdWebuiSettingsWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.settings = load_sd_webui_settings()
        self.init_ui()
        self.load_settings_to_ui(self.settings.get("current_sd_group", "Default"))

    def init_ui(self):
        layout = QVBoxLayout(self)

        top_sd_layout = QHBoxLayout()
        top_sd_layout.addWidget(QLabel("SD API URL:"))
        self.sd_url_input = QLineEdit(self.settings.get("sd_url", ""))
        top_sd_layout.addWidget(self.sd_url_input)
        top_sd_layout.addWidget(QLabel("配置组:"))
        self.sd_group_combo = QComboBox()
        self.sd_group_combo.setMinimumWidth(140)
        self.sd_group_combo.currentTextChanged.connect(self.on_sd_group_changed)
        top_sd_layout.addWidget(self.sd_group_combo)
        self.save_sd_group_btn = QPushButton("保存为新配置组")
        self.save_sd_group_btn.clicked.connect(self.save_as_sd_group)
        top_sd_layout.addWidget(self.save_sd_group_btn)
        self.del_sd_group_btn = QPushButton("删除当前组")
        self.del_sd_group_btn.clicked.connect(self.delete_sd_group)
        top_sd_layout.addWidget(self.del_sd_group_btn)
        layout.addLayout(top_sd_layout)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Checkpoint:"))
        self.sd_model_input = QLineEdit()
        self.sd_model_input.setPlaceholderText("留空则使用 WebUI 当前模型")
        model_layout.addWidget(self.sd_model_input)
        layout.addLayout(model_layout)

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
        self.vae_inputs_list = []
        vae_main_layout.addLayout(self.vae_inputs_container)
        layout.addLayout(vae_main_layout)

        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("Sampler:"))
        self.sampler_input = QLineEdit()
        self.sampler_input.setFixedWidth(100)
        param_layout.addWidget(self.sampler_input)
        param_layout.addWidget(QLabel("Scheduler:"))
        self.scheduler_input = QLineEdit()
        self.scheduler_input.setFixedWidth(100)
        self.scheduler_input.setPlaceholderText("Automatic")
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
        layout.addLayout(param_layout)

        extra_payload_layout = QVBoxLayout()
        extra_payload_layout.addWidget(QLabel("WebUI 附加 Payload (JSON 格式，会被合并到 API 请求中):"))
        self.extra_payload_input = QTextEdit()
        self.extra_payload_input.setMaximumHeight(90)
        self.extra_payload_input.setPlainText(self.settings.get("webui_extra_payload", ""))
        extra_payload_layout.addWidget(self.extra_payload_input)
        layout.addLayout(extra_payload_layout)

        save_layout = QHBoxLayout()
        save_layout.addStretch()
        self.save_btn = QPushButton("保存 SD-WebUI 接口配置")
        self.save_btn.clicked.connect(lambda: self.save_settings(silent=False))
        save_layout.addWidget(self.save_btn)
        layout.addLayout(save_layout)

        self.refresh_sd_groups()

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

    def refresh_sd_groups(self):
        self.sd_group_combo.blockSignals(True)
        self.sd_group_combo.clear()
        groups = list(self.settings.get("sd_config_groups", {}).keys())
        self.sd_group_combo.addItems(groups)
        current = self.settings.get("current_sd_group")
        if current in groups:
            self.sd_group_combo.setCurrentText(current)
        elif groups:
            self.sd_group_combo.setCurrentText(groups[0])
        self.sd_group_combo.blockSignals(False)

    def load_settings_to_ui(self, group_name):
        settings = self.settings.get("sd_config_groups", {}).get(group_name, {})
        self.sd_url_input.setText(self.settings.get("sd_url", ""))
        self.extra_payload_input.setPlainText(self.settings.get("webui_extra_payload", ""))
        self.sd_model_input.setText(settings.get("sd_model", ""))
        self.sampler_input.setText(settings.get("sampler", "Euler a"))
        self.scheduler_input.setText(settings.get("scheduler", "Automatic"))
        self.steps_input.setValue(settings.get("steps", 20))
        self.cfg_input.setValue(settings.get("cfg_scale", 7.0))

        for i in reversed(range(self.vae_inputs_container.count())):
            layout_item = self.vae_inputs_container.itemAt(i)
            if layout_item is None:
                continue
            widget = layout_item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.vae_inputs_list.clear()

        sd_vaes = settings.get("sd_vae", [])
        if isinstance(sd_vaes, str):
            sd_vaes = [sd_vaes] if sd_vaes and sd_vaes.lower() != "automatic" else []
        if not sd_vaes:
            self.add_vae_input_field("")
        else:
            for vae in sd_vaes:
                self.add_vae_input_field(vae)

    def on_sd_group_changed(self, group_name):
        if group_name:
            self.settings["current_sd_group"] = group_name
            self.load_settings_to_ui(group_name)

    def update_current_group_from_ui(self):
        group_name = self.sd_group_combo.currentText() or "Default"
        if group_name not in self.settings["sd_config_groups"]:
            self.settings["sd_config_groups"][group_name] = {}
        valid_vaes = [field.text().strip() for field in self.vae_inputs_list if field.text().strip()]
        self.settings["sd_url"] = self.sd_url_input.text().strip()
        self.settings["webui_extra_payload"] = self.extra_payload_input.toPlainText()
        self.settings["current_sd_group"] = group_name
        self.settings["sd_config_groups"][group_name] = {
            "sd_model": self.sd_model_input.text().strip(),
            "sd_vae": valid_vaes,
            "sampler": self.sampler_input.text().strip(),
            "scheduler": self.scheduler_input.text().strip(),
            "steps": self.steps_input.value(),
            "cfg_scale": self.cfg_input.value(),
        }

    def get_settings(self):
        self.update_current_group_from_ui()
        current = dict(self.settings)
        current["sd_config_groups"] = dict(self.settings.get("sd_config_groups", {}))
        return current

    def save_settings(self, silent=False):
        extra_json = self.extra_payload_input.toPlainText().strip()
        if extra_json:
            try:
                json.loads(extra_json)
            except json.JSONDecodeError as e:
                if not silent:
                    QMessageBox.warning(self, "JSON 格式错误", f"附加 Payload 解析失败，请检查语法:\n{e}")
                return False
        self.update_current_group_from_ui()
        save_sd_webui_settings(self.settings)
        if not silent:
            QMessageBox.information(self, "成功", "SD-WebUI 接口配置已保存。")
        return True

    def save_as_sd_group(self):
        self.update_current_group_from_ui()
        new_name, ok = QInputDialog.getText(self, "保存配置组", "请输入新配置组名称:")
        if ok and new_name.strip():
            group_name = new_name.strip()
            valid_vaes = [field.text().strip() for field in self.vae_inputs_list if field.text().strip()]
            self.settings["sd_config_groups"][group_name] = {
                "sd_model": self.sd_model_input.text().strip(),
                "sd_vae": valid_vaes,
                "sampler": self.sampler_input.text().strip(),
                "scheduler": self.scheduler_input.text().strip(),
                "steps": self.steps_input.value(),
                "cfg_scale": self.cfg_input.value(),
            }
            self.settings["current_sd_group"] = group_name
            self.refresh_sd_groups()
            self.sd_group_combo.setCurrentText(group_name)
            QMessageBox.information(self, "成功", f"配置组 '{group_name}' 已保存！")

    def delete_sd_group(self):
        group_name = self.sd_group_combo.currentText()
        if len(self.settings.get("sd_config_groups", {})) <= 1:
            QMessageBox.warning(self, "警告", "必须保留至少一个配置组！")
            return
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要删除配置组 '{group_name}' 吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            del self.settings["sd_config_groups"][group_name]
            self.settings["current_sd_group"] = list(self.settings["sd_config_groups"].keys())[0]
            self.refresh_sd_groups()
            self.load_settings_to_ui(self.settings["current_sd_group"])


class SdWorkflowWidget(QWidget):
    def __init__(
        self,
        text_config_getter_func=None,
        sd_webui_settings_getter_func=None,
        styles_getter_func=None,
        current_style_name_getter_func=None,
        style_changed_callback=None,
    ):
        super().__init__()
        self.setWindowTitle("AI 自动化绘画工作流")
        self.resize(1000, 900)

        os.makedirs(PROMPTS_DIR, exist_ok=True)
        os.makedirs(NEG_PROMPTS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)

        self.text_config_getter = text_config_getter_func or load_text_api_config_from_file
        self.sd_webui_settings_getter = sd_webui_settings_getter_func or load_sd_webui_settings
        self.styles_getter = styles_getter_func
        self.current_style_name_getter = current_style_name_getter_func
        self.style_changed_callback = style_changed_callback
        self.config = dict(DEFAULT_SD_WORKFLOW_STATE)
        self.worker = None
        self.tag_manager = TagAutocompleteManager()

        self.load_config()
        self.init_ui()

    def load_config(self):
        self.config.update(load_sd_workflow_state())

    def save_config(self):
        save_sd_workflow_state(self.config)

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        llm_group = QGroupBox("大模型接口来源")
        llm_layout = QGridLayout()
        llm_layout.addWidget(QLabel("直接复用“设置 -> 文本分析 API”中的配置。"), 0, 0, 1, 4)
        self.use_nsfw_text_api_cb = QCheckBox("使用文本分析（NSFW）配置")
        self.use_nsfw_text_api_cb.setChecked(bool(self.config.get("use_nsfw_text_api", False)))
        self.use_nsfw_text_api_cb.toggled.connect(self.refresh_text_api_summary)
        llm_layout.addWidget(self.use_nsfw_text_api_cb, 1, 0, 1, 2)
        self.merge_prompt_cb = QCheckBox("启用 System Prompt 兼容模式 (合并到 User)")
        self.merge_prompt_cb.setChecked(self.config.get("merge_system_prompt", True))
        llm_layout.addWidget(self.merge_prompt_cb, 1, 2, 1, 2)
        self.text_api_summary_label = QLabel()
        self.text_api_summary_label.setWordWrap(True)
        llm_layout.addWidget(self.text_api_summary_label, 2, 0, 1, 4)
        llm_group.setLayout(llm_layout)
        main_layout.addWidget(llm_group)

        task_group = QGroupBox("任务与模板设置")
        task_layout = QVBoxLayout()
        theme_style_layout = QFormLayout()
        self.theme_input = QLineEdit(self.config.get("last_used_theme", "中秋主题少女"))
        self.theme_input.setPlaceholderText("例如：中秋主题少女、赛博朋克城市...")
        theme_style_layout.addRow("绘画主题 (必填):", self.theme_input)

        self.style_combo = QComboBox()
        self.style_combo.setMinimumWidth(200)
        self.style_combo.currentTextChanged.connect(self.on_style_changed)
        theme_style_layout.addRow("Prompt 风格预设:", self.style_combo)
        self.load_style_options()
        task_layout.addLayout(theme_style_layout)

        template_ctrl_layout = QHBoxLayout()
        template_ctrl_layout.addWidget(QLabel("正向模板 (交由 LLM 扩写):"))
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
        self.template_editor.setPlaceholderText("在这里编辑需要发给大模型进行细节扩写的【正向提示词】基础模板...")
        self.template_editor.setMaximumHeight(80)
        task_layout.addWidget(self.template_editor)

        neg_template_ctrl_layout = QHBoxLayout()
        neg_template_ctrl_layout.addWidget(QLabel("反向模板 (直接发给 SD 引擎):"))
        self.neg_template_combo = QComboBox()
        self.neg_template_combo.setMinimumWidth(200)
        self.neg_template_combo.currentTextChanged.connect(self.load_negative_template_content)
        neg_template_ctrl_layout.addWidget(self.neg_template_combo)
        self.save_neg_template_btn = QPushButton("保存反向模板")
        self.save_neg_template_btn.clicked.connect(self.save_current_negative_template)
        neg_template_ctrl_layout.addWidget(self.save_neg_template_btn)
        self.save_as_neg_template_btn = QPushButton("反向模板另存...")
        self.save_as_neg_template_btn.clicked.connect(self.save_as_new_negative_template)
        neg_template_ctrl_layout.addWidget(self.save_as_neg_template_btn)
        task_layout.addLayout(neg_template_ctrl_layout)

        self.neg_template_editor = QTextEdit()
        self.neg_template_editor.setPlaceholderText("在这里编辑【反向提示词】内容，这段文本不会经过大模型。")
        self.neg_template_editor.setMaximumHeight(80)
        task_layout.addWidget(self.neg_template_editor)
        task_group.setLayout(task_layout)
        main_layout.addWidget(task_group)

        self.refresh_templates()
        self.refresh_negative_templates()

        workflow_group = QGroupBox("批量生成参数")
        workflow_layout = QVBoxLayout()
        runtime_hint = QLabel("Stable Diffusion WebUI 接口配置已移动到“设置 -> SD-WebUI接口配置”。")
        runtime_hint.setWordWrap(True)
        workflow_layout.addWidget(runtime_hint)

        param_layout = QHBoxLayout()
        param_layout.addWidget(QLabel("大模型请求轮数(Y):"))
        self.loop_count_input = QSpinBox()
        self.loop_count_input.setRange(1, 9999)
        self.loop_count_input.setValue(self.config.get("loop_count", 1))
        param_layout.addWidget(self.loop_count_input)
        param_layout.addWidget(QLabel("单次返回组数(X):"))
        self.count_input = QSpinBox()
        self.count_input.setRange(1, 9999)
        self.count_input.setValue(self.config.get("generate_count", 3))
        param_layout.addWidget(self.count_input)
        workflow_layout.addLayout(param_layout)

        fixed_prompt_layout = QFormLayout()
        self.fixed_prompt_input = QLineEdit(self.config.get("fixed_prompt", ""))
        self.fixed_prompt_input.setPlaceholderText("自动拼接到大模型结果后")
        fixed_prompt_layout.addRow("附加固定正向提示词:", self.fixed_prompt_input)
        self.fixed_neg_prompt_input = QLineEdit(self.config.get("fixed_negative_prompt", ""))
        self.fixed_neg_prompt_input.setPlaceholderText("自动拼接到最终反向提示词后")
        fixed_prompt_layout.addRow("附加固定反向提示词:", self.fixed_neg_prompt_input)
        workflow_layout.addLayout(fixed_prompt_layout)

        self.sd_webui_summary_label = QLabel()
        self.sd_webui_summary_label.setWordWrap(True)
        workflow_layout.addWidget(self.sd_webui_summary_label)
        workflow_group.setLayout(workflow_layout)
        main_layout.addWidget(workflow_group)

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

        self.template_editor_completer = self.tag_manager.setup_text_edit(self.template_editor)
        self.fixed_prompt_completer = self.tag_manager.setup_line_edit(self.fixed_prompt_input)
        self.refresh_text_api_summary()
        self.refresh_sd_webui_summary()

    def closeEvent(self, event):
        self.update_config_from_ui()
        super().closeEvent(event)

    def refresh_text_api_summary(self):
        use_nsfw = bool(self.use_nsfw_text_api_cb.isChecked())
        base_url, _api_key, model = self.text_config_getter(use_nsfw)
        api_name = "文本分析（NSFW）" if use_nsfw else "文本分析 API"
        url_text = base_url or "未配置"
        model_text = model or "未配置"
        self.text_api_summary_label.setText(f"当前来源: {api_name} | Base URL: {url_text} | Model: {model_text}")

    def refresh_sd_webui_summary(self):
        settings = self.sd_webui_settings_getter()
        group_name = str(settings.get("current_sd_group", "Default") or "Default")
        sd_url = str(settings.get("sd_url", "") or "").strip() or "未配置"
        group = settings.get("sd_config_groups", {}).get(group_name, {})
        checkpoint = str(group.get("sd_model", "") or "").strip() or "沿用 WebUI 当前模型"
        sampler = str(group.get("sampler", "Euler a") or "Euler a")
        steps = group.get("steps", 20)
        cfg_scale = group.get("cfg_scale", 7.0)
        self.sd_webui_summary_label.setText(
            f"当前 WebUI 配置组: {group_name} | URL: {sd_url} | Checkpoint: {checkpoint} | Sampler: {sampler} | Steps: {steps} | CFG: {cfg_scale}"
        )

    def _get_style_options(self):
        if callable(self.styles_getter):
            try:
                loaded = self.styles_getter() or {}
                if isinstance(loaded, dict) and loaded:
                    return dict(loaded)
            except Exception:
                pass

        styles_file = os.path.join(BASE_DIR, "conf", "config-styles.json")
        try:
            if os.path.exists(styles_file):
                with open(styles_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict) and loaded:
                    return loaded
        except Exception as e:
            print(f"加载风格文件失败: {e}")
        return {"默认(无附加)": ""}

    def _get_current_style_name(self):
        if callable(self.current_style_name_getter):
            try:
                value = str(self.current_style_name_getter() or "").strip()
                if value:
                    return value
            except Exception:
                pass
        return str(self.config.get("last_used_style_key", "默认(无附加)") or "默认(无附加)")

    def _get_selected_style_prompt(self):
        selected_style_key = self.style_combo.currentText().strip()
        return str(self.style_options.get(selected_style_key, "") or "")

    def update_styles(self, style_names=None, current_style_name=None):
        self.style_options = self._get_style_options()
        if style_names:
            ordered = []
            for name in style_names:
                key = str(name or "").strip()
                if key and key not in ordered:
                    ordered.append(key)
        else:
            ordered = list(self.style_options.keys())
        if "默认(无附加)" not in ordered:
            ordered.insert(0, "默认(无附加)")
            self.style_options.setdefault("默认(无附加)", "")

        selected_style_name = current_style_name or self._get_current_style_name()
        if selected_style_name not in ordered:
            selected_style_name = "默认(无附加)"

        self.style_combo.blockSignals(True)
        self.style_combo.clear()
        self.style_combo.addItems(ordered)
        self.style_combo.setCurrentText(selected_style_name)
        self.style_combo.blockSignals(False)

    def refresh_templates(self):
        self.template_combo.blockSignals(True)
        self.template_combo.clear()
        templates = [f for f in os.listdir(PROMPTS_DIR) if f.endswith(".txt")]
        if templates:
            self.template_combo.addItems(templates)
            last_template = self.config.get("last_used_template", "")
            selected = last_template if last_template in templates else templates[0]
            self.template_combo.setCurrentText(selected)
            self.load_template_content(selected)
        else:
            self.template_combo.addItem("未找到 txt 文件")
            self.template_editor.clear()
        self.template_combo.blockSignals(False)

    def load_template_content(self, filename):
        if not filename or filename == "未找到 txt 文件":
            return
        filepath = os.path.join(PROMPTS_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                self.template_editor.setPlainText(f.read())

    def save_current_template(self):
        filename = self.template_combo.currentText()
        if not filename or filename == "未找到 txt 文件":
            QMessageBox.warning(self, "警告", "当前没有选中有效的模板文件。")
            return
        filepath = os.path.join(PROMPTS_DIR, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
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
                reply = QMessageBox.question(
                    self,
                    "确认覆盖",
                    f"文件 '{filename}' 已存在，是否覆盖？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.No:
                    return
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(self.template_editor.toPlainText())
                self.refresh_templates()
                self.template_combo.setCurrentText(filename)
                QMessageBox.information(self, "成功", f"新模板 '{filename}' 已保存！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def refresh_negative_templates(self):
        self.neg_template_combo.blockSignals(True)
        self.neg_template_combo.clear()
        templates = [f for f in os.listdir(NEG_PROMPTS_DIR) if f.endswith(".txt")]
        if templates:
            self.neg_template_combo.addItems(templates)
            last_template = self.config.get("last_used_negative_template", "")
            selected = last_template if last_template in templates else templates[0]
            self.neg_template_combo.setCurrentText(selected)
            self.load_negative_template_content(selected)
        else:
            self.neg_template_combo.addItem("未找到 txt 文件")
            self.neg_template_editor.clear()
        self.neg_template_combo.blockSignals(False)

    def load_negative_template_content(self, filename):
        if not filename or filename == "未找到 txt 文件":
            return
        filepath = os.path.join(NEG_PROMPTS_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                self.neg_template_editor.setPlainText(f.read())

    def save_current_negative_template(self):
        filename = self.neg_template_combo.currentText()
        if not filename or filename == "未找到 txt 文件":
            QMessageBox.warning(self, "警告", "当前没有选中有效的反向模板文件。")
            return
        filepath = os.path.join(NEG_PROMPTS_DIR, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.neg_template_editor.toPlainText())
            QMessageBox.information(self, "成功", f"反向模板 '{filename}' 已保存！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def save_as_new_negative_template(self):
        new_name, ok = QInputDialog.getText(self, "反向模板另存为", "请输入新反向模板名称 (无需输入 .txt 后缀):")
        if ok and new_name.strip():
            filename = f"{new_name.strip()}.txt"
            filepath = os.path.join(NEG_PROMPTS_DIR, filename)
            if os.path.exists(filepath):
                reply = QMessageBox.question(
                    self,
                    "确认覆盖",
                    f"文件 '{filename}' 已存在，是否覆盖？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.No:
                    return
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(self.neg_template_editor.toPlainText())
                self.refresh_negative_templates()
                self.neg_template_combo.setCurrentText(filename)
                QMessageBox.information(self, "成功", f"新反向模板 '{filename}' 已保存！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def append_log(self, text):
        timestamp = time.strftime("%H:%M:%S")
        self.log_area.append(f"[{timestamp}] {text}")
        scrollbar = self.log_area.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    def load_style_options(self):
        self.update_styles()

    def on_style_changed(self, style_name):
        style_name = str(style_name or "").strip()
        if not style_name:
            return
        if callable(self.style_changed_callback):
            self.style_changed_callback(style_name)
        else:
            self.config["last_used_style_key"] = style_name
            self.config["last_used_style"] = str(self.style_options.get(style_name, "") or "")
            self.save_config()

    def update_config_from_ui(self):
        self.config["last_used_theme"] = self.theme_input.text().strip()
        self.config["last_used_template"] = self.template_combo.currentText()
        self.config["last_used_negative_template"] = self.neg_template_combo.currentText()
        selected_style_key = self.style_combo.currentText()
        if not callable(self.style_changed_callback):
            self.config["last_used_style_key"] = selected_style_key
            self.config["last_used_style"] = self.style_options.get(selected_style_key, "")
        self.config["generate_count"] = self.count_input.value()
        self.config["loop_count"] = self.loop_count_input.value()
        self.config["merge_system_prompt"] = self.merge_prompt_cb.isChecked()
        self.config["use_nsfw_text_api"] = self.use_nsfw_text_api_cb.isChecked()
        self.config["fixed_prompt"] = self.fixed_prompt_input.text().strip()
        self.config["fixed_negative_prompt"] = self.fixed_neg_prompt_input.text().strip()
        self.save_config()

    def start_workflow(self):
        theme = self.theme_input.text().strip()
        if not theme:
            QMessageBox.warning(self, "警告", "请填写绘画主题！")
            return

        template_text = self.template_editor.toPlainText().strip()
        if not template_text:
            QMessageBox.warning(self, "警告", "正向模板内容不能为空！")
            return

        neg_template_text = self.neg_template_editor.toPlainText().strip()
        use_nsfw = bool(self.use_nsfw_text_api_cb.isChecked())
        base_url, api_key, model = self.text_config_getter(use_nsfw)
        if not base_url or not api_key or not model:
            api_name = "文本分析（NSFW）" if use_nsfw else "文本分析 API"
            QMessageBox.warning(self, "警告", f"请先在设置中补全 {api_name} 的 Base URL / API Key / Model。")
            return

        sd_webui_settings = self.sd_webui_settings_getter()
        extra_json = str(sd_webui_settings.get("webui_extra_payload", "") or "").strip()
        if extra_json:
            try:
                json.loads(extra_json)
            except json.JSONDecodeError as e:
                QMessageBox.warning(self, "JSON 格式错误", f"请先修正“设置 -> SD-WebUI接口配置”中的附加 Payload:\n{e}")
                return

        self.update_config_from_ui()
        self.log_area.clear()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_template_btn.setEnabled(False)
        self.save_as_template_btn.setEnabled(False)
        self.save_neg_template_btn.setEnabled(False)
        self.save_as_neg_template_btn.setEnabled(False)

        runtime_config = dict(self.config)
        runtime_config["base_url"] = base_url
        runtime_config["api_key"] = api_key
        runtime_config["model"] = model
        runtime_config["last_used_style_key"] = self.style_combo.currentText().strip()
        runtime_config["last_used_style"] = self._get_selected_style_prompt()
        runtime_config.update(sd_webui_settings)
        self.refresh_text_api_summary()
        self.refresh_sd_webui_summary()
        self.worker = SdWorkflowThread(runtime_config, theme, template_text, neg_template_text)
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
        self.save_template_btn.setEnabled(True)
        self.save_as_template_btn.setEnabled(True)
        self.save_neg_template_btn.setEnabled(True)
        self.save_as_neg_template_btn.setEnabled(True)
