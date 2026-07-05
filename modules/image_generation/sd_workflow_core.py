import base64
import json
import logging
import os
import re
import time

import requests
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from modules.others.api_backend import fetch_llm_json
from utils.prompt_loader import get_prompt_path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_FILE = os.path.join(BASE_DIR, "conf", "config-sd.json")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")
NEG_PROMPTS_DIR = os.path.join(BASE_DIR, "data", "negative_prompts")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CACHE_DIR = os.path.join(BASE_DIR, "cache", "sd-req")
STORY_SEQUENCE_DIR = os.path.join(BASE_DIR, "data", "story_sequences")
SYSTEM_PROMPT_FILE = get_prompt_path("sd-make-system_prompt.md")
STORY_SYSTEM_PROMPT_FILE = get_prompt_path("sd-story-sequence-system_prompt.md")
STORY_PAGE_SYSTEM_PROMPT_FILE = get_prompt_path("sd-story-page-system_prompt.md")

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
    "story_page_count": 6,
    "story_prompt_preset": "平衡",
    "story_prompt_min_words": 250,
    "story_prompt_keyword_count": 30,
    "story_no_character_description": False,
    "story_no_appearance_description": False,
    "story_no_outfit_description": False,
    "last_story_json_path": "",
    "fixed_prompt": "(masterpiece, best quality:1.2), ultra-detailed, highres",
    "fixed_negative_prompt": "(worst quality, low quality:1.4), bad anatomy, deformed, signature, watermark",
    "webui_extra_payload": "{\n  \n}",
}

STORY_PROMPT_PRESETS = {
    "保守": {"min_words": 180, "keyword_count": 16},
    "平衡": {"min_words": 250, "keyword_count": 30},
    "长描述": {"min_words": 400, "keyword_count": 40},
}

STORY_RESOLUTION_PRESETS = [
    (1024, 1536),  # 2:3
    (1536, 1024),  # 3:2
    (1824, 1024),  # 16:9
    (1024, 1824),  # 9:16
    (1344, 1024),  # 4:3
    (1024, 1344),  # 3:4
    (1024, 1024),  # 1:1
]

DEFAULT_SD_WEBUI_SETTINGS = {
    "sd_url": "http://127.0.0.1:7860",
    "current_sd_group": "Default",
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


def _strip_json_fence(reply_text):
    clean_text = str(reply_text or "").strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]
    return clean_text.strip()


def _parse_json_reply(reply_text):
    clean_text = _strip_json_fence(reply_text)
    start_idx = clean_text.find("{")
    end_idx = clean_text.rfind("}")
    if start_idx == -1 or end_idx == -1:
        raise ValueError("模型返回值中未找到 JSON 的大括号结构。")
    return json.loads(clean_text[start_idx : end_idx + 1])


def _coerce_positive_int(value, default_value):
    try:
        parsed = int(value)
        return parsed if parsed > 0 else int(default_value)
    except Exception:
        return int(default_value)


def _normalize_story_resolution(width, height):
    parsed_width = _coerce_positive_int(width, 1024)
    parsed_height = _coerce_positive_int(height, 1536)
    if parsed_width == parsed_height:
        return 1024, 1024

    target_ratio = parsed_width / float(parsed_height or 1)
    best_width, best_height = STORY_RESOLUTION_PRESETS[0]
    best_score = None
    for preset_width, preset_height in STORY_RESOLUTION_PRESETS:
        preset_ratio = preset_width / float(preset_height or 1)
        score = (
            abs(target_ratio - preset_ratio),
            abs(parsed_width - preset_width) + abs(parsed_height - preset_height),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_width, best_height = preset_width, preset_height
    return best_width, best_height


def _sanitize_filename_part(text, max_length=16):
    raw = str(text or "").strip()
    safe_chars = []
    for ch in raw:
        if ch.isalnum() or ch in ("-", "_"):
            safe_chars.append(ch)
        elif ch.isspace():
            safe_chars.append("_")
    safe = "".join(safe_chars).strip("_")
    return safe[: max(1, int(max_length))] or "story"


def normalize_story_sequence(sequence_data, default_theme="", expected_pages=0):
    if not isinstance(sequence_data, dict):
        raise ValueError("故事序列必须是 JSON 对象。")

    raw_pages = sequence_data.get("pages")
    if raw_pages is None:
        raw_pages = sequence_data.get("results")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise ValueError("故事序列中缺少有效的 pages 数组。")

    normalized_pages = []
    for index, raw_page in enumerate(raw_pages, start=1):
        if not isinstance(raw_page, dict):
            continue
        prompt_en = str(
            raw_page.get("prompt_en")
            or raw_page.get("english_prompt")
            or raw_page.get("prompt")
            or ""
        ).strip()
        prompt_zh = str(
            raw_page.get("prompt_zh")
            or raw_page.get("prompt_cn")
            or raw_page.get("chinese_prompt")
            or raw_page.get("translation_zh")
            or ""
        ).strip()
        if not prompt_en:
            raise ValueError(f"第 {index} 页缺少英文 prompt。")
        if not prompt_zh:
            raise ValueError(f"第 {index} 页缺少中文翻译 prompt。")

        normalized_width, normalized_height = _normalize_story_resolution(
            raw_page.get("width"), raw_page.get("height")
        )
        normalized_pages.append(
            {
                "page": _coerce_positive_int(raw_page.get("page"), index),
                "title_zh": str(raw_page.get("title_zh") or raw_page.get("scene_title_zh") or "").strip(),
                "title_en": str(raw_page.get("title_en") or raw_page.get("scene_title_en") or "").strip(),
                "prompt_en": prompt_en,
                "prompt_zh": prompt_zh,
                "width": normalized_width,
                "height": normalized_height,
            }
        )

    normalized_pages.sort(key=lambda item: item.get("page", 0))
    for index, page in enumerate(normalized_pages, start=1):
        page["page"] = index

    if expected_pages and len(normalized_pages) != int(expected_pages):
        raise ValueError(f"故事序列页数不匹配，期望 {expected_pages} 页，实际 {len(normalized_pages)} 页。")

    theme_text = str(sequence_data.get("theme") or default_theme or "").strip()
    return {
        "theme": theme_text,
        "title_zh": str(sequence_data.get("title_zh") or sequence_data.get("story_title_zh") or "").strip(),
        "title_en": str(sequence_data.get("title_en") or sequence_data.get("story_title_en") or "").strip(),
        "page_count": len(normalized_pages),
        "pages": normalized_pages,
    }


def save_story_sequence(sequence_data, target_path=None):
    os.makedirs(STORY_SEQUENCE_DIR, exist_ok=True)
    normalized = normalize_story_sequence(sequence_data)
    final_path = target_path
    if not final_path:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{_sanitize_filename_part(normalized.get('theme', 'story'), max_length=10)}.json"
        final_path = os.path.join(STORY_SEQUENCE_DIR, filename)
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    return final_path, normalized


def load_story_sequence(story_path):
    with open(story_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    return normalize_story_sequence(loaded)


def load_webui_extra_payload(payload_text):
    text = str(payload_text or "").strip()
    if not text:
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("WebUI 附加 Payload 必须是 JSON 对象。")
    return payload


def dump_webui_extra_payload(payload_dict):
    payload = payload_dict if isinstance(payload_dict, dict) else {}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _format_payload_value_text(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return str(value)


def _payload_value_preview(value_text, max_length=90):
    single_line = " ".join(str(value_text or "").split())
    if len(single_line) <= max_length:
        return single_line
    return f"{single_line[: max_length - 3]}..."


def _parse_payload_value_text(value_text):
    raw_text = str(value_text or "")
    stripped = raw_text.strip()
    if not stripped:
        return ""

    lowered = stripped.lower()
    if lowered in ("true", "false", "null"):
        return json.loads(lowered)
    if stripped.startswith("{") or stripped.startswith("[") or (stripped.startswith('"') and stripped.endswith('"')):
        try:
            return json.loads(stripped)
        except Exception as e:
            raise ValueError(f"JSON 值解析失败: {e}") from e
    if re.fullmatch(r"-?\d+", stripped):
        return int(stripped)
    if re.fullmatch(r"-?\d+\.\d+", stripped):
        return float(stripped)
    return raw_text


def normalize_story_outline(sequence_data, default_theme="", expected_pages=0):
    if not isinstance(sequence_data, dict):
        raise ValueError("故事大纲必须是 JSON 对象。")

    raw_pages = sequence_data.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        raise ValueError("故事大纲中缺少有效的 pages 数组。")

    normalized_pages = []
    for index, raw_page in enumerate(raw_pages, start=1):
        if not isinstance(raw_page, dict):
            continue
        normalized_width, normalized_height = _normalize_story_resolution(
            raw_page.get("width"), raw_page.get("height")
        )
        normalized_pages.append(
            {
                "page": _coerce_positive_int(raw_page.get("page"), index),
                "title_zh": str(raw_page.get("title_zh") or raw_page.get("scene_title_zh") or f"第{index}页").strip(),
                "title_en": str(raw_page.get("title_en") or raw_page.get("scene_title_en") or f"Page {index}").strip(),
                "scene_summary_zh": str(
                    raw_page.get("scene_summary_zh")
                    or raw_page.get("summary_zh")
                    or raw_page.get("scene_zh")
                    or ""
                ).strip(),
                "scene_summary_en": str(
                    raw_page.get("scene_summary_en")
                    or raw_page.get("summary_en")
                    or raw_page.get("scene_en")
                    or ""
                ).strip(),
                "width": normalized_width,
                "height": normalized_height,
            }
        )

    normalized_pages.sort(key=lambda item: item.get("page", 0))
    for index, page in enumerate(normalized_pages, start=1):
        page["page"] = index

    if expected_pages and len(normalized_pages) != int(expected_pages):
        raise ValueError(f"故事大纲页数不匹配，期望 {expected_pages} 页，实际 {len(normalized_pages)} 页。")

    return {
        "theme": str(sequence_data.get("theme") or default_theme or "").strip(),
        "title_zh": str(sequence_data.get("title_zh") or sequence_data.get("story_title_zh") or "").strip(),
        "title_en": str(sequence_data.get("title_en") or sequence_data.get("story_title_en") or "").strip(),
        "page_count": len(normalized_pages),
        "pages": normalized_pages,
    }


def _extract_chat_message_and_reason(resp_json):
    choices = resp_json.get("choices", []) if isinstance(resp_json, dict) else []
    if not choices:
        raise ValueError("模型返回中缺少 choices。")

    choice = choices[0] or {}
    message = choice.get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
                elif "text" in item:
                    text_parts.append(str(item.get("text", "")))
            else:
                text_parts.append(str(item))
        content = "".join(text_parts)
    finish_reason = str(
        choice.get("finish_reason")
        or choice.get("stop_reason")
        or resp_json.get("stop_reason")
        or ""
    ).strip().lower()
    return str(content or ""), finish_reason


def _request_chat_completion(base_url, api_key, model, messages, temperature=0.7, force_json=False, timeout=120):
    logger = logging.getLogger("whatai_logger")
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if force_json:
        payload["response_format"] = {"type": "json_object"}

    logger.info("=== 发起故事序列 LLM 请求 ===")
    logger.info(f"请求 URL: {url}")
    logger.info(f"请求 Payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    resp_json = resp.json()
    logger.info(f"=== 故事序列 LLM 原始返回 ===\n{json.dumps(resp_json, ensure_ascii=False, indent=2)}")
    return resp_json


def fetch_llm_reply_with_continuation(
    base_url,
    api_key,
    model,
    system_prompt,
    user_content,
    temperature=0.7,
    merge_system_prompt=True,
    force_json=False,
    max_rounds=8,
):
    if merge_system_prompt:
        messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_content}"}]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    collected_parts = []
    continue_prompt = (
        "Continue exactly from the last character of your previous answer. "
        "Do not restart, do not repeat, do not add explanations, and do not wrap with markdown. "
        "Only output the remaining continuation so the combined result becomes one complete valid JSON object."
    )

    for round_index in range(max_rounds):
        resp_json = _request_chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            force_json=force_json and round_index == 0,
        )
        reply_text, finish_reason = _extract_chat_message_and_reason(resp_json)
        if not reply_text:
            raise ValueError("模型返回为空。")
        collected_parts.append(reply_text)
        messages.append({"role": "assistant", "content": reply_text})

        if finish_reason not in ("length", "max_tokens"):
            return "".join(collected_parts).strip()

        logging.getLogger("whatai_logger").info(
            f"检测到模型返回被截断，finish_reason={finish_reason}，准备继续请求剩余内容..."
        )
        messages.append({"role": "user", "content": continue_prompt})

    raise ValueError("模型多次续写后仍未完成，请减少页数或缩短单页内容要求。")


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

        try:
            return _parse_json_reply(reply_text)
        except Exception as e:
            self.log_signal.emit("❌ 【错误 - JSON 解析失败】模型返回了不规范的 JSON 格式。")
            self.log_signal.emit(f"尝试解析的文本: \n{_strip_json_fence(reply_text)}")
            self.log_signal.emit(f"具体异常: {e}")
            return None

    def _build_sd_payload(self, prompt_text, negative_prompt_text, width, height):
        current_group_name = self.config.get("current_sd_group", "Default")
        sd_settings = self.config.get("sd_config_groups", {}).get(current_group_name, {})
        style_prompt = self.config.get("last_used_style", "").strip()
        fixed_prompt = self.config.get("fixed_prompt", "").strip()
        final_prompt = ", ".join([p for p in [fixed_prompt, str(prompt_text or "").strip(), style_prompt] if p])
        fixed_neg_prompt = self.config.get("fixed_negative_prompt", "").strip()
        final_neg_prompt = ", ".join([p for p in [str(negative_prompt_text or "").strip(), fixed_neg_prompt] if p])

        payload = {
            "prompt": final_prompt,
            "negative_prompt": final_neg_prompt,
            "width": _coerce_positive_int(width, 512),
            "height": _coerce_positive_int(height, 512),
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
        return payload

    def _save_generated_images(self, images_base64, output_subdir="sdmake", prefix="gen"):
        saved_files = []
        for idx, img_b64 in enumerate(images_base64):
            img_data = base64.b64decode(img_b64)
            timestamp = int(time.time())
            date_str = time.strftime("%Y%m%d")
            output_dir = os.path.join(BASE_DIR, "data", date_str, output_subdir)
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{prefix}_{timestamp}_{idx}.png")
            with open(filename, "wb") as img_file:
                img_file.write(img_data)
            saved_files.append(filename)
            self.log_signal.emit(f"图片已保存至: {filename}")
        return saved_files

    def render_prompt_item(self, prompt_text, width, height, negative_prompt_text=None, output_subdir="sdmake", prefix="gen"):
        url = f"{self.config['sd_url'].rstrip('/')}/sdapi/v1/txt2img"
        payload = self._build_sd_payload(prompt_text, negative_prompt_text, width, height)

        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            images_base64 = response.json().get("images", [])
            self._save_generated_images(images_base64, output_subdir=output_subdir, prefix=prefix)
        except Exception as e:
            self.log_signal.emit(f"SD WebUI API 错误: {e}")

    def generate_sd_image(self, llm_data):
        self.render_prompt_item(
            prompt_text=llm_data.get("prompt", "").strip(),
            width=llm_data.get("width", 512),
            height=llm_data.get("height", 512),
            negative_prompt_text=self.neg_template_text.strip(),
            output_subdir="sdmake",
            prefix="gen",
        )

    def stop(self):
        self.is_running = False


class SdStorySequenceThread(QThread):
    log_signal = pyqtSignal(str)
    success_signal = pyqtSignal(str, object)
    finished_signal = pyqtSignal()

    def __init__(
        self,
        runtime_config,
        theme_text,
        page_count,
        style_prompt,
        prompt_min_words,
        keyword_count,
        no_appearance_description=False,
        no_outfit_description=False,
    ):
        super().__init__()
        self.runtime_config = dict(runtime_config or {})
        self.theme_text = str(theme_text or "").strip()
        self.page_count = _coerce_positive_int(page_count, 1)
        self.style_prompt = str(style_prompt or "").strip()
        self.prompt_min_words = _coerce_positive_int(prompt_min_words, 250)
        self.keyword_count = _coerce_positive_int(keyword_count, 30)
        self.no_appearance_description = bool(no_appearance_description)
        self.no_outfit_description = bool(no_outfit_description)

    def _load_prompt_file(self, prompt_path, prompt_name):
        if not os.path.isfile(prompt_path):
            raise FileNotFoundError(f"缺少 {prompt_name} 文件：{prompt_path}")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def _build_outline(self):
        system_prompt = self._load_prompt_file(STORY_SYSTEM_PROMPT_FILE, "故事大纲 System Prompt").replace(
            "{page_count}", str(self.page_count)
        )
        user_lines = [
            f"绘画主题: {self.theme_text}",
            f"页数: {self.page_count}",
        ]
        if self.no_appearance_description:
            user_lines.append("额外约束: 生成故事序列时不要描述角色外观设定，不要写发色、发型、脸部特征、体型等角色外观信息。")
        if self.no_outfit_description:
            user_lines.append("额外约束: 生成故事序列时不要描述角色服装设定，不要写衣装、饰品、鞋袜、配件等内容。")
        if self.style_prompt:
            user_lines.append(f"风格参考: {self.style_prompt}")

        self.log_signal.emit("\n>>> 第 1 步：开始生成故事大纲 JSON...")
        self.log_signal.emit(f"已加载故事大纲提示词: {STORY_SYSTEM_PROMPT_FILE}")
        outline_reply = fetch_llm_reply_with_continuation(
            base_url=self.runtime_config["base_url"],
            api_key=self.runtime_config["api_key"],
            model=self.runtime_config["model"],
            system_prompt=system_prompt,
            user_content="\n".join(user_lines),
            temperature=0.7,
            merge_system_prompt=self.runtime_config.get("merge_system_prompt", True),
            force_json=True,
        )
        return normalize_story_outline(
            _parse_json_reply(outline_reply),
            default_theme=self.theme_text,
            expected_pages=self.page_count,
        )

    def _build_page_prompt(self, outline_data, page_outline):
        system_prompt = self._load_prompt_file(STORY_PAGE_SYSTEM_PROMPT_FILE, "故事单页 System Prompt")
        system_prompt = (
            system_prompt.replace("{min_words}", str(self.prompt_min_words))
            .replace("{keyword_count}", str(self.keyword_count))
            .replace(
                "{character_description_rule}",
                (
                    "Do not describe character appearance details such as hair color, hairstyle, facial features, body type, or other identity-design attributes. Keep character references generic so the user can control them with LoRA or fixed character settings."
                    if self.no_appearance_description
                    else "You may describe character appearance when it materially helps the scene."
                ),
            )
            .replace(
                "{outfit_description_rule}",
                (
                    "Do not describe clothing, accessories, shoes, jewelry, or outfit design details."
                    if self.no_outfit_description
                    else "You may describe clothing or accessories when it materially helps the scene."
                ),
            )
        )
        outline_text = json.dumps(outline_data, ensure_ascii=False, indent=2)
        page_text = json.dumps(page_outline, ensure_ascii=False, indent=2)
        user_lines = [
            f"绘画主题: {self.theme_text}",
            f"故事总标题(中): {outline_data.get('title_zh', '')}",
            f"故事总标题(英): {outline_data.get('title_en', '')}",
            "完整故事大纲(JSON):",
            outline_text,
            "当前需要详细生成的页面(JSON):",
            page_text,
        ]
        if self.style_prompt:
            user_lines.extend(["风格参考:", self.style_prompt])

        page_reply = fetch_llm_reply_with_continuation(
            base_url=self.runtime_config["base_url"],
            api_key=self.runtime_config["api_key"],
            model=self.runtime_config["model"],
            system_prompt=system_prompt,
            user_content="\n".join(user_lines),
            temperature=0.7,
            merge_system_prompt=self.runtime_config.get("merge_system_prompt", True),
            force_json=False,
            max_rounds=10,
        )
        page_data = _parse_json_reply(page_reply)
        page_data["page"] = page_outline.get("page")
        page_data["title_zh"] = str(page_data.get("title_zh") or page_outline.get("title_zh") or "").strip()
        page_data["title_en"] = str(page_data.get("title_en") or page_outline.get("title_en") or "").strip()
        page_data["width"], page_data["height"] = _normalize_story_resolution(
            page_data.get("width"), page_data.get("height")
        )
        return page_data

    def run(self):
        whatai_logger = logging.getLogger("whatai_logger")
        gui_handler = GuiLogHandler(self.log_signal)
        whatai_logger.addHandler(gui_handler)
        try:
            outline_data = self._build_outline()
            self.log_signal.emit(
                f"故事大纲生成完成：{outline_data.get('title_zh') or outline_data.get('title_en') or self.theme_text}"
            )
            final_story = {
                "theme": outline_data.get("theme", self.theme_text),
                "title_zh": outline_data.get("title_zh", ""),
                "title_en": outline_data.get("title_en", ""),
                "pages": [],
            }
            total_pages = len(outline_data.get("pages", []))
            for page_outline in outline_data.get("pages", []):
                page_no = page_outline.get("page", len(final_story["pages"]) + 1)
                self.log_signal.emit(
                    f"\n>>> 第 2 步：正在生成故事页 {page_no}/{total_pages} 的长篇自然语言 prompt..."
                )
                page_data = self._build_page_prompt(outline_data, page_outline)
                final_story["pages"].append(page_data)
                self.log_signal.emit(
                    f"第 {page_no} 页生成完成，英文 prompt 长度: {len(page_data.get('prompt_en', '').split())} words"
                )

            normalized = normalize_story_sequence(
                final_story,
                default_theme=self.theme_text,
                expected_pages=self.page_count,
            )
            target_path, normalized = save_story_sequence(normalized)
            self.log_signal.emit(f"故事序列已保存至: {target_path}")
            self.success_signal.emit(target_path, normalized)
        except Exception as e:
            self.log_signal.emit(f"❌ 故事序列生成失败: {e}")
        finally:
            whatai_logger.removeHandler(gui_handler)
            self.finished_signal.emit()


class SdStoryRenderThread(SdWorkflowThread):
    def __init__(self, config, story_sequence, neg_template_text, story_path=""):
        super().__init__(config, "", "", neg_template_text)
        self.story_sequence = normalize_story_sequence(story_sequence)
        self.story_path = str(story_path or "").strip()

    def run(self):
        whatai_logger = logging.getLogger("whatai_logger")
        gui_handler = GuiLogHandler(self.log_signal)
        whatai_logger.addHandler(gui_handler)
        try:
            pages = list(self.story_sequence.get("pages", []))
            total_pages = len(pages)
            theme = self.story_sequence.get("theme", "") or "未命名主题"
            self.log_signal.emit(f"=== 开始故事序列顺序生成，共 {total_pages} 页 | 主题: {theme} ===")
            if self.story_path:
                self.log_signal.emit(f"当前故事文件: {self.story_path}")

            for index, page in enumerate(pages, start=1):
                if not self.is_running:
                    self.log_signal.emit("任务已被用户中止。")
                    break

                prompt_en = str(page.get("prompt_en", "") or "").strip()
                prompt_zh = str(page.get("prompt_zh", "") or "").strip()
                width = page.get("width", 1024)
                height = page.get("height", 1536)
                page_no = page.get("page", index)
                page_title = str(page.get("title_zh") or page.get("title_en") or "").strip()

                self.log_signal.emit(
                    f"\n--- [故事页 {index}/{total_pages}] 第 {page_no} 页 {page_title} ---"
                )
                self.log_signal.emit(f"中文翻译: {prompt_zh[:120]}")
                self.log_signal.emit(f"英文 Prompt: {prompt_en[:120]}")
                self.log_signal.emit(f"计划尺寸: {width}x{height}")
                self.render_prompt_item(
                    prompt_text=prompt_en,
                    width=width,
                    height=height,
                    negative_prompt_text=self.neg_template_text.strip(),
                    output_subdir="sd-story",
                    prefix=f"page{page_no}",
                )
                if self.is_running:
                    time.sleep(1)

            if self.is_running:
                self.log_signal.emit("\n✅ 故事序列全部生成完毕。")
        except Exception as e:
            self.log_signal.emit(f"故事序列生成线程异常: {e}")
        finally:
            whatai_logger.removeHandler(gui_handler)
            self.finished_signal.emit()


class StorySequenceEditorDialog(QDialog):
    saved_signal = pyqtSignal(str, object)

    def __init__(self, story_path, parent=None):
        super().__init__(parent)
        self.story_path = os.path.abspath(story_path)
        self._story_data = {"theme": "", "title_zh": "", "title_en": "", "pages": []}
        self.setWindowTitle("故事序列编辑器")
        self.resize(1180, 760)
        self.init_ui()
        self.load_from_disk()

    def init_ui(self):
        layout = QVBoxLayout(self)
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("当前文件:"))
        self.path_label = QLineEdit(self.story_path)
        self.path_label.setReadOnly(True)
        path_layout.addWidget(self.path_label)
        layout.addLayout(path_layout)

        tip_label = QLabel("表格里直接编辑每页标题、中文/英文 prompt 与尺寸。保存时会自动整理为标准 JSON。")
        tip_label.setWordWrap(True)
        layout.addWidget(tip_label)

        meta_form = QFormLayout()
        self.theme_input = QLineEdit()
        meta_form.addRow("主题:", self.theme_input)
        self.title_zh_input = QLineEdit()
        meta_form.addRow("故事标题(中):", self.title_zh_input)
        self.title_en_input = QLineEdit()
        meta_form.addRow("Story Title(EN):", self.title_en_input)
        layout.addLayout(meta_form)

        table_btn_layout = QHBoxLayout()
        self.add_row_btn = QPushButton("新增页")
        self.add_row_btn.clicked.connect(self.add_page_row)
        table_btn_layout.addWidget(self.add_row_btn)
        self.del_row_btn = QPushButton("删除选中页")
        self.del_row_btn.clicked.connect(self.delete_selected_page_row)
        table_btn_layout.addWidget(self.del_row_btn)
        self.renumber_btn = QPushButton("页码重排")
        self.renumber_btn.clicked.connect(self.renumber_pages)
        table_btn_layout.addWidget(self.renumber_btn)
        table_btn_layout.addStretch()
        layout.addLayout(table_btn_layout)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["页码", "标题(中)", "Title(EN)", "中文 Prompt", "English Prompt", "宽", "高"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.EditKeyPressed)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.table)

        preview_label = QLabel("当前选中页预览")
        layout.addWidget(preview_label)
        preview_layout = QHBoxLayout()
        self.preview_zh = QTextEdit()
        self.preview_zh.setReadOnly(True)
        self.preview_zh.setPlaceholderText("中文 prompt 预览")
        preview_layout.addWidget(self.preview_zh)
        self.preview_en = QTextEdit()
        self.preview_en.setReadOnly(True)
        self.preview_en.setPlaceholderText("English prompt preview")
        preview_layout.addWidget(self.preview_en)
        layout.addLayout(preview_layout)
        self.table.itemSelectionChanged.connect(self.update_preview_from_selection)
        self.table.itemChanged.connect(lambda _item: self.update_preview_from_selection())

        btn_layout = QHBoxLayout()
        self.reload_btn = QPushButton("重新载入")
        self.reload_btn.clicked.connect(self.load_from_disk)
        btn_layout.addWidget(self.reload_btn)
        self.save_btn = QPushButton("保存到本地")
        self.save_btn.clicked.connect(lambda: self.save_to_disk(show_message=True))
        btn_layout.addWidget(self.save_btn)
        self.start_btn = QPushButton("保存并启动顺序生成")
        btn_layout.addWidget(self.start_btn)
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)
        layout.addLayout(btn_layout)

    def load_from_disk(self):
        if not os.path.isfile(self.story_path):
            self._story_data = {"theme": "", "title_zh": "", "title_en": "", "pages": []}
            self.load_to_ui(self._story_data)
            return
        self._story_data = load_story_sequence(self.story_path)
        self.load_to_ui(self._story_data)

    def load_to_ui(self, story_data):
        normalized = normalize_story_sequence(story_data)
        self.theme_input.setText(normalized.get("theme", ""))
        self.title_zh_input.setText(normalized.get("title_zh", ""))
        self.title_en_input.setText(normalized.get("title_en", ""))
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for page in normalized.get("pages", []):
            self.add_page_row(page, emit_signals=False)
        self.table.blockSignals(False)
        if self.table.rowCount() > 0:
            self.table.selectRow(0)
        self.update_preview_from_selection()

    def add_page_row(self, page_data=None, emit_signals=True):
        page_data = dict(page_data or {})
        row = self.table.rowCount()
        self.table.insertRow(row)
        values = [
            str(page_data.get("page", row + 1)),
            str(page_data.get("title_zh", "") or ""),
            str(page_data.get("title_en", "") or ""),
            str(page_data.get("prompt_zh", "") or ""),
            str(page_data.get("prompt_en", "") or ""),
            str(page_data.get("width", 1024)),
            str(page_data.get("height", 1536)),
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            self.table.setItem(row, col, item)
        if emit_signals:
            self.update_preview_from_selection()

    def delete_selected_page_row(self):
        row = self.table.currentRow()
        if row < 0:
            return
        self.table.removeRow(row)
        self.renumber_pages()
        if self.table.rowCount() > 0:
            self.table.selectRow(min(row, self.table.rowCount() - 1))
        self.update_preview_from_selection()

    def renumber_pages(self):
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is None:
                item = QTableWidgetItem()
                self.table.setItem(row, 0, item)
            item.setText(str(row + 1))
        self.table.blockSignals(False)

    def update_preview_from_selection(self):
        row = self.table.currentRow()
        if row < 0:
            self.preview_zh.clear()
            self.preview_en.clear()
            return
        zh_item = self.table.item(row, 3)
        en_item = self.table.item(row, 4)
        self.preview_zh.setPlainText(zh_item.text() if zh_item else "")
        self.preview_en.setPlainText(en_item.text() if en_item else "")

    def get_normalized_data(self):
        pages = []
        for row in range(self.table.rowCount()):
            pages.append(
                {
                    "page": self.table.item(row, 0).text().strip() if self.table.item(row, 0) else str(row + 1),
                    "title_zh": self.table.item(row, 1).text().strip() if self.table.item(row, 1) else "",
                    "title_en": self.table.item(row, 2).text().strip() if self.table.item(row, 2) else "",
                    "prompt_zh": self.table.item(row, 3).text().strip() if self.table.item(row, 3) else "",
                    "prompt_en": self.table.item(row, 4).text().strip() if self.table.item(row, 4) else "",
                    "width": self.table.item(row, 5).text().strip() if self.table.item(row, 5) else "1024",
                    "height": self.table.item(row, 6).text().strip() if self.table.item(row, 6) else "1536",
                }
            )
        normalized = normalize_story_sequence(
            {
                "theme": self.theme_input.text().strip(),
                "title_zh": self.title_zh_input.text().strip(),
                "title_en": self.title_en_input.text().strip(),
                "pages": pages,
            }
        )
        self._story_data = normalized
        return normalized

    def save_to_disk(self, show_message=False):
        normalized = self.get_normalized_data()
        save_story_sequence(normalized, target_path=self.story_path)
        self.saved_signal.emit(self.story_path, normalized)
        if show_message:
            QMessageBox.information(self, "成功", f"故事序列已保存至:\n{self.story_path}")
        return normalized


class StorySequencePreviewDialog(QDialog):
    def __init__(self, story_path, parent=None):
        super().__init__(parent)
        self.story_path = os.path.abspath(story_path)
        self._story_pages = []
        self.setWindowTitle("故事序列预览")
        self.resize(1100, 760)
        self.init_ui()
        self.load_from_disk()

    def init_ui(self):
        layout = QVBoxLayout(self)
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("当前文件:"))
        self.path_label = QLineEdit(self.story_path)
        self.path_label.setReadOnly(True)
        path_layout.addWidget(self.path_label)
        layout.addLayout(path_layout)

        self.meta_label = QLabel("请选择左侧页码查看详情")
        self.meta_label.setWordWrap(True)
        layout.addWidget(self.meta_label)

        content_layout = QHBoxLayout()
        self.page_list = QListWidget()
        self.page_list.setMinimumWidth(280)
        self.page_list.itemSelectionChanged.connect(self.update_detail_from_selection)
        content_layout.addWidget(self.page_list)

        detail_layout = QVBoxLayout()
        self.zh_preview = QTextEdit()
        self.zh_preview.setReadOnly(True)
        self.zh_preview.setPlaceholderText("中文 prompt 预览")
        detail_layout.addWidget(self.zh_preview)
        self.en_preview = QTextEdit()
        self.en_preview.setReadOnly(True)
        self.en_preview.setPlaceholderText("English prompt preview")
        detail_layout.addWidget(self.en_preview)
        content_layout.addLayout(detail_layout)
        layout.addLayout(content_layout)

        btn_layout = QHBoxLayout()
        self.reload_btn = QPushButton("重新载入")
        self.reload_btn.clicked.connect(self.load_from_disk)
        btn_layout.addWidget(self.reload_btn)
        btn_layout.addStretch()
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.close_btn)
        layout.addLayout(btn_layout)

    def load_story(self, story_path):
        self.story_path = os.path.abspath(story_path)
        self.path_label.setText(self.story_path)
        self.load_from_disk()

    def load_from_disk(self):
        if not os.path.isfile(self.story_path):
            self._story_pages = []
            self.page_list.clear()
            self.meta_label.setText(f"故事文件不存在: {self.story_path}")
            self.zh_preview.clear()
            self.en_preview.clear()
            return
        try:
            story_data = load_story_sequence(self.story_path)
        except Exception as e:
            self._story_pages = []
            self.page_list.clear()
            self.meta_label.setText(f"故事文件读取失败: {e}")
            self.zh_preview.clear()
            self.en_preview.clear()
            return

        self._story_pages = list(story_data.get("pages", []))
        self.page_list.clear()
        for page in self._story_pages:
            title = str(page.get("title_zh") or page.get("title_en") or "").strip()
            label = f"第 {page.get('page', 0)} 页"
            if title:
                label = f"{label} | {title}"
            self.page_list.addItem(label)
        if self.page_list.count() > 0:
            self.page_list.setCurrentRow(0)
        else:
            self.update_detail_from_selection()

    def update_detail_from_selection(self):
        row = self.page_list.currentRow()
        if row < 0 or row >= len(self._story_pages):
            self.meta_label.setText("请选择左侧页码查看详情")
            self.zh_preview.clear()
            self.en_preview.clear()
            return
        page = self._story_pages[row]
        self.meta_label.setText(
            f"第 {page.get('page', row + 1)} 页 | 标题: {page.get('title_zh') or page.get('title_en') or '未命名'} | 尺寸: {page.get('width', 1024)}x{page.get('height', 1536)}"
        )
        self.zh_preview.setPlainText(str(page.get("prompt_zh", "") or ""))
        self.en_preview.setPlainText(str(page.get("prompt_en", "") or ""))


class WebuiExtraPayloadEditorDialog(QDialog):
    saved_signal = pyqtSignal(str, object)

    def __init__(self, payload_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("WebUI 附加 Payload 编辑器")
        self.resize(980, 760)
        self._editing_row = -1
        self._payload_dict = load_webui_extra_payload(payload_text)
        self.init_ui()
        self.load_payload_to_ui(self._payload_dict)

    def init_ui(self):
        layout = QVBoxLayout(self)
        tip_label = QLabel(
            "使用 key-value 表格编辑附加 Payload。普通字符串可直接填写；dict/list 请在下方“值详情”里按 JSON 格式编辑。"
        )
        tip_label.setWordWrap(True)
        layout.addWidget(tip_label)

        btn_layout = QHBoxLayout()
        self.add_row_btn = QPushButton("新增字段")
        self.add_row_btn.clicked.connect(self.add_row)
        btn_layout.addWidget(self.add_row_btn)
        self.del_row_btn = QPushButton("删除选中字段")
        self.del_row_btn.clicked.connect(self.delete_selected_row)
        btn_layout.addWidget(self.del_row_btn)
        self.format_btn = QPushButton("按 JSON 重整")
        self.format_btn.clicked.connect(self.reformat_selected_value)
        btn_layout.addWidget(self.format_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Key", "Value 预览"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self.on_selection_changed)
        layout.addWidget(self.table)

        detail_label = QLabel("值详情")
        layout.addWidget(detail_label)
        self.value_editor = QTextEdit()
        self.value_editor.setPlaceholderText("可输入普通字符串，或输入 dict/list 的 JSON 内容。")
        layout.addWidget(self.value_editor)

        detail_btn_layout = QHBoxLayout()
        self.apply_value_btn = QPushButton("应用当前值到选中字段")
        self.apply_value_btn.clicked.connect(self.apply_value_to_selected_row)
        detail_btn_layout.addWidget(self.apply_value_btn)
        detail_btn_layout.addStretch()
        layout.addLayout(detail_btn_layout)

        save_btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("保存到配置")
        self.save_btn.clicked.connect(lambda: self.save_and_close(close_after=False))
        save_btn_layout.addWidget(self.save_btn)
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.close)
        save_btn_layout.addWidget(self.close_btn)
        layout.addLayout(save_btn_layout)

    def _set_row_value_text(self, row, value_text):
        item = self.table.item(row, 1)
        if item is None:
            item = QTableWidgetItem()
            self.table.setItem(row, 1, item)
        full_text = str(value_text or "")
        item.setData(Qt.ItemDataRole.UserRole, full_text)
        item.setText(_payload_value_preview(full_text))

    def _get_row_value_text(self, row):
        item = self.table.item(row, 1)
        if item is None:
            return ""
        stored = item.data(Qt.ItemDataRole.UserRole)
        if stored is None:
            return item.text()
        return str(stored)

    def _persist_value_editor(self):
        if self._editing_row < 0 or self._editing_row >= self.table.rowCount():
            return
        self._set_row_value_text(self._editing_row, self.value_editor.toPlainText())

    def load_payload_to_ui(self, payload_dict):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for key, value in payload_dict.items():
            self.add_row(key, _format_payload_value_text(value), select_row=False)
        self.table.blockSignals(False)
        if self.table.rowCount() > 0:
            self.table.selectRow(0)
        else:
            self._editing_row = -1
            self.value_editor.clear()

    def add_row(self, key_text="", value_text="", select_row=True):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(key_text or "")))
        self._set_row_value_text(row, value_text)
        if select_row:
            self.table.selectRow(row)

    def delete_selected_row(self):
        row = self.table.currentRow()
        if row < 0:
            return
        self.table.removeRow(row)
        self._editing_row = -1
        if self.table.rowCount() > 0:
            self.table.selectRow(min(row, self.table.rowCount() - 1))
        else:
            self.value_editor.clear()

    def on_selection_changed(self):
        self._persist_value_editor()
        row = self.table.currentRow()
        self._editing_row = row
        if row < 0:
            self.value_editor.clear()
            return
        self.value_editor.setPlainText(self._get_row_value_text(row))

    def apply_value_to_selected_row(self):
        row = self.table.currentRow()
        if row < 0:
            return
        self._set_row_value_text(row, self.value_editor.toPlainText())

    def reformat_selected_value(self):
        row = self.table.currentRow()
        if row < 0:
            return
        try:
            parsed = _parse_payload_value_text(self.value_editor.toPlainText())
        except ValueError as e:
            QMessageBox.warning(self, "值格式错误", str(e))
            return
        self.value_editor.setPlainText(_format_payload_value_text(parsed))
        self._set_row_value_text(row, self.value_editor.toPlainText())

    def get_payload_dict(self):
        self._persist_value_editor()
        payload = {}
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            key_text = str(key_item.text() if key_item else "").strip()
            if not key_text:
                continue
            if key_text in payload:
                raise ValueError(f"检测到重复字段名: {key_text}")
            payload[key_text] = _parse_payload_value_text(self._get_row_value_text(row))
        return payload

    def save_and_close(self, close_after=False):
        try:
            payload = self.get_payload_dict()
        except Exception as e:
            QMessageBox.warning(self, "Payload 错误", str(e))
            return
        payload_text = dump_webui_extra_payload(payload)
        self.saved_signal.emit(payload_text, payload)
        QMessageBox.information(self, "成功", "WebUI 附加 Payload 已保存。")
        if close_after:
            self.close()
