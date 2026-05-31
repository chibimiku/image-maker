import os
import json
import base64
import io
import datetime
import re
import hashlib
from openai import OpenAI
from PIL import Image, ImageGrab

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QCheckBox,
                             QLabel, QPushButton, QTextEdit, QComboBox, QMessageBox, QDoubleSpinBox, QCompleter,
                             QListWidget, QListWidgetItem, QDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QStringListModel
from PyQt5.QtGui import QPixmap, QImage, QColor, QDesktopServices
from PyQt5.QtCore import QUrl

from modules.others.api_backend import generate_image_whatai, generate_image_aigc2d 
from utils.booru_tags import normalize_booru_tags
from utils.wd14_tagger import predict_local_booru_tags, merge_prompt_with_local_booru_tags
from utils.task_runtime import SystemNotifier, TaskCountdown
from utils.image_upscale_runtime import JpgAutoUpscaleThread, list_esrgan_models, normalize_upscale_options
from utils.prompt_loader import read_prompt_file, render_prompt_file, find_missing_prompt_files

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYSTEM_PROMPT_FILE = "single-analyzer-system.md"
STYLE_ANALY_PROMPT_FILE = "style-analy.md"
REFINE_DESC_PROMPT_FILE = "refine-desc.md"
OUTFIT_CHECK_PROMPT_FILE = "single-analyzer-outfit-check.md"


def get_single_analyzer_required_prompt_files(enable_refine=True, enable_outfit_check=False):
    files = [
        SYSTEM_PROMPT_FILE,
        STYLE_ANALY_PROMPT_FILE,
    ]
    if enable_refine:
        files.append(REFINE_DESC_PROMPT_FILE)
    if enable_outfit_check:
        files.append(OUTFIT_CHECK_PROMPT_FILE)
    return files


def get_single_analyzer_missing_prompt_files(enable_refine=True, enable_outfit_check=False):
    return find_missing_prompt_files(
        get_single_analyzer_required_prompt_files(
            enable_refine=enable_refine,
            enable_outfit_check=enable_outfit_check,
        )
    )


def _load_system_prompt():
    return read_prompt_file(SYSTEM_PROMPT_FILE).strip()

def get_style_analyze_prompt(booru_tag_limit):
    limit = int(booru_tag_limit) if str(booru_tag_limit).strip().isdigit() else 30
    if limit <= 0:
        limit = 30
    return render_prompt_file(STYLE_ANALY_PROMPT_FILE, {"booru_tag_limit": str(limit)}).strip()

def append_extra_llm_prompt(base_prompt, extra_llm_prompt):
    extra_text = str(extra_llm_prompt or "").strip()
    if not extra_text:
        return base_prompt
    return f"{base_prompt.rstrip()}\n\n{extra_text}"

def _is_timeout_error(err) -> bool:
    text = str(err).lower()
    return ("timeout" in text) or ("timed out" in text) or ("readtimeout" in text)

def _clone_image_source(image_source):
    if isinstance(image_source, str):
        return os.path.abspath(image_source)
    if isinstance(image_source, Image.Image):
        try:
            return image_source.copy()
        except Exception:
            return image_source
    return image_source

def _describe_image_source(image_source):
    if isinstance(image_source, str):
        return os.path.abspath(image_source)
    if isinstance(image_source, Image.Image):
        return "剪贴板图片"
    return "未知图片源"

def _generate_task_hash(image_source, submit_time, thread_no):
    seed = f"{_describe_image_source(image_source)}|{submit_time.isoformat()}|{thread_no}"
    return hashlib.md5(seed.encode("utf-8")).hexdigest()[:8]

def _to_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default

def _normalize_analysis_result(result_json, fallback_data=None, booru_tag_limit=30):
    if not isinstance(result_json, dict):
        return {}
    fallback_data = fallback_data if isinstance(fallback_data, dict) else {}
    normalized = dict(result_json)
    english_description = (
        result_json.get("english_description")
        or fallback_data.get("english_description")
        or ""
    )
    japanese_title = (
        result_json.get("japanese_title")
        or fallback_data.get("japanese_title")
        or ""
    )
    chinese_title = (
        result_json.get("chinese_title")
        or fallback_data.get("chinese_title")
        or ""
    )
    pixiv_tags = result_json.get("pixiv_tags")
    if pixiv_tags is None:
        pixiv_tags = fallback_data.get("pixiv_tags", [])
    if isinstance(pixiv_tags, str):
        pixiv_tags = [tag.strip() for tag in pixiv_tags.split(",") if tag.strip()]
    if not isinstance(pixiv_tags, list):
        pixiv_tags = []
    short_description = (
        result_json.get("short_description")
        or result_json.get("shortDescription")
        or fallback_data.get("short_description")
        or ""
    )
    booru_tags = (
        result_json.get("booru-tags")
        or result_json.get("booru_tags")
        or result_json.get("booruTags")
        or result_json.get("booru_tag")
        or result_json.get("booruTag")
        or fallback_data.get("booru-tags")
        or []
    )
    limit = int(booru_tag_limit) if str(booru_tag_limit).strip().isdigit() else 30
    if limit <= 0:
        limit = 30
    normalized["english_description"] = str(english_description).strip()
    normalized["japanese_title"] = str(japanese_title).strip()
    normalized["chinese_title"] = str(chinese_title).strip()
    normalized["pixiv_tags"] = [str(tag).strip() for tag in pixiv_tags if str(tag).strip()]
    normalized["short_description"] = str(short_description).strip()
    normalized["booru-tags"] = normalize_booru_tags(booru_tags, limit=limit)
    return normalized

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

def _looks_like_base64_text(value: str) -> bool:
    if not isinstance(value, str):
        return False
    compact = value.strip().replace("\n", "").replace("\r", "")
    if len(compact) < 256:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9+/=]+", compact))

def _sanitize_ui_log_data(value):
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            key_lower = str(key).lower()
            if isinstance(item, str) and ("base64" in key_lower or key_lower.startswith("b64") or key_lower == "data"):
                if _looks_like_base64_text(item) or key_lower != "data":
                    sanitized[key] = "<BASE64_IMAGE_DATA_OMITTED>"
                    continue
            sanitized[key] = _sanitize_ui_log_data(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_ui_log_data(item) for item in value]
    if isinstance(value, str):
        text = re.sub(
            r"(data:image\/[a-zA-Z0-9.+-]+;base64,)[A-Za-z0-9+/=\r\n]+",
            r"\1<BASE64_IMAGE_DATA_OMITTED>",
            value
        )
        if _looks_like_base64_text(text):
            return "<BASE64_IMAGE_DATA_OMITTED>"
        if len(text) > 4000:
            return f"{text[:4000]}...(TRUNCATED, total={len(text)})"
        return text
    return value

def _format_ui_log_json(value) -> str:
    try:
        return json.dumps(_sanitize_ui_log_data(value), ensure_ascii=False, indent=2)
    except Exception:
        return str(value)

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

def step_1_analyze_image(image_source, client, model_name, log_callback=None, booru_tag_limit=30, local_booru_tags=None, extra_llm_prompt="", timeout_seconds=120, status_callback=None):
    mime_type, base64_image = compress_and_encode_image(image_source, log_callback=log_callback)
    if not base64_image:
        if log_callback:
            log_callback("Step 1 失败: 图片压缩或编码失败")
        return None
    try:
        analyze_prompt = get_style_analyze_prompt(booru_tag_limit)
        analyze_prompt = merge_prompt_with_local_booru_tags(analyze_prompt, local_booru_tags)
        analyze_prompt = append_extra_llm_prompt(analyze_prompt, extra_llm_prompt)
        system_prompt = _load_system_prompt()
        response = client.chat.completions.create(
            model=model_name, 
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": analyze_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_image}", "detail": "high"}
                        }
                    ]
                }
            ],
            temperature=0.7, max_completion_tokens=16384, timeout=timeout_seconds
        )
        parsed = json.loads(response.choices[0].message.content)
        fallback_data = {"booru-tags": normalize_booru_tags(local_booru_tags or [], limit=booru_tag_limit)}
        normalized = _normalize_analysis_result(parsed, fallback_data=fallback_data, booru_tag_limit=booru_tag_limit)
        if local_booru_tags:
            normalized["booru_tags_local_candidate"] = normalize_booru_tags(local_booru_tags, limit=booru_tag_limit, output_style="space")
        return normalized
    except Exception as e:
        error_msg = f"Step 1 请求错误: {e}"
        if log_callback:
            log_callback(error_msg)
        if status_callback:
            status_callback("timeout" if _is_timeout_error(e) else "error")
        print(error_msg)
        return None

def step_2_refine_description(original_json_data, client, model_name, booru_tag_limit=30, extra_llm_prompt="", timeout_seconds=120, status_callback=None):
    original_description = original_json_data.get("english_description", "")
    jp_title = original_json_data.get("japanese_title", "")
    cn_title = original_json_data.get("chinese_title", "")
    tags = original_json_data.get("pixiv_tags", [])
    booru_seed_tags = original_json_data.get("booru-tags", [])
    
    tags_str = json.dumps(tags, ensure_ascii=False)
    
    # 构建模板文件路径并读取
    refine_prompt = render_prompt_file(
        REFINE_DESC_PROMPT_FILE,
        {
            "jp_title": jp_title,
            "cn_title": cn_title,
            "original_description": original_description,
            "tags_str": tags_str,
            "booru_tag_limit": str(int(booru_tag_limit) if str(booru_tag_limit).strip().isdigit() else 30),
        }
    ).strip()
    seed_text = ", ".join(normalize_booru_tags(booru_seed_tags, limit=booru_tag_limit))
    if seed_text:
        refine_prompt = (
            f"{refine_prompt}\n\n"
            "booru-tags seed from local model and step1:\n"
            f"{seed_text}\n"
            "Please optimize these booru-tags with your own understanding and keep only final high-quality tags."
        )
    refine_prompt = append_extra_llm_prompt(refine_prompt, extra_llm_prompt)
    
    try:
        system_prompt = _load_system_prompt()
        response = client.chat.completions.create(
            model=model_name, 
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": refine_prompt}
            ],
            temperature=0.7, max_completion_tokens=16384, timeout=timeout_seconds
        )
        final_result_json = json.loads(response.choices[0].message.content)
        final_result_json = _normalize_analysis_result(final_result_json, fallback_data=original_json_data, booru_tag_limit=booru_tag_limit)
        # 将原始描述也存入最终结果，方便后续对比或同时生成
        final_result_json["original_english_description"] = original_description
        return final_result_json
    except Exception as e:
        print(f"Step 2 二次加工时发生错误: {e}")
        if status_callback:
            status_callback("timeout" if _is_timeout_error(e) else "error")
        return None

def step_3_check_outfit_consistency(final_json_data, client, model_name, timeout_seconds=120, outfit_style_override="", status_callback=None):
    fallback_data = dict(final_json_data or {})
    refine_prompt = str(fallback_data.get("english_description") or "").strip()
    original_prompt = str(fallback_data.get("original_english_description") or "").strip()
    outfit_style_override = str(outfit_style_override or "").strip()
    if not refine_prompt and not original_prompt:
        return fallback_data

    prompt_payload = {
        "english_description": refine_prompt,
        "original_english_description": original_prompt,
    }
    user_prompt = render_prompt_file(
        OUTFIT_CHECK_PROMPT_FILE,
        {
            "input_json": json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            "outfit_style_override": outfit_style_override,
        }
    ).strip()

    try:
        system_prompt = _load_system_prompt()
        response = client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_completion_tokens=8192,
            timeout=timeout_seconds
        )
        parsed = json.loads(response.choices[0].message.content)
        checked_refine = str(parsed.get("english_description") or refine_prompt).strip()
        checked_original = str(parsed.get("original_english_description") or original_prompt).strip()
        has_person = _to_bool(parsed.get("has_person"), default=bool(refine_prompt or original_prompt))
        modified = _to_bool(parsed.get("modified"), default=False)
        if checked_refine == refine_prompt and checked_original == original_prompt:
            modified = False

        result = dict(fallback_data)
        result["outfit_check_has_person"] = has_person
        result["outfit_check_modified"] = modified
        result["outfit_check_reason"] = str(parsed.get("reason") or "").strip()

        if modified:
            result["english_description_before_outfit_check"] = refine_prompt
            result["original_english_description_before_outfit_check"] = original_prompt
            result["english_description"] = checked_refine
            result["original_english_description"] = checked_original
        return result
    except Exception as e:
        print(f"Step 3 服装搭配检查时发生错误: {e}")
        if status_callback:
            status_callback("timeout" if _is_timeout_error(e) else "error")
        return None

class WorkerThread(QThread):
    log_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(dict)

    def __init__(self, image_source, api_key, base_url, model_name, enable_refine=True, booru_tag_limit=30, extra_llm_prompt="", timeout_seconds=120, enable_outfit_check=False, outfit_style_override=""):
        super().__init__()
        self.image_source = image_source
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.enable_refine = bool(enable_refine)
        self.booru_tag_limit = int(booru_tag_limit) if str(booru_tag_limit).strip().isdigit() else 30
        if self.booru_tag_limit <= 0:
            self.booru_tag_limit = 30
        self.extra_llm_prompt = str(extra_llm_prompt or "").strip()
        self.timeout_seconds = int(timeout_seconds) if str(timeout_seconds).strip().isdigit() else 120
        if self.timeout_seconds <= 0:
            self.timeout_seconds = 120
        self.enable_outfit_check = bool(enable_outfit_check)
        self.outfit_style_override = str(outfit_style_override or "").strip()
        self.last_status = "idle"
        self._force_cancel_requested = False

    def request_cancel(self, force=False):
        self._force_cancel_requested = bool(force)
        self.requestInterruption()

    def run(self):
        self.last_status = "running"
        if self.isInterruptionRequested():
            self.last_status = "cancelled"
            self.log_signal.emit("任务在开始前已被取消。")
            self.finish_signal.emit({})
            return
        try:
            client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_seconds)
        except Exception as e:
            self.last_status = "error"
            self.log_signal.emit(f"初始化 API 客户端失败: {e}")
            self.finish_signal.emit({})
            return

        self.log_signal.emit(f"请求超时设置: {self.timeout_seconds} 秒")
        self.log_signal.emit(f"正在执行本地 booru tagger 分析（tag 上限: {self.booru_tag_limit}）...")
        local_booru_tags = predict_local_booru_tags(self.image_source, booru_tag_limit=self.booru_tag_limit, log_callback=self.log_signal.emit)
        if self.isInterruptionRequested():
            self.last_status = "cancelled"
            self.log_signal.emit("任务已取消（已停止后续分析步骤）。")
            self.finish_signal.emit({})
            return
        if local_booru_tags:
            preview_tags = ", ".join(local_booru_tags[:10])
            self.log_signal.emit(f"本地 booru tagger 候选标签预览（前 10 个）: {preview_tags}")
        else:
            self.log_signal.emit("本地 booru tagger 候选标签为空，将仅使用大模型继续分析")
        if self.extra_llm_prompt:
            self.log_signal.emit("已启用附加 prompts，LLM 请求将追加重点关注要求")
        self.log_signal.emit(f"正在使用模型 [{self.model_name}] 开始 Step 1: 读取并压缩图片，发送 Vision 请求...")
        stage_status = {"value": "ok"}
        initial_result = step_1_analyze_image(
            self.image_source,
            client,
            self.model_name,
            log_callback=self.log_signal.emit,
            booru_tag_limit=self.booru_tag_limit,
            local_booru_tags=local_booru_tags,
            extra_llm_prompt=self.extra_llm_prompt,
            timeout_seconds=self.timeout_seconds,
            status_callback=lambda status: stage_status.update({"value": status})
        )
        
        if initial_result:
            self.log_signal.emit("Step 1 完成。初步结果已获取。")
            if self.enable_refine:
                if self.isInterruptionRequested():
                    self.last_status = "cancelled"
                    self.log_signal.emit("任务已取消（Step 2 未执行）。")
                    self.finish_signal.emit({})
                    return
                self.log_signal.emit("正在开始 Step 2: 根据中文指令对英文描述进行加工并推断长宽比...")
                stage_status["value"] = "ok"
                final_result = step_2_refine_description(
                    initial_result,
                    client,
                    self.model_name,
                    booru_tag_limit=self.booru_tag_limit,
                    extra_llm_prompt=self.extra_llm_prompt,
                    timeout_seconds=self.timeout_seconds,
                    status_callback=lambda status: stage_status.update({"value": status})
                )
                if final_result:
                    final_result["aspect_ratio"] = calculate_closest_aspect_ratio(self.image_source)
                    initial_tags = initial_result.get("pixiv_tags", [])
                    refined_tags = final_result.get("pixiv_tags", [])
                    final_result["pixiv_tags_first"] = initial_tags
                    final_result["pixiv_tags_second"] = refined_tags
                    if initial_tags and refined_tags:
                        initial_tags_lower = [tag.lower() for tag in initial_tags]
                        intersection_tags = []
                        for tag in refined_tags:
                            if tag.lower() in initial_tags_lower:
                                intersection_tags.append(tag)
                        final_result["pixiv_tags"] = intersection_tags if intersection_tags else refined_tags
                    if local_booru_tags:
                        final_result["booru_tags_local_candidate"] = normalize_booru_tags(local_booru_tags, limit=self.booru_tag_limit, output_style="space")
                else:
                    self.last_status = "timeout" if stage_status.get("value") == "timeout" else "error"
            else:
                self.log_signal.emit("已跳过 Step 2 refine。")
                final_result = dict(initial_result)
                final_result["original_english_description"] = initial_result.get("english_description", "")
                final_result["english_description"] = ""
                final_result["aspect_ratio"] = calculate_closest_aspect_ratio(self.image_source)
                final_result["pixiv_tags_first"] = initial_result.get("pixiv_tags", [])
                final_result["pixiv_tags_second"] = []
                if local_booru_tags:
                    final_result["booru_tags_local_candidate"] = normalize_booru_tags(local_booru_tags, limit=self.booru_tag_limit, output_style="space")
            if final_result and self.enable_outfit_check:
                if self.isInterruptionRequested():
                    self.last_status = "cancelled"
                    self.log_signal.emit("任务已取消（服装搭配检查未执行）。")
                    self.finish_signal.emit({})
                    return
                if self.outfit_style_override:
                    self.log_signal.emit(f"Step 3 已启用服装风格覆盖目标: {self.outfit_style_override}")
                self.log_signal.emit("正在开始 Step 3: 检查人物服装搭配是否协调，并按需修订 prompts...")
                outfit_stage_status = {"value": "ok"}
                outfit_checked_result = step_3_check_outfit_consistency(
                    final_result,
                    client,
                    self.model_name,
                    timeout_seconds=self.timeout_seconds,
                    outfit_style_override=self.outfit_style_override,
                    status_callback=lambda status: outfit_stage_status.update({"value": status})
                )
                if outfit_checked_result:
                    if outfit_checked_result.get("outfit_check_modified"):
                        self.log_signal.emit("Step 3 完成。已检测到人物并修订 prompts 中的服装搭配。")
                    else:
                        if outfit_checked_result.get("outfit_check_has_person"):
                            self.log_signal.emit("Step 3 完成。存在人物，但当前服装搭配无需修订。")
                        else:
                            self.log_signal.emit("Step 3 完成。未检测到人物，prompts 保持不变。")
                    final_result = outfit_checked_result
                else:
                    self.log_signal.emit("Step 3 执行失败，已保留 Step 2 的 prompts 结果。")
            if final_result:
                self.last_status = "success"
            self.finish_signal.emit(final_result if final_result else {})
        else:
            if self.isInterruptionRequested() or self._force_cancel_requested:
                self.last_status = "cancelled"
            else:
                self.last_status = "timeout" if stage_status.get("value") == "timeout" else "error"
            self.log_signal.emit("Step 1 失败，流程终止。")
            self.finish_signal.emit({})

class ImageGenWorkerThread(QThread):
    log_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(list)

    def __init__(self, prompt, model_name, aspect_ratio, instructions, api_type=None, resolution=None, image_paths=None, verbose_debug=False, file_prefix=None):
        super().__init__()
        self.prompt = prompt
        self.model_name = model_name
        self.aspect_ratio = aspect_ratio
        self.instructions = instructions
        self.api_type = api_type
        self.resolution = resolution
        self.image_paths = list(image_paths or [])
        self.verbose_debug = bool(verbose_debug)
        self.file_prefix = str(file_prefix or "").strip()
        self.last_status = "idle"

    def request_cancel(self):
        self.requestInterruption()

    def run(self):
        self.last_status = "running"
        if self.isInterruptionRequested():
            self.last_status = "cancelled"
            self.finish_signal.emit([])
            return
        self.log_signal.emit(f"\n🚀 开始请求生图 API (模型: {self.model_name})...")
        self.log_signal.emit("请耐心等待，这可能需要几十秒的时间...")
        try:
            if self.verbose_debug:
                self.log_signal.emit(
                    "=== 单图调试-完整请求输入 ===\n" + _format_ui_log_json({
                        "api_type": self.api_type,
                        "model": self.model_name,
                        "aspect_ratio": self.aspect_ratio,
                        "resolution": self.resolution,
                        "file_prefix": self.file_prefix,
                        "instructions": self.instructions,
                        "prompt": self.prompt,
                        "image_paths": self.image_paths
                    })
                )
            # 根据api_type调用相应的生成函数
            if self.api_type == "aigc2d":
                result = generate_image_aigc2d(
                    prompt=self.prompt, 
                    image_paths=self.image_paths,
                    model=self.model_name, 
                    aspect_ratio=self.aspect_ratio, 
                    instructions=self.instructions,
                    api_type=self.api_type,
                    resolution=self.resolution,
                    file_prefix=self.file_prefix,
                    return_metadata=self.verbose_debug
                )
            else:
                result = generate_image_whatai(
                    prompt=self.prompt, 
                    image_paths=self.image_paths,
                    model=self.model_name, 
                    aspect_ratio=self.aspect_ratio, 
                    instructions=self.instructions,
                    api_type=self.api_type,
                    resolution=self.resolution,
                    file_prefix=self.file_prefix,
                    return_metadata=self.verbose_debug
                )
            if isinstance(result, dict):
                saved_files = result.get("saved_files", []) or []
                response_payload = {
                    "saved_files": saved_files,
                    "annotation": result.get("annotation", {}),
                    "raw_text": result.get("raw_text", ""),
                }
                server_raw = result.get("server_response_raw")
                if server_raw and isinstance(server_raw, dict) and server_raw:
                    response_payload["server_response_raw"] = server_raw
            else:
                saved_files = result or []
                response_payload = {"saved_files": saved_files}
                server_raw = None

            if self.verbose_debug:
                self.log_signal.emit("=== 单图调试-完整返回 ===\n" + _format_ui_log_json(response_payload))
                if not saved_files and server_raw:
                    self.log_signal.emit("=== 单图调试-服务器原始返回 ===\n" + _format_ui_log_json(server_raw))
                if isinstance(result, dict):
                    replay_script = result.get("replay_script", "")
                    replay_json = result.get("replay_json", "")
                    if replay_script or replay_json:
                        self.log_signal.emit(f"\n📋 请求回放文件（可发送给客服复现）:\n  脚本: {replay_script}\n  数据: {replay_json}")
                if not saved_files:
                    self.log_signal.emit("⚠️ 生图失败：请查看上方的「服务器原始返回」定位问题。")

            if self.isInterruptionRequested():
                self.last_status = "cancelled"
                self.finish_signal.emit([])
                return
            self.last_status = "success" if saved_files else "error"
            self.finish_signal.emit(saved_files)
        except Exception as e:
            self.last_status = "cancelled" if self.isInterruptionRequested() else "error"
            if self.last_status != "cancelled":
                self.log_signal.emit(f"❌ 生图请求发生异常: {e}")
            self.finish_signal.emit([])

class AnalysisHistoryDetailDialog(QDialog):
    def __init__(self, task_record, parent=None):
        super().__init__(parent)
        self.task_record = dict(task_record or {})
        self.setWindowTitle(f"分析任务详情 #{self.task_record.get('thread_no', '?')}")
        self.resize(760, 620)
        layout = QVBoxLayout()

        header = QLabel(self._build_summary_text())
        header.setWordWrap(True)
        layout.addWidget(header)

        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setPlainText(self._build_detail_text())
        layout.addWidget(self.detail_text)

        action_layout = QHBoxLayout()
        action_layout.addStretch()
        self.open_json_btn = QPushButton("打开 JSON 文件")
        json_path = str(self.task_record.get("saved_json_path") or "").strip()
        self.open_json_btn.setEnabled(bool(json_path and os.path.isfile(json_path)))
        self.open_json_btn.clicked.connect(self._open_json_file)
        action_layout.addWidget(self.open_json_btn)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)
        action_layout.addWidget(close_btn)
        layout.addLayout(action_layout)
        self.setLayout(layout)

    def _build_summary_text(self):
        status_text = str(self.task_record.get("status_text") or "未知状态")
        title = str(self.task_record.get("title") or "未命名")
        return f"线程 #{self.task_record.get('thread_no', '?')} | {status_text} | {title}"

    def _build_detail_text(self):
        lines = [
            f"线程号: {self.task_record.get('thread_no', '?')}",
            f"状态: {self.task_record.get('status_text', '未知状态')}",
            f"提交时间: {self.task_record.get('submit_time_text', '-')}",
            f"完成时间: {self.task_record.get('finish_time_text', '-')}",
            f"任务 Hash: {self.task_record.get('task_hash', '-')}",
            f"图片源: {self.task_record.get('source_desc', '-')}",
            f"标题: {self.task_record.get('title', '未命名')}",
            f"JSON 文件: {self.task_record.get('saved_json_path', '-')}",
        ]
        prompt_paths = self.task_record.get("saved_prompt_paths") or []
        if prompt_paths:
            lines.append("Prompt 文件:")
            for path in prompt_paths:
                lines.append(f"- {path}")
        result_json = self.task_record.get("result_json")
        lines.append("")
        if result_json:
            lines.append("结果 JSON:")
            lines.append(_format_ui_log_json(result_json))
        else:
            lines.append("结果 JSON: 当前暂无结果数据。")
        return "\n".join(lines)

    def _open_json_file(self):
        json_path = str(self.task_record.get("saved_json_path") or "").strip()
        if not json_path or not os.path.isfile(json_path):
            QMessageBox.warning(self, "提示", "当前任务还没有可打开的 JSON 文件。")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(json_path))

# --- 单图分析核心界面 Widget ---
class SingleAnalyzerWidget(QWidget):
    def __init__(self, config_getter_func, img_config_getter_func, styles_getter_func, save_img_cfg_callback, ar_policy_getter_func=None, nsfw_default_getter_func=None, nsfw_changed_callback=None, booru_tag_limit_getter_func=None, timeout_getter_func=None, upscale_options_getter_func=None, upscale_options_changed_callback=None, outfit_check_default_getter_func=None, outfit_check_changed_callback=None, outfit_style_history_getter_func=None, outfit_style_default_getter_func=None, outfit_style_changed_callback=None, outfit_style_delete_callback=None):
        super().__init__()
        self.get_text_config = config_getter_func
        self.get_img_config = img_config_getter_func
        self.get_styles = styles_getter_func
        self.save_img_cfg = save_img_cfg_callback
        self.get_nsfw_default = nsfw_default_getter_func
        self.on_nsfw_changed = nsfw_changed_callback
        self.get_booru_tag_limit = booru_tag_limit_getter_func
        self.get_timeout_seconds = timeout_getter_func
        self.get_upscale_options = upscale_options_getter_func
        self.on_upscale_options_changed = upscale_options_changed_callback
        self.get_outfit_check_default = outfit_check_default_getter_func
        self.on_outfit_check_changed = outfit_check_changed_callback
        self.get_outfit_style_history = outfit_style_history_getter_func
        self.get_outfit_style_default = outfit_style_default_getter_func
        self.on_outfit_style_changed = outfit_style_changed_callback
        self.on_outfit_style_deleted = outfit_style_delete_callback
        
        self.image_source = None
        self.current_aspect_ratio = "1:1"
        self.current_orig_desc = ""
        self.current_refine_desc = ""
        self.current_task_hash = ""
        self._active_analysis_threads = []
        self._analysis_thread_seq = 0
        self._image_gen_thread_seq = 0
        self._post_thread_seq = 0
        self._analysis_history = {}

        # 【新增】用来保存正在执行的生图线程池，防止被垃圾回收
        self._active_img_threads = []
        self._auto_gen_groups = {}
        self._img_gen_running = False
        self._img_gen_deadline = None
        self._img_gen_timeout_seconds = 0
        self._img_gen_countdown = TaskCountdown(
            parent=self,
            on_tick=self._on_image_gen_countdown_tick,
            on_timeout=lambda: self.cancel_image_generation(reason="timeout")
        )

        self.get_ar_policy = ar_policy_getter_func
        self._notifier = SystemNotifier(self)
        self._active_post_threads = []
        self._updating_outfit_style_combo = False
        self._outfit_style_history_cache = []
        
        self.initUI()

    def _send_system_notification(self, title, message):
        self._notifier.notify(title, message)
        
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

        nsfw_layout = QHBoxLayout()
        self.use_nsfw_cb = QCheckBox("使用nsfw接口")
        self.use_nsfw_cb.setChecked(bool(self.get_nsfw_default()) if self.get_nsfw_default else False)
        self.use_nsfw_cb.toggled.connect(self.on_use_nsfw_toggled)
        nsfw_layout.addWidget(self.use_nsfw_cb)
        self.enable_outfit_check_cb = QCheckBox("服装搭配检查")
        self.enable_outfit_check_cb.setToolTip("分析出 prompts 后，再额外检查人物服装搭配是否协调；若修订，仅覆盖 prompts 相关字段并保留原值备用。")
        self.enable_outfit_check_cb.setChecked(bool(self.get_outfit_check_default()) if self.get_outfit_check_default else False)
        self.enable_outfit_check_cb.toggled.connect(self.on_enable_outfit_check_toggled)
        nsfw_layout.addWidget(self.enable_outfit_check_cb)
        nsfw_layout.addStretch()
        layout.addLayout(nsfw_layout)

        outfit_style_layout = QHBoxLayout()
        outfit_style_layout.addWidget(QLabel("服装风格覆盖:"))
        self.outfit_style_combo = QComboBox()
        self.outfit_style_combo.setEditable(True)
        self.outfit_style_combo.setInsertPolicy(QComboBox.NoInsert)
        self.outfit_style_completer_model = QStringListModel(self)
        self.outfit_style_completer = QCompleter(self.outfit_style_completer_model, self)
        self.outfit_style_completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.outfit_style_completer.setFilterMode(Qt.MatchContains)
        self.outfit_style_completer.setCompletionMode(QCompleter.PopupCompletion)
        self.outfit_style_combo.setCompleter(self.outfit_style_completer)
        if self.outfit_style_combo.lineEdit() is not None:
            self.outfit_style_combo.lineEdit().setPlaceholderText("留空则不覆盖，例如：维多利亚风格")
            self.outfit_style_combo.lineEdit().setClearButtonEnabled(True)
            self.outfit_style_combo.lineEdit().editingFinished.connect(self._commit_outfit_style_from_editor)
            self.outfit_style_combo.lineEdit().textEdited.connect(self._filter_outfit_style_history_live)
        self.outfit_style_combo.currentTextChanged.connect(self._on_outfit_style_text_changed)
        self.outfit_style_combo.activated.connect(lambda _idx: self._commit_outfit_style_text(self.outfit_style_combo.currentText(), add_to_history=False))
        outfit_style_layout.addWidget(self.outfit_style_combo, stretch=1)
        self.delete_outfit_style_btn = QPushButton("x")
        self.delete_outfit_style_btn.setFixedWidth(28)
        self.delete_outfit_style_btn.setToolTip("删除当前历史项")
        self.delete_outfit_style_btn.clicked.connect(self._delete_current_outfit_style_history)
        outfit_style_layout.addWidget(self.delete_outfit_style_btn)
        layout.addLayout(outfit_style_layout)

        upscale_layout = QHBoxLayout()
        self.enable_jpg_upscale_cb = QCheckBox("生图后自动处理 JPG")
        self.enable_jpg_upscale_cb.toggled.connect(self._persist_upscale_options)
        upscale_layout.addWidget(self.enable_jpg_upscale_cb)
        upscale_layout.addWidget(QLabel("模型:"))
        self.upscale_model_combo = QComboBox()
        self.upscale_model_combo.currentTextChanged.connect(self._persist_upscale_options)
        upscale_layout.addWidget(self.upscale_model_combo)
        self.reload_upscale_models_btn = QPushButton("刷新模型")
        self.reload_upscale_models_btn.clicked.connect(self._reload_upscale_models)
        upscale_layout.addWidget(self.reload_upscale_models_btn)
        upscale_layout.addWidget(QLabel("倍率:"))
        self.upscale_by_spin = QDoubleSpinBox()
        self.upscale_by_spin.setRange(1.0, 8.0)
        self.upscale_by_spin.setSingleStep(0.1)
        self.upscale_by_spin.setValue(2.0)
        self.upscale_by_spin.valueChanged.connect(self._persist_upscale_options)
        upscale_layout.addWidget(self.upscale_by_spin)
        upscale_layout.addWidget(QLabel("WebP目标MB:"))
        self.webp_target_mb_spin = QDoubleSpinBox()
        self.webp_target_mb_spin.setRange(0.1, 100.0)
        self.webp_target_mb_spin.setDecimals(1)
        self.webp_target_mb_spin.setSingleStep(0.5)
        self.webp_target_mb_spin.setValue(10.0)
        self.webp_target_mb_spin.valueChanged.connect(self._persist_upscale_options)
        upscale_layout.addWidget(self.webp_target_mb_spin)
        upscale_layout.addStretch()
        layout.addLayout(upscale_layout)

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

        gen_control_layout = QHBoxLayout()
        self.gen_countdown_label = QLabel("生图超时倒计时: --")
        self.cancel_gen_btn = QPushButton("终止当前生图")
        self.cancel_gen_btn.setEnabled(False)
        self.cancel_gen_btn.clicked.connect(self.cancel_image_generation)
        gen_control_layout.addWidget(self.gen_countdown_label)
        gen_control_layout.addStretch()
        gen_control_layout.addWidget(self.cancel_gen_btn)
        layout.addLayout(gen_control_layout)

        history_header_layout = QHBoxLayout()
        history_header_layout.addWidget(QLabel("历史分析结果:"))
        history_header_layout.addStretch()
        self.apply_history_btn = QPushButton("设为当前结果")
        self.apply_history_btn.setEnabled(False)
        self.apply_history_btn.clicked.connect(self.apply_selected_history_result)
        history_header_layout.addWidget(self.apply_history_btn)
        self.open_history_btn = QPushButton("查看选中结果")
        self.open_history_btn.setEnabled(False)
        self.open_history_btn.clicked.connect(self.open_selected_history_detail)
        history_header_layout.addWidget(self.open_history_btn)
        layout.addLayout(history_header_layout)

        self.history_list = QListWidget()
        self.history_list.setMinimumHeight(150)
        self.history_list.itemSelectionChanged.connect(self._update_history_action_buttons)
        self.history_list.itemDoubleClicked.connect(lambda _item: self.open_selected_history_detail())
        layout.addWidget(self.history_list)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_toolbar_layout = QHBoxLayout()
        log_toolbar_layout.addStretch()
        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.clicked.connect(self.log_text.clear)
        log_toolbar_layout.addWidget(self.clear_log_btn)
        layout.addLayout(log_toolbar_layout)
        layout.addWidget(self.log_text)

        self.setLayout(layout)
        self._reload_upscale_models()
        self.set_upscale_options_defaults(self.get_upscale_options() if self.get_upscale_options else {})
        self.set_outfit_style_options(
            self.get_outfit_style_history() if self.get_outfit_style_history else [],
            self.get_outfit_style_default() if self.get_outfit_style_default else ""
        )

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
            config_path = os.path.join(BASE_DIR, 'conf', 'config-styles.json')
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

    def _next_thread_no(self, attr_name):
        next_no = int(getattr(self, attr_name, 0)) + 1
        setattr(self, attr_name, next_no)
        return next_no

    def _build_thread_prefix(self, thread_kind, thread_no, related_thread_no=None):
        prefix = f"[{thread_kind}#{thread_no}]"
        if related_thread_no is not None:
            prefix = f"[{thread_kind}#{thread_no}|分析线程#{related_thread_no}]"
        return prefix

    def _status_to_text(self, status):
        status_map = {
            "running": "进行中",
            "success": "已完成",
            "timeout": "超时",
            "cancelled": "已取消",
            "error": "失败",
            "idle": "未开始",
        }
        return status_map.get(str(status or "").strip().lower(), "未知状态")

    def _status_to_color(self, status):
        normalized = str(status or "").strip().lower()
        if normalized == "success":
            return QColor("#1b8f3a")
        if normalized == "running":
            return QColor("#0b63c7")
        if normalized in ("timeout", "error", "cancelled"):
            return QColor("#b23a2b")
        return QColor("#444444")

    def _create_history_record(self, thread_no, image_source_snapshot, submit_time, task_hash):
        task_id = f"analysis-{thread_no}"
        submit_time_text = submit_time.strftime("%Y-%m-%d %H:%M:%S")
        return {
            "task_id": task_id,
            "thread_no": thread_no,
            "task_hash": task_hash,
            "status": "running",
            "status_text": self._status_to_text("running"),
            "submit_time_text": submit_time_text,
            "finish_time_text": "-",
            "source_desc": _describe_image_source(image_source_snapshot),
            "title": "处理中",
            "result_json": None,
            "saved_json_path": "",
            "saved_prompt_paths": [],
            "aspect_ratio": "",
            "original_prompt": "",
            "refined_prompt": "",
        }

    def _insert_history_record(self, record):
        task_id = record["task_id"]
        self._analysis_history[task_id] = record
        item = QListWidgetItem()
        item.setData(Qt.UserRole, task_id)
        self.history_list.insertItem(0, item)
        self._refresh_history_item(task_id)
        self.history_list.setCurrentItem(item)

    def _refresh_history_item(self, task_id):
        record = self._analysis_history.get(task_id)
        if not record:
            return
        for row in range(self.history_list.count()):
            item = self.history_list.item(row)
            if item.data(Qt.UserRole) != task_id:
                continue
            status_text = record.get("status_text", "未知状态")
            title = str(record.get("title") or "未命名")
            submit_time_text = record.get("submit_time_text", "-")
            source_desc = str(record.get("source_desc") or "-")
            item.setText(
                f"[{status_text}] 线程#{record.get('thread_no', '?')} / {record.get('task_hash', '--------')}  {submit_time_text}\n"
                f"{title} | {source_desc}"
            )
            item.setForeground(self._status_to_color(record.get("status")))
            break

    def _update_history_record(self, task_id, **kwargs):
        record = self._analysis_history.get(task_id)
        if not record:
            return
        record.update(kwargs)
        record["status_text"] = self._status_to_text(record.get("status"))
        self._refresh_history_item(task_id)

    def _selected_history_record(self):
        item = self.history_list.currentItem()
        if item is None:
            return None
        task_id = item.data(Qt.UserRole)
        return self._analysis_history.get(task_id)

    def _update_history_action_buttons(self):
        record = self._selected_history_record()
        self.open_history_btn.setEnabled(record is not None)
        self.apply_history_btn.setEnabled(bool(record and record.get("status") == "success" and record.get("result_json")))

    def _apply_history_record_to_current_state(self, record):
        if not record:
            return False
        if str(record.get("status") or "").strip().lower() != "success":
            return False
        result_json = record.get("result_json")
        if not isinstance(result_json, dict):
            return False
        self.current_task_hash = str(record.get("task_hash") or result_json.get("task_hash") or "").strip()
        self.current_aspect_ratio = str(record.get("aspect_ratio") or result_json.get("aspect_ratio") or "1:1").strip() or "1:1"
        self.current_orig_desc = str(record.get("original_prompt") or result_json.get("original_english_description") or "").strip()
        self.current_refine_desc = str(record.get("refined_prompt") or result_json.get("english_description") or "").strip()
        self.gen_orig_btn.setEnabled(bool(self.current_orig_desc))
        self.gen_ref_btn.setEnabled(bool(self.current_refine_desc))
        return True

    def apply_selected_history_result(self):
        record = self._selected_history_record()
        if not record:
            QMessageBox.information(self, "提示", "请先在历史分析结果中选择一条记录。")
            return
        if not self._apply_history_record_to_current_state(record):
            QMessageBox.information(self, "提示", "只有已完成并带结果的历史记录才能设为当前结果。")
            return
        self.log_msg(
            f"已将历史分析结果设为当前结果: 线程#{record.get('thread_no', '?')} / {record.get('task_hash', '--------')} / {record.get('title', '未命名')}"
        )

    def open_selected_history_detail(self):
        record = self._selected_history_record()
        if not record:
            QMessageBox.information(self, "提示", "请先在历史分析结果中选择一条记录。")
            return
        dialog = AnalysisHistoryDetailDialog(record, self)
        dialog.exec_()

    def log_msg(self, text, prefix=None):
        message = "" if text is None else str(text)
        if prefix:
            lines = message.splitlines()
            if not lines:
                lines = [message]
            message = "\n".join(f"{prefix} {line}" if line else prefix for line in lines)
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def process_image(self):
        if not self.image_source: return
        self._commit_outfit_style_text(self.outfit_style_combo.currentText(), add_to_history=True)
        missing_prompt_files = get_single_analyzer_missing_prompt_files(
            enable_refine=True,
            enable_outfit_check=self.enable_outfit_check_cb.isChecked(),
        )
        if missing_prompt_files:
            missing_text = "\n".join(missing_prompt_files)
            QMessageBox.warning(self, "缺少 Prompt 文件", f"以下 Prompt 文件不存在，请补齐后再执行：\n{missing_text}")
            self.log_msg(f"❌ 缺少 Prompt 文件，已中止分析：\n{missing_text}")
            return
            
        base_url, api_key, model_name = self.get_text_config(self.use_nsfw_cb.isChecked())
        if not api_key or not model_name:
            QMessageBox.warning(self, "缺少配置", "文本分析 API Key 和 模型名称不能为空！")
            return
        
        image_source_snapshot = _clone_image_source(self.image_source)
        analysis_thread_no = self._next_thread_no("_analysis_thread_seq")
        thread_prefix = self._build_thread_prefix("分析线程", analysis_thread_no)
        self.log_msg("\n" + ("=" * 72))
        self.log_msg("任务已启动...", prefix=thread_prefix)
        timeout_seconds = int(self.get_timeout_seconds()) if self.get_timeout_seconds else 120
        submit_time = datetime.datetime.now()
        ddl = submit_time + datetime.timedelta(seconds=max(1, timeout_seconds))
        task_hash = _generate_task_hash(image_source_snapshot, submit_time, analysis_thread_no)
        self.log_msg(f"图片源: {_describe_image_source(image_source_snapshot)}", prefix=thread_prefix)
        self.log_msg(f"提交时间: {submit_time.strftime('%Y-%m-%d %H:%M:%S')}", prefix=thread_prefix)
        self.log_msg(f"任务 Hash: {task_hash}", prefix=thread_prefix)
        self.log_msg(f"超时设置: {timeout_seconds} 秒（预计超时点: {ddl.strftime('%H:%M:%S')}）", prefix=thread_prefix)
        history_record = self._create_history_record(analysis_thread_no, image_source_snapshot, submit_time, task_hash)
        self._insert_history_record(history_record)
        
        booru_tag_limit = int(self.get_booru_tag_limit()) if self.get_booru_tag_limit else 30
        thread = WorkerThread(
            image_source_snapshot,
            api_key,
            base_url,
            model_name,
            booru_tag_limit=booru_tag_limit,
            timeout_seconds=timeout_seconds,
            enable_outfit_check=self.enable_outfit_check_cb.isChecked(),
            outfit_style_override=self.outfit_style_combo.currentText().strip()
        )
        thread.meta_thread_no = analysis_thread_no
        thread.meta_source_snapshot = image_source_snapshot
        thread.meta_task_id = history_record["task_id"]
        thread.meta_task_hash = task_hash
        self._active_analysis_threads.append(thread)
        thread.log_signal.connect(
            lambda text, t=thread: self.log_msg(
                text,
                prefix=self._build_thread_prefix("分析线程", getattr(t, "meta_thread_no", "?"))
            )
        )
        thread.finish_signal.connect(lambda result_json, t=thread: self.on_process_finished(t, result_json))
        thread.finished.connect(lambda t=thread: self._on_analysis_thread_stopped(t))
        thread.start()

    def on_use_nsfw_toggled(self, checked):
        if self.on_nsfw_changed:
            self.on_nsfw_changed(bool(checked))

    def on_enable_outfit_check_toggled(self, checked):
        if self.on_outfit_check_changed:
            self.on_outfit_check_changed(bool(checked))

    def _reload_upscale_models(self):
        current = self.upscale_model_combo.currentText().strip()
        models = list_esrgan_models()
        self.upscale_model_combo.blockSignals(True)
        self.upscale_model_combo.clear()
        self.upscale_model_combo.addItems(models)
        if current:
            self.upscale_model_combo.setCurrentText(current)
        self.upscale_model_combo.blockSignals(False)
        if not models:
            self.log_msg("⚠️ 未找到 ESRGAN 模型，请确认 data/models/ESRGAN 或 models/ESRGAN 目录")
        self._persist_upscale_options()

    def _collect_upscale_options(self):
        raw = {
            "enabled": bool(self.enable_jpg_upscale_cb.isChecked()),
            "model_name": self.upscale_model_combo.currentText().strip(),
            "upscale_mode": 0,
            "upscale_by": float(self.upscale_by_spin.value()),
            "max_side_length": 0,
            "upscale_to_width": 1024,
            "upscale_to_height": 1024,
            "upscale_crop": False,
            "upscaler_2_name": "",
            "upscaler_2_visibility": 0.0,
            "cache_size": 4,
            "webp_target_mb": float(self.webp_target_mb_spin.value()),
        }
        return normalize_upscale_options(raw)

    def _persist_upscale_options(self):
        if self.on_upscale_options_changed:
            self.on_upscale_options_changed(self._collect_upscale_options())

    def set_upscale_options_defaults(self, options):
        opts = normalize_upscale_options(options)
        self.enable_jpg_upscale_cb.blockSignals(True)
        self.enable_jpg_upscale_cb.setChecked(bool(opts.get("enabled", False)))
        self.enable_jpg_upscale_cb.blockSignals(False)
        self.upscale_by_spin.blockSignals(True)
        self.upscale_by_spin.setValue(float(opts.get("upscale_by", 2.0)))
        self.upscale_by_spin.blockSignals(False)
        self.webp_target_mb_spin.blockSignals(True)
        self.webp_target_mb_spin.setValue(float(opts.get("webp_target_mb", 10.0)))
        self.webp_target_mb_spin.blockSignals(False)
        model_name = str(opts.get("model_name", "")).strip()
        if model_name:
            self.upscale_model_combo.setCurrentText(model_name)

    def set_use_nsfw_default(self, checked):
        self.use_nsfw_cb.blockSignals(True)
        self.use_nsfw_cb.setChecked(bool(checked))
        self.use_nsfw_cb.blockSignals(False)

    def set_outfit_check_default(self, checked):
        self.enable_outfit_check_cb.blockSignals(True)
        self.enable_outfit_check_cb.setChecked(bool(checked))
        self.enable_outfit_check_cb.blockSignals(False)

    def set_outfit_style_options(self, history_items, current_text=""):
        items = []
        for item in (history_items or []):
            text = str(item or "").strip()
            if text and text not in items:
                items.append(text)
        self._outfit_style_history_cache = list(items)
        current_text = str(current_text or "").strip()
        self._apply_outfit_style_combo_items(items, current_text)

    def _apply_outfit_style_combo_items(self, items, current_text):
        self._updating_outfit_style_combo = True
        self.outfit_style_combo.blockSignals(True)
        self.outfit_style_combo.clear()
        self.outfit_style_combo.addItems(items)
        self.outfit_style_combo.setCurrentText(current_text)
        self.outfit_style_combo.blockSignals(False)
        self._updating_outfit_style_combo = False
        self.outfit_style_completer_model.setStringList(list(items))
        self._refresh_outfit_style_delete_btn()

    def _filter_outfit_style_history_live(self, text):
        if self._updating_outfit_style_combo:
            return
        keyword = str(text or "").strip().lower()
        if keyword:
            filtered_items = [item for item in self._outfit_style_history_cache if keyword in item.lower()]
        else:
            filtered_items = list(self._outfit_style_history_cache)
        self._apply_outfit_style_combo_items(filtered_items, text)
        if filtered_items:
            self.outfit_style_combo.showPopup()
        else:
            self.outfit_style_combo.hidePopup()

    def _refresh_outfit_style_delete_btn(self):
        current_text = str(self.outfit_style_combo.currentText() or "").strip()
        history_items = self.get_outfit_style_history() if self.get_outfit_style_history else []
        normalized_history = {str(item or "").strip() for item in (history_items or []) if str(item or "").strip()}
        self.delete_outfit_style_btn.setEnabled(bool(current_text and current_text in normalized_history))

    def _on_outfit_style_text_changed(self, _text):
        if self._updating_outfit_style_combo:
            return
        self._refresh_outfit_style_delete_btn()

    def _commit_outfit_style_from_editor(self):
        self._commit_outfit_style_text(self.outfit_style_combo.currentText(), add_to_history=True)

    def _commit_outfit_style_text(self, text, add_to_history):
        if self._updating_outfit_style_combo:
            return
        value = str(text or "").strip()
        if self.on_outfit_style_changed:
            self.on_outfit_style_changed(value, add_to_history=bool(add_to_history and value))
        self._refresh_outfit_style_delete_btn()

    def _delete_current_outfit_style_history(self):
        value = str(self.outfit_style_combo.currentText() or "").strip()
        if not value:
            return
        if self.on_outfit_style_deleted:
            self.on_outfit_style_deleted(value)

    def _on_analysis_thread_stopped(self, thread):
        if thread in self._active_analysis_threads:
            self._active_analysis_threads.remove(thread)
        self.send_btn.setEnabled(bool(self.image_source))

    def on_process_finished(self, thread, result_json):
        analysis_thread_no = getattr(thread, "meta_thread_no", "?")
        thread_prefix = self._build_thread_prefix("分析线程", analysis_thread_no)
        task_status = getattr(thread, "last_status", "unknown")
        task_id = getattr(thread, "meta_task_id", "")
        finish_time_text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not result_json:
            self._update_history_record(
                task_id,
                status=task_status,
                title="未生成有效结果",
                finish_time_text=finish_time_text,
            )
            if task_status == "timeout":
                self.log_msg("处理失败：请求超时，请检查“请求超时时间”配置是否足够。", prefix=thread_prefix)
                self._send_system_notification("单图分析超时", "任务因请求超时结束，请调整超时配置后重试。")
            elif task_status == "cancelled":
                self.log_msg("任务已取消。", prefix=thread_prefix)
                self._send_system_notification("单图分析已取消", "当前任务已被取消。")
            else:
                self.log_msg("处理失败，未能获取到有效的 JSON 数据。", prefix=thread_prefix)
                self._send_system_notification("单图分析失败", "任务已结束但未获取到有效结果。")
            return
        source_snapshot = getattr(thread, "meta_source_snapshot", None)
        if isinstance(source_snapshot, str):
            result_json["source_image_path"] = os.path.abspath(source_snapshot)
        else:
            result_json["source_image_path"] = ""
        task_hash = str(getattr(thread, "meta_task_hash", "") or "").strip()
        if task_hash:
            result_json["task_hash"] = task_hash

        self.log_msg("========== 最终处理结果 ==========", prefix=thread_prefix)
        self.log_msg(json.dumps(result_json, indent=4, ensure_ascii=False), prefix=thread_prefix)
        
        jp_title = result_json.get("japanese_title", "未命名")
        safe_title = re.sub(r'[\\/*?:"<>|]', "", jp_title).strip() or "未命名"
        
        now = datetime.datetime.now()
        now_str = now.strftime("%Y%m%d-%H%M%S")
        date_str = now.strftime("%Y%m%d") 
        save_dir = os.path.join('data', date_str) 
        
        if not os.path.exists(save_dir): os.makedirs(save_dir) 
            
        safe_task_hash = task_hash or "nohash"
        base_filename = f"{now_str}-{safe_task_hash}-{safe_title}"
        json_filename = f"{base_filename}.json"
        
        try:
            saved_json_path = os.path.abspath(os.path.join(save_dir, json_filename))
            with open(os.path.join(save_dir, json_filename), "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=4)
            self.log_msg(f"✅ 成功！JSON 结果已保存至: {json_filename}", prefix=thread_prefix)
        except Exception as e:
            saved_json_path = ""
            self.log_msg(f"❌ 保存 JSON 文件时出错: {e}", prefix=thread_prefix)

        try:
            raw_ar = result_json.get("aspect_ratio", "2:3")
            self.current_aspect_ratio = self._resolve_ar_for_first_stage(raw_ar)
            self.log_msg(f"📌 确定的图片长宽比: {self.current_aspect_ratio}", prefix=thread_prefix)

            self.current_refine_desc = result_json.get("english_description", "")
            self.current_orig_desc = result_json.get("original_english_description", "")
            self.current_task_hash = safe_task_hash
            
            
            selected_style_name = self.main_style_combo.currentText()
            styles_data = self.get_styles()
            current_fixed_tags = styles_data.get(selected_style_name, "")
            
            # 在风格标签和描述之间添加两个回车
            style_part = f"--ar {self.current_aspect_ratio} {current_fixed_tags}".strip()
            final_prompt = f"{style_part}\n\n{self.current_refine_desc}".strip()
            orig_prompt = f"{style_part}\n\n{self.current_orig_desc}".strip()
            
            txt_filename = f"{base_filename}-prompts.txt"
            orig_txt_filename = f"{base_filename}-original-prompts.txt"
            saved_prompt_paths = [
                os.path.abspath(os.path.join(save_dir, txt_filename)),
                os.path.abspath(os.path.join(save_dir, orig_txt_filename))
            ]
            
            with open(os.path.join(save_dir, txt_filename), "w", encoding="utf-8") as f: f.write(final_prompt)
            with open(os.path.join(save_dir, orig_txt_filename), "w", encoding="utf-8") as f: f.write(orig_prompt)
                
            self.log_msg(f"✅ 成功！两份画幅与提示词文件已保存:\n - {txt_filename}\n - {orig_txt_filename}", prefix=thread_prefix)
            
            self._apply_history_record_to_current_state({
                "status": "success",
                "result_json": result_json,
                "task_hash": safe_task_hash,
                "aspect_ratio": self.current_aspect_ratio,
                "original_prompt": self.current_orig_desc,
                "refined_prompt": self.current_refine_desc,
            })
            self.log_msg("💡 该结果已设为当前手动生图内容，可直接点击下方按钮继续生成图片。", prefix=thread_prefix)
        except Exception as e:
            saved_prompt_paths = []
            self.log_msg(f"❌ 保存提示词 txt 文件时出错: {e}", prefix=thread_prefix)

        self._update_history_record(
            task_id,
            status="success",
            title=str(result_json.get("japanese_title") or result_json.get("chinese_title") or "未命名"),
            finish_time_text=finish_time_text,
            result_json=result_json,
            saved_json_path=saved_json_path,
            saved_prompt_paths=saved_prompt_paths,
            task_hash=safe_task_hash,
            aspect_ratio=self.current_aspect_ratio,
            original_prompt=self.current_orig_desc,
            refined_prompt=self.current_refine_desc,
        )

        # 【新增】自动执行发图逻辑
        auto_targets = []
        if self.auto_gen_orig_cb.isChecked() and str(self.current_orig_desc).strip():
            auto_targets.append("original")
        if self.auto_gen_ref_cb.isChecked() and str(self.current_refine_desc).strip():
            auto_targets.append("refined")
        prompt_bundle = {
            "task_hash": self.current_task_hash,
            "aspect_ratio": self.current_aspect_ratio,
            "original_prompt": self.current_orig_desc,
            "refined_prompt": self.current_refine_desc,
        }
        if auto_targets:
            auto_group_id = f"analysis-{analysis_thread_no}"
            self._auto_gen_groups[auto_group_id] = {
                "expected": 0,
                "finished": 0,
                "cancelled": False,
                "thread_no": analysis_thread_no,
            }
            started_count = 0
            for prompt_type in auto_targets:
                if self.trigger_image_generation(
                    prompt_type,
                    is_auto=True,
                    prompt_bundle=prompt_bundle,
                    analysis_thread_no=analysis_thread_no,
                    auto_group_id=auto_group_id
                ):
                    started_count += 1
            if started_count > 0:
                self._auto_gen_groups[auto_group_id]["expected"] = started_count
                self.log_msg(f"🤖 检测到自动生图任务，共 {started_count} 项，通知将于全部生图结束后发送。", prefix=thread_prefix)
            else:
                self._auto_gen_groups.pop(auto_group_id, None)
                self._send_system_notification("单图分析完成", "任务已完成并生成结果文件。")
        else:
            self._send_system_notification("单图分析完成", "任务已完成并生成结果文件。")

    def _start_image_gen_runtime(self, timeout_seconds):
        self._img_gen_running = True
        self._img_gen_timeout_seconds = max(1, int(timeout_seconds))
        now = datetime.datetime.now()
        self._img_gen_deadline = now + datetime.timedelta(seconds=self._img_gen_timeout_seconds)
        self.cancel_gen_btn.setEnabled(True)
        self.log_msg(f"生图提交时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_msg(f"生图超时设置: {self._img_gen_timeout_seconds} 秒（预计超时点: {self._img_gen_deadline.strftime('%H:%M:%S')}）")
        self._img_gen_countdown.start(self._img_gen_timeout_seconds)

    def _stop_image_gen_runtime(self):
        self._img_gen_running = False
        self._img_gen_deadline = None
        self._img_gen_timeout_seconds = 0
        self._img_gen_countdown.stop()
        self.cancel_gen_btn.setEnabled(False)
        self.gen_countdown_label.setText("生图超时倒计时: --")

    def _on_image_gen_countdown_tick(self, remain_seconds):
        if not self._img_gen_running:
            self.gen_countdown_label.setText("生图超时倒计时: --")
            return
        remain = int(remain_seconds)
        if remain <= 0:
            self.gen_countdown_label.setText("生图超时倒计时: 0 秒")
            self.log_msg("⏰ 生图超时倒计时已到，正在终止当前生图任务...")
            return
        self.gen_countdown_label.setText(f"生图超时倒计时: {remain} 秒")

    def _on_image_thread_stopped(self, thread):
        if thread in self._active_img_threads:
            self._active_img_threads.remove(thread)
        if not self._active_img_threads:
            self.gen_orig_btn.setEnabled(True)
            self.gen_ref_btn.setEnabled(True)
            self._stop_image_gen_runtime()

    def cancel_image_generation(self, reason="manual"):
        if not self._active_img_threads:
            self.log_msg("当前没有正在执行的生图任务。")
            return
        for group in self._auto_gen_groups.values():
            group["cancelled"] = True
        running_threads = list(self._active_img_threads)
        self.log_msg(f"正在请求终止 {len(running_threads)} 个生图任务，等待线程自行退出...")
        self._img_gen_running = False
        self._img_gen_deadline = None
        self._img_gen_timeout_seconds = 0
        self._img_gen_countdown.stop()
        self.cancel_gen_btn.setEnabled(False)
        self.gen_countdown_label.setText("生图超时倒计时: 正在退出...")
        self.gen_orig_btn.setEnabled(False)
        self.gen_ref_btn.setEnabled(False)
        for t in running_threads:
            try:
                if hasattr(t, "request_cancel"):
                    t.request_cancel()
                else:
                    t.requestInterruption()
            except Exception:
                pass
        if reason == "timeout":
            self.log_msg("⏰ 已发出生图超时终止请求，等待当前任务退出。")
            self._send_system_notification("生图任务超时", "自动生图已超时并终止。")
        else:
            self.log_msg("🛑 已发出生图终止请求，等待当前任务退出。")
            self._send_system_notification("生图任务已终止", "当前生图任务已手动取消。")

    def trigger_image_generation(self, prompt_type, is_auto=False, prompt_bundle=None, analysis_thread_no=None, auto_group_id=None):
        self.save_img_cfg()
        
        img_base_url, img_key, model_name, api_type = self.get_img_config()
        if not img_key:
            if is_auto:
                self.log_msg("自动生图已跳过：生图 API Key 不能为空，请检查【全局配置】。")
            else:
                QMessageBox.warning(self, "缺少配置", "生图 API Key 不能为空，请检查【全局配置】。")
            return False
        timeout_seconds = int(self.get_timeout_seconds()) if self.get_timeout_seconds else 120
        
        prompt_context = prompt_bundle or {
            "task_hash": self.current_task_hash,
            "aspect_ratio": self.current_aspect_ratio,
            "original_prompt": self.current_orig_desc,
            "refined_prompt": self.current_refine_desc,
        }
        task_hash = str(prompt_context.get("task_hash") or self.current_task_hash or "").strip()
        prompt_to_use = prompt_context.get("original_prompt", "") if prompt_type == "original" else prompt_context.get("refined_prompt", "")
        if not str(prompt_to_use).strip():
            self.log_msg(f"{prompt_type} 提示词为空，已跳过本次生图。")
            return False
        
        selected_style_name = self.main_style_combo.currentText()
        styles_data = self.get_styles()
        active_instructions = styles_data.get(selected_style_name, "")
        
        self.gen_orig_btn.setEnabled(False)
        self.gen_ref_btn.setEnabled(False)
        
        # 【修改】动态实例化线程对象存放至列表，避免并发勾选导致线程互相覆盖报错
        final_gen_ar = self._resolve_ar_for_second_stage(prompt_context.get("aspect_ratio", self.current_aspect_ratio))

        img_thread = ImageGenWorkerThread(
            prompt=prompt_to_use,
            model_name=model_name,
            aspect_ratio=final_gen_ar,
            instructions=active_instructions,
            api_type=api_type,
            file_prefix=task_hash
        )
        img_thread.meta_thread_no = self._next_thread_no("_image_gen_thread_seq")
        img_thread.meta_analysis_thread_no = analysis_thread_no
        img_thread.meta_is_auto = bool(is_auto)
        img_thread.meta_prompt_type = prompt_type
        img_thread.meta_auto_group_id = auto_group_id
        img_thread.meta_task_hash = task_hash

        self._active_img_threads.append(img_thread)
        if not self._img_gen_running:
            self._start_image_gen_runtime(timeout_seconds)
        
        img_thread.log_signal.connect(
            lambda text, t=img_thread: self.log_msg(
                text,
                prefix=self._build_thread_prefix(
                    "生图线程",
                    getattr(t, "meta_thread_no", "?"),
                    getattr(t, "meta_analysis_thread_no", None)
                )
            )
        )
        img_thread.finish_signal.connect(lambda files, t=img_thread: self.on_image_generation_finished(t, files))
        
        # 清除完成的线程并同步按钮状态
        img_thread.finished.connect(lambda t=img_thread: self._on_image_thread_stopped(t))
        img_thread.start()
        return True

    def on_image_generation_finished(self, thread, saved_files):
        prompt_type = getattr(thread, "meta_prompt_type", "unknown")
        is_auto = bool(getattr(thread, "meta_is_auto", False))
        thread_prefix = self._build_thread_prefix(
            "生图线程",
            getattr(thread, "meta_thread_no", "?"),
            getattr(thread, "meta_analysis_thread_no", None)
        )
        
        if saved_files:
            self.log_msg(f"🎉 成功生成了 {len(saved_files)} 张 {prompt_type} 图片！", prefix=thread_prefix)
            for file_path in saved_files:
                self.log_msg(f"📂 保存路径: {file_path}", prefix=thread_prefix)
            self._start_jpg_postprocess(saved_files, prompt_type)
        else:
            status = getattr(thread, "last_status", "unknown")
            if status == "cancelled":
                self.log_msg(f"🛑 {prompt_type} 生图已取消。", prefix=thread_prefix)
            else:
                self.log_msg(f"⚠️ 未能生成 {prompt_type} 图片，请检查上方日志，或查看日志文件夹（log）的记录。", prefix=thread_prefix)
        auto_group_id = getattr(thread, "meta_auto_group_id", None)
        if is_auto and auto_group_id:
            group = self._auto_gen_groups.get(auto_group_id)
            if group:
                group["finished"] += 1
                if group["finished"] >= group["expected"] and group["expected"] > 0:
                    should_notify = not group.get("cancelled", False)
                    self._auto_gen_groups.pop(auto_group_id, None)
                    if should_notify:
                        self._send_system_notification("单图分析与自动生图完成", "分析与自动生图任务已全部完成。")

    def _start_jpg_postprocess(self, saved_files, prompt_type):
        if not self.enable_jpg_upscale_cb.isChecked():
            return
        jpg_files = [str(path) for path in (saved_files or []) if str(path).lower().endswith((".jpg", ".jpeg"))]
        if not jpg_files:
            return
        options = self._collect_upscale_options()
        if not options.get("model_name"):
            self.log_msg("⚠️ 已启用 JPG 自动处理，但未选择 upscaler 模型，已跳过。")
            return
        thread = JpgAutoUpscaleThread(
            image_paths=jpg_files,
            options=options,
            task_name=f"单图{prompt_type}后处理",
        )
        thread.meta_thread_no = self._next_thread_no("_post_thread_seq")
        self._active_post_threads.append(thread)
        thread.log_signal.connect(
            lambda text, t=thread: self.log_msg(
                text,
                prefix=self._build_thread_prefix("后处理线程", getattr(t, "meta_thread_no", "?"))
            )
        )
        thread.finish_signal.connect(lambda results, t=thread: self._on_postprocess_finished(t, results))
        thread.finished.connect(lambda t=thread: self._cleanup_post_thread(t))
        thread.start()

    def _on_postprocess_finished(self, thread, results):
        success = 0
        webp_count = 0
        for item in results or []:
            if item.get("fixed_png_path") and not item.get("error"):
                success += 1
            if item.get("webp_path"):
                webp_count += 1
        if success > 0:
            self.log_msg(
                f"✅ JPG 自动处理完成，新增 fixed.png: {success} 张，WebP: {webp_count} 张",
                prefix=self._build_thread_prefix("后处理线程", getattr(thread, "meta_thread_no", "?"))
            )

    def _cleanup_post_thread(self, thread):
        if thread in self._active_post_threads:
            self._active_post_threads.remove(thread)
