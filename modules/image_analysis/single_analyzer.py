import os
import json
import datetime
import re
import hashlib
import logging
import tempfile
from openai import OpenAI
from PIL import Image, ImageGrab

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QCheckBox,
                             QLabel, QPushButton, QTextEdit, QComboBox, QMessageBox, QDoubleSpinBox, QCompleter,
                             QListWidget, QListWidgetItem, QDialog, QMenu, QApplication, QScrollArea, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QStringListModel
from PyQt6.QtGui import QPixmap, QColor, QDesktopServices
from PyQt6.QtCore import QUrl

from modules.others.api_backend import generate_image_whatai, generate_image_aigc2d 
from utils.booru_tags import normalize_booru_tags, filter_facial_degrading_tags, filter_facial_degrading_from_text
from utils.wd14_tagger import predict_local_booru_tags, merge_prompt_with_local_booru_tags, merge_prompt_with_pixiv_tag_hints
from utils.styles import style_prompt, style_ref_image, ref_image_valid, build_ref_gen_params
from utils.style_ref_widget import StyleRefModeCombo
from utils.pixiv_tag_matcher import get_local_pixiv_tag_candidates
from utils.task_runtime import SystemNotifier, TaskCountdown
from utils.image_encoding import compress_and_encode_image as _base_compress_and_encode_image
from utils.image_upscale_runtime import JpgAutoUpscaleThread, list_esrgan_models, normalize_upscale_options
from utils.prompt_loader import read_prompt_file, render_prompt_file, find_missing_prompt_files
from utils.llm_retry import call_with_retry, load_retry_settings, format_wait
from utils.output_isolation import resolve_output_target
from modules.image_analysis.dir_batch_selector import DirectoryBatchSelectorDialog

logger = logging.getLogger("whatai_logger")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SYSTEM_PROMPT_FILE = "single-analyzer-system.md"
STYLE_ANALY_PROMPT_FILE = "style-analy.md"
REFINE_DESC_PROMPT_FILE = "refine-desc.md"
OUTFIT_CHECK_PROMPT_FILE = "single-analyzer-outfit-check.md"
REMOVE_PHOTO_STYLE_PROMPT_FILE = "remove-photo-style.md"
RECOMPUTE_PIXIV_TAGS_PROMPT_FILE = "recompute-pixiv-tags.md"
# 剪贴板图片没有磁盘路径，提交分析时在这里落一份快照，右键「重跑」才有源可用。
# 目录位于可清空的 cache/temp 下：清掉后旧记录会退化成「源图不可用」。
CLIPBOARD_SNAPSHOT_DIR = os.path.join("cache", "temp", "single-analyzer-clipboard")


ANALYSIS_GPT_UI_NODE = "analysis_gpt_pipeline"
# 默认链：GPT 首图 → Gemini（源图 + 完整画风图）→ 一次 source-only 质量修订 → 身份定点修订。
# recipe_version 只迁移配方键一次；用户之后的自定义选择继续保留。
GPT_RECIPE_VERSION = 6
STABLE_LOCAL_REGIONS = ("subject_no_face", "shoes", "waist", "thigh")
# 注意：`dual_reference` 键**已废弃**（旧配置里可能还留着，读到会忽略）。
# 重绘的第二张参考由 `analysis_gen.pipeline_steps_from_flags(repaint_ref_mode=...)` 决定，
# 分析 Tab 默认为 "style"（源图 + 完整画风图）；其它参考组合留给无头 CLI 对照实验。
# —— §三十一 实测线锚图会把画面塌成白底线稿 / 铺满「碎玻璃」纹理。
ANALYSIS_GPT_UI_DEFAULTS = {"channel": "gemini", "repaint": True, "structure": False,
                            "local": False, "region": STABLE_LOCAL_REGIONS[0],
                            "regions": list(STABLE_LOCAL_REGIONS),
                            "quality": "high",
                            "size_follow_input": True,
                            "tone": False, "tone_target": "style", "ink": False,
                            "repaint_scope": "full", "first_pass_mode": "generate",
                            "recipe_version": GPT_RECIPE_VERSION}
GPT_RECIPE_KEYS = ("repaint", "structure", "local", "region", "regions", "tone", "tone_target", "ink",
                   "repaint_scope")


def analysis_gpt_ui_path() -> str:
    """UI 记忆写进统一配置 conf/config.json（与 gpt-image-2 Tab 同一个文件）。"""
    return os.path.join(BASE_DIR, "conf", "config.json")


def load_analysis_gpt_ui(path: str = None) -> dict:
    """读取上次的「生图通道 / gpt 工序」选择（缺字段用默认值）。

    配方升级：老配置里没有 `recipe_version`（或版本落后）时，**配方相关的键一次性回到新默认**
    （= 完整画风图重绘 + 两轮身份闭环，额外本地工序关闭），之后保留用户的自定义开关。
    """
    state = dict(ANALYSIS_GPT_UI_DEFAULTS)
    try:
        with open(path or analysis_gpt_ui_path(), encoding="utf-8") as f:
            node = (json.load(f) or {}).get(ANALYSIS_GPT_UI_NODE)
        if isinstance(node, dict):
            state.update({k: node[k] for k in ANALYSIS_GPT_UI_DEFAULTS if k in node})
            if int(node.get("recipe_version") or 0) < GPT_RECIPE_VERSION:
                for key in GPT_RECIPE_KEYS:
                    state[key] = ANALYSIS_GPT_UI_DEFAULTS[key]
    except Exception:  # noqa: BLE001 - 读不到就用默认
        pass
    return state


def save_analysis_gpt_ui(state: dict, path: str = None) -> None:
    """把当前选择写回 conf/config.json（只动 ANALYSIS_GPT_UI_NODE 这一个节点）。"""
    target = path or analysis_gpt_ui_path()
    try:
        data = {}
        if os.path.isfile(target):
            with open(target, encoding="utf-8") as f:
                data = json.load(f) or {}
        node = dict(ANALYSIS_GPT_UI_DEFAULTS)
        node.update({k: state.get(k, v) for k, v in ANALYSIS_GPT_UI_DEFAULTS.items()})
        data[ANALYSIS_GPT_UI_NODE] = node
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception:  # noqa: BLE001 - 记不住选择不该影响生图
        pass


def get_single_analyzer_required_prompt_files(enable_refine=True, enable_outfit_check=False, enable_remove_photo_style=False, enable_recompute_pixiv_tags=False):
    files = [
        SYSTEM_PROMPT_FILE,
        STYLE_ANALY_PROMPT_FILE,
    ]
    if enable_refine:
        files.append(REFINE_DESC_PROMPT_FILE)
    if enable_outfit_check:
        files.append(OUTFIT_CHECK_PROMPT_FILE)
    if enable_remove_photo_style:
        files.append(REMOVE_PHOTO_STYLE_PROMPT_FILE)
    if enable_recompute_pixiv_tags:
        files.append(RECOMPUTE_PIXIV_TAGS_PROMPT_FILE)
    return files


def get_single_analyzer_missing_prompt_files(enable_refine=True, enable_outfit_check=False, enable_remove_photo_style=False, enable_recompute_pixiv_tags=False):
    return find_missing_prompt_files(
        get_single_analyzer_required_prompt_files(
            enable_refine=enable_refine,
            enable_outfit_check=enable_outfit_check,
            enable_remove_photo_style=enable_remove_photo_style,
            enable_recompute_pixiv_tags=enable_recompute_pixiv_tags,
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

def _sanitize_title_text(text, fallback_text=""):
    """清理标题：去掉英文/罗马字/数字/ASCII 标点，只保留中日韩（CJK/假名/全角）字符。

    用于防御性地处理模型偶尔在 japanese_title / chinese_title 中混入的
    「意义不明英文」（如 'rocking'）、罗马字、数字或 ASCII 标点。
    若清理后为空，则回退到 fallback_text（通常是中文标题）或空串。
    """
    text = str(text or "")
    cleaned = "".join(
        ch for ch in text if ord(ch) > 0x7F and not ch.isspace()
    )
    if cleaned:
        return cleaned
    return str(fallback_text or "").replace("\n", "").strip()

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
    normalized["japanese_title"] = _sanitize_title_text(japanese_title, fallback_text=chinese_title)
    normalized["chinese_title"] = _sanitize_title_text(chinese_title, fallback_text="")
    normalized["pixiv_tags"] = [str(tag).strip() for tag in pixiv_tags if str(tag).strip()]
    normalized["short_description"] = str(short_description).strip()
    booru_tags_normalized = normalize_booru_tags(booru_tags, limit=limit)
    normalized["booru-tags"] = filter_facial_degrading_tags(booru_tags_normalized)
    # 清理描述文本中的面部模糊/打码/遮挡相关短语
    normalized["english_description"] = filter_facial_degrading_from_text(normalized["english_description"])
    normalized["short_description"] = filter_facial_degrading_from_text(normalized["short_description"])
    return normalized

def _extract_json_candidates_from_text(text):
    """从可能含前置文本 / 多个拼接 JSON 的文本中提取所有可解析的 JSON 对象（dict）。

    有些模型在 json_object 模式下仍会先输出一个「preparing/status」前置对象再接真正的结果对象，
    导致整段 json.loads 失败。此函数逐字符扫描顶层平衡的 {...} 对象，收集所有能解析出的 dict 候选。
    """
    text = str(text or "").strip()
    candidates = []
    if not text:
        return candidates

    # 先尝试整段直接解析（最常见、最可靠）
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            candidates.append(parsed)
    except Exception:
        pass

    # 逐字符扫描顶层平衡对象
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        j = i
        in_str = False
        esc = False
        found_closing = False
        while j < n:
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(text[i:j + 1])
                            if isinstance(parsed, dict):
                                candidates.append(parsed)
                        except Exception:
                            pass
                        i = j + 1
                        found_closing = True
                        break
            j += 1
        if not found_closing:
            break

    # 去重
    seen = set()
    deduped = []
    for c in candidates:
        try:
            key = json.dumps(c, sort_keys=True, ensure_ascii=False)
        except Exception:
            key = repr(c)
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    return deduped


def _score_analysis_json_candidate(candidate, step_label):
    """给候选 JSON 打分，从拼接/前置文本里挑出真正的分析结果对象。

    结果对象通常包含 english_description 及若干结果字段；前置的 status/message 包装对象应被压低。
    """
    result_keys = (
        "english_description", "japanese_title", "chinese_title",
        "short_description", "pixiv_tags", "booru-tags", "booru_tags",
        "aspect_ratio",
    )
    score = 0
    has_english = False
    for key in result_keys:
        if key in candidate:
            if key == "english_description":
                has_english = True
                score += 12
            else:
                score += 3
    if has_english:
        score += 6
    # 前置状态包装 / 思考结构降低优先级
    if "status" in candidate and "message" in candidate:
        score -= 8
    if not has_english and set(candidate.keys()).issubset({"status", "message", "reason"}):
        score -= 20
    return score


def _log_step_diag(message, log_callback=None):
    """把分析步骤的诊断信息同时送到 GUI 回调与 log/<日期>.log。

    分析链路走的是 OpenAI SDK（不经过 api_backend 的请求日志），此前失败原因只 emit 给
    GUI 面板或被 print 到 stdout；GUI 用 pythonw 启动时 stdout 被重定向到 os.devnull，
    于是日志文件与临时目录里什么都查不到。这里统一双写，保证失败可追溯。
    """
    text = str(message or "")
    if log_callback:
        try:
            log_callback(text)
        except Exception:
            pass
    try:
        logger.info(text)
    except Exception:
        pass


def _summarize_raw_content(content, head=2000, tail=800):
    """长返回做「头+尾」摘要：JSON 被 max_completion_tokens 截断时，尾部才是关键证据。"""
    text = "" if content is None else str(content)
    if len(text) <= head + tail:
        return text
    omitted = len(text) - head - tail
    return f"{text[:head]}\n...<已省略 {omitted} 字符>...\n{text[-tail:]}"


_STEP_FAILURE_MARKERS = ("发生错误:", "请求错误:", "JSON 解析失败:", "响应 content 为 None", "响应异常")


def _looks_like_step_failure(text):
    return any(marker in str(text or "") for marker in _STEP_FAILURE_MARKERS)


def _safe_json_from_response(response, log_callback=None, step_label="Step"):
    """安全地从 API response 中解析 JSON，并在失败时记录原始响应信息用于调试。"""
    try:
        choice = response.choices[0]
    except (IndexError, AttributeError) as e:
        raw_str = str(response)[:2000]
        msg = f"{step_label} 响应异常 (无法获取 choices[0]): {e}\n原始响应(截断): {raw_str}"
        _log_step_diag(msg, log_callback)
        raise ValueError(msg) from e

    finish_reason = getattr(choice, "finish_reason", "unknown")
    _log_step_diag(f"{step_label} finish_reason: {finish_reason}", log_callback)

    content = getattr(choice.message, "content", None)
    if content is None:
        refusal = getattr(choice.message, "refusal", None)
        details = []
        if refusal:
            details.append(f"refusal: {repr(refusal)}")
        details.append(f"finish_reason: {finish_reason}")
        details.append("content 为 None，可能是 API 安全过滤或模型拒绝响应")
        msg = f"{step_label} 响应 content 为 None ({', '.join(details)})"
        _log_step_diag(msg, log_callback)
        raise ValueError(msg)

    content_preview = _summarize_raw_content(content)
    _log_step_diag(f"{step_label} 原始响应内容(截断): {content_preview}", log_callback)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        # 模型可能先输出前置 status/message 对象再接真正的结果对象，导致整段解析失败。
        # 尝试从文本中提取候选并挑选最像结果对象的那个。
        candidates = _extract_json_candidates_from_text(content)
        if candidates:
            best = max(candidates, key=lambda c: _score_analysis_json_candidate(c, step_label))
            _log_step_diag(
                f"{step_label} 原始响应含拼接/前置文本，已自动解析出结果对象 "
                f"(候选 {len(candidates)} 个)",
                log_callback,
            )
            return best
        msg = (
            f"{step_label} JSON 解析失败: {e}\n"
            f"原始内容总长度: {len(str(content))} 字符\n"
            f"原始内容(头+尾): {_summarize_raw_content(content, head=4000, tail=1500)}"
        )
        _log_step_diag(msg, log_callback)
        raise

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
    """单图分析专用的图片编码（保留原有 quality=100 与日志行为）。

    实际实现已抽到公共模块 utils/image_encoding.py，这里仅做薄封装以维持
    本模块原有调用契约与输出行为（quality=100、可选 log_callback）。
    """
    return _base_compress_and_encode_image(
        image_source, max_dim=max_dim, quality=100, log_callback=log_callback
    )

def step_1_analyze_image(image_source, client, model_name, log_callback=None, booru_tag_limit=30, local_booru_tags=None, pixiv_candidates=None, extra_llm_prompt="", timeout_seconds=120, status_callback=None, cancel_check=None):
    mime_type, base64_image = compress_and_encode_image(image_source, log_callback=log_callback)
    if not base64_image:
        if log_callback:
            log_callback("Step 1 失败: 图片压缩或编码失败")
        return None
    try:
        analyze_prompt = get_style_analyze_prompt(booru_tag_limit)
        analyze_prompt = merge_prompt_with_local_booru_tags(analyze_prompt, local_booru_tags)
        analyze_prompt = merge_prompt_with_pixiv_tag_hints(analyze_prompt, pixiv_candidates)
        analyze_prompt = append_extra_llm_prompt(analyze_prompt, extra_llm_prompt)
        system_prompt = _load_system_prompt()
        response = call_with_retry(
            lambda: client.chat.completions.create(
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
            ),
            settings=load_retry_settings(),
            step_label="Step 1 Vision 请求",
            log_callback=log_callback,
            cancel_check=cancel_check,
        )
        parsed = _safe_json_from_response(response, log_callback=log_callback, step_label="Step 1")
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

def step_2_refine_description(original_json_data, client, model_name, booru_tag_limit=30, extra_llm_prompt="", timeout_seconds=120, status_callback=None, log_callback=None, cancel_check=None):
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
        response = call_with_retry(
            lambda: client.chat.completions.create(
                model=model_name,
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": refine_prompt}
                ],
                temperature=0.7, max_completion_tokens=16384, timeout=timeout_seconds
            ),
            settings=load_retry_settings(),
            step_label="Step 2 refine 请求",
            log_callback=log_callback,
            cancel_check=cancel_check,
        )
        final_result_json = _safe_json_from_response(response, log_callback=log_callback, step_label="Step 2")
        final_result_json = _normalize_analysis_result(final_result_json, fallback_data=original_json_data, booru_tag_limit=booru_tag_limit)
        # 将原始描述也存入最终结果，方便后续对比或同时生成
        final_result_json["original_english_description"] = original_description
        return final_result_json
    except Exception as e:
        _log_step_diag(f"Step 2 二次加工时发生错误: {e}", log_callback)
        if status_callback:
            status_callback("timeout" if _is_timeout_error(e) else "error")
        return None

def step_3_check_outfit_consistency(final_json_data, client, model_name, timeout_seconds=120, outfit_style_override="", status_callback=None, log_callback=None, cancel_check=None):
    fallback_data = dict(final_json_data or {})
    refine_prompt = str(fallback_data.get("english_description") or "").strip()
    original_prompt = str(fallback_data.get("original_english_description") or "").strip()
    pixiv_tags = fallback_data.get("pixiv_tags", [])
    outfit_style_override = str(outfit_style_override or "").strip()
    if not refine_prompt and not original_prompt:
        return fallback_data

    prompt_payload = {
        "english_description": refine_prompt,
        "original_english_description": original_prompt,
        "pixiv_tags": pixiv_tags,
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
        response = call_with_retry(
            lambda: client.chat.completions.create(
                model=model_name,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_completion_tokens=8192,
                timeout=timeout_seconds
            ),
            settings=load_retry_settings(),
            step_label="Step 3 服装搭配检查请求",
            log_callback=log_callback,
            cancel_check=cancel_check,
        )
        parsed = _safe_json_from_response(response, log_callback=log_callback, step_label="Step 3")
        checked_refine = str(parsed.get("english_description") or refine_prompt).strip()
        checked_original = str(parsed.get("original_english_description") or original_prompt).strip()
        checked_pixiv_tags = parsed.get("pixiv_tags")
        if not isinstance(checked_pixiv_tags, list):
            checked_pixiv_tags = pixiv_tags
        has_person = _to_bool(parsed.get("has_person"), default=bool(refine_prompt or original_prompt))
        modified = _to_bool(parsed.get("modified"), default=False)
        if checked_refine == refine_prompt and checked_original == original_prompt and checked_pixiv_tags == pixiv_tags:
            modified = False

        result = dict(fallback_data)
        result["outfit_check_has_person"] = has_person
        result["outfit_check_modified"] = modified
        result["outfit_check_reason"] = str(parsed.get("reason") or "").strip()

        if modified:
            result["english_description_before_outfit_check"] = refine_prompt
            result["original_english_description_before_outfit_check"] = original_prompt
            result["pixiv_tags_before_outfit_check"] = list(pixiv_tags)
            result["english_description"] = checked_refine
            result["original_english_description"] = checked_original
            result["pixiv_tags"] = checked_pixiv_tags
        return result
    except Exception as e:
        _log_step_diag(f"Step 3 服装搭配检查时发生错误: {e}", log_callback)
        if status_callback:
            status_callback("timeout" if _is_timeout_error(e) else "error")
        return None


def step_4_remove_photo_style(final_json_data, client, model_name, timeout_seconds=120, status_callback=None, log_callback=None, cancel_check=None):
    fallback_data = dict(final_json_data or {})
    english_description = str(fallback_data.get("english_description") or "").strip()
    original_english_description = str(fallback_data.get("original_english_description") or "").strip()
    short_description = str(fallback_data.get("short_description") or "").strip()
    booru_tags = fallback_data.get("booru-tags", [])
    japanese_title = str(fallback_data.get("japanese_title") or "").strip()
    chinese_title = str(fallback_data.get("chinese_title") or "").strip()
    pixiv_tags = fallback_data.get("pixiv_tags", [])
    aspect_ratio = str(fallback_data.get("aspect_ratio") or "").strip()

    prompt_payload = {
        "english_description": english_description,
        "original_english_description": original_english_description,
        "short_description": short_description,
        "booru-tags": booru_tags,
        "japanese_title": japanese_title,
        "chinese_title": chinese_title,
        "pixiv_tags": pixiv_tags,
        "aspect_ratio": aspect_ratio,
    }
    user_prompt = render_prompt_file(
        REMOVE_PHOTO_STYLE_PROMPT_FILE,
        {
            "input_json": json.dumps(prompt_payload, ensure_ascii=False, indent=2),
        }
    ).strip()

    try:
        system_prompt = _load_system_prompt()
        response = call_with_retry(
            lambda: client.chat.completions.create(
                model=model_name,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_completion_tokens=8192,
                timeout=timeout_seconds
            ),
            settings=load_retry_settings(),
            step_label="Step 4 去除照片风格请求",
            log_callback=log_callback,
            cancel_check=cancel_check,
        )
        parsed = _safe_json_from_response(response, log_callback=log_callback, step_label="Step 4")

        result = dict(fallback_data)
        result["english_description_before_remove_photo"] = english_description
        result["original_english_description_before_remove_photo"] = original_english_description
        result["short_description_before_remove_photo"] = short_description
        result["booru_tags_before_remove_photo"] = list(booru_tags)
        result["pixiv_tags_before_remove_photo"] = list(pixiv_tags)

        new_desc = str(parsed.get("english_description") or english_description).strip()
        new_orig_desc = str(parsed.get("original_english_description") or original_english_description).strip()
        new_short = str(parsed.get("short_description") or short_description).strip()
        new_booru = parsed.get("booru-tags")
        if not isinstance(new_booru, list):
            new_booru = booru_tags
        new_pixiv_tags = parsed.get("pixiv_tags")
        if not isinstance(new_pixiv_tags, list):
            new_pixiv_tags = pixiv_tags

        result["english_description"] = new_desc
        result["original_english_description"] = new_orig_desc
        result["short_description"] = new_short
        result["booru-tags"] = new_booru
        result["pixiv_tags"] = new_pixiv_tags

        return result
    except Exception as e:
        _log_step_diag(f"Step 4 去除照片风格时发生错误: {e}", log_callback)
        if status_callback:
            status_callback("timeout" if _is_timeout_error(e) else "error")
        return None


def step_5_recompute_pixiv_tags(final_json_data, client, model_name, timeout_seconds=120, status_callback=None, log_callback=None, cancel_check=None):
    """Step 5: 基于最终英文描述重新计算 pixiv_tags，确保标签准确反映图片中实际出现的视觉内容。

    当启用了服装搭配检查或去除照片风格等优化选项后，描述与标签可能产生漂移，
    本步骤以最终描述为唯一依据重新生成 pixiv_tags，纠正标签与描述/图片不符的问题。
    """
    fallback_data = dict(final_json_data or {})
    english_description = str(fallback_data.get("english_description") or "").strip()
    short_description = str(fallback_data.get("short_description") or "").strip()
    booru_tags = fallback_data.get("booru-tags", [])
    pixiv_tags = fallback_data.get("pixiv_tags", [])
    if not english_description and not short_description:
        return fallback_data

    prompt_payload = {
        "english_description": english_description,
        "short_description": short_description,
        "booru-tags": booru_tags,
        "pixiv_tags": pixiv_tags,
    }
    user_prompt = render_prompt_file(
        RECOMPUTE_PIXIV_TAGS_PROMPT_FILE,
        {
            "input_json": json.dumps(prompt_payload, ensure_ascii=False, indent=2),
        }
    ).strip()

    try:
        system_prompt = _load_system_prompt()
        response = call_with_retry(
            lambda: client.chat.completions.create(
                model=model_name,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_completion_tokens=8192,
                timeout=timeout_seconds
            ),
            settings=load_retry_settings(),
            step_label="Step 5 重算 pixiv_tags 请求",
            log_callback=log_callback,
            cancel_check=cancel_check,
        )
        parsed = _safe_json_from_response(response, log_callback=log_callback, step_label="Step 5")

        result = dict(fallback_data)
        result["pixiv_tags_before_recompute"] = list(pixiv_tags)

        new_pixiv_tags = parsed.get("pixiv_tags")
        if not isinstance(new_pixiv_tags, list):
            new_pixiv_tags = pixiv_tags
        new_pixiv_tags = [str(tag).strip() for tag in new_pixiv_tags if str(tag).strip()]

        result["pixiv_tags"] = new_pixiv_tags
        result["pixiv_tags_recompute_reason"] = str(parsed.get("reason") or "").strip()
        return result
    except Exception as e:
        _log_step_diag(f"Step 5 重新计算 pixiv_tags 时发生错误: {e}", log_callback)
        if status_callback:
            status_callback("timeout" if _is_timeout_error(e) else "error")
        return None


class WorkerThread(QThread):
    log_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(dict)

    def __init__(self, image_source, api_key, base_url, model_name, enable_refine=True, booru_tag_limit=30, extra_llm_prompt="", timeout_seconds=120, enable_outfit_check=False, outfit_style_override="", remove_photo_style=False):
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
        self.remove_photo_style = bool(remove_photo_style)
        self.last_status = "idle"
        self.last_error = ""
        # 右键「重跑并生图」时由界面指定本次分析完成后要生成的提示词类型（original / refined）
        self.meta_force_gen_targets = []
        self._force_cancel_requested = False

    def request_cancel(self, force=False):
        self._force_cancel_requested = bool(force)
        self.requestInterruption()

    def run(self):
        self.last_status = "running"
        self.last_error = ""

        def step_log(message):
            """步骤级日志：进 GUI 面板，同时记录最近一条失败原因供最终汇总使用。"""
            text = str(message)
            if _looks_like_step_failure(text):
                self.last_error = text
            self.log_signal.emit(text)

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

        # 基于本地 booru 标签匹配 Pixiv 日文标签候选（复用已计算的 booru 标签，避免重复 WD14 推理）
        self.log_signal.emit("正在匹配本地 Pixiv 标签候选...")
        pixiv_candidates = get_local_pixiv_tag_candidates(
            local_booru_tags,
            log_callback=self.log_signal.emit
        )
        if pixiv_candidates:
            self.log_signal.emit(f"本地 Pixiv 标签候选 ({len(pixiv_candidates)} 个): {', '.join(pixiv_candidates)}")
        else:
            self.log_signal.emit("本地 Pixiv 标签候选为空")
        if self.isInterruptionRequested():
            self.last_status = "cancelled"
            self.log_signal.emit("任务已取消（已停止后续分析步骤）。")
            self.finish_signal.emit({})
            return

        if self.extra_llm_prompt:
            self.log_signal.emit("已启用附加 prompts，LLM 请求将追加重点关注要求")
        self.log_signal.emit(f"正在使用模型 [{self.model_name}] 开始 Step 1: 读取并压缩图片，发送 Vision 请求...")
        stage_status = {"value": "ok"}
        initial_result = step_1_analyze_image(
            self.image_source,
            client,
            self.model_name,
            log_callback=step_log,
            booru_tag_limit=self.booru_tag_limit,
            local_booru_tags=local_booru_tags,
            pixiv_candidates=pixiv_candidates,
            extra_llm_prompt=self.extra_llm_prompt,
            timeout_seconds=self.timeout_seconds,
            status_callback=lambda status: stage_status.update({"value": status}),
            cancel_check=self.isInterruptionRequested
        )
        
        if initial_result:
            # 方式1：只有 Step 1 基于原图视觉分析得到的结果才是“原图真实内容”的标签来源；
            # 精修/重算阶段会按目标审美改写描述（例如注入蕾丝宝石手套），因此最终标签一律以这里为准。
            original_pixiv_tags = list(initial_result.get("pixiv_tags", []) or [])
            original_booru_tags = list(initial_result.get("booru-tags", []) or [])
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
                    status_callback=lambda status: stage_status.update({"value": status}),
                    log_callback=step_log,
                    cancel_check=self.isInterruptionRequested
                )
                if final_result:
                    final_result["aspect_ratio"] = calculate_closest_aspect_ratio(self.image_source)
                    initial_tags = original_pixiv_tags
                    refined_tags = final_result.get("pixiv_tags", [])
                    final_result["pixiv_tags_first"] = initial_tags
                    final_result["pixiv_tags_second"] = refined_tags
                    # 方式1：最终 pixiv_tags 以原图为准；refine 按目标审美改写后的标签仅保留在 pixiv_tags_second 供对比，
                    # 不再用它覆盖分析结果（否则会出现原图没有手套却标上“レース手袋”的问题）。
                    final_result["pixiv_tags"] = initial_tags if initial_tags else refined_tags
                    if original_booru_tags:
                        final_result["booru-tags"] = original_booru_tags
                    if local_booru_tags:
                        final_result["booru_tags_local_candidate"] = normalize_booru_tags(local_booru_tags, limit=self.booru_tag_limit, output_style="space")
                else:
                    if self.isInterruptionRequested() or self._force_cancel_requested:
                        self.last_status = "cancelled"
                        self.log_signal.emit("任务已取消（Step 2 未完成）。")
                        self.finish_signal.emit({})
                        return
                    self.last_status = "timeout" if stage_status.get("value") == "timeout" else "error"
                    self.log_signal.emit(
                        f"Step 2 执行失败（{self.last_status}），后续步骤已终止；原因见上方 Step 2 诊断信息。"
                    )
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
                    status_callback=lambda status: outfit_stage_status.update({"value": status}),
                    log_callback=step_log,
                    cancel_check=self.isInterruptionRequested
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
                    if self.isInterruptionRequested() or self._force_cancel_requested:
                        self.last_status = "cancelled"
                        self.log_signal.emit("任务已取消（服装搭配检查未完成）。")
                        self.finish_signal.emit({})
                        return
                    self.log_signal.emit("Step 3 执行失败，已保留 Step 2 的 prompts 结果。")
            if final_result and self.remove_photo_style:
                if self.isInterruptionRequested():
                    self.last_status = "cancelled"
                    self.log_signal.emit("任务已取消（去除照片风格未执行）。")
                    self.finish_signal.emit({})
                    return
                self.log_signal.emit("正在开始 Step 4: 去除照片风格相关提示词...")
                photo_style_stage_status = {"value": "ok"}
                photo_style_result = step_4_remove_photo_style(
                    final_result,
                    client,
                    self.model_name,
                    timeout_seconds=self.timeout_seconds,
                    status_callback=lambda status: photo_style_stage_status.update({"value": status}),
                    log_callback=step_log,
                    cancel_check=self.isInterruptionRequested
                )
                if photo_style_result:
                    self.log_signal.emit("Step 4 完成。已移除照片风格相关提示词。")
                    final_result = photo_style_result
                else:
                    if self.isInterruptionRequested() or self._force_cancel_requested:
                        self.last_status = "cancelled"
                        self.log_signal.emit("任务已取消（去除照片风格未完成）。")
                        self.finish_signal.emit({})
                        return
                    self.log_signal.emit("Step 4 执行失败，已保留之前的 prompts 结果。")
            # Step 5: 当启用了任一优化选项（服装搭配检查 / 去除照片风格）时，
            # 基于最终描述重新计算 pixiv_tags，纠正多阶段修改导致的标签漂移
            if final_result and (self.enable_outfit_check or self.remove_photo_style):
                if self.isInterruptionRequested():
                    self.last_status = "cancelled"
                    self.log_signal.emit("任务已取消（pixiv_tags 重新计算未执行）。")
                    self.finish_signal.emit({})
                    return
                self.log_signal.emit("正在开始 Step 5: 基于最终描述重新计算 pixiv_tags...")
                recompute_stage_status = {"value": "ok"}
                recomputed_result = step_5_recompute_pixiv_tags(
                    final_result,
                    client,
                    self.model_name,
                    timeout_seconds=self.timeout_seconds,
                    status_callback=lambda status: recompute_stage_status.update({"value": status}),
                    log_callback=step_log,
                    cancel_check=self.isInterruptionRequested
                )
                if recomputed_result:
                    self.log_signal.emit("Step 5 完成。pixiv_tags 已根据最终描述重新计算。")
                    final_result = recomputed_result
                else:
                    if self.isInterruptionRequested() or self._force_cancel_requested:
                        self.last_status = "cancelled"
                        self.log_signal.emit("任务已取消（pixiv_tags 重算未完成）。")
                        self.finish_signal.emit({})
                        return
                    self.log_signal.emit("Step 5 执行失败，已保留之前的 pixiv_tags 结果。")
            # 方式1：无论之后是否启用服装搭配检查 / 去除照片风格 / 重算 pixiv_tags，
            # 最终标签一律以原图（Step 1）为准，避免这些阶段基于“被注入目标审美后的描述”
            # 重新生成的标签（如蕾丝宝石手套）污染分析结果。
            if final_result:
                if not final_result.get("pixiv_tags_second"):
                    derived_tags = final_result.get("pixiv_tags", []) or []
                    if derived_tags and derived_tags != original_pixiv_tags:
                        final_result["pixiv_tags_second"] = derived_tags
                if not final_result.get("pixiv_tags_first"):
                    final_result["pixiv_tags_first"] = original_pixiv_tags
                if original_pixiv_tags:
                    final_result["pixiv_tags"] = original_pixiv_tags
                if original_booru_tags:
                    final_result["booru-tags"] = original_booru_tags
                # gpt-image 通道专用短提示词：单独字段，与 Gemini 用的长 prompts 分开（不混用）。
                # 两档：full（≤1400，内容优先）与 short（≤500，要挂画风参考图时用）
                try:
                    from utils.analysis_gpt_prompt import (FIELD_KEY, FIELD_MAX_CHARS, SHORT_FIELD_KEY,
                                                           SHORT_FIELD_MAX_CHARS, build_gpt_image_prompt)
                    desc = str(final_result.get("english_description") or "").strip()
                    if desc:
                        text_cfg = {"base_url": self.base_url, "api_key": self.api_key, "model": self.model_name}
                        self.log_signal.emit("正在生成 gpt-image 专用短提示词（完整档 / 短锚档）...")
                        field = build_gpt_image_prompt(desc, text_cfg=text_cfg, max_chars=FIELD_MAX_CHARS,
                                                       tier="full", log_callback=self.log_signal.emit)
                        if field:
                            final_result[FIELD_KEY] = field
                            self.log_signal.emit(f"gpt-image 完整档 {len(field)} 字符")
                        short_field = build_gpt_image_prompt(desc, text_cfg=text_cfg,
                                                             max_chars=SHORT_FIELD_MAX_CHARS, tier="short",
                                                             log_callback=self.log_signal.emit)
                        if short_field:
                            final_result[SHORT_FIELD_KEY] = short_field
                            self.log_signal.emit(f"gpt-image 短锚档 {len(short_field)} 字符")
                except Exception as exc:  # noqa: BLE001 - 附加步骤失败不影响分析结果
                    self.log_signal.emit(f"gpt-image 短提示词生成失败，已跳过: {type(exc).__name__}: {exc}")
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

    def __init__(self, prompt, model_name, aspect_ratio, instructions, api_type=None, resolution=None, image_paths=None, verbose_debug=False, file_prefix=None, post_instructions=None):
        super().__init__()
        self.prompt = prompt
        self.model_name = model_name
        self.aspect_ratio = aspect_ratio
        self.instructions = instructions
        self.post_instructions = post_instructions or ""
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
                    return_metadata=self.verbose_debug,
                    cancel_check=lambda: self.isInterruptionRequested(),
                    post_instructions=self.post_instructions
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
                    return_metadata=self.verbose_debug,
                    cancel_check=lambda: self.isInterruptionRequested(),
                    post_instructions=self.post_instructions
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

class GptImageGenWorkerThread(QThread):
    """gpt-image-2 通道的生图线程：分析内容锚 + 画风参考图 → 出图 → 按勾选跑工序加工。

    与 Gemini 通道（ImageGenWorkerThread）的区别：
    - 不挂原分析图，只挂画风参考图；
    - 提示词 = 画风 prompt_gpt + 与 Gemini 通道同源的分析描述；
    - 默认用源图 + 完整画风图重绘，必要时做一次 source-only 质量修订，再做身份定点修订。
    见 docs/gpt-image-tid-style/BEST-PIPELINE.md。
    """
    log_signal = pyqtSignal(str)
    finish_signal = pyqtSignal(list)

    def __init__(self, request_payload, steps=None, firmware="", size="1024x1536", quality="high",
                 output_format="png", file_prefix=None, model_name="gpt-image-2", api_type="aigc-2d-gpt",
                 mode="generate", style_ref_path="", style_clauses=None, analysis_result=None):
        super().__init__()
        self.request_payload = dict(request_payload or {})
        self.steps = dict(steps or {})
        self.firmware = str(firmware or "")
        self.size = size
        self.quality = quality
        self.output_format = output_format
        self.file_prefix = str(file_prefix or "").strip()
        self.model_name = model_name
        self.api_type = api_type
        # 首图走「新建图片」（/images/generations + 参考图字段），不走 /images/edits（用户 2026-09-24 要求）
        self.mode = str(mode or "generate")
        # 重绘的第二张参考：画风参考图 + 画风渲染条款（`reference_mode="style"` 时用得上）
        self.style_ref_path = str(style_ref_path or "")
        self.style_clauses = [str(c) for c in (style_clauses or []) if str(c).strip()]
        self.analysis_result = dict(analysis_result or {})
        self.last_status = "idle"

    def request_cancel(self):
        self.requestInterruption()

    def run(self):
        from modules.others.api_backend import generate_image_aigc2d_gpt
        from utils.analysis_gen import first_pass_sub_dir, run_gpt_image_pipeline, save_generation_manifest
        self.last_status = "running"
        if self.isInterruptionRequested():
            self.last_status = "cancelled"
            self.finish_signal.emit([])
            return
        prompt = self.request_payload.get("prompt", "")
        images = list(self.request_payload.get("image_paths") or [])
        active_steps = [k for k, v in self.steps.items() if isinstance(v, dict) and v.get("enabled")]
        self.log_signal.emit("\n🚀 gpt-image-2 生图（%s，提示词 %d 字符，参考图 %d 张：%s）"
                             % (self.model_name, len(prompt), len(images),
                                "画风参考图" if images else "无"))
        try:
            saved = generate_image_aigc2d_gpt(
                prompt=prompt, image_paths=images, model=self.model_name, size=self.size,
                quality=self.quality, output_format=self.output_format, n=1,
                api_type=self.api_type, file_prefix=self.file_prefix or "analysis-gpt",
                # 没有后续工序时首图就是最终产物 → 直接落 data/<日期>/（方便发布）；
                # 有工序时它是中间产物，进 analysis-gpt-image/ 子目录。
                save_sub_dir=first_pass_sub_dir(self.steps),
                return_metadata=False,
                mode=self.mode,
            )
        except Exception as exc:  # noqa: BLE001
            self.log_signal.emit(f"❌ gpt-image-2 生图异常: {exc}")
            self.last_status = "error"
            self.finish_signal.emit([])
            return
        saved = [p for p in (saved or []) if p]
        if not saved:
            self.last_status = "error"
            self.finish_signal.emit([])
            return
        first_image = saved[0]
        def record_request(outputs=None):
            try:
                if os.path.isfile(first_image):
                    target = save_generation_manifest(first_image, self.request_payload,
                        model=self.model_name, size=self.size, quality=self.quality, mode=self.mode,
                        steps=self.steps, firmware=self.firmware, outputs=outputs)
                    self.log_signal.emit(f"[gpt 通道] 首图与工序快照：{target}")
            except Exception as exc:
                self.log_signal.emit(f"[gpt 通道] 请求快照保存失败：{exc}")
        record_request()
        active = [k for k, v in self.steps.items() if isinstance(v, dict) and v.get("enabled")]
        if active and not self.isInterruptionRequested():
            self.log_signal.emit("🧩 工序加工: " + " → ".join(active))
            try:
                saved = run_gpt_image_pipeline(saved, self.steps, firmware=self.firmware,
                                               log_callback=self.log_signal.emit,
                                               style_ref_path=self.style_ref_path,
                                               style_clauses=self.style_clauses) or saved
            except Exception as exc:  # noqa: BLE001 - 工序失败仍保留生图产物
                self.log_signal.emit(f"⚠️ 工序加工失败，已保留生图产物: {type(exc).__name__}: {exc}")
        # 完整画风图只用于第一次重绘。第二次质量修订改用「当前图 + GPT 首图」：
        # 审计从画风图提取具体差异写进文字，但不再次发送画风图，避免参考角色/场景二次侵入。
        if ((self.steps.get("repaint") or {}).get("enabled") and saved and self.style_ref_path
                and os.path.isfile(self.style_ref_path) and not self.isInterruptionRequested()):
            try:
                import json as _json
                from utils.refine_quality import (audit_refine_quality, build_quality_correction_prompt,
                                                  should_refine_quality)
                from modules.others.api_backend import generate_image_repaint
                quality_dir = os.path.dirname(os.path.abspath(saved[-1]))
                actual_prompt = str(self.request_payload.get("prompt") or "")
                quality = audit_refine_quality(first_image, saved[-1], self.style_ref_path,
                                               first_pass_prompt=actual_prompt)
                with open(os.path.join(quality_dir, "refine-quality-audit-0.json"),
                          "w", encoding="utf-8") as f:
                    _json.dump(quality, f, ensure_ascii=False, indent=2)
                if should_refine_quality(quality):
                    quality_prompt = build_quality_correction_prompt(quality)
                    refined = generate_image_repaint(
                        [saved[-1]], resolution="2K", aspect_ratio="auto", prompt=quality_prompt,
                        use_detail_suffix=False, save_sub_dir=quality_dir,
                        file_prefix=(self.file_prefix or "analysis-gpt") + "-quality-refine",
                        extra_reference_paths=[first_image]) or []
                    if refined:
                        saved = [refined[-1]]
                        quality_after = audit_refine_quality(
                            first_image, saved[-1], self.style_ref_path,
                            first_pass_prompt=actual_prompt)
                        with open(os.path.join(quality_dir, "refine-quality-audit-1.json"),
                                  "w", encoding="utf-8") as f:
                            _json.dump(quality_after, f, ensure_ascii=False, indent=2)
                        self.log_signal.emit(
                            "[质量门禁] 已用当前图 + GPT 首图完成一次文字驱动修订；"
                            "画风图未再次发送。")
                else:
                    self.log_signal.emit("[质量门禁] 未发现需定点修复的高置信质量问题。")
            except Exception as exc:
                self.log_signal.emit(
                    f"[质量门禁] 审计/修订失败，保留当前重绘图: {type(exc).__name__}: {exc}")
        # 重绘后的身份处理最多两轮定点修订。每轮都拿当前图与实际首图 prompt 重新审计，
        # 只修上一轮仍不符合的稳定身份特征；最终不回退 GPT 首图，以保留线条修复。
        if ((self.steps.get("repaint") or {}).get("enabled") and saved and self.analysis_result
                and not self.isInterruptionRequested()):
            try:
                import json as _json
                from utils.identity_audit import (audit_image_identity, build_identity_correction_prompt,
                                                  identity_gate_action)
                from modules.others.api_backend import generate_image_repaint
                audit_dir = os.path.dirname(os.path.abspath(saved[-1]))
                current = saved[-1]
                actual_prompt = str(self.request_payload.get("prompt") or "")
                audit = audit_image_identity(current, self.analysis_result, expected_prompt=actual_prompt)
                with open(os.path.join(audit_dir, "identity-audit-0.json"), "w", encoding="utf-8") as f:
                    _json.dump(audit, f, ensure_ascii=False, indent=2)
                for correction_round in range(1, 3):
                    action = identity_gate_action(audit)
                    if action != "correct":
                        if action == "review":
                            self.log_signal.emit("[身份门禁] 审计结果需人工复核；保留当前图。")
                        break
                    correction_prompt = build_identity_correction_prompt(
                        audit, iteration=correction_round, max_iterations=2)
                    corrected = generate_image_repaint(
                        [current], resolution="2K", aspect_ratio="auto", prompt=correction_prompt,
                        use_detail_suffix=False, save_sub_dir=audit_dir,
                        file_prefix=(self.file_prefix or "analysis-gpt") + f"-identity-correct-{correction_round}") or []
                    if not corrected:
                        break
                    current = corrected[-1]
                    audit = audit_image_identity(current, self.analysis_result, expected_prompt=actual_prompt)
                    with open(os.path.join(audit_dir, f"identity-audit-{correction_round}.json"),
                              "w", encoding="utf-8") as f:
                        _json.dump(audit, f, ensure_ascii=False, indent=2)
                    saved = [current]
                    self.log_signal.emit(
                        f"[身份门禁] 第 {correction_round}/2 轮定点修订完成；"
                        f"复审{'仍有差异' if audit.get('mismatch') else '通过'}。")
                self.log_signal.emit("[身份门禁] 按不回退策略保留最后一轮图。")
            except Exception as exc:  # 审计故障不能让已经成功的重绘任务失败
                self.log_signal.emit(f"[身份门禁] 审计/定点修订失败，保留当前重绘图: {type(exc).__name__}: {exc}")
        if self.isRequestInterruption_requested_safe():
            self.last_status = "cancelled"
            self.finish_signal.emit([])
            return
        self.last_status = "success"
        record_request(saved)
        self.finish_signal.emit([p for p in saved if p])

    def isRequestInterruption_requested_safe(self):
        return self.isInterruptionRequested()


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
        source_path = str(self.task_record.get("source_path") or "").strip()
        if source_path:
            origin_text = "剪贴板快照" if str(self.task_record.get("source_origin") or "") == "clipboard" else "原图"
            lines.append(f"{origin_text}路径（右键可重跑）: {source_path}")
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
    def __init__(self, config_getter_func, img_config_getter_func, styles_getter_func, save_img_cfg_callback, ar_policy_getter_func=None, nsfw_default_getter_func=None, nsfw_changed_callback=None, booru_tag_limit_getter_func=None, timeout_getter_func=None, upscale_options_getter_func=None, upscale_options_changed_callback=None, outfit_check_default_getter_func=None, outfit_check_changed_callback=None, remove_photo_style_default_getter_func=None, remove_photo_style_changed_callback=None, outfit_style_history_getter_func=None, outfit_style_default_getter_func=None, outfit_style_changed_callback=None, outfit_style_delete_callback=None, styles_reload_callback=None):
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
        self.get_remove_photo_style_default = remove_photo_style_default_getter_func
        self.on_remove_photo_style_changed = remove_photo_style_changed_callback
        self.get_outfit_style_history = outfit_style_history_getter_func
        self.get_outfit_style_default = outfit_style_default_getter_func
        self.on_outfit_style_changed = outfit_style_changed_callback
        self.on_outfit_style_deleted = outfit_style_delete_callback
        self.on_styles_reload = styles_reload_callback
        
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
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(6)

        # 【布局契约】窗口默认只有 1100x750，而本 Tab 的控件行数远超这个高度：
        # 固定区 = 预览（拖拽落点）+ ② 生成按钮 + 队列 + 日志（用户必须随时看得见，日志以前被
        # gpt 选项行挤成 0 高）；选项区（① 按钮 / 自动生图 / 画风 / 生图通道 / gpt 工序与参数…）
        # 放进滚动容器 —— 它们是一次性设置，滚动看即可。
        # 注意：滚动容器**不接受**拖拽，这样拖图落到选项区时事件会继续上抛到本 Widget（拖拽落点不变）。
        self.image_label = QLabel("请将图片拖拽至此，\n或在窗口内按 Ctrl+V 粘贴")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color : #f0f0f0; border: 2px dashed #aaa; font-size: 16px; }")
        self.image_label.setMinimumHeight(160)
        outer_layout.addWidget(self.image_label)

        self.controls_scroll = QScrollArea()
        self.controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setMinimumHeight(100)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.controls_body = QWidget()
        layout = QVBoxLayout(self.controls_body)
        layout.setContentsMargins(0, 0, 6, 0)
        layout.setSpacing(4)      # 行距压紧：选项区能少滚一点（内容高度直接省掉 20+ 像素）
        self.controls_scroll.setWidget(self.controls_body)
        # 选项区**按内容高度**取尺寸（不给 stretch）：窗口拉大/最大化时多出来的高度要归
        # 队列和日志，而不是把每一行拉高 —— 否则按钮之间被撑出大片空白（用户 2026-09-24 反馈）。
        outer_layout.addWidget(self.controls_scroll, 0)

        self.send_btn = QPushButton("① 发送并生成分析描述")
        self.send_btn.setFixedHeight(40)
        self.send_btn.clicked.connect(self.process_image)
        self.send_btn.setEnabled(False) 
        layout.addWidget(self.send_btn)

        # 终止分析：重试等待期间也能立刻中断
        self.cancel_analysis_btn = QPushButton("🛑 终止当前分析（含等待中的重试）")
        self.cancel_analysis_btn.setToolTip(
            "请求终止所有正在进行的分析任务；若此刻在等待重试，会立刻中断等待。\n"
            "已经发出、正在等待对面返回的那一次请求，要等它自己返回或超时"
            "（「设置 → 图片生成 API → 请求超时时间」）才会真正结束。"
        )
        self.cancel_analysis_btn.setEnabled(False)
        self.cancel_analysis_btn.clicked.connect(self.cancel_active_analysis_tasks)
        layout.addWidget(self.cancel_analysis_btn)

        # 目录批量选择按钮
        self.dir_batch_btn = QPushButton("📁 从目录批量选择图片并分析")
        self.dir_batch_btn.setFixedHeight(36)
        self.dir_batch_btn.clicked.connect(self._open_directory_batch_selector)
        layout.addWidget(self.dir_batch_btn)

        # 【新增】两项自动生成图片的勾选框
        auto_gen_layout = QHBoxLayout()
        self.auto_gen_orig_cb = QCheckBox("分析完成后立即生成图片（基于原始提示词）")
        self.auto_gen_ref_cb = QCheckBox("分析完成后立即生成图片（基于优化提示词）")
        auto_gen_layout.addWidget(self.auto_gen_orig_cb)
        auto_gen_layout.addWidget(self.auto_gen_ref_cb)
        layout.addLayout(auto_gen_layout)

        # 保存到原图同目录
        self.save_to_source_dir_cb = QCheckBox("分析结果保存到原图同目录（不生成图片）")
        self.save_to_source_dir_cb.setToolTip(
            "勾选后，分析结果的 JSON 和 TXT 将保存到原图所在目录，并同步文件修改时间。\n"
            "此时只分析不生成图片，JSON 可用于直接拖入投稿 Server。\n"
            "仅加载硬盘文件时可用，剪贴板图片无实际文件路径。"
        )
        self.save_to_source_dir_cb.setEnabled(False)
        self.save_to_source_dir_cb.toggled.connect(self._on_save_to_source_dir_toggled)
        layout.addWidget(self.save_to_source_dir_cb)

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
        self.remove_photo_style_cb = QCheckBox("去除照片风格")
        self.remove_photo_style_cb.setToolTip("勾选后，在最终产生 prompts 前，再次提交分析模型，去除所有照片类提示词（如 realistic, photo-like 等），但保持原文其他内容不变。\n另外：若检测到自拍挡脸（手/手机/相机/头发/口罩/道具遮住面部），会删除对应遮挡描述与标签，替换为不挡脸的其他动作，只改动作，不动身份、服装与场景。")
        self.remove_photo_style_cb.setChecked(bool(self.get_remove_photo_style_default()) if self.get_remove_photo_style_default else False)
        self.remove_photo_style_cb.toggled.connect(self.on_remove_photo_style_toggled)
        nsfw_layout.addWidget(self.remove_photo_style_cb)
        nsfw_layout.addStretch()
        layout.addLayout(nsfw_layout)

        outfit_style_layout = QHBoxLayout()
        outfit_style_layout.addWidget(QLabel("服装风格覆盖:"))
        self.outfit_style_combo = QComboBox()
        self.outfit_style_combo.setEditable(True)
        self.outfit_style_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.outfit_style_completer_model = QStringListModel(self)
        self.outfit_style_completer = QCompleter(self.outfit_style_completer_model, self)
        self.outfit_style_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.outfit_style_completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.outfit_style_completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
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
        style_select_layout.addWidget(QLabel("参考模式:"))
        self.style_ref_mode_combo = StyleRefModeCombo(self)
        self.style_ref_mode_combo.setMaximumWidth(130)
        style_select_layout.addWidget(self.style_ref_mode_combo)
        self.main_style_combo.currentTextChanged.connect(self._on_style_changed)

        # ---- 生图通道：Gemini（老逻辑）/ gpt-image-2（gpt 专用短锚 + 画风参考图 + 工序加工）----
        from PyQt6.QtWidgets import QRadioButton, QButtonGroup
        channel_row = QHBoxLayout()
        channel_row.addWidget(QLabel("生图通道:"))
        self.gen_channel_group = QButtonGroup(self)
        self.gen_channel_gemini = QRadioButton("Gemini")
        self.gen_channel_gpt = QRadioButton("gpt-image-2 + 艺术风格")
        self.gen_channel_gemini.setChecked(True)
        self.gen_channel_group.addButton(self.gen_channel_gemini, 0)
        self.gen_channel_group.addButton(self.gen_channel_gpt, 1)
        channel_row.addWidget(self.gen_channel_gemini)
        channel_row.addWidget(self.gen_channel_gpt)
        channel_row.addStretch(1)
        self.gen_channel_row = QWidget()
        self.gen_channel_row.setLayout(channel_row)
        layout.addWidget(self.gen_channel_row)

        # gpt-image-2 通道的工序勾选（不选该通道时沿用老逻辑，不受影响）
        pp_row = QHBoxLayout()
        self.gpt_pp_repaint = QCheckBox("重绘提线(Gemini)")
        self.gpt_basic_controls = QWidget()
        basic_row = QHBoxLayout(self.gpt_basic_controls)
        basic_row.setContentsMargins(0, 0, 0, 0)
        basic_row.addWidget(self.gpt_pp_repaint)
        self.gpt_advanced_toggle = QCheckBox("高级选项")
        basic_row.addWidget(self.gpt_advanced_toggle)
        self.gpt_reset_recipe = QPushButton("恢复推荐")
        basic_row.addWidget(self.gpt_reset_recipe)
        channel_row.insertWidget(channel_row.count() - 1, self.gpt_basic_controls)
        self.gpt_advanced_toggle.toggled.connect(self._on_gen_channel_changed)
        self.gpt_reset_recipe.clicked.connect(lambda: self._load_gpt_pipeline_ui(
            dict(ANALYSIS_GPT_UI_DEFAULTS, channel="gpt-image")))
        self.gpt_pp_repaint.setToolTip("把 GPT 首图与完整画风图交给 Gemini 提线；随后按首图 prompt 最多做两轮身份定点修订。")

        self.gpt_pp_structure = QCheckBox("结构线叠加")
        self.gpt_pp_local = QCheckBox("局部重绘贴回")
        self.gpt_pp_region = QComboBox()
        self.gpt_pp_region.setMaximumWidth(220)          # 标签很长：限宽 + tooltip 兜住完整解释
        try:
            from utils.post_process import REGION_LABELS
            # 第一项 = 实测最稳的多区域链（§二十八）：单区域 subject_no_face 线条最好但约 1/3 概率重排主体出鬼影
            self.gpt_pp_region.addItem("四区串联（实验）", list(STABLE_LOCAL_REGIONS))
            for _key, _label in REGION_LABELS.items():
                self.gpt_pp_region.addItem(_label, _key)
            self.gpt_pp_region.setCurrentIndex(0)
        except Exception:  # noqa: BLE001
            self.gpt_pp_region.addItem("头发", "hair")
        self.gpt_pp_region.setToolTip(
            "局部重绘覆盖哪些区域。默认「稳定四区」= 人物保脸 + 鞋/靴带 + 束腰 + 大腿袜带，逐个重绘并羽化贴回；\n"
            "实测（docs/gpt-image-tid-style/BEST-PIPELINE.md §二十八）：单区域 subject_no_face 线条指标最好，\n"
            "但同一输入重跑约 1/3 概率把主体重排（保脸区的原脸 + 新身体 = 双人鬼影），所以默认用更稳的四区链。"
        )
        pp_row.addWidget(QLabel("gpt 工序:"))
        pp_row.addWidget(self.gpt_pp_structure)
        pp_row.addWidget(self.gpt_pp_local)
        pp_row.addWidget(self.gpt_pp_region)
        pp_row.addWidget(QLabel("重绘范围:"))
        self.gpt_pp_scope = QComboBox()
        self.gpt_pp_scope.setMaximumWidth(220)           # 同上：范围标签很长，限宽避免把整行撑爆
        try:
            from utils.post_process import REPAINT_SCOPE_LABELS
            for _key, _label in REPAINT_SCOPE_LABELS.items():
                self.gpt_pp_scope.addItem(_label, _key)
            _i = self.gpt_pp_scope.findData("lines_only")
            self.gpt_pp_scope.setCurrentIndex(_i if _i >= 0 else 0)
        except Exception:  # noqa: BLE001
            self.gpt_pp_scope.addItem("只连通线条（不改色不改内容）", "lines_only")
        self.gpt_pp_scope.setToolTip(
            "在整张图上用提示词限制修复范围，不是像素遮罩。所有档位都要求保留身份和设计；模型仍可能改变细节。"
        )
        self.gpt_pp_region.setToolTip("裁切后重绘再贴回，可能产生错位或鬼影；四区串联是历史实验，不作为默认。")
        pp_row.addWidget(self.gpt_pp_scope)
        self.gpt_first_pass_mode = QComboBox()
        self.gpt_first_pass_mode.addItem("首图：新建接口", "generate")
        self.gpt_first_pass_mode.addItem("首图：参考图接口（历史）", "edits")
        self.gpt_first_pass_mode.setMaximumWidth(180)
        self.gpt_first_pass_mode.setToolTip("历史 V6 首图走 edits。新建接口的参考图字段是中转站扩展；两条路效果不保证相同。拒绝请求后不会切换接口重试。")
        pp_row.addWidget(self.gpt_first_pass_mode)
        pp_row.addStretch(1)
        self.gpt_pp_row = QWidget()
        self.gpt_pp_row.setLayout(pp_row)
        layout.addWidget(self.gpt_pp_row)

        # gpt 通道的「参数」行：画质 / 尺寸 / 首图来源 / 色调 / 加墨。
        # 以前这五组各自占一行，切到 gpt 通道就多出 5 行，把下面固定的队列与日志挤没了
        # （用户 2026-09-24 反馈「UI 混乱、日志看不见」）——现在合并成一行，选项一行、参数一行。
        q_row = QHBoxLayout()
        q_row.addWidget(QLabel("gpt 参数:"))
        self.gpt_quality_combo = QComboBox()
        for _label, _val, _tip in (("high 细节最好(推荐)", "high", "输出 token 约为 medium 的 5 倍，细节/发丝/蕾丝最清楚"),
                                   ("medium 快省", "medium", "细节明显少于 high，重绘放大后容易发糊"),
                                   ("low 预览", "low", "只适合构图预览")):
            self.gpt_quality_combo.addItem(_label, _val)
            self.gpt_quality_combo.setItemData(self.gpt_quality_combo.count() - 1, _tip, 3)  # Qt.ToolTipRole
        self.gpt_quality_combo.setToolTip("首图质量：high 的细节 token 约为 medium 的 5 倍（重绘放大后差别明显）。")
        self.gpt_quality_combo.setCurrentIndex(0)
        self.gpt_quality_combo.setMaximumWidth(180)
        q_row.addWidget(self.gpt_quality_combo)

        self.gpt_size_follow_cb = QCheckBox("尺寸跟随输入图")
        self.gpt_size_follow_cb.setChecked(True)
        self.gpt_size_follow_cb.setToolTip(
            "gpt-image-2 只有 1024x1024 / 1536x1024 / 1024x1536 三档尺寸；\n"
            "勾上后按**输入图的实际比例**挑：横图 → 1536x1024，竖图 → 1024x1536，方图 → 1024x1024。\n"
            "取消勾选则固定用 1024x1536。"
        )
        q_row.addWidget(self.gpt_size_follow_cb)

        self.gpt_pp_tone = QCheckBox("色调校准")
        self.gpt_pp_tone.setToolTip(
            "按参考图匹配亮度与饱和度：目标选画风参考图时画面更深更饱和、线条更清楚（画质优先）；\n"
            "选输入照片则更接近照片原色（色彩保真）。只改亮度/饱和度，不动色相。"
        )
        q_row.addWidget(self.gpt_pp_tone)
        self.gpt_pp_tone_target = QComboBox()
        self.gpt_pp_tone_target.addItem("目标=画风参考图", "style")
        self.gpt_pp_tone_target.addItem("目标=输入照片", "photo")
        self.gpt_pp_tone_target.setToolTip("色调校准的目标：画风参考图 = 画质优先；输入照片 = 色彩保真。")
        q_row.addWidget(self.gpt_pp_tone_target)

        self.gpt_pp_ink = QCheckBox("线条加墨")
        self.gpt_pp_ink.setToolTip("只把已有线条压深，让线明显深于局部底色 —— 解决「线条看着稀碎」的问题。")
        q_row.addWidget(self.gpt_pp_ink)
        q_row.addStretch(1)
        self.gpt_param_row = QWidget()
        self.gpt_param_row.setLayout(q_row)
        layout.addWidget(self.gpt_param_row)

        self.gen_channel_gemini.toggled.connect(lambda _checked: self._on_gen_channel_changed())
        self._on_gen_channel_changed()
        self._load_gpt_pipeline_ui()
        self._connect_gpt_pipeline_signals()

        self.reload_styles_btn = QPushButton("🔄 重新加载配置")
        self.reload_styles_btn.setFixedWidth(120)
        self.reload_styles_btn.clicked.connect(self.reload_styles)
        style_select_layout.addWidget(self.reload_styles_btn)
        
        style_select_layout.addStretch()
        layout.addLayout(style_select_layout)
        layout.addStretch(1)      # 选项区多余的高度收在底部留白里，别把每一行撑开（最大化时最明显）
        

        
        # ---- 以下是固定区（不随选项区滚动）：② 生成图片 / 队列 / 日志，必须一直看得见 ----
        self.bottom_panel = QWidget()
        bottom = QVBoxLayout(self.bottom_panel)
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(6)

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
        bottom.addLayout(gen_img_layout)

        gen_control_layout = QHBoxLayout()
        self.gen_countdown_label = QLabel("生图超时倒计时: --")
        self.cancel_gen_btn = QPushButton("终止当前生图")
        self.cancel_gen_btn.setEnabled(False)
        self.cancel_gen_btn.clicked.connect(self.cancel_image_generation)
        gen_control_layout.addWidget(self.gen_countdown_label)
        gen_control_layout.addStretch()
        gen_control_layout.addWidget(self.cancel_gen_btn)
        bottom.addLayout(gen_control_layout)

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
        bottom.addLayout(history_header_layout)

        self.history_list = QListWidget()
        self.history_list.setMinimumHeight(120)          # 队列常驻：绿色[完成] 就靠它看
        self.history_list.setToolTip(
            "分析队列：每行一条分析任务（批量提交会排进同一列表）。\n"
            "状态 = 该任务**全部工序**的状态：绿色[已完成] 表示分析、生图、后处理都跑完了。\n"
            "在失败/超时的记录上点右键可「重跑分析」或「重跑分析并生图」。"
        )
        self.history_list.itemSelectionChanged.connect(self._update_history_action_buttons)
        self.history_list.itemDoubleClicked.connect(lambda _item: self.open_selected_history_detail())
        self.history_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.history_list.customContextMenuRequested.connect(self._on_history_context_menu)
        bottom.addWidget(self.history_list)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(120)              # 日志常驻：以前被 gpt 选项行挤成 0 高
        log_toolbar_layout = QHBoxLayout()
        log_toolbar_layout.addStretch()
        self.clear_log_btn = QPushButton("清空日志")
        self.clear_log_btn.clicked.connect(self.log_text.clear)
        log_toolbar_layout.addWidget(self.clear_log_btn)
        bottom.addLayout(log_toolbar_layout)
        bottom.addWidget(self.log_text)

        outer_layout.addWidget(self.bottom_panel, 1)   # 多余空间给队列 + 日志（两者都是 Expanding）

        self.setLayout(outer_layout)
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
        self._refresh_style_ref_availability()

    def _refresh_style_ref_availability(self):
        """加载样式列表后按当前样式的参考图是否存在刷新参考模式可用性。"""
        styles_data = self.get_styles() or {}
        name = self.main_style_combo.currentText()
        has_ref = ref_image_valid(style_ref_image(styles_data, name))
        self.style_ref_mode_combo.set_modes_available(has_ref)
        return has_ref

    def _on_style_changed(self, _name=None):
        self._refresh_style_ref_availability()

    def reload_styles(self):
        """重新加载 config-styles.json 配置文件"""
        try:
            if self.on_styles_reload:
                self.on_styles_reload()
            else:
                import json
                config_path = os.path.join(BASE_DIR, 'conf', 'config-styles.json')
                with open(config_path, 'r', encoding='utf-8') as f:
                    styles_data = json.load(f)
                style_keys = list(styles_data.keys())
                self.update_styles(style_keys)
            self.log_msg(f"✅ 已重新加载配置文件")
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
                self.save_to_source_dir_cb.setEnabled(True)
            else:
                self.log_msg("不支持的文件格式，请拖入图片。")

    def keyPressEvent(self, event):
        if (
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
            and event.key() == Qt.Key.Key_V
        ):
            clipboard_img = ImageGrab.grabclipboard()
            if isinstance(clipboard_img, Image.Image):
                self.image_source = clipboard_img
                self.log_msg("已从剪贴板加载图片。")
                self.show_clipboard_preview(clipboard_img)
            else:
                self.log_msg("剪贴板中没有有效的图片。")

    def show_preview(self, filepath):
        pixmap = QPixmap(filepath)
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.send_btn.setEnabled(True)
        self.save_to_source_dir_cb.setEnabled(True)

    def show_clipboard_preview(self, pil_image):
        fp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            pil_image.save(fp, format="PNG")
            fp.close()
            pixmap = QPixmap(fp.name)
        finally:
            try:
                fp.close()
            except Exception:
                pass
            try:
                os.unlink(fp.name)
            except Exception:
                pass
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.send_btn.setEnabled(True)
        # 剪贴板图片无实际文件路径，保存到原图目录不适用；
        # 需先取消勾选，避免 _on_save_to_source_dir_toggled 残留禁用两个自动生图 checkbox
        self.save_to_source_dir_cb.setChecked(False)
        self.save_to_source_dir_cb.setEnabled(False)

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

    def _create_history_record(self, thread_no, image_source_snapshot, submit_time, task_hash, source_path=None):
        task_id = f"analysis-{thread_no}"
        submit_time_text = submit_time.strftime("%Y-%m-%d %H:%M:%S")
        if source_path is None:
            source_path = os.path.abspath(image_source_snapshot) if isinstance(image_source_snapshot, str) else ""
        return {
            "task_id": task_id,
            "thread_no": thread_no,
            "task_hash": task_hash,
            "status": "running",
            "status_text": self._status_to_text("running"),
            "submit_time_text": submit_time_text,
            "finish_time_text": "-",
            "source_desc": _describe_image_source(image_source_snapshot),
            "source_path": str(source_path or ""),
            "source_origin": "clipboard" if isinstance(image_source_snapshot, Image.Image) else "file",
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
        item.setData(Qt.ItemDataRole.UserRole, task_id)
        self.history_list.insertItem(0, item)
        self._refresh_history_item(task_id)
        self.history_list.setCurrentItem(item)

    def _refresh_history_status_text(self, record):
        """状态文案：进行中时把当前阶段也带上（生图 / 后处理），避免"已完成"抢跑。"""
        base = str(record.get("status_text") or self._status_to_text(record.get("status")))
        phase = str(record.get("phase") or "").strip()
        if str(record.get("status") or "").lower() == "running" and phase:
            return f"{base}·{phase}"
        return base

    def _refresh_history_item(self, task_id):
        record = self._analysis_history.get(task_id)
        if not record:
            self.log_msg(f"⚠️ _refresh_history_item: 未找到 task_id={task_id} 的历史记录")
            return
        found = False
        for row in range(self.history_list.count()):
            item = self.history_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) != task_id:
                continue
            found = True
            status_text = self._refresh_history_status_text(record)
            title = str(record.get("title") or "未命名")
            submit_time_text = record.get("submit_time_text", "-")
            source_desc = str(record.get("source_desc") or "-")
            item.setText(
                f"[{status_text}] 线程#{record.get('thread_no', '?')} / {record.get('task_hash', '--------')}  {submit_time_text}\n"
                f"{title} | {source_desc}"
            )
            item.setForeground(self._status_to_color(record.get("status")))
            break
        if not found:
            self.log_msg(f"⚠️ _refresh_history_item: task_id={task_id} 对应的 QListWidgetItem 未找到，列表可能已被清空")

    def _update_history_record(self, task_id, **kwargs):
        if not task_id:
            self.log_msg(f"⚠️ _update_history_record: task_id 为空，无法更新历史记录。kwargs={kwargs}")
            return
        record = self._analysis_history.get(task_id)
        if not record:
            self.log_msg(f"⚠️ _update_history_record: 未找到 task_id={task_id} 的历史记录，可能已被移除或从未创建。kwargs={kwargs}")
            return
        record.update(kwargs)
        record["status_text"] = self._status_to_text(record.get("status"))
        self._refresh_history_item(task_id)

    def _selected_history_record(self):
        item = self.history_list.currentItem()
        if item is None:
            return None
        task_id = item.data(Qt.ItemDataRole.UserRole)
        return self._analysis_history.get(task_id)

    def _guess_final_products(self, task_hash, date_str=None):
        """兜底：在 data/<日期>/ 里找 **-final- 产物（以最终产物为准的判定依据）。"""
        import glob
        day = date_str or datetime.datetime.now().strftime("%Y%m%d")
        pattern = os.path.join("data", day, f"*{task_hash}*-final-*")
        hits = [p for p in glob.glob(pattern) if os.path.isfile(p)]
        return sorted(hits, key=os.path.getmtime)

    def _history_task_ids_for_hash(self, task_hash):
        """同一个 task_hash 可能有多条记录（重跑），全部都要收尾。"""
        h = str(task_hash or "").strip()
        return [tid for tid, rec in self._analysis_history.items()
                if h and str(rec.get("task_hash") or "").strip() == h]

    def _pipeline_pending_for_hash(self, task_hash):
        """该任务是否还有生图/后处理线程在跑。"""
        h = str(task_hash or "").strip()

        def _belongs(thread):
            if str(getattr(thread, "meta_task_hash", "") or "").strip() == h:
                return True
            return h and h in str(getattr(thread, "meta_task_id", "") or "")

        pending = [t for t in list(self._active_img_threads) + list(getattr(self, "_active_post_threads", []))
                   if _belongs(t)]
        return pending

    def _finalize_task_pipeline(self, task_hash, final_products=None, require_products=True):
        """只有「没有待跑线程 + 有最终产物」时，队列才置为绿色已完成。

        `require_products=False`：这条任务本来就不产图（没勾自动生图 / 生图线程一个都没启动），
        只要有线程活着就不再吊着队列，直接标完成 —— 否则它会永远停在「进行中」。
        """
        pending = self._pipeline_pending_for_hash(task_hash)
        if pending:
            kinds = []
            if any(t in self._active_img_threads for t in pending):
                kinds.append("生图")
            if any(t in getattr(self, "_active_post_threads", []) for t in pending):
                kinds.append("后处理")
            for task_id in self._history_task_ids_for_hash(task_hash):
                self._update_history_record(task_id, status="running", phase="+".join(kinds) or "处理中")
            return False
        finalized = False
        for task_id in self._history_task_ids_for_hash(task_hash):
            record = self._analysis_history.get(task_id) or {}
            products = list(final_products or record.get("final_products") or [])
            if not products:
                products = self._guess_final_products(task_hash)     # 以 -final- 产物为准
            if not products and require_products:
                # 生图这条线自己失败了（线程已退出且没产物）→ 标红，别让它永远「进行中·等待最终产物」
                failure = str(record.get("pipeline_error") or "").strip()
                if failure:
                    self._update_history_record(
                        task_id,
                        status="error",
                        phase="",
                        finish_time_text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    )
                    self.log_msg(f"❌ 队列标记失败（{failure}）：{record.get('title', '未命名')}")
                    continue
                self._update_history_record(task_id, status="running", phase="等待最终产物")
                self.log_msg(f"⚠️ 任务 {task_hash} 的线程都结束了，但还没有最终产物，暂不标记完成。")
                continue
            self._update_history_record(
                task_id,
                status="success",
                phase="",
                pipeline_error="",
                finish_time_text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                final_products=products,
            )
            if products:
                self.log_msg(f"✅ 队列标记完成（以最终产物为准）: {os.path.basename(str(products[-1]))}")
            else:
                self.log_msg("✅ 队列标记完成（本任务不含生图工序，分析结果已落盘）")
            finalized = True
        return finalized

    def _update_history_action_buttons(self):
        record = self._selected_history_record()
        self.open_history_btn.setEnabled(record is not None)
        # 生图失败（status=error）但分析结果本身是好的 → 仍然允许「设为当前结果」，别把可用产物锁死
        self.apply_history_btn.setEnabled(bool(
            record and str(record.get("status") or "").strip().lower() != "running"
            and record.get("result_json")))

    def _apply_history_record_to_current_state(self, record):
        if not record:
            return False
        if str(record.get("status") or "").strip().lower() == "running":
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
            QMessageBox.information(self, "提示", "只有已结束（含「分析成功但生图失败」）并带结果的历史记录才能设为当前结果。")
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
        dialog.exec()

    # ==================== 分析队列右键菜单（重跑失败项） ====================

    RERUNNABLE_STATUSES = ("error", "timeout")

    def _history_record_source_path(self, record):
        """取出该记录可重跑的原图路径；剪贴板图片没有路径，返回空串。"""
        path = str((record or {}).get("source_path") or "").strip()
        return os.path.abspath(path) if path else ""

    def _history_record_usable_path(self, record):
        """路径存在才可重跑（原图可能已被移动/删除）。"""
        path = self._history_record_source_path(record)
        return path if path and os.path.isfile(path) else ""

    def _failed_history_records(self):
        """失败/超时的记录，按提交先后排序（新的记录插在列表顶部，这里按线程号还原提交顺序）。"""
        failed = [
            record for record in self._analysis_history.values()
            if str(record.get("status") or "").strip().lower() in self.RERUNNABLE_STATUSES
        ]
        failed.sort(key=lambda record: record.get("thread_no") or 0)
        return failed

    def _on_history_context_menu(self, pos):
        item = self.history_list.itemAt(pos)
        if item is not None:
            self.history_list.setCurrentItem(item)
        record = self._selected_history_record()
        source_path = self._history_record_source_path(record)
        usable_path = self._history_record_usable_path(record)
        failed_records = self._failed_history_records()
        failed_runnable = [r for r in failed_records if self._history_record_usable_path(r)]

        menu = QMenu(self.history_list)
        if record is not None and not usable_path:
            hint = menu.addAction("⚠️ 分析源图不可用（文件已移动/删除，或剪贴板快照被清空）")
            hint.setEnabled(False)
            menu.addSeparator()

        rerun_action = menu.addAction("🔁 重跑分析")
        rerun_action.setEnabled(bool(usable_path))

        gen_menu = menu.addMenu("🔁 重跑分析并生图")
        gen_menu.setEnabled(bool(usable_path))
        gen_original_action = gen_menu.addAction("基于原始提示词")
        gen_refined_action = gen_menu.addAction("基于优化提示词")
        gen_both_action = gen_menu.addAction("原始 + 优化都生成")

        menu.addSeparator()
        rerun_failed_action = menu.addAction(f"🔁 重跑全部失败项（{len(failed_runnable)}）")
        rerun_failed_action.setEnabled(bool(failed_runnable))
        clear_failed_action = menu.addAction(f"🧹 清空失败记录（{len(failed_records)}）")
        clear_failed_action.setEnabled(bool(failed_records))

        menu.addSeparator()
        open_dir_action = menu.addAction("📂 打开源图所在目录")
        open_dir_action.setEnabled(bool(usable_path))
        copy_path_action = menu.addAction("📋 复制源图路径")
        copy_path_action.setEnabled(bool(source_path))
        remove_action = menu.addAction("🗑️ 移除该记录")
        remove_action.setEnabled(bool(record) and str(record.get("status") or "").strip().lower() != "running")

        chosen = menu.exec(self.history_list.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen is rerun_action:
            self._rerun_history_record(record)
        elif chosen is gen_original_action:
            self._rerun_history_record(record, gen_targets=["original"])
        elif chosen is gen_refined_action:
            self._rerun_history_record(record, gen_targets=["refined"])
        elif chosen is gen_both_action:
            self._rerun_history_record(record, gen_targets=["original", "refined"])
        elif chosen is rerun_failed_action:
            self._rerun_failed_history_records()
        elif chosen is clear_failed_action:
            self._clear_failed_history_records()
        elif chosen is open_dir_action:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(usable_path)))
        elif chosen is copy_path_action:
            QApplication.clipboard().setText(source_path)
            self.log_msg(f"📋 已复制源图路径: {source_path}")
        elif chosen is remove_action:
            self._remove_history_record(record)

    def _rerun_history_record(self, record, gen_targets=None):
        """按记录里的源图（原图或剪贴板快照）重新提交一次分析（可选：分析成功后自动生图）。"""
        usable_path = self._history_record_usable_path(record)
        if not usable_path:
            QMessageBox.warning(
                self, "无法重跑",
                "该记录的分析源图不可用：原图已被移动/删除，或剪贴板快照被清理（cache/temp 可清空）。"
            )
            return False
        origin_text = "剪贴板快照" if str(record.get("source_origin") or "") == "clipboard" else "原图"
        self._commit_outfit_style_text(self.outfit_style_combo.currentText(), add_to_history=True)
        self.log_msg("\n" + ("=" * 72))
        self.log_msg(
            f"🔁 重跑分析：线程#{record.get('thread_no', '?')} / {record.get('task_hash', '--------')}"
            f" / {record.get('title', '未命名')}（原状态: {record.get('status_text', '未知')}）"
            f" → {origin_text} {os.path.basename(usable_path)}"
        )
        thread = self._launch_analysis_task(
            _clone_image_source(usable_path),
            gen_targets=gen_targets,
            header_note=f"重跑来源: 线程#{record.get('thread_no', '?')}（{record.get('status_text', '未知')}）",
        )
        if thread is None:
            return False
        if thread.meta_force_gen_targets:
            targets_text = " + ".join(
                "原始提示词" if target == "original" else "优化提示词"
                for target in thread.meta_force_gen_targets
            )
            self.log_msg(f"🤖 本次重跑将在分析成功后自动生图：{targets_text}")
        return True

    def _rerun_failed_history_records(self):
        failed_records = self._failed_history_records()
        if not failed_records:
            QMessageBox.information(self, "提示", "当前没有失败或超时的记录。")
            return
        runnable = [record for record in failed_records if self._history_record_usable_path(record)]
        skipped = len(failed_records) - len(runnable)
        if not runnable:
            QMessageBox.warning(
                self, "无法重跑",
                f"{len(failed_records)} 条失败记录的原图都不可用（剪贴板图片或文件已被移动/删除）。"
            )
            return
        message = (
            f"将重跑 {len(runnable)} 条失败/超时记录（沿用当前勾选的选项与超时设置）。"
        )
        if skipped:
            message += f"\n另有 {skipped} 条因原图不可用会被跳过。"
        message += "\n\n是否继续？"
        if QMessageBox.question(self, "重跑失败项", message) != QMessageBox.StandardButton.Yes:
            return
        started = 0
        for record in runnable:
            if self._rerun_history_record(record):
                started += 1
        self.log_msg(f"🔁 重跑失败项完成：已提交 {started}/{len(runnable)} 条。")

    def _remove_history_record(self, record):
        if not record:
            return
        task_id = record.get("task_id")
        if str(record.get("status") or "").strip().lower() == "running":
            QMessageBox.information(self, "提示", "该任务仍在进行中，请等它结束后再移除。")
            return
        self._analysis_history.pop(task_id, None)
        for row in range(self.history_list.count()):
            item = self.history_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == task_id:
                self.history_list.takeItem(row)
                break
        self._update_history_action_buttons()
        self.log_msg(
            f"🗑️ 已从分析队列移除记录: 线程#{record.get('thread_no', '?')} / {record.get('task_hash', '--------')}"
        )

    def _clear_failed_history_records(self):
        failed_records = self._failed_history_records()
        if not failed_records:
            QMessageBox.information(self, "提示", "当前没有失败或超时的记录。")
            return
        message = (
            f"将从分析队列中移除 {len(failed_records)} 条失败/超时记录。\n"
            "（只清列表，不动磁盘上已保存的文件）\n\n是否继续？"
        )
        if QMessageBox.question(self, "清空失败记录", message) != QMessageBox.StandardButton.Yes:
            return
        for record in list(failed_records):
            self._remove_history_record(record)
        self.log_msg(f"🧹 已清空 {len(failed_records)} 条失败/超时记录。")

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
        image_source_snapshot = _clone_image_source(self.image_source)
        self.log_msg("\n" + ("=" * 72))
        self._launch_analysis_task(image_source_snapshot)

    def _save_clipboard_snapshot(self, pil_image):
        """把剪贴板图片落成 PNG 快照，返回路径（失败返回空串）。

        目的：让「Ctrl+V 粘贴进来」的分析记录也能在队列里右键重跑——
        否则这条记录只有内存里的 PIL 对象，重跑时无源可用。
        """
        try:
            os.makedirs(CLIPBOARD_SNAPSHOT_DIR, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            fd, snapshot_path = tempfile.mkstemp(
                prefix=f"clip-{stamp}-", suffix=".png", dir=CLIPBOARD_SNAPSHOT_DIR
            )
            os.close(fd)
            image = pil_image
            if image.mode not in ("RGB", "RGBA", "L", "P"):
                image = image.convert("RGBA")
            image.save(snapshot_path, format="PNG")
            return os.path.abspath(snapshot_path)
        except Exception as e:
            self.log_msg(f"⚠️ 剪贴板图片快照保存失败，本次记录将无法右键重跑: {e}")
            return ""

    def _launch_analysis_task(self, image_source_snapshot, gen_targets=None, header_note=None):
        """统一的分析任务提交入口：单图分析 / 目录批量 / 右键重跑都走这里。

        负责预检（prompt 文件、文本 API 配置）→ 建历史记录 → 装配 WorkerThread 并启动。
        返回 thread；预检不通过返回 None。
        """
        enable_outfit_check = self.enable_outfit_check_cb.isChecked()
        remove_photo_style = self.remove_photo_style_cb.isChecked()
        # 当启用任一优化选项时，Step 5 会重新计算 pixiv_tags，需要确保对应 prompt 文件存在
        enable_recompute_pixiv_tags = enable_outfit_check or remove_photo_style
        missing_prompt_files = get_single_analyzer_missing_prompt_files(
            enable_refine=True,
            enable_outfit_check=enable_outfit_check,
            enable_remove_photo_style=remove_photo_style,
            enable_recompute_pixiv_tags=enable_recompute_pixiv_tags,
        )
        if missing_prompt_files:
            missing_text = "\n".join(missing_prompt_files)
            QMessageBox.warning(self, "缺少 Prompt 文件", f"以下 Prompt 文件不存在，请补齐后再执行：\n{missing_text}")
            self.log_msg(f"❌ 缺少 Prompt 文件，已中止分析：\n{missing_text}")
            return None

        base_url, api_key, model_name = self.get_text_config(self.use_nsfw_cb.isChecked())
        if not api_key or not model_name:
            QMessageBox.warning(self, "缺少配置", "文本分析 API Key 和 模型名称不能为空！")
            return None

        timeout_seconds = int(self.get_timeout_seconds()) if self.get_timeout_seconds else 120
        booru_tag_limit = int(self.get_booru_tag_limit()) if self.get_booru_tag_limit else 30
        analysis_thread_no = self._next_thread_no("_analysis_thread_seq")
        thread_prefix = self._build_thread_prefix("分析线程", analysis_thread_no)
        submit_time = datetime.datetime.now()
        ddl = submit_time + datetime.timedelta(seconds=max(1, timeout_seconds))
        task_hash = _generate_task_hash(image_source_snapshot, submit_time, analysis_thread_no)

        if header_note:
            self.log_msg(header_note, prefix=thread_prefix)
        self.log_msg("任务已启动...", prefix=thread_prefix)
        self.log_msg(f"图片源: {_describe_image_source(image_source_snapshot)}", prefix=thread_prefix)
        self.log_msg(f"提交时间: {submit_time.strftime('%Y-%m-%d %H:%M:%S')}", prefix=thread_prefix)
        self.log_msg(f"任务 Hash: {task_hash}", prefix=thread_prefix)
        self.log_msg(f"超时设置: {timeout_seconds} 秒（预计超时点: {ddl.strftime('%H:%M:%S')}）", prefix=thread_prefix)
        retry_settings = load_retry_settings()
        if retry_settings.enabled and retry_settings.times > 0:
            self.log_msg(
                f"失败重试: 最多 {retry_settings.times} 次，间隔 {format_wait(retry_settings.interval_seconds)}"
                f"（对面 429 限流 / 5xx / 请求超时时触发，可在「设置 → 文本分析 API」调整）",
                prefix=thread_prefix,
            )
        else:
            self.log_msg(
                "失败重试: 已关闭（对面返回 429/5xx 时会直接失败；可在「设置 → 文本分析 API」开启）",
                prefix=thread_prefix,
            )

        clipboard_snapshot_path = ""
        if isinstance(image_source_snapshot, Image.Image):
            clipboard_snapshot_path = self._save_clipboard_snapshot(image_source_snapshot)
            if clipboard_snapshot_path:
                self.log_msg(
                    f"📋 剪贴板图片已存快照（可在队列里右键重跑）: {clipboard_snapshot_path}",
                    prefix=thread_prefix,
                )

        history_record = self._create_history_record(
            analysis_thread_no, image_source_snapshot, submit_time, task_hash,
            source_path=clipboard_snapshot_path or None,
        )
        self._insert_history_record(history_record)

        thread = WorkerThread(
            image_source_snapshot,
            api_key,
            base_url,
            model_name,
            booru_tag_limit=booru_tag_limit,
            timeout_seconds=timeout_seconds,
            enable_outfit_check=enable_outfit_check,
            outfit_style_override=self.outfit_style_combo.currentText().strip(),
            remove_photo_style=remove_photo_style
        )
        thread.meta_thread_no = analysis_thread_no
        thread.meta_source_snapshot = image_source_snapshot
        thread.meta_task_id = history_record["task_id"]
        thread.meta_task_hash = task_hash
        thread.meta_force_gen_targets = [t for t in (gen_targets or []) if t in ("original", "refined")]
        self._active_analysis_threads.append(thread)
        self._update_analysis_cancel_btn()
        thread.log_signal.connect(
            lambda text, t=thread: self.log_msg(
                text,
                prefix=self._build_thread_prefix("分析线程", getattr(t, "meta_thread_no", "?"))
            )
        )
        thread.finish_signal.connect(lambda result_json, t=thread: self.on_process_finished(t, result_json))
        thread.finished.connect(lambda t=thread: self._on_analysis_thread_stopped(t))
        thread.start()
        return thread

    def on_use_nsfw_toggled(self, checked):
        if self.on_nsfw_changed:
            self.on_nsfw_changed(bool(checked))

    def on_enable_outfit_check_toggled(self, checked):
        if self.on_outfit_check_changed:
            self.on_outfit_check_changed(bool(checked))

    def on_remove_photo_style_toggled(self, checked):
        if self.on_remove_photo_style_changed:
            self.on_remove_photo_style_changed(bool(checked))

    def _on_save_to_source_dir_toggled(self, checked):
        """保存到原图同目录时，禁用自动生图选项"""
        if checked:
            self.auto_gen_orig_cb.setEnabled(False)
            self.auto_gen_ref_cb.setEnabled(False)
        else:
            self.auto_gen_orig_cb.setEnabled(True)
            self.auto_gen_ref_cb.setEnabled(True)

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

    def set_remove_photo_style_default(self, checked):
        self.remove_photo_style_cb.blockSignals(True)
        self.remove_photo_style_cb.setChecked(bool(checked))
        self.remove_photo_style_cb.blockSignals(False)

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

    def _update_analysis_cancel_btn(self):
        if hasattr(self, "cancel_analysis_btn"):
            self.cancel_analysis_btn.setEnabled(bool(self._active_analysis_threads))

    def cancel_active_analysis_tasks(self):
        """终止所有正在进行的分析任务；重试等待期间会立刻中断等待。"""
        threads = list(self._active_analysis_threads)
        if not threads:
            self.log_msg("当前没有正在执行的分析任务。")
            self._update_analysis_cancel_btn()
            return
        self.log_msg(f"🛑 正在请求终止 {len(threads)} 个分析任务（含等待中的重试）...")
        for thread in threads:
            try:
                if hasattr(thread, "request_cancel"):
                    thread.request_cancel(force=True)
                else:
                    thread.requestInterruption()
            except Exception:
                pass
        self._update_analysis_cancel_btn()

    def _on_analysis_thread_stopped(self, thread):
        if thread in self._active_analysis_threads:
            self._active_analysis_threads.remove(thread)
        self.send_btn.setEnabled(bool(self.image_source))
        self._update_analysis_cancel_btn()
        # 兜底检查：如果线程已退出但历史记录仍处于"处理中"状态，说明 on_process_finished
        # 未能成功更新历史记录（可能是 task_id 不匹配或信号丢失），此时强制更新为完成。
        # 【例外】本任务还要自动生图/后处理时，记录是**故意**留在 running 的（phase=生图/后处理中），
        # 那种情况必须跳过兜底，否则会把标题改成「已完成（兜底更新）」并抢跑标绿，
        # 随后生图完成又被改回 running —— 队列文案来回跳（用户 2026-09-24 截图就是这个）。
        task_id = getattr(thread, "meta_task_id", "")
        if task_id:
            record = self._analysis_history.get(task_id)
            if record and record.get("status") == "running":
                task_hash = str(record.get("task_hash") or getattr(thread, "meta_task_hash", "") or "").strip()
                if self._pipeline_pending_for_hash(task_hash):
                    self.log_msg(
                        f"⏳ 分析线程 #{getattr(thread, 'meta_thread_no', '?')} 已退出，"
                        f"但该任务的生图/后处理还在跑：队列保持「进行中」，等管线全部结束后再标完成。"
                    )
                    return
                finish_time_text = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                last_error = str(getattr(thread, "last_error", "") or "").strip()
                warn_text = (
                    f"⚠️ 线程 #{getattr(thread, 'meta_thread_no', '?')} 已退出但历史记录仍为「处理中」，正在兜底更新。"
                )
                if last_error:
                    warn_text += f"\n最后一条错误信息: {last_error}"
                self.log_msg(warn_text)
                _log_step_diag(
                    f"⚠️ 分析线程 #{getattr(thread, 'meta_thread_no', '?')} 已退出但未上报结果"
                    f"（可能 run() 内抛出未捕获异常）"
                    + (f"；最后一条错误信息: {last_error}" if last_error else "")
                )
                self._update_history_record(
                    task_id,
                    status="success",
                    title="已完成（兜底更新）",
                    finish_time_text=finish_time_text,
                )

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
                _log_step_diag(f"{thread_prefix} 处理失败：请求超时（超时设置 {getattr(thread, 'timeout_seconds', '?')} 秒）")
                self._send_system_notification("单图分析超时", "任务因请求超时结束，请调整超时配置后重试。")
            elif task_status == "cancelled":
                self.log_msg("任务已取消。", prefix=thread_prefix)
                _log_step_diag(f"{thread_prefix} 任务已取消。")
                self._send_system_notification("单图分析已取消", "当前任务已被取消。")
            else:
                failure_reason = f"处理失败，未能获取到有效的 JSON 数据。（线程状态: {task_status}）"
                last_error = str(getattr(thread, "last_error", "") or "").strip()
                if last_error:
                    failure_reason += f"\n失败原因: {last_error}"
                self.log_msg(failure_reason, prefix=thread_prefix)
                _log_step_diag(f"{thread_prefix} {failure_reason}")
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
        safe_task_hash = task_hash or "nohash"

        # 确定保存目录和基础文件名（支持保存到原图同目录）
        save_to_source = (
            self.save_to_source_dir_cb.isChecked()
            and isinstance(source_snapshot, str)
            and os.path.isfile(source_snapshot)
        )
        if save_to_source:
            save_dir = os.path.dirname(os.path.abspath(source_snapshot))
            # 从原图文件名提取 key（_ 分割的第一段），用于 publish_server 匹配
            source_basename = os.path.basename(source_snapshot)
            source_key = os.path.splitext(source_basename)[0].split("_")[0]
            base_filename = f"{now_str}-{source_key}-{safe_title}"
        else:
            save_dir = os.path.join('data', date_str)
            base_filename = f"{now_str}-{safe_task_hash}-{safe_title}"

        # 测试模式下把 data/<日期>/ 的产出改道到 data/test-result/<日期>/ 并加前缀，
        # 避免测试假数据（hash_0 / task_00 / test_hash…）混进真实产出目录
        save_dir, base_filename = resolve_output_target(save_dir, base_filename)

        if not os.path.exists(save_dir): os.makedirs(save_dir)
            
        json_filename = f"{base_filename}.json"
        
        try:
            saved_json_path = os.path.abspath(os.path.join(save_dir, json_filename))
            self._last_saved_json_path = saved_json_path
            with open(os.path.join(save_dir, json_filename), "w", encoding="utf-8") as f:
                json.dump(result_json, f, ensure_ascii=False, indent=4)
            self.log_msg(f"✅ 成功！JSON 结果已保存至: {json_filename}", prefix=thread_prefix)
        except Exception as e:
            saved_json_path = ""
            self._last_saved_json_path = ""
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
            current_fixed_tags = style_prompt(styles_data, selected_style_name)
            
            # 在风格标签和描述之间添加两个回车
            style_part = f"--ar {self.current_aspect_ratio} {current_fixed_tags}".strip()
            # 添加面部质量保护后缀，防止因原图打码等原因导致生成模糊面部
            _face_quality_suffix = "detailed face, clear facial features, sharp focus on face"
            refine_desc_with_face = f"{self.current_refine_desc}, {_face_quality_suffix}" if self.current_refine_desc else self.current_refine_desc
            orig_desc_with_face = f"{self.current_orig_desc}, {_face_quality_suffix}" if self.current_orig_desc else self.current_orig_desc
            final_prompt = f"{style_part}\n\n{refine_desc_with_face}".strip()
            orig_prompt = f"{style_part}\n\n{orig_desc_with_face}".strip()
            
            txt_filename = f"{base_filename}-prompts.txt"
            orig_txt_filename = f"{base_filename}-original-prompts.txt"
            saved_prompt_paths = [
                os.path.abspath(os.path.join(save_dir, txt_filename)),
                os.path.abspath(os.path.join(save_dir, orig_txt_filename))
            ]
            self._last_saved_prompt_paths = saved_prompt_paths
            
            with open(os.path.join(save_dir, txt_filename), "w", encoding="utf-8") as f: f.write(final_prompt)
            with open(os.path.join(save_dir, orig_txt_filename), "w", encoding="utf-8") as f: f.write(orig_prompt)

            # gpt-image 通道专用短提示词（单独文件，不与 Gemini 的长 prompts 混用）：完整档 + 短锚档
            for field_key, suffix, label in (("gpt_image_prompt", "-gpt-image-prompts.txt", "完整档"),
                                             ("gpt_image_prompt_short", "-gpt-image-short-prompts.txt", "短锚档")):
                gpt_field = str(result_json.get(field_key) or "").strip()
                if not gpt_field:
                    continue
                gpt_txt_filename = f"{base_filename}{suffix}"
                gpt_txt_path = os.path.abspath(os.path.join(save_dir, gpt_txt_filename))
                with open(gpt_txt_path, "w", encoding="utf-8") as f:
                    f.write(gpt_field)
                saved_prompt_paths.append(gpt_txt_path)
                self.log_msg(f"✅ gpt-image {label}提示词已保存: {gpt_txt_filename}（{len(gpt_field)} 字符）",
                             prefix=thread_prefix)

            self.log_msg(f"✅ 成功！两份画幅与提示词文件已保存:\n - {txt_filename}\n - {orig_txt_filename}", prefix=thread_prefix)
            
            # 保存到原图同目录时，同步文件修改时间
            if save_to_source:
                self._sync_file_times_to_source(saved_json_path, source_snapshot)
                for prompt_path in saved_prompt_paths:
                    self._sync_file_times_to_source(prompt_path, source_snapshot)
                self.log_msg("🕐 分析结果文件的修改时间已同步到与原图一致", prefix=thread_prefix)
            
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
            self._last_saved_prompt_paths = []
            self.log_msg(f"❌ 保存提示词 txt 文件时出错: {e}", prefix=thread_prefix)

        # 后面还会自动生图/跑工序时，这一条**不能**先标完成（队列绿得太早就是这里造成的）
        will_auto_gen = (not save_to_source) and (
            bool(getattr(thread, "meta_force_gen_targets", None))
            or self.auto_gen_orig_cb.isChecked() or self.auto_gen_ref_cb.isChecked()
        )
        pipeline_next = will_auto_gen and bool(
            str(result_json.get("original_english_description") or "").strip()
            or str(result_json.get("english_description") or "").strip()
        )
        self._update_history_record(
            task_id,
            status="running" if pipeline_next else "success",
            phase="生图/后处理中" if pipeline_next else "",
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
        # 直接从 result_json 和 safe_task_hash 构建 prompt_bundle，完全不依赖实例变量
        # 避免多线程竞态条件：多个分析任务几乎同时完成时，实例变量可能被后续任务覆盖
        # 当勾选"保存到原图同目录"时跳过自动生图
        if save_to_source:
            if getattr(thread, "meta_force_gen_targets", None):
                self.log_msg(
                    "⚠️ 已勾选「保存到原图同目录（不生成图片）」，本次重跑请求的生图已跳过。",
                    prefix=thread_prefix,
                )
            self._send_system_notification("单图分析完成", "任务已完成，结果已保存至原图同目录。")
        else:
            local_task_hash = safe_task_hash
            local_aspect_ratio = self._resolve_ar_for_first_stage(result_json.get("aspect_ratio", "2:3"))
            local_orig_desc = result_json.get("original_english_description", "")
            local_refine_desc = result_json.get("english_description", "")
            
            auto_targets = []
            # 右键「重跑分析并生图」时，本次任务自带生成目标，优先于全局勾选框
            forced_targets = [
                target for target in (getattr(thread, "meta_force_gen_targets", None) or [])
                if target in ("original", "refined")
            ]
            if forced_targets:
                if "original" in forced_targets and str(local_orig_desc).strip():
                    auto_targets.append("original")
                if "refined" in forced_targets and str(local_refine_desc).strip():
                    auto_targets.append("refined")
                if not auto_targets:
                    self.log_msg("⚠️ 本次重跑要求生图，但分析结果里没有可用的提示词，已跳过自动生图。", prefix=thread_prefix)
            else:
                if self.auto_gen_orig_cb.isChecked() and str(local_orig_desc).strip():
                    auto_targets.append("original")
                if self.auto_gen_ref_cb.isChecked() and str(local_refine_desc).strip():
                    auto_targets.append("refined")
            prompt_bundle = {
                "task_hash": local_task_hash,
                "aspect_ratio": local_aspect_ratio,
                "original_prompt": local_orig_desc,
                "refined_prompt": local_refine_desc,
                "analysis_json_path": saved_json_path,
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
                    # 一个生图线程都没起来（缺 key / 目标提示词为空…）：本任务到此结束，
                    # 不能因为前面按「还会生图」把记录留在 running，就让它永远吊在「进行中」。
                    self._auto_gen_groups.pop(auto_group_id, None)
                    self.log_msg("⚠️ 自动生图没有启动任何任务（上方有原因），本任务按「只做分析」收尾。",
                                 prefix=thread_prefix)
                    self._finalize_task_pipeline(local_task_hash, require_products=False)
                    self._send_system_notification("单图分析完成", "任务已完成并生成结果文件。")
            else:
                # 勾了自动生图但没拿到可用提示词 → 同样按「只做分析」收尾（同上）
                if (getattr(thread, "meta_force_gen_targets", None)
                        or self.auto_gen_orig_cb.isChecked() or self.auto_gen_ref_cb.isChecked()):
                    self._finalize_task_pipeline(local_task_hash, require_products=False)
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

    def _mark_pipeline_busy(self):
        """进入后处理阶段：保持按钮禁用 + 状态提示，避免看着像已完成。"""
        self.gen_orig_btn.setEnabled(False)
        self.gen_ref_btn.setEnabled(False)
        if self._active_post_threads:
            self.log_msg(f"⏳ 进入后处理阶段（{len(self._active_post_threads)} 个线程）；"
                         "全部结束后按钮才会恢复。")

    def _pipeline_busy(self):
        """还有生图线程或后处理线程在跑 → 整个任务没结束。"""
        return bool(self._active_img_threads or getattr(self, "_active_post_threads", []))

    def _mark_pipeline_idle_if_done(self):
        """只有生图 + 后处理都结束，才恢复按钮并宣告完成。"""
        if self._pipeline_busy():
            pending = []
            if self._active_img_threads:
                pending.append(f"生图 {len(self._active_img_threads)}")
            if getattr(self, "_active_post_threads", []):
                pending.append(f"后处理 {len(self._active_post_threads)}")
            self.log_msg(f"⏳ 仍在执行：{'、'.join(pending)}（全部结束后才算完成）")
            return
        self.gen_orig_btn.setEnabled(True)
        self.gen_ref_btn.setEnabled(True)
        self._stop_image_gen_runtime()
        self.log_msg("✅ 全流程完成（生图 + 后处理）")

    def _on_image_thread_stopped(self, thread):
        """生图线程真正退出（QThread.finished）：清引用 → 队列收尾。

        【关键】`on_image_generation_finished` 里那次 `_finalize_task_pipeline` 必然**失败**（那一刻
        线程自己还在 `_active_img_threads` 里），那时只是把队列标成「进行中·生图」；真正的收尾必须
        放在这里。以前这里什么都不做，只在后处理线程结束时收尾 —— 默认配方根本不产生后处理线程
        （JPG 后处理只对 .jpg 生效），于是队列永远停在「进行中·生图」（用户 2026-09-24 截图）。
        """
        if thread in self._active_img_threads:
            self._active_img_threads.remove(thread)
        task_hash = str(getattr(thread, "meta_task_hash", "") or "").strip()
        if not self._active_img_threads:
            self._stop_image_gen_runtime()
        self._mark_pipeline_idle_if_done()          # 生图 + 后处理都结束才恢复按钮并提示完成
        if task_hash:
            self._finalize_task_pipeline(task_hash)  # 管线没别的线程了 → 有产物就标绿

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

    def _load_gpt_pipeline_ui(self, state=None):
        """恢复上次的通道 / 工序 / 区域 / 画质选择；老配置按「效果最好的配方」升级一次。"""
        state = load_analysis_gpt_ui() if state is None else state
        for widget, key in ((self.gpt_first_pass_mode, "first_pass_mode"),):
            idx = widget.findData(state.get(key, ANALYSIS_GPT_UI_DEFAULTS[key]))
            widget.setCurrentIndex(max(0, idx))
        if state.get("channel") == "gpt-image":
            self.gen_channel_gpt.setChecked(True)
        self.gpt_pp_repaint.setChecked(bool(state.get("repaint")))
        self.gpt_pp_structure.setChecked(bool(state.get("structure")))
        self.gpt_pp_local.setChecked(bool(state.get("local")))
        self.gpt_size_follow_cb.setChecked(bool(state.get("size_follow_input", True)))
        self.gpt_pp_tone.setChecked(bool(state.get("tone")))
        self.gpt_pp_ink.setChecked(bool(state.get("ink")))
        _ti = self.gpt_pp_tone_target.findData(str(state.get("tone_target") or "style"))
        if _ti >= 0:
            self.gpt_pp_tone_target.setCurrentIndex(_ti)
        regions = [str(r) for r in (state.get("regions") or []) if str(r).strip()]
        idx = self.gpt_pp_region.findData(regions) if regions else -1     # 多区域链按列表匹配
        if idx < 0:
            idx = self.gpt_pp_region.findData(str(state.get("region") or ""))
        if idx >= 0:
            self.gpt_pp_region.setCurrentIndex(idx)                       # 找不到就保持默认（稳定四区）
        idx = self.gpt_quality_combo.findData(str(state.get("quality") or "high"))
        if idx >= 0:
            self.gpt_quality_combo.setCurrentIndex(idx)
        if hasattr(self, "gpt_pp_scope"):
            idx = self.gpt_pp_scope.findData(str(state.get("repaint_scope") or "full"))
            if idx >= 0:
                self.gpt_pp_scope.setCurrentIndex(idx)
        self._on_gen_channel_changed()
        self._save_gpt_pipeline_ui()      # 记下配方版本：本次之后用户自己的改动才会被记住

    def _gpt_pipeline_ui_state(self) -> dict:
        data = self.gpt_pp_region.currentData()
        regions = [str(r) for r in data] if isinstance(data, (list, tuple)) else [str(data or "hair")]
        return {
            "channel": "gpt-image" if self._gpt_image_channel_active() else "gemini",
            "repaint": bool(self.gpt_pp_repaint.isChecked()),
            "structure": bool(self.gpt_pp_structure.isChecked()),
            "local": bool(self.gpt_pp_local.isChecked()),
            "region": regions[0],
            "regions": regions,
            "quality": str(self.gpt_quality_combo.currentData() or "high"),
            "size_follow_input": bool(self.gpt_size_follow_cb.isChecked()),
            "tone": bool(self.gpt_pp_tone.isChecked()),
            "tone_target": str(self.gpt_pp_tone_target.currentData() or "style"),
            "ink": bool(self.gpt_pp_ink.isChecked()),
            "repaint_scope": str(getattr(self, "gpt_pp_scope", None).currentData()
                                 if getattr(self, "gpt_pp_scope", None) is not None else "full"),
            "recipe_version": GPT_RECIPE_VERSION,
            "first_pass_mode": self.gpt_first_pass_mode.currentData(),
        }

    def _save_gpt_pipeline_ui(self, *_args):
        """任何选择变化都记下来（下次启动自动恢复）。"""
        save_analysis_gpt_ui(self._gpt_pipeline_ui_state())

    def _connect_gpt_pipeline_signals(self):
        for widget, signal in ((self.gen_channel_gemini, "toggled"),
                               (self.gpt_pp_repaint, "toggled"),
                               (self.gpt_pp_structure, "toggled"),
                               (self.gpt_pp_local, "toggled"),
                               (self.gpt_size_follow_cb, "toggled"),
                               (self.gpt_pp_tone, "toggled"),
                               (self.gpt_pp_ink, "toggled"),
                               (self.gpt_pp_tone_target, "currentIndexChanged"),
                               (self.gpt_pp_scope, "currentIndexChanged"),
                               (self.gpt_pp_region, "currentIndexChanged"),
                               (self.gpt_first_pass_mode, "currentIndexChanged"),
                               (self.gpt_quality_combo, "currentIndexChanged")):
            getattr(widget, signal).connect(self._save_gpt_pipeline_ui)

    def _gpt_image_channel_active(self) -> bool:
        widget = getattr(self, "gen_channel_gpt", None)
        return bool(widget is not None and widget.isChecked())

    def _on_gen_channel_changed(self):
        active = self._gpt_image_channel_active()
        basic = getattr(self, "gpt_basic_controls", None)
        if basic is not None:
            basic.setVisible(active)
        advanced = getattr(self, "gpt_advanced_toggle", None)
        for name in ("gpt_pp_row", "gpt_param_row"):
            row = getattr(self, name, None)
            if row is not None:
                row.setVisible(active and advanced is not None and advanced.isChecked())

    def _build_gpt_image_steps(self) -> dict:
        from utils.analysis_gen import pipeline_steps_from_flags
        region = getattr(self, "gpt_pp_region", None)
        data = region.currentData() if region is not None else None
        regions = [str(r) for r in data] if isinstance(data, (list, tuple)) else [str(data or "hair")]
        tone_target = "style"
        combo = getattr(self, "gpt_pp_tone_target", None)
        if combo is not None:
            tone_target = str(combo.currentData() or "style")
        scope_combo = getattr(self, "gpt_pp_scope", None)
        # 线条优先配方：Gemini 同时接收 GPT 首图和完整画风图；身份偏离交给两轮闭环修订。
        return pipeline_steps_from_flags(
            repaint=bool(getattr(self, "gpt_pp_repaint", None) and self.gpt_pp_repaint.isChecked()),
            structure=bool(getattr(self, "gpt_pp_structure", None) and self.gpt_pp_structure.isChecked()),
            local=bool(getattr(self, "gpt_pp_local", None) and self.gpt_pp_local.isChecked()),
            local_regions=regions,
            tone=bool(getattr(self, "gpt_pp_tone", None) and self.gpt_pp_tone.isChecked()),
            tone_target=tone_target,
            ink=bool(getattr(self, "gpt_pp_ink", None) and self.gpt_pp_ink.isChecked()),
            repaint_ref_mode="style",
            repaint_scope=str(scope_combo.currentData() or "full") if scope_combo is not None else "full",
        )

    def _pipeline_timeout_budget(self, timeout_seconds, steps) -> int:
        """超时预算 = 每道**联网**工序一份；重绘另预留初审和最多两轮定点修订/复审。

        用户要求：不要用一个总体 120 秒掐掉整条链（high 画质首图常要 120 秒以上，
        后面还有重绘/局部重绘），而是「一道工序 120 秒」。
        结构线/色调校准/加墨是本地工序（秒级），不算配额，否则超时窗口会被它们虚高；
        而局部重绘是多区域的**逐个 API 调用**，每个区域各占一份（默认四区链 → 6 份 = 720 秒）。
        """
        steps = steps or {}
        slots = 1                                                   # 首图
        if (steps.get("repaint") or {}).get("enabled"):
            slots += 9  # 重绘 + 质量审计/一次修订/复审 + 身份初审/最多两次修订及复审
        if (steps.get("local") or {}).get("enabled"):
            regions = (steps["local"].get("regions") or [steps["local"].get("region") or "hair"])
            slots += max(1, len([r for r in regions if str(r).strip()]))
        return max(1, int(timeout_seconds or 120)) * slots

    def _start_gpt_image_thread(self, prompt_type, prompt_context, task_hash, selected_style_name,
                               styles_data, is_auto=False, analysis_thread_no=None, auto_group_id=None,
                               timeout_seconds=120):
        """gpt-image-2 通道：画风 prompt_gpt + 身份完整锚 → 新建图片 → 工序加工。"""
        from utils.analysis_gen import build_first_pass_request

        analysis_json_path = prompt_context.get("analysis_json_path") or getattr(self, "_last_saved_json_path", "")
        analysis_result = {}
        try:
            import json as _json
            if analysis_json_path and os.path.isfile(analysis_json_path):
                with open(analysis_json_path, encoding="utf-8") as f:
                    analysis_result = _json.load(f) or {}
        except Exception as exc:  # noqa: BLE001
            self.log_msg(f"[gpt 通道] 读取分析产物失败（改用描述文本）: {exc}")

        # 挂画风参考图时优先使用约 1400 字符的身份完整锚：实测 4400+ 字符分析全文会压弱
        # 画风图，而 500 字符短锚容易漏掉发色/瞳色。完整锚同时保留角色颜色、服装和构图。
        content_text = str(analysis_result.get("gpt_image_prompt") or "").strip()
        if not content_text:
            content_text = str(prompt_context.get("original_prompt", "") if prompt_type == "original"
                               else prompt_context.get("refined_prompt", "") or "").strip()
        if not content_text:
            content_text = str(analysis_result.get("english_description")
                               or analysis_result.get("original_english_description") or "").strip()

        # 首图请求统一从 build_first_pass_request 组装（画风说明 + 内容锚 + 排除句 + 渲染语言条款）。
        # 这里以前自己拼、漏传 extra_clauses，导致 GUI 首图永远没有 RENDERING LANGUAGE 段
        # （CLI 是传了的）——2026-09-24 定位并修掉。
        request_payload = build_first_pass_request(
            styles_data, selected_style_name, analysis_result,
            content_text=content_text, tier="short")
        ref = str(request_payload.get("style_ref_path") or "")
        style_clauses = list(request_payload.get("clauses") or [])
        _clause_note = {"entry": "画风自带", "derived": "按 prompt_gpt 派生", "none": "无"}.get(
            str(request_payload.get("clauses_source") or "none"), "无")
        self.log_msg(f"[gpt 通道] 画风 {selected_style_name or '默认'}：说明 {request_payload['style_chars']} 字符、"
                     f"内容描述 {request_payload['content_chars']} 字符、"
                     f"参考图 {'画风参考图（不挂分析素材图）' if request_payload['image_paths'] else '无'}、"
                     f"渲染条款 {len(style_clauses)} 条（{_clause_note}）")

        steps = self._build_gpt_image_steps()
        enabled_steps = [k for k, v in steps.items() if (v or {}).get("enabled")]
        # 超时按「一道工序一份」算：首图 + 每道工序各给一份（用户要求：不要总体 120s 掐掉整条链）
        step_timeout = int(timeout_seconds or 120)
        timeout_budget = self._pipeline_timeout_budget(step_timeout, steps)
        self.log_msg(f"[gpt 通道] 超时预算 {timeout_budget} 秒 = 每道工序 {step_timeout} 秒 × "
                     f"(首图 1 + 工序 {len(enabled_steps)})")
        firmware = ""
        if steps.get("repaint", {}).get("enabled") or steps.get("local", {}).get("enabled"):
            try:
                from utils.gpt_image_optimize import load_config, read_prompt_relative
                conf = load_config() or {}
                firmware = read_prompt_relative(str(conf.get("system_prompt") or ""))
            except Exception as exc:  # noqa: BLE001
                self.log_msg(f"[gpt 通道] 读取重绘固件失败（将用默认固件）: {exc}")

        self.gen_orig_btn.setEnabled(False)
        self.gen_ref_btn.setEnabled(False)
        quality = str(getattr(self, "gpt_quality_combo", None).currentData()
                      if getattr(self, "gpt_quality_combo", None) else "high") or "high"
        size = None
        if getattr(self, "gpt_size_follow_cb", None) is None or self.gpt_size_follow_cb.isChecked():
            from modules.others.api_backend import pick_gpt_image2_size_for_images
            images_for_size = list(request_payload.get("image_paths") or [])
            source_for_size = str((prompt_context or {}).get("source_image_path") or "").strip()
            if source_for_size and os.path.isfile(source_for_size):
                images_for_size.insert(0, source_for_size)
            if not images_for_size:
                source_for_size = str(self._current_source_path() or "")
                if source_for_size and os.path.isfile(source_for_size):
                    images_for_size.append(source_for_size)
            size = pick_gpt_image2_size_for_images(images_for_size)
            self.log_msg(f"[gpt 通道] 尺寸 {size}（按输入图比例自动选）")

        thread = GptImageGenWorkerThread(
            request_payload=request_payload, steps=steps, firmware=firmware,
            quality=quality, file_prefix=task_hash or "analysis-gpt",
            mode=str(self.gpt_first_pass_mode.currentData() or "generate"),
            style_ref_path=ref,
            style_clauses=style_clauses,
            analysis_result=analysis_result,
            **({"size": size} if size else {}),
        )
        self.log_msg(f"[gpt 通道] 画质 {quality}（high 的细节 token 约为 medium 的 5 倍）")
        thread.meta_thread_no = self._next_thread_no("_image_gen_thread_seq")
        thread.meta_analysis_thread_no = analysis_thread_no
        thread.meta_is_auto = bool(is_auto)
        thread.meta_prompt_type = prompt_type
        thread.meta_auto_group_id = auto_group_id
        thread.meta_task_hash = task_hash
        thread.meta_analysis_json_path = analysis_json_path
        self._active_img_threads.append(thread)
        if not self._img_gen_running:
            self._start_image_gen_runtime(timeout_budget)
        thread.log_signal.connect(
            lambda text, t=thread: self.log_msg(
                text,
                prefix=self._build_thread_prefix("gpt生图线程",
                                                 getattr(t, "meta_thread_no", "?"),
                                                 getattr(t, "meta_analysis_thread_no", None))))
        thread.finish_signal.connect(lambda files, t=thread: self.on_image_generation_finished(t, files))
        thread.finished.connect(lambda t=thread: self._on_image_thread_stopped(t))
        thread.start()
        return True

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
            "analysis_json_path": getattr(self, "_last_saved_json_path", ""),
        }
        task_hash = str(prompt_context.get("task_hash") or self.current_task_hash or "").strip()
        prompt_to_use = prompt_context.get("original_prompt", "") if prompt_type == "original" else prompt_context.get("refined_prompt", "")
        if not str(prompt_to_use).strip():
            # gpt 通道的内容来自「分析产物里的描述」，本窗口没带描述时不该直接跳过
            _json_for_gpt = str(prompt_context.get("analysis_json_path") or "").strip()
            if self._gpt_image_channel_active() and _json_for_gpt and os.path.isfile(_json_for_gpt):
                self.log_msg("[gpt 通道] 本次没有 prompt 文本 → 直接用分析产物里的描述作为内容")
            else:
                self.log_msg(f"{prompt_type} 提示词为空，已跳过本次生图。")
                return False
        # 添加面部质量保护后缀，防止因原图打码等原因导致生成模糊面部
        _face_quality_suffix = "detailed face, clear facial features, sharp focus on face"
        prompt_to_use = f"{prompt_to_use}, {_face_quality_suffix}"
        
        selected_style_name = self.main_style_combo.currentText()
        styles_data = self.get_styles() or {}
        has_ref = ref_image_valid(style_ref_image(styles_data, selected_style_name))
        active_mode = self.style_ref_mode_combo.effective_mode(has_ref)
        active_instructions, post_instructions, style_ref_paths = build_ref_gen_params(
            styles_data, selected_style_name, active_mode
        )
        
        self.gen_orig_btn.setEnabled(False)
        self.gen_ref_btn.setEnabled(False)
        
        # 【修改】动态实例化线程对象存放至列表，避免并发勾选导致线程互相覆盖报错
        final_gen_ar = self._resolve_ar_for_second_stage(prompt_context.get("aspect_ratio", self.current_aspect_ratio))

        self._save_gpt_pipeline_ui()
        # ---- gpt-image-2 通道：用 gpt 专用短锚 + 画风参考图出图，再按勾选跑工序 ----
        if self._gpt_image_channel_active():
            return self._start_gpt_image_thread(
                prompt_type=prompt_type, prompt_context=prompt_context, task_hash=task_hash,
                selected_style_name=selected_style_name, styles_data=styles_data,
                is_auto=is_auto, analysis_thread_no=analysis_thread_no, auto_group_id=auto_group_id,
                timeout_seconds=timeout_seconds)

        img_thread = ImageGenWorkerThread(
            prompt=prompt_to_use,
            model_name=model_name,
            aspect_ratio=final_gen_ar,
            instructions=active_instructions,
            api_type=api_type,
            image_paths=style_ref_paths,
            post_instructions=post_instructions,
            file_prefix=task_hash
        )
        img_thread.meta_thread_no = self._next_thread_no("_image_gen_thread_seq")
        img_thread.meta_analysis_thread_no = analysis_thread_no
        img_thread.meta_is_auto = bool(is_auto)
        img_thread.meta_prompt_type = prompt_type
        img_thread.meta_auto_group_id = auto_group_id
        img_thread.meta_task_hash = task_hash
        img_thread.meta_analysis_json_path = prompt_context.get("analysis_json_path", getattr(self, "_last_saved_json_path", ""))

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
            task_hash_of_thread = str(getattr(thread, "meta_task_hash", "") or "").strip()
            self._last_gen_task_hash = task_hash_of_thread
            self._sync_analysis_mtimes(thread, saved_files)
            self._start_jpg_postprocess(saved_files, prompt_type, task_hash=task_hash_of_thread)
            if task_hash_of_thread:
                # 生图产物先记下（后处理结束后它就是"最终产物"的兜底）
                for _tid in self._history_task_ids_for_hash(task_hash_of_thread):
                    rec = self._analysis_history.get(_tid) or {}
                    rec["final_products"] = list(saved_files)
                    rec["pipeline_error"] = ""
                if not self._finalize_task_pipeline(task_hash_of_thread, final_products=saved_files):
                    self.log_msg("⏳ 生图完成，但管线还没收尾（后处理/线程退出中）：队列保持「进行中」，全部跑完才标记完成。")
        else:
            status = getattr(thread, "last_status", "unknown")
            if status == "cancelled":
                self.log_msg(f"🛑 {prompt_type} 生图已取消。", prefix=thread_prefix)
            else:
                self.log_msg(f"⚠️ 未能生成 {prompt_type} 图片，请检查上方日志，或查看日志文件夹（log）的记录。", prefix=thread_prefix)
            # 没有产物的生图线程一旦退出，队列就不能再等产物了：
            # `_on_image_thread_stopped` 收到这条标记后会把记录标红失败，而不是永远「等待最终产物」
            task_hash_of_thread = str(getattr(thread, "meta_task_hash", "") or "").strip()
            if task_hash_of_thread:
                reason = "生图已取消" if status == "cancelled" else "生图失败（无产物）"
                for _tid in self._history_task_ids_for_hash(task_hash_of_thread):
                    rec = self._analysis_history.get(_tid)
                    if rec is not None:
                        rec["pipeline_error"] = reason
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

    def _sync_file_times_to_source(self, filepath, source_path):
        """将目标文件的修改时间（mtime/atime）同步到与源文件一致。"""
        try:
            if not os.path.exists(filepath) or not os.path.exists(source_path):
                return
            src_stat = os.stat(source_path)
            os.utime(filepath, (src_stat.st_atime, src_stat.st_mtime))
        except Exception:
            pass  # 静默失败，不影响主流程

    def _sync_analysis_mtimes(self, thread, saved_files):
        """将分析文件（json、两个txt）的修改时间同步到与生成的第一张图片一致"""
        analysis_json_path = getattr(thread, "meta_analysis_json_path", None)
        if not analysis_json_path:
            self.log_msg("⏭ 同步 mtime 跳过：未关联分析 JSON 路径（meta_analysis_json_path 为空）")
            return
        if not saved_files:
            self.log_msg("⏭ 同步 mtime 跳过：saved_files 为空，无生成图片")
            return

        first_img = saved_files[0]
        self.log_msg("🕐 开始同步分析文件 mtime，基准图片: %s" % os.path.basename(first_img))
        try:
            if not os.path.exists(first_img):
                self.log_msg("⚠️ 同步 mtime 跳过：图片文件不存在 %s" % first_img)
                return
            if not os.path.exists(analysis_json_path):
                self.log_msg("⚠️ 同步 mtime 跳过：分析 JSON 文件不存在 %s" % analysis_json_path)
                return

            img_mtime = os.stat(first_img).st_mtime
            img_mtime_str = datetime.datetime.fromtimestamp(img_mtime).strftime("%H:%M:%S")
            self.log_msg("  基准图片 mtime: %s" % img_mtime_str)

            synced = 0
            for fpath, label in ((analysis_json_path, "JSON"),
                                  (analysis_json_path.rsplit(".json", 1)[0] + "-prompts.txt", "优化提示词"),
                                  (analysis_json_path.rsplit(".json", 1)[0] + "-original-prompts.txt", "原始提示词")):
                if os.path.exists(fpath):
                    old_mtime = os.stat(fpath).st_mtime
                    old_str = datetime.datetime.fromtimestamp(old_mtime).strftime("%H:%M:%S")
                    os.utime(fpath, (os.stat(fpath).st_atime, img_mtime))
                    synced += 1
                    self.log_msg("  ✓ %s : %s → %s  (%s)" % (label, old_str, img_mtime_str, os.path.basename(fpath)))
                else:
                    self.log_msg("  ✗ %s 文件不存在，跳过 (%s)" % (label, fpath))

            self.log_msg("🕐 同步完成: %d/3 个文件已同步 mtime" % synced)
        except Exception as e:
            self.log_msg("⚠️ 同步分析文件 mtime 失败: %s" % e)

    def _start_jpg_postprocess(self, saved_files, prompt_type, task_hash=""):
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
        thread.meta_task_hash = str(task_hash or getattr(self, "_last_gen_task_hash", "") or "")
        self._active_post_threads.append(thread)
        self._mark_pipeline_busy()
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
        task_hash = str(getattr(thread, "meta_task_hash", "") or "").strip()
        if thread in self._active_post_threads:
            self._active_post_threads.remove(thread)
        self._mark_pipeline_idle_if_done()
        if task_hash:
            self._finalize_task_pipeline(task_hash)

    # ==================== 目录批量选择 ====================

    def _open_directory_batch_selector(self):
        """打开目录批量选择对话框，用户确认后将图片加入分析队列。"""
        self._commit_outfit_style_text(self.outfit_style_combo.currentText(), add_to_history=True)

        # 预检查：缺少的 prompt 文件
        enable_outfit_check = self.enable_outfit_check_cb.isChecked()
        enable_remove_photo_style = self.remove_photo_style_cb.isChecked()
        enable_recompute_pixiv_tags = enable_outfit_check or enable_remove_photo_style
        missing_prompt_files = get_single_analyzer_missing_prompt_files(
            enable_refine=True,
            enable_outfit_check=enable_outfit_check,
            enable_remove_photo_style=enable_remove_photo_style,
            enable_recompute_pixiv_tags=enable_recompute_pixiv_tags,
        )
        if missing_prompt_files:
            missing_text = "\n".join(missing_prompt_files)
            QMessageBox.warning(self, "缺少 Prompt 文件", f"以下 Prompt 文件不存在，请补齐后再执行：\n{missing_text}")
            self.log_msg(f"❌ 缺少 Prompt 文件，已中止批量分析：\n{missing_text}")
            return

        base_url, api_key, model_name = self.get_text_config(self.use_nsfw_cb.isChecked())
        if not api_key or not model_name:
            QMessageBox.warning(self, "缺少配置", "文本分析 API Key 和 模型名称不能为空！")
            return

        dialog = DirectoryBatchSelectorDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        image_paths = dialog.get_selected_paths()
        if not image_paths:
            return

        # Record selected images to analyzed history so they can be excluded in future scans.
        from modules.image_analysis.dir_batch_selector import add_to_analyzed_history
        add_to_analyzed_history(image_paths)

        self._submit_batch_images(image_paths)

    def _submit_batch_images(self, image_paths):
        """将一批图片逐个提交到分析队列。"""
        self.log_msg("\n" + ("=" * 72))
        self.log_msg(f"📁 目录批量选择：共 {len(image_paths)} 张图片，开始逐个提交分析...")

        submitted_count = 0
        for filepath in image_paths:
            if not os.path.isfile(filepath):
                self.log_msg(f"  ⚠️ 跳过（文件不存在）: {filepath}")
                continue

            self.image_source = filepath
            self.show_preview(filepath)

            thread = self._launch_analysis_task(
                _clone_image_source(filepath),
                header_note=f"--- [批量 #{submitted_count + 1}/{len(image_paths)}] {os.path.basename(filepath)} ---",
            )
            if thread is None:
                self.log_msg("  ⚠️ 批量提交中止（配置或 prompt 文件校验未通过）。")
                break
            submitted_count += 1

        self.log_msg("\n" + ("=" * 72))
        self.log_msg(f"📁 目录批量选择：已提交 {submitted_count}/{len(image_paths)} 张图片进入分析队列。")
