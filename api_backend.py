# api_backend.py
from logging import config
import os
import json
import base64
import requests
import re
import logging
import copy
import uuid
import sys
from contextlib import ExitStack
from datetime import datetime
import time
import mimetypes
from utils.booru_tags import normalize_booru_tags

# ================= 1. 日志系统配置 =================
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("whatai_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter(fmt='[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    # 在 Windows 环境尽量确保控制台输出使用 UTF-8，避免 emoji 触发 gbk 编码异常
    try:
        if hasattr(console_handler.stream, "reconfigure"):
            console_handler.stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    console_handler.setFormatter(formatter)
    
    today_date = datetime.now().strftime("%Y-%m-%d")
    log_file_path = os.path.join(LOG_DIR, f"{today_date}.log")
    file_handler = logging.FileHandler(filename=log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ================= 2. 核心功能函数 =================
def load_config(config_path="config-image.json"):
    if not os.path.exists(config_path):
        logger.error(f"未找到配置文件: {config_path}")
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_api_config(config_path="config-image.json", api_type=None):
    """获取指定API类型的配置"""
    config = load_config(config_path)
    # 如果指定了api_type，直接返回对应配置
    if api_type:
        return config.get("apis", {}).get(api_type, {})
    # 否则返回当前API类型的配置
    current_api = config.get("current_api", "whatup")
    return config.get("apis", {}).get(current_api, {})

def to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def _extract_json_object(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {}
    decoder = json.JSONDecoder()
    code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates = code_blocks + [text]
    for candidate in candidates:
        content = candidate.strip()
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        for idx, ch in enumerate(content):
            if ch != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(content[idx:])
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
    return {}

def _normalize_annotation_result(result_json: dict, booru_tag_limit: int = 30) -> dict:
    if not isinstance(result_json, dict):
        return {}
    long_description = (
        result_json.get("long_description")
        or result_json.get("longDescription")
        or result_json.get("description")
        or ""
    )
    short_description = (
        result_json.get("short_description")
        or result_json.get("shortDescription")
        or ""
    )
    booru_tags = result_json.get("booru-tags")
    if booru_tags is None:
        booru_tags = result_json.get("booru_tags")
    if booru_tags is None:
        booru_tags = result_json.get("booruTags")
    if booru_tags is None:
        booru_tags = result_json.get("tags")
    limit = int(booru_tag_limit) if str(booru_tag_limit).strip().isdigit() else 30
    if limit <= 0:
        limit = 30
    normalized = {
        "description": str(long_description).strip(),
        "long_description": str(long_description).strip(),
        "short_description": str(short_description).strip(),
        "booru-tags": normalize_booru_tags(booru_tags, limit=limit)
    }
    if any(normalized.values()):
        return normalized
    return {}

def _looks_like_base64_text(value: str) -> bool:
    if not isinstance(value, str):
        return False
    compact = value.strip().replace("\n", "").replace("\r", "")
    if len(compact) < 256:
        return False
    if not re.fullmatch(r"[A-Za-z0-9+/=]+", compact):
        return False
    return True

def _sanitize_log_data(value):
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            key_lower = str(key).lower()
            if isinstance(item, str) and ("base64" in key_lower or key_lower.startswith("b64") or key_lower == "data"):
                if _looks_like_base64_text(item) or key_lower != "data":
                    sanitized[key] = "<BASE64_IMAGE_DATA_OMITTED>"
                    continue
            sanitized[key] = _sanitize_log_data(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_log_data(item) for item in value]
    if isinstance(value, str):
        sanitized_text = re.sub(
            r"(data:image\/[a-zA-Z0-9.+-]+;base64,)[A-Za-z0-9+/=\r\n]+",
            r"\1<BASE64_IMAGE_DATA_OMITTED>",
            value
        )
        if _looks_like_base64_text(sanitized_text):
            return "<BASE64_IMAGE_DATA_OMITTED>"
        if len(sanitized_text) > 4000:
            return f"{sanitized_text[:4000]}...(TRUNCATED, total={len(sanitized_text)})"
        return sanitized_text
    return value

def _format_safe_log(value) -> str:
    sanitized = _sanitize_log_data(value)
    if isinstance(sanitized, str):
        return sanitized
    try:
        return json.dumps(sanitized, ensure_ascii=False, indent=2)
    except Exception:
        return str(sanitized)

def _log_stage_elapsed(stage_name: str, start_time: float):
    try:
        elapsed = time.perf_counter() - float(start_time)
    except Exception:
        elapsed = 0.0
    logger.info(f"[耗时] {stage_name}: {elapsed:.3f} 秒")

def _save_server_response_json(save_dir: str, file_prefix: str, api_tag: str, resp_json) -> str:
    try:
        os.makedirs(save_dir, exist_ok=True)
        prefix = f"{file_prefix}_" if file_prefix else ""
        filename = f"{prefix}{api_tag}_server_response_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}.json"
        output_path = os.path.join(save_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(resp_json, f, ensure_ascii=False, indent=2)
        logger.warning(f"⚠️ 图片获取失败，已将服务器返回 JSON 保存到: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"保存服务器返回 JSON 失败: {e}")
        return ""

def _response_text_utf8(resp) -> str:
    """统一按 UTF-8 解析响应文本，避免 requests 在部分场景错误推断为 gbk。"""
    if resp is None:
        return ""
    try:
        if hasattr(resp, "content") and resp.content is not None:
            return resp.content.decode("utf-8", errors="replace")
    except Exception:
        pass
    try:
        if getattr(resp, "encoding", None) is None:
            resp.encoding = "utf-8"
        return resp.text or ""
    except Exception:
        return ""

def _save_server_response_raw(save_dir: str, file_prefix: str, api_tag: str, raw_text: str) -> str:
    try:
        os.makedirs(save_dir, exist_ok=True)
        prefix = f"{file_prefix}_" if file_prefix else ""
        filename = f"{prefix}{api_tag}_server_raw_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}.txt"
        output_path = os.path.join(save_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(str(raw_text or ""))
        logger.warning(f"服务器原始响应已保存到: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"保存服务器原始响应失败: {e}")
        return ""

def _as_bool(value, default: bool = False) -> bool:
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

def _to_unmasked_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return str(value)
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except Exception:
            return str(value)
    return str(value)

def _save_debug_http_trace(
    save_dir: str,
    file_prefix: str,
    api_tag: str,
    request_url: str = "",
    request_headers=None,
    request_body=None,
    response=None,
    response_body: str = None,
    note: str = ""
) -> str:
    """
    Debug模式专用：完整保存请求/响应，不做任何脱敏。
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        prefix = f"{file_prefix}_" if file_prefix else ""
        filename = f"{prefix}{api_tag}_http_debug_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}.json"
        output_path = os.path.join(save_dir, filename)

        req_headers = request_headers
        req_body = request_body
        if response is not None and hasattr(response, "request") and response.request is not None:
            if req_headers is None:
                req_headers = dict(getattr(response.request, "headers", {}) or {})
            if req_body is None:
                req_body = getattr(response.request, "body", None)
            if not request_url:
                request_url = str(getattr(response.request, "url", "") or "")

        resp_headers = {}
        resp_status_code = None
        resp_reason = ""
        if response is not None:
            try:
                resp_headers = dict(getattr(response, "headers", {}) or {})
            except Exception:
                resp_headers = {}
            resp_status_code = getattr(response, "status_code", None)
            resp_reason = str(getattr(response, "reason", "") or "")
            if response_body is None:
                response_body = _response_text_utf8(response)

        trace_payload = {
            "timestamp": datetime.now().isoformat(),
            "note": str(note or ""),
            "request": {
                "url": str(request_url or ""),
                "headers": req_headers if req_headers is not None else {},
                "body": _to_unmasked_text(req_body)
            },
            "response": {
                "status_code": resp_status_code,
                "reason": resp_reason,
                "headers": resp_headers,
                "body": _to_unmasked_text(response_body)
            }
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(trace_payload, f, ensure_ascii=False, indent=2)
        logger.warning(f"[DEBUG] 已保存完整HTTP请求/响应: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"[DEBUG] 保存完整HTTP请求/响应失败: {e}")
        return ""

def _aspect_ratio_to_dalle3_size(aspect_ratio: str, allow_auto_for_extreme: bool = True) -> str:
    """
    将常见长宽比映射到 GPT Image 可用 size（3挡）：
    - 1024x1536（竖图）
    - 1024x1024（方图）
    - 1536x1024（横图）
    """
    default_size = "1024x1024"
    if not aspect_ratio or not isinstance(aspect_ratio, str):
        return default_size
    ratio_text = aspect_ratio.strip()
    if ":" in ratio_text:
        try:
            w, h = ratio_text.split(":", 1)
            ratio = float(w) / float(h)
        except Exception:
            return default_size
    else:
        try:
            ratio = float(ratio_text)
        except Exception:
            return default_size
    portrait_ratio = 1024 / 1536
    landscape_ratio = 1536 / 1024
    if allow_auto_for_extreme and (ratio < portrait_ratio or ratio > landscape_ratio):
        return "auto"

    candidates = [
        ("1024x1536", portrait_ratio),
        ("1024x1024", 1.0),
        ("1536x1024", landscape_ratio),
    ]
    return min(candidates, key=lambda item: abs(item[1] - ratio))[0]

def _ensure_openrouter_generations_url(api_base: str, api_path: str = "/v1/images/generations") -> str:
    base = str(api_base or "https://openrouter.ai/api").rstrip("/")
    path = str(api_path or "/v1/images/generations").strip()
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{base}{path}"

def _download_image_with_referer_retry(img_url: str, timeout_s: int = 30, referer: str = None) -> bytes:
    headers = {}
    try:
        resp = requests.get(img_url, timeout=timeout_s, headers=headers)
        resp.raise_for_status()
        return resp.content
    except requests.exceptions.HTTPError as e:
        status_code = getattr(getattr(e, "response", None), "status_code", None)
        if status_code == 403 and referer:
            retry_headers = {"Referer": referer}
            retry_resp = requests.get(img_url, timeout=timeout_s, headers=retry_headers)
            retry_resp.raise_for_status()
            return retry_resp.content
        raise

def _guess_image_mime_type(img_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(img_path)
    if mime_type and str(mime_type).startswith("image/"):
        return str(mime_type)
    return "image/jpeg"

def _existing_image_paths(image_paths: list = None) -> list:
    valid_paths = []
    for img_path in list(image_paths or []):
        if img_path and os.path.exists(img_path):
            valid_paths.append(img_path)
        else:
            logger.warning(f"找不到本地图片文件 {img_path}，已跳过。")
    return valid_paths

def _derive_edits_url_from_generations_url(generations_url: str) -> str:
    normalized = str(generations_url or "").rstrip("/")
    if normalized.endswith("/images/generations"):
        return normalized[:-len("/images/generations")] + "/images/edits"
    if normalized.endswith("/v1"):
        return f"{normalized}/images/edits"
    return f"{normalized}/v1/images/edits"

def _post_images_edits_request(url: str, api_key: str, form_payload: dict, image_paths: list, timeout_val: int):
    used_files = []
    with ExitStack() as stack:
        files = []
        for img_path in image_paths:
            fh = stack.enter_context(open(img_path, "rb"))
            mime_type = _guess_image_mime_type(img_path)
            files.append(("image[]", (os.path.basename(img_path), fh, mime_type)))
            used_files.append({"name": os.path.basename(img_path), "path": img_path, "mime_type": mime_type})
        data_payload = {k: str(v) for k, v in (form_payload or {}).items() if v is not None}
        headers = {"Authorization": f"Bearer {api_key}"}
        resp = requests.post(url, headers=headers, data=data_payload, files=files, timeout=timeout_val)
    return resp, used_files

def generate_image_openai_image(prompt: str, image_paths: list = None, model: str = "gpt-image-2", aspect_ratio: str = "1:1", instructions: str = "", resolution: str = None, api_type: str = None, save_sub_dir: str = None, file_prefix: str = None, return_metadata: bool = False) -> list:
    """
    openai-image 生图接口（OpenAI 风格）：
    - 入口端点：/v1/images/generations
    - 请求体风格：OpenAI Image API
    - 默认模型：gpt-image-2
    """
    normalized_api_type = str(api_type or "").strip().lower()
    if normalized_api_type in {"aigc-2d-gpt", "aigc2d-gpt", "aigc_2d_gpt"}:
        normalized_api_type = "openai-image"
    config = get_api_config(api_type=normalized_api_type or "openai-image")
    api_base = str(config.get("base_url", "https://api.openai.com/v1") or "https://api.openai.com/v1").rstrip("/")
    if api_base.endswith("/v1"):
        url = f"{api_base}/images/generations"
    else:
        url = f"{api_base}/v1/images/generations"
    api_key = config.get("api_key")
    timeout_val = config.get("timeout", 180)
    max_retries = config.get("max_retries", 1)
    debug_dump_full_http = _as_bool(config.get("debug_dump_full_http", False), False)

    if not api_key:
        logger.error("配置文件 config-image.json 中缺少 'api_key' 参数。")
        return []

    valid_image_paths = _existing_image_paths(image_paths)

    output_format = str(config.get("output_format", "png") or "png").lower()
    quality = str(config.get("quality", "high") or "high")
    raw_allow_auto_size = config.get("allow_auto_size", True)
    if isinstance(raw_allow_auto_size, str):
        allow_auto_size = raw_allow_auto_size.strip().lower() not in ("0", "false", "no", "off")
    else:
        allow_auto_size = bool(raw_allow_auto_size)
    request_size = _aspect_ratio_to_dalle3_size(aspect_ratio, allow_auto_for_extreme=allow_auto_size)
    final_prompt = f"{instructions}\n\n{prompt}".strip() if instructions else str(prompt or "")

    use_model = str(model or config.get("model") or "gpt-image-2").strip() or "gpt-image-2"
    payload = {
        "model": use_model,
        "prompt": final_prompt,
        "size": request_size,
        "quality": quality,
        "n": 1,
        "response_format": "b64_json",
        "output_format": output_format
    }
    use_edits_mode = len(valid_image_paths) > 0
    request_url = _derive_edits_url_from_generations_url(url) if use_edits_mode else url
    if use_edits_mode:
        logger.info(f"openai-image 将使用 /images/edits 模式，附件数量: {len(valid_image_paths)}")
    else:
        logger.info("openai-image 使用 /images/generations 模式（无附件）。")

    logger.info("=== 发起 openai-image API 请求 ===")
    logger.info(f"请求 URL: {request_url}")
    logger.info(f"请求数据:\n{_format_safe_log(payload)}")

    # ================= 阶段1：请求并获取 JSON 响应 =================
    stage_json_start = time.perf_counter()
    resp = None
    request_trace_body = payload
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"正在进行第 {attempt} 次重试 (最大重试次数: {max_retries})...")
            if use_edits_mode:
                resp, used_files = _post_images_edits_request(
                    url=request_url,
                    api_key=api_key,
                    form_payload=payload,
                    image_paths=valid_image_paths,
                    timeout_val=timeout_val
                )
                request_trace_body = {"form": payload, "files": used_files}
            else:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                resp = requests.post(request_url, headers=headers, json=payload, timeout=timeout_val)
                request_trace_body = payload
            resp.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            logger.warning(f"网络请求发生异常 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                time.sleep(2)
            else:
                logger.error("达到最大重试次数，openai-image 图片生成请求最终失败。")
                fail_today = datetime.now().strftime("%Y%m%d")
                fail_dir = os.path.join("data", fail_today, save_sub_dir) if save_sub_dir else os.path.join("data", fail_today)
                if debug_dump_full_http:
                    _save_debug_http_trace(
                        save_dir=fail_dir,
                        file_prefix=file_prefix,
                        api_tag="openai-image",
                        request_url=request_url,
                        request_headers={"Authorization": f"Bearer {api_key}"},
                        request_body=request_trace_body,
                        response=resp,
                        note=f"request_exception_retry_exhausted: {e}"
                    )
                if resp is not None:
                    _save_server_response_raw(fail_dir, file_prefix, "openai-image", _response_text_utf8(resp))
                return []

    try:
        if getattr(resp, "encoding", None) is None:
            resp.encoding = "utf-8"
        resp_json = resp.json()
        logger.info(f"=== openai-image 服务器原始返回信息 ===\n{_format_safe_log(resp_json)}")
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"解析 openai-image 返回 JSON 失败: {e}")
        fail_today = datetime.now().strftime("%Y%m%d")
        fail_dir = os.path.join("data", fail_today, save_sub_dir) if save_sub_dir else os.path.join("data", fail_today)
        if debug_dump_full_http:
            _save_debug_http_trace(
                save_dir=fail_dir,
                file_prefix=file_prefix,
                api_tag="openai-image",
                request_url=request_url,
                request_headers={"Authorization": f"Bearer {api_key}"},
                request_body=request_trace_body,
                response=resp,
                note=f"json_parse_error: {e}"
            )
        _save_server_response_raw(fail_dir, file_prefix, "openai-image", _response_text_utf8(resp))
        return []
    _log_stage_elapsed("阶段1-获取JSON响应", stage_json_start)

    today_str = datetime.now().strftime("%Y%m%d")
    save_dir = os.path.join("data", today_str, save_sub_dir) if save_sub_dir else os.path.join("data", today_str)
    os.makedirs(save_dir, exist_ok=True)
    if debug_dump_full_http:
        _save_debug_http_trace(
            save_dir=save_dir,
            file_prefix=file_prefix,
            api_tag="openai-image",
            request_url=request_url,
            request_headers={"Authorization": f"Bearer {api_key}"},
            request_body=request_trace_body,
            response=resp,
            note="success_response_received"
        )
        _save_server_response_json(save_dir, file_prefix, "openai-image", resp_json)

    data_items = resp_json.get("data", [])
    if not data_items:
        logger.warning("openai-image 返回的 JSON 中没有找到 data 节点。")
        _save_server_response_json(save_dir, file_prefix, "openai-image", resp_json)
        _save_server_response_raw(save_dir, file_prefix, "openai-image", _response_text_utf8(resp))
        return []

    saved_files = []
    revised_prompt_parts = []

    # ================= 阶段2：提取并保存图片 =================
    stage_save_start = time.perf_counter()
    for idx, item in enumerate(data_items):
        b64_data = item.get("b64_json")
        revised_prompt = item.get("revised_prompt")
        if revised_prompt:
            revised_prompt_parts.append(str(revised_prompt))

        if b64_data:
            try:
                image_bytes = base64.b64decode(b64_data)
                ext = f".{output_format}" if output_format in ("png", "jpg", "jpeg", "webp") else ".png"
                prefix = f"{file_prefix}_" if file_prefix else ""
                file_name = f"{prefix}output_{datetime.now().strftime('%H%M%S')}_{idx}_{uuid.uuid4().hex[:6]}{ext}"
                file_path = os.path.join(save_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(image_bytes)
                logger.info(f"✅ 成功保存图片 ({ext} 格式): {file_path}")
                saved_files.append(file_path)
                continue
            except Exception as e:
                logger.error(f"写入 Base64 图片失败: {e}")

        img_url = item.get("url")
        if img_url:
            try:
                img_resp = requests.get(img_url, timeout=30)
                img_resp.raise_for_status()
                img_data = img_resp.content
                ext = ".png"
                if img_data.startswith(b"\xff\xd8"):
                    ext = ".jpg"
                elif img_data.startswith(b"RIFF") and img_data[8:12] == b"WEBP":
                    ext = ".webp"
                elif img_data.startswith(b"GIF87a") or img_data.startswith(b"GIF89a"):
                    ext = ".gif"
                prefix = f"{file_prefix}_" if file_prefix else ""
                file_name = f"{prefix}output_{datetime.now().strftime('%H%M%S')}_{idx}_{uuid.uuid4().hex[:6]}{ext}"
                file_path = os.path.join(save_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(img_data)
                logger.info(f"✅ 成功保存图片 ({ext} 格式): {file_path}")
                saved_files.append(file_path)
            except Exception as e:
                logger.error(f"下载图片失败 {img_url}: {e}")
    _log_stage_elapsed("阶段2-提取并保存图片", stage_save_start)
    logger.info(f"[阶段统计] data节点: {len(data_items)}，成功保存: {len(saved_files)}")
    if len(saved_files) == 0:
        _save_server_response_json(save_dir, file_prefix, "openai-image", resp_json)
        _save_server_response_raw(save_dir, file_prefix, "openai-image", _response_text_utf8(resp))

    raw_text = "\n".join(revised_prompt_parts).strip()
    annotation_data = _normalize_annotation_result(_extract_json_object(raw_text))
    if return_metadata:
        return {
            "saved_files": saved_files,
            "annotation": annotation_data,
            "raw_text": raw_text
        }
    return saved_files

def generate_image_openrouter_image(prompt: str, image_paths: list = None, model: str = "gpt-image-1", aspect_ratio: str = "1:1", instructions: str = "", resolution: str = None, api_type: str = None, save_sub_dir: str = None, file_prefix: str = None, return_metadata: bool = False) -> list:
    """
    openrouter-image 生图接口（对齐 useless/image_generate.py）：
    - 入口端点：{base_url}{path}，默认 https://openrouter.ai/api + /v1/images/generations
    - 支持 timeout 重试（requests timeout 触发）
    - 支持下载 URL 时 403 自动带 Referer 重试
    """
    config = get_api_config(api_type=api_type or "openrouter-image")
    api_base = config.get("base_url", "https://openrouter.ai/api")
    api_path = config.get("path", "/v1/images/generations")
    url = _ensure_openrouter_generations_url(api_base, api_path)
    api_key = config.get("api_key")
    timeout_val = int(config.get("timeout", 180) or 180)
    max_retries = int(config.get("max_retries", 2) or 2)
    retry_backoff_s = float(config.get("retry_backoff", 1.0) or 1.0)
    no_download = bool(config.get("no_download", False))
    debug_dump_full_http = _as_bool(config.get("debug_dump_full_http", False), False)
    download_referer = f"{str(api_base).rstrip('/')}/"

    if not api_key:
        logger.error("配置文件 config-image.json 中缺少 'api_key' 参数。")
        return []

    valid_image_paths = _existing_image_paths(image_paths)

    n = int(config.get("n", 1) or 1)
    if n <= 0:
        n = 1
    payload = {
        "prompt": f"{instructions}\n\n{prompt}".strip() if instructions else str(prompt or ""),
        "n": n,
        "model": str(model or config.get("model") or "gpt-image-1").strip() or "gpt-image-1",
        "size": str(config.get("size", "1024x1024") or "1024x1024"),
        "seed": int(config.get("seed", -1) or -1),
        "aspect_ratio": str(aspect_ratio or config.get("aspect_ratio") or "").strip() or None
    }
    if payload["aspect_ratio"] is None:
        payload.pop("aspect_ratio", None)
    response_format = str(config.get("response_format", "") or "").strip()
    if response_format:
        payload["response_format"] = response_format
    output_format = str(config.get("output_format", "") or "").strip()
    if output_format:
        payload["output_format"] = output_format

    use_edits_mode = len(valid_image_paths) > 0
    request_url = _derive_edits_url_from_generations_url(url) if use_edits_mode else url
    if use_edits_mode:
        logger.info(f"openrouter-image 将使用 /images/edits 模式，附件数量: {len(valid_image_paths)}")
    else:
        logger.info("openrouter-image 使用 /images/generations 模式（无附件）。")

    logger.info("=== 发起 openrouter-image API 请求 ===")
    logger.info(f"请求 URL: {request_url}")
    logger.info(f"请求数据:\n{_format_safe_log(payload)}")

    stage_json_start = time.perf_counter()
    resp = None
    request_trace_body = payload
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"正在进行第 {attempt} 次重试 (最大重试次数: {max_retries})...")
            if use_edits_mode:
                resp, used_files = _post_images_edits_request(
                    url=request_url,
                    api_key=api_key,
                    form_payload=payload,
                    image_paths=valid_image_paths,
                    timeout_val=timeout_val
                )
                request_trace_body = {"form": payload, "files": used_files}
            else:
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                resp = requests.post(request_url, headers=headers, json=payload, timeout=timeout_val)
                request_trace_body = payload
            resp.raise_for_status()
            break
        except requests.exceptions.Timeout as e:
            logger.warning(f"请求超时 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                time.sleep(retry_backoff_s * (attempt + 1))
            else:
                logger.error("达到最大重试次数，openrouter-image 图片生成请求最终失败。")
                fail_today = datetime.now().strftime("%Y%m%d")
                fail_dir = os.path.join("data", fail_today, save_sub_dir) if save_sub_dir else os.path.join("data", fail_today)
                if debug_dump_full_http:
                    _save_debug_http_trace(
                        save_dir=fail_dir,
                        file_prefix=file_prefix,
                        api_tag="openrouter-image",
                        request_url=request_url,
                        request_headers={"Authorization": f"Bearer {api_key}"},
                        request_body=request_trace_body,
                        response=resp,
                        note=f"timeout_retry_exhausted: {e}"
                    )
                if resp is not None:
                    _save_server_response_raw(fail_dir, file_prefix, "openrouter-image", _response_text_utf8(resp))
                return []
        except requests.exceptions.RequestException as e:
            logger.error(f"openrouter-image 请求失败: {e}")
            fail_today = datetime.now().strftime("%Y%m%d")
            fail_dir = os.path.join("data", fail_today, save_sub_dir) if save_sub_dir else os.path.join("data", fail_today)
            if debug_dump_full_http:
                _save_debug_http_trace(
                    save_dir=fail_dir,
                    file_prefix=file_prefix,
                    api_tag="openrouter-image",
                    request_url=request_url,
                    request_headers={"Authorization": f"Bearer {api_key}"},
                    request_body=request_trace_body,
                    response=resp,
                    note=f"request_exception: {e}"
                )
            if resp is not None:
                _save_server_response_raw(fail_dir, file_prefix, "openrouter-image", _response_text_utf8(resp))
            return []

    try:
        if getattr(resp, "encoding", None) is None:
            resp.encoding = "utf-8"
        resp_json = resp.json()
        logger.info(f"=== openrouter-image 服务器原始返回信息 ===\n{_format_safe_log(resp_json)}")
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"解析 openrouter-image 返回 JSON 失败: {e}")
        fail_today = datetime.now().strftime("%Y%m%d")
        fail_dir = os.path.join("data", fail_today, save_sub_dir) if save_sub_dir else os.path.join("data", fail_today)
        if debug_dump_full_http:
            _save_debug_http_trace(
                save_dir=fail_dir,
                file_prefix=file_prefix,
                api_tag="openrouter-image",
                request_url=request_url,
                request_headers={"Authorization": f"Bearer {api_key}"},
                request_body=request_trace_body,
                response=resp,
                note=f"json_parse_error: {e}"
            )
        _save_server_response_raw(fail_dir, file_prefix, "openrouter-image", _response_text_utf8(resp))
        return []
    _log_stage_elapsed("阶段1-获取JSON响应", stage_json_start)

    today_str = datetime.now().strftime("%Y%m%d")
    save_dir = os.path.join("data", today_str, save_sub_dir) if save_sub_dir else os.path.join("data", today_str)
    os.makedirs(save_dir, exist_ok=True)
    if debug_dump_full_http:
        _save_debug_http_trace(
            save_dir=save_dir,
            file_prefix=file_prefix,
            api_tag="openrouter-image",
            request_url=request_url,
            request_headers={"Authorization": f"Bearer {api_key}"},
            request_body=request_trace_body,
            response=resp,
            note="success_response_received"
        )
        _save_server_response_json(save_dir, file_prefix, "openrouter-image", resp_json)

    data_items = resp_json.get("data", [])
    if not isinstance(data_items, list) or not data_items:
        logger.warning("openrouter-image 返回的 JSON 中没有找到 data 节点。")
        _save_server_response_json(save_dir, file_prefix, "openrouter-image", resp_json)
        _save_server_response_raw(save_dir, file_prefix, "openrouter-image", _response_text_utf8(resp))
        return []

    saved_files = []
    raw_text_parts = []
    stage_save_start = time.perf_counter()
    for idx, item in enumerate(data_items):
        if not isinstance(item, dict):
            continue
        revised_prompt = item.get("revised_prompt")
        if revised_prompt:
            raw_text_parts.append(str(revised_prompt))

        b64_data = item.get("b64_json")
        if b64_data:
            try:
                image_bytes = base64.b64decode(b64_data)
                prefix = f"{file_prefix}_" if file_prefix else ""
                file_name = f"{prefix}output_{datetime.now().strftime('%H%M%S')}_{idx}_{uuid.uuid4().hex[:6]}.png"
                file_path = os.path.join(save_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(image_bytes)
                logger.info(f"✅ 成功保存图片(.png): {file_path}")
                saved_files.append(file_path)
                continue
            except Exception as e:
                logger.error(f"写入 Base64 图片失败: {e}")

        img_url = item.get("url")
        if img_url:
            if no_download:
                logger.info(f"openrouter-image 返回图片URL(跳过下载): {img_url}")
                raw_text_parts.append(str(img_url))
                continue
            try:
                img_data = _download_image_with_referer_retry(
                    img_url=str(img_url),
                    timeout_s=min(timeout_val, 60),
                    referer=download_referer
                )
                ext = ".png"
                if img_data.startswith(b"\xff\xd8"):
                    ext = ".jpg"
                elif img_data.startswith(b"RIFF") and img_data[8:12] == b"WEBP":
                    ext = ".webp"
                elif img_data.startswith(b"GIF87a") or img_data.startswith(b"GIF89a"):
                    ext = ".gif"
                prefix = f"{file_prefix}_" if file_prefix else ""
                file_name = f"{prefix}output_{datetime.now().strftime('%H%M%S')}_{idx}_{uuid.uuid4().hex[:6]}{ext}"
                file_path = os.path.join(save_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(img_data)
                logger.info(f"✅ 成功保存图片 ({ext} 格式): {file_path}")
                saved_files.append(file_path)
            except Exception as e:
                logger.error(f"下载图片失败 {img_url}: {e}")
    _log_stage_elapsed("阶段2-提取并保存图片", stage_save_start)
    logger.info(f"[阶段统计] data节点: {len(data_items)}，成功保存: {len(saved_files)}")
    if len(saved_files) == 0:
        _save_server_response_json(save_dir, file_prefix, "openrouter-image", resp_json)
        _save_server_response_raw(save_dir, file_prefix, "openrouter-image", _response_text_utf8(resp))

    raw_text = "\n".join(raw_text_parts).strip()
    annotation_data = _normalize_annotation_result(_extract_json_object(raw_text))
    if return_metadata:
        return {
            "saved_files": saved_files,
            "annotation": annotation_data,
            "raw_text": raw_text
        }
    return saved_files

def generate_image_aigc2d_gpt(prompt: str, image_paths: list = None, model: str = "gpt-image-2", aspect_ratio: str = "1:1", instructions: str = "", resolution: str = None, api_type: str = None, save_sub_dir: str = None, file_prefix: str = None, return_metadata: bool = False) -> list:
    """
    AIGC-2D-GPT 生图接口（对齐 useless/app-func.py）：
    - base_url 作为完整请求地址（默认 https://next.aigc2d.com/v1/images/generations）
    - 请求体核心字段：model / prompt / n / size
    - 返回 data[].b64_json 时直接解码保存图片
    - 至少保存一份返回 JSON（无法解析时保存 raw_text 包装 JSON）
    """
    config = get_api_config(api_type=api_type or "aigc-2d-gpt")
    api_base = str(config.get("base_url", "https://next.aigc2d.com/v1/images/generations") or "https://next.aigc2d.com/v1/images/generations").strip().rstrip("/")
    if api_base.endswith("/images/generations") or api_base.endswith("/v1/images/generations"):
        url = api_base
    elif api_base.endswith("/v1"):
        url = f"{api_base}/images/generations"
    else:
        url = f"{api_base}/v1/images/generations"
    api_key = config.get("api_key")
    timeout_val = int(config.get("timeout", 120) or 120)
    max_retries = int(config.get("max_retries", 1) or 1)
    retry_backoff_s = float(config.get("retry_backoff", 1.0) or 1.0)
    debug_dump_full_http = _as_bool(config.get("debug_dump_full_http", False), False)
    n = int(config.get("n", 1) or 1)
    if n <= 0:
        n = 1
    size = str(config.get("size", "") or "").strip()
    if not size:
        size = _aspect_ratio_to_dalle3_size(aspect_ratio, allow_auto_for_extreme=True)

    if not api_key:
        logger.error("配置文件 config-image.json 中缺少 'api_key' 参数。")
        return []

    final_prompt = f"{instructions}\n\n{prompt}".strip() if instructions else str(prompt or "")
    payload = {
        "model": str(model or config.get("model") or "gpt-image-2").strip() or "gpt-image-2",
        "prompt": final_prompt,
        "n": n,
        "size": size
    }
    seed_val = config.get("seed", None)
    if seed_val is not None and str(seed_val).strip() != "":
        try:
            payload["seed"] = int(seed_val)
        except Exception:
            payload["seed"] = seed_val

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    valid_image_paths = _existing_image_paths(image_paths)
    use_edits_mode = len(valid_image_paths) > 0
    request_url = _derive_edits_url_from_generations_url(url) if use_edits_mode else url
    if use_edits_mode:
        logger.info(f"AIGC-2D-GPT 将使用 /images/edits 模式，附件数量: {len(valid_image_paths)}")
    else:
        logger.info("AIGC-2D-GPT 使用 /images/generations 模式（无附件）。")

    logger.info("=== 发起 AIGC-2D-GPT API 请求（app-func 逻辑）===")
    logger.info(f"请求 URL: {request_url}")
    logger.info(f"请求数据:\n{_format_safe_log(payload)}")

    stage_json_start = time.perf_counter()
    resp = None
    last_exc = None
    request_trace_body = payload
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"正在进行第 {attempt} 次重试 (最大重试次数: {max_retries})...")
            if use_edits_mode:
                resp, used_files = _post_images_edits_request(
                    url=request_url,
                    api_key=api_key,
                    form_payload=payload,
                    image_paths=valid_image_paths,
                    timeout_val=timeout_val
                )
                request_trace_body = {"form": payload, "files": used_files}
            else:
                resp = requests.post(request_url, headers=headers, json=payload, timeout=timeout_val)
                request_trace_body = payload
            break
        except requests.exceptions.RequestException as e:
            last_exc = e
            logger.warning(f"AIGC-2D-GPT 网络请求异常 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                time.sleep(retry_backoff_s * (attempt + 1))

    today_str = datetime.now().strftime("%Y%m%d")
    save_dir = os.path.join("data", today_str, save_sub_dir) if save_sub_dir else os.path.join("data", today_str)
    os.makedirs(save_dir, exist_ok=True)

    if resp is None:
        logger.error(f"AIGC-2D-GPT 请求失败，未获得响应对象: {last_exc}")
        if debug_dump_full_http:
            _save_debug_http_trace(
                save_dir=save_dir,
                file_prefix=file_prefix,
                api_tag="aigc-2d-gpt",
                request_url=request_url,
                request_headers={"Authorization": f"Bearer {api_key}"},
                request_body=request_trace_body,
                response=None,
                response_body=str(last_exc or ""),
                note="request_failed_no_response"
            )
        _save_server_response_json(
            save_dir,
            file_prefix,
            "aigc-2d-gpt",
            {"error": str(last_exc or "request failed"), "request_url": url}
        )
        return []

    resp_text = _response_text_utf8(resp)
    if debug_dump_full_http:
        _save_debug_http_trace(
            save_dir=save_dir,
            file_prefix=file_prefix,
            api_tag="aigc-2d-gpt",
            request_url=request_url,
            request_headers={"Authorization": f"Bearer {api_key}"},
            request_body=request_trace_body,
            response=resp,
            response_body=resp_text,
            note=f"http_status={getattr(resp, 'status_code', None)}"
        )

    try:
        if getattr(resp, "encoding", None) is None:
            resp.encoding = "utf-8"
        resp_json = resp.json()
    except Exception:
        resp_json = {"raw_text": resp_text, "status_code": getattr(resp, "status_code", None)}
    _save_server_response_json(save_dir, file_prefix, "aigc-2d-gpt", resp_json)
    logger.info(f"=== AIGC-2D-GPT 服务器原始返回信息 ===\n{_format_safe_log(resp_json)}")
    _log_stage_elapsed("阶段1-获取JSON响应", stage_json_start)

    data_items = resp_json.get("data", []) if isinstance(resp_json, dict) else []
    if not isinstance(data_items, list) or not data_items:
        logger.warning("AIGC-2D-GPT 返回 JSON 中没有 data 节点。")
        _save_server_response_raw(save_dir, file_prefix, "aigc-2d-gpt", resp_text)
        return []

    saved_files = []
    revised_prompt_parts = []
    stage_save_start = time.perf_counter()
    for idx, item in enumerate(data_items):
        if not isinstance(item, dict):
            continue
        revised_prompt = item.get("revised_prompt")
        if revised_prompt:
            revised_prompt_parts.append(str(revised_prompt))

        b64_data = item.get("b64_json")
        if b64_data:
            try:
                image_bytes = base64.b64decode(b64_data)
                prefix = f"{file_prefix}_" if file_prefix else ""
                file_name = f"{prefix}output_{datetime.now().strftime('%H%M%S')}_{idx}_{uuid.uuid4().hex[:6]}.png"
                file_path = os.path.join(save_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(image_bytes)
                logger.info(f"✅ 成功保存 AIGC-2D-GPT 图片(.png): {file_path}")
                saved_files.append(file_path)
                continue
            except Exception as e:
                logger.error(f"AIGC-2D-GPT 写入 Base64 图片失败: {e}")

        img_url = item.get("url")
        if img_url:
            try:
                img_resp = requests.get(str(img_url), timeout=min(timeout_val, 60))
                img_resp.raise_for_status()
                img_data = img_resp.content
                ext = ".png"
                if img_data.startswith(b"\xff\xd8"):
                    ext = ".jpg"
                elif img_data.startswith(b"RIFF") and img_data[8:12] == b"WEBP":
                    ext = ".webp"
                elif img_data.startswith(b"GIF87a") or img_data.startswith(b"GIF89a"):
                    ext = ".gif"
                prefix = f"{file_prefix}_" if file_prefix else ""
                file_name = f"{prefix}output_{datetime.now().strftime('%H%M%S')}_{idx}_{uuid.uuid4().hex[:6]}{ext}"
                file_path = os.path.join(save_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(img_data)
                logger.info(f"✅ 成功下载并保存 AIGC-2D-GPT 图片: {file_path}")
                saved_files.append(file_path)
            except Exception as e:
                logger.error(f"AIGC-2D-GPT 下载图片失败 {img_url}: {e}")
    _log_stage_elapsed("阶段2-提取并保存图片", stage_save_start)
    logger.info(f"[阶段统计] data节点: {len(data_items)}，成功保存: {len(saved_files)}")

    raw_text = "\n".join(revised_prompt_parts).strip()
    annotation_data = _normalize_annotation_result(_extract_json_object(raw_text))
    if return_metadata:
        return {
            "saved_files": saved_files,
            "annotation": annotation_data,
            "raw_text": raw_text
        }
    return saved_files

def generate_image_whatai(prompt: str, image_paths: list = None, model: str = "nano-banana-2", aspect_ratio: str = "1:1", instructions: str = "", resolution = "1K", api_type: str = None, save_sub_dir: str = None, file_prefix: str = None, return_metadata: bool = False) -> list:
    """
    独立出来的图片生成核心逻辑
    """
    normalized_api_type = str(api_type or "").strip().lower()
    if normalized_api_type in {"openai-image", "openai_image"}:
        return generate_image_openai_image(
            prompt=prompt,
            image_paths=image_paths,
            model=model,
            aspect_ratio=aspect_ratio,
            instructions=instructions,
            resolution=resolution,
            api_type=api_type,
            save_sub_dir=save_sub_dir,
            file_prefix=file_prefix,
            return_metadata=return_metadata
        )
    if normalized_api_type in {"aigc-2d-gpt", "aigc2d-gpt", "aigc_2d_gpt"}:
        return generate_image_aigc2d_gpt(
            prompt=prompt,
            image_paths=image_paths,
            model=model,
            aspect_ratio=aspect_ratio,
            instructions=instructions,
            resolution=resolution,
            api_type=api_type,
            save_sub_dir=save_sub_dir,
            file_prefix=file_prefix,
            return_metadata=return_metadata
        )
    if normalized_api_type in {"openrouter-image", "openrouter_image", "openrouter"}:
        return generate_image_openrouter_image(
            prompt=prompt,
            image_paths=image_paths,
            model=model,
            aspect_ratio=aspect_ratio,
            instructions=instructions,
            resolution=resolution,
            api_type=api_type,
            save_sub_dir=save_sub_dir,
            file_prefix=file_prefix,
            return_metadata=return_metadata
        )

    config = get_api_config(api_type=api_type)
    # 兼容原有的 base_url 命名
    api_base = config.get("base_url", "https://api.whatai.cc/v1").rstrip('/')
    api_key = config.get("api_key")
    timeout_val = config.get("timeout", 120)      # <--- 读取超时配置，默认120
    max_retries = config.get("max_retries", 1)    # <--- 读取重试配置，默认1
    debug_dump_full_http = _as_bool(config.get("debug_dump_full_http", False), False)

    # TODO: 处理resolution，但是whatai其实根本不接受这个参数，目前只能放在prompt里让模型自己理解了
    
    if not api_key:
        logger.error("配置文件 config-image.json 中缺少 'api_key' 参数。")
        raise ValueError("配置文件 config-image.json 中缺少 'api_key' 参数。")

    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 将用户保存的 Instructions 和具体的 prompt 拼接传入
    combined_prompt = f"--ar {aspect_ratio} ,  {instructions}  {prompt}"
    content_list = [{"type": "text", "text": combined_prompt}]

    if image_paths:
        for img_path in image_paths:
            if os.path.exists(img_path):
                mime_type = _guess_image_mime_type(img_path)
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{to_base64(img_path)}"}
                })
            else:
                logger.warning(f"找不到本地图片文件 {img_path}，已跳过。")

    data = {
        "aspect_ratio": aspect_ratio,
        "model": model,
        "messages": [{"role": "user", "content": content_list}]
    }

    # 隐藏 Base64 打印日志
    safe_data = copy.deepcopy(data)
    for msg in safe_data.get("messages", []):
        for content in msg.get("content", []):
            if type(content) == dict and content.get("type") == "image_url":
                content["image_url"]["url"] = "<BASE64_IMAGE_DATA_OMITTED>"
    
    logger.info("=== 发起 API 请求 ===")
    logger.info(f"请求数据:\n{json.dumps(safe_data, ensure_ascii=False, indent=2)}")

    # ================= 阶段1：请求并获取 JSON 响应 =================
    stage_json_start = time.perf_counter()
    resp = None
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"正在进行第 {attempt} 次重试 (最大重试次数: {max_retries})...")
            
            resp = requests.post(url, headers=headers, json=data, timeout=timeout_val)
            resp.raise_for_status()
            break  # 如果没有抛出异常，说明请求成功，跳出循环
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"网络请求发生异常 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                time.sleep(2)  # 重试前稍微休息2秒，避免频繁打满后端
            else:
                logger.error("达到最大重试次数，图片生成请求最终失败。")
                fail_today = datetime.now().strftime("%Y%m%d")
                fail_dir = os.path.join("data", fail_today, save_sub_dir) if save_sub_dir else os.path.join("data", fail_today)
                if debug_dump_full_http:
                    _save_debug_http_trace(
                        save_dir=fail_dir,
                        file_prefix=file_prefix,
                        api_tag="whatai",
                        request_url=url,
                        request_headers=headers,
                        request_body=data,
                        response=resp,
                        note=f"request_exception_retry_exhausted: {e}"
                    )
                if resp is not None:
                    _save_server_response_raw(fail_dir, file_prefix, "whatai", _response_text_utf8(resp))
                return []
    # ====================================================================

    try:
        if getattr(resp, "encoding", None) is None:
            resp.encoding = "utf-8"
        resp_json = resp.json()
        
        # 【新增】打印服务器返回的原始完整 JSON 信息
        logger.info(f"=== 服务器原始返回信息 ===\n{_format_safe_log(resp_json)}")
        
        content_str = resp_json["choices"][0]["message"]["content"]
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"解析返回 JSON 失败: {e}")
        fail_today = datetime.now().strftime("%Y%m%d")
        fail_dir = os.path.join("data", fail_today, save_sub_dir) if save_sub_dir else os.path.join("data", fail_today)
        if debug_dump_full_http:
            _save_debug_http_trace(
                save_dir=fail_dir,
                file_prefix=file_prefix,
                api_tag="whatai",
                request_url=url,
                request_headers=headers,
                request_body=data,
                response=resp,
                note=f"json_parse_error: {e}"
            )
        _save_server_response_raw(fail_dir, file_prefix, "whatai", _response_text_utf8(resp))
        return []
    _log_stage_elapsed("阶段1-获取JSON响应", stage_json_start)

    today_str = datetime.now().strftime("%Y%m%d")
    if save_sub_dir:
        save_dir = os.path.join("data", today_str, save_sub_dir)
    else:
        save_dir = os.path.join("data", today_str)
    os.makedirs(save_dir, exist_ok=True)
    if debug_dump_full_http:
        _save_debug_http_trace(
            save_dir=save_dir,
            file_prefix=file_prefix,
            api_tag="whatai",
            request_url=url,
            request_headers=headers,
            request_body=data,
            response=resp,
            note="success_response_received"
        )
        _save_server_response_json(save_dir, file_prefix, "whatai", resp_json)

    # 解析 Markdown 提取图片
    img_urls = re.findall(r'!\[.*?\]\((https?://[^\)]+)\)', content_str)
    annotation_data = _normalize_annotation_result(_extract_json_object(content_str))
    if not img_urls:
        logger.warning("未在返回的文本中找到图片链接。")
        _save_server_response_json(save_dir, file_prefix, "whatai", resp_json)
        _save_server_response_raw(save_dir, file_prefix, "whatai", content_str)
        return []
    saved_files = []

    # ================= 阶段2：下载并保存图片 =================
    stage_download_start = time.perf_counter()
    for idx, img_url in enumerate(img_urls):
        try:
            # 【优化】获取完整的 response 以读取内容
            img_resp = requests.get(img_url, timeout=30)
            img_resp.raise_for_status()
            img_data = img_resp.content
            
            # 【新增】通过二进制文件头（Magic Bytes）准确识别图片格式
            ext = ".png"  # 默认兜底后缀
            if img_data.startswith(b'\xff\xd8'):
                ext = ".jpg"
            elif img_data.startswith(b'\x89PNG\r\n\x1a\n'):
                ext = ".png"
            elif img_data.startswith(b'RIFF') and img_data[8:12] == b'WEBP':
                ext = ".webp"
            elif img_data.startswith(b'GIF87a') or img_data.startswith(b'GIF89a'):
                ext = ".gif"
            
            # 使用识别出的正确后缀保存文件
            prefix = f"{file_prefix}_" if file_prefix else ""
            file_name = f"{prefix}output_{datetime.now().strftime('%H%M%S')}_{idx}_{uuid.uuid4().hex[:6]}{ext}"
            file_path = os.path.join(save_dir, file_name)
            
            with open(file_path, "wb") as f:
                f.write(img_data)
                
            logger.info(f"✅ 成功保存图片 ({ext} 格式): {file_path}")
            saved_files.append(file_path)
            
        except Exception as e:
            logger.error(f"下载图片失败 {img_url}: {e}")
    _log_stage_elapsed("阶段2-下载并保存图片", stage_download_start)
    logger.info(f"[阶段统计] 识别到图片链接: {len(img_urls)}，成功保存: {len(saved_files)}")
    if len(saved_files) == 0:
        _save_server_response_json(save_dir, file_prefix, "whatai", resp_json)
        _save_server_response_raw(save_dir, file_prefix, "whatai", content_str)

    if return_metadata:
        return {
            "saved_files": saved_files,
            "annotation": annotation_data,
            "raw_text": content_str
        }
    return saved_files

def fetch_llm_json(base_url: str, api_key: str, model: str, system_prompt: str, user_content: str, temperature: float = 0.5, merge_system_prompt: bool = True) -> str:
    """
    通用 LLM 对话请求函数，专门用于获取 JSON 格式文本，并记录完整请求和响应日志
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    if merge_system_prompt:
        messages = [
            {"role": "user", "content": f"{system_prompt}\n\n{user_content}"}
        ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        # 如果你使用的大模型 API 兼容 OpenAI，开启此选项能极大提高返回纯 JSON 的概率
        "response_format": { "type": "json_object" } 
    }

    logger.info("=== 发起 LLM 提示词请求 ===")
    logger.info(f"请求 URL: {url}")
    # 打印完整的 payload 到日志中
    logger.info(f"请求 Payload:\n{_format_safe_log(payload)}")

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        resp_json = resp.json()
        
        # 将服务器返回的原始完整 JSON 记录到日志
        logger.info(f"=== 服务器原始返回完整信息 ===\n{_format_safe_log(resp_json)}")
        
        # 提取模型回复的文本
        return resp_json['choices'][0]['message']['content'].strip()
        
    except requests.exceptions.Timeout:
        logger.error("LLM 请求超时。")
        return ""
    except Exception as e:
        logger.error(f"LLM 请求发生异常: {str(e)}")
        if 'resp' in locals():
            logger.error(f"服务器返回信息: {_format_safe_log(resp.text)}")
        return ""

def fetch_cohere_json(system_prompt: str, user_content: str, temperature: float = 0.5) -> str:
    """
    专门用于读取 config-cohere.json 并请求 Cohere API 的函数
    """
    config_path = "config-cohere.json"
    if not os.path.exists(config_path):
        logger.error(f"未找到 Cohere 配置文件: {config_path}")
        return ""
        
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"读取 {config_path} 失败: {e}")
        return ""
        
    base_url = config.get("base_url", "https://api.cohere.com/v1").rstrip('/')
    # 智能补全 URL 路径
    if not base_url.endswith("/chat") and not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    url = f"{base_url}/chat" if not base_url.endswith("/chat") else base_url
    
    api_key = config.get("api_key", "")
    model = config.get("model", "command-r-plus")
    merge_system_prompt = config.get("merge_system_prompt", False)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    # 按照 Cohere API 的要求构造 message 和 preamble (系统提示词)
    if merge_system_prompt:
        message = f"{system_prompt}\n\n{user_content}"
        preamble = ""
    else:
        message = user_content
        preamble = system_prompt
    
    payload = {
        "model": model,
        "message": message,
        "temperature": temperature,
        "response_format": { "type": "json_object" } # 强制要求 Cohere 输出 JSON
    }
    
    if preamble:
        payload["preamble"] = preamble

    logger.info("=== 发起 Cohere LLM 提示词请求 ===")
    logger.info(f"请求 headers: {headers}")
    logger.info(f"请求 URL: {url}")
    logger.info(f"请求 Payload:\n{_format_safe_log(payload)}")

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        resp_json = resp.json()
        
        logger.info(f"=== Cohere 服务器原始返回完整信息 ===\n{_format_safe_log(resp_json)}")
        
        # Cohere V1 Endpoint 返回的文本内容在 'text' 字段中
        return resp_json.get('text', '').strip()
        
    except requests.exceptions.Timeout:
        logger.error("Cohere 请求超时。")
        return ""
    except Exception as e:
        logger.error(f"Cohere 请求发生异常: {str(e)}")
        if 'resp' in locals():
            logger.error(f"服务器返回信息: {_format_safe_log(resp.text)}")
        return ""

def generate_image_aigc2d(prompt: str, image_paths: list = None, model: str = "gemini-3.1-flash-image-preview", aspect_ratio: str = "1:1", instructions: str = "", resolution: str = None, api_type: str = None, save_sub_dir: str = None, file_prefix: str = None, return_metadata: bool = False) -> list:
    """
    AIGC2D 专用的图片生成核心逻辑
    入参跟 generate_image_whatai 保持完全一致
    """
    # 从统一配置文件中加载 aigc2d 配置
    config = get_api_config(api_type="aigc2d" if not api_type else api_type)

    api_base = config.get("base_url", "https://next.aigc2d.com/v1beta").rstrip('/')
    api_key = config.get("api_key")
    timeout_val = config.get("timeout", 180)
    max_retries = config.get("max_retries", 1)
    debug_dump_full_http = _as_bool(config.get("debug_dump_full_http", False), False)
    # 从配置中读取resolution，如果传入了参数则使用传入的参数
    if resolution is None:
        resolution = config.get("resolution", "1K")

    if not api_key:
        logger.error("配置文件 config-image.json 中缺少 'api_key' 参数。")
        return []

    # AIGC2D 接口 URL 拼接规则
    url = f"{api_base}/models/{model}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json"
    }

    # 组合提示词
    combined_prompt = f"{instructions} \n {prompt}".strip() if instructions else prompt
    parts = [{"text": combined_prompt}]

    # 处理传入的参考图片（支持多图，按照入参列表追加）
    if image_paths:
        for img_path in image_paths:
            if os.path.exists(img_path):
                mime_type, _ = mimetypes.guess_type(img_path)
                if not mime_type:
                    mime_type = "image/jpeg"
                parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": to_base64(img_path)
                    }
                })
            else:
                logger.warning(f"找不到本地图片文件 {img_path}，已跳过。")

    # 构造请求 Payload
    payload = {
        "contents": [
            {
                "parts": parts
            }
        ],
        "generationConfig": {
            "imageConfig": {
                "aspectRatio": aspect_ratio,
                "imageSize": resolution
            }
        }
    }

    # 日志脱敏：隐藏 Base64 字符串
    safe_payload = copy.deepcopy(payload)
    for part in safe_payload.get("contents", [])[0].get("parts", []):
        if "inline_data" in part and "data" in part["inline_data"]:
            part["inline_data"]["data"] = "<BASE64_IMAGE_DATA_OMITTED>"
            
    logger.info("=== 发起 AIGC2D API 请求 ===")
    logger.info(f"请求数据:\n{json.dumps(safe_payload, ensure_ascii=False, indent=2)}")

    # ================= 阶段1：请求并获取 JSON 响应 =================
    stage_json_start = time.perf_counter()
    resp = None
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"正在进行第 {attempt} 次重试 (最大重试次数: {max_retries})...")
            
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout_val)
            resp.raise_for_status()
            break  
        except requests.exceptions.RequestException as e:
            logger.warning(f"网络请求发生异常 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                time.sleep(2)
            else:
                logger.error("达到最大重试次数，AIGC2D 图片生成请求最终失败。")
                fail_today = datetime.now().strftime("%Y%m%d")
                fail_dir = os.path.join("data", fail_today, save_sub_dir) if save_sub_dir else os.path.join("data", fail_today)
                if debug_dump_full_http:
                    _save_debug_http_trace(
                        save_dir=fail_dir,
                        file_prefix=file_prefix,
                        api_tag="aigc2d",
                        request_url=url,
                        request_headers=headers,
                        request_body=payload,
                        response=resp,
                        note=f"request_exception_retry_exhausted: {e}"
                    )
                if resp is not None:
                    raw_text = _response_text_utf8(resp)
                    logger.error(f"最后一次响应内容: {_format_safe_log(raw_text)}")
                    _save_server_response_raw(fail_dir, file_prefix, "aigc2d", raw_text)
                return []

    # 解析返回 JSON
    try:
        if getattr(resp, "encoding", None) is None:
            resp.encoding = "utf-8"
        resp_json = resp.json()
        # 记录脱敏后的原始返回（避免返回巨量 base64 撑爆日志）
        safe_resp_json = copy.deepcopy(resp_json)
        for cand in safe_resp_json.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                if "inlineData" in part:
                    part["inlineData"]["data"] = "<BASE64_IMAGE_DATA_OMITTED>"
                elif "inline_data" in part:
                    part["inline_data"]["data"] = "<BASE64_IMAGE_DATA_OMITTED>"
                    
        logger.info(f"=== AIGC2D 服务器返回信息 ===\n{json.dumps(safe_resp_json, ensure_ascii=False, indent=2)}")
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"解析 AIGC2D 返回 JSON 失败: {e}")
        fail_today = datetime.now().strftime("%Y%m%d")
        fail_dir = os.path.join("data", fail_today, save_sub_dir) if save_sub_dir else os.path.join("data", fail_today)
        if debug_dump_full_http:
            _save_debug_http_trace(
                save_dir=fail_dir,
                file_prefix=file_prefix,
                api_tag="aigc2d",
                request_url=url,
                request_headers=headers,
                request_body=payload,
                response=resp,
                note=f"json_parse_error: {e}"
            )
        _save_server_response_raw(fail_dir, file_prefix, "aigc2d", _response_text_utf8(resp))
        return []
    _log_stage_elapsed("阶段1-获取JSON响应", stage_json_start)

    # 提取图片并保存
    today_str = datetime.now().strftime("%Y%m%d")
    if save_sub_dir:
        save_dir = os.path.join("data", today_str, save_sub_dir)
    else:
        save_dir = os.path.join("data", today_str)
    os.makedirs(save_dir, exist_ok=True)
    if debug_dump_full_http:
        _save_debug_http_trace(
            save_dir=save_dir,
            file_prefix=file_prefix,
            api_tag="aigc2d",
            request_url=url,
            request_headers=headers,
            request_body=payload,
            response=resp,
            note="success_response_received"
        )
        _save_server_response_json(save_dir, file_prefix, "aigc2d", resp_json)
    saved_files = []

    candidates = resp_json.get("candidates", [])
    if not candidates:
        logger.warning("AIGC2D 返回的 JSON 中没有找到 candidates 节点。")
        _save_server_response_json(save_dir, file_prefix, "aigc2d", resp_json)
        _save_server_response_raw(save_dir, file_prefix, "aigc2d", _response_text_utf8(resp))
        return []

    model_text_parts = []
    # ================= 阶段2：提取并保存图片 =================
    stage_save_start = time.perf_counter()
    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])

        for part in parts:
            # AIGC2D 的 Gemini API 结构可能是 inlineData 或 inline_data
            inline_data = part.get("inlineData") or part.get("inline_data")
            if inline_data and "data" in inline_data:
                mime_type = inline_data.get("mimeType") or inline_data.get("mime_type") or "image/png"
                ext = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/webp": ".webp",
                }.get(mime_type, ".png")

                # 生成形如：P01_142305-a1b2c3.png 的文件名
                time_str = datetime.now().strftime('%H%M%S')
                random_str = uuid.uuid4().hex[:6]
                prefix = f"{file_prefix}_" if file_prefix else ""
                file_name = f"{prefix}{time_str}-{random_str}{ext}"
                file_path = os.path.join(save_dir, file_name)

                try:
                    image_bytes = base64.b64decode(inline_data["data"])
                    with open(file_path, "wb") as f:
                        f.write(image_bytes)
                    logger.info(f"✅ 成功保存图片 ({ext} 格式): {file_path}")
                    saved_files.append(file_path)
                except Exception as e:
                    logger.error(f"写入图片文件失败: {e}")
                    
            elif "text" in part:
                # 顺手记录一下模型可能返回的额外文本提示
                logger.info(f"模型文本反馈: {part['text']}")
                model_text_parts.append(part["text"])
    _log_stage_elapsed("阶段2-提取并保存图片", stage_save_start)
    logger.info(f"[阶段统计] candidates: {len(candidates)}，成功保存: {len(saved_files)}")
    if len(saved_files) == 0:
        _save_server_response_json(save_dir, file_prefix, "aigc2d", resp_json)
        _save_server_response_raw(save_dir, file_prefix, "aigc2d", _response_text_utf8(resp))

    raw_text = "\n".join(model_text_parts).strip()
    annotation_data = _normalize_annotation_result(_extract_json_object(raw_text))
    if return_metadata:
        return {
            "saved_files": saved_files,
            "annotation": annotation_data,
            "raw_text": raw_text
        }
    return saved_files
