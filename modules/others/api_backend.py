# api_backend.py
from logging import config
import os
import json
import base64
import io
import tempfile
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(BASE_DIR, "conf")
# 图片生成配置已并入统一的 conf/config.json（脚本 key / 文本 API 与图片 API 同处一个文件）；
# LEGACY_CONFIG 仅为兼容老机器上还留着的旧文件，找不到新文件时回退读取。
UNIFIED_CONFIG_PATH = os.path.join(CONFIG_DIR, "config.json")
LEGACY_IMAGE_CONFIG_PATH = os.path.join(CONFIG_DIR, "config-image.json")

# ================= 2. 核心功能函数 =================
NODES_ENV = "IMAGE_MAKER_NODES"
# 逐节点环境变量的后缀（`IMAGE_MAKER_<SLUG>_BASE_URL` 等）
_NODE_ENV_FIELDS = (("_BASE_URL", "base_url"), ("_API_TYPE", "api_type"),
                    ("_ENV_SLUG", "env_slug"), ("_MODEL", "model"),
                    ("_RESOLUTION", "resolution"), ("_TIMEOUT", "timeout"))
# 这两组走顶层配置，不当作图片节点
_NODE_ENV_RESERVED = {"TEXT", "NSFW"}


_NODE_NAME_ALIASES = {"aigc_2d_gpt": "aigc-2d-gpt", "aigc2d_gpt": "aigc-2d-gpt"}


def _canonical_node_name(slug: str, existing=()) -> str:
    """把环境变量推导出来的节点名对齐到配置里真正用的名字（`aigc_2d_gpt` → `aigc-2d-gpt`）。

    否则 `IMAGE_MAKER_AIGC_2D_GPT_BASE_URL` 会生成一个叫 `aigc_2d_gpt` 的新节点，
    而 app / CLI 查的是 `aigc-2d-gpt`，等于没生效。
    """
    norm = lambda value: re.sub(r"[^0-9a-z]+", "", str(value or "").lower())  # noqa: E731
    for name in existing or ():
        if norm(name) == norm(slug):
            return name
    return _NODE_NAME_ALIASES.get(str(slug or "").lower(), str(slug or "").lower())


def nodes_from_env(env=None) -> dict:
    """从环境变量收集「图片 API 节点定义」（`.env` 里就能配节点，不必写进 conf/config.json）。

    - `IMAGE_MAKER_NODES`：JSON，形如 `{"whatup": {"base_url": "...", "model": "...", "env_slug": "aigc2d"}}`；
    - 逐节点：`IMAGE_MAKER_<SLUG>_BASE_URL` / `_MODEL` / `_API_TYPE` / `_ENV_SLUG` / `_RESOLUTION` / `_TIMEOUT`。
    密钥本身仍走 `IMAGE_MAKER_<SLUG>_API_KEY`（或节点里的 `env_slug`）。
    """
    env = os.environ if env is None else env
    nodes: dict = {}
    raw = str(env.get(NODES_ENV) or "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                for node, entry in parsed.items():
                    if isinstance(entry, dict):
                        nodes.setdefault(str(node), {}).update(entry)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{NODES_ENV} 不是合法 JSON，已忽略：{exc}")
    for name, value in env.items():
        if not name.startswith(ENV_KEY_PREFIX) or not value:
            continue
        for suffix, field in _NODE_ENV_FIELDS:
            if not name.endswith(suffix):
                continue
            slug = name[len(ENV_KEY_PREFIX):-len(suffix)]
            if not slug or slug in _NODE_ENV_RESERVED:
                continue
            nodes.setdefault(_canonical_node_name(slug), {}).setdefault(field, str(value).strip())
    for node, entry in nodes.items():
        # 没写 model 时给一个合理的默认，避免下游读空
        if entry.get("base_url") and "model" not in entry:
            entry["model"] = ""
    return nodes


def apply_env_nodes(data: dict) -> dict:
    """把环境变量里的「图片节点定义 + current_api 覆盖」合入一份**已解析**的配置（就地改并返回）。

    `load_config()` 用它；GUI 里已经自己 `json.load` 过配置文件的地方（app.py 的设置页）也要用它 ——
    否则「节点和密钥都只在 `.env` 里」时，界面完全看不到这些节点、
    `IMAGE_MAKER_CURRENT_API` 也不生效：界面会停在配置文件里剩下的那个节点（如 `whatup`），
    自动生图随即报「生图 API Key 不能为空」。只在内存里合，**绝不回写磁盘**。
    """
    if not isinstance(data, dict):
        return data
    current_override = str(os.environ.get("IMAGE_MAKER_CURRENT_API") or "").strip()
    if current_override:
        data["current_api"] = current_override
    env_nodes = nodes_from_env()
    if env_nodes:
        apis = data.setdefault("apis", {})
        # 先把规范化后的节点名对齐到配置文件里已有的名字（避免建出 aigc_2d_gpt 这种查不到的名字）
        for node in list(env_nodes):
            canonical = _canonical_node_name(node, apis.keys())
            if canonical != node:
                env_nodes.setdefault(canonical, {}).update(env_nodes.pop(node))
        for node, entry in env_nodes.items():
            merged = dict(entry)
            merged.update({k: v for k, v in (apis.get(node) or {}).items() if v not in ("", None)})
            apis[node] = merged
    return data


def load_config(config_path=None):
    if config_path is None:
        config_path = UNIFIED_CONFIG_PATH
        if not os.path.exists(config_path) and os.path.exists(LEGACY_IMAGE_CONFIG_PATH):
            logger.warning("未找到 conf/config.json，回退读取已废弃的 conf/config-image.json（建议合并）")
            config_path = LEGACY_IMAGE_CONFIG_PATH
    if not os.path.exists(config_path):
        logger.error(f"未找到配置文件: {config_path}")
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 环境变量里的节点定义：只补缺，不覆盖配置文件里已经写好的值；绝不回写磁盘
    return apply_env_nodes(data)


# 密钥可用环境变量覆盖，避免明文躺在配置文件里（配置文件里留空即可）：
#   IMAGE_MAKER_<节点名>_API_KEY   例：IMAGE_MAKER_AIGC2D_API_KEY / IMAGE_MAKER_AIGC_2D_GPT_API_KEY
#   <节点名>_API_KEY  /  <节点名>_KEY   例：AIGC2D_API_KEY（节点名里的 - / . 会换成 _）
# 优先级：IMAGE_MAKER_* > 通用名 > 配置文件里的值。
ENV_KEY_PREFIX = "IMAGE_MAKER_"

# .env（仓库根目录）兜底装载：父进程环境可能是在 setx 之前启动的旧环境块，
# 靠 .env 才能做到"不管谁怎么拉起进程，密钥都拿得到"。放在模块导入时执行一次。
try:
    from utils.env_loader import ensure_env_loaded as _ensure_env_loaded

    _ensure_env_loaded()
except Exception as _env_exc:  # noqa: BLE001 - 环境装载失败不该影响出图功能
    logger.warning("装载 .env 失败（已忽略）: %s", _env_exc)


def _env_key_names(api_type: str, cfg: dict = None):
    """该节点可用的环境变量名（按优先级）。

    节点名会先规范化（`-` / `.` → `_`）；若配置里给了 `env_slug`，额外把那个名字排在前面，
    这样同一家服务的多个节点（如 `aigc2d` 与 `aigc-2d-gpt`）能共用一把环境变量。
    """
    names = []
    slug = re.sub(r"[^0-9A-Za-z]+", "_", str((cfg or {}).get("env_slug") or "")).strip("_").upper()
    if slug:
        names += [f"{ENV_KEY_PREFIX}{slug}_API_KEY", f"{slug}_API_KEY", f"{slug}_KEY"]
    node_slug = re.sub(r"[^0-9A-Za-z]+", "_", str(api_type or "")).strip("_").upper()
    if node_slug:
        names += [f"{ENV_KEY_PREFIX}{node_slug}_API_KEY", f"{node_slug}_API_KEY", f"{node_slug}_KEY"]
    return tuple(dict.fromkeys(names))


def resolve_api_key(cfg: dict, api_type: str = None) -> str:
    """取该 API 节点的 key：环境变量优先，其次配置文件。返回值只用于请求头，不要打进日志。"""
    for name in _env_key_names(api_type, cfg):
        value = str(os.environ.get(name) or "").strip()
        if value:
            return value
    return str((cfg or {}).get("api_key") or "").strip()


def api_key_source(api_type: str = None, cfg: dict = None) -> str:
    """告知界面 key 从哪来（env:VAR / config / none），只暴露变量名不暴露值。"""
    for name in _env_key_names(api_type, cfg):
        if str(os.environ.get(name) or "").strip():
            return f"env:{name}"
    if str((cfg or {}).get("api_key") or "").strip():
        return "config"
    return "none"


# 顶层「文本分析 API」与「NSFW 文本 API」的 key 同样支持环境变量覆盖。
# 这两个 key 以前只写在 conf/config.json 顶层（base_url / api_key / model），换成环境变量时
# 最容易踩坑：改了 .env、图片节点生效了，文本分析仍拿旧 key 报 401。这里统一成和
# apis.* 节点一样的优先级：真实环境变量 > .env > 配置文件。
TEXT_API_KEY_ENV_NAMES = ("IMAGE_MAKER_TEXT_API_KEY", "TEXT_API_KEY")
NSFW_API_KEY_ENV_NAMES = ("IMAGE_MAKER_NSFW_API_KEY", "NSFW_API_KEY")


def _first_env_value(names):
    """按顺序取第一个非空环境变量，返回 (值, 变量名)；都没有返回 ("", "")。"""
    for name in names:
        value = str(os.environ.get(name) or "").strip()
        if value:
            return value, name
    return "", ""


def resolve_text_api_key(cfg: dict = None) -> str:
    """顶层文本分析 API 的 key：环境变量优先，其次 `cfg`（conf/config.json 读出的 dict）。"""
    value, _name = _first_env_value(TEXT_API_KEY_ENV_NAMES)
    return value or str((cfg or {}).get("api_key") or "").strip()


def resolve_nsfw_api_key(cfg: dict = None) -> str:
    """顶层 NSFW 文本 API 的 key：环境变量优先，其次 `cfg`。"""
    value, _name = _first_env_value(NSFW_API_KEY_ENV_NAMES)
    return value or str((cfg or {}).get("nsfw_api_key") or "").strip()


def text_api_key_source(cfg: dict = None) -> str:
    """文本 key 从哪来（env:VAR / config / none），只暴露变量名不暴露值。"""
    value, name = _first_env_value(TEXT_API_KEY_ENV_NAMES)
    if value:
        return f"env:{name}"
    return "config" if str((cfg or {}).get("api_key") or "").strip() else "none"


def nsfw_api_key_source(cfg: dict = None) -> str:
    """NSFW key 从哪来（env:VAR / config / none）。"""
    value, name = _first_env_value(NSFW_API_KEY_ENV_NAMES)
    if value:
        return f"env:{name}"
    return "config" if str((cfg or {}).get("nsfw_api_key") or "").strip() else "none"


def apply_secret_env_overrides(cfg: dict) -> dict:
    """把顶层 `api_key` / `nsfw_api_key` 换成环境变量里的值（没配就保持原值）。

    只在**内存里**改这一份 dict，绝不回写磁盘 —— 调用方如果把同一个 dict 存回
    conf/config.json，才不会把密钥从环境变量"漏"进配置文件。
    `_text_api_key_source` / `_nsfw_api_key_source` 记录来源（env:VAR / config / none）。
    """
    if not isinstance(cfg, dict):
        return cfg
    cfg["_text_api_key_source"] = text_api_key_source(cfg)
    cfg["_nsfw_api_key_source"] = nsfw_api_key_source(cfg)
    text_key = resolve_text_api_key(cfg)
    nsfw_key = resolve_nsfw_api_key(cfg)
    if text_key:
        cfg["api_key"] = text_key
    if nsfw_key:
        cfg["nsfw_api_key"] = nsfw_key
    return cfg


def get_api_config(config_path=None, api_type=None):
    """获取指定API类型的配置（返回值里的 api_key 已按环境变量优先解析）。"""
    config = load_config(config_path)
    node = api_type if api_type else config.get("current_api", "whatup")
    cfg = dict(config.get("apis", {}).get(node, {}) or {})
    resolved = resolve_api_key(cfg, node)
    # 即使没解析出 key 也保留 `api_key` 这个键（空串）：调用方 `cfg["api_key"]` 才不会 KeyError，
    # 缺 key 的判断统一用真值检查（`not cfg.get("api_key")`），与 key 是空串还是不存在无关。
    cfg["api_key"] = resolved
    cfg["_api_key_source"] = api_key_source(node, cfg)
    return cfg


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

def _headers_for_log(headers: dict) -> str:
    safe = {}
    for k, v in (headers or {}).items():
        kl = str(k).lower()
        if kl in ("authorization", "x-goog-api-key", "api-key", "token"):
            v_str = str(v)
            if len(v_str) > 12:
                safe[k] = v_str[:8] + "..." + v_str[-4:]
            else:
                safe[k] = v_str[:4] + "***" if len(v_str) > 4 else "***"
        else:
            safe[k] = v
    try:
        return json.dumps(safe, ensure_ascii=False)
    except Exception:
        return str(safe)

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
        logger.warning(f"服务器返回 JSON 已保存到: {output_path}")
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

REPLAY_SCRIPT_TEMPLATE = r'''#!/usr/bin/env python3
"""API 请求回放脚本 — 将本文件和同名 .json 数据文件放在同一目录，运行即可复现请求。"""
import json, os, sys, requests

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "{data_filename}")

def main():
    if not os.path.exists(DATA_FILE):
        print(f"错误：找不到数据文件 {{DATA_FILE}}", file=sys.stderr)
        sys.exit(1)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    url = data["url"]
    headers = data["headers"]
    body = data.get("body")

    print(f"=== 请求 URL ===")
    print(url)
    print(f"\n=== 请求 Headers ===")
    print(json.dumps(headers, ensure_ascii=False, indent=2))
    if body:
        print(f"\n=== 请求 Body ===")
        print(json.dumps(body, ensure_ascii=False, indent=2))
    print("\n" + "=" * 60 + "\n正在发送请求...\n")

    resp = requests.post(url, headers=headers, json=body, timeout=120)
    print(f"响应状态码: {{resp.status_code}}")
    print(f"响应 Headers: {{json.dumps(dict(resp.headers), ensure_ascii=False)}}")
    print(f"\n响应 Body:\n{{resp.text[:8000]}}")
    if len(resp.text) > 8000:
        print(f"\n... (共 {{len(resp.text)}} 字符，已截断)")

if __name__ == "__main__":
    main()
'''

def _generate_request_replay(save_dir: str, file_prefix: str, api_tag: str,
                              url: str, headers: dict, body) -> dict:
    """生成请求回放文件：一份 JSON 数据 + 一份独立 .py 脚本。返回 {"json_path": ..., "script_path": ...}"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        prefix = f"{file_prefix}_" if file_prefix else ""
        base_name = f"{prefix}{api_tag}_replay_{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:6]}"

        safe_body = body
        if isinstance(safe_body, dict):
            safe_body = _sanitize_log_data(safe_body)

        request_data = {
            "url": str(url or ""),
            "headers": dict(headers or {}),
            "body": safe_body,
            "api_tag": api_tag,
            "generated_at": datetime.now().isoformat()
        }

        json_path = os.path.join(save_dir, f"{base_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(request_data, f, ensure_ascii=False, indent=2)

        script_path = os.path.join(save_dir, f"{base_name}.py")
        script_content = REPLAY_SCRIPT_TEMPLATE.replace("{data_filename}", f"{base_name}.json")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        logger.info(f"📋 请求回放文件已生成:\n  JSON: {json_path}\n  脚本: {script_path}")
        return {"replay_json": json_path, "replay_script": script_path}
    except Exception as e:
        logger.error(f"生成请求回放文件失败: {e}")
        return {}

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

# ================= gpt-image-2（aigc2d 通道）专用约束 =================
# aigc2d 的 gpt-image-2 只接受 3 档尺寸：方图 / 横图 / 竖图，并且不接受 auto。
# 其余尺寸（含 2K/4K 预设与自定义 WxH）在该通道上会被上游拒绝，这里一律收敛到 3 档。
GPT_IMAGE2_SIZE_SQUARE = "1024x1024"
GPT_IMAGE2_SIZE_LANDSCAPE = "1536x1024"
GPT_IMAGE2_SIZE_PORTRAIT = "1024x1536"
GPT_IMAGE2_SIZES = (GPT_IMAGE2_SIZE_SQUARE, GPT_IMAGE2_SIZE_LANDSCAPE, GPT_IMAGE2_SIZE_PORTRAIT)
GPT_IMAGE2_QUALITIES = ("auto", "low", "medium", "high")
GPT_IMAGE2_OUTPUT_FORMATS = ("png", "jpeg", "webp")
GPT_IMAGE2_MAX_REFERENCE_IMAGES = 16
# gpt-image-2 不支持透明背景，也不接受 input_fidelity（编辑场景自动高保真），这些字段不要外传。
GPT_IMAGE2_UNSUPPORTED_FIELDS = ("background", "input_fidelity")

# ---- gpt-image-2 两个站点（GUI Tab 与 tools/gpt_image2_gen.py 共用同一份映射）----
GPT_IMAGE2_SITE_AIGC2D = "new.aigc2d"
GPT_IMAGE2_SITE_AUTODL = "autodl"
GPT_IMAGE2_SITE_API_TYPES = {
    GPT_IMAGE2_SITE_AIGC2D: "aigc-2d-gpt",
    GPT_IMAGE2_SITE_AUTODL: "autodl",
}
GPT_IMAGE2_SITE_SAVE_SUB_DIRS = {
    GPT_IMAGE2_SITE_AIGC2D: "gpt-image-2",
    GPT_IMAGE2_SITE_AUTODL: "gpt-image-2-autodl",
}
GPT_IMAGE2_SITE_FILE_PREFIXES = {
    GPT_IMAGE2_SITE_AIGC2D: "gptimage2",
    GPT_IMAGE2_SITE_AUTODL: "gptimage2-autodl",
}
# autodl 站点沿用其历史尺寸清单（含 auto / 1792x1024）；aigc2d 站点只有 3 档
GPT_IMAGE2_SITE_SIZES = {
    GPT_IMAGE2_SITE_AIGC2D: GPT_IMAGE2_SIZES,
    GPT_IMAGE2_SITE_AUTODL: ("auto", "1024x1024", "1536x1024", "1792x1024", "1024x1536"),
}
GPT_IMAGE2_SITE_DEFAULT_SIZES = {
    GPT_IMAGE2_SITE_AIGC2D: GPT_IMAGE2_SIZE_PORTRAIT,
    GPT_IMAGE2_SITE_AUTODL: GPT_IMAGE2_SIZE_PORTRAIT,
}

# ---- gpt-image 系列可选模型（GUI 模型下拉 / CLI 共用一份，避免两边漂移）----
# 2026-09 实测 `GET {base}/v1/models` 的真实结果：aigc2d 站点有整族（含 2.5 的 flare/sunburst
# 与它们的 `-c` 按次计费版），autodl 站点只有 gpt-image-2。清单只是常用项，下拉框仍可手输。
GPT_IMAGE2_MODEL_DEFAULT = "gpt-image-2"
GPT_IMAGE2_MODEL_PREFIXES = ("gpt-image", "dall-e", "dalle")
GPT_IMAGE2_SITE_MODELS = {
    GPT_IMAGE2_SITE_AIGC2D: (
        "gpt-image-2",
        "gpt-image-2-c",
        "gpt-image-2.5-flare",
        "gpt-image-2.5-flare-c",
        "gpt-image-2.5-sunburst",
        "gpt-image-2.5-sunburst-c",
        "gpt-image-1.5",
        "gpt-image-1",
        "gpt-image-1-mini",
        "dall-e-3",
    ),
    GPT_IMAGE2_SITE_AUTODL: (
        "gpt-image-2",
    ),
}
# 下拉项的悬浮说明（键为模型名，缺失则只显示模型名）
GPT_IMAGE2_MODEL_NOTES = {
    "gpt-image-2": "基线模型，稳定；尺寸仅 3 档",
    "gpt-image-2-c": "与 gpt-image-2 同款，按次计费通道",
    "gpt-image-2.5-flare": "ChatGPT Images 2.5，偏快（垫图 1024x1536 medium 实测约 30s）",
    "gpt-image-2.5-flare-c": "2.5 flare 的按次计费通道",
    "gpt-image-2.5-sunburst": "ChatGPT Images 2.5，偏精细（同参数实测约 180s）",
    "gpt-image-2.5-sunburst-c": "2.5 sunburst 的按次计费通道",
    "gpt-image-1.5": "上一代 gpt-image",
    "gpt-image-1": "上一代 gpt-image",
    "gpt-image-1-mini": "上一代小模型，便宜快速",
    "dall-e-3": "DALL·E 3（字段与其他 gpt-image 略有差异）",
}


def resolve_models_endpoint(api_base: str, api_type: str = None) -> str:
    """由 base_url 推出 OpenAI 兼容的 `GET /v1/models` 地址（与 images 端点同源推导）。"""
    generations_url = resolve_images_endpoint(api_base, has_images=False, api_type=api_type)
    suffix = "/images/generations"
    if not generations_url.endswith(suffix):
        return f"{generations_url.rstrip('/')}/models"
    return generations_url[: -len(suffix)] + "/models"


def is_gpt_image_model(model_name: str) -> bool:
    """站点 /models 里夹着大量文本/视频模型，只认 gpt-image / dall-e 系列。"""
    return str(model_name or "").strip().lower().startswith(GPT_IMAGE2_MODEL_PREFIXES)


def pick_gpt_image_models(site: str = None, models=None, extra=()) -> list:
    """拼出模型下拉的可选项：站点常用清单 → 额外指定（如配置里已保存的模型）→ 站点实时返回的系列模型。

    `extra` 不做过滤（用户可能手输了自定义模型名）；`models`（站点实时清单）只保留 gpt-image / dall-e 系列。
    """
    ordered: list[str] = []
    for candidate in list(GPT_IMAGE2_SITE_MODELS.get(site, ())) + list(extra or ()):
        text = str(candidate or "").strip()
        if text and text not in ordered:
            ordered.append(text)
    for candidate in (models or []):
        text = str(candidate or "").strip()
        if text and is_gpt_image_model(text) and text not in ordered:
            ordered.append(text)
    return ordered


def list_available_models(api_type: str = None, config_path: str = None, timeout: int = 20) -> list:
    """实时拉取站点模型列表（`GET /v1/models`），返回去重排序后的全部模型 id。

    无 PyQt 依赖，GUI 的「刷新模型列表」与 CLI 都可以直接用；
    调用方一般再用 `pick_gpt_image_models` 过滤出 gpt-image 系列。
    """
    config = get_api_config(config_path=config_path, api_type=api_type)
    api_key = str(config.get("api_key") or "").strip()
    if not api_key:
        raise ValueError(f"apis.{api_type or 'current'} 缺少 api_key，无法获取模型列表")
    url = resolve_models_endpoint(str(config.get("base_url") or ""), api_type=api_type)
    logger.info(f"=== 获取模型列表: GET {url} ===")
    resp = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=int(timeout or 20))
    resp.raise_for_status()
    payload = resp.json()
    rows = payload.get("data") if isinstance(payload, dict) else None
    models = []
    for row in rows or []:
        if isinstance(row, dict) and row.get("id"):
            models.append(str(row["id"]).strip())
        elif isinstance(row, str) and row.strip():
            models.append(row.strip())
    return sorted(set(item for item in models if item))


def _gpt_image2_size_by_ratio(ratio: float) -> str:
    candidates = [
        (GPT_IMAGE2_SIZE_PORTRAIT, 1024 / 1536),
        (GPT_IMAGE2_SIZE_SQUARE, 1.0),
        (GPT_IMAGE2_SIZE_LANDSCAPE, 1536 / 1024),
    ]
    return min(candidates, key=lambda item: abs(item[1] - ratio))[0]

def pick_gpt_image2_size_for_images(image_paths, fallback: str = None, tolerance: float = 0.05) -> str:
    """按**输入图的实际比例**挑 gpt-image-2 的 3 档尺寸（横图→1536x1024，竖图→1024x1536，方图→1024x1024）。

    为什么需要它：gpt-image-2 只有 3 档尺寸，若固定用 1024x1536，输入一张宽图就会输出纵向长图
    （用户 2026-09-23 反馈的问题）。有参考图时按第一张参考图的比例选，没参考图才用 fallback。
    """
    fallback_size = str(fallback or GPT_IMAGE2_SIZE_PORTRAIT)
    for path in (image_paths or []):
        if not path or not os.path.isfile(str(path)):
            continue
        try:
            import cv2
            import numpy as np
            img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            height, width = img.shape[:2]
            if not width or not height:
                continue
            ratio = width / float(height)
            if ratio > 1.0 + tolerance:
                return GPT_IMAGE2_SIZE_LANDSCAPE
            if ratio < 1.0 - tolerance:
                return GPT_IMAGE2_SIZE_PORTRAIT
            return GPT_IMAGE2_SIZE_SQUARE
        except Exception:  # noqa: BLE001 - 读不了就继续尝试下一张
            continue
    return fallback_size


def normalize_gpt_image2_size(size: str = None, aspect_ratio: str = None) -> str:
    """把 size / 长宽比收敛到 gpt-image-2 的 3 档尺寸（永远不返回 auto）。"""
    candidate = str(size or "").strip().lower().replace(" ", "")
    if candidate in GPT_IMAGE2_SIZES:
        return candidate
    if candidate in {"square", "1:1", "方块", "方图"}:
        return GPT_IMAGE2_SIZE_SQUARE
    if candidate in {"landscape", "horizontal", "16:9", "3:2", "横图", "横向"}:
        return GPT_IMAGE2_SIZE_LANDSCAPE
    if candidate in {"portrait", "vertical", "9:16", "2:3", "竖图", "纵向"}:
        return GPT_IMAGE2_SIZE_PORTRAIT

    ratio = None
    if "x" in candidate:
        width_text, _, height_text = candidate.partition("x")
        try:
            width_val, height_val = float(width_text), float(height_text)
            if width_val > 0 and height_val > 0:
                ratio = width_val / height_val
        except ValueError:
            ratio = None
    if ratio is None:
        ratio_text = str(aspect_ratio or "").strip()
        try:
            if ":" in ratio_text:
                width_text, height_text = ratio_text.split(":", 1)
                ratio = float(width_text) / float(height_text)
            elif ratio_text:
                ratio = float(ratio_text)
        except Exception:
            ratio = None
    if ratio is None or ratio <= 0:
        return GPT_IMAGE2_SIZE_SQUARE
    return _gpt_image2_size_by_ratio(ratio)

def build_gpt_image2_payload(prompt: str, model: str = "gpt-image-2", size: str = None, aspect_ratio: str = "1:1", quality: str = None, output_format: str = None, n: int = 1, instructions: str = "", post_instructions: str = "") -> dict:
    """
    组装 gpt-image-2 请求体：/v1/images/generations(JSON) 与 /v1/images/edits(multipart) 共用同一份字段。
    只保留上游真正认识的键（model / prompt / n / size / quality / output_format）。
    """
    prompt_parts = [str(instructions or "").strip(), str(prompt or "").strip(), str(post_instructions or "").strip()]
    try:
        n_val = int(n or 1)
    except (TypeError, ValueError):
        n_val = 1
    payload = {
        "model": str(model or "gpt-image-2").strip() or "gpt-image-2",
        "prompt": "\n\n".join([part for part in prompt_parts if part]),
        "n": max(1, min(10, n_val)),
        "size": normalize_gpt_image2_size(size=size, aspect_ratio=aspect_ratio),
    }
    quality_text = str(quality or "").strip().lower()
    if quality_text in GPT_IMAGE2_QUALITIES and quality_text != "auto":
        payload["quality"] = quality_text
    output_format_text = str(output_format or "").strip().lower()
    if output_format_text in GPT_IMAGE2_OUTPUT_FORMATS:
        payload["output_format"] = output_format_text
    return payload

def gpt_image2_output_extension(output_format: str = None, data: bytes = None) -> str:
    """按 output_format（缺失时按图片魔数）决定落盘后缀。"""
    fmt = str(output_format or "").strip().lower()
    if fmt in {"jpeg", "jpg"}:
        return ".jpg"
    if fmt == "webp":
        return ".webp"
    if fmt == "png":
        return ".png"
    if data:
        if data.startswith(b"\xff\xd8"):
            return ".jpg"
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return ".webp"
        if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
            return ".gif"
    return ".png"

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

def _maybe_compress_image_path(img_path, max_dim=2048, temp_files=None):
    """参考图尺寸超过 max_dim 时压缩为 JPEG 临时文件（与分析流程一致）。

    返回实际使用的路径：图片过大则返回压缩后的临时文件路径，否则返回原路径。
    temp_files 用于收集生成的临时文件，便于调用方在请求完成后清理。
    """
    try:
        from PIL import Image
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        width, height = img.size
        if max(width, height) <= max_dim:
            return img_path
        scaling = max_dim / max(width, height)
        new_size = (int(width * scaling), int(height * scaling))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="ref_compressed_")
        os.close(fd)
        img.save(tmp_path, format="JPEG", quality=95)
        if temp_files is not None:
            temp_files.append(tmp_path)
        logger.info(f"参考图过大已压缩: {img_path} ({width}x{height}) -> {new_size[0]}x{new_size[1]}")
        return tmp_path
    except Exception as e:
        logger.warning(f"参考图压缩失败，使用原图: {img_path} - {e}")
        return img_path

def _cleanup_temp_files(temp_files):
    for path in temp_files or []:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

def to_base64_compressed(path, max_dim=2048):
    """转 base64；图片尺寸超过 max_dim 时先压缩（与分析流程一致）。

    返回 (mime_type, base64)。压缩失败时回退为原始文件 base64。
    """
    try:
        from PIL import Image
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        width, height = img.size
        if max(width, height) > max_dim:
            scaling = max_dim / max(width, height)
            img = img.resize((int(width * scaling), int(height * scaling)), Image.Resampling.LANCZOS)
            logger.info(f"参考图过大已压缩: {path} ({width}x{height}) -> {int(width * scaling)}x{int(height * scaling)}")
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        return "image/jpeg", base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        logger.warning(f"参考图压缩失败，使用原图 base64: {path} - {e}")
        return _guess_image_mime_type(path), to_base64(path)

def _derive_edits_url_from_generations_url(generations_url: str) -> str:
    normalized = str(generations_url or "").rstrip("/")
    if normalized.endswith("/images/generations"):
        return normalized[:-len("/images/generations")] + "/images/edits"
    if normalized.endswith("/v1"):
        return f"{normalized}/images/edits"
    return f"{normalized}/v1/images/edits"

def resolve_images_endpoint(api_base: str, has_images: bool = False, api_type: str = None) -> str:
    """
    统一解析 OpenAI 兼容的 images 端点（GUI / CLI / 测试共用）：
    - aigc-2d-gpt：`/v1/images/generations`（base 已带 /images/* 时原样沿用）
    - 其他（autodl / openai-image）：base 以 `/v1` 结尾用 `/v1/images/generations`，否则 `/v1beta/images/generations`
    - has_images=True 时切换到对应的 `/images/edits`
    """
    base = str(api_base or "").strip().rstrip("/")
    normalized = str(api_type or "").strip().lower()
    if base.endswith("/images/edits"):
        base = base[: -len("/images/edits")] + "/images/generations"
    if base.endswith("/images/generations"):
        generations_url = base
    elif normalized in {"aigc-2d-gpt", "aigc2d-gpt", "aigc_2d_gpt"}:
        generations_url = f"{base}/images/generations" if base.endswith("/v1") else f"{base}/v1/images/generations"
    else:
        generations_url = f"{base}/images/generations" if base.endswith("/v1") else f"{base}/v1beta/images/generations"
    return _derive_edits_url_from_generations_url(generations_url) if has_images else generations_url

def _build_aigc2d_generate_content_url(api_base: str, model_name: str) -> str:
    """
    统一将 AIGC2D 的 base_url 规范化为:
    https://new.aigc2d.com/v1beta/models/{model}:generateContent
    兼容用户可能填写的 /v1、/v1beta、/v1beta/models 以及旧的完整 endpoint。
    """
    base = str(api_base or "").strip().rstrip("/")
    if not base:
        base = "https://new.aigc2d.com/v1beta/models"

    if base.endswith("/models"):
        prefix = base
    elif "/models/" in base:
        prefix = base.split("/models/", 1)[0] + "/models"
    elif base.endswith("/v1beta"):
        prefix = f"{base}/models"
    elif base.endswith("/v1"):
        prefix = f"{base[:-3]}/v1beta/models"
    else:
        prefix = f"{base}/v1beta/models"

    model_text = str(model_name or "").strip() or "gemini-3.1-flash-image-preview"
    return f"{prefix.rstrip('/')}/{model_text}:generateContent"

def _post_images_edits_request(url: str, api_key: str, form_payload: dict, image_paths: list, timeout_val: int):
    used_files = []
    temp_files = []
    try:
        with ExitStack() as stack:
            files = []
            for img_path in image_paths:
                prepared_path = _maybe_compress_image_path(img_path, temp_files=temp_files)
                fh = stack.enter_context(open(prepared_path, "rb"))
                mime_type = _guess_image_mime_type(prepared_path)
                files.append(("image[]", (os.path.basename(prepared_path), fh, mime_type)))
                used_files.append({"name": os.path.basename(prepared_path), "path": prepared_path, "mime_type": mime_type})
            data_payload = {k: str(v) for k, v in (form_payload or {}).items() if v is not None}
            headers = {"Authorization": f"Bearer {api_key}"}
            resp = requests.post(url, headers=headers, data=data_payload, files=files, timeout=timeout_val)
    finally:
        _cleanup_temp_files(temp_files)
    return resp, used_files

def generate_image_openai_image(prompt: str, image_paths: list = None, model: str = "gpt-image-2", aspect_ratio: str = "1:1", instructions: str = "", resolution: str = None, api_type: str = None, save_sub_dir: str = None, file_prefix: str = None, return_metadata: bool = False, size: str = None, quality: str = None, output_format: str = None, config_path: str = None) -> list:
    """
    openai-image 生图接口（OpenAI 风格）：
    - 入口端点：/v1/images/generations
    - 请求体风格：OpenAI Image API
    - 默认模型：gpt-image-2
    """
    normalized_api_type = str(api_type or "").strip().lower()
    if normalized_api_type in {"aigc-2d-gpt", "aigc2d-gpt", "aigc_2d_gpt"}:
        normalized_api_type = "openai-image"
    config = get_api_config(config_path=config_path, api_type=normalized_api_type or "openai-image")
    api_base = str(config.get("base_url", "https://api.openai.com/v1") or "https://api.openai.com/v1").rstrip("/")
    url = resolve_images_endpoint(api_base, has_images=False, api_type=normalized_api_type)
    api_key = config.get("api_key")
    timeout_val = config.get("timeout", 180)
    max_retries = config.get("max_retries", 1)
    debug_dump_full_http = _as_bool(config.get("debug_dump_full_http", False), False)

    if not api_key:
        logger.error("配置文件 conf/config.json 中缺少 'api_key' 参数。")
        return []

    valid_image_paths = _existing_image_paths(image_paths)

    output_format = str(config.get("output_format", "png") or "png").lower()
    quality = str(quality or config.get("quality", "high") or "high")
    raw_allow_auto_size = config.get("allow_auto_size", True)
    if isinstance(raw_allow_auto_size, str):
        allow_auto_size = raw_allow_auto_size.strip().lower() not in ("0", "false", "no", "off")
    else:
        allow_auto_size = bool(raw_allow_auto_size)
    # 尺寸优先级: 显式 size 参数 > 配置 size 字段 > 按长宽比推导
    request_size = str(size or config.get("size") or "").strip()
    if not request_size:
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
    logger.info(f"请求 Headers: {_headers_for_log({'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'})}")
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
                if return_metadata:
                    raw_text = _response_text_utf8(resp) if resp is not None else str(e)
                    replay_info = _generate_request_replay(fail_dir, file_prefix, "openai-image", request_url,
                                                            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, payload)
                    return {
                        "saved_files": [],
                        "annotation": {},
                        "raw_text": raw_text,
                        "server_response_raw": {"error": str(e), "note": "retry_exhausted", "raw_text": raw_text[:2000]},
                        "replay_script": replay_info.get("replay_script", ""),
                        "replay_json": replay_info.get("replay_json", "")
                    }
                return []
    try:
        if getattr(resp, "encoding", None) is None:
            resp.encoding = "utf-8"
        resp_text_raw = _response_text_utf8(resp)
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
        _save_server_response_raw(fail_dir, file_prefix, "openai-image", resp_text_raw)
        if return_metadata:
            replay_info = _generate_request_replay(fail_dir, file_prefix, "openai-image", request_url,
                                                    {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, payload)
            return {
                "saved_files": [],
                "annotation": {},
                "raw_text": resp_text_raw,
                "server_response_raw": resp_json if 'resp_json' in locals() else {"parse_error": str(e), "raw_text": resp_text_raw[:2000]},
                "replay_script": replay_info.get("replay_script", ""),
                "replay_json": replay_info.get("replay_json", "")
            }
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
        logger.warning(f"=== openai-image 服务器完整返回 ===\n{_format_safe_log(resp_json)}")
        _save_server_response_json(save_dir, file_prefix, "openai-image", resp_json)
        _save_server_response_raw(save_dir, file_prefix, "openai-image", _response_text_utf8(resp))
        if return_metadata:
            replay_info = _generate_request_replay(save_dir, file_prefix, "openai-image", request_url,
                                                    {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, payload)
            return {
                "saved_files": [],
                "annotation": {},
                "raw_text": _response_text_utf8(resp),
                "server_response_raw": resp_json,
                "replay_script": replay_info.get("replay_script", ""),
                "replay_json": replay_info.get("replay_json", "")
            }
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
        logger.warning(f"=== openai-image 服务器完整返回(图片提取全部失败) ===\n{_format_safe_log(resp_json)}")
        _save_server_response_json(save_dir, file_prefix, "openai-image", resp_json)
        _save_server_response_raw(save_dir, file_prefix, "openai-image", _response_text_utf8(resp))

    raw_text = "\n".join(revised_prompt_parts).strip()
    annotation_data = _normalize_annotation_result(_extract_json_object(raw_text))
    if return_metadata:
        replay_info = _generate_request_replay(save_dir, file_prefix, "openai-image", request_url,
                                                {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, payload)
        return {
            "saved_files": saved_files,
            "annotation": annotation_data,
            "raw_text": raw_text,
            "server_response_raw": resp_json,
            "replay_script": replay_info.get("replay_script", ""),
            "replay_json": replay_info.get("replay_json", "")
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
        logger.error("配置文件 conf/config.json 中缺少 'api_key' 参数。")
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
    logger.info(f"请求 Headers: {_headers_for_log({'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'})}")
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
                if return_metadata:
                    raw_text = _response_text_utf8(resp) if resp is not None else str(e)
                    replay_info = _generate_request_replay(fail_dir, file_prefix, "openrouter-image", request_url,
                                                            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, payload)
                    return {
                        "saved_files": [],
                        "annotation": {},
                        "raw_text": raw_text,
                        "server_response_raw": {"error": str(e), "note": "timeout_retry_exhausted", "raw_text": raw_text[:2000]},
                        "replay_script": replay_info.get("replay_script", ""),
                        "replay_json": replay_info.get("replay_json", "")
                    }
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
            if return_metadata:
                raw_text = _response_text_utf8(resp) if resp is not None else str(e)
                replay_info = _generate_request_replay(fail_dir, file_prefix, "openrouter-image", request_url,
                                                        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, payload)
                return {
                    "saved_files": [],
                    "annotation": {},
                    "raw_text": raw_text,
                    "server_response_raw": {"error": str(e), "note": "request_exception_retry_exhausted", "raw_text": raw_text[:2000]},
                    "replay_script": replay_info.get("replay_script", ""),
                    "replay_json": replay_info.get("replay_json", "")
                }
            return []

    try:
        if getattr(resp, "encoding", None) is None:
            resp.encoding = "utf-8"
        resp_text_raw = _response_text_utf8(resp)
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
        _save_server_response_raw(fail_dir, file_prefix, "openrouter-image", resp_text_raw)
        if return_metadata:
            replay_info = _generate_request_replay(fail_dir, file_prefix, "openrouter-image", request_url,
                                                    {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, payload)
            return {
                "saved_files": [],
                "annotation": {},
                "raw_text": resp_text_raw,
                "server_response_raw": resp_json if 'resp_json' in locals() else {"parse_error": str(e), "raw_text": resp_text_raw[:2000]},
                "replay_script": replay_info.get("replay_script", ""),
                "replay_json": replay_info.get("replay_json", "")
            }
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
        logger.warning(f"=== openrouter-image 服务器完整返回 ===\n{_format_safe_log(resp_json)}")
        _save_server_response_json(save_dir, file_prefix, "openrouter-image", resp_json)
        _save_server_response_raw(save_dir, file_prefix, "openrouter-image", _response_text_utf8(resp))
        if return_metadata:
            replay_info = _generate_request_replay(save_dir, file_prefix, "openrouter-image", request_url,
                                                    {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, payload)
            return {
                "saved_files": [],
                "annotation": {},
                "raw_text": _response_text_utf8(resp),
                "server_response_raw": resp_json,
                "replay_script": replay_info.get("replay_script", ""),
                "replay_json": replay_info.get("replay_json", "")
            }
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
        logger.warning(f"=== openrouter-image 服务器完整返回(图片提取全部失败) ===\n{_format_safe_log(resp_json)}")
        _save_server_response_json(save_dir, file_prefix, "openrouter-image", resp_json)
        _save_server_response_raw(save_dir, file_prefix, "openrouter-image", _response_text_utf8(resp))

    raw_text = "\n".join(raw_text_parts).strip()
    annotation_data = _normalize_annotation_result(_extract_json_object(raw_text))
    if return_metadata:
        replay_info = _generate_request_replay(save_dir, file_prefix, "openrouter-image", request_url,
                                                {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, payload)
        return {
            "saved_files": saved_files,
            "annotation": annotation_data,
            "raw_text": raw_text,
            "server_response_raw": resp_json,
            "replay_script": replay_info.get("replay_script", ""),
            "replay_json": replay_info.get("replay_json", "")
        }
    return saved_files

def generate_image_aigc2d_gpt(prompt: str, image_paths: list = None, model: str = "gpt-image-2", aspect_ratio: str = "1:1", instructions: str = "", resolution: str = None, api_type: str = None, save_sub_dir: str = None, file_prefix: str = None, return_metadata: bool = False, size: str = None, quality: str = None, output_format: str = None, n: int = None, mode: str = None, cancel_check: callable = None, post_instructions: str = "", log_callback: callable = None, config_path: str = None) -> list:
    """
    aigc2d 的 gpt-image-2 专用通道（与 Gemini 的 /v1beta/models 通道完全分离）：
    - 纯文生图：POST {base}/v1/images/generations，JSON 体 = model / prompt / n / size(/quality/output_format)
    - 带参考图（生图垫图 or 编辑）：POST {base}/v1/images/edits，multipart，图片字段 image[]，最多 16 张
    - size 只有 3 档：1024x1024(方) / 1536x1024(横) / 1024x1536(竖)，不接受 auto 与自定义尺寸
    - 不支持 background=transparent 与 input_fidelity，这两个字段不发送
    - high 画质单张常见 3~5 分钟，超时取配置 timeout（默认 600s）
    - 返回 data[].b64_json（或无 b64 时的 data[].url），落盘后会另存一份服务器原始 JSON
    """
    resolved_api_type = api_type or "aigc-2d-gpt"
    config = get_api_config(config_path=config_path, api_type=resolved_api_type)
    api_base = str(config.get("base_url", "https://new.aigc2d.com/v1/images/generations") or "https://new.aigc2d.com/v1/images/generations").strip().rstrip("/")
    url = resolve_images_endpoint(api_base, has_images=False, api_type=resolved_api_type)
    api_key = config.get("api_key")
    timeout_val = int(config.get("timeout", 600) or 600)
    max_retries = int(config.get("max_retries", 1) or 1)
    retry_backoff_s = float(config.get("retry_backoff", 1.0) or 1.0)
    debug_dump_full_http = _as_bool(config.get("debug_dump_full_http", False), False)

    def _emit(message: str):
        try:
            logger.info(str(message))
        except Exception:
            pass
        if log_callback:
            try:
                log_callback(str(message))
            except Exception:
                pass

    def _is_cancelled() -> bool:
        if not cancel_check:
            return False
        try:
            return bool(cancel_check())
        except Exception:
            return False

    if not api_key:
        logger.error("配置文件 conf/config.json 中缺少 'api_key' 参数。")
        return []

    # 尺寸优先级：显式 size 参数 > 配置 size 字段 > 按长宽比收敛；最终一定是 3 档之一
    request_size = normalize_gpt_image2_size(
        size=size or config.get("size"),
        aspect_ratio=aspect_ratio,
    )
    request_quality = str(quality or config.get("quality", "") or "").strip().lower()
    request_output_format = str(output_format or config.get("output_format", "") or "").strip().lower()
    request_n = n if n is not None else config.get("n", 1)

    payload = build_gpt_image2_payload(
        prompt=prompt,
        model=str(model or config.get("model") or "gpt-image-2"),
        size=request_size,
        aspect_ratio=aspect_ratio,
        quality=request_quality,
        output_format=request_output_format,
        n=request_n,
        instructions=instructions,
        post_instructions=post_instructions,
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    valid_image_paths = _existing_image_paths(image_paths)
    mode_text = str(mode or "").strip().lower()
    force_edit = mode_text in {"edit", "edits", "image-edit", "编辑", "图片编辑"}
    # 「新建图片」模式（用户 2026-09-24 要求）：首图走 /images/generations（不是 edits），
    # 但画风参考图仍然带上 —— 实测中转站的 generations 端点接受 `image` 字段（base64 data URI），
    # 这样模型是"从文字新建一张图"，而不是"把参考图当原图来编辑"（edits 会把参考图的角色/构图搬过来）。
    force_generate = mode_text in {"generate", "generation", "new", "生图", "新建", "新建图片"}
    if force_edit and not valid_image_paths:
        logger.error("AIGC-2D-GPT(edit) 需要至少 1 张参考图/原图，但未收到有效图片路径。")
        _emit("编辑模式需要至少 1 张图片，已中止。")
        return []
    if len(valid_image_paths) > GPT_IMAGE2_MAX_REFERENCE_IMAGES:
        logger.warning(
            f"AIGC-2D-GPT 参考图 {len(valid_image_paths)} 张，超过上游上限 "
            f"{GPT_IMAGE2_MAX_REFERENCE_IMAGES} 张，只发送前 {GPT_IMAGE2_MAX_REFERENCE_IMAGES} 张。"
        )
        valid_image_paths = valid_image_paths[:GPT_IMAGE2_MAX_REFERENCE_IMAGES]

    # 带图就走 edits（gpt-image-2 的参考图/编辑入口），否则走 generations；
    # 显式 mode=generate/new 时即使带图也走 generations，把图放进 JSON 的 `image` 字段。
    use_edits_mode = len(valid_image_paths) > 0 and not force_generate
    request_url = _derive_edits_url_from_generations_url(url) if use_edits_mode else url
    if use_edits_mode:
        _emit(f"模式=/images/edits（参考图 {len(valid_image_paths)} 张）  尺寸={payload['size']}  画质={payload.get('quality', '服务端默认')}")
    elif valid_image_paths:
        images_b64 = []
        for path in valid_image_paths:
            try:
                mime, b64 = to_base64_compressed(path)
                images_b64.append(f"data:{mime or 'image/png'};base64,{b64}")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"参考图编码失败（已跳过）: {path} -> {exc}")
        if images_b64:
            payload["image"] = images_b64[0] if len(images_b64) == 1 else images_b64
        _emit(f"模式=/images/generations（新建图片 + 参考图 {len(images_b64)} 张）  "
              f"尺寸={payload['size']}  画质={payload.get('quality', '服务端默认')}")
    else:
        _emit(f"模式=/images/generations（无参考图）  尺寸={payload['size']}  画质={payload.get('quality', '服务端默认')}")

    logger.info(f"=== 发起 gpt-image(aigc2d) API 请求  model={payload.get('model')} ===")
    logger.info(f"请求 URL: {request_url}")
    logger.info(f"请求 Headers: {_headers_for_log(headers)}")
    logger.info(f"请求数据:\n{_format_safe_log(payload)}")

    if _is_cancelled():
        _emit("请求已被取消。")
        return []

    stage_json_start = time.perf_counter()
    resp = None
    last_exc = None
    request_trace_body = payload

    def _post_once():
        """发一次请求（edits 走 multipart，generations 走 JSON），返回 (resp, trace_body)。"""
        if use_edits_mode:
            r, files = _post_images_edits_request(url=request_url, api_key=api_key, form_payload=payload,
                                                  image_paths=valid_image_paths, timeout_val=timeout_val)
            return r, {"form": payload, "files": files}
        return requests.post(request_url, headers=headers, json=payload, timeout=timeout_val), payload

    for attempt in range(max_retries + 1):
        if _is_cancelled():
            logger.info("AIGC-2D-GPT 请求在重试前被取消。")
            _emit("请求已被取消。")
            return []
        try:
            if attempt > 0:
                logger.info(f"正在进行第 {attempt} 次重试 (最大重试次数: {max_retries})...")
            resp, request_trace_body = _post_once()
            # 只有 429/5xx 这类可重试状态才抛异常触发重试；4xx（含审核拦截）直接落盘原始返回
            status_code = getattr(resp, "status_code", None)
            if isinstance(status_code, int) and (status_code == 429 or 500 <= status_code < 600):
                resp.raise_for_status()
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
        if return_metadata:
            replay_info = _generate_request_replay(save_dir, file_prefix, "aigc-2d-gpt", request_url, headers, payload)
            return {
                "saved_files": [],
                "annotation": {},
                "raw_text": str(last_exc or "request failed"),
                "server_response_raw": {"error": str(last_exc or "request failed"), "note": "no_response"},
                "replay_script": replay_info.get("replay_script", ""),
                "replay_json": replay_info.get("replay_json", "")
            }
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
        resp_text_raw = _response_text_utf8(resp)
        resp_json = resp.json()
    except Exception:
        resp_json = {"raw_text": resp_text, "status_code": getattr(resp, "status_code", None)}
    _save_server_response_json(save_dir, file_prefix, "aigc-2d-gpt", resp_json)
    logger.info(f"=== AIGC-2D-GPT 服务器原始返回信息 ===\n{_format_safe_log(resp_json)}")
    _log_stage_elapsed("阶段1-获取JSON响应", stage_json_start)

    data_items = resp_json.get("data", []) if isinstance(resp_json, dict) else []
    # 中转站按请求随机挑上游：某些通道不认我们发的字段（实测 `/images/generations` 的 `image` 字段会
    # 被判成 `Unknown parameter: 'image'`，同一份请求换个通道就正常）。这类「上游报错 + 空 data」
    # 重发一次基本就好，所以这里单独再试几次，别让画风生图随机失败。
    upstream_retries = int(config.get("upstream_error_retries", 2) or 2)
    while (not isinstance(data_items, list) or not data_items) and upstream_retries > 0 \
            and isinstance(resp_json, dict) and resp_json.get("error"):
        upstream_retries -= 1
        err_text = str((resp_json.get("error") or {}).get("message") or resp_json.get("error"))[:200]
        logger.warning(f"AIGC-2D-GPT 上游返回错误（{err_text}），重新发起请求（还剩 {upstream_retries} 次）…")
        _emit(f"上游返回错误，重发请求：{err_text}")
        time.sleep(max(0.5, retry_backoff_s))
        if _is_cancelled():
            logger.info("AIGC-2D-GPT 请求在重发前被取消。")
            return []
        try:
            resp, request_trace_body = _post_once()
        except requests.exceptions.RequestException as e:
            logger.warning(f"AIGC-2D-GPT 重发失败: {e}")
            break
        try:
            if getattr(resp, "encoding", None) is None:
                resp.encoding = "utf-8"
            resp_text_raw = _response_text_utf8(resp)
            resp_text = resp_text_raw
            resp_json = resp.json()
        except Exception:  # noqa: BLE001
            resp_json = {"raw_text": resp_text, "status_code": getattr(resp, "status_code", None)}
        _save_server_response_json(save_dir, file_prefix, "aigc-2d-gpt", resp_json)
        logger.info(f"=== AIGC-2D-GPT 服务器原始返回信息（重发后） ===\n{_format_safe_log(resp_json)}")
        data_items = resp_json.get("data", []) if isinstance(resp_json, dict) else []
        stage_json_start = time.perf_counter()
    # 最后兜底：某些上游通道压根不认 generations 的 `image` 字段（`Unknown parameter: 'image'`），
    # 重发也一直落到这类通道 → 改用 `/images/edits`（multipart 传参考图）。语义上 edits 更像
    # 「把参考图当原图编辑」，所以**只在报错时回退**，并且日志里写明，方便排查产物差异。
    if (not isinstance(data_items, list) or not data_items) and isinstance(resp_json, dict) \
            and resp_json.get("error") and not use_edits_mode \
            and "image" in str(resp_json.get("error")).lower():
        logger.warning("AIGC-2D-GPT 上游不认 generations 的 image 字段 → 回退 /images/edits 重发一次")
        _emit("上游不支持「新建图片」模式的参考图字段，已回退到编辑端点重发")
        use_edits_mode = True
        request_url = _derive_edits_url_from_generations_url(url)
        try:
            resp, request_trace_body = _post_once()
            if getattr(resp, "encoding", None) is None:
                resp.encoding = "utf-8"
            resp_text = _response_text_utf8(resp)
            resp_json = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"AIGC-2D-GPT 回退 edits 重发失败: {exc}")
            resp_json = resp_json if isinstance(resp_json, dict) else {}
        else:
            _save_server_response_json(save_dir, file_prefix, "aigc-2d-gpt", resp_json)
            logger.info(f"=== AIGC-2D-GPT 服务器原始返回信息（edits 回退后） ===\n{_format_safe_log(resp_json)}")
        data_items = resp_json.get("data", []) if isinstance(resp_json, dict) else []
    if not isinstance(data_items, list) or not data_items:
        logger.warning("AIGC-2D-GPT 返回 JSON 中没有 data 节点。")
        logger.warning(f"=== AIGC-2D-GPT 服务器完整返回 ===\n{_format_safe_log(resp_json)}")
        _save_server_response_raw(save_dir, file_prefix, "aigc-2d-gpt", resp_text)
        if return_metadata:
            replay_info = _generate_request_replay(save_dir, file_prefix, "aigc-2d-gpt", request_url, headers, payload)
            return {
                "saved_files": [],
                "annotation": {},
                "raw_text": resp_text,
                "server_response_raw": resp_json,
                "replay_script": replay_info.get("replay_script", ""),
                "replay_json": replay_info.get("replay_json", "")
            }
        return []

    saved_files = []
    revised_prompt_parts = []
    stage_save_start = time.perf_counter()
    resp_output_format = ""
    if isinstance(resp_json, dict):
        resp_output_format = str(resp_json.get("output_format") or "")
    output_format_for_ext = str(payload.get("output_format") or resp_output_format or "").strip().lower()
    for idx, item in enumerate(data_items):
        if not isinstance(item, dict):
            continue
        revised_prompt = item.get("revised_prompt")
        if revised_prompt:
            revised_prompt_parts.append(str(revised_prompt))

        b64_data = item.get("b64_json")
        if b64_data:
            try:
                if isinstance(b64_data, str) and b64_data.startswith("data:"):
                    # 部分中转会带 data:image/png;base64, 前缀，先剥掉
                    b64_data = b64_data.split(",", 1)[-1]
                image_bytes = base64.b64decode(b64_data)
                ext = gpt_image2_output_extension(output_format_for_ext, image_bytes)
                prefix = f"{file_prefix}_" if file_prefix else ""
                file_name = f"{prefix}output_{datetime.now().strftime('%H%M%S')}_{idx}_{uuid.uuid4().hex[:6]}{ext}"
                file_path = os.path.join(save_dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(image_bytes)
                logger.info(f"✅ 成功保存 AIGC-2D-GPT 图片({ext}): {file_path}")
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
                ext = gpt_image2_output_extension(output_format_for_ext, img_data)
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
    if len(saved_files) == 0:
        logger.warning(f"=== AIGC-2D-GPT 服务器完整返回(图片提取全部失败) ===\n{_format_safe_log(resp_json)}")
        _save_server_response_json(save_dir, file_prefix, "aigc-2d-gpt", resp_json)
        _save_server_response_raw(save_dir, file_prefix, "aigc-2d-gpt", resp_text)

    raw_text = "\n".join(revised_prompt_parts).strip()
    annotation_data = _normalize_annotation_result(_extract_json_object(raw_text))
    if return_metadata:
        replay_info = _generate_request_replay(save_dir, file_prefix, "aigc-2d-gpt", request_url, headers, payload)
        return {
            "saved_files": saved_files,
            "annotation": annotation_data,
            "raw_text": raw_text,
            "server_response_raw": resp_json,
            "replay_script": replay_info.get("replay_script", ""),
            "replay_json": replay_info.get("replay_json", "")
        }
    return saved_files

def generate_image_whatai(prompt: str, image_paths: list = None, model: str = "nano-banana-2", aspect_ratio: str = "1:1", instructions: str = "", resolution = "1K", api_type: str = None, save_sub_dir: str = None, file_prefix: str = None, return_metadata: bool = False, cancel_check: callable = None, post_instructions: str = "") -> list:
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
    if normalized_api_type in {"autodl"}:
        # autodl 站点(OpenAI 兼容): gpt-image-2 /images/generations + /images/edits
        return generate_image_openai_image(
            prompt=prompt,
            image_paths=image_paths,
            model=model,
            aspect_ratio=aspect_ratio,
            instructions=instructions,
            resolution=resolution,
            api_type="autodl",
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
            return_metadata=return_metadata,
            cancel_check=cancel_check,
            post_instructions=post_instructions
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
        logger.error("配置文件 conf/config.json 中缺少 'api_key' 参数。")
        raise ValueError("配置文件 conf/config.json 中缺少 'api_key' 参数。")

    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 将用户保存的 Instructions 和具体的 prompt 拼接传入
    _face_quality_boost = "detailed face, clear facial features, sharp focus on face"
    combined_prompt = f"--ar {aspect_ratio} ,  {instructions}  {prompt}, {_face_quality_boost}"
    content_list = [{"type": "text", "text": combined_prompt}]

    if image_paths:
        for img_path in image_paths:
            if os.path.exists(img_path):
                mime_type, b64_data = to_base64_compressed(img_path)
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}
                })
            else:
                logger.warning(f"找不到本地图片文件 {img_path}，已跳过。")

    # 交错模式：在图片之后紧跟补充指令（仅当调用方显式传入时启用）
    if post_instructions:
        content_list.append({"type": "text", "text": post_instructions})

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
    
    logger.info("=== 发起 whatai API 请求 ===")
    logger.info(f"请求 URL: {url}")
    logger.info(f"请求 Headers: {_headers_for_log(headers)}")
    logger.info(f"请求数据:\n{json.dumps(safe_data, ensure_ascii=False, indent=2)}")

    # ================= 阶段1：请求并获取 JSON 响应 =================
    stage_json_start = time.perf_counter()
    resp = None
    for attempt in range(max_retries + 1):
        try:
            if callable(cancel_check) and cancel_check():
                logger.info("收到外部取消请求，停止重试（阶段1-请求前）。")
                if return_metadata:
                    return {
                        "saved_files": [],
                        "annotation": {},
                        "raw_text": "cancelled",
                        "server_response_raw": {"error": "cancelled_before_request"},
                    }
                return []
            if attempt > 0:
                logger.info(f"正在进行第 {attempt} 次重试 (最大重试次数: {max_retries})...")
            
            resp = requests.post(url, headers=headers, json=data, timeout=timeout_val)
            resp.raise_for_status()
            break  # 如果没有抛出异常，说明请求成功，跳出循环
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"网络请求发生异常 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
            if callable(cancel_check) and cancel_check():
                logger.info("收到外部取消请求，停止重试。")
                if return_metadata:
                    return {
                        "saved_files": [],
                        "annotation": {},
                        "raw_text": str(e),
                        "server_response_raw": {"error": "cancelled", "raw_text": str(e)[:2000]},
                    }
                return []
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
                if return_metadata:
                    raw_text = _response_text_utf8(resp) if resp is not None else str(e)
                    replay_info = _generate_request_replay(fail_dir, file_prefix, "whatai", url, headers, data)
                    return {
                        "saved_files": [],
                        "annotation": {},
                        "raw_text": raw_text,
                        "server_response_raw": {"error": str(e), "note": "retry_exhausted", "raw_text": raw_text[:2000]},
                        "replay_script": replay_info.get("replay_script", ""),
                        "replay_json": replay_info.get("replay_json", "")
                    }
                return []
    # ====================================================================

    try:
        if getattr(resp, "encoding", None) is None:
            resp.encoding = "utf-8"
        resp_text = _response_text_utf8(resp)
        resp_json = resp.json()
        
        logger.info(f"=== 服务器原始返回信息 ===\n{_format_safe_log(resp_json)}")
        
        content_str = resp_json["choices"][0]["message"]["content"]
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"解析返回 JSON 失败: {e}")
        fail_today = datetime.now().strftime("%Y%m%d")
        fail_dir = os.path.join("data", fail_today, save_sub_dir) if save_sub_dir else os.path.join("data", fail_today)
        raw_text = _response_text_utf8(resp) if resp is not None else str(e)
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
        _save_server_response_raw(fail_dir, file_prefix, "whatai", raw_text)
        if return_metadata:
            replay_info = _generate_request_replay(fail_dir, file_prefix, "whatai", url, headers, data)
            return {
                "saved_files": [],
                "annotation": {},
                "raw_text": raw_text,
                "server_response_raw": resp_json if 'resp_json' in locals() else {"parse_error": str(e), "raw_text": raw_text[:2000]},
                "replay_script": replay_info.get("replay_script", ""),
                "replay_json": replay_info.get("replay_json", "")
            }
        return []
    _log_stage_elapsed("阶段1-获取JSON响应", stage_json_start)

    if callable(cancel_check) and cancel_check():
        logger.info("收到外部取消请求，跳过图片下载（阶段2-下载前）。")
        if return_metadata:
            return {
                "saved_files": [],
                "annotation": {},
                "raw_text": "cancelled_after_response",
                "server_response_raw": {"error": "cancelled_before_download"},
            }
        return []

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
        logger.warning(f"=== whatai 服务器完整返回 ===\n{_format_safe_log(resp_json)}")
        _save_server_response_json(save_dir, file_prefix, "whatai", resp_json)
        _save_server_response_raw(save_dir, file_prefix, "whatai", content_str)
        if return_metadata:
            replay_info = _generate_request_replay(save_dir, file_prefix, "whatai", url, headers, data)
            return {
                "saved_files": [],
                "annotation": {},
                "raw_text": content_str,
                "server_response_raw": resp_json,
                "replay_script": replay_info.get("replay_script", ""),
                "replay_json": replay_info.get("replay_json", "")
            }
        return []
    saved_files = []

    # ================= 阶段2：下载并保存图片 =================
    stage_download_start = time.perf_counter()
    for idx, img_url in enumerate(img_urls):
        if callable(cancel_check) and cancel_check():
            logger.info("收到外部取消请求，停止图片下载。")
            break
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
        logger.warning(f"=== whatai 服务器完整返回(图片下载全部失败) ===\n{_format_safe_log(resp_json)}")
        _save_server_response_json(save_dir, file_prefix, "whatai", resp_json)
        _save_server_response_raw(save_dir, file_prefix, "whatai", content_str)

    if return_metadata:
        replay_info = _generate_request_replay(save_dir, file_prefix, "whatai", url, headers, data)
        return {
            "saved_files": saved_files,
            "annotation": annotation_data,
            "raw_text": content_str,
            "server_response_raw": resp_json,
            "replay_script": replay_info.get("replay_script", ""),
            "replay_json": replay_info.get("replay_json", "")
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
    logger.info(f"请求 Headers: {_headers_for_log(headers)}")
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
    config_path = os.path.join(CONFIG_DIR, "config-cohere.json")
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
    logger.info(f"请求 URL: {url}")
    logger.info(f"请求 Headers: {_headers_for_log(headers)}")
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

def generate_image_aigc2d(prompt: str, image_paths: list = None, model: str = "gemini-3.1-flash-image-preview", aspect_ratio: str = "1:1", instructions: str = "", resolution: str = None, api_type: str = None, save_sub_dir: str = None, file_prefix: str = None, return_metadata: bool = False, log_callback=None, cancel_check: callable = None, post_instructions: str = "", prompt_suffix: str = "", face_quality_boost: bool = True) -> list:
    """
    AIGC2D 专用的图片生成核心逻辑
    入参跟 generate_image_whatai 保持完全一致；
    `prompt_suffix` 为追加段落（重绘通道用它挂「细节强化要求」，见 utils/gpt_image_optimize.py）；
    `face_quality_boost=False` 时不追加「detailed face…」（重绘时 prompt 已自带更强约束）。
    """
    # 从统一配置文件中加载 aigc2d 配置
    config = get_api_config(api_type="aigc2d" if not api_type else api_type)

    def _log(msg: str) -> None:
        if log_callback:
            log_callback(msg)

    api_base = str(config.get("base_url", "https://new.aigc2d.com/v1beta/models/") or "https://new.aigc2d.com/v1beta/models/").strip()
    api_key = config.get("api_key")
    timeout_val = config.get("timeout", 180)
    max_retries = config.get("max_retries", 1)
    debug_dump_full_http = _as_bool(config.get("debug_dump_full_http", False), False)
    # 从配置中读取resolution，如果传入了参数则使用传入的参数
    if resolution is None:
        resolution = config.get("resolution", "1K")

    if not api_key:
        _log("[生成/api] 错误: conf/config.json 中缺少 api_key")
        logger.error("配置文件 conf/config.json 中缺少 'api_key' 参数。")
        return []

    _log(f"[生成/api] 接口: {api_base.rstrip('/')}")
    _log(f"[生成/api] 模型: {model}, 分辨率: {resolution}, 超时: {timeout_val}s")

    # AIGC2D 接口 URL 统一规范：/v1beta/models/{model}:generateContent
    url = _build_aigc2d_generate_content_url(api_base, model)
    headers = {
        "x-goog-api-key": str(api_key),
        "Content-Type": "application/json"
    }

    # 组合提示词
    _face_quality_boost = ", detailed face, clear facial features, sharp focus on face"
    boost = _face_quality_boost if face_quality_boost else ""
    combined_prompt = f"{instructions} \n {prompt}{boost}".strip() if instructions else f"{prompt}{boost}"
    # 重绘（gpt-image 产物优化）通道：用独立段落追加细节强化要求，替代逗号后缀
    suffix_text = str(prompt_suffix or "").strip()
    if suffix_text:
        combined_prompt = f"{combined_prompt}\n\n{suffix_text}"
    parts = [{"text": combined_prompt}]

    # 处理传入的参考图片（支持多图，按照入参列表追加）
    if image_paths:
        for img_path in image_paths:
            if os.path.exists(img_path):
                mime_type, b64_data = to_base64_compressed(img_path)
                parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": b64_data
                    }
                })
            else:
                logger.warning(f"找不到本地图片文件 {img_path}，已跳过。")

    # 交错模式：在图片之后紧跟补充指令（仅当调用方显式传入时启用）
    if post_instructions:
        parts.append({"text": post_instructions})

    # 构造请求 Payload
    # 宽高比：空 / auto 时不传 aspectRatio —— 官方默认行为是按输入图比例输出，
    # 这样换任何比例的源图都不会被拉伸或裁切（显式比例才下发该字段）。
    image_config = {"imageSize": resolution}
    ratio_text = str(aspect_ratio or "").strip()
    if ratio_text and ratio_text.lower() not in {"auto", "自动", "跟随源图", "keep", "same", "original", "none", "null"}:
        image_config = {"aspectRatio": ratio_text, "imageSize": resolution}
    payload = {
        "contents": [
            {
                # Gemini generateContent 的 contents 项必须带 role（user/model）；上游新 API 缺失即 400
                # "Please use a valid role: user, model."（2026-09-21 起复现，补上后恢复正常）
                "role": "user",
                "parts": parts
            }
        ],
        "generationConfig": {
            "imageConfig": image_config
        }
    }

    # 日志脱敏：隐藏 Base64 字符串
    safe_payload = copy.deepcopy(payload)
    for part in safe_payload.get("contents", [])[0].get("parts", []):
        if "inline_data" in part and "data" in part["inline_data"]:
            part["inline_data"]["data"] = "<BASE64_IMAGE_DATA_OMITTED>"
            
    logger.info("=== 发起 AIGC2D API 请求 ===")
    logger.info(f"请求 URL: {url}")
    logger.info(f"请求 Headers: {_headers_for_log(headers)}")
    logger.info(f"请求数据:\n{json.dumps(safe_payload, ensure_ascii=False, indent=2)}")
    _log(f"[生成/api] 发送请求: {url}")
    _log(f"[生成/api] 图片参考: {len(image_paths or [])} 张, Prompt: {len(combined_prompt)} chars")

    # ================= 阶段1：请求并获取 JSON 响应 =================
    stage_json_start = time.perf_counter()
    resp = None
    for attempt in range(max_retries + 1):
        try:
            if callable(cancel_check) and cancel_check():
                logger.info("收到外部取消请求，停止 AIGC2D 重试（阶段1-请求前）。")
                _log("[生成/api] 收到取消请求，停止重试。")
                if return_metadata:
                    return {
                        "saved_files": [],
                        "annotation": {},
                        "raw_text": "cancelled",
                        "server_response_raw": {"error": "cancelled_before_request"},
                    }
                return []
            if attempt > 0:
                _log(f"[生成/api] 第 {attempt} 次重试...")
            
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout_val)
            resp.raise_for_status()
            _log(f"[生成/api] 响应: {resp.status_code}, 耗时 {time.perf_counter() - stage_json_start:.1f}s")
            break  
        except requests.exceptions.RequestException as e:
            _log(f"[生成/api] 请求异常 ({attempt + 1}/{max_retries + 1}): {e}")
            if callable(cancel_check) and cancel_check():
                logger.info("收到外部取消请求，停止 AIGC2D 重试。")
                _log("[生成/api] 收到取消请求，停止重试。")
                if return_metadata:
                    return {
                        "saved_files": [],
                        "annotation": {},
                        "raw_text": str(e),
                        "server_response_raw": {"error": "cancelled", "raw_text": str(e)[:2000]},
                    }
                return []
            if attempt < max_retries:
                time.sleep(2)
            else:
                logger.error("达到最大重试次数，AIGC2D 图片生成请求最终失败。")
                _log(f"[生成/api] 请求最终失败: {e}")
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
                if return_metadata:
                    raw_text = _response_text_utf8(resp) if resp is not None else str(e)
                    replay_info = _generate_request_replay(fail_dir, file_prefix, "aigc2d", url, headers, payload)
                    return {
                        "saved_files": [],
                        "annotation": {},
                        "raw_text": raw_text,
                        "server_response_raw": {"error": str(e), "note": "retry_exhausted", "raw_text": raw_text[:2000]},
                        "replay_script": replay_info.get("replay_script", ""),
                        "replay_json": replay_info.get("replay_json", "")
                    }
                return []

    # 解析返回 JSON
    try:
        if getattr(resp, "encoding", None) is None:
            resp.encoding = "utf-8"
        resp_text_raw = _response_text_utf8(resp)
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
        _save_server_response_raw(fail_dir, file_prefix, "aigc2d", resp_text_raw)
        if return_metadata:
            replay_info = _generate_request_replay(fail_dir, file_prefix, "aigc2d", url, headers, payload)
            return {
                "saved_files": [],
                "annotation": {},
                "raw_text": resp_text_raw,
                "server_response_raw": resp_json if 'resp_json' in locals() else {"parse_error": str(e), "raw_text": resp_text_raw[:2000]},
                "replay_script": replay_info.get("replay_script", ""),
                "replay_json": replay_info.get("replay_json", "")
            }
        return []
    _log_stage_elapsed("阶段1-获取JSON响应", stage_json_start)

    if callable(cancel_check) and cancel_check():
        logger.info("收到外部取消请求，跳过 AIGC2D 图片保存（阶段2-保存前）。")
        _log("[生成/api] 收到取消请求，跳过图片保存。")
        if return_metadata:
            return {
                "saved_files": [],
                "annotation": {},
                "raw_text": "cancelled_after_response",
                "server_response_raw": {"error": "cancelled_before_save"},
            }
        return []

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
        logger.warning(f"=== AIGC2D 服务器完整返回 ===\n{_format_safe_log(resp_json)}")
        _save_server_response_json(save_dir, file_prefix, "aigc2d", resp_json)
        _save_server_response_raw(save_dir, file_prefix, "aigc2d", _response_text_utf8(resp))
        if return_metadata:
            replay_info = _generate_request_replay(save_dir, file_prefix, "aigc2d", url, headers, payload)
            return {
                "saved_files": [],
                "annotation": {},
                "raw_text": _response_text_utf8(resp),
                "server_response_raw": resp_json,
                "replay_script": replay_info.get("replay_script", ""),
                "replay_json": replay_info.get("replay_json", "")
            }
        return []

    model_text_parts = []
    # ================= 阶段2：提取并保存图片 =================
    stage_save_start = time.perf_counter()
    for candidate in candidates:
        if callable(cancel_check) and cancel_check():
            logger.info("收到外部取消请求，停止 AIGC2D 图片提取。")
            _log("[生成/api] 收到取消请求，停止图片提取。")
            break
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
                    # 空片段 / 思考图（Gemini 3 会回 thought 图片）不能当产物保存，
                    # 否则会留下一堆 0 字节 png（历史遗留：tmp_repaint 里几百个空文件）
                    if part.get("thought"):
                        logger.info("[跳过] 思考图片段（thought image），不作为产物")
                        continue
                    raw_b64 = str(inline_data.get("data") or "").strip()
                    image_bytes = base64.b64decode(raw_b64) if raw_b64 else b""
                    if len(image_bytes) < 1024:
                        logger.warning(f"[跳过] 返回片段为空或过小（{len(image_bytes)} 字节），不写文件")
                        continue
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
        logger.warning(f"=== AIGC2D 服务器完整返回(图片提取全部失败) ===\n{_format_safe_log(resp_json)}")
        _save_server_response_json(save_dir, file_prefix, "aigc2d", resp_json)
        _save_server_response_raw(save_dir, file_prefix, "aigc2d", _response_text_utf8(resp))

    raw_text = "\n".join(model_text_parts).strip()
    annotation_data = _normalize_annotation_result(_extract_json_object(raw_text))
    if return_metadata:
        replay_info = _generate_request_replay(save_dir, file_prefix, "aigc2d", url, headers, payload)
        return {
            "saved_files": saved_files,
            "annotation": annotation_data,
            "raw_text": raw_text,
            "server_response_raw": resp_json,
            "replay_script": replay_info.get("replay_script", ""),
            "replay_json": replay_info.get("replay_json", "")
        }
    return saved_files


def generate_image_repaint(source_paths, api_type: str = None, config_path: str = None, model: str = None, resolution: str = None, aspect_ratio: str = None, prompt: str = None, prompt_suffix: str = None, use_detail_suffix: bool = None, repeat: int = None, save_sub_dir: str = None, file_prefix: str = None, cancel_check: callable = None, log_callback=None, return_metadata: bool = False, extra_reference_paths=None) -> list:
    """gpt-image 产物优化：对 `source_paths` 逐张走 Gemini 重绘提线。

    模型 / 分辨率 / 宽高比 / prompt / 输出目录**全部来自 `prompts/gpt-image-optimize/config.json`**
    （本函数只在这些入参显式传入时才覆盖配置，没有任何写死的模型名或尺寸）。

    设计要点（理论见 `docs/gpt-image-optimize/README.md`）：
    - prompt 不写死在本文件：默认从 `prompts/gpt-image-optimize/` 装载，也可由调用方直接传 `prompt`
    - 每张源图独立一次请求，输出前缀按源文件名派生，便于回溯「哪张产物被重绘成哪张」
    - 宽高比默认 `auto`（不下发字段，由模型匹配输入图尺寸），换任意比例源图都不会变形
    - 单张失败不影响其余（返回已成功的产物），失败原因照旧落 `log/<日期>.log`
    - 走 `/v1beta/models/{model}:generateContent`（与 `generate_image_aigc2d` 同一通道）

    返回落盘后的重绘产物路径列表。
    """
    from utils.gpt_image_optimize import (
        ASPECT_RATIO_AUTO,
        DEFAULTS as REPAINT_DEFAULTS,
        build_repaint_prompt,
        load_config,
        plan_output,
        resolve_aspect_ratio,
    )

    conf = load_config()
    # 所有兜底都取自 prompts/gpt-image-optimize/config.json 的同义默认值（单一事实来源）
    resolved_model = str(model or conf.get("model") or REPAINT_DEFAULTS["model"])
    resolved_resolution = str(resolution or conf.get("resolution") or REPAINT_DEFAULTS["resolution"])
    # 宽高比默认 auto：不传字段，由模型按输入图比例输出
    resolved_aspect = resolve_aspect_ratio(
        {"aspect_ratio": aspect_ratio} if aspect_ratio else conf, ASPECT_RATIO_AUTO
    )
    resolved_api_type = str(api_type or conf.get("api_type") or REPAINT_DEFAULTS["api_type"])
    try:
        resolved_repeat = max(1, int(repeat if repeat is not None else conf.get("repeat") or 1))
    except (TypeError, ValueError):
        resolved_repeat = 1

    if prompt is None:
        conf_for_prompt = dict(conf)
        if use_detail_suffix is not None:
            conf_for_prompt["use_detail_suffix"] = use_detail_suffix
        resolved_prompt = build_repaint_prompt(conf_for_prompt)
    else:
        resolved_prompt = str(prompt or "")
    if prompt_suffix is not None:
        resolved_suffix = str(prompt_suffix or "")
    elif use_detail_suffix is False:
        resolved_suffix = ""
    else:
        suffix_relative = str(conf.get("detail_suffix") or "").strip()
        try:
            from utils.gpt_image_optimize import read_prompt_relative
            resolved_suffix = read_prompt_relative(suffix_relative) if suffix_relative else ""
        except Exception:  # noqa: BLE001 - 后缀缺失不影响主 prompt
            resolved_suffix = ""

    valid_sources = [p for p in (source_paths or []) if p and os.path.isfile(p)]
    for path in [p for p in (source_paths or []) if p and not os.path.isfile(p)]:
        logger.warning(f"重绘源图不存在，已跳过: {path}")
    if not valid_sources:
        logger.error("重绘需要至少 1 张已存在的源图。")
        return []

    all_saved = []
    extra_refs = [p for p in (extra_reference_paths or []) if p and os.path.isfile(p)]
    multi = len(valid_sources) > 1
    for index, source in enumerate(valid_sources, start=1):
        plan = plan_output(source, conf)
        sub_dir = save_sub_dir or plan["save_sub_dir"]
        if file_prefix:
            base_prefix = f"{file_prefix}_{index:02d}" if multi else str(file_prefix)
        else:
            base_prefix = f"{plan['file_prefix']}_{index:02d}" if multi else plan["file_prefix"]
        logger.info(f"=== 重绘 {index}/{len(valid_sources)}: {source} -> {resolved_model} @{resolved_resolution}"
                    f"{'（双参考 +%d）' % len(extra_refs) if extra_refs else ''} ===")
        saved = generate_image_aigc2d(
            prompt=resolved_prompt,
            image_paths=[source] + extra_refs,
            model=resolved_model,
            aspect_ratio=resolved_aspect,
            resolution=resolved_resolution,
            api_type=resolved_api_type,
            save_sub_dir=sub_dir,
            file_prefix=base_prefix,
            return_metadata=False,
            log_callback=log_callback,
            cancel_check=cancel_check,
            prompt_suffix=resolved_suffix,
            face_quality_boost=False,
        ) or []
        if not saved:
            logger.warning(f"重绘第 {index} 张未返回图片（源图 {source}），继续下一张。")
            continue
        for extra in range(2, resolved_repeat + 1):
            more = generate_image_aigc2d(
                prompt=resolved_prompt,
                image_paths=[source] + extra_refs,
                model=resolved_model,
                aspect_ratio=resolved_aspect,
                resolution=resolved_resolution,
                api_type=resolved_api_type,
                save_sub_dir=sub_dir,
                file_prefix=f"{base_prefix}_r{extra}",
                return_metadata=False,
                log_callback=log_callback,
                cancel_check=cancel_check,
                prompt_suffix=resolved_suffix,
                face_quality_boost=False,
            ) or []
            saved.extend(more)
        all_saved.extend(saved)
    logger.info(f"=== 重绘完成: 源图 {len(valid_sources)} 张 -> 产物 {len(all_saved)} 张 ===")
    if return_metadata:
        return {
            "saved_files": all_saved,
            "model": resolved_model,
            "resolution": resolved_resolution,
            "aspect_ratio": resolved_aspect,
            "prompt_chars": len(resolved_prompt),
            "sources": valid_sources,
        }
    return all_saved
