# -*- coding: utf-8 -*-
"""文本模型调用与「分析产物 → gpt-image 专用短提示词」的公共实现。

为什么要独立字段：分析产物默认是给 Gemini 通道用的（11k 画风 + 3.8~4.4k 长描述 ≈ 15k 字符）。
gpt-image 通道在这个量级会断连（实测 ConnectionResetError），而且长文本会把「画风参考图」挤掉
（见 docs/gpt-image-tid-style/README.md §4.5 / §4.16）。所以针对 gpt-image 单独出一份短描述字段。

- `call_text_model` / `normalize_chat_base` / `load_text_api_config`：文本通道（OpenAI 兼容）调用
- `build_gpt_image_prompt`：把长描述压成 ≤`FIELD_MAX_CHARS` 字符、保留构图/姿势/服装/道具的短提示词
- `compose_gpt_image_prompt`：画风短版（`prompt_gpt`）+ 该字段的拼装与长度护栏
"""
import json
import os
import time

import requests

from utils.env_loader import ensure_env_loaded  # noqa: F401  (触发 .env 装载)
from utils.prompt_loader import read_prompt_file

FIELD_KEY = "gpt_image_prompt"
FIELD_MAX_CHARS = 1400
# 第二档：短内容锚（配画风参考图用，短文本才压得住参考图）
SHORT_FIELD_KEY = "gpt_image_prompt_short"
SHORT_FIELD_MAX_CHARS = 500
# 画风短版(≤680) + 内容字段(≤1400) 正常不超过 ~2100；超过这个数画风参考图基本失效、且请求有断连风险
COMPOSED_WARN_CHARS = 2500
SYSTEM_PROMPT_FILE = "analysis-gpt-prompt-system.md"
SHORT_SYSTEM_PROMPT_FILE = "analysis-gpt-prompt-short-system.md"
USER_PROMPT_FILE = "analysis-gpt-prompt-user.md"
# 挂画风参考图时必须显式排除参考图角色特征：gpt-image 会把参考图当"要编辑的原图"，
# 短提示词下它会连发色/瞳色/角色设计一起搬过来（实测 W4 产物变成参考图的粉发金瞳）。
STYLE_REF_EXCLUSION = (
    "Use the attached reference image for its palette, lighting, brushwork and rendering grammar only. "
    "Do not copy its character, face, hairstyle, hair colour, eye colour, outfit, pose, props or composition; "
    "keep the subject, colours of the subject's own design and the scene exactly as described above."
    "\n\nCHARACTER-INTRINSIC FEATURES ARE NOT PART OF THE STYLE: keep the subject's OWN hair colour, eye colour, skin tone, body proportions and outfit colours exactly as described by the content; the style reference must only change the rendering language (palette temperature of the environment, brushwork, line character, edge treatment, texture), never the subject's intrinsic colouring or design."
)
FALLBACK_BASE_URL = "https://new.aigc2d.com/v1"
DEFAULT_TIMEOUT = 180


def normalize_chat_base(base_url: str) -> str:
    """把 conf 里各种形态的 base_url 归一成可拼 /chat/completions 的地址。"""
    url = str(base_url or "").strip().rstrip("/")
    if not url:
        return ""
    for tail in ("/v1beta/models", "/chat/completions"):
        if url.endswith(tail):
            url = url[: -len(tail)]
            break
    if not url.endswith("/v1"):
        url = url + "/v1"
    return url


def load_text_api_config(config_path: str = None) -> dict:
    """读 conf/config.json 的文本通道配置（顶层 base_url/model + 环境变量里的 key）。"""
    from modules.others.api_backend import resolve_text_api_key

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = config_path or os.path.join(base_dir, "conf", "config.json")
    cfg = {}
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f) or {}
    except Exception:  # noqa: BLE001
        cfg = {}
    return {
        "base_url": str(cfg.get("base_url") or FALLBACK_BASE_URL).strip(),
        "model": str(cfg.get("model") or "gpt-5.6-luna").strip(),
        "api_key": resolve_text_api_key(cfg),
    }


def call_text_model(base_url, api_key, model, system_prompt, user_prompt, timeout=DEFAULT_TIMEOUT,
                    max_tokens=4000, image_path=""):
    """POST /chat/completions（支持推理模型与可选附图）。"""
    url = f"{normalize_chat_base(base_url)}/chat/completions"
    if image_path:
        from utils.style_gpt import to_data_url
        content = [{"type": "text", "text": user_prompt},
                   {"type": "image_url", "image_url": {"url": to_data_url(image_path)}}]
    else:
        content = user_prompt
    payload = {"model": model,
               "messages": [{"role": "system", "content": system_prompt},
                            {"role": "user", "content": content}],
               "max_completion_tokens": max_tokens}
    resp = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                         json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(str(data["error"])[:300])
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("响应没有 choices")
    return str((choices[0].get("message") or {}).get("content") or "").strip()


def sanitize_short_prompt(text: str, max_chars: int = FIELD_MAX_CHARS) -> str:
    """去掉围栏/标签前缀，压空白，按句子边界收到 max_chars。"""
    import re

    t = str(text or "").strip()
    t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t).strip()
    t = re.sub(r"^(short prompt|prompt|gpt-image prompt)\s*[:：]\s*", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_chars:
        cut = t[:max_chars]
        for sep in (". ", "; ", ", "):
            if sep in cut:
                cut = cut[: cut.rfind(sep)]
                break
        t = cut.strip().rstrip(",;.")
    return t


def build_gpt_image_prompt(description: str, text_cfg: dict = None, max_chars: int = FIELD_MAX_CHARS,
                           attempts: int = 3, timeout: float = DEFAULT_TIMEOUT, log_callback=None,
                           tier: str = "full") -> str:
    """用文本模型把长描述压成 gpt-image 专用短提示词（构图/姿势/服装/道具/镜头不变）。

    tier="full"  → ≤1400 字符的完整内容锚（不挂参考图、或内容优先时用）
    tier="short" → ≤500 字符的短内容锚（要挂画风参考图时用；文本越短，参考图越能起作用）
    """
    desc = str(description or "").strip()
    if not desc:
        return ""
    cfg = text_cfg or load_text_api_config()
    if not (cfg.get("base_url") and cfg.get("api_key") and cfg.get("model")):
        raise RuntimeError("缺少文本 API 配置（conf/config.json 的 base_url/model + IMAGE_MAKER_TEXT_API_KEY）")
    tier = str(tier or "full").lower()
    if tier == "short":
        system_prompt = read_prompt_file(SHORT_SYSTEM_PROMPT_FILE)
        limit = min(int(max_chars), SHORT_FIELD_MAX_CHARS)
    else:
        system_prompt = read_prompt_file(SYSTEM_PROMPT_FILE)
        limit = int(max_chars)
    user_prompt = read_prompt_file(USER_PROMPT_FILE).replace("{description}", desc)
    last = ""
    for attempt in range(1, max(1, attempts) + 1):
        try:
            raw = call_text_model(cfg["base_url"], cfg["api_key"], cfg["model"], system_prompt, user_prompt,
                                  timeout=timeout)
            out = sanitize_short_prompt(raw, limit)
            if out:
                return out
            last = "模型返回空内容"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
            if log_callback:
                log_callback(f"gpt-image 短提示词生成失败（{tier}，第 {attempt} 次）：{last}")
            time.sleep(2)
    raise RuntimeError(f"gpt-image 短提示词生成失败（{tier}）：{last}")


def compose_gpt_image_prompt(style_prompt_gpt: str, field_text: str, style_ref_exclusion: bool = False) -> str:
    """拼成最终发给 gpt-image 的提示词：画风短版 → 内容字段 →（挂参考图时）参考图角色排除句。"""
    style = str(style_prompt_gpt or "").strip()
    body = str(field_text or "").strip()
    parts = [p for p in (style, body) if p]
    if style_ref_exclusion:
        parts.append(STYLE_REF_EXCLUSION)
    return "\n\n".join(parts)


def resolve_content_field(result: dict, tier: str = "full") -> str:
    """从分析产物里取对应档位的内容字段（缺失时回退到另一档）。"""
    data = result or {}
    if str(tier).lower() == "short":
        return str(data.get(SHORT_FIELD_KEY) or data.get(FIELD_KEY) or "").strip()
    return str(data.get(FIELD_KEY) or data.get(SHORT_FIELD_KEY) or "").strip()


def composed_length_warning(composed: str) -> str:
    """长度护栏：超过 COMPOSED_WARN_CHARS 就提醒（画风参考图会失效、长文本有断连风险）。"""
    n = len(str(composed or ""))
    if n > COMPOSED_WARN_CHARS:
        return (f"提示词 {n} 字符，超过 {COMPOSED_WARN_CHARS}：画风参考图会失效，且长文本请求可能被中转站断连"
                f"（实测 15k 字符会 ConnectionResetError）")
    return ""
