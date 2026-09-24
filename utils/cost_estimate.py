# -*- coding: utf-8 -*-
"""提示词长度护栏 + 单张成本估算（给 GUI 显示，避免用户写出会爆的提示词）。

两件事：
1. **长度预算**：官网上限 32000 字符，但本中转站在 **15000 字符左右就会断连**（实测 15,662 字符的
   组合直接 ConnectionResetError）；而「画风参考图能不能起作用」还另有一条**软上限 ~2000 字符**
   （超过它参考图会被长文本压住，实测相似度 0.323 → 0.107）。所以 UI 要同时显示软/硬两条线。
2. **成本估算**：价格来自中转站公开的 `GET /api/pricing`（模型倍率 + 分组倍率），token 用量取自
   本项目实测（`data/**/*_server_response*.json` 的 usage）。公式是 new-api 的标准口径：
       quota = (文本token + 输入图token×image_ratio) × model_ratio
               + 输出token × completion_ratio × model_ratio
       USD   = quota × 分组倍率 / 500000
   按次计费模型（`quota_type=1`，如 gemini 图片、`gpt-image-2-c`）：USD = model_price × 分组倍率。

实测路由分组（响应头 `X-Routing-Group`）：gpt-image-2 → `Openai-Gpt-1`（倍率 0.88236）；
gemini-3-pro-image-preview → `Discounted-Banana-1`（倍率 0.110295）。
"""
import json
import math
import os
import time

import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRICING_URL = "https://new.aigc2d.com/api/pricing"
CACHE_PATH = os.path.join(BASE_DIR, "cache", "pricing_cache.json")
CACHE_TTL_SECONDS = 12 * 3600
USD_PER_QUOTA = 500000.0

# ---- 提示词长度护栏（字符） ----
PROMPT_SOFT_LIMIT = 2000      # 超过它：画风参考图基本失效（实测）
PROMPT_HARD_LIMIT = 15000     # 超过它：本中转站会断连（实测 15,662 直接 reset）
PROMPT_OFFICIAL_LIMIT = 32000  # 官方文档 maxLength
CHARS_PER_TEXT_TOKEN = 3.2    # 实测：1325 字符 → 417 text token

# ---- 实测路由分组（可由 /api/pricing 的 group_ratio 刷新倍率） ----
ROUTES = {
    "openai_gpt": {"group": "Openai-Gpt-1", "ratio": 0.88236},
    "gemini_image": {"group": "Discounted-Banana-1", "ratio": 0.110295},
}

# ---- 内置兜底价目（网络不可用 / 缓存过期时用），来自 /api/pricing 实测快照 2026-09-22 ----
FALLBACK_PRICING = {
    "gpt-image-2": {"quota_type": 0, "model_ratio": 2.5, "completion_ratio": 6.0, "image_ratio": 1.6,
                    "model_price": 0.0},
    "gpt-image-2.5-flare": {"quota_type": 0, "model_ratio": 2.5, "completion_ratio": 6.0, "image_ratio": 1.6,
                            "model_price": 0.0},
    "gpt-image-2-c": {"quota_type": 1, "model_ratio": 0.0, "completion_ratio": 0.0, "image_ratio": 1.0,
                      "model_price": 0.12},
    "gemini-3-pro-image-preview": {"quota_type": 1, "model_ratio": 0.0, "completion_ratio": 0.0,
                                   "image_ratio": 1.0, "model_price": 0.33},
    "gemini-3.1-flash-image-preview": {"quota_type": 1, "model_ratio": 0.0, "completion_ratio": 0.0,
                                       "image_ratio": 1.0, "model_price": 0.1655},
    "gemini-2.5-flash-image": {"quota_type": 1, "model_ratio": 0.0, "completion_ratio": 0.0,
                               "image_ratio": 1.0, "model_price": 0.15},
    # 文本通道（分析链路的 Step1~5 与字段生成都用 luna）
    "gpt-5.6-luna": {"quota_type": 0, "model_ratio": 0.1, "completion_ratio": 6.0,
                     "image_ratio": 1.0, "model_price": 0.0},
    "gpt-5.6-sol": {"quota_type": 0, "model_ratio": 2.5, "completion_ratio": 6.0,
                    "image_ratio": 1.0, "model_price": 0.0},
}

# ---- 实测 token 用量（1024x1536 单张） ----
OUTPUT_TOKENS_BY_QUALITY = {"low": 343, "medium": 1105, "high": 5500}
REF_IMAGE_INPUT_TOKENS = 1105   # 一张 1024x1536 参考图 ≈ 1105 image token
BASE_PIXELS = 1024 * 1536


# ------------------------------------------------------------------ 价目

def _load_cache():
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def fetch_pricing(force: bool = False, timeout: float = 30):
    """取价目表（模型倍率 + 分组倍率），带本地缓存；失败回退内置快照。"""
    cache = _load_cache()
    fresh = cache.get("fetched_at") and (time.time() - float(cache["fetched_at"]) < CACHE_TTL_SECONDS)
    if cache.get("models") and fresh and not force:
        return cache
    try:
        j = requests.get(PRICING_URL, timeout=timeout).json()
        models = {}
        for m in (j.get("data") or []):
            name = str(m.get("model_name") or "")
            if not name:
                continue
            models[name] = {
                "quota_type": m.get("quota_type"),
                "model_ratio": m.get("model_ratio") or 0.0,
                "completion_ratio": m.get("completion_ratio") or 0.0,
                "image_ratio": m.get("image_ratio") or 1.0,
                "model_price": m.get("model_price") or 0.0,
            }
        cache = {"fetched_at": time.time(), "group_ratio": j.get("group_ratio") or {}, "models": models}
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        return cache
    except Exception:  # noqa: BLE001
        if cache.get("models"):
            return cache
        return {"fetched_at": 0, "group_ratio": {}, "models": dict(FALLBACK_PRICING)}


def model_price_entry(model: str, pricing: dict = None) -> dict:
    pricing = pricing or fetch_pricing()
    entry = (pricing.get("models") or {}).get(model)
    if not entry:
        entry = FALLBACK_PRICING.get(model)
    return entry or {"quota_type": 0, "model_ratio": 1.0, "completion_ratio": 6.0,
                     "image_ratio": 1.0, "model_price": 0.0}


def route_ratio(route: str, pricing: dict = None) -> float:
    """分组倍率：优先用实测路由分组，其次用价目表里的同名倍率。"""
    info = ROUTES.get(route)
    if not info:
        return 1.0
    pricing = pricing or fetch_pricing()
    return float((pricing.get("group_ratio") or {}).get(info["group"], info["ratio"]))


def route_group(route: str) -> str:
    return str((ROUTES.get(route) or {}).get("group") or "")


# ------------------------------------------------------------------ 提示词预算

def text_tokens_for(prompt_chars: int) -> int:
    return int(math.ceil(max(0, int(prompt_chars)) / CHARS_PER_TEXT_TOKEN))


def prompt_budget(prompt_chars: int, style_chars: int = 0) -> dict:
    """给出提示词长度诊断：总计 / 软硬上限 / 还能输入多少。"""
    total = max(0, int(prompt_chars)) + max(0, int(style_chars))
    return {
        "user_chars": max(0, int(prompt_chars)),
        "style_chars": max(0, int(style_chars)),
        "total_chars": total,
        "soft_limit": PROMPT_SOFT_LIMIT,
        "hard_limit": PROMPT_HARD_LIMIT,
        "official_limit": PROMPT_OFFICIAL_LIMIT,
        "remaining_soft": max(0, PROMPT_SOFT_LIMIT - total),
        "remaining_hard": max(0, PROMPT_HARD_LIMIT - total),
        "over_soft": total > PROMPT_SOFT_LIMIT,
        "over_hard": total > PROMPT_HARD_LIMIT,
    }


def format_budget(budget: dict) -> str:
    b = budget
    parts = [f"提示词 {b['total_chars']:,} 字符"]
    if b["style_chars"]:
        parts[0] += f"（画风段 {b['style_chars']:,} + 你的输入 {b['user_chars']:,}）"
    parts.append(f"建议 ≤{b['soft_limit']:,}（画风参考图才生效）")
    parts.append(f"硬上限 {b['hard_limit']:,}（超了中转站会断连）")
    if b["over_hard"]:
        parts.append(f"⚠ 已超硬上限 {b['total_chars'] - b['hard_limit']:,} 字符")
    elif b["over_soft"]:
        parts.append(f"⚠ 已超建议值，还能砍 {b['total_chars'] - b['soft_limit']:,} 字符让参考图生效")
    else:
        parts.append(f"还能输入约 {b['remaining_soft']:,} 字符（按建议）")
    return " ｜ ".join(parts)


# ------------------------------------------------------------------ 成本估算

def _scale_by_size(tokens: int, width: int, height: int) -> int:
    return int(round(tokens * (max(1, width) * max(1, height)) / BASE_PIXELS))


def estimate_gpt_image_cost(prompt_chars: int, size: str = "1024x1536", quality: str = "medium",
                            ref_images: int = 1, model: str = "gpt-image-2", pricing: dict = None) -> dict:
    """gpt-image 出图单张成本（USD）。token 计费走公式，按次计费走 model_price。"""
    pricing = pricing or fetch_pricing()
    entry = model_price_entry(model, pricing)
    ratio = route_ratio("openai_gpt", pricing)
    try:
        w, h = (int(x) for x in str(size).lower().split("x"))
    except Exception:  # noqa: BLE001
        w, h = 1024, 1536
    if entry.get("quota_type") == 1:
        usd = float(entry.get("model_price") or 0.0) * ratio
        return {"model": model, "usd": usd, "billed": "per_call", "group": route_group("openai_gpt"),
                "group_ratio": ratio, "price_per_call": entry.get("model_price")}
    text_tok = text_tokens_for(prompt_chars)
    img_in_tok = _scale_by_size(REF_IMAGE_INPUT_TOKENS, w, h) * max(0, int(ref_images))
    out_tok = _scale_by_size(OUTPUT_TOKENS_BY_QUALITY.get(str(quality).lower(), 1105), w, h)
    model_ratio = float(entry.get("model_ratio") or 2.5)
    completion_ratio = float(entry.get("completion_ratio") or 6.0)
    image_ratio = float(entry.get("image_ratio") or 1.6)
    quota = ((text_tok + img_in_tok * image_ratio) * model_ratio
             + out_tok * completion_ratio * model_ratio)
    usd = quota * ratio / USD_PER_QUOTA
    return {"model": model, "usd": usd, "billed": "token", "group": route_group("openai_gpt"),
            "group_ratio": ratio, "text_tokens": text_tok, "image_input_tokens": img_in_tok,
            "output_tokens": out_tok}


def estimate_gemini_repaint_cost(model: str = "gemini-3-pro-image-preview", repeat: int = 1,
                                 pricing: dict = None) -> dict:
    """Gemini 图片通道单次重绘成本（USD，按次计费）。"""
    pricing = pricing or fetch_pricing()
    entry = model_price_entry(model, pricing)
    ratio = route_ratio("gemini_image", pricing)
    price = float(entry.get("model_price") or 0.33) if entry.get("quota_type") == 1 \
        else float(entry.get("model_ratio") or 0.0) * ratio / 1000.0
    return {"model": model, "usd": price * ratio * max(1, int(repeat)), "billed": "per_call",
            "group": route_group("gemini_image"), "group_ratio": ratio, "price_per_call": price}


def estimate_text_cost(prompt_chars: int, completion_tokens: int = 1500,
                       model: str = "gpt-5.6-luna", pricing: dict = None) -> dict:
    """文本模型单次调用成本（分析链路用）。"""
    pricing = pricing or fetch_pricing()
    entry = model_price_entry(model, pricing)
    ratio = route_ratio("openai_gpt", pricing)
    text_tok = text_tokens_for(prompt_chars)
    model_ratio = float(entry.get("model_ratio") or 0.1)
    completion_ratio = float(entry.get("completion_ratio") or 6.0)
    quota = (text_tok + int(completion_tokens) * completion_ratio) * model_ratio
    return {"model": model, "usd": quota * ratio / USD_PER_QUOTA, "group_ratio": ratio,
            "text_tokens": text_tok, "completion_tokens": int(completion_tokens)}


def estimate_pipeline(prompt_chars: int, size: str = "1024x1536", quality: str = "medium",
                      ref_images: int = 1, gpt_model: str = "gpt-image-2",
                      repaint_model: str = "gemini-3-pro-image-preview",
                      include_repaint: bool = True, include_structure: bool = True,
                      include_local: bool = True, local_model: str = "gemini-3-pro-image-preview",
                      include_analysis: bool = False, cny_rate: float = 7.2,
                      pricing: dict = None) -> dict:
    """整条流水线的单张成本拆分（USD + 人民币折算）。"""
    pricing = pricing or fetch_pricing()
    steps = []
    first = estimate_gpt_image_cost(prompt_chars, size=size, quality=quality, ref_images=ref_images,
                                    model=gpt_model, pricing=pricing)
    steps.append({"key": "generate", "label": f"首图 {gpt_model} @{size}/{quality}", **first})
    if include_repaint:
        rp = estimate_gemini_repaint_cost(repaint_model, pricing=pricing)
        steps.append({"key": "repaint", "label": f"全图重绘 {repaint_model} 2K", **rp})
    if include_structure:
        steps.append({"key": "structure", "label": "结构线叠加（纯本地）", "usd": 0.0, "billed": "local",
                      "group": "-", "group_ratio": 0.0})
    if include_local:
        lp = estimate_gemini_repaint_cost(local_model, pricing=pricing)
        steps.append({"key": "local", "label": f"局部重绘+羽化贴回 {local_model} 2K", **lp})
    if include_analysis:
        analysis_calls = 5 + 2  # Step1~5 + 两档字段生成
        per_call = estimate_text_cost(prompt_chars=2500, completion_tokens=2000, pricing=pricing)
        steps.append({"key": "analysis", "label": f"分析链路（{analysis_calls} 次文本调用）",
                      "usd": per_call["usd"] * analysis_calls, "billed": "token",
                      "group": per_call["group_ratio"], "group_ratio": per_call["group_ratio"],
                      "call_count": analysis_calls, "per_call_usd": per_call["usd"]})
    total = sum(float(s.get("usd") or 0.0) for s in steps)
    return {"steps": steps, "total_usd": total, "total_cny": total * float(cny_rate),
            "cny_rate": float(cny_rate), "prompt_chars": int(prompt_chars)}


def format_pipeline_cost(est: dict, show_steps: bool = True) -> str:
    if show_steps:
        detail = " + ".join(f"{s['label'].split(' ')[0]} ${s['usd']:.4f}" for s in est["steps"])
    else:
        detail = ""
    return (f"单张全工序估算 ≈ ${est['total_usd']:.3f}（≈¥{est['total_cny']:.2f}）"
            + (f" ｜ {detail}" if detail else ""))
