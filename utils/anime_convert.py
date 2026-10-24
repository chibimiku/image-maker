# -*- coding: utf-8 -*-
"""「转化为二次元」：把当前输入的真人照片作为唯一参考，走图片 edit 接口重绘成
日系二次元美少女商业插画（提示词全文固定在 `prompts/anime-convert/edit-prompt.md`）。

通道路由与图片分析页的「生图通道」单选一致：
- `channel="gpt-image"`   → `generate_image_aigc2d_gpt`（带图自动走 `/v1/images/edits`，
  模型/画质/超时读 aigc-2d-gpt 配置，尺寸按输入照片比例收敛到 3 档之一）；
- `channel="gemini"`      → 跟随全局生图配置的 api_type：`aigc2d` 走
  `generate_image_aigc2d`（generateContent + inline_data），其余走 `generate_image_whatai`
  （其内部再按 api_type 分发 openai-image / autodl / openrouter / whatai）。

与画风系统无关：不挂画风参考图、不拼渲染条款，prompt 就是固件全文（逐字发送）。
Gemini 通道显式 `face_quality_boost=False`：正文已含完整面部要求，不再追加英文后缀。
"""
import os

ANIME_CONVERT_PROMPT_FILE = "anime-convert/edit-prompt.md"
ANIME_CONVERT_SAVE_SUB_DIR = "anime-convert"
ANIME_CONVERT_FILE_PREFIX = "anime-convert"
ANIME_CONVERT_CHANNELS = ("gemini", "gpt-image")


def anime_convert_prompt() -> str:
    """固件全文（strip 后逐字发送，不再做任何拼接）。"""
    from utils.prompt_loader import read_prompt_file
    return read_prompt_file(ANIME_CONVERT_PROMPT_FILE).strip()


def normalize_anime_convert_channel(channel: str) -> str:
    key = str(channel or "").strip().lower()
    return "gpt-image" if key in {"gpt-image", "gpt", "gpt_image", "gpt-image-2"} else "gemini"


def build_anime_convert_request(source_path: str, *, channel: str, model_name: str = "",
                                api_type: str = "", resolution: str = None) -> dict:
    """解析一次「转化二次元」的请求参数（只算不发）：比例锁输入照片，产物落独立子目录。"""
    if not source_path or not os.path.isfile(source_path):
        raise FileNotFoundError(f"转化二次元需要一张存在的输入照片: {source_path}")
    from utils.post_process import snapped_aspect_ratio
    channel_key = normalize_anime_convert_channel(channel)
    request = {
        "channel": channel_key,
        "source": os.path.abspath(source_path),
        "prompt": anime_convert_prompt(),
        # 显式下发最接近的官方比例，锁住输出方向/比例 = 输入照片比例
        "aspect_ratio": snapped_aspect_ratio(source_path),
        "save_sub_dir": ANIME_CONVERT_SAVE_SUB_DIR,
        "file_prefix": ANIME_CONVERT_FILE_PREFIX,
    }
    if channel_key == "gpt-image":
        from modules.others.api_backend import pick_gpt_image2_size_for_images
        request["size"] = pick_gpt_image2_size_for_images([source_path])
    else:
        request["model"] = str(model_name or "").strip()
        request["api_type"] = str(api_type or "").strip()
        request["resolution"] = resolution
    return request


def write_anime_convert_record(request: dict, outputs, error: str = "") -> str:
    """在首个产物旁留请求快照（channel/prompt/比例/源图 hash），便于事后复核；无产物时不写。"""
    import json
    from utils.image_identity import file_sha256
    files = [p for p in (outputs or []) if p and os.path.isfile(p)]
    if not files:
        return ""
    record = {
        "kind": "anime-convert",
        "channel": request.get("channel"),
        "model": request.get("model") or "",
        "api_type": request.get("api_type") or "",
        "aspect_ratio": request.get("aspect_ratio") or "",
        "size": request.get("size") or "",
        "source": request.get("source") or "",
        "source_sha256": file_sha256(request.get("source") or "") if request.get("source") and os.path.isfile(request.get("source")) else "",
        "prompt_chars": len(str(request.get("prompt") or "")),
        "outputs": [os.path.abspath(p) for p in files],
    }
    if error:
        record["error"] = error
    target = os.path.abspath(str(files[0]) + ".anime-convert.json")
    with open(target, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return target


def run_anime_convert(source_path: str, *, channel: str, model_name: str = "", api_type: str = "",
                      resolution: str = None, log_callback=None, cancel_check=None) -> list:
    """发送一次「转化二次元」edit 请求，返回落盘产物路径列表（失败/取消返回空列表或抛错）。"""
    log = log_callback or (lambda message: None)
    request = build_anime_convert_request(source_path, channel=channel, model_name=model_name,
                                          api_type=api_type, resolution=resolution)
    prompt = request["prompt"]
    aspect = request["aspect_ratio"]
    common = {"prompt": prompt, "image_paths": [request["source"]],
              "aspect_ratio": aspect, "save_sub_dir": request["save_sub_dir"],
              "file_prefix": request["file_prefix"], "cancel_check": cancel_check,
              "log_callback": log}
    if request["channel"] == "gpt-image":
        from modules.others.api_backend import generate_image_aigc2d_gpt
        log(f"[转化二次元] gpt 通道 edit：比例 {aspect} → 尺寸 {request['size']}（模型/画质读 aigc-2d-gpt 配置）")
        saved = generate_image_aigc2d_gpt(**common) or []
    else:
        resolved_api = str(api_type or "").strip().lower()
        if resolved_api == "aigc2d":
            from modules.others.api_backend import generate_image_aigc2d
            log(f"[转化二次元] Gemini 通道 edit（aigc2d）：模型 {model_name or '配置默认'}，比例 {aspect}")
            saved = generate_image_aigc2d(model=str(model_name or "").strip(), api_type=resolved_api,
                                          resolution=resolution, face_quality_boost=False, **common) or []
        else:
            from modules.others.api_backend import generate_image_whatai
            log(f"[转化二次元] Gemini 通道 edit（{api_type or 'whatai'}）：模型 {model_name or '配置默认'}，比例 {aspect}")
            saved = generate_image_whatai(model=str(model_name or "").strip(),
                                          api_type=str(api_type or "").strip(), **common) or []
    record_path = write_anime_convert_record(request, saved)
    if record_path:
        log(f"[转化二次元] 请求快照: {record_path}")
    return list(saved)
