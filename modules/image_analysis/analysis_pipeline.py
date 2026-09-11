# -*- coding: utf-8 -*-
"""Headless 单图全链路分析（无 GUI）。

把原先根目录 analyze_fashion.py 的功能下沉为模块：复刻 modules/image_analysis/single_analyzer.py
的 WorkerThread LLM 分析流程（Step 1~5），并支持「保存到原图同目录」的投稿落地格式
（JSON 含 source_image_path/task_hash + prompts TXT，可直接拖入「投稿 Server」）。

只做分析，不产生任何新图。
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
from typing import Callable

from openai import OpenAI

from modules.image_analysis.single_analyzer import (
    calculate_closest_aspect_ratio,
    step_1_analyze_image,
    step_2_refine_description,
    step_3_check_outfit_consistency,
    step_4_remove_photo_style,
    step_5_recompute_pixiv_tags,
)
from utils.booru_tags import normalize_booru_tags
from utils.pixiv_tag_matcher import get_local_pixiv_tag_candidates
from utils.wd14_tagger import predict_local_booru_tags

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "conf", "config.json")

LogCallback = Callable[[str], None] | None


def _log(log_callback: LogCallback, msg: str) -> None:
    if log_callback:
        log_callback(msg)
    else:
        print(msg, flush=True)


def load_config() -> dict:
    """读取 conf/config.json（文本分析与通用开关配置）。"""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_image_step1(image_path: str, api_type: str = "aigc2d", timeout_seconds: int = 240,
                        log_callback: LogCallback = None) -> dict | None:
    """出图后单图 Step 1 分析（不写文件），返回分析结果或 None。

    批量生图后自动分析用（原 data/fashion_pipeline_sexy.py / analyze_backless.py 的
    analyze_and_save 逻辑），API 配置从 conf/config-image.json 的指定 api 取。
    """
    from modules.others.api_backend import get_api_config

    cfg = get_api_config(api_type=api_type)
    base_url = str(cfg.get("base_url", "") or "").strip()
    api_key = cfg.get("api_key", "")
    if "/v1beta/models/" in base_url:
        base_url = base_url.split("/v1beta/models/")[0] + "/v1"
    model = "gemini-2.5-flash"

    if not api_key or not base_url:
        _log(log_callback, "⚠️ 跳过分析：API 配置缺失（api_key/base_url）")
        return None

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)
    return step_1_analyze_image(
        image_path, client, model,
        log_callback=(lambda m: _log(log_callback, m)),
        timeout_seconds=timeout_seconds,
    )


def save_step1_analysis(result: dict | None, image_path: str,
                        log_callback: LogCallback = None) -> str | None:
    """把 Step 1 结果保存为 <原图basename>_analysis.json（与原批量脚本一致），返回路径。"""
    if not result:
        return None
    json_path = os.path.join(
        os.path.dirname(os.path.abspath(image_path)),
        f"{os.path.splitext(os.path.basename(image_path))[0]}_analysis.json",
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    _log(log_callback, f"📁 分析结果已保存: {json_path}")
    return json_path


def analyze_single_image(image_path: str, config: dict, timeout_seconds: int = 300,
                         log_callback: LogCallback = None) -> dict | None:
    """对单图执行完整 LLM 分析链（Step 1~5），返回分析结果 JSON，失败返回 None。"""
    base_url = str(config.get("base_url", "")).strip()
    api_key = str(config.get("api_key", "")).strip()
    model_name = str(config.get("model", "")).strip()
    booru_tag_limit = int(config.get("booru_tag_limit", 30))
    enable_outfit_check = bool(config.get("enable_outfit_check_single", False))
    remove_photo_style = bool(config.get("remove_photo_style_single", False))
    outfit_style_override = str(config.get("outfit_style_override_single", "") or "").strip()
    enable_recompute_pixiv_tags = enable_outfit_check or remove_photo_style

    if not api_key or not model_name:
        raise RuntimeError("API Key 或模型名为空，请检查 conf/config.json")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)

    _log(log_callback, "== 本地 booru tagger ==")
    local_booru_tags = predict_local_booru_tags(
        image_path, booru_tag_limit=booru_tag_limit,
        log_callback=(lambda m: _log(log_callback, m)),
    )
    _log(log_callback, "== 本地 Pixiv 标签候选 ==")
    pixiv_candidates = get_local_pixiv_tag_candidates(
        local_booru_tags, log_callback=(lambda m: _log(log_callback, m)),
    )

    _log(log_callback, f"== Step 1: Vision 分析 (模型 {model_name}) ==")
    initial_result = step_1_analyze_image(
        image_path,
        client,
        model_name,
        log_callback=(lambda m: _log(log_callback, m)),
        booru_tag_limit=booru_tag_limit,
        local_booru_tags=local_booru_tags,
        pixiv_candidates=pixiv_candidates,
        timeout_seconds=timeout_seconds,
    )
    if not initial_result:
        _log(log_callback, "❌ Step 1 失败，流程终止。")
        return None

    original_pixiv_tags = list(initial_result.get("pixiv_tags", []) or [])
    original_booru_tags = list(initial_result.get("booru-tags", []) or [])

    _log(log_callback, "== Step 2: refine 二次加工 ==")
    final_result = step_2_refine_description(
        initial_result,
        client,
        model_name,
        booru_tag_limit=booru_tag_limit,
        timeout_seconds=timeout_seconds,
    )
    if not final_result:
        _log(log_callback, "❌ Step 2 失败，流程终止。")
        return None
    final_result["aspect_ratio"] = calculate_closest_aspect_ratio(image_path)
    initial_tags = original_pixiv_tags
    refined_tags = final_result.get("pixiv_tags", [])
    final_result["pixiv_tags_first"] = initial_tags
    final_result["pixiv_tags_second"] = refined_tags
    final_result["pixiv_tags"] = initial_tags if initial_tags else refined_tags
    if original_booru_tags:
        final_result["booru-tags"] = original_booru_tags
    if local_booru_tags:
        final_result["booru_tags_local_candidate"] = normalize_booru_tags(
            local_booru_tags, limit=booru_tag_limit, output_style="space"
        )

    if final_result and enable_outfit_check:
        _log(log_callback, f"== Step 3: 服装搭配检查 (override: {outfit_style_override or '无'}) ==")
        final_result = step_3_check_outfit_consistency(
            final_result,
            client,
            model_name,
            timeout_seconds=timeout_seconds,
            outfit_style_override=outfit_style_override,
        )
        if final_result is None:
            _log(log_callback, "⚠️ Step 3 执行失败，已保留 Step 2 结果。")

    if final_result and remove_photo_style:
        _log(log_callback, "== Step 4: 去除照片风格 ==")
        final_result = step_4_remove_photo_style(
            final_result, client, model_name, timeout_seconds=timeout_seconds
        )
        if final_result is None:
            _log(log_callback, "⚠️ Step 4 执行失败，已保留之前的 prompts 结果。")

    if final_result and enable_recompute_pixiv_tags:
        _log(log_callback, "== Step 5: 重新计算 pixiv_tags ==")
        final_result = step_5_recompute_pixiv_tags(
            final_result, client, model_name, timeout_seconds=timeout_seconds
        )
        if final_result is None:
            _log(log_callback, "⚠️ Step 5 执行失败，已保留之前的 pixiv_tags 结果。")

    # 最终标签一律以原图（Step 1）为准
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

    return final_result


def safe_title_from(result_json: dict) -> str:
    jp_title = result_json.get("japanese_title", "未命名")
    return re.sub(r'[\\/*?:"<>|]', "", jp_title).strip() or "未命名"


def ensure_unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    idx = 1
    while True:
        candidate = f"{stem}_{idx}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def save_result_to_source(result_json: dict, image_path: str,
                          log_callback: LogCallback = None) -> str:
    """把分析结果保存到原图同目录（投稿 Server 兼容格式），返回 JSON 路径。"""
    result_json = dict(result_json)
    result_json["source_image_path"] = os.path.abspath(image_path)
    task_hash = hashlib.md5(
        (os.path.abspath(image_path) + str(datetime.datetime.now().isoformat())).encode("utf-8")
    ).hexdigest()[:8]
    result_json["task_hash"] = task_hash

    safe_title = safe_title_from(result_json)
    now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.dirname(os.path.abspath(image_path))
    source_basename = os.path.basename(image_path)
    source_key = os.path.splitext(source_basename)[0].split("_")[0]
    base_filename = f"{now_str}-{source_key}-{safe_title}"

    json_path = ensure_unique_path(os.path.join(save_dir, f"{base_filename}.json"))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=4)
    _log(log_callback, f"📁 JSON 结果已保存至: {json_path}")

    # 提示词 TXT（画幅 + 画风 + 描述）
    raw_ar = result_json.get("aspect_ratio", "2:3")
    style_part = f"--ar {raw_ar}".strip()
    face_suffix = "detailed face, clear facial features, sharp focus on face"
    refine_desc = result_json.get("english_description", "")
    orig_desc = result_json.get("original_english_description", "")
    refine_desc_with_face = f"{refine_desc}, {face_suffix}" if refine_desc else refine_desc
    orig_desc_with_face = f"{orig_desc}, {face_suffix}" if orig_desc else orig_desc
    final_prompt = f"{style_part}\n\n{refine_desc_with_face}".strip()
    orig_prompt = f"{style_part}\n\n{orig_desc_with_face}".strip()

    txt_path = ensure_unique_path(os.path.join(save_dir, f"{base_filename}-prompts.txt"))
    orig_txt_path = ensure_unique_path(os.path.join(save_dir, f"{base_filename}-original-prompts.txt"))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(final_prompt)
    with open(orig_txt_path, "w", encoding="utf-8") as f:
        f.write(orig_prompt)
    _log(log_callback, f"📄 提示词已保存: {os.path.basename(txt_path)} | {os.path.basename(orig_txt_path)}")

    return json_path


def find_images(directory: str) -> list[str]:
    supported = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    files = []
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith(supported):
            files.append(os.path.join(directory, name))
    return files
