# -*- coding: utf-8 -*-
"""autodl 站点 MiniMax-H3 视频生成后端.

端点(OpenAI 兼容封装于 autodl):
  创建: POST {base}/v2/video_generation
  查询: GET  {base}/v2/query/video_generation/{task_id}
支持模式:
  text      文生视频(仅 prompt, ratio 必填)
  first     图生视频-首帧(text + image role=first_frame)
  first_last 图生视频-首尾帧(text + first_frame + last_frame)
  reference 多模态参考生视频(text + reference_image(s) [+ reference_audio])
配置: conf/config-video.json
"""
import os
import json
import time
import base64
import mimetypes

import requests

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(MODULE_DIR))
CONF_PATH = os.path.join(PROJECT_ROOT, "conf", "config-video.json")

MODELS = ["MiniMax-H3", "MiniMax-H3-Max"]
RESOLUTIONS = ["480P", "768P", "2K"]
RATIOS = ["adaptive", "16:9", "9:16", "4:3", "3:4", "1:1", "21:9"]
MODES = [
    ("text", "文生视频"),
    ("first", "图生视频-首帧"),
    ("first_last", "图生视频-首尾帧"),
    ("reference", "多模态参考生视频(参考图/音频)"),
]


def load_config(config_path=None):
    path = config_path or CONF_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到视频配置: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_text_config(config=None):
    return config if config is not None else load_config()


def _b64_data_uri(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/png" if str(path).lower().endswith((".png", ".jpg", ".jpeg", ".webp")) else "application/octet-stream"
    return f"data:{mime};base64,{b64}"


def build_content(mode, prompt, first_frame=None, last_frame=None,
                  reference_images=None, reference_videos=None, reference_audio=None):
    """构建 MiniMax H3 content 数组; 首项必须为非空 text."""
    content = [{"type": "text", "text": prompt}]
    if mode == "first":
        content.append({"type": "image_url", "image_url": {"url": _b64_data_uri(first_frame)},
                        "role": "first_frame"})
    elif mode == "first_last":
        content.append({"type": "image_url", "image_url": {"url": _b64_data_uri(first_frame)},
                        "role": "first_frame"})
        content.append({"type": "image_url", "image_url": {"url": _b64_data_uri(last_frame)},
                        "role": "last_frame"})
    elif mode == "reference":
        for p in (reference_images or []):
            content.append({"type": "image_url", "image_url": {"url": _b64_data_uri(p)},
                            "role": "reference_image"})
        for v in (reference_videos or []):
            content.append({"type": "video_url", "video_url": {"url": _b64_data_uri(v)},
                            "role": "reference_video"})
        if reference_audio:
            content.append({"type": "audio_url", "audio_url": {"url": _b64_data_uri(reference_audio)},
                            "role": "reference_audio"})
    return content


def create_task(mode, prompt, first_frame=None, last_frame=None, reference_images=None,
                reference_videos=None, reference_audio=None, model=None, resolution=None,
                duration=None, ratio=None, callback_url=None, aigc_watermark=None, config=None):
    cfg = get_text_config(config)
    url = cfg["base_url"].rstrip("/") + cfg["create_path"]
    headers = {"Authorization": "Bearer " + cfg["api_key"], "Content-Type": "application/json"}
    payload = {
        "model": model or cfg.get("model", "MiniMax-H3"),
        "content": build_content(mode, prompt, first_frame, last_frame, reference_images,
                                 reference_videos, reference_audio),
        "resolution": resolution or cfg.get("resolution", "768P"),
        "duration": int(duration or cfg.get("duration", 8)),
        "ratio": ratio or cfg.get("ratio", "adaptive"),
    }
    if callback_url:
        payload["callback_url"] = callback_url
    if aigc_watermark is not None:
        payload["aigc_watermark"] = bool(aigc_watermark)
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    task_id = data.get("task_id") or data.get("data", {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"创建任务失败, 响应: {json.dumps(data, ensure_ascii=False)[:400]}")
    return task_id, data, payload


def query_task(task_id, config=None):
    cfg = get_text_config(config)
    url = cfg["base_url"].rstrip("/") + cfg["query_path_template"].format(task_id=task_id)
    headers = {"Authorization": "Bearer " + cfg["api_key"]}
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json().get("task", {})


def download(url, out_path, timeout=180):
    resp = requests.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    return out_path


def poll_and_download(task_id, out_path, config=None, progress_cb=None, cancel_check=None):
    cfg = get_text_config(config)
    interval = int(cfg.get("poll_interval_s", 6))
    timeout = int(cfg.get("poll_timeout_s", 900))
    t0 = time.time()
    while True:
        task = query_task(task_id, cfg)
        status = task.get("status")
        if progress_cb:
            progress_cb(status, task, int(time.time() - t0))
        if cancel_check and cancel_check():
            return None, "cancelled"
        if status == "succeeded":
            url = (task.get("content") or {}).get("url")
            if not url:
                return None, "no_url"
            download(url, out_path)
            return out_path, "succeeded"
        if status in ("failed", "cancelled"):
            return None, status
        if time.time() - t0 > timeout:
            return None, "timeout"
        time.sleep(interval)
