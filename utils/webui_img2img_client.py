#!/usr/bin/env python3
"""
Stable Diffusion WebUI img2img client.
"""

from __future__ import annotations

import base64
import datetime as dt
import os
import time
from typing import Any, Dict, List, Optional

import requests


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_output_dir(base_output_dir: str) -> str:
    day = dt.datetime.now().strftime("%Y%m%d")
    output_dir = os.path.join(base_output_dir, day, "webui-img2img")
    _ensure_dir(output_dir)
    return output_dir


class WebuiImg2ImgClient:
    def __init__(self, base_url: str = "http://127.0.0.1:7860", timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> Dict[str, Any]:
        # A1111/Forge usually exposes this endpoint.
        url = f"{self.base_url}/sdapi/v1/options"
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return {
            "ok": True,
            "base_url": self.base_url,
            "sd_model_checkpoint": data.get("sd_model_checkpoint", ""),
            "sd_vae": data.get("sd_vae", ""),
        }

    def img2img_image_file(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 28,
        cfg_scale: float = 4.0,
        denoising_strength: float = 0.72,
        num_images: int = 1,
        seed: int = -1,
        width: Optional[int] = None,
        height: Optional[int] = None,
        sampler_name: str = "Euler a",
        scheduler: str = "Automatic",
        sd_model: str = "",
        sd_vae: str = "Automatic",
        extra_payload: Optional[Dict[str, Any]] = None,
        return_base64: bool = False,
        output_dir: str = "data",
    ) -> Dict[str, Any]:
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"输入图片不存在: {image_path}")
        if not prompt.strip():
            raise ValueError("prompt 不能为空")

        with open(image_path, "rb") as f:
            init_image_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload: Dict[str, Any] = {
            "prompt": prompt.strip(),
            "negative_prompt": negative_prompt.strip(),
            "init_images": [init_image_b64],
            "steps": int(steps),
            "cfg_scale": float(cfg_scale),
            "denoising_strength": float(denoising_strength),
            "batch_size": int(num_images),
            "seed": int(seed),
            "sampler_name": sampler_name.strip() or "Euler a",
            "scheduler": scheduler.strip() or "Automatic",
            "override_settings": {},
        }
        if width:
            payload["width"] = int(width)
        if height:
            payload["height"] = int(height)
        if sd_model.strip():
            payload["override_settings"]["sd_model_checkpoint"] = sd_model.strip()
        if sd_vae.strip():
            payload["override_settings"]["sd_vae"] = sd_vae.strip()
        if extra_payload and isinstance(extra_payload, dict):
            for key, value in extra_payload.items():
                if key == "override_settings" and isinstance(value, dict):
                    payload["override_settings"].update(value)
                else:
                    payload[key] = value
        if not payload["override_settings"]:
            payload.pop("override_settings", None)

        url = f"{self.base_url}/sdapi/v1/img2img"
        start = time.time()
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        images_b64 = data.get("images", []) or []

        saved = self._save_images(images_b64, image_path, output_dir)
        if return_base64:
            for i, item in enumerate(saved):
                item["image_base64"] = images_b64[i]
        return {
            "ok": True,
            "elapsed_sec": round(time.time() - start, 3),
            "outputs": saved,
            "info": data.get("info", ""),
            "parameters": data.get("parameters", {}),
        }

    def _save_images(self, images_b64: List[str], source_path: str, output_dir: str) -> List[Dict[str, str]]:
        save_dir = _to_output_dir(output_dir)
        source_name = os.path.splitext(os.path.basename(source_path))[0] or "image"
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        results: List[Dict[str, str]] = []
        for idx, image_b64 in enumerate(images_b64, start=1):
            raw = base64.b64decode(image_b64)
            filename = f"{ts}-{source_name}-{idx}.png"
            file_path = os.path.join(save_dir, filename)
            with open(file_path, "wb") as f:
                f.write(raw)
            results.append({"index": str(idx), "file_path": file_path})
        return results
