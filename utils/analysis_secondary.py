"""Resolve the configured secondary text channel for post-vision editing."""
from __future__ import annotations

import json
from pathlib import Path


def secondary_text_config(config=None):
    from modules.others.api_backend import apply_secret_env_overrides

    if config is None:
        path = Path(__file__).resolve().parents[1] / "conf" / "config.json"
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
    cfg = apply_secret_env_overrides(dict(config))
    return {
        "base_url": str(cfg.get("nsfw_base_url") or "").strip(),
        "api_key": str(cfg.get("nsfw_api_key") or "").strip(),
        "model": str(cfg.get("nsfw_model") or "").strip(),
    }


def create_secondary_text_client(config, *, client_factory, timeout_seconds):
    cfg = dict(config)
    missing = [key for key in ("base_url", "api_key", "model") if not cfg.get(key)]
    if missing:
        raise ValueError("DeepSeek 次级文本通道配置缺失：" + ", ".join(missing)
                         + "；请检查设置 → NSFW 文本 API。")
    client = client_factory(api_key=cfg["api_key"], base_url=cfg["base_url"],
                            timeout=timeout_seconds)
    return client, cfg["model"]
