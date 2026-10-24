"""Shared GPT image credentials and opt-in post-generation workflow settings."""
import json
import os
from pathlib import Path
import re
import tempfile

CONFIG_PATH = Path(__file__).resolve().parents[1] / "conf" / "config.json"
NODE = "gpt_image_workflow"
KEY_ENV = "IMAGE_MAKER_GPT_IMAGE_API_KEY"
DEFAULTS = {"dedicated_key_enabled": True, "postprocess_enabled": False}


def load_settings(config=None, config_path=None):
    if config is None:
        try:
            config = json.loads(Path(config_path or CONFIG_PATH).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = {}
    node = config.get(NODE, {}) if isinstance(config, dict) else {}
    node = node if isinstance(node, dict) else {}
    result = {k: node.get(k, v) is True for k, v in DEFAULTS.items()}
    result["same_as_gemini"] = node.get("same_as_gemini", not result["dedicated_key_enabled"]) is True
    result["dedicated_key_enabled"] = not result["same_as_gemini"]
    result["base_url"] = str(node.get("base_url") or "").strip()
    try:
        result["timeout_seconds"] = max(30, min(3600, int(node.get("timeout_seconds", 600))))
    except (TypeError, ValueError):
        result["timeout_seconds"] = 600
    return result


def save_settings(changes, config_path=None):
    target = Path(config_path or CONFIG_PATH)
    # Do not overwrite a damaged configuration with an empty replacement.
    config = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    node = dict(config.get(NODE) or {})
    node.update({k: bool(v) for k, v in changes.items() if k in DEFAULTS})
    if "same_as_gemini" in changes:
        node["same_as_gemini"] = bool(changes["same_as_gemini"])
        node["dedicated_key_enabled"] = not node["same_as_gemini"]
    if "base_url" in changes:
        node["base_url"] = str(changes["base_url"] or "").strip()
    if "timeout_seconds" in changes:
        node["timeout_seconds"] = max(30, min(3600, int(changes["timeout_seconds"])))
    config[NODE] = node
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, ensure_ascii=False, indent=4), encoding="utf-8")
    return load_settings(config)


def postprocess_enabled(config_path=None):
    return load_settings(config_path=config_path)["postprocess_enabled"]


def generation_timeout_budget(api_config, *, mode="generate", has_images=False):
    """GUI must outlast all allowed image POSTs, then allow local delivery."""
    from modules.others.api_backend import resolve_image_max_retries
    seconds = max(1, int(api_config.get("timeout") or 600))
    upstream = max(0, int(api_config.get("upstream_error_retries", 2) or 2))
    requests = 1 + resolve_image_max_retries(api_config, 1) + upstream
    return seconds * requests + 30


def dedicated_key():
    # api_backend loads .env once on import; GUI saves update this process directly.
    return str(os.environ.get(KEY_ENV) or "").strip()


def apply_dedicated_key(cfg, settings):
    cfg = dict(cfg)
    if settings["dedicated_key_enabled"]:
        cfg["_gpt_dedicated_key"] = True
        cfg["api_key"] = dedicated_key()
        cfg["_api_key_source"] = "env:" + KEY_ENV if cfg["api_key"] else "none"
        if settings.get("base_url"):
            cfg["base_url"] = settings["base_url"]
    return cfg


def save_dedicated_key(value, env_path=None):
    """Merge one secret into the ignored .env; never put it in JSON settings."""
    from utils.env_loader import default_env_path
    value = str(value or "").strip()
    if not value or any(c in value for c in "\r\n\x00"):
        raise ValueError("请输入有效的 GPT 专用 key")
    target = Path(env_path or default_env_path())
    text = target.read_text(encoding="utf-8") if target.exists() else ""
    pattern = re.compile(r"^\s*(?:export\s+)?" + re.escape(KEY_ENV) + r"\s*=")
    lines = [line for line in text.splitlines() if not pattern.match(line)]
    lines.append(KEY_ENV + "=" + value)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False)
    try:
        with handle:
            handle.write("\n".join(lines) + "\n")
        os.replace(handle.name, target)
    finally:
        if os.path.exists(handle.name):
            os.unlink(handle.name)
    os.environ[KEY_ENV] = value
