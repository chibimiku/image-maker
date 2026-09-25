# -*- coding: utf-8 -*-
"""三图质量审计：GPT 首图（内容锚）/ 当前重绘 / 画风参考图。"""
from __future__ import annotations

import json
import os
import re
import time

from utils.analysis_gpt_prompt import call_text_model, load_text_api_config


def _prompt(name: str) -> str:
    from utils.prompt_loader import read_prompt_file
    return read_prompt_file("gpt-image-optimize/" + name).strip()


def _json_object(text: str) -> dict:
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text or "").strip(), flags=re.I).strip()
    try:
        value = json.loads(raw)
    except Exception:  # noqa: BLE001
        match = re.search(r"\{.*\}", raw, flags=re.S)
        value = json.loads(match.group(0)) if match else {}
    return value if isinstance(value, dict) else {}


def _items(data: dict, key: str, fields: tuple[str, ...]) -> list[dict]:
    out = []
    for item in data.get(key) or []:
        if not isinstance(item, dict):
            continue
        confidence = max(0.0, min(1.0, float(item.get("confidence") or 0.0)))
        clean = {field: str(item.get(field) or "").strip() for field in fields}
        if confidence >= 0.65 and all(clean.values()):
            clean["confidence"] = confidence
            out.append(clean)
    return out


def normalize_quality_audit(value: dict) -> dict:
    data = value if isinstance(value, dict) else {}
    result = {
        "structural_issues": _items(data, "structural_issues", ("region", "observed", "repair")),
        "line_issues": _items(data, "line_issues", ("region", "observed", "repair")),
        "background_drift": _items(data, "background_drift", ("region", "original", "candidate", "repair")),
        "style_gaps": _items(data, "style_gaps", ("aspect", "candidate", "target", "repair")),
    }
    findings = sum((result[key] for key in ("structural_issues", "line_issues", "background_drift", "style_gaps")), [])
    result["needs_refine"] = bool(data.get("needs_refine")) and bool(findings)
    severity = str(data.get("severity") or ("major" if findings else "none")).lower()
    result["severity"] = severity if severity in {"none", "minor", "major"} else "minor"
    result["confidence"] = max(0.0, min(1.0, float(data.get("confidence") or 0.0)))
    result["protected_features"] = [str(v).strip() for v in (data.get("protected_features") or [])
                                      if str(v).strip()][:20]
    result["summary"] = str(data.get("summary") or "").strip()
    return result


def _proxy(path: str, prefix: str) -> str:
    from PIL import Image
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(root, "cache", "temp")
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, f"{prefix}-{os.getpid()}-{int(time.time() * 1000)}.jpg")
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
        im.save(target, "JPEG", quality=90, optimize=True)
    return target


def _detail_sheet(path: str) -> str:
    from PIL import Image
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(root, "cache", "temp")
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, f"quality-sheet-{os.getpid()}-{int(time.time() * 1000)}.jpg")
    with Image.open(path) as source:
        im = source.convert("RGB")
        width, height = im.size
        tile_size = 512
        sheet = Image.new("RGB", (tile_size * 3, tile_size * 3), "white")
        for row in range(3):
            for col in range(3):
                left, top = round(width * col / 3), round(height * row / 3)
                right, bottom = round(width * (col + 1) / 3), round(height * (row + 1) / 3)
                tile = im.crop((left, top, right, bottom))
                tile.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
                x = col * tile_size + (tile_size - tile.width) // 2
                y = row * tile_size + (tile_size - tile.height) // 2
                sheet.paste(tile, (x, y))
        sheet.save(target, "JPEG", quality=92, optimize=True)
    return target


def audit_refine_quality(original_path: str, candidate_path: str, style_path: str,
                         first_pass_prompt: str = "", text_cfg: dict | None = None,
                         timeout: int = 240) -> dict:
    paths = [original_path, candidate_path, style_path]
    if not all(os.path.isfile(path) for path in paths):
        raise FileNotFoundError("质量审计的三张输入图必须都存在")
    proxies = []
    try:
        proxies = [_proxy(path, f"quality-{i}") for i, path in enumerate(paths, 1)]
        proxies.append(_detail_sheet(candidate_path))
        cfg = text_cfg or load_text_api_config()
        user = ("Images are attached in the exact order defined by the system prompt.\n"
                "ACTUAL GPT FIRST-PASS PROMPT (use only to understand intended content):\n" +
                str(first_pass_prompt or "")[:6500])
        raw = call_text_model(cfg["base_url"], cfg["api_key"], cfg["model"],
                              _prompt("refine-quality-audit-system.md"), user,
                              timeout=timeout, max_tokens=3000, image_paths=proxies)
    finally:
        for path in proxies:
            try:
                os.remove(path)
            except OSError:
                pass
    result = normalize_quality_audit(_json_object(raw))
    result.update({"original": os.path.abspath(original_path), "candidate": os.path.abspath(candidate_path),
                   "style_reference": os.path.abspath(style_path), "raw": raw})
    return result


def build_quality_correction_prompt(audit: dict) -> str:
    repairs = []
    for key in ("structural_issues", "line_issues"):
        for item in (audit or {}).get(key) or []:
            repairs.append(f"[{item.get('region')}] {item.get('repair')}")
    for item in (audit or {}).get("background_drift") or []:
        repairs.append(f"[BACKGROUND: {item.get('region')}] Restore {item.get('original')}. {item.get('repair')}")
    for item in (audit or {}).get("style_gaps") or []:
        repairs.append(f"[STYLE: {item.get('aspect')}] {item.get('repair')}")
    if not repairs:
        return ""
    protected = [str(v).strip() for v in ((audit or {}).get("protected_features") or []) if str(v).strip()]
    return (_prompt("refine-quality-correction.md")
            .replace("{protected_features}", "- " + "\n- ".join(protected) if protected else "- Everything not listed under REPAIRS.")
            .replace("{repairs}", "- " + "\n- ".join(repairs)))


def should_refine_quality(audit: dict, min_confidence: float = 0.72) -> bool:
    if not (audit or {}).get("needs_refine"):
        return False
    keys = ("structural_issues", "line_issues", "background_drift", "style_gaps")
    return any(float(item.get("confidence") or 0.0) >= min_confidence
               for key in keys for item in ((audit or {}).get(key) or []))
