# -*- coding: utf-8 -*-
"""重绘后的角色一致性审计与定点修订。"""
from __future__ import annotations

import json
import os
import re
import time

from utils.analysis_gpt_prompt import call_text_model, load_text_api_config

def _prompt(name: str) -> str:
    from utils.prompt_loader import read_prompt_file
    return read_prompt_file("gpt-image-optimize/" + name).strip()


def expected_character_text(analysis_result: dict, max_chars: int = 6500) -> str:
    data = analysis_result or {}
    text = str(data.get("gpt_image_prompt") or data.get("english_description") or
               data.get("original_english_description") or "").strip()
    return text[:max_chars]


def _json_object(text: str) -> dict:
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I).strip()
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except Exception:  # noqa: BLE001
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            return {}
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else {}


def normalize_audit(value: dict) -> dict:
    data = value if isinstance(value, dict) else {}
    differences = []
    for item in data.get("differences") or []:
        if not isinstance(item, dict):
            continue
        feature = re.sub(r"[^a-z0-9_]+", "_", str(item.get("feature") or "").lower()).strip("_")
        expected = str(item.get("expected") or "").strip()
        observed = str(item.get("observed") or "").strip()
        correction = str(item.get("correction") or "").strip()
        confidence = max(0.0, min(1.0, float(item.get("confidence") or 0.0)))
        if feature and expected and observed and confidence >= 0.65:
            differences.append({"feature": feature, "expected": expected, "observed": observed,
                                "correction": correction, "confidence": confidence})
    mismatch = bool(data.get("mismatch")) and bool(differences)
    severity = str(data.get("severity") or ("major" if mismatch else "none")).lower()
    if severity not in {"none", "minor", "major"}:
        severity = "minor" if mismatch else "none"
    anchors = [str(v).strip() for v in (data.get("stable_anchors") or []) if str(v).strip()]
    return {"mismatch": mismatch, "severity": severity,
            "confidence": max(0.0, min(1.0, float(data.get("confidence") or 0.0))),
            "stable_anchors": anchors[:20], "differences": differences,
            "summary": str(data.get("summary") or "").strip()}


def audit_image_identity(image_path: str, analysis_result: dict, text_cfg: dict = None,
                         timeout: int = 180, expected_prompt: str = "") -> dict:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)
    expected = str(expected_prompt or "").strip() or expected_character_text(analysis_result)
    if not expected:
        raise ValueError("分析产物缺少角色描述")
    cfg = text_cfg or load_text_api_config()
    user = "ACTUAL GPT FIRST-PASS PROMPT:\n" + expected
    audit_image = image_path
    proxy_path = ""
    try:
        from PIL import Image
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(root_dir, "cache", "temp")
        os.makedirs(cache_dir, exist_ok=True)
        proxy_path = os.path.join(cache_dir, "identity-audit-%d-%d.jpg" % (os.getpid(), int(time.time() * 1000)))
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
            im.save(proxy_path, "JPEG", quality=88, optimize=True)
        audit_image = proxy_path
        raw = call_text_model(cfg["base_url"], cfg["api_key"], cfg["model"],
                              _prompt("identity-audit-system.md"), user,
                              timeout=timeout, max_tokens=1800, image_path=audit_image)
    finally:
        if proxy_path and os.path.isfile(proxy_path):
            try:
                os.remove(proxy_path)
            except OSError:
                pass
    result = normalize_audit(_json_object(raw))
    result.update({"image": os.path.abspath(image_path), "model": cfg.get("model", ""),
                   "expected_character": expected, "raw": raw})
    return result


def build_identity_correction_prompt(audit: dict, iteration: int = 1, max_iterations: int = 2) -> str:
    differences = list((audit or {}).get("differences") or [])
    if not differences:
        return ""
    fixes = []
    for item in differences:
        instruction = str(item.get("correction") or "").strip()
        if not instruction:
            instruction = f"Change {item['feature']} from {item['observed']} to {item['expected']}."
        fixes.append(instruction)
    anchors = [str(v).strip() for v in ((audit or {}).get("stable_anchors") or []) if str(v).strip()]
    anchor_text = "- " + "\n- ".join(anchors) if anchors else "- Preserve every identity-bearing feature not listed below."
    feature_names = " ".join(str(item.get("feature") or "").lower() for item in differences)
    structural_terms = ("skirt", "dress", "neckline", "sleeve", "outfit", "garment", "coat", "cape",
                        "shoe", "boot", "stocking", "glove", "hair_length", "hairstyle")
    structural = any(any(term in str(item.get("feature") or "").lower() for term in structural_terms)
                     for item in differences)
    structural_exception = ""
    if structural:
        structural_exception = (
            "STRUCTURAL EXCEPTION FOR THE LISTED FEATURE: the existing geometry of that feature is incorrect and "
            "is not locked. Remove or reshape only the conflicting garment/hair geometry, including obsolete "
            "trailing fabric or length when listed. Reconstruct the small newly exposed area from adjacent image "
            "evidence so no ghost edge, duplicate contour or cut-out seam remains. This exception applies only to "
            "the listed feature; all other structure stays locked.")
    if "skirt" in feature_names and any(word in feature_names for word in ("length", "design", "shape")):
        structural_exception += (
            " SKIRT-LENGTH GEOMETRY: interpret 'short skirt' literally: the visible primary hem must terminate "
            "around the upper thighs while seated. Delete every incorrect skirt panel and train continuing below "
            "that hem; do not retain it as back drapery, an underskirt, a separate train or fabric behind the legs. "
            "Keep both existing legs unobstructed and unchanged. Fill the removed area with a seamless continuation "
            "of the already visible chair, floor and background using only adjacent image evidence.")
    return (_prompt("identity-correction.md")
            .replace("{iteration}", str(max(1, int(iteration))))
            .replace("{max_iterations}", str(max(1, int(max_iterations))))
            .replace("{identity_anchors}", anchor_text)
            .replace("{structural_exception}", structural_exception)
            .replace("{corrections}", "- " + "\n- ".join(fixes)))


def should_correct(audit: dict, min_confidence: float = 0.78) -> bool:
    if not (audit or {}).get("mismatch"):
        return False
    return any(float(item.get("confidence") or 0.0) >= min_confidence
               for item in (audit.get("differences") or []))


def identity_gate_action(audit: dict, max_local_differences: int = 2) -> str:
    """返回 accept/correct/review；检测到差异时只做二次定点修订，不回退首图。"""
    data = audit or {}
    if data.get("audit_error") or str(data.get("severity") or "") == "unknown":
        return "review"
    if not data.get("mismatch"):
        return "accept"
    differences = [d for d in (data.get("differences") or [])
                   if float(d.get("confidence") or 0.0) >= 0.78]
    return "correct" if differences else "review"
