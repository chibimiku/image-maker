# -*- coding: utf-8 -*-
"""三图质量审计：GPT 首图（内容锚）/ 当前重绘 / 画风参考图。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid

from utils.analysis_gpt_prompt import call_text_model, load_text_api_config
from utils.send_budget import guarded_text_call


def _prompt(name: str) -> str:
    from utils.prompt_loader import read_prompt_file
    return read_prompt_file("gpt-image-optimize/" + name).strip()


def file_sha256(path: str) -> str:
    """Content hash of one artefact; used to bind every audit to the pixels it judged.

    Audits that name a file are not proof that the *selected* file was judged:
    a later repair rewrites the selection. Recording this hash in every audit
    result lets a report prove (or disprove) the binding without trusting mtime.
    """
    if not path or not os.path.isfile(path):
        return ""
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


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


def normalize_quality_audit(value: dict, schema: str = "quality") -> dict:
    """规范化质量/人体/终审审计结论。

    第四轮 P0.2：**不允许把缺字段默认成「否 / none」再伪装成有效结论**。
    合法结论必须给出 boolean ``needs_refine``、枚举严重度（none/minor/major），
    每个 ``*_issues``/``background_drift``/``style_gaps`` 字段都必须是 list。
    任一缺失时 ``conclusion_valid=false``、``severity="unknown"``，
    门禁据此只能判 review_required。

    各审计的**真实 schema**不同，必须分开要求（第四轮 P0.2 修订）：
    * ``schema="quality"`` / ``"final_review"``：``needs_refine`` + ``severity`` +
      四个问题集合（对应 `refine-quality-audit-system.md` / `final-quality-audit-system.md`）；
    * ``schema="anatomy"``：``hand-audit-system.md`` 只要求
      ``structural_issues`` / ``character_limb_inventory`` / ``summary`` / ``needs_refine`` /
      ``ownership_uncertain``，**没有 severity，也不要求画风字段** —— 拿画风字段去要求人体审计
      会让每一张图都「缺结论」，那不是审计的问题。
    """
    data = value if isinstance(value, dict) else {}
    schema = str(schema or "quality")
    required_lists = ("structural_issues",) if schema == "anatomy" else (
        "structural_issues", "line_issues", "background_drift", "style_gaps")
    require_severity = schema != "anatomy"
    result = {
        "structural_issues": _items(data, "structural_issues", ("region", "observed", "repair")),
        "line_issues": _items(data, "line_issues", ("region", "observed", "repair")),
        "background_drift": _items(data, "background_drift", ("region", "original", "candidate", "repair")),
        "style_gaps": _items(data, "style_gaps", ("aspect", "candidate", "target", "repair")),
    }
    findings = sum((result[key] for key in ("structural_issues", "line_issues", "background_drift", "style_gaps")), [])
    raw_needs = data.get("needs_refine")
    has_bool_needs = isinstance(raw_needs, bool)
    severity_raw = str(data.get("severity") or "").strip().lower()
    severity_present = bool(severity_raw)
    severity_ok = severity_raw in {"none", "minor", "major"}
    missing_lists = [key for key in required_lists if not isinstance(data.get(key), list)]
    missing = ([] if has_bool_needs else ["needs_refine"])
    if require_severity and not severity_ok:
        missing.append("severity")
    if severity_present and not severity_ok:
        missing.append("severity(非法枚举)")
    missing += missing_lists
    result["needs_refine"] = (bool(raw_needs) if has_bool_needs else False) and bool(findings)
    result["severity"] = severity_raw if severity_ok else ("unknown" if require_severity else "")
    result["severity_required"] = require_severity
    result["confidence"] = max(0.0, min(1.0, float(data.get("confidence") or 0.0)))
    result["protected_features"] = [str(v).strip() for v in (data.get("protected_features") or [])
                                      if str(v).strip()][:20]
    result["summary"] = str(data.get("summary") or "").strip()
    result["schema"] = schema
    result["conclusion_valid"] = not missing
    result["conclusion_missing"] = bool(missing)
    result["missing_conclusion_fields"] = missing
    return result


def _proxy(path: str, prefix: str, crop_box=None) -> str:
    from PIL import Image
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(root, "cache", "temp")
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, f"{prefix}-{os.getpid()}-{uuid.uuid4().hex}.jpg")
    with Image.open(path) as im:
        im = im.convert("RGB")
        if crop_box:
            width, height = im.size
            im = im.crop(tuple(round(v * (width if i % 2 == 0 else height))
                               for i, v in enumerate(crop_box)))
        im.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
        im.save(target, "JPEG", quality=90, optimize=True)
    return target


def _detail_sheet(path: str) -> str:
    from PIL import Image
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(root, "cache", "temp")
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, f"quality-sheet-{os.getpid()}-{uuid.uuid4().hex}.jpg")
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
                         timeout: int = 240, proportion_clauses=None,
                         final_review: bool = False, style_targets: str = "",
                         authorized_changes=None) -> dict:
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
        proportion_target = [str(v).strip() for v in (proportion_clauses or []) if str(v).strip()]
        if proportion_target:
            user += ("\n\nEXPLICIT BODY-PROPORTION TARGETS (these override Image 1 when its anatomy "
                     "violates them):\n- " + "\n- ".join(proportion_target))
        if final_review:
            user += "\n\nSELECTED RENDERING TARGETS:\n" + str(style_targets or "")
            user += "\n\nAUTHORIZED DESIGN VARIATIONS:\n" + "\n".join(authorized_changes or [])
        raw = guarded_text_call("quality-audit", call_text_model, cfg["base_url"], cfg["api_key"],
                                cfg["model"],
                                _prompt("final-quality-audit-system.md" if final_review else "refine-quality-audit-system.md"), user,
                                timeout=timeout, max_tokens=5000 if final_review else 3000, image_paths=proxies)
    finally:
        for path in proxies:
            try:
                os.remove(path)
            except OSError:
                pass
    value = _json_object(raw)
    if not all(isinstance(value.get(key), list) for key in
               ("structural_issues", "line_issues", "background_drift", "style_gaps")):
        raise ValueError("质量审计未返回完整有效的缺陷列表")
    result = normalize_quality_audit(value)
    result["needs_refine"] = any(result[key] for key in
                                ("structural_issues", "line_issues", "background_drift", "style_gaps"))
    result["needs_review"] = bool(value.get("ownership_uncertain"))
    result["ownership_uncertain"] = value.get("ownership_uncertain") or []
    result.update({"original": os.path.abspath(original_path), "candidate": os.path.abspath(candidate_path),
                   "style_reference": os.path.abspath(style_path), "raw": raw})
    # Bind the verdict to the exact pixels reviewed, and say which images were sent.
    result.update({"candidate_sha256": file_sha256(candidate_path),
                   "original_sha256": file_sha256(original_path),
                   "style_reference_sha256": file_sha256(style_path),
                   "audit_images": [os.path.abspath(path) for path in paths],
                   "audit_image_sha256": [file_sha256(path) for path in paths]})
    return result


def write_style_reference_absent_audit(candidate: str, audit_call, output_dir: str) -> dict:
    """没有画风参考图可用时的只读终审：仍然审，但明确记录缺的是哪张图。

    参考图不参与生成，不代表可以跳过终审；这里不做任何修补，只落一条
    ``needs_refine=True`` 的结论，让门禁按「未通过」处理。
    """
    audit = {"needs_refine": True, "severity": "unknown", "audit_error": "",
             "structural_issues": [], "line_issues": [], "background_drift": [], "style_gaps": [],
             "candidate": os.path.abspath(candidate), "candidate_sha256": file_sha256(candidate),
             "summary": "本配置没有可用的画风参考图（style reference），"
                        "终审只完成「有审计记录」这一步，不给出通过结论。",
             "style_reference_absent": True, "audit_only": True}
    # needs_review 让门禁按「未通过」处理：只有 needs_refine 而没有具体缺陷条目时，
    # should_refine_quality 会判成不需要修订，那等于默认放行。
    audit["needs_review"] = True
    audit["ownership_uncertain"] = [{"region": "whole image",
                                     "reason": "没有画风参考图，无法完成画风对照终审"}]
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "final-quality-audit.json"), "w", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)
    return audit


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


QUALITY_FINDING_LABELS = {"structural_issues": "结构", "line_issues": "线条",
                        "background_drift": "背景漂移", "style_gaps": "画风"}


def quality_failure_details(audit: dict) -> str:
    """Human-readable evidence, including lower-confidence retained findings."""
    audit = audit or {}
    lines = [f"严重度={audit.get('severity', 'unknown')}；审计置信度={audit.get('confidence', 0)}"]
    for key, label in QUALITY_FINDING_LABELS.items():
        for item in audit.get(key) or []:
            region = item.get("region") or item.get("aspect") or "未注明区域"
            observed = item.get("observed") or item.get("candidate") or "未说明"
            target = item.get("original") or item.get("target")
            expected = f"；目标：{target}" if target else ""
            lines.append(f"{label} [{region}] {observed}（置信度 {item.get('confidence', 0)}）{expected}；建议：{item.get('repair') or '人工复核'}")
    for item in audit.get("ownership_uncertain") or []:
        lines.append("归属待确认：" + (f"[{item.get('region', '')}] {item.get('reason', '')}"
                                    if isinstance(item, dict) else str(item)))
    if audit.get("audit_error"):
        lines.append("审计错误：" + str(audit["audit_error"]))
    if audit.get("summary"):
        lines.append("摘要：" + str(audit["summary"]))
    return "\n".join(lines)


def final_quality_decision(audit: dict) -> dict:
    """Relax only explicitly minor rendering findings; anatomy stays strict."""
    audit = audit or {}
    if audit.get("audit_error") or audit.get("needs_review") or audit.get("ownership_uncertain"):
        return {"policy": "minor-rendering-v1", "action": "review", "reason": "审计失败或归属不明确"}
    if not should_refine_quality(audit):
        return {"policy": "minor-rendering-v1", "action": "accept", "reason": "无高置信缺陷"}
    hard_findings = any(float(item.get("confidence") or 0) >= .72
                        for key in ("structural_issues", "background_drift")
                        for item in audit.get(key) or [])
    if audit.get("severity") == "minor" and not hard_findings:
        return {"policy": "minor-rendering-v1", "action": "accept_with_warning",
                "reason": "仅轻微线条/画风差异，保留警告并通过；结构与背景门禁未放宽"}
    return {"policy": "minor-rendering-v1", "action": "repair", "reason": "结构、背景或非轻微质量缺陷"}


def record_quality_audit(audit: dict, path: str, log_callback=None, *, final=False) -> dict:
    """Persist original evidence plus a reviewable decision and text sidecar."""
    audit = dict(audit)
    if final:
        audit["gate_decision"] = final_quality_decision(audit)
    details = quality_failure_details(audit)
    report = (("门禁：" + audit["gate_decision"]["reason"] + "\n") if final else "") + details
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)
    with open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8") as stream:
        stream.write(report + "\n")
    if log_callback:
        log_callback(f"[质量审计] {report}\n审计记录：{os.path.abspath(path)}")
    return audit


def match_final_canvas(path, size):
    """Allow tiny Gemini rounding deficits by extending edge pixels, never stretching."""
    from PIL import Image
    with Image.open(path) as source:
        if source.size == size:
            return path
        dx, dy = size[0] - source.width, size[1] - source.height
        if not (0 <= dx <= 8 and 0 <= dy <= 8
                and max(dx / size[0], dy / size[1]) <= .002):
            raise RuntimeError(f"兜底图片尺寸变化 {size} → {source.size}，保留候选图，禁止发布")
        source = source.convert("RGB")
        left, top = dx // 2, dy // 2
        canvas = Image.new("RGB", size)
        canvas.paste(source, (left, top))
        if left:
            canvas.paste(source.crop((0, 0, 1, source.height)).resize((left, source.height)), (0, top))
        if dx - left:
            canvas.paste(source.crop((source.width - 1, 0, source.width, source.height)).resize(
                (dx - left, source.height)), (left + source.width, top))
        if top:
            canvas.paste(canvas.crop((0, top, size[0], top + 1)).resize((size[0], top)), (0, 0))
        if dy - top:
            canvas.paste(canvas.crop((0, top + source.height - 1, size[0], top + source.height)).resize(
                (size[0], dy - top)), (0, top + source.height))
        target = os.path.splitext(path)[0] + "-canvas.png"
        canvas.save(target)
        return target


def review_final_candidate(candidate, audit_call, output_dir, operation=None,
                           cancel_check=None, log_callback=None, audit_only: bool = False):
    """Bounded source-only rescue; cache paid operations before any subsequent audit.

    ``audit_only=True`` is the verifiable no-repair path: the current frozen
    candidate is audited exactly once, the audit JSON is written, and the
    function returns without generating anything. A candidate that would have
    triggered a rescue still comes back unchanged, so the caller must treat the
    audit findings as a gate failure instead of silently repairing the image.
    """
    from PIL import Image
    from utils.gpt_image_optimize import load_config
    from modules.others.api_backend import generate_image_repaint
    cfg = load_config().get("final_candidate_repair") or {}
    operation = operation or (lambda key, call: call())
    os.makedirs(output_dir, exist_ok=True)
    limit = 0 if audit_only else max(0, min(2, int(cfg.get("max_repairs", 1))))
    task_match = re.match(r"^([0-9a-fA-F]{8})(?=[^0-9a-fA-F]|$)", os.path.basename(candidate))
    repair_prefix = (task_match.group(1) + "-" if task_match else "") + "final-rescue"
    for round_no in range(limit + 1):
        if cancel_check and cancel_check():
            raise RuntimeError("最终兜底已取消")
        current = candidate
        audit = operation(f"final_review-v2-audit-{round_no}", lambda: audit_call(current))
        if not isinstance(audit, dict):
            audit = {"needs_refine": False, "severity": "unknown", "audit_error": "审计未返回结果"}
        audit.setdefault("candidate", os.path.abspath(current))
        audit.setdefault("candidate_sha256", file_sha256(current))
        audit["audit_only"] = bool(audit_only)
        audit_path = os.path.join(output_dir, f"final-quality-audit-{round_no}.json")
        audit = record_quality_audit(audit, audit_path, log_callback, final=True)
        decision = final_quality_decision(audit)
        if audit_only:
            return current, audit
        if decision["action"] == "review":
            raise RuntimeError("最终审计需人工确认，保留候选图：\n" + quality_failure_details(audit) + "\n审计记录：" + audit_path)
        if decision["action"] in {"accept", "accept_with_warning"}:
            return current, audit
        repairs = []
        for key in ("structural_issues", "line_issues", "background_drift", "style_gaps"):
            for item in audit.get(key) or []:
                if float(item.get("confidence") or 0) >= .72:
                    repairs.append(f"[{item.get('region') or item.get('aspect')}] "
                                   f"{item.get('observed') or item.get('candidate')}: {item.get('repair')}")
        prompt = _prompt("final-candidate-repair.md").replace("{repairs}", "\n".join(repairs))
        with open(os.path.join(output_dir, f"final-repair-{round_no + 1}.txt"), "w", encoding="utf-8") as stream:
            stream.write(prompt)
        if round_no == limit:
            raise RuntimeError("最终复审仍有缺陷，已达到兜底次数上限，禁止发布：\n" + quality_failure_details(audit) + "\n审计记录：" + audit_path)
        with Image.open(current) as image:
            size = image.size
        resolution = "4K" if max(size) > 3072 else "2K" if max(size) > 1536 else "1K"
        if log_callback:
            log_callback(f"[最终兜底] {len(repairs)} 项明确缺陷，Gemini 定点修复 {round_no + 1}/{limit}，{resolution}")
        def repaint():
            paths = generate_image_repaint([current], model=cfg.get("model"),
                resolution=resolution, aspect_ratio="auto", repeat=1, prompt=prompt,
                use_detail_suffix=False, save_sub_dir=output_dir,
                file_prefix=f"{repair_prefix}-{round_no + 1}", cancel_check=cancel_check,
                log_callback=log_callback)
            if not paths:
                raise RuntimeError("最终兜底 Gemini 未返回图片")
            return paths
        candidate = operation(f"final_review-v2-repair-{round_no + 1}", repaint)[-1]
        candidate = match_final_canvas(candidate, size)
    raise RuntimeError("最终兜底未完成")


def repair_style_guard(request: dict) -> str:
    """Rendering-only fields; never recycle sampled pose/palette instructions."""
    from utils.style_gpt import parse_fields
    text = str(request.get("style_text") or "")
    if not text:  # Legacy checkpoints begin with the eight style fields.
        text = str(request.get("prompt") or "").split("\n\n", 1)[0]
    fields = parse_fields(text)
    targets = "\n".join(f"{key}: {fields[key]}" for key in
                        ("Lighting", "Brushwork", "Edges", "Texture", "Detail level", "Avoid")
                        if fields.get(key))
    return _prompt("repair-style-guard.md").replace("{style_targets}", targets) if targets else ""


def audit_hand_quality(candidate_path: str, text_cfg: dict | None = None,
                       timeout: int = 240) -> dict:
    """Source-only per-character anatomy check; retain the legacy API name."""
    proxies = []
    try:
        proxies.append(_proxy(candidate_path, "hands"))
        proxies.append(_detail_sheet(candidate_path))
        proxies.append(_proxy(candidate_path, "anatomy-left", (0, .25, .65, 1)))
        proxies.append(_proxy(candidate_path, "anatomy-right", (.35, .25, 1, 1)))
        cfg = text_cfg or load_text_api_config()
        raw = guarded_text_call("anatomy-audit", call_text_model, cfg["base_url"], cfg["api_key"],
                                cfg["model"],
                                _prompt("hand-audit-system.md"), "Trace every character's limbs and inspect all visible hands.",
                                timeout=timeout, max_tokens=4000, image_paths=proxies)
        value = _json_object(raw)
        if not isinstance(value.get("structural_issues"), list):
            raise ValueError("手部审计未返回有效 structural_issues")
        # 人体审计的真实 schema（hand-audit-system.md）没有 severity，也不含画风字段，
        # 所以按 `schema="anatomy"` 规范化：只要求 needs_refine + structural_issues + 归属结论。
        result = normalize_quality_audit(value, schema="anatomy")
        # A contradictory needs_refine=false must not hide a confident defect.
        result["needs_refine"] = bool(result["structural_issues"])
        result.update(candidate=os.path.abspath(candidate_path), raw=raw,
                      candidate_sha256=file_sha256(candidate_path))
        result["character_limb_inventory"] = value.get("character_limb_inventory", [])
        result["ownership_uncertain"] = value.get("ownership_uncertain") or []
        result["needs_review"] = bool(result["ownership_uncertain"])
        return result
    finally:
        for path in proxies:
            try:
                os.remove(path)
            except OSError:
                pass


def build_hand_correction_prompt(audit: dict) -> str:
    repairs = [f"[{item['region']}] {item['repair']}"
               for item in audit.get("structural_issues", [])
               if float(item.get("confidence") or 0) >= 0.72]
    return _prompt("hand-correction.md").replace("{repairs}", "- " + "\n- ".join(repairs)) if repairs else ""


def should_refine_quality(audit: dict, min_confidence: float = 0.72) -> bool:
    if not (audit or {}).get("needs_refine"):
        return False
    keys = ("structural_issues", "line_issues", "background_drift", "style_gaps")
    return any(float(item.get("confidence") or 0.0) >= min_confidence
               for key in keys for item in ((audit or {}).get(key) or []))
