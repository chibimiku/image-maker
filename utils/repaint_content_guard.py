"""原图/重绘双图审计与一次局部新增清除；无效或整体替换回退原图。"""
import json
import math
import os

from utils.prompt_loader import read_prompt_file, render_prompt_file

CORE_FIELDS = ("same_artwork", "identity_preserved", "pose_preserved",
               "composition_preserved", "major_objects_preserved", "garment_design_preserved")


def assess_report(report):
    """不完整、互相矛盾或低置信结论不能放行。"""
    if not isinstance(report, dict):
        return "fallback"
    confidence = report.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0.8 <= confidence <= 1:
        return "fallback"
    if any(report.get(k) is not True for k in CORE_FIELDS) or report.get("global_replacement") is not False:
        return "fallback"
    rendering = report.get("rendering")
    if not isinstance(rendering, dict) or rendering.get("source_medium_preserved") is not True or rendering.get("uniform_ink_or_cel_conversion") is not False:
        return "fallback"
    added = report.get("added_elements")
    missing = report.get("missing_or_replaced_elements")
    if not isinstance(added, list) or not isinstance(missing, list) or missing:
        return "fallback"
    if added:
        if report.get("eligible_for_local_cleanup") is not True or report.get("verdict") not in ("pass", "fail"):
            return "fallback"
        for item in added:
            if not isinstance(item, dict) or not all(str(item.get(k) or "").strip() for k in ("element", "location", "source_evidence", "removal_instruction")):
                return "fallback"
            value = item.get("confidence")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.8 <= value <= 1:
                return "fallback"
        return "cleanup"
    return "accept" if report.get("verdict") == "pass" else "fallback"


def audit_pair(before, after, *, cancel_check=None):
    from utils.analysis_gpt_prompt import load_text_api_config, call_text_model
    from utils.first_image_audit import extract_json_object
    from utils.send_budget import guarded_text_call
    if callable(cancel_check) and cancel_check():
        raise RuntimeError("内容检查前已取消")
    cfg = load_text_api_config()
    prompt = read_prompt_file("gpt-image-optimize/repaint-global-content-audit.md")
    raw = guarded_text_call("repaint-content-" + os.path.basename(after), call_text_model,
                            cfg["base_url"], cfg["api_key"], cfg["model"], prompt,
                            "Image 1 is SOURCE, Image 2 is RESULT. Judge the actual two images.",
                            timeout=180, max_tokens=4500, image_paths=[before, after])
    return {"report": extract_json_object(raw), "raw": raw, "model": cfg["model"],
            "images": [before, after], "prompt": prompt}


def _save_record(after, record):
    path = str(after) + "-content-audit.json"
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def guard_repaint(before, after, *, repaint_options=None, log_callback=None, cancel_check=None):
    """最多两次审计、一次清除；失败不覆盖任何产物，交付原图。"""
    log = log_callback or (lambda message: None)
    record = {"source": str(before), "repaint": str(after), "selected": str(before)}
    try:
        audit = audit_pair(before, after, cancel_check=cancel_check)
        from utils.audit_display import summary, details
        log("[重绘内容审计] " + summary(audit.get("report")) + "\n" + details(audit.get("report")))
        record["audit"] = audit
        action = assess_report(audit.get("report"))
        record["action"] = action
        if action == "accept":
            record["selected"] = str(after)
            log("[重绘内容检查] 通过，交付重绘图（修线收益不作为内容通过的条件）。")
        elif action == "cleanup":
            if callable(cancel_check) and cancel_check():
                raise RuntimeError("清除前已取消")
            from modules.others.api_backend import generate_image_repaint
            from utils.gpt_image_optimize import source_model_aspect_ratio
            report = audit["report"]
            findings = json.dumps({"additions": report["added_elements"],
                                   "uncertain_do_not_delete": report.get("uncertain", [])},
                                  ensure_ascii=False, indent=2)
            prompt = render_prompt_file("gpt-image-optimize/repaint-remove-added-elements.md", {"findings": findings})
            options = dict(repaint_options or {})
            options.update(source_paths=[before], extra_reference_paths=[after], prompt=prompt,
                           prompt_suffix="", use_detail_suffix=False, repeat=1, return_metadata=False,
                           content_guard=False, request_timeout=360, retry_limit=0,
                           aspect_ratio=source_model_aspect_ratio(after),
                           file_prefix=os.path.splitext(os.path.basename(after))[0] + "-cleanup",
                           cancel_check=cancel_check, log_callback=log_callback)
            record["cleanup_prompt"] = prompt
            log("[重绘内容检查] 发现明确局部新增元素，清除一次；Image 1=原图，Image 2=重绘底图。")
            # 每个清除用独立预算run，禁止与首轮的run_id冲突。
            from utils import send_budget
            options["budget_run_id"] = (send_budget.current_run_id() + "-cleanup-" + os.path.basename(after)) if send_budget.budget_active() else None
            cleaned = generate_image_repaint(**options) or []
            record["cleanup_outputs"] = cleaned
            if cleaned:
                verified = audit_pair(before, cleaned[0], cancel_check=cancel_check)
                log("[清除复审] " + summary(verified.get("report")) + "\n" + details(verified.get("report")))
                record["cleanup_audit"] = verified
                if assess_report(verified.get("report")) == "accept":
                    record["selected"] = str(cleaned[0])
                    log("[重绘内容检查] 清除后复核通过。")
                else:
                    log("[重绘内容检查] 清除后复核未通过，回退原图；全部产物保留。")
            else:
                log("[重绘内容检查] 清除无产物，回退原图。")
        else:
            log("[重绘内容检查] 内容被替换或结论无法放行，回退原图；不发清除请求。")
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        log(f"[重绘内容检查] 无法完成，回退原图：{record['error']}")
    try:
        _save_record(after, record)
    except OSError as exc:
        log(f"[重绘内容检查] 审计记录保存失败：{exc}")
    return record["selected"]


def source_repaint_prompt():
    """通用源图保真固件；角色、物件由实际输入图决定。"""
    return render_prompt_file("gpt-image-optimize/repaint-source-geometry-v2.md", {
        "source_contract": read_prompt_file("gpt-image-optimize/repaint-source-content-checklist.md"),
        "style_role": read_prompt_file("gpt-image-optimize/repaint-source-geometry-source-role.md"),
    })
