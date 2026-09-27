"""One substantive, source-faithful compliance revision after an explicit block."""
import json
from pathlib import Path


def is_moderation_block(result):
    if not isinstance(result, dict):
        return False
    body = result.get("server_response_raw") or {}
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return False
    return error.get("code") in {"moderation_blocked", "content_policy_violation"}


def review_request(prompt, analysis_result, image_paths):
    from utils import analysis_gpt_prompt as text_api
    cfg = text_api.load_text_api_config()
    from utils.prompt_loader import read_prompt_file
    system = read_prompt_file("first-image-safe-review.md")
    source = str((analysis_result or {}).get("source_image_path") or "")
    refs = ([source] if source and Path(source).is_file() else []) + list(image_paths or [])
    raw = text_api.call_text_model(cfg["base_url"], cfg["api_key"], cfg["model"], system,
        json.dumps({"request": prompt, "source_analysis": analysis_result or {},
                    "image_roles": "first is source photo if available; others are style references"}, ensure_ascii=False),
        image_paths=list(dict.fromkeys(refs)), max_tokens=4500)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    plan = json.loads(raw)
    if not isinstance(plan, dict):
        raise ValueError("安全复核必须返回 JSON 对象")
    if plan.get("retry_allowed") is True:
        if plan.get("reference_policy") not in {"none", "keep"}:
            raise ValueError("安全替代必须明确参考图保留策略")
        if (not isinstance(plan.get("changes"), list) or
                not isinstance(plan.get("safe_rendering_clauses"), list) or
                not isinstance(plan.get("prompt"), str)):
            raise ValueError("安全替代的修改说明、渲染条款或提示词格式无效")
    plan["version"] = 2
    return plan


def apply_safe_plan(payload, steps, plan):
    """Remove old style instructions throughout subsequent stages for alternatives."""
    if not plan or plan.get("retry_allowed") is not True:
        return
    payload["prompt"] = plan["prompt"]
    payload["safe_alternative"] = True
    # The new full prompt supersedes old clothing overrides and render targets.
    for key in ("generation_clauses", "identity_correction_clauses", "post_adjustment",
                "style_prompt", "prompt_gpt", "style_text"):
        payload.pop(key, None)
    clauses = [str(c) for c in (plan.get("safe_rendering_clauses") or []) if isinstance(c, str)]
    payload["clauses"] = clauses
    if plan.get("reference_policy", "none") == "none":
        payload["image_paths"] = []
        payload["style_ref_path"] = ""
        payload["repaint_reference_mode"] = "none"
        payload["face_hair_refine"] = False
        if "repaint" in steps:
            steps["repaint"]["reference_mode"] = "none"
        if "tone" in steps:
            steps["tone"]["enabled"] = False


def generate_first_image(generate, *, prompt, analysis_result=None, checkpoint=None,
                         log_callback=None, plan_callback=None, **kwargs):
    """Return (files, effective prompt); never retry vague errors or switch routes."""
    def log(message):
        if log_callback:
            log_callback(message)
    if kwargs.get("cancel_check") and kwargs["cancel_check"]():
        return [], prompt
    # Successful paid retries are reusable even if subsequent stages failed.
    state = checkpoint.data.setdefault("first_safe_review", {}) if checkpoint else {}
    if state.get("outputs") and all(Path(p).is_file() for p in state["outputs"]):
        if plan_callback:
            plan_callback(state["plan"])
        return state["outputs"], state["plan"]["prompt"]
    if not state:
        response = generate(prompt=prompt, return_metadata=True, **kwargs)
        files = response.get("saved_files", []) if isinstance(response, dict) else response or []
        if files or not is_moderation_block(response):
            return files, prompt
        if checkpoint is None:
            import hashlib
            from datetime import datetime
            from utils.generation_checkpoint import GenerationCheckpoint
            from utils.output_isolation import resolve_output_target
            directory = Path("data") / datetime.now().strftime("%Y%m%d") / str(kwargs.get("save_sub_dir") or "")
            directory, _ = resolve_output_target(str(directory), "safe-review")
            key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
            checkpoint = GenerationCheckpoint(str(Path(directory) / ("first-safe-review-" + key + ".json")))
            state = checkpoint.data.setdefault("first_safe_review", {})
        state.update(original_prompt=prompt, blocked=True,
                     original_image_paths=list(kwargs.get("image_paths") or []))
        if checkpoint:
            checkpoint.save()
    if "plan" not in state or (state["plan"].get("version", 1) < 2 and not state.get("retry_attempted")):
        log("首图被审核拦截：请求文字接口复核素材与无依据扩写。")
        state["plan"] = review_request(prompt, analysis_result, kwargs.get("image_paths"))
        if checkpoint:
            checkpoint.save()
    if kwargs.get("cancel_check") and kwargs["cancel_check"]():
        return [], prompt
    plan = state["plan"]
    revised = str(plan.get("prompt") or "").strip()
    if (plan.get("retry_allowed") is not True or not plan.get("changes") or
            not revised or revised == prompt or len(revised) > 15000):
        log("安全复核未提供可提交的实质修改，保留首图失败。")
        return [], prompt
    if state.get("retry_attempted"):
        log("安全版本已提交过一次，停止重复请求。")
        return [], revised
    if plan_callback:
        plan_callback(plan)
    state["retry_attempted"] = True
    if checkpoint:
        checkpoint.save()
    log("安全复核完成，提交一次实质修改后的服装插画请求：" + str(plan.get("reason") or ""))
    retry_kwargs = dict(kwargs)
    if plan.get("reference_policy", "none") == "none":
        retry_kwargs["image_paths"] = []
    response = generate(prompt=revised, return_metadata=True, **retry_kwargs)
    files = response.get("saved_files", []) if isinstance(response, dict) else response or []
    state["outputs"] = list(files)
    state["retry_blocked"] = is_moderation_block(response)
    if checkpoint:
        checkpoint.save()
    return files, revised
