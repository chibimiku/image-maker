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
    system = (Path(__file__).resolve().parents[1] / "prompts/first-image-safe-review.md").read_text(encoding="utf-8")
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
    return plan


def generate_first_image(generate, *, prompt, analysis_result=None, checkpoint=None,
                         log_callback=None, **kwargs):
    """Return (files, effective prompt); never retry vague errors or switch routes."""
    def log(message):
        if log_callback:
            log_callback(message)
    if kwargs.get("cancel_check") and kwargs["cancel_check"]():
        return [], prompt
    # Successful paid retries are reusable even if subsequent stages failed.
    state = checkpoint.data.setdefault("first_safe_review", {}) if checkpoint else {}
    if state.get("outputs") and all(Path(p).is_file() for p in state["outputs"]):
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
        state.update(original_prompt=prompt, blocked=True)
        if checkpoint:
            checkpoint.save()
    if "plan" not in state:
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
    state["retry_attempted"] = True
    if checkpoint:
        checkpoint.save()
    log("安全复核完成，提交一次实质修改后的服装插画请求：" + str(plan.get("reason") or ""))
    response = generate(prompt=revised, return_metadata=True, **kwargs)
    files = response.get("saved_files", []) if isinstance(response, dict) else response or []
    state["outputs"] = list(files)
    state["retry_blocked"] = is_moderation_block(response)
    if checkpoint:
        checkpoint.save()
    return files, revised
