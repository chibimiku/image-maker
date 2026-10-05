"""Independent clothing-design presets; no Qt, API calls or art-style fields."""
from __future__ import annotations

import hashlib
import json

from utils.prompt_loader import read_prompt_file, render_prompt_file

POLICIES = (("replace", "完整转化"), ("reinterpret", "改款"), ("fill_missing", "只补全"))


def wardrobe_presets() -> dict:
    data = json.loads(read_prompt_file("wardrobe/presets.json"))
    if data.get("schema_version") != "image-maker.wardrobe.v1":
        raise ValueError("不支持的衣装预设版本")
    return data["presets"]


def build_wardrobe_spec(name: str = "", policy: str = "replace") -> dict:
    """Snapshot effective text at submission; off never loads/changes prompts."""
    name = str(name or "").strip()
    if not name:
        return {}
    if policy not in dict(POLICIES):
        raise ValueError(f"未知衣装应用策略: {policy}")
    presets = wardrobe_presets()
    if name not in presets:
        raise ValueError(f"未知衣装预设: {name}")
    entry = presets[name]
    prompt = render_prompt_file("wardrobe/application.md", {
        "label": entry["label"], "policy": policy,
        "policy_rule": read_prompt_file(f"wardrobe/policy-{policy}.md").strip(),
        "eligibility": read_prompt_file(entry["eligibility_file"]).strip()
                       if entry.get("eligibility_file") else "",
        "design": "\n\n".join(read_prompt_file(path).strip() for path in
                                  entry.get("prompt_files", [entry.get("prompt_file")]) if path),
    }).strip()
    return {"schema_version": "image-maker.wardrobe.v1", "name": name,
            "label": entry["label"], "policy": policy, "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "provenance": entry.get("provenance", "hand_written")}


def validate_wardrobe_style(styles, name, spec):
    """Reject two clothing authorities; keep historic art entries readable."""
    if not spec:
        return
    wardrobe_prompt(spec)
    from utils.styles import is_wardrobe_style
    if is_wardrobe_style(name, (styles or {}).get(name)):
        raise ValueError("该旧预设控制服装，请在穿衣风格中选择；不能与另一衣装目标同时使用。")


def wardrobe_prompt(spec: dict | None) -> str:
    if not spec:
        return ""
    if spec.get("schema_version") != "image-maker.wardrobe.v1":
        raise ValueError("不支持的衣装任务快照版本")
    prompt = str(spec.get("prompt") or "").strip()
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if not prompt or digest != spec.get("prompt_sha256"):
        raise ValueError("衣装任务快照缺失或校验失败")
    return prompt


def apply_wardrobe(prompt: str, spec: dict | None = None) -> str:
    block = wardrobe_prompt(spec)
    if block:
        count = str(prompt or "").count(block)
        if count > 1:
            raise ValueError("衣装规则重复注入")
        if count == 1:
            return prompt
    return str(prompt or "") + "\n\n" + block if block else prompt


def wardrobe_reference_guard(spec):
    if not wardrobe_prompt(spec):
        return ""
    return render_prompt_file("wardrobe/reference-guard.md", {"label": spec["label"]}).strip()


def wardrobe_continuity(spec: dict | None) -> str:
    if not wardrobe_prompt(spec):
        return ""
    return render_prompt_file("wardrobe/continuity.md", {"label": spec["label"]}).strip()


def record_wardrobe_outputs(paths, spec, prompt, *, stage="generation"):
    """Write clothing-only request snapshots beside direct GUI/CLI outputs."""
    if not wardrobe_prompt(spec):
        return
    from pathlib import Path
    from utils.atomic_io import write_json_atomic
    for path in paths or []:
        candidate = Path(path)
        record = {"schema_version": "image-maker.wardrobe-output.v1", "stage": stage,
                  "wardrobe": dict(spec), "prompt": str(prompt or ""),
                  "prompt_sha256": hashlib.sha256(str(prompt or "").encode("utf-8")).hexdigest(),
                  "image_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                  "effect_reviewed": False}
        write_json_atomic(str(candidate.with_suffix(".wardrobe.json")), record, ensure_ascii=False, indent=2)


def wardrobe_audit_key(candidate: str, spec: dict, content: str) -> str:
    from utils.refine_quality import file_sha256
    wardrobe_prompt(spec)
    inputs = [file_sha256(candidate), spec["prompt_sha256"], str(content or ""),
              read_prompt_file("wardrobe/audit.md"), "wardrobe-audit-v1"]
    return hashlib.sha256(json.dumps(inputs, ensure_ascii=False).encode("utf-8")).hexdigest()


def audit_wardrobe(candidate: str, spec: dict, content: str, text_cfg=None) -> dict:
    """Read-only clothing check, bound to the actual candidate pixels."""
    from utils.analysis_gpt_prompt import call_text_model, load_text_api_config
    from utils.send_budget import guarded_text_call
    from utils.refine_quality import _proxy, file_sha256, _json_object
    import os
    block = wardrobe_prompt(spec)
    if not block:
        raise ValueError("衣装审计需要有效目标")
    proxy = _proxy(candidate, "wardrobe")
    try:
        cfg = text_cfg or load_text_api_config()
        raw = guarded_text_call("wardrobe-audit", call_text_model,
                                cfg["base_url"], cfg["api_key"], cfg["model"],
                                read_prompt_file("wardrobe/audit.md"),
                                "USER CONTENT:\n" + str(content or "") + "\n\n" + block,
                                timeout=240, max_tokens=2000, image_path=proxy)
    finally:
        if os.path.isfile(proxy):
            os.remove(proxy)
    data = _json_object(raw)
    checks = data.get("checks")
    dimensions = {"silhouette", "construction", "trim", "coordination", "explicit_constraints"}
    valid = (isinstance(data.get("conforms"), bool) and isinstance(data.get("uncertain"), bool)
             and isinstance(data.get("summary"), str) and bool(data["summary"].strip())
             and isinstance(checks, list) and len(checks) == len(dimensions)
             and all(isinstance(c, dict) and c.get("result") in {"pass", "fail", "not_visible"}
                     and isinstance(c.get("evidence"), str) and bool(c["evidence"].strip()) for c in checks)
             and {c.get("dimension") for c in checks} == dimensions)
    accepted = (valid and data["conforms"] and not data["uncertain"]
                and not any(c["result"] == "fail" for c in checks)
                and any(c["result"] == "pass" for c in checks if c["dimension"] != "explicit_constraints"))
    return {**data, "accepted": bool(accepted), "conclusion_valid": bool(valid),
            "audit_key": wardrobe_audit_key(candidate, spec, content),
            "candidate": os.path.abspath(candidate), "candidate_sha256": file_sha256(candidate),
            "wardrobe_sha256": spec["prompt_sha256"], "raw": raw}


def record_wardrobe_audit(audit: dict, path: str):
    from utils.atomic_io import write_json_atomic
    write_json_atomic(path, audit, indent=2, ensure_ascii=False)
    from pathlib import Path
    lines = [str(audit.get("summary") or "衣装审计缺少有效结论")]
    for check in audit.get("checks") or []:
        if isinstance(check, dict):
            lines.append(f"{check.get('dimension')}: {check.get('result')} — {check.get('evidence')}")
    Path(path).with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")
    if not audit.get("accepted"):
        raise RuntimeError("衣装目标未通过或无法确认，保留候选图：" + lines[0] + "\n审计记录：" + path)
