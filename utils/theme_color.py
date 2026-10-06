"""First-generation theme colour contracts; independent of source-image editing."""
import hashlib
import json
from pathlib import Path
import re
import uuid

from utils.prompt_loader import read_prompt_file, render_prompt_file
from utils.atomic_io import write_json_atomic

BASE = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def profiles():
    return json.loads(read_prompt_file("color-knowledge/theme-color-profiles-v1.json"))["profiles"]


def direction_catalog():
    return json.loads(read_prompt_file("color-knowledge/theme-color-direction-v2.json"))


def freeze_theme_color(selection=None):
    selection = dict(selection or {})
    name = selection.get("profile_id") or ""
    active_palette = bool(name and name != "off")
    tone = str(selection.get("tone") or "")
    area = str(selection.get("area") or "") if active_palette else ""
    if not active_palette and not tone:
        return {}
    profile = next((p for p in profiles() if p["id"] == name), None) if active_palette else {}
    if active_palette and profile is None:
        raise ValueError("配色方案不存在，请重新选择")
    environment = str(selection.get("environment") or "").strip()
    accents = str(selection.get("accent_regions") or "").strip()
    if active_palette and (not environment or not accents or selection.get("confirmed") is not True):
        raise ValueError("请在配色详情中填写正文已有的环境和点缀区域，并确认允许改色")
    names = lambda text: {r.strip().casefold() for r in re.split(r"[,;，；\n]", text) if r.strip()}
    if active_palette and names(environment) & names(accents):
        raise ValueError("主色区域和点缀区域不能相同")
    if max(len(environment), len(accents)) > 500:
        raise ValueError("配色区域描述请限制在 500 字符内")
    prompt = render_prompt_file("color-knowledge/theme-color-contract-v1.md", {
        "base": profile["base"], "accent": profile["accent"],
        "environment": environment, "accent_regions": accents}).strip() if active_palette else ""
    pages = {146, 79} if active_palette else set()
    auxiliary_regions = str(selection.get("auxiliary_regions") or "").strip()
    if tone or area:
        catalog = direction_catalog()
        instructions = []
        tone_source = None
        for group, value in (("tones", tone), ("areas", area)):
            if not value:
                continue
            entry = next((e for e in catalog[group] if e["id"] == value), None)
            if entry is None:
                raise ValueError("未知色调或主次选项: " + value)
            instructions.append(entry["instruction"])
            pages.update(entry["pages"])
            if group == "tones":
                source_tones = json.loads(read_prompt_file("color-knowledge/book-generation-knowledge-v3.json"))["tones"]
                tone_source = {"mapping_note": entry["mapping_note"], "reference_tones": [
                    t for t in source_tones if t["code"] in entry["ncd_candidates"]]}
                if {t["code"] for t in tone_source["reference_tones"]} != set(entry["ncd_candidates"]):
                    raise ValueError("色调来源目录缺项，禁止无来源补齐")
        auxiliary_binding = ""
        if area == "three-level":
            if not auxiliary_regions or len(auxiliary_regions) > 500:
                raise ValueError("三层主次需要单独填写正文已有的辅助色区域（最多 500 字符）")
            if names(auxiliary_regions) & (names(environment) | names(accents)):
                raise ValueError("辅助色区域不能和主色、强调色区域相同")
            auxiliary = catalog["auxiliary"].get(name)
            if not auxiliary:
                raise ValueError("该色板尚未指定辅助色，不能按排列顺序推断")
            auxiliary_binding = render_prompt_file("color-knowledge/theme-color-auxiliary-v2.md",
                {"colour": auxiliary["colour"], "regions": auxiliary_regions}).strip()
        if area:
            prompt = render_prompt_file("color-knowledge/theme-color-binding-v2.md", {
                "base": profile["base"], "accent": profile["accent"],
                "environment": environment, "accent_regions": accents,
                "auxiliary_binding": auxiliary_binding}).strip()
        direction = render_prompt_file("color-knowledge/theme-color-direction-v2.md",
                                      {"instructions": "\n".join(instructions)}).strip()
        prompt = "\n\n".join(p for p in (prompt, direction) if p)
    snapshot = {"schema_version": 1, "profile": profile, "selection": selection,
                "scope": "text_only_first_generation", "prompt": prompt,
                "source_pages": sorted(pages), "validation": "not_reviewed",
                "swatch_source": "designer_approximation_not_book_measurement"}
    if tone or area:
        snapshot["schema_version"] = 2
        snapshot["direction"] = {"tone": tone, "area": area,
                                 "auxiliary_regions": auxiliary_regions if area == "three-level" else ""}
        if tone_source:
            snapshot["direction"]["tone_source"] = tone_source
        if area == "three-level":
            snapshot["direction"]["auxiliary"] = auxiliary
    snapshot["plan_hash"] = digest(snapshot)
    return snapshot


def validate_snapshot(snapshot):
    if snapshot and digest({k: v for k, v in snapshot.items() if k != "plan_hash"}) != snapshot.get("plan_hash"):
        raise ValueError("配色快照校验失败，禁止使用已变更的条款")


def apply_theme_color(prompt, snapshot, *, image_paths=(), mode="generate", post_enabled=False, content_prompt=None):
    if not snapshot:
        return prompt
    validate_snapshot(snapshot)
    if image_paths or mode != "generate":
        raise ValueError("主题配色首版只支持文字生图，请移除内容/画风参考图并选择生图模式")
    if post_enabled:
        raise ValueError("主题配色首版只生成首图，请关闭重绘和后处理；后续工序的配色保留尚未验证")
    content = (prompt if content_prompt is None else content_prompt).casefold()
    selection = snapshot["selection"]
    keys = ["environment", "accent_regions"] if snapshot.get("profile") else []
    if (snapshot.get("direction") or {}).get("area") == "three-level":
        keys.append("auxiliary_regions")
    regions = [r.strip() for key in keys
               for r in re.split(r"[,;，；\n]", str(selection.get(key) or "")) if r.strip()]
    missing = [r for r in regions if not re.search(r"(?<![a-z0-9_])" + re.escape(r.casefold()) + r"(?![a-z0-9_])", content)]
    if missing:
        raise ValueError("配色落点未在正文中找到，请复制正文里的物体短语：" + ", ".join(missing))
    return prompt + "\n\n" + snapshot["prompt"]


def theme_style(style, snapshot):
    """Remove only the known eight-field Palette section, never edit stored styles."""
    if not snapshot:
        return style
    validate_snapshot(snapshot)
    if not snapshot.get("profile"):
        return style
    return re.sub(r"(?im)^Palette\s*:[^\n]*\n?", "", style).strip()


def record_request(snapshot, prompt, *, model="", channel="", context=None, root=None):
    if not snapshot:
        return {}
    validate_snapshot(snapshot)
    folder = Path(root) if root else BASE / "cache/temp/theme-color"
    folder.mkdir(parents=True, exist_ok=True)
    record = {"request_id": uuid.uuid4().hex, "color_plan": snapshot, "prompt": prompt,
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "model": model, "channel": channel, "context": context or {}, "outputs": [],
              "status": "prepared", "colour_review": "not_performed"}
    path = folder / (record["request_id"] + ".json")
    write_json_atomic(str(path), record)
    return {"path": str(path), "record": record}


def record_outputs(request_record, paths):
    if not request_record:
        return
    record = dict(request_record["record"])
    record["outputs"] = [{"path": str(Path(p).resolve()),
                          "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in paths]
    record["status"] = "generated" if paths else "no_output"
    write_json_atomic(request_record["path"], record)
    for path in paths:
        write_json_atomic(str(Path(path).with_suffix(".color-plan.json")), record)


def record_failure(request_record, error_type):
    if request_record:
        record = {**request_record["record"], "status": "failed", "error_type": error_type}
        write_json_atomic(request_record["path"], record)


def recommend_theme(description, *, caller=None, cache_dir=None):
    from utils.analysis_gpt_prompt import load_text_api_config, call_text_model
    cfg = load_text_api_config()
    system = read_prompt_file("color-knowledge/theme-color-select-v1.md")
    candidates = profiles()
    cache_key = digest({"description": description, "profiles": candidates, "system": system,
                        "model": cfg["model"], "base_url": cfg["base_url"]})
    # 测试注入 caller 不触碰真实缓存；只复用完整校验通过的成功推荐。
    cache_path = None if caller else (Path(cache_dir) if cache_dir else BASE / "cache/temp/theme-color-select") / (cache_key + ".json")
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("cache_key") == cache_key and cached.get("result", {}).get("profile_id") in {"off", *(p["id"] for p in candidates)}:
            return cached["result"]
    value = (caller or call_text_model)(cfg["base_url"], cfg["api_key"], cfg["model"],
        system, json.dumps({"description": description, "candidates": candidates}, ensure_ascii=False),
        max_tokens=2000, timeout=120)
    result = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", value.strip()))
    if not isinstance(result, dict):
        raise ValueError("推荐没有返回 JSON 对象")
    if result.get("profile_id") not in {"off", *(p["id"] for p in profiles())} or not isinstance(result.get("reason"), str):
        raise ValueError("推荐返回了非法方案或缺少理由")
    result = {"profile_id": result["profile_id"], "reason": result["reason"][:1000]}
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(str(cache_path), {"cache_key": cache_key, "result": result,
                                          "description": description, "model": cfg["model"]})
    return result
