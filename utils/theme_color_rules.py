"""Book palette bindings and validated, cached text-only rule proposals."""
import hashlib
import json
from pathlib import Path
import re

from utils.atomic_io import write_json_atomic
from utils.prompt_loader import read_prompt_file, render_prompt_file


def region_names(value):
    return [part.strip() for part in re.split(r"[,;，；\n]", value) if part.strip()]


def freeze_book_palette(selection, profile):
    from utils.theme_color import digest, direction_catalog
    if selection.get("confirmed") is not True:
        raise ValueError("请确认允许将所选书籍色板应用到这些已有区域")
    bindings = selection.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("请点自动生成规则，或填写至少一个书籍颜色落点")
    colours = {colour["order"]: colour for colour in profile["colours"]}
    used, seen, roles, lines = set(), set(), {}, []
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValueError("颜色落点必须为对象")
        order, regions, role = binding.get("colour_order"), binding.get("regions"), binding.get("role", "")
        if type(order) is not int or order not in colours or order in used:
            raise ValueError("书籍色板的颜色编号不存在或重复")
        if not isinstance(regions, str) or not regions.strip() or len(regions) > 500:
            raise ValueError("颜色落点必须是正文已有短语（最多500字符）")
        if role not in ("", "base", "auxiliary", "accent"):
            raise ValueError("未知主次角色")
        names = {name.casefold() for name in region_names(regions)}
        if names & seen:
            raise ValueError("不同颜色落点不能重复")
        if role:
            if role in roles:
                raise ValueError("同一主次角色只能指定一种颜色")
            roles[role] = colours[order]["name"]
        seen.update(names)
        used.add(order)
        # The order addresses a swatch; it is NEVER interpreted as hierarchy.
        lines.append(f"Colour {order}: {colours[order]['name']} ONLY on: {regions.strip()}. "
                     + (f"Proposed role for this task: {role}." if role else "No hierarchy role specified."))
    catalog = direction_catalog()
    instructions, pages = [], {profile["source"]["pdf_page"]}
    tone, area = selection.get("tone", ""), selection.get("area", "")
    for group, value in (("tones", tone), ("areas", area)):
        if value:
            entry = next((x for x in catalog[group] if x["id"] == value), None)
            if not entry:
                raise ValueError("未知色调或主次选项")
            instructions.append(entry["instruction"])
            pages.update(entry["pages"])
    needed = {"base", "accent", "auxiliary"} if area == "three-level" else {"base", "accent"} if area else set()
    if needed - roles.keys():
        raise ValueError("主次规则需要在颜色表中指定主色/强调色，三层还需要辅助色；未指定不能按顺序猜")
    prompt = render_prompt_file("color-knowledge/book-theme-contract-v1.md", {
        "colours": " / ".join(c["name"] for c in profile["colours"]),
        "bindings": "\n".join(lines)}).strip()
    if instructions:
        prompt += "\n\n" + render_prompt_file("color-knowledge/theme-color-direction-v2.md", {
            "instructions": "\n".join(instructions)}).strip()
    plan = {"schema_version": 3, "profile": profile, "selection": selection,
            "scope": "text_only_first_generation", "prompt": prompt,
            "source_pages": sorted(pages), "validation": "not_reviewed",
            "swatch_source": "per_colour_source_in_book_catalog",
            "role_source": "user_or_model_task_proposal_not_book_order",
            "direction": {"tone": tone, "area": area},
            "unused_colour_orders": sorted(colours.keys() - used)}
    plan["plan_hash"] = digest(plan)
    return plan


def validate_rule_proposal(value, description, locked_profile=""):
    from utils.theme_color import profiles, freeze_theme_color, apply_theme_color
    if not isinstance(value, dict):
        raise ValueError("自动规则没有返回JSON对象")
    profile_id = value.get("profile_id")
    if locked_profile and profile_id != locked_profile:
        raise ValueError("自动规则不能更换用户选定的色板")
    profile = next((p for p in profiles() if p["id"] == profile_id), None)
    if not profile:
        raise ValueError("自动规则必须选择目录已有方案")
    result = {"profile_id": profile_id, "tone": value.get("tone", ""), "area": value.get("area", ""),
              "confirmed": True}
    if profile.get("origin") == "book_case":
        bindings = value.get("bindings")
        if not isinstance(bindings, list) or any(not isinstance(b, dict) for b in bindings):
            raise ValueError("自动规则缺少颜色落点列表")
        result["bindings"] = [{"colour_order": b.get("colour_order"), "regions": b.get("regions"),
                               "role": b.get("role", "")} for b in bindings]
    else:
        for key in ("environment", "accent_regions", "auxiliary_regions"):
            if not isinstance(value.get(key, ""), str):
                raise ValueError("自动规则落点必须为文本")
            result[key] = value.get(key, "").strip()
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("自动规则缺少建议理由")
    # No model-supplied prompt, HEX, permission flag or palette data is trusted.
    plan = freeze_theme_color(result)
    apply_theme_color(description, plan)
    result["reason"] = reason[:1000]
    return result


def generate_theme_rules(description, profile_id="", *, caller=None, cache_dir=None):
    from utils.analysis_gpt_prompt import load_text_api_config, call_text_model
    from utils.theme_color import BASE, digest, profiles, direction_catalog
    description = str(description or "").strip()
    if not description:
        raise ValueError("请先填写当前生图正文")
    candidates = profiles()
    if profile_id:
        candidates = [p for p in candidates if p["id"] == profile_id]
        if not candidates:
            raise ValueError("选定色板不在目录中")
    else:
        candidates = [p for p in candidates if p.get("origin") == "book_case"]
    system = read_prompt_file("color-knowledge/theme-color-auto-rules-v1.md")
    cfg = load_text_api_config()
    payload = {"description": description, "locked_profile_id": profile_id,
               "candidates": candidates, "directions": direction_catalog()}
    key = digest({"payload": payload, "system": system, "model": cfg["model"], "base_url": cfg["base_url"]})
    path = None if caller else (Path(cache_dir) if cache_dir else BASE / "cache/temp/theme-color-rules") / (key + ".json")
    if path and path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("cache_key") == key:
            return validate_rule_proposal(cached.get("result"), description, profile_id)
    raw = (caller or call_text_model)(cfg["base_url"], cfg["api_key"], cfg["model"], system,
        json.dumps(payload, ensure_ascii=False), max_tokens=3500, timeout=120)
    value = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()))
    result = validate_rule_proposal(value, description, profile_id)
    result["description_sha256"] = hashlib.sha256(description.encode()).hexdigest()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(str(path), {"cache_key": key, "result": result, "model": cfg["model"]})
    return result
