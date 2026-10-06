"""Optional prompt-only atmosphere, frozen independently of source analysis."""
import copy
import hashlib
import json

from utils.prompt_loader import read_prompt_file, render_prompt_file


def catalog():
    return json.loads(read_prompt_file("color-knowledge/refine-atmosphere-v1.json"))


def _hash(plan):
    body = {k: v for k, v in plan.items() if k != "plan_hash"}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def freeze(selection=None):
    selection = dict(selection or {})
    if "plan_hash" in selection:
        if selection["plan_hash"] != _hash(selection):
            raise ValueError("生成氛围快照校验失败")
        return copy.deepcopy(selection)
    chosen = {key: str(selection.get(key) or "") for key in ("mood", "season")}
    if not any(chosen.values()):
        return {}
    entries = catalog()
    instructions, pages = [], []
    for key, group in (("mood", "moods"), ("season", "seasons")):
        if not chosen[key]:
            continue
        entry = next((e for e in entries[group] if e["id"] == chosen[key]), None)
        if entry is None:
            raise ValueError("未知生成氛围选项: " + chosen[key])
        instructions.append(entry["instruction"])
        pages.append(entry["page"])
    plan = {"version": 1, "selection": chosen, "source_pages": sorted(set(pages)),
            "scope": "refined_generation_prompt", "prompt": render_prompt_file(
                "color-knowledge/refine-atmosphere-contract-v1.md",
                {"instructions": "\n".join(instructions)}).strip()}
    plan["compact_prompt"] = render_prompt_file(
        "color-knowledge/refine-atmosphere-compact-v1.md",
        {"choices": "; ".join(key + "=" + value for key, value in chosen.items() if value)}).strip()
    plan["plan_hash"] = _hash(plan)
    return plan


def refine_note(plan):
    if not plan:
        return ""
    plan = freeze(plan)
    return "\n\n" + render_prompt_file("color-knowledge/refine-atmosphere-note-v1.md", {"prompt": plan["prompt"]})


def apply_result(result, plan=None, fields=("english_description", "short_description")):
    plan = freeze(plan if plan is not None else result.get("generation_atmosphere"))
    if not plan:
        return result
    result["generation_atmosphere"] = plan
    for field in fields:
        suffix = plan["compact_prompt"] if field.startswith("gpt_image_") else plan["prompt"]
        text = str(result.get(field) or "").strip()
        if text and suffix not in text:
            result["factual_" + field] = text
            result[field] = text + "\n\n" + suffix
    return result
