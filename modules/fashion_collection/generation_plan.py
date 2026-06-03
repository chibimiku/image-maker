from __future__ import annotations

import json
import os

from .theme_profiles import ThemeProfile


GENERIC_PROMPT = "请生成一位可爱梦幻的少女，全身像，站姿自然，画面干净，突出服装整体搭配感。"
GENERIC_INSTRUCTIONS = "请严格参考输入的服饰图片完成一位少女角色的穿搭组合，保持主服装、鞋子、袜子、发饰和包袋的款式与颜色协调一致，输出日系少女插画风格。"


PART_LABELS = {
    "dress": "连衣裙",
    "shoes": "鞋子",
    "socks": "袜子",
    "hair_accessory": "发饰",
    "bag": "包袋",
}


def load_styles_config(styles_path: str) -> dict[str, str]:
    if not styles_path or not os.path.isfile(styles_path):
        return {}
    with open(styles_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    return {str(k): str(v or "") for k, v in payload.items()}


def split_style_tokens(style_value: str) -> list[str]:
    text = str(style_value or "").replace("，", ",").replace("\n", ",")
    return [token.strip() for token in text.split(",") if token.strip()]


def resolve_style_bundle(style_value: str, styles_data: dict[str, str], theme_profile: ThemeProfile | None = None) -> tuple[list[str], str]:
    tokens = split_style_tokens(style_value)
    resolved_names: list[str] = []
    resolved_texts: list[str] = []

    if not tokens and theme_profile:
        tokens = list(theme_profile.default_style_names)

    for token in tokens:
        if token in styles_data:
            resolved_names.append(token)
            style_text = str(styles_data.get(token) or "").strip()
            if style_text:
                resolved_texts.append(style_text)
        else:
            resolved_texts.append(token)

    seen_texts: list[str] = []
    for text in resolved_texts:
        if text and text not in seen_texts:
            seen_texts.append(text)
    return resolved_names, "\n\n".join(seen_texts).strip()


def resolve_prompt_and_instructions(
    prompt: str,
    instructions: str,
    theme_profile: ThemeProfile | None,
    style_text: str,
) -> tuple[str, str]:
    final_prompt = str(prompt or "").strip()
    final_instructions = str(instructions or "").strip()

    if not final_prompt:
        final_prompt = (theme_profile.default_prompt if theme_profile else "") or GENERIC_PROMPT
    if not final_instructions:
        final_instructions = (theme_profile.default_instructions if theme_profile else "") or GENERIC_INSTRUCTIONS

    if theme_profile:
        final_prompt = f"{theme_profile.title}\n{final_prompt}".strip()
    if style_text:
        final_instructions = f"{final_instructions}\n\n画风参考：\n{style_text}".strip()
    return final_prompt, final_instructions


def build_scene_and_character_description(bundle, theme_profile: ThemeProfile | None, character_count: int = 1) -> tuple[str, str]:
    normalized_count = 2 if int(character_count or 1) >= 2 else 1
    theme_title = theme_profile.title if theme_profile else "少女时尚"
    part_names = [PART_LABELS.get(asset.part, asset.part) for asset in bundle.assets]
    part_summary = "、".join(part_names) if part_names else "服饰"
    if theme_profile and theme_profile.key == "sweet-lolita":
        scene_text = (
            "场景设定：欧式花园下午茶与甜点茶会氛围，暖阳从花架和玻璃温室间洒下，"
            f"背景有玫瑰、蕾丝桌布与甜点陈列，整体与{part_summary}的甜美华丽感呼应。"
        )
        if normalized_count == 2:
            character_text = (
                "主角描述：两位气质协调的少女同框，一位为主视觉中心，另一位作为陪伴角色，"
                "身高和体态略有区分，互动自然，强调甜美洛丽塔姐妹感与精致配饰细节。"
            )
        else:
            character_text = (
                "主角描述：一位面容精致、气质甜美的少女作为主角，体态轻盈，姿态优雅，"
                "发型与发饰呼应裙装细节，整体表现出梦幻、可爱、精心打扮后的茶会大小姐感。"
            )
        return scene_text, character_text

    scene_text = (
        f"场景设定：围绕{theme_title}与{part_summary}营造统一的时尚插画场景，背景简洁但具有空间层次，"
        "让服装和配饰成为视觉重点。"
    )
    if normalized_count == 2:
        character_text = (
            "主角描述：两位主角同框，服装主题一致但姿态与表情略有区分，形成主次层次，"
            "表现协调互动与成套穿搭的呼应关系。"
        )
    else:
        character_text = (
            "主角描述：一位主角居中出镜，人物设定与服装风格保持一致，面部、发型、姿态和配饰都围绕服装主题展开。"
        )
    return scene_text, character_text


def build_reference_prompt(base_prompt: str, bundle, scene_text: str = "", character_text: str = "") -> str:
    prompt_lines = [str(base_prompt or "").strip()]
    if scene_text:
        prompt_lines.extend(["", scene_text.strip()])
    if character_text:
        prompt_lines.extend(["", character_text.strip()])
    prompt_lines.extend(["", "服饰参考清单："])
    for asset in bundle.assets:
        prompt_lines.append(f"- {PART_LABELS.get(asset.part, asset.part)}: {asset.item.title}")
        if asset.item.brand:
            prompt_lines.append(f"  品牌: {asset.item.brand}")
        if asset.prompt_hint:
            prompt_lines.append(f"  要点: {asset.prompt_hint}")
    return "\n".join(line for line in prompt_lines if line is not None).strip()
