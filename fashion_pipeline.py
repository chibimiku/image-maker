"""
Fashion Auto-Collect & Generate Pipeline

一站式流水线：
  1. 从指定站点自动搜索并采集服饰素材
  2. 通过 LLM 生成摄影构图/姿势描述
  3. 用角色参考图+服饰素材图作为参考，生成少女插画

用法:
    python fashion_pipeline.py \\
        --site hybrid \\
        --theme "甜美洛丽塔" \\
        --parts dress,shoes,socks,hair_accessory,bag \\
        --character-image "d:/nikki-comic.jpg" \\
        --extra-prompt "夏日花园午后场景" \\
        --instructions "日系插画风格，高跟凉鞋" \\
        --style puracotte-style \\
        --characters 1 \\
        --ratio 3:4 \\
        --resolution 2K \\
        --api aigc2d \\
        --output-prefix nikki_sweet_summer
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from modules.fashion_collection.collection_service import FashionCollectionService
from modules.fashion_collection.generation_plan import (
    PART_LABELS,
    build_reference_prompt,
    build_scene_and_character_description,
    load_styles_config,
    resolve_prompt_and_instructions,
    resolve_style_bundle,
)
from modules.fashion_collection.theme_profiles import get_theme_profile
from modules.others.api_backend import (
    fetch_llm_json,
    generate_image_aigc2d,
    get_api_config,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_SITE = "hybrid"
DEFAULT_THEME = "甜美洛丽塔"
DEFAULT_PARTS = "dress,shoes,socks,hair_accessory,bag"
DEFAULT_RATIO = "3:4"
DEFAULT_RESOLUTION = "2K"
DEFAULT_API = "aigc2d"


def parse_parts(raw: str) -> list[str]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    valid = {"dress", "shoes", "socks", "hair_accessory", "bag"}
    return [p for p in parts if p in valid]


def _get_llm_config() -> tuple[str, str, str]:
    """Extract LLM chat config from config-image.json for composition generation."""
    cfg = get_api_config(api_type="aigc2d")
    base_url = str(cfg.get("base_url", "https://next.aigc2d.com/v1beta/models/") or "").strip()
    # For chat, convert the image gen URL to the v1 chat endpoint
    if "/v1beta/models/" in base_url:
        base_url = base_url.split("/v1beta/models/")[0] + "/v1"
    api_key = cfg.get("api_key", "")
    model = "gemini-3.1-flash-image-preview"  # Chat-capable model
    return base_url, api_key, model


def _load_prompt_file(relative_path: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    full_path = os.path.join(PROJECT_ROOT, relative_path)
    if not os.path.isfile(full_path):
        print(f"[警告] Prompt 文件不存在: {full_path}")
        return ""
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()


def generate_composition(
    theme: str,
    character_desc: str,
    items_summary: str,
    ratio: str,
    characters: int,
) -> str:
    """Ask LLM to generate a photography composition / pose description."""

    char_note = (
        "画面中有两位主角同框互动，请描述她们各自的位置、姿态和视线方向。"
        if int(characters) >= 2
        else "画面中只有一位主角。"
    )

    # Load prompts from template files
    system = _load_prompt_file("prompts/fashion-composition-system.md")
    user_template = _load_prompt_file("prompts/fashion-composition-user.md")

    if not system:
        print("[构图] System prompt 模板缺失，跳过构图生成。")
        return ""

    user = (user_template or "").replace("{{theme}}", theme)
    user = user.replace("{{character_desc}}", character_desc)
    user = user.replace("{{items_summary}}", items_summary)
    user = user.replace("{{ratio}}", ratio)
    user = user.replace("{{char_note}}", char_note)

    try:
        base_url, api_key, model = _get_llm_config()
        print(f"[构图] 正在请求 LLM 生成摄影构图方案...")
        raw_json = fetch_llm_json(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=system,
            user_content=user,
            temperature=0.8,
            merge_system_prompt=True,
        )
        data = json.loads(raw_json) if raw_json else {}
        if not data:
            print("[构图] LLM 返回为空，使用默认构图。")
            return ""

        lines: list[str] = []
        labels = {
            "composition_type": "构图类型",
            "camera_angle": "镜头角度",
            "pose_description": "角色姿态",
            "focal_point": "视觉焦点",
            "depth_of_field": "景深",
            "lighting": "光线",
            "overall_mood": "画面情绪",
        }
        for key, label in labels.items():
            val = str(data.get(key, "")).strip()
            if val:
                lines.append(f"{label}: {val}")

        negative = str(data.get("negative_prompt_hints", "")).strip()
        if negative:
            lines.append(f"需避免: {negative}")

        result = "摄影构图指导:\n" + "\n".join(f"- {line}" for line in lines)
        print(f"[构图] 已生成 ({len(result)} 字符)")
        return result
    except Exception as e:
        print(f"[构图] LLM 请求失败: {e}，将使用默认构图。")
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="Fashion Auto-Collect & Generate Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--site", default=DEFAULT_SITE, help="站点: lolibrary / wear / mayla / hybrid")
    parser.add_argument("--theme", default=DEFAULT_THEME, help="采集主题")
    parser.add_argument("--brand", default="", help="品牌 slug/key（lolibrary 必填）")
    parser.add_argument("--parts", default=DEFAULT_PARTS, help="目标部位，逗号分隔")
    parser.add_argument("--max-pages", type=int, default=1, help="每部位扫描页数")
    parser.add_argument("--extra-prompt", default="", help="额外出图 Prompt")
    parser.add_argument("--instructions", default="", help="系统约束/instructions")
    parser.add_argument("--style", default="", help="画风预设名，留空则用主题默认")
    parser.add_argument("--characters", type=int, default=1, help="主角人数 1 或 2")
    parser.add_argument("--ratio", default=DEFAULT_RATIO, help="出图比例")
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION, help="分辨率: 1K / 2K / 4K")
    parser.add_argument("--api", default=DEFAULT_API, help="API 类型")
    parser.add_argument("--output-prefix", default="fashion_pipeline", help="输出文件前缀")
    parser.add_argument("--character-image", default="", help="角色参考图路径（如 d:/nikki-comic.jpg）")
    parser.add_argument("--character-desc", default="", help="角色文字描述（配合角色图使用）")
    parser.add_argument("--no-composition", action="store_true", help="跳过构图生成")
    parser.add_argument("--no-collect", action="store_true", help="跳过采集，直接从已有目录读取素材")
    parser.add_argument("--bundle-dir", default="", help="已有 bundle 目录（配合 --no-collect 使用）")

    args = parser.parse_args()
    parts = parse_parts(args.parts)
    if not parts:
        print("错误：未指定有效部位。可用: dress, shoes, socks, hair_accessory, bag")
        sys.exit(1)

    theme_profile = get_theme_profile(args.theme)
    if theme_profile:
        print(f"[主题] {theme_profile.title} (key={theme_profile.key})")
    else:
        print(f"[主题] 自定义: {args.theme}")

    # Validate character image
    character_image = ""
    if args.character_image:
        if os.path.isfile(args.character_image):
            character_image = os.path.abspath(args.character_image)
            print(f"[角色图] {character_image}")
        else:
            print(f"[警告] 角色图不存在: {args.character_image}")

    # -----------------------------------------------------------------------
    # Step 1 — Collect fashion items
    # -----------------------------------------------------------------------
    if args.no_collect and args.bundle_dir:
        print(f"[采集] 跳过，使用已有目录: {args.bundle_dir}")
        output_dir = args.bundle_dir
    else:
        service = FashionCollectionService(timeout=8)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(PROJECT_ROOT, "data", "fashion-pipeline", args.site, timestamp)
        brand = args.brand
        if args.site == "lolibrary" and not brand:
            if theme_profile and theme_profile.dress_brands:
                brand = theme_profile.dress_brands[0]
            else:
                brand = "angelic-pretty"

        print(f"[采集] 站点={args.site}, 品牌={brand or '不限'}, 部位={parts}, 主题={args.theme}")
        print(f"[采集] 输出目录={output_dir}")

        bundle = service.collect_bundle(
            site_key=args.site,
            brand_slug=brand,
            output_dir=output_dir,
            max_pages=args.max_pages,
            preferred_parts=parts,
            theme=args.theme,
        )
        print(f"[采集] 完成: {len(bundle.assets)} 件素材, 缺失: {bundle.missing_parts or '无'}")
        for a in bundle.assets:
            print(f"  [{a.part}] {a.item.title} -> {a.local_path}")

    # -----------------------------------------------------------------------
    # Step 2 — Build prompt & instructions
    # -----------------------------------------------------------------------
    styles_data = load_styles_config(os.path.join(PROJECT_ROOT, "conf", "config-styles.json"))
    resolved_style_names, style_text = resolve_style_bundle(
        args.style, styles_data, theme_profile=theme_profile,
    )
    if resolved_style_names:
        print(f"[画风] {', '.join(resolved_style_names)}")

    final_prompt, final_instructions = resolve_prompt_and_instructions(
        args.extra_prompt, args.instructions,
        theme_profile=theme_profile, style_text=style_text,
    )

    if args.no_collect and args.bundle_dir:
        composed_extra = final_prompt
    else:
        scene_text, character_text = build_scene_and_character_description(
            bundle, theme_profile=theme_profile, character_count=args.characters,
        )
        final_prompt = build_reference_prompt(
            final_prompt or (theme_profile.default_prompt if theme_profile else ""),
            bundle,
            scene_text,
            character_text,
        )
        composed_extra = "\n\n".join([t for t in [final_prompt, scene_text, character_text] if t])
        print(f"[场景] {scene_text[:120]}...")
        print(f"[主角] {character_text[:120]}...")

    # -----------------------------------------------------------------------
    # Step 3 — Add character reference to prompt
    # -----------------------------------------------------------------------
    if character_image and args.character_desc:
        char_intro = (
            f"角色设定：请以提供的角色参考图为原型，绘制该角色。{args.character_desc}\n"
            "保持角色的面部特征、发型、发色和整体气质一致，"
            "但为她换上采集到的服饰穿搭。"
        )
        final_prompt = char_intro + "\n\n" + final_prompt

        final_instructions = (
            "请严格参照角色参考图的面部特征、发型发色来绘制该角色，"
            "确保角色辨识度（face consistency），"
            "同时为她穿上参考服饰图中的全套穿搭。\n\n"
            + final_instructions
        )

    # -----------------------------------------------------------------------
    # Step 4 — Generate photography composition via LLM
    # -----------------------------------------------------------------------
    composition_text = ""
    if not args.no_composition:
        items_summary = ""
        if not (args.no_collect and args.bundle_dir):
            items_summary = "、".join(
                f"{PART_LABELS.get(a.part, a.part)}「{a.item.title}」"
                for a in bundle.assets
            )
        character_summary = args.character_desc or final_prompt[:200]
        composition_text = generate_composition(
            theme=args.theme,
            character_desc=character_summary,
            items_summary=items_summary,
            ratio=args.ratio,
            characters=args.characters,
        )

    if composition_text:
        final_instructions = f"{final_instructions}\n\n{composition_text}".strip()
        print(f"[构图] 已注入构图指导到 instructions")

    print(f"[Prompt 预览]\n{final_prompt[:400]}...\n")
    print(f"[Instructions 预览]\n{final_instructions[:400]}...\n")

    # -----------------------------------------------------------------------
    # Step 5 — Collect reference images
    # -----------------------------------------------------------------------
    image_paths: list[str] = []

    # Character image goes FIRST so the model registers the character identity
    if character_image:
        image_paths.append(character_image)
        print(f"[参考图] 角色图 1 张")

    if not args.no_collect:
        for a in bundle.assets:
            if os.path.isfile(a.local_path):
                image_paths.append(a.local_path)
        print(f"[参考图] 服饰 {sum(1 for a in bundle.assets if os.path.isfile(a.local_path))} 张")
    else:
        for root, _dirs, files in os.walk(args.bundle_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    image_paths.append(os.path.join(root, f))

    for p in image_paths:
        print(f"  {p}")

    # -----------------------------------------------------------------------
    # Step 6 — Generate image
    # -----------------------------------------------------------------------
    print(f"\n[生成] 开始调用 API={args.api}, 分辨率={args.resolution}, 比例={args.ratio}, 参考图共{len(image_paths)}张")
    result = generate_image_aigc2d(
        prompt=final_prompt,
        image_paths=image_paths or None,
        aspect_ratio=args.ratio,
        instructions=final_instructions,
        resolution=args.resolution,
        api_type=args.api,
        save_sub_dir="fashion-pipeline",
        file_prefix=args.output_prefix,
        return_metadata=True,
    )

    saved_files: list[str] = []
    if isinstance(result, dict):
        saved_files = result.get("saved_files") or []
    elif isinstance(result, list):
        saved_files = result

    print(f"\n[完成] 生成 {len(saved_files)} 张图片:")
    for f in saved_files:
        print(f"  {f}")

    # Save metadata
    meta_dir = (
        os.path.dirname(saved_files[0]) if saved_files
        else (output_dir if (not args.no_collect or args.bundle_dir) else ".")
    )
    if not os.path.isabs(meta_dir):
        meta_dir = os.path.join(PROJECT_ROOT, meta_dir)
    meta_path = os.path.join(meta_dir, f"{args.output_prefix}_pipeline_meta.json")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump({
            "site": args.site,
            "theme": args.theme,
            "parts": parts,
            "style": resolved_style_names,
            "composition_text": composition_text,
            "character_image": character_image,
            "prompt": final_prompt,
            "instructions": final_instructions,
            "reference_count": len(image_paths),
            "saved_files": saved_files,
        }, mf, ensure_ascii=False, indent=2)
    print(f"[元数据] {meta_path}")


if __name__ == "__main__":
    main()
