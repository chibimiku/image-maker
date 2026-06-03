import argparse
import json
import os
import sys
from datetime import datetime

from modules.fashion_collection.collection_service import FashionCollectionService
from modules.fashion_collection.generation_plan import (
    build_reference_prompt,
    build_scene_and_character_description,
    load_styles_config,
    resolve_prompt_and_instructions,
    resolve_style_bundle,
)
from modules.fashion_collection.make_pic_bridge import export_bundle_manifest, export_make_pic_state
from modules.fashion_collection.models import PART_BAG, PART_DRESS, PART_HAIR_ACCESSORY, PART_SHOES, PART_SOCKS
from modules.fashion_collection.theme_profiles import get_theme_profile
from modules.others.api_backend import generate_image_aigc2d


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="抓取一套服饰并调用 AIGC2D 生成 1 张少女图片")
    parser.add_argument("--site", choices=["lolibrary", "wear", "hybrid"], default="wear", help="采集站点")
    parser.add_argument("--brand", default="", help="品牌 slug/key；wear 可留空")
    parser.add_argument("--theme", default="", help="采集主题，如 甜美洛丽塔")
    parser.add_argument("--style", default="", help="画风预设名或直接的 style 文本，可用逗号分隔多个")
    parser.add_argument("--pages", type=int, default=1, help="扫描页数")
    parser.add_argument("--parts", nargs="+", default=[PART_DRESS, PART_SHOES, PART_SOCKS, PART_HAIR_ACCESSORY, PART_BAG], help="目标部位")
    parser.add_argument("--character-count", type=int, choices=[1, 2], default=1, help="主角人数，支持 1 或 2 人")
    parser.add_argument("--aspect-ratio", default="2:3", help="输出比例")
    parser.add_argument("--resolution", default="", help="覆盖输出分辨率，如 2K；为空时跟配置")
    parser.add_argument("--prompt", default="", help="用户主提示词；为空时按主题或默认值生成")
    parser.add_argument("--instructions", default="", help="附加系统风格/约束；为空时按主题或默认值生成")
    parser.add_argument("--output-subdir", default="fashion-generate", help="生成图片保存子目录")
    parser.add_argument("--file-prefix", default="fashion_combo", help="输出文件前缀")
    parser.add_argument("--export-state", action="store_true", help="同时导出 make-pic 状态文件")
    return parser.parse_args()


def main():
    args = parse_args()
    service = FashionCollectionService()
    theme_profile = get_theme_profile(args.theme)
    styles_data = load_styles_config(os.path.join(BASE_DIR, "conf", "config-styles.json"))
    resolved_style_names, style_text = resolve_style_bundle(args.style, styles_data, theme_profile=theme_profile)
    final_prompt_base, final_instructions = resolve_prompt_and_instructions(
        args.prompt,
        args.instructions,
        theme_profile=theme_profile,
        style_text=style_text,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_DIR, "data", "fashion-collector", args.site, timestamp)

    bundle = service.collect_bundle(
        site_key=args.site,
        brand_slug=args.brand,
        output_dir=output_dir,
        max_pages=max(1, int(args.pages)),
        preferred_parts=list(args.parts),
        theme=args.theme,
    )
    if bundle.missing_parts:
        print(f"缺失部位: {', '.join(bundle.missing_parts)}", file=sys.stderr)
    if not bundle.assets:
        raise RuntimeError("未采集到可用素材，无法继续生成。")

    manifest_path = export_bundle_manifest(bundle, bundle.output_dir)
    print(f"采集清单: {manifest_path}")

    scene_text, character_text = build_scene_and_character_description(
        bundle,
        theme_profile=theme_profile,
        character_count=args.character_count,
    )
    composed_extra_prompt = "\n\n".join(
        [text for text in [final_prompt_base, scene_text, character_text] if str(text or "").strip()]
    ).strip()

    if args.export_state:
        state_path = export_make_pic_state(
            bundle,
            state_path=os.path.join(BASE_DIR, "cache", "last_state.json"),
            instructions=final_instructions,
            extra_prompt=composed_extra_prompt,
            aspect_ratio=args.aspect_ratio,
        )
        print(f"已导出 make-pic 状态文件: {state_path}")
    if resolved_style_names:
        print(f"已应用画风预设: {', '.join(resolved_style_names)}")
    elif style_text:
        print("已应用自定义 style 文本")
    if theme_profile:
        print(f"当前主题: {theme_profile.title}")
    print(scene_text)
    print(character_text)

    final_prompt = build_reference_prompt(final_prompt_base, bundle, scene_text=scene_text, character_text=character_text)
    image_paths = [asset.local_path for asset in bundle.assets if os.path.isfile(asset.local_path)]

    result = generate_image_aigc2d(
        prompt=final_prompt,
        image_paths=image_paths,
        aspect_ratio=args.aspect_ratio,
        instructions=final_instructions,
        resolution=args.resolution.strip() or None,
        api_type="aigc2d",
        save_sub_dir=args.output_subdir,
        file_prefix=args.file_prefix,
        return_metadata=True,
    )

    if isinstance(result, dict):
        saved_files = result.get("saved_files") or []
        raw_path = os.path.join(bundle.output_dir, f"{args.file_prefix}_aigc2d_result.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"AIGC2D 原始结果: {raw_path}")
    else:
        saved_files = result or []

    if not saved_files:
        raise RuntimeError("AIGC2D 未返回可保存图片。")

    print("生成图片:")
    for path in saved_files:
        print(path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"运行失败: {exc}", file=sys.stderr)
        sys.exit(1)
