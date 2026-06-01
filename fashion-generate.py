import argparse
import json
import os
import sys
from datetime import datetime

from modules.fashion_collection.collection_service import FashionCollectionService
from modules.fashion_collection.make_pic_bridge import export_bundle_manifest, export_make_pic_state
from modules.fashion_collection.models import PART_DRESS, PART_SHOES, PART_SOCKS
from modules.others.api_backend import generate_image_aigc2d


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="抓取一套服饰并调用 AIGC2D 生成 1 张少女图片")
    parser.add_argument("--site", choices=["lolibrary", "wear"], default="wear", help="采集站点")
    parser.add_argument("--brand", default="", help="品牌 slug/key；wear 可留空")
    parser.add_argument("--pages", type=int, default=1, help="扫描页数")
    parser.add_argument("--parts", nargs="+", default=[PART_DRESS, PART_SHOES, PART_SOCKS], help="目标部位")
    parser.add_argument("--aspect-ratio", default="2:3", help="输出比例")
    parser.add_argument("--resolution", default="", help="覆盖输出分辨率，如 2K；为空时跟配置")
    parser.add_argument("--prompt", default="请生成一位可爱梦幻的少女，全身像，站姿自然，画面干净，突出服装整体搭配感。", help="用户主提示词")
    parser.add_argument("--instructions", default="请严格参考输入的服饰图片完成一位少女角色的穿搭组合，保持连衣裙、鞋子、袜子的款式与颜色协调一致，输出日系少女插画风格。", help="附加系统风格/约束")
    parser.add_argument("--output-subdir", default="fashion-generate", help="生成图片保存子目录")
    parser.add_argument("--file-prefix", default="fashion_combo", help="输出文件前缀")
    parser.add_argument("--export-state", action="store_true", help="同时导出 make-pic 状态文件")
    return parser.parse_args()


def main():
    args = parse_args()
    service = FashionCollectionService()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_DIR, "data", "fashion-collector", args.site, timestamp)

    bundle = service.collect_bundle(
        site_key=args.site,
        brand_slug=args.brand,
        output_dir=output_dir,
        max_pages=max(1, int(args.pages)),
        preferred_parts=list(args.parts),
    )
    if bundle.missing_parts:
        print(f"缺失部位: {', '.join(bundle.missing_parts)}", file=sys.stderr)
    if not bundle.assets:
        raise RuntimeError("未采集到可用素材，无法继续生成。")

    manifest_path = export_bundle_manifest(bundle, bundle.output_dir)
    print(f"采集清单: {manifest_path}")

    if args.export_state:
        state_path = export_make_pic_state(
            bundle,
            state_path=os.path.join(BASE_DIR, "cache", "last_state.json"),
            instructions=args.instructions,
            extra_prompt=args.prompt,
            aspect_ratio=args.aspect_ratio,
        )
        print(f"已导出 make-pic 状态文件: {state_path}")

    prompt_lines = [args.prompt.strip(), "", "服饰参考清单："]
    for asset in bundle.assets:
        prompt_lines.append(f"- {asset.part}: {asset.item.title}")
        if asset.prompt_hint:
            prompt_lines.append(f"  要点: {asset.prompt_hint}")
    final_prompt = "\n".join(line for line in prompt_lines if line is not None).strip()
    image_paths = [asset.local_path for asset in bundle.assets if os.path.isfile(asset.local_path)]

    result = generate_image_aigc2d(
        prompt=final_prompt,
        image_paths=image_paths,
        aspect_ratio=args.aspect_ratio,
        instructions=args.instructions.strip(),
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
