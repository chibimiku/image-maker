"""Standalone inventory/contact-sheet/validation helper; Python 3.10+ and Pillow."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
from manifest_core import make_inventory, load_manifest


def contact_sheets(document, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = document["images"]
    root = Path(document["source_root"])
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for start in range(0, len(records), 24):
        sheet = Image.new("RGB", (960, 1440), "#eeeeee")
        draw = ImageDraw.Draw(sheet)
        for offset, record in enumerate(records[start:start + 24]):
            x, y = (offset % 4) * 240, (offset // 4) * 240
            if record["readable"]:
                with Image.open(root / record["path"]) as image:
                    thumb = ImageOps.exif_transpose(image).convert("RGB")
                    thumb.thumbnail((224, 208))
                sheet.paste(thumb, (x + (240 - thumb.width) // 2, y + 4))
            else:
                draw.text((x + 8, y + 60), "UNREADABLE", fill="red", font=font)
            # 序号对应 images 列表的 1-based 索引；完整相对路径在清单里。
            label = record["path"].encode("ascii", "replace").decode()[:26]
            draw.text((x + 8, y + 214), f"{start + offset + 1:04d}  {label}", fill="black", font=font)
        target = directory / f"contact-{start // 24 + 1:03d}.jpg"
        if target.exists():
            raise ValueError(f"联系表已存在，请更换输出目录: {target}")
        sheet.save(target, quality=92)
    print(f"联系表：{directory.resolve()}；索引对应清单 images 顺序")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inv = commands.add_parser("inventory")
    inv.add_argument("source_root")
    inv.add_argument("--style-name", required=True)
    inv.add_argument("--count", type=int, default=12)
    inv.add_argument("--recursive", action="store_true")
    inv.add_argument("--output", required=True)
    inv.add_argument("--sheets")
    valid = commands.add_parser("validate")
    valid.add_argument("manifest")
    valid.add_argument("--source-root")
    args = parser.parse_args()
    try:
        if args.command == "inventory":
            if not 2 <= args.count <= 100:
                raise ValueError("count 须为 2–100")
            document = make_inventory(args.source_root, args.style_name, args.count, args.recursive)
            path = Path(args.output)
            if path.exists():
                raise ValueError("输出清单已存在，请换一个名称")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
            if args.sheets:
                contact_sheets(document, args.sheets)
            print(f"待审 {len(document['images'])} 张；draft 不可直接训练：{path.resolve()}")
        else:
            result = load_manifest(args.manifest, args.source_root)
            print(f"校验通过：{result['style_name']}；已审 {result['reviewed_count']}，入选 {result['selected_count']}")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"清单错误：{exc}\n")


if __name__ == "__main__":
    main()
