"""画风筛图清单：生成待审清单、验证 agent 结果、复制训练数据。"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.image_analysis.style_dataset import make_inventory, load_manifest, materialize_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory", help="生成待视觉审查清单（不可直接训练）")
    inventory.add_argument("source_root")
    inventory.add_argument("--style-name", required=True)
    inventory.add_argument("--count", type=int, default=12)
    inventory.add_argument("--recursive", action="store_true")
    inventory.add_argument("--output", required=True)
    for name in ("validate", "copy"):
        command = commands.add_parser(name)
        command.add_argument("manifest")
        command.add_argument("--source-root", help="跨机器时重定位源目录，仍校验哈希和完整覆盖")
        if name == "copy":
            command.add_argument("--output-root", default="data/style-datasets")
    args = parser.parse_args()
    try:
        if args.command == "inventory":
            if not 2 <= args.count <= 100:
                raise ValueError("--count 须为 2–100")
            document = make_inventory(args.source_root, args.style_name, args.count, args.recursive)
            output = Path(args.output)
            if output.exists():
                raise ValueError("输出文件已存在，请换一个名称")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"已生成待审清单，共 {len(document['images'])} 张：{output.resolve()}")
        else:
            validated = load_manifest(args.manifest, args.source_root)
            print(f"校验通过：{validated['style_name']}；已审 {validated['reviewed_count']}，入选 {validated['selected_count']}")
            if args.command == "copy":
                print(json.dumps(materialize_dataset(validated, args.output_root), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"筛图清单错误：{exc}\n")


if __name__ == "__main__":
    main()
