# -*- coding: utf-8 -*-
"""把历史目录里「没被交付的候选」改名成 `rejected-<原名>`，让同 hash 前缀恢复唯一。

配套 `utils/rejected_candidates.py`（判定与计划）与 `tools/repair_image_paths.py`（把台账里
指向旧名的 `candidates[].file` 改到新名）。**默认预演，不删任何文件**。

例：
  # 预演：只列要改哪些
  python tools/rename_rejected_candidates.py --dir data/20261007

  # 真的改，并把清单写成 JSON（供 repair_image_paths 消费）
  python tools/rename_rejected_candidates.py --dir data/20261007 --apply \
      --json data/test-result/20261007/rejected-rename/report.json

  # 台账跟着改
  python tools/repair_image_paths.py --dir data/20261007 --report <上面那份> --apply
"""
import datetime
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

try:  # 控制台编码兜底（清单里有中文）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from utils.rejected_candidates import (  # noqa: E402
    DEFAULT_PREFIX,
    DEFAULT_PUBLISH_DB,
    apply_plan,
    plan,
)


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(
        prog="rename_rejected_candidates",
        description="把历史目录里没被交付的 -aN 候选改名成 rejected-<原名>，"
                    "让 publish_server 的「同 hash 前缀恰好一张」判据恢复。",
    )
    parser.add_argument("--dir", dest="dirs", action="append", default=[],
                        help="要整理的目录，可重复传；默认取 data/<今天>")
    parser.add_argument("--recursive", action="store_true", help="递归子目录（默认只看顶层）")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX,
                        help=f"改名用的前缀，默认 {DEFAULT_PREFIX!r}")
    parser.add_argument("--db", dest="db_path", default="",
                        help=f"发布队列 DB，默认 {DEFAULT_PUBLISH_DB}")
    parser.add_argument("--no-publish-db", action="store_true",
                        help="不读发布队列（在册的文件就失去保护，一般别用）")
    parser.add_argument("--apply", action="store_true", help="真的改名；不加只预演")
    parser.add_argument("--json", dest="json_out", default="", help="把清单写成 JSON")
    parser.add_argument("--quiet", action="store_true", help="只打印汇总")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    dirs = [d for d in (args.dirs or []) if d] or \
        [os.path.join("data", datetime.datetime.now().strftime("%Y%m%d"))]
    for directory in dirs:
        if not os.path.isdir(directory):
            print(f"⚠️ 目录不存在，跳过：{directory}")

    result = plan(dirs, db_path=args.db_path or None, prefix=args.prefix,
                  recursive=args.recursive, use_publish_db=not args.no_publish_db)
    renames = result["renames"]

    print(f"扫描目录：{'、'.join(dirs)}   （{'递归' if args.recursive else '只看顶层'}）")
    print(f"同 hash 前缀 ≥2 张的分组：{result['groups']}　｜　要改名的非交付候选：{len(renames)}")
    print(f"发布队列：{'不读（--no-publish-db）' if args.no_publish_db else result['publish_db'] or '（未提供）'}")
    print(f"改名前缀：{result['prefix']}")

    if renames and not args.quiet:
        print("\n改名清单：")
        for index, row in enumerate(renames, 1):
            print(f"  {index:>3}. {os.path.basename(row['old'])}")
            print(f"        → {os.path.basename(row['new'])}")
            print(f"        hash {row['hash']} / 第 {row['round']} 轮   sha256 {str(row['sha256'])[:16]}…")

    fallback = [row for row in result["kept"] if row["why"] == "last-one-fallback"]
    if fallback:
        print(f"\n⚠️ 整组都是非交付候选，按 mtime 留了最新的一张（{len(fallback)} 组）：")
        for row in fallback:
            print(f"  - {os.path.basename(row['path'])}")

    kept_why = {}
    for row in result["kept"]:
        kept_why[row["why"]] = kept_why.get(row["why"], 0) + 1
    if kept_why and not args.quiet:
        print("\n保持原样（按原因计）： " + "　".join(f"{k}={v}" for k, v in sorted(kept_why.items())))

    for row in result["skipped"]:
        print(f"⚠️ 跳过 {row.get('path') or row.get('directory')}：{row['reason']}")

    if not renames:
        print("\n没有需要改名的候选。")

    applied = {}
    if args.apply and renames:
        applied = apply_plan(result)
        print(f"\n✅ 已改名 {len(applied['renamed'])} 个（{'、'.join(os.path.basename(r['old']) for r in applied['renamed'][:3])}"
              f"{'…' if len(applied['renamed']) > 3 else ''}）")
        for row in applied["refused"]:
            print(f"  ⚠️ 放弃改名 {os.path.basename(row.get('old', ''))}：{row['reason']}")
    elif renames:
        print("\n（预演模式，未改动任何文件；要真的改加 --apply）")

    if args.json_out:
        payload = {**result, "directories": dirs, "apply": bool(args.apply),
                   "applied": applied}
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"\n📝 清单已写入：{args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
