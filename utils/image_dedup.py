"""同目录图片按 SHA-256 去重：删掉「扩展名撒谎」的重复副本，并给出清单。

## 为什么需要

`utils.image_identity` 记了背景：aigc2d 的 Gemini 网关**自己决定回 PNG 还是 JPEG**
（逐日横跳，客户端无法干预），而发布名过去把后缀写死成 `.png`、`promote_candidate`
又用 `copy2` 原样复制字节 —— 于是同目录会留下两张**字节完全相同**的文件：

    <hash>_<时间>-first-<token>.png          ← 其实是 JPEG
    <hash>_<时间>-first-<token>-aN_….jpg     ← 后缀没撒谎的那张

用户 2026-10-07 反馈「看着大小一样」，正是因为它们本来就是同一个字节流。

这些假 PNG 不只是难看，还会**真的坏事**：

- `publish_server.find_image_for_json` 反向找图时要求"同 hash 前缀的图片恰好一张"，
  两张就返回 None（"无法从结果 JSON 确定对应原图"）；
- `publish_server.add_to_queue` 按 MD5 判定"内容已存在"，第二张会被拒绝入队。

## 规则（保守：宁可漏删，不可误删）

1. 只处理 `IMAGE_EXTENSIONS` 里的文件，**按 (目录, SHA-256) 分组**（不跨目录合并）；
2. **只删「扩展名撒谎」的那份**，且同组里必须存在一张**后缀没撒谎**的同哈希文件当幸存者
   —— 找不到幸存者就一张都不删（宁可留着也不猜）；
3. 幸存者次序：后缀没撒谎 > 发布名（名字里带 `-first-`，队列与投稿记录指着它）
   > 名字短的 > 名字靠前的；
4. 全组都是真后缀的重复（例如两张 `.jpg`）默认**只报告不删**，`mode="all"` 才收；
5. 每组至少留一张，**从不删空一个哈希组**；
6. 真删之前重新校验一次：文件还在、哈希没变、幸存者还在且哈希一致，三条缺一不删。

## 命令行

    python tools/image_dedup.py --dir data/20261006 --dir data/20261007          # 预演
    python tools/image_dedup.py --dir data/20261007 --apply                      # 真的删
    python tools/image_dedup.py --dir data/20261007 --apply --json 报告.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from utils.image_identity import (
    IMAGE_EXTENSIONS,
    canonical_extension,
    file_sha256,
    sniff_format,
    declared_format,
)

MODE_MISNAMED = "misnamed"
MODE_ALL = "all"
VALID_MODES = (MODE_MISNAMED, MODE_ALL)


def _iter_images(directory: str, recursive: bool = False):
    directory = os.path.abspath(str(directory))
    if not os.path.isdir(directory):
        return
    if recursive:
        for root, _dirs, files in os.walk(directory):
            for name in files:
                if name.lower().endswith(IMAGE_EXTENSIONS):
                    yield os.path.join(root, name)
        return
    try:
        entries = os.listdir(directory)
    except OSError:
        return
    for name in entries:
        if not name.lower().endswith(IMAGE_EXTENSIONS):
            continue
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            yield path


def _is_misnamed_row(row: dict) -> bool:
    return bool(row.get("declared") and row.get("actual") and row["declared"] != row["actual"])


def _survivor_key(row: dict):
    """幸存者排序键：越小越优先被留下。"""
    name = str(row.get("name") or "")
    return (
        0 if not _is_misnamed_row(row) else 1,      # 后缀没撒谎的优先
        0 if "-first-" in name else 1,              # 发布名（记录指着它）优先
        len(name),
        name,
    )


def scan(directories, recursive: bool = False) -> list:
    """按 (目录, SHA-256) 分组的扫描结果。返回 [{"key", "directory", "sha256", "files":[...]}]。"""
    if isinstance(directories, (str, os.PathLike)):
        directories = [directories]
    groups = {}
    for directory in directories or []:
        for path in sorted(_iter_images(directory, recursive=recursive)):
            digest = file_sha256(path)
            if not digest:
                continue
            row = {
                "path": path,
                "name": os.path.basename(path),
                "directory": os.path.dirname(path),
                "size": os.path.getsize(path),
                "declared": declared_format(path),
                "actual": sniff_format(path),
                "sha256": digest,
            }
            groups.setdefault((row["directory"], digest), []).append(row)
    return [{"key": list(key), "directory": key[0], "sha256": key[1], "files": files}
            for key, files in sorted(groups.items())]


def plan(directories, recursive: bool = False, mode: str = MODE_MISNAMED) -> dict:
    """算出要删哪些、留哪些。**不碰磁盘**，只读。

    返回 `{"mode", "groups", "removable", "skipped", "info"}`：

    - `removable`：可以删的「假后缀」重复（附幸存者 `kept`）；
    - `skipped`：本想删但没有合法幸存者的（附 `reason`）；
    - `info`：全是真后缀的重复，默认只报告。
    """
    mode = str(mode or MODE_MISNAMED).lower()
    if mode not in VALID_MODES:
        raise ValueError(f"mode 必须是 {VALID_MODES}，收到 {mode!r}")
    removable, skipped, info, groups = [], [], [], []
    for group in scan(directories, recursive=recursive):
        files = list(group["files"])
        if len(files) < 2:
            continue
        ordered = sorted(files, key=_survivor_key)
        survivor = ordered[0]
        others = ordered[1:]
        truthful = [row for row in files if not _is_misnamed_row(row)]
        record = {
            "directory": group["directory"],
            "sha256": group["sha256"],
            "size": survivor["size"],
            "kept": survivor["path"],
            "files": [row["name"] for row in ordered],
        }
        for row in others:
            if not _is_misnamed_row(row):
                # 后缀没撒谎的重复：默认只报告
                if mode == MODE_ALL:
                    removable.append({
                        **row,
                        "duplicate_kind": "identical",
                        "kept": survivor["path"],
                        "reason": "内容与幸存者完全相同（后缀都真实）",
                    })
                    record.setdefault("removed", []).append(row["name"])
                else:
                    info.append({**row, "kept": survivor["path"],
                                 "note": "后缀都真实，默认保留；要一并收敛用 --mode all"})
                continue
            # 后缀撒谎的那份：必须有真后缀的幸存者才敢删
            if not truthful:
                skipped.append({**row, "reason": "no_truthful_twin",
                                "note": "整组后缀都在撒谎，找不到可信的幸存者，一张都不删"})
                continue
            removable.append({
                **row,
                "duplicate_kind": "misnamed",
                "kept": survivor["path"],
                "reason": "与幸存者字节相同，且自己是假后缀",
            })
            record.setdefault("removed", []).append(row["name"])
        groups.append(record)
    return {"mode": mode, "groups": groups, "removable": removable,
            "skipped": skipped, "info": info}


def apply_plan(plan_data: dict) -> dict:
    """执行删除。删之前重新校验：文件还在、哈希没变、幸存者在且哈希一致。"""
    deleted, refused = [], []
    for row in plan_data.get("removable") or []:
        path = row.get("path") or ""
        kept = row.get("kept") or ""
        expected = row.get("sha256") or ""
        if not os.path.isfile(path):
            refused.append({**row, "reason": "文件已不在"})
            continue
        if file_sha256(path) != expected:
            refused.append({**row, "reason": "哈希在计划之后变了，放弃删除"})
            continue
        if not kept or not os.path.isfile(kept) or file_sha256(kept) != expected:
            refused.append({**row, "reason": "幸存者缺失或哈希不一致，放弃删除"})
            continue
        try:
            os.remove(path)
        except OSError as exc:
            refused.append({**row, "reason": f"删除失败：{exc}"})
            continue
        deleted.append({**row, "freed": row.get("size") or 0})
    return {
        "deleted": deleted,
        "refused": refused,
        "freed_bytes": sum(int(row.get("freed") or 0) for row in deleted),
    }


def _human_size(num: int) -> str:
    value = float(num or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.2f} GB"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image_dedup",
        description="按 SHA-256 找出同目录里字节相同的图片，删掉「扩展名撒谎」的假后缀重复，并输出清单。",
    )
    parser.add_argument("--dir", dest="dirs", action="append", default=[],
                        help="要扫描的目录，可重复传；默认取 data/<今天>")
    parser.add_argument("--recursive", action="store_true", help="递归子目录（默认只看顶层）")
    parser.add_argument("--mode", choices=VALID_MODES, default=MODE_MISNAMED,
                        help="misnamed（默认）只删假后缀重复；all 连真后缀重复也一起收敛")
    parser.add_argument("--apply", action="store_true",
                        help="真的删除；不加只预演（默认预演，不碰磁盘）")
    parser.add_argument("--json", dest="json_out", default="",
                        help="把完整清单写成 JSON 到该路径")
    parser.add_argument("--quiet", action="store_true", help="只打印汇总，不逐条列")
    return parser


def main(argv=None) -> int:
    import datetime
    args = build_parser().parse_args(argv)
    dirs = [d for d in (args.dirs or []) if d] or \
        [os.path.join("data", datetime.datetime.now().strftime("%Y%m%d"))]
    missing = [d for d in dirs if not os.path.isdir(d)]
    for directory in missing:
        print(f"⚠️ 目录不存在，跳过：{directory}")

    result = plan(dirs, recursive=args.recursive, mode=args.mode)
    removable = result["removable"]

    print(f"扫描目录：{'、'.join(dirs)}   （{'递归' if args.recursive else '只看顶层'}）")
    print(f"模式：{args.mode}　｜　内容相同的分组：{len(result['groups'])}　｜　"
          f"可删的重复副本：{len(removable)}")

    if removable and not args.quiet:
        print("\n可删清单：")
        for index, row in enumerate(removable, 1):
            print(f"  {index:>3}. [{row['duplicate_kind']}] {row['name']}")
            print(f"       路径   : {row['path']}")
            print(f"       后缀声称: .{row['declared']}　真实字节: {row['actual']}"
                  f"　{_human_size(row['size'])}　sha256 {row['sha256'][:16]}…")
            print(f"       幸存者 : {os.path.basename(row['kept'])}")

    if result["skipped"]:
        print(f"\n⚠️ 想删但没敢删（{len(result['skipped'])} 个）：")
        for row in result["skipped"]:
            print(f"  - {row['name']}：{row['note']}")

    if result["info"]:
        print(f"\nℹ️ 后缀都真实、内容相同的重复（默认保留，{len(result['info'])} 个）：")
        for row in result["info"]:
            print(f"  - {row['name']}　（同 {os.path.basename(row['kept'])}）")

    if not removable:
        print("\n没有需要删除的重复副本。")

    freed = 0
    if args.apply and removable:
        outcome = apply_plan(result)
        freed = outcome["freed_bytes"]
        print(f"\n✅ 已删除 {len(outcome['deleted'])} 个重复副本，释放 {_human_size(freed)}")
        for row in outcome["refused"]:
            print(f"  ⚠️ 放弃删除 {row['name']}：{row['reason']}")
        result["applied"] = outcome
    elif removable:
        print("\n（预演模式，未改动任何文件；要真的删加 --apply）")

    if args.json_out:
        payload = {**result, "directories": dirs, "apply": bool(args.apply), "freed_bytes": freed}
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\n📝 清单已写入：{args.json_out}")
    return 0


if __name__ == "__main__":       # pragma: no cover - 便于 `python -m utils.image_dedup`
    sys.exit(main())
