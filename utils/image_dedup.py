"""同目录图片按 SHA-256 去重：删掉重复副本，并给出清单。

## 为什么需要

`utils.image_identity` 记了背景：aigc2d 的 Gemini 网关**自己决定回 PNG 还是 JPEG**
（逐日横跳，客户端无法干预），而发布名过去把后缀写死成 `.png`、`promote_candidate`
又用 `copy2` 原样复制字节 —— 于是同目录会留下两张**字节完全相同**的文件：

    <hash>_<时间>-first-<token>.png          ← 其实是 JPEG
    <hash>_<时间>-first-<token>-aN_….jpg     ← 后缀没撒谎的那张

用户 2026-10-07 反馈「看着大小一样」，正是因为它们本来就是同一个字节流。
（发布那一侧已按用户选择「选中即 move 到发布名」改成**改名交付**，不再产生同内容副本；
本工具依旧负责**既有**重复的收敛。）

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

## `mode="identical"`：字节相同的重复（含"候选 + 发布副本"）

`misnamed` 只认「后缀撒谎」，而发布名后缀被纠正之后（以及今天之前的历史产物里），
同一张图会以**两个都合规**的名字并存（当时 `utils.first_image_audit.promote_candidate`
把选中的候选 `copy2` 成发布名，`mtime` 都一模一样）—— 那种重复 `misnamed` 收不了，要 `identical`：

    优先留下**发布队列（`data/publish/publish_queue.db`）在册的那一张**（已发/在队都算，
    删了会让发布器找不到文件），没有在册的就留**名字短的**（短名 = `<hash>_<时间>-first-<token>`
    发布名，`-aN` 候选名更长），最后按名字排序兜底。

用户 2026-10-07 定的规则：*如果有一张已经发了就留发了的这张，否则两张都没发就留名字短的*。
一条链里有两份都在发布库里（理论不该出现，`add_to_queue` 按 MD5 拦着）时**整组一张都不删**。

## 命令行

    python tools/image_dedup.py --dir data/20261006 --dir data/20261007          # 预演
    python tools/image_dedup.py --dir data/20261007 --apply                      # 真的删
    python tools/image_dedup.py --dir data/20261007 --apply --json 报告.json
    python tools/image_dedup.py --dir data/20261007 --mode identical             # 字节相同也收敛
    python tools/image_dedup.py --dir data/20261007 --mode identical --no-publish-db
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
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
MODE_IDENTICAL = "identical"
VALID_MODES = (MODE_MISNAMED, MODE_ALL, MODE_IDENTICAL)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PUBLISH_DB = os.path.join(BASE_DIR, "data", "publish", "publish_queue.db")

# 发布队列里算「已经发出去/正在发」的状态；`pending` 是在队但还没发，
# 一样不能删（删了发布器就找不到文件了），只是报告里分开写。
PUBLISHED_STATUS = ("uploading", "uploaded")


def _norm_path(path: str) -> str:
    text = str(path or "").replace("/", os.sep).replace("\\", os.sep)
    return os.path.normcase(os.path.abspath(text)) if text else ""


def publish_index(db_path: str | None = None) -> dict:
    """发布队列按「归一化路径」和「小写文件名」建的索引（只读，绝不改库）。

    返回 `{key: {"path", "pixiv_status", "chichipui_status", "uploaded", "status"}}`。
    库不存在 / 读不出来时返回空索引 —— 那就等价于"谁都没发过"，按名字长短裁决。
    """
    index: dict = {}
    path = str(db_path or DEFAULT_PUBLISH_DB)
    if not path or not os.path.isfile(path):
        return index
    conn = None
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT image_path, pixiv_status, chichipui_status FROM publish_queue"
        ).fetchall()
    except (sqlite3.Error, OSError):
        return index
    finally:
        if conn is not None:
            conn.close()
    for image_path, pixiv, chichi in rows:
        name = os.path.basename(str(image_path or "").replace("\\", "/"))
        if not name:
            continue
        uploaded = str(pixiv or "").lower() in PUBLISHED_STATUS \
            or str(chichi or "").lower() in PUBLISHED_STATUS
        row = {
            "path": image_path,
            "pixiv_status": pixiv,
            "chichipui_status": chichi,
            "uploaded": uploaded,
            "status": ("uploaded" if uploaded else "pending"),
        }
        index[_norm_path(image_path)] = row
        index.setdefault(name.lower(), row)      # 文件名兜底（发布名带时间+随机 token，不会撞）
    return index


def publish_state(row: dict, index: dict) -> dict:
    """这个文件在发布队列里的状态：`published` = 在册（已发或在队）。"""
    if not index:
        return {"published": False, "uploaded": False, "queue_status": ""}
    hit = index.get(_norm_path(row.get("path"))) or index.get(str(row.get("name") or "").lower())
    if not hit:
        return {"published": False, "uploaded": False, "queue_status": ""}
    return {"published": True, "uploaded": bool(hit.get("uploaded")),
            "queue_status": str(hit.get("status") or ""),
            "queue_path": hit.get("path")}


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
    """幸存者排序键：越小越优先被留下。

    次序（用户 2026-10-07 定的规则）：**发布队列在册的** > 后缀没撒谎的
    > 发布名（名字里带 `-first-`）> 名字短的 > 名字靠前。
    """
    name = str(row.get("name") or "")
    return (
        0 if row.get("published") else 1,           # 已发/在队的优先（删了发布器就找不到它）
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


def plan(directories, recursive: bool = False, mode: str = MODE_MISNAMED,
         use_publish_db: bool = True, db_path: str | None = None) -> dict:
    """算出要删哪些、留哪些。**不碰磁盘**，只读。

    返回 `{"mode", "groups", "removable", "skipped", "info"}`：

    - `removable`：可以删的重复副本（附幸存者 `kept`）；
    - `skipped`：本想删但没有合法幸存者（或发布库里对不上人）的（附 `reason`）；
    - `info`：全是真后缀的重复，`misnamed` 模式下只报告。

    `mode="identical"` 时先读发布队列（`use_publish_db=False` 或 `db_path` 可覆盖），
    让**在册的那一张**优先当幸存者；其余模式不读库，行为与从前完全一致。
    """
    mode = str(mode or MODE_MISNAMED).lower()
    if mode not in VALID_MODES:
        raise ValueError(f"mode 必须是 {VALID_MODES}，收到 {mode!r}")
    index = publish_index(db_path) if (mode == MODE_IDENTICAL and use_publish_db) else {}
    removable, skipped, info, groups = [], [], [], []
    for group in scan(directories, recursive=recursive):
        files = list(group["files"])
        if len(files) < 2:
            continue
        if index:
            files = [{**row, **publish_state(row, index)} for row in files]
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
        if index:
            record["kept_published"] = bool(survivor.get("published"))
            record["kept_queue_status"] = str(survivor.get("queue_status") or "")
            record["publish_checked"] = True

        if mode == MODE_IDENTICAL:
            # 字节相同就收敛，不再看"后缀有没有撒谎"
            cited = [row for row in ordered if row.get("published")]
            if len(cited) > 1:
                # 一组里两份都在发布库里（`add_to_queue` 按 MD5 拦着，理论上不该出现）
                # —— 分不清该留谁，整组一张都不删
                for row in ordered:
                    skipped.append({**row, "reason": "multiple_published",
                                    "note": "同内容有两份都在发布队列里，一张都不删"})
                record["skipped"] = "multiple_published"
                groups.append(record)
                continue
            if _is_misnamed_row(survivor):
                record["kept_misnamed"] = True
            for row in others:
                removable.append({
                    **row,
                    "duplicate_kind": "identical",
                    "kept": survivor["path"],
                    "kept_published": bool(survivor.get("published")),
                    "kept_queue_status": str(survivor.get("queue_status") or ""),
                    "reason": "与幸存者字节完全相同（SHA-256 一致）"
                              + ("；幸存者在发布队列里" if survivor.get("published") else ""),
                })
                record.setdefault("removed", []).append(row["name"])
            groups.append(record)
            continue

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
    """执行删除。删之前重新校验：**不在发布队列里**、文件还在、哈希没变、幸存者在且哈希一致。"""
    deleted, refused = [], []
    for row in plan_data.get("removable") or []:
        path = row.get("path") or ""
        kept = row.get("kept") or ""
        expected = row.get("sha256") or ""
        if row.get("published"):
            # 双保险：计划阶段就不该把在册的那份列为待删，真到这一步一律不删
            refused.append({**row, "reason": "该文件在发布队列里有记录，放弃删除"})
            continue
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
        description="按 SHA-256 找出同目录里字节相同的图片，删掉重复副本（附幸存者），并输出清单。",
    )
    parser.add_argument("--dir", dest="dirs", action="append", default=[],
                        help="要扫描的目录，可重复传；默认取 data/<今天>")
    parser.add_argument("--recursive", action="store_true", help="递归子目录（默认只看顶层）")
    parser.add_argument("--mode", choices=VALID_MODES, default=MODE_MISNAMED,
                        help="misnamed（默认）只删假后缀重复；"
                             "identical 连字节相同的真后缀重复也收敛（先留发布队列在册的那张，"
                             "否则留名字短的）；all = identical 但不看发布队列")
    parser.add_argument("--db", dest="db_path", default="",
                        help=f"发布队列 DB，默认 {DEFAULT_PUBLISH_DB}（只有 identical 模式读它）")
    parser.add_argument("--no-publish-db", action="store_true",
                        help="identical 模式也不读发布队列，纯按后缀/名字长短裁决")
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

    result = plan(dirs, recursive=args.recursive, mode=args.mode,
                  use_publish_db=not args.no_publish_db,
                  db_path=args.db_path or None)
    removable = result["removable"]

    print(f"扫描目录：{'、'.join(dirs)}   （{'递归' if args.recursive else '只看顶层'}）")
    print(f"模式：{args.mode}　｜　内容相同的分组：{len(result['groups'])}　｜　"
          f"可删的重复副本：{len(removable)}")
    if args.mode == MODE_IDENTICAL:
        db = args.db_path or DEFAULT_PUBLISH_DB
        print(f"发布队列：{'不读（--no-publish-db）' if args.no_publish_db else db}")

    if removable and not args.quiet:
        print("\n可删清单：")
        for index, row in enumerate(removable, 1):
            print(f"  {index:>3}. [{row['duplicate_kind']}] {row['name']}")
            print(f"       路径   : {row['path']}")
            print(f"       后缀声称: .{row['declared']}　真实字节: {row['actual']}"
                  f"　{_human_size(row['size'])}　sha256 {row['sha256'][:16]}…")
            print(f"       幸存者 : {os.path.basename(row['kept'])}"
                  + (f"　（发布队列：{row.get('kept_queue_status')}）" if row.get("kept_published") else ""))

    if result["skipped"]:
        print(f"\n⚠️ 想删但没敢删（{len(result['skipped'])} 个）：")
        for row in result["skipped"]:
            print(f"  - {row['name']}：{row.get('note') or row.get('reason')}")

    if result["info"]:
        print(f"\nℹ️ 后缀都真实、内容相同的重复（默认保留，{len(result['info'])} 个）：")
        for row in result["info"]:
            print(f"  - {row['name']}　（同 {os.path.basename(row['kept'])}）")

    misnamed_kept = [row for row in result["groups"] if row.get("kept_misnamed")]
    if misnamed_kept:
        print(f"\n⚠️ 幸存者自己后缀撒谎（扩展名与字节不符，{len(misnamed_kept)} 组）：")
        for row in misnamed_kept:
            print(f"  - {os.path.basename(row['kept'])}")

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
