# -*- coding: utf-8 -*-
"""把「记录里指向已删文件的路径」改指到实际在盘上的那一份。默认预演，加 --apply 才写。

## 场景

`tools/image_dedup.py` 删掉了「后缀撒谎」的假 PNG（`.png` 里装着 JPEG）。图像本身以真后缀
（`-aN_….jpg`）活着、没丢内容，但**台账里还写着被删的那个路径**，于是路径悬空。

本脚本**只改指向已删文件的路径字段**，别的一个字不动：

1. `*-first-image-audit.json` 的 `selected_file`
2. `publish_queue.db` 的 `image_path`

## 台账怎么修（不依赖外部清单，自证）

台账自己就记着答案：`selected_round` = 第几次尝试被选中，`candidates[].round` /
`candidates[].file` 列着每次尝试各自的文件。所以「实际在盘的那一份」就是
`round == selected_round` 的那个候选文件。改之前核一遍：**该文件必须真的存在**，
否则跳过并报告。找不到自证信息时退回 `--report` 的映射。

## 发布队列 DB

`image_path` 这一列没有自证信息，走 `--report`（`tools/image_dedup.py --json` 产出的
去重清单）里的「已删路径 → 幸存路径」映射。没有覆盖到的行一律不动。

## 用法

    # 预演（默认不写任何文件）
    python tools/repair_image_paths.py --dir data/20261006 --dir data/20261007

    # 真的写
    python tools/repair_image_paths.py --dir data/20261006 --dir data/20261007 --apply \
        --report data/test-result/dedup-20261006-07-report.json
"""
import argparse
import glob
import json
import os
import sqlite3
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

try:  # 控制台编码兜底（清单里有中文）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

DEFAULT_DB = os.path.join(BASE, "data", "publish", "publish_queue.db")
LEDGER_GLOB = "**/*-first-image-audit.json"


def _norm(path):
    """路径比较用：绝对化 + 大小写归一（Windows 下盘符大小写会不一致）。"""
    return os.path.normcase(os.path.abspath(str(path or "")))


def survivor_from_ledger(payload):
    """台账自证：`selected_round` 那次尝试的候选文件 = 实际选中的那一份。"""
    if not isinstance(payload, dict):
        return ""
    round_index = payload.get("selected_round")
    for row in payload.get("candidates") or []:
        if isinstance(row, dict) and row.get("round") == round_index:
            return str(row.get("file") or "")
    return ""


def load_report_mapping(report_path):
    """读去重清单的「已删路径 → 幸存路径」；幸存者必须真的在盘上。"""
    if not report_path or not os.path.isfile(report_path):
        return {}
    with open(report_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    mapping = {}
    for row in ((data.get("applied") or {}).get("deleted") or []):
        old, new = row.get("path"), row.get("kept")
        if old and new and os.path.isfile(new):
            mapping[_norm(old)] = str(new)
    return mapping


def _ledger_files(directory):
    """目录下所有首次审计台账（含子目录；测试隔离产物也会一起修）。"""
    pattern = os.path.join(str(directory), LEDGER_GLOB)
    return sorted(glob.glob(pattern, recursive=True))


def repair_ledgers(directories, mapping, apply=False):
    """修 `selected_file`。本来就指对的一张不动，找不到落地目标的只报告。"""
    changed, intact, skipped = [], [], []
    for directory in directories or []:
        for record_path in _ledger_files(directory):
            try:
                with open(record_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, ValueError) as exc:
                skipped.append({"record": record_path, "reason": f"台账读不出来：{exc}"})
                continue
            if not isinstance(payload, dict):
                skipped.append({"record": record_path, "reason": "台账根节点不是对象"})
                continue
            selected = str(payload.get("selected_file") or "")
            if not selected:
                skipped.append({"record": record_path, "reason": "台账没有 selected_file"})
                continue
            if os.path.isfile(selected):
                intact.append({"record": record_path, "path": selected})
                continue
            # 台账自证优先；它给的目标**必须真的在盘上**才算数，否则退回清单映射
            target = survivor_from_ledger(payload)
            if not target or not os.path.isfile(target):
                target = mapping.get(_norm(selected), "")
            if not target or not os.path.isfile(target):
                skipped.append({"record": record_path, "old": selected,
                                "reason": "selected_file 不存在，也找不到在盘的目标文件"})
                continue
            row = {"record": record_path, "old": selected, "new": target}
            if apply:
                payload["selected_file"] = target
                # 与 utils.first_image_audit.write_audit_record 的落盘口径保持一致
                temp = record_path + ".repair-tmp"
                with open(temp, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
                os.replace(temp, record_path)
            changed.append(row)
    return {"changed": changed, "intact": intact, "skipped": skipped}


def repair_publish_db(db_path, mapping, apply=False):
    """修 `publish_queue.image_path`：只改能映射到在盘文件的那些行。"""
    if not db_path:
        return {"changed": [], "intact": [], "skipped": [], "present": False}
    if not os.path.isfile(db_path):
        return {"changed": [], "intact": [],
                "skipped": [{"db": db_path, "reason": "数据库不存在，跳过"}], "present": False}
    changed, intact, skipped = [], [], []
    connection = sqlite3.connect(db_path)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(publish_queue)")}
        if "image_path" not in columns:
            return {"changed": [], "intact": [],
                    "skipped": [{"db": db_path, "reason": "表里没有 image_path 列"}], "present": True}
        rows = connection.execute("SELECT uuid, image_path FROM publish_queue").fetchall()
        for item_uuid, image_path in rows:
            stored = str(image_path or "")
            if not stored:
                continue
            if os.path.isfile(stored):
                intact.append({"uuid": item_uuid, "path": stored})
                continue
            target = mapping.get(_norm(stored), "")
            if not target or not os.path.isfile(target):
                skipped.append({"uuid": item_uuid, "old": stored,
                                "reason": "image_path 不存在且不在映射里"})
                continue
            changed.append({"uuid": item_uuid, "old": stored, "new": target})
            if apply:
                connection.execute(
                    "UPDATE publish_queue SET image_path = ?, "
                    "updated_at = datetime('now','localtime') WHERE uuid = ?",
                    [target, item_uuid])
        if apply and changed:
            connection.commit()
    finally:
        connection.close()
    return {"changed": changed, "intact": intact, "skipped": skipped, "present": True}


def build_parser():
    parser = argparse.ArgumentParser(
        prog="repair_image_paths",
        description="把审计台账与发布队列里指向已删文件的路径，改指到实际在盘的那一份。",
    )
    parser.add_argument("--dir", dest="dirs", action="append", default=[],
                        help="要修的记录目录（递归找 *-first-image-audit.json），可重复传")
    parser.add_argument("--report", default="", help="去重清单 JSON（已删路径 → 幸存路径）")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"发布队列 DB，默认 {DEFAULT_DB}")
    parser.add_argument("--no-db", action="store_true", help="完全不碰发布队列 DB")
    parser.add_argument("--apply", action="store_true", help="真的写回；不加只预演")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    directories = [d for d in (args.dirs or []) if d]
    missing = [d for d in directories if not os.path.isdir(d)]
    for directory in missing:
        print(f"⚠️ 目录不存在，跳过：{directory}")
    directories = [d for d in directories if os.path.isdir(d)]

    mapping = load_report_mapping(args.report)
    print(f"扫描目录：{'、'.join(directories) or '（无）'}")
    print(f"映射条目：{len(mapping)}"
          + (f"（来自 {args.report}）" if mapping else "（未提供 --report 或清单为空）"))

    ledgers = repair_ledgers(directories, mapping, apply=args.apply)
    print(f"\n【审计台账 selected_file】需要改 {len(ledgers['changed'])} 条，"
          f"本来就对 {len(ledgers['intact'])} 条")
    for row in ledgers["changed"]:
        print(f"  - {os.path.basename(row['record'])}")
        print(f"      旧: {row['old']}")
        print(f"      新: {row['new']}")
    for row in ledgers["skipped"]:
        print(f"  ⚠️ 跳过 {os.path.basename(row.get('record', ''))}：{row['reason']}")

    db_result = None
    if not args.no_db:
        db_result = repair_publish_db(args.db, mapping, apply=args.apply)
        if db_result["present"]:
            print(f"\n【发布队列 image_path】需要改 {len(db_result['changed'])} 条，"
                  f"本来就对 {len(db_result['intact'])} 条")
            for row in db_result["changed"]:
                print(f"  - {row['uuid'][:8]}…  旧: {row['old']}\n            新: {row['new']}")
            for row in db_result["skipped"]:
                print(f"  ⚠️ 跳过 {str(row.get('uuid', ''))[:8]}…：{row['reason']}")
        else:
            print(f"\n【发布队列】{db_result['skipped'][0]['reason'] if db_result['skipped'] else '跳过'}")

    total = len(ledgers["changed"]) + (len(db_result["changed"]) if db_result else 0)
    if not args.apply and total:
        print(f"\n（预演模式，未改动任何文件；要真的写加 --apply）")
    elif args.apply:
        print(f"\n✅ 已改写 {total} 条记录")
    else:
        print("\n没有需要改的记录。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
