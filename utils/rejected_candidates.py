"""把「没被交付的候选」从任务 hash 前缀里摘出来（历史产物的手工整理，不删文件）。

## 为什么

`publish_server.find_image_for_json()` 反向找图时要求「同 hash 前缀的图片**恰好一张**」
（`publish_server.py:233`），多轮任务里被审计打回的 `-aN` 候选留在同一目录就变成两张、
返回 `None`，于是「拖投稿 JSON 入队」报"无法从结果 JSON 确定对应原图"。

发布链本身已经在 2026-10-07 改成「选中即 move 到发布名」（`utils.first_image_audit.
promote_candidate`），**新产物不会再产生同内容副本**；但**历史目录**里被拒的候选还在原地，
而且 20261007 早上那次「假 PNG 去重」把发布名删掉之后，有些任务顶层只剩 `-aN` 文件。
本模块负责把这些**非交付**的历史候选改名成 `rejected-<原名>`：

- `e.split("_")[0]` 不再等于任务 hash → 不再参与「恰好一张」的判据；
- 文件名仍带着原名的全部信息（hash / 时间 / `-aN` / token 都在），随时去掉前缀就能还原；
- `publish_server.find_metadata_json()` 还能认出来（它除了首字段外还会找名字里**独立的 8 位 hex**）；
- **一个文件都不删**。

## 「非交付」怎么判定（三查，缺一不动）

候选 = 名字形如 `<hash>_<时间>-first-<token>-a<N>_…` 的图片。满足以下**任一**条件就保持原样：

1. **在发布队列里**（`data/publish/publish_queue.db`，任何状态）—— 发布器按 `image_path`
   发图，动了就发不出去；
2. **是某份审计台账的 `selected_file`** —— 那是这次任务真正交付的那张（早上那次修复后，
   很多台账的 `selected_file` 正指着 `-aN` 文件）；
3. 同 hash 前缀只剩它一张（搬走就等于让这个 hash 一张都匹配不到，反而更糟）。

另外**每组至少留一张**匹配前缀的文件：万一整组都是"非交付"候选，就按 mtime 最新的那张留着。

## 用法

    python tools/rename_rejected_candidates.py --dir data/20261007            # 预演
    python tools/rename_rejected_candidates.py --dir data/20261007 --apply \
        --json data/test-result/.../report.json
    python tools/repair_image_paths.py --dir data/20261007 --report <上面那份> --apply

改名之后**台账里的路径要跟着改**（`candidates[].file` 会指向旧名），所以配套
`tools/repair_image_paths.py`（它现在也认 `renames` 清单、也会修 `candidates[].file`）。
"""

from __future__ import annotations

import glob
import json
import os
import re
import sqlite3

from utils.image_identity import IMAGE_EXTENSIONS, file_sha256

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PUBLISH_DB = os.path.join(BASE_DIR, "data", "publish", "publish_queue.db")
DEFAULT_PREFIX = "rejected-"
LEDGER_GLOB = "**/*-first-image-audit.json"

# `<hash>_<时间>-first-<token>-a<N>_….<ext>`：只有带 `-aN` 的才是"候选"，发布名不带
CANDIDATE_RE = re.compile(
    r"^(?P<hash>[0-9a-f]{8})_\d{6}-first-[0-9a-f]{6}-a(?P<round>\d+)_", re.IGNORECASE)


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(str(path or "")))


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


def published_paths(db_path: str | None = None) -> tuple:
    """发布队列在册的文件：返回 `(归一化绝对路径集合, 小写文件名集合)`（只读）。"""
    paths, names = set(), set()
    path = str(db_path or DEFAULT_PUBLISH_DB)
    if not path or not os.path.isfile(path):
        return paths, names
    connection = None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rows = connection.execute("SELECT image_path FROM publish_queue").fetchall()
    except (sqlite3.Error, OSError):
        return paths, names
    finally:
        if connection is not None:
            connection.close()
    for (image_path,) in rows:
        text = str(image_path or "")
        if not text:
            continue
        paths.add(_norm(text))
        names.add(os.path.basename(text.replace("\\", "/")).lower())
    return paths, names


def ledger_selected(directory: str) -> dict:
    """目录下（含子目录）所有台账的 `selected_file`：`{归一化路径: 台账路径}`。"""
    selected = {}
    for record_path in sorted(glob.glob(os.path.join(str(directory), LEDGER_GLOB),
                                         recursive=True)):
        try:
            with open(record_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        path = str(payload.get("selected_file") or "")
        if path and os.path.isfile(path):
            selected.setdefault(_norm(path), record_path)
    return selected


def _already_prefixed(path: str, prefix: str) -> bool:
    return os.path.basename(path).startswith(prefix)


def plan(directories, db_path: str | None = None, prefix: str = DEFAULT_PREFIX,
         recursive: bool = False, use_publish_db: bool = True) -> dict:
    """算出要把哪些历史候选改名成 `<prefix><原名>`。**不碰磁盘**，只读。

    返回 `{"renames": [{"old", "new", "hash", "round", "sha256", "size", "why"}],
    "kept": [...], "skipped": [...], "groups": N}`。
    """
    prefix = str(prefix or DEFAULT_PREFIX)
    if isinstance(directories, (str, os.PathLike)):
        directories = [directories]
    published, published_names = published_paths(db_path) if use_publish_db else (set(), set())
    renames, kept, skipped, groups = [], [], [], 0
    seen = set()

    for directory in directories or []:
        directory = os.path.abspath(str(directory))
        if not os.path.isdir(directory):
            skipped.append({"directory": directory, "reason": "目录不存在"})
            continue
        selected = ledger_selected(directory)
        by_hash = {}
        for path in _iter_images(directory, recursive=recursive):
            if path in seen:
                continue
            name = os.path.basename(path)
            if _already_prefixed(path, prefix):
                continue                      # 已经摘出去过的，不再动
            match = CANDIDATE_RE.match(name)
            key = (name.split("_")[0] or "").lower() if not match else match.group("hash").lower()
            if not key:
                continue
            by_hash.setdefault(key, []).append(path)

        for key, paths in sorted(by_hash.items()):
            if len(paths) < 2:
                # 同前缀只有一张：动它只会让这个 hash 一张都匹配不到
                for path in paths:
                    kept.append({"path": path, "why": "only-one"})
                continue
            groups += 1
            movable = []
            for path in sorted(paths):
                name = os.path.basename(path)
                if not CANDIDATE_RE.match(name):
                    kept.append({"path": path, "why": "publish-name"})
                    continue
                if _norm(path) in published or name.lower() in published_names:
                    kept.append({"path": path, "why": "published"})
                    continue
                if _norm(path) in selected:
                    kept.append({"path": path, "why": "ledger-selected",
                                 "ledger": selected[_norm(path)]})
                    continue
                movable.append(path)

            # 每组至少留一张匹配前缀的：全是"非交付"时按 mtime 最新的那张留
            if len(movable) == len(paths) and movable:
                movable.sort(key=lambda p: os.path.getmtime(p))
                newest = movable.pop()
                kept.append({"path": newest, "why": "last-one-fallback"})

            for path in movable:
                new_path = os.path.join(os.path.dirname(path), prefix + os.path.basename(path))
                if os.path.exists(new_path):
                    skipped.append({"path": path, "new": new_path,
                                    "reason": "目标名已存在，放弃改名"})
                    continue
                renames.append({
                    "old": path,
                    "new": new_path,
                    "hash": key,
                    "round": CANDIDATE_RE.match(os.path.basename(path)).group("round"),
                    "sha256": file_sha256(path),
                    "size": os.path.getsize(path),
                    "why": "同 hash 前缀里没被交付的候选（多轮任务里被审计打回的那一轮）",
                })
                seen.add(path)

    return {"prefix": prefix, "groups": groups, "renames": renames,
            "kept": kept, "skipped": skipped,
            "publish_db_used": bool(use_publish_db),
            "publish_db": str(db_path or DEFAULT_PUBLISH_DB) if use_publish_db else ""}


def apply_plan(plan_data: dict) -> dict:
    """执行改名。改之前重新校验：文件还在、SHA-256 没变、目标名没被占。"""
    renamed, refused = [], []
    for row in plan_data.get("renames") or []:
        old, new = str(row.get("old") or ""), str(row.get("new") or "")
        if not old or not new:
            refused.append({**row, "reason": "缺少旧名或新名"})
            continue
        if not os.path.isfile(old):
            refused.append({**row, "reason": "文件已不在"})
            continue
        if file_sha256(old) != (row.get("sha256") or ""):
            refused.append({**row, "reason": "哈希在计划之后变了，放弃改名"})
            continue
        if os.path.exists(new):
            refused.append({**row, "reason": "目标名已存在，放弃改名"})
            continue
        try:
            os.replace(old, new)
        except OSError as exc:
            refused.append({**row, "reason": f"改名失败：{exc}"})
            continue
        renamed.append({**row, "freed": 0})
    return {"renamed": renamed, "refused": refused,
            "bytes_moved": sum(int(row.get("size") or 0) for row in renamed)}
