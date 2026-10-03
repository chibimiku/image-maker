# -*- coding: utf-8 -*-
"""付费实验的**发送前**原子预算账本（第四轮 P0.4）。

为什么不能在成功之后才记账：底层 `requests` 会在异常/超时时自动重发，一次「调用」
可能已经扣了两次费；而 `api-calls.jsonl` 是产物落盘之后才写的，失败与断线根本不会留下
条目。所以名额必须在**真正发出 HTTP 之前**、以原子方式持久预留：

* 预留 = 用 ``O_CREAT | O_EXCL`` 抢一个编号槽位文件（Windows 上同样原子），
  抢不到就换下一个编号；编号用满即拒绝发送；
* 账本是文件系统本身（``slots/*.lock`` + ``images.jsonl``），所以**跨进程、跨 CLI 调用、
  跨重启累计**，不依赖任何常驻内存状态；
* 结算单独追加一条事件，区分「成功」「失败」「超时/未知」——未知扣费单列，不与成功混算；
* 同一个 ``run_id`` 已经派发过就拒绝再次执行；断点恢复只允许复用成功产物。

文本审计另有额度（默认 64 次、每阶段最多 2 次），走同一个目录的 ``text-slots/``。

环境变量（由实验 wrapper 注入，只影响被注入的那个子进程）：

* ``IMAGE_MAKER_SEND_BUDGET_DIR``：账本目录；未设置时本模块完全惰性（正常 GUI 不受影响）
* ``IMAGE_MAKER_SEND_BUDGET_RUN_ID``：本次计划 run 的编号
* ``IMAGE_MAKER_SEND_BUDGET_LIMIT``：图片尝试硬上限
* ``IMAGE_MAKER_TEXT_BUDGET_LIMIT`` / ``IMAGE_MAKER_TEXT_BUDGET_PER_STAGE``：文本审计额度
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os

ENV_DIR = "IMAGE_MAKER_SEND_BUDGET_DIR"
ENV_RUN = "IMAGE_MAKER_SEND_BUDGET_RUN_ID"
ENV_LIMIT = "IMAGE_MAKER_SEND_BUDGET_LIMIT"
ENV_TEXT_LIMIT = "IMAGE_MAKER_TEXT_BUDGET_LIMIT"
ENV_TEXT_PER_STAGE = "IMAGE_MAKER_TEXT_BUDGET_PER_STAGE"

DEFAULT_TEXT_LIMIT = 64
DEFAULT_TEXT_PER_STAGE = 2

IMAGE_LEDGER = "images.jsonl"
TEXT_LEDGER = "audit-calls.jsonl"
IMAGE_SLOTS = "slots"
TEXT_SLOTS = "text-slots"


class SendBudgetExhausted(RuntimeError):
    """预算用尽（或同一 run 已派发）：调用方必须**不要**发送。"""


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def budget_dir() -> str:
    return str(os.environ.get(ENV_DIR) or "").strip()


def budget_active() -> bool:
    return bool(budget_dir())


def current_run_id() -> str:
    return str(os.environ.get(ENV_RUN) or "").strip()


def _limit(default: int = 0) -> int:
    raw = str(os.environ.get(ENV_LIMIT) or "").strip()
    if not raw:
        return int(default)
    try:
        return max(0, int(float(raw)))
    except (TypeError, ValueError):
        return int(default)


def _write_jsonl(path: str, entry: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: str) -> list:
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _slot_dir(root: str, kind: str) -> str:
    return os.path.join(root, IMAGE_SLOTS if kind == "image" else TEXT_SLOTS)


def _used_slots(root: str, kind: str) -> list:
    folder = _slot_dir(root, kind)
    if not os.path.isdir(folder):
        return []
    names = sorted(name for name in os.listdir(folder) if name.endswith(".json"))
    rows = []
    for name in names:
        path = os.path.join(folder, name)
        try:
            with open(path, encoding="utf-8") as handle:
                rows.append(json.load(handle))
        except (OSError, json.JSONDecodeError):
            rows.append({"_unreadable": path})
    return rows


def _claim_slot(root: str, kind: str, limit: int, record: dict) -> dict:
    """原子抢一个编号槽位；编号全部占用时抛 ``SendBudgetExhausted``。"""
    folder = _slot_dir(root, kind)
    os.makedirs(folder, exist_ok=True)
    for index in range(1, limit + 1):
        path = os.path.join(folder, f"{index:03d}.json")
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        except OSError as exc:  # 目录不可写等：绝不能降级成「没有预算照样发」
            raise SendBudgetExhausted(f"预算槽位不可写：{path}（{exc}）") from exc
        payload = dict(record)
        payload.update({"slot": index, "slot_path": path, "reserved_at": _now()})
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        return payload
    raise SendBudgetExhausted(
        f"{kind} 预算已用满：{limit} 个槽位全部被占用（{folder}）。本轮没有补跑额度，"
        "不允许为了补齐样本继续发送。")


def state(root: str = "", kind: str = "image", limit: int = 0) -> dict:
    root = root or budget_dir()
    if not root:
        return {"active": False, "used": 0, "remaining": 0, "limit": 0, "slots": []}
    limit = limit or _limit()
    rows = _used_slots(root, kind)
    ledger = _read_jsonl(os.path.join(root, IMAGE_LEDGER if kind == "image" else TEXT_LEDGER))
    settled = {}
    for row in ledger:
        if row.get("event") == "settled" and row.get("attempt_id"):
            settled[row["attempt_id"]] = row
    return {"active": True, "dir": root, "kind": kind, "limit": limit, "used": len(rows),
            "remaining": max(0, limit - len(rows)),
            "slots": rows,
            "dispatched_run_ids": [row.get("run_id") for row in rows if row.get("run_id")],
            "settlements": settled}


def reserve_image_attempt(*, run_id: str = "", operation: str = "", model: str = "",
                          resolution: str = "", aspect_ratio: str = "", prompt_sha256: str = "",
                          prompt_chars: int = 0, source_sha256: str = "",
                          reference_images: list | None = None, note: str = "") -> dict:
    """在**发出 HTTP 之前**预留一次图片尝试。未启用预算时返回 ``{"active": False}``。"""
    root = budget_dir()
    if not root:
        return {"active": False}
    limit = _limit()
    rid = str(run_id or current_run_id() or "")
    for row in _used_slots(root, "image"):
        if rid and row.get("run_id") == rid:
            raise SendBudgetExhausted(
                f"run_id={rid} 已经派发过（槽位 {row.get('slot')}），拒绝重复执行同一计划 run")
    record = {"event": "reserved", "attempt_id": "", "run_id": rid, "operation": operation,
              "model": model, "resolution": resolution, "aspect_ratio": aspect_ratio,
              "prompt_sha256": prompt_sha256, "prompt_chars": int(prompt_chars or 0),
              "source_sha256": source_sha256, "reference_images": reference_images or [],
              "note": note, "pid": os.getpid()}
    if not rid:
        record["run_id"] = "unlabeled"
    claimed = _claim_slot(root, "image", limit, record)
    attempt_id = hashlib.sha256(
        f"{root}|{claimed['slot']}|{claimed.get('run_id')}|{claimed['reserved_at']}".encode("utf-8")
    ).hexdigest()[:16]
    claimed["attempt_id"] = attempt_id
    try:
        with open(claimed["slot_path"], "w", encoding="utf-8") as handle:
            json.dump(claimed, handle, ensure_ascii=False, indent=2)
    except OSError:
        pass
    _write_jsonl(os.path.join(root, IMAGE_LEDGER),
                 {"event": "reserved", "at": _now(), "attempt_id": attempt_id,
                  "slot": claimed["slot"], "run_id": claimed.get("run_id"),
                  "used_after": len(_used_slots(root, "image")), "limit": limit,
                  "operation": operation, "model": model, "resolution": resolution,
                  "prompt_sha256": prompt_sha256, "prompt_chars": int(prompt_chars or 0),
                  "source_sha256": source_sha256})
    return claimed


def settle_image_attempt(reservation: dict, *, status: str, output_path: str = "",
                         output_sha256: str = "", error: str = "", billed: str = "") -> dict:
    """结算一次尝试。``status`` ∈ success/failed/timeout/refused/unknown/cancelled。"""
    root = budget_dir()
    entry = {"event": "settled", "at": _now(),
             "attempt_id": str((reservation or {}).get("attempt_id") or ""),
             "slot": (reservation or {}).get("slot"),
             "run_id": (reservation or {}).get("run_id", ""),
             "status": str(status or "unknown"), "billed": billed or ("success" if status == "success" else "unknown"),
             "output_path": output_path, "output_sha256": output_sha256,
             "error": str(error or "")[:500]}
    if root:
        _write_jsonl(os.path.join(root, IMAGE_LEDGER), entry)
    return entry


def reserve_text_attempt(*, stage: str, run_id: str = "", purpose: str = "") -> dict:
    """文本审计额度：全局默认 64 次、每阶段最多 2 次（含失败）。"""
    root = budget_dir()
    if not root:
        return {"active": False}
    total_limit = int(str(os.environ.get(ENV_TEXT_LIMIT) or DEFAULT_TEXT_LIMIT))
    per_stage = int(str(os.environ.get(ENV_TEXT_PER_STAGE) or DEFAULT_TEXT_PER_STAGE))
    used = _used_slots(root, "text")
    if len(used) >= total_limit:
        raise SendBudgetExhausted(f"文本审计额度已用满（{total_limit} 次）")
    same_stage = [row for row in used if row.get("stage") == stage]
    if len(same_stage) >= per_stage:
        raise SendBudgetExhausted(f"阶段 {stage} 的文本审计次数已达上限（{per_stage} 次）")
    claimed = _claim_slot(root, "text", total_limit, {
        "event": "reserved", "run_id": run_id or current_run_id(), "stage": stage,
        "purpose": purpose, "pid": os.getpid()})
    _write_jsonl(os.path.join(root, TEXT_LEDGER),
                 {"event": "reserved", "at": _now(), "attempt_id": claimed.get("slot_path"),
                  "slot": claimed["slot"], "stage": stage, "run_id": claimed.get("run_id")})
    return claimed


def settle_text_attempt(reservation: dict, *, status: str, error: str = "") -> dict:
    root = budget_dir()
    entry = {"event": "settled", "at": _now(), "slot": (reservation or {}).get("slot"),
             "stage": (reservation or {}).get("stage", ""), "status": status,
             "error": str(error or "")[:300]}
    if root:
        _write_jsonl(os.path.join(root, TEXT_LEDGER), entry)
    return entry


def guarded_text_call(stage: str, fn, *args, **kwargs):
    """给一次文本审计 HTTP 调用套上额度（未启用预算时完全透传）。

    ``stage`` 由「本次计划 run + 审计种类」组成，所以「每阶段最多 2 次」= 每个 run
    的每类审计最多 2 次（含失败），正好允许一次失败重试；全局另有 64 次上限。
    """
    if not budget_active():
        return fn(*args, **kwargs)
    label = f"{current_run_id() or 'unlabeled'}/{stage}"
    reservation = reserve_text_attempt(stage=label, purpose=stage)
    try:
        value = fn(*args, **kwargs)
    except BaseException as exc:
        settle_text_attempt(reservation, status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    settle_text_attempt(reservation, status="success")
    return value


def can_reuse_success(root: str, run_id: str) -> str:
    """断点恢复：只复用**成功**产物；返回可复用的产物路径，没有则返回空串。"""
    if not root:
        return ""
    for row in _read_jsonl(os.path.join(root, IMAGE_LEDGER)):
        if row.get("event") == "settled" and row.get("run_id") == run_id \
                and row.get("status") == "success" and row.get("output_path") \
                and os.path.isfile(str(row["output_path"])):
            return str(row["output_path"])
    return ""
