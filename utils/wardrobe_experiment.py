# -*- coding: utf-8 -*-
"""衣装实验族的持久预算账本与快照确定性检查（无 Qt、无网络、无生图依赖）。

为什么单独成模块：第二轮任务书要求「**发送前**按每次真实 HTTP 尝试扣减」预算，
失败/超时/备用/探针/重试都算，SDK 隐藏重试必须禁用或在同一边界计数，崩溃后
已发出但结果未知的尝试也不能重置。这些规则必须能被离线测试（见
`tests/test_wardrobe_experiment.py`），所以从 CLI 里抽出来放在这里。

用法概览：

    ledger = ExperimentLedger.load(path, budgets)
    attempt = ledger.plan_attempt("generation", slot="C03-gpt-r1-v3_routed")  # 发送前落盘
    ... 真实发送 ...
    ledger.finish_attempt(attempt, status="success", http_status=200, model="gpt-image-2")

账本只记录请求元数据与结果，**不保存任何凭据**；落盘用 `utils.atomic_io`。
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from typing import Iterable

from utils.atomic_io import write_json_atomic

SCHEMA_VERSION = "image-maker.wardrobe-experiment.v1"
# 预算按真实 HTTP 发送扣减；键名与任务书表格一致
BUDGET_KEYS = ("text_http_attempts", "generation_http_attempts", "vision_http_attempts",
               "model_list_http_attempts")
ATTEMPT_KINDS = ("text", "generation", "vision", "model_list")
# 生成/视觉的最小重试间隔：防止把同一份请求在循环里反复打（本轮不允许任何自动重试）
DEFAULT_MAX_ATTEMPTS_PER_SLOT = 1


class BudgetExceeded(RuntimeError):
    """发送前就超出预算：必须停下，不得先发再判。"""


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def default_budgets() -> dict:
    return {"text_http_attempts": 2, "generation_http_attempts": 32, "vision_http_attempts": 20,
            "model_list_http_attempts": 1}


def normalize_budgets(raw: dict | None) -> dict:
    """只认已知的四个上限；缺省取任务书值，负数一律按 0（不允许负预算当无限）。"""
    budgets = default_budgets()
    for key in BUDGET_KEYS:
        if raw and key in raw and raw[key] is not None:
            try:
                budgets[key] = max(0, int(raw[key]))
            except (TypeError, ValueError):
                pass
    return budgets


def resolve_run_dir(output_root: str, run_id: str, explicit: str = "", *,
                    ledger_name: str = "ledger.json") -> str:
    """解析本轮输出目录：**跨天复跑必须落在同一个原目录与原账本上**。

    CLI 的默认口径是 `output_root/<当天日期>/<run-id>`，于是同一条实验在第二天
    重跑会凭空多出一个目录、一份新账本（预算被悄悄重置）。任务书要求「先在现有
    工具内做最小、可测试的『指定已有 run 目录』恢复支持」：给 `--run-dir` 就直接
    用那个目录，不再按当天日期推导。

    显式目录必须已经存在且带着账本——否则就是打错路径去开一个新预算，必须拒绝。
    """
    if explicit:
        target = os.path.abspath(explicit)
        if not os.path.isdir(target):
            raise FileNotFoundError(f"--run-dir 指向的目录不存在: {target}")
        if not os.path.isfile(os.path.join(target, ledger_name)):
            raise FileNotFoundError(
                f"--run-dir 目录里没有 {ledger_name}: {target}（避免把新目录当成原账本继续跑）")
        return target
    return os.path.abspath(os.path.join(output_root, time.strftime("%Y%m%d"), str(run_id or "")))


def check_ledger_budgets(existing: dict, per_spec: dict) -> list:
    """账本上限只能等于 spec，或高于它——后者是**用户显式授权追加**留下的痕迹。

    低于 spec 一定有问题（被悄悄改小、或跑的是别的 spec）；高于 spec 不算篡改，
    因为追加额度只走 `ExperimentLedger.grant_budget()`，账本里有对应的授权记录。
    追加本身绝不允许重置已消耗：这里只比较上限，`spent` 由 attempts 决定。
    """
    issues = []
    current = normalize_budgets(existing)
    wanted = normalize_budgets(per_spec)
    for key in BUDGET_KEYS:
        if int(current[key]) < int(wanted[key]):
            issues.append({"check": "ledger_budget_not_below_spec", "budget": key,
                           "ledger": int(current[key]), "spec": int(wanted[key]),
                           "detail": "账本上限低于 spec：禁止改小上限或另开新账本"})
    return issues


def budget_key_for(kind: str) -> str:
    kind = str(kind or "").strip()
    if kind not in ATTEMPT_KINDS:
        raise ValueError(f"未知的尝试类型: {kind}")
    return {
        "text": "text_http_attempts",
        "generation": "generation_http_attempts",
        "vision": "vision_http_attempts",
        "model_list": "model_list_http_attempts",
    }[kind]


class ExperimentLedger:
    """按「发送前扣减」计数的持久账本。"""

    def __init__(self, data: dict = None, path: str = ""):
        self.data = data or {"schema_version": SCHEMA_VERSION, "created_at": now_iso(),
                             "budgets": default_budgets(), "attempts": [], "recovered": []}
        self.data.setdefault("schema_version", SCHEMA_VERSION)
        self.data.setdefault("budgets", default_budgets())
        self.data.setdefault("attempts", [])
        self.data.setdefault("recovered", [])
        self.path = path

    # ---------------- 读写 ----------------
    @classmethod
    def load(cls, path: str, budgets: dict = None, create: bool = True,
             trust_persisted_budgets: bool = False) -> "ExperimentLedger":
        """读账本。

        `budgets` 默认只允许**收紧**（防止换一个上限把已消耗抹掉）。但已经有用户
        显式授权追加额度的账本不该被 spec 上限压回去：那种场景传
        `trust_persisted_budgets=True`，上限以账本里的值为准（`spent` 从来只由
        attempts 决定，不受这个开关影响）。
        """
        data = None
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        elif not create:
            raise FileNotFoundError(path)
        ledger = cls(data, path)
        if budgets and not trust_persisted_budgets:
            # 只允许收紧或与任务书一致：已经消耗的数量不能被换一个上限抹掉
            merged = {}
            for key, value in normalize_budgets(budgets).items():
                merged[key] = min(int(ledger.data["budgets"].get(key, value)), value)
            ledger.data["budgets"] = merged
        return ledger

    def save(self) -> None:
        if not self.path:
            return
        self.data["updated_at"] = now_iso()
        write_json_atomic(self.path, self.data)

    # ---------------- 预算 ----------------
    def spent(self, kind: str) -> int:
        return sum(1 for item in self.data["attempts"] if item.get("kind") == kind)

    def remaining(self, kind: str) -> int:
        return int(self.data["budgets"][budget_key_for(kind)]) - self.spent(kind)

    def summary(self) -> dict:
        return {kind: {"limit": int(self.data["budgets"][budget_key_for(kind)]),
                       "spent": self.spent(kind), "remaining": self.remaining(kind)}
                for kind in ATTEMPT_KINDS}

    def require(self, kind: str) -> None:
        """发送前检查；不够就抛 BudgetExceeded（绝不允许先发再判）。"""
        if self.remaining(kind) <= 0:
            raise BudgetExceeded(f"{kind} 预算已用尽（上限 {self.data['budgets'][budget_key_for(kind)]}）")

    # ---------------- 尝试 ----------------
    def plan_attempt(self, kind: str, slot: str = "", label: str = "", requested_at: str = "") -> dict:
        """在真实发送**之前**记账并落盘；崩溃后该记录会被恢复逻辑视为已消耗。"""
        self.require(kind)
        same_slot = [item for item in self.data["attempts"]
                     if item.get("kind") == kind and item.get("slot") == slot]
        attempt = {
            "attempt_index": len(self.data["attempts"]) + 1,
            "kind": kind,
            "slot": str(slot or ""),
            "label": str(label or ""),
            "slot_attempt": len(same_slot) + 1,
            "status": "inflight",
            "planned_at": requested_at or now_iso(),
            "budget_after_plan": self.remaining(kind) - 1,
        }
        self.data["attempts"].append(attempt)
        self.save()
        return attempt

    def finish_attempt(self, attempt: dict, status: str = "success", http_status=None,
                       model: str = "", error: str = "", elapsed_ms: int = 0, extra: dict = None) -> dict:
        target = None
        for item in self.data["attempts"]:
            if item is attempt or (item.get("attempt_index") == attempt.get("attempt_index")
                                   and item.get("kind") == attempt.get("kind")):
                target = item
                break
        if target is None:
            raise KeyError("账本里找不到该尝试记录")
        target.update({"status": str(status or "unknown"), "finished_at": now_iso(),
                       "http_status": http_status, "model": str(model or ""),
                       "error": str(error or "")[:800], "elapsed_ms": int(elapsed_ms or 0)})
        if extra:
            target.update(extra)
        self.save()
        return target

    def recover_inflight(self, reason: str = "previous process ended before a result was recorded") -> list:
        """崩溃恢复：把上次遗留的 inflight 记录标成 unknown 并保持已消耗。"""
        recovered = []
        for item in self.data["attempts"]:
            if item.get("status") == "inflight":
                item["status"] = "unknown"
                item["recovered_at"] = now_iso()
                item["recover_reason"] = str(reason)
                recovered.append(item)
        if recovered:
            self.data["recovered"].extend(
                {"attempt_index": item["attempt_index"], "kind": item["kind"], "slot": item.get("slot"),
                 "reason": reason} for item in recovered)
            self.save()
        return recovered

    def granted_limit(self, kind: str) -> int:
        """该预算上**授权过的**最高上限（没有授权就返回任务书默认值）。

        为什么需要它：`load()` 的老口径是「只允许收紧」，会把用户后来授权的
        追加额度按 spec 上限压回去——2026-10-05 的 vision 额度就是这样在评分
        中途从 31 掉回 25、白丢 6 个槽位。上限以账本里授权过的最高值为准。
        """
        key = budget_key_for(kind)
        limit = int(default_budgets()[key])
        for record in self.data.get("budget_grants") or []:
            if record.get("budget") == key and record.get("new_limit") is not None:
                limit = max(limit, int(record["new_limit"]))
        return limit

    def assert_budget_floor(self, kind: str) -> int:
        """把上限抬回授权过的最高值（只抬不降，绝不改已消耗）。"""
        key = budget_key_for(kind)
        granted = self.granted_limit(kind)
        if int(self.data["budgets"].get(key, 0)) < granted:
            self.data["budgets"][key] = granted
            self.save()
        return int(self.data["budgets"][key])

    def mark_skipped(self, reason: str) -> None:
        """停止条件（解析/校准/审查失败等）：把原因写进账本，便于复核。"""
        self.data["stop_reason"] = str(reason or "")
        self.save()

    def grant_budget(self, kind: str, limit: int, *, authorized_by: str, reason: str,
                     source: str = "") -> dict:
        """**用户显式授权**的追加额度：只抬上限，不重置已消耗。

        为什么单独做成一条记录而不是直接改数字：任务书禁止「改上限、换 run-id、
        删账本」来补结果。只有用户在新的对话里明确授权追加时，才允许用这个方法，
        而且必须把授权人、时间、理由与旧值/新值一并留痕，`spent` 与已有 attempts
        完全不动。不是授权场景请勿调用。
        """
        key = budget_key_for(kind)
        if not str(authorized_by or "").strip() or not str(reason or "").strip():
            raise ValueError("追加额度必须记录授权人与理由")
        new_limit = max(0, int(limit))
        previous = int(self.data["budgets"].get(key, 0))
        if new_limit <= previous:
            raise ValueError(f"追加额度必须高于当前上限（当前 {previous}，要求 {new_limit}）")
        granted_at = now_iso()
        self.data["budgets"][key] = new_limit
        self.data.setdefault("budget_grants", []).append({
            "budget": key, "kind": kind, "previous_limit": previous, "new_limit": new_limit,
            "delta": new_limit - previous, "spent_at_grant": self.spent(kind),
            "authorized_by": str(authorized_by), "reason": str(reason), "source": str(source),
            "granted_at": granted_at,
        })
        self.data["updated_at"] = granted_at
        self.save()
        return {"budget": key, "previous_limit": previous, "new_limit": new_limit,
                "spent": self.spent(kind), "remaining": self.remaining(kind), "granted_at": granted_at}

    def record_override(self, kind: str, *, authorized_by: str, reason: str,
                        assumed_hypothesis: str, known_risk: str, accepted_at: str = "",
                        source: str = "") -> dict:
        """记录一次**用户显式放行**（例如校准未通过但用户接受已知风险继续）。

        放行不是跳过：必须写明放行的门、依据、假设与已知风险，一条都不能空。
        记录追加在 `budget_grants` 里，调用方还要把它挂到对应阶段状态上，
        门禁与报告才看得见「这道门是被放行的，不是通过的」。
        """
        fields = {"kind": str(kind or "").strip(), "authorized_by": str(authorized_by or "").strip(),
                  "reason": str(reason or "").strip(),
                  "assumed_hypothesis": str(assumed_hypothesis or "").strip(),
                  "known_risk": str(known_risk or "").strip()}
        missing = [name for name, value in fields.items() if not value]
        if missing:
            raise ValueError(f"放行记录缺字段: {missing}")
        record = {**fields, "accepted_at": str(accepted_at or "").strip() or now_iso(),
                  "source": str(source or ""), "granted_at": now_iso()}
        # 放行记录单独放 `gate_overrides`：它跟额度无关，混进 `budget_grants`
        # 会让账本里出现 budget/new_limit 全是 None 的条目（2026-10-05 踩过）
        self.data.setdefault("gate_overrides", []).append(record)
        self.data["updated_at"] = record["granted_at"]
        self.save()
        return record

    def attempt_count(self) -> int:
        return len(self.data["attempts"])


# ---------------------------------------------------------------------------
# 快照与展开 prompt 的确定性检查
# ---------------------------------------------------------------------------

LEAK_TOKENS = ("expected", "gold_", "\"gold\"", "score", "评分", "test-score", "evaluator",
               "resolver-cases", "calibration-cases", "preferred", "oracle")


def snapshot_hash_check(manifest: dict, pack_dir: str) -> dict:
    """逐份重算 pack 里的 prompt 快照 hash / 字符数，与 manifest 对账。"""
    rows = []
    for item in manifest.get("snapshots") or []:
        path = os.path.join(pack_dir, item["file"])
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        rows.append({"file": item["file"], "case_id": item.get("case_id"),
                     "variant": item.get("variant"),
                     "sha256_expected": item.get("sha256"), "sha256_actual": sha256_text(text),
                     "characters_expected": item.get("characters"), "characters_actual": len(text),
                     "match": (sha256_text(text) == item.get("sha256")
                               and len(text) == item.get("characters"))})
    return {"ok": all(row["match"] for row in rows), "rows": rows}


BLOCK_HEAD_RE = re.compile(r"^[A-Z][A-Z0-9 _/+\-]{6,}:")
TARGET_TOKENS = ("lolita", "op dress", "jsk", "bell skirt", "a-line skirt", "skirt volume",
                 "layered skirt", "near-knee")


def added_blocks(prompt: str, base_prompt: str) -> list:
    """取 prompt 里超出 base 的追加块（按空行分段，逐段判断是否为新增）。"""
    extra = prompt[len(base_prompt):] if prompt.startswith(base_prompt) else prompt
    return [block.strip() for block in re.split(r"\n\s*\n", extra) if block.strip()]


def deterministic_prompt_checks(prompt: str, base_prompt: str, *, variant: str,
                                gold_action: str, framing: str, role: str = "",
                                strict: bool = True) -> dict:
    """保留分支/近景分支/候选分支的确定性检查（不看模型、不看图）。

    - preserve：新增块不得含目标衣装设计指令（Lolita / OP / JSK / 裙型裙量等）；
    - upper_chest：新增块不得要求全身衣装或裙鞋，且最后一段必须是 frame-bust 的收尾约束；
    - v4_candidate：追加块里不得再出现「设计块优先/软覆盖」式条款；
    - 任何分支：不得出现评价器/预期字段泄漏。

    `strict=False`（用于冻结控制臂）：把检查结果标成 `defect_expected`，
    只报告不判失败；候选臂必须 `strict=True`，任何一条都不许放宽。
    """
    blocks = added_blocks(prompt, base_prompt)
    joined = "\n\n".join(blocks).lower()
    issues = []
    if gold_action == "preserve":
        hits = [token for token in TARGET_TOKENS if token in joined]
        if hits:
            issues.append({"check": "preserve_no_target_design_block", "hits": hits,
                           "detail": "保留分支的追加块里出现了目标衣装设计指令"})
    if framing == "upper_chest":
        for token in ("full-body", "full body", "including both feet", "shoes", "skirt volume"):
            if token in joined:
                issues.append({"check": "close_portrait_no_full_outfit", "hits": [token],
                               "detail": "近景分支的追加块里出现了全身/裙鞋要求"})
        if blocks and "FINAL FRAMING REQUIREMENT" not in blocks[-1]:
            issues.append({"check": "frame_clause_last", "hits": [blocks[-1][:60]],
                           "detail": "最后一段不是画幅约束"})
        if "above the bust midpoint" not in joined:
            issues.append({"check": "crop_boundary_explicit", "hits": [],
                           "detail": "缺少明确的裁剪下缘约束"})
    if variant == "v4_candidate":
        for token in ("design block above applies", "overrides the design block",
                      "overrides generic source-outfit"):
            if token in joined or token in prompt.lower():
                issues.append({"check": "candidate_no_soft_override", "hits": [token],
                               "detail": "候选里仍带「设计块优先/软覆盖」式条款"})
    leaked = [token for token in LEAK_TOKENS if token.lower() in prompt.lower()]
    if leaked:
        issues.append({"check": "no_expectation_leak", "hits": leaked,
                       "detail": "prompt 里出现了评价器/预期字段"})
    if re.search(r"\{[a-z_]+\}", prompt):
        issues.append({"check": "no_placeholder", "hits": re.findall(r"\{[a-z_]+\}", prompt),
                       "detail": "prompt 残留占位符"})
    for issue in issues:
        issue["defect_expected"] = not strict
        issue["severity"] = "defect_expected" if not strict else "fail"
    failed = [issue for issue in issues if issue["severity"] == "fail"]
    return {"ok": not failed, "issues": issues, "added_blocks": blocks,
            "variant": variant, "gold_action": gold_action, "framing": framing,
            "strict": bool(strict), "expected_defects": [issue for issue in issues
                                                         if not issue["severity"] == "fail"]}


SECRET_KEYS = ("authorization", "api_key", "apikey", "x-goog-api-key", "token", "secret")


def scrub_request_snapshot(payload, secret_values: Iterable[str] = ()) -> dict:
    """请求快照脱敏：去掉认证头与密钥值，长 base64 换成 hash 占位。"""
    secrets = [str(value) for value in (secret_values or []) if value]

    def walk(node):
        if isinstance(node, dict):
            out = {}
            for key, value in node.items():
                if str(key).lower() in SECRET_KEYS:
                    out[key] = "<REDACTED>"
                else:
                    out[key] = walk(value)
            return out
        if isinstance(node, list):
            return [walk(item) for item in node]
        if isinstance(node, str):
            text = node
            for secret in secrets:
                if secret and secret in text:
                    text = text.replace(secret, "<REDACTED>")
            if len(text) > 2048:
                return {"__len__": len(text), "__sha256__": sha256_text(text), "__head__": text[:64]}
            return text
        return node

    return {"redacted": walk(payload), "scrubbed_at": now_iso()}


def content_block_order(messages) -> list:
    """记录标签与图片内容块的顺序（label、类型、图片 hash），便于复现真实发送顺序。"""
    order = []
    if isinstance(messages, dict):
        messages = [messages]
    for message in messages or []:
        role = str((message or {}).get("role") or "")
        content = (message or {}).get("content")
        if isinstance(content, str):
            order.append({"role": role, "type": "text", "sha256": sha256_text(content),
                          "preview": content[:80]})
            continue
        for block in content or []:
            if not isinstance(block, dict):
                continue
            kind = str(block.get("type") or "")
            if kind == "text":
                text = str(block.get("text") or "")
                label = ""
                for pattern in (r"\[(?:IMAGE|IMAGE )?([A-Z0-9\-_]+)\]",
                                r"IMAGE LABEL:\s*\[([^\]]+)\]"):
                    match = re.search(pattern, text)
                    if match:
                        label = match.group(1)
                        break
                order.append({"role": role, "type": "text", "label": label,
                              "sha256": sha256_text(text), "preview": text[:80]})
            elif kind == "image_url":
                url = str(((block.get("image_url") or {}).get("url")) or "")
                order.append({"role": role, "type": "image_url",
                              "sha256": sha256_text(url), "bytes_estimate": len(url)})
            else:
                order.append({"role": role, "type": kind or "unknown"})
    return order


def resolver_request_payload(cases: list, system_prompt: str) -> dict:
    """按 resolve.md 构造批量解析请求体：**只带输入字段，剔除 expected**。"""
    payload_cases = []
    for case in cases or []:
        payload_cases.append({"id": case.get("id"), "policy": case.get("policy"),
                              "source_facts": case.get("source_facts"),
                              "user_request": case.get("user_request")})
    return {"system": system_prompt,
            "user": json.dumps({"cases": payload_cases}, ensure_ascii=False, indent=2),
            "expected_fields_stripped": True, "case_count": len(payload_cases)}


def gold_routes(plan: dict) -> dict:
    """生图阶段的授权路由（oracle）：只用预登记 gold，不采用模型解析输出。"""
    routes = {}
    for case in plan.get("cases") or []:
        routes[case["case_id"]] = {"gold_action": case.get("gold_action"),
                                   "gold_presence": case.get("gold_presence"),
                                   "framing": case.get("framing"),
                                   "role": case.get("role"), "repeats": case.get("repeats")}
    return routes


def validate_resolver_output(raw_text: str, cases: list, expected_by_id: dict = None) -> dict:
    """校验解析返回：12 个 ID、真布尔、四个字段齐全；全部一致才算通过。"""
    issues = []
    text = str(raw_text or "")
    start, end = text.find("{"), text.rfind("}")
    data = None
    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end + 1])
        except Exception as exc:  # noqa: BLE001
            issues.append(f"JSON 解析失败: {type(exc).__name__}: {exc}")
    else:
        issues.append("响应里找不到 JSON 对象")
    decisions = (data or {}).get("decisions") if isinstance(data, dict) else None
    if not isinstance(decisions, list):
        issues.append("缺少 decisions 数组")
        decisions = []
    ids_expected = [case.get("id") for case in (cases or [])]
    ids_seen = [item.get("id") for item in decisions if isinstance(item, dict)]
    if len(ids_seen) != len(set(ids_seen)):
        issues.append("ID 重复")
    missing = [item for item in ids_expected if item not in ids_seen]
    extra = [item for item in ids_seen if item not in ids_expected]
    if missing:
        issues.append(f"缺少 ID: {missing}")
    if extra:
        issues.append(f"多出 ID: {extra}")
    rows = []
    for item in decisions:
        if not isinstance(item, dict):
            issues.append("decision 不是对象")
            continue
        row_issues = []
        if not isinstance(item.get("explicit_keep_original"), bool):
            row_issues.append("explicit_keep_original 不是真布尔")
        for field in ("action", "clothing_presence", "framing"):
            if not isinstance(item.get(field), str) or not item.get(field):
                row_issues.append(f"{field} 不是非空字符串")
        row = {"id": item.get("id"), "action": item.get("action"),
               "clothing_presence": item.get("clothing_presence"),
               "explicit_keep_original": item.get("explicit_keep_original"),
               "framing": item.get("framing"), "issues": row_issues, "matches": None}
        if expected_by_id and item.get("id") in expected_by_id:
            expected = expected_by_id[item["id"]]
            row["expected"] = expected
            checked = [field for field in ("action", "clothing_presence", "explicit_keep_original",
                                           "framing") if field in expected]
            row["checked_fields"] = checked
            row["matches"] = all(item.get(field) == expected.get(field) for field in checked)
            if not row["matches"]:
                row_issues.append("与本地 gold 不一致")
        elif expected_by_id:
            # 出现本地 gold 里没有的 ID：不能算通过
            row_issues.append("本地 gold 里没有这个 ID")
            row["matches"] = False
        rows.append(row)
    matched = sum(1 for row in rows if row.get("matches") is True)
    return {"valid": not issues and all(not row["issues"] for row in rows),
            "issues": issues, "rows": rows, "matched": matched, "total": len(cases or []),
            "raw": text}


# ---------------------------------------------------------------------------
# Round3 风格的人物任务 spec：校验、组装、配对与门禁
# ---------------------------------------------------------------------------

SPEC_REQUIRED_CASE_FIELDS = ("case_id", "character_id", "role", "policy", "branch", "gold_action",
                             "gold_presence", "framing", "repeats", "variants")


def spec_hash(spec_dir: str, *, characters_file: str = "", base_pack: str = "") -> str:
    """spec 指纹：计划 + 人物卡 + base-pack + 全部模板文件一起摘要。

    就绪结论必须绑定这个 hash：任何素材改动都会让旧的就绪结论失效。

    **同族新实验（round4 及以上）**：模板/人物卡所在目录不再写死为 spec 的兄弟目录
    `../variants`、`../characters`，改由 plan 的 `spec_digest` 段声明
    （`declared_files` 逐个文件、`declared_dirs` 逐棵目录树）。旧 plan 没有这一段时
    完全走原来的默认值，round2/round3 的指纹不会变。
    """
    parts = []
    plan_path = os.path.join(spec_dir, "plan.json")
    plan = json.load(open(plan_path, "r", encoding="utf-8")) if os.path.isfile(plan_path) else {}
    parts.append(("plan.json", sha256_file(plan_path) if os.path.isfile(plan_path) else ""))
    root = os.path.dirname(spec_dir)
    digest_spec = plan.get("spec_digest") or {}
    chars = characters_file or str(digest_spec.get("characters_file")
                                   or plan.get("characters_file") or "")
    if chars and os.path.isfile(chars):
        parts.append((chars.replace("\\", "/"), sha256_file(chars)))
    packs = [base_pack or str(digest_spec.get("base_pack") or plan.get("base_pack") or "")]
    packs += [str(item) for item in (digest_spec.get("base_packs") or [])]
    for pack in packs:
        if not pack:
            continue
        for name in sorted(os.listdir(pack)) if os.path.isdir(pack) else []:
            path = os.path.join(pack, name)
            if os.path.isfile(path):
                parts.append((os.path.join(pack, name).replace("\\", "/"), sha256_file(path)))
    for root_dir, _dirs, files in os.walk(spec_dir):
        for name in sorted(files):
            if not name.endswith((".md", ".json")):
                continue
            path = os.path.join(root_dir, name)
            relative = os.path.relpath(path, os.path.dirname(spec_dir)).replace("\\", "/")
            if any(relative == item[0] for item in parts):
                continue
            parts.append((relative, sha256_file(path)))
    directories = [str(item) for item in (digest_spec.get("declared_dirs") or [])]
    for name in (digest_spec.get("declared_files") or []):
        name = str(name)
        if os.path.isfile(name):
            parts.append((name.replace("\\", "/"), sha256_file(name)))
    if not digest_spec:
        directories = [os.path.join(root, folder) for folder in ("variants", "characters")]
    for directory in directories:
        for root_dir, _dirs, files in os.walk(directory):
            for name in sorted(files):
                path = os.path.join(root_dir, name)
                if name.endswith((".md", ".json")):
                    parts.append((os.path.relpath(path, root).replace("\\", "/"), sha256_file(path)))
    digest = hashlib.sha256()
    for name, digest_value in sorted(parts):
        digest.update(f"{name}:{digest_value}\n".encode("utf-8"))
    return digest.hexdigest()


def render_template(text: str, values: dict) -> str:
    """安全的占位符替换：只替换已知键，模板里出现未知花括号不改动。"""
    out = str(text or "")
    for key, value in (values or {}).items():
        out = out.replace("{" + str(key) + "}", str(value))
    return out


def load_character_test(spec_dir: str) -> dict:
    plan = json.load(open(os.path.join(spec_dir, "plan.json"), "r", encoding="utf-8"))
    characters_file = plan.get("characters_file") or os.path.join(os.path.dirname(spec_dir),
                                                                  "characters.json")
    with open(characters_file, "r", encoding="utf-8") as handle:
        characters = json.load(handle)
    return {"plan": plan, "characters_doc": characters, "characters_file": characters_file,
            "by_id": {item["id"]: item for item in characters.get("characters") or []},
            "by_case": {item["case_id"]: item for item in characters.get("characters") or []}}


def assert_online_supported(plan: dict) -> None:
    """同族新实验的在线阶段守卫：只放行显式声明支持在线执行的计划。

    round4 的规划文件本身写着 `cli_compatible: false` / `online_execution_authorized: false`，
    在没有拿到用户新额度与执行指示之前，任何在线子命令都不许被「顺手跑一下」。
    """
    schema = str((plan or {}).get("schema_version") or "")
    if schema.startswith("image-maker.wardrobe-round4"):
        if (plan or {}).get("cli_compatible") is not True:
            raise RuntimeError(
                "该计划声明 cli_compatible=false：在线阶段需要先由用户授权并按本族规范"
                "把候选包/模板/变体定义补齐后重新冻结，不能直接执行")
        if (plan or {}).get("online_execution_authorized") is not True:
            raise RuntimeError("该计划未获在线执行授权（online_execution_authorized != true）")


def plan_variant_ids(plan: dict) -> list:
    """计划里声明的变体 ID（按声明顺序）；没有变体定义时返回空表。"""
    return [str(item.get("id")) for item in (plan or {}).get("variants") or [] if item.get("id")]


def validate_character_spec(spec_dir: str, repo_root: str = ".") -> dict:
    """按 spec 校验人物卡与用例：字段、枚举、数量、引用、base-pack hash。

    缺字段/错枚举/数量不符一律 ok=False（禁止默认通过）。

    **round4 家族**：规划文件的 schema 与 round3 不同（`image-maker.wardrobe-round4.planning.v1`），
    结构也不一样（14 个直接采样槽位、没有 control/candidate 两臂、没有 base-pack）。
    这里**不做**「按 round4 规则校验并放行」的假装实现：遇到非 round3 的 schema 一律
    `ok=False` 并在 `unsupported_schema` 里明说「本工具尚未实现该 schema 的校验」，
    避免把「未校验」当成「校验通过」。
    """
    plan = json.load(open(os.path.join(spec_dir, "plan.json"), "r", encoding="utf-8")) \
        if os.path.isfile(os.path.join(spec_dir, "plan.json")) else {}
    schema = str(plan.get("schema_version") or "")
    if schema and schema != "image-maker.wardrobe-round3.v1":
        return {"ok": False, "issues": [f"本工具只实现 round3 的 spec 校验，收到 schema={schema!r}"],
                "unsupported_schema": schema, "case_count": len(plan.get("cases") or []),
                "planned_generation_slots": 0, "characters": [], "variants": plan_variant_ids(plan),
                "base_pack_rows": [], "spec_hash": spec_hash(spec_dir)}
    issues = []
    doc = load_character_test(spec_dir)
    plan, characters = doc["plan"], doc["characters_doc"]
    allowed = plan.get("allowed") or {}
    cases = plan.get("cases") or []

    # --- 人物卡 ---
    character_rows = characters.get("characters") or []
    if plan.get("schema_version") != "image-maker.wardrobe-round3.v1":
        issues.append(f"plan.schema_version 不是 round3: {plan.get('schema_version')!r}")
    if characters.get("schema_version") != "image-maker.wardrobe-character-test.v1":
        issues.append(f"characters.schema_version 不是角色测试版: {characters.get('schema_version')!r}")
    declared_characters = len(character_rows)
    if declared_characters != len(cases):
        issues.append(f"人物数 {declared_characters} 与用例数 {len(cases)} 不一致")
    for character in character_rows:
        for field in ("id", "case_id", "label", "policy", "gold_action", "framing", "repeats",
                      "identity", "source_clothing", "request", "protected"):
            if character.get(field) in (None, "", []):
                issues.append(f"{character.get('id', '?')} 缺少字段 {field}")
        if not isinstance(character.get("protected"), list) or len(character.get("protected") or []) < 3:
            issues.append(f"{character.get('id', '?')} protected 少于 3 条")
        if character.get("policy") not in (allowed.get("policies") or []):
            issues.append(f"{character.get('id', '?')} policy 非法: {character.get('policy')!r}")
        if character.get("gold_action") not in (allowed.get("gold_actions") or []):
            issues.append(f"{character.get('id', '?')} gold_action 非法: {character.get('gold_action')!r}")
        if character.get("framing") not in (allowed.get("framings") or []):
            issues.append(f"{character.get('id', '?')} framing 非法: {character.get('framing')!r}")
        if not isinstance(character.get("repeats"), int) or int(character.get("repeats") or 0) < 1:
            issues.append(f"{character.get('id', '?')} repeats 必须是正整数")

    # --- 用例 ---
    seen_case_ids = set()
    seen_characters = set()
    planned_slots = 0
    for case in cases:
        missing = [field for field in SPEC_REQUIRED_CASE_FIELDS if case.get(field) in (None, "", [])]
        if missing:
            issues.append(f"{case.get('case_id', '?')} 用例缺字段: {missing}")
            continue
        if case["case_id"] in seen_case_ids:
            issues.append(f"用例 ID 重复: {case['case_id']}")
        seen_case_ids.add(case["case_id"])
        if case["character_id"] in seen_characters:
            issues.append(f"人物被多个用例复用: {case['character_id']}")
        seen_characters.add(case["character_id"])
        allowed_field_keys = {"role": "roles", "policy": "policies", "branch": "branches",
                              "gold_action": "gold_actions", "gold_presence": "gold_presence",
                              "framing": "framings"}
        for field, allowed_key in allowed_field_keys.items():
            if case[field] not in (allowed.get(allowed_key) or []):
                issues.append(f"{case['case_id']} {field} 非法: {case[field]!r}")
        for variant in case["variants"]:
            if variant not in (allowed.get("variants") or []):
                issues.append(f"{case['case_id']} variant 非法: {variant!r}")
        if not isinstance(case["repeats"], int) or case["repeats"] < 1:
            issues.append(f"{case['case_id']} repeats 必须是正整数")
        character = doc["by_case"].get(case["case_id"])
        if not character:
            issues.append(f"{case['case_id']} 在人物卡里找不到对应人物")
        else:
            for field in ("policy", "gold_action", "framing"):
                if character.get(field) != case.get(field):
                    issues.append(f"{case['case_id']} 用例 {field} 与人物卡不一致")
            if isinstance(case["repeats"], int) and isinstance(character.get("repeats"), int) \
                    and case["repeats"] != character["repeats"]:
                issues.append(f"{case['case_id']} repeats 与人物卡不一致")
        planned_slots += len(case["variants"]) * len(plan.get("channels") or []) * int(case["repeats"] or 0)
    budget = (plan.get("budgets") or {}).get("generation_http_attempts")
    if budget is not None and planned_slots != int(budget):
        issues.append(f"计划生图槽位 {planned_slots} 与预算 {budget} 不一致")

    # --- base-pack ---
    base_pack = plan.get("base_pack") or ""
    base_manifest_path = os.path.join(base_pack, "manifest.json")
    base_rows = []
    if not os.path.isfile(base_manifest_path):
        issues.append(f"缺少 base-pack manifest: {base_manifest_path}")
    else:
        base_manifest = json.load(open(base_manifest_path, "r", encoding="utf-8"))
        if not base_manifest.get("not_complete_generation_requests"):
            issues.append("base-pack 未声明 not_complete_generation_requests")
        if base_manifest.get("actual_network_attempts") not in (0, None):
            issues.append("base-pack 声明了非零在线尝试")
        for row in base_manifest.get("rows") or []:
            path = os.path.join(base_pack, row["file"])
            if not os.path.isfile(path):
                issues.append(f"base-pack 缺少文件 {row['file']}")
                continue
            actual = sha256_file(path)
            matched = actual == row.get("sha256")
            if not matched:
                issues.append(f"base-pack {row['file']} hash 与 manifest 不一致")
            base_rows.append({"case_id": row["case_id"], "file": row["file"],
                              "sha256_expected": row.get("sha256"), "sha256_actual": actual,
                              "match": matched})
        if len(base_rows) != len(cases):
            issues.append(f"base-pack 行数 {len(base_rows)} 与用例数 {len(cases)} 不一致")

    # --- 变体定义 ---
    variants = {item["id"]: item for item in plan.get("variants") or []}
    for required in ("v3_control", "v4_candidate"):
        if required not in variants:
            issues.append(f"plan 缺少变体定义 {required}")
    if variants.get("v3_control", {}).get("role") != "control":
        issues.append("v3_control 的角色必须是 control（冻结控制）")
    if variants.get("v4_candidate", {}).get("role") != "candidate":
        issues.append("v4_candidate 的角色必须是 candidate")
    if not variants.get("v3_control", {}).get("known_defects"):
        issues.append("v3_control 未声明已知缺陷（README 要求冻结缺陷只报告）")

    # --- 模板文件 ---
    plan_dir = os.path.dirname(spec_dir)
    template_files = plan.get("template_files") or [
        "variants/preserve.md", "variants/redesign-v3.md", "variants/redesign-v4.md",
        "variants/visible-v3.md", "variants/visible-v4.md",
        "variants/visible-v4-redesign.md", "variants/design.md",
        "variants/frame-full.md", "variants/frame-full-v4.md",
        "variants/frame-bust-v3.md", "variants/frame-bust-v4.md",
        "variants/policy-reinterpret.md", "variants/policy-fill_missing.md",
        "variants/policy-replace.md",
        "spec/review.md", "spec/score.md", "spec/calibrate.md",
        "spec/calibration-cases.json"]
    missing_templates = [name for name in template_files
                         if not os.path.isfile(os.path.join(plan_dir, name))]
    if missing_templates:
        issues.append(f"缺少模板文件: {missing_templates}")

    return {"ok": not issues, "issues": issues, "case_count": len(cases),
            "planned_generation_slots": planned_slots,
            "characters": sorted(seen_characters), "variants": sorted(variants),
            "base_pack_rows": base_rows, "spec_hash": spec_hash(spec_dir)}


def compose_case_prompt(spec_dir: str, case: dict, character: dict, variant: str) -> str:
    """组装一份完整生图 prompt：base + 分支块 +（v4）tailoring + 画幅收尾。"""
    plan = json.load(open(os.path.join(spec_dir, "plan.json"), "r", encoding="utf-8"))
    root = os.path.dirname(spec_dir)
    base_pack = plan.get("base_pack") or ""
    base = read_text(os.path.join(base_pack, f"{case['case_id']}-base.txt"))
    branch = case["branch"]
    if variant == "v3_control":
        branch_file = {"preserve": "preserve.md", "redesign": "redesign-v3.md",
                       "visible": "visible-v3.md"}[branch]
        if branch == "visible" and case["gold_action"] == "redesign":
            branch_file = "visible-v3-redesign.md"
    else:
        if branch == "visible" and case["gold_action"] == "redesign":
            branch_file = "visible-v4-redesign.md"
        else:
            branch_file = {"preserve": "preserve.md", "redesign": "redesign-v4.md",
                           "visible": "visible-v4.md"}[branch]
    values = {"rendering": plan.get("rendering") or character.get("rendering") or "",
              "identity": character["identity"], "source_clothing": character["source_clothing"],
              "request": character["request"], "action": str(case["gold_action"]).upper(),
              "policy": case["policy"], "design": ""}
    if branch == "redesign":
        policy_file = f"policy-{case['policy']}.md"
        values["policy_rule"] = read_text(os.path.join(root, "variants", policy_file)).strip()
        values["design"] = read_text(os.path.join(root, "variants", "design.md")).strip()
    if branch == "redesign" and variant == "v4_candidate":
        values["tailoring"] = read_text(os.path.join(root, "characters",
                                                    f"{case['case_id']}.md")).strip()
    branch_text = read_text(os.path.join(root, "variants", branch_file))
    blocks = [base.strip(), render_template(branch_text, values).strip()]
    if branch == "redesign" and variant == "v4_candidate" and values.get("tailoring"):
        # redesign-v4.md 已经把 tailoring 放在正文里；避免重复插入
        pass
    if branch == "preserve" and variant == "v4_candidate":
        blocks.append(read_text(os.path.join(root, "characters", f"{case['case_id']}.md")).strip())
    frame_file = ("frame-bust-v4.md" if variant == "v4_candidate" else "frame-bust-v3.md") \
        if case["framing"] == "upper_chest" else \
        ("frame-full-v4.md" if variant == "v4_candidate" else "frame-full.md")
    frame_text = read_text(os.path.join(root, "variants", frame_file))
    blocks.append(render_template(frame_text, values).strip())
    return "\n\n".join(block for block in blocks if block)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def compose_all(spec_dir: str) -> dict:
    """把 6 个用例 × 2 个版本组装成完整 prompt（返回 file/文本/hash/字符数）。"""
    doc = load_character_test(spec_dir)
    rows = []
    for case in doc["plan"]["cases"]:
        character = doc["by_case"][case["case_id"]]
        for variant in case["variants"]:
            text = compose_case_prompt(spec_dir, case, character, variant)
            rows.append({
                "case_id": case["case_id"], "character_id": character["id"],
                "variant": variant, "role": case["role"], "policy": case["policy"],
                "branch": case["branch"], "gold_action": case["gold_action"],
                "gold_presence": case["gold_presence"], "framing": case["framing"],
                "repeats": case["repeats"],
                "file": f"{case['case_id']}-{variant}.txt",
                "sha256": sha256_text(text), "characters": len(text), "prompt": text,
            })
    return {"spec_hash": spec_hash(spec_dir), "rows": rows}


def build_request_order(cases: list, channels: list, variants: list, *, seed: int = 0) -> list:
    """按 spec 生成生图槽位顺序：每 case×variant×channel×repeat 一个槽位，确定性打乱。"""
    slots = []
    for case in cases:
        for variant in variants:
            for channel in channels:
                for repeat in range(1, int(case["repeats"]) + 1):
                    slots.append({
                        "sample_id": f"{case['case_id']}-{channel}-r{repeat}-{variant}",
                        "pair_id": f"{case['case_id']}-{channel}-r{repeat}",
                        "case_id": case["case_id"], "channel": channel, "repeat": repeat,
                        "variant": variant, "prompt_file": f"{case['case_id']}-{variant}.txt"})
    rng = random.Random(seed)
    rng.shuffle(slots)
    return slots


def build_pairs_exact_balance(order: list, *, seed: int = 0) -> dict:
    """A/B 精确平衡：每个 pair 的 control/candidate 各占 A 一半。"""
    groups = []
    seen = set()
    for row in order:
        if row["pair_id"] in seen:
            continue
        seen.add(row["pair_id"])
        members = [item for item in order if item["pair_id"] == row["pair_id"]]
        groups.append({"pair_id": row["pair_id"], "case_id": row["case_id"],
                       "channel": row["channel"], "repeat": row["repeat"], "members": members})
    rng = random.Random(seed)
    rng.shuffle(groups)
    pairs = []
    for index, group in enumerate(groups):
        control = next((m for m in group["members"] if m["variant"] == "v3_control"), None)
        candidate = next((m for m in group["members"] if m["variant"] == "v4_candidate"), None)
        if not control or not candidate:
            continue
        # 精确平衡：偶数序号 A=control，奇数序号 A=candidate
        first_control = index % 2 == 0
        assignment = {"A": control["variant"], "B": candidate["variant"]} if first_control else \
            {"A": candidate["variant"], "B": control["variant"]}
        pairs.append({"pair_id": group["pair_id"], "case_id": group["case_id"],
                      "channel": group["channel"], "repeat": group["repeat"],
                      "assignment": assignment,
                      "sample_id": {"A": control["sample_id"] if first_control else candidate["sample_id"],
                                    "B": candidate["sample_id"] if first_control else control["sample_id"]}})
    a_control = sum(1 for item in pairs if item["assignment"]["A"] == "v3_control")
    balance = {"pairs": len(pairs), "a_control": a_control, "a_candidate": len(pairs) - a_control,
               "exact_balance": a_control == len(pairs) // 2 and len(pairs) % 2 == 0,
               "seed": seed}
    return {"pairs": pairs, "balance": balance}


def calibration_ok_with_overrides(calibration: dict) -> dict:
    """校准门禁的判定：通过，或**有用户显式放行记录**。

    放行不是「静默跳过」：调用方必须在 `--grant-override` 里写明假设与已知风险，
    记录进 `ledger.budget_grants` 并同时写进 `calibration_overrides`。这里只认带
    完整字段（kind/authorized_by/reason/assumed_hypothesis/known_risk/accepted_at）
    的放行，缺一项就仍按未通过处理。
    """
    calibration = calibration or {}
    if calibration.get("ok") is True:
        return {"ok": True, "overridden": False, "reason": "校准通过"}
    required = ("kind", "authorized_by", "reason", "assumed_hypothesis", "known_risk", "accepted_at")
    for record in calibration.get("calibration_overrides") or []:
        if all(str(record.get(field) or "").strip() for field in required):
            return {"ok": True, "overridden": True,
                    "reason": f"用户显式放行（{record['authorized_by']}）：{record['reason']}",
                    "record": record}
    return {"ok": False, "overridden": False, "reason": "看图校准未通过且没有显式放行记录"}


def candidate_readiness(spec_dir: str, state: dict, prepare_report: dict = None,
                       review: dict = None, calibration: dict = None) -> dict:
    """候选就绪门禁：控制缺陷只报告，候选审查/prepare/校准失败拦一切生成入口。

    审查结论还必须**绑定当前素材**：素材一改（spec 指纹变了），上一次审查的
    ready=true 就不再算数——否则改完模板直接沿用旧结论就能放行生图。
    """
    reasons = []
    notes = []
    plan = json.load(open(os.path.join(spec_dir, "plan.json"), "r", encoding="utf-8"))
    current_hash = spec_hash(spec_dir)
    state = state or {}
    if not state.get("prepare_ok"):
        reasons.append("prepare 未通过（spec/hash/确定性检查）")
    if state.get("spec_hash") != current_hash:
        reasons.append("就绪结论与当前 spec 指纹不匹配（素材已变更）")
    review = review if review is not None else state.get("review") or {}
    if review.get("candidate_ready") is not True:
        reasons.append("候选审查未就绪（review.ready=false 或缺失）")
    elif review.get("spec_hash") != current_hash:
        reasons.append(f"审查结论来自旧素材（审查指纹 {str(review.get('spec_hash'))[:12]}，"
                       f"当前 {current_hash[:12]}）：素材已变更，必须重跑审查")
    if review.get("blocking"):
        reasons.append(f"候选审查仍有 blocking: {len(review['blocking'])} 条")
    calibration = calibration if calibration is not None else state.get("calibrate") or {}
    gate = calibration_ok_with_overrides(calibration)
    if not gate["ok"]:
        reasons.append("看图校准未通过或未执行")
    elif gate["overridden"]:
        notes.append(gate["reason"])
    return {"ready": not reasons, "reasons": reasons, "overrides": notes, "spec_hash": current_hash,
            "control_id": "v3_control", "candidate_id": "v4_candidate",
            "control_role": "frozen_control_known_defects_reported_only"}


def scoring_conditions(case: dict, character: dict) -> dict:
    """评分输入的真实条件：人物/原衣物/用户要求/protected 原样带上，隐藏版本身份。"""
    return {
        "case_id": case["case_id"], "role": case["role"],
        "authorized_clothing_action": case["gold_action"],
        "framing_requirement": ("close portrait from the upper chest upward"
                                if case["framing"] == "upper_chest"
                                else "full-body view including both feet"),
        "identity": character["identity"],
        "source_clothing": character["source_clothing"],
        "user_request": character["request"],
        "protected": list(character.get("protected") or []),
        "note": ("Judge each image only against the conditions above. Do not borrow garments, "
                 "colours or accessories from the other image. Without a source image, exact face "
                 "identity and exact original cut cannot be verified."),
    }


def frozen_pack_digest(pack: dict) -> str:
    body = {key: value for key, value in pack.items() if key != "sha256"}
    return sha256_text(json.dumps(body, ensure_ascii=False, sort_keys=True))


def load_frozen_generation_pack(path: str) -> dict:
    """Validate a generation-only snapshot before creating output or sending."""
    from utils.wardrobe import wardrobe_prompt
    with open(path, encoding="utf-8") as handle:
        pack = json.load(handle)
    if pack.get("schema_version") != "image-maker.wardrobe-generation-pack.v1":
        raise ValueError("Unsupported frozen generation pack")
    if pack.get("sha256") != frozen_pack_digest(pack):
        raise ValueError("Frozen pack checksum mismatch")
    requests = pack.get("requests") or []
    ids = [row["sample_id"] for row in requests]
    if not requests or len(set(ids)) != len(ids):
        raise ValueError("Empty pack or duplicate sample IDs")
    if pack.get("generation_http_attempts") != len(requests):
        raise ValueError("Generation budget must match frozen slots")
    for row in requests:
        if row.get("channel") not in {"gpt", "gemini"}:
            raise ValueError("Unsupported generation channel")
        if sha256_text(row["prompt"]) != row["prompt_sha256"]:
            raise ValueError("Prompt checksum mismatch: " + row["sample_id"])
        params = row.get("params") or {}
        if not params.get("model") or not params.get("api_type"):
            raise ValueError("Missing frozen model or API node")
        if params.get("n", 1) != 1 or params.get("face_quality_boost", False):
            raise ValueError("Only one image call per frozen slot is permitted")
        block = wardrobe_prompt(row["wardrobe"])
        if row["prompt"].count(block) != 1:
            raise ValueError("Wardrobe block must appear exactly once")
        refs = row.get("image_paths") or []
        if len(refs) != len(row.get("reference_sha256") or []):
            raise ValueError("Reference manifest mismatch")
        for image, digest in zip(refs, row["reference_sha256"]):
            if sha256_file(image) != digest:
                raise ValueError("Reference checksum mismatch: " + image)
    return pack
