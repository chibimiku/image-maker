#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""衣装实验族 CLI（参数化：预算、授权路由、快照、生图、看图校准与评价）。

为什么是「族」而不是一次性脚本：第二轮任务书要求同一套编排同时支撑
「解析 → 就绪审查 → 看图校准 → 32 个生图槽 → 16 组匿名评价」，
并带**发送前扣减**的持久预算、崩溃后不重置、失败原样保留。
所有用例/矩阵/prompt 都从 `--spec` 指向的目录与文件读，运行逻辑里不硬编码 C03、
不硬编码模型与目录（模型与参数从 spec 里的 generation 段或首轮 run-manifest 读）。

子命令：
  --prepare             只做快照核对 + 确定性检查 + 预算/参数解析（不发任何请求）
  --list-models         实时模型清单核验（独立预算 1 次，单列，不算推理请求）
  --resolver            12 例授权解析（文本预算 1 次，gold 不进请求）
  --review              一次文本就绪审查（文本预算 1 次）
  --calibrate           已有图看图校准（视觉预算）
  --generate            按 spec 顺序生图（生图预算 32 次，一次尝试=一次 HTTP）
  --score               16 组匿名评价（视觉预算）
  --run                 依次执行 resolver → review → calibrate → generate → score
  --mock               离线跑通预算/失败/恢复（不联网），用于在线前自检
  --status / --ledger   只看账本

预算一律「发送前扣减」：每次真实 HTTP 之前先在 send_budget 的原子槽位里占位并结算，
崩溃后 inflight 视为已消耗。图片通道额外设 IMAGE_MAKER_IMAGE_MAX_RETRIES=0 关闭隐藏重试。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils import send_budget  # noqa: E402
from utils.atomic_io import write_json_atomic  # noqa: E402
from utils.wardrobe_experiment import (  # noqa: E402
    BudgetExceeded,
    ExperimentLedger,
    assert_online_supported,
    build_pairs_exact_balance,
    build_request_order,
    candidate_readiness,
    check_ledger_budgets,
    compose_all,
    content_block_order,
    deterministic_prompt_checks,
    gold_routes,
    load_character_test,
    normalize_budgets,
    read_text,
    resolve_run_dir,
    resolver_request_payload,
    scoring_conditions,
    scrub_request_snapshot,
    sha256_file,
    sha256_text,
    snapshot_hash_check,
    spec_hash,
    validate_character_spec,
    validate_resolver_output,
)

DEFAULT_SPEC = os.path.join("prompts", "wardrobe", "round3", "spec")
DEFAULT_PACK = os.path.join("prompts", "wardrobe", "round3", "pack")
# 追加额度只认这四类（与任务书预算表一一对应）
BUDGET_KINDS = ("text", "generation", "vision", "model_list")
# 允许被用户显式放行的门（必须有假设与已知风险；放行 ≠ 通过）
OVERRIDE_KINDS = ("calibrate",)


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _log(message: str) -> None:
    print(message, flush=True)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, data) -> str:
    write_json_atomic(path, data)
    return path


def _append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def text_api_config() -> dict:
    from utils.analysis_gpt_prompt import load_text_api_config
    return load_text_api_config()


def _currency_free_config_summary() -> dict:
    """配置摘要：只给端点/模型与凭据来源，绝不输出密钥值。"""
    from modules.others import api_backend
    summary = {}
    for api_type in ("aigc-2d-gpt", "aigc2d"):
        cfg = api_backend.get_api_config(config_path="conf/config.json", api_type=api_type)
        summary[api_type] = {"base_url": cfg.get("base_url"), "model": cfg.get("model"),
                             "timeout": cfg.get("timeout"), "resolution": cfg.get("resolution"),
                             "key_present": bool(str(cfg.get("api_key") or "").strip()),
                             "key_source": api_backend.api_key_source(api_type=api_type, cfg=cfg)}
    text_cfg = text_api_config()
    summary["text"] = {"base_url": text_cfg.get("base_url"), "model": text_cfg.get("model"),
                       "key_present": bool(str(text_cfg.get("api_key") or "").strip())}
    return summary


def http_post_once(url: str, *, headers: dict, payload: dict, timeout: int):
    """**单次** HTTP POST：不重试、不跟随备用端点（备用由调用方显式记账）。"""
    import requests
    started = time.perf_counter()
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return response, elapsed_ms


def parse_json_object(text: str):
    body = str(text or "").strip()
    body = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", body)
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(body[start:end + 1])
    except Exception:  # noqa: BLE001
        return None


def scrub_credential_files(out_dir: str) -> dict:
    """清掉后端生成的「请求回放」文件：它们的 Authorization 头里带未掩码的 key。

    `return_metadata=True` 会让 `api_backend` 另外写一份 `*_replay_*.json` + `*.py`
    （内含真实密钥）。本轮要求「绝不存凭据」，所以每次生成后立刻删除，
    并在账本目录留一条记录说明删了什么、为什么。请求正文本身仍可从
    `samples.json` 的 prompt 全文与 `*_server_response_*.json` 复核。
    """
    removed = []
    for root, _dirs, files in os.walk(out_dir):
        for name in files:
            if "_replay_" in name and name.endswith((".json", ".py")):
                path = os.path.join(root, name)
                try:
                    os.remove(path)
                    removed.append(os.path.basename(path))
                except OSError:
                    pass
    return {"removed_count": len(removed), "removed": sorted(removed)[:50]}


# ---------------------------------------------------------------------------
# 实验上下文
# ---------------------------------------------------------------------------

class Experiment:
    def __init__(self, args):
        self.args = args
        self.spec_dir = os.path.abspath(args.spec)
        self.pack_dir = os.path.abspath(args.pack)
        self.plan = _read_json(os.path.join(self.spec_dir, "plan.json"))
        self.manifest = self._load_manifest()
        # 轮次标签从 plan 的 schema/round_label 派生，不再从目录名猜（目录名一改就会把
        # run-id 与 `_spec_source_files` 的路径前缀一起带到别的轮次上）。
        self.plan_schema = str(self.plan.get("schema_version") or "")
        slug = str(self.plan.get("round_label") or "").strip()
        if not slug:
            slug = ("round3" if self.plan_schema.endswith("-round3.v1") else
                    "round2" if self.plan_schema.endswith("-round2.v1") else
                    re.sub(r"[^a-z0-9]+", "-", self.plan_schema.replace("image-maker.wardrobe-", "")
                           .replace(".v1", "").replace(".planning", "")).strip("-") or "experiment")
        self.round_label = slug
        self.pack_root_label = str(self.plan.get("pack_root_label")
                                   or os.path.dirname(self.spec_dir).replace("\\", "/"))
        self.run_id = args.run_id or f"wardrobe-{slug}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        # 跨天复跑必须回到原目录与原账本：--run-dir 优先，默认才按当天日期推导
        self.out_dir = resolve_run_dir(args.output_root, self.run_id,
                                       getattr(args, "run_dir", "") or "")
        if getattr(args, "run_dir", ""):
            self.run_dir_source = "explicit --run-dir"
        else:
            self.run_dir_source = f"default output_root/{datetime.now().strftime('%Y%m%d')}/run-id"
        # 尚未支持 schema 的计划**连目录都不建**：2026-10-05 的 round4 冒烟测试里，
        # `--compose/--generate` 在守卫拦住请求之前已经在 data/test-result 下留下了
        # 两个空的 `<run-id>` 目录（没有账本、没有产物），清理起来还要靠人工判断。
        # 计划本身的 schema 与 run-dir 的名字无关，所以这里先拦住，保持产物目录干净。
        unsupported = validate_character_spec(self.spec_dir).get("unsupported_schema")
        if unsupported:
            raise RuntimeError(f"plan.schema_version={unsupported}：本工具只实现 round3 的 "
                               "spec 校验、组装与在线守卫；拒绝为该计划创建 run 目录或账本")
        os.makedirs(self.out_dir, exist_ok=True)
        ledger_path = os.path.join(self.out_dir, "ledger.json")
        existing_ledger = _read_json(ledger_path) if os.path.isfile(ledger_path) else None
        if existing_ledger:
            # 原账本的上限必须与 spec 一致：不接受「换目录顺手换上限」
            budget_issues = check_ledger_budgets(existing_ledger.get("budgets"), self.plan.get("budgets"))
            if budget_issues:
                raise RuntimeError("账本预算与 spec 不一致，拒绝继续：" +
                                   "; ".join(f"{item['budget']} 账本={item['ledger']} spec={item['spec']}"
                                             for item in budget_issues))
        self.ledger = ExperimentLedger.load(ledger_path, normalize_budgets(self.plan.get("budgets")),
                                            trust_persisted_budgets=bool(existing_ledger))
        if self.ledger.data.get("run_id") != self.run_id:
            self.ledger.data["run_id"] = self.run_id
        if existing_ledger:
            # 账本上的**实际**上限为准（含用户显式授权的追加额度），
            # 绝不用 spec 的上限把已授权的额度压回去、也绝不用它重置已消耗
            self.ledger.data["budgets"] = normalize_budgets(existing_ledger.get("budgets"))
            for kind in BUDGET_KINDS:
                self.ledger.assert_budget_floor(kind)
        self.recovered = self.ledger.recover_inflight()
        self.slots_dir = os.path.join(self.out_dir, "budget-slots")
        # 生图通道关闭隐藏重试（一次付费动作 = 一次 HTTP）
        os.environ["IMAGE_MAKER_IMAGE_MAX_RETRIES"] = "0"
        os.environ["IMAGE_MAKER_TEST_OUTPUT"] = "1"
        self.backend_sub_dir = self._resolve_backend_sub_dir()
        self.gold = gold_routes(self.plan)
        self.state_path = os.path.join(self.out_dir, "state.json")
        self.state = _read_json(self.state_path) if os.path.isfile(self.state_path) else {
            "run_id": self.run_id, "created_at": datetime.now().isoformat()}

    # ---------------- 状态 ----------------
    def _resolve_backend_sub_dir(self) -> str:
        """后端落盘用的相对子目录：相对于当天 `data/<日期>`。

        `--run-dir` 可能指向别的盘/别的日期（跨天恢复）。那时 `relpath` 会直接抛
        `ValueError: path is on mount 'C:'`，所以这里：能算相对路径就算，不能算就
        退回「相对仓库根」，再不能算就用绝对路径——落盘位置始终确定、可复核，
        绝不因为换目录而丢产物。
        """
        day_root = os.path.abspath(os.path.join("data", datetime.now().strftime("%Y%m%d")))
        for base in (day_root, BASE_DIR):
            try:
                return os.path.relpath(self.out_dir, base).replace("\\", "/")
            except ValueError:
                continue
        return self.out_dir.replace("\\", "/")

    def _load_manifest(self) -> dict:
        """读取 pack manifest；round3 这类「由 spec 组装」的实验允许首轮不存在。"""
        path = os.path.join(self.pack_dir, "manifest.json")
        if os.path.isfile(path):
            return _read_json(path)
        if self.plan.get("characters_file"):
            return {"status": "composed_from_spec", "snapshots": [], "source_files": {},
                    "plan_sha256": "", "prior_results_sha256": ""}
        raise FileNotFoundError(path)

    def is_composed_spec(self) -> bool:
        return bool(self.plan.get("characters_file"))

    def compose_pack(self) -> dict:
        """按 spec 组装 12 份完整 prompt 并写进 pack 目录（含 hash 与逐处差异）。

        round3 的 base-pack 只是共同基础文本，**不是**最终生图请求；
        这里把 base + 分支块 +（v4）tailoring + 画幅收尾 拼成完整请求。
        """
        composed = compose_all(self.spec_dir)
        os.makedirs(self.pack_dir, exist_ok=True)
        rows = []
        for row in composed["rows"]:
            path = os.path.join(self.pack_dir, row["file"])
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(row["prompt"])
            base = read_text(os.path.join(self.plan["base_pack"], f"{row['case_id']}-base.txt"))
            strict = row["variant"] == "v4_candidate"
            checks = deterministic_prompt_checks(row["prompt"], base, variant=row["variant"],
                                                 gold_action=row["gold_action"],
                                                 framing=row["framing"], strict=strict)
            rows.append({**{key: value for key, value in row.items() if key != "prompt"},
                         "checks": checks})
        manifest = {
            "status": "composed_from_spec", "composed_at": datetime.now().isoformat(),
            "not_complete_generation_requests": False,
            "plan_sha256": sha256_file(os.path.join(self.spec_dir, "plan.json")),
            "characters_sha256": sha256_file(os.path.join(self.spec_dir, "..",
                                                          "characters.json").replace("\\", "/"))
            if os.path.isfile(os.path.join(self.spec_dir, "..", "characters.json")) else "",
            "spec_hash": composed["spec_hash"],
            "variants": [item["id"] for item in self.plan.get("variants") or []],
            "control_id": "v3_control", "candidate_id": "v4_candidate",
            "snapshot_count": len(rows),
            "generation_slots": sum(row["repeats"] for row in rows
                                    if row["variant"] == "v4_candidate")
            * len(self.plan.get("channels") or []),
            "snapshots": [{"case_id": row["case_id"], "character_id": row["character_id"],
                           "variant": row["variant"], "role": row["role"],
                           "gold_action": row["gold_action"], "framing": row["framing"],
                           "file": row["file"], "sha256": row["sha256"],
                           "characters": row["characters"]} for row in rows],
            "source_files": self._spec_source_files(),
        }
        manifest["request_order"] = [{
            **slot, "prompt_sha256": next(row["sha256"] for row in rows
                                          if row["file"] == slot["prompt_file"])
        } for slot in build_request_order(self.plan["cases"], self.plan.get("channels") or [],
                                         ["v3_control", "v4_candidate"],
                                         seed=int(self.plan.get("pairing", {}).get("seed") or 0))]
        pairs = build_pairs_exact_balance(manifest["request_order"],
                                          seed=int(self.plan.get("pairing", {}).get("seed") or 0))
        manifest["pairing"] = pairs["balance"]
        _write_json(os.path.join(self.pack_dir, "manifest.json"), manifest)
        self.manifest = manifest
        self._write_diffs(composed)
        return manifest

    def _spec_source_files(self) -> dict:
        """记录参与组装的 spec 文件 hash（模板/人物卡/base-pack/规则）。

        路径前缀取自 plan 的 `pack_root_label`（缺省＝spec 的父目录），
        不再写死 `prompts/wardrobe/round3`。
        """
        rows = {}
        root = os.path.dirname(self.spec_dir)
        prefix = self.pack_root_label

        def label(relative: str) -> str:
            return os.path.join(prefix, relative).replace("\\", "/")

        for relative in ("characters.json", "base-pack/manifest.json"):
            path = os.path.join(root, relative)
            if os.path.isfile(path):
                rows[label(relative)] = sha256_file(path)
        for folder in ("variants", "spec", "characters"):
            full = os.path.join(root, folder)
            for name in sorted(os.listdir(full)) if os.path.isdir(full) else []:
                path = os.path.join(full, name)
                if os.path.isfile(path):
                    rows[label(os.path.join(folder, name))] = sha256_file(path)
        for name in sorted(os.listdir(self.plan.get("base_pack") or ".")) \
                if os.path.isdir(self.plan.get("base_pack") or "") else []:
            if name.endswith(".txt"):
                path = os.path.join(self.plan["base_pack"], name)
                rows[os.path.relpath(path, BASE_DIR).replace("\\", "/")] = sha256_file(path)
        return rows

    def _arm_checks(self, row: dict) -> dict:
        base = read_text(os.path.join(self.plan["base_pack"], f"{row['case_id']}-base.txt"))
        return deterministic_prompt_checks(row["prompt"], base, variant=row["variant"],
                                          gold_action=row["gold_action"], framing=row["framing"],
                                          strict=row["variant"] == "v4_candidate")

    def _write_diffs(self, composed: dict) -> dict:
        """逐处差异：同 case 的 control 与 candidate 对比（段落级 + 字符/hash）。"""
        by_case = {}
        for row in composed["rows"]:
            by_case.setdefault(row["case_id"], {})[row["variant"]] = row
        diffs = []
        for case_id, arms in sorted(by_case.items()):
            control, candidate = arms.get("v3_control"), arms.get("v4_candidate")
            if not control or not candidate:
                continue
            control_blocks = [block for block in control["prompt"].split("\n\n") if block.strip()]
            candidate_blocks = [block for block in candidate["prompt"].split("\n\n") if block.strip()]
            control_checks = self._arm_checks(control)
            candidate_checks = self._arm_checks(candidate)
            diffs.append({
                "case_id": case_id,
                "control": {"file": control["file"], "sha256": control["sha256"],
                            "characters": control["characters"],
                            "block_count": len(control_blocks),
                            "checks_ok": control_checks["ok"],
                            "expected_defects": [item["check"] for item in control_checks["issues"]]},
                "candidate": {"file": candidate["file"], "sha256": candidate["sha256"],
                              "characters": candidate["characters"],
                              "block_count": len(candidate_blocks),
                              "checks_ok": candidate_checks["ok"],
                              "issues": [item["check"] for item in candidate_checks["issues"]]},
                "only_in_control": [block[:160] for block in control_blocks
                                    if block not in candidate_blocks],
                "only_in_candidate": [block[:160] for block in candidate_blocks
                                      if block not in control_blocks],
                "characters_delta": candidate["characters"] - control["characters"],
                "same_sha256": control["sha256"] == candidate["sha256"],
            })
        report = {"spec_hash": composed["spec_hash"], "generated_at": datetime.now().isoformat(),
                  "cases": diffs,
                  "note": ("两臂共享 base 与画法正文；差异只在分支块组织、v4 的 tailoring 段与画幅收尾。"
                           "控制臂的 includes 属预登记缺陷，只报告不阻塞。")}
        _write_json(os.path.join(self.out_dir, "arm-diff.json"), report)
        return report

    def write_offline_suite(self) -> dict:
        """离线准备自检：spec 校验 + 确定性检查 + 配对平衡（不发任何请求）。"""
        validation = validate_character_spec(self.spec_dir)
        composed = compose_all(self.spec_dir)
        order = build_request_order(self.plan["cases"], self.plan.get("channels") or [],
                                    ["v3_control", "v4_candidate"],
                                    seed=int(self.plan.get("pairing", {}).get("seed") or 0))
        pairs = build_pairs_exact_balance(order, seed=int(self.plan.get("pairing", {}).get("seed") or 0))
        checks = []
        for row in composed["rows"]:
            base = read_text(os.path.join(self.plan["base_pack"], f"{row['case_id']}-base.txt"))
            checks.append({"case_id": row["case_id"], "variant": row["variant"],
                           "checks": deterministic_prompt_checks(
                               row["prompt"], base, variant=row["variant"],
                               gold_action=row["gold_action"], framing=row["framing"],
                               strict=row["variant"] == "v4_candidate")})
        report = {"spec_hash": composed["spec_hash"], "validation": validation,
                  "checks": checks, "pairing": pairs["balance"],
                  "order_slots": len(order),
                  "candidate_failures": [(item["case_id"], item["variant"],
                                          [issue["check"] for issue in item["checks"]["issues"]])
                                         for item in checks
                                         if item["variant"] == "v4_candidate"
                                         and not item["checks"]["ok"]],
                  "candidate_ok": all(item["checks"]["ok"] for item in checks
                                      if item["variant"] == "v4_candidate"),
                  "written_at": datetime.now().isoformat()}
        _write_json(os.path.join(self.out_dir, "offline-suite.json"), report)
        return report

    def candidate_gate(self) -> dict:
        """候选就绪门禁：控制缺陷只报告；候选/prepare/校准失败拦所有生成入口。"""
        review = self.stage("review")
        calibration = self.stage("calibrate")
        gate = candidate_readiness(self.spec_dir, {
            "prepare_ok": self.state.get("prepare_ok"),
            "spec_hash": self.state.get("spec_hash"),
            "review": {"candidate_ready": review.get("candidate_ready"),
                       "blocking": review.get("blocking") or [],
                       "spec_hash": review.get("spec_hash")},
            "calibrate": calibration,
        })
        _write_json(os.path.join(self.out_dir, "candidate-gate.json"), gate)
        return gate

    def save_state(self) -> None:
        self.state["updated_at"] = datetime.now().isoformat()
        _write_json(self.state_path, self.state)

    def set_stage(self, stage: str, payload: dict) -> None:
        self.state.setdefault("stages", {})[stage] = payload
        self.save_state()

    def stage(self, name: str) -> dict:
        return (self.state.get("stages") or {}).get(name) or {}

    def stage_case(self, name: str, case_id: str) -> dict:
        """取某阶段里某条用例的小结（历史归档用）。"""
        return ((self.stage(name).get("cases") or {}).get(case_id)) or {}

    # ---------------- 预算 ----------------
    def _budget_root(self, kind: str) -> str:
        return os.path.join(self.slots_dir, kind)

    def reserve(self, kind: str, label: str, limit_key: str, **meta):
        """发送前占用一个原子槽位；占用即消耗（失败/放弃都不退还）。"""
        self.ledger.require(kind)
        root = self._budget_root(kind)
        old_dir = os.environ.get(send_budget.ENV_DIR)
        old_limit = os.environ.get(send_budget.ENV_LIMIT)
        old_per_stage = os.environ.get(send_budget.ENV_TEXT_PER_STAGE)
        os.environ[send_budget.ENV_DIR] = root
        os.environ[send_budget.ENV_LIMIT] = str(int(self.ledger.data["budgets"][limit_key]))
        if kind == "text":
            # `send_budget` 另有「每阶段最多 2 次」的默认护栏（防连打）。本轮账本
            # 是唯一账本，它的上限与追加授权已经覆盖这个问题，所以把该护栏对齐到
            # **本账本允许的总次数**——不然用户显式追加的额度会被这一层静默吞掉。
            os.environ[send_budget.ENV_TEXT_PER_STAGE] = str(
                int(self.ledger.data["budgets"][limit_key]))
        try:
            if kind == "generation":
                reservation = send_budget.reserve_image_attempt(
                    run_id=label, operation=str(meta.get("operation") or ""),
                    model=str(meta.get("model") or ""), prompt_sha256=str(meta.get("prompt_sha256") or ""),
                    prompt_chars=int(meta.get("prompt_chars") or 0), note=str(meta.get("note") or ""))
            else:
                stage = f"{kind}:{label}"
                reservation = send_budget.reserve_text_attempt(stage=stage, run_id=label,
                                                               purpose=str(meta.get("operation") or ""))
            attempt = self.ledger.plan_attempt(kind, slot=label,
                                              label=str(meta.get("operation") or ""))
            return {"reservation": reservation, "ledger_attempt": attempt}
        finally:
            if old_dir is None:
                os.environ.pop(send_budget.ENV_DIR, None)
            else:
                os.environ[send_budget.ENV_DIR] = old_dir
            if old_limit is None:
                os.environ.pop(send_budget.ENV_LIMIT, None)
            else:
                os.environ[send_budget.ENV_LIMIT] = old_limit
            if old_per_stage is None:
                os.environ.pop(send_budget.ENV_TEXT_PER_STAGE, None)
            else:
                os.environ[send_budget.ENV_TEXT_PER_STAGE] = old_per_stage

    def settle(self, kind: str, reserved: dict, *, status: str, **extra) -> None:
        root = self._budget_root(kind)
        old_dir = os.environ.get(send_budget.ENV_DIR)
        os.environ[send_budget.ENV_DIR] = root
        try:
            if kind == "generation":
                send_budget.settle_image_attempt(reserved.get("reservation") or {}, status=status,
                                                 output_path=str(extra.get("output_path") or ""),
                                                 output_sha256=str(extra.get("output_sha256") or ""),
                                                 error=str(extra.get("error") or ""))
            else:
                send_budget.settle_text_attempt(reserved.get("reservation") or {}, status=status,
                                                error=str(extra.get("error") or ""))
        finally:
            if old_dir is None:
                os.environ.pop(send_budget.ENV_DIR, None)
            else:
                os.environ[send_budget.ENV_DIR] = old_dir
        self.ledger.finish_attempt(reserved["ledger_attempt"], status=status,
                                   http_status=extra.get("http_status"),
                                   model=str(extra.get("model") or ""),
                                   error=str(extra.get("error") or ""),
                                   elapsed_ms=int(extra.get("elapsed_ms") or 0),
                                   extra=extra.get("extra"))

    def spent_marker(self, kind: str, label: str) -> bool:
        """该槽位之前是否已经发过（用于避免重复发送同一计划槽位）。"""
        return any(item.get("kind") == kind and item.get("slot") == label
                   for item in self.ledger.data["attempts"])

    # ---------------- 生图参数 ----------------
    def spec_generation_params(self) -> dict:
        """从 spec 读生图参数；缺省时回落到首轮 run-manifest 的冻结设置。

        运行逻辑里不硬编码模型名：模型/尺寸/画质都来自 spec 或首轮清单，
        核对不到就报错停止，绝不自动换模型。
        """
        cached = self.state.get("generation_params")
        if cached:
            return cached
        spec_params = self.plan.get("generation_params") or {}
        if spec_params:
            self.state["generation_params"] = spec_params
            self.save_state()
            return spec_params
        manifest_path = "data/test-result/20261005/wardrobe-pilot-v1/run-manifest.json"
        if not os.path.isfile(manifest_path):
            raise RuntimeError("找不到首轮 run-manifest，无法确定冻结的生图参数")
        prior = _read_json(manifest_path)["frozen_generation_params"]
        params = {"gpt": dict(prior["gpt"]), "gemini": dict(prior["gemini"])}
        params["source"] = "prior_run_manifest:data/test-result/20261005/wardrobe-pilot-v1/run-manifest.json"
        self.state["generation_params"] = params
        self.save_state()
        return params

    # ---------------- 快照 ----------------
    def base_prompt(self, case_id: str) -> str:
        """当前 spec 的基座文本。

        - 由 spec 组装的实验（round3）：base-pack 里的 `<case>-base.txt`；
        - 早期实验（round2）：rendering + 两个换行 + content（从 test-plan.json 读）。
        """
        if self.is_composed_spec():
            path = os.path.join(self.plan["base_pack"], f"{case_id}-base.txt")
            return read_text(path) if os.path.isfile(path) else ""
        plan = _read_json(os.path.join("prompts", "wardrobe", "test-plan.json"))
        rendering = plan["rendering"]
        content = ""
        for case in plan["cases"]:
            if case["id"] == case_id:
                content = case["content"]
                break
        return rendering + "\n\n" + content

    def verify(self) -> dict:
        """快照 hash + 基座对账 + 确定性检查 + gold 路由（全部离线）。

        由 spec 组装的实验会先用 `compose_pack()` 重新生成 12 份完整 prompt，
        所以这里校验的永远是「当前 spec 组出来的请求」，而不是历史文件。
        候选臂必须 strict 通过；控制臂按 `strict=False` 只报告预登记缺陷。
        """
        composed = None
        if self.is_composed_spec():
            composed = self.compose_pack()
        pack_hashes = snapshot_hash_check(self.manifest, self.pack_dir)
        source_hash_rows = []
        for relative, expected in (self.manifest.get("source_files") or {}).items():
            path = relative.replace("\\", os.sep)
            actual = sha256_file(path) if os.path.isfile(path) else ""
            source_hash_rows.append({"file": relative, "expected": expected, "actual": actual,
                                     "match": actual == expected})
        strict_pass = True
        prior_results = self.manifest.get("prior_results_sha256", "")
        prior_path = "data/test-result/20261005/wardrobe-pilot-v1/results.json"
        prior_actual = sha256_file(prior_path) if os.path.isfile(prior_path) else ""
        checks = []
        prior_by_id = {}
        if os.path.isfile(prior_path):
            prior = _read_json(prior_path)
            for sample in prior.get("samples") or []:
                prior_by_id[sample["sample_id"]] = sample
        for item in self.manifest.get("snapshots") or []:
            prompt = _read_text(os.path.join(self.pack_dir, item["file"]))
            base = self.base_prompt(item["case_id"])
            strict = item.get("variant") != "v3_control"
            result = deterministic_prompt_checks(prompt, base, variant=item["variant"],
                                                 gold_action=item["gold_action"],
                                                 framing=item["framing"], strict=strict)
            if item["variant"] == "v4_candidate" and not result["ok"]:
                strict_pass = False
            prior_comparison = None
            prior_sample = prior_by_id.get(f"{item['case_id']}-lolita-{item.get('channel', 'gpt')}")
            if prior_sample:
                prior_comparison = {"prior_prompt_sha256": prior_sample.get("prompt_sha256"),
                                    "snapshot_sha256": item["sha256"],
                                    "identical": prior_sample.get("prompt_sha256") == item["sha256"]}
            checks.append({"file": item["file"], "case_id": item["case_id"],
                           "variant": item["variant"], "role": item.get("role"),
                           "gold_action": item["gold_action"], "framing": item["framing"],
                           "sha256": item["sha256"], "characters": item["characters"],
                           "deterministic": result, "prior_snapshot": prior_comparison,
                           "severity": "candidate" if item["variant"] == "v4_candidate" else "control"})
        offline = self.write_offline_suite()
        gate_state = {
            "prepare_ok": bool(pack_hashes["ok"] and strict_pass and offline["validation"]["ok"]
                               and offline["pairing"]["exact_balance"]),
            "spec_hash": self.state.get("spec_hash"),
            "review": {"candidate_ready": (self.stage("review") or {}).get("candidate_ready"),
                       "blocking": (self.stage("review") or {}).get("blocking") or [],
                       "spec_hash": (self.stage("review") or {}).get("spec_hash")},
            "calibrate": self.stage("calibrate"),
        }
        gate = candidate_readiness(self.spec_dir, gate_state)
        report = {
            "run_id": self.run_id, "generated_at": datetime.now().isoformat(),
            "spec_dir": self.spec_dir, "pack_dir": self.pack_dir,
            "out_dir": self.out_dir, "run_dir_source": getattr(self, "run_dir_source", ""),
            "composed_from_spec": bool(composed),
            "plan_sha256": {"expected": self.manifest.get("plan_sha256"),
                            "actual": sha256_file(os.path.join(self.spec_dir, "plan.json"))},
            "spec_hash": spec_hash(self.spec_dir),
            "pack_snapshot_hash_check": pack_hashes,
            "source_file_hash_check": {"ok": all(row["match"] for row in source_hash_rows),
                                       "rows": source_hash_rows},
            "prior_results_sha256": {"expected": prior_results, "actual": prior_actual,
                                     "match": prior_results == prior_actual},
            "deterministic_checks": checks,
            "candidate_strict_pass": strict_pass,
            "offline_suite": {key: offline[key] for key in
                              ("candidate_ok", "candidate_failures", "pairing", "order_slots")},
            "gold_routes": self.gold,
            "budgets": self.ledger.data["budgets"],
            "config_summary": _currency_free_config_summary(),
            "output_isolation": {"isolated_dir": self.out_dir,
                                 "backend_sub_dir": self.backend_sub_dir,
                                 "env": os.environ.get("IMAGE_MAKER_TEST_OUTPUT")},
            "candidate_gate": gate,
        }
        ok = (pack_hashes["ok"] and all(row["match"] for row in source_hash_rows)
              and report["plan_sha256"]["expected"] == report["plan_sha256"]["actual"]
              and strict_pass and offline["validation"]["ok"]
              and offline["pairing"]["exact_balance"])
        report["ok"] = bool(ok)
        self.state["prepare_ok"] = bool(ok)
        self.state["spec_hash"] = report["spec_hash"]
        self.save_state()
        _write_json(os.path.join(self.out_dir, "prepare-report.json"), report)
        return report


# ---------------------------------------------------------------------------
# 各阶段
# ---------------------------------------------------------------------------

def cmd_compose(exp: Experiment) -> int:
    """按 spec 组装 12 份完整 prompt + 逐处差异 + 离线自检（不发任何请求）。

    同族新实验（schema 不是 round3）的组装规则不同：本工具**不假装**能组装，
    先做 schema 检查并明确报错，避免用 round3 的组装路径硬套出一个说不出依据的文件。
    """
    validation = validate_character_spec(exp.spec_dir)
    if validation.get("unsupported_schema"):
        _log(f"[unsupported] plan.schema_version={validation['unsupported_schema']}："
             "本工具只实现 round3 的组装与校验，拒绝按 round3 规则组装该计划。"
             "round4 的草案由离线准备任务的请求生成器产出，见 "
             "prompts/wardrobe/round4/requests/manifest.json")
        return 1
    manifest = exp.compose_pack()
    offline = exp.write_offline_suite()
    _log(json.dumps({"spec_hash": manifest["spec_hash"][:16],
                     "snapshots": manifest["snapshot_count"],
                     "slots": len(manifest.get("request_order") or []),
                     "pairing": manifest.get("pairing")}, ensure_ascii=False, indent=2))
    for row in manifest["snapshots"]:
        _log(f"  {row['case_id']} {row['variant']:12s} {row['file']:24s} "
             f"chars={row['characters']:5d} sha256={row['sha256'][:12]}")
    _log(f"候选 strict 通过: {offline['candidate_ok']} 失败项: {offline['candidate_failures']}")
    _log(f"pack: {exp.pack_dir}")
    return 0 if offline["candidate_ok"] and offline["validation"]["ok"] else 1


def cmd_prepare(exp: Experiment) -> int:
    validation = validate_character_spec(exp.spec_dir)
    if validation.get("unsupported_schema"):
        _log(f"[unsupported] plan.schema_version={validation['unsupported_schema']}："
             "prepare 需要 round3 的组装与校验规则，本工具尚未实现该 schema。"
             "**结论：未通过（不是通过）**；不要用本命令的退出码声称该计划已 prepare。")
        return 1
    report = exp.verify()
    _log(json.dumps({key: report[key] for key in ("ok", "spec_hash", "candidate_strict_pass",
                                                  "offline_suite")}, ensure_ascii=False, indent=2))
    _log(f"快照 {len(report['pack_snapshot_hash_check']['rows'])} 份，hash 全对: "
         f"{report['pack_snapshot_hash_check']['ok']}")
    _log(f"源码文件 hash 全对: {report['source_file_hash_check']['ok']}")
    for item in report["deterministic_checks"]:
        defects = [issue["check"] for issue in item["deterministic"]["issues"]]
        _log(f"  {item['file']:26s} {item['variant']:12s} severity={item['severity']:9s} "
             f"ok={item['deterministic']['ok']} issues={defects}")
    if report.get("candidate_gate"):
        _log(f"候选门禁: {json.dumps(report['candidate_gate'], ensure_ascii=False)}")
    _log(f"预算: {json.dumps(exp.ledger.summary(), ensure_ascii=False)}")
    _log(f"报告: {os.path.join(exp.out_dir, 'prepare-report.json')}")
    return 0 if report["ok"] else 1


def cmd_list_models(exp: Experiment) -> int:
    """模型清单核验：**只有一次** HTTP 机会。

    10:08 的第一次真实调用已经发出（预算已消耗），但脚本在本地解析返回值时崩了，
    清单内容没有落盘。按任务书的计数规则，这一次不能因为「本地报错」就重发，
    所以这里把该槽位如实标成 incomplete，并改用**纯本地**证据核对模型 ID
    （首轮成功生图的记录 + `conf/config.json` 的已缓存模型清单）。
    """
    from modules.others import api_backend
    label = "model-list"
    params = exp.spec_generation_params()
    wanted = [value["model"] for key, value in params.items()
              if key in ("gpt", "gemini") and isinstance(value, dict) and value.get("model")]
    local_evidence = exp._local_model_evidence(wanted)
    stage = exp.stage("model_list")
    if not stage:
        # 唯一一次尝试已经用过（账本里已有该槽位）：不再发第二次
        attempts = [item for item in exp.ledger.data["attempts"]
                    if item.get("kind") == "model_list" and item.get("slot") == label]
        if attempts:
            inflight = [item for item in attempts if item.get("status") == "inflight"]
            for item in inflight:
                exp.ledger.finish_attempt(item, status="incomplete_local_error",
                                          error="第一次清单调用已发出，本地解析阶段报错；按规则不重发",
                                          extra={"extra": {"spent_by_rule": True}})
        stage = {"count": None, "found": {name: local_evidence[name]["available"] for name in wanted},
                 "error": "list-models 调用已消耗但内容未落盘；改用本地证据核对",
                 "local_evidence": local_evidence}
        exp.set_stage("model_list", stage)
        _log(json.dumps(stage, ensure_ascii=False, indent=2))
        return 0 if all(stage["found"].values()) else 1
    _log(json.dumps(stage, ensure_ascii=False, indent=2))
    return 0 if stage.get("found") and all(stage["found"].values()) else 1


    def _local_model_evidence(self, wanted: list) -> dict:
        """纯本地核对模型 ID 是否曾被成功使用/在缓存清单里登记（不发任何请求）。

        证据目录来自 plan 的 `prior_runs`（缺省＝首轮 pilot 目录），不再写死某个轮次。
        """
        evidence = {name: {"available": False, "sources": []} for name in wanted}
        try:
            config = _read_json(os.path.join("conf", "config.json"))
            cached = set(config.get("cached_models") or [])
            for entry in (config.get("apis") or {}).values():
                if entry.get("model"):
                    cached.add(str(entry["model"]))
            for name in wanted:
                if name in cached:
                    evidence[name]["available"] = True
                    evidence[name]["sources"].append("conf/config.json cached_models/apis")
        except Exception:  # noqa: BLE001
            pass
        prior_dirs = list(self.plan.get("prior_runs") or [])
        if not prior_dirs:
            prior_dirs = ["data/test-result/20261005/wardrobe-pilot-v1"]
        for prior_dir in prior_dirs:
            if not os.path.isdir(prior_dir):
                continue
            for name in wanted:
                hits = 0
                for file_name in os.listdir(prior_dir):
                    if not file_name.endswith("_server_response_*.json") and not file_name.endswith(".json"):
                        continue
                    try:
                        text = _read_text(os.path.join(prior_dir, file_name))
                    except OSError:
                        continue
                    if f'"model": "{name}"' in text or f'"modelVersion": "{name}"' in text:
                        hits += 1
                if hits:
                    evidence[name]["available"] = True
                    evidence[name]["sources"].append(f"prior-run server responses ({hits} files)")
        return evidence


def cmd_resolver(exp: Experiment) -> int:
    cases = _read_json(os.path.join(exp.spec_dir, "resolver-cases.json"))["cases"]
    system_prompt = _read_text(os.path.join(exp.spec_dir, "resolve.md"))
    payload = resolver_request_payload(cases, system_prompt)
    cfg = text_api_config()
    from utils.analysis_gpt_prompt import normalize_chat_base
    url = f"{normalize_chat_base(cfg['base_url'])}/chat/completions"
    body = {"model": cfg["model"],
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": payload["user"]}],
            "max_completion_tokens": 4000}
    snapshot = scrub_request_snapshot({"url": url, "body": body}, secret_values=[cfg.get("api_key") or ""])
    _write_json(os.path.join(exp.out_dir, "resolver-request.json"),
                {"payload": payload, "snapshot": snapshot,
                 "body_sha256": sha256_text(json.dumps(body, ensure_ascii=False, sort_keys=True)),
                 "expected_stripped": "expected" not in payload["user"]})
    reserved = exp.reserve("text", "resolver-batch", "text_http_attempts", operation="resolver")
    started = datetime.now()
    error, raw = "", ""
    try:
        response, elapsed_ms = http_post_once(url, headers={"Authorization": f"Bearer {cfg['api_key']}",
                                                            "Content-Type": "application/json"},
                                              payload=body, timeout=300)
        if response.status_code != 200:
            error = f"HTTP {response.status_code}: {response.text[:500]}"
        else:
            data = response.json()
            choices = data.get("choices") or []
            raw = str((choices[0].get("message") or {}).get("content") or "") if choices else ""
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((datetime.now() - started).total_seconds() * 1000)
        error = f"{type(exc).__name__}: {exc}"
    expected_by_id = {case["id"]: case["expected"] for case in cases}
    validation = validate_resolver_output(raw, cases, expected_by_id=expected_by_id)
    exp.settle("text", reserved, status="success" if raw and not error else "error",
               error=error, http_status=200 if raw else None, model=cfg["model"],
               elapsed_ms=elapsed_ms if 'elapsed_ms' in locals() else 0)
    record = {"request": payload, "snapshot": snapshot, "error": error, "raw": raw,
              "validation": validation, "written_at": datetime.now().isoformat()}
    _write_json(os.path.join(exp.out_dir, "resolver-result.json"), record)
    exp.set_stage("resolver", {"ok": validation["valid"] and not error,
                               "matched": validation["matched"], "total": validation["total"],
                               "error": error, "issues": validation["issues"]})
    _log(f"[resolver] matched={validation['matched']}/{validation['total']} "
         f"valid={validation['valid']} error={error[:120]}")
    return 0 if (validation["valid"] and not error) else 1


def cmd_review(exp: Experiment) -> int:
    system_prompt = _read_text(os.path.join(exp.spec_dir, "review.md"))
    prepare = exp.verify() if not os.path.isfile(os.path.join(exp.out_dir, "prepare-report.json")) \
        else _read_json(os.path.join(exp.out_dir, "prepare-report.json"))
    evidence = {
        "plan": exp.plan,
        "gold_routes": exp.gold,
        "pack_manifest": exp.manifest,
        "prepare_report": {key: prepare.get(key) for key in
                           ("ok", "pack_snapshot_hash_check", "source_file_hash_check",
                            "prior_results_sha256", "deterministic_checks", "v2_channel_identity")},
        "expanded_prompts": {item["file"]: _read_text(os.path.join(exp.pack_dir, item["file"]))
                             for item in exp.manifest["snapshots"]},
        "note": exp.plan.get("review_context", "Review the supplied experimental snapshots."),
    }
    cfg = text_api_config()
    from utils.analysis_gpt_prompt import normalize_chat_base
    url = f"{normalize_chat_base(cfg['base_url'])}/chat/completions"
    body = {"model": cfg["model"],
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)}],
            "max_completion_tokens": 5000}
    snapshot = scrub_request_snapshot({"url": url, "body": body}, secret_values=[cfg.get("api_key") or ""])
    reserved = exp.reserve("text", "review", "text_http_attempts", operation="review")
    error, raw = "", ""
    elapsed_ms = 0
    try:
        response, elapsed_ms = http_post_once(url, headers={"Authorization": f"Bearer {cfg['api_key']}",
                                                            "Content-Type": "application/json"},
                                              payload=body, timeout=300)
        if response.status_code != 200:
            error = f"HTTP {response.status_code}: {response.text[:500]}"
        else:
            choices = response.json().get("choices") or []
            raw = str((choices[0].get("message") or {}).get("content") or "") if choices else ""
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    parsed = parse_json_object(raw)
    ready = (parsed or {}).get("ready") is True and not (parsed or {}).get("blocking")
    blocking = [item for item in ((parsed or {}).get("blocking") or []) if item]
    control_defects = (parsed or {}).get("control_defects_reported") or []
    exp.settle("text", reserved, status="success" if raw and not error else "error", error=error,
               model=cfg["model"], elapsed_ms=elapsed_ms)
    _write_json(os.path.join(exp.out_dir, "review-result.json"),
                {"request": snapshot, "error": error, "raw": raw, "parsed": parsed,
                 "ready": ready, "candidate_ready": bool(ready and not error),
                 "blocking": blocking, "control_defects_reported": control_defects,
                 "control_role": "frozen_control_known_defects_reported_only",
                 "candidate_role": "v4_candidate", "spec_hash": spec_hash(exp.spec_dir),
                 "written_at": datetime.now().isoformat()})
    exp.set_stage("review", {"ok": bool(ready and not error), "ready": ready,
                             "candidate_ready": bool(ready and not error),
                             "blocking": blocking, "control_defects_reported": control_defects,
                             "spec_hash": spec_hash(exp.spec_dir), "error": error})
    _log(f"[review] candidate_ready={bool(ready and not error)} blocking={len(blocking)} "
         f"control_defects={len(control_defects)} error={error[:120]}")
    return 0 if (ready and not error) else 1


def _data_url(path: str) -> str:
    with open(path, "rb") as handle:
        data = handle.read()
        mime = "image/jpeg" if data.startswith(bytes([255, 216, 255])) else "image/png"
        return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def _vision_request(exp: Experiment, *, system_prompt: str, user_text: str, images: list,
                    label_prefix: str, case_label: str) -> dict:
    """构造带图 messages：文字标签 + 图片按给定顺序交替；记录内容块顺序与原图 hash。"""
    content = [{"type": "text", "text": user_text}]
    image_meta = []
    for label, path in images:
        content.append({"type": "text", "text": f"IMAGE LABEL: [{label}] ({case_label})"})
        content.append({"type": "image_url", "image_url": {"url": _data_url(path)}})
        image_meta.append({"label": label, "path": os.path.abspath(path),
                           "sha256": sha256_file(path), "bytes": os.path.getsize(path),
                           "proxy_sha256": sha256_text(_data_url(path))})
    body = {"model": exp.state.get("vision_model") or "",
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": content}],
            "max_completion_tokens": 3000}
    return {"body": body, "content_order": content_block_order(body["messages"]),
            "images": image_meta, "label_prefix": label_prefix}


def _vision_model(exp: Experiment) -> str:
    model = str(exp.state.get("vision_model") or "").strip()
    if not model:
        raise RuntimeError("视觉模型未登记：先运行 --pick-vision-model（本轮只允许登记一次）")
    return model


def cmd_pick_vision_model(exp: Experiment) -> int:
    """登记本轮唯一的视觉评价模型（不发送任何请求）。

    只做“登记”而不是“尝试多个”：任务书要求冻结一个确认支持图片的评价模型，
    并在预检里说明来源。这里登记的就是首轮已确认能读图的文本视觉端点模型。
    """
    candidates = [item for item in str(exp.args.vision_model or "").split(",") if item.strip()]
    model = candidates[0] if candidates else str(text_api_config().get("model") or "")
    if not model:
        _log("[错误] 没有可登记的视觉模型")
        return 1
    exp.state["vision_model"] = model
    exp.state["vision_model_candidates"] = candidates or [model]
    exp.save_state()
    _log(f"[vision] 本轮登记视觉模型: {model}（来源: "
         f"{'--vision-model' if candidates else 'conf 文本模型'}）")
    return 0


def extract_observations(raw: str) -> dict:
    """**纯本地**取观察记录：按 label 建索引，不看规则、不发请求。

    为什么单独要有这个：`validate_calibration(rules=None)` 只做「响应是否可解析」
    的检查、不返回 observations，于是单图调用（一次一个 label）拿不到读数。
    这里只负责把 label 归一化成大写键，判据仍然全部由 `validate_calibration` 承担。
    """
    parsed = parse_json_object(raw)
    if not parsed:
        return {}
    rows = parsed.get("observations") or []
    if isinstance(rows, dict):
        rows = [rows]
    out = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip().upper()
        if label:
            out[label] = item
    return out


def calibrate_case_aggregate(case_id: str, observations: dict, error: str, rules: dict) -> dict:
    """把「每张图一次调用」的读数合成该用例的判定，复用同一套 `validate_calibration`。"""
    merged = {"observations": [observations[label] for label in sorted(observations)]}
    return validate_calibration(case_id, json.dumps(merged, ensure_ascii=False), error, rules=rules)


def calibrate_observations(exp: Experiment, case: dict, *, user_text: str, per_image: bool,
                           mode: str, system_prompt: str, model: str, url: str,
                           auth_header: dict, timeout: int = 300) -> dict:
    """跑一条校准用例（每张图一次调用或一次发两张），返回读数、逐调用记录与错误。

    预算在**每次真实发送前**分别占用一个槽位；失败同样占用。
    """
    observations, calls, error = {}, [], ""
    if per_image:
        groups = [((item["label"], item["image"]),) for item in case["images"]]
    else:
        groups = [tuple((item["label"], item["image"]) for item in case["images"])]
    for group in groups:
        labels = [item[0] for item in group]
        suffix = labels[0] if per_image else "merged"
        slot = f"calibration:{case['id']}:{suffix}"
        request = _vision_request(exp, system_prompt=system_prompt,
                                  user_text=user_text.format(label=labels[0], test_id=case["id"]),
                                  images=list(group), label_prefix=case["id"],
                                  case_label=case["id"])
        request["body"]["model"] = model
        reserved = exp.reserve("vision", slot, "vision_http_attempts", operation="calibration")
        call_error, raw, elapsed_ms = "", "", 0
        try:
            response, elapsed_ms = http_post_once(url, headers=auth_header,
                                                  payload=request["body"], timeout=timeout)
            if response.status_code != 200:
                call_error = f"HTTP {response.status_code}: {response.text[:400]}"
            else:
                choices = response.json().get("choices") or []
                raw = str((choices[0].get("message") or {}).get("content") or "") if choices else ""
        except Exception as exc:  # noqa: BLE001
            call_error = f"{type(exc).__name__}: {exc}"
        found = extract_observations(raw)
        observations.update(found)
        calls.append({"slot": slot, "labels": labels, "mode": mode, "error": call_error,
                      "elapsed_ms": elapsed_ms, "raw": raw, "observations": found,
                      "parsed": bool(parse_json_object(raw)), "user_text": user_text,
                      "images": request["images"], "content_order": request["content_order"],
                      "request_snapshot": scrub_request_snapshot(request, secret_values=[])})
        exp.settle("vision", reserved, status="success" if raw and not call_error else "error",
                   error=call_error, model=model, elapsed_ms=elapsed_ms)
        _log(f"[calibrate] {case['id']} [{labels[0]}] parsed={bool(parse_json_object(raw))} "
             f"error={call_error[:100]}")
        if call_error:
            error = "; ".join(filter(None, [error, call_error]))
    return {"observations": observations, "calls": calls, "error": error}


def cmd_calibrate_case(exp: Experiment) -> int:
    """只重跑**指定的一条**校准用例（任务书：不得把三项校准整体重跑当成一次）。"""
    case_id = str(getattr(exp.args, "calibrate_case", "") or "").strip()
    cases = _read_json(os.path.join(exp.spec_dir, "calibration-cases.json"))["tests"]
    case = next((item for item in cases if item["id"] == case_id), None)
    if not case:
        _log(f"[error] 校准用例里没有 {case_id!r}；可选 {[item['id'] for item in cases]}")
        return 1
    if exp.ledger.remaining("vision") < len(case["images"]):
        _log(f"[budget] 视觉额度不足以重跑 {case_id}（剩余 {exp.ledger.remaining('vision')}）")
        return 1
    system_prompt = _read_text(os.path.join(exp.spec_dir, "calibrate.md"))
    model = _vision_model(exp)
    cfg = text_api_config()
    from utils.analysis_gpt_prompt import normalize_chat_base
    url = f"{normalize_chat_base(cfg['base_url'])}/chat/completions"
    user_text = str(getattr(exp.args, "calibrate_user_text", "")
                    or "Read the attached image. This is image label [{label}] of test {test_id}.")
    result = calibrate_observations(
        exp, case, user_text=user_text, per_image=not exp.args.calibrate_merged,
        mode="per_image" if not exp.args.calibrate_merged else "merged", system_prompt=system_prompt,
        model=model, url=url, auth_header={"Authorization": f"Bearer {cfg['api_key']}",
                                           "Content-Type": "application/json"})
    path = os.path.join(exp.out_dir, f"calibration-{case_id}.json")
    entry = _read_json(path) if os.path.isfile(path) else {}
    observations = {}
    for call in entry.get("calls") or []:
        observations.update(call.get("observations") or extract_observations(call.get("raw") or ""))
    observations.update(result["observations"])          # 重跑到的 label 覆盖旧读数
    previous_error = entry.get("error") or ""
    validation = calibrate_case_aggregate(case_id, observations, result["error"],
                                          case.get("validation"))
    entry.update({"test_id": case_id, "mode": "per_image", "model": model, "url": url,
                  "expected_local_only": case["expected_observations_local_only"],
                  "images": case["images"], "calls": (entry.get("calls") or []) + result["calls"],
                  "call_count": len((entry.get("calls") or []) + result["calls"]),
                  "retry_of_single_case": True, "retry_user_text": user_text,
                  "previous_error": previous_error, "validation": validation,
                  "retried_at": datetime.now().isoformat(),
                  "file": f"calibration-{case_id}.json",
                  "observations_merged": {label: observations[label] for label in sorted(observations)}})
    _write_json(path, entry)
    exp.state["calibrate_mode"] = "per_image"
    stage = dict(exp.stage("calibrate") or {})
    cases_state = dict(stage.get("cases") or {})
    cases_state[case_id] = validation
    stage.update({"ok": all((cases_state.get(other) or {}).get("ok") is True
                            for other in {item["id"] for item in cases}),
                  "mode": "per_image", "slots_used": exp.ledger.spent("vision"),
                  "cases": cases_state,
                  "single_case_retries": list(stage.get("single_case_retries") or []) + [case_id]})
    exp.set_stage("calibrate", stage)
    _log(f"[calibrate] {case_id} 单项重跑 ok={validation['ok']} issues={validation['issues']}")
    return 0 if validation["ok"] else 1


def cmd_calibrate_recompute(exp: Experiment) -> int:
    """**纯本地**重算已有校准读数：只用已落盘的原始回复，绝不再发请求。

    用途：汇总逻辑修好后（例如单图调用的 observations 提取），不必重发 8 次调用，
    直接把缓存里的读数重新合成判定与阶段状态。没有原始回复的用例会被如实标成
    「无法重算」，不会伪造通过。
    """
    calib_path = os.path.join(exp.spec_dir, "calibration-cases.json")
    rules_by_id = {case["id"]: case.get("validation")
                   for case in (_read_json(calib_path).get("tests") or [])}
    results, slots = [], {}
    for case_id, rules in rules_by_id.items():
        path = os.path.join(exp.out_dir, f"calibration-{case_id}.json")
        if not os.path.isfile(path):
            _log(f"[recompute] {case_id} 没有已落盘的原始回复，跳过（不重发请求）")
            results.append({"test_id": case_id, "validation": {"ok": False,
                                                               "issues": ["没有可重算的原始回复"]}})
            continue
        entry = _read_json(path)
        observations, error = {}, ""
        for call in entry.get("calls") or []:
            observations.update(extract_observations(call.get("raw") or ""))
            if call.get("error"):
                error = "; ".join(filter(None, [error, str(call["error"])]))
        validation = calibrate_case_aggregate(case_id, observations, error, rules)
        entry.update({"mode": entry.get("mode", "per_image"), "validation": validation,
                      "recomputed_at": datetime.now().isoformat(),
                      "recompute_note": "汇总逻辑修正后用缓存原始回复重算；未发生新的 HTTP 调用"})
        _write_json(path, entry)
        results.append(entry)
        slots[case_id] = [f"{case_id}:{label}" for label in sorted(observations)]
        _log(f"[recompute] {case_id} ok={validation['ok']} labels={sorted(observations)} "
             f"issues={validation['issues']}")
    all_ok = all(item["validation"]["ok"] for item in results)
    exp.set_stage("calibrate", {"ok": all_ok, "mode": exp.state.get("calibrate_mode", "per_image"),
                                "slots_used": exp.ledger.spent("vision"),
                                "cases": {item["test_id"]: item["validation"] for item in results},
                                "call_slots": slots,
                                "recomputed": True})
    exp.save_state()
    _log(f"[recompute] 校准 {'通过' if all_ok else '未通过'}（视觉尝试 "
         f"{exp.ledger.spent('vision')}/{exp.ledger.data['budgets']['vision_http_attempts']}，本次未新增）")
    return 0 if all_ok else 1


def cmd_calibrate(exp: Experiment) -> int:
    if exp.state.get("prepare_ok") is not True:
        exp.ledger.mark_skipped("calibrate blocked: prepare not ok")
        _log("[gate] prepare 未通过，拒绝校准（校准失败/未做都不允许继续到生图）")
        return 1
    review = exp.stage("review")
    if review and review.get("candidate_ready") is False:
        exp.ledger.mark_skipped("calibrate blocked: candidate review not ready")
        _log("[gate] 候选审查 ready=false，拒绝校准与后续生图")
        return 1
    calib_path = os.path.join(exp.spec_dir, "calibration-cases.json")
    if not os.path.isfile(calib_path):
        exp.ledger.mark_skipped("calibrate blocked: calibration-cases.json missing")
        _log("[gate] 缺少校准用例文件，拒绝继续（也拒绝直接跳去生图）")
        return 1
    cases = _read_json(calib_path)["tests"]
    incomplete = [case.get("id") for case in cases
                  if not case.get("images") or not case.get("validation")]
    if incomplete:
        exp.ledger.mark_skipped(f"calibrate blocked: 用例缺规则/图片 {incomplete}")
        _log(f"[gate] 校准用例缺 validation 规则或图片，拒绝发请求：{incomplete}")
        return 1
    system_prompt = _read_text(os.path.join(exp.spec_dir, "calibrate.md"))
    model = _vision_model(exp)
    cfg = text_api_config()
    from utils.analysis_gpt_prompt import normalize_chat_base
    url = f"{normalize_chat_base(cfg['base_url'])}/chat/completions"
    per_image = not bool(getattr(exp.args, "calibrate_merged", False))
    mode = "per_image" if per_image else "merged"
    history_path = os.path.join(exp.out_dir, "calibration-history.json")
    history = _read_json(history_path) if os.path.isfile(history_path) else {"entries": []}
    history["entries"].append({
        "mode": exp.state.get("calibrate_mode", "merged"),
        "recorded_at": datetime.now().isoformat(),
        "note": "上一轮校准的逐例结论（原始读数保存在 calibration-<id>.json）",
        "cases": {case["id"]: exp.stage_case("calibrate", case["id"]) for case in cases}})
    results = []
    for case in cases:
        # 单图调用：每次只发一张图（A、B 各自一次）。两张图一起发时模型会把
        # 裁剪位置与配饰读窜（第一轮三例实测），拆开读是修数据质量、不是放松判据。
        observations, calls, failed = {}, [], False
        planned = [(item["label"], item["image"]) for item in case["images"]] if per_image \
            else [((case["images"][0]["label"], case["images"][0]["image"]),
                   (case["images"][1]["label"], case["images"][1]["image"]))]
        for group in planned:
            labels = [group[0]] if per_image else [item[0] for item in group]
            images = [(group[0], group[1])] if per_image else list(group)
            suffix = labels[0] if per_image else "merged"
            slot = f"calibration:{case['id']}:{suffix}"
            request = _vision_request(exp, system_prompt=system_prompt,
                                      user_text=f"Read the attached image. This is image label "
                                                f"[{labels[0]}] of test {case['id']}.",
                                      images=images, label_prefix=case["id"], case_label=case["id"])
            request["body"]["model"] = model
            reserved = exp.reserve("vision", slot, "vision_http_attempts", operation="calibration")
            error, raw, elapsed_ms = "", "", 0
            try:
                response, elapsed_ms = http_post_once(
                    url, headers={"Authorization": f"Bearer {cfg['api_key']}",
                                  "Content-Type": "application/json"},
                    payload=request["body"], timeout=300)
                if response.status_code != 200:
                    error = f"HTTP {response.status_code}: {response.text[:400]}"
                else:
                    choices = response.json().get("choices") or []
                    raw = str((choices[0].get("message") or {}).get("content") or "") if choices else ""
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
            per_call = {"parsed": bool(parse_json_object(raw)),
                        "observations": extract_observations(raw)}
            for label, item in per_call["observations"].items():
                observations[label] = item
            calls.append({"slot": slot, "labels": labels, "mode": mode, "error": error,
                          "elapsed_ms": elapsed_ms, "raw": raw,
                          "observations": per_call.get("observations") or {},
                          "parsed": bool(per_call.get("parsed")),
                          "images": request["images"], "content_order": request["content_order"],
                          "request_snapshot": scrub_request_snapshot(
                              request, secret_values=[cfg.get("api_key") or ""])})
            exp.settle("vision", reserved, status="success" if raw and not error else "error",
                       error=error, model=model, elapsed_ms=elapsed_ms)
            _log(f"[calibrate] {case['id']} [{labels[0]}] parsed={bool(per_call.get('parsed'))} "
                 f"error={error[:100]}")
            if not raw or error:
                failed = True
        merged_raw = "\n".join(f"# {call['slot']}\n{call['raw']}" for call in calls)
        merged_error = "; ".join(call["error"] for call in calls if call["error"])
        validation = calibrate_case_aggregate(case["id"], observations, merged_error,
                                              case.get("validation"))
        entry = {"test_id": case["id"], "mode": mode, "model": model, "url": url,
                 "expected_local_only": case["expected_observations_local_only"],
                 "images": case["images"], "calls": calls, "call_count": len(calls),
                 "raw": merged_raw, "error": merged_error, "failed_request": failed,
                 "validation": validation,
                 "file": f"calibration-{case['id']}.json"}
        _write_json(os.path.join(exp.out_dir, entry["file"]), entry)
        results.append(entry)
        _log(f"[calibrate] {case['id']} ok={validation['ok']} calls={len(calls)} "
             f"error={merged_error[:100]}")
    history["updated_at"] = datetime.now().isoformat()
    _write_json(history_path, history)
    all_ok = all(item["validation"]["ok"] for item in results)
    exp.state["calibrate_mode"] = mode
    exp.set_stage("calibrate", {"ok": all_ok, "mode": mode, "slots_used": exp.ledger.spent("vision"),
                                "cases": {item["test_id"]: item["validation"] for item in results},
                                "call_slots": {item["test_id"]: [c["slot"] for c in item["calls"]]
                                               for item in results}})
    return 0 if all_ok else 1


CALIBRATION_RULES = {
    "CAL-C03": {"A": {"must": ["uniform", "school"], "must_not": ["lolita", "lace dress"],
                      "expect": "uniform"},
                "B": {"must": ["dress"], "must_not": [], "expect": "decorative_dress"}},
    "CAL-C06": {"A": {"must": [], "expect": "tight_crop"},
                "B": {"must": [], "expect": "wider_crop"}},
}


CALIBRATION_RULES = {
    "CAL-C03": {"A": {"must": ["uniform", "school"], "must_not": ["lolita", "lace dress"],
                      "expect": "uniform"},
                "B": {"must": ["dress"], "must_not": [], "expect": "decorative_dress"}},
    "CAL-C06": {"A": {"must": [], "expect": "tight_crop"},
                "B": {"must": [], "expect": "wider_crop"}},
}


def _term_present(term: str, text: str) -> bool:
    """词项是否出现在文本里：**按词匹配**，不做裸子串。

    为什么必须这样：`red` 作为裸子串会命中 `colored` / `patterned` 这类词——
    2026-10-05 的 CAL-ACC 重跑就因此把两张纯黑图判成「读出了红色」。
    单字词走词边界匹配，多词短语仍按连续子串匹配。
    """
    needle = str(term or "").lower().strip()
    if not needle:
        return False
    if " " in needle:
        return needle in str(text or "").lower()
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])",
                     str(text or "").lower()) is not None


def validate_calibration(test_id: str, raw: str, error: str, rules: dict = None) -> dict:
    """按校准用例自带的规则核对可见描述，不做关键词放行的粗糙判断。

    判据来自 `calibration-cases.json` 每条用例的 `validation` 段：
    - `A_terms` / `B_terms`：必须出现在该标签描述里的词；
    - `A_forbidden` / `both_forbidden`：出现即失败（例如把黑裙读成粉色）;
    - `require_distinct_edges`：两个标签的画面下缘必须不同（裁剪宽度校准）；
    - `must_use_terms`：至少一个标签里要出现的位置描述词。
    **没有规则就判失败**——不允许默认通过。
    """
    parsed = parse_json_object(raw)
    issues = []
    if error:
        issues.append(f"请求失败: {error[:160]}")
    if not parsed:
        issues.append("响应不是可解析 JSON")
        return {"ok": False, "issues": issues, "parsed": None}
    observations = parsed.get("observations") or []
    labels = {str(item.get("label") or "").upper(): item for item in observations
              if isinstance(item, dict)}
    for label in ("A", "B"):
        if label not in labels:
            issues.append(f"缺少 {label} 的观察记录")
            continue
        if not str(labels[label].get("clothing") or "").strip():
            issues.append(f"{label} 没有给出可见服装描述")
        if not str(labels[label].get("lower_frame_edge") or "").strip():
            issues.append(f"{label} 没有给出画面下缘位置")
    if issues:
        return {"ok": False, "issues": issues, "parsed": parsed}
    rules = rules or {}
    if not rules:
        issues.append("校准用例没有提供 validation 规则，禁止默认通过")
        return {"ok": False, "issues": issues, "parsed": parsed, "observations": labels}
    text_a = json.dumps(labels.get("A"), ensure_ascii=False).lower()
    text_b = json.dumps(labels.get("B"), ensure_ascii=False).lower()
    for term in rules.get("A_terms") or []:
        if not _term_present(term, text_a):
            issues.append(f"A 的描述里没有读到「{term}」")
    for term in rules.get("B_terms") or []:
        if not _term_present(term, text_b):
            issues.append(f"B 的描述里没有读到「{term}」")
    for term in rules.get("A_forbidden") or []:
        if _term_present(term, text_a):
            issues.append(f"A 的描述里出现了不该有的「{term}」")
    for term in rules.get("both_forbidden") or []:
        if _term_present(term, text_a) or _term_present(term, text_b):
            issues.append(f"出现了不该有的「{term}」（颜色/类别读错）")
    if rules.get("require_distinct_edges"):
        edge_a = str(labels["A"].get("lower_frame_edge") or "").strip().lower()
        edge_b = str(labels["B"].get("lower_frame_edge") or "").strip().lower()
        if edge_a == edge_b:
            issues.append("两张图的画面下缘被读成同一位置，裁剪校准不成立")
    if rules.get("must_use_terms"):
        combined = text_a + text_b
        if not any(str(term).lower() in combined for term in rules["must_use_terms"]):
            issues.append("没有可用的裁剪位置描述")
    return {"ok": not issues, "issues": issues, "parsed": parsed,
            "observations": labels, "rules": rules}


def _generation_params(exp: Experiment) -> dict:
    return exp.spec_generation_params()


def _find_prior_sample(case_id: str, channel: str) -> dict:
    path = "data/test-result/20261005/wardrobe-pilot-v1/results.json"
    if not os.path.isfile(path):
        return {}
    prior = _read_json(path)
    for sample in prior.get("samples") or []:
        if sample.get("sample_id") == f"{case_id}-lolita-{channel}":
            return sample
    return {}


def _call_generation(exp: Experiment, sample: dict, params: dict) -> list:
    from modules.others import api_backend
    if sample["channel"] == "gpt":
        return api_backend.generate_image_aigc2d_gpt(
            prompt=sample["prompt"], image_paths=[], model=params["gpt"]["model"],
            size=params["gpt"]["size"], quality=params["gpt"]["quality"],
            output_format=params["gpt"]["output_format"], n=params["gpt"]["n"],
            mode=params["gpt"]["mode"], api_type=params["gpt"]["api_type"],
            save_sub_dir=exp.backend_sub_dir, file_prefix=f"{sample['sample_id']}-",
            return_metadata=True, log_callback=lambda message: _log(f"    [gpt] {message}")) or {}
    return api_backend.generate_image_aigc2d(
        prompt=sample["prompt"], image_paths=[], model=params["gemini"]["model"],
        aspect_ratio=params["gemini"]["aspect_ratio"], resolution=params["gemini"]["resolution"],
        api_type=params["gemini"]["api_type"], save_sub_dir=exp.backend_sub_dir,
        file_prefix=f"{sample['sample_id']}-",
        face_quality_boost=params["gemini"]["face_quality_boost"],
        return_metadata=True, log_callback=lambda message: _log(f"    [gemini] {message}")) or {}


def cmd_generate(exp: Experiment) -> int:
    gate = exp.candidate_gate()
    if not gate["ready"]:
        exp.ledger.mark_skipped("candidate gate blocked generate: " + "; ".join(gate["reasons"]))
        _log("[gate] 候选未就绪，拒绝所有生图入口：")
        for reason in gate["reasons"]:
            _log(f"  - {reason}")
        return 1
    if exp.state.get("prepare_ok") is not True:
        exp.ledger.mark_skipped("prepare not ok")
        _log("[gate] prepare 未通过，拒绝生图")
        return 1
    params = _generation_params(exp)
    # 计划槽位必须来自 spec（case 数量/版本/repeat 由 spec 决定，不硬编码）
    if exp.is_composed_spec():
        expected_slots = sum(len(case["variants"]) * len(exp.plan.get("channels") or [])
                             * int(case["repeats"]) for case in exp.plan["cases"])
        if len(exp.manifest.get("request_order") or []) != expected_slots:
            _log(f"[gate] 计划槽位 {len(exp.manifest.get('request_order') or [])} 与 spec 计算的 "
                 f"{expected_slots} 不一致，停止")
            return 1
    samples_path = os.path.join(exp.out_dir, "samples.json")
    samples = _read_json(samples_path) if os.path.isfile(samples_path) else {
        "run_id": exp.run_id, "generated_at": datetime.now().isoformat(), "samples": []}
    by_id = {item["sample_id"]: item for item in samples.get("samples") or []}
    # 清理没有产物文件的伪成功（保留失败/未知记录）
    for item in list(by_id.values()):
        if item.get("status") == "success" and not all(os.path.isfile(path)
                                                      for path in item.get("saved_files") or []):
            item["status"] = "reused_missing_file"
            item.pop("saved_files", None)
    budget_exhausted = False
    for row in exp.manifest.get("request_order") or []:
        sample_id = row["sample_id"]
        existing = by_id.get(sample_id) or {}
        if existing.get("status") == "success":
            _log(f"[reuse] {sample_id} 已有成功产物，跳过")
            continue
        prompt = _read_text(os.path.join(exp.pack_dir, row["prompt_file"]))
        spec = {"sample_id": sample_id, "pair_id": row["pair_id"], "case_id": row["case_id"],
                "channel": row["channel"], "repeat": row["repeat"], "variant": row["variant"],
                "gold_action": exp.gold.get(row["case_id"], {}).get("gold_action"),
                "framing": exp.gold.get(row["case_id"], {}).get("framing"),
                "prompt_file": row["prompt_file"], "prompt": prompt,
                "prompt_sha256": sha256_text(prompt), "prompt_chars": len(prompt),
                "attempted_at": datetime.now().isoformat()}
        try:
            reserved = exp.reserve("generation", sample_id, "generation_http_attempts",
                                   operation=f"generate:{row['channel']}",
                                   model=params[row["channel"]]["model"],
                                   prompt_sha256=spec["prompt_sha256"], prompt_chars=len(prompt))
        except BudgetExceeded as exc:
            budget_exhausted = True
            spec.update({"status": "not_sent_budget_exhausted", "error": str(exc)})
            by_id[sample_id] = spec
            samples["samples"] = list(by_id.values())
            _write_json(samples_path, samples)
            _log(f"[budget] {sample_id} 未发送：{exc}")
            break
        try:
            started = datetime.now()
            import time as _time
            t0 = _time.perf_counter()
            payload = _call_generation(exp, spec, params)
            elapsed_ms = int((_time.perf_counter() - t0) * 1000)
            saved = list((payload or {}).get("saved_files") or []) if isinstance(payload, dict) else list(payload or [])
            raw_text = str((payload or {}).get("raw_text") or "") if isinstance(payload, dict) else ""
            spec["saved_files"] = [os.path.abspath(path) for path in saved]
            spec["status"] = "success" if saved else "failed_no_image"
            spec["elapsed_ms"] = elapsed_ms
            spec["raw_text"] = raw_text[:2000]
            spec["started_at"] = started.isoformat()
            spec["finished_at"] = datetime.now().isoformat()
        except Exception as exc:  # noqa: BLE001
            spec.update({"saved_files": [], "status": "error",
                         "error": f"{type(exc).__name__}: {exc}",
                         "finished_at": datetime.now().isoformat()})
        exp.settle("generation", reserved, status=spec["status"],
                   output_path=(spec.get("saved_files") or [""])[0],
                   error=spec.get("error", ""), model=params[spec["channel"]]["model"],
                   elapsed_ms=spec.get("elapsed_ms", 0),
                   extra={"prompt_sha256": spec["prompt_sha256"]})
        by_id[sample_id] = spec
        samples["samples"] = list(by_id.values())
        samples["updated_at"] = datetime.now().isoformat()
        _write_json(samples_path, samples)
        _log(f"[{spec['status']}] {sample_id} files={len(spec.get('saved_files') or [])} "
             f"elapsed={spec.get('elapsed_ms')}ms err={spec.get('error', '')[:100]}")
    success = sum(1 for item in by_id.values() if item.get("status") == "success")
    hygiene = scrub_credential_files(exp.out_dir)
    exp.state["credential_hygiene"] = hygiene
    exp.save_state()
    _write_json(os.path.join(exp.out_dir, "credential-hygiene.json"),
                {**hygiene, "reason": ("后端的生成回放文件（*_replay_*.json/_*.py）会在 Authorization 头里"
                                       "写入未掩码的 API key；每次生成后删除，避免凭据落盘。")})
    exp.set_stage("generate", {"attempted": exp.ledger.spent("generation"),
                               "success": success, "budget_exhausted": budget_exhausted,
                               "credential_hygiene": hygiene})
    _log(f"[generate] 成功 {success} / 槽位 {len(exp.manifest.get('request_order') or [])}，"
         f"生图尝试 {exp.ledger.spent('generation')}/{exp.ledger.data['budgets']['generation_http_attempts']}")
    return 0 if not budget_exhausted else 1


def build_pairs(exp: Experiment) -> dict:
    """16 组匿名配对：A/B 精确平衡（control/candidate 各 8），不透露版本身份。"""
    seed = int(exp.plan.get("pairing", {}).get("seed")
               or exp.manifest.get("request_order_seed") or 20261005)
    samples = (_read_json(os.path.join(exp.out_dir, "samples.json")) or {}).get("samples") or []
    by_id = {item["sample_id"]: item for item in samples}
    spec_pairs = build_pairs_exact_balance(exp.manifest.get("request_order") or [], seed=seed)
    pairs = []
    for spec_pair in spec_pairs["pairs"]:
        images, entries = {}, []
        for label in ("A", "B"):
            sample_id = spec_pair["sample_id"][label]
            sample = by_id.get(sample_id) or {}
            image = (sample.get("saved_files") or [None])[0]
            images[label] = image
            entries.append({"label": label, "sample_id": sample_id,
                            "variant": spec_pair["assignment"][label],
                            "image": image, "status": sample.get("status")})
        usable = all(item["image"] and os.path.isfile(item["image"]) for item in entries)
        gold = exp.gold.get(spec_pair["case_id"], {})
        pairs.append({"pair_id": spec_pair["pair_id"], "case_id": spec_pair["case_id"],
                      "channel": spec_pair["channel"], "repeat": spec_pair["repeat"],
                      "gold_action": gold.get("gold_action"), "framing": gold.get("framing"),
                      "role": gold.get("role"),
                      "status": "ready" if usable else "skipped_missing_image",
                      "assignment": dict(spec_pair["assignment"]),
                      "images": images, "members": entries})
    result = {"run_id": exp.run_id, "seed": seed, "generated_at": datetime.now().isoformat(),
              "pairs": pairs, "balance": spec_pairs["balance"],
              "ready": sum(1 for item in pairs if item["status"] == "ready")}
    _write_json(os.path.join(exp.out_dir, "pairs-blind.json"), result)
    return result


def cmd_pairs(exp: Experiment) -> int:
    result = build_pairs(exp)
    balance = result.get("balance") or {}
    _log(json.dumps({"pairs": len(result["pairs"]), "ready": result["ready"],
                     "balance": balance}, ensure_ascii=False))
    _log(f"配对: {os.path.join(exp.out_dir, 'pairs-blind.json')}")
    if not result["ready"]:
        return 1
    expected = int(exp.plan.get("pairing", {}).get("pair_slots") or len(result["pairs"]))
    if len(result["pairs"]) != expected:
        _log(f"[gate] 配对数 {len(result['pairs'])} 与 spec 声明的 {expected} 不一致")
        return 1
    if not balance.get("exact_balance"):
        _log("[gate] A/B 没有精确平衡（spec 要求 control/candidate 各占一半）")
        return 1
    return 0


def cmd_score(exp: Experiment) -> int:
    gate = exp.candidate_gate()
    if not gate["ready"]:
        _log("[gate] 候选未就绪，拒绝评分入口：" + "；".join(gate["reasons"]))
        return 1
    system_prompt = _read_text(os.path.join(exp.spec_dir, "score.md"))
    pairs = (_read_json(os.path.join(exp.out_dir, "pairs-blind.json")) if
             os.path.isfile(os.path.join(exp.out_dir, "pairs-blind.json")) else build_pairs(exp))
    if not pairs.get("balance", {}).get("exact_balance"):
        _log("[gate] 配对未达到 spec 要求的 A/B 精确平衡，拒绝评分")
        return 1
    cfg = text_api_config()
    model = _vision_model(exp)
    doc = load_character_test(exp.spec_dir) if exp.is_composed_spec() else {"by_case": {}}
    meta_path = os.path.join(exp.out_dir, "sample-meta.json")
    meta = _read_json(meta_path) if os.path.isfile(meta_path) else {}
    from utils.analysis_gpt_prompt import normalize_chat_base
    url = f"{normalize_chat_base(cfg['base_url'])}/chat/completions"
    scores_path = os.path.join(exp.out_dir, "scores.json")
    scores = _read_json(scores_path) if os.path.isfile(scores_path) else {
        "run_id": exp.run_id, "model": model, "records": []}
    done = {item["pair_id"]: item for item in scores.get("records") or []}
    stop = False
    for pair in pairs["pairs"]:
        if pair["pair_id"] in done:
            _log(f"[reuse] {pair['pair_id']} 已有评分记录")
            continue
        if pair["status"] != "ready":
            record = {"pair_id": pair["pair_id"], "case_id": pair["case_id"], "channel": pair["channel"],
                      "status": "not_scored_missing_image", "assignment": pair["assignment"],
                      "images": pair["images"], "recorded_at": datetime.now().isoformat()}
            done[pair["pair_id"]] = record
            scores["records"] = list(done.values())
            _write_json(scores_path, scores)
            continue
        # 评分必须拿到真实人物/衣物/要求/protected：缺任何一项都不发请求
        character = (doc.get("by_case") or {}).get(pair["case_id"]) or meta.get(pair["case_id"]) or {}
        case_for_scoring = {"case_id": pair["case_id"], "role": pair.get("role"),
                            "gold_action": pair.get("gold_action"), "framing": pair.get("framing")}
        missing_conditions = [field for field in ("identity", "source_clothing", "request", "protected")
                              if not character.get(field)]
        if missing_conditions:
            record = {"pair_id": pair["pair_id"], "case_id": pair["case_id"],
                      "channel": pair["channel"], "assignment": pair["assignment"],
                      "images": pair["images"], "status": "not_scored_missing_conditions",
                      "missing_conditions": missing_conditions,
                      "recorded_at": datetime.now().isoformat()}
            done[pair["pair_id"]] = record
            scores["records"] = list(done.values())
            _write_json(scores_path, scores)
            _log(f"[score] {pair['pair_id']} 缺少评分条件 {missing_conditions}，拒绝借对照图发明锚点")
            continue
        conditions = scoring_conditions(case_for_scoring, character)
        user_text = (
            f"CASE {conditions['case_id']} ({conditions['role']}).\n"
            f"Authorized clothing action: {conditions['authorized_clothing_action'].upper()}.\n"
            f"Framing requirement: {conditions['framing_requirement']}.\n"
            f"Identity facts that must not change: {conditions['identity']}\n"
            f"Described source clothing: {conditions['source_clothing']}\n"
            f"User request for this case: {conditions['user_request']}\n"
            "Protected anchors (check each independently; do not add anchors that are not listed):\n"
            + "\n".join(f"- {item}" for item in conditions["protected"]) + "\n"
            f"{conditions['note']}\n"
            "Two anonymized images follow, labelled in the text block immediately before each image. "
            "Judge every dimension separately and report not_visible where the crop hides it.")
        request = _vision_request(exp, system_prompt=system_prompt, user_text=user_text,
                                  images=[("A", pair["images"]["A"]), ("B", pair["images"]["B"])],
                                  label_prefix=pair["pair_id"], case_label=pair["pair_id"])
        request["body"]["model"] = model
        entry = {"pair_id": pair["pair_id"], "case_id": pair["case_id"], "channel": pair["channel"],
                 "gold_action": pair["gold_action"], "framing": pair["framing"],
                 "assignment": pair["assignment"], "images": pair["images"],
                 "scoring_conditions": conditions,
                 "content_order": request["content_order"], "image_meta": request["images"],
                 "request_snapshot": scrub_request_snapshot(request, secret_values=[cfg.get("api_key") or ""]),
                 "model": model, "called_at": datetime.now().isoformat()}
        try:
            reserved = exp.reserve("vision", f"score:{pair['pair_id']}", "vision_http_attempts",
                                   operation="score")
        except BudgetExceeded as exc:
            _log(f"[budget] 评分停止：{exc}")
            stop = True
            break
        error, raw, elapsed_ms = "", "", 0
        try:
            response, elapsed_ms = http_post_once(url, headers={"Authorization": f"Bearer {cfg['api_key']}",
                                                                "Content-Type": "application/json"},
                                                  payload=request["body"], timeout=300)
            if response.status_code != 200:
                error = f"HTTP {response.status_code}: {response.text[:400]}"
            else:
                choices = response.json().get("choices") or []
                raw = str((choices[0].get("message") or {}).get("content") or "") if choices else ""
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        parsed = parse_json_object(raw)
        entry.update({"raw": raw, "error": error, "elapsed_ms": elapsed_ms,
                      "status": "scored" if parsed else "unscored",
                      "parsed": parsed})
        exp.settle("vision", reserved, status="success" if parsed else "error", error=error,
                   model=model, elapsed_ms=elapsed_ms)
        done[pair["pair_id"]] = entry
        scores["records"] = list(done.values())
        scores["updated_at"] = datetime.now().isoformat()
        _write_json(scores_path, scores)
        _log(f"[score] {pair['pair_id']} {'ok' if parsed else 'FAILED'} {error[:100]}")
    scored = sum(1 for item in done.values() if item.get("status") == "scored")
    exp.set_stage("score", {"scored": scored, "records": len(done), "stopped_on_budget": stop})
    _log(f"[score] 有效评分 {scored} / 记录 {len(done)}；视觉尝试 "
         f"{exp.ledger.spent('vision')}/{exp.ledger.data['budgets']['vision_http_attempts']}")
    return 0


def cmd_mock(exp: Experiment) -> int:
    """离线自检：预算、扣减时机、失败保留、崩溃恢复、无隐藏重试路径。"""
    _log("[mock] 开始离线自检（不发任何 HTTP）")
    mock_dir = os.path.join(exp.out_dir, "mock")
    os.makedirs(mock_dir, exist_ok=True)
    checks = []

    def check(name: str, ok: bool, detail: str = ""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        _log(f"  {'OK ' if ok else 'FAIL'} {name} {detail}")

    ledger_path = os.path.join(mock_dir, "ledger.json")
    if os.path.isfile(ledger_path):
        os.remove(ledger_path)
    ledger = ExperimentLedger.load(ledger_path, {"generation_http_attempts": 2,
                                                 "vision_http_attempts": 2,
                                                 "text_http_attempts": 1,
                                                 "model_list_http_attempts": 1})
    first = ledger.plan_attempt("generation", slot="mock-1")
    check("发送前扣减", ledger.remaining("generation") == 1, f"remaining={ledger.remaining('generation')}")
    ledger.finish_attempt(first, status="error", error="mock-failure")
    second = ledger.plan_attempt("generation", slot="mock-2")
    check("失败仍消耗预算", ledger.spent("generation") == 2)
    try:
        ledger.plan_attempt("generation", slot="mock-3")
        check("超预算前停止", False, "第 3 次竟然被放行")
    except BudgetExceeded:
        check("超预算前停止", True)
    reloaded = ExperimentLedger.load(ledger_path, {"generation_http_attempts": 2})
    recovered = reloaded.recover_inflight()
    check("崩溃恢复不重置预算", reloaded.spent("generation") == 2 and
          [item["slot"] for item in recovered] == ["mock-2"],
          f"recovered={[item['slot'] for item in recovered]}")
    # 隐藏重试：生图通道已强制一次发送
    check("关闭隐藏重试的环境变量", os.environ.get("IMAGE_MAKER_IMAGE_MAX_RETRIES") == "0")
    # 主账本不受 mock 影响
    check("mock 不污染主账本", exp.ledger.attempt_count() == 0,
          f"主账本尝试={exp.ledger.attempt_count()}")
    ok = all(item["ok"] for item in checks)
    _write_json(os.path.join(exp.out_dir, "mock-report.json"), {"ok": ok, "checks": checks})
    _log(f"[mock] {'通过' if ok else '未通过'}；报告 {os.path.join(exp.out_dir, 'mock-report.json')}")
    return 0 if ok else 1


def command_grant_budget(args, experiment: Experiment) -> int:
    """把**用户显式授权**的追加额度写进原账本（只抬上限，不重置已消耗、不换 run-id）。"""
    kind = str(args.grant_budget or "").strip()
    if kind not in BUDGET_KINDS:
        _log(f"[error] --grant-budget 只认 {BUDGET_KINDS}，收到 {kind!r}")
        return 1
    try:
        result = experiment.ledger.grant_budget(
            kind, int(args.grant_limit), authorized_by=args.authorized_by,
            reason=args.grant_reason, source=args.grant_source)
    except (ValueError, BudgetExceeded) as exc:
        _log(f"[error] {type(exc).__name__}: {exc}")
        return 1
    _log("[grant-budget] " + json.dumps(result, ensure_ascii=False))
    _log("[grant-budget] 账本: " + os.path.join(experiment.out_dir, "ledger.json"))
    return 0


def command_grant_override(args, experiment: Experiment) -> int:
    """把用户显式放行写进账本，并挂到被放行阶段的判定里（放行 ≠ 通过）。"""
    kind = str(args.grant_override or "").strip()
    if kind not in OVERRIDE_KINDS:
        _log(f"[error] --grant-override 只认 {OVERRIDE_KINDS}，收到 {kind!r}")
        return 1
    try:
        record = experiment.ledger.record_override(
            kind, authorized_by=args.override_by, reason=args.override_reason,
            assumed_hypothesis=args.override_assumption, known_risk=args.override_risk,
            accepted_at=args.override_accepted_at, source=args.override_source)
    except ValueError as exc:
        _log(f"[error] {exc}")
        return 1
    stage = dict(experiment.stage(kind) or {})
    stage_key = "calibration_overrides" if kind == "calibrate" else "overrides"
    stage[stage_key] = list(stage.get(stage_key) or []) + [record]
    experiment.set_stage(kind, stage)
    experiment.state["overrides"] = list(experiment.state.get("overrides") or []) + [record]
    experiment.save_state()
    _log("[grant-override] " + json.dumps(record, ensure_ascii=False))
    _log(f"[grant-override] 已挂到阶段 {kind}；门禁会把它作为放行（不是通过）报告")
    return 0


def cmd_ledger(exp: Experiment) -> int:
    _log(json.dumps({"run_id": exp.run_id, "out_dir": exp.out_dir,
                     "run_dir_source": getattr(exp, "run_dir_source", ""),
                     "budgets": exp.ledger.summary(),
                     "attempts": exp.ledger.data["attempts"],
                     "recovered": exp.ledger.data.get("recovered", [])},
                    ensure_ascii=False, indent=2)[:20000])
    return 0


def command_frozen_pack(args) -> int:
    """Generation only: immutable requests, persistent attempts, no text/vision calls."""
    from pathlib import Path
    from utils.wardrobe_experiment import load_frozen_generation_pack
    try:
        pack = load_frozen_generation_pack(args.frozen_pack)
        if not (args.prepare or args.generate or args.status):
            raise ValueError("Frozen pack requires --prepare, --generate or --status")
        if any(getattr(args, flag, False) for flag in
               ("run", "resolver", "review", "calibrate", "score", "list_models", "mock")):
            raise ValueError("Frozen pack supports generation only")
        if not args.run_dir:
            raise ValueError("Frozen pack requires a fixed --run-dir")
        out = Path(args.run_dir).resolve()
        out.relative_to(Path(BASE_DIR, "data", "test-result").resolve())
    except (ValueError, KeyError, OSError) as exc:
        _log(f"[preflight] {exc}")
        return 1
    rows = [row for row in pack["requests"] if not args.channel or row["channel"] == args.channel]
    if args.prepare:
        _log(f"[ready] {len(rows)} frozen requests; 0 HTTP; pack {pack['sha256']}; output {out}")
        return 0
    if args.status:
        state_path = out / "samples.json"
        _log(json.dumps(_read_json(str(state_path)) if state_path.exists() else
                        {"status": "not_started", "slots": len(rows)}, ensure_ascii=False, indent=2))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "frozen-pack.json"
    if manifest_path.exists() and _read_json(str(manifest_path)).get("sha256") != pack["sha256"]:
        _log("[error] Output directory belongs to another frozen pack")
        return 1
    _write_json(str(manifest_path), pack)
    state_path = out / "samples.json"
    state = _read_json(str(state_path)) if state_path.exists() else {"pack_sha256": pack["sha256"], "samples": []}
    if state.get("pack_sha256") != pack["sha256"]:
        _log("[error] Sample state belongs to another frozen pack")
        return 1
    by_id = {row["sample_id"]: row for row in state["samples"]}
    # ---- 用户显式授权的追加额度 -------------------------------------------
    # 冻结包的默认上限就是包里的槽位数，跑满之后**没有任何补跑路径**：第 N+1 次发送
    # 会被 send_budget 拦下。任务书允许「用户主动追加授权」这种有效授权，所以这里只
    # 认一个显式的 `--grant-frozen-budget generation`，且必须带授权人、理由与来源；
    # 缺任何一项就拒绝，不把「无授权的重发」混进来。上限只抬不降、spent 不动、
    # run-dir 不换，授权记录落进 ledger.json 供复核。
    grant = None
    if str(getattr(args, "grant_frozen_budget", "") or "").strip():
        from utils.wardrobe_experiment import ExperimentLedger, budget_key_for, now_iso
        kind = str(args.grant_frozen_budget).strip()
        if kind != "generation":
            _log(f"[error] --grant-frozen-budget 只认 generation，收到 {kind!r}")
            return 1
        if not (str(args.authorized_by or "").strip() and str(args.grant_reason or "").strip()
                and str(args.grant_source or "").strip()):
            _log("[error] 追加额度必须同时给出 --authorized-by / --grant-reason / --grant-source")
            return 1
        ledger_path = out / "ledger.json"
        ledger = ExperimentLedger.load(str(ledger_path),
                                       {"generation_http_attempts": int(pack["generation_http_attempts"])},
                                       trust_persisted_budgets=bool(ledger_path.exists()))
        bkey = budget_key_for(kind)
        # 上限的基线是**冻结包里的槽位数**：新建 ledger 时若落到任务书默认值（32），
        # 追加就变成「从 32 抬到 42」，多出的 8 个额度来路不明。已有账本不回写
        # （授权记录是审计材料，不改写历史）。
        if not ledger_path.exists():
            ledger.data["budgets"][bkey] = max(int(ledger.data["budgets"].get(bkey, 0)),
                                                int(pack["generation_http_attempts"]))
            ledger.save()
        # 真实派发次数以本次实验**自己的发送账本**为准：`send-budget/slots/` 里每个
        # 槽位文件就是一次已经发出的请求。新建的 ledger 若记着 0 次消耗，上限就只是
        # 一个跟实际发送无关的数字，所以先按槽位文件对账（只对齐消耗，不重置、不删）。
        slots_dir = out / "send-budget" / "slots"
        used_slots = len([name for name in os.listdir(slots_dir)
                          if name.endswith(".json")]) if slots_dir.is_dir() else 0
        if used_slots and ledger.spent(kind) != used_slots:
            _log(f"[grant] 对账：本次实验已发出 {used_slots} 次生图，ledger 记录 {ledger.spent(kind)} 次，"
                 "按实际发送次数对齐消耗")
            ledger.data["budgets"][bkey] = max(int(ledger.data["budgets"].get(bkey, 0)), used_slots)
            ledger.save()
        if int(ledger.data["budgets"].get(bkey, 0)) >= int(args.grant_limit):
            # 重复执行同一条授权命令必须幂等：上限已经到位就不再写第二条授权记录
            _log(f"[grant] 上限已是 {int(ledger.data['budgets'][bkey])}（≥ 要求的 {int(args.grant_limit)}），"
                 "沿用既有授权，不重复登记")
        else:
            ledger.grant_budget(kind, int(args.grant_limit), authorized_by=args.authorized_by,
                                reason=args.grant_reason, source=args.grant_source)
            ledger.save()
        limit = int(ledger.data["budgets"][bkey])
        grant = {"limit": limit, "index": len(ledger.data.get("budget_grants") or []),
                 "granted_at": now_iso()}
        _log(f"[grant] generation 上限 {grant['limit']}（{bkey}）；"
             f"已消耗 {ledger.spent('generation')}，剩余 {ledger.remaining('generation')}；"
             f"账本 {ledger_path}")
        failed_slots = sorted(slot for slot, item in by_id.items()
                              if item.get("status") not in (None, "success"))
        picked = [row for row in by_id.items() if row[1].get("channel") == (args.channel or "gemini")]
        if args.channel is None:
            picked = list(by_id.items())
        retry_ids = {slot for slot, item in picked if item.get("status") not in (None, "success")}
        if str(args.retry_slots or "").strip():
            wanted = {name.strip() for name in str(args.retry_slots).split(",") if name.strip()}
            unknown = sorted(wanted - set(by_id))
            if unknown:
                _log(f"[error] --retry-slots 里有本次实验不认识的槽位: {unknown}")
                return 1
            retry_ids &= wanted
        if not retry_ids:
            _log(f"[grant] 没有可补跑的失败槽位（本通道已尝试的失败项：{failed_slots or '无'}）")
            return 0
        rows = [row for row in rows if row["sample_id"] in retry_ids]
        _log(f"[grant] 本次只补跑 {len(rows)} 个已失败的槽位：{sorted(retry_ids)}")
    env_keys = [send_budget.ENV_DIR, send_budget.ENV_LIMIT, send_budget.ENV_RUN,
                send_budget.ENV_TEXT_LIMIT, "IMAGE_MAKER_IMAGE_MAX_RETRIES"]
    old_env = {key: os.environ.get(key) for key in env_keys}
    effective_limit = grant["limit"] if grant else int(pack["generation_http_attempts"])
    os.environ.update({send_budget.ENV_DIR: str(out / "send-budget"),
                       send_budget.ENV_LIMIT: str(effective_limit),
                       send_budget.ENV_TEXT_LIMIT: "0", "IMAGE_MAKER_IMAGE_MAX_RETRIES": "0"})
    try:
        from modules.others import api_backend
        for row in rows:
            slot = row["sample_id"]
            prior = by_id.get(slot)
            if prior and prior.get("status") == "success":
                # 成功产物按文件 hash 复用；文件被改动过就不算成功，需重新生成
                files = prior.get("saved_files") or []
                hashes = prior.get("output_sha256") or []
                if not files or len(files) != len(hashes) or any(not os.path.isfile(p) or sha256_file(p) != h
                                    for p, h in zip(files, hashes)):
                    prior["status"] = "missing_or_changed_output"
            prior_attempts = []
            if prior:
                reuse = prior.get("status") in ("success", "missing_or_changed_output")
                if reuse or not grant:
                    # 默认口径：已尝试过的槽位（含失败）不重发；补跑时成功的照样复用
                    _log(f"[reuse/skip] {slot}: {prior.get('status')}; no resend")
                    continue
                # 拿到用户显式追加授权后，只有已失败的槽位才会走到这里：把上一次的
                # 尝试记录挪进 attempts 留痕，再按新的授权代次重新派发一次，不覆盖
                # 「曾经失败过」这个事实。
                prior_attempts = list(prior.get("attempts") or [])
                prior_attempts.append(
                    {key: prior.get(key) for key in
                     ("status", "saved_files", "output_sha256", "started_at", "finished_at",
                      "rerun_label") if key in prior})
            # 已派发过的槽位（失败/未知）只有拿到**用户显式追加授权**才允许再发一次。
            # send_budget 按 run_id 去重，所以第 2 次派发用带授权代次后缀的标签，
            # 让两次尝试在 images.jsonl 里可以分辨，而不是伪装成同一次。
            label = slot if not grant else f"{slot}#g{grant['index']}"
            os.environ[send_budget.ENV_RUN] = label
            reservation = send_budget.reserve_image_attempt(
                run_id=label, operation="frozen-wardrobe-generation", model=row["params"]["model"],
                prompt_sha256=row["prompt_sha256"], prompt_chars=len(row["prompt"]),
                note=("user-authorized frozen rerun" if grant else ""))
            result = {"sample_id": slot, "case_id": row["case_id"], "family": row["family"],
                      "channel": row["channel"], "status": "pending_unknown", "saved_files": [],
                      "prompt_sha256": row["prompt_sha256"], "started_at": datetime.now().isoformat()}
            if prior_attempts:
                # 这里必须带上归档：下面的 by_id[slot] = result 会把整条旧记录换掉，
                # 不显式搬过来，上一轮失败的证据就在账本里丢了。
                result["attempts"] = prior_attempts
            if grant:
                result.update({"rerun": True, "rerun_label": label,
                               "rerun_granted_at": grant["granted_at"]})
            by_id[slot] = result
            state["samples"] = list(by_id.values())
            _write_json(str(state_path), state)
            try:
                params = dict(row["params"])
                common = dict(prompt=row["prompt"], image_paths=row["image_paths"],
                              save_sub_dir=str(out), file_prefix=slot, return_metadata=False,
                              log_callback=lambda message: _log(str(message)))
                fn = api_backend.generate_image_aigc2d_gpt if row["channel"] == "gpt" else api_backend.generate_image_aigc2d
                saved = fn(**common, **params) or []
                result.update(status="success" if saved else "failed_no_image",
                              saved_files=[os.path.abspath(p) for p in saved],
                              output_sha256=[sha256_file(p) for p in saved])
            except Exception as exc:
                result.update(status="error", error=f"{type(exc).__name__}: {exc}")
            finally:
                scrub_credential_files(str(out))
            result["finished_at"] = datetime.now().isoformat()
            send_budget.settle_image_attempt(reservation, status=result["status"],
                                            output_path=(result["saved_files"] or [""])[0],
                                            error=result.get("error", ""))
            _write_json(str(state_path), state)
            _log(f"[{result['status']}] {slot}")
        state["samples"] = list(by_id.values())
        _write_json(str(state_path), state)
    except send_budget.SendBudgetExhausted as exc:
        _log(f"[budget] {exc}")
        return 1
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return 0 if all(by_id.get(row["sample_id"], {}).get("status") == "success" for row in rows) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wardrobe_experiment.py",
                                    description="衣装实验族 CLI（预算/授权/生图/评价）")
    parser.add_argument("--spec", default=DEFAULT_SPEC, help="实验 spec 目录（plan.json 等）")
    parser.add_argument("--frozen-pack", default="", help="已准备好的独立衣装生图快照；只支持 prepare/generate/status")
    parser.add_argument("--channel", choices=("gpt", "gemini"), help="只执行冻结包的指定通道")
    parser.add_argument("--pack", default=DEFAULT_PACK, help="prompt 快照目录（manifest.json 等）")
    parser.add_argument("--output-root", default=os.path.join("data", "test-result"),
                        help="隔离输出根目录（默认 data/test-result）")
    parser.add_argument("--run-dir", default="",
                        help="显式指定已有 run 目录（跨天复跑时锁定原目录与原账本；"
                             "不填才按 output-root/当天日期/run-id 推导）")
    parser.add_argument("--run-id", default="", help="沿用同一个输出目录即可恢复；不复用就换 id")
    parser.add_argument("--vision-model", default="", help="登记本轮视觉评价模型（逗号分隔取第一个）")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--resolver", action="store_true")
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--pick-vision-model", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibrate-merged", action="store_true",
                        help="校准一次发两张图（第一轮的旧口径；默认改为每张图一次调用）")
    parser.add_argument("--calibrate-recompute", action="store_true",
                        help="纯本地：用已落盘的原始回复重算校准判定（不发任何请求）")
    parser.add_argument("--calibrate-case", default="",
                        help="只重跑指定的**一条**校准用例（如 CAL-ACC），不整体重跑")
    parser.add_argument("--calibrate-user-text", default="",
                        help="单项重跑时替换提问文本（{label}/{test_id} 占位；不得写入本地 gold 预期）")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--pairs", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--run", action="store_true", help="resolver → review → calibrate → generate → score")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--ledger", action="store_true")
    parser.add_argument("--compose-pack", action="store_true",
                        help="按 spec 组装完整 prompt 快照 + 逐处差异 + 离线自检（不发请求）")
    parser.add_argument("--grant-budget", default="",
                        help="用户显式授权后追加额度（text/generation/vision/model_list）；"
                             "只抬上限，不重置已消耗、不换 run-id")
    parser.add_argument("--grant-limit", type=int, default=0, help="追加后的新上限（必须高于当前值）")
    parser.add_argument("--grant-frozen-budget", default="",
                        help="冻结包专用：用户显式授权后追加额度（目前只支持 generation）。"
                             "只抬上限、不重置已消耗、不换 run-dir；重新派发已失败槽位时"
                             "用带 #gN 后缀的标签，避开同一 run_id 的重复派发拦截")
    parser.add_argument("--retry-slots", default="",
                        help="配合 --grant-frozen-budget：只补跑这些 sample_id（逗号分隔）。"
                             "不填则补跑所选通道里所有已失败的槽位；成功产物仍然复用")
    parser.add_argument("--authorized-by", default="", help="授权人（对话里的用户明确答复）")
    parser.add_argument("--grant-reason", default="", help="授权理由（写进账本，便于复核）")
    parser.add_argument("--grant-source", default="", help="授权来源（对话时间/原话摘要）")
    parser.add_argument("--grant-override", default="",
                        help="用户显式放行某道门（如 calibrate）；必须同时给出理由、假设与已知风险")
    parser.add_argument("--override-by", default="")
    parser.add_argument("--override-reason", default="")
    parser.add_argument("--override-assumption", default="", help="放行时所依赖的假设")
    parser.add_argument("--override-risk", default="", help="放行后已知会受影响的结论")
    parser.add_argument("--override-accepted-at", default="", help="用户接受风险的时间/轮次")
    parser.add_argument("--override-source", default="", help="接受风险的对话来源")
    parser.add_argument("--gate", action="store_true", help="只看候选就绪门禁")
    return parser


STAGES = (("resolver", cmd_resolver), ("review", cmd_review), ("calibrate", cmd_calibrate),
          ("generate", cmd_generate), ("score", cmd_score))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.frozen_pack:
        return command_frozen_pack(args)
    try:
        experiment = Experiment(args)
    except (BudgetExceeded, FileNotFoundError, RuntimeError) as exc:
        _log(f"[error] {type(exc).__name__}: {exc}")
        return 1
    if args.status or args.ledger:
        return cmd_ledger(experiment)
    if args.compose_pack:
        return cmd_compose(experiment)
    if args.grant_budget:
        return command_grant_budget(args, experiment)
    if args.grant_override:
        return command_grant_override(args, experiment)
    if args.calibrate_recompute:
        return cmd_calibrate_recompute(experiment)
    if args.calibrate_case:
        return cmd_calibrate_case(experiment)
    if args.prepare:
        return cmd_prepare(experiment)
    # ---- 在线阶段守卫 -------------------------------------------------------
    # 会真的发出 HTTP 的子命令（含 --run 与 --mock 之外的联网路径）先检查计划是否
    # 明确声明支持在线执行。round4 这类 `cli_compatible: false` 的规划文件即使被
    # 手滑传进来，也必须在**发出任何请求之前**停住。
    online_requested = any(getattr(args, flag, False) for flag in (
        "run", "resolver", "review", "calibrate", "generate", "score", "pairs",
        "calibrate_recompute"))
    if args.pick_vision_model:
        online_requested = True
    if online_requested:
        try:
            assert_online_supported(experiment.plan)
        except RuntimeError as exc:
            _log(f"[gate] 在线阶段被拒（未发出任何请求）：{exc}")
            return 1
    if args.gate:
        gate = experiment.candidate_gate()
        _log(json.dumps(gate, ensure_ascii=False, indent=2))
        return 0 if gate["ready"] else 1
    if args.list_models:
        return cmd_list_models(experiment)
    if args.mock:
        return cmd_mock(experiment)
    if args.pick_vision_model:
        return cmd_pick_vision_model(experiment)
    exit_code = 0
    if args.run:
        gate = experiment.candidate_gate()
        if not gate["ready"]:
            _log("[gate] 候选未就绪，--run 在生图前停止：" + "；".join(gate["reasons"]))
            experiment.ledger.mark_skipped("candidate gate blocked --run")
            return 1
        for name, handler in STAGES:
            code = handler(experiment)
            _log(f"== 阶段 {name} 退出码 {code}")
            if code != 0:
                experiment.ledger.mark_skipped(f"stage {name} failed with code {code}")
                return code
        return 0
    for flag, handler in (("resolver", cmd_resolver), ("review", cmd_review),
                          ("calibrate", cmd_calibrate), ("generate", cmd_generate),
                          ("pairs", cmd_pairs), ("score", cmd_score)):
        if getattr(args, flag):
            exit_code = handler(experiment) or exit_code
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
