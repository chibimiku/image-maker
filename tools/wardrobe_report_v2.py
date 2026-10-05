# -*- coding: utf-8 -*-
"""衣装实验的**参数化汇总**（按 spec 读用例条件与条数，不硬编码 C01..C06）。

用法：
    python tools/wardrobe_report_v2.py --run-dir data/test-result/20261005/wardrobe-round3-r1
    python tools/wardrobe_report_v2.py --run-dir ... --spec prompts/wardrobe/round3/spec

产物（写在 run 目录里）：
  - summary-v2.json：分通道/分用例统计、候选通过标准判定、missing/uncertain 单列
  - gallery-wardrobe.html：16 组匿名并排 + 原始评价响应（单文件离线）

判定口径：
  * 每个用例的成功条件来自 spec（gold_action 决定要检查的维度）；
  * `fail` → failed；`uncertain` 或字段缺失 → unproven（绝不默认通过）；
  * `missing`（没拿到评分/不可读）单列，不折算成 fail 或 pass。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

sys.stdout.reconfigure(encoding="utf-8")
from PIL import Image  # noqa: E402

# 用例必须通过的评分字段：按**该轮实际使用的评分口径**选一套。
# round3 的 score.md 逐字段输出 person_count/identity_anchors/specified_items/…；
# round2/首轮的 score.md 只输出 clothing_action/framing/protected_anchors。
# 这两张表是**历史口径的内建后备值**：新轮次改为在 plan.json 的 `report_fields` 里声明
# （schema → {字段集 → 字段表}），由 `field_map_for_plan()` 读取，不再往这里加分支。
FIELDS_ROUND3 = {
    "preserve": ("clothing_action", "identity_anchors", "specified_items", "framing"),
    "fill": ("clothing_action", "identity_anchors", "specified_items", "framing"),
    "redesign": ("clothing_action", "identity_anchors", "specified_items", "framing"),
}
FIELDS_ROUND2 = {
    "preserve": ("clothing_action", "protected_anchors"),
    "fill": ("clothing_action", "protected_anchors"),
    "redesign": ("clothing_action", "protected_anchors"),
}
ROUND2_FRAMING = {"upper_chest": ("framing",), "full": ("framing",)}
FRAMING_FIELDS = {"upper_chest": ("framing",), "full": ("framing",)}

# 计划里可以声明的条件来源（按顺序尝试）；每个来源都会先按运行目录里真实出现的
# case_id 校验覆盖面，不匹配就换下一个，绝不默认套用别的轮次。
PLAN_CONDITION_CANDIDATES = (
    os.path.join("prompts", "wardrobe", "round4", "spec"),
    os.path.join("prompts", "wardrobe", "round3", "spec"),
)


def field_map_for_plan(plan: dict) -> dict:
    """该计划声明的评分字段口径（按 schema 选一套，缺失时回落到内建历史口径）。

    round4 的 `report_fields` 直接声明字段集名（`required_fields_round4`）与字段表，
    评价模板必须覆盖人物数量、场景、配饰等全部必要条件；这里只做读取与校验，
    不替任何一轮发明字段。
    """
    report_fields = (plan or {}).get("report_fields") or {}
    field_set = str(report_fields.get("field_set") or "").strip()
    tables = {}
    for name, mapping in (report_fields.get("tables") or {}).items():
        tables[str(name)] = {str(key): tuple(value or ()) for key, value in (mapping or {}).items()}
    if not tables:
        tables = {"required_fields_round3": FIELDS_ROUND3, "required_fields_round2": FIELDS_ROUND2}
    if not field_set:
        schema = str((plan or {}).get("schema_version") or "")
        field_set = ("required_fields_round3" if schema.endswith("-round3.v1")
                     else "required_fields_round2")
    return {"field_set": field_set, "tables": tables,
            "framing_fields": dict(report_fields.get("framing_fields") or FRAMING_FIELDS)}


CANDIDATE_VARIANTS = ("v4_candidate", "v3_routed", "lolita")
CONTROL_VARIANTS = ("v3_control", "v2_frozen", "off")


def required_fields_for(conditions: dict, assignment: dict) -> tuple:
    """按配对里实际出现的版本选择评分口径。

    评分口径跟着**那一次评价实际用的 score 模板**走：字段集名与字段表来自该轮计划的
    `report_fields`（内建后备值见 `field_map_for_plan`）。round3 的一对是
    v4_candidate/v3_control，历史轮次是 v3_routed/v2_frozen（round2）或 lolita/off（首轮）。
    """
    variant_names = set((assignment or {}).values())
    if variant_names & {"v4_candidate", "v3_control"}:
        return tuple(conditions.get("required_fields_round3") or ())
    return tuple(conditions.get("required_fields_round2") or ())


def _candidate_variant(per_arm: dict) -> str:
    for name in CANDIDATE_VARIANTS:
        if name in per_arm:
            return name
    return ""


def _control_variant(per_arm: dict) -> str:
    for name in CONTROL_VARIANTS:
        if name in per_arm:
            return name
    return ""


def load(path: str, default=None):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return default if default is not None else {}


def thumb(path: str, max_side: int = 620) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def case_conditions_from_spec(spec_dir: str) -> dict:
    """按 spec 读用例条件与条数（不硬编码旧用例编号）。

    字段口径不再写死 `required_fields_round3`：字段集名与字段表来自 plan 的
    `report_fields`（见 `field_map_for_plan`），所以新轮次只要在计划里声明就能用。
    每个用例同时带上全部字段集（历史轮次的目录结构仍然可读）。
    """
    doc = load(os.path.join(spec_dir, "plan.json"))
    plan = doc
    field_map = field_map_for_plan(plan)
    conditions = {}
    characters_file = plan.get("characters_file") or ""
    characters = []
    if characters_file and os.path.isfile(characters_file):
        characters = (load(characters_file, {}) or {}).get("characters") or []
    by_case = {item.get("case_id"): item for item in characters}
    for case in plan.get("cases") or []:
        case_id = case.get("case_id") or case.get("id")
        character = by_case.get(case_id) or {}
        gold_action = case.get("gold_action") or case.get("action") or "fill"
        framing = case.get("framing") or "full"
        row = {
            "role": case.get("role"), "policy": case.get("policy"), "branch": case.get("branch"),
            "gold_action": gold_action, "gold_presence": case.get("gold_presence"),
            "framing": framing, "repeats": case.get("repeats") or 1,
            "variants": case.get("variants") or [],
            "expected_generation_slots": len(case.get("variants") or [])
            * len(plan.get("channels") or []) * int(case.get("repeats") or 1),
            "identity": character.get("identity") or plan.get("identity"),
            "source_clothing": character.get("source_clothing") or case.get("source_clothing"),
            "protected": character.get("protected") or [],
            "field_set": field_map["field_set"],
        }
        for name, table in field_map["tables"].items():
            row[name] = list(table.get(gold_action, ("clothing_action",))) + list(
                field_map["framing_fields"].get(framing, ()))
        # 历史字段名在内建口径下必须继续可用（round2 报告脚本不因为改造而变口径）
        row.setdefault("required_fields_round3",
                       list(FIELDS_ROUND3.get(gold_action, ("clothing_action",)))
                       + list(FRAMING_FIELDS.get(framing, ())))
        row.setdefault("required_fields_round2",
                       list(FIELDS_ROUND2.get(gold_action, ("clothing_action",)))
                       + list(ROUND2_FRAMING.get(framing, ())))
        conditions[case_id] = row
    return conditions


def config_from_run(run_dir: str) -> dict:
    """round2 运行目录：用例条件来自该轮的 plan。"""
    plan_path = os.path.join("prompts", "wardrobe", "test-plan.json")
    plan = load(plan_path, {})
    conditions = {}
    for case in plan.get("cases") or []:
        target = case.get("target_type")
        gold_action = {"lolita": "redesign", "preserve": "preserve",
                       "visible_detail": "fill"}.get(target, "fill")
        conditions[case["id"]] = {
            "role": case.get("name"), "policy": case.get("policy"), "gold_action": gold_action,
            "framing": "upper_chest" if target == "visible_detail" else "full",
            "repeats": 1, "variants": ["off", "lolita"],
            "expected_generation_slots": 4,
            "required_fields_round3": list(FIELDS_ROUND3.get(gold_action, ("clothing_action",)))
            + list(FRAMING_FIELDS.get("upper_chest" if target == "visible_detail" else "full", ())),
            "required_fields_round2": list(FIELDS_ROUND2.get(gold_action, ("clothing_action",)))
            + list(ROUND2_FRAMING.get("upper_chest" if target == "visible_detail" else "full", ())),
            "identity": None, "source_clothing": None, "protected": case.get("protected")}
    return conditions


def resolve_conditions(run_dir: str, spec_dir: str = "") -> tuple:
    """挑一组与本次运行**实际用例**匹配的条件来源，并显式声明覆盖范围。

    运行目录里的样本 ID 决定这一轮用哪套 case（round3 是 T01..T06，round2 是
    C03/C04/C06/C01/C02；round4 是 U/H/W/R/GP/GF）。条件源必须**包含**运行里出现的
    全部 case；覆盖面差异单独写进 `covered_cases`，不靠目录名猜、也不默认套用别的 spec。
    候选来源见 `PLAN_CONDITION_CANDIDATES`（新轮次排在前，逐个按覆盖面校验）。
    """
    samples = load(os.path.join(run_dir, "samples.json"), {"samples": []}).get("samples") or []
    run_cases = {item.get("case_id") for item in samples if item.get("case_id")}
    if not run_cases:
        pairs = load(os.path.join(run_dir, "pairs-blind.json"), {"pairs": []}).get("pairs") or []
        run_cases = {item.get("case_id") for item in pairs if item.get("case_id")}
    candidates = [spec_dir] + list(PLAN_CONDITION_CANDIDATES)
    tried = []
    for candidate in candidates:
        if not candidate or not os.path.isfile(os.path.join(candidate, "plan.json")):
            continue
        plan = load(os.path.join(candidate, "plan.json"))
        if not plan.get("cases"):
            continue
        # 条件来源必须能给出人物/衣物/保护项：人物卡在 plan 里声明，或与 spec 同级的
        # `characters.json`。两者都没有的 spec（例如还没补人物卡的 round4 规划目录）
        # 直接跳过，避免用它给出「没有人物/衣物信息」的假条件。
        characters_path = plan.get("characters_file") or os.path.join(
            os.path.dirname(candidate), "characters.json")
        if not os.path.isfile(characters_path):
            tried.append({"spec": candidate, "skipped": "缺少人物卡", "path": characters_path})
            continue
        conditions = case_conditions_from_spec(candidate)
        tried.append({"spec": candidate, "cases": sorted(conditions)})
        if run_cases and run_cases <= set(conditions):
            return conditions, candidate, sorted(run_cases)
    fallback = config_from_run(run_dir)
    if run_cases and run_cases <= set(fallback):
        return fallback, "prompts/wardrobe/test-plan.json", sorted(run_cases)
    raise SystemExit(f"找不到与运行用例匹配的条件来源: 运行里是 {sorted(run_cases)}，"
                     f"已尝试 {tried}，round2 plan 是 {sorted(fallback)}")


def judge_arm(record: dict, conditions: dict) -> dict:
    """按用例条件判一条记录的候选结果：pass/failed/unproven/missing。"""
    assignment = (record or {}).get("assignment") or {}
    required = required_fields_for(conditions, assignment)
    if not required:
        return {"outcome": "missing", "fields": {}, "reason": "没有可用的评分字段口径"}
    if not record or record.get("status") != "scored" or not record.get("parsed"):
        return {"outcome": "missing", "fields": {}, "reason": record.get("status") or "no_record",
                "error": record.get("error", "")}
    parsed = record.get("parsed") or {}
    roles = {str(item.get("label") or "").upper(): item for item in parsed.get("images") or []
             if isinstance(item, dict)}
    per_arm = {}
    for label, variant in assignment.items():
        per_arm[variant] = roles.get(label) or {}
    candidate_variant = _candidate_variant(per_arm)
    arm = per_arm.get(candidate_variant) or {}
    fields = {name: arm.get(name) for name in required}
    missing = [name for name, value in fields.items() if value in (None, "")]
    failing = [name for name, value in fields.items() if value == "fail"]
    uncertain = [name for name, value in fields.items() if value == "uncertain"]
    if missing:
        outcome = "unproven"
    elif failing:
        outcome = "failed"
    elif uncertain:
        outcome = "unproven"
    else:
        outcome = "pass"
    control_variant = _control_variant(per_arm)
    control_fields = {name: (per_arm.get(control_variant) or {}).get(name)
                      for name in required}
    return {"outcome": outcome, "fields": fields, "missing_fields": missing,
            "failing_fields": failing, "uncertain_fields": uncertain,
            "control_fields": control_fields, "required_fields": list(required),
            "preferred": parsed.get("preferred"), "reason": parsed.get("reason"),
            "model": record.get("model"), "assignment": assignment,
            "full_outfit_verifiable": parsed.get("full_outfit_verifiable")}


def summarise(run_dir: str, spec_dir: str = "") -> dict:
    conditions, condition_source, covered_cases = resolve_conditions(run_dir, spec_dir)
    samples = load(os.path.join(run_dir, "samples.json"), {"samples": []}).get("samples") or []
    pairs = load(os.path.join(run_dir, "pairs-blind.json"), {"pairs": []}).get("pairs") or []
    scores = load(os.path.join(run_dir, "scores.json"), {"records": []}).get("records") or []
    ledger = load(os.path.join(run_dir, "ledger.json"), {})
    review = load(os.path.join(run_dir, "review-result.json"), {})
    gate = load(os.path.join(run_dir, "candidate-gate.json"), {})
    score_by_pair = {item["pair_id"]: item for item in scores}

    per_case = {}
    for case_id, condition in conditions.items():
        case_samples = [item for item in samples if item.get("case_id") == case_id]
        per_case[case_id] = {
            "covered_this_run": case_id in covered_cases,
            "expected_generation_slots": condition["expected_generation_slots"],
            "attempted_slots": len(case_samples),
            "success": sum(1 for item in case_samples if item.get("status") == "success"),
            "by_variant": {variant: sum(1 for item in case_samples
                                        if item.get("variant") == variant
                                        and item.get("status") == "success")
                           for variant in condition["variants"]},
        }

    judgements = []
    for pair in pairs:
        condition = conditions.get(pair["case_id"])
        if not condition:
            judgements.append({"pair_id": pair["pair_id"], "case_id": pair["case_id"],
                               "outcome": "unproven", "reason": "spec 里没有这个用例的条件"})
            continue
        record = score_by_pair.get(pair["pair_id"]) or {}
        judged = judge_arm(record, condition)
        judgements.append({"pair_id": pair["pair_id"], "case_id": pair["case_id"],
                           "channel": pair.get("channel"), "repeat": pair.get("repeat"),
                           "gold_action": condition["gold_action"], "framing": condition["framing"],
                           **judged})
    counts = {name: sum(1 for item in judgements if item["outcome"] == name)
              for name in ("pass", "failed", "unproven", "missing")}
    control_only = load(os.path.join(run_dir, "control-defects.json"), {})
    waves = {}
    for item in samples:
        wave = item.get("wave") or "unspecified"
        row = waves.setdefault(str(wave), {"attempted": 0, "success": 0,
                                           "by_case": {}, "by_channel": {}})
        row["attempted"] += 1
        if item.get("status") == "success":
            row["success"] += 1
        case_key = str(item.get("case_id") or item.get("target_family") or "unknown")
        row["by_case"][case_key] = row["by_case"].get(case_key, 0) + 1
        channel = str(item.get("channel") or "unknown")
        row["by_channel"][channel] = row["by_channel"].get(channel, 0) + 1
    paired = any((item.get("assignment") or {}) for item in pairs)
    return {
        "run_dir": os.path.abspath(run_dir), "written_at": datetime.now().isoformat(),
        "condition_source": condition_source,
        "spec_condition_count": len(conditions),
        "covered_cases": covered_cases,
        "uncovered_cases": sorted(set(conditions) - set(covered_cases)),
        "expected_pair_slots": sum(condition["expected_generation_slots"] // 2
                                   for case_id, condition in conditions.items()
                                   if case_id in covered_cases),
        "case_conditions": conditions,
        # 直接采样（无 control/candidate 配对）的轮次按波次汇总；paired 为 false 时
        # 下面 `judgements` 的判定**不代表**该轮结论，必须看人工逐图记录。
        "sampling_mode": "paired_arms" if paired else "direct_sampling_no_pairing",
        "waves": waves,
        "generation": {"total": len(samples),
                       "success": sum(1 for item in samples if item.get("status") == "success"),
                       "per_case": per_case},
        "pairs": {"total": len(pairs), "ready": sum(1 for item in pairs if item.get("status") == "ready"),
                  "balance": load(os.path.join(run_dir, "pairs-blind.json")).get("balance")},
        "scoring": {"records": len(scores),
                    "scored": sum(1 for item in scores if item.get("status") == "scored"),
                    "not_scored": sum(1 for item in scores if item.get("status") != "scored")},
        "judgements": judgements,
        "counts": counts,
        "candidate_verdict": ("passed" if counts["failed"] == 0 and counts["unproven"] == 0
                              and counts["missing"] == 0 and judgements else
                              "failed" if counts["failed"] else "unproven"),
        "review": {"candidate_ready": review.get("candidate_ready"),
                   "blocking_count": len(review.get("blocking") or []),
                   "control_defects_reported": review.get("control_defects_reported")},
        "candidate_gate": gate,
        "control_defects": control_only,
        "budget": {"budgets": ledger.get("budgets"),
                   "attempts_by_kind": _count_by_kind(ledger.get("attempts") or [])},
    }


def _count_by_kind(attempts: list) -> dict:
    counts = {}
    for item in attempts:
        kind = item.get("kind") or "unknown"
        counts.setdefault(kind, {"total": 0, "success": 0, "failed": 0})
        counts[kind]["total"] += 1
        status = item.get("status")
        if status == "success":
            counts[kind]["success"] += 1
        elif status != "inflight":
            counts[kind]["failed"] += 1
    return counts


def build_gallery(run_dir: str, summary: dict) -> str:
    pairs = load(os.path.join(run_dir, "pairs-blind.json"), {"pairs": []}).get("pairs") or []
    scores = {item["pair_id"]: item for item in
              load(os.path.join(run_dir, "scores.json"), {"records": []}).get("records") or []}
    judgement_by_pair = {item["pair_id"]: item for item in summary["judgements"]}
    cards = []
    for pair in pairs:
        record = scores.get(pair["pair_id"]) or {}
        judged = judgement_by_pair.get(pair["pair_id"]) or {}
        figures = []
        for label in ("A", "B"):
            variant = (pair.get("assignment") or {}).get(label)
            image = (pair.get("images") or {}).get(label)
            if image and os.path.isfile(image):
                figures.append(f'<figure><img src="{thumb(image)}" alt="{pair["pair_id"]}-{label}">'
                               f'<figcaption><b>{label}</b> · {variant}<br>'
                               f'<span class=path>{os.path.basename(image)}</span></figcaption></figure>')
            else:
                figures.append(f'<figure class=missing><div class=ph>无产物</div>'
                               f'<figcaption><b>{label}</b> · {variant}</figcaption></figure>')
        raw = record.get("raw") or record.get("error") or "（无评分记录）"
        conditions = (summary["case_conditions"].get(pair["case_id"]) or {})
        cards.append(f"""
<section class=card>
  <h3>{pair['pair_id']} <small>{conditions.get('role')} · gold={conditions.get('gold_action')} ·
  framing={conditions.get('framing')}</small></h3>
  <p class=meta>匿名映射：{'，'.join(f"{k}={v}" for k, v in (pair.get('assignment') or {}).items())}｜
  判定：<b>{judged.get('outcome', 'missing')}</b>｜
  候选字段：{json.dumps(judged.get('fields'), ensure_ascii=False)}</p>
  <p class=meta>条件（人物/衣物/要求/protected 原样取自 spec）：
  {json.dumps({k: conditions.get(k) for k in ('identity', 'source_clothing', 'protected')}, ensure_ascii=False)[:600]}</p>
  <div class=row>{''.join(figures)}</div>
  <details><summary>原始评价响应</summary><pre>{raw}</pre></details>
</section>""")
    head = f"""<!doctype html><html lang=zh><head><meta charset=utf-8>
<title>衣装实验对照（{summary['generation']['success']}/{summary['generation']['total']}）</title>
<style>
body{{font-family:system-ui,'Segoe UI',sans-serif;margin:22px;background:#faf8f6;color:#222}}
h1{{font-size:20px}} h3{{margin:0 0 6px;font-size:15px}} small{{font-weight:400;color:#666}}
.card{{background:#fff;border:1px solid #e6e0da;border-radius:10px;padding:12px;margin:14px 0}}
.row{{display:flex;gap:12px;flex-wrap:wrap}} figure{{margin:0;width:300px}}
figure img{{width:300px;border-radius:8px;border:1px solid #ddd;display:block}}
figcaption{{font-size:12px;color:#444;margin-top:6px}} .path{{color:#888;font-size:11px}}
.meta{{font-size:12px;color:#555}} pre{{white-space:pre-wrap;font-size:12px;background:#f6f4f2;
padding:10px;border-radius:6px;max-height:380px;overflow:auto}} .missing .ph{{width:300px;height:400px;
display:flex;align-items:center;justify-content:center;background:#f1eeeb;color:#999;border-radius:8px}}
</style></head><body>
<h1>衣装实验：control vs candidate（{summary['run_dir']}）</h1>
<p class=meta>生图成功 {summary['generation']['success']}/{summary['generation']['total']}，
配对 {summary['pairs']['ready']}/{summary['pairs']['total']}，
评分 {summary['scoring']['scored']}/{summary['scoring']['records']}；
判定 pass={summary['counts']['pass']} failed={summary['counts']['failed']}
unproven={summary['counts']['unproven']} missing={summary['counts']['missing']}</p>
<p class=meta>用例条件来源：{summary['condition_source']}（{summary['spec_condition_count']} 个用例）</p>
"""
    html = head + "".join(cards) + "\n</body></html>"
    path = os.path.join(run_dir, "gallery-wardrobe.html")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(html)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--spec", default="")
    args = parser.parse_args(argv)
    run_dir = os.path.abspath(args.run_dir)
    summary = summarise(run_dir, args.spec)
    with open(os.path.join(run_dir, "summary-v2.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    gallery = build_gallery(run_dir, summary)
    print(json.dumps({key: summary[key] for key in
                      ("condition_source", "spec_condition_count", "generation", "pairs",
                       "scoring", "counts", "candidate_verdict")}, ensure_ascii=False, indent=2))
    print("gallery:", gallery, os.path.getsize(gallery), "bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
