#!/usr/bin/env python
"""画风相似度**严格受控实验** CLI（协议 1.0 / 2026-10-06）。

只做参数解析与阶段编排；数据冻结、计算、统计、报告在
`utils/style_experiment_controlled*.py`，相似度数值一律来自共享入口
`utils.style_similarity.compare_images` 与
`modules.image_analysis.style_comparison.compare_state_in_process`。

```powershell
# 1) 冻结数据/参数/主指标/门槛（不读任何分数）
python -u tools/style_similarity_experiment.py freeze

# 2) 跑各阶段（可重复执行：已完成阶段直接复用）
python -u tools/style_similarity_experiment.py run --stages E0,E1,E2,E3

# 3) 汇总并写报告（Markdown / results.json / HTML / 失败证据）
python -u tools/style_similarity_experiment.py report

# 4) 重跑某阶段（保留原结果，追加账本）
python -u tools/style_similarity_experiment.py run --stages E1 --force
```
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import style_experiment_controlled as core  # noqa: E402
from utils import style_experiment_controlled_phases as phases  # noqa: E402
from utils.style_experiment_controlled import _confounds  # noqa: E402

STAGES = ("E0", "E1", "E2", "E3", "E4", "E5")


def load_ledger():
    ledger = core.AttemptLedger(core.run_root() / "ledger" / "attempts.jsonl").load()
    return ledger


def load_plan():
    plan = core.read_json(core.run_root() / "PLAN" / "plan-full.json")
    if not plan:
        raise SystemExit("尚未冻结计划：先运行 `freeze`")
    return plan


def command_freeze(args):
    run_dir = core.run_root()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ledger").mkdir(parents=True, exist_ok=True)
    ledger = load_ledger()
    print(f"[freeze] 运行目录 {run_dir}", flush=True)
    plan = core.build_plan(ledger, progress=lambda message: print("[plan] " + message, flush=True))
    freeze = core.write_freezes(plan, ledger, progress=lambda message: print("[freeze] " + message, flush=True))
    sheets = core.near_duplicate_contact_sheets(
        plan, run_dir / "PLAN" / "near-duplicate-review")
    plan["near_duplicate_contact_sheets"] = sheets
    core.atomic_json(run_dir / "PLAN" / "plan-full.json", plan)
    print(f"[freeze] 近重复候选接触表 {len(sheets)} 张（待人工核查）", flush=True)
    print(f"[freeze] 代码/提示词 {len(freeze['code_and_prompts_sha256'])} 项 hash 已冻结；"
          f"权重 {len(freeze['weights'])} 项；E3 图样 "
          f"{len(plan['e3_frozen']['records'])} 张", flush=True)
    return 0


def command_run(args):
    plan = load_plan()
    ledger = load_ledger()
    stages = [stage.strip().upper() for stage in args.stages.split(",") if stage.strip()]
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        raise SystemExit(f"未知阶段：{unknown}")
    if args.force:
        for stage in stages:
            marker = core.stage_result_path(stage)
            if marker.is_file():
                shutil.copy2(marker, marker.with_suffix(".json.prev"))
                marker.unlink()
                print(f"[force] 已归档并删除 {marker}", flush=True)
    results = phases.run_all(ledger, plan, progress=lambda message: print("[run] " + message, flush=True),
                             stages=stages, locate_regions=not args.no_locate)
    for stage in stages:
        value = results.get(stage) or {}
        print(f"[{stage}] status={value.get('status')}", flush=True)
    return 0


def command_report(args):
    from utils.style_experiment_annotation import write_annotation_assets
    from utils.style_experiment_reports import write_reports
    plan = load_plan()
    ledger = load_ledger()
    results = {"protocol_version": core.PROTOCOL_VERSION, "protocol_date": core.PROTOCOL_DATE,
               "run_id": plan["run_id"], "plan": _plan_view(plan)}
    for stage in STAGES:
        results[stage] = core.read_json(core.stage_result_path(stage), {}) or {}
    results["stages"] = core.stage_statuses(results, plan)
    results["confounds"] = _confounds(plan)
    results["deviations"] = deviations(results)
    results["ledger_summary"] = ledger.summarize()
    results["ledger_rows"] = ledger.rows
    results["gaps"] = core.collect_gaps(results, plan)
    results["next_step"] = core.next_step(results)
    results["summary_template"] = core.summary_template(results)
    results["e5_images"] = [{"label": row["image_id"] + "（历史 Gemini 产物）", "path": row["path"]}
                            for row in plan["e0_images"] if row["kind"] == "generated"]
    # E4 人工标注入口（含自动叠图与逐图 hash）作为可交付件
    e4_images = [{"image_id": row["image_id"], "kind": row["kind"], "path": row["path"],
                  "sha256": row.get("sha256"), "style_id": row.get("style_id"),
                  "history_failure": row.get("history_failure"), "overlay": row.get("overlay")}
                 for row in (results.get("E4") or {}).get("rows") or []]
    if e4_images:
        results["e4_annotation_assets"] = write_annotation_assets(e4_images, core.run_root())
    files = write_reports(results, plan, core.run_root())
    files.update(results.get("e4_annotation_assets") or {})
    print(json.dumps(files, ensure_ascii=False, indent=2), flush=True)
    return 0


def _plan_view(plan) -> dict:
    """报告用的计划视图：去掉超大的图片列表，保留计数与来源。"""
    view = {key: value for key, value in plan.items()
            if key not in ("split_records", "e0_images", "e3_frozen", "e5_states", "styles_config_snapshot")}
    view["split"] = {style: {"source": payload["source"], "pool_size": payload["pool_size"],
                             "usable_pool": payload["usable_pool"],
                             "references": _brief(payload["references"]),
                             "queries": _brief(payload["queries"]),
                             "extras": _brief(payload["extras"]),
                             "near_duplicate_excluded": payload["near_duplicate_excluded"],
                             "near_duplicate_pairs": payload["near_duplicate_pairs"]}
                     for style, payload in plan["split"].items()}
    view["e3"] = {**plan["e3"], "frozen_count": len(plan.get("e3_frozen", {}).get("records") or [])}
    return view


def _brief(rows):
    return [{"image_id": row["image_id"], "path": row["path"], "sha256": row["sha256"],
             "width": row["width"], "height": row["height"], "framing": row["framing"],
             "brightness": row["brightness"], "background_density": row["background_density"],
             "dominant_hue": row.get("dominant_hue")} for row in rows]


def deviations(results) -> list[dict]:
    plan = results.get("plan") or {}
    e2_rule = ((plan.get("e2") or {}).get("selection_rule") or {})
    items = [
        {"what": "E2 的 P/N（困难对照）选择依据由确定性自动匹配规则生成，不是「不知道指标值的标注者」填写",
         "impact": "困难对照的选择依据缺少独立人工背书；规则只读取冻结属性，未读取任何指标值",
         "handling": "在 E2 结果与报告第 5 节单列为协议偏离；人工标注列为待项"},
        {"what": "第 4 种画风用 Renian 替代协议优先清单里的 TID",
         "impact": "与协议原始画风清单不完全一致，未测风格不得外推",
         "handling": "在看分数之前决定并记录在 protocol-lock.json 的 style_substitution"},
        {"what": "E3 使用参数化可控绘制图样而非真实插画",
         "impact": "只证明测量机制会响应指定变化，不代表真实插画效度",
         "handling": "协议 §7 明确允许，并在 E3 结果与报告中声明"},
        {"what": "E4 实际运行 8 张（6 个可控案例 + 2 张历史 Gemini 图），协议写的是 10 张（6 个案例 + 4 张历史图）",
         "impact": "历史图分母比协议少 2 张（那次历史测试只产出 2 张图），覆盖率分母相应变小",
         "handling": "在 E4 结果与报告中写明实际分母 8 与差额 2 的来源，不把 8 说成 10"},
        {"what": "E4 人工标注与 E5 人类盲评尚未提供",
         "impact": "自动定位可用率、人工间重复性、候选排序效度均无法验收",
         "handling": "登记 awaiting_human，并交付盲评包与标注入口，未用模型代填"},
        {"what": "E5 的视觉复核只完成「参考图正序 / 倒序」两种呈现顺序，候选级顺序敏感性无法检验",
         "impact": "每个画风只有 1 个候选，候选顺序对评分的影响未测",
         "handling": "在 E5 结果中写明检验的是参考图顺序，不冒充候选级顺序敏感性"},
        {"what": "E5 新增 18 个生图槽位没有费用授权",
         "impact": "E5 只有历史 2 张产物，最多 1 组组内比较，远少于 24 个有效独立题",
         "handling": "登记 awaiting_budget，未发出任何新增生图请求"},
    ]
    if results.get("E0", {}).get("critical_failures"):
        items.append({"what": "E0 存在未通过的关键项",
                      "impact": "实现可用性门槛未达成，需要按新代码版本重跑受影响计算",
                      "handling": "见 E0-result.json 的逐项结果"})
    return items


def command_complete_e5_orders(args):
    """补齐 E5 的参考图倒序复核（正序已在主运行完成）。

    用途：初次运行因状态文件路径错误只完成正序时，用同一条共享入口补跑倒序，
    作为协议 §9 要求的「正序/倒序各运行一次」的顺序敏感性证据。
    """
    from utils.style_experiment_controlled_phases import e5_batch_review
    plan = load_plan()
    ledger = load_ledger()
    result = e5_batch_review(ledger, plan, progress=lambda message: print("[e5] " + message, flush=True))
    print(json.dumps(result, ensure_ascii=False, indent=2)[:2000], flush=True)
    return 0


def command_status(args):
    plan = core.read_json(core.run_root() / "PLAN" / "plan-full.json", {}) or {}
    ledger = load_ledger()
    summary = {"run_root": str(core.run_root()), "plan_frozen": bool(plan),
               "stages": {}}
    for stage in STAGES:
        value = core.read_json(core.stage_result_path(stage), {})
        summary["stages"][stage] = (value or {}).get("status", "planned")
    summary["ledger"] = ledger.summarize()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="画风相似度严格受控实验（协议 1.0）")
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="冻结数据/参数/主指标/门槛（不读任何分数）")
    freeze.set_defaults(func=command_freeze)
    run = sub.add_parser("run", help="执行阶段")
    run.add_argument("--stages", default=",".join(STAGES))
    run.add_argument("--force", action="store_true", help="归档并删除该阶段结果后重跑（账本继续追加）")
    run.add_argument("--no-locate", action="store_true", help="E4 不调用文本模型生成自动定位")
    run.set_defaults(func=command_run)
    report = sub.add_parser("report", help="汇总并写 Markdown / results.json / HTML / 失败证据")
    report.set_defaults(func=command_report)
    status = sub.add_parser("status", help="只看阶段与账本摘要")
    status.set_defaults(func=command_status)
    orders = sub.add_parser("complete-e5-orders", help="补齐 E5 参考图倒序复核（共享入口，会调用文本端点）")
    orders.set_defaults(func=command_complete_e5_orders)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
