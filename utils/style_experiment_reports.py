"""受控实验的报告输出：结果 Markdown、自包含 HTML、results.json。

只整理 `utils.style_experiment_controlled_phases` 已算出的、带真实覆盖率与真实缺项的状态；
不在这里计算、修补或美化任何指标。
"""

from __future__ import annotations

import base64
import html as html_module
import json
from pathlib import Path

from utils.style_experiment_controlled import ACCEPTANCE, MAIN_METRIC, atomic_json, now

STAGE_STATUS = ("planned", "running", "complete", "partial", "awaiting_human", "awaiting_budget", "blocked")
STATUS_LABEL = {"pass": "通过", "fail": "未通过", "error": "错误", "ok": "通过",
                "partial": "部分", "complete": "完成", "planned": "未运行"}


def _fmt(value, digits=6):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
        return text if text else "0"
    return str(value)


def _table(headers, rows):
    lines = ["| " + " | ".join(str(item) for item in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(
            item if isinstance(item, str) else _fmt(item) for item in row) + " |")
    return "\n".join(lines)


def _image_uri(path, max_bytes=5_000_000):
    path = Path(str(path))
    if not path.is_file():
        return None
    data = path.read_bytes()
    if len(data) > max_bytes:
        return None
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def stage_table(stages) -> str:
    rows = []
    for name in ("E0", "E1", "E2", "E3", "E4", "E5"):
        stage = stages.get(name) or {}
        rows.append([name, STATUS_LABEL.get(stage.get("status"), stage.get("status", "未运行")),
                     stage.get("planned", "—"), stage.get("actual", "—"),
                     stage.get("independent_works", "—"),
                     ", ".join(stage.get("missing_ids") or []) or "—",
                     (stage.get("reason") or "").strip()])
    return _table(["阶段", "状态", "计划", "实际", "独立作品/主体", "缺项 ID", "原因"], rows)


def purpose_conclusions(results) -> list[dict]:
    """用途结论：逐项填「通过 / 仅可辅助 / 未验证」，依据到具体实验/ID。"""
    e0 = results.get("E0") or {}
    e1 = results.get("E1") or {}
    e2 = results.get("E2") or {}
    e3 = results.get("E3") or {}
    e4 = results.get("E4") or {}
    e5 = results.get("E5") or {}
    primary = ((e1.get("analysis") or {}).get(MAIN_METRIC)) or {}
    preference = ((e2.get("analysis") or {}).get("preference")) or {}
    e3_gate = ((e3.get("analysis") or {}).get("gate")) or {}
    e4_coverage = (e4.get("coverage") or {})
    e4_gate = (e4.get("gate") or {})
    blind = (e5.get("blind") or {})
    e0_failures = e0.get("critical_failures") or []
    retrieval_ok = (primary.get("top1_accuracy") or 0) >= ACCEPTANCE["style_retrieval_assist"]["value"] \
        and all((value.get("accuracy") or 0) >= 2 / 3 for value in (primary.get("per_style") or {}).values())
    contrast_ok = bool(preference) and all((preference.get(metric, {}).get("rate") or 0) >= 0.75
                                           for metric in ("gram", "adain", "lpips", "csd"))
    return [
        {"purpose": "实现可用（同图 / 对称 / 缓存 / 入口一致 / 失败状态）",
         "verdict": "通过" if not e0_failures and e0.get("status") == "complete" else "未通过",
         "evidence": ("E0 全部关键项通过（" + str(len(e0.get("checks") or [])) + " 项检查）"
                      if not e0_failures else "未通过：" + "; ".join(e0_failures))},
        {"purpose": "全图八组作为画风检索辅助",
         "verdict": "仅可辅助" if retrieval_ok else ("不能用" if (primary.get("top1_accuracy") or 0) < 0.5 else "仅可辅助（未过预注册门槛）"),
         "evidence": (f"E1 主指标 {MAIN_METRIC} Top-1={_fmt(primary.get('top1_accuracy'), 4)}"
                      f"（门槛 {ACCEPTANCE['style_retrieval_assist']['value']}）；"
                      f"E2 困难对照 CSD 正确偏好率={_fmt((preference.get('csd') or {}).get('rate'), 4)}"
                      f"（门槛 0.75，{'达标' if contrast_ok else '未达标'}）")},
        {"purpose": "自动定位默认参与局部可靠测量",
         "verdict": "未验证",
         "evidence": (f"E4 自动可用 {e4_coverage.get('usable')}/{e4_coverage.get('images')}"
                      f"（可用率 {_fmt(e4_coverage.get('usable_rate'), 4)}）；"
                      f"人工确认 {e4_coverage.get('confirmed')}；人工间重复性未提供 → {e4_gate.get('reason', '')}")},
        {"purpose": "人工确认定位后的局部细项测量",
         "verdict": "仅可辅助（机制已验证，真实图未验证）" if e3_gate.get("passed") else "未验证",
         "evidence": (f"E3 受控图样目标方向命中 {_fmt(e3_gate.get('direction_hit'), 4)}"
                      f"（门槛 {ACCEPTANCE['local_measurement_assist']['value']}）；"
                      f"负对照 {'全部通过' if e3_gate.get('negative_controls_passed') else '未全部通过'}；"
                      f"未验证维度 " + "、".join(sorted(((e3.get('analysis') or {}).get('diagnostics') or {}))))},
        {"purpose": "自动选最佳候选（候选排序）",
         "verdict": "未验证",
         "evidence": (f"人类盲评有效独立题 {blind.get('valid_independent_questions')}"
                      f"（门槛 {ACCEPTANCE['candidate_ranking_assist']['min_items']}）；"
                      f"视觉模型已复核 {sum(len(v) for v in ((e5.get('review') or {}).get('runs') or {}).values())} 次运行，"
                      f"但两人一致性、方向一致率均无法计算")},
    ]


def markdown_report(results, paths) -> str:
    stages = results.get("stages", {})
    plan = results.get("plan") or {}
    e0 = results.get("E0") or {}
    e1 = results.get("E1") or {}
    e2 = results.get("E2") or {}
    e3 = results.get("E3") or {}
    e4 = results.get("E4") or {}
    e5 = results.get("E5") or {}
    lines = []
    add = lines.append
    add("# 画风相似度严格受控实验（结果）")
    add("")
    add(f"> 协议 {results.get('protocol_version')}（{results.get('protocol_date')}）· "
        f"运行 id `{results.get('run_id')}` · 生成 {results.get('generated_at')} · "
        f"设备 {plan.get('device')} / {plan.get('precision')}")
    add("")
    add("**本文件是执行结果，不是通过声明。** 计算成功、`status=ok`、单元测试通过都不等于画风效度成立；"
        "所有未完成、缺人类输入、缺额度的项按实际状态列出。")
    add("")

    add("## 1. 阶段状态")
    add("")
    add(stage_table(stages))
    add("")
    add("协议偏离（单列）：")
    add("")
    add(_table(["#", "偏离", "影响", "处理"],
               [[index + 1, item.get("what"), item.get("impact"), item.get("handling")]
                for index, item in enumerate(results.get("deviations") or [])]))
    add("")

    add("## 2. 数据来源、split、近重复核查、混杂分布")
    add("")
    add(f"随机种子 `{plan.get('seed')}`；主指标 **{MAIN_METRIC}**（预注册）；"
        f"设备/精度 **{plan.get('device')}/{plan.get('precision')}**（整场唯一，未中途切换）。")
    add("")
    split = plan.get("split") or {}
    add(_table(["画风", "数据源", "候选池", "可用池", "参考/query", "预留 E2 池", "自动剔除重复", "待人工核查的模糊对"],
               [[style, payload["source"]["source_root"], payload["pool_size"], payload["usable_pool"],
                 f"{len(payload['references'])}/{len(payload['queries'])}", len(payload["extras"]),
                 len(payload["near_duplicate_excluded"]), payload.get("fuzzy_pairs_awaiting_review")]
                for style, payload in sorted(split.items())]))
    add("")
    add("确定性重复剔除规则与阈值、分层选样规则见 "
        "[sampling-rule.md](../../../prompts/style-extraction/similarity-validation-v1/sampling-rule.md)；"
        "被剔除项逐条记录在 `PLAN/images.json`。")
    add("")
    if plan.get("style_substitution"):
        add("**画风替换（在看分数之前决定）**：" + plan["style_substitution"]["reason"])
        add("")
    confounds = results.get("confounds") or {}
    if confounds:
        add("混杂分布（如实列出不平衡，不声称已消除）：")
        add("")
        add(_table(["画风", "分类计数"], [[style, ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))]
                                       for style, counts in sorted(confounds.items())]))
        add("")
    add("冻结包：" + "、".join(f"[{Path(item).name}](<{item}>)" for item in paths.get("plan_files", [])))
    add("")
    add("代码/提示词 SHA-256、权重清单、依赖版本：`PLAN/code-freeze.json`、`PLAN/weights.json`、`PLAN/runtime.json`。")
    add("")
    add("E5 冻结的生成参数（历史产物来源，未新增请求）：`PLAN/e5-plan.json`；"
        "历史请求回放见 `data/test-result/20261006/style-similarity-gemini/*/*_aigc2d_replay_*.json`（已脱敏）。")
    add("")

    add("## 3. E0 实现一致性与错误传播")
    add("")
    add(_table(["检查项", "结果", "证据"],
               [[row["check"], STATUS_LABEL.get(row["status"], row["status"]),
                 json.dumps(row.get("detail") or row.get("error"), ensure_ascii=False)[:420]]
                for row in e0.get("checks", [])]))
    add("")
    if e0.get("critical_failures"):
        add("**关键失败**：" + "、".join(e0["critical_failures"]))
        add("")

    add("## 4. E1 原作品画风辨别（主要效度实验）")
    add("")
    counts = e1.get("counts") or {}
    add(f"query {counts.get('queries')} 个 × 4 个画风参考集 → 聚合单元 {counts.get('aggregate_units')}，"
        f"逐图配对 {counts.get('image_pairs')}。主指标 **{e1.get('main_metric')}**。")
    add("")
    rows = []
    for metric, payload in sorted((e1.get("analysis") or {}).items()):
        statistics = payload.get("counts") or {}
        rows.append([metric, _fmt(payload.get("top1_accuracy"), 4), _fmt(payload.get("apportioned_accuracy"), 4),
                     f"{statistics.get('correct')}/{statistics.get('total')}",
                     f"{statistics.get('tie')}/{statistics.get('wrong')}/{statistics.get('missing')}",
                     f"[{_fmt((payload.get('wilson_95') or {}).get('low'), 3)}, {_fmt((payload.get('wilson_95') or {}).get('high'), 3)}]",
                     f"[{_fmt((payload.get('bootstrap_by_query') or {}).get('low'), 3)}, "
                     f"{_fmt((payload.get('bootstrap_by_query') or {}).get('high'), 3)}]"])
    add(_table(["指标", "Top-1", "分摊并列", "命中/总数", "并列/错误/缺失", "Wilson 95%", "cluster bootstrap 95%（query 簇）"], rows))
    add("")
    primary = (e1.get("analysis") or {}).get(MAIN_METRIC) or {}
    add("各画风命中率（主指标）：" + "、".join(
        f"{style} {value.get('correct')}/{value.get('total')}"
        for style, value in sorted((primary.get("per_style") or {}).items())))
    add("")
    confusion = primary.get("confusion") or {}
    if confusion:
        styles = sorted(confusion)
        add("4×4 混淆矩阵（行=真实画风，列=预测）：")
        add("")
        add(_table(["真实\\预测"] + styles,
                   [[style] + [confusion.get(style, {}).get(other, 0) for other in styles] for style in styles]))
        add("")
    wrong = [row for row in primary.get("rows", []) if row["status"] in ("wrong", "tie", "missing")]
    add("错误 / 并列 / 缺失的 query ID：" + ("；".join(
        f"`{row['query_id']}`（真实 {row['query_style']}，"
        f"{'并列 ' + '/'.join(row['top']) if row.get('top') and len(row['top']) > 1 else '预测 ' + str(row.get('predicted'))}）"
        for row in wrong) or "无"))
    add("")
    same = ((e1.get("distributions") or {}).get("same_style") or {}).get(MAIN_METRIC) or {}
    different = ((e1.get("distributions") or {}).get("different_style") or {}).get(MAIN_METRIC) or {}
    add(_table(["主指标分布", "n", "均值", "中位数", "p10", "p90", "标准化间隔"],
               [["同风格", same.get("count"), same.get("mean"), same.get("median"), same.get("p10"),
                 same.get("p90"), same.get("standardized_separation")],
                ["异风格", different.get("count"), different.get("mean"), different.get("median"),
                 different.get("p10"), different.get("p90"), different.get("standardized_separation")]]))
    add("")
    add("**结论**：" + str(e1.get("conclusion")))
    add("")
    add("试验集只作诊断：区间按 query/work 取样（不能把 576 个相关配对当 576 个独立样本），"
        "同风格参考集等权，不把不同画风的参考图混进一个均值。")
    add("")

    add("## 5. E2 内容与配色干扰对照")
    add("")
    preference = ((e2.get("analysis") or {}).get("preference")) or {}
    add(_table(["指标", "正确偏好", "n", "正确率", "并列", "错误", "配色相近分块（n/正确率）", "景别+亮暗+背景匹配分块（n/正确率）"],
               [[metric, value.get("correct"), value.get("n"), _fmt(value.get("rate"), 4),
                 value.get("tie"), value.get("wrong"),
                 f"{(value.get('split_by_colour_match') or {}).get('n')}/"
                 f"{_fmt((value.get('split_by_colour_match') or {}).get('rate'), 3)}",
                 f"{(value.get('split_by_composition_match') or {}).get('n')}/"
                 f"{_fmt((value.get('split_by_composition_match') or {}).get('rate'), 3)}"]
                for metric, value in sorted(preference.items())]))
    add("")
    transforms = ((e2.get("analysis") or {}).get("transform_summary")) or {}
    add("确定性干扰（N0–N4）后仍把 anchor 判为自身画风的比例：")
    add("")
    add(_table(["变体", "仍为自身画风", "翻转", "缺失", "预测正确率", "翻转到的画风"],
               [[name, value.get("same_style"), value.get("other_style"), value.get("missing"),
                 _fmt(value.get("prediction_accuracy"), 4),
                 ", ".join(sorted({item["predicted"] for item in value.get("flips", [])})) or "—"]
                for name, value in sorted(transforms.items())]))
    add("")
    add("N1/N2 是配色/明度敏感性诊断，不要求指标不变；N3/N4 涉及边界与插值，不预设画风必然相同；"
        "不同尺度的分数不得直接相减比较。变体参数、掩膜与 hash 见 `E2/variants-frozen.json`，"
        "人工核查接触表见 `E2/contact-sheets/`。")
    add("")

    add("## 6. E3 局部测量的受控响应")
    add("")
    analysis = e3.get("analysis") or {}
    gate = analysis.get("gate") or {}
    add(f"- 目标响应方向命中：**{_fmt(gate.get('direction_hit'), 4)}**（门槛 {ACCEPTANCE['local_measurement_assist']['value']}）")
    add(f"- 负对照：{'全部通过' if gate.get('negative_controls_passed') else '未全部通过'}"
        f"（{len(analysis.get('negative_controls') or [])} 个案例级负对照）")
    add(f"- 受控干预维度：{', '.join(((e3.get('coverage') or {}).get('controlled_metrics')) or [])}")
    add(f"- 诊断维度（无对应干预，**不得**宣称已验证）："
        f"{', '.join(sorted(analysis.get('diagnostics') or {}))}")
    add("")
    add("剂量响应（贴近度越低表示与该干预的原图差异越大）：")
    add("")
    add(_table(["案例", "干预", "目标项", "剂量贴近度（原→弱→中→强）", "单调", "判定"],
               [[row["case_id"], row["intervention"], row.get("target"),
                 " → ".join(_fmt(value, 4) for value in (row.get("dosage") or {}).get("closeness", [])),
                 _fmt(row.get("monotonic")),
                 STATUS_LABEL.get(row.get("status"), row.get("status"))]
                for row in analysis.get("responses", [])]))
    add("")
    add("负对照明细（要求贴近度精确为 1.0）：")
    add("")
    add(_table(["案例", "负对照", "期望不变项", "实测贴近度", "判定"],
               [[row["case_id"], row["control"], ", ".join(row["expect_no_change"]),
                 ", ".join(f"{key}={_fmt(value, 8)}" for key, value in row["closeness"].items()),
                 STATUS_LABEL.get(row["status"], row["status"])]
                for row in analysis.get("negative_controls", [])]))
    add("")
    add("非目标响应矩阵（每个干预对 13 项的平均贴近度；越大 = 越不敏感）：")
    add("")
    matrix = analysis.get("non_target_response_matrix") or {}
    metrics = sorted({key for value in matrix.values() for key in value})
    add(_table(["干预"] + metrics, [[name] + [_fmt(value.get(metric), 3) for metric in metrics]
                                    for name, value in sorted(matrix.items())]))
    add("")
    couplings = (plan.get("e3") or {}).get("declared_couplings") or {}
    if couplings:
        add("预注册的预期耦合：" + json.dumps(couplings, ensure_ascii=False))
        add("")
    add("限制：机制测试基于参数化可控图样（标注坐标与像素同源），**不代表真实插画的效度**；"
        "真实图效度由 E1/E2/E4/E5 承担。")
    add("")

    add("## 7. E4 自动定位与人工标注误差")
    add("")
    coverage = (e4.get("coverage") or {})
    add(f"分母含全部 **{coverage.get('images')}** 张（6 个可控案例 + 4 张历史图，历史失败图保留）："
        f"自动返回 {coverage.get('returned')}、可用 {coverage.get('usable')}、人工确认 {coverage.get('confirmed')}；"
        f"返回率 {_fmt(coverage.get('return_rate'), 4)}、可用率 {_fmt(coverage.get('usable_rate'), 4)}。")
    add("")
    add(_table(["图", "类型", "自动提案", "视角", "未完成维度"],
               [[row["image_id"], row["kind"], row.get("proposal") or (row.get("error") or "—")[:70],
                 row.get("pose") or "—", ", ".join(row.get("incomplete_dimensions") or []) or "—"]
                for row in (e4.get("rows") or [])]))
    add("")
    add(_table(["维度", "自动可用", "总数", "覆盖率"],
               [[metric, value["available"], value["total"], _fmt(value["coverage"], 4)]
                for metric, value in sorted((e4.get("per_dimension_coverage") or {}).items())]))
    add("")
    geometry = (e4.get("geometry_vs_synthetic_truth") or {}).get("rows") or []
    if geometry:
        add("自动眼睑曲线 vs 可控图样解析真值（相对眼宽；**只对合成案例有意义**）：")
        add("")
        rows = []
        for row in geometry:
            for name, value in sorted((row.get("eyes") or {}).items()):
                rows.append([row["image_id"], name, _fmt(value.get("mean_over_eye_width"), 4),
                             _fmt(value.get("p95_over_eye_width"), 4)])
        add(_table(["图", "曲线", "平均偏差 / 眼宽", "95 分位 / 眼宽"], rows))
        add("")
    add("人工部分：**" + str(((e4.get("awaiting_human") or {}).get("note"))) + "**")
    add("")
    add("A↔H1 / H1↔H2 / H1↔H1-repeat 的偏差与排名翻转率：**未计算（awaiting_human）**。"
        "标注入口见 `E4/human-annotation-entry.html`，schema 见 `E4/annotation-brief.json`。")
    add("")
    overlays = [row.get("overlay") for row in (e4.get("rows") or [])]
    add("自动候选叠图（未确认，仅供人工核查）：" +
        "、".join(f"`{Path(item).name}`" for item in overlays if item))
    add("")

    add("## 8. E5 真实 Gemini 候选排序与独立复核")
    add("")
    runs = ((e5.get("review") or {}).get("runs")) or {}
    add(_table(["画风", "呈现顺序", "最佳候选", "候选数", "门禁触发", "选择状态"],
               [[style, variant, value.get("best_id"), len(value.get("rows") or []),
                 ", ".join(
                     f"{row['id']}:{','.join(key for key, flag in (row.get('gates') or {}).items() if flag)}"
                     for row in (value.get("rows") or [])
                     if any((row.get("gates") or {}).values())) or "无",
                 value.get("selection_status")]
                for style, variants in sorted(runs.items())
                for variant, value in sorted(variants.items())]))
    add("")
    sensitivity = (e5.get("review") or {}).get("order_sensitivity") or {}
    add("顺序敏感性：" + json.dumps(sensitivity, ensure_ascii=False)[:800])
    add("")
    budget = e5.get("budget") or {}
    add(f"新增生成槽位：计划 **{budget.get('planned_slots')}**，已授权 **{budget.get('authorised_slots')}**，"
        f"状态 `{budget.get('status')}`。冻结参数：{json.dumps(budget.get('frozen_parameters'), ensure_ascii=False)}")
    add("")
    blind = e5.get("blind") or {}
    add(f"盲评包：`{blind.get('questionnaire')}`（题数 {blind.get('valid_independent_questions')}，"
        f"门槛 {ACCEPTANCE['candidate_ranking_assist']['min_items']}），状态 **awaiting_human**；"
        "作答模板与隐藏映射见同目录 `blind-answers-template.json` / `blind-map.json`。")
    add("")
    add("视觉模型与人类的一致率、两人分歧率、重复题一致性：**未计算（缺人类作答）**。"
        "视觉复核与自动定位同属同端点模型族，**不是独立第三方评审**。")
    add("")

    add("## 9. 尝试账本摘要")
    add("")
    add("```json")
    add(json.dumps(results.get("ledger_summary") or {}, ensure_ascii=False, indent=2))
    add("```")
    add("")
    add("计费：本轮唯一产生外部请求的是「E4 自动区域定位」（文本/视觉端点）与「E5 视觉复核」（同族端点）；"
        "两者按次计费但账户分项单价未知，统一记 `cost=unknown`、`may_have_charged=true`，"
        "且不声称符合费用上限。**没有发出任何新增生图请求**（18 槽位待额度）。")
    add("")

    add("## 10. 用途结论")
    add("")
    add(_table(["用途", "结论", "依据"],
               [[row["purpose"], row["verdict"], row["evidence"]] for row in results.get("purpose_conclusions") or []]))
    add("")

    add("## 11. 尚需人工 / 额度 / 数据的具体项目")
    add("")
    add(_table(["#", "类型", "具体缺项", "阻塞的实验"],
               [[index + 1, item["kind"], item["what"], item["blocks"]]
                for index, item in enumerate(results.get("gaps") or [])]))
    add("")

    add("## 12. 下一轮唯一优先改进与验证方案")
    add("")
    add(str(results.get("next_step", "")))
    add("")
    add("## 附：结果摘要模板（按协议 §12 填写）")
    add("")
    add("```text")
    add(str(results.get("summary_template", "")).strip())
    add("```")
    return "\n".join(lines)


def html_report(results, paths) -> str:
    plan = results.get("plan") or {}
    parts = ["<!doctype html><meta charset='utf-8'><title>画风相似度严格受控实验</title>",
             "<style>body{font-family:system-ui;margin:24px;max-width:1500px;line-height:1.55}"
             "table{border-collapse:collapse;margin:12px 0}td,th{border:1px solid #ccc;padding:6px;font-size:13px}"
             "img{max-height:300px;border:1px solid #ddd}figure{display:inline-block;margin:6px;vertical-align:top}"
             "code,pre{background:#f6f6f6;padding:2px 4px;overflow-wrap:anywhere;white-space:pre-wrap}</style>",
             "<h1>画风相似度严格受控实验</h1>",
             f"<p>协议 {results.get('protocol_version')} · 运行 <code>{results.get('run_id')}</code> · "
             f"{results.get('generated_at')} · 设备 {plan.get('device')}/{plan.get('precision')}</p>",
             "<h2>阶段状态</h2>"]
    stages = results.get("stages") or {}
    parts.append("<table><tr><th>阶段</th><th>状态</th><th>计划</th><th>实际</th><th>缺项</th><th>原因</th></tr>")
    for name in ("E0", "E1", "E2", "E3", "E4", "E5"):
        stage = stages.get(name) or {}
        parts.append("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            name, html_module.escape(str(stage.get("status"))), html_module.escape(str(stage.get("planned"))),
            html_module.escape(str(stage.get("actual"))),
            html_module.escape(", ".join(stage.get("missing_ids") or []) or "—"),
            html_module.escape(str(stage.get("reason") or ""))))
    parts.append("</table>")
    parts.append("<h2>用途结论</h2><table><tr><th>用途</th><th>结论</th><th>依据</th></tr>")
    for row in results.get("purpose_conclusions") or []:
        parts.append("<tr><td>{}</td><td><b>{}</b></td><td>{}</td></tr>".format(
            html_module.escape(row["purpose"]), html_module.escape(row["verdict"]),
            html_module.escape(row["evidence"])))
    parts.append("</table>")

    primary = ((results.get("E1") or {}).get("analysis") or {}).get(MAIN_METRIC) or {}
    confusion = primary.get("confusion") or {}
    if confusion:
        styles = sorted(confusion)
        parts.append("<h2>E1 主指标混淆矩阵（行=真实，列=预测）</h2><table><tr><th>真实\\预测</th>" +
                     "".join(f"<th>{html_module.escape(style)}</th>" for style in styles) + "</tr>")
        for style in styles:
            parts.append(f"<tr><th>{html_module.escape(style)}</th>" +
                         "".join(f"<td>{confusion[style].get(other, 0)}</td>" for other in styles) + "</tr>")
        parts.append("</table>")
    parts.append("<h2>E2 干扰变体接触表（人工核查人物结构）</h2>")
    for sheet in (results.get("E2") or {}).get("contact_sheets") or []:
        uri = _image_uri(sheet)
        if uri:
            parts.append(f"<figure><figcaption>{html_module.escape(Path(sheet).name)}</figcaption><img src='{uri}'></figure>")
    parts.append("<h2>E3 可绘图样接触表</h2>")
    for sheet in (results.get("E3") or {}).get("contact_sheets") or []:
        uri = _image_uri(sheet)
        if uri:
            parts.append(f"<figure><figcaption>{html_module.escape(Path(sheet).name)}</figcaption><img src='{uri}'></figure>")
    parts.append("<h2>E4 自动定位叠图（未确认）</h2>")
    for row in (results.get("E4") or {}).get("rows") or []:
        uri = _image_uri(row.get("overlay"))
        if uri:
            parts.append(f"<figure><figcaption>{html_module.escape(row['image_id'])} · "
                         f"{html_module.escape(row['kind'])}</figcaption><img src='{uri}'></figure>")
    parts.append("<h2>E5 历史 Gemini 产物与参考画风图</h2>")
    for entry in results.get("e5_images") or []:
        uri = _image_uri(entry.get("path"))
        if uri:
            parts.append(f"<figure><figcaption>{html_module.escape(entry.get('label', ''))}</figcaption>"
                         f"<img src='{uri}'></figure>")
    parts.append("<h2>近重复候选接触表（待人工核查）</h2>")
    for sheet in plan.get("near_duplicate_contact_sheets") or []:
        uri = _image_uri(sheet)
        if uri:
            parts.append(f"<figure><figcaption>{html_module.escape(Path(sheet).name)}</figcaption><img src='{uri}'></figure>")
    parts.append("<h2>证据与账本路径</h2><ul>")
    for item in results.get("evidence_paths") or []:
        parts.append("<li><code>" + html_module.escape(str(item)) + "</code></li>")
    parts.append("</ul>")
    return "\n".join(parts)


def write_failure_evidence(results, directory) -> list[str]:
    """把失败/拒绝/未完成逐条写成证据文件（不含任何凭据）。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    failures = []
    for row in (results.get("E4") or {}).get("rows") or []:
        if row.get("proposal") == "failed":
            failures.append({"stage": "E4", "image_id": row["image_id"], "kind": "region_proposal_failed",
                             "error_class": row.get("error_class"), "error": row.get("error"),
                             "image_sha256": row.get("sha256"), "image_path": row.get("path")})
    for name, stage in (results.get("stages") or {}).items():
        if stage.get("status") != "complete":
            failures.append({"stage": name, "kind": "stage_not_complete", "status": stage.get("status"),
                             "reason": stage.get("reason"), "missing_ids": stage.get("missing_ids")})
    for row in results.get("ledger_rows") or []:
        if row.get("status") == "failed":
            failures.append({"stage": row.get("stage"), "job_id": row.get("job_id"), "kind": "attempt_failed",
                             "attempt_id": row.get("attempt_id"), "error_class": row.get("error_class"),
                             "error": row.get("error"), "may_have_charged": row.get("may_have_charged")})
    target = directory / "failures.json"
    atomic_json(target, {"generated_at": now(), "count": len(failures), "failures": failures,
                         "note": "原始响应/回放保存在各阶段自己的目录；此处不含任何密钥。"})
    return [str(target)]


def write_reports(results, plan, run_dir) -> dict:
    run_dir = Path(run_dir)
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results["purpose_conclusions"] = purpose_conclusions(results)
    results["generated_at"] = now()
    files = write_failure_evidence(results, run_dir / "evidence")
    results["evidence_paths"] = files + [
        str(run_dir / "ledger" / "attempts.jsonl"), str(run_dir / "PLAN"),
        str(run_dir / "E2" / "contact-sheets"), str(run_dir / "E3" / "contact-sheets"),
        str(run_dir / "E4" / "overlays"), str(run_dir / "E5" / "blind"),
        str(run_dir / "E0" / "entries"), str(run_dir / "E1")]
    markdown = markdown_report(results, {"plan_files": [str(path) for path in
                                                        sorted((run_dir / "PLAN").glob("*.json"))]})
    (results_dir / "RESULTS.md").write_text(markdown, encoding="utf-8")
    (results_dir / "report.html").write_text(html_report(results, {}), encoding="utf-8")
    atomic_json(results_dir / "results.json", results)
    docs_copy = run_dir.parents[3] / "docs/style-extraction/DEEPSEEK-SIMILARITY-CONTROLLED-RESULTS-20261006.md"
    docs_copy.write_text(markdown, encoding="utf-8")
    return {"markdown": str(results_dir / "RESULTS.md"), "html": str(results_dir / "report.html"),
            "json": str(results_dir / "results.json"), "docs_markdown": str(docs_copy),
            "failure_evidence": files[0]}
