# -*- coding: utf-8 -*-
"""最终门禁的纯判定：把「上游已经判红」「下游审计只是补结论」和「根本没有结论」分开。

背景（第三轮 P0.3）：第二轮出现过 ``status=complete`` 同时还留着
`质量修订产生重大漂移` 的 ``audit_error``，因为状态赋值会把早先的
``review_required`` 覆盖回 ``complete``；也有 arm 在最后一张图生成之后仍用旧图
的审计结论充当通过。两件事都不是画风问题，而是判定问题。

第四轮 P0.2（第三轮复核第 5 条实测复现）：**只带正确 candidate_sha256、没有任何
结论字段的审计会被判 pass** —— 绑定正确只是必要条件。缺字段、字符串 "false"、
不合法枚举、错误的集合类型、空 JSON、``unknown``、异常都必须无法通过。
判定前先验证原响应，绝不在本模块把缺字段默认成「否 / none」再当成有效结论。

本模块只做判定，不生成、不修图、不写盘：

* 上游（质量三重审计）判红后，整体状态只能是 ``review_required``；
* 图片被后续工序替换后，针对旧图的身份/人体/终审结论视为失效（``stale``），
  ``unknown`` 不等于 ``pass``；
* 只有「上游无红 + 每张审计都带完整可解释结论 + 绑定同一最终图 SHA + 各审计自身
  通过」才可能得到 ``complete``；quality 未运行时记 ``not_run``（是否参与门禁沿
  现有配置，本轮沿用「不阻断」）。
"""
from __future__ import annotations

import os

from utils.refine_quality import file_sha256, should_refine_quality, final_quality_decision, quality_failure_details

MISSING = "missing"
STALE = "stale"
INVALID = "invalid"
NOT_RUN = "not_run"

# 审计严重度的合法枚举（`normalize_*` 只允许这三个值；"unknown" 表示结论不可用）
VALID_SEVERITIES = ("none", "minor", "major")
FINDING_KEYS = ("structural_issues", "line_issues", "background_drift", "style_gaps")
# 各审计按自己的真实 schema 校验：人体审计（hand-audit-system.md）没有 severity，
# 也不要求画风字段；质量/终审（*-quality-audit-system.md）两者都要。
ANATOMY_SCHEMA = {"require_severity": False, "required_lists": ("structural_issues",)}
QUALITY_SCHEMA = {"require_severity": True, "required_lists": FINDING_KEYS}

# ``not_run`` 只允许出现在 quality 上（是否参与门禁沿现有配置）；其余三个审计
# 不允许跳过，缺记录一律按阻断处理。
NON_BLOCKING_VERDICTS = {NOT_RUN}


def _real_bool(value):
    """只接受真正的 JSON boolean；字符串 "false"/0 这类一律不算合法结论。"""
    return value if isinstance(value, bool) else None


def _severity(value) -> str:
    text = str(value or "").strip().lower()
    return text if text in VALID_SEVERITIES else ""


def _invalid(name: str, reason: str, **extra) -> dict:
    item = {"audit": name, "verdict": INVALID, "reason": reason, "conclusion_valid": False}
    item.update(extra)
    return item


def _finding_lists(binding: dict, required=FINDING_KEYS):
    """返回 (是否要求的问题集合字段都是 list, 缺失/类型错误的字段名)。"""
    missing = [key for key in required if not isinstance(binding.get(key), list)]
    return (not missing), missing


def _validate_common(name: str, binding: dict, *, require_needs_refine: bool = True):
    """结论完整性检查：返回 (错误原因, 附加字段)；无错误时错误原因为空串。

    每个审计按**自己的真实 schema** 校验（PROTOCOL：不要求人体审计带画风字段）：

    * ``anatomy``（hand-audit-system.md）：``needs_refine`` + ``structural_issues`` +
      归属结论；该审计没有 severity 字段，所以不要求，但**如果给了就必须是合法枚举**。
    * ``final_review`` / ``quality``：``needs_refine`` + 合法 severity + 四个问题集合。
    """
    schema = ANATOMY_SCHEMA if name == "anatomy" else QUALITY_SCHEMA
    if require_needs_refine:
        if _real_bool(binding.get("needs_refine")) is None:
            return ("结论缺少合法 boolean needs_refine（不能把缺字段当成「不需要修订」）", {})
    severity_present = str(binding.get("severity") or "").strip() != ""
    severity = _severity(binding.get("severity"))
    if (schema["require_severity"] or severity_present) and not severity:
        return ("结论缺少合法严重度（合法值只有 %s）" % "/".join(VALID_SEVERITIES),
                {"severity_raw": str(binding.get("severity"))[:60]})
    ok, missing = _finding_lists(binding, schema["required_lists"])
    if not ok:
        return ("结论缺少可解释的问题集合：" + ", ".join(missing), {"missing_fields": missing})
    extra = {"conclusion_valid": True}
    if severity:
        extra["severity"] = severity
    else:
        # anatomy 的严重度由 needs_refine 承载（该审计 schema 没有 severity 字段）
        extra["severity"] = "major" if binding.get("needs_refine") else "none"
        extra["severity_source"] = "needs_refine（该审计 schema 无 severity 字段）"
    return ("", extra)


def _verdict(binding: dict | None, name: str, final_sha256: str) -> dict:
    """把一个审计对象折算成 (状态, 原因)。状态取 pass/fail/unknown/stale/missing/invalid/not_run。"""
    if not isinstance(binding, dict) or not binding:
        if name == "quality":
            # 质量审计可能按画风配置关闭（没有三图审计、也没有报错）。这不等于通过，
            # 也不等于失败：记 not_run，是否参与门禁沿现有配置（本轮沿用不阻断）。
            return {"audit": name, "verdict": NOT_RUN, "conclusion_valid": False,
                    "reason": "本次没有跑三图质量审计（未参与门禁）"}
        return {"audit": name, "verdict": MISSING, "conclusion_valid": False,
                "reason": "没有该审计记录"}
    if name == "quality" and not any(binding.get(key) for key in
                                     ("before", "after")) and not binding.get("audit_error"):
        return {"audit": name, "verdict": NOT_RUN, "conclusion_valid": False,
                "reason": "本次没有跑三图质量审计（未参与门禁）"}
    audit_sha = str(binding.get("candidate_sha256") or "")
    audit_path = str(binding.get("candidate") or binding.get("image") or "")
    if name != "quality" and binding.get("audit_error"):
        # 审计本身失败的记录往往连绑定哈希都没有；先报「审计失败」比先报「绑定不明」更可操作，
        # 两者都不能判 pass。
        return {"audit": name, "verdict": "unknown", "conclusion_valid": False,
                "reason": "审计本身失败：" + str(binding.get("audit_error"))[:200]}
    if name != "quality" and not audit_sha:
        return {"audit": name, "verdict": STALE, "conclusion_valid": False,
                "reason": "审计未记录被审图片的内容哈希，无法证明它审的是最终图",
                "audited_image": os.path.basename(audit_path)}
    if name != "quality" and (not final_sha256 or audit_sha != final_sha256):
        return {"audit": name, "verdict": STALE, "conclusion_valid": False,
                "reason": "审计绑定的是另一张图（后续修图使旧结论失效）",
                "audited_image": os.path.basename(audit_path),
                "audited_sha256": audit_sha, "final_sha256": final_sha256}
    if binding.get("audit_error"):
        return {"audit": name, "verdict": "unknown", "conclusion_valid": False,
                "reason": "审计本身失败：" + str(binding.get("audit_error"))[:200]}
    if name == "identity":
        # 绑定正确之后还要有真正的结论：mismatch 必须是 JSON boolean、严重度必须在
        # 枚举里、差异集合必须是 list；说不一致就必须给出可解释的差异条目。
        mismatch = _real_bool(binding.get("mismatch"))
        if mismatch is None:
            return _invalid(name, "结论缺少合法 boolean mismatch（字符串 \"false\" 或缺失都不算结论）",
                            mismatch_raw=str(binding.get("mismatch"))[:60])
        severity = _severity(binding.get("severity"))
        if not severity:
            return _invalid(name, "结论缺少合法严重度（合法值只有 %s）" % "/".join(VALID_SEVERITIES),
                            severity_raw=str(binding.get("severity"))[:60])
        if not isinstance(binding.get("differences"), list):
            return _invalid(name, "结论缺少合法 differences 列表")
        differences = list(binding.get("differences") or [])
        if mismatch and not differences:
            return _invalid(name, "结论声明不一致，却没有可解释的差异条目", severity=severity)
        if not mismatch and severity != "none":
            return _invalid(name, "结论自相矛盾：mismatch=false 却给出 severity=%s" % severity,
                            severity=severity)
        if binding.get("conclusion_missing") or binding.get("conclusion_valid") is False:
            return _invalid(name, "原响应缺少结论字段，规范化不得把它默认成「通过」", severity=severity)
        if mismatch:
            return {"audit": name, "verdict": "fail", "conclusion_valid": True,
                    "reason": "身份有差异（severity=%s）" % severity}
        return {"audit": name, "verdict": "pass", "conclusion_valid": True, "reason": "身份审计通过"}
    if name == "quality":
        # The before-audit is repair history, not evidence against a repaired image.
        # Match the GUI: minor local findings continue to anatomy/final review.
        latest = binding.get("after") if isinstance(binding.get("after"), dict) else binding.get("before") or {}
        if not isinstance(latest, dict) or not latest:
            return _invalid(name, "质量审计记录里没有可用的 before/after 结论")
        error, extra = _validate_common(name, latest)
        if error:
            return _invalid(name, error, **extra)
        drift = any(float(item.get("confidence") or 0) >= .72
                    for item in latest.get("background_drift") or [])
        if latest.get("audit_error") or latest.get("needs_review"):
            return {"audit": name, "verdict": "unknown", "conclusion_valid": False,
                    "reason": quality_failure_details(latest)}
        if should_refine_quality(latest) and (latest.get("severity") != "minor" or drift):
            return {"audit": name, "verdict": "fail", "conclusion_valid": True,
                    "reason": "质量三重审计仍报高置信缺陷\n" + quality_failure_details(latest)}
        return {"audit": name, "verdict": "pass", "conclusion_valid": True,
                "reason": "最新质量审计无阻断性缺陷；局部问题由人体与最终复核判定"}
    # anatomy / final_review：必须有合法 needs_refine + 可解释严重度 + 问题集合，
    # 并且要有归属结论（ownership_uncertain 列表或 needs_review 布尔，二者之一）。
    error, extra = _validate_common(name, binding)
    if error:
        return _invalid(name, error, **extra)
    if not isinstance(binding.get("ownership_uncertain"), list) and _real_bool(binding.get("needs_review")) is None:
        return _invalid(name, "结论缺少归属结论（ownership_uncertain 列表与 needs_review 布尔都没有）")
    if isinstance(binding.get("ownership_uncertain"), list) and binding.get("ownership_uncertain") \
            and not binding.get("needs_review"):
        return _invalid(name, "归属结论不完整：有归属待确认条目却不是 needs_review=true")
    if _real_bool(binding.get("needs_review")) and binding.get("needs_review"):
        return {"audit": name, "verdict": "fail", "conclusion_valid": True, "reason": "归属不明确，需人工复核"}
    if name == "final_review":
        decision = final_quality_decision(binding)
        return {"audit": name, "verdict": "pass" if decision["action"] in
                {"accept", "accept_with_warning"} else "fail",
                "conclusion_valid": True, "severity": extra.get("severity"),
                "warning": decision["action"] == "accept_with_warning",
                "reason": decision["reason"] + ("\n" + quality_failure_details(binding)
                                               if decision["action"] != "accept" else "")}
    if should_refine_quality(binding):
        return {"audit": name, "verdict": "fail", "conclusion_valid": True,
                "severity": extra.get("severity"), "severity_source": extra.get("severity_source"),
                "reason": "存在高置信缺陷（severity=%s）" % extra.get("severity")}
    return {"audit": name, "verdict": "pass", "conclusion_valid": True,
            "severity": extra.get("severity"), "severity_source": extra.get("severity_source"),
            "reason": "未发现高置信缺陷"}


def evaluate_final_gate(*, final_image: str = "", final_sha256: str = "", identity=None,
                        anatomy=None, quality=None, final_review=None,
                        identity_action: str = "", quality_audit_error: str = "",
                        upstream_failed: bool = False, pipeline_failed: bool = False) -> dict:
    """返回 ``{status, reasons, audits, text}``；``text`` 可 diag 打印。

    参数都允许为 None/""：缺什么就报 ``missing``/``invalid``，绝不当成通过。
    """
    sha = str(final_sha256 or "")
    if not sha and final_image:
        sha = file_sha256(final_image)
    reasons: list[str] = []
    if pipeline_failed or not str(final_image or "") or not sha:
        return {"status": "pipeline_failed", "reasons": ["没有可用的最终图"],
                "audits": [], "final_image": str(final_image or ""), "final_sha256": sha,
                "text": "pipeline_failed"}
    if upstream_failed or quality_audit_error:
        reasons.append("上游质量门禁已判红：" + (str(quality_audit_error)[:200] or "upstream_failed"))
    if identity_action and identity_action != "accept":
        reasons.append("身份门禁动作为 %s（非 accept）" % identity_action)
    audits = [_verdict(quality, "quality", sha), _verdict(identity, "identity", sha),
              _verdict(anatomy, "anatomy", sha), _verdict(final_review, "final_review", sha)]
    for item in audits:
        if item["verdict"] != "pass" and item["verdict"] not in NON_BLOCKING_VERDICTS:
            reasons.append("%s: %s（%s）" % (item["audit"], item["verdict"], item["reason"]))
    passed = all(item["verdict"] == "pass" or item["verdict"] in NON_BLOCKING_VERDICTS
                 for item in audits)
    status = "complete" if (passed and not reasons) else "review_required"
    text = "%s | %s" % (status, "; ".join(reasons) if reasons else "全部审计通过且都绑定最终图")
    return {"status": status, "reasons": reasons, "audits": audits,
            "final_image": os.path.abspath(final_image), "final_sha256": sha, "text": text}
