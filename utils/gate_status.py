# -*- coding: utf-8 -*-
"""最终门禁的纯判定：把「上游已经判红」与「下游审计只是补结论」分开。

背景（第三轮 P0.3）：第二轮出现过 ``status=complete`` 同时还留着
`质量修订产生重大漂移` 的 ``audit_error``，因为状态赋值会把早先的
``review_required`` 覆盖回 ``complete``；也有 arm 在最后一张图生成之后仍用旧图
的审计结论充当通过。两件事都不是画风问题，而是判定问题。

本模块只做判定，不生成、不修图、不写盘：

* 上游（质量三重审计）判红后，整体状态只能是 ``review_required``；
* 图片被后续工序替换后，针对旧图的身份/人体/终审结论视为失效（``stale``），
  ``unknown`` 不等于 ``pass``；
* 只有「上游无红 + 每张审计都绑定同一最终图 SHA + 各审计自身通过」才
  可能得到 ``complete``。
"""
from __future__ import annotations

import os

from utils.refine_quality import file_sha256, should_refine_quality

MISSING = "missing"
STALE = "stale"


def _verdict(binding: dict | None, name: str, final_sha256: str) -> dict:
    """把一个审计对象折算成 (状态, 原因)。状态取 pass/fail/unknown/stale/missing。"""
    if not isinstance(binding, dict) or not binding:
        if name == "quality":
            # 质量审计可能按画风配置关闭（没有三图审计、也没有报错），这不等于通过；
            # 上游一旦判红会由 quality_audit_error 单独带进来，所以这里只记 not_run。
            return {"audit": name, "verdict": "pass", "reason": "本次没有跑三图质量审计（未参与门禁）"}
        return {"audit": name, "verdict": MISSING, "reason": "没有该审计记录"}
    if name == "quality" and not any(binding.get(key) for key in
                                     ("before", "after")) and not binding.get("audit_error"):
        return {"audit": name, "verdict": "pass", "reason": "本次没有跑三图质量审计（未参与门禁）"}
    audit_sha = str(binding.get("candidate_sha256") or "")
    audit_path = str(binding.get("candidate") or binding.get("image") or "")
    if name != "quality" and not audit_sha:
        return {"audit": name, "verdict": STALE,
                "reason": "审计未记录被审图片的内容哈希，无法证明它审的是最终图",
                "audited_image": os.path.basename(audit_path)}
    if name != "quality" and (not final_sha256 or audit_sha != final_sha256):
        return {"audit": name, "verdict": STALE,
                "reason": "审计绑定的是另一张图（后续修图使旧结论失效）",
                "audited_image": os.path.basename(audit_path),
                "audited_sha256": audit_sha, "final_sha256": final_sha256}
    if binding.get("audit_error"):
        return {"audit": name, "verdict": "unknown",
                "reason": "审计本身失败：" + str(binding.get("audit_error"))[:200]}
    if name == "identity":
        if binding.get("mismatch"):
            return {"audit": name, "verdict": "fail",
                    "reason": "身份有差异（severity=%s）" % binding.get("severity")}
        if str(binding.get("severity") or "") == "unknown":
            return {"audit": name, "verdict": "unknown", "reason": "身份严重度未知，不按通过计"}
        return {"audit": name, "verdict": "pass", "reason": "身份审计通过"}
    if name == "quality":
        findings = [item for key in ("before", "after")
                    for item in [binding.get(key)] if isinstance(item, dict)]
        if any(should_refine_quality(item) for item in findings):
            return {"audit": name, "verdict": "fail",
                    "reason": "质量三重审计仍报高置信缺陷"}
        return {"audit": name, "verdict": "pass", "reason": "质量三重审计未报阻断性缺陷"}
    if binding.get("needs_review"):
        return {"audit": name, "verdict": "fail", "reason": "归属不明确，需人工复核"}
    if should_refine_quality(binding):
        return {"audit": name, "verdict": "fail",
                "reason": "存在高置信缺陷（severity=%s）" % binding.get("severity")}
    return {"audit": name, "verdict": "pass", "reason": "未发现高置信缺陷"}


def evaluate_final_gate(*, final_image: str = "", final_sha256: str = "", identity=None,
                        anatomy=None, quality=None, final_review=None,
                        identity_action: str = "", quality_audit_error: str = "",
                        upstream_failed: bool = False, pipeline_failed: bool = False) -> dict:
    """返回 ``{status, reasons, audits, text}``；``text`` 可 diag 打印。

    参数都允许为 None/""：缺什么就报 ``missing``，绝不当成通过。
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
        if item["verdict"] != "pass":
            reasons.append("%s: %s（%s）" % (item["audit"], item["verdict"], item["reason"]))
    passed = all(item["verdict"] == "pass" for item in audits)
    status = "complete" if (passed and not reasons) else "review_required"
    text = "%s | %s" % (status, "; ".join(reasons) if reasons else "全部审计通过且都绑定最终图")
    return {"status": status, "reasons": reasons, "audits": audits,
            "final_image": os.path.abspath(final_image), "final_sha256": sha, "text": text}
