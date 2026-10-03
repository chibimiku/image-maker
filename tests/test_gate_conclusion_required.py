# -*- coding: utf-8 -*-
"""第四轮 P0.2 回归：**缺结论不能通过**，且规范化入口不得把缺字段伪装成有效结论。

第三轮复核第 5 条实测复现：给 ``evaluate_final_gate`` 的 identity/anatomy/final_review
各只带一个正确的 ``candidate_sha256``，门禁返回 ``complete``。绑定正确只是必要条件；
结论本身必须存在、类型合法、并且和绑定一致。

本文件全程离线，不联网、不调用任何图片接口。
"""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

EMPTY = {"structural_issues": [], "line_issues": [], "background_drift": [], "style_gaps": []}


def _image(path, color="white", size=(64, 64)):
    Image.new("RGB", size, color).save(path)
    return str(path)


def clean_quality():
    return {"before": {"needs_refine": False, "severity": "none", "ownership_uncertain": [],
                       "needs_review": False, **EMPTY}}


def clean_identity(sha):
    return {"mismatch": False, "severity": "none", "differences": [], "confidence": .9,
            "stable_anchors": [], "candidate_sha256": sha}


def clean_anatomy(sha):
    return {"needs_refine": False, "severity": "none", "ownership_uncertain": [],
            "needs_review": False, "candidate_sha256": sha, **EMPTY}


def clean_final(sha):
    return {"needs_refine": False, "severity": "none", "ownership_uncertain": [],
            "needs_review": False, "candidate_sha256": sha, **EMPTY}


class ShaOnlyAuditTests(unittest.TestCase):
    """第三轮复核第 5 条：只有 SHA 的审计不是结论。"""

    def test_sha_only_audits_cannot_reach_complete(self):
        from utils.gate_status import evaluate_final_gate
        from utils.refine_quality import file_sha256
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png")
            sha = file_sha256(final)          # 绑定完全正确
            gate = evaluate_final_gate(
                final_image=final,
                identity={"candidate_sha256": sha},
                anatomy={"candidate_sha256": sha},
                final_review={"candidate_sha256": sha},
                quality={})
            self.assertEqual(gate["status"], "review_required")
            verdicts = {item["audit"]: item["verdict"] for item in gate["audits"]}
            self.assertEqual(verdicts["identity"], "invalid")
            self.assertEqual(verdicts["anatomy"], "invalid")
            self.assertEqual(verdicts["final_review"], "invalid")
            self.assertEqual(verdicts["quality"], "not_run")  # 未跑质量审计只记 not_run
            self.assertNotIn("complete", gate["text"])

    def test_sha_only_with_the_right_hash_still_fails(self):
        """哈希完全正确也不够：结论字段缺失同样拦截。"""
        from utils.gate_status import evaluate_final_gate
        from utils.refine_quality import file_sha256
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png")
            sha = file_sha256(final)
            gate = evaluate_final_gate(
                final_image=final, identity={"candidate_sha256": sha},
                anatomy={"candidate_sha256": sha}, final_review={"candidate_sha256": sha},
                quality={})
            self.assertEqual(gate["status"], "review_required")


class MalformedConclusionTests(unittest.TestCase):
    """字符串 "false"、不合法枚举、错误集合类型、空 JSON、unknown、异常都不算通过。"""

    def _gate(self, final, **overrides):
        from utils.gate_status import evaluate_final_gate
        from utils.refine_quality import file_sha256
        sha = file_sha256(final)
        payload = {"identity": clean_identity(sha), "anatomy": clean_anatomy(sha),
                   "final_review": clean_final(sha), "quality": clean_quality()}
        payload.update(overrides)
        payload["final_image"] = final
        return evaluate_final_gate(**payload)

    def test_string_false_mismatch_is_not_a_conclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png")
            from utils.refine_quality import file_sha256
            sha = file_sha256(final)
            gate = self._gate(final, identity={"mismatch": "false", "severity": "none",
                                               "differences": [], "candidate_sha256": sha})
            self.assertEqual(gate["status"], "review_required")
            self.assertEqual({i["audit"]: i["verdict"] for i in gate["audits"]}["identity"], "invalid")

    def test_invalid_severity_enum_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png")
            from utils.refine_quality import file_sha256
            sha = file_sha256(final)
            bad = dict(clean_anatomy(sha))
            bad["severity"] = "catastrophic"
            gate = self._gate(final, anatomy=bad)
            self.assertEqual(gate["status"], "review_required")

    def test_unknown_severity_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png")
            from utils.refine_quality import file_sha256
            sha = file_sha256(final)
            bad = dict(clean_final(sha))
            bad["severity"] = "unknown"
            gate = self._gate(final, final_review=bad)
            self.assertEqual(gate["status"], "review_required")

    def test_wrong_collection_type_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png")
            from utils.refine_quality import file_sha256
            sha = file_sha256(final)
            bad = dict(clean_anatomy(sha))
            bad["structural_issues"] = "none"      # 字符串不是集合
            gate = self._gate(final, anatomy=bad)
            self.assertEqual(gate["status"], "review_required")

    def test_convicted_mismatch_without_differences_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png")
            from utils.refine_quality import file_sha256
            sha = file_sha256(final)
            bad = dict(clean_identity(sha))
            bad["mismatch"] = True                 # 说不一致却没有差异条目
            gate = self._gate(final, identity=bad)
            self.assertEqual(gate["status"], "review_required")

    def test_audit_error_and_unknown_still_block(self):
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png")
            gate = self._gate(final, anatomy={"audit_error": "RuntimeError: 503"})
            self.assertEqual(gate["status"], "review_required")
            self.assertEqual({i["audit"]: i["verdict"] for i in gate["audits"]}["anatomy"], "unknown")

    def test_empty_json_audit_cannot_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png")
            gate = self._gate(final, identity={}, anatomy={}, final_review={})
            self.assertEqual(gate["status"], "review_required")
            verdicts = {i["audit"]: i["verdict"] for i in gate["audits"]}
            self.assertEqual(verdicts["identity"], "missing")
            self.assertEqual(verdicts["anatomy"], "missing")
            self.assertEqual(verdicts["final_review"], "missing")


class NormalizerMustNotFabricateTests(unittest.TestCase):
    """规范化入口先验证原响应，禁止把缺字段默认成「否 / none」。"""

    def test_identity_normalizer_marks_missing_fields(self):
        from utils.identity_audit import normalize_audit
        for raw, expected_severity in (
                ({}, "unknown"),
                ({"mismatch": False}, "unknown"),
                ({"mismatch": "false", "severity": "none"}, "none"),  # 字符串不是 boolean
                ({"mismatch": False, "severity": "severe"}, "unknown"),
                ({"mismatch": False, "severity": "none", "differences": "none"}, "none")):
            with self.subTest(raw=raw):
                audit = normalize_audit(raw)
                self.assertFalse(audit["conclusion_valid"])
                self.assertTrue(audit["conclusion_missing"])
                self.assertTrue(audit["missing_conclusion_fields"])
                self.assertEqual(audit["severity"], expected_severity)

    def test_identity_normalizer_keeps_a_complete_verdict(self):
        from utils.identity_audit import normalize_audit
        audit = normalize_audit({"mismatch": False, "severity": "none", "differences": []})
        self.assertTrue(audit["conclusion_valid"])
        self.assertEqual(audit["severity"], "none")

    def test_quality_normalizer_marks_missing_fields(self):
        from utils.refine_quality import normalize_quality_audit
        for raw in ({}, {"needs_refine": False},
                    {"needs_refine": False, "severity": "none"},   # 缺四个集合
                    {"needs_refine": False, "severity": "n/a", "structural_issues": [],
                     "line_issues": [], "background_drift": [], "style_gaps": []},
                    {"needs_refine": "false", "severity": "none", "structural_issues": [],
                     "line_issues": [], "background_drift": [], "style_gaps": []}):
            with self.subTest(raw=raw):
                audit = normalize_quality_audit(raw)
                self.assertFalse(audit["conclusion_valid"])
                self.assertTrue(audit["conclusion_missing"])

    def test_quality_normalizer_keeps_a_complete_verdict(self):
        from utils.refine_quality import normalize_quality_audit
        audit = normalize_quality_audit({"needs_refine": False, "severity": "none", **EMPTY})
        self.assertTrue(audit["conclusion_valid"])
        self.assertEqual(audit["severity"], "none")

    def test_refused_model_response_becomes_invalid_not_clean(self):
        """模型回拒绝话术（空 JSON）时，审计结论必须是「无效」而不是「清白的」。"""
        import utils.refine_quality as quality
        with tempfile.TemporaryDirectory() as folder:
            candidate = _image(Path(folder) / "candidate.png")
            original = quality.call_text_model
            quality.call_text_model = lambda *a, **k: "抱歉，我无法分析这张图片。"
            try:
                with self.assertRaises(ValueError):
                    quality.audit_refine_quality(candidate, candidate, candidate, final_review=True,
                                                 text_cfg={"base_url": "t", "api_key": "", "model": "t"})
            finally:
                quality.call_text_model = original


class PolicyCompatibilityTests(unittest.TestCase):
    """今天已有的质量政策必须保留：minor 放行、after 优先、结构/背景与上游失败拦截。"""

    def _gate(self, final, **overrides):
        from utils.gate_status import evaluate_final_gate
        from utils.refine_quality import file_sha256
        sha = file_sha256(final)
        payload = {"identity": clean_identity(sha), "anatomy": clean_anatomy(sha),
                   "final_review": clean_final(sha), "quality": clean_quality()}
        payload.update(overrides)
        payload["final_image"] = final
        return evaluate_final_gate(**payload)

    def test_explicit_clean_pass_is_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png")
            gate = self._gate(final)
            self.assertEqual(gate["status"], "complete", gate["text"])
            self.assertTrue(all(item["verdict"] == "pass" for item in gate["audits"]))

    def test_minor_line_and_style_warning_still_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png")
            from utils.refine_quality import file_sha256
            sha = file_sha256(final)
            minor = {"needs_refine": True, "severity": "minor", "ownership_uncertain": [],
                     "needs_review": False, "candidate_sha256": sha, **EMPTY,
                     "style_gaps": [{"aspect": "line weight", "candidate": "pale",
                                     "target": "firmer", "repair": "thicken", "confidence": .9}]}
            gate = self._gate(final, final_review=minor)
            self.assertEqual(gate["status"], "complete")
            self.assertTrue({i["audit"]: i.get("warning", False) for i in gate["audits"]}["final_review"])

    def test_major_structural_finding_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png")
            from utils.refine_quality import file_sha256
            sha = file_sha256(final)
            major = {"needs_refine": True, "severity": "major", "ownership_uncertain": [],
                     "needs_review": False, "candidate_sha256": sha, **EMPTY,
                     "structural_issues": [{"region": "arm", "observed": "extra limb",
                                            "repair": "remove", "confidence": .95}]}
            gate = self._gate(final, final_review=major)
            self.assertEqual(gate["status"], "review_required")

    def test_background_drift_in_quality_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png")
            drift = {"needs_refine": True, "severity": "minor", "ownership_uncertain": [],
                     "needs_review": False, **EMPTY,
                     "background_drift": [{"region": "wall", "original": "yellow",
                                           "candidate": "white", "repair": "restore",
                                           "confidence": .9}]}
            gate = self._gate(final, quality={"after": drift})
            self.assertEqual(gate["status"], "review_required")

    def test_after_wins_over_a_dirty_before(self):
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png")
            dirty = {"needs_refine": True, "severity": "major", "ownership_uncertain": [],
                     "needs_review": False, **EMPTY,
                     "structural_issues": [{"region": "hand", "observed": "six fingers",
                                            "repair": "remove", "confidence": .97}]}
            gate = self._gate(final, quality={"before": dirty, "after": {
                "needs_refine": False, "severity": "none", "ownership_uncertain": [],
                "needs_review": False, **EMPTY}})
            self.assertEqual(gate["status"], "complete")

    def test_upstream_failure_is_kept(self):
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png")
            gate = self._gate(final, upstream_failed=True, quality_audit_error="major drift")
            self.assertEqual(gate["status"], "review_required")
            self.assertTrue(any("上游质量门禁已判红" in r for r in gate["reasons"]))

    def test_identity_action_other_than_accept_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png")
            gate = self._gate(final, identity_action="correct")
            self.assertEqual(gate["status"], "review_required")


class AuditRecordShapeTests(unittest.TestCase):
    """真实审计产物的形态必须与门禁要求一致（避免把新检查写成对真实输出永远失败）。"""

    def test_real_hand_audit_shape_satisfies_the_gate(self):
        """hand-audit-system.md 的真实 schema：没有 severity，也不含画风字段。"""
        import utils.refine_quality as quality
        with tempfile.TemporaryDirectory() as folder:
            candidate = _image(Path(folder) / "candidate.png")
            original = quality.call_text_model
            quality.call_text_model = lambda *a, **k: json.dumps(
                {"needs_refine": False, "structural_issues": [],
                 "character_limb_inventory": [{"owner": "one girl", "arms": "ok", "legs": "ok"}],
                 "ownership_uncertain": [], "summary": "clean"})
            try:
                audit = quality.audit_hand_quality(candidate,
                                                   text_cfg={"base_url": "t", "api_key": "", "model": "t"})
            finally:
                quality.call_text_model = original
            self.assertTrue(audit["conclusion_valid"])
            self.assertEqual(audit["severity"], "")
            self.assertEqual(audit["missing_conclusion_fields"], [])
            from utils.gate_status import evaluate_final_gate
            from utils.refine_quality import file_sha256
            gate = evaluate_final_gate(
                final_image=candidate, identity=clean_identity(file_sha256(candidate)),
                anatomy=audit, quality={}, final_review=clean_final(file_sha256(candidate)))
            self.assertEqual(gate["status"], "complete", gate["text"])
            anatomy_verdict = {item["audit"]: item for item in gate["audits"]}["anatomy"]
            self.assertEqual(anatomy_verdict["verdict"], "pass")
            self.assertIn("needs_refine", anatomy_verdict.get("severity_source", ""))

    def test_anatomy_without_findings_collection_is_invalid(self):
        """连 structural_issues 都没有的人体审计，不是「人体没问题」的结论。"""
        from utils.gate_status import evaluate_final_gate
        from utils.refine_quality import file_sha256
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png")
            sha = file_sha256(final)
            gate = evaluate_final_gate(
                final_image=final, identity=clean_identity(sha),
                anatomy={"needs_refine": False, "ownership_uncertain": [], "needs_review": False,
                         "candidate_sha256": sha},
                quality={}, final_review=clean_final(sha))
            self.assertEqual(gate["status"], "review_required")
            self.assertEqual({item["audit"]: item["verdict"] for item in gate["audits"]}["anatomy"],
                             "invalid")


if __name__ == "__main__":
    unittest.main()
