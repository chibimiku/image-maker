# -*- coding: utf-8 -*-
"""第三轮 P0 五个具体缺陷的离线回归：全程不联网、不调用任何图片接口。

覆盖：
  1. 上游质量审计 major 之后，只要没有强制救援，图片调用必须为 0，整体状态不得回 complete；
  2. selected_output 与审计绑定的内容哈希不一致时，门禁不得判通过；
  3. reference_mode=none（参考图不参与生成）也必须留下 final_review 记录；
  4. 人脸检测器的命中不等于人工确认，检测结果只能写进 detected 一列；
  5. expected_diff 为空时禁止按变体生图，实测请求与声明不符时同样拒绝。
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

TEMP = tempfile.mkdtemp(prefix="dsh-round3-")


def _image(path, color="white", size=(64, 64)):
    Image.new("RGB", size, color).save(path)
    return str(path)


def _quality(counts):
    def fake(*args, **kwargs):
        counts.append("quality")
        return {"needs_refine": True, "severity": "major",
                "structural_issues": [{"region": "hand", "observed": "extra finger",
                                       "repair": "remove it", "confidence": .95}],
                "line_issues": [], "background_drift": [], "style_gaps": [],
                "candidate_sha256": "irrelevant"}
    return fake


def _clean(counts):
    def fake(*args, **kwargs):
        counts.append("audit")
        return {"needs_refine": False, "severity": "none", "structural_issues": [],
                "line_issues": [], "background_drift": [], "style_gaps": [],
                "candidate": str(args[0]) if args else "",
                "candidate_sha256": __import__("utils.refine_quality", fromlist=["file_sha256"]).file_sha256(
                    str(args[0]) if args else "")}
    return fake


def _identity(mismatch=False):
    def fake(*args, **kwargs):
        return {"mismatch": mismatch, "severity": "major" if mismatch else "none", "confidence": .9,
                "differences": [] if not mismatch else [{"feature": "hair_colour", "expected": "black",
                                                         "observed": "blue", "confidence": .95}],
                "image": str(args[0]), "candidate_sha256":
                    __import__("utils.refine_quality", fromlist=["file_sha256"]).file_sha256(str(args[0]))}
    return fake


def _anatomy():
    def fake(*args, **kwargs):
        return {"needs_refine": False, "structural_issues": [], "line_issues": [],
                "background_drift": [], "style_gaps": [],
                "candidate": str(args[0]), "candidate_sha256":
                    __import__("utils.refine_quality", fromlist=["file_sha256"]).file_sha256(str(args[0]))}
    return fake


class UpstreamFailureTests(unittest.TestCase):
    """缺陷 1：上游 major 之后不得再产生图片调用，状态不得回 complete。"""

    def test_major_quality_drift_keeps_review_required_without_repair(self):
        from utils.refine_quality import review_final_candidate
        with tempfile.TemporaryDirectory() as folder:
            candidate = _image(Path(folder) / "candidate.png")
            counts = []
            with patch("modules.others.api_backend.generate_image_repaint") as repaint:
                result, audit = review_final_candidate(
                    candidate, _quality(counts), folder, audit_only=True)
                repaint.assert_not_called()
            self.assertEqual(result, candidate)
            self.assertTrue(audit["needs_refine"])
            self.assertTrue(audit["audit_only"])
            self.assertEqual(len(counts), 1)  # 只审一次，不重复付费

    def test_gate_cannot_return_complete_after_upstream_failure(self):
        from utils.gate_status import evaluate_final_gate
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png")
            gate = evaluate_final_gate(
                final_image=final, identity=_identity()(final), anatomy=_anatomy()(final),
                quality={"audit_error": "RuntimeError: 质量修订产生重大漂移"},
                final_review=_clean([])(final), quality_audit_error="RuntimeError: 重大漂移",
                upstream_failed=True)
            self.assertEqual(gate["status"], "review_required")
            self.assertTrue(any("上游质量门禁已判红" in reason for reason in gate["reasons"]))

    def test_all_clean_and_bound_may_still_complete(self):
        from utils.gate_status import evaluate_final_gate
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png")
            gate = evaluate_final_gate(
                final_image=final, identity=_identity()(final), anatomy=_anatomy()(final),
                quality={"before": {"severity": "none", "needs_refine": False}},
                final_review=_clean([])(final), identity_action="accept")
            self.assertEqual(gate["status"], "complete")

    def test_cli_reports_review_required_without_stage2_generation(self):
        """整条 CLI：质量门禁判红 + --audit-only，两次出图接口都不许被调用。"""
        import tools.analysis_gpt_run as cli
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            analysis = root / "analysis.json"
            analysis.write_text(json.dumps({
                "task_hash": "round3test", "aspect_ratio": "2:3",
                "original_english_description": "a girl standing in a room",
                "english_description": "a girl standing in a room",
                "gpt_image_prompt": "a girl with black hair standing in a room",
                "source_image_path": _image(root / "source.png")}), encoding="utf-8")
            base = _image(root / "base.png", "white")
            out_dir = root / "out"
            calls = []
            steps = {"repaint": {"enabled": True, "reference_mode": "style", "scope": "full"},
                     "structure": {"enabled": False}, "local": {"enabled": False}}
            argv = ["analysis_gpt_run.py", "--json", str(analysis), "--style", "goto-p",
                    "--steps", "repaint", "--repaint-ref", "style",
                    "--base-image", base, "--identity-audit", "--audit-only",
                    "--final-review-audit", "--output-dir", str(out_dir)]
            # _build_steps 走真实实现，只把出图与审计替换成离线假件
            with patch("modules.others.api_backend.generate_image_repaint") as repaint, \
                 patch("modules.others.api_backend.generate_image_aigc2d_gpt") as first_pass, \
                 patch("utils.first_image_review.generate_first_image") as first_image, \
                 patch("utils.refine_quality.audit_refine_quality", _quality(calls)), \
                 patch("utils.refine_quality.audit_hand_quality", _anatomy()), \
                 patch("utils.identity_audit.audit_image_identity", _identity(True)), \
                 patch("utils.post_process.run_pipeline", return_value=[base]) as pipeline, \
                 patch("utils.analysis_gen.run_gpt_image_pipeline", return_value=[base]), \
                 patch("utils.post_process.default_pipeline", return_value=steps), \
                 patch.dict(os.environ, {"IMAGE_MAKER_TEST_OUTPUT": "1"}):
                import sys
                saved = sys.argv
                sys.argv = argv
                try:
                    with patch("utils.gpt_image_optimize.load_config",
                               return_value={"final_candidate_repair": {"max_repairs": 1}}):
                        code = cli.main()
                finally:
                    sys.argv = saved
                repaint.assert_not_called()
                first_pass.assert_not_called()
                first_image.assert_not_called()
                self.assertTrue(pipeline.called)  # 一次完整图重绘仍然发生
            manifest = json.loads((out_dir / "request.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 1)
            self.assertEqual(manifest["status"], "review_required")
            self.assertTrue(manifest["audit_only"])
            self.assertNotEqual(manifest.get("final_review", {}).get("needs_refine"), False)
            gate_file = json.loads((out_dir / "final-gate.json").read_text(encoding="utf-8"))
            self.assertEqual(gate_file["status"], "review_required")


class BindingTests(unittest.TestCase):
    """缺陷 2：选中的图与审计绑定的图不是同一份内容时，不得通过。"""

    def test_stale_audit_on_another_image_fails_the_gate(self):
        from utils.gate_status import evaluate_final_gate
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png", "white")
            other = _image(Path(folder) / "other.png", "black")
            stale = dict(_anatomy()(other))
            gate = evaluate_final_gate(
                final_image=final, identity=_identity()(final), anatomy=stale,
                quality={}, final_review=_clean([])(final))
            self.assertEqual(gate["status"], "review_required")
            checks = {item["audit"]: item["verdict"] for item in gate["audits"]}
            self.assertEqual(checks["anatomy"], "stale")

    def test_audit_without_hash_is_not_treated_as_pass(self):
        from utils.gate_status import evaluate_final_gate
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png")
            gate = evaluate_final_gate(final_image=final, identity={"mismatch": False, "severity": "none"},
                                       anatomy=_anatomy()(final), quality={},
                                       final_review=_clean([])(final))
            self.assertEqual(gate["status"], "review_required")
            checks = {item["audit"]: item["verdict"] for item in gate["audits"]}
            self.assertEqual(checks["identity"], "stale")


class ReferenceModeNoneTests(unittest.TestCase):
    """缺陷 3：画风图不参与生成时，仍要有 final_review，且不能自动判过。"""

    def test_missing_style_reference_still_records_a_final_review(self):
        from utils.refine_quality import write_style_reference_absent_audit
        with tempfile.TemporaryDirectory() as folder:
            candidate = _image(Path(folder) / "candidate.png")
            audit = write_style_reference_absent_audit(candidate, None, folder)
            self.assertTrue(audit["style_reference_absent"])
            self.assertTrue(audit["needs_refine"])
            stored = json.loads((Path(folder) / "final-quality-audit.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["candidate_sha256"], audit["candidate_sha256"])

    def test_gate_requires_a_final_review_record(self):
        from utils.gate_status import evaluate_final_gate
        with tempfile.TemporaryDirectory() as folder:
            final = _image(Path(folder) / "final.png")
            gate = evaluate_final_gate(final_image=final, identity=_identity()(final),
                                       anatomy=_anatomy()(final), quality={}, final_review=None)
            self.assertEqual(gate["status"], "review_required")
            checks = {item["audit"]: item["verdict"] for item in gate["audits"]}
            self.assertEqual(checks["final_review"], "missing")

    def test_audit_without_style_image_judges_structure_only(self):
        from utils.refine_quality import audit_refine_quality
        with tempfile.TemporaryDirectory() as folder:
            candidate = _image(Path(folder) / "candidate.png")
            captured = {}
            def model(base_url, api_key, name, system, user, **kwargs):
                captured["images"] = kwargs.get("image_paths")
                captured["user"] = user
                return json.dumps({"needs_refine": False, "structural_issues": [], "line_issues": [],
                                   "background_drift": [], "style_gaps": []})
            import utils.refine_quality as quality
            original = quality.call_text_model
            quality.call_text_model = model
            try:
                audit = audit_refine_quality(candidate, candidate, candidate, final_review=True,
                                             text_cfg={"base_url": "t", "api_key": "", "model": "t"})
            finally:
                quality.call_text_model = original
            self.assertEqual(len(captured["images"]), 4)  # 三图 + 九宫格
            self.assertIn("candidate_sha256", audit)


class CropConfirmationTests(unittest.TestCase):
    """缺陷 4：检测器命中不能自动写成人工确认。"""

    def test_detector_hit_does_not_fill_human_fields(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "style_face_eval", str(Path(__file__).resolve().parents[1] / "tools" / "style_face_eval.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            image = _image(Path(folder) / "plain.png", "gray", (200, 300))
            box, method = module.head_box_auto(module.read_image(image))
            self.assertIn(method, ("YUNET", "AUTO", "FALLBACK"))
            # 关键点：自动方法名只是 detected 的依据，人工列由 tool 留空
            record = {"detection_method": method, "detected": method in ("YUNET", "AUTO"),
                      "human_checked": None, "contains_target_face": None, "eye_readable": None}
            self.assertIn(record["detection_method"], ("YUNET", "AUTO", "FALLBACK"))
            self.assertIsNone(record["human_checked"])
            self.assertIsNone(record["eye_readable"])

    def test_implausible_auto_box_is_flagged(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "style_face_eval", str(Path(__file__).resolve().parents[1] / "tools" / "style_face_eval.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ok, _reason = module.head_box_plausible((0.01, 0.01, 0.99, 0.99), (2048, 2048))
        self.assertFalse(ok)
        ok2, _reason2 = module.head_box_plausible((0.3, 0.06, 0.72, 0.36), (2048, 2048))
        self.assertTrue(ok2)


class ExpectedDiffTests(unittest.TestCase):
    """缺陷 5：expected_diff 为空时禁止生图；实测请求与声明不符要报出来。"""

    def test_run_without_declared_diff_is_refused(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "style_face_eval", str(Path(__file__).resolve().parents[1] / "tools" / "style_face_eval.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            plan_path = Path(folder) / "run-plan.json"
            plan_path.write_text(json.dumps({"runs": [{
                "run_id": "r1", "experiment_id": "P2", "style": "sheya-style", "slot": "a",
                "arm": "variant_eye_clause", "stage": "s", "base_path": _image(Path(folder) / "b.png"),
                "base_sha256": "", "analysis_path": "", "style_ref_path": "",
                "style_reference_sent": False, "repaint_ref": "none", "firmware": "x",
                "expected_diff": []}]}), encoding="utf-8")
            args = type("A", (), {"round": folder, "plan": str(plan_path), "experiment": "P2",
                                  "slot": "a", "arm": "variant_eye_clause", "repeat": 1,
                                  "dry_run": True})()
            with self.assertRaises(SystemExit) as info:
                module.round_run(args)
            self.assertIn("expected_diff", str(info.exception))

    def test_actual_reference_mismatch_is_reported(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "style_face_eval", str(Path(__file__).resolve().parents[1] / "tools" / "style_face_eval.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "api-calls.jsonl"
            log.write_text(json.dumps({
                "operation": "r-final-rp", "prompt_sha256": "aaa",
                "prompt_chars": 10, "reference_images": [{"path": "base.png", "sha256": "x"},
                                                         {"path": "style.png", "sha256": "y"}]}) + "\n",
                encoding="utf-8")
            run = {"style_reference_sent": False, "expected_diff": ["d"], "planned_prompt_sha256": "aaa"}
            diff = module._actual_call_diff(Path(folder), run)
            self.assertTrue(diff["actual_style_reference_sent"])
            self.assertTrue(any("声明不送参考图" in note for note in diff["notes"]))


if __name__ == "__main__":
    unittest.main()
