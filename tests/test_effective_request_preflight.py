# -*- coding: utf-8 -*-
"""第四轮 P0.3 回归：有效请求共享组装 + 严格对账（零 HTTP、零付费）。

三件事：
  1. `--dry-run`/预检与**实际发送**共用同一份组装结果（不是各算一遍）；
  2. 假后端跑完整路径时，每个 run 只有一次重绘请求，没有任何首图/救援图片请求；
  3. `round-diffcheck` 必须拦下第三轮 P1 的真实整段文字差异，同时让 P2 的
     单句替换通过（旧版只看参考图数量，把 P1 误判成 `all_declared_differences_confirmed`）。
"""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ROUND4 = ROOT / "data" / "exp" / "eye-face-hair-round4-20261001"
ROUND3 = ROOT / "data" / "exp" / "eye-face-hair-round3-20261001"


def _load_module():
    spec = importlib.util.spec_from_file_location("style_face_eval", str(ROOT / "tools" / "style_face_eval.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AssemblySharingTests(unittest.TestCase):
    """预检组装出的有效请求必须就是真正发出去的那一份。"""

    def setUp(self):
        if not (ROUND4 / "run-plan.json").is_file():
            self.skipTest("round-4 run plan not built")
        self.module = _load_module()
        self.plan = json.loads((ROUND4 / "run-plan.json").read_text(encoding="utf-8"))
        self.entry = next(r for r in self.plan["runs"] if r["run_id"] == "E1-sheya-a-control-r1")

    def _entry_for(self, output_dir):
        entry = dict(self.entry)
        entry["output_dir"] = str(output_dir)
        return entry

    def _freeze(self, entry):
        code = self.module._round4_call_cli(self.module._round4_argv(entry) + ["--effective-request-only"])
        self.assertIn(code, (0, 1))
        path = Path(entry["output_dir"]) / "effective-request.json"
        self.assertTrue(path.is_file(), "预检必须落盘有效请求")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_dry_run_request_equals_the_dispatched_request(self):
        import modules.others.api_backend as backend
        import utils.post_process as pp
        with tempfile.TemporaryDirectory() as folder:
            frozen = self._freeze(self._entry_for(Path(folder) / "freeze"))
            captured = {}
            fake_product = Path(folder) / "out.png"
            fake_product.write_bytes((ROUND4 / "inputs" / "a" / "base.png").read_bytes())

            def fake_repaint(**kwargs):
                captured.update(kwargs)
                return [str(fake_product)]

            with patch.object(backend, "generate_image_repaint", fake_repaint), \
                    patch("utils.refine_quality.audit_refine_quality",
                          lambda original, candidate, style, **kw: self._quality(candidate)), \
                    patch("utils.refine_quality.audit_hand_quality", self._anatomy), \
                    patch("utils.identity_audit.audit_image_identity", self._identity), \
                    patch.dict(os.environ, {"IMAGE_MAKER_SEND_BUDGET_DIR": "",
                                            "IMAGE_MAKER_EFFECTIVE_REQUEST_DIR": ""}):
                self.module._round4_call_cli(self.module._round4_argv(self._entry_for(Path(folder) / "run")))
            self.assertTrue(captured, "假后端必须被调用一次（重绘）")
            self.assertEqual(captured["prompt"], frozen["prompt"])
            self.assertEqual(list(captured["source_paths"]), list(frozen["source_paths"]))
            self.assertEqual(list(captured.get("extra_reference_paths") or []),
                             list(frozen.get("extra_reference_paths") or []))
            self.assertEqual(captured["resolution"], frozen["resolution"])
            self.assertEqual(captured["aspect_ratio"], frozen["aspect_ratio"])
            self.assertIs(captured["use_detail_suffix"], False)
            self.assertIs(frozen["detail_suffix_applied"], False)
            self.assertEqual(frozen["n"], 1)

    @staticmethod
    def _quality(candidate):
        import utils.refine_quality as quality
        return {"needs_refine": False, "severity": "none", "structural_issues": [], "line_issues": [],
                "background_drift": [], "style_gaps": [], "ownership_uncertain": [],
                "needs_review": False, "candidate": os.path.abspath(candidate),
                "candidate_sha256": quality.file_sha256(candidate)}

    @staticmethod
    def _anatomy(candidate, **kwargs):
        import utils.refine_quality as quality
        data = AssemblySharingTests._quality(candidate)
        return data

    @staticmethod
    def _identity(image_path, analysis, **kwargs):
        import utils.refine_quality as quality
        return {"mismatch": False, "severity": "none", "differences": [], "stable_anchors": [],
                "confidence": .9, "image": os.path.abspath(image_path),
                "candidate_sha256": quality.file_sha256(image_path)}


class PreflightDispatchTests(unittest.TestCase):
    """假后端预检：每个 run 只有一次重绘请求，且没有其他图片通道。"""

    def test_preflight_dispatches_exactly_one_repaint_per_run(self):
        if not (ROUND4 / "run-plan.json").is_file():
            self.skipTest("round-4 run plan not built")
        module = _load_module()
        plan = json.loads((ROUND4 / "run-plan.json").read_text(encoding="utf-8"))
        entry = dict(next(r for r in plan["runs"] if r["run_id"] == "E1-sheya-a-control-r1"))
        with tempfile.TemporaryDirectory() as folder:
            entry["output_dir"] = str(Path(folder) / "run")
            frozen = json.loads(Path(entry["effective_request_path"]).read_text(encoding="utf-8"))
            entry["effective_request"] = {
                "prompt": frozen["prompt"], "source_paths": frozen["source_paths"],
                "extra_reference_paths": frozen["extra_reference_paths"],
                "reference_images": frozen["reference_images"], "model": frozen["model"],
                "resolution": frozen["resolution"], "aspect_ratio": frozen["aspect_ratio"],
                "use_detail_suffix": frozen["use_detail_suffix"],
                "detail_suffix_applied": frozen["detail_suffix_applied"],
                "n": frozen["n"], "repeat": frozen["repeat"]}
            mini = {"problems": [], "runs": [entry]}
            plan_path = Path(folder) / "plan.json"
            plan_path.write_text(json.dumps(mini), encoding="utf-8")
            out = Path(folder) / "out"
            args = type("A", (), {"round": str(ROUND4), "plan": str(plan_path),
                                  "out": str(out), "run": ""})()
            code = module.round4_preflight(args)
            report = json.loads((out / "preflight-report.json").read_text(encoding="utf-8"))
        self.assertEqual(code, 0, report.get("failures"))
        self.assertEqual(report["verdict"], "pass")
        self.assertEqual(report["summary"]["image_dispatches"], 1)
        self.assertEqual(report["runs"][0]["tripwire_hits"], [])
        self.assertEqual(report["real_http_requests"], 0)
        assertions = {item["assertion"]: item["ok"] for item in report["assertions"]}
        self.assertTrue(assertions["run 只有一次重绘图片请求"])
        self.assertTrue(assertions["最终 prompt 与冻结的有效请求逐字相同"])
        self.assertTrue(assertions["没有额外参考图进入生成请求"])


class DiffCheckStrictnessTests(unittest.TestCase):
    """第三轮真实请求：P1 必须被拦下，P2 的单句替换必须通过。"""

    def setUp(self):
        if not (ROUND3 / "run-plan.json").is_file():
            self.skipTest("round-3 evidence not present")
        self.module = _load_module()
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self):
        args = type("A", (), {"round": str(ROUND3), "plan": str(ROUND3 / "run-plan.json"),
                              "out": self.tmp.name, "strict": True})()
        code = self.module.round_diffcheck(args)
        report = json.loads((Path(self.tmp.name) / "request-diff-report.json").read_text(encoding="utf-8"))
        return code, report

    def test_round3_p1_text_difference_is_cannot_compare(self):
        code, report = self._run()
        pairs = {(pair["experiment"], pair["style"], pair["slot"]): pair for pair in report["pairs"]}
        p1 = pairs[("P1", "goto-p", "b")]
        self.assertEqual(p1["verdict"], "cannot_compare")
        self.assertEqual(p1["declaration"]["mode"], "phrasing_only")
        self.assertTrue(any("整段文字" in text for text in p1["problems"]), p1["problems"])
        self.assertNotEqual(report["summary"]["verdict"], "all_declared_differences_confirmed")
        self.assertEqual(code, 1)          # --strict 必须非零

    def test_round3_p2_single_sentence_replacement_passes(self):
        code, report = self._run()
        pairs = {(pair["experiment"], pair["style"], pair["slot"]): pair for pair in report["pairs"]}
        for slot in ("a", "b"):
            pair = pairs[("P2", "sheya-style", slot)]
            self.assertEqual(pair["declaration"]["mode"], "single_sentence_clause")
            self.assertEqual(pair["verdict"], "can_compare", pair["problems"])
            self.assertEqual(pair["problems"], [])
        self.assertEqual(report["summary"]["pairs_can_compare"], 2)

    def test_missing_request_makes_a_pair_incomparable(self):
        plan = {"runs": [
            {"run_id": "x-a", "experiment_id": "X", "style": "s", "slot": "a", "arm": "control",
             "clauses": ["A"], "style_reference_sent": False,
             "output_dir": str(Path(self.tmp.name) / "nothing" / "control")},
            {"run_id": "x-b", "experiment_id": "X", "style": "s", "slot": "a", "arm": "variant",
             "clauses": ["B"], "style_reference_sent": False,
             "output_dir": str(Path(self.tmp.name) / "nothing" / "variant")},
        ]}
        plan_path = Path(self.tmp.name) / "plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        args = type("A", (), {"round": self.tmp.name, "plan": str(plan_path),
                              "out": str(Path(self.tmp.name) / "out"), "strict": True})()
        code = self.module.round_diffcheck(args)
        report = json.loads((Path(self.tmp.name) / "out" / "request-diff-report.json")
                            .read_text(encoding="utf-8"))
        self.assertEqual(report["pairs"][0]["verdict"], "cannot_compare")
        self.assertTrue(any("缺少真实请求记录" in text for text in report["problems"]))
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
