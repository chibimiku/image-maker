import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils.color_experiment import book_ablation_plan, freeze_book_ablation, validate_book_ablation_review, book_ablation_report, sanitize_book_ablation_replays, book_sample_attempts

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "prompts/color-knowledge/book-palette-ablation-v1.json"


class BookAblationTests(unittest.TestCase):
    def plan(self):
        with patch("utils.analysis_gpt_prompt.load_text_api_config", return_value={"model": "test-review", "base_url": "https://test.invalid", "api_key": "unused"}):
            return book_ablation_plan(str(SPEC))

    def test_ten_requests_no_reference_and_no_promotion(self):
        plan = self.plan()
        self.assertEqual(len(plan["requests"]), 10)
        self.assertEqual(len({r["id"] for r in plan["requests"]}), 10)
        self.assertTrue(all(not r["image_paths"] and not r["face_quality_boost"] for r in plan["requests"]))
        self.assertNotIn("unused", str(plan))

    def test_only_rule_instructions_differ_between_variants(self):
        plan = self.plan()
        for prefix in ("C13", "C16"):
            a, b = [g for g in plan["groups"] if g["id"].startswith(prefix)]
            self.assertEqual(a["compiled"]["selection"]["bindings"], b["compiled"]["selection"]["bindings"])
            self.assertEqual(a["colours"], b["colours"])
            self.assertNotIn("Keep the assigned base", a["prompt"])
            self.assertIn("Keep the assigned base", b["prompt"])
            self.assertEqual(a["prompt"].split("PRESERVE:")[-1], b["prompt"].split("PRESERVE:")[-1])

    def test_v2_common_guard_and_original_protocol_unchanged(self):
        old = self.plan()
        with patch("utils.analysis_gpt_prompt.load_text_api_config", return_value={"model": "test-review", "base_url": "https://test.invalid", "api_key": "unused"}):
            new = book_ablation_plan(str(SPEC.with_name("book-palette-ablation-v2.json")))
        self.assertEqual(len(new["requests"]), 10)
        self.assertNotEqual(old["plan_hash"], new["plan_hash"])
        for a, b in zip(old["requests"], new["requests"]):
            self.assertEqual(a["id"], b["id"])
            for key in ("model", "resolution", "aspect_ratio", "image_paths"):
                self.assertEqual(a[key], b[key])
            self.assertNotIn("SCENE AND SUBJECT COUNT LOCK", a["prompt"])
            self.assertEqual(b["prompt"].count("SCENE AND SUBJECT COUNT LOCK"), 1)
        for g in new["groups"]:
            self.assertIn("style-only reference", g["prompt"])
            self.assertIn("If no content reference exists", g["prompt"])
            self.assertIn("never draw swatches", g["prompt"])
        self.assertIn("protected_ok concerns intrinsic", new["review_system"])
        self.assertIn("one young woman", new["content_constraints"])

    def test_report_keeps_unattempted_denominator(self):
        plan = self.plan()
        with tempfile.TemporaryDirectory() as tmp:
            from utils.atomic_io import write_json_atomic
            write_json_atomic(str(Path(tmp) / "book-frozen.json"), plan)
            with patch("utils.color_experiment._isolated_out", return_value=Path(tmp)):
                result = book_ablation_report(tmp)
            self.assertEqual(result["planned"], 10)
            self.assertTrue(all(g["usable"] == 0 and not g["promoted"] for g in result["groups"]))

    def test_review_requires_explicit_bool_not_string(self):
        row = {"id": "Q01", "colour_status": "conform", "region_status": "conform", "area_status": "conform",
               "protected_ok": True, "content_ok": True,
               "extra_colour_areas": [], "protection_issues": [], "content_issues": [], "unverifiable_reason": [], "evidence": []}
        validate_book_ablation_review({"images": [row]}, ["Q01"])
        row["protected_ok"] = "true"
        with self.assertRaises(ValueError):
            validate_book_ablation_review({"images": [row]}, ["Q01"])

    def test_single_variant_report_writes_all_deliverables_without_comparison(self):
        plan = self.plan()
        plan["spec"]["variants"] = {"simple": plan["spec"]["variants"]["simple"]}
        plan["groups"] = [g for g in plan["groups"] if g.get("variant") == "simple"]
        ids = {g["id"] for g in plan["groups"]}
        plan["requests"] = [r for r in plan["requests"] if r["group_id"] in ids]
        plan["planned_images"] = len(plan["requests"])
        with tempfile.TemporaryDirectory() as tmp:
            from utils.atomic_io import write_json_atomic
            root = Path(tmp)
            write_json_atomic(str(root / "book-frozen.json"), plan)
            with patch("utils.color_experiment._isolated_out", return_value=root):
                result = book_ablation_report(tmp)
            self.assertEqual(result["planned"], 4)
            self.assertEqual(result["variant_comparisons"], [])
            self.assertTrue(all(g["generated"] == 0 and g["usable"] == 0 for g in result["groups"]))
            for name in ("book-summary.json", "book-per-image.json", "call-ledger.json", "evaluation.html", "RUN-REPORT.md"):
                self.assertTrue((root / name).is_file(), name)
            self.assertIn("没有同轮变体对照", (root / "RUN-REPORT.md").read_text(encoding="utf-8"))

    def test_three_variants_report_includes_every_pair(self):
        plan = self.plan()
        plan["spec"]["variants"]["third"] = {}
        for scheme in plan["spec"]["schemes"]:
            group = copy.deepcopy(next(g for g in plan["groups"] if g["id"] == scheme["id"] + "-simple"))
            group.update(id=scheme["id"] + "-third", variant="third")
            plan["groups"].append(group)
            for request in list(plan["requests"]):
                if request["group_id"] == scheme["id"] + "-simple":
                    extra = copy.deepcopy(request)
                    extra.update(id=group["id"] + "-r" + str(request["round"]), group_id=group["id"])
                    plan["requests"].append(extra)
        plan["planned_images"] = len(plan["requests"])
        with tempfile.TemporaryDirectory() as tmp:
            from utils.atomic_io import write_json_atomic
            write_json_atomic(str(Path(tmp) / "book-frozen.json"), plan)
            with patch("utils.color_experiment._isolated_out", return_value=Path(tmp)):
                result = book_ablation_report(tmp)
        self.assertEqual(len(result["variant_comparisons"]), 6)
        for scheme in ("C13", "C16"):
            pairs = [{key for key in c if key.endswith("_usable")} for c in result["variant_comparisons"] if c["scheme"] == scheme]
            self.assertEqual(pairs, [{"simple_usable", "knowledge_usable"}, {"simple_usable", "third_usable"}, {"knowledge_usable", "third_usable"}])

    def test_duplicate_image_ids_rejected(self):
        with self.assertRaises(ValueError):
            validate_book_ablation_review({"images": [{"id": "Q01"}, {"id": "Q01"}]}, ["Q01", "Q02"])

    def test_backend_replay_headers_are_removed(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples" / "sample" / "sample_replay_01.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"headers": {"x-goog-api-key": "test-secret"}, "body": {"prompt": "same"}}), encoding="utf-8")
            self.assertEqual(sanitize_book_ablation_replays(tmp), 1)
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(value["headers"], {})
            self.assertEqual(value["body"], {"prompt": "same"})
            self.assertNotIn("test-secret", path.read_text(encoding="utf-8"))

    def test_retry_history_preserves_failed_first_attempt(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            retry = root / "attempts" / "attempt-02"
            retry.mkdir(parents=True)
            initial = {"status": "failed", "outputs": []}
            (root / "result.json").write_text(json.dumps(initial), encoding="utf-8")
            (retry / "result.json").write_text(json.dumps({"status": "success", "outputs": ["image.png"]}), encoding="utf-8")
            history = book_sample_attempts(root)
            self.assertEqual([a["status"] for a in history], ["failed", "success"])
            self.assertEqual(json.loads((root / "result.json").read_text()), initial)

    def test_nested_retry_replay_is_sanitized(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples" / "sample" / "attempts" / "attempt-02" / "sample_replay_01.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"headers": {"Authorization": "secret"}, "body": {}}), encoding="utf-8")
            self.assertEqual(sanitize_book_ablation_replays(tmp), 1)
            self.assertEqual(json.loads(path.read_text())["headers"], {})


if __name__ == "__main__":
    unittest.main()
