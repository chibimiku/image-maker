import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from utils.color_extract import build_generation_knowledge, compile_generation_palette
from utils.color_experiment import book_ablation_plan, book_generation_body, book_grayscale_proxy, book_ablation_report, validate_book_ablation_review

ROOT = Path(__file__).resolve().parents[1]


class ScopedRulesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.knowledge = build_generation_knowledge(str(ROOT / "data/color-knowledge"),
            str(ROOT / "docs/261003-color-improve/book-html/pages"),
            str(ROOT / "prompts/color-knowledge/book-image-chart-palettes-v1.json"),
            str(ROOT / "prompts/color-knowledge/book-scoped-rules-v1.json"))
        cls.selection = json.loads((ROOT / "prompts/color-knowledge/book-generation-example-v3.json").read_text(encoding="utf-8"))

    def compile(self, **changes):
        selection = copy.deepcopy(self.selection)
        selection.update(changes)
        return compile_generation_palette(self.knowledge, selection)

    def test_counts_not_inflated_and_tags_are_retrieval_only(self):
        self.assertEqual(self.knowledge["count"], 109)
        extension = self.knowledge["scoped_rules"]
        self.assertEqual(len(extension["generation_rules"]), 7)
        self.assertEqual(len(extension["audit_rules"]), 3)
        self.assertEqual(len(extension["retrieval_metadata"]), 8)
        self.assertTrue(all(r["auto_inject"] is False for r in extension["retrieval_metadata"]))

    def test_audits_do_not_enter_generation_prompt(self):
        result = self.compile()
        plain = self.compile(audit_rules=[])
        self.assertEqual(result["prompt"], plain["prompt"])
        self.assertEqual(len(result["scoped_rule_snapshot"]["audits"]), 3)
        self.assertIn("At canal water", result["prompt"])
        self.assertNotIn("snow", result["prompt"])

    def test_material_absent_is_rejected(self):
        with self.assertRaises(ValueError):
            self.compile(scoped_rules=[{"id": "glass-transmission", "regions": ["canal water"]}])

    def test_material_fact_is_explicit_and_snapshotted(self):
        result = self.compile(region_facts={"canal water": {"material": "glass"}},
            scoped_rules=[{"id": "glass-transmission", "regions": ["canal water"]}])
        self.assertEqual(result["scoped_rule_snapshot"]["applied"][0]["facts"]["canal water"]["material"], "glass")

    def test_material_rule_colour_conflict_is_rejected(self):
        with self.assertRaises(ValueError):
            self.compile(region_facts={"belt": {"material": "white_porcelain", "existing_highlights": True}},
                scoped_rules=[{"id": "porcelain-gloss", "regions": ["belt"]}])

    def test_protected_or_added_regions_rejected(self):
        for region in ("skin", "new glass cup"):
            with self.assertRaises(ValueError):
                self.compile(scoped_rules=[{"id": "palette-hue-budget", "regions": [region]}])

    def test_audit_id_cannot_be_generation_rule(self):
        with self.assertRaises(ValueError):
            self.compile(scoped_rules=[{"id": "perceived-depth", "regions": ["canal water"]}])

    def test_duplicate_or_unknown_scoped_requests_rejected(self):
        request = {"id": "palette-hue-budget", "regions": ["canal water"]}
        with self.assertRaises(ValueError):
            self.compile(scoped_rules=[request, request])
        with self.assertRaises(ValueError):
            self.compile(audit_rules=["unknown"])

    def test_local_tone_scope_and_global_conflict(self):
        with self.assertRaises(ValueError):
            self.compile(region_tones={"skin": "L"})
        with self.assertRaises(ValueError):
            self.compile(tone_code="L")
        with self.assertRaises(ValueError):
            self.compile(include_relationship_rules=False, arrangement_mode="none")

    def test_reference_plan_is_eight_no_style_images_and_narrow_bank(self):
        with patch("utils.analysis_gpt_prompt.load_text_api_config", return_value={"model": "test", "base_url": "https://test.invalid"}):
            plan = book_ablation_plan(str(ROOT / "prompts/color-knowledge/book-content-anchor-v1.json"))
        self.assertEqual(len(plan["requests"]), 8)
        self.assertEqual(sum(bool(r["image_paths"]) for r in plan["requests"]), 4)
        self.assertTrue(all(g["compiled"]["selection"]["style_reference_images"] == [] for g in plan["groups"]))
        self.assertTrue(all("opposite riverbank ground" in g["compiled"]["selection"]["protected_regions"] for g in plan["groups"]))
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "book-frozen.json").write_text(json.dumps(plan), encoding="utf-8")
            with patch("utils.color_experiment._isolated_out", return_value=Path(tmp)):
                report = book_ablation_report(tmp)
            self.assertEqual(report["planned"], 8)
            self.assertTrue(all(g["generated"] == 0 for g in report["groups"]))

    def test_body_retains_real_attachment_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.png"
            Image.new("RGB", (40, 30), "red").save(path)
            request = {"prompt": "same", "image_paths": [str(path)], "aspect_ratio": "3:2", "resolution": "1K"}
            body = book_generation_body(request)
            self.assertEqual(len(body["contents"][0]["parts"]), 2)
            self.assertTrue(body["contents"][0]["parts"][1]["inline_data"]["data"])

    def test_grayscale_proxy_never_changes_colour_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "source.png"
            Image.new("RGB", (40, 30), "red").save(path)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            gray = book_grayscale_proxy(root, path)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            with Image.open(gray) as image:
                self.assertEqual(image.mode, "L")

    def test_extended_review_never_accepts_missing_count_checks(self):
        row = {"id": "Q01", "colour_status": "conform", "region_status": "conform", "area_status": "conform",
               "protected_ok": True, "content_ok": True,
               "extra_colour_areas": [], "protection_issues": [], "content_issues": [], "unverifiable_reason": [], "evidence": []}
        with self.assertRaises(ValueError):
            validate_book_ablation_review({"images": [row]}, ["Q01"], content_anchor=True)
        row.update(visible_person_count=1, source_count_ok=True, source_layout_status="conform",
                   grayscale_status="partial", grayscale_evidence=["low edge contrast"])
        validate_book_ablation_review({"images": [row]}, ["Q01"], content_anchor=True)
        row["visible_person_count"] = True
        with self.assertRaises(ValueError):
            validate_book_ablation_review({"images": [row]}, ["Q01"], content_anchor=True)


if __name__ == "__main__":
    unittest.main()
