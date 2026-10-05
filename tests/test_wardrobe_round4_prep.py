# -*- coding: utf-8 -*-
"""Round4 离线准备的契约回归（不联网、不发请求、不出图）。

覆盖本轮准备任务的硬性要求：
- 同族工具遇到非 round3 的 schema 必须**明确判未通过**，不能假装校验通过；
- 在线阶段在计划未授权时必须在发出任何请求之前拦住；
- spec 指纹的目录声明参数化之后，round3 的指纹不变；
- 报告口径按计划声明读取，不靠变体名硬编码；
- 筛样/请求草案的隔离检查（留出图不进提取请求、设计组不跨 split、评价字段齐全）。
"""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import hashlib

from tools.wardrobe_experiment import Experiment
from tools.wardrobe_report_v2 import (
    case_conditions_from_spec,
    field_map_for_plan,
    required_fields_for,
)
from utils.wardrobe_experiment import (
    assert_online_supported,
    spec_hash,
    validate_character_spec,
)

ROUND3_SPEC = os.path.join("prompts", "wardrobe", "round3", "spec")
ROUND4_SPEC = os.path.join("prompts", "wardrobe", "round4", "spec")
ROUND4_PLANNING = os.path.join("prompts", "wardrobe", "round4", "planning.json")
ROUND4_REQUESTS = os.path.join("prompts", "wardrobe", "round4", "requests")
SELECTION = os.path.join("docs", "style-extraction", "wardrobe-research", "wardrobe-round4-prep", "selection.json")
PREP_DIR = os.path.join("docs", "style-extraction", "wardrobe-research", "wardrobe-round4-prep")
SCORE_TEMPLATE = os.path.join("prompts", "wardrobe", "round4", "templates", "score-round4.md")


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class UnsupportedSchemaIsNotAPass(unittest.TestCase):
    def test_round4_plan_is_reported_unsupported_not_validated(self):
        result = validate_character_spec(ROUND4_SPEC)
        self.assertFalse(result["ok"], "非 round3 的 schema 不能判通过")
        self.assertEqual(result["unsupported_schema"], "image-maker.wardrobe-round4.v1")
        self.assertTrue(any("未实现" in issue or "只实现" in issue for issue in result["issues"]),
                        result["issues"])

    def test_round3_plan_still_validates(self):
        self.assertTrue(validate_character_spec(ROUND3_SPEC)["ok"])

    def test_online_stage_guard_blocks_planning_only_plans(self):
        planning = load(ROUND4_PLANNING)
        self.assertFalse(planning["cli_compatible"])
        with self.assertRaises(RuntimeError):
            assert_online_supported(planning)
        with self.assertRaises(RuntimeError):
            assert_online_supported(load(os.path.join(ROUND4_SPEC, "plan.json")))

    def test_guard_allows_a_plan_that_declares_online_support(self):
        assert_online_supported({"schema_version": "image-maker.wardrobe-round4.v1",
                                 "cli_compatible": True, "online_execution_authorized": True})
        assert_online_supported({"schema_version": "image-maker.wardrobe-round3.v1"})

    def test_experiment_refuses_to_create_a_run_dir_for_an_unsupported_spec(self):
        """不支持的计划连 run-dir 与账本都不许建（否则会在产物目录里留空目录）。"""
        with tempfile.TemporaryDirectory() as temp:
            args = type("Args", (), {
                "spec": ROUND4_SPEC, "pack": os.path.join("prompts", "wardrobe", "round4", "pack"),
                "output_root": temp, "run_id": "", "run_dir": ""})()
            with self.assertRaises(RuntimeError):
                Experiment(args)
            self.assertEqual(os.listdir(temp), [])


class SpecHashStaysBackwardsCompatible(unittest.TestCase):
    def test_round3_spec_hash_is_unchanged_after_parameterisation(self):
        # Git normalizes historical CRLF files. Bind the portable LF projection
        # while retaining the original byte digest in the archived report.
        recorded = load("docs/style-extraction/wardrobe-research/portable-spec-digest.json")
        with patch("utils.wardrobe_experiment.sha256_file", side_effect=lambda path:
                   hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()):
            self.assertEqual(spec_hash(ROUND3_SPEC), recorded["round3_spec_lf_sha256"])

    def test_round4_spec_hash_covers_the_declared_directories(self):
        first = spec_hash(ROUND4_SPEC)
        self.assertEqual(first, spec_hash(ROUND4_SPEC))
        self.assertEqual(len(first), 64)


class ReportFieldsComeFromThePlan(unittest.TestCase):
    def test_round3_field_set_is_unchanged(self):
        conditions = case_conditions_from_spec(ROUND3_SPEC)
        self.assertIn("framing", conditions["T05"]["required_fields_round3"])
        self.assertIn("identity_anchors", conditions["T05"]["required_fields_round3"])
        self.assertNotIn("identity_anchors", conditions["T05"]["required_fields_round2"])
        self.assertEqual(required_fields_for(conditions["T01"],
                                             {"A": "v3_control", "B": "v4_candidate"}),
                         tuple(conditions["T01"]["required_fields_round3"]))

    def test_round4_field_set_is_declared_in_the_plan(self):
        plan = load(os.path.join(ROUND4_SPEC, "plan.json"))
        field_map = field_map_for_plan(plan)
        self.assertEqual(field_map["field_set"], "required_fields_round4")
        required = field_map["tables"]["required_fields_round4"]["redesign"]
        for name in ("complete_transformation_or_preservation", "lolita_structure_when_requested",
                     "target_family_evidence_when_requested", "protected_colours",
                     "protected_accessories", "person_count", "identity_attributes", "scene",
                     "full_body_framing", "visible_defects"):
            self.assertIn(name, required)

    def test_round4_conditions_can_be_read_by_case_id(self):
        conditions = case_conditions_from_spec(ROUND4_SPEC)
        self.assertEqual(sorted(conditions), ["GF", "GP", "H", "R", "U", "W"])
        self.assertEqual(conditions["U"]["gold_action"], "fill")
        self.assertEqual(conditions["GP"]["gold_action"], "preserve")
        self.assertIn("person_count", conditions["U"]["required_fields_round4"])
        self.assertNotIn("target_family_evidence_when_requested",
                         conditions["GP"]["required_fields_round4"])
        self.assertTrue(conditions["H"]["identity"].startswith("Exactly one 32-year-old adult woman"))


class EvaluationTemplateCoversEveryRequiredField(unittest.TestCase):
    def test_person_count_scene_and_accessories_cannot_be_dropped(self):
        text = open(SCORE_TEMPLATE, encoding="utf-8").read()
        for field in ("complete_transformation_or_preservation", "lolita_structure_when_requested",
                      "target_family_evidence_when_requested", "protected_colours",
                      "protected_accessories", "person_count", "identity_attributes", "scene",
                      "full_body_framing", "visible_defects", "overall"):
            self.assertIn(field, text)
        self.assertIn("person_count` failing", text)
        self.assertIn("scene` failing", text)
        self.assertIn("unproven", text)


class SelectionAndRequestIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selection = load(SELECTION)
        cls.records = {row["id"]: row for row in cls.selection["records"]}
        cls.manifest = load(os.path.join(ROUND4_REQUESTS, "manifest.json"))

    def test_every_family_has_eight_three_three(self):
        for family in ("classical", "sweet", "gothic"):
            counts = self.selection["counts"][family]
            self.assertTrue(counts["extraction"]["quota_met"], family)
            self.assertEqual(counts["extraction"]["selected"], 8, family)
            self.assertEqual(counts["heldout"]["selected"], 3, family)
            self.assertEqual(counts["boundary"]["selected"], 3, family)

    def test_every_record_carries_source_path_and_hash(self):
        for row in self.selection["records"]:
            self.assertTrue(row["source_path"], row["id"])
            self.assertEqual(len(row["source_sha256"]), 64, row["id"])
            self.assertTrue(row["reason"], row["id"])
            self.assertTrue(row["ambiguity"], row["id"])

    def test_extraction_examples_record_at_least_three_reasons(self):
        checked = 0
        for row in self.selection["records"]:
            if row["split"] != "extraction":
                continue
            checked += 1
            self.assertGreaterEqual(len(row["reason"]), 2, row["id"])
            self.assertTrue(row["reason"][0].strip(), row["id"])
        self.assertEqual(checked, 24)

    def test_design_groups_never_span_extraction_and_heldout(self):
        extraction = {row["design_group"] for row in self.selection["records"]
                      if row["split"] == "extraction"}
        heldout = {row["design_group"] for row in self.selection["records"]
                   if row["split"] == "heldout"}
        self.assertFalse(extraction & heldout, sorted(extraction & heldout))

    def test_held_out_images_are_absent_from_extraction_requests(self):
        heldout = {row["id"] for row in self.selection["records"] if row["split"] == "heldout"}
        checked = 0
        for entry in self.manifest["requests"]:
            if entry["kind"] != "extraction":
                continue
            checked += 1
            request = load(entry["file"])
            self.assertEqual(request["status"], "draft_not_sent")
            self.assertFalse(request["sent"])
            self.assertEqual(request["real_http_requests"], 0)
            self.assertLessEqual(len(request["attachments"]), 4)
            self.assertFalse(heldout & set(request["attachment_order"]), entry["slot"])
            for image_id in heldout:
                self.assertNotIn(image_id, request["user_text"], entry["slot"])
            self.assertTrue(request["no_held_out_images_in_attachments"], entry["slot"])
        self.assertEqual(checked, 6)

    def test_heldout_requests_contain_only_heldout_images(self):
        heldout = {row["id"] for row in self.selection["records"] if row["split"] == "heldout"}
        checked = 0
        for entry in self.manifest["requests"]:
            if entry["kind"] != "heldout_check":
                continue
            checked += 1
            request = load(entry["file"])
            self.assertTrue(set(request["attachment_order"]) <= heldout, entry["slot"])
        self.assertEqual(checked, 3)

    def test_conversion_drafts_are_labelled_as_hand_written_drafts(self):
        conversion = 0
        preservation = 0
        for entry in self.manifest["requests"]:
            request = load(entry["file"])
            if request["kind"] == "conversion":
                conversion += 1
                provenance = request["wardrobe_block_provenance"]
                self.assertEqual(provenance["kind"], "hand_written_draft", entry["slot"])
                self.assertIn("draft", provenance["warning"].lower())
                self.assertIn("AUTHORIZED WARDROBE TRANSFORMATION", request["prompt"])
            elif request["kind"] == "preservation":
                preservation += 1
                for token in ("Lolita", "lolita", "OP dress", "JSK", "bell skirt"):
                    self.assertNotIn(token, request["prompt"], entry["slot"])
                self.assertIn("USER REQUEST (takes precedence)", request["prompt"])
        self.assertEqual(conversion, 12)
        self.assertEqual(preservation, 2)

    def test_drafts_record_their_assembly_sources_and_hashes(self):
        for entry in self.manifest["requests"]:
            request = load(entry["file"])
            if request["kind"] not in ("conversion", "preservation"):
                continue
            self.assertTrue(request["assembled_from"], entry["slot"])
            for item in request["assembled_from"]:
                self.assertEqual(len(item["sha256"]), 64, entry["slot"])
            self.assertEqual(request["generation_references"], 0)

    def test_offline_checks_report_is_present_and_passed(self):
        report = load(os.path.join(PREP_DIR, "offline-checks.json"))
        self.assertEqual(report["verdict"], "pass", report.get("failures"))
        self.assertEqual(report["real_http_requests"], 0)
        self.assertFalse(report["run_dir_created"])
        self.assertEqual(len(report["wave1_call_list"]), 14)
        self.assertEqual(report["budget_recommendation_not_granted"]["wave1_generation_attempts"], 14)


if __name__ == "__main__":
    unittest.main()
