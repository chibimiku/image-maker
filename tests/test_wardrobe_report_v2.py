# -*- coding: utf-8 -*-
"""参数化汇总的离线回归：按 spec 条件判定、missing/uncertain 绝不默认通过。"""
import json
import os
import unittest

from tools.wardrobe_report_v2 import (
    case_conditions_from_spec,
    judge_arm,
    required_fields_for,
    resolve_conditions,
)

SPEC = os.path.join("prompts", "wardrobe", "round3", "spec")


def scored_record(assignment, per_label, status="scored"):
    return {"pair_id": "T01-gpt-r1", "case_id": "T01", "status": status,
            "assignment": assignment,
            "parsed": {"images": [{"label": label, **values} for label, values in per_label.items()],
                       "preferred": "A"}}


class ReportContracts(unittest.TestCase):
    def setUp(self):
        self.conditions = case_conditions_from_spec(SPEC)

    def test_round3_conditions_come_from_spec(self):
        self.assertEqual(sorted(self.conditions), ["T01", "T02", "T03", "T04", "T05", "T06"])
        t05 = self.conditions["T05"]
        self.assertEqual(t05["framing"], "upper_chest")
        self.assertIn("framing", t05["required_fields_round3"])
        self.assertIn("identity_anchors", t05["required_fields_round3"])
        self.assertNotIn("identity_anchors", t05["required_fields_round2"])
        self.assertTrue(t05["identity"].startswith("Exactly one 26-year-old adult woman"))
        self.assertTrue(t05["protected"])

    def test_fields_follow_the_scoring_template_of_that_round(self):
        round3 = required_fields_for(self.conditions["T01"], {"A": "v3_control", "B": "v4_candidate"})
        self.assertIn("specified_items", round3)
        legacy = required_fields_for(self.conditions["T01"], {"A": "v2_frozen", "B": "v3_routed"})
        self.assertIn("protected_anchors", legacy)
        self.assertNotIn("specified_items", legacy)

    def test_pass_requires_every_required_field_to_pass(self):
        record = scored_record({"A": "v3_control", "B": "v4_candidate"}, {
            "A": {"clothing_action": "fail", "identity_anchors": "pass",
                  "specified_items": "pass", "framing": "pass"},
            "B": {"clothing_action": "pass", "identity_anchors": "pass",
                  "specified_items": "pass", "framing": "pass"}})
        self.assertEqual(judge_arm(record, self.conditions["T01"])["outcome"], "pass")

    def test_missing_field_is_unproven_not_pass(self):
        record = scored_record({"A": "v3_control", "B": "v4_candidate"}, {
            "A": {"clothing_action": "pass", "identity_anchors": "pass",
                  "specified_items": "pass", "framing": "pass"},
            "B": {"clothing_action": "pass", "framing": "pass"}})
        judged = judge_arm(record, self.conditions["T01"])
        self.assertEqual(judged["outcome"], "unproven")
        self.assertIn("identity_anchors", judged["missing_fields"])

    def test_uncertain_and_fail_are_not_interchangeable(self):
        uncertain = scored_record({"A": "v3_control", "B": "v4_candidate"}, {
            "A": {"clothing_action": "pass", "identity_anchors": "pass",
                  "specified_items": "pass", "framing": "pass"},
            "B": {"clothing_action": "uncertain", "identity_anchors": "pass",
                  "specified_items": "pass", "framing": "pass"}})
        self.assertEqual(judge_arm(uncertain, self.conditions["T01"])["outcome"], "unproven")
        failing = scored_record({"A": "v3_control", "B": "v4_candidate"}, {
            "A": {"clothing_action": "pass", "identity_anchors": "pass",
                  "specified_items": "pass", "framing": "pass"},
            "B": {"clothing_action": "pass", "identity_anchors": "fail",
                  "specified_items": "pass", "framing": "pass"}})
        self.assertEqual(judge_arm(failing, self.conditions["T01"])["outcome"], "failed")

    def test_unscored_pair_is_missing(self):
        record = {"pair_id": "T05-gpt-r1", "status": "not_scored_missing_conditions",
                  "assignment": {"A": "v3_control", "B": "v4_candidate"}}
        judged = judge_arm(record, self.conditions["T05"])
        self.assertEqual(judged["outcome"], "missing")

    def test_candidate_failure_is_not_offset_by_the_control_arm(self):
        record = scored_record({"A": "v4_candidate", "B": "v3_control"}, {
            "A": {"clothing_action": "fail", "identity_anchors": "pass",
                  "specified_items": "pass", "framing": "pass"},
            "B": {"clothing_action": "pass", "identity_anchors": "pass",
                  "specified_items": "pass", "framing": "pass"}})
        judged = judge_arm(record, self.conditions["T01"])
        self.assertEqual(judged["outcome"], "failed")
        self.assertEqual(judged["control_fields"]["clothing_action"], "pass")

    def test_condition_source_must_cover_the_run_cases(self):
        conditions, source, covered = resolve_conditions(
            "docs/style-extraction/wardrobe-research/wardrobe-round2-r1")
        self.assertIn("C03", conditions)
        self.assertEqual(sorted(covered), ["C01", "C02", "C03", "C04", "C06"])
        self.assertIn("test-plan.json", source.replace("\\", "/"))


if __name__ == "__main__":
    unittest.main()
