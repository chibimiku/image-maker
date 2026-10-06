"""清色重复验证（v3）与通用协议支持的离线回归。

覆盖：候选追加正文的逐字编译与校验、重复条件（同一 variant 多次）、
冻结包声明的平衡展示顺序、通用 comparisons 解码、缺项不冒充成功、
以及成功缓存的零联网复用。全部在临时目录里跑，不联网、不写实验目录。
"""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils import color_direction_pack as dp

SPEC = dp.BASE / "prompts/color-knowledge/theme-color-clear-repeat-v3-spec.json"
PACK = dp.BASE / "prompts/color-knowledge/theme-color-clear-repeat-v3-pack.json"
SCENES = dp.BASE / "prompts/color-knowledge/theme-color-clear-repeat-v3-scenes.json"
REVIEW = dp.BASE / "prompts/color-knowledge/theme-color-clear-repeat-v3-review.md"
CANDIDATE = dp.BASE / "prompts/color-knowledge/theme-color-clear-candidate-v3.md"


class BuildAndVerifyTests(unittest.TestCase):
    def test_pack_registered_with_registered_budget(self):
        pack = dp.load_pack(PACK)
        approval = dp.APPROVED_PACKS[pack["pack_hash"]]
        self.assertEqual(approval["out_dir"], pack["out_dir"])
        self.assertEqual(approval["images"], pack["image_http_budget"])
        self.assertEqual(approval["reviews"], pack["review_http_budget"])
        self.assertEqual(sorted(slot["variant_id"] for slot in pack["slots"]),
                         sorted(approval["variants"]))
        self.assertIs(pack["automatic_retry"], False)

    def test_candidate_text_is_the_only_append_and_matches_its_source(self):
        pack = dp.load_pack(PACK)
        text_id = "clear-candidate-v3"
        on_disk = (dp.BASE / pack["candidate_text_sources"][text_id]).read_text(encoding="utf-8")
        self.assertEqual(on_disk.rstrip("\n"), pack["candidate_texts"][text_id])
        carrying = {slot["variant_id"] for slot in pack["slots"] if slot["candidate_append"]}
        self.assertEqual(carrying, set(pack["candidate_append_variants"]))
        for slot in pack["slots"]:
            self.assertEqual(slot["candidate_append_sha256"],
                             dp._sha256_text(slot["candidate_append"]))

    def test_c_differs_from_b_only_by_the_appended_text(self):
        pack = dp.load_pack(PACK)
        by_id = {slot["id"]: slot for slot in pack["slots"]}
        for repeat in ("1", "2", "3"):
            base, candidate = by_id["B" + repeat], by_id["C" + repeat]
            self.assertEqual(base["color_plan"], candidate["color_plan"])
            self.assertEqual(base["plan_hash"] if "plan_hash" in base else
                             base["color_plan"]["plan_hash"], candidate["color_plan"]["plan_hash"])
            self.assertEqual(base["selection"], candidate["selection"])
            self.assertEqual(candidate["prompt"],
                             dp.compose_slot_prompt(base["prompt"], candidate["candidate_append"]))
            self.assertNotEqual(base["prompt_sha256"], candidate["prompt_sha256"])

    def test_verify_recompiles_offline_with_zero_problems(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {"IMAGE_MAKER_COLOUR_TEST_OUTPUT_ROOT": folder}):
            result = dp.verify_offline(Path(folder) / "out", pack_path=str(PACK),
                                       scenes_path=str(SCENES), review_path=str(REVIEW),
                                       log=lambda *_: None)
            self.assertEqual(result["problems"], [])
            self.assertEqual(len(result["slots"]), 9)
            self.assertTrue(result["pack_hash_ok"])

    def test_verify_rejects_a_forged_append(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {"IMAGE_MAKER_COLOUR_TEST_OUTPUT_ROOT": folder}):
            forged = json.loads(PACK.read_text(encoding="utf-8"))
            forged["slots"][6]["candidate_append"] = "tampered text"
            forged_path = Path(folder) / "forged-pack.json"
            forged_path.write_text(json.dumps(forged, ensure_ascii=False), encoding="utf-8")
            result = dp.verify_offline(Path(folder) / "out", pack_path=str(forged_path),
                                       scenes_path=str(SCENES), review_path=str(REVIEW),
                                       log=lambda *_: None)
            joined = " ".join(result["problems"])
            self.assertTrue("candidate_append" in joined or "prompt" in joined, joined)

    def test_builder_reproduces_the_frozen_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "rebuilt-pack.json"
            built = dp.build_pack(SPEC, out_pack_path=target, log=lambda *_: None)
            self.assertEqual(built["pack_hash"], dp.load_pack(PACK)["pack_hash"])
            self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["pack_hash"],
                             dp.load_pack(PACK)["pack_hash"])


class DisplayOrderTests(unittest.TestCase):
    def test_declared_balanced_order_is_used_verbatim(self):
        pack = dp.load_pack(PACK)
        pack = copy.deepcopy(pack)
        pack["review_groups"] = [group for group in pack["review_groups"] if group["id"] == "R2"]
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {"IMAGE_MAKER_COLOUR_TEST_OUTPUT_ROOT": folder}):
            out = Path(folder) / "out"
            ledger = dp.load_ledger(out, pack)
            for slot_id in ("B2", "C2", "A2"):
                ledger["slots"][slot_id] = {"slot_id": slot_id, "status": "generated", "attempts": 1,
                                            "outputs": [{"path": "x.jpg", "sha256": "0" * 8}]}
            dp.save_ledger(out, ledger)
            with patch.object(dp, "slot_outputs_intact", return_value=True), \
                    patch.object(dp, "_review_once") as sender:
                sender.return_value = ("reviewed", {"parsed": {"images": []}, "raw": "{}"})
                dp.review_groups(out, pack, {"review": {}}, log=lambda *_: None)
                mapping = json.loads((out / "reviews" / "R2" / "mapping.json").read_text(encoding="utf-8"))
            self.assertEqual(mapping["neutral_to_slot"], {"N1": "B2", "N2": "C2", "N3": "A2"})

    def test_order_without_declaration_still_shuffles(self):
        pack = copy.deepcopy(dp.load_pack(PACK))
        for group in pack["review_groups"]:
            group.pop("neutral_order", None)
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {"IMAGE_MAKER_COLOUR_TEST_OUTPUT_ROOT": folder}):
            out = Path(folder) / "out"
            ledger = dp.load_ledger(out, pack)
            for slot_id in ("A1", "B1", "C1"):
                ledger["slots"][slot_id] = {"slot_id": slot_id, "status": "generated", "attempts": 1,
                                            "outputs": [{"path": "x.jpg", "sha256": "0" * 8}]}
            dp.save_ledger(out, ledger)
            with patch.object(dp, "slot_outputs_intact", return_value=True), \
                    patch.object(dp, "_review_once") as sender:
                sender.return_value = ("reviewed", {"parsed": {"images": []}, "raw": "{}"})
                dp.review_groups(out, pack, {"review": {}}, only={"R1"}, log=lambda *_: None)
                mapping = json.loads((out / "reviews" / "R1" / "mapping.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(mapping["neutral_to_slot"].values()), ["A1", "B1", "C1"])
            self.assertEqual(sorted(mapping["neutral_to_slot"]), ["N1", "N2", "N3"])


def observation(neutral_id, content="pass", palette="conform", reds=("closed notebook cover", "mug")):
    bindings = [{"region": "后墙", "region_id": "back wall", "visible": "yes", "status": "conform"}]
    bindings += [{"region": name, "region_id": name, "visible": "yes", "status": "conform"} for name in reds]
    return {"id": neutral_id, "content_status": content, "content_issues": [],
            "palette_status": palette, "binding_observations": bindings,
            "protected_hues_status": "pass", "protected_hues_issues": [],
            "detail_readability": "pass", "unauthorized_additions": [], "uncertainty": []}


def review_view(mapping, more_distinct, confounded=False, pair=("N1", "N2")):
    return [{"review_id": "R1", "status": "reviewed", "neutral_to_slot": mapping,
             "parsed": {"images": [observation(neutral) for neutral in mapping],
                        "chroma_pairwise": [{"ids": list(pair),
                                             "more_distinct": more_distinct,
                                             "confounded": confounded}]}}]


class ComparisonDecodeTests(unittest.TestCase):
    def test_candidate_more_distinct_counts_as_direction(self):
        pack = dp.load_pack(PACK)
        row = review_view({"N1": "A1", "N2": "B1", "N3": "C1"}, "N2")
        result = dp._decode_directions(pack, row, {})["clear_b_vs_a"]
        self.assertEqual(result["observed_direction_count"], 1)
        self.assertEqual(result["candidate_usable_success"], 1)
        self.assertEqual(result["usable_success"], 1)

    def test_confounded_pair_is_not_a_full_control_success(self):
        pack = dp.load_pack(PACK)
        row = review_view({"N1": "A1", "N2": "B1", "N3": "C1"}, "N2", confounded=True)
        result = dp._decode_directions(pack, row, {})["clear_b_vs_a"]
        self.assertEqual(result["observed_direction_count"], 1)
        self.assertEqual(result["candidate_usable_success"], 1)
        self.assertEqual(result["usable_success"], 0)

    def test_baseline_wins_is_not_a_direction(self):
        pack = dp.load_pack(PACK)
        row = review_view({"N1": "A1", "N2": "B1", "N3": "C1"}, "N1")
        result = dp._decode_directions(pack, row, {})["clear_b_vs_a"]
        self.assertEqual(result["observed_direction_count"], 0)
        self.assertEqual(result["scenes"][0]["verdict"], "opposite_or_unclear")

    def test_content_failure_is_not_usable_even_when_direction_seen(self):
        pack = dp.load_pack(PACK)
        rows = review_view({"N1": "A1", "N2": "B1", "N3": "C1"}, "N2")
        rows[0]["parsed"]["images"][1] = observation("N2", content="fail")
        result = dp._decode_directions(pack, rows, {})["clear_b_vs_a"]
        self.assertEqual(result["scenes"][0]["verdict"], "content_failure")
        self.assertEqual(result["usable_success"], 0)

    def test_missing_pair_entry_is_missing_evidence(self):
        pack = dp.load_pack(PACK)
        rows = [{"review_id": "R1", "status": "reviewed", "neutral_to_slot": {"N1": "A1", "N2": "B1"},
                 "parsed": {"images": [observation("N1"), observation("N2")], "chroma_pairwise": []}}]
        result = dp._decode_directions(pack, rows, {})["clear_b_vs_a"]
        self.assertEqual(result["scenes"][0]["verdict"], "missing_evidence")
        self.assertEqual(result["valid_samples"], 0)

    def test_all_three_comparisons_are_decoded_from_the_same_group(self):
        pack = dp.load_pack(PACK)
        mapping = {"N1": "A1", "N2": "B1", "N3": "C1"}
        pairs = [{"ids": ["N1", "N2"], "more_distinct": "N2", "confounded": False},
                 {"ids": ["N1", "N3"], "more_distinct": "N3", "confounded": False},
                 {"ids": ["N2", "N3"], "more_distinct": "N3", "confounded": False}]
        rows = [{"review_id": "R1", "status": "reviewed", "neutral_to_slot": mapping,
                 "parsed": {"images": [observation(n) for n in mapping], "chroma_pairwise": pairs}}]
        directions = dp._decode_directions(pack, rows, {})
        self.assertEqual(set(directions), {"clear_b_vs_a", "clear_c_vs_a", "clear_c_vs_b"})
        for name in directions:
            self.assertEqual(directions[name]["observed_direction_count"], 1, name)
        # B 相对 A 的方向取自 N1/N2，C 相对 B 的方向取自 N2/N3：不能串用
        self.assertEqual(directions["clear_c_vs_b"]["scenes"][0]["pairwise"]["ids"], ["N2", "N3"])

    def test_legacy_two_scene_pack_still_decodes(self):
        pack = dp.load_pack(dp.BASE / "prompts/color-knowledge/theme-color-chroma-validation-v2-pack.json")
        self.assertIsNone(pack.get("comparisons"))
        rows = [{"review_id": "S1-tone", "status": "reviewed",
                 "neutral_to_slot": {"N1": "S1-A", "N2": "S1-B", "N3": "S1-C"},
                 "parsed": {"images": [observation("N1"), observation("N2"), observation("N3")],
                            "chroma_pairwise": [{"ids": ["N1", "N2"], "more_chromatic": "N1",
                                                 "confounded": True}]}}]
        directions = dp._decode_directions(pack, rows, {})
        self.assertEqual(set(directions), {"clear", "muted"})
        self.assertEqual(directions["clear"]["scenes"][0]["slot_id"], "S1-B")


class CacheAndOfflineTests(unittest.TestCase):
    def test_second_run_reuses_generated_slots_without_network(self):
        pack = dp.load_pack(PACK)
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {"IMAGE_MAKER_COLOUR_TEST_OUTPUT_ROOT": folder}), \
                patch("modules.others.api_backend.generate_image_aigc2d",
                      side_effect=AssertionError("network")) as sender:
            out = Path(folder) / "out"
            ledger = dp.load_ledger(out, pack)
            for slot in pack["slots"]:
                ledger["slots"][slot["id"]] = {"slot_id": slot["id"], "status": "generated",
                                               "attempts": 1, "outputs": []}
            dp.save_ledger(out, ledger)
            with patch.object(dp, "slot_outputs_intact", return_value=True):
                result = dp.run_slots(out, pack, {"generation": {}}, log=lambda *_: None)
            sender.assert_not_called()
            self.assertTrue(all(row.get("reused") for row in result["results"]))
            self.assertEqual(result["ledger"]["counters"]["image_sends"], 0)

    def test_preflight_and_report_send_nothing(self):
        pack = dp.load_pack(PACK)
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {"IMAGE_MAKER_COLOUR_TEST_OUTPUT_ROOT": folder}), \
                patch("modules.others.api_backend.generate_image_aigc2d", side_effect=AssertionError("network")), \
                patch("utils.analysis_gpt_prompt.call_text_model", side_effect=AssertionError("network")):
            runtime = {"generation": {}, "review": {}, "runtime_hash": "offline"}
            mock = dp.run_slots(folder, pack, runtime, dry_run=True, log=lambda *_: None)
            review_mock = dp.review_groups(folder, pack, runtime, dry_run=True, log=lambda *_: None)
            self.assertEqual(len(mock["results"]), 9)
            self.assertEqual(len(review_mock["results"]), 3)
            dp.render_report(folder, pack, runtime, log=lambda *_: None)
            results = json.loads((Path(folder) / "RESULTS.json").read_text(encoding="utf-8"))
            self.assertEqual(set(results["directions"]),
                             {"clear_b_vs_a", "clear_c_vs_a", "clear_c_vs_b"})
            self.assertEqual(results["counters"], {"image_sends": 0, "review_sends": 0})
            self.assertTrue((Path(folder) / "gallery.html").is_file())

    def test_review_template_does_not_leak_conditions(self):
        text = REVIEW.read_text(encoding="utf-8").lower()
        for forbidden in ("baseline", "candidate", "variant", "condition a", "condition b", "condition c"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
