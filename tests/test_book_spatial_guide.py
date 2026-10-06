"""独立位置图（spatial_guide）研究：附件角色、顺序、冻结与旧协议不变式。"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils.color_experiment import (BOOK_SPATIAL_GUIDE_ROLE, book_ablation_plan, book_generation_body,
                                    book_preflight, book_reference_images, freeze_book_ablation)

ROOT = Path(__file__).resolve().parents[1]
SPATIAL_SPEC = ROOT / "prompts/color-knowledge/book-spatial-guide-v1.json"
LEGACY_HASHES = {
    "book-palette-ablation-v1.json": "3ed42e3e69d386b758e1ce5d7c76955d4b7a0efdcd81a70ec25ee90b74d09549",
    "book-palette-ablation-v2.json": "467b653f2acb6abde4fd75c02efe3aa5d30dfd23413a172ee8c896f4b3b3496c",
    "book-region-contract-v1.json": "8051d5fd7129c9465f35ab544a6bdf286a23a32fb2da3f64969664068eb70abb",
}
TEXT_CFG = {"model": "test-review", "base_url": "https://test.invalid", "api_key": "unused"}


def legacy_plan(name):
    with patch("utils.analysis_gpt_prompt.load_text_api_config", return_value=dict(TEXT_CFG)):
        return book_ablation_plan(str(ROOT / "prompts/color-knowledge" / name))


def spatial_plan(spec_path=None):
    with patch("utils.analysis_gpt_prompt.load_text_api_config", return_value=dict(TEXT_CFG)):
        return book_ablation_plan(str(spec_path or SPATIAL_SPEC))


def isolated(root):
    return patch("utils.color_experiment._isolated_out", return_value=Path(root))


class LegacyProtocolTests(unittest.TestCase):
    def test_frozen_legacy_plan_is_still_reproduced(self):
        frozen = ROOT / "data/test-result/20261005/book-region-contract-v1/book-frozen.json"
        if not frozen.is_file():
            self.skipTest("frozen legacy plan is not present in this checkout")
        stored = json.loads(frozen.read_text(encoding="utf-8"))
        config = {"model": stored["review_model"], "base_url": stored["review_base_url"], "api_key": "unused"}
        with patch("utils.analysis_gpt_prompt.load_text_api_config", return_value=config):
            plan = book_ablation_plan(str(ROOT / "prompts/color-knowledge/book-region-contract-v1.json"))
        self.assertEqual(plan["plan_hash"], stored["plan_hash"])
        self.assertEqual([r["id"] for r in plan["requests"]], [r["id"] for r in stored["requests"]])

    def test_legacy_requests_still_have_one_content_attachment_only(self):
        plan = legacy_plan("book-region-contract-v1.json")
        for request in plan["requests"]:
            self.assertNotIn("attachments", request)
            self.assertEqual([a["role"] for a in book_reference_images(request)], ["content"])
        self.assertNotIn("spatial_guide", plan["groups"][0])
        self.assertEqual(plan["groups"][0]["prompt"].count("SPATIAL GUIDE ROLES"), 0)
        self.assertNotIn("image_roles", json.dumps(plan["groups"]))


class SpatialGuidePlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = spatial_plan()

    def test_four_requests_two_branches(self):
        self.assertEqual(len(self.plan["requests"]), 4)
        self.assertEqual([g["id"] for g in self.plan["groups"]], ["C16-source", "C16-source_locator"])
        self.assertEqual(sorted({r["group_id"] for r in self.plan["requests"]}),
                         ["C16-source", "C16-source_locator"])

    def test_branch_a_has_no_guide_text_or_attachment(self):
        group = next(g for g in self.plan["groups"] if g["id"] == "C16-source")
        self.assertNotIn("SPATIAL GUIDE ROLES", group["prompt"])
        self.assertEqual([e["role"] for e in group["attachments"]], ["content"])
        for request in self.plan["requests"]:
            if request["group_id"] == "C16-source":
                self.assertEqual(len(request["image_paths"]), 1)

    def test_branch_b_carries_the_locator_and_its_role_text(self):
        group = next(g for g in self.plan["groups"] if g["id"] == "C16-source_locator")
        self.assertIn("SPATIAL GUIDE ROLES", group["prompt"])
        self.assertIn("not a native API mask", group["prompt"])
        self.assertEqual([(e["order"], e["role"]) for e in group["attachments"]],
                         [(1, "content"), (2, "spatial_guide")])
        self.assertEqual(group["attachments"][1]["size"], [1264, 848])

    def test_shared_text_contract_identical_in_both_branches(self):
        first, second = self.plan["groups"]
        self.assertEqual(first["prompt"].split("PRESERVE:")[-1], second["prompt"].split("PRESERVE:")[-1])
        for phrase in ("upper-right of the frame", "lower-right corner", "opposite riverbank ground"):
            self.assertIn(phrase, first["prompt"])
            self.assertIn(phrase, second["prompt"])

    def test_attachment_records_cover_path_hash_size_and_order(self):
        for request in self.plan["requests"]:
            entries = request["attachments"]
            self.assertEqual([e["order"] for e in entries], list(range(1, len(entries) + 1)))
            self.assertEqual([e["path"] for e in entries], list(request["image_paths"]))
            for entry in entries:
                self.assertEqual(entry["size"], [1264, 848])
                self.assertEqual(len(entry["sha256"]), 64)
            roles = [e["role"] for e in entries]
            self.assertEqual(roles[0], "content")
            self.assertEqual(roles.count("spatial_guide"), 1 if len(roles) > 1 else 0)

    def test_locator_is_never_labelled_content(self):
        guide = self.plan["spatial_guide"]
        self.assertEqual(guide["role"], BOOK_SPATIAL_GUIDE_ROLE)
        self.assertEqual(guide["native_mask_support"], "not_confirmed")
        for request in self.plan["requests"]:
            for entry in request["attachments"]:
                if entry["path"] == guide["absolute_path"]:
                    self.assertEqual(entry["role"], "spatial_guide")

    def test_request_body_sends_images_in_declared_order_without_mask_field(self):
        request = next(r for r in self.plan["requests"] if r["group_id"] == "C16-source_locator")
        body = book_generation_body(request)
        self.assertEqual(len(body["contents"][0]["parts"]), 3)
        self.assertIn("text", body["contents"][0]["parts"][0])
        self.assertEqual([set(p) for p in body["contents"][0]["parts"][1:]], [{"inline_data"}, {"inline_data"}])
        self.assertEqual(set(body), {"contents", "generationConfig"})
        self.assertEqual(set(body["generationConfig"]["imageConfig"]), {"aspectRatio", "imageSize"})

    def test_guide_change_or_role_conflict_stops(self):
        for change, message in (({"sha256": "0" * 64}, "hash mismatch"),
                                ({"role": "content"}, "role"),
                                ({"path": "prompts/color-knowledge/book-content-anchor-v1.md"}, "hash")):
            spec = json.loads(SPATIAL_SPEC.read_text(encoding="utf-8"))
            spec["spatial_guide"].update(change)
            spec.pop("extends", None)
            base = json.loads((ROOT / "prompts/color-knowledge/book-palette-ablation-v1.json").read_text(encoding="utf-8"))
            merged = {**base, **spec}
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "spec.json"
                path.write_text(json.dumps({**merged, "out_dir": str(Path(tmp) / "out")}), encoding="utf-8")
                with self.assertRaises(ValueError) as caught:
                    spatial_plan(path)
                self.assertIn(message, str(caught.exception))

    def test_guide_that_equals_the_content_source_stops(self):
        spec = json.loads(SPATIAL_SPEC.read_text(encoding="utf-8"))
        spec.pop("extends", None)
        base = json.loads((ROOT / "prompts/color-knowledge/book-palette-ablation-v1.json").read_text(encoding="utf-8"))
        merged = {**base, **spec}
        content = merged["content_reference"]
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.json"
            manifest.write_text(json.dumps({
                "role": BOOK_SPATIAL_GUIDE_ROLE,
                "source_path": content["path"],
                "source_sha256": content["sha256"],
                "canvas_size": [1264, 848],
                "locator_sha256": content["sha256"],
            }), encoding="utf-8")
            merged["spatial_guide"] = dict(merged["spatial_guide"], path=content["path"], sha256=content["sha256"],
                                           canvas_size=[1264, 848], manifest=str(manifest),
                                           manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
            path = Path(tmp) / "spec.json"
            path.write_text(json.dumps({**merged, "out_dir": str(Path(tmp) / "out")}), encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                spatial_plan(path)
        self.assertIn("different files", str(caught.exception))

    def test_variant_needs_content_reference_alongside_the_guide(self):
        spec = json.loads(SPATIAL_SPEC.read_text(encoding="utf-8"))
        spec.pop("extends", None)
        base = json.loads((ROOT / "prompts/color-knowledge/book-palette-ablation-v1.json").read_text(encoding="utf-8"))
        merged = {**base, **spec}
        merged["variants"] = copy.deepcopy(merged["variants"])
        merged["variants"]["source_locator"]["uses_content_reference"] = False
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spec.json"
            path.write_text(json.dumps({**merged, "out_dir": str(Path(tmp) / "out")}), encoding="utf-8")
            with self.assertRaises(ValueError) as caught:
                spatial_plan(path)
        self.assertIn("content source", str(caught.exception))

    def test_plan_freeze_is_stable_and_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with isolated(tmp), patch("utils.analysis_gpt_prompt.load_text_api_config", return_value=dict(TEXT_CFG)):
                frozen = freeze_book_ablation(str(SPATIAL_SPEC), tmp)
            self.assertEqual(frozen["plan_hash"], self.plan["plan_hash"])
            self.assertEqual(frozen["spatial_guide"]["sha256"], self.plan["spatial_guide"]["sha256"])


class BackendKwargTests(unittest.TestCase):
    def test_attachment_metadata_is_not_passed_to_the_backend(self):
        from modules.others import api_backend
        calls = []

        def fake_generate(**kwargs):
            calls.append(kwargs)
            return {"saved_files": [], "server_response_raw": {}}

        with tempfile.TemporaryDirectory() as tmp:
            with isolated(Path(tmp)), \
                 patch("utils.analysis_gpt_prompt.load_text_api_config", return_value=dict(TEXT_CFG)), \
                 patch.object(api_backend, "generate_image_aigc2d", side_effect=fake_generate), \
                 patch.object(api_backend, "get_api_config",
                              return_value={"api_key": "test", "base_url": "https://new.aigc2d.com/v1beta/models/"}):
                from utils.color_experiment import run_book_ablation
                run_book_ablation(str(SPATIAL_SPEC), tmp, log=lambda *_: None)
        self.assertTrue(calls)
        for kwargs in calls:
            self.assertNotIn("attachments", kwargs)
            self.assertNotIn("id", kwargs)
            self.assertNotIn("group_id", kwargs)
        by_prompt = {("SPATIAL GUIDE ROLES" in kwargs["prompt"], tuple(Path(p).name for p in kwargs["image_paths"]))
                     for kwargs in calls}
        self.assertIn((False, ("book-C13-simple-r1_105300-2c8d70.jpg",)), by_prompt)
        self.assertIn((True, ("book-C13-simple-r1_105300-2c8d70.jpg", "protected-region-locator.png")), by_prompt)


class PreflightTests(unittest.TestCase):
    def test_preflight_records_roles_and_sends_nothing(self):
        from modules.others import api_backend
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with isolated(root), \
                 patch("utils.analysis_gpt_prompt.load_text_api_config", return_value=dict(TEXT_CFG)), \
                 patch.object(api_backend, "get_api_config",
                              return_value={"api_key": "test", "base_url": "https://new.aigc2d.com/v1beta/models/"}):
                checks = book_preflight(str(SPATIAL_SPEC), tmp, log=lambda *_: None)
            self.assertTrue(checks["ok"])
            self.assertFalse(checks["sent"])
            self.assertTrue((root / "PREFLIGHT-CHECKS.json").is_file())
            self.assertEqual(checks["budget"]["planned_image_calls"], 4)
            self.assertEqual(checks["budget"]["planned_review_calls"], 2)
            roles = [row["role"] for row in checks["request_bodies"][0]["attachments"]]
            self.assertEqual(roles, ["content"] if len(roles) == 1 else ["content", "spatial_guide"])
            self.assertTrue(all(len(row["sha256"]) == 64 for entry in checks["request_bodies"] for row in entry["attachments"]))
            self.assertTrue(all(row["verified_on_disk"] for entry in checks["request_bodies"] for row in entry["attachments"]))
            self.assertEqual(checks["spatial_guide"]["sha256"],
                             "39ea2e9438dda924ae2eeed825dec25475eaafbe8612a949a0a214d6d4735dab")

    def test_preflight_stops_when_the_locator_disappears(self):
        from modules.others import api_backend
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = json.loads(SPATIAL_SPEC.read_text(encoding="utf-8"))
            spec.pop("extends", None)
            base = json.loads((ROOT / "prompts/color-knowledge/book-palette-ablation-v1.json").read_text(encoding="utf-8"))
            spec = {**base, **spec}
            spec["spatial_guide"] = dict(spec["spatial_guide"], path="data/test-result/20261005/nope/missing.png")
            spec["out_dir"] = str(root / "out")
            path = root / "spec.json"
            path.write_text(json.dumps(spec), encoding="utf-8")
            with isolated(root), \
                 patch("utils.analysis_gpt_prompt.load_text_api_config", return_value=dict(TEXT_CFG)), \
                 patch.object(api_backend, "get_api_config",
                              return_value={"api_key": "test", "base_url": "https://new.aigc2d.com/v1beta/models/"}):
                with self.assertRaises(ValueError):
                    book_preflight(str(path), str(root / "out"), log=lambda *_: None)


if __name__ == "__main__":
    unittest.main()
