"""App 二阶段色调/主次冻结包的离线回归：校验、预算、复用、暂停、0 联网预演。

全部用例都在临时根目录里跑（`IMAGE_MAKER_COLOUR_TEST_OUTPUT_ROOT`），
不读真实密钥、不发任何联网请求；网络入口被打桩成「一调就失败」。
"""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils import color_direction_pack as dp

NETWORK_IMAGE = "modules.others.api_backend.generate_image_aigc2d"
NETWORK_REVIEW = "utils.analysis_gpt_prompt.call_text_model"


def _boom(*_args, **_kwargs):
    raise AssertionError("本用例不允许联网")


class DirectionPackTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        os.environ["IMAGE_MAKER_COLOUR_TEST_OUTPUT_ROOT"] = str(self.root)
        self.out = self.root / "run"
        self.out.mkdir(parents=True, exist_ok=True)
        self.pack = dp.load_pack()

    def tearDown(self):
        os.environ.pop("IMAGE_MAKER_COLOUR_TEST_OUTPUT_ROOT", None)
        self._tmp.cleanup()

    # --- 离线校验 ---------------------------------------------------------

    def test_verify_offline_passes_and_writes_record(self):
        result = dp.verify_offline(self.out, log=lambda *_a: None)
        self.assertEqual(result["problems"], [], result["problems"])
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["pack_hash_ok"])
        self.assertEqual(len(result["slots"]), 10)
        self.assertTrue((self.out / "offline-verification.json").is_file())
        self.assertEqual(result["source_files"] and
                         {row["status"] for row in result["source_files"]}, {"match"})

    def test_recompile_matches_frozen_prompt_for_every_slot(self):
        from utils.theme_color import apply_theme_color, freeze_theme_color, validate_snapshot
        for slot in self.pack["slots"]:
            plan = freeze_theme_color(slot["selection"])
            self.assertEqual(plan, slot["color_plan"], slot["id"])
            validate_snapshot(slot["color_plan"])
            prompt = apply_theme_color(slot["content_prompt"], plan, image_paths=(),
                                       mode="generate", post_enabled=False)
            self.assertEqual(prompt, slot["prompt"], slot["id"])

    def test_frozen_pack_is_published_byte_identically(self):
        first = dp.publish_frozen_pack(self.out, log=lambda *_a: None)
        self.assertTrue(first["byte_identical"])
        source = (dp.BASE / dp.PACK_PATH).read_bytes()
        self.assertEqual((self.out / "FROZEN-PACK.json").read_bytes(), source)
        second = dp.publish_frozen_pack(self.out, log=lambda *_a: None)
        self.assertTrue(second["reused"])
        self.assertEqual((self.out / "FROZEN-PACK.json").read_bytes(), source)

    def test_pack_hash_matches_the_expected_value(self):
        self.assertEqual(dp.pack_digest(self.pack), dp.EXPECTED_PACK_HASH)
        self.assertEqual(self.pack["pack_hash"], dp.EXPECTED_PACK_HASH)

    def test_verify_flags_a_changed_prompt(self):
        pack = json.loads(json.dumps(self.pack))
        pack["slots"][0]["prompt_sha256"] = "0" * 64
        path = self.root / "tampered.json"
        path.write_text(json.dumps(pack, ensure_ascii=False), encoding="utf-8")
        result = dp.verify_offline(self.out / "tampered", pack_path=path, log=lambda *_a: None)
        self.assertNotEqual(result["status"], "passed")
        self.assertTrue(any("prompt_sha256" in item for item in result["problems"]))

    # --- 运行参数冻结 -----------------------------------------------------

    def test_freeze_runtime_reads_config_and_keeps_secrets_out(self):
        image_cfg = {"base_url": "https://new.aigc2d.com/v1beta/models/", "model": "gemini-3-pro-image-preview",
                     "default_aspect_ratio": "1:1", "resolution": "2K", "timeout": 120, "max_retries": 1,
                     "api_key": "SECRET-IMAGE-KEY", "_api_key_source": "env:IMAGE_MAKER_AIGC2D_API_KEY"}
        text_cfg = {"base_url": "https://new.aigc2d.com/v1", "model": "gpt-5.6-luna",
                    "api_key": "SECRET-TEXT-KEY"}
        with patch("modules.others.api_backend.get_api_config", return_value=image_cfg), \
                patch("utils.analysis_gpt_prompt.load_text_api_config", return_value=text_cfg):
            runtime = dp.freeze_runtime(self.out, self.pack, log=lambda *_a: None)
        self.assertEqual(runtime["status"], "frozen", runtime["problems"])
        self.assertEqual(runtime["generation"]["model"], "gemini-3-pro-image-preview")
        self.assertEqual(runtime["generation"]["aspect_ratio"], "1:1")
        self.assertEqual(runtime["generation"]["resolution"], "2K")
        self.assertEqual(runtime["review"]["model"], "gpt-5.6-luna")
        self.assertTrue(runtime["runtime_hash"])
        text = (self.out / "RUNTIME-FROZEN.json").read_text(encoding="utf-8")
        self.assertNotIn("SECRET-IMAGE-KEY", text)
        self.assertNotIn("SECRET-TEXT-KEY", text)
        self.assertEqual(dp.load_runtime(self.out)["runtime_hash"], runtime["runtime_hash"])

    # --- 0 联网预演 -------------------------------------------------------

    def test_mock_preflight_sends_nothing(self):
        with patch(NETWORK_IMAGE, side_effect=_boom), patch(NETWORK_REVIEW, side_effect=_boom):
            runtime = {"generation": {"model": "gemini-3-pro-image-preview", "resolution": "2K",
                                      "aspect_ratio": "1:1"},
                       "review": {"model": "gpt-5.6-luna", "endpoint_without_credentials": "https://new.aigc2d.com/v1",
                                  "max_completion_tokens": dp.REVIEW_MAX_TOKENS,
                                  "timeout_seconds": dp.REVIEW_TIMEOUT_SECONDS}}
            mock = dp.run_slots(self.out, self.pack, runtime, dry_run=True, log=lambda *_a: None)
            review = dp.review_groups(self.out, self.pack, runtime, dry_run=True, log=lambda *_a: None)
        self.assertEqual(mock["ledger"]["counters"]["image_sends"], 0)
        self.assertEqual(review["ledger"]["counters"]["review_sends"], 0)
        self.assertEqual(len(mock["results"]), 10)
        self.assertEqual(len(review["results"]), 4)
        self.assertTrue((self.out / "requests" / "S1-A" / "planned-request.json").is_file())

    # --- 成功复用 / 不重发 -------------------------------------------------

    def _write_ledger(self, ledger):
        dp.save_ledger(self.out, ledger)

    def _base_ledger(self, **slots):
        ledger = dp.load_ledger(self.out, self.pack)
        ledger["pack_hash"] = self.pack["pack_hash"]
        ledger["slots"] = slots
        return ledger

    def test_generated_slot_is_reused_without_sending(self):
        from PIL import Image
        target = self.out / "images" / "S1-A-1.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), (90, 120, 150)).save(target)
        ledger = self._base_ledger(**{"S1-A": {"slot_id": "S1-A", "status": "generated", "attempts": 1,
                                               "outputs": [{"path": str(target),
                                                            "sha256": dp._sha256_file(target)}]}})
        self._write_ledger(ledger)
        with patch(NETWORK_IMAGE, side_effect=_boom) as spy:
            result = dp.run_slots(self.out, self.pack, {"generation": {}}, only={"S1-A"}, log=lambda *_a: None)
        self.assertFalse(spy.called)
        self.assertEqual(result["results"][0]["status"], "generated")
        self.assertTrue(result["results"][0]["reused"])
        self.assertEqual(result["ledger"]["counters"]["image_sends"], 0)

    def test_generated_slot_with_missing_file_is_not_silently_reused(self):
        ledger = self._base_ledger(**{"S1-A": {"slot_id": "S1-A", "status": "generated", "attempts": 1,
                                               "outputs": [{"path": str(self.out / "images" / "gone.png"),
                                                            "sha256": "0" * 64}]}})
        self._write_ledger(ledger)
        with patch(NETWORK_IMAGE, return_value={"saved_files": [], "server_response_raw": {"candidates": []}}):
            result = dp.run_slots(self.out, self.pack, {"generation": {}}, only={"S1-A"}, log=lambda *_a: None)
        self.assertNotEqual(result["results"][0].get("reused"), True)

    def test_failed_slot_is_not_resent(self):
        ledger = self._base_ledger(**{"S1-B": {"slot_id": "S1-B", "status": "failed", "attempts": 1,
                                               "error_type": "no_image_in_response"}})
        self._write_ledger(ledger)
        with patch(NETWORK_IMAGE, side_effect=_boom) as spy:
            result = dp.run_slots(self.out, self.pack, {"generation": {}}, only={"S1-B"}, log=lambda *_a: None)
        self.assertFalse(spy.called)
        self.assertEqual(result["results"][0]["status"], "failed")
        self.assertEqual(result["ledger"]["counters"]["image_sends"], 0)

    def test_sending_or_unknown_state_pauses_and_never_resends(self):
        for state in ("sending", "unknown_after_send"):
            out = self.root / ("pause-" + state)
            out.mkdir(parents=True, exist_ok=True)
            ledger = self._base_ledger(**{"S1-C": {"slot_id": "S1-C", "status": state, "attempts": 1}})
            dp.save_ledger(out, ledger)
            with patch(NETWORK_IMAGE, side_effect=_boom) as spy:
                result = dp.run_slots(out, self.pack, {"generation": {}}, only={"S1-C"}, log=lambda *_a: None)
            self.assertFalse(spy.called)
            self.assertTrue(result["paused"])
            self.assertEqual(result["ledger"]["counters"]["image_sends"], 0)

    def test_image_budget_stops_further_sends(self):
        ledger = self._base_ledger()
        ledger["counters"]["image_sends"] = 10
        self._write_ledger(ledger)
        with patch(NETWORK_IMAGE, side_effect=_boom) as spy:
            result = dp.run_slots(self.out, self.pack, {"generation": {}}, only={"S1-A"}, log=lambda *_a: None)
        self.assertFalse(spy.called)
        self.assertEqual(result["ledger"]["counters"]["image_sends"], 10)

    def test_authorize_resend_is_explicit_and_keeps_history(self):
        ledger = self._base_ledger(**{"S1-A": {"slot_id": "S1-A", "status": "unknown_after_send",
                                               "attempts": 1, "error_type": "transport_error"}})
        ledger["paused"] = {"slot_id": "S1-A", "status": "unknown_after_send", "reason": "x"}
        self._write_ledger(ledger)
        entry = dp.authorize_resend(self.out, "S1-A", note="用户授权：后端修好后补发", log=lambda *_a: None)
        self.assertEqual(entry["status"], "planned")
        self.assertEqual(entry["attempts"], 1)
        reloaded = dp.load_ledger(self.out, self.pack)
        self.assertEqual(reloaded["counters"]["authorized_extra_sends"], 1)
        self.assertIsNone(reloaded["paused"])
        self.assertEqual(reloaded["previous_attempts"][0]["status"], "unknown_after_send")
        self.assertEqual(reloaded["previous_attempts"][0]["moved_reason"], "human_authorised_resend")
        # 授权重发后，有效预算 = 冻结包预算 + 1，槽位可以真的再发一次
        with patch(NETWORK_IMAGE, return_value={"saved_files": [], "server_response_raw": {"candidates": []}}) as spy:
            dp.run_slots(self.out, self.pack, {"generation": {}}, only={"S1-A"}, log=lambda *_a: None)
        self.assertTrue(spy.called)
        self.assertEqual(dp.load_ledger(self.out, self.pack)["slots"]["S1-A"]["attempts"], 2)

    def test_authorize_resend_needs_a_reason_and_a_sent_state(self):
        ledger = self._base_ledger(**{"S1-B": {"slot_id": "S1-B", "status": "planned", "attempts": 0}})
        self._write_ledger(ledger)
        with self.assertRaises(ValueError):
            dp.authorize_resend(self.out, "S1-B", note="", log=lambda *_a: None)
        with self.assertRaises(ValueError):
            dp.authorize_resend(self.out, "S1-B", note="随便发", log=lambda *_a: None)

    # --- 评审 -------------------------------------------------------------

    def test_review_groups_without_images_are_skipped(self):
        with patch(NETWORK_REVIEW, side_effect=_boom) as spy:
            result = dp.review_groups(self.out, self.pack, {"review": {}}, log=lambda *_a: None)
        self.assertFalse(spy.called)
        self.assertEqual(result["ledger"]["counters"]["review_sends"], 0)
        self.assertTrue(all(row["status"] == "skipped_no_images" for row in result["results"]))

    def test_completed_review_is_reused(self):
        group = self.pack["review_groups"][0]
        folder = self.out / "reviews" / group["id"]
        folder.mkdir(parents=True, exist_ok=True)
        rules_hash = dp._sha256_text((dp.BASE / dp.REVIEW_PROMPT_PATH).read_text(encoding="utf-8"))
        (folder / "review.json").write_text(json.dumps(
            {"review_id": group["id"], "rules_hash": rules_hash, "response": {"images": []},
             "neutral_to_slot": {}}, ensure_ascii=False), encoding="utf-8")
        ledger = dp.load_ledger(self.out, self.pack)
        ledger["pack_hash"] = self.pack["pack_hash"]
        ledger["reviews"] = {group["id"]: {"review_id": group["id"], "status": "reviewed", "attempts": 1}}
        dp.save_ledger(self.out, ledger)
        with patch(NETWORK_REVIEW, side_effect=_boom) as spy:
            result = dp.review_groups(self.out, self.pack, {"review": {}}, only={group["id"]},
                                      log=lambda *_a: None)
        self.assertFalse(spy.called)
        self.assertTrue(result["results"][0]["reused"])
        self.assertEqual(result["ledger"]["counters"]["review_sends"], 0)

    def test_failed_review_is_not_resent(self):
        group = self.pack["review_groups"][0]
        ledger = dp.load_ledger(self.out, self.pack)
        ledger["pack_hash"] = self.pack["pack_hash"]
        ledger["reviews"] = {group["id"]: {"review_id": group["id"], "status": "parse_error", "attempts": 1}}
        dp.save_ledger(self.out, ledger)
        with patch(NETWORK_REVIEW, side_effect=_boom) as spy:
            result = dp.review_groups(self.out, self.pack, {"review": {}}, only={group["id"]},
                                      log=lambda *_a: None)
        self.assertFalse(spy.called)
        self.assertEqual(result["results"][0]["status"], "parse_error")
        self.assertEqual(result["ledger"]["counters"]["review_sends"], 0)

    # --- 证据提取与状态 ---------------------------------------------------

    def test_http_evidence_extraction_matches_the_sent_prompt(self):
        prompt = self.pack["slots"][0]["prompt"]
        payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                   "generationConfig": {"imageConfig": {"aspectRatio": "1:1", "imageSize": "2K"}}}
        log_file = self.root / "fake.log"
        log_file.write_text(
            "[2026-10-05 12:00:00] INFO - === 发起 AIGC2D API 请求 ===\n"
            "[2026-10-05 12:00:00] INFO - 请求 URL: https://new.aigc2d.com/v1beta/models/x:generateContent\n"
            "[2026-10-05 12:00:00] INFO - 请求 Headers: {\"x-goog-api-key\": \"abcd1234...wxyz\"}\n"
            "[2026-10-05 12:00:00] INFO - 请求数据:\n"
            + json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
            "[2026-10-05 12:00:01] INFO - 响应: 200\n", encoding="utf-8")
        evidence = dp._http_evidence("S1-A", prompt, log_file=log_file)
        self.assertEqual(evidence["status"], "captured")
        self.assertTrue(evidence["prompt_matches_sent_request"])
        self.assertEqual(evidence["image_config"]["imageSize"], "2K")
        self.assertNotIn("abcd1234", json.dumps(evidence, ensure_ascii=False))

    def test_status_reports_remaining_budget(self):
        ledger = dp.load_ledger(self.out, self.pack)
        ledger["pack_hash"] = self.pack["pack_hash"]
        ledger["counters"]["image_sends"] = 3
        ledger["counters"]["review_sends"] = 1
        dp.save_ledger(self.out, ledger)
        state = dp.status(self.out, self.pack)
        self.assertEqual(state["remaining_image_budget"], 7)
        self.assertEqual(state["remaining_review_budget"], 3)
        self.assertEqual([row["status"] for row in state["slots"]], ["planned"] * 10)

    def test_output_directory_must_stay_under_test_result(self):
        with self.assertRaises(ValueError):
            dp._resolve_out(dp.BASE / "data" / "20261005" / "not-allowed")


if __name__ == "__main__":
    unittest.main()
