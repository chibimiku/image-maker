import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from utils import color_direction_pack as dp
from utils.request_replay import safe_request_snapshot, REDACTED


class EvidenceGateTests(unittest.TestCase):
    def observation(self):
        return {"content_status": "pass", "palette_status": "conform",
                "protected_hues_status": "pass", "detail_readability": "pass",
                "unauthorized_additions": [], "binding_observations": [
                    {"region": "mug", "region_id": "mug", "visible": "yes", "status": "conform"}]}

    def test_all_required_checks_and_regions(self):
        good = self.observation()
        self.assertTrue(dp._usability_gate(good, ["mug"])["usable"])
        for key in ("content_status", "palette_status", "protected_hues_status", "detail_readability"):
            for bad in ("uncertain", "fail", "partial", None):
                observation = copy.deepcopy(good)
                observation[key] = bad
                self.assertFalse(dp._usability_gate(observation)["usable"], (key, bad))
        self.assertFalse(dp._usability_gate(good, ["mug", "notebook"])["usable"])
        self.assertFalse(dp._usability_gate({})["usable"])

    def test_next_pack_recompiles_offline(self):
        stem = dp.BASE / "prompts/color-knowledge/theme-color-chroma-validation-v2"
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"IMAGE_MAKER_COLOUR_TEST_OUTPUT_ROOT": folder}):
            result = dp.verify_offline(Path(folder) / "out", pack_path=str(stem)+"-pack.json",
                                      scenes_path=str(stem)+"-scenes.json", review_path=str(stem)+"-review.md", log=lambda *_: None)
            self.assertEqual(result["problems"], [])
            self.assertEqual(len(result["slots"]), 6)

    def test_chroma_pair_decodes_without_brightness(self):
        result = dp._decode_tone([{"ids": ["N1", "N2"], "more_chromatic": "N2"}],
                                 {"N1": "S1-A", "N2": "S1-B"}, "S1-A", "S1-B", "more_chromatic")
        self.assertTrue(result["other_is_lighter"])

    def test_visible_direction_survives_failed_palette_but_is_not_success(self):
        pack = dp.load_pack()
        observation = self.observation()
        observation.update(id="N2", palette_status="partial")
        review = [{"review_id": "S1-tone", "status": "reviewed",
                   "neutral_to_slot": {"N1": "S1-A", "N2": "S1-C"},
                   "parsed": {"images": [observation], "tone_pairwise": [
                       {"ids": ["N1", "N2"], "lighter": "N1", "confounded": False}]}}]
        result = dp._decode_directions(pack, review, {})["deep"]
        self.assertEqual(result["observed_direction_count"], 1)
        self.assertEqual(result["candidate_usable_success"], 0)
        self.assertEqual(result["usable_success"], 0)

    def test_next_pack_preflight_and_report_send_nothing(self):
        pack = dp.load_pack(dp.BASE / "prompts/color-knowledge/theme-color-chroma-validation-v2-pack.json")
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"IMAGE_MAKER_COLOUR_TEST_OUTPUT_ROOT": folder}), \
                patch("modules.others.api_backend.generate_image_aigc2d", side_effect=AssertionError("network")), \
                patch("utils.analysis_gpt_prompt.call_text_model", side_effect=AssertionError("network")):
            runtime = {"generation": {}, "review": {}, "runtime_hash": "offline"}
            dp.run_slots(folder, pack, runtime, dry_run=True, log=lambda *_: None)
            dp.review_groups(folder, pack, runtime, dry_run=True, log=lambda *_: None)
            dp.render_report(folder, pack, runtime, log=lambda *_: None)
            results = json.loads((Path(folder) / "RESULTS.json").read_text(encoding="utf-8"))
            self.assertEqual(set(results["directions"]), {"clear", "muted"})
            self.assertEqual(results["counters"], {"image_sends": 0, "review_sends": 0})
            self.assertTrue((Path(folder) / "gallery.html").is_file())


class ReplayTests(unittest.TestCase):
    def test_no_credentials_persist_and_inputs_intact(self):
        headers = {"Authorization": "Bearer SECRET-A", "X-Goog-Api-Key": "SECRET-B", "Content-Type": "application/json"}
        body = {"messages": [{"text": "literal prompt", "api_key": "SECRET-C"}]}
        frozen = safe_request_snapshot("https://user:SECRET-D@example.org/x?key=SECRET-E&mode=generate", headers, body)
        text = json.dumps(frozen)
        for suffix in "ABCDE":
            self.assertNotIn("SECRET-"+suffix, text)
        self.assertEqual(headers["Authorization"], "Bearer SECRET-A")
        self.assertEqual(body["messages"][0]["api_key"], "SECRET-C")
        self.assertEqual(frozen["body"]["messages"][0]["text"], "literal prompt")
        self.assertEqual(frozen["headers"]["Authorization"], REDACTED)

    def test_backend_replay_requires_env_and_keeps_auth_out_of_stdout(self):
        from modules.others.api_backend import _generate_request_replay
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as folder:
            paths = _generate_request_replay(folder, "test", "gemini", "https://example.org/x",
                                              {"Authorization": "Bearer ORIGINAL-SECRET"}, {"prompt": "literal"})
            data = Path(paths["replay_json"]).read_text(encoding="utf-8")
            script = Path(paths["replay_script"]).read_text(encoding="utf-8")
            self.assertNotIn("ORIGINAL-SECRET", data+script)
            namespace = {"__file__": paths["replay_script"], "__name__": "test_replay"}
            exec(compile(script, paths["replay_script"], "exec"), namespace)
            with patch.dict(os.environ, {}, clear=True), patch("requests.post") as post:
                with self.assertRaises(SystemExit):
                    namespace["main"]()
                post.assert_not_called()
            with patch.dict(os.environ, {"IMAGE_MAKER_REPLAY_AUTHORIZATION": "RUNTIME-SECRET"}), patch("requests.post") as post:
                post.return_value.status_code = 200
                post.return_value.headers = {}
                post.return_value.text = "ok"
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    namespace["main"]()
                self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer RUNTIME-SECRET")
                self.assertNotIn("RUNTIME-SECRET", output.getvalue())


if __name__ == "__main__":
    unittest.main()
