import copy
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from utils import refine_atmosphere as atmosphere
from modules.image_analysis import single_analyzer as analyzer

SOURCE = {"english_description": "One person in a blue dress beside a window.",
          "short_description": "One person in a blue dress.", "pixiv_tags": ["ドレス"],
          "booru-tags": ["blue_dress"], "aspect_ratio": "16:9"}


class AtmosphereTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_off_has_no_template_io_and_no_changes(self):
        source = copy.deepcopy(SOURCE)
        with patch.object(atmosphere, "catalog", side_effect=AssertionError("off read catalog")):
            self.assertEqual(atmosphere.freeze(), {})
            self.assertIs(atmosphere.apply_result(source), source)
        self.assertEqual(source, SOURCE)

    def test_options_independent_and_snapshot_immutable(self):
        for selection, pages in (({"mood": "relaxed"}, [53]), ({"season": "spring"}, [37]),
                                 ({"mood": "tense", "season": "winter"}, [37, 53])):
            plan = atmosphere.freeze(selection)
            self.assertEqual(plan["source_pages"], pages)
            self.assertEqual(atmosphere.freeze(plan), plan)
            broken = copy.deepcopy(plan)
            broken["prompt"] += " altered"
            with self.assertRaises(ValueError):
                atmosphere.freeze(broken)
        with self.assertRaises(ValueError):
            atmosphere.freeze({"season": "invented"})

    def test_application_preserves_source_and_tags_and_is_idempotent(self):
        source = copy.deepcopy(SOURCE)
        source["original_english_description"] = SOURCE["english_description"]
        plan = atmosphere.freeze({"mood": "relaxed", "season": "spring"})
        atmosphere.apply_result(source, plan)
        once = copy.deepcopy(source)
        atmosphere.apply_result(source, plan)
        self.assertEqual(source, once)
        self.assertEqual(source["factual_english_description"], SOURCE["english_description"])
        self.assertEqual(source["original_english_description"], SOURCE["english_description"])
        self.assertEqual(source["pixiv_tags"], SOURCE["pixiv_tags"])
        self.assertIn("palette choices", source["english_description"])
        self.assertIn("do not add snow", source["english_description"])

    def test_short_anchor_uses_small_appendix_without_truncating_identity(self):
        plan = atmosphere.freeze({"mood": "melancholic", "season": "autumn"})
        source = {"gpt_image_prompt_short": "x" * 500}
        atmosphere.apply_result(source, plan, fields=("gpt_image_prompt_short",))
        self.assertTrue(source["gpt_image_prompt_short"].startswith("x" * 500))
        self.assertLess(len(source["gpt_image_prompt_short"]), 680)
        self.assertNotIn(plan["prompt"], source["gpt_image_prompt_short"])

    def request(self, selection=None, response=None):
        requests = []
        def create(**kwargs):
            requests.append(kwargs)
            if response is not None:
                return response
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=
                SimpleNamespace(content=json.dumps(SOURCE), refusal=None))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with tempfile.TemporaryDirectory() as root, patch.object(analyzer, "BASE_DIR", root):
            result = analyzer.step_2_refine_description(copy.deepcopy(SOURCE), client, "test", atmosphere=selection)
            from pathlib import Path
            diagnostic = json.loads(next(Path(root).rglob("*.json")).read_text(encoding="utf-8"))
        return result, requests, diagnostic

    def test_refine_request_has_separate_direction_and_frozen_diagnostic(self):
        result, requests, diagnostic = self.request({"season": "spring"})
        self.assertEqual(len(requests), 1)
        self.assertIn("不是原图事实", requests[0]["messages"][1]["content"])
        self.assertEqual(result["english_description"], SOURCE["english_description"])
        self.assertEqual(result["generation_atmosphere"]["selection"]["season"], "spring")
        self.assertEqual(diagnostic["step1_result"], SOURCE)
        self.assertEqual(diagnostic["request"], requests[0])

    def test_off_refine_has_no_atmosphere_field(self):
        result, requests, _ = self.request()
        self.assertNotIn("generation_atmosphere", result)
        self.assertNotIn("User-selected generation atmosphere", requests[0]["messages"][1]["content"])

    def test_refusal_still_stops(self):
        response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="content_filter", message=
            SimpleNamespace(content=None, refusal="blocked"))])
        result, requests, diagnostic = self.request({"mood": "pleasant"}, response)
        self.assertIsNone(result)
        self.assertEqual(len(requests), 1)
        self.assertTrue(diagnostic["error"])

    def test_worker_freezes_choice_and_keeps_it_through_later_stages(self):
        selection = {"mood": "relaxed"}
        worker = analyzer.WorkerThread("source.png", "fake", "http://unused", "fake", atmosphere=selection,
                                       enable_outfit_check=True, gpt_prompts=True, use_fallback=False)
        selection["mood"] = "tense"
        results = []
        worker.finish_signal.connect(results.append)
        with patch.object(analyzer, "OpenAI"), patch.object(worker, "_log_fallback_plan"), \
             patch.object(analyzer, "predict_local_booru_tags", return_value=[]), \
             patch.object(analyzer, "get_local_pixiv_tag_candidates", return_value=[]), \
             patch.object(analyzer, "step_1_analyze_image", return_value=copy.deepcopy(SOURCE)) as step1, \
             patch.object(analyzer, "step_2_refine_description", return_value=copy.deepcopy(SOURCE)) as step2, \
             patch.object(analyzer, "step_3_check_outfit_consistency", return_value=copy.deepcopy(SOURCE)), \
             patch.object(analyzer, "step_5_recompute_pixiv_tags", return_value=copy.deepcopy(SOURCE)), \
             patch.object(analyzer, "calculate_closest_aspect_ratio", return_value="16:9"), \
             patch("utils.analysis_gpt_prompt.build_gpt_image_prompt", return_value="Identity anchor.") as compact:
            worker.run()
        self.assertEqual(worker.last_status, "success")
        self.assertNotIn("atmosphere", step1.call_args.kwargs)
        self.assertEqual(step2.call_args.kwargs["atmosphere"]["selection"]["mood"], "relaxed")
        self.assertEqual(results[0]["pixiv_tags"], SOURCE["pixiv_tags"])
        self.assertIn("mood=relaxed", results[0]["gpt_image_prompt_short"])
        self.assertEqual(compact.call_args.args[0], SOURCE["english_description"])

    def test_widget_defaults_off_and_restores_independent_choices(self):
        from utils.refine_atmosphere_widget import RefineAtmosphereSelector
        widget = RefineAtmosphereSelector()
        self.assertEqual(widget.snapshot(), {})
        widget.restore({"mood": "relaxed", "season": "spring"})
        self.assertEqual(widget.selection(), {"mood": "relaxed", "season": "spring"})
        widget.restore({"mood": "invalid", "season": "winter"})
        self.assertEqual(widget.selection(), {"mood": "", "season": "winter"})

    def test_headless_uses_the_same_frozen_plan(self):
        from modules.image_analysis import analysis_pipeline as pipeline
        cfg = {"base_url": "http://unused", "api_key": "fake", "model": "fake",
               "refine_atmosphere": {"mood": "relaxed", "season": "winter"}}
        with patch("modules.others.api_backend.apply_secret_env_overrides", side_effect=lambda c: c), \
             patch.object(pipeline, "OpenAI"), \
             patch.object(pipeline, "predict_local_booru_tags", return_value=[]), \
             patch.object(pipeline, "get_local_pixiv_tag_candidates", return_value=[]), \
             patch.object(pipeline, "step_1_analyze_image", return_value=copy.deepcopy(SOURCE)), \
             patch.object(pipeline, "step_2_refine_description", return_value=copy.deepcopy(SOURCE)) as refine, \
             patch.object(pipeline, "calculate_closest_aspect_ratio", return_value="16:9"):
            result = pipeline.analyze_single_image("source.png", cfg, compute_gpt_prompts=False)
        self.assertIn("relaxed", result["english_description"])
        self.assertEqual(result["generation_atmosphere"], refine.call_args.kwargs["atmosphere"])
        self.assertEqual(result["pixiv_tags"], SOURCE["pixiv_tags"])
        self.assertNotIn("gpt_image_prompt", result)


if __name__ == "__main__":
    unittest.main()
