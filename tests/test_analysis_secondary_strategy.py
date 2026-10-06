"""Offline end-to-end coverage for observe-then-secondary-edit/v1."""
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PyQt6.QtWidgets import QApplication
from modules.image_analysis import single_analyzer as module
from modules.image_analysis import analysis_pipeline as pipeline
from utils.analysis_secondary import secondary_text_config, create_secondary_text_client
from utils.analysis_fallback import FallbackConfig, is_refusal_text
from utils.llm_retry import RetrySettings


SOURCE = {
    "english_description": "A photograph of a female character in Lolita clothing. Her face is blurred and a hand covers one eye.",
    "short_description": "A female character with a blurred face.",
    "booru-tags": ["covered_face", "dress"],
    "pixiv_tags": ["ドレス"], "japanese_title": "窓辺", "chinese_title": "窗边",
}
EDITED = dict(SOURCE, english_description="An illustration of a female character in rococo-inspired fashion with clear facial rendering. A hand covers one eye.")
SECONDARY = {"base_url": "https://secondary.invalid/v1", "api_key": "secondary-secret", "model": "secondary-text"}


class FakeClient:
    def __init__(self, payload, *, finish_reason="stop", refusal=None):
        self.calls = []
        self.response = S(choices=[S(finish_reason=finish_reason,
                                    message=S(content=json.dumps(payload), refusal=refusal))])
        self.chat = S(completions=S(create=self.create))
        self.base_url = "https://primary.invalid/v1"

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class SecondaryStrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qapp = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "source.png"
        Image.new("RGB", (64, 32)).save(self.image)
        for target, value in (("BASE_DIR", str(self.root)),
                              ("predict_local_booru_tags", lambda *a, **k: []),
                              ("get_local_pixiv_tag_candidates", lambda *a, **k: []),
                              ("load_retry_settings", lambda: RetrySettings(enabled=False, times=0, interval_seconds=0))):
            patcher = patch.object(module, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_vision_retains_observed_occlusion_and_ignores_generation_extra(self):
        client = FakeClient(SOURCE)
        result = module.step_1_analyze_image(str(self.image), client, "vision-model",
                                             extra_llm_prompt="GENERATE_ONLY_OVERRIDE", use_fallback=False)
        self.assertEqual(result["english_description"], SOURCE["english_description"])
        self.assertEqual(result["booru-tags"], SOURCE["booru-tags"])
        prompt = client.calls[0]["messages"][1]["content"][0]["text"]
        self.assertNotIn("GENERATE_ONLY_OVERRIDE", prompt)
        self.assertNotIn("MUST imagine", prompt)
        self.assertNotIn("Always use the word 'girl'", prompt)
        self.assertIn("Do not convert photographs", prompt)

    def test_worker_routes_second_stage_to_secondary_text_only(self):
        primary, secondary = FakeClient(SOURCE), FakeClient(EDITED)
        created = []

        def factory(**kwargs):
            created.append(kwargs)
            return secondary if kwargs["base_url"] == SECONDARY["base_url"] else primary

        with patch.object(module, "OpenAI", factory):
            thread = module.WorkerThread(str(self.image), "primary-key", "https://primary.invalid/v1",
                                         "vision-model", secondary_config=SECONDARY,
                                         use_fallback=False, gpt_prompts=False)
            results = []
            thread.finish_signal.connect(results.append)
            thread.run()
        self.assertEqual(thread.last_status, "success")
        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(len(secondary.calls), 1)
        self.assertEqual(secondary.calls[0]["model"], SECONDARY["model"])
        self.assertTrue(all(isinstance(message["content"], str) for message in secondary.calls[0]["messages"]))
        result = results[0]
        self.assertEqual(result["original_english_description"], SOURCE["english_description"])
        self.assertEqual(result["source_analysis"]["english_description"], SOURCE["english_description"])
        self.assertEqual(result["booru-tags"], SOURCE["booru-tags"])
        self.assertEqual(result["analysis_strategy"], "observe-then-secondary-edit/v1")
        self.assertEqual(result["aspect_ratio"], "16:9")
        user_prompt = secondary.calls[0]["messages"][1]["content"]
        for clause in ("清晰面部绘制目标", "girl", "精确措辞", "rococo-inspired fashion"):
            self.assertIn(clause, user_prompt)
        snapshot = json.loads(next((self.root / "cache/temp/analysis-refine").glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(snapshot["step1_result"]["english_description"], SOURCE["english_description"])
        self.assertNotIn("secondary-secret", json.dumps(snapshot))

    def test_primary_soft_refusal_stops_without_editing(self):
        client = FakeClient({"english_description": "I cannot help with that request."})
        states = []
        self.assertIsNone(module.step_1_analyze_image(str(self.image), client, "vision",
                                                     use_fallback=False, status_callback=states.append))
        self.assertEqual(states, ["refused"])
        self.assertEqual(len(client.calls), 1)

    def test_soft_refusal_switches_to_backup_using_same_observation_request(self):
        primary = FakeClient({"english_description": "I cannot help with that request."})
        backup = FakeClient(SOURCE)
        backup.base_url = "https://backup.invalid/v1"
        cfg = FallbackConfig(enabled=True, base_url=backup.base_url, api_key="backup-key", model="backup-vision")
        with patch.object(module, "load_fallback_config", return_value=cfg), \
                patch("utils.analysis_fallback._make_client", return_value=backup):
            result = module.step_1_analyze_image(str(self.image), primary, "vision", use_fallback=True)
        self.assertEqual(result["english_description"], SOURCE["english_description"])
        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(len(backup.calls), 1)
        self.assertEqual(primary.calls[0]["messages"], backup.calls[0]["messages"])

    def test_ordinary_observation_is_not_misclassified_as_refusal(self):
        for description in ("A safety pin on a dress.", "Hair colour is unknown.", "A girl wears a safety helmet."):
            self.assertFalse(is_refusal_text(description), description)

    def test_secondary_refusal_preserves_source_diagnostic_and_stops(self):
        primary = FakeClient(SOURCE)
        secondary = FakeClient({"english_description": "I cannot help with that request."})
        with patch.object(module, "OpenAI", side_effect=[primary, secondary]):
            thread = module.WorkerThread(str(self.image), "key", "https://primary.invalid/v1", "vision",
                                         secondary_config=SECONDARY, use_fallback=False, gpt_prompts=False)
            results = []
            thread.finish_signal.connect(results.append)
            thread.run()
        self.assertEqual(thread.last_status, "refused")
        self.assertEqual(results, [{}])
        self.assertEqual(len(secondary.calls), 1)
        snapshot = json.loads(next((self.root / "cache/temp/analysis-refine").glob("*.json")).read_text(encoding="utf-8"))
        self.assertEqual(snapshot["step1_result"]["english_description"], SOURCE["english_description"])
        self.assertTrue(snapshot["error"])

    def test_missing_secondary_never_reuses_primary(self):
        primary = FakeClient(SOURCE)
        with patch.object(module, "OpenAI", return_value=primary) as factory:
            thread = module.WorkerThread(str(self.image), "key", "https://primary.invalid/v1", "vision",
                                         secondary_config={}, use_fallback=False, gpt_prompts=False)
            results = []
            thread.finish_signal.connect(results.append)
            thread.run()
        self.assertEqual(thread.last_status, "error")
        self.assertEqual(results, [{}])
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(len(primary.calls), 1)

    def test_headless_uses_secondary_from_supplied_config(self):
        primary, secondary = FakeClient(SOURCE), FakeClient(EDITED)
        config = {"base_url": "https://primary.invalid/v1", "api_key": "key", "model": "vision",
                  "nsfw_base_url": SECONDARY["base_url"], "nsfw_api_key": SECONDARY["api_key"],
                  "nsfw_model": SECONDARY["model"], "fallback_enabled": False}
        with patch.object(pipeline, "OpenAI", side_effect=[primary, secondary]), \
                patch.object(pipeline, "predict_local_booru_tags", return_value=[]), \
                patch.object(pipeline, "get_local_pixiv_tag_candidates", return_value=[]), \
                patch("modules.others.api_backend.apply_secret_env_overrides", side_effect=lambda cfg: dict(cfg)):
            result = pipeline.analyze_single_image(str(self.image), config, compute_gpt_prompts=False)
        self.assertIsNotNone(result)
        self.assertEqual(secondary.calls[0]["model"], SECONDARY["model"])
        self.assertEqual(result["original_english_description"], SOURCE["english_description"])

    def test_secondary_resolution_keeps_text_model_and_requires_own_fields(self):
        config = {"model": "vision", "api_key": "primary", "nsfw_base_url": SECONDARY["base_url"],
                  "nsfw_api_key": "secondary", "nsfw_model": "deepseek-text-only"}
        with patch("modules.others.api_backend.apply_secret_env_overrides", side_effect=lambda cfg: dict(cfg)):
            cfg = secondary_text_config(config)
        self.assertEqual(cfg["model"], "deepseek-text-only")
        self.assertEqual(cfg["api_key"], "secondary")
        with self.assertRaises(ValueError):
            create_secondary_text_client({}, client_factory=lambda **kwargs: self.fail("unexpected client"), timeout_seconds=30)


if __name__ == "__main__":
    unittest.main()
