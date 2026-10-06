import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils import color_experiment as ce


class TargetedRecheckTests(unittest.TestCase):
    def setUp(self):
        self.resolver = patch.object(ce, "_resolve_retention_out", side_effect=lambda value: Path(value).resolve())
        self.resolver.start()
        self.addCleanup(self.resolver.stop)
        self.spec = ce.BASE / "prompts/color-knowledge/fixed-palette-retention-v2.json"

    def test_authorization_required(self):
        with self.assertRaises(ValueError):
            ce.retention_recheck_run("unused")

    def test_three_calls_no_retries_and_success_reused(self):
        from requests import Response
        with tempfile.TemporaryDirectory() as folder:
            cfg = {"model": "mock", "base_url": "https://example.invalid/v1", "api_key": "test-key"}
            def respond(*args, **kwargs):
                self.assertFalse(kwargs["allow_redirects"])
                if "candidate_dimensions" in kwargs["json"]["messages"][1]["content"][0]["text"]:
                    value = {"candidate_shoes": [{"screen_side": side, "status": "contained"} for side in ("left", "right")],
                             "scope": "framing_diagnostic_only_not_production_gate"}
                else:
                    value = {"images": [{"id": q, "main_colour_status": "conform", "colour_status": "conform",
                                         "accent_waistband": "present", "accent_left_shoe_bow": "present",
                                         "accent_right_shoe_bow": "present", "protected_colours_status": "retained",
                                         "reference_influence": {"status": "none_observed", "causal_attribution": "not_established"}}
                                        for q in ("Q01", "Q02")]}
                response = Response()
                response.status_code = 200
                response._content = json.dumps({"choices": [{"message": {"content": json.dumps(value)}}],
                                                "usage": {"total_tokens": 42}}).encode()
                return response
            with patch.object(ce, "_text_cfg", return_value=cfg), patch("requests.Session.post", side_effect=respond) as post:
                first = ce.retention_recheck_run(folder, spec_path=self.spec, authorized=True, log=lambda *args: None)
                second = ce.retention_recheck_run(folder, spec_path=self.spec, authorized=True, log=lambda *args: None)
                self.assertEqual(post.call_count, 3)
                self.assertEqual(first["text_reservations"], 3)
                self.assertEqual(second["image_calls"], 0)
            request = json.loads((Path(folder) / "vision/colour-A.request.json").read_text(encoding="utf-8"))
            self.assertNotIn("test-key", json.dumps(request))
            self.assertEqual(len(request["payload"]["messages"][1]["content"]), 6)

    def test_http_failure_preserved_without_retry(self):
        from requests import Response
        with tempfile.TemporaryDirectory() as folder:
            cfg = {"model": "mock", "base_url": "https://example.invalid/v1", "api_key": "test-key"}
            response = Response()
            response.status_code = 503
            response._content = b'{"error":"temporary failure"}'
            with patch.object(ce, "_text_cfg", return_value=cfg), patch("requests.Session.post", return_value=response) as post:
                result = ce.retention_recheck_run(folder, spec_path=self.spec, authorized=True, log=lambda *args: None)
                self.assertEqual(post.call_count, 3)
                self.assertTrue(all(row["status"] == "failed_or_unknown" for row in result["reviews"]))
                ce.retention_recheck_run(folder, spec_path=self.spec, authorized=True, log=lambda *args: None)
                self.assertEqual(post.call_count, 3)


if __name__ == "__main__":
    unittest.main()
