"""Offline coverage for CLI result-log pickup and publication."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.analysis_gpt_run import _history_entries, run_history_command


class HistoryCliTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.process = self.root / "data" / "20260928" / "analysis-gpt-image" / "run"
        self.process.mkdir(parents=True)
        self.first = self.process / "65ebea5b_first.png"
        self.candidate = self.process / "65ebea5b-final-rescue.jpg"
        self.first.write_bytes(b"first")
        self.candidate.write_bytes(b"candidate")
        self.checkpoint = self.process / "generation-checkpoint.json"
        self.checkpoint.write_text(json.dumps({
            "snapshot": {"file_prefix": "65ebea5b", "request_payload": {"style_name": "test"}},
            "status": "error", "current_stage": "final_review",
            "last_outputs": [str(self.candidate)],
            "stages": {"first": {"status": "success", "outputs": [str(self.first)]},
                       "hands": {"status": "success", "outputs": [str(self.first)]},
                       "final_review": {"status": "error"}},
            "operations": {"final_review-v2-repair-1": [str(self.candidate)]},
        }), encoding="utf-8")
        self.log = self.root / "analysis-cli-results.json"
        self.log.write_text(json.dumps({"version": 1, "kind": "analysis-cli-results",
            "items": [{"source": "source.png", "status": "error", "task_hash": "65ebea5b",
                       "checkpoints": [str(self.checkpoint)]}]}), encoding="utf-8")

    def test_list_from_result_log_exposes_candidate_and_checkpoint(self):
        self.assertEqual(_history_entries(str(self.log))[0]["checkpoint"], str(self.checkpoint))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(run_history_command("list", str(self.log), json_output=True), 0)
        data = json.loads(output.getvalue())
        self.assertEqual(data[0]["status"], "error")
        self.assertTrue(any(item["path"] == str(self.candidate) and item["stage"] == "final_review"
                            for item in data[0]["artifacts"]))

    def test_resume_uses_selected_image_and_keeps_original_checkpoint(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run_history_command("list", str(self.log), json_output=True)
        candidates = json.loads(output.getvalue())[0]["artifacts"]
        pick = next(item["pick"] for item in candidates if item["path"] == str(self.candidate))
        with patch("tools.analysis_gpt_run.resume_app_checkpoint", return_value=0) as resume:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_history_command("resume", str(self.log), pick=pick), 0)
        branch = Path(resume.call_args.args[0])
        self.assertTrue(branch.is_file())
        self.assertEqual(json.loads(branch.read_text(encoding="utf-8"))["pickup_source"],
                         str(self.candidate))
        self.assertEqual(json.loads(self.checkpoint.read_text(encoding="utf-8"))["status"], "error")

    def test_manual_publish_uses_original_date(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run_history_command("list", str(self.log), json_output=True)
        pick = next(item["pick"] for item in json.loads(output.getvalue())[0]["artifacts"]
                    if item["path"] == str(self.candidate))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_history_command("publish", str(self.log), pick=pick), 0)
        finals = list(self.process.parent.parent.glob("65ebea5b_*.jpg"))
        self.assertEqual(len(finals), 1)
        self.assertEqual(finals[0].read_bytes(), b"candidate")

    def test_experiment_request_can_list_but_cannot_stage_resume(self):
        request = self.process / "request.json"
        request.write_text(json.dumps({"analysis_json": "analysis.json", "steps": {},
                                       "source": "source.png", "status": "review_required",
                                       "style": "test", "selected_output": str(self.candidate)}),
                           encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(run_history_command("list", str(request), json_output=True), 0)
        candidate = next(item for item in json.loads(output.getvalue())[0]["artifacts"]
                         if item["path"] == str(self.candidate))
        with self.assertRaisesRegex(ValueError, "没有逐阶段断点"):
            run_history_command("resume", str(request), pick=candidate["pick"])

    def test_resume_result_log_points_back_to_attempt_checkpoint(self):
        result_log = self.process / "resume-result-generation-checkpoint-pickup-1.json"
        result_log.write_text(json.dumps({"status": "error", "checkpoint": str(self.checkpoint),
                                          "pickup_source": str(self.candidate)}), encoding="utf-8")
        self.assertEqual(_history_entries(str(result_log))[0]["checkpoint"], str(self.checkpoint))


if __name__ == "__main__":
    unittest.main()
