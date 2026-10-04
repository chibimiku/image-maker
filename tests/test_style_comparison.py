import json
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
from PIL import Image
from modules.image_analysis.style_comparison import (DIMENSIONS, calculate_ranking,
    comparison_inputs, resume_seed, import_style_candidate, write_comparison_report, StyleComparisonWorker,
    LEGACY_DIMENSIONS)


class StyleComparisonTests(unittest.TestCase):
    def score(self, ident, value=8, **changes):
        return {"id": ident, **{key: value for key in DIMENSIONS}, "subject_fidelity": 10,
                "reference_content_copied": False, "major_structure_defect": False,
                "explicit_subject_mismatch": False, "uncertain": False, "reason": "清晰线条",
                "dimension_evidence": {key: {"reference": "源图外轮廓较强", "candidate": "候选外轮廓较强", "difference": "内线略重"} for key in DIMENSIONS}, **changes}

    def test_formula_gates_and_tie(self):
        candidates = [{"id": ident, "path": "x"} for ident in ("b", "a", "c")]
        result = calculate_ranking(candidates, [self.score("b"), self.score("a"), self.score("c", 10, reference_content_copied=True)])
        self.assertEqual(result["best_id"], "a")
        self.assertEqual(result["rows"][1]["total"], 8.3)
        self.assertFalse(result["rows"][0]["eligible"])

    def test_invalid_or_missing_scores_never_select(self):
        candidates = [{"id": "a", "path": "x"}]
        for scores in ([], [self.score("unknown")], [self.score("a", float("nan"))], [self.score("a", True)], [self.score("a", 11)]):
            with self.assertRaises(ValueError):
                calculate_ranking(candidates, scores)
        self.assertIsNone(calculate_ranking(candidates, [self.score("a", uncertain=True)])["best_id"])

    def test_hash_report_import_and_selected_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref, image = root / "ref.png", root / "output.png"
            Image.new("RGB", (20, 20), "red").save(ref)
            Image.new("RGB", (20, 20), "blue").save(image)
            entry = {"prompt": "GEMINI", "prompt_gpt": "GPT", "repaint_clauses": ["LINES"], "motif_clauses": ["OPTIONAL"]}
            record = {"generated_files": {"gemini_direct": [str(image)]}, "prompts_used": "ROUND ONE",
                      "style_reference": str(ref), "test_prompt": "girl beside lake",
                      "prompt_variants": {"gpt_image_prompt_valid": True, "style_entry": entry}}
            state = {"dataset": {"images": [str(ref)]}, "test_images": {"round_1": record}, "file_prefix": "new-style", "parameters": {"total_rounds": 3, "images_per_round": 4}}
            candidates, _, _, digest = comparison_inputs(state, "model", "endpoint")
            scores = [self.score(candidates[0]["id"])]
            result = calculate_ranking(candidates, scores)
            result.update(input_hash=digest, model="model", endpoint="endpoint", assessments=scores)
            state["automatic_comparison"] = result
            report = write_comparison_report(state, result, str(root))
            text = Path(report).read_text(encoding="utf-8")
            self.assertIn("data:image/png;base64", text)
            self.assertIn("ROUND ONE", text)
            self.assertIn("运行参数", text)
            self.assertIn("每轮检查图片数</th><td>4", text)
            self.assertIn("LoRA", text)
            self.assertIn("未计算", text)
            self.assertIn("源图外轮廓较强", text)
            seed, restored = resume_seed(state, candidates[0]["id"], "parent.json")
            self.assertEqual(seed["iterations"][-1]["art_style_prompts"], "ROUND ONE")
            from modules.image_analysis.style_analyzer import StyleIterativeWorkerThread
            worker = StyleIterativeWorkerThread([str(ref)], "key", "endpoint", "model", existing_state=seed)
            self.assertEqual(worker._get_current_prompts(seed["iterations"]), "ROUND ONE")
            self.assertNotIn("final_test_images", seed)
            config = root / "styles.json"
            config.write_text(json.dumps({"old": {"prompt": "PRESERVE"}}), encoding="utf-8")
            imported = import_style_candidate(state, candidates[0]["id"], "new-style", str(config), str(root / "refs"))
            self.assertEqual(imported["repaint_clauses"], ["LINES"])
            self.assertFalse(imported["motif_enabled"])
            self.assertTrue(Path(imported["ref_image"]).exists())
            self.assertEqual(json.loads(config.read_text())["old"]["prompt"], "PRESERVE")
            with self.assertRaises(ValueError):
                import_style_candidate(state, candidates[0]["id"], "new-style", str(config), str(root / "refs"))
            Image.new("RGB", (20, 20), "green").save(image)
            self.assertNotEqual(comparison_inputs(state, "model", "endpoint")[3], digest)

    def test_worker_cache_avoids_second_paid_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "ref.png"
            Image.new("RGB", (12, 12)).save(image)
            state = {"dataset": {"images": [str(image)]}, "test_images": {"round_1": {
                "generated_files": {"gemini_direct": [str(image)]}, "style_reference": str(image),
                "test_prompt": "a girl", "prompts_used": "BASE"}}}
            path = root / "state.json"
            path.write_text(json.dumps(state), encoding="utf-8")
            score = self.score("round_1/gemini_direct/0")
            response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"assessments": [score]})))])
            with patch("modules.image_analysis.style_comparison.OpenAI") as client:
                create = client.return_value.chat.completions.create
                create.return_value = response
                worker = StyleComparisonWorker(str(path), ("endpoint", "secret", "model"))
                results = []
                worker.completed.connect(lambda result, error: results.append((result, error)))
                worker.run()
                worker.run()
                self.assertEqual(create.call_count, 1)
                self.assertFalse(results[0][0]["cached"])
                self.assertTrue(results[1][0]["cached"])
                self.assertTrue(Path(results[1][0]["report_path"]).exists())
                self.assertEqual(results[1][1], "")

    def test_batched_comparison_recovers_completed_batches_and_exposes_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "ref.png"
            Image.new("RGB", (12, 12)).save(image)
            records = {f"round_{i}": {"generated_files": {"gemini_direct": [str(image)]},
                       "style_reference": str(image), "test_prompt": "girl", "prompts_used": f"ROUND {i}"} for i in range(5)}
            path = root / "state.json"
            path.write_text(json.dumps({"dataset": {"images": [str(image)]}, "test_images": records}), encoding="utf-8")
            requests = []
            def respond(**kwargs):
                content = kwargs["messages"][0]["content"]
                ids = [item["text"].splitlines()[0][10:] for item in content if item.get("text", "").startswith("CANDIDATE ")]
                requests.append(ids)
                self.assertLessEqual(len(ids), 4)
                self.assertEqual(sum(item.get("text", "").startswith("STYLE DATASET REFERENCE") for item in content), 1)
                if len(requests) == 2:
                    raise RuntimeError("temporary provider error")
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"assessments": [self.score(i) for i in ids]})))])
            with patch("modules.image_analysis.style_comparison.OpenAI") as client:
                create = client.return_value.chat.completions.create
                create.side_effect = respond
                worker = StyleComparisonWorker(str(path), ("endpoint", "key", "model"))
                completed = []
                worker.completed.connect(lambda result, error: completed.append((result, error)))
                worker.run()
                self.assertIn("temporary", completed[-1][1])
                worker.run()
                self.assertEqual(create.call_count, 3)
                self.assertEqual(len(completed[-1][0]["rows"]), 5)
                self.assertEqual(len(completed[-1][0]["tied_best_ids"]), 5)
                self.assertEqual(completed[-1][0]["selection_status"], "manual_review_tie")
                worker.run()
                self.assertEqual(create.call_count, 3)

    def test_gui_selection_and_resume_keep_source(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        from modules.image_analysis.style_analyzer import StyleAnalyzerWidget
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "ref.png"
            Image.new("RGB", (12, 12)).save(image)
            record = {"generated_files": {"gemini_direct": [str(image)]}, "style_reference": str(image),
                      "test_prompt": "a girl", "prompts_used": "SELECTED ROUND"}
            state = {"dataset": {"images": [str(image)]}, "test_images": {"round_2": record}, "file_prefix": "new-style"}
            path = root / "source.json"
            original = json.dumps(state)
            path.write_text(original, encoding="utf-8")
            widget = StyleAnalyzerWidget(lambda: ("endpoint", "secret", "model"))
            widget._loaded_json_path = str(path)
            result = calculate_ranking([{ "id": "round_2/gemini_direct/0", "path": str(image)}], [self.score("round_2/gemini_direct/0")])
            widget.on_comparison_completed(result, "")
            self.assertEqual(widget.selected_comparison_row()["id"], result["best_id"])
            self.assertTrue(widget.import_style_btn.isEnabled())
            self.assertIsNotNone(widget.image_list.parent())
            with patch("modules.image_analysis.style_analyzer.QInputDialog.getInt", return_value=(4, True)), patch.object(widget, "start_analysis") as start:
                widget.resume_selected_version()
                start.assert_called_once()
            self.assertEqual(widget.total_rounds_spin.value(), 4)
            self.assertEqual(widget._existing_state["iterations"][0]["art_style_prompts"], "SELECTED ROUND")
            self.assertNotEqual(widget._output_dir, str(root))
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            widget.close()

    def test_v2_group_weights_evidence_and_legacy_scores(self):
        candidate = [{"id": "a", "path": "x"}]
        score = self.score("a", 0, color_logic=10, subject_fidelity=0)
        result = calculate_ranking(candidate, [score])
        self.assertEqual(result["rows"][0]["style_score"], 2)
        self.assertEqual(result["rows"][0]["total"], 1.7)
        self.assertEqual(len(result["dimensions"]), 12)
        score.pop("dimension_evidence")
        with self.assertRaisesRegex(ValueError, "每个维度"):
            calculate_ranking(candidate, [score])
        legacy = {**self.score("a"), **{key: 8 for key in LEGACY_DIMENSIONS}}
        old = calculate_ranking(candidate, [legacy], version="style-comparison-v1")
        self.assertEqual(old["rows"][0]["total"], 8.3)
        self.assertEqual(old["dimensions"], list(LEGACY_DIMENSIONS))


if __name__ == "__main__":
    unittest.main()
