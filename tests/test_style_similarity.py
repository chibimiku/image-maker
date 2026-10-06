import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw
from utils import style_similarity as similarity
from utils.style_metrics import devices, inventory


class StyleSimilarityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.first = self.root / "图一.png"
        self.second = self.root / "图二.png"
        image = Image.new("RGB", (96, 64), "white")
        ImageDraw.Draw(image).rectangle((10, 10, 70, 50), fill="red", outline="black", width=3)
        image.save(self.first)
        Image.new("RGB", (96, 64), "blue").save(self.second)

    def tearDown(self):
        self.temp.cleanup()

    def unavailable(self):
        keys = ("vgg19", "lpips_alex", "lpips_alex_trunk", "csd", "clip_vit_l14")
        return patch.object(inventory, "list_inventory", return_value={k: {"sha256_matches": False} for k in keys})

    def test_local_identity_symmetry_and_difference(self):
        identity = similarity.local_pair(str(self.first), str(self.first))
        forward = similarity.local_pair(str(self.first), str(self.second))
        backward = similarity.local_pair(str(self.second), str(self.first))
        for m in similarity.LOCAL_METRICS:
            self.assertEqual(identity[m]["value"], 1)
            self.assertAlmostEqual(forward[m]["value"], backward[m]["value"])
            self.assertGreaterEqual(forward[m]["value"], 0)
            self.assertLessEqual(forward[m]["value"], 1)
        self.assertLess(forward["tone"]["value"], 1)

    def test_corrupt_local_image_keeps_errors(self):
        bad = self.root / "bad.png"
        bad.write_text("invalid")
        result = similarity.local_pair(str(bad), str(self.first))
        self.assertTrue(all(r["status"] == "error" and r["value"] is None for r in result.values()))

    def test_manifest_detects_changes_and_duplicate_ids(self):
        inputs = similarity.image_manifest([self.first], [self.second])
        inputs["candidates"].append(inputs["candidates"][0])
        with self.assertRaises(ValueError):
            similarity.validate_manifest(inputs)
        inputs["candidates"].pop()
        self.second.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            similarity.validate_manifest(inputs)

    def test_all_candidates_all_references_missing_never_zero(self):
        inputs = similarity.image_manifest([self.first, self.second], [self.first, self.second])
        with self.unavailable(), patch.object(inventory, "preflight", return_value={m: [] for m in similarity.METRICS}):
            result = similarity.compare_images(inputs, "cpu")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(result["rows"]), 2)
        for row in result["rows"]:
            self.assertEqual(len(row["pairs"]), 2)
            self.assertEqual(set(row["summary"]), set(similarity.ALL_METRICS))
            for m in similarity.METRICS:
                self.assertIsNone(row["summary"][m]["mean"])
            for m in similarity.LOCAL_METRICS:
                self.assertEqual(row["summary"][m]["count"], 2)
        self.assertEqual(result["metric_contract"]["version"], similarity.VERSION)

    def test_extraction_calls_shared_engine_and_keeps_state(self):
        from modules.image_analysis import style_deep_comparison as extraction
        state = self.root / "state.json"
        state.write_text(json.dumps({"dataset": {"images": [str(self.second)]},
            "test_images": {"round_1": {"generated_files": {"gemini_direct": [str(self.first)]}}}, "keep": 123}))
        with self.unavailable(), patch.object(inventory, "preflight", return_value={m: [] for m in similarity.METRICS}), patch.object(similarity, "compare_images", wraps=similarity.compare_images) as engine:
            extracted = extraction.compute(state, "cpu")
            direct = similarity.compare_images(similarity.image_manifest([self.first], [self.second]), "cpu")
        self.assertEqual(engine.call_count, 2)
        self.assertEqual(extracted["rows"][0]["pairs"], direct["rows"][0]["pairs"])
        self.assertEqual(json.loads(state.read_text(encoding="utf-8"))["keep"], 123)
        self.assertIn("明度层次", Path(extracted["report_path"]).read_text(encoding="utf-8"))

    def test_cli_similarity_uses_shared_engine(self):
        from tools.style_metrics_verify import main
        output = self.root / "cli.json"
        with self.unavailable(), patch.object(inventory, "preflight", return_value={m: [] for m in similarity.METRICS}), patch.object(similarity, "compare_images", wraps=similarity.compare_images) as engine:
            code = main(["--similarity", str(self.first), str(self.second), "--reference", str(self.first), "--device", "cpu", "--out", str(output)])
        self.assertEqual(code, 1)
        engine.assert_called_once()
        self.assertEqual(len(json.loads(output.read_text(encoding="utf-8"))["rows"][0]["pairs"]), 2)

    def test_ui_freezes_paths_and_uses_extraction_worker(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        from unittest.mock import MagicMock
        from modules.image_analysis import style_similarity_tab as tab
        app = QApplication.instance() or QApplication([])
        widget = tab.StyleSimilarityWidget()
        worker = MagicMock()
        try:
            widget.paths[0].setText(str(self.first))
            widget.paths[1].setText(str(self.second))
            with tempfile.TemporaryDirectory() as directory, patch.object(tab, "create_worker", return_value=worker) as create:
                # Redirect only the saved temporary state; no live GUI is started.
                with patch.object(tab, "atomic_json", wraps=tab.atomic_json) as save:
                    widget.start()
                state_path = create.call_args.args[0]
                state = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertEqual(state["dataset"]["images"], [str(self.second)])
                self.assertEqual(state["test_images"]["pair"]["generated_files"]["input"], [str(self.first)])
                self.assertFalse(widget.run_button.isEnabled())
                self.assertTrue(all(not b.isEnabled() for b in widget.choose_buttons))
                worker.start.assert_called_once()
                widget.cancel()
                worker.requestInterruption.assert_called_once()
                widget.stopped()
                self.assertTrue(widget.run_button.isEnabled())
        finally:
            widget.close()

    def test_ui_worker_and_eight_metrics(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        from modules.image_analysis.style_similarity_tab import StyleSimilarityWidget
        app = QApplication.instance() or QApplication([])
        widget = StyleSimilarityWidget()
        try:
            widget.paths[0].setText(str(self.first))
            widget.paths[1].setText(str(self.second))
            with self.unavailable(), patch.object(inventory, "preflight", return_value={m: [] for m in similarity.METRICS}):
                result = similarity.compare_images(similarity.image_manifest([self.first], [self.second]), "cpu")
            widget.completed(result, "")
            self.assertEqual(widget.table.rowCount(), 8)
            self.assertTrue(widget.export_button.isEnabled())
            widget.paths[0].setText(str(self.second))
            self.assertFalse(widget.export_button.isEnabled())
            self.assertEqual(widget.table.rowCount(), 0)
        finally:
            widget.close()


if __name__ == "__main__":
    unittest.main()
