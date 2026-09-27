"""Bounded final rescue without paid network calls."""
import tempfile
import os
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from utils.refine_quality import review_final_candidate, match_final_canvas


class FinalRescueTests(unittest.TestCase):
    def test_cached_rescue_publication_retains_task_hash(self):
        from utils.analysis_gen import publish_final_output
        with tempfile.TemporaryDirectory() as folder:
            source = str(Path(folder) / "final-rescue-1.png")
            Image.new("RGB", (8, 8), "white").save(source)
            result = publish_final_output(source, style_name="style", task_hash="681024b4",
                                          final_dir=str(Path(folder) / "published"))
            self.assertTrue(Path(result).name.startswith("681024b4_"))

    def test_two_pixel_rounding_preserves_content_without_stretch(self):
        with tempfile.TemporaryDirectory() as folder:
            source = str(Path(folder) / "source.png")
            Image.new("RGB", (1694, 2528), "red").save(source)
            result = match_final_canvas(source, (1696, 2528))
            with Image.open(result) as image:
                self.assertEqual(image.size, (1696, 2528))
                self.assertEqual(image.getpixel((0, 0)), (255, 0, 0))
            with self.assertRaises(RuntimeError):
                match_final_canvas(source, (1800, 2528))
    def test_gui_first_image_uses_isolated_checkpoint_directory(self):
        from modules.image_analysis.single_analyzer import GptImageGenWorkerThread
        worker = GptImageGenWorkerThread({"prompt": "test", "image_paths": []},
                                        analysis_result={"task_hash": "isolated"})
        with patch.dict(os.environ, {"IMAGE_MAKER_TEST_OUTPUT": "1"}), \
             patch("utils.generation_checkpoint.GenerationCheckpoint") as checkpoint, \
             patch("modules.others.api_backend.generate_image_aigc2d_gpt", return_value=[]) as generate:
            checkpoint.return_value.begin.return_value = (True, [])
            checkpoint.return_value.data = {}
            worker.run()
        directory = generate.call_args.kwargs["save_sub_dir"]
        self.assertIn(os.path.join("data", "test-result"), directory)
        self.assertTrue(os.path.isabs(directory))
        self.assertEqual(directory, os.path.dirname(worker.checkpoint_path))

    def test_repair_then_reaudit_and_preserve_size(self):
        with tempfile.TemporaryDirectory() as folder:
            source = str(Path(folder) / "source.png")
            fixed = str(Path(folder) / "fixed.png")
            Image.new("RGB", (2048, 2048)).save(source)
            Image.new("RGB", (2048, 2048)).save(fixed)
            defect = {"needs_refine": True, "structural_issues": [
                {"region": "left hand", "observed": "extra finger", "repair": "remove extra finger", "confidence": .9}]}
            with patch("utils.gpt_image_optimize.load_config", return_value={"final_candidate_repair": {"model": "test-model"}}), \
                 patch("modules.others.api_backend.generate_image_repaint", return_value=[fixed]) as repaint:
                result, _ = review_final_candidate(source, lambda p: defect if p == source else {}, folder)
                self.assertEqual(result, fixed)
                self.assertEqual(repaint.call_args.kwargs["resolution"], "2K")
                self.assertEqual(repaint.call_args.kwargs["repeat"], 1)
                self.assertEqual(repaint.call_args.args[0], [source])
                self.assertTrue((Path(folder) / "final-repair-1.txt").exists())

    def test_uncertain_anatomy_does_not_generate(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("modules.others.api_backend.generate_image_repaint") as repaint:
                with self.assertRaises(RuntimeError):
                    review_final_candidate("unused", lambda p: {"needs_review": True}, folder)
                repaint.assert_not_called()


if __name__ == "__main__":
    unittest.main()
