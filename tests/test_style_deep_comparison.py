import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from modules.image_analysis.style_deep_comparison import aggregate, input_manifest, publish, statistics_distance, compute, resolve_backend, LIMITS
from utils.style_metrics import gram, adain


class DeepComparisonTests(unittest.TestCase):
    def test_report_keeps_partial_visual_gates_and_pending_candidates_visible(self):
        from modules.image_analysis.style_deep_comparison import write_report
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "reference.png"
            Image.new("RGB", (16, 16), "red").save(image)
            result = {"inputs": {"references": [{"path": str(image)}]}, "rows": [
                {"id": name, "path": str(image), "summary": {m: {"mean": 1} for m in ("gram", "adain", "lpips", "csd")}} for name in ("copied", "pending")]}
            state = {"automatic_comparison_status": {"status": "failed", "evaluated": 1, "expected": 2,
                "partial_assessments": [{"id": "copied", "reference_content_copied": True, "reason": "复制了书本"}]}}
            report = Path(write_report(result, directory, state)).read_text(encoding="utf-8")
            for text in ("1/2", "未选出最佳版本", "排除：复制参考内容", "待复核", "原始参考数据", "reference.png", "复制了书本"):
                self.assertIn(text, report)

    def test_statistics_match_deployed_formula(self):
        generator = torch.Generator().manual_seed(47)
        a = {k: torch.rand(1, c, 12, 9, generator=generator) for k, c in (("a", 4), ("b", 8))}
        b = {k: torch.rand(v.shape, generator=generator) for k, v in a.items()}
        def compact(features):
            return ({k: gram.gram_matrix(v) for k, v in features.items()}, {k: adain.feature_stats(v) for k, v in features.items()})
        for metric, run in (("gram", gram.gram_distance), ("adain", adain.adain_distance)):
            measured, _ = statistics_distance(compact(a), compact(b), metric)
            self.assertAlmostEqual(measured, run(a, b, layers=a.keys())["distance"], places=12)
            self.assertEqual(statistics_distance(compact(a), compact(a), metric)[0], 0)

    def test_missing_or_nonfinite_never_becomes_zero_mean(self):
        for outcome in ({"status": "unavailable", "value": None}, {"status": "ok", "value": float("nan")}):
            pairs = [{"csd": {"status": "ok", "value": .8}}, {"csd": outcome}]
            summary = aggregate(pairs, "csd", 2)
            self.assertIsNone(summary["mean"])
            self.assertEqual(summary["count"], 1)
        self.assertIsNone(aggregate([], "csd", 0)["mean"])
        complete = aggregate([{"csd": {"status": "ok", "value": v}} for v in (.6, .8)], "csd", 2)
        self.assertAlmostEqual(complete["mean"], .7)

    def test_publish_preserves_other_fields_and_detects_input_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (16, 16), "red").save(image)
            path = root / "state.json"
            state = {"dataset": {"images": [str(image)]}, "test_images": {"round_1": {"generated_files": {"gemini_direct": [str(image)]}}}, "unrelated": "preserve"}
            path.write_text(json.dumps(state), encoding="utf-8")
            inputs = input_manifest(state)
            result = {"version": "test", "backend": {"actual": "cpu", "precision": "fp32"}, "input_hash": "hash", "models": {}, "inputs": inputs, "rows": []}
            publish(path, inputs, result, root / "result.json")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["unrelated"], "preserve")
            report = Path(result["report_path"]).read_text(encoding="utf-8")
            self.assertIn("data:image/", report)
            self.assertIn(";base64,", report)
            self.assertIn("Gatys", report)
            before = path.read_bytes()
            Image.new("RGB", (16, 16), "blue").save(image)
            with self.assertRaises(ValueError):
                publish(path, inputs, result, root / "result.json")
            self.assertEqual(before, path.read_bytes())

    def test_ui_keeps_local_metrics_separate_and_locks_controls(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        from modules.image_analysis.style_analyzer import StyleAnalyzerWidget
        app = QApplication.instance() or QApplication([])
        widget = StyleAnalyzerWidget(lambda: ("", "", ""))
        try:
            self.assertEqual(widget.deep_device_combo.currentData(), "auto-npu")
            self.assertEqual({widget.deep_device_combo.itemData(i) for i in range(widget.deep_device_combo.count())}, {"auto-npu", "auto", "cuda", "npu", "cpu"})
            widget.set_running_state(True)
            self.assertFalse(widget.deep_btn.isEnabled())
            self.assertFalse(widget.deep_device_combo.isEnabled())
            widget.set_running_state(False)
            self.assertTrue(widget.deep_btn.isEnabled())
            widget._deep_result = {"report_path": "old-report"}
            widget.deep_table.setRowCount(1)
            widget.clear_imported_state()
            self.assertEqual(widget._deep_result, {})
            self.assertEqual(widget.deep_table.rowCount(), 0)
        finally:
            widget.close()

    def test_npu_priority_never_probes_cuda_if_npu_is_available(self):
        from utils.style_metrics import devices
        with patch.object(devices, "resolve_device", return_value=devices.Backend("npu", "npu", "openvino-npu", precision="fp16")) as resolve:
            backend = resolve_backend("auto-npu")
            resolve.assert_called_once_with("npu")
            self.assertEqual(backend.actual, "npu")
            self.assertEqual(backend.requested, "auto-npu")
        with patch.object(devices, "resolve_device", side_effect=[devices.DeviceUnavailableError("offline"), devices.Backend("cuda", "cuda", "torch-cuda")]) as resolve:
            backend = resolve_backend("auto-npu")
            self.assertEqual([c.args[0] for c in resolve.call_args_list], ["npu", "cuda"])
            self.assertTrue(backend.fallback)
            self.assertIn("offline", backend.note)

    def test_missing_weights_report_partial_without_fake_values(self):
        from utils.style_metrics import devices, inventory
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (12, 12), "red").save(image)
            path = root / "state.json"
            path.write_text(json.dumps({"dataset": {"images": [str(image)]}, "test_images": {"r": {"generated_files": {"gemini_direct": [str(image)]}}}}), encoding="utf-8")
            missing = {m: [{"reason": "test missing"}] for m in ("gram", "adain", "lpips", "csd")}
            weights = {k: {"sha256_matches": False} for k in ("vgg19", "lpips_alex", "lpips_alex_trunk", "csd", "clip_vit_l14")}
            with patch.object(devices, "resolve_device", return_value=devices.Backend("cpu", "cpu", "torch-cpu")), patch.object(inventory, "list_inventory", return_value=weights), patch.object(inventory, "preflight", return_value=missing):
                result = compute(path, "cpu")
            self.assertEqual(result["status"], "partial")
            for metric in missing:
                self.assertIsNone(result["rows"][0]["summary"][metric]["mean"])
                self.assertEqual(result["rows"][0]["pairs"][0][metric]["status"], "unavailable")

    def test_visual_report_shows_actual_deep_status_and_values(self):
        from modules.image_analysis.style_comparison import calculate_ranking, LEGACY_DIMENSIONS, write_comparison_report
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (12, 12), "red").save(image)
            ident = "round_1/gemini_direct/0"
            state = {"dataset": {"images": [str(image)]}, "test_images": {"round_1": {"generated_files": {"gemini_direct": [str(image)]}, "test_prompt": "girl"}}}
            score = {"id": ident, **{k: 8 for k in LEGACY_DIMENSIONS}, "subject_fidelity": 9, "reason": "test",
                     **{k: False for k in ("reference_content_copied", "major_structure_defect", "explicit_subject_mismatch", "uncertain")}}
            visual = calculate_ranking([{"id": ident, "path": str(image)}], [score], "style-comparison-v1")
            visual.update(model="test", input_hash="hash")
            state["deep_feature_comparison"] = {"report_path": str(root / "deep.html"), "backend": {"actual": "npu", "precision": "fp16"},
                "rows": [{"id": ident, "summary": {m: {"status": "ok", "mean": .25} for m in ("gram", "adain", "lpips", "csd")}}]}
            report = Path(write_comparison_report(state, visual, directory)).read_text(encoding="utf-8")
            self.assertIn("已计算 1/1", report)
            self.assertIn("npu / fp16", report)
            self.assertIn("<td>0.25</td>", report)
            self.assertIn("真实深度特征指标", report)


if __name__ == "__main__":
    unittest.main()
