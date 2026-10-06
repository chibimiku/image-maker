import torch  # Load before Qt/OpenVINO on Windows.
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from utils.style_metrics import gram, runner
from utils.style_metrics.config import GRAM_FORMULA_VERSION, GRAM_LEGACY_VERSION
from utils.style_metrics.pairs import compare_pairs


class GramRevisionTests(unittest.TestCase):
    def feature_pair(self):
        a = torch.tensor([[[[1., 2.], [3., 4.]], [[2., 0.], [1., 3.]]]])
        return a, a + 1

    def test_matches_raw_gram_paper_formula(self):
        a, b = self.feature_pair()
        raw_a, raw_b = a.flatten(2), b.flatten(2)
        delta = raw_a @ raw_a.transpose(1, 2) - raw_b @ raw_b.transpose(1, 2)
        expected = float(delta.double().square().sum() / (4 * 2**2 * 4**2))
        result = gram.gram_distance({"x": a}, {"x": b}, layers=["x"])
        self.assertAlmostEqual(result["distance"], expected, places=12)
        self.assertEqual(result["formula_version"], GRAM_FORMULA_VERSION)

    def test_normalized_and_raw_paths_equivalent(self):
        a, b = self.feature_pair()
        normalized = gram.gram_distance({"x": a}, {"x": b}, ["x"])
        raw = gram.gram_distance({"x": a}, {"x": b}, ["x"], normalize=False)
        self.assertAlmostEqual(normalized["distance"], raw["distance"], places=12)

    def test_legacy_reproduces_extra_channel_factor(self):
        a, b = self.feature_pair()
        new = gram.gram_distance({"x": a}, {"x": b}, ["x"])
        old = gram.gram_distance({"x": a}, {"x": b}, ["x"], formula_version=GRAM_LEGACY_VERSION)
        self.assertAlmostEqual(new["distance"], old["distance"] * a.shape[1]**2)

    def test_repeating_spatial_samples_does_not_reduce_distance(self):
        a, b = self.feature_pair()
        first = gram.gram_distance({"x": a}, {"x": b}, ["x"])["distance"]
        second = gram.gram_distance({"x": a.repeat(1, 1, 2, 2)}, {"x": b.repeat(1, 1, 2, 2)}, ["x"])["distance"]
        self.assertAlmostEqual(first, second, places=12)

    def test_mixed_layers_require_per_layer_conversion(self):
        a, b = self.feature_pair()
        fa, fb = {"x": a, "y": a.repeat(1, 2, 1, 1)}, {"x": b, "y": b.repeat(1, 2, 1, 1)}
        old = gram.gram_distance(fa, fb, fa.keys(), formula_version=GRAM_LEGACY_VERSION)
        new = gram.gram_distance(fa, fb, fa.keys())
        expected = sum(row["contribution"] * row["channels"]**2 for row in old["layers"].values())
        self.assertAlmostEqual(new["distance"], expected)

    def test_unknown_version_is_rejected(self):
        with self.assertRaises(ValueError):
            gram.gram_distance({}, {}, formula_version="unknown")

    def test_compact_gui_statistics_matches_v2(self):
        from modules.image_analysis.style_deep_comparison import statistics_distance
        a, b = self.feature_pair()
        measured, _ = statistics_distance(({"x": gram.gram_matrix(a)}, {}), ({"x": gram.gram_matrix(b)}, {}), "gram")
        self.assertAlmostEqual(measured, gram.gram_distance({"x": a}, {"x": b}, ["x"])["distance"])

    def test_runner_records_formula_version(self):
        a, b = self.feature_pair()
        self.assertEqual(runner.run_gram({"x": a}, {"x": b}, ["x"]).detail["formula_version"], GRAM_FORMULA_VERSION)


class PairModeTests(unittest.TestCase):
    def test_missing_dependency_is_unavailable_without_substitute(self):
        error = ModuleNotFoundError("No module named 'clip'", name="clip")
        outcome = runner.outcome_from_error("csd", error)
        self.assertEqual(outcome.status, "unavailable")
        self.assertIsNone(outcome.value)
        self.assertEqual(outcome.detail["missing_dependency"], "clip")

    def test_csd_wrapper_keeps_missing_dependency_unavailable(self):
        backend = SimpleNamespace(actual="cpu")
        error = ModuleNotFoundError("No module named 'clip'", name="clip")
        with patch("utils.style_metrics.csd_metric.compare", side_effect=error):
            outcome = runner.run_csd(None, None, backend)
        self.assertEqual(outcome.status, "unavailable")
        self.assertIsNone(outcome.value)

    def test_all_pairs_measured_without_false_identity_requirement(self):
        from PIL import Image
        backend = SimpleNamespace(actual="cpu", requested="cpu", backend="torch-cpu", precision="fp32")
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for i, color in enumerate(("red", "blue", "green")):
                path = Path(folder) / f"{i}.png"
                Image.new("RGB", (8, 8), color).save(path)
                paths.append(path)
            before = [p.read_bytes() for p in paths]
            def measure(a, b, _backend):
                return runner.MetricOutcome("lpips", "ok", 0.0 if a is b else 0.8)
            with patch("utils.style_metrics.inventory.preflight", return_value={}), patch("utils.style_metrics.runner.run_lpips", side_effect=measure):
                records = compare_pairs([(paths[0], paths[1]), (paths[1], paths[2])], ["lpips"], backend)
            self.assertEqual([row["value"] for row in records], [0.8, 0.8])
            self.assertTrue(all(row["status"] == "ok" for row in records))
            self.assertEqual([p.read_bytes() for p in paths], before)

    def test_missing_weights_have_no_fake_value(self):
        backend = SimpleNamespace(actual="cpu", requested="cpu", backend="torch-cpu", precision="fp32")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.bin"
            path.write_bytes(b"input")
            with patch("utils.style_metrics.inventory.preflight", return_value={"gram": [{"reason": "missing"}]}):
                row = compare_pairs([(path, path)], ["gram"], backend)[0]
            self.assertEqual(row["status"], "unavailable")
            self.assertIsNone(row["value"])

    def test_pair_cli_uses_separate_mode_and_honours_no_compare(self):
        from tools import style_metrics_verify as cli
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "result.json"
            with patch.object(cli, "run", return_value={"results": [], "summary": {"all_ok": True}}) as run:
                self.assertEqual(cli.main(["--pair", "a", "b", "--no-compare-cpu", "--out", str(output)]), 0)
                self.assertFalse(run.call_args.args[0].compare_cpu)


if __name__ == "__main__":
    unittest.main()
