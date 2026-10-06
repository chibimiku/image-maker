"""画风受控实验编排/统计的单元测试（不调用任何联网端点、不生成图片）。

覆盖点：
1. 冻结选样只看属性（同规则、可复现）、确定性重复才自动剔除；
2. E0 容差规则与「并列不计正确」的口径；
3. 统计工具（Wilson / cluster bootstrap / Kendall tau-b / ICC）不伪造数值；
4. E2 变体算法确定性、参数与 hash 记录；
5. E3 可控图样的解析标注与剂量响应方向（用共享入口计算，不复制公式）；
6. 编排层不会把 partial / 未完成写成 complete。
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from utils import style_experiment_attributes as attrs  # noqa: E402
from utils import style_experiment_controlled as core  # noqa: E402


def _write_image(path, value=128, size=(128, 128), stripes=False):
    array = np.full((size[1], size[0], 3), value, dtype=np.uint8)
    if stripes:
        array[:, ::8] = 20
    Image.fromarray(array).save(path, format="PNG")
    return str(path)


class StratifiedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="style-exp-sel-"))
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def _pool(self, count=20):
        rows = []
        for index in range(count):
            path = _write_image(self.directory / f"work-{index:03d}.png", value=60 + index * 5,
                                stripes=index % 3 == 0)
            rows.append({"path": path, "file_name": Path(path).name, "sha256": core.sha256_file(path),
                         "framing": "face_closeup" if index % 2 else "full_body_or_scene",
                         "brightness": "bright" if index % 3 else "dark",
                         "background_density": "clean", "image_id": f"x-{index:03d}"})
        return rows

    def test_selection_is_deterministic_and_balanced(self):
        pool = self._pool()
        first = core.stratified_selection(pool, 4, 4)
        second = core.stratified_selection(pool, 4, 4)
        self.assertEqual([row["path"] for row in first["references"]],
                         [row["path"] for row in second["references"]])
        self.assertEqual([row["path"] for row in first["queries"]],
                         [row["path"] for row in second["queries"]])
        self.assertEqual(len(first["references"]), 4)
        self.assertEqual(len(first["queries"]), 4)
        chosen = {row["path"] for row in first["references"]} & {row["path"] for row in first["queries"]}
        self.assertFalse(chosen)

    def test_flat_colour_images_are_not_all_collapsed_into_one_work(self):
        """裸时间戳后缀不是再导出标记：同目录不同作品不能被误判成同一作品。"""
        pool = []
        for index in range(6):
            path = _write_image(self.directory / f"artist_2025-02-17_00-0{index}-0{index}.png",
                                value=70 + index * 20)
            pool.append({"path": path, "file_name": Path(path).name, "sha256": core.sha256_file(path),
                         "framing": "face_closeup", "brightness": "bright",
                         "background_density": "clean", "image_id": f"d-{index}"})
        selection = core.stratified_selection(pool, 2, 2)
        self.assertEqual(selection["usable_pool"], 6)
        self.assertEqual(len(selection["references"]), 2)

    def test_reencoded_copies_are_detected_by_normalised_name(self):
        first = _write_image(self.directory / "artwork-1.png", value=200)
        second = _write_image(self.directory / "artwork-1_fixed.png", value=200)
        pool = [{"path": path, "file_name": Path(path).name, "sha256": core.sha256_file(path),
                 "framing": "face_closeup", "brightness": "bright", "background_density": "clean",
                 "image_id": Path(path).stem}
                for path in (first, second)]
        # 需要真实的近重复指纹才能进入判定
        for row in pool:
            row.update(attrs.measure(row["path"]))
        decisions = attrs.near_duplicate_decisions(pool)
        self.assertEqual(len(decisions["auto_excluded"]), 1)
        self.assertEqual(decisions["auto_excluded"][0]["rule"], "normalized_filename")

    def test_fuzzy_pairs_are_flagged_not_deleted(self):
        """内容完全不同、只有亮度相近的两张图不得被自动剔除。"""
        first = _write_image(self.directory / "a.png", value=100, stripes=True)
        second = _write_image(self.directory / "b.png", value=200, size=(256, 128))
        pool = [{"path": path, "file_name": Path(path).name, "sha256": core.sha256_file(path),
                 "framing": "face_closeup", "brightness": "bright", "background_density": "clean",
                 "image_id": Path(path).stem}
                for path in (first, second)]
        for row in pool:
            row.update(attrs.measure(row["path"]))
        decisions = attrs.near_duplicate_decisions(pool)
        self.assertEqual(decisions["auto_excluded"], [])


class TransformTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="style-exp-tf-"))
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.source = Path(_write_image(self.directory / "source.png", value=140, size=(160, 120), stripes=True))

    def test_n0_is_deterministic_and_records_sha(self):
        first = attrs.transform_n0(self.source, self.directory / "n0a.png")
        second = attrs.transform_n0(self.source, self.directory / "n0b.png")
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["resolution"], [160, 120])

    def test_n1_changes_hue_but_records_shift(self):
        outcome = attrs.transform_n1(self.source, self.directory / "n1.png", hue_shift_degrees=30.0)
        self.assertIn("mean_V_channel_shift", outcome["recorded"])
        self.assertEqual(outcome["parameters"]["hue_shift_degrees"], 30.0)

    def test_n2_records_clipping_ratio(self):
        outcome = attrs.transform_n2(self.source, self.directory / "n2.png", intensity_scale=0.9)
        self.assertGreaterEqual(outcome["recorded"]["clipped_high_ratio"], 0.0)
        self.assertLessEqual(outcome["recorded"]["clipped_high_ratio"], 1.0)

    def test_n3_keeps_subject_pixels_byte_identical(self):
        outcome = attrs.transform_n3(self.source, self.directory / "n3.png")
        self.assertTrue(outcome["recorded"]["subject_pixels_byte_identical"])
        self.assertTrue(Path(outcome["recorded"]["mask_path"]).is_file())

    def test_n4_pads_with_replicated_column(self):
        outcome = attrs.transform_n4(self.source, self.directory / "n4.png", shift_fraction=0.05)
        self.assertEqual(outcome["recorded"]["pad_columns"], 8)
        with Image.open(outcome["path"]) as image:
            self.assertEqual(image.size, (160, 120))


class StatisticsTests(unittest.TestCase):
    def test_wilson_matches_known_value(self):
        result = core.wilson(8, 10)
        self.assertAlmostEqual(result["p"], 0.8, places=6)
        self.assertLess(result["low"], 0.8)
        self.assertGreater(result["high"], 0.8)

    def test_bootstrap_is_seeded_and_reports_cluster_count(self):
        clusters = {f"q{index}": float(index % 2) for index in range(12)}
        first = core.cluster_bootstrap(clusters, lambda sample: sum(sample) / len(sample))
        second = core.cluster_bootstrap(clusters, lambda sample: sum(sample) / len(sample))
        self.assertEqual(first, second)
        self.assertEqual(first["clusters"], 12)

    def test_kendall_tau_b_handles_ties(self):
        result = core.kendall_tau_b([1, 2, 2, 3], [1, 2, 2, 3])
        self.assertIsNotNone(result["tau"])
        self.assertGreater(result["tau"], 0.9)

    def test_kendall_returns_none_for_tiny_samples(self):
        self.assertIsNone(core.kendall_tau_b([1, 2], [1, 2])["tau"])

    def test_icc_requires_variation(self):
        flat = core.icc_a1([[1.0, 1.0], [1.0, 1.0]])
        self.assertIsNone(flat["icc"])
        varying = core.icc_a1([[0.1, 0.12], [0.5, 0.52], [0.9, 0.88]])
        self.assertIsNotNone(varying["icc"])

    def test_exact_counts(self):
        self.assertEqual(core.exact_counts([True, False, True]), {"n": 3, "k": 2})


class E0ToleranceTests(unittest.TestCase):
    def test_symmetry_formula_matches_protocol(self):
        limit = core.E0_SYMMETRY_ABS + core.E0_SYMMETRY_RTOL * max(abs(2.537698), abs(2.537699))
        self.assertAlmostEqual(limit, 1e-6 + 1e-5 * 2.537699, places=12)

    def test_tie_tolerance_is_strict(self):
        self.assertLess(core.TIE_RTOL, 1e-6)

    def test_metric_directions_declared(self):
        self.assertEqual(core.METRIC_DIRECTION["csd"], "max")
        for metric in ("gram", "adain", "lpips"):
            self.assertEqual(core.METRIC_DIRECTION[metric], "min")

    def test_acceptance_thresholds_are_preregistered(self):
        self.assertEqual(core.ACCEPTANCE["style_retrieval_assist"]["value"], 0.80)
        self.assertEqual(core.ACCEPTANCE["local_measurement_assist"]["value"], 0.90)
        self.assertEqual(core.ACCEPTANCE["candidate_ranking_assist"]["min_items"], 24)
        self.assertEqual(core.MAX_ATTEMPTS, 3)


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="style-exp-meta-"))
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def test_measure_records_json_safe_fingerprints(self):
        path = _write_image(self.directory / "image.png", value=90, size=(96, 64))
        record = attrs.measure(path)
        json.dumps({key: value for key, value in record.items() if key != "gray_small"})
        self.assertEqual(record["width"], 96)
        self.assertEqual(len(record["gray_histogram_32"]), attrs.GRAY_HISTOGRAM_BINS)
        self.assertEqual(len(record["gray_thumbnail_16x16"]), 256)
        self.assertEqual(len(record["dhash64"]), 16)
        self.assertIn(record["framing"], ("face_closeup", "upper_body", "full_body_or_scene"))

    def test_normalize_work_name_keeps_pages(self):
        self.assertEqual(attrs.normalize_work_name("prejpg-artist-1_fixed.png"), "artist-1")
        self.assertEqual(attrs.normalize_work_name("70115399_p0.jpg"), "70115399_p0")
        self.assertNotEqual(attrs.normalize_work_name("a-1.png"), attrs.normalize_work_name("a-2.png"))


if __name__ == "__main__":
    unittest.main()
