import copy
import json
from pathlib import Path
import tempfile
import unittest

from utils.color_extract import measure_book_image_chart, build_generation_knowledge

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "prompts/color-knowledge/book-image-chart-rois-v1.json"


class ChartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chart = measure_book_image_chart(str(MANIFEST), str(ROOT))

    def test_fifty_source_positions_not_merged_by_label(self):
        self.assertEqual(self.chart["count"], 50)
        self.assertEqual(self.chart["swatch_count"], 150)
        self.assertEqual(sum(p["label"] == "悠闲的" for p in self.chart["palettes"]), 2)
        self.assertEqual(sum(p["label"] == "理性的" for p in self.chart["palettes"]), 2)

    def test_near_white_is_measured_not_discarded(self):
        colour = self.chart["palettes"][0]["colours"][1]
        self.assertEqual(colour["hex_source"], "scan_measured")
        self.assertGreater(min(colour["measured_rgb"]), 235)

    def test_all_roles_unassigned_and_source_deduplicated(self):
        self.assertEqual(self.chart["source_pages"], [147, 74])
        self.assertTrue(all(c["role"] is None for p in self.chart["palettes"] for c in p["colours"]))

    def test_measurement_is_repeatable(self):
        self.assertEqual(self.chart, measure_book_image_chart(str(MANIFEST), str(ROOT)))

    def test_every_sample_is_strictly_inside_its_colour_cell(self):
        for p in self.chart["palettes"]:
            x0, y0, x1, y1 = p["source"]["bbox"]
            for c in p["colours"]:
                sx0, sy0, sx1, sy1 = c["sample_bbox"]
                self.assertGreater(sx0, x0 + (c["order"] - 1) * (x1 - x0) / 3)
                self.assertLess(sx1, x0 + c["order"] * (x1 - x0) / 3)
                self.assertTrue(y0 < sy0 < sy1 < y1)

    def test_changed_source_hash_or_duplicate_roi_is_rejected(self):
        source = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for change in ("hash", "duplicate", "outside"):
            data = copy.deepcopy(source)
            if change == "hash":
                data["source_image_sha256"] = "0" * 64
            elif change == "duplicate":
                data["rows"].append(data["rows"][0])
            else:
                data["rows"][0][1] = -1
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "manifest.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    measure_book_image_chart(str(path), str(ROOT))

    def test_v2_includes_chart_without_changing_v1(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chart.json"
            path.write_text(json.dumps(self.chart), encoding="utf-8")
            knowledge = build_generation_knowledge(str(ROOT / "data/color-knowledge"),
                str(ROOT / "docs/261003-color-improve/book-html/pages"), str(path))
            self.assertEqual((knowledge["count"], knowledge["version"]), (109, 2))
            self.assertEqual(len({p["id"] for p in knowledge["palettes"]}), 109)


if __name__ == "__main__":
    unittest.main()
