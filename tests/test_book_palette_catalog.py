import json
import tempfile
import unittest
from pathlib import Path

from utils.color_extract import build_book_palette_catalog, export_book_palette_catalog


class BookPaletteCatalogTests(unittest.TestCase):
    def test_all_existing_cases_are_included(self):
        root = Path(__file__).resolve().parents[1]
        data = root / "data/color-knowledge"
        source = json.loads((data / "palette_library.json").read_text(encoding="utf-8"))
        catalog = build_book_palette_catalog(str(data))
        self.assertEqual(catalog["count"], 31)
        self.assertEqual(catalog["swatch_count"], 155)
        self.assertEqual([p["id"] for p in catalog["palettes"]], [p["slug"] for p in source["palettes"]])
        self.assertTrue(all(p["generation_validation"] == "untested" for p in catalog["palettes"]))
        self.assertTrue(all(c["role"] is None for p in catalog["palettes"] for c in p["colours"]))

    def test_measured_nominal_and_missing_values_stay_distinct(self):
        root = Path(__file__).resolve().parents[1]
        catalog = build_book_palette_catalog(str(root / "data/color-knowledge"))
        colours = [c for p in catalog["palettes"] for c in p["colours"]]
        self.assertEqual(sum(c["hex_source"] == "scan_measured" for c in colours), 10)
        self.assertTrue(any(c["hex_source"] == "name_only" and c["hex"] is None for c in colours))
        for c in colours:
            if c["measured_hex"]:
                self.assertEqual(c["hex"], c["measured_hex"])
                self.assertEqual(c["hex_source"], "scan_measured")

    def test_export_is_offline_and_does_not_modify_library(self):
        root = Path(__file__).resolve().parents[1]
        data = root / "data/color-knowledge"
        before = (data / "palette_library.json").read_bytes()
        with tempfile.TemporaryDirectory() as folder:
            result = export_book_palette_catalog(str(data), str(Path(folder) / "catalog.json"))
            html = Path(result["gallery"]).read_text(encoding="utf-8")
            self.assertEqual(html.count('<section>'), 31)
            self.assertNotIn('src="http', html)
            self.assertEqual(result["swatch_count"], 155)
        self.assertEqual((data / "palette_library.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
