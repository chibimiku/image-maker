import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from modules.image_analysis.style_dataset import (SCORE_KEYS, make_inventory, validate_manifest,
    materialize_dataset, load_manifest)


class StyleDatasetManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "原图"
        self.root.mkdir()
        for index, color in enumerate(("red", "blue", "green")):
            Image.new("RGB", (32 + index, 40), color).save(self.root / f"图{index}.png")
        self.document = make_inventory(self.root, "test-style", 2)
        self.document.update(agent={"name": "vision-agent", "model": "model", "review_method": "联系表及原图"},
                             selection_summary="相同画法，保留不同配色", reference_reason="原图清晰；角色内容须独立保持")
        for index, record in enumerate(self.document["images"]):
            record.update(inspected=True, decision="selected" if index < 2 else "rejected",
                          reason="清晰成稿" if index < 2 else "与已选内容近似，保留变化更大的样本",
                          scores={key: 8 for key in SCORE_KEYS}, tags=["framing:half-body"])
        selected = [r["path"] for r in self.document["images"][:2]]
        self.document.update(selected_order=selected[::-1], reference_image=selected[0])

    def tearDown(self):
        self.temp.cleanup()

    def test_validate_copy_preserves_order_and_source_and_audit(self):
        original = {p.name: p.read_bytes() for p in self.root.iterdir()}
        checked = validate_manifest(self.document)
        self.assertEqual(checked["selected_count"], 2)
        self.assertEqual(checked["rows"][0]["selection_score"], 8)
        output = Path(self.temp.name) / "datasets"
        copied = materialize_dataset(checked, output)
        self.assertEqual([Path(p).name for p in copied["image_paths"]], self.document["selected_order"])
        self.assertTrue(Path(copied["reference_path"]).is_file())
        self.assertEqual({p.name: p.read_bytes() for p in self.root.iterdir()}, original)
        self.assertTrue((Path(copied["directory"]) / "import-audit.json").exists())
        self.assertNotEqual(materialize_dataset(checked, output)["directory"], copied["directory"])

    def test_unseen_incomplete_fabricated_or_changed_input_rejected(self):
        changes = [lambda d: d["images"][0].update(inspected=False),
                   lambda d: d["images"][0].update(decision="needs_review"),
                   lambda d: d["images"].pop(),
                   lambda d: d["images"][0].update(sha256="0" * 64),
                   lambda d: d["images"][0].update(path="../escape.png"),
                   lambda d: d["images"][0].update(path="C:/escape.png"),
                   lambda d: d["images"][0].update(duplicate_of=[]),
                   lambda d: d["images"][0]["scores"].update(completeness=True),
                   lambda d: d["images"][0]["scores"].update(completeness=float("nan")),
                   lambda d: d.update(selected_order=[d["selected_order"][0]] * 2),
                   lambda d: d.update(reference_image="missing.png"),
                   lambda d: d.update(schema_version="unknown")]
        for change in changes:
            document = copy.deepcopy(self.document)
            change(document)
            with self.assertRaises((ValueError, OSError)):
                validate_manifest(document)
        Image.new("RGB", (10, 10), "white").save(self.root / "图0.png")
        with self.assertRaisesRegex(ValueError, "元数据不匹配"):
            validate_manifest(self.document)

    def test_duplicate_content_shortfall_and_unreadable(self):
        (self.root / "duplicate.png").write_bytes((self.root / "图0.png").read_bytes())
        document = make_inventory(self.root, "test-style", 3)
        # 补上完全重复项的审查，选中重复文件和原文件时必须拒绝。
        previous = {r["path"]: r for r in self.document["images"]}
        document.update({k: v for k, v in self.document.items() if k not in ("images", "requested_count")})
        for r in document["images"]:
            if r["path"] in previous:
                r.update(previous[r["path"]])
            else:
                r.update(inspected=True, decision="selected", reason="相同内容", scores={key: 8 for key in SCORE_KEYS})
        document["selected_order"].append("duplicate.png")
        with self.assertRaisesRegex(ValueError, "完全重复"):
            validate_manifest(document)
        duplicate = next(r for r in document["images"] if r["path"] == "duplicate.png")
        duplicate.update(decision="rejected", duplicate_of="图0.png")
        document["selected_order"].remove("duplicate.png")
        with self.assertRaisesRegex(ValueError, "shortfall_reason"):
            validate_manifest(document)
        document["shortfall_reason"] = "只找到两张有代表性的清晰作品"
        self.assertEqual(validate_manifest(document)["selected_count"], 2)
        bad = self.root / "坏图.jpg"
        bad.write_bytes(b"not an image")
        broken = next(r for r in make_inventory(self.root, "test-style")["images"] if r["path"] == "坏图.jpg")
        broken.update(decision="rejected", reason="文件不能解码")
        document["images"].append(broken)
        validate_manifest(document)

    def test_relocation_and_nested_same_names(self):
        moved = Path(self.temp.name) / "moved"
        import shutil
        shutil.copytree(self.root, moved)
        checked = validate_manifest(self.document, moved)
        self.assertEqual(Path(checked["source_root"]), moved)
        for name in ("a", "b"):
            (moved / name).mkdir()
            Image.new("RGB", (12, 13), "yellow" if name == "a" else "purple").save(moved / name / "same.png")
        recursive = make_inventory(moved, "test-style", 2, True)
        self.assertIn("a/same.png", [r["path"] for r in recursive["images"]])
        with self.assertRaisesRegex(ValueError, "覆盖不完整"):
            validate_manifest({**self.document, "recursive": True}, moved)

    def test_app_import_is_fresh_training_and_provenance_is_saved(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt6.QtWidgets import QApplication
        from modules.image_analysis.style_analyzer import StyleAnalyzerWidget, StyleIterativeWorkerThread, StyleDatasetImportDialog
        app = QApplication.instance() or QApplication([])
        manifest = Path(self.temp.name) / "selection-manifest.json"
        manifest.write_text(json.dumps(self.document, ensure_ascii=False), encoding="utf-8")
        dialog = StyleDatasetImportDialog(str(manifest))
        self.assertTrue(dialog.ok.isEnabled())
        self.assertEqual(dialog.table.rowCount(), 3)
        copied = materialize_dataset(load_manifest(manifest), Path(self.temp.name) / "datasets")
        widget = StyleAnalyzerWidget(lambda: ("", "", ""))
        widget._existing_state = {"iterations": ["OLD"]}
        widget._output_dir = "old-run"
        widget.apply_dataset_selection(copied, str(manifest))
        self.assertIsNone(widget._existing_state)
        self.assertEqual(widget._output_dir, "")
        self.assertEqual(widget._get_image_paths(), copied["image_paths"])
        self.assertEqual(widget.test_ref_input.text(), copied["reference_path"])
        worker = StyleIterativeWorkerThread(copied["image_paths"], "", "", "", dataset_selection=widget._dataset_selection)
        self.assertEqual(worker._build_state(0, [])["dataset_selection"]["source_manifest"], str(manifest))
        dialog.close()
        widget.close()

    def test_portable_skill_validator_matches_app(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual((root / "modules/image_analysis/style_dataset.py").read_bytes(),
                         (root / "docs/skills/style-dataset-curator/scripts/manifest_core.py").read_bytes())


if __name__ == "__main__":
    unittest.main()
