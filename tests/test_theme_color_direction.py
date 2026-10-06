import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from utils.theme_color import freeze_theme_color, apply_theme_color, theme_style, record_request, record_outputs, validate_snapshot

BASE = {"profile_id": "cool-blue-red", "environment": "sky", "accent_regions": "ribbon", "confirmed": True}
CONTENT = "One person under sky wearing a ribbon, with river water and a white dress."


class DirectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def test_old_selection_keeps_old_prompt_and_off_is_unchanged(self):
        from utils.prompt_loader import render_prompt_file
        plan = freeze_theme_color(BASE)
        expected = render_prompt_file("color-knowledge/theme-color-contract-v1.md", {
            "base": plan["profile"]["base"], "accent": plan["profile"]["accent"],
            "environment": "sky", "accent_regions": "ribbon"}).strip()
        self.assertEqual(plan["prompt"], expected)
        self.assertEqual(plan["schema_version"], 1)
        self.assertEqual(freeze_theme_color({"profile_id": "", "area": "balanced"}), {})

    def test_tones_work_without_palette_or_recolour_permission(self):
        for tone in ("bright", "deep", "clear", "muted"):
            plan = freeze_theme_color({"tone": tone})
            self.assertEqual(plan["profile"], {})
            self.assertEqual(plan["direction"]["tone"], tone)
            self.assertTrue(plan["direction"]["tone_source"]["reference_tones"])
            self.assertIn("retain all colours", apply_theme_color(CONTENT, plan))
            self.assertEqual(theme_style("Palette: blue\nBrushwork: ink", plan), "Palette: blue\nBrushwork: ink")
        with self.assertRaises(ValueError):
            freeze_theme_color({"tone": "neon"})

    def test_tone_changes_preserve_palette_and_bindings(self):
        bright = freeze_theme_color({**BASE, "tone": "bright"})
        deep = freeze_theme_color({**BASE, "tone": "deep"})
        self.assertEqual(bright["profile"], deep["profile"])
        self.assertEqual(bright["selection"]["environment"], deep["selection"]["environment"])
        self.assertNotEqual(bright["plan_hash"], deep["plan_hash"])
        self.assertIn("do not turn the scene into night", deep["prompt"])

    def test_balanced_replaces_legacy_small_accent_clause(self):
        plan = freeze_theme_color({**BASE, "area": "balanced"})
        self.assertNotIn("as a small, concentrated accent", plan["prompt"])
        self.assertNotIn("as the large environmental base", plan["prompt"])
        self.assertIn("comparable visual prominence", plan["prompt"])
        self.assertIn("preserve the composition", plan["prompt"])
        self.assertIn(79, plan["source_pages"])
        self.assertNotRegex(plan["prompt"], r"\d+%")

    def test_auxiliary_is_explicit_and_must_exist(self):
        plan = freeze_theme_color({**BASE, "area": "three-level", "auxiliary_regions": "river water"})
        self.assertEqual(plan["direction"]["auxiliary"]["colour"], "muted blue-grey")
        self.assertIn("not a book-measured role", plan["direction"]["auxiliary"]["provenance"])
        self.assertIn("river water", apply_theme_color(CONTENT, plan))
        with self.assertRaises(ValueError):
            apply_theme_color("One person under sky wearing a ribbon.", plan)
        for region in ("", "sky", "ribbon"):
            with self.assertRaises(ValueError):
                freeze_theme_color({**BASE, "area": "three-level", "auxiliary_regions": region})

    def test_tone_only_keeps_text_only_scope_and_freezes_evidence(self):
        plan = freeze_theme_color({"tone": "deep"})
        for kwargs in ({"image_paths": ["source.png"]}, {"post_enabled": True}, {"mode": "edit"}):
            with self.assertRaises(ValueError):
                apply_theme_color(CONTENT, plan, **kwargs)
        altered = copy.deepcopy(plan)
        altered["direction"]["tone"] = "bright"
        with self.assertRaises(ValueError):
            validate_snapshot(altered)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "candidate.png"
            path.write_bytes(b"model output; no transformations")
            request = record_request(plan, apply_theme_color(CONTENT, plan), root=Path(root) / "requests")
            record_outputs(request, [str(path)])
            self.assertEqual(path.read_bytes(), b"model output; no transformations")
            record = json.loads(path.with_suffix(".color-plan.json").read_text(encoding="utf-8"))
            self.assertEqual(record["color_plan"], plan)
            self.assertEqual(record["colour_review"], "not_performed")

    def test_ui_persists_independent_tone_and_preserves_unrelated_configuration(self):
        import utils.theme_color_widget as ui
        with tempfile.TemporaryDirectory() as root, patch.object(ui, "BASE", Path(root)):
            config = Path(root) / "conf/config.json"
            config.parent.mkdir()
            config.write_text('{"unrelated":{"keep":1}}', encoding="utf-8")
            widget = ui.ThemeColorSelector(scope="test")
            widget._region_selection = {"tone": "deep"}
            self.assertTrue(widget.is_active())
            widget._persist()
            restored = ui.ThemeColorSelector(scope="test")
            self.assertEqual(restored.snapshot(), widget.snapshot())
            self.assertEqual(json.loads(config.read_text())["unrelated"], {"keep": 1})

    def test_gpt_tone_only_submits_once_and_stops_in_repaint(self):
        import utils.theme_color_widget as ui
        import modules.image_generation.gpt_image2_tab as gt
        with tempfile.TemporaryDirectory() as root, patch.object(ui, "BASE", Path(root)), \
             patch.object(gt, "_read_config_image", return_value={}), \
             patch.object(gt.GptImage2Widget, "_start_pricing_refresh"), \
             patch.object(gt.GptImage2Widget, "load_defaults"), patch.object(gt.GptImage2Widget, "save_defaults"), \
             patch.object(gt.GptImage2Widget, "current_style_block", return_value=("Palette: blue\nBrushwork: ink", "")):
            widget = gt.GptImage2Widget()
            widget.color_selector._region_selection = {"tone": "deep"}
            widget.prompt_edit.setPlainText(CONTENT)
            with patch.object(widget, "repaint_enabled", return_value=False), patch.object(widget, "post_pipeline_steps", return_value={}):
                _, params = widget.build_request()
            self.assertEqual(params["prompt"].count("USER-SELECTED COLOUR DIRECTION"), 1)
            self.assertIn("Palette: blue", params["prompt"])
            self.assertNotIn("color_plan", params)
            self.assertEqual(widget.prompt_edit.toPlainText(), CONTENT)
            widget.mode_combo.setCurrentText(gt.MODE_REPAINT)
            with patch.object(gt.QMessageBox, "warning") as warning, patch.object(widget, "build_repaint_backend_request") as repaint:
                widget.generate()
                warning.assert_called_once()
                repaint.assert_not_called()
            widget.deleteLater()

    def test_dialog_accepts_tone_and_cancel_preserves_previous_selection(self):
        import utils.theme_color_widget as ui
        from PyQt6.QtWidgets import QComboBox, QDialog
        with tempfile.TemporaryDirectory() as root, patch.object(ui, "BASE", Path(root)):
            widget = ui.ThemeColorSelector(scope="test")
            def accept(dialog):
                for combo in dialog.findChildren(QComboBox):
                    index = combo.findData("deep")
                    if index >= 0:
                        combo.setCurrentIndex(index)
                return QDialog.DialogCode.Accepted
            with patch.object(QDialog, "exec", new=accept):
                widget.details()
            self.assertEqual(widget.snapshot()["direction"]["tone"], "deep")
            saved = widget.selection()
            with patch.object(QDialog, "exec", return_value=QDialog.DialogCode.Rejected):
                widget.details()
            self.assertEqual(widget.selection(), saved)


if __name__ == "__main__":
    unittest.main()
