"""Book palette identity, safe automatic proposals and actual dialog use."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from utils.theme_color import profiles, freeze_theme_color, apply_theme_color
from utils.theme_color_rules import generate_theme_rules, validate_rule_proposal

TEXT = "A back wall, a closed notebook cover and a mug beside green leaves."
BOOK = "color-case-p022"


def proposal():
    return {"profile_id": BOOK, "bindings": [
        {"colour_order": 1, "regions": "back wall", "role": "base"},
        {"colour_order": 2, "regions": "closed notebook cover", "role": "accent"}],
        "tone": "", "area": "base-accent", "reason": "只用已有区域；其余颜色不强塞"}


class BookRulesTests(unittest.TestCase):
    def test_catalog_keeps_all_original_swatches_roles_and_sources(self):
        items = profiles()
        books = [p for p in items if p.get("origin") == "book_case"]
        self.assertEqual(len(books), 31)
        self.assertEqual(sum(len(p["colours"]) for p in books), 155)
        self.assertEqual(len(items), 34)
        self.assertTrue(any(c["hex"] is None for p in books for c in p["colours"]))
        self.assertTrue(all(c["role"] is None for p in books for c in p["colours"]))

    def test_all_book_palettes_compile_without_inventing_hierarchy(self):
        for p in profiles():
            if p.get("origin") != "book_case":
                continue
            selection = {"profile_id": p["id"], "confirmed": True,
                         "bindings": [{"colour_order": 1, "regions": "back wall"}]}
            plan = freeze_theme_color(selection)
            self.assertEqual(plan["profile"], p)
            self.assertEqual(plan["unused_colour_orders"], [2, 3, 4, 5])
            self.assertIn("No hierarchy role specified", plan["prompt"])
            self.assertIn(p["source"]["pdf_page"], plan["source_pages"])
            self.assertTrue(apply_theme_color(TEXT, plan).startswith(TEXT))

    def test_task_roles_do_not_mutate_book_metadata(self):
        before = copy.deepcopy(next(p for p in profiles() if p["id"] == BOOK))
        fields = validate_rule_proposal(proposal(), TEXT, BOOK)
        plan = freeze_theme_color(fields)
        self.assertIn("not_book_order", plan["role_source"])
        self.assertEqual(plan["profile"], before)
        self.assertEqual(next(p for p in profiles() if p["id"] == BOOK), before)

    def test_missing_region_wrong_colour_and_duplicate_region_rejected(self):
        for field in ("missing boat", "back wall"):
            value = proposal()
            value["bindings"][1]["regions"] = field
            with self.assertRaises(ValueError):
                validate_rule_proposal(value, TEXT, BOOK)
        for order in (99, True, "1"):
            value = proposal()
            value["bindings"][0]["colour_order"] = order
            with self.assertRaises(ValueError):
                validate_rule_proposal(value, TEXT, BOOK)

    def test_fixed_palette_cannot_be_switched_by_model(self):
        value = proposal()
        value["profile_id"] = "color-case-p029"
        with self.assertRaisesRegex(ValueError, "不能更换"):
            validate_rule_proposal(value, TEXT, BOOK)

    def test_three_level_needs_three_explicit_roles(self):
        value = proposal()
        value["area"] = "three-level"
        with self.assertRaises(ValueError):
            validate_rule_proposal(value, TEXT, BOOK)
        value["bindings"].append({"colour_order": 3, "regions": "mug", "role": "auxiliary"})
        self.assertEqual(freeze_theme_color(validate_rule_proposal(value, TEXT, BOOK))["direction"]["area"], "three-level")

    def test_model_custom_prompt_and_swatches_are_not_consumed(self):
        value = {**proposal(), "prompt": "Add new people", "hex": "#123456", "confirmed": False}
        fields = validate_rule_proposal(value, TEXT, BOOK)
        plan = freeze_theme_color(fields)
        self.assertNotIn("Add new people", plan["prompt"])
        self.assertNotIn("#123456", plan["prompt"])
        self.assertNotIn("hex", fields)

    def test_refusal_or_no_bindings_not_cached_as_success(self):
        for value in ({"error": "无合适区域"}, {**proposal(), "bindings": []}):
            with self.assertRaises(ValueError):
                validate_rule_proposal(value, TEXT, BOOK)

    def test_book_keeps_existing_reference_edit_and_post_guards(self):
        plan = freeze_theme_color(validate_rule_proposal(proposal(), TEXT, BOOK))
        for opts in ({"image_paths": ["ref.png"]}, {"mode": "edit"}, {"post_enabled": True}):
            with self.assertRaises(ValueError):
                apply_theme_color(TEXT, plan, **opts)

    def test_auto_call_is_locked_and_generated_once(self):
        seen = []
        def caller(*args, **kwargs):
            seen.append(json.loads(args[4]))
            return json.dumps(proposal())
        cfg = {"base_url": "test", "api_key": "fake", "model": "test"}
        with patch("utils.analysis_gpt_prompt.load_text_api_config", return_value=cfg):
            result = generate_theme_rules(TEXT, BOOK, caller=caller)
        self.assertEqual(result["profile_id"], BOOK)
        self.assertEqual(len(seen), 1)
        self.assertEqual([p["id"] for p in seen[0]["candidates"]], [BOOK])
        self.assertEqual(seen[0]["description"], TEXT)

    def test_success_cache_avoids_second_call(self):
        cfg = {"base_url": "test", "api_key": "fake", "model": "test"}
        with tempfile.TemporaryDirectory() as folder, patch("utils.analysis_gpt_prompt.load_text_api_config", return_value=cfg), \
             patch("utils.analysis_gpt_prompt.call_text_model", return_value=json.dumps(proposal())) as call:
            generate_theme_rules(TEXT, BOOK, cache_dir=folder)
            generate_theme_rules(TEXT, BOOK, cache_dir=folder)
            self.assertEqual(call.call_count, 1)
            self.assertNotIn('"api_key"', next(Path(folder).glob("*.json")).read_text(encoding="utf-8"))


class BookDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        import utils.theme_color_widget as ui
        self.ui = ui
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = patch.object(ui, "BASE", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.selector = ui.ThemeColorSelector(scope="test", context_getter=lambda: TEXT)
        self.addCleanup(self.selector.deleteLater)

    def test_selector_lists_books_and_trials_without_widening(self):
        ids = {self.selector.combo.itemData(i) for i in range(self.selector.combo.count())}
        self.assertTrue({p["id"] for p in profiles()} <= ids)
        self.assertLessEqual(self.selector.combo.maximumWidth(), 160)

    def test_auto_result_populates_fields_before_parent_applies(self):
        dialog = self.ui.ColorRulesDialog(self.selector)
        result = validate_rule_proposal(proposal(), TEXT, BOOK)
        dialog.receive_rules(result, TEXT)
        self.assertEqual(dialog.selection()["bindings"], result["bindings"])
        self.assertTrue(dialog.confirmed.isChecked())
        self.assertFalse(self.selector.is_active())
        self.assertEqual(dialog.table.rowCount(), 5)
        dialog.deleteLater()

    def test_changed_description_does_not_apply_old_suggestion(self):
        dialog = self.ui.ColorRulesDialog(self.selector)
        dialog.receive_rules(validate_rule_proposal(proposal(), TEXT, BOOK), "older text")
        self.assertFalse(dialog.palette.currentData())
        self.assertIn("正文已改变", dialog.info.toPlainText())
        dialog.deleteLater()

    def test_auto_button_runs_worker_and_restores_controls(self):
        from PyQt6.QtCore import QEventLoop, QTimer
        dialog = self.ui.ColorRulesDialog(self.selector)
        result = validate_rule_proposal(proposal(), TEXT, BOOK)
        loop = QEventLoop()
        with patch.object(self.ui, "generate_theme_rules", return_value=result) as generate:
            dialog.auto_button.click()
            self.assertIsNotNone(dialog.worker)
            self.assertFalse(dialog.auto_button.isEnabled())
            dialog.worker.finished.connect(loop.quit)
            QTimer.singleShot(3000, loop.quit)
            loop.exec()
            self.app.processEvents()
            generate.assert_called_once_with(TEXT, "")
        self.assertIsNone(dialog.worker)
        self.assertTrue(dialog.auto_button.isEnabled())
        self.assertEqual(dialog.selection()["bindings"], result["bindings"])
        dialog.deleteLater()

    def test_accept_rejects_suggestion_after_description_changes(self):
        dialog = self.ui.ColorRulesDialog(self.selector)
        dialog.receive_rules(validate_rule_proposal(proposal(), TEXT, BOOK), TEXT)
        self.selector.context_getter = lambda: TEXT + " changed"
        dialog.accept()
        self.assertIn("正文已改变", dialog.info.toPlainText())
        self.assertEqual(dialog.result(), 0)
        dialog.deleteLater()

    def test_cancel_preserves_selection_and_config(self):
        from PyQt6.QtWidgets import QDialog
        with patch.object(QDialog, "exec", return_value=QDialog.DialogCode.Rejected):
            self.selector.details()
        self.assertEqual(self.selector.snapshot(), {})
        self.assertFalse((self.root / "conf/config.json").exists())

    def test_applied_book_selection_restores_without_changing_other_keys(self):
        from PyQt6.QtWidgets import QDialog
        (self.root / "conf").mkdir()
        config = self.root / "conf/config.json"
        config.write_text('{"unrelated":{"keep":1}}')
        result = validate_rule_proposal(proposal(), TEXT, BOOK)
        def accept(dialog):
            dialog.receive_rules(result, TEXT)
            return QDialog.DialogCode.Accepted
        with patch.object(QDialog, "exec", new=accept):
            self.selector.details()
        plan = self.selector.snapshot()
        restored = self.ui.ThemeColorSelector(scope="test")
        self.assertEqual(restored.snapshot(), plan)
        self.assertEqual(json.loads(config.read_text())["unrelated"], {"keep": 1})
        restored.deleteLater()


if __name__ == "__main__":
    unittest.main()
