import json
import os
from pathlib import Path
from unittest.mock import patch
import unittest
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from utils.theme_color import (freeze_theme_color, apply_theme_color, theme_style,
                               record_request, record_outputs, recommend_theme)


def selection():
    return {"profile_id": "cool-blue-red", "environment": "sky, river water",
            "accent_regions": "waist ribbon sash", "confirmed": True}


class ThemeColorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_off_does_not_read_prompts_or_write_snapshots(self):
        tmp_path = self.root
        app = self.app
        with patch("utils.theme_color.read_prompt_file", side_effect=AssertionError("off read")):
            assert freeze_theme_color({}) == {}
            assert apply_theme_color("original\n", {}) == "original\n"
            assert theme_style("Palette: red\n", {}) == "Palette: red\n"
            assert record_request({}, "original", root=tmp_path) == {}
        assert not list(tmp_path.iterdir())


    def test_scope_missing_regions_and_tampering_stop_before_generation(self):
        tmp_path = self.root
        app = self.app
        snapshot = freeze_theme_color(selection())
        text = "One person, sky, river water, waist ribbon sash, white dress."
        result = apply_theme_color(text, snapshot)
        assert result.startswith(text)
        assert result.count("THEME COLOUR DESIGN") == 1
        for kwargs in ({"image_paths": ["style.png"]}, {"mode": "edit"}, {"post_enabled": True}):
            with self.assertRaises(ValueError):
                apply_theme_color(text, snapshot, **kwargs)
        with self.assertRaisesRegex(ValueError, "未在正文"):
            apply_theme_color("A city skyline.", snapshot)
        with self.assertRaisesRegex(ValueError, "校验失败"):
            apply_theme_color(text, {**snapshot, "prompt": "new"})
        for changes in ({"confirmed": False}, {"accent_regions": ""}, {"profile_id": "unknown"}):
            with self.assertRaises(ValueError):
                freeze_theme_color({**selection(), **changes})
        with self.assertRaises(ValueError):
            freeze_theme_color({**selection(), "accent_regions": "sky"})
        bow = freeze_theme_color({**selection(), "accent_regions": "bow"})
        with self.assertRaises(ValueError):
            apply_theme_color("sky and river water beside an elbow", bow)


    def test_style_palette_priority_preserves_other_drawing_fields(self):
        tmp_path = self.root
        app = self.app
        style = "Palette: bright green\nLighting: daylight\nBrushwork: ink\nKeep lace connected."
        changed = theme_style(style, freeze_theme_color(selection()))
        assert "bright green" not in changed
        assert "Lighting: daylight\nBrushwork: ink\nKeep lace connected." == changed
        assert style.startswith("Palette:")


    def test_snapshots_attach_to_actual_outputs_and_remain_unreviewed(self):
        tmp_path = self.root
        app = self.app
        snapshot = freeze_theme_color(selection())
        request = record_request(snapshot, "actual prompt", model="test-model", channel="test", root=tmp_path / "requests")
        candidate = tmp_path / "candidate.jpg"
        candidate.write_bytes(b"unchanged-model-image")
        record_outputs(request, [str(candidate)])
        saved = json.loads(candidate.with_suffix(".color-plan.json").read_text(encoding="utf-8"))
        assert saved["color_plan"] == snapshot
        assert saved["prompt"] == "actual prompt"
        assert saved["colour_review"] == "not_performed"
        assert candidate.read_bytes() == b"unchanged-model-image"
        assert saved["outputs"][0]["path"] == str(candidate.resolve())
        assert "api_key" not in saved


    def test_recommendation_is_closed_and_does_not_rewrite_description(self):
        tmp_path = self.root
        app = self.app
        cfg = {"base_url": "test", "api_key": "not-real", "model": "test"}
        seen = []
        def caller(*args, **kwargs):
            seen.append(json.loads(args[4]))
            return '{"profile_id":"cool-blue-red","reason":"冷蓝环境"}'
        with patch("utils.analysis_gpt_prompt.load_text_api_config", return_value=cfg):
            assert recommend_theme("A river", caller=caller)["profile_id"] == "cool-blue-red"
            assert seen[0]["description"] == "A river"
            with self.assertRaises(ValueError):
                recommend_theme("A river", caller=lambda *a, **k: '{"profile_id":"new","reason":"x"}')


    def test_selector_persists_only_own_node_and_defaults_off(self):
        tmp_path = self.root
        app = self.app
        import utils.theme_color_widget as ui
        (tmp_path / "conf").mkdir()
        config = tmp_path / "conf/config.json"
        config.write_text(json.dumps({"unrelated": {"keep": 1}}), encoding="utf-8")
        with patch.object(ui, "BASE", tmp_path):
            widget = ui.ThemeColorSelector(scope="test")
            assert widget.snapshot() == {}
            assert json.loads(config.read_text())["unrelated"] == {"keep": 1}
            widget._region_selection = selection()
            widget.combo.setCurrentIndex(widget.combo.findData("cool-blue-red"))
            assert widget.snapshot()["profile"]["id"] == "cool-blue-red"
            saved = json.loads(config.read_text())
            assert saved["unrelated"] == {"keep": 1}
            restored = ui.ThemeColorSelector(scope="test")
            assert restored.snapshot() == widget.snapshot()
            widget.deleteLater()
            restored.deleteLater()


    def test_gpt_request_uses_frozen_theme_once_without_backend_metadata(self):
        tmp_path = self.root
        app = self.app
        import modules.image_generation.gpt_image2_tab as gt
        import utils.theme_color_widget as ui
        with patch.object(ui, "BASE", tmp_path), patch.object(gt, "_read_config_image", return_value={}), \
             patch.object(gt.GptImage2Widget, "_start_pricing_refresh"), \
             patch.object(gt.GptImage2Widget, "load_defaults"), patch.object(gt.GptImage2Widget, "save_defaults"), \
             patch.object(gt.GptImage2Widget, "current_style_block", return_value=("", "")):
            (tmp_path / "conf").mkdir()
            widget = gt.GptImage2Widget()
            widget.prompt_edit.setPlainText("One woman near sky and river water, waist ribbon sash, white dress.")
            widget.color_selector._region_selection = selection()
            widget.color_selector.combo.setCurrentIndex(widget.color_selector.combo.findData("cool-blue-red"))
            with patch.object(widget, "repaint_enabled", return_value=False), patch.object(widget, "post_pipeline_steps", return_value={}):
                _, params = widget.build_request()
                assert params["prompt"].count("THEME COLOUR DESIGN") == 1
                assert "color_plan" not in params
                assert widget._prepared_color["scope"] == "text_only_first_generation"
                widget.mode_combo.setCurrentText(gt.MODE_EDIT)
                with self.assertRaises(ValueError):
                    widget.build_request()
            widget.deleteLater()

    def test_prompt_cell_submits_theme_and_keeps_original_edit_text(self):
        import modules.image_generation.prompt_generator as pg
        from unittest.mock import Mock
        content = "One person by sky and river water wearing a waist ribbon sash."
        worker = Mock()
        snapshot = freeze_theme_color(selection())
        cell = pg.PromptCellWidget(content, lambda: "Palette: green\nBrushwork: ink",
            lambda: ("test", "not-real", "test-model", "aigc2d"), lambda: None,
            color_getter_func=lambda: snapshot)
        with patch.object(pg, "ImageGenWorkerThread", return_value=worker) as factory, \
             patch("utils.theme_color.record_request", return_value={}):
            cell.generate_image()
            sent = factory.call_args.kwargs
            assert sent["prompt"].count("THEME COLOUR DESIGN") == 1
            assert "Palette:" not in sent["instructions"]
            assert cell.text_edit.toPlainText() == content
            worker.start.assert_called_once()
        cell.deleteLater()

    def test_worker_writes_the_submission_snapshot_without_backend_extra_fields(self):
        from modules.image_generation.gpt_image2_tab import GptImage2Worker
        snapshot = freeze_theme_color(selection())
        request = record_request(snapshot, "actual text", root=self.root / "requests")
        candidate = self.root / "original.jpg"
        candidate.write_bytes(b"model output")
        seen = []
        def backend(**kwargs):
            seen.append(kwargs)
            return [str(candidate)]
        worker = GptImage2Worker(backend, {"prompt": "actual text"}, color_record=request)
        worker.run()
        assert seen == [{"prompt": "actual text"}]
        saved = json.loads(candidate.with_suffix(".color-plan.json").read_text(encoding="utf-8"))
        assert saved["color_plan"] == snapshot
        assert saved["status"] == "generated"

    def test_successful_recommendations_reuse_the_same_cache(self):
        from unittest.mock import Mock
        cfg = {"base_url": "test", "api_key": "not-real", "model": "test"}
        response = Mock(return_value='{"profile_id":"off","reason":"没有落点"}')
        with patch("utils.analysis_gpt_prompt.load_text_api_config", return_value=cfg), \
             patch("utils.analysis_gpt_prompt.call_text_model", response):
            first = recommend_theme("same description", cache_dir=self.root)
            assert recommend_theme("same description", cache_dir=self.root) == first
            response.assert_called_once()
            recommend_theme("another description", cache_dir=self.root)
            assert response.call_count == 2
