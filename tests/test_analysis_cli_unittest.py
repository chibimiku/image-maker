"""无网络的启动和分析 CLI/UI 选项映射冒烟。"""
import os
import datetime
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QComboBox


class AnalysisCliSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_startup_widgets_do_not_save_sd_config_and_video_has_output_dir(self):
        from modules.image_generation.sd_workflow_tab import SdWorkflowWidget
        from modules.video_generation.video_gen_tab import VideoGenWidget

        with patch("modules.image_generation.sd_workflow_tab.save_sd_workflow_state") as save:
            sd = SdWorkflowWidget()
            self.assertFalse(sd._syncing)
            save.assert_not_called()
        sd.update_config_from_ui = lambda: None
        sd.close()
        video = VideoGenWidget()
        self.assertTrue(video.out_dir_edit.text())
        video.close()

    def test_cli_applies_analysis_ui_options(self):
        import json
        from pathlib import Path
        from modules.image_analysis.single_analyzer import SingleAnalyzerWidget
        from tools.analysis_gpt_run import apply_analysis_tab_options

        styles = json.loads((Path(__file__).resolve().parents[1] / "conf/config-styles.json").read_text(encoding="utf-8"))
        with patch("modules.image_analysis.single_analyzer.save_analysis_gpt_ui") as save_ui:
            tab = SingleAnalyzerWidget(
                config_getter_func=lambda nsfw=False: ("", "", ""),
                img_config_getter_func=lambda: ("", "", "", "aigc2d"),
                styles_getter_func=lambda: styles, save_img_cfg_callback=lambda: None,
                upscale_options_getter_func=lambda: {}, persist_ui_state=False)
            tab.update_styles(list(styles))
        options = {
            "style": "sheya-style", "reference_mode": "priority", "channel": "gpt",
            "nsfw": True, "outfit_check": True, "remove_photo_style": True,
            "save_to_source": False, "jpg_upscale": False, "repaint": True,
            "structure": True, "local": True, "tone": True, "ink": True,
            "size_follow_input": False, "quality": "medium", "region": "stable-four",
            "scope": "person_only", "first_pass_mode": "edits", "tone_target": "photo",
            "outfit_style": "Victorian", "upscale_model": "", "upscale_by": 3.0,
            "webp_target_mb": 8.0,
        }
        with patch("modules.image_analysis.single_analyzer.save_analysis_gpt_ui") as save_ui:
            plan = apply_analysis_tab_options(tab, options)
            save_ui.assert_not_called()
        self.assertEqual(plan["reference_mode"], "priority")
        self.assertEqual(plan["gpt_steps"]["local"]["regions"],
                         ["subject_no_face", "shoes", "waist", "thigh"])
        self.assertEqual(plan["gpt_steps"]["repaint"]["scope"], "person_only")
        self.assertEqual(plan["gpt_steps"]["tone"]["tone_target"], "photo")
        self.assertEqual(tab.gpt_quality_combo.currentData(), "medium")
        self.assertEqual(tab.gpt_first_pass_mode.currentData(), "edits")
        self.assertEqual(tab.outfit_style_combo.currentText(), "Victorian")
        tab.close()

    def test_auto_generation_checks_persist_across_widget_restart(self):
        import json
        from pathlib import Path
        from modules.image_analysis.single_analyzer import SingleAnalyzerWidget

        def make_tab():
            return SingleAnalyzerWidget(
                config_getter_func=lambda: {}, img_config_getter_func=lambda: {},
                styles_getter_func=lambda: {}, save_img_cfg_callback=lambda: None,
                upscale_options_getter_func=lambda: {},
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            with patch("modules.image_analysis.single_analyzer.analysis_gpt_ui_path",
                       return_value=str(config_path)), patch(
                           "modules.image_analysis.single_analyzer.list_esrgan_models",
                           return_value=[]):
                first = make_tab()
                first.update_styles(["默认(无附加)", "style-A"])
                first.main_style_combo.setCurrentText("随机")
                first.auto_gen_orig_cb.setChecked(True)
                first.auto_gen_ref_cb.setChecked(True)
                saved = json.loads(config_path.read_text(encoding="utf-8"))
                self.assertTrue(saved["analysis_gpt_pipeline"]["auto_gen_original"])
                self.assertTrue(saved["analysis_gpt_pipeline"]["auto_gen_refined"])
                first.close()

                second = make_tab()
                second.update_styles(["默认(无附加)", "style-A"])
                self.assertEqual(second.main_style_combo.currentText(), "随机")
                self.assertTrue(second.auto_gen_orig_cb.isChecked())
                self.assertTrue(second.auto_gen_ref_cb.isChecked())
                second.auto_gen_orig_cb.setChecked(False)
                second.close()

                third = make_tab()
                third.update_styles(["默认(无附加)", "style-A"])
                self.assertEqual(third.main_style_combo.currentText(), "随机")
                self.assertFalse(third.auto_gen_orig_cb.isChecked())
                self.assertTrue(third.auto_gen_ref_cb.isChecked())
                third.close()

    def test_manual_pickup_publish_does_not_create_queue_task(self):
        import json
        from pathlib import Path
        from modules.image_analysis.single_analyzer import SingleAnalyzerWidget

        with tempfile.TemporaryDirectory() as temp_dir:
            process = Path(temp_dir) / "data" / "20260928" / "analysis-gpt-image" / "65ebea5b-run"
            process.mkdir(parents=True)
            checkpoint = process / "generation-checkpoint.json"
            checkpoint.write_text(json.dumps({"snapshot": {"file_prefix": "65ebea5b",
                "request_payload": {"style_name": "test"}}}), encoding="utf-8")
            candidate = process / "candidate.jpg"
            candidate.write_bytes(b"image")

            class PickupDialog:
                checkpoint_path = str(checkpoint)
                selected_artifact = {"path": str(candidate), "label": "候选图", "stage": "final_review"}
                action = "publish"

                def __init__(self, *_args):
                    pass

                def exec(self):
                    return 1

            with patch("modules.image_analysis.single_analyzer.list_esrgan_models", return_value=[]), \
                 patch("modules.image_analysis.single_analyzer.save_analysis_gpt_ui"), \
                 patch("modules.image_analysis.history_pickup.discover_generation_checkpoints",
                       return_value=[str(checkpoint)]), \
                 patch("modules.image_analysis.history_pickup.HistoryPickupDialog", PickupDialog), \
                 patch("modules.image_analysis.single_analyzer.QMessageBox.information"), \
                 patch("utils.analysis_gen.publish_final_output",
                       return_value=str(process.parent.parent / "65ebea5b_final.jpg")) as publish:
                tab = SingleAnalyzerWidget(
                    config_getter_func=lambda: {}, img_config_getter_func=lambda: {},
                    styles_getter_func=lambda: {}, save_img_cfg_callback=lambda: None,
                    upscale_options_getter_func=lambda: {}, persist_ui_state=False,
                )
                unrelated = tab._create_history_record(1, "", datetime.datetime.now(), "other123")
                unrelated["status"] = "error"
                tab._insert_history_record(unrelated)
                with patch.object(tab, "_import_generation_checkpoint",
                                  side_effect=AssertionError("publish must not import a task")):
                    self.assertTrue(tab._pickup_generation_history(unrelated))
                self.assertEqual(tab.history_list.count(), 1)
                self.assertEqual(len(tab._analysis_history), 1)
                self.assertNotIn("manual_published", unrelated)
                self.assertEqual(publish.call_args.kwargs["task_hash"], "65ebea5b")
                tab.close()

    def test_gemini_queue_item_cannot_open_another_tasks_gpt_history(self):
        from pathlib import Path
        from modules.image_analysis.single_analyzer import SingleAnalyzerWidget

        with tempfile.TemporaryDirectory() as temp_dir:
            other = Path(temp_dir) / "data" / "20260928" / "analysis-gpt-image" / "65ebea5b-run"
            other.mkdir(parents=True)
            (other / "generation-checkpoint.json").write_text("{}", encoding="utf-8")
            with patch("modules.image_analysis.single_analyzer.BASE_DIR", temp_dir), \
                 patch("modules.image_analysis.single_analyzer.list_esrgan_models", return_value=[]), \
                 patch("modules.image_analysis.single_analyzer.save_analysis_gpt_ui"), \
                 patch("modules.image_analysis.single_analyzer.QMessageBox.information") as info, \
                 patch("modules.image_analysis.single_analyzer.QMessageBox.warning") as warn, \
                 patch("modules.image_analysis.history_pickup.HistoryPickupDialog") as pickup:
                tab = SingleAnalyzerWidget(
                    config_getter_func=lambda: {}, img_config_getter_func=lambda: {},
                    styles_getter_func=lambda: {}, save_img_cfg_callback=lambda: None,
                    upscale_options_getter_func=lambda: {}, persist_ui_state=False,
                )
                gemini = tab._create_history_record(1, "", datetime.datetime.now(), "gemini01")
                self.assertFalse(tab._pickup_generation_history(gemini))
                # 别的任务的断点不能被打开，也不能吞掉成"拾取历史失败"
                pickup.assert_not_called()
                warn.assert_not_called()
                # 这条 Gemini 记录既没有提示词、也没有可用源图 → 只提示不能重新分析，不新建任务
                self.assertIn("无法重新分析", info.call_args.args[1])
                self.assertEqual(tab.history_list.count(), 0)
                self.assertEqual(tab._analysis_history, {})
                tab.close()

    def test_random_style_resolves_at_task_entry_and_stays_fixed_for_generation(self):
        from modules.image_analysis.single_analyzer import SingleAnalyzerWidget, RANDOM_STYLE_LABEL

        styles = {
            "默认(无附加)": {"prompt": "", "enabled": True},
            "style-A": {"prompt": "A", "enabled": True},
            "style-B": {"prompt": "B", "enabled": True},
            "disabled": {"prompt": "X", "enabled": False},
        }

        class Signal:
            def connect(self, *_args):
                pass

        class Worker:
            def __init__(self, *_args, **_kwargs):
                self.log_signal = Signal()
                self.finish_signal = Signal()
                self.finished = Signal()

            def start(self):
                pass

        with patch("modules.image_analysis.single_analyzer.list_esrgan_models", return_value=[]), \
             patch("modules.image_analysis.single_analyzer.save_analysis_gpt_ui"), \
             patch("modules.image_analysis.single_analyzer.WorkerThread", Worker), \
             patch("modules.image_analysis.single_analyzer.get_single_analyzer_missing_prompt_files",
                   return_value=[]), \
             patch("modules.image_analysis.single_analyzer.random.choice", return_value="style-B") as choice:
            tab = SingleAnalyzerWidget(
                config_getter_func=lambda *_args: ("https://example.invalid", "key", "model"),
                img_config_getter_func=lambda: ("https://example.invalid", "key", "model", "aigc2d"),
                styles_getter_func=lambda: styles, save_img_cfg_callback=lambda: None,
                upscale_options_getter_func=lambda: {}, persist_ui_state=False,
            )
            tab.update_styles(["默认(无附加)", "style-A", "style-B"])
            self.assertEqual(tab.main_style_combo.itemText(1), RANDOM_STYLE_LABEL)
            tab.main_style_combo.setCurrentText(RANDOM_STYLE_LABEL)
            worker = tab._launch_analysis_task("sample.jpg")
            self.assertEqual(worker.meta_style_name, "style-B")
            self.assertEqual(choice.call_count, 1)
            self.assertNotIn("disabled", choice.call_args.args[0])
            record = tab._analysis_history[worker.meta_task_id]
            self.assertEqual(record["style_name"], "style-B")
            self.assertIn("画风：style-B", tab.history_list.item(0).text())

            worker.last_status = "success"
            tab.auto_gen_orig_cb.setChecked(False)
            tab.auto_gen_ref_cb.setChecked(False)
            with tempfile.TemporaryDirectory() as output_dir, \
                 patch("modules.image_analysis.single_analyzer.resolve_output_target",
                       return_value=(output_dir, "result")):
                tab.on_process_finished(worker, {
                    "english_description": "a girl", "original_english_description": "a girl",
                    "japanese_title": "test", "aspect_ratio": "1:1",
                })
                result = json.loads((Path(output_dir) / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(result["generation_style_name"], "style-B")

            tab.main_style_combo.setCurrentText("style-A")
            tab.gen_channel_gpt.setChecked(True)
            prompt_bundle = {"task_hash": worker.meta_task_hash, "style_name": worker.meta_style_name,
                             "aspect_ratio": "1:1", "refined_prompt": "a girl"}
            with patch.object(tab, "_start_gpt_image_thread", return_value=True) as start, \
                 patch("modules.image_analysis.single_analyzer.ImageGenWorkerThread",
                       side_effect=AssertionError("test must not start an API thread")):
                self.assertTrue(tab.trigger_image_generation("refined", is_auto=True,
                                                              prompt_bundle=prompt_bundle))
            self.assertEqual(start.call_args.kwargs["selected_style_name"], "style-B")
            self.assertEqual(choice.call_count, 1)
            self.assertEqual(record["generation_styles"], ["style-B"])
            tab.main_style_combo.setCurrentText(RANDOM_STYLE_LABEL)
            with patch.object(tab, "_start_gpt_image_thread", return_value=True) as start, \
                 patch("modules.image_analysis.single_analyzer.ImageGenWorkerThread",
                       side_effect=AssertionError("test must not start an API thread")):
                self.assertTrue(tab.trigger_image_generation("refined", prompt_bundle={
                    "task_hash": worker.meta_task_hash, "aspect_ratio": "1:1",
                    "refined_prompt": "a girl"}))
            self.assertEqual(start.call_args.kwargs["selected_style_name"], "style-B")
            self.assertEqual(choice.call_count, 1)
            tab.close()

    def test_random_single_style_is_not_overwritten_by_global_style_sync(self):
        from app import AppWindow

        def combo(random=False):
            widget = QComboBox()
            widget.addItems(["style-A", "style-B"] + (["随机"] if random else []))
            widget.setCurrentText("随机" if random else "style-A")
            return widget

        single = SimpleNamespace(main_style_combo=combo(random=True))
        window = SimpleNamespace(
            _style_sync_enabled=True, last_used_style="style-A",
            single_analyzer_tab=single,
            prompt_generator_tab=SimpleNamespace(main_style_combo=combo()),
            batch_analyzer_tab=SimpleNamespace(main_style_combo=combo()),
            image_edit_tab=SimpleNamespace(main_style_combo=combo()),
            char_design_tab=SimpleNamespace(main_style_combo=combo()),
            single_gen_debug_tab=SimpleNamespace(main_style_combo=combo()),
            sd_workflow_tab=SimpleNamespace(style_combo=combo()),
            save_text_config=lambda **_kwargs: None,
        )
        AppWindow.sync_selected_style(window, "style-B")
        self.assertEqual(single.main_style_combo.currentText(), "随机")
        self.assertEqual(window.prompt_generator_tab.main_style_combo.currentText(), "style-B")


if __name__ == "__main__":
    unittest.main()
