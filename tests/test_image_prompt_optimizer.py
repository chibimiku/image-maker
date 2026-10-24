import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication, QPlainTextEdit, QPushButton, QWidget
from utils import image_prompt_optimizer as runtime
from utils import image_prompt_optimizer_widget as ui
from utils.analysis_fallback import FallbackConfig, load_fallback_config


def response(text, finish="stop"):
    return NS(choices=[NS(finish_reason=finish, message=NS(content=text, refusal=None))])


class PromptOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait_finished(self, control):
        deadline = time.monotonic() + 3
        while control.busy and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertFalse(control.busy)

    def test_invalid_results(self):
        cases = [("", "stop"), ("{}", "stop"), ("not json", "stop"),
                 (json.dumps({"optimized_prompt": "x" * 2001}), "stop"),
                 (json.dumps({"optimized_prompt": "A complete illustration"}), "length"),
                 (json.dumps({"optimized_prompt": "I cannot help with that request."}), "stop")]
        for text, finish in cases:
            with self.subTest(text=text[:50]), self.assertRaises(ValueError):
                runtime.parse_optimized_prompt(response(text, finish))

    def test_primary_and_refusal_fallback(self):
        calls = []
        class Client:
            def __init__(self, primary):
                self.primary = primary
                self.base_url = "https://primary.invalid/v1" if primary else "https://fallback.invalid/v1"
                self.chat = NS(completions=NS(create=self.create))
            def create(self, **kwargs):
                calls.append((self.primary, kwargs))
                return response(None, "content_filter") if self.primary else response(json.dumps({"optimized_prompt": "A horizontal Halloween garden illustration."}))
            def close(self):
                pass
        cfg = {"base_url": "https://primary.invalid/v1", "api_key": "test", "model": "configured-primary"}
        fb = FallbackConfig(enabled=True, base_url="https://fallback.invalid/v1", api_key="test", model="configured-fallback")
        with patch.object(runtime, "load_text_api_config", return_value=cfg), patch.object(runtime, "OpenAI", return_value=Client(True)), patch.object(runtime, "load_fallback_config", return_value=fb), patch("utils.analysis_fallback._make_client", return_value=Client(False)):
            self.assertIn("Halloween", runtime.optimize_image_prompt("万圣节清新", {"size": "1536x1024", "style": "puracotte"}))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1]["messages"], calls[1][1]["messages"])
        self.assertEqual(calls[1][1]["model"], "configured-fallback")
        self.assertEqual(json.loads(calls[0][1]["messages"][1]["content"])["brief"], "万圣节清新")

    def test_text_fallback_keeps_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"nsfw_base_url": "https://example.invalid/v1", "nsfw_model": "deepseek-v4-pro", "nsfw_api_key": "test"}), encoding="utf-8")
            with patch("utils.analysis_fallback._first_env_value", return_value=("", "")):
                self.assertEqual(load_fallback_config(str(path), require_vision=False).model, "deepseek-v4-pro")

    def test_primary_success_does_not_dispatch_fallback(self):
        cfg = {"base_url": "https://example.invalid/v1", "api_key": "test", "model": "configured"}
        client = NS(chat=NS(completions=NS(create=lambda **kw: response(json.dumps({"optimized_prompt": "A fresh garden illustration."})))), close=lambda: None)
        with patch.object(runtime, "load_text_api_config", return_value=cfg), patch.object(runtime, "OpenAI", return_value=client), patch.object(runtime, "load_fallback_config", return_value=FallbackConfig()), patch("utils.analysis_fallback._make_client") as fallback:
            self.assertIn("garden", runtime.optimize_image_prompt("清新"))
            fallback.assert_not_called()

    def test_thread_undo_stale_and_failure(self):
        owner = QWidget(); owner.generate_btn = QPushButton(owner)
        editor = QPlainTextEdit(owner); editor.setPlainText("万圣节清新")
        threads, messages = [], []
        release = threading.Event()
        def optimize(*args, **kwargs):
            threads.append(threading.get_ident()); release.wait(2)
            return "A horizontal Halloween illustration."
        context = {"style": "puracotte"}
        attachments = ["original.png"]
        control = ui.PromptOptimizerControl(owner, editor, lambda: context, messages.append,
                                           state_getter=lambda: tuple(attachments))
        with patch.object(ui, "optimize_image_prompt", side_effect=optimize):
            control.start(); self.assertFalse(owner.generate_btn.isEnabled()); self.assertFalse(control.button.isEnabled())
            control.start(); release.set(); self.wait_finished(control)
            self.assertEqual(len(threads), 1); self.assertNotEqual(threads[0], threading.get_ident())
            self.assertEqual(editor.toPlainText(), "A horizontal Halloween illustration.")
            editor.undo(); self.assertEqual(editor.toPlainText(), "万圣节清新")
            release.clear(); control.start(); editor.setPlainText("新文案"); release.set(); self.wait_finished(control)
            self.assertEqual(editor.toPlainText(), "新文案")
            release.clear(); control.start(); context["style"] = "other"; release.set(); self.wait_finished(control)
            self.assertEqual(editor.toPlainText(), "新文案")
            release.clear(); control.start(); attachments[0] = "replacement.png"; release.set(); self.wait_finished(control)
            self.assertEqual(editor.toPlainText(), "新文案")
        with patch.object(ui, "optimize_image_prompt", side_effect=RuntimeError("offline failure")):
            control.start(); self.wait_finished(control)
        self.assertEqual(editor.toPlainText(), "新文案"); self.assertTrue(owner.generate_btn.isEnabled())
        self.assertIn("offline failure", messages[-1]); owner.close()

    def test_two_tabs_context_and_repaint_disabled(self):
        import modules.image_generation.gpt_image2_tab as gpt
        import modules.image_generation.single_gen_debug_tab as gemini
        with tempfile.TemporaryDirectory() as tmp, patch.object(gpt, "CONFIG_IMAGE_FILE", str(Path(tmp) / "config.json")), patch.object(gemini, "SINGLE_DEBUG_UI_STATE_FILE", str(Path(tmp) / "state.json")):
            Path(tmp, "config.json").write_text("{}", encoding="utf-8")
            widget = gpt.GptImage2Widget(); widget.prompt_edit.setPlainText("Halloween")
            widget.size_combo.setCurrentIndex(widget.size_combo.findData("1536x1024"))
            self.assertTrue(widget.optimize_prompt_btn.isEnabled())
            self.assertEqual(widget._prompt_optimizer_context()["size"], "1536x1024")
            widget.mode_combo.setCurrentText(gpt.MODE_REPAINT); self.assertFalse(widget.optimize_prompt_btn.isEnabled()); widget.close()
            widget = gemini.SingleGenDebugWidget(lambda: ("", "", "", "aigc2d"), lambda: {}, lambda: None)
            widget.prompt_edit.setPlainText("Halloween"); widget.aspect_ratio_combo.setCurrentText("16:9")
            self.assertTrue(widget.optimize_prompt_btn.isEnabled())
            self.assertEqual(widget._prompt_optimizer_context()["aspect_ratio"], "16:9"); widget.close()


if __name__ == "__main__":
    unittest.main()
