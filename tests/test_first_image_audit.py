# -*- coding: utf-8 -*-
"""Gemini 直出的「审计 + 重画」离线回归。

覆盖两层：
1) 判定层 `utils.first_image_audit`：严格结论校验、重试判定、重画指令、候选选择、落盘提升；
2) 接线层 `modules.image_analysis.single_analyzer`：开关默认值、UI 契约、线程真的走审计分支。

全部离线：`generate` 与 `audit` 都是假函数，不发任何 API 请求。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image

from utils import first_image_audit as fa


def _audit_json(**overrides):
    payload = {
        "content_ok": True, "style_ok": True, "copied_from_reference": False,
        "verdict": "pass", "content_issues": [], "style_issues": [],
        "copied_elements": [], "uncertain": [], "confidence": 0.9, "summary": "ok",
    }
    payload.update(overrides)
    return payload


COPied_FAIL = _audit_json(
    content_ok=False, style_ok=True, copied_from_reference=True, verdict="fail",
    content_issues=[{"element": "pose", "expected": "crouching", "observed": "seated reading",
                     "severity": "major", "confidence": 0.9}],
    copied_elements=["antlers", "rose frame"], summary="copied the sample")


class NormalizeAuditTests(unittest.TestCase):
    def test_missing_fields_are_not_a_pass(self):
        report = fa.normalize_audit({"verdict": "pass"})
        self.assertFalse(report["conclusion_valid"])
        self.assertEqual(report["verdict"], "unknown")
        self.assertTrue(fa.should_retry(report)[0])

    def test_non_boolean_ok_flags_are_missing_conclusions(self):
        report = fa.normalize_audit(_audit_json(content_ok="true"))
        self.assertFalse(report["conclusion_valid"])
        self.assertIn("content_ok", report["missing_conclusion_fields"])

    def test_pass_with_major_issue_is_treated_as_fail(self):
        report = fa.normalize_audit(_audit_json(content_issues=[
            {"element": "hair", "expected": "blonde", "observed": "white",
             "severity": "major", "confidence": 0.9}]))
        self.assertEqual(report["verdict"], "fail")
        self.assertTrue(fa.should_retry(report)[0])

    def test_low_confidence_major_does_not_force_a_redraw(self):
        report = fa.normalize_audit(_audit_json(content_ok=False, verdict="uncertain",
                                                content_issues=[
                                                    {"element": "hair", "expected": "blonde",
                                                     "observed": "white", "severity": "major",
                                                     "confidence": 0.4}]))
        self.assertEqual(report["major_issues"], [])
        retry, reason = fa.should_retry(report)
        self.assertFalse(retry, reason)

    def test_uncertain_alone_does_not_trigger_a_redraw(self):
        report = fa.normalize_audit({"verdict": "pass"})
        # 三个布尔都缺 → 结论无效（会补画一次）
        self.assertFalse(report["conclusion_valid"])
        self.assertTrue(fa.should_retry(report)[0])
        # 结论完整但只报了「有些地方看不清」→ 不重画（抽卡解决不了不确定）
        valid = fa.normalize_audit(_audit_json(verdict="uncertain", uncertain=["eye colour"]))
        self.assertTrue(valid["conclusion_valid"])
        self.assertFalse(fa.should_retry(valid)[0])

    def test_a_confident_style_failure_also_triggers_a_redraw(self):
        """用户要的就是这条：跟参考画风过于不一致时重画。"""
        report = fa.normalize_audit(_audit_json(verdict="uncertain", content_ok=True,
                                                style_ok=False, copied_from_reference=False,
                                                style_issues=[{"element": "line_weight",
                                                               "expected": "delicate",
                                                               "observed": "heavy generic ink",
                                                               "severity": "major",
                                                               "confidence": 0.9}]))
        self.assertTrue(report["conclusion_valid"])
        retry, reason = fa.should_retry(report)
        self.assertTrue(retry)
        self.assertIn("画法", reason)

    def test_copied_reference_content_always_triggers_a_redraw(self):
        report = fa.normalize_audit(COPied_FAIL)
        retry, reason = fa.should_retry(report)
        self.assertTrue(retry)
        self.assertIn("复制", reason)

    def test_json_extraction_handles_fenced_and_noisy_replies(self):
        self.assertEqual(fa.extract_json_object('```json\n{"verdict":"pass"}\n```')["verdict"], "pass")
        self.assertEqual(fa.extract_json_object('noise {"verdict":"fail"} tail')["verdict"], "fail")
        self.assertEqual(fa.extract_json_object("not json"), {})


class RetryDirectiveTests(unittest.TestCase):
    def test_retry_prompt_carries_reasons_content_and_forbidden_items(self):
        report = fa.normalize_audit(COPied_FAIL)
        text = fa.build_retry_prompt(report, "CONTENT-ANCHOR-TEXT",
                                     forbidden=["antlers", "rose frame"], attempt=2, max_attempts=2)
        self.assertIn("attempt 2 of 2", text)
        self.assertIn("CONTENT-ANCHOR-TEXT", text)
        self.assertIn("antlers", text)
        self.assertIn("rose frame", text)
        self.assertIn("pose: expected crouching; the picture showed seated reading", text)

    def test_retry_prompt_falls_back_to_default_forbidden_items(self):
        text = fa.build_retry_prompt(fa.normalize_audit(COPied_FAIL), "C")
        self.assertIn(fa.DEFAULT_FORBIDDEN_ITEMS[0], text)

    def test_retry_prompt_accepts_a_dict_without_a_valid_conclusion(self):
        text = fa.build_retry_prompt({}, "C")
        self.assertIn("could not confirm", text)


class CandidateSelectionTests(unittest.TestCase):
    @staticmethod
    def _record(round_index, ok, content_ok=True, style_ok=True):
        payload = _audit_json(verdict="pass" if ok else "fail", content_ok=content_ok,
                              style_ok=style_ok) if ok else COPied_FAIL
        return {"round": round_index, "file": f"a{round_index}.png",
                "audit": fa.normalize_audit(payload)}

    def test_passing_candidate_beats_an_unverified_one(self):
        best = fa.choose_best_candidate([self._record(1, True),
                                         {"round": 2, "file": "b.png", "audit": {}}])
        self.assertEqual(best["file"], "a1.png")

    def test_best_of_failures_prefers_content_then_style(self):
        first = self._record(1, False)
        second = {"round": 2, "file": "b.png",
                  "audit": fa.normalize_audit(COPied_FAIL | {"style_ok": False})}
        self.assertEqual(fa.choose_best_candidate([first, second])["file"], "a1.png")

    def test_no_candidates_yields_empty(self):
        self.assertEqual(fa.choose_best_candidate([]), {})


class GenerateWithAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.publish = self.root / "data"
        self.publish.mkdir()

    def _fake_generate(self, prompts, audits, files):
        calls = []

        def generate(prompt, round_index):
            calls.append(prompt)
            name = self.publish / f"cand-{round_index}.png"
            Image.new("RGB", (16, 16)).save(name)
            return [str(name)]

        def audit(candidate, prompt, round_index):
            audits.append(candidate)
            return files[min(round_index - 1, len(files) - 1)]

        return calls, generate, audit

    def test_fail_then_pass_writes_the_retry_directive_into_the_second_prompt(self):
        calls, generate, audit = self._fake_generate([], [], [COPied_FAIL, _audit_json()])
        outcome = fa.generate_with_audit(generate, prompt="FIRST-PROMPT", max_attempts=2,
                                         audit=audit, content="CONTENT-ANCHOR",
                                         candidate_dir=str(self.publish),
                                         publish_dir=str(self.publish),
                                         publish_name=lambda r: f"final-a{r}.png")
        self.assertEqual(outcome["stopped_reason"], "passed")
        self.assertFalse(outcome["suspended"])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], "FIRST-PROMPT")
        self.assertIn("attempt 2 of 2", calls[1])
        self.assertTrue((self.publish / "final-a2.png").is_file())
        self.assertTrue((self.publish / "cand-1.png").is_file())   # 被拒候选保留
        records = list(self.publish.glob("*first-image-audit*.json"))
        self.assertEqual(len(records), 1)
        record = json.loads(records[0].read_text(encoding="utf-8"))
        self.assertEqual(record["stopped_reason"], "passed")
        self.assertEqual(record["selected_round"], 2)
        self.assertEqual([row["round"] for row in record["candidates"]], [1, 2])

    def test_stops_at_the_cap_and_marks_the_task_for_review(self):
        calls, generate, audit = self._fake_generate([], [], [COPied_FAIL, COPied_FAIL])
        outcome = fa.generate_with_audit(generate, prompt="P", max_attempts=2, audit=audit,
                                        content="C", candidate_dir=str(self.publish),
                                        publish_dir=str(self.publish),
                                        publish_name="final.png")
        self.assertEqual(outcome["stopped_reason"], "exhausted")
        self.assertTrue(outcome["suspended"])
        self.assertEqual(len(calls), 2)          # 没有第三次
        self.assertEqual(len(outcome["audits"]), 2)
        self.assertTrue((self.publish / "final.png").is_file())

    def test_audit_exception_is_not_silently_a_pass(self):
        """审计调用失败：不许当通过，但也不该再花钱重画 —— 交付这张并记下失败原因。

        （2026-10-07 用户反馈「有结果的也变红」：把 ReadTimeout 当"要重画"会白花一张图，
        最后还按「额度用尽、全部不合格」把已经出好的图标红。）
        """
        def generate(prompt, round_index):
            name = self.publish / f"cand-{round_index}.png"
            Image.new("RGB", (16, 16)).save(name)
            return [str(name)]

        def audit(candidate, prompt, round_index):
            raise RuntimeError("audit endpoint down")

        outcome = fa.generate_with_audit(generate, prompt="P", max_attempts=2, audit=audit,
                                         content="C", candidate_dir=str(self.publish),
                                         publish_dir=str(self.publish), publish_name="final.png")
        self.assertEqual(outcome["stopped_reason"], "audit_failed")
        self.assertFalse(outcome["suspended"])          # 审计没跑成 ≠ 图不合格
        self.assertFalse(fa.should_retry(outcome["candidates"][0]["audit"])[0])
        self.assertIn("audit endpoint down", json.dumps(outcome["candidates"][0]["audit"]))
        self.assertTrue((self.publish / "final.png").is_file())

    def test_cancel_before_the_second_round_stops_the_redraw(self):
        calls, generate, audit = self._fake_generate([], [], [COPied_FAIL, _audit_json()])
        state = {"checks": 0}

        def cancel_check():
            state["checks"] += 1
            # 一次循环查三遍：前两遍（轮次开头 / 审计之后）放行，第三遍才判取消
            return state["checks"] >= 3

        outcome = fa.generate_with_audit(generate, prompt="P", max_attempts=3, audit=audit,
                                        content="C", candidate_dir=str(self.publish),
                                        publish_dir=str(self.publish), publish_name="final.png",
                                        cancel_check=cancel_check)
        self.assertEqual(outcome["stopped_reason"], "cancelled")
        self.assertEqual(len(calls), 1)

    def test_cancel_during_the_audit_keeps_the_picture_and_does_not_fail_the_run(self):
        """倒计时在审计期间到点：图已经出好，按「未审完」交付，不许重画也不许算失败。"""
        calls, generate, audit = self._fake_generate([], [], [_audit_json()])
        state = {"audit_done": False}

        def cancel_check():
            return state["audit_done"]

        real_audit = audit

        def audit_and_cancel(path, prompt, round_index):
            result = real_audit(path, prompt, round_index)
            state["audit_done"] = True     # 审计一返回，倒计时到点
            return result

        outcome = fa.generate_with_audit(generate, prompt="P", max_attempts=2,
                                        audit=audit_and_cancel, content="C",
                                        candidate_dir=str(self.publish),
                                        publish_dir=str(self.publish), publish_name="final.png",
                                        cancel_check=cancel_check)
        self.assertEqual(outcome["stopped_reason"], "audit_cancelled")
        self.assertEqual(len(calls), 1)                    # 不再重画
        self.assertFalse(outcome["suspended"])
        self.assertTrue((self.publish / "final.png").is_file())
        self.assertIn("cancel", json.dumps(outcome["candidates"][0]["audit"]).lower())

    def test_tight_budget_skips_the_second_round_instead_of_being_cut_off(self):
        """剩余时间不够再跑一轮「审计+重画」→ 提前收工、用眼前这张（而不是跑一半被掐掉）。"""
        calls, generate, audit = self._fake_generate([], [], [COPied_FAIL, _audit_json()])
        outcome = fa.generate_with_audit(generate, prompt="P", max_attempts=2, audit=audit,
                                        content="C", candidate_dir=str(self.publish),
                                        publish_dir=str(self.publish), publish_name="final.png",
                                        retry_budget=30, audit_timeout=180)
        self.assertEqual(outcome["stopped_reason"], "budget_tight")
        self.assertEqual(len(calls), 1)
        self.assertTrue((self.publish / "final.png").is_file())

    def test_without_an_audit_callback_only_one_image_is_drawn(self):
        calls, generate, _audit = self._fake_generate([], [], [_audit_json()])
        outcome = fa.generate_with_audit(generate, prompt="P", max_attempts=2, audit=None,
                                        candidate_dir=str(self.publish),
                                        publish_dir=str(self.publish), publish_name="final.png")
        self.assertEqual(outcome["stopped_reason"], "audit_disabled")
        self.assertEqual(len(calls), 1)


class AuditSettingsTests(unittest.TestCase):
    def test_defaults_are_on_with_two_attempts(self):
        settings = fa.audit_settings({})
        self.assertTrue(settings["enabled"])
        self.assertEqual(settings["max_attempts"], 2)
        # 单次审计的秒数上限（实盘 ReadTimeout=180 就卡在这个口径上）
        self.assertEqual(settings["timeout_seconds"], fa.DEFAULT_AUDIT_TIMEOUT)

    def test_config_can_disable_and_cap(self):
        settings = fa.audit_settings({fa.CONFIG_ENABLE_KEY: False,
                                      fa.CONFIG_ATTEMPTS_KEY: 99})
        self.assertFalse(settings["enabled"])
        self.assertEqual(settings["max_attempts"], 3)      # 硬上限 3

    def test_timeout_is_configurable_and_clamped(self):
        self.assertEqual(fa.audit_settings({fa.CONFIG_TIMEOUT_KEY: 300})["timeout_seconds"], 300)
        self.assertEqual(fa.audit_settings({fa.CONFIG_TIMEOUT_KEY: 5})["timeout_seconds"], 60)
        self.assertEqual(fa.audit_settings({fa.CONFIG_TIMEOUT_KEY: 99999})["timeout_seconds"], 600)
        self.assertEqual(fa.audit_settings({fa.CONFIG_TIMEOUT_KEY: "abc"})["timeout_seconds"],
                         fa.DEFAULT_AUDIT_TIMEOUT)

    def test_resolve_audit_content_prefers_the_full_anchor(self):
        result = {"gpt_image_prompt": "FULL", "gpt_image_prompt_short": "SHORT",
                  "english_description": "LONG"}
        self.assertEqual(fa.resolve_audit_content(result), "FULL")
        self.assertEqual(fa.resolve_audit_content(result, tier="short"), "SHORT")


class StyleExclusionFieldTests(unittest.TestCase):
    def test_reference_content_exclusions_survive_normalization(self):
        from utils.styles import normalize_style_entry, reference_content_items
        entry = normalize_style_entry({"prompt": "x",
                                       "reference_content_exclusions": ["antlers", " rose frame ", ""]})
        self.assertEqual(entry["reference_content_exclusions"], ["antlers", "rose frame"])
        self.assertEqual(reference_content_items({"s": entry}, "s"), ["antlers", "rose frame"])
        self.assertEqual(normalize_style_entry("plain text")["reference_content_exclusions"], [])


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "Qt 不可用")
class AnalyzerWiringTests(unittest.TestCase):
    """接线层：UI 契约 + 回调组装（不发出图/审计请求）。"""

    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.qapp = QApplication.instance() or QApplication([])

    def _analyzer(self):
        from modules.image_analysis import single_analyzer as sa
        widget = sa.SingleAnalyzerWidget(
            config_getter_func=lambda: {},
            img_config_getter_func=lambda: ({}, "", "", "", "aigc2d"),
            styles_getter_func=lambda: {},
            save_img_cfg_callback=lambda *a, **k: None,
            persist_ui_state=False,
        )
        self.addCleanup(widget.deleteLater)
        return widget

    def test_audit_controls_are_visible_and_default_on(self):
        widget = self._analyzer()
        self.assertTrue(widget.gemini_audit_retry.isChecked())
        self.assertFalse(widget.gemini_audit_retry.isHidden())
        self.assertEqual(widget.gemini_audit_attempts.currentData(), 2)

    def test_gpt_channel_greys_the_audit_controls_out(self):
        widget = self._analyzer()
        widget.gen_channel_gpt.setChecked(True)
        self.assertFalse(widget.gemini_audit_retry.isEnabled())
        self.assertFalse(widget.gemini_audit_retry.isHidden())
        widget.gen_channel_gemini.setChecked(True)
        self.assertTrue(widget.gemini_audit_retry.isEnabled())

    def test_unchecking_the_switch_greys_the_attempt_cap(self):
        widget = self._analyzer()
        widget.gemini_audit_retry.setChecked(False)
        self.assertFalse(widget.gemini_audit_attempts.isEnabled())

    def test_attempt_cap_is_persisted_with_the_ui_state(self):
        widget = self._analyzer()
        widget.gemini_audit_attempts.setCurrentIndex(1)      # 最多 3 张
        state = widget._gpt_pipeline_ui_state()
        self.assertEqual(state["gemini_audit_attempts"], 3)
        self.assertTrue(state["gemini_audit_retry"])

    def test_worker_thread_without_the_audit_keeps_the_plain_path(self):
        from modules.image_analysis import single_analyzer as sa
        thread = sa.ImageGenWorkerThread(prompt="p", model_name="m", aspect_ratio="1:1",
                                        instructions="", image_paths=[])
        self.assertFalse(thread.audit_retry)
        self.assertIsNone(thread.audit_callback)
        self.assertEqual(thread.audit_max_attempts, 1)

    def test_build_first_image_audit_callback_reads_the_analysis_json(self):
        from modules.image_analysis import single_analyzer as sa
        root = Path(tempfile.mkdtemp())
        analysis = root / "task.json"
        analysis.write_text(json.dumps({"gpt_image_prompt": "FULL-ANCHOR",
                                        "gpt_image_prompt_short": "SHORT-ANCHOR"}),
                            encoding="utf-8")
        style_ref = root / "ref.png"
        Image.new("RGB", (8, 8)).save(style_ref)
        captured = {}

        def fake_audit(image_path, analysis_result, text_cfg=None, **kwargs):
            captured.update({"image": image_path, "result": analysis_result, **kwargs})
            return _audit_json()

        with patch.object(fa, "audit_first_image", fake_audit):
            callback = sa.build_first_image_audit_callback(
                {"source_image_path": ""}, str(analysis), str(style_ref),
                style_name="myc0t0xin",
                styles={"myc0t0xin": {"ref_image": str(style_ref),
                                      "reference_content_exclusions": ["antlers"]}},
                log_callback=lambda message: None)
            self.assertTrue(callable(callback))
            report = callback("candidate.png", "PROMPT-USED", 1)
        self.assertEqual(report["verdict"], "pass")
        self.assertEqual(captured["image"], "candidate.png")
        self.assertIn("FULL-ANCHOR", captured["content"])
        self.assertIn("PROMPT-USED", captured["content"])
        self.assertEqual(captured["forbidden"], ["antlers"])
        self.assertEqual(captured["style_ref_path"], os.path.abspath(style_ref))

    def test_callback_is_none_without_a_json_or_reference(self):
        from modules.image_analysis import single_analyzer as sa
        self.assertIsNone(sa.build_first_image_audit_callback({}, "", "", style_name="x",
                                                              styles={}))

    def test_publish_name_keeps_the_hash_in_the_first_field(self):
        from modules.image_analysis import single_analyzer as sa
        name = sa.build_publish_name("b51aa600", "first", ".jpg")
        self.assertTrue(name.startswith("b51aa600_"))
        self.assertIn("-first-", name)
        self.assertTrue(name.endswith(".jpg"))


@unittest.skipUnless(os.environ.get("QT_QPA_PLATFORM") == "offscreen", "Qt 不可用")
class WorkerThreadAuditTests(unittest.TestCase):
    """线程级接线：真的经过 `generate_with_audit`（出图函数被 monkeypatch，不发请求）。"""

    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.qapp = QApplication.instance() or QApplication([])

    def setUp(self):
        from modules.image_analysis import single_analyzer as sa
        self.sa = sa
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.publish = Path(self.temp.name) / "board"
        self.publish.mkdir()
        self.calls = []

        def fake_generate(**kwargs):
            self.calls.append(kwargs)
            name = Path(kwargs["save_sub_dir"]) / f"{kwargs['file_prefix']}.png"
            Image.new("RGB", (12, 12)).save(name)
            return {"saved_files": [str(name)], "annotation": {}, "raw_text": ""}

        self._original = sa.generate_image_aigc2d
        sa.generate_image_aigc2d = fake_generate
        self.addCleanup(lambda: setattr(sa, "generate_image_aigc2d", self._original))

    def _thread(self, audits, attempts=2):
        thread = self.sa.ImageGenWorkerThread(
            prompt="PROMPT", model_name="m", aspect_ratio="1:1", instructions="",
            api_type="aigc2d", image_paths=[], file_prefix="b51aa600",
            audit_retry=True, audit_callback=lambda path, prompt, rnd: audits[rnd - 1],
            audit_max_attempts=attempts, publish_dir=str(self.publish),
            publish_name="b51aa600_000000-first-abcdef.png")
        results = []
        thread.finish_signal.connect(results.append)
        thread.run()
        return thread, results

    def test_second_round_is_drawn_with_the_retry_directive_and_only_one_publish_file(self):
        thread, results = self._thread([COPied_FAIL, _audit_json()])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[0]["file_prefix"], "b51aa600_000000-first-abcdef-a1")
        self.assertEqual(self.calls[1]["file_prefix"], "b51aa600_000000-first-abcdef-a2")
        self.assertIn("attempt 2 of 2", self.calls[1]["prompt"])
        self.assertEqual(thread.last_status, "success")
        self.assertEqual(results[0],
                         [str(self.publish / "b51aa600_000000-first-abcdef.png")])
        # 两张候选都留在目录里，发布名只有一份
        names = sorted(p.name for p in self.publish.iterdir() if p.suffix == ".png")
        self.assertEqual(names, ["b51aa600_000000-first-abcdef-a1.png",
                                 "b51aa600_000000-first-abcdef-a2.png",
                                 "b51aa600_000000-first-abcdef.png"])

    def test_promotion_failure_falls_back_to_the_candidate_file(self):
        outcome = fa.generate_with_audit(
            lambda prompt, round_index: [str(self.publish / "missing.png")],
            prompt="P", max_attempts=1, audit=None,
            candidate_dir=str(self.publish), publish_dir=str(self.publish),
            publish_name="final.png")
        self.assertEqual(outcome["published"], str(self.publish / "missing.png"))

    def test_two_failures_mark_the_run_as_suspended(self):
        thread, results = self._thread([COPied_FAIL, COPied_FAIL])
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(thread.audit_summary["suspended"])
        self.assertEqual(thread.audit_summary["stopped_reason"], "exhausted")
        self.assertEqual(len(results[0]), 1)


if __name__ == "__main__":
    unittest.main()
