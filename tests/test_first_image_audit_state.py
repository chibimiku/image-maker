# -*- coding: utf-8 -*-
"""首图审计的「失败分级」与队列状态机（用户 2026-10-07 反馈「有结果的也变红」）。

实盘证据（`data/20261007/*-first-image-audit.json`）：
- `038c50fb_015015-…`：第 1 张判 fail → 重画第 2 张 → **审计请求本身 ReadTimeout**
  → 旧逻辑按「重画额度用尽、全部不合格」把已经出好图的队列标红；
- `dd86135c_013205-…` / `953e2666_014644-…`：第 1 张 fail、第 2 张审计期间倒计时到点
  → 旧逻辑按「没有产物」标红。

本文件把「审计把图打回」与「审计自己没跑完」这两件事钉开：前者才需要人工复核，
后者（超时/400/被取消/时间不够）图是好的，按「已完成（审计未完成）」交付、只提示不判失败。
"""

import datetime
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

import modules.image_analysis.single_analyzer as sa
from modules.image_analysis.single_analyzer import (
    ImageGenWorkerThread,
    SingleAnalyzerWidget,
    resolve_first_image_audit_timeout,
)
from utils import first_image_audit as fa


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _analyzer(monkeypatch, tmp_path):
    monkeypatch.setattr(sa, "list_esrgan_models", lambda: ["realesrgan-x4plus"])
    monkeypatch.setattr(sa, "analysis_gpt_ui_path", lambda: str(tmp_path / "ui_state.json"))
    widget = SingleAnalyzerWidget(
        config_getter_func=lambda: {"base_url": "http://x", "api_key": "k", "model": "m"},
        img_config_getter_func=lambda: {"base_url": "http://x", "api_key": "k",
                                        "model": "m", "api_type": "aigc2d"},
        styles_getter_func=lambda: {},
        save_img_cfg_callback=lambda *a, **k: None,
        ar_policy_getter_func=lambda: {"override_second": "", "policy": "keep"},
        nsfw_default_getter_func=lambda: False,
        upscale_options_getter_func=lambda: {},
        outfit_style_history_getter_func=lambda: [],
        outfit_style_default_getter_func=lambda: "",
    )
    return widget


def _seed_record(widget, task_hash="abc123", task_id="analysis-1"):
    """真录一条队列记录（会插进列表）——队列行文案也在这条链上被断言。"""
    record = widget._create_history_record(1, "C:/tmp/girl.png", datetime.datetime.now(), task_hash,
                                           source_path="C:/tmp/girl.png")
    record["status"] = "running"
    record["status_text"] = widget._status_to_text("running")
    widget._insert_history_record(record)
    return record["task_id"]


class _GenThread:
    """生图线程替身：带审计摘要，够队列收尾用。"""

    def __init__(self, task_hash="abc123", audit_summary=None, status="success"):
        self.meta_task_hash = task_hash
        self.meta_task_id = ""
        self.meta_thread_no = 2
        self.meta_analysis_thread_no = 1
        self.meta_prompt_type = "refined"
        self.meta_is_auto = False
        self.meta_auto_group_id = None
        self.meta_analysis_json_path = ""
        self.last_status = status
        self.failed_stage = ""
        self.checkpoint_path = ""
        self.audit_summary = audit_summary or {}
        self.retry_events = []


def _audit_summary(stopped_reason, *, suspended=False, rounds=2):
    return {
        "stopped_reason": stopped_reason,
        "suspended": suspended,
        "published": "D:/board/abc123_000000-first-abcdef.png",
        "candidates": [{"round": index + 1, "file": f"D:/board/cand-{index + 1}.png", "audit": {}}
                       for index in range(rounds)],
    }


# ── 1. 「审计打回」与「审计没跑完」必须分开 ─────────────────────────────────

def test_audit_timeout_does_not_burn_the_redraw_and_keeps_the_picture():
    calls = []
    outcome = fa.generate_with_audit(
        lambda prompt, round_index: calls.append(round_index) or ["D:/board/cand-1.png"],
        prompt="P", max_attempts=2,
        audit=lambda *a: (_ for _ in ()).throw(
            TimeoutError("HTTPSConnectionPool(host='new.aigc2d.com'): Read timed out. (read timeout=180)")),
        content="C", publish_dir="", publish_name="")
    assert calls == [1]                                  # 不再白花第二张
    assert outcome["stopped_reason"] == "audit_failed"
    assert outcome["suspended"] is False                 # 审计没跑成 ≠ 图不合格
    assert outcome["published"] == "D:/board/cand-1.png"  # 产物照常交付
    assert "timed out" in outcome["candidates"][0]["audit"]["audit_error"].lower()


def test_missing_verdict_fields_still_triggers_one_redraw():
    """模型漏答字段（不是接口失败）仍然补画一次 —— 这是原设计，别被上面的修复顺手删掉。"""
    calls = []
    incomplete = {"summary": "no booleans"}

    def generate(prompt, round_index):
        calls.append(round_index)
        return [f"D:/board/cand-{round_index}.png"]

    outcome = fa.generate_with_audit(generate, prompt="P", max_attempts=2, audit=lambda *a: incomplete,
                                     content="C", publish_dir="", publish_name="")
    assert calls == [1, 2]
    assert outcome["stopped_reason"] == "exhausted"


def test_only_explicitly_rejected_candidates_are_suspended():
    rejected = {"conclusion_valid": True, "verdict": "fail", "content_ok": False, "style_ok": False,
                "copied_from_reference": False,
                "major_issues": [{"element": "hair", "severity": "major", "confidence": 0.9}]}
    unknown = {"conclusion_valid": False, "verdict": "unknown", "audit_error": "ReadTimeout"}
    assert fa.suspended([{"file": "a", "audit": rejected}, {"file": "b", "audit": rejected}]) is True
    assert fa.suspended([{"file": "a", "audit": rejected}, {"file": "b", "audit": unknown}]) is False
    assert fa.suspended([{"file": "a", "audit": unknown}]) is False


# ── 2. 队列状态机：审计超时/取消不再标红 ───────────────────────────────────

def test_queue_turns_green_with_warning_when_the_audit_timed_out(qapp, monkeypatch, tmp_path):
    widget = _analyzer(monkeypatch, tmp_path)
    task_id = _seed_record(widget)
    thread = _GenThread(audit_summary=_audit_summary("audit_failed"))
    widget._active_img_threads.append(thread)

    widget.on_image_generation_finished(thread, ["D:/board/abc123_000000-first-abcdef.png"])
    widget._on_image_thread_stopped(thread)

    record = widget._analysis_history[task_id]
    assert record["status"] == "success"                 # 绿，不再红
    assert record["status_text"] == "已完成"
    assert record["pipeline_error"] == ""
    assert "审计" in record["pipeline_warning"]
    text = widget.history_list.item(0).text()
    assert "⚠️" in text and "审计" in text               # 提示仍然看得见
    widget.close()


def test_queue_turns_green_when_the_audit_was_cancelled_mid_flight(qapp, monkeypatch, tmp_path):
    widget = _analyzer(monkeypatch, tmp_path)
    task_id = _seed_record(widget)
    thread = _GenThread(audit_summary=_audit_summary("audit_cancelled"))
    widget._active_img_threads.append(thread)

    widget.on_image_generation_finished(thread, ["D:/board/abc123_000000-first-abcdef.png"])
    widget._on_image_thread_stopped(thread)

    record = widget._analysis_history[task_id]
    assert record["status"] == "success"
    assert "审计" in record["pipeline_warning"]
    widget.close()


def test_queue_still_marks_explicitly_rejected_candidates_for_review(qapp, monkeypatch, tmp_path):
    """所有候选都被审计明确打回 → 仍然保持红色人工复核（这条不能被上面的修复放宽）。"""
    widget = _analyzer(monkeypatch, tmp_path)
    task_id = _seed_record(widget)
    thread = _GenThread(audit_summary=_audit_summary("exhausted", suspended=True))
    widget._active_img_threads.append(thread)

    widget.on_image_generation_finished(thread, ["D:/board/abc123_000000-first-abcdef.png"])
    widget._on_image_thread_stopped(thread)

    record = widget._analysis_history[task_id]
    assert record["status"] == "error"
    assert "首图审计未通过" in record["pipeline_error"]
    widget.close()


# ── 3. 超时预算与审计时限 ─────────────────────────────────────────────────

def test_timeout_budget_covers_the_audit_calls(qapp, monkeypatch, tmp_path):
    """倒计时预算 = 单张 × 张数 + 每次审计的窗口 —— 审计时间不能再被漏算。"""
    widget = _analyzer(monkeypatch, tmp_path)
    captured = {}
    monkeypatch.setattr(widget, "_start_image_gen_runtime",
                        lambda seconds: captured.update({"budget": seconds}))
    monkeypatch.setattr(sa, "ImageGenWorkerThread", _CapturingThread)
    monkeypatch.setattr(widget, "_gpt_image_channel_active", lambda: False)
    monkeypatch.setattr(widget, "gemini_audit_retry", type("C", (), {"isChecked": lambda self: True})())
    monkeypatch.setattr(widget, "gemini_audit_attempts",
                        type("C", (), {"currentData": lambda self: 2})())
    monkeypatch.setattr(sa, "build_first_image_audit_callback",
                        lambda *a, **k: (lambda candidate, prompt, round_index: {}))
    monkeypatch.setattr(sa, "first_image_audit_settings",
                        lambda *a, **k: {"enabled": True, "max_attempts": 2, "timeout_seconds": 180})
    monkeypatch.setattr(widget, "_note_generation_style", lambda *a, **k: None)
    monkeypatch.setattr(widget, "_note_generation_params", lambda *a, **k: None)
    monkeypatch.setattr(widget, "validate_wardrobe_style", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(widget, "get_timeout_seconds", lambda: 120)

    context = {"task_hash": "abc123", "refined_prompt": "a girl", "original_prompt": "a girl",
               "style_name": "", "aspect_ratio": "1:1", "analysis_json_path": "x.json"}
    try:
        widget.trigger_image_generation("refined", prompt_bundle=context, channel="gemini")
    except Exception as exc:  # noqa: BLE001 - 别的接线缺件不掩盖预算断言
        pytest.skip(f"接线依赖缺失，跳过预算断言: {exc}")
    assert captured.get("budget") == 120 * 2 + 180 * 2
    widget.close()


class _CapturingThread:
    def __init__(self, **kwargs):
        _CapturingThread.last = kwargs
        self.log_signal = _SignalStub()
        self.finish_signal = _SignalStub()
        self.finished = _SignalStub()
        self.meta_thread_no = 1
        self.meta_analysis_thread_no = 1
        self.audit_timeout = kwargs.get("audit_timeout")
        self.audit_retry_budget = None

    def start(self):
        pass

    def deleteLater(self):
        pass


class _SignalStub:
    def connect(self, *args, **kwargs):
        return None


def test_audit_timeout_resolution_defaults_and_clamps():
    assert resolve_first_image_audit_timeout(120, None) == 180
    assert resolve_first_image_audit_timeout(600, None) == 600
    assert resolve_first_image_audit_timeout(120, {"timeout_seconds": 420}) == 420
    assert resolve_first_image_audit_timeout(120, {"timeout_seconds": 10}) == 60
    assert resolve_first_image_audit_timeout(120, {"timeout_seconds": 99999}) == 600
    assert resolve_first_image_audit_timeout("abc", None) == 180


def test_thread_passes_the_budget_and_audit_timeout_down(monkeypatch, tmp_path):
    seen = {}
    real = fa.generate_with_audit

    def spy(generate, **kwargs):
        seen.update(kwargs)
        return real(generate, **kwargs)

    monkeypatch.setattr(fa, "generate_with_audit", spy)
    monkeypatch.setattr(sa, "generate_image_aigc2d",
                        lambda **kwargs: {"saved_files": [], "annotation": {}, "raw_text": ""})
    thread = ImageGenWorkerThread(
        prompt="P", model_name="m", aspect_ratio="1:1", instructions="", api_type="aigc2d",
        image_paths=[], file_prefix="abc123", audit_retry=True,
        audit_callback=lambda *a: {}, audit_max_attempts=2, audit_timeout=240)
    thread.audit_retry_budget = 512.0
    thread.run()
    assert seen["retry_budget"] == 512.0
    assert seen["audit_timeout"] == 240


def test_remaining_budget_shrinks_with_the_countdown(qapp, monkeypatch, tmp_path):
    widget = _analyzer(monkeypatch, tmp_path)
    assert widget._remaining_image_gen_budget() is None          # 没在计时
    widget._img_gen_running = True
    widget._img_gen_deadline = datetime.datetime.now() + datetime.timedelta(seconds=90)
    assert 80 <= widget._remaining_image_gen_budget() <= 92
    widget._img_gen_deadline = datetime.datetime.now() - datetime.timedelta(seconds=5)
    assert widget._remaining_image_gen_budget() == 0.0           # 不会变成负数
    widget.close()
