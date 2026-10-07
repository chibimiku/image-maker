# -*- coding: utf-8 -*-
"""Gemini 重试要在队列上看得见（用户 2026-10-06 要求）。

覆盖四件事：
1. 只认真正代表"又发了一次"的日志（网络重试 / 首图审计重画），别的"重试"字样不许误伤；
2. 任务记录上的重试事件是幂等的、次数按"第几次"算；
3. 队列行会多出一行「已发起重试…｜任务 ID xxxx」，任务详情也有对应字段；
4. 标题栏给的是**实时**在跑数（线程号只增不减，不能拿它当线程数）。
"""

import datetime
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

import modules.image_analysis.single_analyzer as sa
from modules.image_analysis.single_analyzer import (
    AnalysisHistoryDetailDialog,
    SingleAnalyzerWidget,
    build_gemini_retry_note,
    detect_gemini_retry,
    record_gemini_retry,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_widget(monkeypatch, tmp_path):
    monkeypatch.setattr(sa, "list_esrgan_models", lambda: ["realesrgan-x4plus"])
    # 别读/写用户的界面记忆文件
    monkeypatch.setattr(sa, "analysis_gpt_ui_path", lambda: str(tmp_path / "gpt_ui_state.json"))
    return SingleAnalyzerWidget(
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


def _insert_record(widget, thread_no, status, task_hash, **extra):
    record = widget._create_history_record(
        thread_no, "C:/tmp/girl.png", datetime.datetime.now(), task_hash, source_path="C:/tmp/girl.png")
    record["status"] = status
    record["status_text"] = widget._status_to_text(status)
    record.update(extra)
    widget._insert_history_record(record)
    return record


class _FakeThread:
    def __init__(self, task_id="", task_hash=""):
        self.meta_task_id = task_id
        self.meta_task_hash = task_hash
        self.meta_thread_no = 1
        self.meta_analysis_thread_no = 1


# ── 1. 日志识别 ────────────────────────────────────────────────────────────

def test_detects_backend_network_retry_line():
    """api_backend 的 AIGC2D 重试行（真实日志格式）。"""
    assert detect_gemini_retry("[生成/api] 第 2 次重试...") == ("network", 2)


def test_detects_audit_redraw_line():
    """first_image_audit 的第二次出图 —— 那就是审计打回后的重画。"""
    assert detect_gemini_retry("🎨 第 2/3 次出图…") == ("audit", 2)


@pytest.mark.parametrize("line", [
    "🔁 重跑 Gemini 生图：线程#3 / abc12345 → 新队列任务 线程#4",
    "失败重试: 最多 5 次，间隔 5 分钟",
    "⏳ 等待重试（第 1/5 次）：剩余 4 分 59 秒",
    "请求终止所有正在进行的分析任务；若此刻在等待重试，会立刻中断等待",
    "📁 重试失败文件：3 个",
])
def test_other_retry_wording_is_ignored(line):
    assert detect_gemini_retry(line) is None


# ── 2. 记录上的重试事件 ────────────────────────────────────────────────────

def test_network_retry_is_recorded_once_per_attempt():
    record = {}
    wrote, count, _stamp = record_gemini_retry(record, "network", 1)
    assert wrote is True and count == 1
    # 同一个 attempt 重复送到不算第二次
    wrote, count, _stamp = record_gemini_retry(record, "network", 1)
    assert wrote is False and count == 1
    wrote, count, _stamp = record_gemini_retry(record, "network", 2)
    assert wrote is True and count == 2
    assert [row["attempt"] for row in record["gemini_retry_events"]] == [1, 2]
    assert "已发起重试" in record["gemini_retry_note"]
    assert "Gemini 请求重试 2 次" in record["gemini_retry_note"]


def test_audit_redraw_and_network_retry_are_summarised_together():
    record = {}
    record_gemini_retry(record, "audit", 2, at=datetime.datetime(2026, 10, 6, 14, 32, 7))
    record_gemini_retry(record, "network", 1, at=datetime.datetime(2026, 10, 6, 14, 33, 9))
    note = build_gemini_retry_note(record)
    assert "Gemini 请求重试 1 次" in note
    assert "首图审计重画 1 次" in note
    assert "最近 14:33:09" in note


def test_unknown_retry_kind_is_rejected():
    record = {}
    wrote, count, _stamp = record_gemini_retry(record, "whatever", 1)
    assert wrote is False and count == 0
    assert build_gemini_retry_note(record) == ""


# ── 3. 队列行与任务详情 ────────────────────────────────────────────────────

def test_queue_row_shows_retry_hint_with_task_id(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch, tmp_path)
    record = _insert_record(widget, 7, "running", "abc12345")
    widget._note_gemini_retry("abc12345", "network", 1)
    widget._note_gemini_retry("abc12345", "audit", 2)

    text = widget.history_list.item(0).text()
    assert "已发起重试" in text
    assert "任务 ID abc12345" in text
    assert "Gemini 请求重试 1 次" in text
    assert "首图审计重画 1 次" in text
    # 原有的状态/线程号/任务号不能被顶掉
    assert "线程#7" in text and "[进行中" in text
    assert record["gemini_retry_note"] in widget.history_list.item(0).toolTip()
    widget.close()


def test_queue_row_without_retry_stays_two_lines(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch, tmp_path)
    _insert_record(widget, 1, "success", "hash0001")
    assert widget.history_list.item(0).text().count("\n") == 1
    widget.close()


def test_repeated_retry_log_lines_do_not_duplicate_the_hint(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch, tmp_path)
    _insert_record(widget, 2, "running", "hash0002")
    widget._note_gemini_retry("hash0002", "network", 1)
    widget._note_gemini_retry("hash0002", "network", 1)
    text = widget.history_list.item(0).text()
    assert text.count("已发起重试") == 1
    widget.close()


def test_task_details_list_every_retry_and_the_task_id(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch, tmp_path)
    record = _insert_record(widget, 3, "success", "hash0003")
    widget._note_gemini_retry("hash0003", "network", 1)
    widget._note_gemini_retry("hash0003", "audit", 2)

    dialog = AnalysisHistoryDetailDialog(record, widget)
    detail = dialog._build_detail_text()
    assert "已发起重试: 2 次" in detail
    assert "重试任务 ID: hash0003" in detail
    assert "Gemini 请求重试 1 次" in detail and "首图审计重画 1 次" in detail
    assert "已发起重试" in dialog._build_summary_text()
    dialog.close()
    widget.close()


def test_thread_log_hook_marks_the_record(qapp, monkeypatch, tmp_path):
    """生图线程的日志出口负责认重试 —— 模拟线程真实吐出的两行日志。"""
    widget = _make_widget(monkeypatch, tmp_path)
    _insert_record(widget, 4, "running", "hash0004")
    thread = _FakeThread(task_hash="hash0004")

    widget._on_gemini_thread_log(thread, "[生成/api] 第 1 次重试...")
    widget._on_gemini_thread_log(thread, "🎨 第 2/3 次出图…")

    text = widget.history_list.item(0).text()
    assert "Gemini 请求重试 1 次" in text
    assert "首图审计重画 1 次" in text
    widget.close()


# ── 4. 队列摘要（线程号只增不减，实时数要看这里） ──────────────────────────

def test_queue_summary_counts_only_live_tasks(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch, tmp_path)
    _insert_record(widget, 1, "success", "hashA")
    _insert_record(widget, 2, "error", "hashB")
    running = _insert_record(widget, 3, "running", "hashC")

    widget._active_analysis_threads.append(_FakeThread(task_id=running["task_id"]))
    widget._refresh_queue_summary()
    assert "队列 3 条" in widget.history_summary_label.text()
    assert "进行中 1" in widget.history_summary_label.text()
    assert "分析 1" in widget.history_summary_label.text()

    # 线程退出 → 线程号序号不回收（还是 #3），但实时在跑数必须归零
    widget._active_analysis_threads.clear()
    widget._refresh_queue_summary()
    assert "进行中 0" in widget.history_summary_label.text()
    assert "线程#3" in widget.history_list.item(0).text()
    widget.close()


def test_queue_summary_counts_post_process_threads_too(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch, tmp_path)
    record = _insert_record(widget, 1, "running", "hashD")
    record["status"] = "running"
    widget._active_post_threads.append(_FakeThread(task_hash="hashD"))
    widget._refresh_queue_summary()
    assert "进行中 1" in widget.history_summary_label.text()
    assert "后处理 1" in widget.history_summary_label.text()
    widget.close()


def test_finished_task_numbers_are_not_recycled(qapp, monkeypatch, tmp_path):
    """线程号是累计序号：跑完不回收（用户看到的「线程#37」就是它），回收的是实时计数。"""
    widget = _make_widget(monkeypatch, tmp_path)
    for _ in range(5):
        _insert_record(widget, widget._next_thread_no("_analysis_thread_seq"), "success", "hashX")
    assert widget._analysis_thread_seq == 5
    assert "进行中 0" in widget.history_summary_label.text()
    widget.close()
