"""分析链路失败必须留下可追溯的证据。

背景：Step 2~5 曾把异常吞掉、只 print 到 stdout（GUI 用 pythonw 启动时 stdout 被
重定向到 os.devnull），且 `_safe_json_from_response` 传的是 log_callback=None，
导致「Step 2 之后线程直接报处理失败」时，log/<日期>.log 与临时目录里查不到任何原因。
本文件锁定修复后的行为：失败原因既进 GUI 回调，也进 whatai_logger（落盘日志）。

注意：这些用例会故意触发 whatai_logger 的真实 handler，因此 pytest 运行时会在
`log/<当天日期>.log` 里追加几行 Step 2 失败诊断——这正是被测行为本身。
"""

import logging
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

import modules.image_analysis.single_analyzer as single_analyzer_module
from modules.image_analysis.single_analyzer import (
    WorkerThread,
    step_2_refine_description,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _FakeHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture()
def captured_log():
    handler = _FakeHandler()
    single_analyzer_module.logger.addHandler(handler)
    try:
        yield handler
    finally:
        single_analyzer_module.logger.removeHandler(handler)


class _FakeMessage:
    def __init__(self, content, refusal=None):
        self.content = content
        self.refusal = refusal


class _FakeChoice:
    def __init__(self, content, finish_reason="stop", refusal=None):
        self.message = _FakeMessage(content, refusal=refusal)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, content, finish_reason="stop", refusal=None):
        self.choices = [_FakeChoice(content, finish_reason=finish_reason, refusal=refusal)]


class _FakeCompletions:
    def __init__(self, response):
        self._response = response

    def create(self, **kwargs):
        return self._response


class _FakeClient:
    def __init__(self, response):
        self.chat = type("_Chat", (), {"completions": _FakeCompletions(response)})()


STEP1_RESULT = {
    "english_description": "a girl standing in a doorway",
    "japanese_title": "扉の前にて",
    "chinese_title": "门前",
    "short_description": "girl by a door",
    "pixiv_tags": ["女の子"],
    "booru-tags": ["1girl"],
}


def test_step_2_json_failure_is_visible_in_gui_and_file_log(captured_log):
    """非法 JSON：返回 None，但原因要同时进 GUI 回调与落盘日志。"""
    gui_messages = []
    client = _FakeClient(_FakeResponse("这不是 JSON，只是一段自然语言解释。", finish_reason="stop"))

    result = step_2_refine_description(
        STEP1_RESULT, client, "test-model", log_callback=gui_messages.append
    )

    assert result is None
    assert any("Step 2 JSON 解析失败" in m for m in gui_messages), gui_messages
    assert any("原始内容总长度" in m for m in gui_messages), gui_messages
    assert any("Step 2 JSON 解析失败" in m for m in captured_log.messages), captured_log.messages


def test_step_2_empty_content_is_visible_in_gui_and_file_log(captured_log):
    """content 为 None（安全过滤/拒绝响应）：同样要留下 finish_reason 与说明。"""
    gui_messages = []
    client = _FakeClient(_FakeResponse(None, finish_reason="content_filter", refusal="blocked"))

    result = step_2_refine_description(
        STEP1_RESULT, client, "test-model", log_callback=gui_messages.append
    )

    assert result is None
    assert any("content 为 None" in m for m in gui_messages), gui_messages
    assert any("content_filter" in m for m in gui_messages), gui_messages
    assert any("content 为 None" in m for m in captured_log.messages), captured_log.messages


def test_step_2_api_error_is_visible_in_gui_and_file_log(captured_log):
    """SDK 抛异常（如 429/5xx）：异常文本要进 GUI 回调与落盘日志。"""

    class _BoomCompletions:
        def create(self, **kwargs):
            raise RuntimeError("upstream 502 bad gateway")

    gui_messages = []
    client = type("_Client", (), {"chat": type("_Chat", (), {"completions": _BoomCompletions()})()})()

    result = step_2_refine_description(
        STEP1_RESULT, client, "test-model", log_callback=gui_messages.append
    )

    assert result is None
    assert any("502 bad gateway" in m for m in gui_messages), gui_messages
    assert any("502 bad gateway" in m for m in captured_log.messages), captured_log.messages


def test_worker_thread_reports_step_2_failure_reason(qapp, monkeypatch):
    """Step 2 返回 None 时，线程不再静默失败：要有明确的失败行与 error 状态。"""

    class _FakeOpenAI:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(single_analyzer_module, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(single_analyzer_module, "predict_local_booru_tags", lambda *a, **k: [])
    monkeypatch.setattr(single_analyzer_module, "get_local_pixiv_tag_candidates", lambda *a, **k: [])
    monkeypatch.setattr(single_analyzer_module, "step_1_analyze_image", lambda *a, **k: dict(STEP1_RESULT))
    monkeypatch.setattr(single_analyzer_module, "step_2_refine_description", lambda *a, **k: None)

    thread = WorkerThread("dummy.png", "key", "https://example.invalid/v1", "test-model")
    log_lines = []
    thread.log_signal.connect(log_lines.append)

    thread.run()

    assert thread.last_status == "error"
    assert any("Step 2 执行失败" in line for line in log_lines), log_lines
    assert any("正在开始 Step 2" in line for line in log_lines), log_lines
