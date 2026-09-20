"""LLM 失败重试：429 限流 / 5xx / 超时必须按配置等待后重试，配置类错误立刻失败。"""

import json
import os

import pytest

from utils.llm_retry import (
    DEFAULT_RETRY_INTERVAL_SECONDS,
    DEFAULT_RETRY_TIMES,
    RetrySettings,
    call_with_retry,
    classify_retryable_error,
    format_wait,
    load_retry_settings,
    wait_with_cancel,
)


class RateLimitBoom(Exception):
    """模拟对面返回的 429（消息照抄真实 case）。"""

    status_code = 429

    def __init__(self):
        super().__init__(
            "Error code: 429 - {'error': {'message': 'The request rate limit has been reached. "
            "Please try again later. (request id: 20260912114327242280748llD8BZX8)', "
            "'type': 'new_api_error', 'param': '', 'code': ''}}"
        )


class ServerBoom(Exception):
    status_code = 503


class BadRequestBoom(Exception):
    status_code = 400


class TimeoutBoom(Exception):
    pass


class _FakeClock:
    """替掉 time.sleep：不真的等，只累计时间。"""

    def __init__(self):
        self.slept = 0.0
        self.calls = []

    def __call__(self, seconds):
        self.slept += float(seconds)
        self.calls.append(float(seconds))


def test_classify_recognises_server_side_failures():
    assert classify_retryable_error(RateLimitBoom())[0] is True
    assert "429" in classify_retryable_error(RateLimitBoom())[1]
    assert classify_retryable_error(ServerBoom())[0] is True
    assert classify_retryable_error(TimeoutBoom("Read timed out."))[0] is True
    assert classify_retryable_error(TimeoutError("timed out"))[0] is True
    assert classify_retryable_error(ConnectionResetError("connection reset by peer"))[0] is True


def test_classify_refuses_client_side_failures():
    retryable, reason = classify_retryable_error(BadRequestBoom("bad prompt"))
    assert retryable is False
    assert "400" in reason
    assert classify_retryable_error(ValueError("json 解析失败"))[0] is False


def test_load_retry_settings_defaults_and_overrides(tmp_path):
    missing = tmp_path / "nope.json"
    settings = load_retry_settings(str(missing))
    assert settings.enabled is True
    assert settings.times == DEFAULT_RETRY_TIMES == 5
    assert settings.interval_seconds == DEFAULT_RETRY_INTERVAL_SECONDS == 300

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "text_retry_enabled": False,
                "text_retry_times": 9,
                "text_retry_interval_seconds": 30,
            }
        ),
        encoding="utf-8",
    )
    settings = load_retry_settings(str(config))
    assert settings.enabled is False
    assert settings.times == 9
    assert settings.interval_seconds == 30

    # 坏值要能兜住，不能让分析因为配置写坏而崩
    config.write_text(json.dumps({"text_retry_times": "abc", "text_retry_interval_seconds": None}), encoding="utf-8")
    settings = load_retry_settings(str(config))
    assert settings.times == DEFAULT_RETRY_TIMES
    assert settings.interval_seconds == DEFAULT_RETRY_INTERVAL_SECONDS


def test_call_with_retry_retries_429_then_succeeds():
    clock = _FakeClock()
    logs = []
    attempts = {"n": 0}

    def call():
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise RateLimitBoom()
        return "ok"

    result = call_with_retry(
        call,
        settings=RetrySettings(times=5, interval_seconds=300),
        step_label="Step 1 Vision 请求",
        log_callback=logs.append,
        sleep_func=clock,
    )

    assert result == "ok"
    assert attempts["n"] == 3
    assert clock.slept == 600.0
    assert sum("429 限流" in line for line in logs) == 2
    assert any("第 1/5 次重试" in line for line in logs)


def test_call_with_retry_gives_up_and_reraises_original_error():
    clock = _FakeClock()
    logs = []
    attempts = {"n": 0}

    def call():
        attempts["n"] += 1
        raise RateLimitBoom()

    with pytest.raises(RateLimitBoom):
        call_with_retry(
            call,
            settings=RetrySettings(times=5, interval_seconds=300),
            log_callback=logs.append,
            sleep_func=clock,
        )

    assert attempts["n"] == 6  # 首次 + 5 次重试
    assert clock.slept == 1500.0
    assert any("已重试 5 次仍失败" in line for line in logs)


def test_call_with_retry_does_not_retry_client_errors_or_when_disabled():
    attempts = {"n": 0}

    def bad_request():
        attempts["n"] += 1
        raise BadRequestBoom("400")

    with pytest.raises(BadRequestBoom):
        call_with_retry(bad_request, settings=RetrySettings(times=5, interval_seconds=300), sleep_func=_FakeClock())
    assert attempts["n"] == 1

    attempts["n"] = 0

    def rate_limited():
        attempts["n"] += 1
        raise RateLimitBoom()

    logs = []
    with pytest.raises(RateLimitBoom):
        call_with_retry(
            rate_limited,
            settings=RetrySettings(enabled=False, times=5, interval_seconds=300),
            log_callback=logs.append,
            sleep_func=_FakeClock(),
        )
    assert attempts["n"] == 1
    assert any("已关闭" in line for line in logs)


def test_call_with_retry_aborts_when_cancelled_during_wait():
    clock = _FakeClock()
    logs = []
    state = {"cancelled": False}

    def call():
        state["cancelled"] = True  # 第一次失败后立刻「用户取消」
        raise RateLimitBoom()

    with pytest.raises(RateLimitBoom):
        call_with_retry(
            call,
            settings=RetrySettings(times=5, interval_seconds=300),
            log_callback=logs.append,
            cancel_check=lambda: state["cancelled"],
            sleep_func=clock,
        )

    assert clock.slept == 0.0
    assert any("等待重试期间被取消" in line for line in logs)


def test_wait_with_cancel_reports_progress_every_minute():
    clock = _FakeClock()
    reports = []
    finished = wait_with_cancel(300, sleep_func=clock, report=reports.append, report_interval=60)

    assert finished is True
    assert clock.slept == 300.0
    assert reports == [240.0, 180.0, 120.0, 60.0]
    assert format_wait(300) == "5 分钟"
    assert format_wait(45) == "45 秒"
    assert format_wait(90) == "1 分 30 秒"


def test_step_1_analyze_image_retries_on_429(monkeypatch, tmp_path):
    """Step 1 拿到 429 时应按配置重试，最终成功返回结果。"""
    from PIL import Image

    import modules.image_analysis.single_analyzer as single_analyzer_module

    monkeypatch.setattr(
        single_analyzer_module, "load_retry_settings",
        lambda *a, **k: RetrySettings(times=3, interval_seconds=0),
    )

    image_path = tmp_path / "girl.png"
    Image.new("RGB", (8, 8)).save(image_path)

    payload = json.dumps(
        {
            "english_description": "a girl",
            "japanese_title": "少女",
            "chinese_title": "少女",
            "pixiv_tags": ["女の子"],
            "booru-tags": ["1girl"],
        }
    )

    class _Message:
        content = payload

    class _Choice:
        message = _Message()
        finish_reason = "stop"

    class _Response:
        choices = [_Choice()]

    class _Completions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls <= 2:
                raise RateLimitBoom()
            return _Response()

    completions = _Completions()
    client = type("_Client", (), {"chat": type("_Chat", (), {"completions": completions})()})()

    logs = []
    result = single_analyzer_module.step_1_analyze_image(
        str(image_path), client, "test-model", log_callback=logs.append
    )

    assert completions.calls == 3
    assert result["english_description"] == "a girl"
    assert sum("429 限流" in line for line in logs) == 2
