"""备用分析端点：第一选择拒绝分析图片时，把同一份请求改发给第二个端点。

对应用户 2026-09-30 反馈：图片分析模块（gpt-5.6-luna 之类）偶尔会拒绝分析图片
（`finish_reason=content_filter` / `refusal` / content 为 None）。这类失败重试无用
（同一张图必然被同样拒绝），只能换模型 —— 于是有了「备用方案」。

覆盖点：
1. `should_use_fallback` 的触发判定（被拒才切 / any 模式全切 / 逻辑错误不切）；
2. 配置解析优先级（`fallback_*` > 复用 NSFW 通道 > 环境变量 > 没有）；
3. `call_with_refusal_fallback` 的三个分支：成功透传、被拒切备用、备用也失败时两个原因都报；
4. 「备用端点就是第一选择本身」时不重发（分析 Tab 勾 nsfw + 备用复用 NSFW 的场景）；
5. `step_1_analyze_image` 端到端：第一选择被拒 → 备用端点拿到**同一份请求体**并返回结果。
"""

import json
import os

import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from utils.analysis_fallback import (
    FallbackConfig,
    FALLBACK_API_KEY_ENV_NAMES,
    TRIGGER_ANY,
    TRIGGER_REFUSAL,
    build_fallback_kwargs,
    call_with_refusal_fallback,
    is_refusal_error,
    load_fallback_config,
    mask_secret,
    resolve_fallback_api_key,
    resolve_fallback_trigger,
    should_use_fallback,
)


class RefusalError(ValueError):
    """模拟 `_safe_json_from_response` 在 refusal / content_filter 时抛的错。"""

    def __init__(self, message="Step 1 响应 content 为 None (refusal: 'I cannot help with that.', finish_reason: content_filter)"):
        super().__init__(message)


class TransportBoom(ConnectionError):
    """模拟网络层失败（非拒绝类）。"""


@pytest.fixture(scope="session")
def qapp():
    """界面用例需要 QApplication（offscreen），本模块自带一份，不依赖别的测试模块。"""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# --------------------------------------------------------------------------
# 1. 触发判定
# --------------------------------------------------------------------------


def test_refusal_markers_trigger_only_in_refusal_mode():
    assert should_use_fallback(RefusalError(), TRIGGER_REFUSAL) is True
    assert should_use_fallback(ValueError("响应 content 为 None，可能是 API 安全过滤或模型拒绝响应")) is True
    assert should_use_fallback(ValueError("Sorry, I can't help with that image.")) is True
    assert should_use_fallback(ValueError("该请求触发内容审核，已拦截")) is True
    # 非拒绝类失败：默认不切（重试/报错交给上层）
    assert should_use_fallback(TransportBoom("connection reset by peer"), TRIGGER_REFUSAL) is False


def test_any_trigger_switches_on_every_failure():
    assert should_use_fallback(TransportBoom("connection reset"), TRIGGER_ANY) is True
    assert should_use_fallback(RuntimeError("whatever"), TRIGGER_ANY) is True


def test_is_refusal_error_is_the_conservative_predicate():
    assert is_refusal_error(RefusalError()) is True
    assert is_refusal_error(TransportBoom("timeout")) is False


def test_mask_secret_never_leaks_full_key():
    assert mask_secret("") == "(空)"
    assert mask_secret("sk-abcd") == "*******"
    assert mask_secret("sk-1234567890abcdef") == "sk-1...cdef"


# --------------------------------------------------------------------------
# 2. 配置解析
# --------------------------------------------------------------------------


def _write_config(tmp_path, payload):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(path)


def test_explicit_fallback_keys_win_over_nsfw_reuse(tmp_path):
    path = _write_config(
        tmp_path,
        {
            "fallback_enabled": True,
            "fallback_base_url": "https://api.deepseek.com",
            "fallback_api_key": "sk-fallback",
            "fallback_model": "deepseek-flash",
            "fallback_trigger": TRIGGER_ANY,
            "nsfw_base_url": "https://nsfw.example/v1",
            "nsfw_api_key": "sk-nsfw",
            "nsfw_model": "nsfw-model",
        },
    )
    cfg = load_fallback_config(path)
    assert cfg.enabled is True
    assert (cfg.base_url, cfg.api_key, cfg.model) == ("https://api.deepseek.com", "sk-fallback", "deepseek-flash")
    assert cfg.trigger == TRIGGER_ANY
    assert cfg.key_source == "config"


def test_nsfw_channel_is_reused_when_fallback_fields_are_empty(tmp_path):
    """界面没填备用方案时，复用「文本分析（NSFW）」通道（本机指向 deepseek，省得存两份 key）。

    2026-09-30 实测该端点：`deepseek-v4-pro` 明确回「image is unsupported」，
    `deepseek-flash` 能正确读图并给出结构化结果 —— 所以复用时会把已知不看图的模型换掉。
    """
    path = _write_config(
        tmp_path,
        {
            "nsfw_base_url": "https://api.deepseek.com",
            "nsfw_api_key": "sk-nsfw",
            "nsfw_model": "deepseek-v4-pro",
        },
    )
    cfg = load_fallback_config(path)
    assert cfg.enabled is True
    assert cfg.model == "deepseek-flash"          # 自动换成能读图的
    assert cfg.base_url == "https://api.deepseek.com"
    assert cfg.extra["model_swapped_from"] == "deepseek-v4-pro"
    assert "NSFW" in cfg.source_label


def test_nsfw_reuse_keeps_models_that_can_read_images(tmp_path):
    path = _write_config(
        tmp_path,
        {
            "nsfw_base_url": "https://api.deepseek.com",
            "nsfw_api_key": "sk-nsfw",
            "nsfw_model": "deepseek-flash",
        },
    )
    cfg = load_fallback_config(path)
    assert cfg.model == "deepseek-flash"
    assert cfg.extra["model_swapped_from"] == ""


def test_env_key_overrides_config_and_missing_config_is_disabled(tmp_path, monkeypatch):
    """key 只在 .env 里也能用：端点填了就算配好，key 从环境变量解析（界面不该因为为空而误判成没配）。"""
    path = _write_config(tmp_path, {"fallback_base_url": "https://api.deepseek.com", "fallback_model": "deepseek-flash"})
    cfg_without_key = load_fallback_config(path)
    assert cfg_without_key.base_url == "https://api.deepseek.com"
    assert cfg_without_key.missing() == ["api_key"]     # 差 key → 分析时不会切换，但界面能提示该配哪个变量

    monkeypatch.setenv(FALLBACK_API_KEY_ENV_NAMES[0], "sk-from-env")
    cfg = load_fallback_config(path)
    assert cfg.enabled is True
    assert cfg.api_key == "sk-from-env"
    assert cfg.missing() == []
    assert cfg.key_source == f"env:{FALLBACK_API_KEY_ENV_NAMES[0]}"
    assert resolve_fallback_api_key({}) == "sk-from-env"
    # 日志/界面里只允许出现变量名，不允许出现密钥值
    assert "sk-from-env" not in cfg.describe()
    assert FALLBACK_API_KEY_ENV_NAMES[0] in cfg.describe()
    monkeypatch.delenv(FALLBACK_API_KEY_ENV_NAMES[0], raising=False)


def test_nothing_configured_means_disabled(tmp_path, monkeypatch):
    for name in FALLBACK_API_KEY_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name in ("IMAGE_MAKER_NSFW_API_KEY", "NSFW_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    path = _write_config(tmp_path, {"base_url": "https://new.aigc2d.com/v1", "api_key": "", "model": "gpt-5.6-luna"})
    cfg = load_fallback_config(path)
    assert cfg.enabled is False and cfg.missing() == ["base_url", "api_key", "model"]


def test_fallback_key_falls_back_to_nsfw_key(tmp_path, monkeypatch):
    """专用 key 变量没配时借用 NSFW 那把（本机 NSFW 就指向 deepseek，不用重复存 key）。"""
    for name in FALLBACK_API_KEY_ENV_NAMES:
        monkeypatch.setenv(name, "")
    path = _write_config(
        tmp_path,
        {
            "fallback_base_url": "https://api.deepseek.com",
            "fallback_model": "deepseek-flash",
            "nsfw_api_key": "sk-nsfw",
        },
    )
    cfg = load_fallback_config(path)
    assert cfg.api_key == "sk-nsfw"
    assert cfg.key_source == "nsfw"
    assert cfg.missing() == []


def test_trigger_env_var_and_default(tmp_path, monkeypatch):
    monkeypatch.delenv("IMAGE_MAKER_FALLBACK_TEXT_TRIGGER", raising=False)
    assert resolve_fallback_trigger({}) == TRIGGER_REFUSAL
    assert resolve_fallback_trigger({"fallback_trigger": "bogus"}) == TRIGGER_REFUSAL
    monkeypatch.setenv("IMAGE_MAKER_FALLBACK_TEXT_TRIGGER", "any")
    assert resolve_fallback_trigger({"fallback_trigger": "refusal"}) == TRIGGER_ANY


# --------------------------------------------------------------------------
# 3. 调用分支
# --------------------------------------------------------------------------


class _FakeClient:
    """够用的 OpenAI 假客户端替身（不触发网络，可断言收到的 kwargs）。"""

    def __init__(self, base_url="https://primary.example/v1", response=None, error=None, calls=None):
        self.base_url = base_url
        self._response = response
        self._error = error
        self.calls = calls if calls is not None else []
        self.chat = type("_Chat", (), {"completions": self})()

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response

    def count(self):
        return len(self.calls)


def _ok_response(payload):
    message = type("Message", (), {"content": json.dumps(payload), "refusal": None})()
    choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
    return type("Response", (), {"choices": [choice], "model": "test-model", "id": "req-1"})()


def test_success_path_never_touches_fallback():
    primary = _FakeClient(response=_ok_response({"ok": True}))
    response = call_with_refusal_fallback(
        call=primary.chat.completions.create,
        request_kwargs={"model": "m", "messages": []},
        fallback=FallbackConfig(enabled=True, base_url="https://backup.example/v1", api_key="k", model="backup-model"),
    )
    assert response.choices[0].message.content
    assert primary.count() == 1


def test_refusal_switches_to_fallback_with_same_body(monkeypatch, tmp_path):
    """被拒 → 备用端点收到同一份请求体，只换了 model（图/提示词一字不改）。"""
    primary = _FakeClient(error=RefusalError())
    backup = _FakeClient(base_url="https://backup.example/v1", response=_ok_response({"english_description": "a girl"}))
    monkeypatch.setattr("utils.analysis_fallback._make_client", lambda *a, **k: backup)

    logs = []
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
    response = call_with_refusal_fallback(
        call=primary.chat.completions.create,
        request_kwargs={"model": "primary-model", "messages": messages, "temperature": 0.7},
        fallback=FallbackConfig(enabled=True, base_url="https://backup.example/v1", api_key="k", model="backup-model"),
        timeout_seconds=30,
        log_callback=logs.append,
        step_label="Step 1 Vision 请求",
    )

    assert response is backup._response
    assert backup.count() == 1
    sent = backup.calls[0]
    assert sent["model"] == "backup-model"          # 只换模型/端点
    assert sent["messages"] == messages             # 同一张图、同一段提示词
    assert sent["temperature"] == 0.7
    assert any("备用端点返回成功" in line for line in logs)


def test_fallback_endpoint_failure_reports_both_reasons(monkeypatch):
    primary = _FakeClient(error=RefusalError("Step 1 响应 content 为 None (finish_reason: content_filter)"))
    backup = _FakeClient(base_url="https://backup.example/v1", error=RuntimeError("401 Invalid token"))
    monkeypatch.setattr("utils.analysis_fallback._make_client", lambda *a, **k: backup)

    logs = []
    with pytest.raises(RuntimeError) as excinfo:
        call_with_refusal_fallback(
            call=primary.chat.completions.create,
            request_kwargs={"model": "primary-model"},
            fallback=FallbackConfig(enabled=True, base_url="https://backup.example/v1", api_key="k", model="backup-model"),
            log_callback=logs.append,
        )
    text = str(excinfo.value)
    assert "第一选择失败" in text and "content_filter" in text
    assert "备用端点" in text and "401 Invalid token" in text
    assert any("备用端点也失败" in line for line in logs)


def test_same_endpoint_is_not_retried(monkeypatch):
    """备用端点 == 第一选择（例如分析 Tab 勾了 nsfw，而备用方案正是复用 NSFW 通道）→ 不重发。"""
    primary = _FakeClient(base_url="https://api.deepseek.com", error=RefusalError())
    created = []
    monkeypatch.setattr("utils.analysis_fallback._make_client", lambda *a, **k: created.append(k) or primary)

    logs = []
    with pytest.raises(RefusalError):
        call_with_refusal_fallback(
            call=primary.chat.completions.create,
            request_kwargs={"model": "deepseek-v4-pro"},
            fallback=FallbackConfig(
                enabled=True, base_url="https://api.deepseek.com", api_key="k", model="deepseek-v4-pro",
            ),
            primary_client=primary,
            primary_base_url="https://api.deepseek.com",
            log_callback=logs.append,
        )
    assert primary.count() == 1        # 只发了第一选择那一次
    assert created == []               # 没有新建备用客户端
    assert any("同一个" in line for line in logs)


def test_non_refusal_error_is_not_switched_by_default(monkeypatch):
    primary = _FakeClient(error=TransportBoom("connection reset by peer"))
    monkeypatch.setattr("utils.analysis_fallback._make_client", lambda *a, **k: pytest.fail("不应创建备用客户端"))
    with pytest.raises(TransportBoom):
        call_with_refusal_fallback(
            call=primary.chat.completions.create,
            request_kwargs={"model": "primary-model"},
            fallback=FallbackConfig(enabled=True, base_url="https://backup.example/v1", api_key="k", model="backup-model"),
        )


def test_use_fallback_false_disables_switch(monkeypatch):
    primary = _FakeClient(error=RefusalError())
    monkeypatch.setattr("utils.analysis_fallback._make_client", lambda *a, **k: pytest.fail("不应创建备用客户端"))
    with pytest.raises(RefusalError):
        call_with_refusal_fallback(
            call=primary.chat.completions.create,
            request_kwargs={"model": "primary-model"},
            fallback=FallbackConfig(enabled=True, base_url="https://backup.example/v1", api_key="k", model="backup-model"),
            use_fallback=False,
        )


def test_unsupported_response_format_is_retried_without_it(monkeypatch):
    """备用端点不认 `response_format` → 去掉该参数再试一次（而不是直接失败）。"""

    class _Picky(_FakeClient):
        def create(self, **kwargs):
            self.calls.append(kwargs)
            if "response_format" in kwargs:
                raise RuntimeError("Unsupported parameter: response_format")
            return self._response

    backup = _Picky(base_url="https://backup.example/v1", response=_ok_response({"ok": 1}))
    primary = _FakeClient(error=RefusalError())
    monkeypatch.setattr("utils.analysis_fallback._make_client", lambda *a, **k: backup)

    response = call_with_refusal_fallback(
        call=primary.chat.completions.create,
        request_kwargs={"model": "primary-model", "response_format": {"type": "json_object"}},
        fallback=FallbackConfig(enabled=True, base_url="https://backup.example/v1", api_key="k", model="backup-model"),
    )
    assert response is backup._response
    assert backup.count() == 2
    assert "response_format" in backup.calls[0] and "response_format" not in backup.calls[1]


def test_build_fallback_kwargs_does_not_mutate_caller():
    original = {"model": "primary-model", "response_format": {"type": "json_object"}, "messages": []}
    payload = build_fallback_kwargs(
        FallbackConfig(enabled=True, base_url="https://b/v1", api_key="k", model="backup-model", response_format=False),
        original,
    )
    assert payload["model"] == "backup-model"
    assert "response_format" not in payload
    assert original["model"] == "primary-model" and "response_format" in original


# --------------------------------------------------------------------------
# 4. Step 1 端到端
# --------------------------------------------------------------------------


def _make_png(tmp_path):
    from PIL import Image

    path = tmp_path / "girl.png"
    Image.new("RGB", (8, 8), "white").save(path)
    return str(path)


def test_step_1_switches_to_backup_when_primary_refuses(monkeypatch, tmp_path):
    import modules.image_analysis.single_analyzer as single_analyzer_module
    from utils.llm_retry import RetrySettings

    monkeypatch.setattr(
        single_analyzer_module, "load_retry_settings",
        lambda *a, **k: RetrySettings(enabled=False, times=0, interval_seconds=0),
    )
    monkeypatch.setattr(
        single_analyzer_module, "load_fallback_config",
        lambda *a, **k: FallbackConfig(
            enabled=True, base_url="https://backup.example/v1", api_key="sk-backup", model="deepseek-flash",
        ),
    )

    payload = {
        "english_description": "a girl",
        "japanese_title": "少女",
        "chinese_title": "少女",
        "pixiv_tags": ["女の子"],
        "booru-tags": ["1girl"],
    }
    primary = _FakeClient(base_url="https://primary.example/v1", error=RefusalError())
    backup = _FakeClient(base_url="https://backup.example/v1", response=_ok_response(payload))
    monkeypatch.setattr("utils.analysis_fallback._make_client", lambda *a, **k: backup)

    logs = []
    statuses = []
    result = single_analyzer_module.step_1_analyze_image(
        _make_png(tmp_path),
        primary,
        "gpt-5.6-luna",
        log_callback=logs.append,
        status_callback=statuses.append,
        use_fallback=True,
    )

    assert result["english_description"] == "a girl"
    assert primary.count() == 1
    assert backup.count() == 1
    assert backup.calls[0]["model"] == "deepseek-flash"
    # 图片以 data URI 原样重发（同一张图；编码格式取决于压缩结果，这里只校验它是同一张内联图）
    image_part = backup.calls[0]["messages"][1]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/")
    assert ";base64," in image_part["image_url"]["url"]
    assert any("备用端点" in line for line in logs)


def test_step_1_reports_refused_status_when_fallback_unavailable(monkeypatch, tmp_path):
    import modules.image_analysis.single_analyzer as single_analyzer_module
    from utils.llm_retry import RetrySettings

    monkeypatch.setattr(
        single_analyzer_module, "load_retry_settings",
        lambda *a, **k: RetrySettings(enabled=False, times=0, interval_seconds=0),
    )
    monkeypatch.setattr(single_analyzer_module, "load_fallback_config", lambda *a, **k: FallbackConfig())

    primary = _FakeClient(error=RefusalError())
    statuses = []
    result = single_analyzer_module.step_1_analyze_image(
        _make_png(tmp_path), primary, "gpt-5.6-luna", status_callback=statuses.append,
    )
    assert result is None
    # 「被拒」与「对面坏了」要分开标，日志/队列里能一眼区分
    assert statuses == ["refused"]


def test_analyzer_widget_fallback_checkbox_defaults_on_and_is_exposed(qapp):
    """分析 Tab 的勾选框只借一行里的一个控件（不新增常驻行，布局契约见 AGENTS.md）。"""
    import modules.image_analysis.single_analyzer as single_analyzer_module
    widget = single_analyzer_module.SingleAnalyzerWidget(
        config_getter_func=lambda *a, **k: ("https://api.example/v1", "sk-x", "model-x"),
        img_config_getter_func=lambda: ("", "sk-y", "img-model", "aigc2d"),
        styles_getter_func=lambda: {},
        save_img_cfg_callback=lambda: None,
        persist_ui_state=False,
    )
    assert widget.use_fallback_cb.isChecked() is True
    widget.set_use_fallback_default(False)
    assert widget.use_fallback_cb.isChecked() is False

    captured = []
    widget.on_fallback_changed = captured.append
    widget.use_fallback_cb.setChecked(True)
    assert captured == [True]
    assert widget.use_fallback_cb.parent() is not None
    widget.close()
