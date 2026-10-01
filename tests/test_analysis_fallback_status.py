"""第一选择「拒绝分析」时的工序状态：拒绝不能把整条任务判成失败。

对应用户 2026-09-30 反馈：跑图片分析时 `gpt-5.6-luna` 拒绝分析，改用备用端点
（`deepseek-v4.1`）再解析，结果**队列里这条任务被置位成失败**。

根因有两层，都在这里钉住：

1. **拒绝不是异常**。`finish_reason=content_filter` / `refusal` / content 为 None 都是
   HTTP 200 + 正常 body，`call_with_refusal_fallback` 原来只在 `call()` 抛异常时才会
   切备用端点 —— 于是备用端点永远不会被调用，任务在解析那一步炸成失败/被拒
   （`test_primary_refusal_response_really_switches_to_backup`）。
2. **拒绝判定必须覆盖"软拒绝"**：模型回一句 `抱歉，我无法分析这张图片`（JSON 合法、
   `finish_reason=stop`）同样要切成备用端点（`test_soft_refusal_text_is_detected`）。

备用端点接住之后，`WorkerThread.last_status` 必须是 `success` —— 队列才会标绿，
而不是把"备用方案已生效"写成失败（`test_worker_thread_reports_success_when_backup_analyzes`）。
"""

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import modules.image_analysis.single_analyzer as single_analyzer_module
from utils.analysis_fallback import FallbackConfig, call_with_refusal_fallback, is_refusal_text
from utils.llm_retry import RetrySettings


# --------------------------------------------------------------------------
# 假客户端 / 假响应
# --------------------------------------------------------------------------


class _Completions:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        self.owner.calls.append(kwargs)
        if self.owner.error:
            raise self.owner.error
        return self.owner.response


class _Chat:
    def __init__(self, owner):
        self.completions = _Completions(owner)


class _FakeClient:
    def __init__(self, base_url="https://primary.example/v1", error=None, response=None):
        self.base_url = base_url
        self.error = error
        self.response = response
        self.calls = []
        self.chat = _Chat(self)

    def count(self):
        return len(self.calls)


def _response(payload=None, content=None, finish_reason="stop", refusal=None):
    body = content if content is not None else json.dumps(payload or {}, ensure_ascii=False)
    message = type("Message", (), {"content": body, "refusal": refusal})()
    choice = type("Choice", (), {"message": message, "finish_reason": finish_reason})()
    return type("Response", (), {"choices": [choice], "model": "test-model", "id": "req-1"})()


def _refusal_response():
    """`gpt-5.6-luna` 拒绝分析的形态：HTTP 200，但 content 为 None + content_filter。"""
    return _response(content=None, finish_reason="content_filter", refusal="I cannot help with that.")


def _soft_refusal_response():
    """软拒绝：JSON 合法、finish_reason=stop，只是描述字段里写了拒绝话术。"""
    return _response({"english_description": "抱歉，我无法分析这张图片。", "japanese_title": "无法分析"})


ANALYSIS_PAYLOAD = {
    "english_description": "a girl standing in a white dress",
    "japanese_title": "少女",
    "chinese_title": "少女",
    "pixiv_tags": ["女の子"],
    "booru-tags": ["1girl"],
}


def _make_png(tmp_path):
    from PIL import Image

    path = tmp_path / "girl.png"
    Image.new("RGB", (8, 8), "white").save(path)
    return str(path)


def _backup_config():
    return FallbackConfig(
        enabled=True, base_url="https://api.deepseek.com", api_key="sk-backup", model="deepseek-v4.1")


def _inspect_response(response):
    """跟 `single_analyzer._step_label_refusal_text` 同口径的极简版（只看 refusal/content）。"""
    choice = response.choices[0]
    if choice.message.refusal:
        return choice.message.refusal
    return choice.message.content or ""


# --------------------------------------------------------------------------
# 1. 判定函数
# --------------------------------------------------------------------------


def test_soft_refusal_text_is_detected():
    assert is_refusal_text("抱歉，我无法分析这张图片。") is True
    assert is_refusal_text("I cannot help with that request.") is True
    assert is_refusal_text("N/A") is True
    assert is_refusal_text("") is False
    # 正常英文描述不能被当成拒绝
    assert is_refusal_text("a girl standing in a white dress, long black hair, holding a parasol") is False


# --------------------------------------------------------------------------
# 2. 拒绝响应（没有异常）也必须切成备用端点
# --------------------------------------------------------------------------


def test_primary_refusal_response_really_switches_to_backup(monkeypatch):
    """这一条就是本次 bug 的核心：拒绝走的是"正常返回"，必须照样切备用端点。"""
    primary = _FakeClient(response=_refusal_response())
    backup = _FakeClient(base_url="https://api.deepseek.com", response=_response(ANALYSIS_PAYLOAD))
    monkeypatch.setattr("utils.analysis_fallback._make_client", lambda *a, **k: backup)

    def refusal_text(response):
        choice = response.choices[0]
        if choice.message.refusal:
            return choice.message.refusal
        return choice.message.content or ""

    logs = []
    response = call_with_refusal_fallback(
        call=primary.chat.completions.create,
        request_kwargs={"model": "gpt-5.6-luna", "messages": []},
        fallback=_backup_config(),
        primary_client=primary,
        log_callback=logs.append,
        refusal_text=refusal_text,
    )

    assert response is backup.response
    assert primary.count() == 1 and backup.count() == 1
    assert backup.calls[0]["model"] == "deepseek-v4.1"
    assert any("第一选择被拒" in line for line in logs)
    assert any("备用端点返回成功" in line for line in logs)


def test_soft_refusal_response_also_switches(monkeypatch):
    """软拒绝（content 合法、finish_reason=stop）同样要切，而不是把拒绝当结果收下。"""
    primary = _FakeClient(response=_soft_refusal_response())
    backup = _FakeClient(base_url="https://api.deepseek.com", response=_response(ANALYSIS_PAYLOAD))
    monkeypatch.setattr("utils.analysis_fallback._make_client", lambda *a, **k: backup)

    def refusal_text(response):
        payload = json.loads(response.choices[0].message.content)
        return payload.get("english_description", "")

    response = call_with_refusal_fallback(
        call=primary.chat.completions.create,
        request_kwargs={"model": "gpt-5.6-luna", "messages": []},
        fallback=_backup_config(),
        primary_client=primary,
        refusal_text=refusal_text,
    )
    assert response is backup.response
    assert backup.count() == 1


def test_normal_primary_response_never_touches_backup(monkeypatch):
    """第一选择正常返回时不能产生任何额外请求（原契约不变）。"""
    primary = _FakeClient(response=_response(ANALYSIS_PAYLOAD))
    monkeypatch.setattr("utils.analysis_fallback._make_client", lambda *a, **k: pytest.fail("不应创建备用客户端"))

    def refusal_text(response):
        payload = json.loads(response.choices[0].message.content)
        return payload.get("english_description", "")

    response = call_with_refusal_fallback(
        call=primary.chat.completions.create,
        request_kwargs={"model": "gpt-5.6-luna", "messages": []},
        fallback=_backup_config(),
        primary_client=primary,
        refusal_text=refusal_text,
    )
    assert response is primary.response
    assert primary.count() == 1


def test_backup_refusal_is_not_treated_as_success(monkeypatch):
    """备用端点自己回拒绝话术时，不能把它当成功交上去。"""
    primary = _FakeClient(response=_refusal_response())
    backup = _FakeClient(base_url="https://api.deepseek.com", response=_soft_refusal_response())
    monkeypatch.setattr("utils.analysis_fallback._make_client", lambda *a, **k: backup)

    def refusal_text(response):
        choice = response.choices[0]
        if choice.message.refusal:
            return choice.message.refusal
        return choice.message.content or ""

    with pytest.raises(RuntimeError) as excinfo:
        call_with_refusal_fallback(
            call=primary.chat.completions.create,
            request_kwargs={"model": "gpt-5.6-luna", "messages": []},
            fallback=_backup_config(),
            primary_client=primary,
            refusal_text=refusal_text,
        )
    assert "备用端点" in str(excinfo.value)


# --------------------------------------------------------------------------
# 3. Step 1 / WorkerThread 端到端：备用端点接住 → 任务成功
# --------------------------------------------------------------------------


def _patch_offline(monkeypatch, tmp_path, primary, backup):
    """把 Step 1 的联网依赖全部换掉（不碰重试、不碰 WD14）。"""
    monkeypatch.setattr(single_analyzer_module, "OpenAI", lambda **kwargs: primary)
    monkeypatch.setattr(
        single_analyzer_module, "load_retry_settings",
        lambda *a, **k: RetrySettings(enabled=False, times=0, interval_seconds=0),
    )
    monkeypatch.setattr(single_analyzer_module, "load_fallback_config", lambda *a, **k: _backup_config())
    monkeypatch.setattr(single_analyzer_module, "predict_local_booru_tags", lambda *a, **k: [])
    monkeypatch.setattr(single_analyzer_module, "get_local_pixiv_tag_candidates", lambda *a, **k: [])
    monkeypatch.setattr("utils.analysis_fallback._make_client", lambda *a, **k: backup)


def test_step_1_returns_backup_result_when_primary_is_refused(monkeypatch, tmp_path):
    primary = _FakeClient(response=_refusal_response())
    backup = _FakeClient(base_url="https://api.deepseek.com", response=_response(ANALYSIS_PAYLOAD))
    _patch_offline(monkeypatch, tmp_path, primary, backup)

    statuses = []
    result = single_analyzer_module.step_1_analyze_image(
        _make_png(tmp_path), primary, "gpt-5.6-luna",
        status_callback=statuses.append, timeout_seconds=30, use_fallback=True,
    )

    assert result["english_description"] == "a girl standing in a white dress"
    assert primary.count() == 1 and backup.count() == 1
    # 备用端点接住了 → 不该再给上层报「被拒」
    assert statuses == []


def test_step_1_reports_refused_only_when_backup_also_fails(monkeypatch, tmp_path):
    primary = _FakeClient(response=_refusal_response())
    backup = _FakeClient(base_url="https://api.deepseek.com", error=RuntimeError("401 Invalid token"))
    _patch_offline(monkeypatch, tmp_path, primary, backup)

    statuses = []
    result = single_analyzer_module.step_1_analyze_image(
        _make_png(tmp_path), primary, "gpt-5.6-luna",
        status_callback=statuses.append, timeout_seconds=30, use_fallback=True,
    )

    assert result is None
    assert statuses == ["refused"]      # 两边都被拒/失败 → 队列才允许标红


def test_worker_thread_reports_success_when_backup_analyzes(monkeypatch, tmp_path):
    """用户看到的那条：第一选择被拒、备用端点解析成功 → 线程状态必须是 success。"""
    primary = _FakeClient(response=_refusal_response())
    backup = _FakeClient(base_url="https://api.deepseek.com", response=_response(ANALYSIS_PAYLOAD))
    _patch_offline(monkeypatch, tmp_path, primary, backup)

    logs = []
    results = []
    thread = single_analyzer_module.WorkerThread(
        _make_png(tmp_path), "sk-primary", "https://primary.example/v1", "gpt-5.6-luna",
        enable_refine=False, timeout_seconds=30, use_fallback=True,
    )
    thread.log_signal.connect(logs.append)
    thread.finish_signal.connect(results.append)
    thread.run()          # 同步跑，不起事件循环

    assert thread.last_status == "success"
    # 本次没开 Step 2 refine，内容锚放在 original_english_description（english_description 留给精修）
    assert results and results[0].get("original_english_description") == "a girl standing in a white dress"
    assert backup.count() == 1
    assert any("备用端点返回成功" in line for line in logs)
    assert not any("Step 1 失败，流程终止" in line for line in logs)


def test_worker_thread_reports_refused_when_nothing_can_analyze(monkeypatch, tmp_path):
    """两边都救不回来时才允许标红，且状态是「被拒」而不是笼统的「失败」。"""
    primary = _FakeClient(response=_refusal_response())
    backup = _FakeClient(base_url="https://api.deepseek.com", error=RuntimeError("401 Invalid token"))
    _patch_offline(monkeypatch, tmp_path, primary, backup)

    results = []
    thread = single_analyzer_module.WorkerThread(
        _make_png(tmp_path), "sk-primary", "https://primary.example/v1", "gpt-5.6-luna",
        enable_refine=False, timeout_seconds=30, use_fallback=True,
    )
    thread.finish_signal.connect(results.append)
    thread.run()

    assert thread.last_status == "refused"
    assert results == [{}]
