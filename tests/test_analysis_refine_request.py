import json
import os
from types import SimpleNamespace

import pytest

from modules.image_analysis import single_analyzer as module
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

STEP1_RESULT = {"english_description": "A person in a blue dress seated by a window.",
                "japanese_title": "窓辺", "chinese_title": "窗边",
                "pixiv_tags": ["ドレス"], "booru-tags": ["blue_dress"]}


def _FakeResponse(content, finish_reason="stop", refusal=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=content, refusal=refusal))])


@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def diagnostic_root(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "BASE_DIR", str(tmp_path))
    return tmp_path / "cache" / "temp" / "analysis-refine"


def client_for(call):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=call)),
                           api_key="secret-not-to-be-saved")


def test_editor_request_preserves_source_and_uses_own_system(diagnostic_root):
    requests = []
    source = dict(STEP1_RESULT, aspect_ratio="16:9")

    def call(**kwargs):
        requests.append(kwargs)
        return _FakeResponse(json.dumps(source))

    result = module.step_2_refine_description(source, client_for(call), "test-model")
    assert result["original_english_description"] == source["english_description"]
    assert len(requests) == 1
    messages = requests[0]["messages"]
    assert "text editor" in messages[0]["content"]
    assert "image analyzer" not in messages[0]["content"]
    assert source["english_description"] in messages[1]["content"]
    assert '16:9' in messages[1]["content"]
    assert "保留遮挡" in messages[1]["content"]
    assert "必须将其完全删除" not in messages[1]["content"]
    saved = json.loads(next(diagnostic_root.glob("*.json")).read_text(encoding="utf-8"))
    assert saved["request"] == requests[0]
    assert saved["step1_result"] == source
    assert "secret-not-to-be-saved" not in json.dumps(saved)
    assert len(saved["prompt_sha256"]) == 64


@pytest.mark.parametrize("failure", ["http400", "filtered", "soft", "refusal", "refusal_with_json"])
def test_refusal_stops_once_and_retains_evidence(diagnostic_root, failure):
    calls, statuses, logs = [], [], []

    def call(**kwargs):
        calls.append(kwargs)
        if failure == "http400":
            raise RuntimeError("Error code: 400 content_filter request id: test-123")
        if failure == "filtered":
            return _FakeResponse(None, finish_reason="content_filter")
        if failure == "refusal":
            return _FakeResponse(None, refusal="blocked")
        if failure == "refusal_with_json":
            return _FakeResponse(json.dumps(STEP1_RESULT), refusal="blocked")
        return _FakeResponse(json.dumps({"english_description": "I can't help with that"}))

    assert module.step_2_refine_description(
        STEP1_RESULT, client_for(call), "test-model",
        status_callback=statuses.append, log_callback=logs.append) is None
    assert statuses == ["refused"]
    assert len(calls) == 1
    path = next(diagnostic_root.glob("*.json"))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["error"]
    assert saved["step1_result"] == STEP1_RESULT
    assert "[error]" in path.with_suffix(".txt").read_text(encoding="utf-8")
    assert any("请求快照" in line for line in logs)


def test_empty_editor_result_is_not_reported_as_success(diagnostic_root):
    statuses = []
    result = module.step_2_refine_description(
        STEP1_RESULT, client_for(lambda **kwargs: _FakeResponse("{}")), "test-model",
        status_callback=statuses.append)
    assert result is None
    assert statuses == ["error"]


def test_worker_keeps_step2_refused_state(monkeypatch, qapp):
    monkeypatch.setattr(module, "OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(module, "predict_local_booru_tags", lambda *a, **k: [])
    monkeypatch.setattr(module, "get_local_pixiv_tag_candidates", lambda *a, **k: [])
    monkeypatch.setattr(module, "step_1_analyze_image", lambda *a, **k: dict(STEP1_RESULT))

    def refine(*args, **kwargs):
        kwargs["status_callback"]("refused")
        return None

    monkeypatch.setattr(module, "step_2_refine_description", refine)
    thread = module.WorkerThread("dummy.png", "key", "https://example.invalid/v1", "test-model")
    results = []
    thread.finish_signal.connect(results.append)
    thread.run()
    assert thread.last_status == "refused"
    assert results == [{}]
