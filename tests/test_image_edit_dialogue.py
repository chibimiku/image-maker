import copy
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from modules.image_generation import image_edit_dialogue as rt


@pytest.fixture
def bundle(tmp_path):
    paths = []
    for index, colour in enumerate(("red", "blue")):
        path = tmp_path / f"source-{index}.png"
        Image.new("RGB", (64, 96), colour).save(path)
        paths.append(str(path))
    session = rt.new_session()
    ids = rt.register_images(session, paths)
    config = {"model": "gemini-configured", "api_key": "secret", "resolution": "2K", "base_url": "https://example.invalid"}
    return session, ids, config, tmp_path / "session.json"


def route(ids):
    return {"action": "edit", "message": "逐张调整背景", "tasks": [
        {"image_id": image_id, "version": "latest", "instruction": f"Change background of #{image_id} to white"}
        for image_id in ids]}


def plan(**overrides):
    return {"decision": "edit", "summary": "只调整背景", "prompt": "Change only the background to white. Preserve the subject.",
            "sexualized": False, "age_assessment": "not_applicable", **overrides}


def install_api(monkeypatch, tmp_path, ids, fail_first=False, cancel=None):
    calls, stages = [], []
    def analyse(stage, context, attachments, config):
        stages.append((stage, copy.deepcopy(context), attachments))
        return (route(ids) if stage == "route" else
                {"verdict": "pass", "summary": "背景已变白，其余保持", "issues": []} if stage == "review" else plan())
    def generate(**kwargs):
        calls.append(kwargs)
        if fail_first and len(calls) == 1:
            return {"saved_files": []}
        path = tmp_path / f"output-{len(calls)}.png"
        Image.new("RGB", (64, 96), "white").save(path)
        if cancel:
            cancel[0] = True
        return {"saved_files": [str(path)], "server_response_raw": {"usageMetadata": {"totalTokenCount": 10}}}
    monkeypatch.setattr(rt, "request_json", analyse)
    monkeypatch.setattr(rt, "generate_image_aigc2d", generate)
    return calls, stages


def test_serial_image_specific_generation_and_durable_lineage(bundle, monkeypatch, tmp_path):
    session, ids, config, path = bundle
    calls, stages = install_api(monkeypatch, tmp_path, ids)
    snapshots = []
    runner = rt.DialogueRunner(session, path, config, config, snapshots.append)
    runner.run("背景变白", ids)
    result = rt.load_session(path)
    assert len(calls) == 2
    assert [stage for stage, *_ in stages] == ["route", "plan", "review", "plan", "review"]
    for index, call in enumerate(calls):
        assert call["image_paths"] == [session["images"][index]["original"]]
        assert call["retry_limit"] == 0 and call["face_quality_boost"] is False
        assert call["aspect_ratio"] == "auto"
        assert call["config_snapshot"] == config
    assert all(len(row["versions"]) == 2 for row in result["images"])
    assert result["images"][0]["versions"][1]["parent_version"] == 0
    assert result["turns"][-1]["tasks"][0]["status"] == "success"
    assert "secret" not in path.read_text(encoding="utf-8")
    assert any(e["session"]["turns"][-1]["tasks"][0]["status"] == "generating"
               for e in snapshots if e["session"]["turns"][-1]["tasks"])


@pytest.mark.parametrize("change", [
    lambda r: r["tasks"][0].update(image_id=999),
    lambda r: r["tasks"].append(r["tasks"][0]),
    lambda r: r["tasks"][0].update(version=9),
    lambda r: r["tasks"][0].update(image_id=True),
    lambda r: r["tasks"][0].update(instruction=""),
    lambda r: r.update(action="reply"),
])
def test_invalid_route_rejected_before_generation(bundle, change):
    session, ids, *_ = bundle
    value = route(ids)
    change(value)
    with pytest.raises(ValueError):
        rt.validate_route(value, session, ids)


@pytest.mark.parametrize("age", ["minor", "uncertain", "not_applicable"])
def test_sexualized_edits_require_clear_adult(age):
    assert rt.validate_plan(plan(sexualized=True, age_assessment=age))["decision"] == "skip"


def test_malformed_safety_fields_rejected():
    with pytest.raises(ValueError):
        rt.validate_plan(plan(sexualized="false"))
    with pytest.raises(ValueError):
        rt.validate_plan(plan(prompt=""))
    with pytest.raises(ValueError, match="拒绝"):
        rt.validate_plan(plan(prompt="I cannot help with that request."))


def test_reply_and_clarification_do_not_generate(bundle, monkeypatch):
    session, ids, config, path = bundle
    monkeypatch.setattr(rt, "request_json", lambda *a: {"action": "clarify", "message": "你希望调整哪张？", "tasks": []})
    monkeypatch.setattr(rt, "generate_image_aigc2d", lambda **kw: pytest.fail("must not generate"))
    rt.DialogueRunner(session, path, config, config).run("你觉得哪张好", ids)
    assert rt.load_session(path)["turns"][-1]["status"] == "clarify"


def test_stop_retains_inflight_paid_result_and_skips_next(bundle, monkeypatch, tmp_path):
    session, ids, config, path = bundle
    cancel = [False]
    calls, _ = install_api(monkeypatch, tmp_path, ids, cancel=cancel)
    runner = rt.DialogueRunner(session, path, config, config, cancelled=lambda: cancel[0])
    runner.run("背景变白", ids)
    assert len(calls) == 1
    tasks = rt.load_session(path)["turns"][-1]["tasks"]
    assert [t["status"] for t in tasks] == ["success", "cancelled"]


def test_retry_only_failures_uses_original_input_and_plan(bundle, monkeypatch, tmp_path):
    session, ids, config, path = bundle
    calls, stages = install_api(monkeypatch, tmp_path, ids, fail_first=True)
    rt.DialogueRunner(session, path, config, config).run("背景变白", ids)
    session = rt.load_session(path)
    first = session["turns"][-1]
    assert [t["status"] for t in first["tasks"]] == ["error", "success"]
    rt.DialogueRunner(session, path, config, config).run("背景变白", [ids[0]], retry_turn_id=first["id"])
    assert len(calls) == 3
    assert calls[-1]["image_paths"] == calls[0]["image_paths"]
    assert [stage for stage, *_ in stages] == ["route", "plan", "plan", "review", "review"]
    session = rt.load_session(path)
    assert len(session["images"][1]["versions"]) == 2
    assert session["turns"][0]["tasks"][0]["status"] == "error"


def test_followup_uses_latest_and_explicit_original(bundle, monkeypatch, tmp_path):
    session, ids, config, path = bundle
    calls, stages = install_api(monkeypatch, tmp_path, [ids[0]])
    rt.DialogueRunner(session, path, config, config).run("背景变白", ids)
    session = rt.load_session(path)
    rt.DialogueRunner(session, path, config, config).run("再改淡蓝", ids)
    assert calls[-1]["image_paths"] == [str(tmp_path / "output-1.png")]
    assert [context for stage, context, _ in stages if stage == "plan"][-1]["history"][-1]["user"] == "背景变白"
    session = rt.load_session(path)
    rt.DialogueRunner(session, path, config, config).run("从原图改淡蓝", ids, preferred_versions={ids[0]: 0})
    assert calls[-1]["image_paths"] == [session["images"][0]["original"]]


def test_mutated_input_blocks_dispatch(bundle, monkeypatch, tmp_path):
    session, ids, config, path = bundle
    calls, _ = install_api(monkeypatch, tmp_path, ids)
    Image.new("RGB", (64, 96), "black").save(session["images"][0]["original"])
    rt.DialogueRunner(session, path, config, config).run("背景变白", ids)
    assert not calls
    assert "被修改" in rt.load_session(path)["turns"][-1]["message"]


def test_recovery_never_resumes_requests(bundle):
    session, ids, config, path = bundle
    session["turns"] = [{"id": "old", "user": "hello", "status": "running", "tasks": [
        {"status": "generating", "image_id": ids[0], "version": 0, "instruction": "edit",
         "input": session["images"][0]["versions"][0], "attempts": []}]}]
    rt.save_session(session, path)
    restored = rt.load_session(path)
    assert restored["turns"][0]["status"] == "interrupted"
    assert restored["turns"][0]["tasks"][0]["status"] == "interrupted"


def test_failed_image_cannot_fall_back_to_source(bundle, monkeypatch, tmp_path):
    session, ids, config, path = bundle
    install_api(monkeypatch, tmp_path, [ids[0]])
    monkeypatch.setattr(rt, "generate_image_aigc2d", lambda **kw: {"saved_files": kw["image_paths"]})
    rt.DialogueRunner(session, path, config, config).run("背景变白", ids)
    restored = rt.load_session(path)
    assert restored["turns"][-1]["tasks"][0]["status"] == "error"
    assert len(restored["images"][0]["versions"]) == 1


def test_plan_skip_does_not_generate(bundle, monkeypatch):
    session, ids, config, path = bundle
    monkeypatch.setattr(rt, "request_json", lambda stage, *a: route(ids) if stage == "route" else plan(sexualized=True, age_assessment="uncertain"))
    monkeypatch.setattr(rt, "generate_image_aigc2d", lambda **kw: pytest.fail("must skip"))
    rt.DialogueRunner(session, path, config, config).run("性感些", ids)
    assert all(t["status"] == "skip" for t in rt.load_session(path)["turns"][-1]["tasks"])


@pytest.mark.parametrize("bad_review", ["fail", "timeout", "malformed"])
def test_output_review_never_retries_paid_generation(bundle, monkeypatch, tmp_path, bad_review):
    session, ids, config, path = bundle
    calls, _ = install_api(monkeypatch, tmp_path, [ids[0]])
    original = rt.request_json
    def analyse(stage, *args):
        if stage == "review":
            if bad_review == "timeout":
                raise TimeoutError("review timeout")
            if bad_review == "malformed":
                return {}
            return {"verdict": "fail", "summary": "右侧边框新增断裂", "issues": ["右侧边框断裂"]}
        return original(stage, *args)
    monkeypatch.setattr(rt, "request_json", analyse)
    rt.DialogueRunner(session, path, config, config).run("改背景", ids)
    assert len(calls) == 1
    restored = rt.load_session(path)
    task = restored["turns"][-1]["tasks"][0]
    assert task["status"] == "success"  # image received, independent from review verdict
    assert task["reviews"][0]["verdict"] in {"fail", "uncertain"}
    assert len(restored["images"][0]["versions"]) == 2
    assert "1 项复核异常" in restored["turns"][-1]["message"]


def test_backend_snapshot_freezes_node_and_disables_hidden_boost(monkeypatch, tmp_path):
    import requests
    from modules.others import api_backend as backend
    calls = []
    response = requests.Response()
    response.status_code = 200
    response._content = b'{"candidates": []}'
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return response
    monkeypatch.setattr(backend.requests, "post", post)
    monkeypatch.setattr(backend, "get_api_config", lambda **kw: pytest.fail("snapshot must be used"))
    monkeypatch.setattr(backend, "_generate_request_replay", lambda *a, **kw: {})
    result = backend.generate_image_aigc2d(prompt="only recolour", model="gemini-configured", aspect_ratio="auto",
        face_quality_boost=False, retry_limit=0, save_sub_dir=str(tmp_path), return_metadata=True,
        config_snapshot={"base_url": "https://frozen.invalid/v1beta/models/", "api_key": "test-key", "timeout": 42, "resolution": "2K"})
    assert len(calls) == 1 and calls[0][0].startswith("https://frozen.invalid/")
    assert calls[0][1]["timeout"] == 42
    payload = calls[0][1]["json"]
    assert payload["contents"][0]["parts"][0]["text"] == "only recolour"
    assert "aspectRatio" not in payload["generationConfig"]["imageConfig"]
    assert result["saved_files"] == []


@pytest.mark.parametrize("mutation", [lambda s: s.update(id="../../escape"),
                                    lambda s: s["images"].append({}),
                                    lambda s: s.update(turns=[{}])])
def test_corrupt_session_is_rejected(bundle, mutation):
    session, _, _, path = bundle
    mutation(session)
    rt.save_session(session, path)
    with pytest.raises(ValueError):
        rt.load_session(path)


def test_config_is_validated_before_any_dispatch(bundle, monkeypatch):
    session, ids, config, path = bundle
    config["model"] = "gpt-image-configured"
    monkeypatch.setattr(rt, "request_json", lambda *a: pytest.fail("must not dispatch"))
    with pytest.raises(ValueError, match="Gemini"):
        rt.DialogueRunner(session, path, config, config).run("背景变白", ids)


def test_real_worker_ui_thread_and_shared_tab_guards(bundle, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from modules.image_generation import image_edit as ui
    from modules.image_generation import image_edit_dialogue_widget as chat_ui
    app = QApplication.instance() or QApplication([])
    session, ids, config, path = bundle
    monkeypatch.setattr(ui, "IMAGE_EDIT_UI_STATE_FILE", str(tmp_path / "ui-state.json"))
    monkeypatch.setattr(ui, "PROMPT_DIR", str(tmp_path / "prompts"))
    monkeypatch.setattr(rt, "default_session_path", lambda: str(path))
    monkeypatch.setattr(chat_ui, "get_api_config", lambda **kw: dict(config))
    release = threading.Event()
    threads = []
    calls, stages = install_api(monkeypatch, tmp_path, [ids[1]])
    previous = rt.request_json
    def blocked(*args):
        threads.append(threading.get_ident())
        release.wait(3)
        return previous(*args)
    monkeypatch.setattr(rt, "request_json", blocked)
    widget = ui.ImageEditWidget(lambda: (config["base_url"], "test", "configured-vision"), lambda: ("", "", "", "gpt"), lambda: {})
    widget._add_images_from_paths([row["original"] for row in session["images"]])
    widget.image_list.item(1).setCheckState(Qt.CheckState.Checked)
    panel = widget.dialogue_panel
    widget.edit_modes.setCurrentIndex(1)
    panel.input.setPlainText("这张背景变白")
    panel.send()
    try:
        assert panel.busy and not widget.start_btn.isEnabled()
        assert not widget.add_btn.isEnabled()
        widget.start_processing()
        assert not widget.processing
        panel.send()
        release.set()
        deadline = time.monotonic() + 5
        while panel.busy and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        assert not panel.busy
        assert len(calls) == 1
        assert all(t != threading.get_ident() for t in threads)
        assert panel.session["turns"][-1]["allowed_ids"] == [2]
        assert panel.version_list.count() == 3
        assert "已返回图片" in panel.transcript.toPlainText()
        old_id = panel.session["id"]
        panel.new_conversation()
        assert panel.session["id"] != old_id
        assert (path.parent / f"conversation-{old_id[:12]}.json").exists()
        assert rt.load_session(path)["id"] == panel.session["id"]
    finally:
        release.set()
        if panel.worker:
            panel.worker.wait(5000)
            app.processEvents()
        widget.close()


@pytest.mark.parametrize("finish,text", [("length", "{}"), ("content_filter", "{}"), ("stop", "not json"), ("stop", "[]")])
def test_bad_model_json_no_retry(monkeypatch, finish, text):
    calls = []
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.chat = NS(completions=NS(create=self.create))
        def create(self, **kwargs):
            calls.append(kwargs)
            return NS(choices=[NS(finish_reason=finish, message=NS(content=text, refusal=None))])
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(rt, "OpenAI", Client)
    with pytest.raises(ValueError):
        rt.request_json("route", {"user": "hello"}, [], {"model": "vision", "api_key": "secret", "base_url": "https://example.invalid"})
    assert len(calls) == 1
