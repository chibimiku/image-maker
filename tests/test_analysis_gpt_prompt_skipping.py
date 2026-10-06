"""只出 Gemini 图的任务不该再算 gpt-image 专用短提示词。

对应用户 2026-10-01 反馈：只生成 Gemini 的任务，`gpt_image_prompt` / `gpt_image_prompt_short`
这两个字段（各要一次文本模型调用）根本用不上，却在每次分析收尾都照跑一遍。

契约（`SingleAnalyzerWidget._compute_gpt_prompts_for_task`）：

| 情况 | 是否算 gpt prompts |
|---|---|
| 生图通道单选 Gemini，且不自动生图 | ❌ 跳过 |
| 生图通道单选 Gemini，但勾了自动生图 | ✅ 照算（那次生图的通道可能是 gpt） |
| 生图通道 = gpt-image | ✅ 照算 |
| 本次带强制生图目标（重跑分析并生图） | ✅ 照算（不猜那次用哪个通道） |
"""

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

import modules.image_analysis.single_analyzer as single_analyzer_module
from modules.image_analysis.single_analyzer import SingleAnalyzerWidget
from utils.llm_retry import RetrySettings


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_widget(monkeypatch):
    monkeypatch.setattr(single_analyzer_module, "list_esrgan_models", lambda: ["realesrgan-x4plus"])
    return SingleAnalyzerWidget(
        config_getter_func=lambda *_a, **_k: ("https://example.invalid/v1", "test-key", "test-model"),
        img_config_getter_func=lambda *_a, **_k: ("https://example.invalid/v1", "image-key", "image-model", "openai"),
        styles_getter_func=lambda: {"默认风格": "masterpiece"},
        save_img_cfg_callback=lambda *_a, **_k: None,
        nsfw_default_getter_func=lambda: False,
        upscale_options_getter_func=lambda: {},
        outfit_style_history_getter_func=lambda: [],
        outfit_style_default_getter_func=lambda: "",
        persist_ui_state=False,
    )


# --------------------------------------------------------------------------
# 1. 开关判定
# --------------------------------------------------------------------------


def test_gemini_only_without_auto_gen_skips_gpt_prompts(qapp, monkeypatch):
    widget = _make_widget(monkeypatch)
    widget.gen_channel_gpt.setChecked(False)
    widget.gen_channel_gemini.setChecked(True)
    widget.auto_gen_orig_cb.setChecked(False)
    widget.auto_gen_ref_cb.setChecked(False)

    assert widget._compute_gpt_prompts_for_task() is False
    widget.close()


def test_gpt_image_channel_always_computes(qapp, monkeypatch):
    widget = _make_widget(monkeypatch)
    widget.gen_channel_gpt.setChecked(True)
    widget.auto_gen_orig_cb.setChecked(False)
    widget.auto_gen_ref_cb.setChecked(False)

    assert widget._compute_gpt_prompts_for_task() is True
    widget.close()


def test_gemini_channel_with_auto_gen_still_computes(qapp, monkeypatch):
    """勾了自动生图就算——那次生图可能走 gpt 通道，不能把字段抽掉。"""
    widget = _make_widget(monkeypatch)
    widget.gen_channel_gpt.setChecked(False)
    widget.gen_channel_gemini.setChecked(True)
    widget.auto_gen_ref_cb.setChecked(True)

    assert widget._compute_gpt_prompts_for_task() is True
    widget.close()


def test_forced_gen_targets_always_compute(qapp, monkeypatch):
    widget = _make_widget(monkeypatch)
    widget.gen_channel_gpt.setChecked(False)
    widget.gen_channel_gemini.setChecked(True)
    widget.auto_gen_orig_cb.setChecked(False)
    widget.auto_gen_ref_cb.setChecked(False)

    assert widget._compute_gpt_prompts_for_task(forced_targets=["refined"]) is True
    # 非法目标不算「强制生图」
    assert widget._compute_gpt_prompts_for_task(forced_targets=["bogus"]) is False
    widget.close()


def test_launch_passes_the_flag_to_worker(qapp, monkeypatch, tmp_path):
    from PIL import Image

    widget = _make_widget(monkeypatch)
    image_path = tmp_path / "girl.png"
    Image.new("RGB", (4, 4)).save(image_path)
    created = {}

    class _Signal:
        def connect(self, *_args, **_kwargs):
            return None

    class _FakeWorkerThread:
        def __init__(self, *args, **kwargs):
            created.update(kwargs)
            self.log_signal = _Signal()
            self.finish_signal = _Signal()
            self.finished = _Signal()
            self.meta_force_gen_targets = []

        def start(self):
            pass

    monkeypatch.setattr(single_analyzer_module, "WorkerThread", _FakeWorkerThread)
    widget.gen_channel_gpt.setChecked(False)
    widget.gen_channel_gemini.setChecked(True)
    widget.auto_gen_orig_cb.setChecked(False)
    widget.auto_gen_ref_cb.setChecked(False)

    widget._launch_analysis_task(str(image_path))
    assert created["gpt_prompts"] is False

    widget.gen_channel_gpt.setChecked(True)
    widget._launch_analysis_task(str(image_path))
    assert created["gpt_prompts"] is True
    widget.close()


# --------------------------------------------------------------------------
# 2. 端到端：Gemini-only 跑完整条分析，一次 gpt-image 调用都不许发生
# --------------------------------------------------------------------------


class _Completions:
    def __init__(self, owner):
        self.owner = owner

    def create(self, **kwargs):
        self.owner.calls.append(kwargs)
        return self.owner.response


class _Chat:
    def __init__(self, owner):
        self.completions = _Completions(owner)


class _FakeClient:
    def __init__(self, payload):
        body = json.dumps(payload, ensure_ascii=False)
        message = type("Message", (), {"content": body, "refusal": None})()
        choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
        self.response = type("Response", (), {"choices": [choice], "model": "m", "id": "req-1"})()
        self.base_url = "https://example.invalid/v1"
        self.calls = []
        self.chat = _Chat(self)


PAYLOAD = {
    "english_description": "a girl standing in a white dress",
    "japanese_title": "少女",
    "chinese_title": "少女",
    "booru-tags": ["1girl"],
}


def _make_png(tmp_path):
    from PIL import Image

    path = tmp_path / "girl.png"
    Image.new("RGB", (8, 8), "white").save(path)
    return str(path)


@pytest.mark.parametrize("gpt_prompts, expect_calls", [(False, 0), (True, 2)])
def test_worker_thread_only_calls_gpt_prompt_builder_when_asked(
        monkeypatch, tmp_path, gpt_prompts, expect_calls):
    """`gpt_prompts=False` 时一次 `build_gpt_image_prompt` 都不能发生（每档一次文本模型调用）。"""
    import utils.analysis_gpt_prompt as gpt_prompt

    calls = []

    def fake_build(desc, text_cfg=None, max_chars=1400, tier="full", log_callback=None, **kwargs):
        calls.append(tier)
        return f"{tier} prompt"

    monkeypatch.setattr(gpt_prompt, "build_gpt_image_prompt", fake_build)
    # Step 2 也换掉：本用例只关心收尾的 gpt 专用字段
    monkeypatch.setattr(
        single_analyzer_module, "step_2_refine_description",
        lambda data, *a, **k: dict(data, english_description=k.get("english_description")
                                   or "a girl standing in a white dress"),
    )
    client = _FakeClient(PAYLOAD)
    monkeypatch.setattr(single_analyzer_module, "OpenAI", lambda **kw: client)
    monkeypatch.setattr(single_analyzer_module, "load_retry_settings",
                        lambda *a, **k: RetrySettings(enabled=False, times=0, interval_seconds=0))
    monkeypatch.setattr(single_analyzer_module, "predict_local_booru_tags", lambda *a, **k: [])
    monkeypatch.setattr(single_analyzer_module, "get_local_pixiv_tag_candidates", lambda *a, **k: [])

    logs = []
    results = []
    thread = single_analyzer_module.WorkerThread(
        _make_png(tmp_path), "sk", "https://example.invalid/v1", "test-model",
        enable_refine=True, timeout_seconds=30, use_fallback=False, gpt_prompts=gpt_prompts,
    )
    thread.log_signal.connect(logs.append)
    thread.finish_signal.connect(results.append)
    thread.run()

    assert thread.last_status == "success"
    assert calls == (["full", "short"] if expect_calls else [])
    if not gpt_prompts:
        assert "已跳过 gpt-image 专用短提示词" in "\n".join(logs)
        assert "gpt_image_prompt" not in results[0]
    else:
        assert results[0]["gpt_image_prompt"] == "full prompt"
        assert results[0]["gpt_image_prompt_short"] == "short prompt"


# --------------------------------------------------------------------------
# 3. 判据本身（GUI 与无头链路共用）
# --------------------------------------------------------------------------


@pytest.mark.parametrize("channel, will_generate, forced, enabled, expected", [
    ("gpt-image", False, None, True, True),      # gpt 通道 → 要
    ("gpt-image", False, None, False, False),    # 配置关掉 → 恒否
    ("gemini", False, None, True, False),        # 只出 Gemini 图且不自动生图 → 跳过
    ("gemini", True, None, True, True),          # 有生图意图 → 照算
    ("gemini", False, ["refined"], True, True),  # 强制生图目标 → 照算
    ("", False, None, True, False),              # 通道未知且没有生图意图 → 跳过
    ("", False, ["bogus"], True, False),         # 非法目标不算强制
])
def test_should_build_gpt_prompts_rules(channel, will_generate, forced, enabled, expected):
    from utils.analysis_gpt_prompt import should_build_gpt_prompts

    assert should_build_gpt_prompts(
        channel=channel, will_generate=will_generate, forced_targets=forced, enabled=enabled,
    ) is expected


def test_pipeline_accepts_compute_flag_and_skips_builder(monkeypatch, tmp_path):
    """无头链路：`compute_gpt_prompts=False` 时整段跳过，不调 `build_gpt_image_prompt`。"""
    import modules.image_analysis.analysis_pipeline as pipeline
    from PIL import Image

    image_path = tmp_path / "girl.png"
    Image.new("RGB", (8, 8), "white").save(image_path)

    monkeypatch.setattr(pipeline, "predict_local_booru_tags", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "get_local_pixiv_tag_candidates", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "step_1_analyze_image", lambda *a, **k: dict(PAYLOAD))
    monkeypatch.setattr(pipeline, "step_2_refine_description",
                        lambda data, *a, **k: dict(data, english_description="a girl"))
    monkeypatch.setattr(pipeline, "OpenAI", lambda **k: _FakeClient(PAYLOAD))
    monkeypatch.setattr(pipeline, "calculate_closest_aspect_ratio", lambda *a, **k: "2:3")

    import utils.analysis_gpt_prompt as gpt_prompt
    monkeypatch.setattr(gpt_prompt, "build_gpt_image_prompt",
                        lambda *a, **k: pytest.fail("只出 Gemini 图的任务不该算 gpt prompts"))

    logs = []
    result = pipeline.analyze_single_image(
        str(image_path),
        {"base_url": "https://example.invalid/v1", "api_key": "sk", "model": "m",
         "nsfw_api_key": "secondary-test-key", "nsfw_base_url": "https://secondary.invalid/v1",
         "nsfw_model": "secondary-test-model",
         "enable_gpt_image_prompt_single": True},
        timeout_seconds=30,
        log_callback=logs.append,
        compute_gpt_prompts=False,
    )

    assert result is not None
    assert "gpt_image_prompt" not in result
    assert any("跳过 gpt-image 专用短提示词" in line for line in logs)
