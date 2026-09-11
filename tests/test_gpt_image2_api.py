"""gpt-image-2（new.aigc2d / autodl 双站点）尺寸收敛、请求体组装、端点路由与 Tab 冒烟测试。"""
import base64
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

import modules.image_generation.gpt_image2_tab as gpt_image2_tab
from modules.image_generation.gpt_image2_tab import (
    MODE_EDIT,
    MODE_GENERATE,
    SITE_AIGC2D,
    SITE_AUTODL,
    GptImage2Widget,
)
from modules.others import api_backend
from modules.others.api_backend import (
    GPT_IMAGE2_MAX_REFERENCE_IMAGES,
    GPT_IMAGE2_SIZE_LANDSCAPE,
    GPT_IMAGE2_SIZE_PORTRAIT,
    GPT_IMAGE2_SIZE_SQUARE,
    GPT_IMAGE2_SIZES,
    build_gpt_image2_payload,
    generate_image_aigc2d_gpt,
    generate_image_openai_image,
    gpt_image2_output_extension,
    normalize_gpt_image2_size,
)

TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)

TAB_CONFIG = {
    "apis": {
        "aigc-2d-gpt": {
            "base_url": "https://example.invalid/v1",
            "api_key": "test-key",
            "model": "gpt-image-2",
            "default_aspect_ratio": "2:3",
        },
        "autodl": {
            "base_url": "https://autodl.invalid/api/v1",
            "api_key": "autodl-key",
            "model": "gpt-image-2",
        },
    },
    "gpt_image2": {
        "site": SITE_AIGC2D,
        "mode": MODE_GENERATE,
        "sites": {
            SITE_AIGC2D: {"model": "gpt-image-2", "size": GPT_IMAGE2_SIZE_PORTRAIT, "quality": "medium", "output_format": "png", "n": 3},
            SITE_AUTODL: {"model": "gpt-image-2", "size": "1792x1024", "quality": "low", "output_format": "jpeg", "n": 1},
        },
    },
}


def _fake_api_config(**_kwargs):
    return {"base_url": "https://example.invalid/v1", "api_key": "test-key", "timeout": 30, "max_retries": 0}


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200
        self.encoding = "utf-8"

    def json(self):
        return self._payload

    @property
    def text(self):
        return json.dumps(self._payload)

    def raise_for_status(self):
        pass


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ---------------- 尺寸：aigc2d 通道只允许 3 档，永不 auto ----------------
@pytest.mark.parametrize(
    "aspect_ratio, expected",
    [
        ("1:1", GPT_IMAGE2_SIZE_SQUARE),
        ("16:9", GPT_IMAGE2_SIZE_LANDSCAPE),
        ("3:2", GPT_IMAGE2_SIZE_LANDSCAPE),
        ("4:3", GPT_IMAGE2_SIZE_LANDSCAPE),
        ("21:9", GPT_IMAGE2_SIZE_LANDSCAPE),
        ("9:16", GPT_IMAGE2_SIZE_PORTRAIT),
        ("2:3", GPT_IMAGE2_SIZE_PORTRAIT),
        ("3:4", GPT_IMAGE2_SIZE_PORTRAIT),
    ],
)
def test_normalize_size_from_aspect_ratio(aspect_ratio, expected):
    assert normalize_gpt_image2_size(aspect_ratio=aspect_ratio) == expected


@pytest.mark.parametrize(
    "size",
    ["1024x1024", "1536x1024", "1024x1536", "square", "landscape", "portrait", "横图", "纵向"],
)
def test_normalize_size_passthrough(size):
    assert normalize_gpt_image2_size(size=size) in GPT_IMAGE2_SIZES


def test_normalize_size_never_returns_auto_or_custom():
    for raw in ("auto", "", None, "2048x2048", "3840x2160", "2160x3840", "garbage", "12x"):
        assert normalize_gpt_image2_size(size=raw) in GPT_IMAGE2_SIZES


# ---------------- 请求体 ----------------
def test_payload_keeps_supported_fields_only():
    payload = build_gpt_image2_payload(
        prompt="a cat",
        model="gpt-image-2",
        size=GPT_IMAGE2_SIZE_PORTRAIT,
        quality="medium",
        output_format="jpeg",
        n=3,
        instructions="style head",
        post_instructions="style tail",
    )
    assert payload["model"] == "gpt-image-2"
    assert payload["size"] == GPT_IMAGE2_SIZE_PORTRAIT
    assert payload["quality"] == "medium"
    assert payload["output_format"] == "jpeg"
    assert payload["n"] == 3
    assert payload["prompt"] == "style head\n\na cat\n\nstyle tail"
    # gpt-image-2 不支持这两个字段，绝不能出现在请求体里
    assert "background" not in payload
    assert "input_fidelity" not in payload
    assert "response_format" not in payload


def test_payload_drops_auto_quality_and_bad_format():
    payload = build_gpt_image2_payload(prompt="p", quality="auto", output_format="bmp", n=0)
    assert "quality" not in payload
    assert "output_format" not in payload
    assert payload["n"] == 1


def test_payload_n_is_clamped():
    assert build_gpt_image2_payload(prompt="p", n=999)["n"] == 10
    assert build_gpt_image2_payload(prompt="p", n="oops")["n"] == 1


def test_output_extension_by_format_and_magic_bytes():
    assert gpt_image2_output_extension("jpeg") == ".jpg"
    assert gpt_image2_output_extension("webp") == ".webp"
    assert gpt_image2_output_extension("png") == ".png"
    assert gpt_image2_output_extension("", b"\xff\xd8\xff\xe0") == ".jpg"
    assert gpt_image2_output_extension("", b"RIFF\x00\x00\x00\x00WEBP") == ".webp"
    assert gpt_image2_output_extension("", b"\x89PNG\r\n\x1a\n") == ".png"


# ---------------- aigc2d 端点路由（不发真实请求） ----------------
def test_generation_without_reference_uses_json_generations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(api_backend, "get_api_config", _fake_api_config)
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeResponse(
            {"data": [{"b64_json": base64.b64encode(TINY_PNG).decode()}], "output_format": "png", "size": "1024x1536"}
        )

    monkeypatch.setattr(api_backend.requests, "post", fake_post)

    saved = api_backend.generate_image_aigc2d_gpt(
        prompt="一位少女站在海边",
        size="auto",            # 上游不收 auto，必须被收敛
        aspect_ratio="2:3",
        quality="medium",
        output_format="png",
        save_sub_dir="gpt-image-2-test",
        file_prefix="unit",
    )

    assert captured["url"] == "https://example.invalid/v1/images/generations"
    assert captured["kwargs"]["json"]["size"] == GPT_IMAGE2_SIZE_PORTRAIT
    assert captured["kwargs"]["json"]["quality"] == "medium"
    assert "files" not in captured["kwargs"]
    assert len(saved) == 1
    assert saved[0].endswith(".png")
    assert Path(saved[0]).read_bytes() == TINY_PNG


def test_generation_with_reference_uses_multipart_edits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(api_backend, "get_api_config", _fake_api_config)
    ref_path = tmp_path / "ref.png"
    ref_path.write_bytes(TINY_PNG)
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeResponse({"data": [{"b64_json": base64.b64encode(TINY_PNG).decode()}], "output_format": "png"})

    monkeypatch.setattr(api_backend.requests, "post", fake_post)

    saved = api_backend.generate_image_aigc2d_gpt(
        prompt="把背景换成日落海滩",
        image_paths=[str(ref_path)],
        aspect_ratio="2:3",
        save_sub_dir="gpt-image-2-test",
        file_prefix="unit-edit",
    )

    assert captured["url"] == "https://example.invalid/v1/images/edits"
    assert captured["kwargs"]["data"]["size"] == GPT_IMAGE2_SIZE_PORTRAIT
    assert captured["kwargs"]["data"]["model"] == "gpt-image-2"
    assert captured["kwargs"]["files"][0][0] == "image[]"
    assert len(saved) == 1


def test_edit_mode_without_images_aborts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(api_backend, "get_api_config", _fake_api_config)
    called = {"post": False}

    def fake_post(url, **kwargs):  # pragma: no cover - 不应该被调用
        called["post"] = True
        return _FakeResponse({"data": []})

    monkeypatch.setattr(api_backend.requests, "post", fake_post)

    assert api_backend.generate_image_aigc2d_gpt(prompt="p", mode="edit") == []
    assert called["post"] is False


# ---------------- Tab ----------------
class _SignalStub:
    def connect(self, *_args, **_kwargs):
        pass


class _FakeWorker:
    """记录后端与参数、不真正发请求的 Worker 替身。"""

    last_backend = None
    last_params = None

    def __init__(self, backend, params, parent=None):
        _FakeWorker.last_backend = backend
        _FakeWorker.last_params = dict(params)
        self.backend = backend
        self.log = _SignalStub()
        self.done = _SignalStub()
        self.error = _SignalStub()
        self.finished = _SignalStub()

    def start(self):
        pass

    def isRunning(self):
        return False

    def requestInterruption(self):
        pass

    def deleteLater(self):
        pass


def _write_tab_config(tmp_path, payload=None):
    config_path = tmp_path / "config-image.json"
    config_path.write_text(json.dumps(payload if payload is not None else TAB_CONFIG, ensure_ascii=False), encoding="utf-8")
    return config_path


@pytest.fixture
def tab(qapp, tmp_path, monkeypatch):
    config_path = _write_tab_config(tmp_path)
    monkeypatch.setattr(gpt_image2_tab, "CONFIG_IMAGE_FILE", str(config_path))
    monkeypatch.setattr(gpt_image2_tab, "GptImage2Worker", _FakeWorker)
    return GptImage2Widget()


def test_tab_has_two_sites_with_per_site_sizes(tab):
    assert tab.site_combo.count() == 2
    assert [tab.site_combo.itemData(i) for i in range(2)] == [SITE_AIGC2D, SITE_AUTODL]
    assert tab.mode_combo.count() == 2
    assert [tab.mode_combo.itemText(i) for i in range(2)] == [MODE_GENERATE, MODE_EDIT]
    # aigc2d 站点只给 3 档
    assert tab.current_site() == SITE_AIGC2D
    assert [tab.size_combo.itemData(i) for i in range(tab.size_combo.count())] == list(GPT_IMAGE2_SIZES)


def test_tab_loads_per_site_defaults(tab):
    assert tab.size_combo.currentData() == GPT_IMAGE2_SIZE_PORTRAIT
    assert tab.quality_combo.currentData() == "medium"
    assert tab.n_spin.value() == 3


def test_switching_site_repopulates_size_model_and_n(tab):
    tab.site_combo.setCurrentIndex(tab.site_combo.findData(SITE_AUTODL))
    sizes = [tab.size_combo.itemData(i) for i in range(tab.size_combo.count())]
    assert "auto" in sizes and "1792x1024" in sizes
    assert tab.size_combo.currentData() == "1792x1024"
    assert tab.quality_combo.currentData() == "low"
    assert tab.output_format_combo.currentText() == "jpeg"
    # autodl 一次一张，没有 n 参数
    assert tab.n_spin.isEnabled() is False
    assert "autodl" in tab.api_hint.text()


def test_aigc2d_request_uses_aigc2d_backend(tab, tmp_path):
    image_path = tmp_path / "ref.png"
    image_path.write_bytes(TINY_PNG)
    tab._add_paths([str(image_path)])
    tab.prompt_edit.setPlainText("把背景换成海边")

    backend, params = tab.build_request()

    assert backend is generate_image_aigc2d_gpt
    assert params["api_type"] == "aigc-2d-gpt"
    assert params["size"] == GPT_IMAGE2_SIZE_PORTRAIT
    assert params["quality"] == "medium"
    assert params["n"] == 3
    assert params["mode"] == "generate"
    assert params["image_paths"] == [str(image_path)]
    assert params["save_sub_dir"] == "gpt-image-2"


def test_autodl_request_uses_openai_backend(tab, tmp_path):
    tab.site_combo.setCurrentIndex(tab.site_combo.findData(SITE_AUTODL))
    image_path = tmp_path / "ref.png"
    image_path.write_bytes(TINY_PNG)
    tab._add_paths([str(image_path)])
    tab.prompt_edit.setPlainText("换成海边背景")
    tab.mode_combo.setCurrentText(MODE_EDIT)

    backend, params = tab.build_request()

    assert backend is generate_image_openai_image
    assert params["api_type"] == "autodl"
    assert params["size"] == "1792x1024"
    assert params["quality"] == "low"
    assert params["output_format"] == "jpeg"
    assert "n" not in params          # autodl 通道一次一张
    assert params["image_paths"] == [str(image_path)]
    assert params["save_sub_dir"] == "gpt-image-2-autodl"


def test_generate_wires_worker_with_selected_site(tab, monkeypatch):
    monkeypatch.setattr(gpt_image2_tab, "get_api_config", lambda api_type=None: {"api_key": "k", "base_url": "u"})
    tab.prompt_edit.setPlainText("一只猫")
    tab.site_combo.setCurrentIndex(tab.site_combo.findData(SITE_AUTODL))

    tab.generate()

    assert _FakeWorker.last_backend is generate_image_openai_image
    assert _FakeWorker.last_params["api_type"] == "autodl"


def test_legacy_flat_config_node_is_migrated(qapp, tmp_path, monkeypatch):
    legacy = {
        "apis": {"aigc-2d-gpt": {"base_url": "u", "api_key": "k", "model": "gpt-image-2"}},
        "gpt_image2": {"mode": "生图", "size": "1536x1024", "quality": "low", "output_format": "webp", "n": 2},
    }
    config_path = _write_tab_config(tmp_path, legacy)
    monkeypatch.setattr(gpt_image2_tab, "CONFIG_IMAGE_FILE", str(config_path))
    widget = GptImage2Widget()
    assert widget.size_combo.currentData() == GPT_IMAGE2_SIZE_LANDSCAPE
    assert widget.quality_combo.currentData() == "low"
    assert widget.output_format_combo.currentText() == "webp"
    assert widget.n_spin.value() == 2


def test_tab_caps_reference_images(tab, monkeypatch):
    monkeypatch.setattr(gpt_image2_tab.QMessageBox, "information", lambda *args, **kwargs: None)
    base = Path(gpt_image2_tab.CONFIG_IMAGE_FILE).parent
    paths = []
    for index in range(GPT_IMAGE2_MAX_REFERENCE_IMAGES + 3):
        p = base / f"img_{index}.png"
        p.write_bytes(TINY_PNG)
        paths.append(str(p))
    tab._add_paths(paths)
    assert len(tab.image_paths) == GPT_IMAGE2_MAX_REFERENCE_IMAGES

    tab.clear_images()
    assert tab.image_paths == []
    assert tab.image_list.count() == 0
