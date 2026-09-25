"""gpt-image-2（new.aigc2d / autodl 双站点）尺寸收敛、请求体组装、端点路由与 Tab 冒烟测试。"""
import base64
import json
import logging
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QMimeData, Qt, QUrl
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication, QListWidgetItem

import modules.image_generation.gpt_image2_tab as gpt_image2_tab
from modules.image_generation.gpt_image2_tab import (
    MODE_CHOICES,
    MODE_EDIT,
    MODE_GENERATE,
    MODE_REPAINT,
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
    generate_image_repaint,
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
            # 与真实部署一致：aigc-2d-gpt 与 aigc2d 共用一把环境变量（env_slug）
            "env_slug": "aigc2d",
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
def test_moderation_rejection_is_not_retried_or_rerouted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(api_backend, "get_api_config", _fake_api_config)
    calls = []
    def reject(url, **kwargs):
        calls.append(url)
        return _FakeResponse({"error": {"code": "moderation_blocked",
            "message": "Image rejected by safety system"}})
    monkeypatch.setattr(api_backend.requests, "post", reject)
    assert api_backend.generate_image_aigc2d_gpt(prompt="test", mode="generate") == []
    assert calls == ["https://example.invalid/v1/images/generations"]


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
    last_repaint_params = None
    last_init = None

    def __init__(self, backend, params, parent=None, repaint_params=None):
        _FakeWorker.last_backend = backend
        _FakeWorker.last_params = dict(params)
        _FakeWorker.last_repaint_params = dict(repaint_params) if repaint_params else None
        _FakeWorker.last_init = {"backend": backend, "params": dict(params),
                                 "repaint_params": repaint_params}
        self.backend = backend
        self.params = dict(params)
        self.repaint_params = repaint_params
        self.log = _SignalStub()
        self.done = _SignalStub()
        self.error = _SignalStub()
        self.raw_done = _SignalStub()
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
    assert tab.mode_combo.count() == len(MODE_CHOICES)
    assert [tab.mode_combo.itemText(i) for i in range(tab.mode_combo.count())] == list(MODE_CHOICES)
    assert MODE_REPAINT in MODE_CHOICES
    # aigc2d 站点只给 3 档
    assert tab.current_site() == SITE_AIGC2D
    size_values = [tab.size_combo.itemData(i) for i in range(tab.size_combo.count())]
    # aigc2d：三档固定尺寸 + 「自动（跟随参考图比例）」
    assert [v for v in size_values if v in GPT_IMAGE2_SIZES] == list(GPT_IMAGE2_SIZES)
    assert tab.SIZE_FOLLOW_INPUT in size_values


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


# ---------------- 模型下拉（gpt-image 家族，含 2.5） ----------------
def _combo_texts(combo):
    return [combo.itemText(index) for index in range(combo.count())]


def test_model_dropdown_lists_site_gpt_image_family(tab):
    models = _combo_texts(tab.model_combo)
    # aigc2d 站点有 2.5 的 flare / sunburst 及其 -c 计费版
    assert models[:6] == [
        "gpt-image-2",
        "gpt-image-2-c",
        "gpt-image-2.5-flare",
        "gpt-image-2.5-flare-c",
        "gpt-image-2.5-sunburst",
        "gpt-image-2.5-sunburst-c",
    ]
    assert "dall-e-3" in models
    assert tab.model_combo.currentText() == "gpt-image-2"
    assert tab.current_model() == "gpt-image-2"

    flare_index = models.index("gpt-image-2.5-flare")
    note = tab.model_combo.itemData(flare_index, Qt.ItemDataRole.ToolTipRole)
    assert "ChatGPT Images 2.5" in note       # 悬浮提示说明每个模型的定位


def test_model_dropdown_of_autodl_has_no_25_series(tab):
    tab.site_combo.setCurrentIndex(tab.site_combo.findData(SITE_AUTODL))
    models = _combo_texts(tab.model_combo)
    assert models == ["gpt-image-2"]          # 实测 autodl 只有 gpt-image-2
    assert not any("2.5" in name for name in models)


def test_model_combo_stays_editable_and_reaches_request(tab):
    # 下拉里点一个 2.5
    tab.model_combo.setCurrentText("gpt-image-2.5-sunburst")
    tab.prompt_edit.setPlainText("少女跳芭蕾")
    _backend, params = tab.build_request()
    assert params["model"] == "gpt-image-2.5-sunburst"

    # 手输清单里没有的模型名也能用，并会被记进配置
    tab.model_combo.setCurrentText("gpt-image-9-custom")
    assert tab.current_model() == "gpt-image-9-custom"
    _backend, params = tab.build_request()
    assert params["model"] == "gpt-image-9-custom"

    tab.save_defaults()
    data = json.loads(Path(gpt_image2_tab.CONFIG_IMAGE_FILE).read_text(encoding="utf-8"))
    assert data["gpt_image2"]["sites"][SITE_AIGC2D]["model"] == "gpt-image-9-custom"


def test_fetched_models_extend_dropdown_and_skip_non_image_models(tab):
    tab._on_models_fetched(
        ["gpt-image-2", "gpt-image-2.5-flare", "gpt-image-9-preview", "claude-opus-4-6", "Kimi-K3", "Z-Image-Turbo"]
    )

    models = _combo_texts(tab.model_combo)
    assert "gpt-image-9-preview" in models        # 站点新增的模型自动进下拉
    assert "claude-opus-4-6" not in models        # 文本/视频模型不进
    assert "Kimi-K3" not in models
    assert "Z-Image-Turbo" not in models
    assert tab.current_model() == "gpt-image-2"   # 当前选择不受刷新影响
    assert "gpt-image/dall-e" in tab.log_view.toPlainText()

    # 每个站点各自记住自己的实时清单
    tab.site_combo.setCurrentIndex(tab.site_combo.findData(SITE_AUTODL))
    assert _combo_texts(tab.model_combo) == ["gpt-image-2"]
    tab.site_combo.setCurrentIndex(tab.site_combo.findData(SITE_AIGC2D))
    assert "gpt-image-9-preview" in _combo_texts(tab.model_combo)


def test_refresh_models_requires_api_key(tab, monkeypatch):
    warned = []
    monkeypatch.setattr(gpt_image2_tab.QMessageBox, "warning", lambda *args, **kwargs: warned.append(args))
    monkeypatch.setattr(gpt_image2_tab, "get_api_config", lambda api_type=None: {"api_key": "", "base_url": "u"})

    tab.refresh_models()

    assert warned and "缺少 api_key" in str(warned[0])


def test_models_fetch_error_is_logged_and_warns(tab, monkeypatch):
    warned = []
    monkeypatch.setattr(gpt_image2_tab.QMessageBox, "warning", lambda *args, **kwargs: warned.append(args))

    tab._on_models_fetch_error("HTTPError: 401 Unauthorized")

    assert "获取模型列表失败: HTTPError: 401 Unauthorized" in tab.log_view.toPlainText()
    assert warned and "HTTPError" in str(warned[0])


# ---------------- /v1/models 端点与模型清单（api_backend） ----------------
def test_resolve_models_endpoint_follows_images_channel():
    assert api_backend.resolve_models_endpoint("https://example.invalid/v1", api_type="aigc-2d-gpt") == (
        "https://example.invalid/v1/models"
    )
    assert api_backend.resolve_models_endpoint("https://autodl.invalid/api/v1", api_type="autodl") == (
        "https://autodl.invalid/api/v1/models"
    )
    # 旧配置可能直接填到 /images/generations，也要能收敛回 /models
    assert api_backend.resolve_models_endpoint(
        "https://example.invalid/v1/images/generations", api_type="aigc-2d-gpt"
    ) == "https://example.invalid/v1/models"


def test_pick_gpt_image_models_keeps_order_filters_and_extras():
    choices = api_backend.pick_gpt_image_models(
        site=SITE_AIGC2D,
        models=["claude-opus-4-6", "gpt-image-9-preview", "dall-e-3"],
        extra=["my-custom-model"],
    )
    assert choices[:3] == ["gpt-image-2", "gpt-image-2-c", "gpt-image-2.5-flare"]
    assert "my-custom-model" in choices
    assert "gpt-image-9-preview" in choices
    assert "claude-opus-4-6" not in choices
    # autodl 常用清单里没有 2.5（实测站点也只有 gpt-image-2）
    assert api_backend.pick_gpt_image_models(site=SITE_AUTODL) == ["gpt-image-2"]
    # 站点自己报上来的 gpt-image 系模型一律并入（刷新能让新模型自动出现在下拉里）
    assert api_backend.pick_gpt_image_models(site=SITE_AUTODL, models=["Kimi-K3", "gpt-image-9"]) == [
        "gpt-image-2",
        "gpt-image-9",
    ]


def test_list_available_models_gets_v1_models(monkeypatch, tmp_path):
    config_path = _write_tab_config(tmp_path)
    captured = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeResponse(
            {"data": [{"id": "gpt-image-2"}, {"id": "gpt-image-2.5-flare"}, {"id": "gpt-image-2"}, {"id": "Kimi-K3"}]}
        )

    monkeypatch.setattr(api_backend.requests, "get", fake_get)

    models = api_backend.list_available_models(api_type="aigc-2d-gpt", config_path=str(config_path), timeout=5)

    assert models == ["Kimi-K3", "gpt-image-2", "gpt-image-2.5-flare"]     # 去重 + 排序
    assert captured["url"] == "https://example.invalid/v1/models"
    assert captured["kwargs"]["headers"]["Authorization"] == "Bearer test-key"
    assert captured["kwargs"]["timeout"] == 5


def test_list_available_models_requires_api_key(tmp_path):
    config_path = _write_tab_config(tmp_path)
    with pytest.raises(ValueError, match="api_key"):
        api_backend.list_available_models(api_type="missing-api-type", config_path=str(config_path))


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
    assert tab.image_grid.count() == 0
    assert tab.image_grid.empty() is True


# ---------------- 参考图缩略图网格（删除 / 拖拽重排 / 预览） ----------------
def _write_ref_png(path, color=0x3366CC):
    """TINY_PNG 的 IDAT 校验不过 Qt libpng，缩略图测试要用真 PNG。"""
    image = QImage(16, 16, QImage.Format.Format_RGB32)
    image.fill(color)
    assert image.save(str(path))
    return str(path)


def _ref_images(tab, tmp_path, count=3):
    paths = [_write_ref_png(tmp_path / f"ref_{index}.png", 0x204060 + index * 0x102030) for index in range(count)]
    tab._add_paths(paths)
    return paths


def test_reference_grid_shows_thumbnails_with_ordinal_captions(tab, tmp_path):
    paths = _ref_images(tab, tmp_path)

    assert tab.image_paths == paths
    cards = tab.image_grid.cards()
    assert len(cards) == len(paths)

    for index, card in enumerate(cards):
        # 缩略图（而不是裸路径文本）
        assert card.thumb_label.pixmap() is not None
        assert card.thumb_label.pixmap().isNull() is False
        assert card.caption_label.text().startswith(f"{index + 1}.")
        assert card.path == paths[index]
        # 说明文字保留文件后缀，但绝不显示目录
        assert card.caption_label.text().endswith("png")
        assert os.sep not in card.caption_label.text()
        # 完整路径只在 tooltip 里
        assert paths[index] in card.toolTip()
        assert paths[index] not in card.caption_label.text()


def test_reference_card_delete_button_removes_only_that_image(tab, tmp_path, qapp):
    paths = _ref_images(tab, tmp_path)

    tab.image_grid.card_at(1).delete_btn.click()
    qapp.processEvents()

    assert tab.image_paths == [paths[0], paths[2]]
    assert tab.image_grid.count() == 2
    # 序号重排后仍然连续
    assert [card.caption_label.text()[:2] for card in tab.image_grid.cards()] == ["1.", "2."]


def test_reference_card_drop_reorders_and_request_follows_order(tab, tmp_path):
    paths = _ref_images(tab, tmp_path)

    # 把第 1 张拖到第 3 张的位置
    tab.image_grid.card_at(0).dropped.emit(0, tab.image_grid.card_at(2))

    assert tab.image_paths == [paths[1], paths[2], paths[0]]

    _backend, params = tab.build_request()
    assert params["image_paths"] == [paths[1], paths[2], paths[0]]


def test_reference_grid_move_image_clamps_and_ignores_bad_index(tab, tmp_path):
    paths = _ref_images(tab, tmp_path)

    tab.image_grid.move_image(2, 0)
    assert tab.image_paths == [paths[2], paths[0], paths[1]]

    tab.image_grid.move_image(9, 0)
    tab.image_grid.move_image(0, 0)
    assert tab.image_paths == [paths[2], paths[0], paths[1]]


def test_reference_grid_clear_emits_images_changed(tab, tmp_path):
    _ref_images(tab, tmp_path)
    received = []
    tab.image_grid.images_changed.connect(received.append)

    tab.clear_images()

    assert tab.image_paths == []
    assert received == [[]]


def test_reference_click_previews_and_double_click_opens_dir(tab, tmp_path, monkeypatch):
    paths = _ref_images(tab, tmp_path)
    opened = []
    monkeypatch.setattr(gpt_image2_tab.os, "startfile", lambda path: opened.append(path))

    tab.image_grid.image_clicked.emit(paths[1])
    tab.image_grid.image_double_clicked.emit(paths[2])

    assert tab.preview_label.pixmap().isNull() is False
    assert opened == [str(tmp_path)]


def test_tab_drop_event_adds_images_from_urls(tab, tmp_path):
    image = tmp_path / "dropped.png"
    image.write_bytes(TINY_PNG)

    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile(str(image))])

    class _DropEvent:
        def __init__(self, mime):
            self._mime = mime
            self.accepted = False
            self.ignored = False

        def mimeData(self):
            return self._mime

        def acceptProposedAction(self):
            self.accepted = True

        def ignore(self):
            self.ignored = True

    event = _DropEvent(mime_data)
    tab.dropEvent(event)

    assert event.accepted is True
    assert tab.image_paths == [os.path.normpath(str(image))]


# ---------------- 密钥来源（环境变量优先，配置兜底） ----------------
def test_env_var_key_names_normalize_node_id():
    """节点名里的 - / . 要转成下划线再拼变量名。"""
    names = api_backend._env_key_names("aigc-2d-gpt")
    assert names[0] == "IMAGE_MAKER_AIGC_2D_GPT_API_KEY"
    assert "AIGC_2D_GPT_KEY" in names
    assert api_backend._env_key_names("") == ()


def test_env_slug_lets_two_nodes_share_one_env_var(monkeypatch):
    """同一家服务的两个节点（aigc2d / aigc-2d-gpt）可用 env_slug 共用一把环境变量。"""
    cfg = {"api_key": "", "env_slug": "aigc2d"}
    monkeypatch.setenv("IMAGE_MAKER_AIGC2D_API_KEY", "sk-SHARED-000000000000000000000000000")
    assert api_backend._env_key_names("aigc-2d-gpt", cfg)[0] == "IMAGE_MAKER_AIGC2D_API_KEY"
    assert api_backend.resolve_api_key(cfg, "aigc-2d-gpt") == "sk-SHARED-000000000000000000000000000"
    assert api_backend.api_key_source("aigc-2d-gpt", cfg) == "env:IMAGE_MAKER_AIGC2D_API_KEY"


def test_env_var_overrides_config_key(monkeypatch):
    """设了环境变量就用环境变量，不用配置文件里那把。"""
    cfg = {"api_key": "sk-FROM-CONFIG-000000000000000000000000000"}
    monkeypatch.delenv("IMAGE_MAKER_AIGC2D_API_KEY", raising=False)
    assert api_backend.resolve_api_key(cfg, "aigc2d") == cfg["api_key"]
    assert api_backend.api_key_source("aigc2d", cfg) == "config"

    monkeypatch.setenv("IMAGE_MAKER_AIGC2D_API_KEY", "sk-FROM-ENV-111111111111111111111111111")
    assert api_backend.resolve_api_key(cfg, "aigc2d") == "sk-FROM-ENV-111111111111111111111111111"
    assert api_backend.api_key_source("aigc2d", cfg) == "env:IMAGE_MAKER_AIGC2D_API_KEY"


def test_config_key_may_be_empty_when_env_var_supplies_it(monkeypatch, tmp_path):
    """配置文件留空 + 环境变量给 key：也能正常解析（不再报「缺少 api_key」）。"""
    monkeypatch.setenv("IMAGE_MAKER_AIGC_2D_GPT_API_KEY", "sk-FROM-ENV-222222222222222222222222222")
    conf = tmp_path / "config.json"
    conf.write_text(json.dumps({
        "current_api": "aigc-2d-gpt",
        "apis": {"aigc-2d-gpt": {"base_url": "https://example.invalid/v1", "api_key": "", "model": "gpt-image-2"}},
    }), encoding="utf-8")

    cfg = api_backend.get_api_config(config_path=str(conf), api_type="aigc-2d-gpt")
    assert cfg["api_key"] == "sk-FROM-ENV-222222222222222222222222222"
    assert cfg["_api_key_source"] == "env:IMAGE_MAKER_AIGC_2D_GPT_API_KEY"

    # 没有环境变量时，空 key 就是空
    monkeypatch.delenv("IMAGE_MAKER_AIGC_2D_GPT_API_KEY")
    cfg2 = api_backend.get_api_config(config_path=str(conf), api_type="aigc-2d-gpt")
    assert cfg2["api_key"] == ""
    assert cfg2["_api_key_source"] == "none"


def test_get_api_config_does_not_mutate_cached_config(tmp_path):
    """get_api_config 返回副本：调用方改它不会污染后续读取（含 load_config 的缓存）。

    自带 tmp 配置：不依赖开发机 conf/config.json 或 .env 里有没有 aigc2d 节点
    （没有节点时拿到的是空 dict，断言会退化成"空 dict 不等于被改过的 dict"而失去意义）。
    """
    from modules.others import api_backend as backend

    conf = tmp_path / "config.json"
    conf.write_text(json.dumps({
        "current_api": "aigc-2d-gpt",
        "apis": {"aigc-2d-gpt": {"base_url": "https://example.invalid/v1",
                                 "api_key": "sk-ORIGINAL", "model": "gpt-image-2"}},
    }), encoding="utf-8")

    cfg = backend.get_api_config(config_path=str(conf), api_type="aigc-2d-gpt")
    assert cfg["api_key"] == "sk-ORIGINAL"
    cfg["api_key"] = "sk-MUTATED"
    again = backend.get_api_config(config_path=str(conf), api_type="aigc-2d-gpt")
    assert again["api_key"] == "sk-ORIGINAL", "调用方改返回值不能污染缓存里的配置"


def test_tab_hint_shows_key_source(tab, monkeypatch):
    """界面上的 Key 状态要标出「来自配置」还是「来自哪个环境变量」，且不含密钥本身。

    这里自己定义节点（`IMAGE_MAKER_NODES`，含 `env_slug=aigc2d`）：Tab 的 key 解析走的是
    `api_backend.get_api_config`（默认 conf/config.json + 环境变量合并），不会用 Tab 自己的
    临时配置，所以断言不能依赖开发机上的 `.env` / 真实配置。
    """
    monkeypatch.setenv("IMAGE_MAKER_NODES", json.dumps({
        "aigc-2d-gpt": {"base_url": "https://example.invalid/v1", "model": "gpt-image-2",
                        "env_slug": "aigc2d"},
    }))
    monkeypatch.setenv("IMAGE_MAKER_AIGC2D_API_KEY", "sk-FROM-ENV-333333333333333333333333333")
    tab.refresh_api_hint()
    text = tab.api_hint.text()
    assert "IMAGE_MAKER_AIGC2D_API_KEY" in text
    assert "sk-FROM-ENV" not in text
    assert "未配置" not in text


def test_key_hint_mentions_both_sources(tab):
    """缺 key 的提示要同时给出两条路：配置文件 与环境变量名。"""
    text = tab._key_hint("aigc-2d-gpt")
    assert "conf/config.json" in text
    assert "IMAGE_MAKER_AIGC_2D_GPT_API_KEY" in text


# ---------------- 日志栏（UI 诊断） ----------------
def test_mask_secret_hides_middle_of_key():
    # 用长度与真实 key 相同的**假** key（51 字符），只验证掩码规则，绝不写真实凭证进仓库
    fake_key = "sk-TESTKEYabcdefghijklmnopqrstuvwxyz0123456789ABCDE"
    assert len(fake_key) == 51
    masked = gpt_image2_tab.mask_secret(fake_key)
    assert masked.startswith("sk-TES")
    assert masked.endswith("len=51)")
    assert "abcdefghijklmnopqrstuvwxyz" not in masked


def test_format_request_dump_contains_endpoint_model_and_images(tab, tmp_path):
    image = tmp_path / "ref.png"
    image.write_bytes(TINY_PNG)
    params = {
        "prompt": "少女跳芭蕾",
        "image_paths": [str(image)],
        "model": "gpt-image-2.5-flare",
        "size": GPT_IMAGE2_SIZE_PORTRAIT,
        "quality": "medium",
        "output_format": "png",
        "n": 2,
    }
    lines = gpt_image2_tab.format_request_dump(
        params, {"base_url": "https://example.invalid/v1", "api_key": "sk-secret-key-value", "timeout": 600}, SITE_AIGC2D, "aigc-2d-gpt", MODE_GENERATE
    )
    text = "\n".join(lines)
    assert "https://example.invalid/v1/images/edits" in text
    assert "gpt-image-2.5-flare" in text
    assert str(image) in text
    assert "sk-secret-key-value" not in text          # 明文 key 不能进日志
    assert "少女跳芭蕾" in text


def test_backend_log_handler_forwards_records():
    captured = []
    handler = gpt_image2_tab.BackendLogHandler(captured.append)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    record = logging.LogRecord("whatai_logger", logging.WARNING, __file__, 1, "服务器返回 JSON 已保存到: x.json", None, None)
    handler.emit(record)
    assert captured == ["WARNING 服务器返回 JSON 已保存到: x.json"]


def test_backend_logger_info_records_reach_sink():
    """真挂到 api_backend 的 logger 上，确认 INFO 级请求日志也会进日志栏。"""
    captured = []
    handler = gpt_image2_tab.BackendLogHandler(captured.append)
    backend_logger = logging.getLogger("whatai_logger")
    backend_logger.addHandler(handler)
    try:
        backend_logger.info("=== 发起 gpt-image(aigc2d) API 请求  model=gpt-image-2.5-flare ===")
    finally:
        backend_logger.removeHandler(handler)
    assert any("model=gpt-image-2.5-flare" in item for item in captured)


def test_generate_prints_request_details_to_log(tab, monkeypatch):
    monkeypatch.setattr(gpt_image2_tab, "get_api_config", lambda api_type=None: {"api_key": "k", "base_url": "https://example.invalid/v1", "model": "gpt-image-2"})
    tab.prompt_edit.setPlainText("少女跳芭蕾舞")
    tab.generate()

    log_text = tab.log_view.toPlainText()
    assert "开始生成" in log_text
    assert "端点=https://example.invalid/v1/images/generations" in log_text
    assert "将要调用: generate_image_aigc2d_gpt" in log_text
    assert "参考图: 无" in log_text


def test_empty_result_logs_troubleshooting_steps(tab):
    tab.on_done([])
    log_text = tab.log_view.toPlainText()
    assert "[结果] 服务器没有返回可用图片" in log_text
    assert "moderation_blocked" in log_text
    assert "log/<日期>.log" in log_text
    assert "无图片" in tab.status_label.text()


def test_on_error_logs_hint(tab, monkeypatch):
    monkeypatch.setattr(gpt_image2_tab.QMessageBox, "critical", lambda *args, **kwargs: None)
    tab.on_error("HTTPError: 500 server error")
    log_text = tab.log_view.toPlainText()
    assert "调用失败: HTTPError: 500 server error" in log_text
    assert "429/5xx 会自动重试" in log_text


# ---------------- gpt-image 产物优化（Gemini 重绘提线） ----------------
@pytest.fixture
def repaint_config(tmp_path, monkeypatch):
    """把重绘 prompt / 配置的读写重定向到临时文件，避免测试污染 prompts/gpt-image-optimize/。"""
    from utils import gpt_image_optimize

    prompt_dir = tmp_path / "gpt-image-optimize"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "repaint-system.md").write_text("MAIN-PROMPT-BODY\n", encoding="utf-8")
    (prompt_dir / "repaint-detail-suffix.md").write_text("DETAIL-SUFFIX-BODY\n", encoding="utf-8")
    config_path = prompt_dir / "config.json"
    config_path.write_text(json.dumps({"model": "gemini-3-pro-image-preview", "resolution": "2K"},
                                      ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(gpt_image_optimize, "CONFIG_RELATIVE_PATH", "gpt-image-optimize/config.json")
    monkeypatch.setattr(gpt_image_optimize, "read_prompt_file",
                        lambda rel: (tmp_path / str(rel).replace("\\", "/")).read_text(encoding="utf-8"))
    return config_path


def test_repaint_prompt_is_loaded_from_prompts_dir():
    from utils import gpt_image_optimize

    prompt = gpt_image_optimize.build_repaint_prompt()
    # 固件里必须有的关键约束（改了 prompt 也该保留这些语义）
    assert "PRESERVE EVERYTHING THE CHARACTER IS WEARING OR CARRYING" in prompt
    assert "gloves and any hand coverings" in prompt
    assert "tights, pantyhose, stockings" in prompt
    assert "tears, tear streaks" in prompt
    assert "five fingers" in prompt.lower()
    assert "scallop" in prompt
    assert len(prompt) > 1500


def test_repaint_prompt_suffix_can_be_disabled(repaint_config):
    from utils import gpt_image_optimize

    with_suffix = gpt_image_optimize.build_repaint_prompt({"use_detail_suffix": True})
    without = gpt_image_optimize.build_repaint_prompt({"use_detail_suffix": False})
    assert "DETAIL-SUFFIX-BODY" in with_suffix
    assert "DETAIL-SUFFIX-BODY" not in without
    assert len(with_suffix) > len(without)


def test_repaint_plan_prefix_derives_from_source_name():
    from utils import gpt_image_optimize

    plan = gpt_image_optimize.plan_output("data/20260919/x/gpt-image-2_output_193138_0_d09c70.png")
    assert plan["file_prefix"] == "repaint_gpt-image-2_output_193138_0_d09c70"
    assert plan["save_sub_dir"] == "gpt_image_repaint"


def test_tab_repaint_checkbox_toggles_chaining(tab):
    """勾选 = 链式重绘；生图与编辑模式都生效，重绘模式恒定触发。"""
    tab.repaint_check.setChecked(False)
    assert tab.repaint_enabled() is False        # 未勾选则不触发
    tab.repaint_check.setChecked(True)
    assert tab.repaint_enabled() is True         # 生图模式勾了 = 链式重绘
    tab.mode_combo.setCurrentText(MODE_EDIT)
    assert tab.repaint_enabled() is True         # 编辑模式同样支持链式重绘
    tab.mode_combo.setCurrentText(MODE_REPAINT)
    assert tab.repaint_enabled() is True         # 重绘模式恒定触发（与勾选框无关）
    tab.mode_combo.setCurrentText(MODE_GENERATE)
    tab.repaint_check.setChecked(False)


def test_generate_with_repaint_checkbox_chains_repaint(tab, monkeypatch, repaint_config):
    monkeypatch.setattr(gpt_image2_tab, "get_api_config", lambda api_type=None: {"api_key": "k", "base_url": "u"})
    tab.prompt_edit.setPlainText("一只猫")
    tab.repaint_check.setChecked(True)
    tab.repaint_repeat_spin.setValue(2)

    tab.generate()

    assert _FakeWorker.last_backend is generate_image_aigc2d_gpt
    rp = _FakeWorker.last_repaint_params
    assert rp is not None
    assert rp["model"] == "gemini-3-pro-image-preview"
    assert rp["resolution"] == "2K"
    assert rp["repeat"] == 2
    assert "source_paths" not in rp        # 源图由 Worker 用生图结果填充
    tab.repaint_check.setChecked(False)


def test_generate_without_repaint_keeps_single_stage(tab, monkeypatch, repaint_config):
    monkeypatch.setattr(gpt_image2_tab, "get_api_config", lambda api_type=None: {"api_key": "k", "base_url": "u"})
    tab.prompt_edit.setPlainText("一只猫")
    tab.repaint_check.setChecked(False)

    tab.generate()

    assert _FakeWorker.last_backend is generate_image_aigc2d_gpt
    assert _FakeWorker.last_repaint_params is None


def test_repaint_mode_wires_repaint_backend_with_dragged_images(tab, monkeypatch, tmp_path, repaint_config):
    monkeypatch.setattr(gpt_image2_tab, "get_api_config", lambda api_type=None: {"api_key": "k", "base_url": "u"})
    source = tmp_path / "product.png"
    source.write_bytes(TINY_PNG)
    tab._add_paths([str(source)])
    tab.mode_combo.setCurrentText(MODE_REPAINT)

    tab.generate()

    assert _FakeWorker.last_backend is generate_image_repaint
    params = _FakeWorker.last_params
    assert params["source_paths"] == [str(source)]
    assert params["model"] == "gemini-3-pro-image-preview"


def test_repaint_mode_without_images_warns_and_does_not_call_backend(tab, monkeypatch, repaint_config):
    warnings = []
    monkeypatch.setattr(gpt_image2_tab.QMessageBox, "warning", lambda *args, **kwargs: warnings.append(args))
    tab.mode_combo.setCurrentText(MODE_REPAINT)
    _FakeWorker.last_backend = None

    tab.generate()

    assert warnings, "应当提示重绘模式需要图片"
    assert _FakeWorker.last_backend is None


def test_repaint_results_button_uses_result_list(tab, monkeypatch, tmp_path, repaint_config):
    monkeypatch.setattr(gpt_image2_tab, "get_api_config", lambda api_type=None: {"api_key": "k", "base_url": "u"})
    source = tmp_path / "result.png"
    source.write_bytes(TINY_PNG)
    item = QListWidgetItem(str(source))
    item.setData(Qt.ItemDataRole.UserRole, str(source))
    tab.result_list.addItem(item)

    tab.repaint_selected_results()

    assert _FakeWorker.last_backend is generate_image_repaint
    assert _FakeWorker.last_params["source_paths"] == [str(source)]


def test_repaint_output_prefix_appended_to_aigc2d_requests(monkeypatch, tmp_path):
    """后端重绘：逐张调用 Gemini 通道，并且用独立段落式后缀（不再追加逗号脸部分支）。"""
    calls = []

    def _fake_generate(**kwargs):
        calls.append(kwargs)
        return [str(tmp_path / f"out_{len(calls)}.png")]

    monkeypatch.setattr(api_backend, "generate_image_aigc2d", _fake_generate)
    source_a = tmp_path / "a.png"
    source_b = tmp_path / "b.png"
    source_a.write_bytes(TINY_PNG)
    source_b.write_bytes(TINY_PNG)

    saved = api_backend.generate_image_repaint(
        source_paths=[str(source_a), str(source_b)],
        model="gemini-3-pro-image-preview",
        resolution="2K",
        repeat=1,
        save_sub_dir="tmp_repaint",
    )

    assert len(saved) == 2
    assert len(calls) == 2
    for call in calls:
        assert call["model"] == "gemini-3-pro-image-preview"
        assert call["resolution"] == "2K"
        assert call["face_quality_boost"] is False
        assert call["image_paths"] and len(call["image_paths"]) == 1
        assert "PRESERVE EVERYTHING THE CHARACTER IS WEARING OR CARRYING" in call["prompt"] or \
            "You are a CONSERVATIVE restoration model" in call["prompt"]
    assert calls[0]["file_prefix"].endswith("01")
    assert calls[1]["file_prefix"].endswith("02")


def test_repaint_backend_skips_missing_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(api_backend, "generate_image_aigc2d", lambda **kwargs: [])
    assert api_backend.generate_image_repaint(source_paths=[str(tmp_path / "nope.png")]) == []


def test_repaint_repeat_multiplies_requests(monkeypatch, tmp_path):
    calls = []

    def _fake_generate(**kwargs):
        calls.append(kwargs)
        return [str(tmp_path / f"r{len(calls)}.png")]

    monkeypatch.setattr(api_backend, "generate_image_aigc2d", _fake_generate)
    source = tmp_path / "a.png"
    source.write_bytes(TINY_PNG)

    saved = api_backend.generate_image_repaint(source_paths=[str(source)], repeat=3, save_sub_dir="tmp_repaint")

    assert len(saved) == 3
    assert len(calls) == 3


def test_repaint_mode_fills_firmware_into_prompt_box_and_restores_on_leave(tab):
    """切到重绘模式要把固件正文填进提示词框：用户得看得到实际在用的重绘提示词。

    否则用户自己写一句描述就发出去，固件里的"修手 / 保住蕾丝结构"全没了，
    自然体现不出修复效果。
    """
    mine = "一只猫坐在窗台上"
    tab.mode_combo.setCurrentText(MODE_GENERATE)
    tab.prompt_edit.setPlainText(mine)

    tab.mode_combo.setCurrentText(MODE_REPAINT)
    text = tab.prompt_edit.toPlainText()
    assert len(text) > 1500, "重绘模式下提示词框应填入固件正文"
    # 固件主 prompt：默认已换成画风中性的保守修复版（旧版是 PRESERVE EVERYTHING … 的 cel 固件）
    assert ("You are a CONSERVATIVE restoration model" in text
            or "PRESERVE EVERYTHING THE CHARACTER IS WEARING OR CARRYING" in text)
    assert "do not change the image content" in text or "NON-NEGOTIABLE COMPLETENESS CHECK" in text
    assert text != mine
    # 这份内容会被当作覆盖提示词发出去（等价于固件原文）
    params = tab.build_repaint_request(["x.png"])
    assert params.get("prompt", "").startswith("You are a CONSERVATIVE restoration model") or \
        params.get("prompt", "").startswith("Repaint this image")

    # 离开重绘模式应还原原来的文字，避免固件串到生图框
    tab.mode_combo.setCurrentText(MODE_GENERATE)
    assert tab.prompt_edit.toPlainText() == mine

    # 再进重绘模式：重新填入干净的固件（不带上次的临时修改）
    tab.mode_combo.setCurrentText(MODE_REPAINT)
    refilled = tab.prompt_edit.toPlainText()
    assert "You are a CONSERVATIVE restoration model" in refilled or "PRESERVE EVERYTHING" in refilled
    tab.mode_combo.setCurrentText(MODE_GENERATE)


def test_repaint_mode_allows_empty_prompt_meaning_use_firmware(tab, monkeypatch, tmp_path, repaint_config):
    """清空提示词框时仍能重绘（后端回退用固件原文）。"""
    monkeypatch.setattr(gpt_image2_tab, "get_api_config", lambda api_type=None: {"api_key": "k", "base_url": "u"})
    source = tmp_path / "product.png"
    source.write_bytes(TINY_PNG)
    tab._add_paths([str(source)])
    tab.mode_combo.setCurrentText(MODE_REPAINT)
    tab.prompt_edit.setPlainText("")          # 清空

    tab.generate()

    assert _FakeWorker.last_backend is generate_image_repaint
    assert "prompt" not in _FakeWorker.last_params   # 不传 prompt → 后端用固件
    tab.mode_combo.setCurrentText(MODE_GENERATE)


def test_repaint_switch_is_visible_without_expanding(tab):
    """功能开关（勾选框）必须常驻可见；折叠的只能是参数区。"""
    assert tab.repaint_check.isHidden() is False       # 不用展开就能看到并勾选
    assert tab.repaint_toggle_btn.isHidden() is False
    assert tab.repaint_panel.isHidden() is True        # 参数默认收起，不占高度
    assert tab.prompt_edit.minimumHeight() >= 140      # 提示词框有足够的最小高度

    tab.repaint_toggle_btn.setChecked(True)
    assert tab.repaint_panel.isHidden() is False
    tab.repaint_toggle_btn.setChecked(False)
    assert tab.repaint_panel.isHidden() is True


def test_repaint_notice_only_shows_when_action_needed(tab, monkeypatch):
    """正常情况下不显示任何状态复述；只有缺 key / 强制比例这类才出提醒。

    重绘用的是 `prompts/gpt-image-optimize/config.json` 里配的节点（默认 aigc2d），
    这里自己给一个 key，免得测试结果取决于开发机 .env 里有没有真密钥。
    """
    from utils.gpt_image_optimize import ASPECT_RATIO_AUTO

    monkeypatch.setenv("IMAGE_MAKER_AIGC2D_API_KEY", "sk-TEST-000000000000000000000000000")
    tab._refresh_repaint_hint()   # 构造时还没这个 key，重算一次提示

    assert tab.repaint_notice.isHidden() is True
    assert tab.repaint_notice.text() == ""

    tab.repaint_aspect_combo.setCurrentIndex(tab.repaint_aspect_combo.findData("16:9"))
    assert "被强制为 16:9" in tab.repaint_notice.text()
    assert tab.repaint_notice.isHidden() is False

    tab.repaint_aspect_combo.setCurrentIndex(tab.repaint_aspect_combo.findData(ASPECT_RATIO_AUTO))
    assert tab.repaint_notice.text() == ""
    assert tab.repaint_notice.isHidden() is True


def test_repaint_checkbox_enabled_in_generate_and_edit_mode(tab):
    """勾选框在生图/编辑模式都可用；只有重绘模式它没有意义（恒定走重绘）→ 置灰但仍可见。"""
    assert tab.mode_combo.currentText() == MODE_GENERATE
    assert tab.repaint_check.isEnabled() is True

    tab.mode_combo.setCurrentText(MODE_EDIT)
    assert tab.repaint_check.isEnabled() is True, "编辑模式也要能挂重绘（gpt-image edits 分辨率同样低）"
    assert tab.repaint_check.isHidden() is False
    tab.repaint_check.setChecked(False)
    assert tab.repaint_enabled() is False, "未勾选时不重绘"
    tab.repaint_check.setChecked(True)
    assert tab.repaint_enabled() is True, "编辑模式勾选后应链式重绘"
    tab.repaint_check.setChecked(False)

    tab.mode_combo.setCurrentText(MODE_REPAINT)
    assert tab.repaint_check.isEnabled() is False, "重绘模式下勾选框无意义，应灰色不可用"
    assert tab.repaint_check.isHidden() is False, "灰掉但不能消失"
    assert tab.repaint_enabled() is True, "重绘模式恒定走重绘"

    tab.mode_combo.setCurrentText(MODE_GENERATE)


def test_edit_mode_with_repaint_chains_repaint(tab, monkeypatch, tmp_path, repaint_config):
    """编辑 mode + 勾选重绘：先走 gpt-image edits，再把产物送去重绘。"""
    monkeypatch.setattr(gpt_image2_tab, "get_api_config", lambda api_type=None: {"api_key": "k", "base_url": "u"})
    source = tmp_path / "src.png"
    source.write_bytes(TINY_PNG)
    tab._add_paths([str(source)])
    tab.mode_combo.setCurrentText(MODE_EDIT)
    tab.prompt_edit.setPlainText("把背景换成日落")
    tab.repaint_check.setChecked(True)

    tab.generate()

    assert _FakeWorker.last_backend is generate_image_aigc2d_gpt
    assert _FakeWorker.last_params["mode"] == "edit"
    rp = _FakeWorker.last_repaint_params
    assert rp is not None, "编辑模式勾选后也应进入重绘阶段"
    assert rp["model"] == "gemini-3-pro-image-preview"
    tab.repaint_check.setChecked(False)
    tab.mode_combo.setCurrentText(MODE_GENERATE)


def test_repaint_options_always_usable(tab):
    """重绘参数在三种模式下都可调（编辑模式也会用到）。"""
    for mode in MODE_CHOICES:
        tab.mode_combo.setCurrentText(mode)
        for widget in (tab.repaint_model_combo, tab.repaint_resolution_combo,
                       tab.repaint_aspect_combo, tab.repaint_repeat_spin):
            assert widget.isEnabled() is True, f"{mode} 模式下参数应可用"
    tab.mode_combo.setCurrentText(MODE_GENERATE)


def test_reference_grid_is_compact_when_empty(tab, tmp_path):
    """空参考图时不占缩略图高度（否则把提示词框挤到最小高度）。"""
    empty_max = tab.image_grid.maximumHeight()
    assert empty_max < 100, f"空网格仍占 {empty_max}px"

    from PyQt6.QtGui import QImage
    image_path = tmp_path / "ref.png"
    QImage(64, 64, QImage.Format.Format_RGB32).save(str(image_path))
    tab._add_paths([str(image_path)])
    assert tab.image_grid.maximumHeight() > 100, "有缩略图后网格应恢复高度"

    tab.clear_images()
    assert tab.image_grid.maximumHeight() == empty_max, "清空后应回到紧凑高度"


def test_repaint_notice_is_height_limited(tab):
    """提醒文字过长会在窄宽度下换行把面板撑高，这里锁住高度上限。"""
    assert tab.repaint_notice.maximumHeight() <= 64


def test_aigc2d_channel_appends_prompt_suffix_and_can_skip_face_boost(monkeypatch, tmp_path):
    """prompt_suffix 以独立段落追加；face_quality_boost=False 时不追加逗号分支。"""
    captured = {}

    class _Resp:
        status_code = 200
        encoding = "utf-8"

        def json(self):
            return {"candidates": [{"content": {"parts": [{"inlineData": {"data": "", "mimeType": "image/png"}}]}}]}

        @property
        def text(self):
            return "{}"

        def raise_for_status(self):
            return None

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["payload"] = json
        return _Resp()

    monkeypatch.setattr(api_backend.requests, "post", _fake_post)
    monkeypatch.setattr(api_backend, "get_api_config", lambda api_type=None, config_path=None: {
        "base_url": "https://new.aigc2d.com/v1beta/models/", "api_key": "k", "timeout": 30, "max_retries": 0,
    })

    api_backend.generate_image_aigc2d(
        prompt="MAIN-PROMPT",
        model="gemini-3-pro-image-preview",
        api_type="aigc2d",
        save_sub_dir="tmp_repaint",
        prompt_suffix="SUFFIX-PART",
        face_quality_boost=False,
    )

    text = captured["payload"]["contents"][0]["parts"][0]["text"]
    assert text.startswith("MAIN-PROMPT")
    assert "SUFFIX-PART" in text
    assert "\n\nSUFFIX-PART" in text
    assert "detailed face" not in text


def test_aigc2d_channel_downloads_file_data_image(monkeypatch, tmp_path):
    """Gemini 3 偶尔以 fileData.fileUri 返回图片，不能把有效响应误判为无产物。"""
    image_bytes = b"\xff\xd8" + (b"x" * 2048)

    class _PostResp:
        status_code = 200
        encoding = "utf-8"

        def json(self):
            return {"candidates": [{"content": {"parts": [{"fileData": {
                "fileUri": "https://assets.example/generated-image",
                "mimeType": "image/jpeg",
            }}]}}]}

        @property
        def text(self):
            return "{}"

        def raise_for_status(self):
            return None

    class _GetResp:
        content = image_bytes

        def raise_for_status(self):
            return None

    monkeypatch.setattr(api_backend.requests, "post", lambda *args, **kwargs: _PostResp())
    monkeypatch.setattr(api_backend.requests, "get", lambda *args, **kwargs: _GetResp())
    monkeypatch.setattr(api_backend, "get_api_config", lambda api_type=None, config_path=None: {
        "base_url": "https://new.aigc2d.com/v1beta/models/", "api_key": "k",
        "timeout": 30, "max_retries": 0,
    })
    monkeypatch.chdir(tmp_path)

    saved = api_backend.generate_image_aigc2d(
        prompt="P", model="gemini-3-pro-image-preview", api_type="aigc2d",
        save_sub_dir="tmp_repaint", face_quality_boost=False,
    )

    assert len(saved) == 1
    assert saved[0].endswith(".jpg")
    assert Path(saved[0]).read_bytes() == image_bytes


# ---------------- 宽高比：auto = 跟随输入图（不能写死比例） ----------------
def _capture_image_config(monkeypatch, **kwargs):
    captured = {}

    class _Resp:
        status_code = 200
        encoding = "utf-8"

        def json(self):
            return {"candidates": [{"content": {"parts": [{"inlineData": {"data": "", "mimeType": "image/png"}}]}}]}

        @property
        def text(self):
            return "{}"

        def raise_for_status(self):
            return None

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["payload"] = json
        return _Resp()

    monkeypatch.setattr(api_backend.requests, "post", _fake_post)
    monkeypatch.setattr(api_backend, "get_api_config", lambda api_type=None, config_path=None: {
        "base_url": "https://new.aigc2d.com/v1beta/models/", "api_key": "k", "timeout": 30, "max_retries": 0,
    })
    api_backend.generate_image_aigc2d(prompt="P", model="m", api_type="aigc2d",
                                      save_sub_dir="tmp_repaint", face_quality_boost=False, **kwargs)
    return captured["payload"]["generationConfig"]["imageConfig"]


def test_auto_aspect_ratio_omits_field_so_output_follows_input(monkeypatch):
    """auto 时必须完全不出现 aspectRatio —— 官方默认行为就是匹配输入图尺寸。"""
    for value in ("auto", "", None, "AUTO", "跟随源图"):
        image_config = _capture_image_config(monkeypatch, aspect_ratio=value)
        assert "aspectRatio" not in image_config, f"aspect_ratio={value!r} 不该下发 aspectRatio"
        assert image_config["imageSize"] == "1K"


def test_explicit_aspect_ratio_is_sent(monkeypatch):
    image_config = _capture_image_config(monkeypatch, aspect_ratio="16:9", resolution="2K")
    assert image_config == {"aspectRatio": "16:9", "imageSize": "2K"}


def test_repaint_default_aspect_ratio_is_auto(repaint_config):
    from utils import gpt_image_optimize

    assert gpt_image_optimize.load_config().get("aspect_ratio") == gpt_image_optimize.ASPECT_RATIO_AUTO
    assert gpt_image_optimize.resolve_aspect_ratio({}) == gpt_image_optimize.ASPECT_RATIO_AUTO
    assert gpt_image_optimize.resolve_aspect_ratio({"aspect_ratio": "16:9"}) == "16:9"


def test_repaint_backend_passes_auto_aspect_by_default(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(api_backend, "generate_image_aigc2d",
                        lambda **kwargs: calls.append(kwargs) or [str(tmp_path / "o.png")])
    source = tmp_path / "in.png"
    source.write_bytes(TINY_PNG)

    api_backend.generate_image_repaint(source_paths=[str(source)], save_sub_dir="tmp_repaint")

    assert calls[0]["aspect_ratio"] == api_backend_auto_aspect()


def api_backend_auto_aspect():
    from utils.gpt_image_optimize import ASPECT_RATIO_AUTO
    return ASPECT_RATIO_AUTO


def test_image_aspect_ratio_label_reads_pixels(tmp_path):
    from PIL import Image

    from utils.gpt_image_optimize import image_aspect_ratio_label

    wide = tmp_path / "wide.png"
    Image.new("RGB", (1600, 900)).save(wide)
    assert image_aspect_ratio_label(str(wide)) == "16:9"
    square = tmp_path / "square.png"
    Image.new("RGB", (512, 512)).save(square)
    assert image_aspect_ratio_label(str(square)) == "1:1"
    assert image_aspect_ratio_label(str(tmp_path / "nope.png")) == ""


# ---------------- 画风选择：prompt_gpt + 参考图顺序 / 角色分工 ----------------

def test_style_combo_lists_styles_with_default_first(tab):
    from modules.image_generation.gpt_image2_tab import STYLE_NONE
    assert tab.style_combo.count() >= 2
    assert tab.style_combo.itemText(0) == STYLE_NONE
    assert tab.style_combo.findText("tid") > 0


def test_style_with_content_image_adds_role_block_and_orders_style_ref_last(tab, tmp_path):
    """选了画风 + 用户自己的图：内容图在前、画风参考图最后，并在提示词里写明分工。"""
    user_img = tmp_path / "user.png"
    user_img.write_bytes(TINY_PNG)
    tab.image_grid.add_paths([str(user_img)])
    tab.style_combo.setCurrentText("tid")
    tab.prompt_edit.setPlainText("a girl standing in shallow water")
    _backend, params = tab.build_request()
    prompt = params["prompt"]
    assert "IMAGE ROLES" in prompt
    assert "Palette:" in prompt
    assert "a girl standing in shallow water" in prompt
    assert len(params["image_paths"]) == 2
    assert params["image_paths"][0] == str(user_img)
    assert params["image_paths"][-1].endswith("tid.png")


def test_style_only_no_content_image_skips_role_block(tab):
    tab.style_combo.setCurrentText("tid")
    tab.prompt_edit.setPlainText("a girl standing in shallow water")
    _backend, params = tab.build_request()
    assert "IMAGE ROLES" not in params["prompt"]
    assert "Palette:" in params["prompt"]
    assert params["image_paths"] and params["image_paths"][-1].endswith("tid.png")


def test_style_none_keeps_prompt_and_images_as_is(tab, tmp_path):
    from modules.image_generation.gpt_image2_tab import STYLE_NONE
    user_img = tmp_path / "user2.png"
    user_img.write_bytes(TINY_PNG)
    tab.image_grid.add_paths([str(user_img)])
    tab.style_combo.setCurrentText(STYLE_NONE)
    tab.prompt_edit.setPlainText("plain prompt text")
    _backend, params = tab.build_request()
    assert params["prompt"] == "plain prompt text"
    assert params["image_paths"] == [str(user_img)]


def test_style_info_line_renders(tab):
    tab.style_combo.setCurrentText("noir-art-style")
    assert "字符" in tab.style_info_label.text()


def test_styles_all_have_gpt_prompt_except_placeholder():
    """除了「默认(无附加)」占位项，所有画风都应有 gpt-image 专用短版 prompt_gpt。"""
    import json
    from modules.image_generation.gpt_image2_tab import CONFIG_STYLES_FILE, STYLE_NONE
    data = json.load(open(CONFIG_STYLES_FILE, encoding="utf-8"))
    missing = [n for n, e in data.items()
               if isinstance(e, dict) and n != STYLE_NONE and not str(e.get("prompt_gpt") or "").strip()]
    assert missing == [], f"这些画风缺 prompt_gpt: {missing}"


def test_style_only_adds_scene_from_text_clause(tab):
    """只挂画风参考图时，必须声明场景/道具/构图来自文字（防参考图场景泄漏）。"""
    from utils.styles import STYLE_REF_SCENE_FROM_TEXT
    tab.style_combo.setCurrentText("tid")
    tab.prompt_edit.setPlainText("a girl standing in shallow water at dusk")
    _backend, params = tab.build_request()
    assert STYLE_REF_SCENE_FROM_TEXT in params["prompt"]
    assert "a girl standing in shallow water at dusk" in params["prompt"]




def test_style_choice_is_remembered(tab, tmp_path, monkeypatch):
    """画风选择要随 Tab 默认值持久化，并在下次构造时恢复。"""
    tab.style_combo.setCurrentText("tid")
    tab.save_defaults()
    fresh = gpt_image2_tab.GptImage2Widget()
    assert fresh.style_combo.currentText() == "tid"




def test_pick_size_follows_input_orientation(tmp_path):
    """宽图不能输出成竖图：尺寸按输入图实际比例挑。"""
    import cv2
    import numpy as np
    from modules.others.api_backend import pick_gpt_image2_size_for_images

    def _img(name, w, h):
        p = tmp_path / name
        cv2.imwrite(str(p), np.full((h, w, 3), 200, np.uint8))
        return str(p)

    assert pick_gpt_image2_size_for_images([_img("wide.png", 1600, 900)]) == "1536x1024"
    assert pick_gpt_image2_size_for_images([_img("tall.png", 900, 1600)]) == "1024x1536"
    assert pick_gpt_image2_size_for_images([_img("sq.png", 1000, 1000)]) == "1024x1024"
    assert pick_gpt_image2_size_for_images([], fallback="1024x1536") == "1024x1536"


def test_size_combo_has_follow_input_option(tab):
    """尺寸下拉要有「自动（跟随参考图比例）」这一项。"""
    values = [tab.size_combo.itemData(i) for i in range(tab.size_combo.count())]
    assert tab.SIZE_FOLLOW_INPUT in values
    tab.size_combo.setCurrentIndex(tab.size_combo.findData(tab.SIZE_FOLLOW_INPUT))
    assert tab.size_combo.currentData() == tab.SIZE_FOLLOW_INPUT


def test_resolve_request_size_uses_reference_orientation(tab, tmp_path):
    import cv2
    import numpy as np
    wide = tmp_path / "wide_ref.png"
    cv2.imwrite(str(wide), np.full((600, 1200, 3), 210, np.uint8))
    tab.size_combo.setCurrentIndex(tab.size_combo.findData(tab.SIZE_FOLLOW_INPUT))
    grid = tab.image_grid
    for method in ("set_paths", "add_paths", "add_images", "set_images"):
        if hasattr(grid, method):
            getattr(grid, method)([str(wide)])
            break
    assert tab.resolve_request_size() == "1536x1024"


# ---------------- 后处理流水线（结构线叠加 / 局部重绘）勾选框 ----------------

def test_post_process_switches_visible_and_default_off(tab):
    """两道后处理工序各有独立勾选框，默认关闭；参数放折叠区里。"""
    assert tab.structure_check.parent() is not None
    assert tab.local_repaint_check.parent() is not None
    assert tab.structure_check.isChecked() is False
    assert tab.local_repaint_check.isChecked() is False
    assert tab.post_panel.isVisible() is False
    assert tab.post_toggle_btn.isChecked() is False


def test_post_pipeline_steps_follow_checkboxes(tab):
    """勾选框与参数要真正反映到流水线配置上（未勾选的工序不会执行）。"""
    steps = tab.post_pipeline_steps()
    assert steps["structure"]["enabled"] is False
    assert steps["local"]["enabled"] is False
    tab.structure_check.setChecked(True)
    tab.structure_strength_spin.setValue(0.35)
    tab.local_repaint_check.setChecked(True)
    tab.local_region_combo.setCurrentIndex(tab.local_region_combo.findData("face"))
    tab.local_feather_spin.setValue(64)
    steps = tab.post_pipeline_steps()
    assert steps["structure"]["enabled"] is True
    assert abs(steps["structure"]["strength"] - 0.35) < 1e-6
    assert steps["local"]["enabled"] is True
    assert steps["local"]["region"] == "face"
    assert steps["local"]["feather"] == 64


def test_post_process_defaults_persist(tab, tmp_path):
    """后处理勾选与参数要按站点持久化。"""
    tab.structure_check.setChecked(True)
    tab.local_repaint_check.setChecked(True)
    tab.local_region_combo.setCurrentIndex(tab.local_region_combo.findData("skirt"))
    tab.save_defaults()
    data = json.load(open(gpt_image2_tab.CONFIG_IMAGE_FILE, encoding="utf-8"))
    node = data[gpt_image2_tab.CONFIG_NODE]["post_process"]
    assert node["structure_enabled"] is True
    assert node["local_enabled"] is True
    assert node["local_region"] == "skirt"
    assert node["local_feather"] == tab.local_feather_spin.value()


def test_dead_dual_reference_switch_is_removed(tab, tmp_path):
    """「重绘用双参考（源图+线锚图）」是死开关，2026-09-24 已删。

    它以前只读写配置、从不进 `post_pipeline_steps()`（该 Tab 的 repaint 步骤恒 disabled），
    所以勾不勾都没作用；线锚图只留给无头 CLI 的 `--repaint-ref line_anchor|both`
    （§三十一 实测线锚图会把画面塌成白底线稿 / 铺满「碎玻璃」纹理）。
    """
    assert not hasattr(tab, "post_dual_check")
    steps = tab.post_pipeline_steps()
    # 该 Tab 的 repaint 工序恒不参与流水线（重绘是独立「重绘模式」），那个死键在这里毫无作用
    assert steps["repaint"]["enabled"] is False
    tab.save_defaults()
    node = json.load(open(gpt_image2_tab.CONFIG_IMAGE_FILE, encoding="utf-8"))[gpt_image2_tab.CONFIG_NODE]
    assert "post_dual" not in node


def test_old_config_with_dead_dual_key_still_loads(qapp, tmp_path, monkeypatch):
    """老配置里残留的 `post_dual` 键要能被安静忽略（升级后不能因为死键报错）。"""
    payload = json.loads(json.dumps(TAB_CONFIG, ensure_ascii=False))
    payload[gpt_image2_tab.CONFIG_NODE]["post_dual"] = True
    config_path = _write_tab_config(tmp_path, payload)
    monkeypatch.setattr(gpt_image2_tab, "CONFIG_IMAGE_FILE", str(config_path))
    monkeypatch.setattr(gpt_image2_tab, "GptImage2Worker", _FakeWorker)
    widget = GptImage2Widget()
    assert not hasattr(widget, "post_dual_check")
    assert widget.current_site() == SITE_AIGC2D
