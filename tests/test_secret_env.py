# -*- coding: utf-8 -*-
"""顶层「文本分析 API」/「NSFW 文本 API」密钥的环境变量解析。

背景（2026-09-21 实际踩坑）：把图片节点的 key 换进 .env 后，图片分析 Step 1 依旧 401。
原因是 Step 1 走的是 conf/config.json **顶层**的 `base_url / api_key / model`，
而这把 key 当时没有任何环境变量通道 —— 改了 .env 也不会生效。

这组用例锁住两件事：
1. 顶层文本 / NSFW 密钥和 `apis.*` 节点同一条优先级：环境变量 > .env > conf/config.json；
2. `apply_secret_env_overrides` 只在内存里改，绝不把解析结果回写盘上的配置。
"""

from __future__ import annotations

import json
import os
import types
from pathlib import Path

import pytest

# 导入 app.py 前关掉 onnxruntime 预热（本机 onnxruntime 预热会 access violation）并走无头 Qt，
# 与 tests/test_gui_entry.py / tests/test_pyqt6_smoke.py 同一套开关。
os.environ.setdefault("IMAGE_MAKER_SKIP_ONNXRUNTIME_PRELOAD", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.others import api_backend

TEXT_ENVS = api_backend.TEXT_API_KEY_ENV_NAMES
NSFW_ENVS = api_backend.NSFW_API_KEY_ENV_NAMES


@pytest.fixture(autouse=True)
def _clean_secret_envs(monkeypatch):
    """本机 .env 会在导入 api_backend 时装载这些变量，用例里先清干净再自己设。"""
    for name in TEXT_ENVS + NSFW_ENVS:
        monkeypatch.delenv(name, raising=False)
    yield


# ---------------------------------------------------------------- 文本 key

def test_text_key_env_overrides_config(monkeypatch):
    monkeypatch.setenv("IMAGE_MAKER_TEXT_API_KEY", "sk-FROM-ENV")
    cfg = {"api_key": "sk-FROM-CONFIG"}

    assert api_backend.resolve_text_api_key(cfg) == "sk-FROM-ENV"
    assert api_backend.text_api_key_source(cfg) == "env:IMAGE_MAKER_TEXT_API_KEY"


def test_text_key_falls_back_to_config():
    cfg = {"api_key": "sk-FROM-CONFIG"}

    assert api_backend.resolve_text_api_key(cfg) == "sk-FROM-CONFIG"
    assert api_backend.resolve_text_api_key({}) == ""
    assert api_backend.text_api_key_source(cfg) == "config"
    assert api_backend.text_api_key_source({}) == "none"


def test_text_key_accepts_generic_env_name(monkeypatch):
    """没写 IMAGE_MAKER_ 前缀的通用名也认（本地临时覆盖用）。"""
    monkeypatch.setenv("TEXT_API_KEY", "sk-GENERIC")

    assert api_backend.resolve_text_api_key({"api_key": "sk-CONFIG"}) == "sk-GENERIC"
    assert api_backend.text_api_key_source({}) == "env:TEXT_API_KEY"


# ---------------------------------------------------------------- NSFW key

def test_nsfw_key_env_overrides_config(monkeypatch):
    monkeypatch.setenv("IMAGE_MAKER_NSFW_API_KEY", "sk-NSFW-ENV")
    cfg = {"nsfw_api_key": "sk-NSFW-CONFIG"}

    assert api_backend.resolve_nsfw_api_key(cfg) == "sk-NSFW-ENV"
    assert api_backend.nsfw_api_key_source(cfg) == "env:IMAGE_MAKER_NSFW_API_KEY"


def test_nsfw_key_falls_back_to_config():
    cfg = {"nsfw_api_key": "sk-NSFW-CONFIG"}

    assert api_backend.resolve_nsfw_api_key(cfg) == "sk-NSFW-CONFIG"
    assert api_backend.nsfw_api_key_source(cfg) == "config"
    assert api_backend.nsfw_api_key_source({}) == "none"


# ------------------------------------------------- 只改内存、不落盘

def test_apply_secret_env_overrides_rewrites_keys_and_marks_source(monkeypatch):
    monkeypatch.setenv("IMAGE_MAKER_TEXT_API_KEY", "sk-TEXT-ENV")
    monkeypatch.setenv("IMAGE_MAKER_NSFW_API_KEY", "sk-NSFW-ENV")
    cfg = {"api_key": "", "nsfw_api_key": ""}

    api_backend.apply_secret_env_overrides(cfg)

    assert cfg["api_key"] == "sk-TEXT-ENV"
    assert cfg["nsfw_api_key"] == "sk-NSFW-ENV"
    assert cfg["_text_api_key_source"] == "env:IMAGE_MAKER_TEXT_API_KEY"
    assert cfg["_nsfw_api_key_source"] == "env:IMAGE_MAKER_NSFW_API_KEY"


def test_apply_secret_env_overrides_keeps_config_values_without_env():
    cfg = {"api_key": "sk-CONFIG", "nsfw_api_key": "sk-NSFW-CONFIG"}

    api_backend.apply_secret_env_overrides(cfg)

    assert cfg["api_key"] == "sk-CONFIG"
    assert cfg["nsfw_api_key"] == "sk-NSFW-CONFIG"
    assert cfg["_text_api_key_source"] == "config"


def test_apply_secret_env_overrides_never_leaks_into_disk(monkeypatch, tmp_path):
    """调用方若把同一份 dict 写回文件，也不能把密钥从环境变量漏进配置文件。"""
    monkeypatch.setenv("IMAGE_MAKER_TEXT_API_KEY", "sk-TEXT-ENV")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"api_key": "", "model": "gpt-5.6-luna"}), encoding="utf-8")
    before = config_path.read_bytes()

    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    api_backend.apply_secret_env_overrides(loaded)
    assert loaded["api_key"] == "sk-TEXT-ENV"          # 内存里生效
    assert config_path.read_bytes() == before          # 盘上没动


def test_apply_secret_env_overrides_works_on_a_copy():
    """analysis_pipeline 用的是 `dict(config)` 副本，必须保证调用方原 dict 不被污染。"""
    caller_cfg = {"api_key": ""}
    api_backend.apply_secret_env_overrides(dict(caller_cfg))

    assert caller_cfg == {"api_key": ""}


# --------------------------------------------------------- 具体接入点

def test_sd_workflow_text_config_reads_env_key(monkeypatch, tmp_path):
    from modules.image_generation import sd_workflow_core

    (tmp_path / "conf").mkdir()
    (tmp_path / "conf" / "config.json").write_text(
        json.dumps({
            "base_url": "https://new.aigc2d.com/v1",
            "api_key": "",
            "model": "gpt-5.6-luna",
            "nsfw_base_url": "https://api.deepseek.com",
            "nsfw_api_key": "",
            "nsfw_model": "deepseek-v4-pro",
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(sd_workflow_core, "BASE_DIR", str(tmp_path))
    monkeypatch.setenv("IMAGE_MAKER_TEXT_API_KEY", "sk-TEXT-ENV")
    monkeypatch.setenv("IMAGE_MAKER_NSFW_API_KEY", "sk-NSFW-ENV")

    base_url, api_key, model = sd_workflow_core.load_text_api_config_from_file()
    assert (base_url, api_key, model) == ("https://new.aigc2d.com/v1", "sk-TEXT-ENV", "gpt-5.6-luna")

    nsfw_base, nsfw_key, nsfw_model = sd_workflow_core.load_text_api_config_from_file(use_nsfw=True)
    assert (nsfw_base, nsfw_key, nsfw_model) == ("https://api.deepseek.com", "sk-NSFW-ENV", "deepseek-v4-pro")


def test_sd_workflow_text_config_still_reads_config_file(monkeypatch, tmp_path):
    from modules.image_generation import sd_workflow_core

    (tmp_path / "conf").mkdir()
    (tmp_path / "conf" / "config.json").write_text(
        json.dumps({"base_url": "https://example.com/v1", "api_key": "sk-CONFIG", "model": "m"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sd_workflow_core, "BASE_DIR", str(tmp_path))

    assert sd_workflow_core.load_text_api_config_from_file()[1] == "sk-CONFIG"


# ------------------------------------------------------ GUI 侧（可选依赖）

class _FakeLineEdit:
    def __init__(self, text=""):
        self._text = text
        self.placeholders = []
        self.tooltips = []

    def text(self):
        return self._text

    def setPlaceholderText(self, value):  # noqa: N802 - Qt 命名
        self.placeholders.append(value)

    def setToolTip(self, value):  # noqa: N802 - Qt 命名
        self.tooltips.append(value)


class _FakeCombo:
    def __init__(self, text=""):
        self._text = text

    def currentText(self):  # noqa: N802 - Qt 命名
        return self._text


def _app_window_stub():
    return types.SimpleNamespace(
        url_input=_FakeLineEdit("https://new.aigc2d.com/v1"),
        key_input=_FakeLineEdit(""),
        model_combo=_FakeCombo("gpt-5.6-luna"),
        nsfw_url_input=_FakeLineEdit("https://api.deepseek.com"),
        nsfw_key_input=_FakeLineEdit(""),
        nsfw_model_combo=_FakeCombo("deepseek-v4-pro"),
        img_url_input=_FakeLineEdit("https://new.aigc2d.com/v1beta/models/"),
        img_key_input=_FakeLineEdit(""),
        img_model_combo=_FakeCombo("gemini-3-pro-image-preview"),
        api_type_combo=_FakeCombo("aigc2d"),
    )


@pytest.fixture(scope="module")
def app_window_cls():
    pytest.importorskip("PyQt6")
    import app as app_module

    return app_module.AppWindow


def test_app_get_text_config_prefers_env_when_box_empty(monkeypatch, app_window_cls):
    """界面输入框留空（配置里也留空）时，分析仍然拿得到 .env 里的 key。"""
    monkeypatch.setenv("IMAGE_MAKER_TEXT_API_KEY", "sk-TEXT-ENV")
    monkeypatch.setenv("IMAGE_MAKER_NSFW_API_KEY", "sk-NSFW-ENV")
    stub = _app_window_stub()

    assert app_window_cls.get_text_config(stub)[1] == "sk-TEXT-ENV"
    assert app_window_cls.get_text_config(stub, True)[1] == "sk-NSFW-ENV"


def test_app_get_text_config_uses_box_value_without_env(app_window_cls):
    stub = _app_window_stub()
    stub.key_input = _FakeLineEdit("sk-TYPED-IN-GUI")

    assert app_window_cls.get_text_config(stub)[1] == "sk-TYPED-IN-GUI"


def test_app_key_hint_exposes_variable_name_only(monkeypatch, app_window_cls):
    monkeypatch.setenv("IMAGE_MAKER_TEXT_API_KEY", "sk-TEXT-ENV-SECRET")
    stub = _app_window_stub()

    app_window_cls._apply_text_key_hints(stub, {})

    assert "IMAGE_MAKER_TEXT_API_KEY" in stub.key_input.placeholders[0]
    assert all("sk-TEXT-ENV-SECRET" not in text for text in stub.key_input.placeholders)
    assert all("sk-TEXT-ENV-SECRET" not in text for text in stub.key_input.tooltips)


def test_app_key_hint_silent_without_env(app_window_cls):
    stub = _app_window_stub()

    app_window_cls._apply_text_key_hints(stub, {})

    assert stub.key_input.placeholders == []
    assert stub.key_input.tooltips == []


# ------------------------------------------- 图片节点 key（自动生图的空值误判）

def test_app_img_config_resolves_env_key_when_box_empty(monkeypatch, app_window_cls):
    """`apis.aigc2d.api_key` 留空（值在 .env）时，注入给分析 Tab 的生图配置仍拿得到 key。

    回归：不解析环境变量的话，single_analyzer 的自动生工会直接跳过并报
    「生图 API Key 不能为空，请检查【全局配置】」。
    """
    monkeypatch.setenv("IMAGE_MAKER_AIGC2D_API_KEY", "sk-IMG-ENV")
    stub = _app_window_stub()

    base_url, api_key, model, api_type = app_window_cls.get_img_config(stub)

    assert api_key == "sk-IMG-ENV"
    assert api_type == "aigc2d"
    assert base_url == "https://new.aigc2d.com/v1beta/models/"
    assert model == "gemini-3-pro-image-preview"


def test_app_img_config_keeps_box_key_without_env(app_window_cls):
    stub = _app_window_stub()
    stub.api_type_combo = _FakeCombo("TESTNODE")
    stub.img_key_input = _FakeLineEdit("sk-TYPED-IN-GUI")

    assert app_window_cls.get_img_config(stub)[1] == "sk-TYPED-IN-GUI"


def test_app_img_key_hint_exposes_variable_name_only(monkeypatch, app_window_cls):
    monkeypatch.setenv("IMAGE_MAKER_AIGC2D_API_KEY", "sk-IMG-ENV-SECRET")
    stub = _app_window_stub()

    app_window_cls._apply_img_key_hint(stub, {}, "aigc2d")

    assert "IMAGE_MAKER_AIGC2D_API_KEY" in stub.img_key_input.placeholders[0]
    assert all("sk-IMG-ENV-SECRET" not in text for text in stub.img_key_input.placeholders)
    assert all("sk-IMG-ENV-SECRET" not in text for text in stub.img_key_input.tooltips)


def test_img_key_hint_names_variable_when_nothing_configured(app_window_cls):
    """两边都没有 key 时，界面要说清"该用哪个环境变量"，而不是只留一个空密码框。"""
    stub = _app_window_stub()
    stub.api_type_combo = _FakeCombo("aigc-2d-gpt")

    app_window_cls._apply_img_key_hint(stub, {}, "aigc-2d-gpt")

    assert "IMAGE_MAKER_AIGC_2D_GPT_API_KEY" in stub.img_key_input.placeholders[0]


def test_img_key_hint_silent_when_node_has_own_key(app_window_cls):
    stub = _app_window_stub()
    stub.img_key_input = _FakeLineEdit("")

    app_window_cls._apply_img_key_hint(stub, {"api_key": "sk-IN-CONFIG"}, "aigc2d")

    assert stub.img_key_input.placeholders == [""]


def test_nodes_from_env_define_and_reuse_key(tmp_path, monkeypatch):
    """节点定义只写在环境变量里：IMAGE_MAKER_NODES + env_slug 复用同一把钥匙。"""
    import json as _json
    from modules.others import api_backend
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(_json.dumps({"current_api": "nowhere", "apis": {}}), encoding="utf-8")
    monkeypatch.setenv("IMAGE_MAKER_NODES", _json.dumps({
        "whatup": {"base_url": "https://api.whatai.cc/v1", "model": "nano-banana-2", "env_slug": "aigc2d"}}))
    monkeypatch.setenv("IMAGE_MAKER_AIGC2D_API_KEY", "sk-TEST-abcdef")
    data = api_backend.load_config(str(cfg_file))
    assert data["apis"]["whatup"]["base_url"] == "https://api.whatai.cc/v1"
    cfg = api_backend.get_api_config(str(cfg_file), "whatup")
    assert cfg["api_key"] == "sk-TEST-abcdef"
    assert cfg["_api_key_source"] == "env:IMAGE_MAKER_AIGC2D_API_KEY"


def test_per_node_env_naming_aligns_to_config_node(tmp_path, monkeypatch):
    """逐节点环境变量 `IMAGE_MAKER_AIGC_2D_GPT_BASE_URL` 要落到 app 查的 `aigc-2d-gpt` 节点上。"""
    import json as _json
    from modules.others import api_backend
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(_json.dumps({"apis": {}}), encoding="utf-8")
    monkeypatch.setenv("IMAGE_MAKER_AIGC_2D_GPT_BASE_URL", "https://new.aigc2d.com/v1")
    data = api_backend.load_config(str(cfg_file))
    assert "aigc-2d-gpt" in data["apis"]
    assert data["apis"]["aigc-2d-gpt"]["base_url"] == "https://new.aigc2d.com/v1"


def test_env_nodes_never_override_config_values(tmp_path, monkeypatch):
    """环境变量只补缺，不覆盖配置文件里已写好的值。"""
    import json as _json
    from modules.others import api_backend
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(_json.dumps({"apis": {"whatup": {"base_url": "https://from-config/"}}}),
                        encoding="utf-8")
    monkeypatch.setenv("IMAGE_MAKER_WHATUP_BASE_URL", "https://from-env/")
    data = api_backend.load_config(str(cfg_file))
    assert data["apis"]["whatup"]["base_url"] == "https://from-config/"


# ------------------------------- 设置页（全局配置）也要读环境变量
#
# 用户 2026-09-23 实际报错：分析完点「生图」时提示
# 「自动生图已跳过：生图 API Key 不能为空，请检查【全局配置】」。
# 根因：app.py 的设置页用 json.load 直读 conf/config.json，看不见 .env 里
# `IMAGE_MAKER_NODES` 定义的节点，`IMAGE_MAKER_CURRENT_API` 也不生效 ——
# 界面停在配置文件里剩下的 whatup 节点上，key 自然是空的。

@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


ENV_NODES = {
    "aigc2d": {"base_url": "https://new.aigc2d.com/v1beta/models/",
               "api_type": "aigc2d", "env_slug": "aigc2d",
               "model": "gemini-3-pro-image-preview", "resolution": "2K"},
    "aigc-2d-gpt": {"base_url": "https://new.aigc2d.com/v1", "api_type": "aigc-2d-gpt",
                    "env_slug": "aigc2d", "model": "gpt-image-2"},
}


def _write_min_config(tmp_path):
    """最小配置：只有一个 whatup 节点且 key 为空（模拟用户现场）。"""
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({
        "current_api": "whatup",
        "apis": {"whatup": {"base_url": "https://api.whatai.cc/v1", "api_key": "",
                            "model": "nano-banana-2"}},
        "base_url": "https://new.aigc2d.com/v1", "api_key": "", "model": "gpt-5.6-luna",
    }, ensure_ascii=False), encoding="utf-8")
    return cfg


def _build_app_window(app_module, cfg_path, monkeypatch):
    for name in ("CONFIG_FILE", "CONFIG_IMAGE_FILE"):
        monkeypatch.setattr(app_module, name, str(cfg_path))
    window = app_module.AppWindow()
    return window


@pytest.fixture
def env_only_deployment(tmp_path, monkeypatch):
    """模拟「节点 + 密钥都在 .env 里」的部署：config.json 里只有空 key 的 whatup。"""
    cfg = _write_min_config(tmp_path)
    monkeypatch.setenv("IMAGE_MAKER_NODES", json.dumps(ENV_NODES))
    monkeypatch.setenv("IMAGE_MAKER_CURRENT_API", "aigc2d")
    monkeypatch.setenv("IMAGE_MAKER_AIGC2D_API_KEY", "sk-IMG-ENV-SECRET")
    return cfg


def _app_module():
    pytest.importorskip("PyQt6")
    import app as app_module

    return app_module


def test_settings_page_reads_env_only_nodes(qapp, env_only_deployment, monkeypatch):
    """节点与密钥都只在环境变量里时：下拉能看到、当前节点跟着 CURRENT_API 走、key 解析得到。"""
    app_module = _app_module()
    window = _build_app_window(app_module, env_only_deployment, monkeypatch)
    try:
        items = [window.api_type_combo.itemText(i) for i in range(window.api_type_combo.count())]
        assert "aigc2d" in items and "aigc-2d-gpt" in items, "环境变量里定义的节点必须出现在下拉里"
        assert window.api_type_combo.currentText() == "aigc2d", "IMAGE_MAKER_CURRENT_API 要生效"
        assert window.img_url_input.text() == "https://new.aigc2d.com/v1beta/models/"
        assert window.img_model_combo.currentText() == "gemini-3-pro-image-preview"
        # 密钥绝不显示在界面：输入框留空，只提示变量名
        assert window.img_key_input.text() == ""
        assert "IMAGE_MAKER_AIGC2D_API_KEY" in window.img_key_input.placeholderText()
        assert "sk-IMG-ENV-SECRET" not in window.img_key_input.placeholderText()
        # 自动生图拿到的配置里必须有 key，否则会被误判成"没配 key"而跳过
        base_url, api_key, model, api_type = window.get_img_config()
        assert api_key == "sk-IMG-ENV-SECRET"
        assert (base_url, model, api_type) == ("https://new.aigc2d.com/v1beta/models/",
                                               "gemini-3-pro-image-preview", "aigc2d")
    finally:
        window.close()


def test_settings_page_resolves_shared_env_slug(qapp, env_only_deployment, monkeypatch):
    """切到共用 `env_slug` 的节点时也要拿得到 key（aigc-2d-gpt → IMAGE_MAKER_AIGC2D_API_KEY）。"""
    app_module = _app_module()
    window = _build_app_window(app_module, env_only_deployment, monkeypatch)
    try:
        window.api_type_combo.setCurrentText("aigc-2d-gpt")
        qapp.processEvents()
        assert window.img_url_input.text() == "https://new.aigc2d.com/v1"
        assert window.get_img_config()[1] == "sk-IMG-ENV-SECRET"
    finally:
        window.close()


def test_save_image_config_does_not_freeze_env_node(qapp, env_only_deployment, monkeypatch):
    """保存设置页不能把环境变量里的**节点定义**抄进配置文件。

    抄进去以后配置就压过 `.env`（`load_config` 是"配置优先、环境变量补缺"），
    用户改了 `.env` 却看不到变化（等于把环境变量变成一次性导入）；
    同时也要留一份界面记忆（超时/比例策略），别把整条节点丢掉。
    """
    app_module = _app_module()
    window = _build_app_window(app_module, env_only_deployment, monkeypatch)
    try:
        window.api_type_combo.setCurrentText("aigc-2d-gpt")
        qapp.processEvents()
        window.save_image_config(silent=True)
        data = json.loads(Path(str(env_only_deployment)).read_text(encoding="utf-8"))
        node = data["apis"]["aigc-2d-gpt"]
        for key in ("base_url", "api_key", "model", "env_slug", "api_type"):
            assert key not in node, f"{key} 由环境变量提供，不该写进配置文件"
        assert node["timeout"] == window.img_timeout_spin.value()      # 界面记忆照常保存
        assert "default_aspect_ratio" in node
        # 保存之后仍然解析得到 env 里的 key（节点定义没被冻结）
        assert window.get_img_config()[1] == "sk-IMG-ENV-SECRET"
    finally:
        window.close()


def test_settings_page_locks_env_owned_fields(qapp, env_only_deployment, monkeypatch):
    """环境变量提供的字段在界面上置灰只读（改了也不生效，就别让用户白改）。"""
    app_module = _app_module()
    window = _build_app_window(app_module, env_only_deployment, monkeypatch)
    try:
        assert window.img_url_input.text() == "https://new.aigc2d.com/v1beta/models/"
        assert window.img_url_input.isEnabled() is False
        assert "环境变量" in window.img_url_input.toolTip()
        assert window.img_model_combo.isEnabled() is False
        assert window.img_key_input.isEnabled() is False        # key 由 env 提供 → 置灰
        # 切到配置里真正存在的节点（whatup 没配 key）→ 恢复可编辑
        window.api_type_combo.setCurrentText("whatup")
        qapp.processEvents()
        assert window.img_url_input.isEnabled() is True
        assert window.img_key_input.isEnabled() is True
    finally:
        window.close()
