# -*- coding: utf-8 -*-
"""「转化二次元」= 图片分析页的独立动作：输入真人照片 → 图片 edit 接口 → 日系二次元插画。

用户 2026-10-24 要求：提示词为给定全文（落盘 `prompts/anime-convert/edit-prompt.md`，
逐字发送、不改写），通道跟随界面「生图通道」单选（Gemini / gpt-image）。

本文件钉住四件事：
1. 固件全文在盘且关键章节（【一、身份一致性】…【十五、需要避免的效果】）齐全；
2. 请求组装：比例锁输入照片、产物落 data/<日期>/anime-convert/、前缀固定；
3. 通道路由：gpt-image → aigc-2d-gpt 的 /images/edits（带图）；
   gemini → 全局生图 api_type（aigc2d generateContent / whatai 家族），且不追加英文面部后缀；
4. 界面接线：按钮常驻底部生成行、加载图片后才可用、无图/忙时给出明确提示。
"""

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from utils import anime_convert as ac

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_FILE = os.path.join(BASE, "prompts", "anime-convert", "edit-prompt.md")
GPT_SIZES = {"1024x1024", "1536x1024", "1024x1536"}


@pytest.fixture()
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def portrait_photo(tmp_path):
    """800x1200（2:3 竖图）的临时输入照片。"""
    from PIL import Image
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (800, 1200), color=(120, 90, 80)).save(str(path), format="JPEG")
    return str(path)


# ---------- 1. 固件全文 ----------

def test_prompt_file_exists_with_all_fifteen_sections():
    assert os.path.isfile(PROMPT_FILE)
    text = ac.anime_convert_prompt()
    assert text.startswith("请将我上传的真人照片作为唯一的人物参考")
    for i, title in enumerate(
            ["人物身份与外形一致性", "姿势、构图与镜头", "服装与配饰的精细绘制", "整体二次元绘画风格",
             "面部结构与皮肤表现", "眼睛形状与宽高比例", "虹膜、瞳孔与多层次高光", "睫毛与眼部线条",
             "头发整体结构与发丝分组", "头发的多层次高光与打光", "统一的面部、头发与环境光照",
             "色彩与整体氛围", "背景与梦幻装饰", "最终画面质量", "需要避免的效果"], start=1):
        numeral = "一二三四五六七八九十"[i - 1] if i <= 10 else "十" + "一二三四五"[i - 11]
        assert f"【{numeral}、{title}】" in text
    assert len(text) > 2000
    assert text.rstrip().endswith("准确保留真人参考照片中的人物身份、姿势、穿搭和辨识度。")


def test_prompt_is_sent_verbatim_without_template_holes():
    text = ac.anime_convert_prompt()
    assert "{" not in text and "}" not in text   # 固件全文：没有占位符，逐字发送


# ---------- 2. 请求组装 ----------

def test_normalize_channel_accepts_aliases():
    assert ac.normalize_anime_convert_channel("gpt-image") == "gpt-image"
    assert ac.normalize_anime_convert_channel("GPT") == "gpt-image"
    assert ac.normalize_anime_convert_channel("gemini") == "gemini"
    assert ac.normalize_anime_convert_channel("") == "gemini"


def test_build_request_gemini_locks_aspect_and_output_dir(portrait_photo):
    request = ac.build_anime_convert_request(
        portrait_photo, channel="gemini", model_name="gemini-3-pro-image-preview", api_type="aigc2d")
    assert request["channel"] == "gemini"
    assert request["aspect_ratio"] == "2:3"          # 800x1200 → 2:3
    assert request["save_sub_dir"] == ac.ANIME_CONVERT_SAVE_SUB_DIR == "anime-convert"
    assert request["file_prefix"] == "anime-convert"
    assert request["model"] == "gemini-3-pro-image-preview"
    assert request["api_type"] == "aigc2d"
    assert request["prompt"] == ac.anime_convert_prompt()
    assert "size" not in request                       # 尺寸只属于 gpt 通道


def test_build_request_gpt_picks_size_from_photo(portrait_photo):
    request = ac.build_anime_convert_request(portrait_photo, channel="gpt-image")
    assert request["channel"] == "gpt-image"
    assert request["size"] in GPT_SIZES
    assert request["size"] == "1024x1536"              # 竖图 → 竖档
    assert request["aspect_ratio"] == "2:3"


def test_build_request_requires_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ac.build_anime_convert_request(str(tmp_path / "missing.jpg"), channel="gemini")


# ---------- 3. 通道路由（发函数打桩，不发网络） ----------

def _fake_backend(monkeypatch, name, outputs):
    """把 modules.others.api_backend 里的发送函数替换成记录参数的桩。"""
    calls = []

    def _fake(**kwargs):
        calls.append(kwargs)
        return list(outputs)

    monkeypatch.setattr(f"modules.others.api_backend.{name}", _fake)
    return calls


def test_run_routes_gpt_channel_to_edits(monkeypatch, portrait_photo, tmp_path):
    out = tmp_path / "anime-convert_120000-abcdef.jpg"
    out.write_bytes(b"fake-jpeg-bytes")
    calls = _fake_backend(monkeypatch, "generate_image_aigc2d_gpt", [str(out)])
    saved = ac.run_anime_convert(portrait_photo, channel="gpt-image", log_callback=lambda m: None)
    assert saved == [str(out)]
    assert len(calls) == 1
    sent = calls[0]
    assert sent["image_paths"] == [os.path.abspath(portrait_photo)]   # 带图 → /images/edits
    assert sent["prompt"] == ac.anime_convert_prompt()
    assert sent["aspect_ratio"] == "2:3"
    assert sent["save_sub_dir"] == "anime-convert"
    assert sent["file_prefix"] == "anime-convert"
    record_path = str(out) + ".anime-convert.json"
    assert os.path.isfile(record_path)
    record = json.loads(open(record_path, encoding="utf-8").read())
    assert record["kind"] == "anime-convert" and record["channel"] == "gpt-image"
    assert record["prompt_chars"] == len(ac.anime_convert_prompt())
    assert os.path.isfile(record["source"]) and record["source_sha256"]


def test_run_routes_gemini_aigc2d_without_face_suffix(monkeypatch, portrait_photo, tmp_path):
    out = tmp_path / "anime-convert_120001-bcdef0.jpg"
    out.write_bytes(b"fake-jpeg-bytes")
    calls = _fake_backend(monkeypatch, "generate_image_aigc2d", [str(out)])
    saved = ac.run_anime_convert(portrait_photo, channel="gemini",
                                 model_name="gemini-3-pro-image-preview", api_type="aigc2d",
                                 log_callback=lambda m: None)
    assert saved == [str(out)]
    sent = calls[0]
    assert sent["model"] == "gemini-3-pro-image-preview"
    assert sent["api_type"] == "aigc2d"
    assert sent["face_quality_boost"] is False          # 正文已含面部要求，不追加英文后缀
    assert sent["image_paths"] == [os.path.abspath(portrait_photo)]
    assert sent["prompt"] == ac.anime_convert_prompt()


def test_run_routes_gemini_other_api_types_to_whatai(monkeypatch, portrait_photo, tmp_path):
    out = tmp_path / "anime-convert_120002-cdef01.jpg"
    out.write_bytes(b"fake-jpeg-bytes")
    calls = _fake_backend(monkeypatch, "generate_image_whatai", [str(out)])
    saved = ac.run_anime_convert(portrait_photo, channel="gemini",
                                 model_name="gpt-image-2", api_type="autodl",
                                 log_callback=lambda m: None)
    assert saved == [str(out)]
    sent = calls[0]
    assert sent["model"] == "gpt-image-2" and sent["api_type"] == "autodl"
    assert sent["image_paths"] == [os.path.abspath(portrait_photo)]
    assert sent["prompt"] == ac.anime_convert_prompt()


def test_run_no_output_writes_no_record(monkeypatch, portrait_photo, tmp_path):
    calls = _fake_backend(monkeypatch, "generate_image_aigc2d_gpt", [])
    saved = ac.run_anime_convert(portrait_photo, channel="gpt-image", log_callback=lambda m: None)
    assert saved == [] and len(calls) == 1
    leftovers = [p for p in os.listdir(str(tmp_path)) if p.endswith(".anime-convert.json")]
    assert leftovers == []


# ---------- 4. 界面接线 ----------

@pytest.fixture
def widget(qapp, monkeypatch, tmp_path):
    import modules.image_analysis.single_analyzer as sa
    classic = ("http://img", "imgkey", "gemini-3-pro-image-preview", "aigc2d")
    w = sa.SingleAnalyzerWidget(
        config_getter_func=lambda: {"base_url": "http://x", "api_key": "k", "model": "m"},
        img_config_getter_func=lambda: classic,
        styles_getter_func=lambda: {"tid": {"prompt": "FULL", "prompt_gpt": "Palette: pale"}},
        save_img_cfg_callback=lambda *a, **k: None,
        ar_policy_getter_func=lambda: {"override_second": "", "policy": "keep"},
        nsfw_default_getter_func=lambda: False,
        upscale_options_getter_func=lambda: {},
        outfit_style_history_getter_func=lambda: [],
        outfit_style_default_getter_func=lambda: "",
    )
    w.update_styles(["tid"])
    monkeypatch.setattr(sa.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    # 测试里不真正起 QThread：只验证接线与线程登记
    monkeypatch.setattr(sa.AnimeConvertWorkerThread, "start", lambda self: None)
    return w


def test_button_lives_in_generate_row_and_follows_preview(widget, portrait_photo):
    assert widget.anime_convert_btn.text() == "转化二次元"
    assert not widget.anime_convert_btn.isEnabled()          # 初始无图：禁用
    widget.image_source = portrait_photo                     # 拖拽/粘贴路径会先记录源再刷新预览
    widget.show_preview(portrait_photo)
    assert widget.anime_convert_btn.isEnabled()
    widget.image_source = None
    widget._sync_anime_convert_btn()
    assert not widget.anime_convert_btn.isEnabled()


def test_trigger_without_image_only_warns(widget):
    widget.image_source = None
    widget.trigger_anime_convert()
    assert widget._active_img_threads == []


def test_trigger_registers_gemini_thread(widget, portrait_photo):
    import modules.image_analysis.single_analyzer as sa
    widget.image_source = portrait_photo
    widget.gen_channel_gemini.setChecked(True)
    widget.trigger_anime_convert()
    assert len(widget._active_img_threads) == 1
    thread = widget._active_img_threads[-1]
    assert isinstance(thread, sa.AnimeConvertWorkerThread)
    assert thread.channel == "gemini"
    assert thread.api_type == "aigc2d"
    assert thread.model_name == "gemini-3-pro-image-preview"
    assert thread.meta_task_hash == ""                      # 独立动作：不关联队列任务
    assert not widget.gen_orig_btn.isEnabled()
    assert not widget.anime_convert_btn.isEnabled()


def test_trigger_registers_gpt_thread(widget, portrait_photo, monkeypatch):
    widget.image_source = portrait_photo
    widget.gen_channel_gpt.setChecked(True)
    monkeypatch.setattr("modules.others.api_backend.get_api_config",
                        lambda *a, **k: {"api_key": "k", "timeout": 600})
    widget.trigger_anime_convert()
    thread = widget._active_img_threads[-1]
    assert thread.channel == "gpt-image"
    assert thread.source_path == os.path.abspath(portrait_photo)


def test_trigger_skipped_while_busy(widget, portrait_photo):
    widget.image_source = portrait_photo
    widget._active_img_threads.append(object())             # 已有任务在跑
    before = len(widget._active_img_threads)
    widget.trigger_anime_convert()
    assert len(widget._active_img_threads) == before        # 没有新线程


def test_finish_recovers_button_when_idle(widget, portrait_photo):
    widget.image_source = portrait_photo
    widget.gen_channel_gemini.setChecked(True)
    widget.trigger_anime_convert()                          # 登记线程（start 已打桩不真正启动）
    thread = widget._active_img_threads[-1]
    widget._on_anime_convert_finished(thread, [portrait_photo])   # 日志路径不抛错
    widget._active_img_threads.remove(thread)
    widget._mark_pipeline_idle_if_done()
    assert widget.gen_orig_btn.isEnabled()
    assert widget.anime_convert_btn.isEnabled()
