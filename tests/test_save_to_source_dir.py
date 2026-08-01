"""单元测试：分析结果保存到原图同目录功能

覆盖：
  1. _on_save_to_source_dir_toggled —— 勾选时禁用自动生图选项
  2. save_to_source 模式下的文件名格式 —— publish_server 兼容性
  3. _sync_file_times_to_source —— 文件时间同步
"""

import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

import modules.image_analysis.single_analyzer as single_analyzer_module
from modules.image_analysis.single_analyzer import SingleAnalyzerWidget


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def text_config():
    return ("https://example.invalid/v1", "test-key", "test-model")


def image_config():
    return ("https://example.invalid/v1", "image-key", "image-model", "openai")


def styles_config():
    return {"默认风格": "masterpiece", "维多利亚": "victorian dress"}


def ar_policy():
    return {
        "default_aspect_ratio": "1:1",
        "override_first": "不覆盖(沿用原逻辑)",
        "override_second": "3:4",
    }


def _make_widget() -> SingleAnalyzerWidget:
    return SingleAnalyzerWidget(
        config_getter_func=text_config,
        img_config_getter_func=image_config,
        styles_getter_func=styles_config,
        save_img_cfg_callback=lambda *args, **kwargs: None,
        ar_policy_getter_func=ar_policy,
        nsfw_default_getter_func=lambda: False,
        upscale_options_getter_func=lambda: {},
        outfit_style_history_getter_func=lambda: [],
        outfit_style_default_getter_func=lambda: "",
    )


# ---------------------------------------------------------------------------
# 测试 1：勾选框联动 —— 勾选时自动生图选项变灰
# ---------------------------------------------------------------------------

def test_toggle_disables_auto_gen_checkboxes(qapp):
    """勾选"保存到原图同目录"时，两个自动生图勾选框应被禁用（变灰）。

    取消勾选后应恢复可用。
    """
    widget = _make_widget()

    # 初始状态：全部可用
    assert widget.auto_gen_orig_cb.isEnabled()
    assert widget.auto_gen_ref_cb.isEnabled()
    assert not widget.save_to_source_dir_cb.isChecked()

    # 模拟勾选
    widget.save_to_source_dir_cb.setChecked(True)
    assert not widget.auto_gen_orig_cb.isEnabled(), \
        "勾选后原始提示词生图选项应被禁用"
    assert not widget.auto_gen_ref_cb.isEnabled(), \
        "勾选后优化提示词生图选项应被禁用"

    # 取消勾选
    widget.save_to_source_dir_cb.setChecked(False)
    assert widget.auto_gen_orig_cb.isEnabled(), \
        "取消勾选后原始提示词生图选项应恢复"
    assert widget.auto_gen_ref_cb.isEnabled(), \
        "取消勾选后优化提示词生图选项应恢复"

    widget.close()


# ---------------------------------------------------------------------------
# 测试 2：文件名格式 —— publish_server 兼容性
# ---------------------------------------------------------------------------

def test_save_to_source_filename_format(qapp, monkeypatch, tmp_path):
    """验证 save_to_source 模式下的文件名符合 publish_server 的匹配规则。

    publish_server.find_metadata_json 的匹配逻辑：
      - 图片名 `{key}_{suffix}.jpg`  → 取 `_` 前第一段作为 key
      - JSON   `YYYYMMDD-HHMMSS-{key}-{title}.json`
        → 按 `-` 拆分，parts[2] == key 则匹配

    本测试模拟 on_process_finished 中的文件名生成逻辑，验证格式正确性。
    """
    monkeypatch.setattr(single_analyzer_module, "list_esrgan_models", lambda: ["realesrgan-x4plus"])

    widget = _make_widget()
    widget.save_to_source_dir_cb.setChecked(True)

    # 模拟原图路径
    image_name = "ee721fed_003711-18b7de.jpg"
    source_image = tmp_path / image_name
    source_image.write_bytes(b"fake image")

    # 模拟 trigger_image_generation 不做任何事
    monkeypatch.setattr(widget, "trigger_image_generation", lambda *a, **kw: True)
    # 禁用系统通知，避免测试环境中出错
    monkeypatch.setattr(widget, "_send_system_notification", lambda *a: None)

    from unittest.mock import Mock

    mock_thread = Mock()
    mock_thread.meta_thread_no = 0
    mock_thread.meta_task_hash = "abc123"
    mock_thread.meta_task_id = "task_0"
    mock_thread.last_status = "success"
    mock_thread.meta_source_snapshot = str(source_image)

    result_json = {
        "japanese_title": "紫苑の天光",
        "english_description": "a girl under the sky",
        "original_english_description": "original prompt",
        "aspect_ratio": "2:3",
    }

    widget.on_process_finished(mock_thread, result_json)

    saved_path = widget._last_saved_json_path
    assert saved_path, "应生成 JSON 文件路径"

    saved_dir = os.path.dirname(saved_path)
    assert os.path.normpath(saved_dir) == os.path.normpath(str(tmp_path)), \
        f"JSON 应保存在原图目录 {tmp_path}，实际保存在 {saved_dir}"

    json_basename = os.path.basename(saved_path)
    # 文件名格式：YYYYMMDD-HHMMSS-{key}-{safe_title}.json
    name_no_ext = os.path.splitext(json_basename)[0]
    parts = name_no_ext.split("-")

    assert len(parts) >= 3, f"文件名至少应有 3 段 '-' 分隔，实际：{parts}"
    assert len(parts[0]) == 8 and parts[0].isdigit(), \
        f"第一段应为 YYYYMMDD 日期，实际：{parts[0]}"
    assert len(parts[1]) == 6 and parts[1].isdigit(), \
        f"第二段应为 HHMMSS 时间，实际：{parts[1]}"

    source_key = image_name.split("_")[0]
    assert parts[2] == source_key, \
        f"第三段（key）应为 '{source_key}'（从原图文件名提取），实际：{parts[2]}。" \
        f"\n  publish_server 匹配规则：图片 {image_name} 的 key={source_key}，" \
        f"JSON {json_basename} 按 '-' 拆分后 parts[2]={parts[2]}。二者必须一致。"

    widget.close()


# ---------------------------------------------------------------------------
# 测试 3：文件时间同步
# ---------------------------------------------------------------------------

def test_sync_file_times_to_source(tmp_path):
    """验证 _sync_file_times_to_source 将目标文件的 mtime/atime 同步到源文件一致。

    同时验证：源文件不存在 / 目标文件不存在时静默返回，不抛异常。
    """
    widget = _make_widget()

    # ---- 正常同步 ----
    src = tmp_path / "source.png"
    dst = tmp_path / "target.json"

    src.write_bytes(b"fake image content")
    dst.write_text('{"key": "value"}')

    # 给源文件设置一个特殊的时间戳（往过去偏移 3600 秒）
    old_src_stat = os.stat(str(src))
    target_mtime = old_src_stat.st_mtime - 3600
    os.utime(str(src), (old_src_stat.st_atime, target_mtime))

    # 确认目标文件初始时间与源不同
    dst_stat_before = os.stat(str(dst))
    assert abs(dst_stat_before.st_mtime - target_mtime) > 1, \
        "同步前目标文件 mtime 应与源文件不同"

    # 执行同步
    widget._sync_file_times_to_source(str(dst), str(src))

    dst_stat_after = os.stat(str(dst))
    src_stat = os.stat(str(src))
    assert dst_stat_after.st_mtime == src_stat.st_mtime, \
        f"同步后 mtime 应一致：dst={dst_stat_after.st_mtime}, src={src_stat.st_mtime}"
    assert dst_stat_after.st_atime == src_stat.st_atime, \
        f"同步后 atime 应一致：dst={dst_stat_after.st_atime}, src={src_stat.st_atime}"

    # ---- 源文件不存在：静默返回 ----
    widget._sync_file_times_to_source(str(dst), str(tmp_path / "nonexistent.png"))
    # 不应抛异常，dst 时间不应被改变
    dst_stat_unchanged = os.stat(str(dst))
    assert dst_stat_unchanged.st_mtime == dst_stat_after.st_mtime

    # ---- 目标文件不存在：静默返回 ----
    widget._sync_file_times_to_source(str(tmp_path / "nonexistent.json"), str(src))
    # 不应抛异常

    # ---- 两个都不存在：静默返回 ----
    widget._sync_file_times_to_source(
        str(tmp_path / "no_a.json"),
        str(tmp_path / "no_b.png"),
    )
    # 不应抛异常

    widget.close()
