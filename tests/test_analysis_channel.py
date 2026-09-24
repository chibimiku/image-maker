# -*- coding: utf-8 -*-
"""分析 Tab 的「生图通道」单选框 + gpt-image-2 分支（#4）。

只测逻辑接线，不真的调 API：`GptImageGenWorkerThread` 会被换成只记录参数的假类。
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import modules.image_analysis.single_analyzer as sa  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


class _SignalStub:
    def connect(self, *args, **kwargs):
        return None


class _FakeGptWorker:
    last = {}

    def __init__(self, **kwargs):
        _FakeGptWorker.last = kwargs
        self.log_signal = _SignalStub()
        self.finish_signal = _SignalStub()
        self.finished = _SignalStub()
        self.meta_thread_no = None
        self.meta_analysis_thread_no = None

    def start(self):
        _FakeGptWorker.last["started"] = True

    def deleteLater(self):
        pass


@pytest.fixture()
def analyzer(qapp, monkeypatch, tmp_path):
    classic = {"base_url": "http://x", "api_key": "k", "model": "m", "api_type": "aigc2d"}
    monkeypatch.setattr(sa, "list_esrgan_models", lambda: ["realesrgan-x4plus"])
    monkeypatch.setattr(sa, "GptImageGenWorkerThread", _FakeGptWorker)
    # 必须在构造 widget 之前隔离：initUI 里会读 UI 记忆文件
    monkeypatch.setattr(sa, "analysis_gpt_ui_path", lambda: str(tmp_path / "gpt_ui_state.json"))
    widget = sa.SingleAnalyzerWidget(
        config_getter_func=lambda: {"base_url": "http://x", "api_key": "k", "model": "m"},
        img_config_getter_func=lambda: classic,
        styles_getter_func=lambda: {"tid": {"prompt": "FULL", "prompt_gpt": "Palette: pale",
                                            "ref_image": os.path.join(BASE, "data", "style-ref", "tid.png")}},
        save_img_cfg_callback=lambda *a, **k: None,
        ar_policy_getter_func=lambda: {"override_second": "", "policy": "keep"},
        nsfw_default_getter_func=lambda: False,
        upscale_options_getter_func=lambda: {},
        outfit_style_history_getter_func=lambda: [],
        outfit_style_default_getter_func=lambda: "",
    )
    widget.update_styles(["tid"])
    # 线程相关的控件在测试环境里不真正启动
    monkeypatch.setattr(sa, "_start_image_gen_runtime", lambda *a, **k: None, raising=False)
    return widget


def test_channel_radios_exist_and_default_to_gemini(analyzer):
    assert analyzer.gen_channel_gemini.isChecked() is True
    assert analyzer.gen_channel_gpt.isChecked() is False
    # 默认（Gemini 通道）不显示 gpt 工序行（窗口未显示，用 isHidden 判断显隐意图）
    assert analyzer.gpt_pp_row.isHidden() is True
    assert analyzer.gen_channel_row.isHidden() is False


def test_selecting_gpt_channel_shows_post_process_row(analyzer):
    analyzer.gen_channel_gpt.setChecked(True)
    assert analyzer.gpt_pp_row.isHidden() is False
    analyzer.gen_channel_gemini.setChecked(True)
    assert analyzer.gpt_pp_row.isHidden() is True


def test_build_gpt_image_steps_follows_checkboxes(analyzer):
    """默认 = §三十一 的配方（重绘 + 结构线 + 色调校准 + 加墨，重绘范围=只连通线条）。

    局部重绘（裁切→重绘→贴回）已被证实会把画面拼坏（§三十），**默认关闭**；取消/勾选都能生效。
    """
    analyzer.gen_channel_gpt.setChecked(True)
    steps = analyzer._build_gpt_image_steps()
    assert steps["repaint"]["enabled"] is True
    assert steps["repaint"]["scope"] == "lines_only"
    assert steps["structure"]["enabled"] is True
    assert steps["local"]["enabled"] is False            # 默认不再做裁切贴回
    assert steps["tone"]["enabled"] is True and steps["tone"]["tone_target"] == "style"
    assert steps["ink"]["enabled"] is True
    # 打开局部重绘 → 用稳定四区链（单区域 subject_no_face 约 1/3 概率重排主体出鬼影）
    analyzer.gpt_pp_local.setChecked(True)
    steps = analyzer._build_gpt_image_steps()
    assert steps["local"]["enabled"] is True
    assert steps["local"]["regions"] == ["subject_no_face", "shoes", "waist", "thigh"]
    assert steps["local"]["region"] == "subject_no_face"
    # 单独选一个区域时不再是四区链
    analyzer.gpt_pp_region.setCurrentIndex(max(0, analyzer.gpt_pp_region.findData("shoes")))
    steps = analyzer._build_gpt_image_steps()
    assert steps["local"]["regions"] == ["shoes"]
    # 全部关掉
    for cb, key in ((analyzer.gpt_pp_repaint, "repaint"), (analyzer.gpt_pp_structure, "structure"),
                    (analyzer.gpt_pp_local, "local"), (analyzer.gpt_pp_tone, "tone"),
                    (analyzer.gpt_pp_ink, "ink")):
        cb.setChecked(False)
    steps = analyzer._build_gpt_image_steps()
    assert not any(steps[k]["enabled"] for k in steps)
    # 重绘范围可切换
    if hasattr(analyzer, "gpt_pp_scope"):
        analyzer.gpt_pp_scope.setCurrentIndex(max(0, analyzer.gpt_pp_scope.findData("person_only")))
        analyzer.gpt_pp_repaint.setChecked(True)
        assert analyzer._build_gpt_image_steps()["repaint"]["scope"] == "person_only"


def test_gpt_timeout_budget_counts_network_slots_only(analyzer):
    """超时预算按**联网调用**份数算：首图 + 重绘 + 每个局部区域；本地工序（结构线/色调/加墨）不占份额。"""
    analyzer.gen_channel_gpt.setChecked(True)
    steps = analyzer._build_gpt_image_steps()          # 默认：首图 + 重绘（局部关闭，结构线/色调/加墨是本地）
    assert analyzer._pipeline_timeout_budget(120, steps) == 240
    analyzer.gpt_pp_local.setChecked(True)             # 打开四区链 → 再多 4 份
    steps = analyzer._build_gpt_image_steps()
    assert analyzer._pipeline_timeout_budget(120, steps) == 720
    analyzer.gpt_pp_repaint.setChecked(False)          # 只留首图 + 四区
    steps = analyzer._build_gpt_image_steps()
    assert analyzer._pipeline_timeout_budget(120, steps) == 600
    analyzer.gpt_pp_local.setChecked(False)            # 只剩首图（加墨/色调/结构线都是本地工序）
    steps = analyzer._build_gpt_image_steps()
    assert analyzer._pipeline_timeout_budget(120, steps) == 120


def test_gpt_channel_request_uses_analysis_description_and_style_ref(analyzer, tmp_path):
    """gpt 通道：提示词 = 画风 prompt_gpt（最前）+ **与 Gemini 同源的分析描述** + 排除句；参考图只有画风图。

    用户 2026-09-24 要求：内容用分析素材得到的那段描述（Gemini 通道发的是同一份），
    不再用 gpt 专用短锚，也没有「重新构图」开关。
    """
    import json
    analysis = {
        "gpt_image_prompt_short": "SHORT-ANCHOR-ONLY",
        "gpt_image_prompt": "FULL " * 200,
        "english_description": "LONG " * 500,
    }
    json_path = tmp_path / "result.json"
    json_path.write_text(json.dumps(analysis, ensure_ascii=False), encoding="utf-8")
    analyzer.gen_channel_gpt.setChecked(True)
    analyzer._last_saved_json_path = str(json_path)
    analyzer.current_aspect_ratio = "2:3"
    analyzer.current_task_hash = "hash123"
    ok = analyzer.trigger_image_generation(
        "refined", is_auto=False,
        prompt_bundle={"task_hash": "hash123", "aspect_ratio": "2:3",
                       "original_prompt": "orig desc", "refined_prompt": "refined desc",
                       "analysis_json_path": str(json_path)})
    assert ok is True
    payload = _FakeGptWorker.last["request_payload"]
    assert payload["prompt"].startswith("Palette: pale")        # 画风在最前
    assert "refined desc" in payload["prompt"]                  # 内容=与 Gemini 同源的分析描述
    assert "SHORT-ANCHOR-ONLY" not in payload["prompt"]         # 不再用 gpt 短锚
    assert "IMAGE ROLES" not in payload["prompt"]               # 只有画风参考图，不需要分工块
    from utils.analysis_gpt_prompt import STYLE_REF_EXCLUSION
    assert STYLE_REF_EXCLUSION in payload["prompt"]
    assert len(payload["image_paths"]) == 1 and payload["image_paths"][0].endswith("tid.png")
    assert _FakeGptWorker.last.get("started") is True
    # 首图走「新建图片」（/images/generations + 参考图字段），不走 edits
    assert _FakeGptWorker.last.get("mode") == "generate"
    # 重绘的第二张参考 = 画风参考图（不是线锚图）：修「碎玻璃/塌线稿」那处
    assert _FakeGptWorker.last.get("style_ref_path", "").endswith("tid.png")
    assert _FakeGptWorker.last["steps"]["repaint"]["reference_mode"] == "style"


def test_gpt_channel_falls_back_to_analysis_json_description(analyzer, tmp_path):
    """prompt_bundle 里没有描述文本时，退回分析产物里的描述字段（老产物兼容）。"""
    import json
    json_path = tmp_path / "old.json"
    json_path.write_text(json.dumps({"english_description": "sentence. " * 200}), encoding="utf-8")
    analyzer.gen_channel_gpt.setChecked(True)
    ok = analyzer.trigger_image_generation(
        "refined", prompt_bundle={"task_hash": "h", "aspect_ratio": "2:3",
                                  "original_prompt": "", "refined_prompt": "",
                                  "analysis_json_path": str(json_path)})
    assert ok is True
    payload = _FakeGptWorker.last["request_payload"]
    assert payload["content_chars"] > 0
    assert "sentence." in payload["prompt"]


def test_gpt_channel_local_step_is_2k_without_detail_boost(analyzer):
    """局部重绘统一 2K（不把细节区升 4K），并把每道工序的超时预算算给运行时。"""
    analyzer.gen_channel_gpt.setChecked(True)
    analyzer.gpt_pp_repaint.setChecked(True)
    analyzer.gpt_pp_structure.setChecked(True)
    analyzer.gpt_pp_local.setChecked(True)
    steps = analyzer._build_gpt_image_steps()
    assert steps["local"]["resolution"] == "2K"
    assert steps["local"]["detail_boost"] is False
    # 超时预算 = 每份联网调用 120 秒 ×（首图 1 + 重绘 1 + 四区各 1）= 720 —— 不再用总体 120 秒掐掉整条链
    assert analyzer._pipeline_timeout_budget(120, steps) == 720


def _first_pass_payload(analyzer, tmp_path, styles, style_name):
    """让 GUI 走一次首图组装，返回 (payload, worker 关键字参数)。"""
    import json
    analyzer.get_styles = lambda: styles
    analyzer.update_styles(list(styles))
    analyzer.main_style_combo.setCurrentText(style_name)
    json_path = tmp_path / "result.json"
    json_path.write_text(json.dumps({"english_description": "desc " * 20}), encoding="utf-8")
    analyzer.gen_channel_gpt.setChecked(True)
    analyzer._last_saved_json_path = str(json_path)
    ok = analyzer.trigger_image_generation(
        "refined", prompt_bundle={"task_hash": "h", "aspect_ratio": "2:3",
                                  "original_prompt": "orig", "refined_prompt": "refined desc",
                                  "analysis_json_path": str(json_path)})
    assert ok is True
    return _FakeGptWorker.last["request_payload"], _FakeGptWorker.last


def test_gpt_first_pass_includes_derived_render_clauses(analyzer, tmp_path):
    """没有手写 repaint_clauses 的画风：首图也必须带 RENDERING LANGUAGE 段（按 prompt_gpt 派生）。

    这就是 2026-09-24 那个 bug：GUI 的 `_start_gpt_image_thread` 算了 style_clauses 却没传给
    `build_gpt_image_request`，于是 GUI 首图永远比 CLI 少一截渲染语言约束。
    """
    styles = {"tid": {"prompt": "FULL", "prompt_gpt": "Palette: pale\nLighting: soft key",
                      "ref_image": os.path.join(BASE, "data", "style-ref", "tid.png")}}
    payload, kwargs = _first_pass_payload(analyzer, tmp_path, styles, "tid")
    assert "RENDERING LANGUAGE (follow exactly):" in payload["prompt"]
    assert payload["clauses_source"] == "derived"
    assert len(payload["clauses"]) >= 2
    assert "Palette: pale" in payload["prompt"]                 # 画风仍在最前
    # 同一批条款要一路传给重绘（首图与重绘用同一份，别只在一边生效）
    assert kwargs.get("style_clauses") == payload["clauses"]
    assert kwargs.get("style_ref_path", "").endswith("tid.png")


def test_gpt_first_pass_prefers_handwritten_clauses(analyzer, tmp_path):
    """画风自带 repaint_clauses 时用原条款（tinkle 就是这种），不要被派生结果覆盖。"""
    written = ["Use clean tapered linework with dark navy-blue contours.",
               "Keep deep blue-violet anchors so the image never becomes pale or grey."]
    styles = {"tinkle": {"prompt": "FULL", "prompt_gpt": "Palette: sapphire",
                         "ref_image": os.path.join(BASE, "data", "style-ref", "tid.png"),
                         "repaint_clauses": written}}
    payload, kwargs = _first_pass_payload(analyzer, tmp_path, styles, "tinkle")
    assert payload["clauses_source"] == "entry"
    assert payload["clauses"] == written
    for clause in written:
        assert clause in payload["prompt"]
    assert kwargs.get("style_clauses") == written



def test_gpt_quality_combo_defaults_to_high(analyzer):
    """首图质量默认 high（medium 的细节 token 只有 high 的 1/5，重绘放大后必糊）。"""
    assert analyzer.gpt_quality_combo.currentData() == "high"
    assert analyzer.gpt_quality_row.isHidden() is True      # 未选 gpt 通道时隐藏
    analyzer.gen_channel_gpt.setChecked(True)
    assert analyzer.gpt_quality_row.isHidden() is False


def test_gpt_channel_passes_quality_to_worker(analyzer, tmp_path):
    import json
    json_path = tmp_path / "r.json"
    json_path.write_text(json.dumps({"gpt_image_prompt_short": "a woman"}), encoding="utf-8")
    analyzer.gen_channel_gpt.setChecked(True)
    analyzer.gpt_quality_combo.setCurrentIndex(analyzer.gpt_quality_combo.findData("high"))
    analyzer.trigger_image_generation(
        "refined", prompt_bundle={"task_hash": "h", "aspect_ratio": "2:3", "original_prompt": "o",
                                  "refined_prompt": "r", "analysis_json_path": str(json_path)})
    assert _FakeGptWorker.last.get("quality") == "high"


def test_gpt_pipeline_choices_are_remembered(analyzer, tmp_path, monkeypatch):
    """勾选的通道/工序/区域/画质要写进 conf/config.json，并在新实例里恢复。"""
    state_file = tmp_path / "config.json"
    monkeypatch.setattr(sa, "analysis_gpt_ui_path", lambda: str(state_file))
    analyzer._save_gpt_pipeline_ui()

    analyzer.gen_channel_gpt.setChecked(True)
    analyzer.gpt_pp_repaint.setChecked(True)
    analyzer.gpt_pp_structure.setChecked(True)
    analyzer.gpt_pp_local.setChecked(True)
    analyzer.gpt_pp_region.setCurrentIndex(analyzer.gpt_pp_region.findData("subject_no_face"))
    analyzer.gpt_quality_combo.setCurrentIndex(analyzer.gpt_quality_combo.findData("high"))

    saved = sa.load_analysis_gpt_ui(str(state_file))
    assert saved["channel"] == "gpt-image"
    assert saved["repaint"] and saved["structure"] and saved["local"]
    assert saved["region"] == "subject_no_face" and saved["quality"] == "high"

    # 新实例（模拟重启）应恢复同样的选择
    fresh = sa.SingleAnalyzerWidget(
        config_getter_func=lambda: {"base_url": "http://x", "api_key": "k", "model": "m"},
        img_config_getter_func=lambda: {"base_url": "http://x", "api_key": "k", "model": "m",
                                        "api_type": "aigc2d"},
        styles_getter_func=lambda: {}, save_img_cfg_callback=lambda *a, **k: None,
        ar_policy_getter_func=lambda: {"override_second": ""}, nsfw_default_getter_func=lambda: False,
        upscale_options_getter_func=lambda: {}, outfit_style_history_getter_func=lambda: [],
        outfit_style_default_getter_func=lambda: "",
    )
    assert fresh.gen_channel_gpt.isChecked() is True
    assert fresh.gpt_pp_repaint.isChecked() and fresh.gpt_pp_structure.isChecked()
    assert fresh.gpt_pp_local.isChecked()
    assert fresh.gpt_pp_region.currentData() == "subject_no_face"
    assert fresh.gpt_quality_combo.currentData() == "high"


def test_gpt_pipeline_memory_defaults_when_missing(tmp_path):
    state = sa.load_analysis_gpt_ui(str(tmp_path / "nope.json"))
    assert state == sa.ANALYSIS_GPT_UI_DEFAULTS


def test_gpt_first_pass_dir_follows_steps():
    """首图落盘目录规则：无后续工序 → data/<日期>（最终产物）；有工序 → analysis-gpt-image（中间产物）。"""
    from utils.analysis_gen import first_pass_sub_dir
    assert first_pass_sub_dir({"repaint": {"enabled": False}, "structure": {"enabled": False},
                               "local": {"enabled": False}}) == ""
    assert first_pass_sub_dir({"structure": {"enabled": True}}) == "analysis-gpt-image"
    assert first_pass_sub_dir({"local": {"enabled": True}}) == "analysis-gpt-image"


def test_pipeline_final_dir_is_date_root_by_default():
    from utils.post_process import date_output_dir
    assert os.path.basename(date_output_dir()) == __import__("datetime").datetime.now().strftime("%Y%m%d")


def _seed_record(analyzer, task_hash="abc123", task_id="analysis-1"):
    analyzer._analysis_history[task_id] = {
        "task_id": task_id, "thread_no": 1, "task_hash": task_hash, "status": "running",
        "status_text": "进行中", "title": "t", "submit_time_text": "-", "finish_time_text": "-",
        "source_desc": "-",
    }
    return task_id


def test_queue_marks_success_only_with_final_product(analyzer):
    """有最终产物 + 没有待跑线程 → 才置「已完成」。"""
    task_id = _seed_record(analyzer)
    ok = analyzer._finalize_task_pipeline("abc123", final_products=["data/x-final-sline50.png"])
    assert ok is True
    rec = analyzer._analysis_history[task_id]
    assert rec["status"] == "success" and rec["final_products"]


def test_queue_stays_running_while_post_process_running(analyzer):
    """后处理线程还在跑 → 必须保持「进行中·后处理」。"""
    task_id = _seed_record(analyzer)

    class _T:
        meta_task_hash = "abc123"
        meta_task_id = "analysis-1"

    analyzer._active_post_threads.append(_T())
    ok = analyzer._finalize_task_pipeline("abc123", final_products=["data/x.png"])
    assert ok is False
    assert analyzer._analysis_history[task_id]["status"] == "running"
    assert "后处理" in analyzer._analysis_history[task_id]["phase"]
    analyzer._active_post_threads.clear()


def test_queue_stays_running_without_final_product(analyzer):
    """线程都结束了但没有最终产物 → 不能标绿（提示等待最终产物）。"""
    task_id = _seed_record(analyzer, task_hash="nofile999")
    ok = analyzer._finalize_task_pipeline("nofile999")
    assert ok is False
    assert analyzer._analysis_history[task_id]["status"] == "running"
    assert analyzer._analysis_history[task_id]["phase"] == "等待最终产物"


def test_history_status_text_shows_phase(analyzer):
    assert analyzer._refresh_history_status_text({"status": "running", "status_text": "进行中",
                                                  "phase": "生图+后处理"}) == "进行中·生图+后处理"
    assert analyzer._refresh_history_status_text({"status": "success", "status_text": "已完成",
                                                  "phase": ""}) == "已完成"


def test_analysis_size_follows_input_orientation(analyzer, tmp_path, monkeypatch):
    """分析 Tab 的 gpt 通道要按输入图比例选尺寸（宽图不能出竖图）。"""
    import cv2
    import numpy as np
    from modules.others import api_backend

    wide = tmp_path / "wide_source.png"
    cv2.imwrite(str(wide), np.full((600, 1200, 3), 200, np.uint8))
    monkeypatch.setattr(sa, "GptImageGenWorkerThread", _FakeGptWorker)
    _FakeGptWorker.last = {}
    analyzer.gen_channel_gpt.setChecked(True)
    analyzer.gpt_size_follow_cb.setChecked(True)
    import json
    js = tmp_path / "r.json"
    js.write_text(json.dumps({"gpt_image_prompt_short": "a woman"}), encoding="utf-8")
    analyzer.trigger_image_generation("refined", prompt_bundle={
        "task_hash": "h", "aspect_ratio": "2:3", "original_prompt": "o", "refined_prompt": "r",
        "analysis_json_path": str(js), "source_image_path": str(wide)})
    assert _FakeGptWorker.last.get("size") == "1536x1024"


def test_analysis_size_checkbox_default_on(analyzer):
    assert analyzer.gpt_size_follow_cb.isChecked() is True


def test_pick_size_helper_used_by_tab_default():
    from modules.others.api_backend import pick_gpt_image2_size_for_images
    assert pick_gpt_image2_size_for_images([], fallback="1536x1024") == "1536x1024"
