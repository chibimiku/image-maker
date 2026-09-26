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


def test_channel_radios_exist_and_default_to_gpt_recipe(analyzer):
    assert analyzer.gen_channel_gpt.isChecked() is True
    assert analyzer.gen_channel_gemini.isChecked() is False
    # gpt 是新装/配方升级后的默认通道；高级参数仍默认折叠。
    assert analyzer.gpt_pp_row.isHidden() is True
    assert analyzer.gen_channel_row.isHidden() is False


def test_selecting_gpt_channel_shows_post_process_row(analyzer):
    analyzer.gen_channel_gpt.setChecked(True)
    analyzer.gpt_advanced_toggle.setChecked(True)
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
    assert steps["repaint"]["scope"] == "full"
    assert steps["structure"]["enabled"] is False
    assert steps["local"]["enabled"] is False            # 默认不再做裁切贴回
    assert steps["tone"]["enabled"] is False and steps["tone"]["tone_target"] == "style"
    assert steps["ink"]["enabled"] is False
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
    """预算覆盖首图、重绘、质量/身份门禁和局部区域；本地工序不占份额。"""
    analyzer.gen_channel_gpt.setChecked(True)
    steps = analyzer._build_gpt_image_steps()          # 默认：首图 + 重绘（局部关闭，结构线/色调/加墨是本地）
    assert analyzer._pipeline_timeout_budget(120, steps) == 1200
    analyzer.gpt_pp_local.setChecked(True)             # 打开四区链 → 再多 4 份
    steps = analyzer._build_gpt_image_steps()
    assert analyzer._pipeline_timeout_budget(120, steps) == 1680
    analyzer.gpt_pp_repaint.setChecked(False)          # 只留首图 + 四区
    steps = analyzer._build_gpt_image_steps()
    assert analyzer._pipeline_timeout_budget(120, steps) == 600
    analyzer.gpt_pp_local.setChecked(False)            # 只剩首图（加墨/色调/结构线都是本地工序）
    steps = analyzer._build_gpt_image_steps()
    assert analyzer._pipeline_timeout_budget(120, steps) == 120


def test_gpt_channel_request_uses_identity_complete_anchor_and_style_ref(analyzer, tmp_path):
    """挂画风图时使用身份完整锚，避免全文压弱画风，也避免短锚漏发色/瞳色。"""
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
    assert "FULL FULL" in payload["prompt"]                    # 约 1400 字符身份完整锚
    assert "refined desc" not in payload["prompt"]
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
    # 重绘链还含初审、最多两轮修订及每轮复审。
    assert analyzer._pipeline_timeout_budget(120, steps) == 1680


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
    assert kwargs.get("analysis_result", {}).get("english_description", "").startswith("desc ")


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
    assert analyzer.gpt_param_row.isHidden() is True        # 未选 gpt 通道时隐藏
    analyzer.gen_channel_gpt.setChecked(True)
    analyzer.gpt_advanced_toggle.setChecked(True)
    assert analyzer.gpt_param_row.isHidden() is False


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


def test_restore_recommended_recipe_and_advanced_do_not_reset_user_edits(analyzer):
    analyzer.gen_channel_gpt.setChecked(True)
    assert not analyzer.gpt_pp_repaint.isHidden()
    analyzer.gpt_advanced_toggle.setChecked(True)
    analyzer.gpt_pp_ink.setChecked(True)
    analyzer.gpt_advanced_toggle.setChecked(False)
    assert analyzer._build_gpt_image_steps()["ink"]["enabled"]
    analyzer.gpt_reset_recipe.click()
    steps = analyzer._build_gpt_image_steps()
    assert [k for k, v in steps.items() if v["enabled"]] == ["repaint"]
    assert steps["repaint"]["reference_mode"] == "style"
    assert steps["repaint"]["scope"] == "full"
    assert analyzer.gen_channel_gpt.isChecked()


def test_gpt_first_pass_dir_follows_steps():
    """首图落盘目录规则：无后续工序 → data/<日期>（最终产物）；有工序 → analysis-gpt-image（中间产物）。"""
    from utils.analysis_gen import first_pass_sub_dir
    assert first_pass_sub_dir({"repaint": {"enabled": False}, "structure": {"enabled": False},
                               "local": {"enabled": False}}) == ""
    assert first_pass_sub_dir({"structure": {"enabled": True}}) == "analysis-gpt-image"
    assert first_pass_sub_dir({"local": {"enabled": True}}) == "analysis-gpt-image"
    per_run = first_pass_sub_dir({"repaint": {"enabled": True}}, "hash-tid")
    assert os.path.dirname(per_run) == "analysis-gpt-image"
    assert os.path.basename(per_run).startswith("hash-tid-")


def test_publish_final_keeps_process_files_out_of_date_root(tmp_path):
    from utils.analysis_gen import publish_final_output
    process_dir = tmp_path / "analysis-gpt-image" / "run-1"
    process_dir.mkdir(parents=True)
    selected = process_dir / "quality-refine.jpg"
    selected.write_bytes(b"final-image")
    (process_dir / "first-pass.png").write_bytes(b"first")
    (process_dir / "identity-audit-0.json").write_text("{}", encoding="utf-8")
    final_dir = tmp_path / "20260925"

    published = publish_final_output(
        str(selected), style_name="tid:v2", process_dir=str(process_dir), final_dir=str(final_dir))

    assert os.path.dirname(published) == str(final_dir)
    assert os.path.basename(published).startswith("tid-v2-")
    assert open(published, "rb").read() == b"final-image"
    assert sorted(p.name for p in final_dir.iterdir()) == [os.path.basename(published)]
    assert (process_dir / "first-pass.png").exists()
    assert (process_dir / "identity-audit-0.json").exists()
    assert (process_dir / "published-final.json").exists()


def test_publish_final_preserves_task_hash_for_metadata_link(tmp_path):
    from utils.analysis_gen import publish_final_output
    process_dir = tmp_path / "analysis-gpt-image" / "run-1"
    process_dir.mkdir(parents=True)
    selected = process_dir / "960cddea-quality-refine_105842-1e7c96.jpg"
    selected.write_bytes(b"final-image")

    published = publish_final_output(
        str(selected), style_name="ajicoma", process_dir=str(process_dir),
        final_dir=str(tmp_path / "20260926"))

    assert os.path.basename(published).startswith("960cddea_ajicoma-")


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


class _FakeGenThread:
    """生图线程替身：只需要队列收尾用到的 meta_* 字段。"""

    def __init__(self, task_hash="abc123", status="success", auto_group_id=None):
        self.meta_task_hash = task_hash
        self.meta_task_id = ""
        self.meta_thread_no = 2
        self.meta_analysis_thread_no = 1
        self.meta_prompt_type = "refined"
        self.meta_is_auto = True
        self.meta_auto_group_id = auto_group_id
        self.meta_analysis_json_path = ""
        self.last_status = status


class _FakeAnalysisThread:
    """分析线程替身（用于 _on_analysis_thread_stopped 的兜底逻辑）。"""

    def __init__(self, task_id, task_hash, status="success"):
        self.meta_task_id = task_id
        self.meta_task_hash = task_hash
        self.meta_thread_no = 1
        self.meta_source_snapshot = None
        self.last_status = status
        self.last_error = ""


def test_queue_turns_green_when_image_gen_thread_exits(analyzer):
    """生图线程退出时必须给队列收尾。

    用户 2026-09-24 截图：日志已经打完「🕐 同步完成」，队列却永远停在「进行中·生图」。
    原因是 `_finalize_task_pipeline` 只在 `on_image_generation_finished` 里调过一次，而那一刻
    生图线程自己还在 `_active_img_threads` 里 → 判定"还有线程在跑"→ 返回 False；线程真正退出时
    （`_on_image_thread_stopped`）没有人再调它。不勾「生图后自动处理 JPG」时根本没有后处理线程，
    于是这条记录再也没有机会变绿。
    """
    task_id = _seed_record(analyzer)
    analyzer._analysis_history[task_id]["phase"] = "生图/后处理中"
    thread = _FakeGenThread("abc123")
    analyzer._active_img_threads.append(thread)

    analyzer.on_image_generation_finished(thread, ["data/20260924/x-final-sline50.png"])
    assert analyzer._analysis_history[task_id]["status"] == "running"      # 线程还没退出 → 先别标绿

    analyzer._on_image_thread_stopped(thread)
    record = analyzer._analysis_history[task_id]
    assert record["status"] == "success" and record["phase"] == ""
    assert analyzer._status_to_color(record["status"]).name() == "#1b8f3a"  # 绿字 = 全部工序完成


def test_image_thread_exit_keeps_running_while_post_process_alive(analyzer):
    """生图线程退出但后处理还在跑 → 仍是「进行中·后处理」，不能抢跑标绿。"""
    task_id = _seed_record(analyzer)
    thread = _FakeGenThread("abc123")
    analyzer._active_img_threads.append(thread)

    class _PostThread:
        meta_task_hash = "abc123"
        meta_task_id = ""

    post = _PostThread()
    analyzer._active_post_threads.append(post)
    analyzer.on_image_generation_finished(thread, ["data/20260924/x-final-sline50.png"])
    analyzer._on_image_thread_stopped(thread)

    record = analyzer._analysis_history[task_id]
    assert record["status"] == "running" and "后处理" in record["phase"]

    analyzer._cleanup_post_thread(post)
    assert analyzer._analysis_history[task_id]["status"] == "success"
    analyzer._active_post_threads.clear()


def test_analysis_thread_fallback_does_not_clobber_live_pipeline(analyzer):
    """分析线程退出时的「兜底更新」不能覆盖正在跑生图的记录。

    分析完成时如果本任务还要自动生图，记录会被**故意**留在 running（phase=生图/后处理中）；
    以前兜底逻辑只看 `status == running` 就当"线程死了没上报"，于是把它标成
    「已完成（兜底更新）」，随后生图完成又被改回 running —— 队列文案因此来回跳。
    """
    task_id = _seed_record(analyzer)
    record = analyzer._analysis_history[task_id]
    record["phase"] = "生图/后处理中"
    record["title"] = "白发少女"

    gen_thread = _FakeGenThread("abc123")
    analyzer._active_img_threads.append(gen_thread)
    analysis_thread = _FakeAnalysisThread(task_id, "abc123")

    analyzer._on_analysis_thread_stopped(analysis_thread)
    assert record["status"] == "running" and record["title"] == "白发少女"
    assert "兜底更新" not in record["title"]

    # 管线真死了（没有任何线程）→ 兜底仍然生效
    analyzer._active_img_threads.clear()
    analyzer._on_analysis_thread_stopped(analysis_thread)
    assert record["status"] == "success" and record["title"] == "已完成（兜底更新）"


def test_generation_failure_marks_record_error(analyzer):
    """生图彻底失败（没有任何产物）→ 队列标红失败，不能永远「进行中·等待最终产物」。"""
    task_id = _seed_record(analyzer)
    record = analyzer._analysis_history[task_id]
    record["phase"] = "生图/后处理中"
    record["title"] = "白发少女"

    thread = _FakeGenThread("abc123", status="error")
    analyzer._active_img_threads.append(thread)
    analyzer.on_image_generation_finished(thread, [])
    analyzer._on_image_thread_stopped(thread)

    record = analyzer._analysis_history[task_id]
    assert record["status"] == "error"
    assert record["phase"] == ""
    assert record["title"] == "白发少女"        # 标题保留分析标题，失败信息在状态/日志里


def test_repaint_reference_is_always_style_image(analyzer, tmp_path):
    """「重绘用双参考（线锚图）」在分析 Tab 是死开关，2026-09-24 已删；重绘第二张参考恒为画风图。

    以前 `gpt_pp_dual` 只读写 conf 状态、没进布局也没人读：`_build_gpt_image_steps()` 从不传
    `dual_reference`，实际恒定走 `repaint_ref_mode="style"`。线锚图只留给无头 CLI
    （`--repaint-ref line_anchor|both`）——§三十一 实测它会把画面塌成白底线稿/铺「碎玻璃」。
    """
    assert not hasattr(analyzer, "gpt_pp_dual")
    assert "dual_reference" not in sa.ANALYSIS_GPT_UI_DEFAULTS
    analyzer.gen_channel_gpt.setChecked(True)
    steps = analyzer._build_gpt_image_steps()
    assert steps["repaint"]["reference_mode"] == "style"
    # UI 记忆里也不再写这个键（否则旧死键会一直留在 conf/config.json）
    assert "dual_reference" not in analyzer._gpt_pipeline_ui_state()


def test_window_growth_goes_to_queue_and_log_not_option_rows(analyzer):
    """窗口拉高/最大化时，多出来的高度必须给队列 + 日志，不能把选项区每一行撑开。

    用户 2026-09-24 反馈「更新之后界面最大化之后不正常了」：选项区当时是 stretch=1 且没有
    末尾 stretch，于是多余的像素被平均分到每一行上 —— 按钮之间被撑出大片空白、行高被拉高。
    """
    outer = analyzer.layout()
    idx_scroll = outer.indexOf(analyzer.controls_scroll)
    idx_bottom = outer.indexOf(analyzer.bottom_panel)
    assert idx_scroll >= 0 and idx_bottom >= 0
    assert outer.stretch(idx_scroll) == 0        # 选项区按内容取高
    assert outer.stretch(idx_bottom) == 1        # 队列 + 日志吃掉多余空间

    body_layout = analyzer.controls_body.layout()
    assert body_layout.itemAt(body_layout.count() - 1).spacerItem() is not None   # 末尾留白 stretch

    analyzer.resize(1080, 1040)
    analyzer.show()
    QApplication.processEvents()
    assert analyzer.send_btn.height() <= 40                      # 固定 40，不能长高
    assert analyzer.auto_gen_orig_cb.height() <= 24              # 勾选框保持一行
    assert analyzer.controls_scroll.viewport().height() >= 250   # 选项区拿到接近内容的高度
    assert analyzer.log_text.height() > analyzer.log_text.minimumHeight()   # 日志吃掉多余空间
    analyzer.close()


def test_gpt_option_rows_are_compact_and_log_stays_visible(analyzer):
    """gpt 通道只多两行工序/参数；选项区可滚动，队列 + 日志固定在下方（用户反馈日志被挤没了）。"""
    assert analyzer.gpt_pp_row.isHidden() is True
    assert analyzer.gpt_param_row.isHidden() is True
    analyzer.gen_channel_gpt.setChecked(True)
    analyzer.gpt_advanced_toggle.setChecked(True)
    assert analyzer.gpt_pp_row.isHidden() is False
    analyzer.gpt_advanced_toggle.setChecked(True)
    assert analyzer.gpt_param_row.isHidden() is False
    analyzer.gen_channel_gemini.setChecked(True)
    assert analyzer.gpt_param_row.isHidden() is True

    assert analyzer.log_text.minimumHeight() >= 100
    assert analyzer.history_list.minimumHeight() >= 100
    assert analyzer.controls_scroll.widget() is analyzer.controls_body


def test_option_area_does_not_steal_file_drops(analyzer, tmp_path):
    """选项区收进滚动容器后，拖图落点不能变。

    滚动容器（含 viewport）一律**不接受**拖拽：这样拖到选项区的图片会被 Qt 上抛到
    最近的接受拖拽的祖先（本 Widget），和改动前一样能加载。
    """
    from PyQt6.QtCore import QMimeData, QUrl
    from PIL import Image as _PILImage

    assert analyzer.controls_scroll.acceptDrops() is False
    assert analyzer.controls_scroll.viewport().acceptDrops() is False
    assert analyzer.acceptDrops() is True

    image_path = tmp_path / "drop-me.png"
    _PILImage.new("RGB", (12, 12), "white").save(image_path)
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(image_path))])

    class _DropStub:
        def mimeData(self):
            return mime

    analyzer.dropEvent(_DropStub())
    assert os.path.normpath(str(analyzer.image_source)) == os.path.normpath(str(image_path))


def test_queue_lifecycle_from_analysis_to_green(analyzer, monkeypatch):
    """端到端（不调 API）：分析完成 → 显示「进行中·生图/后处理中」→ 生图线程退出 → 绿[已完成]。

    这就是用户要的契约：绿色[完成] == 这个任务的所有工序都跑完了。
    """
    task_id = _seed_record(analyzer)
    analyzer.auto_gen_ref_cb.setChecked(True)
    analysis_thread = _FakeAnalysisThread(task_id, "abc123")
    gen_threads = []

    def _fake_trigger(prompt_type, is_auto=False, prompt_bundle=None, analysis_thread_no=None,
                      auto_group_id=None):
        thread = _FakeGenThread(str(prompt_bundle.get("task_hash") or ""), auto_group_id=auto_group_id)
        analyzer._active_img_threads.append(thread)
        gen_threads.append(thread)
        return True

    monkeypatch.setattr(analyzer, "trigger_image_generation", _fake_trigger)

    analyzer.on_process_finished(analysis_thread, {
        "japanese_title": "白发少女", "english_description": "refined desc",
        "original_english_description": "orig desc", "aspect_ratio": "2:3",
    })
    record = analyzer._analysis_history[task_id]
    assert record["status"] == "running" and record["phase"] == "生图/后处理中"
    assert record["title"] == "白发少女"
    assert len(gen_threads) == 1

    # 分析线程随后退出：记录正在跑管线，兜底不得抢跑
    analyzer._on_analysis_thread_stopped(analysis_thread)
    assert record["status"] == "running" and "兜底" not in record["title"]

    # 生图 + 工序全部结束
    analyzer.on_image_generation_finished(gen_threads[0], ["data/20260924/x-final-rp+sline50.png"])
    analyzer._on_image_thread_stopped(gen_threads[0])
    assert record["status"] == "success" and record["phase"] == ""
    assert record["final_products"] == ["data/20260924/x-final-rp+sline50.png"]


def test_auto_gen_that_never_starts_does_not_hang_queue(analyzer, monkeypatch):
    """勾了自动生图但一个线程都没起来（缺 key 等）→ 记录按「只做分析」收尾，不能吊在「进行中」。"""
    task_id = _seed_record(analyzer)
    analyzer.auto_gen_ref_cb.setChecked(True)
    monkeypatch.setattr(analyzer, "trigger_image_generation", lambda *a, **k: False)
    analysis_thread = _FakeAnalysisThread(task_id, "abc123")

    analyzer.on_process_finished(analysis_thread, {
        "japanese_title": "白发少女", "english_description": "refined desc",
        "original_english_description": "orig desc", "aspect_ratio": "2:3",
    })

    record = analyzer._analysis_history[task_id]
    assert record["status"] == "success" and record["phase"] == ""
    assert not analyzer._auto_gen_groups


def test_failed_pipeline_record_still_offers_analysis_result(analyzer):
    """生图失败（status=error）时，分析结果本身仍然可用：「设为当前结果」不能一起锁死。"""
    task_id = _seed_record(analyzer)
    record = analyzer._analysis_history[task_id]
    record.update({"status": "error", "status_text": "失败",
                   "result_json": {"english_description": "refined desc",
                                   "original_english_description": "orig desc",
                                   "aspect_ratio": "2:3"},
                   "original_prompt": "orig desc", "refined_prompt": "refined desc"})
    assert analyzer._apply_history_record_to_current_state(record) is True
    assert analyzer.current_refine_desc == "refined desc"


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
