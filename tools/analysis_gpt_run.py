# -*- coding: utf-8 -*-
"""分析产物 → gpt-image-2（带画风）→ 后处理工序：**无头命令行版**（与 GUI「分析图片并生图」同一套逻辑）。

例：
  python tools/analysis_gpt_run.py --json "data/20260921/sucai/xxx.json" --style tinkle \
      --quality high --size auto --steps repaint,structure,local --region subject_no_face

流程（调用 GUI 用的同一批函数）：
  1. 读分析 JSON → 短锚内容（`utils.analysis_gen.resolve_content_text`）
  2. 画风 `prompt_gpt` + 画风参考图（`utils.styles`）→ 组请求（`analysis_gen.build_gpt_image_request`）
  3. gpt-image-2 出首图（`api_backend.generate_image_aigc2d_gpt`），尺寸「auto」按输入图比例选
  4. 后处理：重绘（可选双参考）→ 结构线叠加 → 局部重绘+羽化贴回（`utils.post_process.run_pipeline`）
     `--output-dir` 保存完整过程；加 `--publish-final` 时只把最终选中图复制到 `data/<日期>/`
"""
import argparse
import datetime
import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

from utils.refine_quality import file_sha256 as final_sha256  # noqa: E402  （门禁按内容哈希绑定图片）


def _metrics(path):
    try:
        import cv2
        import numpy as np
        sys.path.insert(0, os.path.join(BASE, "tests"))
        import style_render_metrics as m
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return ""
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, int(w * 900 / h)), 900), interpolation=cv2.INTER_AREA)
        tmp = os.path.join(os.path.dirname(path), "_m.png")
        cv2.imwrite(tmp, small)
        cur = m.analyze(tmp)
        os.remove(tmp)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1].mean()
        return (f"亮度 {gray.mean():.1f} 饱和 {sat:.1f} 段长 {cur['line_avg_len']:.1f} "
                f"端点 {cur['line_endpoint_density']:.2f} 长线 {cur['line_long_ratio']:.3f}")
    except Exception:  # noqa: BLE001
        return ""


def run_app_batch(cases_path, text_model=""):
    """Run an explicit test manifest through app.py's actual analysis tab commands."""
    os.environ["IMAGE_MAKER_TEST_OUTPUT"] = "1"
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if text_model:
        from utils import analysis_gpt_prompt
        original_config = analysis_gpt_prompt.load_text_api_config
        analysis_gpt_prompt.load_text_api_config = lambda *args, **kwargs: {
            **original_config(*args, **kwargs), "model": text_model}
    from utils import llm_retry
    llm_retry.load_retry_settings = lambda *args, **kwargs: llm_retry.RetrySettings(True, 2, 15)
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer
    from app import AppWindow
    cases_path = os.path.abspath(cases_path)
    output_dir = os.path.dirname(cases_path)
    cases = json.load(open(cases_path, encoding="utf-8"))
    qt = QApplication.instance() or QApplication([])
    window = AppWindow()
    if text_model:
        window.model_combo.blockSignals(True)
        window.model_combo.setCurrentText(text_model)
        window.model_combo.blockSignals(False)
    tab = window.single_analyzer_tab
    # Test overrides are in-memory and must not change saved user preferences.
    for widget, value in ((tab.remove_photo_style_cb, True), (tab.save_to_source_dir_cb, False),
                          (tab.auto_gen_orig_cb, False), (tab.auto_gen_ref_cb, False),
                          (tab.gen_channel_gpt, True), (tab.gpt_pp_repaint, True),
                          (tab.gpt_pp_structure, False), (tab.gpt_pp_local, False),
                          (tab.gpt_pp_tone, False), (tab.gpt_pp_ink, False)):
        widget.blockSignals(True)
        widget.setChecked(value)
        widget.blockSignals(False)
    tab.get_timeout_seconds = lambda: 300
    tab._send_system_notification = lambda *args, **kwargs: None
    log = open(os.path.join(output_dir, "app-batch.log"), "a", encoding="utf-8")
    original_log = tab.log_msg
    def log_message(text, *args, **kwargs):
        log.write(str(text) + "\n")
        log.flush()
        print(str(text)[:450], flush=True)
        return original_log(text, *args, **kwargs)
    tab.log_msg = log_message
    results = []
    state = {"index": 0, "thread": None}
    def tick():
        if state["thread"] is not None:
            if tab._active_analysis_threads or tab._pipeline_busy():
                return
            thread = state["thread"]
            record = dict(tab._analysis_history.get(thread.meta_task_id) or {})
            results.append(dict(cases[state["index"]], analysis_status=thread.last_status,
                                task_hash=thread.meta_task_hash, record=record))
            with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as stream:
                json.dump(results, stream, ensure_ascii=False, indent=2, default=str)
            state["thread"] = None
            state["index"] += 1
        if state["index"] >= len(cases):
            timer.stop()
            log.close()
            qt.quit()
            return
        case = cases[state["index"]]
        if not os.path.isfile(case["source"]):
            raise FileNotFoundError(case["source"])
        if tab.main_style_combo.findText(case["style"]) < 0:
            raise ValueError("画风不在 enabled 列表: " + case["style"])
        tab.main_style_combo.blockSignals(True)
        tab.main_style_combo.setCurrentText(case["style"])
        tab.main_style_combo.blockSignals(False)
        tab.image_source = case["source"]
        state["thread"] = tab._launch_analysis_task(
            case["source"], gen_targets=["refined"],
            header_note=f"随机全链路测试 {state['index'] + 1}/{len(cases)}: {case['style']}")
        if state["thread"] is None:
            raise RuntimeError("app 分析命令未启动")
    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(1000)
    QTimer.singleShot(0, tick)
    qt.exec()
    return 0 if all(r["record"].get("final_products") and not r["record"].get("pipeline_error")
                    for r in results) else 1


def resume_app_checkpoint(checkpoint_path, text_model="", test_output=True):
    """Resume the GUI worker's cached stages without repeating analysis or paid images."""
    if test_output:
        os.environ["IMAGE_MAKER_TEST_OUTPUT"] = "1"
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if text_model:
        from utils import analysis_gpt_prompt
        original_config = analysis_gpt_prompt.load_text_api_config
        analysis_gpt_prompt.load_text_api_config = lambda *args, **kwargs: {
            **original_config(*args, **kwargs), "model": text_model}
    from PyQt6.QtWidgets import QApplication
    from modules.image_analysis.single_analyzer import GptImageGenWorkerThread
    qt = QApplication.instance() or QApplication([])
    data = json.load(open(checkpoint_path, encoding="utf-8"))
    worker = GptImageGenWorkerThread(**data["snapshot"], checkpoint_path=checkpoint_path)
    outputs = []
    worker.finish_signal.connect(outputs.extend)
    worker.log_signal.connect(print)
    worker.run()
    result = {"status": worker.last_status, "outputs": outputs,
              "checkpoint": os.path.abspath(checkpoint_path), "failed_stage": worker.failed_stage,
              "pickup_source": data.get("pickup_source", "")}
    folder = os.path.dirname(os.path.abspath(checkpoint_path))
    attempt_log = os.path.join(folder, "resume-result-" + Path(checkpoint_path).stem + ".json")
    for result_path in (attempt_log, os.path.join(folder, "resume-result.json")):
        with open(result_path, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    print(f"续跑结果日志: {attempt_log}", flush=True)
    return 0 if worker.last_status == "success" else 1


def load_styles(styles_file: str = "") -> dict:
    """读画风表：默认 `conf/config-styles.json`；给了 `styles_file` 就用它整份替换。

    受控实验（`data/exp/*`）要验证「只改一个画法字段」的因果效果，又不能动正式配置：
    调用方把「正式条目 + 一个预先声明的改动」另存成一份实验画风表，用 `--styles-file` 传进来。
    这里只是换读取来源，组合逻辑仍由 `utils.analysis_gen` / `utils.style_gpt` 统一决定。
    """
    path = os.path.abspath(styles_file) if styles_file else os.path.join(BASE, "conf", "config-styles.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def request_capture(dump_dir: str):
    """在 fetch 点旁路记录 Gemini 生图线程的真实构造参数，供受控实验核对请求快照。

    返回 ``(restore, count)``；`dump_dir` 为空时不改动任何行为。
    """
    if not dump_dir:
        return (lambda: None), (lambda: 0)
    from modules.image_analysis import single_analyzer as _sa
    real_thread_cls = _sa.ImageGenWorkerThread
    captured = []

    class _CapturingThread(real_thread_cls):  # type: ignore[misc, valid-type]
        def __init__(self, prompt, model_name, aspect_ratio, instructions, *a, **kw):
            snapshot = {"kind": "gemini-image-request", "prompt": prompt,
                        "model": model_name, "aspect_ratio": aspect_ratio,
                        "instructions": instructions,
                        "post_instructions": kw.get("post_instructions") or "",
                        "image_paths": list(kw.get("image_paths") or []),
                        "file_prefix": kw.get("file_prefix") or "",
                        "api_type": kw.get("api_type") or ""}
            captured.append(snapshot)
            os.makedirs(dump_dir, exist_ok=True)
            with open(os.path.join(dump_dir, f"gemini-request-{len(captured)}.json"),
                      "w", encoding="utf-8") as stream:
                json.dump(snapshot, stream, ensure_ascii=False, indent=2)
            print(f"[请求快照] Gemini 生图请求 {len(captured)} 已落盘到 {dump_dir}", flush=True)
            super().__init__(prompt, model_name, aspect_ratio, instructions, *a, **kw)

    _sa.ImageGenWorkerThread = _CapturingThread

    def restore():
        _sa.ImageGenWorkerThread = real_thread_cls

    return restore, (lambda: len(captured))


def _relocate_logged_outputs(tab, out_dir: str) -> list:
    """把这次运行写出的图片从 `data/<日期>/` 搬进实验目录，返回新路径。

    受控实验要求产物留在实验目录（PROTOCOL.md §5「不向 data/<日期>/ 发布」）。
    Gemini 直出的落盘位置由 `_resolve_save_dir` 决定，这里按本次日志里的「保存路径」行搬家。
    """
    import re
    import shutil

    candidates = []
    for record in (getattr(tab, "_analysis_history", None) or {}).values():
        candidates.extend(record.get("outputs") or [])
    candidates.extend(getattr(tab, "_cli_generated_files", []) or [])
    moved = []
    for path in candidates:
        path = str(path or "")
        if not path or not os.path.isfile(path):
            continue
        target = os.path.join(out_dir, os.path.basename(path))
        try:
            if os.path.abspath(path) != os.path.abspath(target):
                shutil.move(path, target)
            moved.append(target)
        except OSError as exc:
            print(f"  ⚠️ 产物搬家失败（保留原路径）: {exc}", flush=True)
            moved.append(os.path.abspath(path))
    return moved


def generate_from_fixed_analysis(sources, options, tab, image_config, styles,
                                request_dump="", out_dir="", saved_json_path=""):
    """用**固定分析 JSON** 跑 Gemini 直出：不重新分析、不写新的分析产物。

    受控实验要求 control 与 variant 共用同一份分析 JSON（同一段内容锚、同一个画幅），
    所以这里直接把 JSON 当成产物读入，构建与「分析完成后自动生图」完全一样的
    prompt_bundle（`original_english_description` / `english_description` / `task_hash`
    / `aspect_ratio`），再走 `trigger_image_generation`。输出文件名沿用原 task hash，
    与投稿 JSON 仍能对上。
    """
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer

    analysis_json = os.path.abspath(options["analysis_json"])
    with open(analysis_json, encoding="utf-8") as stream:
        result = json.load(stream)
    bundle = {
        "task_hash": str(result.get("task_hash") or "").strip(),
        "style_name": options.get("style") or "",
        "aspect_ratio": str(result.get("aspect_ratio") or "").strip(),
        "original_prompt": str(result.get("original_english_description") or "").strip(),
        "refined_prompt": str(result.get("english_description") or "").strip(),
        "analysis_json_path": analysis_json,
        "source_image_path": os.path.abspath(sources[0]),
    }
    if not bundle["refined_prompt"] and not bundle["original_prompt"]:
        raise ValueError("固定分析 JSON 里没有可用的提示词字段")
    generation = options["generate"]
    targets = ["original", "refined"] if generation == "both" else [generation]
    targets = [t for t in targets if t in ("original", "refined")]
    print(json.dumps({"mode": "fixed-analysis-json", "analysis_json": analysis_json,
                      "task_hash": bundle["task_hash"], "targets": targets,
                      "style": bundle["style_name"], "aspect_ratio": bundle["aspect_ratio"]},
                     ensure_ascii=False, indent=2), flush=True)
    qt = QApplication.instance() or QApplication([])
    original_log = tab.log_msg

    def print_log(message, *args, **kwargs):
        print(str(message)[:700], flush=True)
        return original_log(message, *args, **kwargs)

    tab.log_msg = print_log
    tab._cli_generated_files = []
    tab._last_saved_json_path = saved_json_path or ""
    _emit = tab.on_image_generation_finished

    def _capture_outputs(thread, saved_files):
        # Gemini 直出恒落 `data/<日期>/`；这里收到信号就立刻搬进实验目录，
        # 免得后续（含看门狗强制退出）拿不到产物路径。
        relocated = []
        for path in (saved_files or []):
            target = path
            if out_dir and path and os.path.isfile(path):
                candidate = os.path.join(out_dir, os.path.basename(path))
                try:
                    if os.path.abspath(path) != os.path.abspath(candidate):
                        shutil.move(path, candidate)
                    target = candidate
                except OSError as exc:
                    print(f"  ⚠️ 产物搬家失败（保留原路径）: {exc}", flush=True)
            if target not in tab._cli_generated_files:
                tab._cli_generated_files.append(target)
            relocated.append(target)
        return _emit(thread, relocated)

    tab.on_image_generation_finished = _capture_outputs
    # 产物落到 --output-dir（受控实验目录），不写 data/<日期>/；
    # 「保存到原图同目录」必须开着，否则 _resolve_save_dir 会回落到 data/<日期>/。
    save_cb = getattr(tab, "save_to_source_dir_cb", None)
    previous_save_flag = save_cb.isChecked() if save_cb is not None else None
    if save_cb is not None:
        old = save_cb.blockSignals(True)
        save_cb.setChecked(True)
        save_cb.blockSignals(old)
    restore, count = request_capture(request_dump)
    started = []
    try:
        for source, prompt_type in ((os.path.abspath(p), t) for p in sources for t in targets):
            tab.image_source = source
            if tab.trigger_image_generation(prompt_type, is_auto=False, prompt_bundle=bundle):
                started.append((source, prompt_type))
        if not started:
            print("❌ 固定 JSON 生图没有启动任何任务（生图 API Key 或提示词为空）", flush=True)
            return 1
        # 不用 QTimer：无头模式下 Qt 事件循环偶尔不会被计时器唤醒（进程一直挂着）。
        # 改成显式 processEvents 轮询，超时用 os._exit 硬收尾，CLI 一定结束。
        started_at = time.time()
        deadline = started_at + max(240.0, float(options.get("timeout") or 300) * 4)
        idle_hits = 0
        seen_busy = False
        last_note = 0.0
        ok = False
        while True:
            qt.processEvents()
            busy = bool(tab._active_img_threads) or bool(tab._pipeline_busy())
            if busy:
                seen_busy = True
                idle_hits = 0
            else:
                idle_hits += 1
                if seen_busy and idle_hits * 0.5 >= 3:
                    print("✅ 固定 JSON 生图线程与后处理已收尾", flush=True)
                    ok = True
                    break
                if not seen_busy and time.time() - started_at > 60:
                    print("❌ 固定 JSON 生图既没有启动线程也没有产物", flush=True)
                    break
            if time.time() - started_at > 300 and time.time() - last_note > 60:
                last_note = time.time()
                print(f"    …等待生图/后处理线程结束（{int(time.time() - started_at)}s，"
                      f"busy={busy}）", flush=True)
            if time.time() > deadline:
                ok = bool(getattr(tab, "_cli_generated_files", []))
                print("⏱ 固定 JSON 生图等待超时，按当前产物收尾（不记为全链成功）", flush=True)
                break
            time.sleep(0.5)
        if not ok:
            # 已经不指望线程正常收尾：直接结束进程，避免 CLI 卡在那里
            restore()
            tab.on_image_generation_finished = _emit
            products = [p for p in (getattr(tab, "_cli_generated_files", []) or []) if p]
            print(json.dumps({"source": os.path.abspath(sources[0]), "status": "error",
                              "task_hash": bundle["task_hash"], "analysis_json": analysis_json,
                              "outputs": products, "error": "固定 JSON 生图未能在预算内收尾"},
                             ensure_ascii=False, default=str), flush=True)
            sys.stdout.flush()
            os._exit(1)
    finally:
        restore()
        tab.on_image_generation_finished = _emit
        if save_cb is not None and previous_save_flag is not None:
            old = save_cb.blockSignals(True)
            save_cb.setChecked(previous_save_flag)
            save_cb.blockSignals(old)
        if request_dump:
            print(f"[请求快照] Gemini 生图请求共 {count()} 份，目录 {request_dump}", flush=True)
    products = []
    for record in (getattr(tab, "_analysis_history", None) or {}).values():
        if str(record.get("task_hash") or "") == bundle["task_hash"]:
            products.extend(p for p in (record.get("final_products") or []) if p)
    if not products and out_dir:
        products = _relocate_logged_outputs(tab, out_dir)
    products = [p for p in products if p]
    print(json.dumps({"source": os.path.abspath(sources[0]), "status": "success" if products else "error",
                      "task_hash": bundle["task_hash"], "analysis_json": analysis_json,
                      "outputs": products, "error": "" if products else "没有最终产物"},
                     ensure_ascii=False, default=str), flush=True)
    return 0 if products else 1


def apply_analysis_tab_options(tab, options):
    """把 CLI 参数映射到真实分析 Tab；所有赋值不改用户保存的 GUI 偏好。"""
    from utils.styles import ref_image_valid, style_ref_image

    def checked(name, value):
        widget = getattr(tab, name)
        old = widget.blockSignals(True)
        try:
            widget.setChecked(bool(value))
        finally:
            widget.blockSignals(old)

    def selected(name, value):
        widget = getattr(tab, name)
        if name == "gpt_pp_region" and value == "stable-four":
            from modules.image_analysis.single_analyzer import STABLE_LOCAL_REGIONS
            value = list(STABLE_LOCAL_REGIONS)
        index = widget.findData(value)
        if index < 0:
            raise ValueError(f"{name} 不支持选项 {value!r}")
        old = widget.blockSignals(True)
        try:
            widget.setCurrentIndex(index)
        finally:
            widget.blockSignals(old)

    style = options.get("style")
    if style:
        index = tab.main_style_combo.findText(style)
        if index < 0:
            raise ValueError(f"画风不在启用列表: {style}")
        old = tab.main_style_combo.blockSignals(True)
        tab.main_style_combo.setCurrentIndex(index)
        tab.main_style_combo.blockSignals(old)
    style = tab.main_style_combo.currentText()
    has_ref = ref_image_valid(style_ref_image(tab.get_styles(), style))
    mode = options.get("reference_mode", "off")
    if mode != "off" and not has_ref:
        raise ValueError(f"画风 {style!r} 没有可用参考图，不能使用 {mode} 模式")
    tab.style_ref_mode_combo.set_mode(mode, has_ref=has_ref)
    for key, widget in (("nsfw", "use_nsfw_cb"), ("outfit_check", "enable_outfit_check_cb"),
                        ("remove_photo_style", "remove_photo_style_cb"),
                        ("save_to_source", "save_to_source_dir_cb"),
                        ("jpg_upscale", "enable_jpg_upscale_cb"),
                        ("repaint", "gpt_pp_repaint"), ("structure", "gpt_pp_structure"),
                        ("local", "gpt_pp_local"), ("tone", "gpt_pp_tone"),
                        ("ink", "gpt_pp_ink"), ("size_follow_input", "gpt_size_follow_cb")):
        checked(widget, options[key])
    checked("gen_channel_gpt" if options["channel"] == "gpt" else "gen_channel_gemini", True)
    for key, widget in (("quality", "gpt_quality_combo"),
                        ("region", "gpt_pp_region"), ("scope", "gpt_pp_scope"),
                        ("first_pass_mode", "gpt_first_pass_mode"),
                        ("tone_target", "gpt_pp_tone_target")):
        selected(widget, options[key])
    tab.outfit_style_combo.blockSignals(True)
    tab.outfit_style_combo.setCurrentText(options.get("outfit_style", ""))
    tab.outfit_style_combo.blockSignals(False)
    if options.get("jpg_upscale"):
        model = options.get("upscale_model") or ""
        if model and tab.upscale_model_combo.findText(model) < 0:
            raise ValueError(f"未找到 JPG 处理模型: {model}")
        if model:
            tab.upscale_model_combo.setCurrentText(model)
    for key, widget in (("upscale_by", tab.upscale_by_spin),
                        ("webp_target_mb", tab.webp_target_mb_spin)):
        old = widget.blockSignals(True)
        widget.setValue(options[key])
        widget.blockSignals(old)
    tab._on_gen_channel_changed()
    plan = {"style": style, "reference_mode": tab.style_ref_mode_combo.selected_mode(),
            "channel": options["channel"], "upscale": tab._collect_upscale_options()}
    if options["channel"] == "gpt":
        plan["gpt_steps"] = tab._build_gpt_image_steps()
    return plan


def run_app_workflow(sources, options):
    """无头运行真实 UI 的单图分析 + 自动生图链，保留门禁/断点/发布语义。"""
    if options.get("test_output"):
        os.environ["IMAGE_MAKER_TEST_OUTPUT"] = "1"
    from utils.output_isolation import resolve_output_target
    log_dir, log_name = resolve_output_target(
        os.path.join("data", datetime.datetime.now().strftime("%Y%m%d")),
        "analysis-cli-results-" + datetime.datetime.now().strftime("%H%M%S") +
        "-" + uuid.uuid4().hex[:6] + ".json")
    result_log = os.path.abspath(os.path.join(log_dir, log_name))
    run_record = {"version": 1, "kind": "analysis-cli-results", "items": [],
                  "sources": [os.path.abspath(p) for p in sources], "status": "running",
                  "started_at": datetime.datetime.now().isoformat(timespec="seconds")}
    def save_result_log():
        # 走重试封装：D 盘 tmp->target 的 os.replace 会偶发 WinError 5，
        # 一旦在收尾时抛出，整个 CLI 会以 exit 1 结束（图其实已经好了）。
        from utils.atomic_io import write_json_atomic

        write_json_atomic(result_log, run_record, indent=2, ensure_ascii=False, default=str)
    if not options.get("dry_run"):
        os.makedirs(log_dir, exist_ok=True)
        save_result_log()
        print(f"结果日志: {result_log}", flush=True)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer
    from modules.image_analysis.single_analyzer import (SingleAnalyzerWidget,
        get_single_analyzer_missing_prompt_files)
    from modules.others.api_backend import (get_api_config, load_config, resolve_text_api_key,
                                            resolve_nsfw_api_key)
    from utils.styles import enabled_style_names

    qt = QApplication.instance() or QApplication([])
    config = load_config()
    styles = load_styles(options.get("styles_file") or "")
    options = dict(options)
    if not options.get("style"):
        saved_style = str(config.get("last_used_style") or "")
        if saved_style in enabled_style_names(styles):
            options["style"] = saved_style
    def text_config(nsfw=False):
        if nsfw:
            return (config.get("nsfw_base_url") or config.get("base_url") or "",
                    resolve_nsfw_api_key(config),
                    options.get("text_model") or config.get("nsfw_model") or config.get("model") or "")
        return (config.get("base_url") or "", resolve_text_api_key(config),
                options.get("text_model") or config.get("model") or "")
    image_api = options.get("image_api") or (
        "aigc-2d-gpt" if options["channel"] == "gpt" else config.get("current_api") or "aigc2d")
    def image_config():
        cfg = get_api_config(api_type=image_api)
        return (cfg.get("base_url") or "", cfg.get("api_key") or "",
                options.get("image_model") or cfg.get("model") or "", image_api)
    tab = SingleAnalyzerWidget(
        config_getter_func=text_config, img_config_getter_func=image_config,
        styles_getter_func=lambda: styles, save_img_cfg_callback=lambda: None,
        ar_policy_getter_func=lambda: {"override_first": options["aspect_ratio_first"],
                                       "override_second": options["aspect_ratio_second"]},
        booru_tag_limit_getter_func=lambda: options["booru_tag_limit"],
        timeout_getter_func=lambda: options["timeout"],
        upscale_options_getter_func=lambda: config.get("upscale_options") or {},
        persist_ui_state=False,
    )
    tab.update_styles(enabled_style_names(styles))
    tab._send_system_notification = lambda *_args, **_kwargs: None
    plan = apply_analysis_tab_options(tab, options)
    generation = options["generate"]
    fixed_json = str(options.get("analysis_json") or "")
    if fixed_json:
        # 受控实验路径：固定分析 JSON → 只生图，不重新分析（不产生新的分析产物）。
        if generation == "none":
            raise ValueError("--analysis-json 用于固定 JSON 生图；要只分析请去掉该参数")
        plan = dict(plan, mode="fixed-analysis-json", analysis_json=os.path.abspath(fixed_json),
                    targets=["original", "refined"] if generation == "both" else [generation])
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        if options.get("dry_run"):
            return 0
        if not str(options.get("style") or ""):
            raise ValueError("--analysis-json 模式必须显式指定 --style")
        if not os.path.isfile(fixed_json):
            raise FileNotFoundError(fixed_json)
        _, image_key, _, _ = tab.get_img_config()
        if not image_key:
            raise ValueError("生图 API Key 缺失；请配置 .env 或 conf/config.json")
        try:
            return generate_from_fixed_analysis(
                sources, options, tab, image_config, styles,
                request_dump=str(options.get("request_dump") or ""),
                out_dir=str(options.get("output_dir") or ""),
                saved_json_path=os.path.abspath(fixed_json))
        finally:
            tab.close()
    if options["save_to_source"] and generation != "none":
        raise ValueError("--save-to-source 仅分析；请同时指定 --generate none")
    if options["booru_tag_limit"] <= 0:
        raise ValueError("--booru-tag-limit 必须大于零")
    if not 1.0 <= options["upscale_by"] <= 8.0 or not 0.1 <= options["webp_target_mb"] <= 100.0:
        raise ValueError("JPG 处理参数超出 UI 支持范围")
    missing = get_single_analyzer_missing_prompt_files(
        enable_refine=True,
        enable_outfit_check=options["outfit_check"],
        enable_remove_photo_style=options["remove_photo_style"],
        enable_recompute_pixiv_tags=options["outfit_check"] or options["remove_photo_style"],
    )
    if missing:
        raise ValueError("缺少分析提示词文件: " + ", ".join(missing))
    if generation != "none" and not options.get("dry_run"):
        _, image_key, _, _ = tab.get_img_config()
        if not image_key:
            raise ValueError("生图 API Key 缺失；请配置 .env 或 conf/config.json")
    if not options.get("dry_run"):
        _, text_key, text_model = tab.get_text_config(options["nsfw"])
        if not text_key or not text_model:
            raise ValueError("文本分析 API Key 或模型缺失")
    plan.update({"sources": [os.path.abspath(p) for p in sources],
                 "generate": generation, "timeout": options["timeout"],
                 "dry_run": bool(options.get("dry_run"))})
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if options.get("dry_run"):
        return 0

    # 使用 gen_targets 强制目标，避免 GUI 记忆中的自动生图勾选改变 CLI 行为。
    targets = [] if generation == "none" else (["original", "refined"] if generation == "both" else [generation])
    for widget in (tab.auto_gen_orig_cb, tab.auto_gen_ref_cb):
        old = widget.blockSignals(True)
        widget.setChecked(False)
        widget.blockSignals(old)
    original_log = tab.log_msg
    def print_log(message, *args, **kwargs):
        print(str(message)[:700], flush=True)
        return original_log(message, *args, **kwargs)
    tab.log_msg = print_log
    # Gemini 通道的请求快照：生图线程构造参数就是真正发给接口的 (prompt / instructions /
    # post_instructions / 参考图 / 比例 / 模型)，在 fetch 点旁路落盘，供受控实验核对
    # 「control 与 variant 只差一个位置」。未指定 --request-dump 时行为不变。
    dump_dir = str(options.get("request_dump") or "")
    if dump_dir:
        from modules.image_analysis import single_analyzer as _sa
        real_thread_cls = _sa.ImageGenWorkerThread
        captured = []

        class _CapturingThread(real_thread_cls):  # type: ignore[misc, valid-type]
            def __init__(self, prompt, model_name, aspect_ratio, instructions, *a, **kw):
                snapshot = {"kind": "gemini-image-request", "prompt": prompt,
                            "model": model_name, "aspect_ratio": aspect_ratio,
                            "instructions": instructions,
                            "post_instructions": kw.get("post_instructions") or "",
                            "image_paths": list(kw.get("image_paths") or []),
                            "file_prefix": kw.get("file_prefix") or "",
                            "api_type": kw.get("api_type") or ""}
                captured.append(snapshot)
                os.makedirs(dump_dir, exist_ok=True)
                with open(os.path.join(dump_dir, f"gemini-request-{len(captured)}.json"),
                          "w", encoding="utf-8") as stream:
                    json.dump(snapshot, stream, ensure_ascii=False, indent=2)
                print(f"[请求快照] Gemini 生图请求 {len(captured)} 已落盘到 {dump_dir}", flush=True)
                super().__init__(prompt, model_name, aspect_ratio, instructions, *a, **kw)

        _sa.ImageGenWorkerThread = _CapturingThread
    else:
        _sa = None
        real_thread_cls = None
    results = []
    import signal
    state = {"index": 0, "thread": None, "cancelled": False, "cancel_sent": False}
    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda _sig, _frame: state.update(cancelled=True))
    timer = QTimer()
    def tick():
        if state["cancelled"]:
            if not state["cancel_sent"]:
                state["cancel_sent"] = True
                tab.cancel_active_analysis_tasks()
                if tab._active_img_threads:
                    tab.cancel_image_generation()
            if not tab._active_analysis_threads and not tab._pipeline_busy():
                timer.stop()
                qt.quit()
            return
        if state["thread"] is not None:
            if tab._active_analysis_threads or tab._pipeline_busy():
                return
            thread = state["thread"]
            record = dict(tab._analysis_history.get(thread.meta_task_id) or {})
            result = {"source": sources[state["index"]], "status": record.get("status"),
                      "task_hash": record.get("task_hash"),
                      "analysis_json": record.get("saved_json_path"),
                      "prompt_files": record.get("saved_prompt_paths") or [],
                      "checkpoints": record.get("generation_checkpoints") or [],
                      "outputs": record.get("final_products") or [],
                      "error": record.get("pipeline_error") or ""}
            results.append(result)
            run_record["items"].append(result)
            save_result_log()
            print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
            state["thread"] = None
            state["index"] += 1
        if state["index"] >= len(sources):
            timer.stop()
            qt.quit()
            return
        source = os.path.abspath(sources[state["index"]])
        tab.image_source = source
        state["thread"] = tab._launch_analysis_task(source, gen_targets=targets,
                                                    header_note=f"CLI {state['index'] + 1}/{len(sources)}")
        if state["thread"] is None:
            failed = {"source": source, "status": "error", "error": "分析预检未通过"}
            results.append(failed)
            run_record["items"].append(failed)
            save_result_log()
            state["index"] += 1
    timer.timeout.connect(tick)
    timer.start(500)
    QTimer.singleShot(0, tick)
    try:
        qt.exec()
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        if _sa is not None and real_thread_cls is not None:
            _sa.ImageGenWorkerThread = real_thread_cls
        if dump_dir:
            print(f"[请求快照] Gemini 生图请求共 {len(captured)} 份，目录 {dump_dir}", flush=True)
    if state["cancelled"]:
        exit_code = 130
    else:
        exit_code = 0 if len(results) == len(sources) and all(
            r.get("status") == "success" and (generation == "none" or r.get("outputs")) and not r.get("error")
            for r in results) else 1
    run_record["status"] = "cancelled" if exit_code == 130 else "success" if exit_code == 0 else "error"
    run_record["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    save_result_log()
    print(f"结果日志: {result_log}", flush=True)
    return exit_code


def _history_entries(target):
    """Resolve a CLI result log, checkpoint, request manifest, or task folder."""
    target = os.path.abspath(target)
    if os.path.isdir(target):
        checkpoints = sorted(Path(target).rglob("generation-checkpoint*.json"))
        if checkpoints:
            return [{"checkpoint": str(path), "source": str(path.parent), "status": ""}
                    for path in checkpoints]
        request = Path(target) / "request.json"
        if request.is_file():
            return [{"manifest": str(request), "source": target, "status": ""}]
        raise ValueError("目录中没有 generation-checkpoint*.json 或 request.json")
    if not os.path.isfile(target):
        raise FileNotFoundError(target)
    with open(target, encoding="utf-8") as stream:
        data = json.load(stream)
    if isinstance(data, dict) and isinstance(data.get("stages"), dict) and data.get("snapshot"):
        return [{"checkpoint": target, "source": os.path.dirname(target),
                 "status": data.get("status", "")}]
    if isinstance(data, dict) and data.get("checkpoint") and "status" in data:
        return [{"checkpoint": str(data["checkpoint"]),
                 "source": data.get("pickup_source") or os.path.dirname(target),
                 "status": data.get("status", "")}]
    if isinstance(data, dict) and data.get("analysis_json") and isinstance(data.get("steps"), dict):
        return [{"manifest": target, "source": data.get("source", ""),
                 "status": data.get("status", "")}]
    items = data.get("items") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("不是 CLI 结果日志、工序断点或实验请求清单")
    entries = []
    for item in items:
        if not isinstance(item, dict):
            continue
        record = item.get("record") if isinstance(item.get("record"), dict) else item
        paths = record.get("checkpoints") or record.get("generation_checkpoints") or []
        common = {"source": item.get("source", ""), "status": record.get("status", ""),
                  "task_hash": record.get("task_hash", ""),
                  "analysis_json": record.get("analysis_json") or record.get("saved_json_path") or "",
                  "outputs": record.get("outputs") or record.get("final_products") or [],
                  "prompt_files": record.get("prompt_files") or record.get("saved_prompt_paths") or [],
                  "error": record.get("error") or record.get("pipeline_error") or ""}
        for path in paths:
            if not os.path.isabs(path):
                path = os.path.abspath(path) if os.path.isfile(path) else os.path.join(os.path.dirname(target), path)
            entries.append({"checkpoint": path, **common})
        if not paths:
            entries.append(common)
    return entries


def _history_date_dir(path):
    for parent in Path(path).resolve().parents:
        if (len(parent.name) == 8 and parent.name.isdigit()
                and parent.parent.name in ("data", "test-result")):
            return str(parent)
    return ""


def run_history_command(action, target, task=1, pick=0, json_output=False):
    """List saved process data, resume from chosen pixels, or publish them."""
    from modules.image_analysis.history_pickup import list_process_artifacts, branch_from_artifact
    from utils.analysis_gen import publish_final_output

    entries = _history_entries(target)
    if not entries:
        raise ValueError("结果日志没有任务")
    details = []
    for number, entry in enumerate(entries, 1):
        path = entry.get("checkpoint") or entry.get("manifest")
        artifacts = list_process_artifacts(path) if path and os.path.isfile(path) else []
        details.append({"task": number, **entry,
                        "artifacts": [{"pick": i, **item} for i, item in enumerate(artifacts, 1)]})
    if action == "list":
        if json_output:
            print(json.dumps(details, ensure_ascii=False, indent=2))
        else:
            for entry in details:
                print(f"任务 {entry['task']}  {entry.get('status') or '-'}  "
                      f"{entry.get('task_hash') or '-'}  {entry.get('source') or '-'}")
                print(f"  断点/清单: {entry.get('checkpoint') or entry.get('manifest') or '(无)'}")
                if entry.get("analysis_json"):
                    print(f"  分析 JSON: {entry['analysis_json']}")
                if entry.get("outputs"):
                    print(f"  最终产物: {', '.join(entry['outputs'])}")
                if entry.get("error"):
                    print(f"  错误: {entry['error']}")
                for item in entry["artifacts"]:
                    timestamp = datetime.datetime.fromtimestamp(item["mtime"]).strftime("%Y-%m-%d %H:%M:%S")
                    marker = "图" if item["image"] else "文件"
                    print(f"  {item['pick']:>2} [{marker}] {timestamp}  {item['label']}  {item['path']}")
        return 0
    if not 1 <= task <= len(details):
        raise ValueError(f"--task 须为 1..{len(details)}")
    entry = details[task - 1]
    if not 1 <= pick <= len(entry["artifacts"]):
        raise ValueError("先用 --history-list 查看编号，再传 --pick <编号>")
    artifact = entry["artifacts"][pick - 1]
    if not artifact["image"] or not os.path.isfile(artifact["path"]):
        raise ValueError("--pick 必须指向仍存在的过程图")
    if action == "resume":
        if not entry.get("checkpoint"):
            raise ValueError("实验 request.json 没有逐阶段断点；请用完整分析 CLI 产出的 generation-checkpoint.json 续跑")
        branch, next_stage = branch_from_artifact(entry["checkpoint"], artifact["path"], artifact["stage"])
        print(f"新断点: {branch}\n续跑工序: {next_stage}", flush=True)
        is_test = "test-result" in {part.lower() for part in Path(branch).parts}
        return resume_app_checkpoint(branch, test_output=is_test)
    path = entry.get("checkpoint") or entry.get("manifest")
    with open(path, encoding="utf-8") as stream:
        data = json.load(stream)
    snapshot = data.get("snapshot") or {}
    style = (snapshot.get("request_payload") or {}).get("style_name") or data.get("style", "")
    task_hash = entry.get("task_hash") or snapshot.get("file_prefix", "")
    process_dir = os.path.dirname(path)
    analysis_json = str(data.get("analysis_json") or "")
    if not task_hash and analysis_json and os.path.isfile(analysis_json):
        with open(analysis_json, encoding="utf-8") as stream:
            task_hash = str((json.load(stream) or {}).get("task_hash") or "")
    published = publish_final_output(
        artifact["path"], style_name=style, process_dir=process_dir,
        final_dir=_history_date_dir(path) or _history_date_dir(analysis_json),
        task_hash=task_hash, prefer_symlink=True)
    print(f"人工发布: {published} ({'软链' if os.path.islink(published) else '副本'})", flush=True)
    return 0


def main():
    if any(flag in sys.argv for flag in ("--history-list", "--history-resume", "--history-publish")):
        history = argparse.ArgumentParser(description="列出 CLI 结果历史，选图续跑或人工发布")
        actions = history.add_mutually_exclusive_group(required=True)
        actions.add_argument("--history-list", metavar="日志/断点/目录")
        actions.add_argument("--history-resume", metavar="日志/断点/目录")
        actions.add_argument("--history-publish", metavar="日志/断点/目录")
        history.add_argument("--task", type=int, default=1, help="结果日志中的任务编号（从 1 开始）")
        history.add_argument("--pick", type=int, default=0, help="--history-list 显示的过程文件编号")
        history.add_argument("--history-json", action="store_true", help="list 输出结构化 JSON")
        args = history.parse_args()
        action = "list" if args.history_list else "resume" if args.history_resume else "publish"
        target = args.history_list or args.history_resume or args.history_publish
        try:
            return run_history_command(action, target, args.task, args.pick, args.history_json)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"历史操作失败: {exc}", file=sys.stderr, flush=True)
            return 1
    if "--app-resume-checkpoint" in sys.argv:
        index = sys.argv.index("--app-resume-checkpoint")
        text_model = sys.argv[sys.argv.index("--text-model") + 1] if "--text-model" in sys.argv else ""
        return resume_app_checkpoint(sys.argv[index + 1], text_model)
    if "--app-batch-cases" in sys.argv:
        index = sys.argv.index("--app-batch-cases")
        text_model = sys.argv[sys.argv.index("--text-model") + 1] if "--text-model" in sys.argv else ""
        return run_app_batch(sys.argv[index + 1], text_model)
    ap = argparse.ArgumentParser(description="分析产物 → gpt-image-2 → 工序（无头）")
    ap.add_argument("--json", required=True, help="分析产物 JSON（或同目录的 -prompts.txt 不必传）")
    ap.add_argument("--style", default="", help="画风名（config-styles.json 的键）")
    ap.add_argument("--quality", default="high", choices=["low", "medium", "high"])
    ap.add_argument("--size", default="auto", help="auto=按输入图比例挑；也可写 1024x1536 等")
    ap.add_argument("--steps", default="repaint",
                    help="要跑的工序，逗号分隔：repaint,structure,local（空 = 只出首图）")
    ap.add_argument("--region", default="subject_no_face", help="局部重绘区域（默认 subject_no_face）")
    ap.add_argument("--feather", type=int, default=40)
    ap.add_argument("--structure-strength", type=float, default=0.5)
    ap.add_argument("--content-image", default="",
                    help="指定内容参考图（默认用分析原图）；可传一张更小的图以规避中转站大载荷 reset")
    ap.add_argument("--content-ref", default="off", choices=["on", "off"],
                    help="首图是否把**分析原图**当内容参考图挂上。off（默认）=分析图只用来出文字；"
                         "on=把原图当内容参考图（视觉锚定最强，代价是出图基本就是原图的画风重绘）")
    ap.add_argument("--use-source-base", action="store_true",
                    help="首图直接用分析原图（保线条，走 §⑲ E 路线），跳过 gpt-image 生成")
    ap.add_argument("--first-pass-mode", default="generate", choices=["generate", "edits"],
                    help="首图端点：generate（默认，/images/generations + image 字段，= app.py 当前契约）"
                         "或 edits（历史路径，把画风参考图当输入图编辑；实测画风注入更强但线条更碎）")
    ap.add_argument("--no-dual", action="store_true", help="重绘不用双参考（默认用）")
    ap.add_argument("--white-contrast", type=float, default=0.0,
                    help="白色/高亮材质区的局部对比增强量（0=关，建议 0.4~0.7；用于高光压平结构的情况）")
    ap.add_argument("--ink", action="store_true",
                    help="最后给线条加墨（只压线条像素），让线重新比底色深；线-底对比目标默认 12")
    ap.add_argument("--ink-target", type=float, default=10.0, help="线条与局部底色的目标分离度（亮度级）")
    ap.add_argument("--ink-max-darken", type=float, default=40.0)
    ap.add_argument("--preset", default="", choices=["", "quality", "fidelity"],
                    help="预设：quality=画质优先（色调目标取画风参考图 + 线条加墨，线-底对比 −20，色彩略偏参考图）；"
                         "fidelity=色彩保真（色调目标取输入照片、不加墨）")
    ap.add_argument("--tone-target", default="style", choices=["photo", "style"],
                    help="色调校准的目标取谁：style=画风参考图（默认，深色高对比氛围更像参考图，线-底对比 −16.7）、"
                         "photo=输入照片（更亮，线-底对比 −11.2）")
    ap.add_argument("--tone", action="store_true",
                    help="最后加一道确定性色调校准（压发白高光、提中对比与色度；目标亮度取输入图）")
    ap.add_argument("--tone-contrast", type=float, default=1.08)
    ap.add_argument("--tone-chroma", type=float, default=1.10,
                    help="色度增益上限；若给了 --tone-saturation 则按参考图饱和度求解（推荐）")
    ap.add_argument("--highlight-strength", type=float, default=0.85,
                    help="高光压缩比例（<1 越多压得越狠，建议 0.75~0.80 再压 3~5%%）")
    ap.add_argument("--skin-warm", type=float, default=0.0, help="肤色暖化强度（0~1，建议 0.3~0.5）")
    ap.add_argument("--sat-scale", type=float, default=1.0, help="目标饱和度系数（0.9~0.95 再降一点）")
    ap.add_argument("--tone-saturation", type=float, default=None,
                    help="目标饱和度（默认自动取输入照片的饱和度）")
    ap.add_argument("--local-resolution", default="2K", choices=["1K", "2K", "4K"],
                    help="局部重绘分辨率（默认全区域 2K，与 app.py 的工序契约一致）")
    ap.add_argument("--detail-boost", action="store_true",
                    help="把细节区（waist/thigh/shoes/face/hair）额外升到 4K。默认关 —— "
                         "2026-09-24 的契约是局部重绘统一 2K（4K 慢且不稳，实测收益不稳定）")
    ap.add_argument("--no-detail-boost", action="store_true", help="（保留参数）显式关闭细节区升分辨率")
    ap.add_argument("--repaint-ref", default="style",
                    choices=["line", "style", "style-neutral", "both", "none"],
                    help="重绘额外参考：style（默认，完整画风图）/ none / style-neutral / line / both")
    ap.add_argument("--repaint-style-ref", default="",
                    help="仅覆盖 Gemini 重绘使用的画风参考图；用于无角色风格板等受控实验，不改首图参考图")
    ap.add_argument("--repaint-no-style-clauses", action="store_true",
                    help="重绘仍可挂画风参考图，但不追加该画风的渲染条款；用于分离图片参考与文字条款的影响")
    ap.add_argument("--repaint-text-without-image", action="store_true",
                    help="受控实验用：重绘只发送源图（不送画风参考图），但保留与挂图模式相同的文字段落，"
                         "仅把「Image 1/Image 2」这类指代换成具名指代。与 --repaint-ref none 联用时，"
                         "唯一变量就是「参考图是否送入生成请求」。")
    ap.add_argument("--repaint-clauses-only", action="store_true",
                    help="受控实验用：不发送画风参考图，也不提参考图，只把画风条目里的条款当成"
                         "本次重绘的文字规格。用于「只替换一条画法句」的对照实验。")
    ap.add_argument("--repaint-clauses-file", default="",
                    help="受控实验用：用这份 JSON（字符串数组）替换本次重绘的条款，"
                         "让实验计划里冻结的那一份逐字生效。")
    ap.add_argument("--repaint-scope",
                    default="full",
                    choices=["full", "person_only", "person_noface", "details", "lines_only"],
                    help="重绘的「编辑范围」：不裁切不贴回，只在提示词里要求模型保留不该动的部分（§三十一）。"
                         "lines_only（默认）= 只把线条连通、不改色不改内容；person_only = 只重绘人物、背景保持；"
                         "person_noface = 人物可动但脸保持；details = 只修手/丝带/系带/鞋带/项链；full = 不额外限制")
    ap.add_argument("--extra-region", default="", help="额外的局部重绘区域（逗号分隔），例如 shoes")
    ap.add_argument("--firmware", default="prompts/gpt-image-optimize/repaint-system-conservative-v5.md")
    ap.add_argument("--source-image", default="", help="覆盖分析 JSON 里的 original 图路径")
    ap.add_argument("--content-field", default="gpt_image_prompt",
                    help="内容字段；默认身份完整锚 gpt_image_prompt，可指定 english_description 或短锚做对照")
    ap.add_argument("--prompt-file", default="", help="完整首图提示词快照（复现实验用）")
    ap.add_argument("--base-image", default="", help="跳过首图，对已有 GPT 产物跑工序")
    ap.add_argument("--output-dir", default="", help="隔离实验产物和请求清单的目录")
    ap.add_argument("--publish-final", action="store_true",
                    help="只把身份门禁后的最终选中图复制到 data/<当天>/；其余产物留在 --output-dir")
    ap.add_argument("--dry-run", action="store_true", help="保存请求清单，不调用图片接口")
    ap.add_argument("--prompt-recipe", choices=["legacy", "reference"], default="legacy")
    ap.add_argument("--styles-file", default="",
                    help="换用另一份画风表（整份替换 conf/config-styles.json）。受控实验用："
                         "把「正式条目 + 一个预先声明的改动」另存成实验画风表，避免改正式配置。")
    ap.add_argument("--first-pass-only", action="store_true",
                    help="只出首图并落盘请求/生成清单（受控实验复用首图用），不跑重绘与任何审计")
    ap.add_argument("--repeat", type=int, default=1, help="同一首图独立复跑后处理次数")
    ap.add_argument("--identity-audit", action="store_true",
                    help="审计首图和最终图是否偏离分析出的角色/服装特征，并保存 JSON")
    ap.add_argument("--identity-correct", action="store_true",
                    help="最终图有高置信身份差异时，最多调用两次 Gemini 做定点修订，每次后重新审计")
    quality_group = ap.add_mutually_exclusive_group()
    quality_group.add_argument("--quality-refine", dest="quality_refine", action="store_true", default=True,
                               help="默认：重绘后做三图+九宫格质量审计，必要时用当前图+GPT首图修订一次")
    quality_group.add_argument("--no-quality-refine", dest="quality_refine", action="store_false",
                               help="关闭重绘后的质量审计与一次定点修订；注意它只关这一处，"
                                    "最终兜底仍可能改图，要连审计一起关请用 --audit-only")
    ap.add_argument("--audit-only", action="store_true",
                    help="只读审计模式：质量、身份、人体、最终复核全部照做并落盘，但本次运行不再产生"
                         "任何修图（质量修订、身份定点修订、最终兜底救援都不调用图片接口）。"
                         "受控实验用；某个候选因此被判定为 review_required 时保持候选，不回绿。")
    ap.add_argument("--final-review-audit", action="store_true",
                    help="即使画风参考图没有参与生成（--repaint-ref none），仍用配置里的参考图做只读终审；"
                         "审计输入不改变生成请求，只让门禁结论有据可依。")
    args = ap.parse_args()

    from modules.others.api_backend import (generate_image_aigc2d_gpt,
                                            pick_gpt_image2_size_for_images)
    from utils import post_process as pp
    from utils.analysis_gen import (build_first_pass_request, first_pass_sub_dir,
                                    publish_final_output, resolve_content_text,
                                    run_face_hair_style_refine, run_gpt_image_pipeline)

    identity_gate_failed = False  # 身份门禁判红后仍继续跑人体/终审，但状态不回绿
    result = json.load(open(args.json, encoding="utf-8"))
    source = args.source_image or str(result.get("source_image_path") or "")
    print(f"[1/4] 分析产物: {os.path.basename(args.json)}")
    print(f"      原图: {source if os.path.isfile(source) else '(缺失)'}")
    print(f"      短锚: {len(resolve_content_text(result, 'short'))} 字符")

    styles = load_styles(args.styles_file)
    content_ref = ""
    if args.content_ref == "on":
        content_ref = args.content_image if (args.content_image and os.path.isfile(args.content_image)) else source
        content_ref = content_ref if (content_ref and os.path.isfile(content_ref)) else ""
    # 首图请求与 GUI 走**同一个组装点**（画风说明 + 内容锚 + 排除句 + 渲染语言条款），
    # 内容锚沿用分析产物里的 gpt 字段（GUI 那条链用与 Gemini 同源的分析描述）。
    payload = build_first_pass_request(styles, args.style, result, tier="short",
                                       content_text=str(result.get(args.content_field) or ""),
                                       prompt_recipe=args.prompt_recipe,
                                       content_image_path=content_ref)
    if args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as f:
            payload["prompt"] = f.read()
    ref = str(payload.get("style_ref_path") or "")
    repaint_style_ref = (os.path.abspath(args.repaint_style_ref)
                         if args.repaint_style_ref else ref)
    if args.repaint_style_ref and not os.path.isfile(repaint_style_ref):
        raise FileNotFoundError(repaint_style_ref)
    proportion_clauses = list(payload.get("proportion_clauses") or [])
    style_clauses = list(payload.get("clauses") or []) + proportion_clauses
    if args.repaint_no_style_clauses:
        style_clauses = []
    if args.repaint_clauses_file:
        # 受控实验：本次重绘的条款以冻结文件为准，避免「声明改了、实发没改」。
        with open(args.repaint_clauses_file, encoding="utf-8") as f:
            style_clauses = [str(c).strip() for c in (json.load(f) or []) if str(c).strip()]
        print(f"      受控模式: 重绘条款取自 {os.path.basename(args.repaint_clauses_file)}"
              f"（{len(style_clauses)} 条）")
    if args.repaint_ref == "style-neutral":
        from utils.style_gpt import resolve_neutral_repaint_clauses
        style_clauses, _ = resolve_neutral_repaint_clauses((styles or {}).get(args.style))
        style_clauses += proportion_clauses
    print(f"[2/4] 画风 {args.style or '(无)'}：说明 {payload['style_chars']} 字符 / "
          f"内容锚 {payload['content_chars']} 字符 / 参考图 {'画风图' if ref else '无'} / "
          f"渲染条款 {len(style_clauses)} 条（{payload.get('clauses_source')}）")
    print(f"      提示词总长 {len(payload['prompt'])} 字符（软上限 2000 / 硬上限 15000）| "
          f"参考图 {len(payload['image_paths'])} 张（内容图 {'有' if content_ref else '无'} + 画风图 "
          f"{'有' if ref else '无'}）")

    if args.preset == "quality":
        args.tone = True
        args.tone_target = "style"
        args.ink = True
        args.ink_target = max(float(args.ink_target), 8.0)
    elif args.preset == "fidelity":
        args.tone = True
        args.tone_target = "photo"
        args.ink = False

    steps = pp.default_pipeline()
    wanted = [s.strip() for s in str(args.steps or "").split(",") if s.strip()]
    for key in ("repaint", "structure", "local"):
        steps[key]["enabled"] = key in wanted
    steps.setdefault("contrast", {})
    steps["contrast"].update({"enabled": float(args.white_contrast) > 0,
                              "amount": float(args.white_contrast), "protect_white": True})
    if float(args.white_contrast) > 0:
        wanted = wanted + ["contrast"]
    steps.setdefault("ink", {})
    steps["ink"].update({"enabled": bool(args.ink), "target_sep": args.ink_target,
                         "max_darken": args.ink_max_darken})
    if args.ink:
        wanted = wanted + ["ink"]
    steps.setdefault("tone", {})
    tone_ref = ref if (args.tone_target == "style" and ref) else source
    steps["tone"].update({"enabled": bool(args.tone), "reference_path": tone_ref,
                          "contrast": args.tone_contrast, "chroma": args.tone_chroma,
                          "highlight_strength": args.highlight_strength,
                          "skin_warm": args.skin_warm, "sat_target_scale": args.sat_scale})
    if args.tone:
        wanted = wanted + ["tone"]
    steps["structure"]["strength"] = args.structure_strength
    regions = [r.strip() for r in str(args.region or "").split(",") if r.strip()]
    regions += [r.strip() for r in str(args.extra_region or "").split(",") if r.strip()]
    steps["local"]["regions"] = regions or ["subject_no_face"]
    steps["local"]["region"] = (steps["local"]["regions"] or ["hair"])[0]
    steps["local"]["feather"] = args.feather
    steps["local"]["resolution"] = args.local_resolution
    steps["local"]["detail_boost"] = bool(args.detail_boost) and not bool(args.no_detail_boost)
    steps["repaint"]["reference_mode"] = ("none" if args.no_dual else
                                          {"line": "line_anchor", "style": "style",
                                           "style-neutral": "style_neutral",
                                           "both": "both", "none": "none"}[args.repaint_ref])
    steps["repaint"]["scope"] = str(args.repaint_scope)
    steps["repaint"]["text_without_image"] = bool(args.repaint_text_without_image)
    steps["repaint"]["clauses_without_image"] = bool(args.repaint_clauses_only)
    if args.repaint_clauses_only and args.repaint_ref == "style":
        print("      受控模式: --repaint-clauses-only 会主动不挂画风参考图，"
              "--repaint-ref style 的图片仍然不会送出")
    if args.repaint_text_without_image and args.repaint_ref != "none":
        print("      受控模式: --repaint-text-without-image 需要与 --repaint-ref none 联用，"
              "否则参考图仍会送出，两臂的差异就不再是唯一变量")
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    size = args.size
    if str(size).lower() == "auto":
        size = pick_gpt_image2_size_for_images([source] if os.path.isfile(source) else payload["image_paths"])
    manifest = {"analysis_json": os.path.abspath(args.json), "content_field": args.content_field,
                "source": source, "style": args.style, "request": payload,
                "size": size, "quality": args.quality, "mode": args.first_pass_mode,
                "steps": steps, "firmware": args.firmware, "base": args.base_image,
                "repaint_style_ref": repaint_style_ref,
                "outputs": [], "quality_refine": {}, "identity": {},
                "published_final": "", "status": "planned"}
    def save_manifest():
        if output_dir:
            with open(os.path.join(output_dir, "request.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
    save_manifest()
    if args.dry_run:
        print("预检完成；未调用图片接口。")
        return 0
    if args.base_image:
        if not os.path.isfile(args.base_image):
            raise FileNotFoundError(args.base_image)
        base_path = os.path.abspath(args.base_image)
    elif args.use_source_base:
        if not os.path.isfile(source):
            print("❌ --use-source-base 需要可用的原图")
            return 1
        base_path = source
        print(f"[3/4] 跳过 gpt-image 生成，直接用原图作为工序输入")
    else:
        size = args.size
        if str(size).lower() == "auto":
            refs_for_size = list(payload.get("image_paths") or [])
            if source and os.path.isfile(source):
                refs_for_size.insert(0, source)
            size = pick_gpt_image2_size_for_images(refs_for_size)
        print(f"[3/4] gpt-image-2 出首图（quality={args.quality}, size={size}, "
              f"端点={'/images/generations（新建图片）' if args.first_pass_mode == 'generate' else '/images/edits'}）…")
        from utils.first_image_review import generate_first_image, apply_safe_plan
        saved, payload["prompt"] = generate_first_image(generate_image_aigc2d_gpt,
            analysis_result=result,
            plan_callback=lambda plan: apply_safe_plan(payload, steps, plan,
                original_style_ref=repaint_style_ref, restore_reference=not args.no_dual),
            prompt=payload["prompt"], image_paths=list(payload.get("image_paths") or []),
            model="gpt-image-2", size=size, quality=args.quality, output_format="png", n=1,
            api_type="aigc-2d-gpt", file_prefix=os.path.splitext(os.path.basename(args.json))[0][:24],
            save_sub_dir=output_dir or first_pass_sub_dir(steps),
            mode=args.first_pass_mode,
            log_callback=print)
        if not saved:
            manifest["status"] = "first_pass_failed"
            save_manifest()
            print("❌ 首图生成失败（看 log/<日期>.log）")
            return 1
        if payload.get("safe_alternative"):
            from utils.first_image_review import safe_repaint_firmware
            args.firmware = safe_repaint_firmware(payload, args.firmware)
            repaint_style_ref = str(payload.get("style_ref_path") or "")
            style_clauses = list(payload.get("clauses") or []) + proportion_clauses
            args.identity_audit = True
            args.identity_correct = True
            args.quality_refine = bool(payload.get("safe_style_restore"))
            manifest["firmware"] = args.firmware
            manifest["repaint_style_ref"] = repaint_style_ref
        base_path = saved[0]
        print(f"      首图: {os.path.relpath(base_path, BASE)}  {_metrics(base_path)}")

    if args.first_pass_only and not args.output_dir:
        print("❌ --first-pass-only 需要 --output-dir（请求快照与首图要按实验目录留存）")
        return 1
    manifest["base"] = base_path
    if args.first_pass_only:
        # 受控实验用：只要首图 + 请求/生成清单，重绘与审计留给复用同一首图的两次运行。
        # `--identity-audit` 仍会执行并落盘，这样门禁结论在首图阶段也是记录在案的。
        if args.identity_audit or args.identity_correct:
            from utils.identity_audit import audit_image_identity
            try:
                first_audit = audit_image_identity(base_path, result, expected_prompt=payload["prompt"])
            except Exception as exc:  # 审计是质量门，不应让已成功的首图报废
                first_audit = {"mismatch": False, "severity": "unknown", "confidence": 0.0,
                               "differences": [], "audit_error": f"{type(exc).__name__}: {exc}"}
            manifest["identity"]["first"] = first_audit
            if output_dir:
                with open(os.path.join(output_dir, "identity-first.json"), "w", encoding="utf-8") as f:
                    json.dump(first_audit, f, ensure_ascii=False, indent=2)
            if first_audit.get("audit_error"):
                first_status = "审计失败"
            elif first_audit.get("mismatch"):
                first_status = "有差异"
            else:
                first_status = "通过"
            print(f"      首图身份审计: {first_status} "
                  f"({first_audit.get('severity')}, {float(first_audit.get('confidence') or 0):.2f})")
        manifest["status"] = "first_pass_complete"
        manifest["selected_output"] = base_path
        save_manifest()
        print("[首图-only] 已产出首图，未跑重绘与审计（按实验声明）")
        print("最终产物:")
        print(f"  {os.path.relpath(base_path, BASE)}")
        print(f"    {_metrics(base_path)}")
        return 0 if not (manifest["identity"].get("first") or {}).get("audit_error") else 1
    manifest["status"] = "first_pass_complete"
    if args.identity_audit or args.identity_correct:
        from utils.identity_audit import audit_image_identity, build_identity_correction_prompt
        try:
            first_audit = audit_image_identity(base_path, result, expected_prompt=payload["prompt"])
            first_audit["correction_prompt"] = build_identity_correction_prompt(first_audit, 1, 2)
        except Exception as exc:  # 审计是质量门，不应让已成功的生图链报废
            first_audit = {"mismatch": False, "severity": "unknown", "confidence": 0.0,
                           "differences": [], "audit_error": f"{type(exc).__name__}: {exc}"}
        manifest["identity"]["first"] = first_audit
        if output_dir:
            with open(os.path.join(output_dir, "identity-first.json"), "w", encoding="utf-8") as f:
                json.dump(first_audit, f, ensure_ascii=False, indent=2)
        first_status = "审计失败" if first_audit.get("audit_error") else ("有差异" if first_audit["mismatch"] else "通过")
        print(f"      首图身份审计: {first_status} "
              f"({first_audit['severity']}, {first_audit['confidence']:.2f})")
    save_manifest()

    print(f"[4/4] 工序: {', '.join(wanted) or '(无)'} | 区域 {regions} | "
          f"重绘参考 {steps['repaint']['reference_mode']} | 画风条款 {len(style_clauses)} 条")
    outs = []
    for trial in range(max(1, args.repeat)):
        work_dir = os.path.join(output_dir or os.path.dirname(os.path.abspath(base_path)),
                                f"steps-{trial + 1}")
        try:
            if payload.get("safe_alternative"):
                outs.extend(run_gpt_image_pipeline([base_path], steps, firmware=args.firmware,
                    style_ref_path=repaint_style_ref, style_clauses=style_clauses,
                    final_dir=output_dir, work_dir=work_dir, strict=True,
                    log_callback=lambda m: print("      ", m)))
            else:
                outs.extend(pp.run_pipeline([base_path], steps, firmware=args.firmware,
                    style_ref_path=repaint_style_ref, style_clauses=style_clauses,
                    final_dir=output_dir, work_dir=work_dir, resume=False,
                    log_callback=lambda m: print("      ", m)))
        except Exception as exc:
            manifest["status"] = "pipeline_failed"
            manifest["pipeline_error"] = f"{type(exc).__name__}: {exc}"
            save_manifest()
            print("❌ 安全替代重绘失败，已保留首图和断点：" + str(exc))
            return 1
    manifest["outputs"] = outs
    if ((steps.get("repaint") or {}).get("enabled") and payload.get("face_hair_refine")
            and outs and repaint_style_ref
            and os.path.isfile(repaint_style_ref)):
        try:
            face_hair = run_face_hair_style_refine(
                outs[-1], repaint_style_ref, style_clauses=style_clauses,
                output_dir=output_dir or os.path.dirname(outs[-1]),
                file_prefix="face-hair-style")
            if face_hair:
                outs.extend(face_hair)
                manifest["outputs"] = outs
                print("      五官画风修订: 已用完整画风图只修订面部与头发绘画语法")
        except Exception as exc:
            print(f"      五官画风修订失败，保留首次重绘图: {type(exc).__name__}: {exc}")
    effective_quality_refine = args.quality_refine and not bool(payload.get("skip_quality_refine"))
    if args.audit_only:
        # 只读审计模式：质量审计照做（报告要两侧对称），但一次修图都不发。
        effective_quality_refine = True
        args.identity_correct = False
        print("      审计模式: 已开启 --audit-only，质量/身份/人体/终审只审计不改图")
    if args.quality_refine and not effective_quality_refine:
        print("      质量门禁: 当前画风配置保留首次完整画风图重绘，跳过二次质量修订")
    if effective_quality_refine and outs and repaint_style_ref and os.path.isfile(repaint_style_ref):
        from modules.others.api_backend import generate_image_repaint
        from utils.refine_quality import (audit_refine_quality, build_quality_correction_prompt,
                                          should_refine_quality, repair_style_guard)
        try:
            quality_audit = audit_refine_quality(
                base_path, outs[-1], repaint_style_ref, first_pass_prompt=payload["prompt"],
                proportion_clauses=proportion_clauses)
            manifest["quality_refine"]["before"] = quality_audit
            if output_dir:
                with open(os.path.join(output_dir, "refine-quality-audit-0.json"),
                          "w", encoding="utf-8") as f:
                    json.dump(quality_audit, f, ensure_ascii=False, indent=2)
            if should_refine_quality(quality_audit) and not args.audit_only:
                quality_prompt = build_quality_correction_prompt(quality_audit)
                quality_prompt += "\n\n" + repair_style_guard(payload)
                refined = generate_image_repaint(
                    [outs[-1]], resolution="2K", aspect_ratio=pp.snapped_aspect_ratio(outs[-1]),
                    prompt=quality_prompt,
                    use_detail_suffix=False, save_sub_dir=output_dir or os.path.dirname(outs[-1]),
                    file_prefix="quality-refine", extra_reference_paths=[base_path]) or []
                if refined:
                    outs.extend(refined)
                    quality_after = audit_refine_quality(
                        base_path, outs[-1], repaint_style_ref, first_pass_prompt=payload["prompt"],
                        proportion_clauses=proportion_clauses)
                    manifest["quality_refine"]["after"] = quality_after
                    if output_dir:
                        with open(os.path.join(output_dir, "refine-quality-audit-1.json"),
                                  "w", encoding="utf-8") as f:
                            json.dump(quality_after, f, ensure_ascii=False, indent=2)
                    if should_refine_quality(quality_after) and (
                            quality_after.get("severity") == "major" or any(
                                float(item.get("confidence") or 0) >= .72
                                for item in quality_after.get("background_drift", []))):
                        raise RuntimeError("质量修订产生重大漂移，禁止继续修身份或发布")
                    print("      质量门禁: 已用当前图 + GPT 首图做一次文字驱动修订（未再次发送画风图）")
            else:
                print("      质量门禁: 未发现需定点修复的高置信问题")
        except Exception as exc:
            manifest["quality_refine"]["audit_error"] = f"{type(exc).__name__}: {exc}"
            print(f"      质量门禁失败，保留当前图: {type(exc).__name__}: {exc}")
        manifest["outputs"] = outs
        manifest["selected_output"] = outs[-1]
        save_manifest()
        if manifest["quality_refine"].get("audit_error"):
            # 质量门禁失败：只停止「再改图」，下游人体/终审仍要对当前候选出结论，
            # 否则两条臂的门禁覆盖不对称、报告无法比较。
            manifest["status"] = "review_required"
            run_downstream_audits = True
    quality_audit_error = str(manifest["quality_refine"].get("audit_error") or "")
    # 只读审计模式下重绘仍是「未修订」的当前图，状态必须保持红色而不是回落。
    if args.audit_only and should_refine_quality(manifest["quality_refine"].get("before") or {}):
        quality_audit_error = quality_audit_error or "审计模式：质量审计仍有高置信缺陷，未自动修订"
    manifest["audit_only"] = bool(args.audit_only)
    if (args.identity_audit or args.identity_correct) and outs:
        from utils.identity_audit import (audit_image_identity, build_identity_correction_prompt,
                                          identity_gate_action)
        from modules.others.api_backend import generate_image_repaint
        if manifest.get("status") == "review_required":
            # 质量门禁已判定漂移：身份阶段只审计、不再生图（不叠加二次修改），
            # 但审计结论要落盘，报告才能给出 a/b 对称的门禁结果。
            args.identity_correct = False
        try:
            final_audit = audit_image_identity(outs[-1], result, expected_prompt=payload["prompt"])
            final_audit["correction_prompt"] = build_identity_correction_prompt(final_audit, 1, 2)
        except Exception as exc:  # 保留最终图并把审计故障写进清单
            final_audit = {"mismatch": False, "severity": "unknown", "confidence": 0.0,
                           "differences": [], "audit_error": f"{type(exc).__name__}: {exc}",
                           "correction_prompt": ""}
        # 审计结论必须绑定它实际审的那张图的内容哈希，否则后续换图后无法证明结论属于哪一张。
        final_audit.setdefault("image", os.path.abspath(outs[-1]))
        final_audit["candidate_sha256"] = final_sha256(outs[-1])
        manifest["identity"]["final"] = final_audit
        if output_dir:
            with open(os.path.join(output_dir, "identity-final.json"), "w", encoding="utf-8") as f:
                json.dump(final_audit, f, ensure_ascii=False, indent=2)
        final_status = "审计失败" if final_audit.get("audit_error") else ("有差异" if final_audit["mismatch"] else "通过")
        print(f"      最终图身份审计: {final_status} "
              f"({final_audit['severity']}, {final_audit['confidence']:.2f})")
        action = identity_gate_action(final_audit)
        manifest["identity"]["action"] = action
        manifest["selected_output"] = outs[-1]
        current = outs[-1]
        current_audit = final_audit
        correction_rounds = []
        effective_identity_correct = args.identity_correct and not bool(payload.get("skip_identity_refine"))
        if args.identity_correct and not effective_identity_correct:
            print("      身份门禁: 当前画风明确允许角色设计变体；仅记录审计，不执行身份回改")
        if effective_identity_correct:
            for correction_round in range(1, 3):
                if identity_gate_action(current_audit) != "correct":
                    break
                prompt = build_identity_correction_prompt(
                    current_audit, iteration=correction_round, max_iterations=2)
                from utils.refine_quality import repair_style_guard
                prompt += "\n\n" + repair_style_guard(payload)
                identity_clauses = [str(c).strip() for c in
                                    (payload.get("identity_correction_clauses") or [])
                                    if str(c).strip()]
                if identity_clauses:
                    prompt += "\n\nSTYLE-SPECIFIC CORRECTION GUARDS:\n- " + "\n- ".join(identity_clauses)
                corrected = generate_image_repaint(
                    [current], resolution="2K", aspect_ratio=pp.snapped_aspect_ratio(current),
                    prompt=prompt,
                    use_detail_suffix=False, save_sub_dir=output_dir or os.path.dirname(current),
                    file_prefix=f"identity-correct-{correction_round}") or []
                if not corrected:
                    break
                current = corrected[-1]
                outs.extend(corrected)
                try:
                    current_audit = audit_image_identity(
                        current, result, expected_prompt=payload["prompt"])
                    current_audit["correction_prompt"] = build_identity_correction_prompt(
                        current_audit, min(2, correction_round + 1), 2)
                except Exception as exc:
                    current_audit = {"mismatch": False, "severity": "unknown", "confidence": 0.0,
                                     "stable_anchors": [], "differences": [],
                                     "audit_error": f"{type(exc).__name__}: {exc}"}
                correction_rounds.append({"round": correction_round, "image": current,
                                          "audit": current_audit})
                if output_dir:
                    with open(os.path.join(output_dir, f"identity-corrected-{correction_round}.json"),
                              "w", encoding="utf-8") as f:
                        json.dump(current_audit, f, ensure_ascii=False, indent=2)
                print(f"      定点修订 {correction_round}/2 复审: "
                      f"{'仍有差异' if current_audit.get('mismatch') else '通过'} "
                      f"({current_audit.get('severity')}, {float(current_audit.get('confidence') or 0):.2f})")
        manifest["identity"]["correction_rounds"] = correction_rounds
        manifest["selected_output"] = current
        if correction_rounds and identity_gate_action(current_audit) != "accept":
            print("      身份门禁: 两轮后仍有差异，按不回退策略保留最后一轮修订图")
        if not bool(payload.get("skip_identity_refine")) and identity_gate_action(current_audit) != "accept":
            manifest["outputs"] = outs
            save_manifest()
            identity_gate_failed = True
            print("      身份门禁未通过：保留候选图，禁止发布；人体/终审仍出结论供对照")
    post_adjustment = payload.get("post_adjustment") or {}
    if outs and post_adjustment:
        from utils.analysis_gen import apply_style_post_adjustment
        source_for_adjustment = str(manifest.get("selected_output") or outs[-1])
        adjusted = apply_style_post_adjustment(
            source_for_adjustment, post_adjustment,
            output_dir=output_dir or os.path.dirname(source_for_adjustment),
            file_prefix="style-adjusted")
        if adjusted:
            outs.append(adjusted)
            manifest["outputs"] = outs
            manifest["selected_output"] = adjusted
            print("      画风确定性收尾: 已应用色彩、曝光或结构线专用校正")
    gate_failed_before_close = (manifest.get("status") == "review_required"
                                or bool(quality_audit_error)
                                or identity_gate_failed)
    selected = str(manifest.get("selected_output") or (outs[-1] if outs else ""))
    manifest["status"] = "complete" if (outs and selected) else "pipeline_failed"
    if gate_failed_before_close:
        # 上游已有门禁判红：下游人体/终审只是补结论，不允许把状态改回 complete
        manifest["status"] = "review_required"
    if selected:
        from utils.refine_quality import audit_hand_quality, should_refine_quality
        try:
            anatomy = audit_hand_quality(selected)
            manifest["anatomy_review"] = anatomy
            if should_refine_quality(anatomy) or anatomy.get("needs_review"):
                manifest["status"] = "review_required"
            if output_dir:
                with open(os.path.join(output_dir, "anatomy-audit.json"), "w", encoding="utf-8") as f:
                    json.dump(anatomy, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            manifest["anatomy_review"] = {"audit_error": f"{type(exc).__name__}: {exc}"}
            manifest["status"] = "review_required"
    pre_final_sha = final_sha256(selected) if selected else ""
    if (steps.get("repaint") or {}).get("enabled") and selected and (args.final_review_audit or args.audit_only):
        # 只读终审：候选图与首图/源图对比。参考图只在没有别的东西可比时充当 Image 3，
        # 并且明确记下用了哪张，避免把「参考图没进生成」读成「没有终审」。
        audit_style_ref = repaint_style_ref if (repaint_style_ref and os.path.isfile(repaint_style_ref)) else ""
        final_audit_used_style = bool(audit_style_ref)
        if not audit_style_ref and os.path.isfile(base_path):
            audit_style_ref = base_path
        from utils.refine_quality import (audit_refine_quality, should_refine_quality, repair_style_guard,
                                          review_final_candidate, write_style_reference_absent_audit)
        final_dir = output_dir or os.path.dirname(selected)
        if not audit_style_ref:
            manifest["final_review"] = write_style_reference_absent_audit(selected, None, final_dir)
            manifest["status"] = "review_required"
            print("      最终复核: 没有可用参考图，只记录「未完成终审」，不判通过")
        else:
            try:
                selected, final_quality = review_final_candidate(
                    selected, lambda candidate: audit_refine_quality(
                    base_path, candidate, audit_style_ref, first_pass_prompt=payload["prompt"],
                    proportion_clauses=proportion_clauses, final_review=True,
                    style_targets=repair_style_guard(payload),
                    authorized_changes=payload.get("generation_clauses") or []),
                    final_dir, log_callback=print, audit_only=True)
                final_quality["audit_style_reference_is_generation_reference"] = final_audit_used_style
                manifest["final_review"] = final_quality
                manifest["selected_output"] = selected
                outs = [selected]
                if should_refine_quality(final_quality) or final_quality.get("needs_review"):
                    manifest["status"] = "review_required"
                if output_dir:
                    with open(os.path.join(output_dir, "final-quality-audit.json"), "w", encoding="utf-8") as f:
                        json.dump(final_quality, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                manifest["final_review"] = {"audit_error": f"{type(exc).__name__}: {exc}"}
                manifest["status"] = "review_required"
            if manifest["status"] == "review_required":
                print("      最终复核未通过：保留候选图和审计记录，禁止发布")
    elif ((steps.get("repaint") or {}).get("enabled") and selected and repaint_style_ref
            and os.path.isfile(repaint_style_ref)
            and payload.get("repaint_reference_mode", "style") != "none"):
        from utils.refine_quality import audit_refine_quality, should_refine_quality, repair_style_guard
        try:
            from utils.refine_quality import review_final_candidate
            selected, final_quality = review_final_candidate(
                selected, lambda candidate: audit_refine_quality(
                base_path, candidate, repaint_style_ref, first_pass_prompt=payload["prompt"],
                proportion_clauses=proportion_clauses, final_review=True,
                style_targets=repair_style_guard(payload),
                authorized_changes=payload.get("generation_clauses") or []),
                output_dir or os.path.dirname(selected), log_callback=print,
                audit_only=bool(args.audit_only))
            manifest["final_review"] = final_quality
            manifest["selected_output"] = selected
            outs = [selected]
            if should_refine_quality(final_quality) or final_quality.get("needs_review"):
                manifest["status"] = "review_required"
            if output_dir:
                with open(os.path.join(output_dir, "final-quality-audit.json"), "w", encoding="utf-8") as f:
                    json.dump(final_quality, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            manifest["final_review"] = {"audit_error": f"{type(exc).__name__}: {exc}"}
            manifest["status"] = "review_required"
        if manifest["status"] == "review_required":
            print("      最终复核未通过：保留候选图和审计记录，禁止发布")
    # 兜底救援一旦真的换了图，之前绑在旧图上的身份/人体审计立即失效：
    # 必须对新图重新出结论，否则「通过」会被按旧图沿用（第二轮的实际缺陷）。
    if (final_sha256(selected) and final_sha256(selected) != pre_final_sha):
        print("      最终图已被兜底替换：身份与人体审计对新图重新出结论")
        if (args.identity_audit or args.identity_correct) and selected:
            from utils.identity_audit import audit_image_identity, build_identity_correction_prompt
            try:
                rebound = audit_image_identity(selected, result, expected_prompt=payload["prompt"])
                rebound["correction_prompt"] = build_identity_correction_prompt(rebound, 1, 2)
            except Exception as exc:  # noqa: BLE001
                rebound = {"mismatch": False, "severity": "unknown", "confidence": 0.0,
                           "differences": [], "audit_error": f"{type(exc).__name__}: {exc}",
                           "correction_prompt": ""}
            rebound["rebound_after_rescue"] = True
            manifest["identity"]["final"] = rebound
            manifest["identity"]["action"] = identity_gate_action(rebound)
            if output_dir:
                with open(os.path.join(output_dir, "identity-final.json"), "w", encoding="utf-8") as f:
                    json.dump(rebound, f, ensure_ascii=False, indent=2)
        from utils.refine_quality import audit_hand_quality
        try:
            anatomy = audit_hand_quality(selected)
        except Exception as exc:  # noqa: BLE001
            anatomy = {"audit_error": f"{type(exc).__name__}: {exc}"}
        anatomy["rebound_after_rescue"] = True
        manifest["anatomy_review"] = anatomy
        if output_dir:
            with open(os.path.join(output_dir, "anatomy-audit.json"), "w", encoding="utf-8") as f:
                json.dump(anatomy, f, ensure_ascii=False, indent=2)
    else:
        anatomy = manifest.get("anatomy_review")
    from utils.gate_status import evaluate_final_gate
    gate = evaluate_final_gate(
        final_image=selected, identity=(manifest.get("identity") or {}).get("final"),
        anatomy=anatomy, quality=manifest.get("quality_refine"),
        final_review=manifest.get("final_review"),
        identity_action=str((manifest.get("identity") or {}).get("action") or ""),
        quality_audit_error=quality_audit_error, upstream_failed=gate_failed_before_close)
    if gate["status"] == "complete" and not manifest.get("status") == "pipeline_failed":
        manifest["status"] = "complete"
    else:
        manifest["status"] = gate["status"] if gate["status"] == "pipeline_failed" else "review_required"
    manifest["final_gate"] = gate
    if output_dir:
        with open(os.path.join(output_dir, "final-gate.json"), "w", encoding="utf-8") as f:
            json.dump(gate, f, ensure_ascii=False, indent=2)
    print("      最终门禁: " + gate["text"])
    if manifest["status"] == "review_required":
        print("      人体/最终门禁未通过或审计失败：候选图保留，不视为全链成功")
    if args.publish_final and manifest["status"] == "complete" and selected and os.path.isfile(selected):
        published = publish_final_output(
            selected, style_name=args.style, process_dir=output_dir or os.path.dirname(selected))
        manifest["published_final"] = published
        print(f"      发布最终图: {os.path.relpath(published, BASE)}")
    save_manifest()
    print("\n最终产物:")
    for path in outs:
        try:
            print(f"  {os.path.relpath(path, BASE)}")
        except ValueError:  # 跨盘符（例如测试用的临时目录）
            print(f"  {path}")
        print(f"    {_metrics(path)}")
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
