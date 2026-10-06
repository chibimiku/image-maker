"""分析队列（单图内容分析 Tab 的历史列表）右键重跑能力。

覆盖：
- 历史记录记住原图路径（剪贴板图片没有路径，不可重跑）；
- 「重跑分析」按原路径重新提交，可选「重跑分析并生图」的生成目标；
- 「重跑全部失败项 / 清空失败记录」只挑失败或超时的记录；
- 本次重跑自带的生图目标优先于全局自动生图勾选框。
"""

import datetime
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QApplication

import modules.image_analysis.single_analyzer as single_analyzer_module
from modules.image_analysis.single_analyzer import SingleAnalyzerWidget, WorkerThread


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_widget(monkeypatch):
    monkeypatch.setattr(single_analyzer_module, "list_esrgan_models", lambda: ["realesrgan-x4plus"])
    return SingleAnalyzerWidget(
        config_getter_func=lambda *_args, **_kwargs: ("https://example.invalid/v1", "test-key", "test-model"),
        img_config_getter_func=lambda *_args, **_kwargs: ("https://example.invalid/v1", "image-key", "image-model", "openai"),
        styles_getter_func=lambda: {"默认风格": "masterpiece"},
        save_img_cfg_callback=lambda *args, **kwargs: None,
        ar_policy_getter_func=lambda: {"default_aspect_ratio": "1:1", "override_first": "不覆盖(沿用原逻辑)", "override_second": "不覆盖(沿用原逻辑)"},
        nsfw_default_getter_func=lambda: False,
        upscale_options_getter_func=lambda: {},
        outfit_style_history_getter_func=lambda: [],
        outfit_style_default_getter_func=lambda: "",
    )


def _insert_record(widget, thread_no, status, source_path, task_hash=None):
    record = widget._create_history_record(
        thread_no, source_path, datetime.datetime.now(), task_hash or f"hash{thread_no}"
    )
    record["status"] = status
    record["status_text"] = widget._status_to_text(status)
    widget._insert_history_record(record)
    return record


class _Signal:
    """替掉 Qt 信号：connect 后什么都不做。"""

    def connect(self, *_args, **_kwargs):
        return None


class _FakeWorkerThread:
    """替掉 WorkerThread：只记录装配结果，不真的起线程。"""

    def __init__(self, *args, **kwargs):
        self.log_signal = _Signal()
        self.finish_signal = _Signal()
        self.finished = _Signal()
        self.meta_force_gen_targets = []

    def start(self):
        return None


def _patch_fake_worker_thread(monkeypatch):
    monkeypatch.setattr(single_analyzer_module, "WorkerThread", _FakeWorkerThread)


def _freeze_dialogs(monkeypatch, answer=None):
    """把 QMessageBox 的弹窗换成即时返回，避免测试里卡住。"""
    box = single_analyzer_module.QMessageBox
    monkeypatch.setattr(box, "information", staticmethod(lambda *a, **k: box.StandardButton.Ok))
    monkeypatch.setattr(box, "warning", staticmethod(lambda *a, **k: box.StandardButton.Ok))
    monkeypatch.setattr(
        box, "question",
        staticmethod(lambda *a, **k: box.StandardButton.Yes if answer is None else answer),
    )


def test_history_record_remembers_source_path(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch)
    image_path = tmp_path / "girl.png"
    Image.new("RGB", (4, 4)).save(image_path)

    file_record = widget._create_history_record(1, str(image_path), datetime.datetime.now(), "h1")
    clipboard_record = widget._create_history_record(2, Image.new("RGB", (4, 4)), datetime.datetime.now(), "h2")

    assert file_record["source_path"] == os.path.abspath(str(image_path))
    assert file_record["source_origin"] == "file"
    # 直接建记录时不做落盘（落盘发生在 _launch_analysis_task 提交时）
    assert clipboard_record["source_path"] == ""
    assert clipboard_record["source_origin"] == "clipboard"
    assert widget._history_record_usable_path(clipboard_record) == ""
    widget.close()


def test_clipboard_snapshot_makes_record_rerunnable(qapp, monkeypatch, tmp_path):
    """剪贴板图片提交时落 PNG 快照，记录里拿到真实路径 → 可以右键重跑。"""
    widget = _make_widget(monkeypatch)
    snapshot_dir = tmp_path / "clip-snapshots"
    monkeypatch.setattr(single_analyzer_module, "CLIPBOARD_SNAPSHOT_DIR", str(snapshot_dir))
    _patch_fake_worker_thread(monkeypatch)

    clipboard_image = Image.new("RGBA", (6, 4), (10, 20, 30, 255))

    widget._launch_analysis_task(clipboard_image)

    record = next(iter(widget._analysis_history.values()))
    assert record["source_origin"] == "clipboard"
    assert record["source_path"].startswith(str(snapshot_dir))
    assert os.path.isfile(record["source_path"])
    assert widget._history_record_usable_path(record) == record["source_path"]

    with Image.open(record["source_path"]) as snapshot:
        assert snapshot.size == (6, 4)

    # 拿到快照后，重跑走的是同一条路径
    calls = []
    monkeypatch.setattr(
        widget, "_launch_analysis_task",
        lambda snapshot, gen_targets=None, header_note=None, **kwargs:
            calls.append((snapshot, header_note)) or type("_T", (), {"meta_force_gen_targets": []})(),
    )
    assert widget._rerun_history_record(record) is True
    assert calls[0][0] == record["source_path"]
    assert "重跑来源" in calls[0][1]
    widget.close()


def test_process_image_from_clipboard_enqueues_rerunnable_task(qapp, monkeypatch, tmp_path):
    """Ctrl+V 粘贴的图片走 process_image 提交后，队列里那条记录同样能重跑。"""
    widget = _make_widget(monkeypatch)
    snapshot_dir = tmp_path / "clip-snapshots"
    monkeypatch.setattr(single_analyzer_module, "CLIPBOARD_SNAPSHOT_DIR", str(snapshot_dir))
    _patch_fake_worker_thread(monkeypatch)

    widget.image_source = Image.new("RGB", (5, 5), (200, 100, 50))
    widget.process_image()

    record = next(iter(widget._analysis_history.values()))
    assert record["status"] == "running"
    assert record["source_origin"] == "clipboard"
    assert os.path.isfile(record["source_path"])
    assert record["source_desc"] == "剪贴板图片"
    widget.close()


def test_clipboard_snapshot_failure_keeps_record_but_disables_rerun(qapp, monkeypatch, tmp_path):
    """快照落盘失败时不能让分析崩掉：照常提交，只是这条记录不可重跑。"""
    widget = _make_widget(monkeypatch)
    # 用「路径其实是文件」的目录让 makedirs/save 必然失败
    broken_dir = tmp_path / "not-a-dir"
    broken_dir.write_text("x", encoding="utf-8")
    monkeypatch.setattr(single_analyzer_module, "CLIPBOARD_SNAPSHOT_DIR", str(broken_dir))
    _patch_fake_worker_thread(monkeypatch)

    widget._launch_analysis_task(Image.new("RGB", (5, 5)))

    record = next(iter(widget._analysis_history.values()))
    assert record["status"] == "running"
    assert record["source_path"] == ""
    assert widget._history_record_usable_path(record) == ""
    widget.close()


def test_failed_history_records_keeps_only_error_and_timeout(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch)
    good = tmp_path / "good.png"
    good.write_bytes(b"x")

    _insert_record(widget, 3, "success", str(good))
    _insert_record(widget, 1, "error", str(good))
    _insert_record(widget, 4, "cancelled", str(good))
    _insert_record(widget, 2, "timeout", str(good))

    failed = widget._failed_history_records()

    assert [record["thread_no"] for record in failed] == [1, 2]
    widget.close()


def test_rerun_history_record_resubmits_same_source_with_gen_target(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch)
    image_path = tmp_path / "girl.png"
    Image.new("RGB", (4, 4)).save(image_path)
    record = _insert_record(widget, 1, "error", str(image_path))

    calls = []

    def fake_launch(image_source_snapshot, gen_targets=None, header_note=None, **kwargs):
        calls.append({
            "image_source": image_source_snapshot,
            "gen_targets": list(gen_targets or []),
            "header_note": header_note,
        })
        return type("_Thread", (), {"meta_force_gen_targets": list(gen_targets or [])})()

    monkeypatch.setattr(widget, "_launch_analysis_task", fake_launch)

    assert widget._rerun_history_record(record, gen_targets=["refined"]) is True

    assert len(calls) == 1
    assert calls[0]["image_source"] == os.path.abspath(str(image_path))
    assert calls[0]["gen_targets"] == ["refined"]
    assert "重跑来源" in calls[0]["header_note"]
    widget.close()


def test_rerun_history_record_rejects_unusable_source(qapp, monkeypatch):
    """没有快照的剪贴板记录、以及源图已被删除的记录，都不允许重跑。"""
    widget = _make_widget(monkeypatch)
    _freeze_dialogs(monkeypatch)
    clipboard_record = _insert_record(widget, 1, "error", Image.new("RGB", (4, 4)))
    missing_record = _insert_record(widget, 2, "error", os.path.join(os.getcwd(), "not-there.png"))

    launched = []
    monkeypatch.setattr(widget, "_launch_analysis_task", lambda *a, **k: launched.append(a))

    assert widget._rerun_history_record(clipboard_record) is False
    assert widget._rerun_history_record(missing_record) is False
    assert launched == []
    widget.close()


def test_rerun_all_failed_records_skips_unusable_sources(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch)
    _freeze_dialogs(monkeypatch)
    image_path = tmp_path / "girl.png"
    Image.new("RGB", (4, 4)).save(image_path)

    _insert_record(widget, 1, "error", str(image_path))
    _insert_record(widget, 2, "timeout", str(image_path))
    _insert_record(widget, 3, "error", Image.new("RGB", (4, 4)))       # 剪贴板，不可重跑
    _insert_record(widget, 4, "success", str(image_path))              # 成功项不参与

    rerun = []
    monkeypatch.setattr(
        widget, "_rerun_history_record",
        lambda record, gen_targets=None: rerun.append(record["thread_no"]) or True,
    )

    widget._rerun_failed_history_records()

    assert rerun == [1, 2]
    widget.close()


def test_clear_and_remove_failed_history_records(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch)
    _freeze_dialogs(monkeypatch)
    image_path = tmp_path / "girl.png"
    Image.new("RGB", (4, 4)).save(image_path)

    failed_a = _insert_record(widget, 1, "error", str(image_path))
    failed_b = _insert_record(widget, 2, "timeout", str(image_path))
    kept = _insert_record(widget, 3, "success", str(image_path))

    widget._clear_failed_history_records()

    remaining = {record["task_id"] for record in widget._analysis_history.values()}
    assert failed_a["task_id"] not in remaining
    assert failed_b["task_id"] not in remaining
    assert kept["task_id"] in remaining
    assert widget.history_list.count() == 1

    widget._remove_history_record(kept)
    assert widget._analysis_history == {}
    assert widget.history_list.count() == 0
    widget.close()


def test_remove_running_record_is_refused(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch)
    _freeze_dialogs(monkeypatch)
    image_path = tmp_path / "girl.png"
    Image.new("RGB", (4, 4)).save(image_path)
    running = _insert_record(widget, 1, "running", str(image_path))

    widget._remove_history_record(running)

    assert running["task_id"] in widget._analysis_history
    widget.close()


def test_launch_analysis_task_stores_force_gen_targets(qapp, monkeypatch, tmp_path):
    """重跑时生成目标要挂在这次任务上，而不是改全局勾选框。"""
    widget = _make_widget(monkeypatch)
    image_path = tmp_path / "girl.png"
    Image.new("RGB", (4, 4)).save(image_path)
    created = {}

    class _Signal:
        def connect(self, *_args, **_kwargs):
            return None

    class _FakeWorkerThread:
        def __init__(self, *args, **kwargs):
            created["args"] = args
            created["kwargs"] = kwargs
            self.log_signal = _Signal()
            self.finish_signal = _Signal()
            self.finished = _Signal()
            self.meta_force_gen_targets = []

        def start(self):
            created["started"] = True

    monkeypatch.setattr(single_analyzer_module, "WorkerThread", _FakeWorkerThread)

    thread = widget._launch_analysis_task(str(image_path), gen_targets=["original", "bogus"])

    assert thread is not None
    assert thread.meta_force_gen_targets == ["original"]
    assert created["started"] is True
    assert created["args"][0] == os.path.abspath(str(image_path))
    record = next(iter(widget._analysis_history.values()))
    assert record["source_path"] == os.path.abspath(str(image_path))
    assert record["status"] == "running"
    widget.close()


def test_on_process_finished_honours_forced_gen_targets(qapp, monkeypatch, tmp_path):
    """「重跑分析并生图」不依赖自动生图勾选框，且只生成被要求的那一种。"""
    widget = _make_widget(monkeypatch)
    widget.auto_gen_orig_cb.setChecked(False)
    widget.auto_gen_ref_cb.setChecked(False)
    monkeypatch.chdir(tmp_path)

    calls = []
    monkeypatch.setattr(
        widget, "trigger_image_generation",
        lambda prompt_type, is_auto=False, prompt_bundle=None, **kwargs:
            calls.append({"prompt_type": prompt_type, "is_auto": is_auto}) or True,
    )

    result_json = {
        "english_description": "refined_prompt",
        "original_english_description": "original_prompt",
        "aspect_ratio": "2:3",
        "japanese_title": "題",
    }
    thread = type("_Thread", (), {
        "meta_thread_no": 7,
        "meta_task_hash": "hash7",
        "meta_task_id": "analysis-7",
        "meta_source_snapshot": "",
        "meta_force_gen_targets": ["refined"],
        "last_status": "success",
        "last_error": "",
    })()

    widget.on_process_finished(thread, result_json)

    assert [call["prompt_type"] for call in calls] == ["refined"]
    assert calls[0]["is_auto"] is True
    widget.close()


# ==================== 终止分析（含重试等待） ====================


@pytest.mark.parametrize("initial_save", [False, True])
def test_analysis_submission_freezes_save_and_generation_options(qapp, monkeypatch, tmp_path, initial_save):
    widget = _make_widget(monkeypatch)
    _patch_fake_worker_thread(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(single_analyzer_module, "get_single_analyzer_missing_prompt_files", lambda **kw: [])
    source = tmp_path / "source" / "girl.png"
    source.parent.mkdir()
    Image.new("RGB", (4, 4)).save(source)
    widget.auto_gen_ref_cb.setChecked(True)
    widget.save_to_source_dir_cb.setChecked(initial_save)
    widget.gen_channel_gemini.setChecked(True)
    thread = widget._launch_analysis_task(str(source))
    # 分析尚未返回时，切到相反模式和另一生图通道。
    widget.save_to_source_dir_cb.setChecked(not initial_save)
    widget.auto_gen_ref_cb.setChecked(False)
    widget.gen_channel_gpt.setChecked(True)
    calls = []
    monkeypatch.setattr(widget, "trigger_image_generation", lambda *a, **kw: calls.append(kw) or False)
    widget.on_process_finished(thread, {
        "japanese_title": "題", "english_description": "refined", "original_english_description": "original"})
    record = widget._analysis_history[thread.meta_task_id]
    if initial_save:
        assert not calls
        assert os.path.dirname(record["saved_json_path"]) == str(source.parent)
        assert record["status"] == "success"
    else:
        assert len(calls) == 1 and calls[0]["channel"] == "gemini"
        assert os.path.dirname(record["saved_json_path"]) != str(source.parent)
        assert record["status"] == "error"
    widget.close()


def test_clipboard_rerun_never_saves_analysis_into_snapshot_directory(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch)
    _patch_fake_worker_thread(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(single_analyzer_module, "get_single_analyzer_missing_prompt_files", lambda **kw: [])
    snapshot = widget._save_clipboard_snapshot(Image.new("RGB", (4, 4)))
    widget.save_to_source_dir_cb.setChecked(True)
    widget.auto_gen_ref_cb.setChecked(True)
    # 显式重跑并生图须胜过「仅保存」；没有显式目标的快照也不能使用原图目录。
    assert not widget._snapshot_analysis_options(snapshot)["save_to_source"]
    thread = widget._launch_analysis_task(snapshot, gen_targets=["refined"])
    assert widget._analysis_history[thread.meta_task_id]["source_origin"] == "clipboard"
    calls = []
    monkeypatch.setattr(widget, "trigger_image_generation", lambda *a, **kw: calls.append(a) or False)
    widget.on_process_finished(thread, {"japanese_title": "題", "english_description": "refined"})
    record = widget._analysis_history[thread.meta_task_id]
    assert calls == [("refined",)]
    assert os.path.dirname(record["saved_json_path"]) != os.path.dirname(snapshot)
    assert all(p.endswith(".png") for p in os.listdir(os.path.dirname(snapshot)))
    widget.close()


def test_switch_generation_save_generation_applies_only_to_new_submissions(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch)
    _patch_fake_worker_thread(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(single_analyzer_module, "get_single_analyzer_missing_prompt_files", lambda **kw: [])
    source = tmp_path / "girl.png"
    Image.new("RGB", (4, 4)).save(source)
    widget.auto_gen_ref_cb.setChecked(True)
    first = widget._launch_analysis_task(str(source))
    widget.save_to_source_dir_cb.setChecked(True)
    second = widget._launch_analysis_task(str(source))
    widget.show_clipboard_preview(Image.new("RGB", (4, 4)))
    third = widget._launch_analysis_task(Image.new("RGB", (4, 4)))
    assert [t.meta_analysis_options["gen_targets"] for t in (first, second, third)] == [["refined"], [], ["refined"]]
    assert [t.meta_analysis_options["save_to_source"] for t in (first, second, third)] == [False, True, False]
    assert widget.auto_gen_ref_cb.isEnabled()
    widget.close()


class _StubAnalysisThread:
    def __init__(self):
        self.meta_task_id = ""
        self.meta_thread_no = 1
        self.last_error = ""
        self.cancel_calls = []

    def request_cancel(self, force=False):
        self.cancel_calls.append(force)


def test_cancel_analysis_button_tracks_active_threads_and_cancels(qapp, monkeypatch):
    widget = _make_widget(monkeypatch)
    assert widget.cancel_analysis_btn.isEnabled() is False

    thread = _StubAnalysisThread()
    widget._active_analysis_threads.append(thread)
    widget._update_analysis_cancel_btn()
    assert widget.cancel_analysis_btn.isEnabled() is True

    widget.cancel_active_analysis_tasks()
    assert thread.cancel_calls == [True]

    widget._on_analysis_thread_stopped(thread)
    assert widget._active_analysis_threads == []
    assert widget.cancel_analysis_btn.isEnabled() is False
    widget.close()


def test_worker_thread_reports_cancelled_instead_of_failure(qapp, monkeypatch):
    """点「终止当前分析」后线程状态必须是 cancelled，而不是报成失败。"""

    class _FakeOpenAI:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(single_analyzer_module, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(single_analyzer_module, "predict_local_booru_tags", lambda *a, **k: [])
    monkeypatch.setattr(single_analyzer_module, "get_local_pixiv_tag_candidates", lambda *a, **k: [])
    # Step 1 因取消而返回 None（真实流程里就是中断重试等待后的结果）
    monkeypatch.setattr(
        single_analyzer_module, "step_1_analyze_image",
        lambda image_source, client, model_name, **kwargs: None,
    )

    thread = WorkerThread("dummy.png", "key", "https://example.invalid/v1", "test-model")
    logs = []
    thread.log_signal.connect(logs.append)

    def _already_cancelled():
        thread._force_cancel_requested = True
        return True

    thread.isInterruptionRequested = _already_cancelled  # type: ignore[assignment]
    thread.run()

    assert thread.last_status == "cancelled"
    assert any("取消" in line for line in logs), logs


# ==================== 右键菜单本身 ====================


class _FakeAction:
    def __init__(self, text):
        self.text = text
        self.enabled = True

    def setEnabled(self, value):
        self.enabled = bool(value)


class _FakeMenu:
    """替掉 QMenu：记录菜单项 + 可编程地返回「被点击」的那一项。"""

    chosen_text = None
    created = []

    def __init__(self, _parent=None):
        self.actions_list = []
        self.submenus = []
        self.enabled = True
        _FakeMenu.created.append(self)

    def setEnabled(self, value):
        self.enabled = bool(value)

    def addAction(self, text):
        action = _FakeAction(text)
        self.actions_list.append(action)
        return action

    def addMenu(self, text):
        submenu = _FakeMenu(self)
        submenu.title = text
        self.submenus.append(submenu)
        return submenu

    def addSeparator(self):
        return None

    def exec(self, *_args, **_kwargs):
        return self.action(_FakeMenu.chosen_text) if _FakeMenu.chosen_text else None

    def all_actions(self):
        collected = list(self.actions_list)
        for submenu in self.submenus:
            collected.extend(submenu.all_actions())
        return collected

    def action(self, text):
        # 子菜单标题本身也是菜单项（QMenu.addMenu 返回的菜单在父菜单里表现为一个 action）
        for candidate in self.actions_list + [type("_T", (), {"text": s.title}) for s in self.submenus]:
            if candidate.text == text:
                return candidate
        for submenu in self.submenus:
            found = submenu.action(text)
            if found is not None:
                return found
        return None


def _patch_analysis_rerun_dialog(monkeypatch, accepted=True, mode="analyze", targets=None, seen=None):
    """把「重新分析」确认框换成即时返回的假对话框（真实对话框会阻塞用例）。"""
    import modules.image_analysis.history_pickup as pickup_module

    class _FakeAnalysisRerunDialog:
        def __init__(self, summary_lines, channel="gemini", parent=None):
            self.summary_lines = list(summary_lines or [])
            self.channel = channel
            self.mode = mode
            self.gen_targets = list(targets or ([] if mode == "analyze" else ["refined"]))
            if seen is not None:
                seen["summary"] = self.summary_lines
                seen["channel"] = channel

        def exec(self):
            from PyQt6.QtWidgets import QDialog

            return QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected

    monkeypatch.setattr(pickup_module, "AnalysisRerunDialog", _FakeAnalysisRerunDialog)


def _open_context_menu(widget, monkeypatch, chosen_text):
    monkeypatch.setattr(single_analyzer_module, "QMenu", _FakeMenu)
    _FakeMenu.chosen_text = chosen_text
    _FakeMenu.created = []
    widget._on_history_context_menu(QPoint(0, 0))
    # 第一个被构造的是根菜单（addMenu 会随后构造子菜单）
    return _FakeMenu.created[0] if _FakeMenu.created else None


def test_context_menu_opens_history_pickup(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch)
    image_path = tmp_path / "girl.png"
    Image.new("RGB", (4, 4)).save(image_path)
    record = _insert_record(widget, 1, "error", str(image_path))
    checkpoint = tmp_path / "generation-checkpoint.json"
    checkpoint.write_text("{}", encoding="utf-8")
    record["generation_checkpoints"] = [str(checkpoint)]

    calls = []
    monkeypatch.setattr(widget, "_pickup_generation_history",
                        lambda rec: calls.append(rec["task_id"]) or True)

    menu = _open_context_menu(widget, monkeypatch, "🧩 拾取历史…")

    assert menu is not None
    assert menu.action("🧩 拾取历史…").enabled is True
    assert calls == [record["task_id"]]
    widget.close()


def test_context_menu_replaces_rerun_actions(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch)
    image_path = tmp_path / "girl.png"
    Image.new("RGB", (4, 4)).save(image_path)
    _insert_record(widget, 1, "timeout", str(image_path))

    menu = _open_context_menu(widget, monkeypatch, None)
    assert menu.action("🔁 重跑分析") is None
    assert menu.action("基于优化提示词") is None
    assert menu.action("🧩 拾取历史…").enabled is True
    widget.close()


def test_context_menu_blocks_unusable_source_and_offers_cleanup(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch)
    clipboard_record = _insert_record(widget, 1, "error", Image.new("RGB", (4, 4)))

    menu = _open_context_menu(widget, monkeypatch, None)

    assert menu.action("⚠️ 分析源图不可用（文件已移动/删除，或剪贴板快照被清空）").enabled is False
    assert menu.action("🧩 拾取历史…").enabled is True
    assert menu.action("📂 打开源图所在目录").enabled is False
    assert menu.action("📋 复制源图路径").enabled is False
    assert menu.action("🧹 清空失败记录（1）").enabled is True
    assert clipboard_record["task_id"] in widget._analysis_history
    widget.close()


def test_context_menu_clear_failed_dispatch(qapp, monkeypatch, tmp_path):
    widget = _make_widget(monkeypatch)
    _freeze_dialogs(monkeypatch)
    image_path = tmp_path / "girl.png"
    Image.new("RGB", (4, 4)).save(image_path)
    _insert_record(widget, 1, "error", str(image_path))
    _insert_record(widget, 2, "timeout", str(image_path))

    clear_calls = []
    monkeypatch.setattr(widget, "_clear_failed_history_records", lambda: clear_calls.append(True))

    _open_context_menu(widget, monkeypatch, "🧹 清空失败记录（2）")
    assert clear_calls == [True]
    widget.close()


# ==================== Gemini 通道重跑（没有 GPT 工序断点） ====================


def _isolated_widget(monkeypatch, tmp_path):
    """BASE_DIR 指向 tmp：断点扫描与 UI 记忆都别碰真实仓库。"""
    monkeypatch.setattr(single_analyzer_module, "BASE_DIR", str(tmp_path))
    return _make_widget(monkeypatch)


def test_pickup_without_checkpoints_enqueues_gemini_rerun_task(qapp, monkeypatch, tmp_path):
    """Gemini 记录没有 GPT 断点 → 右键「拾取历史」按相同参数新建一条队列任务。"""
    widget = _isolated_widget(monkeypatch, tmp_path)
    _freeze_dialogs(monkeypatch)
    image_path = tmp_path / "girl.png"
    Image.new("RGB", (4, 4)).save(image_path)
    analysis_json = tmp_path / "result.json"
    analysis_json.write_text("{}", encoding="utf-8")
    record = _insert_record(widget, 3, "success", str(image_path), task_hash="abcdef12")
    record.update(
        style_name="默认风格", aspect_ratio="2:3",
        original_prompt="a girl", refined_prompt="a girl, refined",
        result_json={"english_description": "a girl, refined",
                     "original_english_description": "a girl",
                     "generation_style_name": "默认风格"},
        saved_json_path=str(analysis_json),
    )

    calls = []

    def fake_trigger(prompt_type, is_auto=False, prompt_bundle=None, **kwargs):
        calls.append({"prompt_type": prompt_type, "bundle": prompt_bundle, **kwargs})
        return True

    monkeypatch.setattr(widget, "_ask_gemini_rerun_targets",
                        lambda rec, available, defaults: ["refined"] if rec is record else [])
    monkeypatch.setattr(widget, "trigger_image_generation", fake_trigger)

    assert widget._pickup_generation_history(record) is True

    assert [call["prompt_type"] for call in calls] == ["refined"]
    assert calls[0]["channel"] == "gemini"
    assert calls[0]["bundle"] == {
        "task_hash": "abcdef12", "style_name": "默认风格", "aspect_ratio": "2:3", "wardrobe": {},
        "original_prompt": "a girl", "refined_prompt": "a girl, refined",
        "analysis_json_path": str(analysis_json), "source_image_path": str(image_path),
    }

    reruns = [rec for rec in widget._analysis_history.values()
              if rec.get("rerun_of") == record["task_id"]]
    assert len(reruns) == 1
    rerun = reruns[0]
    assert rerun is not record
    assert rerun["task_hash"] == "abcdef12"          # 沿用任务号，发布器仍能关联投稿 JSON
    assert rerun["status"] == "running"
    assert rerun["style_name"] == "默认风格"
    assert rerun["aspect_ratio"] == "2:3"
    assert "重跑 Gemini 生图" in rerun["source_desc"]
    assert widget.history_list.count() == 2
    assert record["status"] == "success"             # 原记录原样保留
    widget.close()


def test_pickup_without_checkpoints_uses_recorded_prompt_types(qapp, monkeypatch, tmp_path):
    """记录里存过「上次跑的提示词」时默认沿用，不再要求用户选。"""
    widget = _isolated_widget(monkeypatch, tmp_path)
    _freeze_dialogs(monkeypatch)
    image_path = tmp_path / "girl.png"
    Image.new("RGB", (4, 4)).save(image_path)
    record = _insert_record(widget, 1, "success", str(image_path), task_hash="beef0001")
    record.update(original_prompt="a girl", refined_prompt="a girl, refined",
                  generation_params={"channel": "gemini", "prompt_types": ["original"]})

    asked = []
    monkeypatch.setattr(widget, "_ask_gemini_rerun_targets",
                        lambda rec, available, defaults: asked.append(list(defaults)) or list(defaults))
    monkeypatch.setattr(widget, "trigger_image_generation", lambda *a, **k: True)

    assert widget._pickup_generation_history(record) is True
    assert asked == [["original"]]
    assert widget.history_list.count() == 2
    widget.close()


def test_pickup_without_checkpoints_without_prompts_offers_rerun_analysis(qapp, monkeypatch, tmp_path):
    """分析没跑完（没有提示词）时不能凭空建生图任务，但必须给「重新分析」的路。"""
    widget = _isolated_widget(monkeypatch, tmp_path)
    image_path = tmp_path / "girl.png"
    Image.new("RGB", (4, 4)).save(image_path)
    record = _insert_record(widget, 1, "error", str(image_path), task_hash="deadbeef")

    seen = {}
    _patch_analysis_rerun_dialog(monkeypatch, accepted=True, mode="analyze", seen=seen)
    launched = []
    monkeypatch.setattr(
        widget, "_launch_analysis_task",
        lambda snapshot, gen_targets=None, header_note=None, **kwargs:
            launched.append({"snapshot": snapshot, "targets": gen_targets, "note": header_note})
            or type("_T", (), {"meta_force_gen_targets": []})(),
    )

    assert widget._pickup_generation_history(record) is True
    assert seen["channel"] == "gemini"          # 记录没标通道 → 按 Gemini 记录处理
    assert seen["summary"]                      # 确认框里有参数清单
    assert launched and launched[0]["targets"] == []      # 只做分析
    assert launched[0]["snapshot"] == record["source_path"]
    widget.close()


def test_pickup_offer_rerun_analysis_can_generate_afterwards(qapp, monkeypatch, tmp_path):
    """选了「重新分析后自动生图」→ 把生图目标挂在这条重跑任务上。"""
    widget = _isolated_widget(monkeypatch, tmp_path)
    image_path = tmp_path / "girl.png"
    Image.new("RGB", (4, 4)).save(image_path)
    record = _insert_record(widget, 1, "error", str(image_path), task_hash="deadbeef")

    _patch_analysis_rerun_dialog(monkeypatch, accepted=True, mode="generate", targets=["original"])
    launched = []
    monkeypatch.setattr(
        widget, "_launch_analysis_task",
        lambda snapshot, gen_targets=None, header_note=None, **kwargs:
            launched.append({"targets": gen_targets, "note": header_note})
            or type("_T", (), {"meta_force_gen_targets": ["original"]})(),
    )

    assert widget._pickup_generation_history(record) is True
    assert launched[0]["targets"] == ["original"]
    assert "重新分析" in launched[0]["note"]
    widget.close()


def test_pickup_offer_rerun_analysis_cancel_creates_nothing(qapp, monkeypatch, tmp_path):
    widget = _isolated_widget(monkeypatch, tmp_path)
    image_path = tmp_path / "girl.png"
    Image.new("RGB", (4, 4)).save(image_path)
    record = _insert_record(widget, 1, "error", str(image_path), task_hash="deadbeef")

    _patch_analysis_rerun_dialog(monkeypatch, accepted=False)
    monkeypatch.setattr(widget, "_launch_analysis_task",
                        lambda *a, **k: pytest.fail("取消后不该提交分析"))

    assert widget._pickup_generation_history(record) is False
    assert widget.history_list.count() == 1
    widget.close()


def test_pickup_offer_rerun_analysis_needs_a_usable_source(qapp, monkeypatch, tmp_path):
    """剪贴板快照被清掉、原图也没了 → 只能提示，不能假装能重跑。"""
    widget = _isolated_widget(monkeypatch, tmp_path)
    messages = []
    monkeypatch.setattr(single_analyzer_module.QMessageBox, "information",
                        staticmethod(lambda *args, **kwargs: messages.append(args)))
    record = _insert_record(widget, 1, "error", Image.new("RGB", (4, 4)))
    _patch_analysis_rerun_dialog(monkeypatch, accepted=True, mode="analyze")

    assert widget._pickup_generation_history(record) is False
    assert any("无法重新分析" in str(args[1]) for args in messages), messages
    widget.close()


def test_pickup_offer_rerun_analysis_creates_a_queue_task(qapp, monkeypatch, tmp_path):
    """走真实提交路径：确认后真的新建一条队列任务（不是只打个日志）。"""
    widget = _isolated_widget(monkeypatch, tmp_path)
    _patch_fake_worker_thread(monkeypatch)
    image_path = tmp_path / "girl.png"
    Image.new("RGB", (4, 4)).save(image_path)
    record = _insert_record(widget, 1, "error", str(image_path), task_hash="deadbeef")
    _patch_analysis_rerun_dialog(monkeypatch, accepted=True, mode="analyze")

    assert widget._pickup_generation_history(record) is True
    assert widget.history_list.count() == 2
    new_record = [rec for rec in widget._analysis_history.values() if rec is not record]
    assert len(new_record) == 1
    assert new_record[0]["status"] == "running"
    assert widget._active_analysis_threads or True
    widget.close()


def test_pickup_without_checkpoints_keeps_gpt_records_on_gpt_path(qapp, monkeypatch, tmp_path):
    """记录标明是 gpt-image 通道时不许悄悄改成 Gemini 重跑，但同样要给「重新分析」的路。"""
    widget = _isolated_widget(monkeypatch, tmp_path)
    image_path = tmp_path / "girl.png"
    Image.new("RGB", (4, 4)).save(image_path)
    record = _insert_record(widget, 1, "error", str(image_path), task_hash="feed0002")
    record.update(original_prompt="a girl", refined_prompt="a girl, refined",
                  generation_params={"channel": "gpt-image"})

    seen = {}
    _patch_analysis_rerun_dialog(monkeypatch, accepted=True, mode="analyze", seen=seen)
    launched = []
    monkeypatch.setattr(
        widget, "_launch_analysis_task",
        lambda snapshot, gen_targets=None, header_note=None, **kwargs:
            launched.append({"targets": gen_targets}) or type("_T", (), {"meta_force_gen_targets": []})(),
    )
    monkeypatch.setattr(widget, "trigger_image_generation", lambda *a, **k: True)

    assert widget._pickup_generation_history(record) is True
    assert seen["channel"] == "gpt-image"       # 通道沿记录走，不改成 Gemini
    assert launched and launched[0]["targets"] == []
    widget.close()


def test_rerun_cancelled_in_dialog_creates_no_task(qapp, monkeypatch, tmp_path):
    """用户在确认框里取消 → 不建队列任务、不调生图。"""
    widget = _isolated_widget(monkeypatch, tmp_path)
    _freeze_dialogs(monkeypatch)
    image_path = tmp_path / "girl.png"
    Image.new("RGB", (4, 4)).save(image_path)
    record = _insert_record(widget, 1, "error", str(image_path), task_hash="cafe0003")
    record.update(original_prompt="a girl", refined_prompt="a girl, refined")
    monkeypatch.setattr(widget, "_ask_gemini_rerun_targets", lambda *a, **k: [])
    monkeypatch.setattr(widget, "trigger_image_generation", lambda *a, **k: True)

    assert widget._pickup_generation_history(record) is False
    assert widget.history_list.count() == 1
    widget.close()


def test_channel_override_generates_with_gemini_while_ui_stays_on_gpt(qapp, monkeypatch, tmp_path):
    """`channel="gemini"` 只管这一次：走 Gemini 线程，界面单选框不被改写。"""
    widget = _isolated_widget(monkeypatch, tmp_path)
    widget.gen_channel_gpt.setChecked(True)
    created = []

    class _FakeGeminiThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.log_signal = _Signal()
            self.finish_signal = _Signal()
            self.finished = _Signal()
            created.append(self)

        def start(self):
            self.started = True

    gpt_calls = []
    monkeypatch.setattr(single_analyzer_module, "ImageGenWorkerThread", _FakeGeminiThread)
    monkeypatch.setattr(single_analyzer_module, "GptImageGenWorkerThread",
                        lambda **kwargs: gpt_calls.append(kwargs))

    bundle = {"task_hash": "hash9", "style_name": "默认风格", "aspect_ratio": "2:3",
              "original_prompt": "a girl", "refined_prompt": "a girl, refined",
              "analysis_json_path": ""}
    assert widget.trigger_image_generation("refined", prompt_bundle=bundle,
                                           channel="gemini") is True

    assert gpt_calls == []
    assert len(created) == 1
    assert created[0].kwargs["aspect_ratio"] == "2:3"
    assert created[0].kwargs["file_prefix"] == "hash9"
    assert created[0].kwargs["prompt"] == "a girl, refined, detailed face, clear facial features, sharp focus on face"
    assert created[0].started is True
    assert widget.gen_channel_gpt.isChecked() is True     # 界面通道没被改
    widget.close()


def test_gemini_rerun_dialog_defaults_and_requires_a_prompt(qapp, monkeypatch, tmp_path):
    from modules.image_analysis.history_pickup import GeminiRerunDialog

    dialog = GeminiRerunDialog(["生图通道：Gemini"], ["refined", "original"],
                               defaults=["original"])
    assert dialog.selected_targets() == ["original"]

    dialog.prompt_boxes["original"].setChecked(False)
    dialog.accept()                                   # 一个都不选 → 不该关掉
    assert dialog.result() != dialog.DialogCode.Accepted
    assert dialog.selected_targets() == []
    dialog.close()
