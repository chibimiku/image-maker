"""Real-thread UI regressions; no generation or model API calls."""
import os
import threading
from copy import deepcopy

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from modules.image_generation import gpt_image2_tab as tab
from utils.gui_shutdown import running_threads


@pytest.fixture(scope='session')
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def widget(qapp, monkeypatch):
    monkeypatch.setattr(tab, 'get_api_config', lambda **kwargs: {'api_key': ''})
    for method in ('load_defaults', 'save_defaults', 'save_repaint_defaults',
                   '_refresh_budget', '_start_pricing_refresh'):
        monkeypatch.setattr(tab.GptImage2Widget, method, lambda self: None)
    w = tab.GptImage2Widget()
    yield w
    w.refine_timer.stop()
    w.deleteLater()
    qapp.processEvents()


def test_missing_styles_during_init_keeps_window_usable(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(tab, 'CONFIG_STYLES_FILE', str(tmp_path / 'missing-styles.json'))
    monkeypatch.setattr(tab, 'get_api_config', lambda **kwargs: {'api_key': ''})
    for method in ('load_defaults', 'save_defaults', 'save_repaint_defaults',
                   '_refresh_budget', '_start_pricing_refresh'):
        monkeypatch.setattr(tab.GptImage2Widget, method, lambda self: None)
    w = tab.GptImage2Widget()
    try:
        assert 'missing-styles.json' in w.log_view.toPlainText()
        assert w.generate_btn.isEnabled()
    finally:
        w.refine_timer.stop()
        w.deleteLater()
        qapp.processEvents()


class HeldResult(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(list)

    def __init__(self, *args, **kwargs):
        super().__init__(kwargs.get('parent'))
        self.release = threading.Event()
        self.entered = threading.Event()
        self.delete_calls = []
        self.steps = deepcopy(args[1]) if len(args) > 1 else {}
        self.style_ref_path = kwargs.get('style_ref_path')

    def run(self):
        self.entered.set()
        self.done.emit(['fake-result.jpg'])
        self.release.wait(5)

    def deleteLater(self):
        self.delete_calls.append(self.isRunning())
        super().deleteLater()


def pump_until(qapp, predicate):
    for _ in range(100):
        qapp.processEvents()
        if predicate():
            return
        QTest.qWait(10)
    assert predicate()


def test_result_before_thread_exit_keeps_busy_and_does_not_delete(widget, qapp, monkeypatch):
    monkeypatch.setattr(tab, 'PostProcessWorker', HeldResult)
    outputs = []
    monkeypatch.setattr(widget, '_show_results', outputs.append)
    widget._start_post_process(['input.jpg'], {'steps': {'ink': {'enabled': True}}, 'style_ref_path': 'style-A'})
    worker = widget._post_worker
    try:
        pump_until(qapp, lambda: bool(outputs))
        assert worker.isRunning() and widget._post_worker is worker
        assert not worker.delete_calls
        assert not widget.generate_btn.isEnabled()
        assert not widget.repaint_results_btn.isEnabled()
        assert not widget.cancel_btn.isEnabled()  # Local post-processing is not cancellable.
        old_generation = QThread()
        widget._worker = old_generation
        widget._on_worker_finished(old_generation)
        assert not widget.generate_btn.isEnabled()  # Generating finished; post is still busy.
    finally:
        worker.release.set()
        worker.wait(2000)
    pump_until(qapp, lambda: widget._post_worker is None)
    assert worker.delete_calls == [False]
    assert widget.generate_btn.isEnabled() and widget.repaint_results_btn.isEnabled()


def test_all_paid_entry_points_refuse_reentry(widget, monkeypatch):
    worker = QThread()
    widget._worker = worker
    def unexpected(*args, **kwargs):
        pytest.fail('Busy UI attempted to construct a second paid worker')
    monkeypatch.setattr(tab, 'GptImage2Worker', unexpected)
    monkeypatch.setattr(widget, 'current_repaint_config', unexpected)
    widget.generate()
    widget.repaint_selected_results()
    widget.retry_failed_post_steps()
    widget.retry_refine()
    assert widget._worker is worker
    widget._worker = None
    widget._post_worker = worker
    widget.generate()
    widget.repaint_selected_results()
    widget.retry_failed_post_steps()
    assert widget._post_worker is worker
    widget._post_worker = None


def test_late_finished_cannot_clear_new_worker(widget):
    old, current = QThread(), QThread()
    widget._worker = current
    widget._on_worker_finished(old)
    assert widget._worker is current and not widget.generate_btn.isEnabled()
    widget._on_worker_finished(current)
    assert widget._worker is None and widget.generate_btn.isEnabled()


def test_post_recipe_freezes_flags_and_reference_before_result(widget, monkeypatch):
    flags = {'tone': {'enabled': True}, 'ink': {'enabled': False}}
    reference = ['style-A']
    monkeypatch.setattr(widget, 'post_pipeline_steps', lambda: flags)
    monkeypatch.setattr(widget, 'tone_reference_path', lambda: reference[0])
    monkeypatch.setattr(widget, 'current_style_block', lambda: ('style', reference[0]))
    recipe = widget._freeze_post_recipe()
    flags['ink']['enabled'] = True
    flags['tone']['enabled'] = False
    reference[0] = 'style-B'
    calls = []
    monkeypatch.setattr(widget, '_start_post_process', lambda paths, frozen: calls.append(frozen))
    widget.on_done(['input.jpg'], recipe)
    assert calls == [recipe]
    assert recipe['steps']['ink']['enabled'] is False
    assert recipe['steps']['tone'] == {'enabled': True, 'reference_path': 'style-A'}
    assert recipe['style_ref_path'] == 'style-A'


def test_models_return_to_requested_site_without_overwriting_current(widget, monkeypatch):
    monkeypatch.setattr(widget, 'current_site', lambda: tab.SITE_AUTODL)
    calls = []
    monkeypatch.setattr(widget, '_set_model_choices', lambda *args: calls.append(args))
    widget._on_models_fetched(['aigc-only'], tab.SITE_AIGC2D)
    assert widget._live_models[tab.SITE_AIGC2D] == ['aigc-only'] and not calls
    assert tab.SITE_AUTODL not in widget._live_models
    widget._on_models_fetched(['gpt-image-2'], tab.SITE_AUTODL)
    assert calls and widget._live_models[tab.SITE_AUTODL] == ['gpt-image-2']


def test_pricing_result_waits_for_finished_before_cleanup(widget, qapp, monkeypatch):
    worker = HeldResult(widget)
    widget._pricing_worker = worker
    budgets = []
    monkeypatch.setattr(widget, '_refresh_budget', lambda: budgets.append(True))
    worker.done.connect(lambda _: widget._on_pricing_ready(True))
    worker.finished.connect(lambda: widget._on_pricing_finished(worker))
    worker.start()
    try:
        pump_until(qapp, lambda: bool(budgets))
        assert worker.isRunning() and widget._pricing_worker is worker
        assert not worker.delete_calls
    finally:
        worker.release.set()
        worker.wait(2000)
    pump_until(qapp, lambda: widget._pricing_worker is None)
    assert worker.delete_calls == [False]


def test_close_waits_for_parentless_worker_and_delivers_result(widget, qapp):
    worker = HeldResult()
    widget._external_workers = {'active': [worker]}
    results = []
    worker.done.connect(results.append)
    widget.show()
    worker.start()
    assert worker.entered.wait(2)
    try:
        assert worker in running_threads(widget)
        assert widget.close() is False
        assert widget.isVisible() and not widget.isEnabled()
        pump_until(qapp, lambda: bool(results))
        assert results == [['fake-result.jpg']]
    finally:
        worker.release.set()
        worker.wait(2000)
    pump_until(qapp, lambda: not widget.isVisible())
    assert not worker.isRunning()


def test_close_waits_for_next_stage_started_by_queued_result(widget, qapp):
    first, second = HeldResult(), HeldResult()
    widget._external_workers = [first]
    def next_stage(_):
        widget._external_workers = [second]
        second.start()
    first.done.connect(next_stage)
    widget.show()
    first.start()
    assert first.entered.wait(2)
    try:
        assert widget.close() is False
        pump_until(qapp, lambda: second.isRunning())
        first.release.set()
        first.wait(2000)
        QTest.qWait(250)
        assert widget.isVisible() and second.isRunning()
    finally:
        first.release.set()
        second.release.set()
        first.wait(2000)
        second.wait(2000)
    pump_until(qapp, lambda: not widget.isVisible())
