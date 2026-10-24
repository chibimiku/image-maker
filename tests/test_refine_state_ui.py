import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PIL import Image
from PyQt6.QtWidgets import QApplication
from utils import refine_state as state
from modules.image_generation import gpt_image2_tab as tab

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')

@pytest.fixture
def case(tmp_path):
    root = tmp_path / '20261009'
    root.mkdir()
    paths = [root / name for name in ('source.png', 'person.png', 'background.png', 'final-tone.png')]
    for p in paths:
        Image.new('RGB', (120, 80)).save(p)
    source, person, bg, final = map(str, paths)
    record = {'source': source, 'source_sha256': hashlib.sha256(paths[0].read_bytes()).hexdigest(),
              'selected': person, 'hair_check': {'decision': 'skip'}, 'stages': [
                  {'key': 'person', 'kind': 'image', 'status': 'accepted', 'input': source, 'outputs': [person], 'options': {'aspect_ratio': '3:2'}},
                  {'key': 'background', 'kind': 'image', 'status': 'rejected', 'input': person, 'outputs': [bg], 'options': {'aspect_ratio': '3:2'}},
                  {'key': 'background-audit', 'kind': 'text', 'status': 'returned', 'report': {'verdict': 'fail', 'unacceptable_changes': ['Window fragments remain.']}}]}
    rp = root / 'gpt_image_repaint' / 'test-refine.json'
    dump(rp, record)
    dump(root / 'pipeline-steps/run/pipeline-manifest.json', {'items': [{'source': person, 'final': final, 'steps': []}]})
    return paths, rp, record

def test_final_local_product_maps_through_manifest_and_exposes_actual_failure(case):
    paths, rp, record = case
    found = state.find_record(str(paths[3]))
    assert Path(found[0]) == rp
    assert state.default_stage(found[1]) == 'background'
    text = state.status_text(found[1])
    assert '背景：未通过' in text and 'Window fragments remain.' in text
    assert '头发：无需补修' in text and '光照：尚未执行' in text

def test_exact_path_lookup_does_not_match_similar_filename(case):
    paths, _, _ = case
    assert state.find_record(str(paths[0].with_name('source-copy.png'))) is None

def test_explicit_child_run_supersedes_displayed_old_product(case):
    paths, rp, record = case
    child = dict(record, parent_record=str(rp), selected=str(paths[2]), manual_override={'stage': 'background'}, stages=[])
    cp = rp.with_name('child-refine.json')
    dump(cp, child)
    found = state.find_record(str(paths[3]))
    assert Path(found[0]) == cp
    assert '人工采用' in state.status_text(found[1])

def test_force_cannot_adopt_an_unreturned_candidate(case):
    _, rp, record = case
    record['stages'][1]['status'] = 'requested'
    dump(rp, record)
    with pytest.raises(ValueError, match='没有已返回'):
        state.resume_plan(rp, 'background', True)

@pytest.fixture
def widget(monkeypatch):
    from utils import gpt_workflow_settings
    monkeypatch.setattr(gpt_workflow_settings, "postprocess_enabled", lambda **kw: True)
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(tab.GptImage2Widget, 'load_defaults', lambda self: None)
    monkeypatch.setattr(tab.GptImage2Widget, 'save_defaults', lambda self: None)
    monkeypatch.setattr(tab.GptImage2Widget, '_refresh_budget', lambda self: None)
    monkeypatch.setattr(tab.GptImage2Widget, '_start_pricing_refresh', lambda self: None)
    w = tab.GptImage2Widget()
    yield w
    w.refine_timer.stop()
    w.deleteLater()
    app.processEvents()

def test_ui_shows_refine_status_for_tone_product_and_enables_stage_actions(widget, case):
    paths, rp, _ = case
    widget._show_results([str(paths[3])])
    assert Path(widget._refine_record_path) == rp
    assert widget.refine_stage_combo.currentData() == 'background'
    assert widget.refine_retry_btn.isEnabled() and widget.refine_force_btn.isEnabled()
    assert 'Window fragments remain.' in widget.refine_status_view.toPlainText()
    widget.refine_stage_combo.setCurrentIndex(widget.refine_stage_combo.findData('lighting'))
    assert not widget.refine_force_btn.isEnabled()


def test_finished_record_disables_adopt_and_continue_even_when_selecting_old_candidate(widget, case, monkeypatch):
    paths, rp, record = case
    widget._show_results([str(paths[3])])
    assert widget.refine_force_btn.isEnabled()
    record['status'] = 'finished'
    dump(rp, record)
    widget.refresh_refine_status()
    assert not widget.refine_force_btn.isEnabled()
    assert not widget.refine_continue_btn.isEnabled()
    assert widget.refine_retry_btn.isEnabled()  # Explicit redraw remains available.
    widget.refine_stage_combo.setCurrentIndex(widget.refine_stage_combo.findData('background'))
    assert not widget.refine_force_btn.isEnabled()
    monkeypatch.setattr(tab, 'RefineRetryWorker', lambda *a, **k: pytest.fail('Completed chain must not advance'))
    widget.continue_refine()
    widget.retry_refine(force=True)


def test_hybrid_ui_lists_models_and_clothing_checkpoint(widget, case):
    paths, rp, record = case
    record.update(workflow='hybrid', status='stopped', stopped_at='clothing', diagnosis={'has_people': True},
                  targeted_check={'decision': 'skip'},
                  stages=[{'key': 'whole', 'kind': 'image', 'status': 'accepted', 'options': {'model': 'flash'}},
                          {'key': 'background', 'kind': 'image', 'status': 'accepted', 'options': {'model': 'pro'}},
                          {'key': 'clothing', 'kind': 'image', 'status': 'rejected', 'input': str(paths[2]),
                           'outputs': [str(paths[1])], 'options': {'model': 'flash'}}])
    dump(rp, record)
    widget._show_results([str(paths[1])])
    assert widget.refine_stage_combo.count() == 4
    assert widget.refine_stage_combo.currentData() == 'clothing'
    assert '背景：通过（pro）' in widget.refine_status_view.toPlainText()


def test_hybrid_v9_stage_order_and_final_checkpoint():
    from utils.refine_state import stages_for_record, default_stage
    record = {'workflow': 'hybrid', 'version': 9, 'status': 'finished',
              'diagnosis': {'has_people': False}, 'targeted_check': {'decision': 'skip'}}
    assert stages_for_record(record) == ('targeted', 'background', 'clothing', 'whole')
    assert default_stage(record) == 'whole'
    record['version'] = 8
    assert stages_for_record(record) == ('targeted', 'whole', 'background', 'clothing')
    assert default_stage(record) == 'background'


def test_stage_model_controls_save_and_feed_automatic_request(widget, monkeypatch):
    saves = []
    monkeypatch.setattr(tab, 'save_repaint_config', lambda cfg: saves.append(dict(cfg)))
    widget.refine_background_model_combo.setCurrentText('custom-background')
    widget.refine_clothing_model_combo.setCurrentText('custom-clothing')
    assert saves[-1]['refine_background_model'] == 'custom-background'
    assert saves[-1]['refine_clothing_model'] == 'custom-clothing'
    widget.mode_combo.setCurrentText(tab.MODE_GENERATE)
    request = widget.build_repaint_request([])
    assert request['refine_background_model'] == 'custom-background'
    assert request['refine_clothing_model'] == 'custom-clothing'
    widget.mode_combo.setCurrentText(tab.MODE_REPAINT)
    assert 'refine_background_model' not in widget.build_repaint_request([])


def test_whole_ui_only_offers_combined_stage_and_preserves_failure_controls(widget, case):
    paths, rp, record = case
    record.update(workflow='whole', version=6, status='stopped', selected=str(paths[0]),
                  stages=[{'key': 'whole', 'kind': 'image', 'status': 'rejected',
                           'input': str(paths[0]), 'outputs': [str(paths[1])],
                           'options': {'aspect_ratio': '3:2'}},
                          {'key': 'whole-audit', 'kind': 'text', 'status': 'returned',
                           'report': {'verdict': 'fail', 'unacceptable_changes': ['Extra hand.']}}])
    dump(rp, record)
    widget._show_results([str(paths[1])])
    assert widget.refine_stage_combo.count() == 1
    assert widget.refine_stage_combo.currentData() == 'whole'
    assert widget.refine_retry_btn.isEnabled() and widget.refine_force_btn.isEnabled()
    text = widget.refine_status_view.toPlainText()
    assert '整体优化：未通过' in text and 'Extra hand.' in text
    assert '背景：尚未执行' not in text


def test_targeted_rejected_whole_accepted_shows_completed_and_defaults_to_whole(widget, case):
    paths, rp, record = case
    record.update(workflow='targeted_whole', version=7, status='finished', selected=str(paths[1]),
                  targeted_check={'decision': 'repair'},
                  material_guidance={'materials': [{'region': 'jacket', 'material': 'leather'}]},
                  stages=[{'key': 'targeted', 'kind': 'image', 'status': 'rejected'},
                          {'key': 'whole', 'kind': 'image', 'status': 'accepted',
                           'input': str(paths[0]), 'outputs': [str(paths[1])]}])
    dump(rp, record)
    widget._show_results([str(paths[1])])
    assert widget.refine_stage_combo.count() == 2
    assert widget.refine_stage_combo.currentData() == 'whole'
    text = widget.refine_status_view.toPlainText()
    assert 'Refine：已完成' in text and '针对性修复：未通过' in text
    assert 'jacket / leather' in text

def test_ui_manual_force_uses_explicit_stage_in_worker_without_local_post_processing(widget, case, monkeypatch):
    class Signal:
        def __init__(self): self.callbacks = []
        def connect(self, cb): self.callbacks.append(cb)
    class Worker:
        def __init__(self, path, stage, force, parent):
            self.args = (path, stage, force)
            self.log, self.done, self.error, self.finished = [Signal() for _ in range(4)]
            self.running = False
        def start(self): self.running = True
        def isRunning(self): return self.running
    monkeypatch.setattr(tab, 'RefineRetryWorker', Worker)
    widget._show_results([str(case[0][3])])
    widget.retry_refine(True)
    assert widget._worker.args == (str(case[1]), 'background', True)
    assert not widget.refine_retry_btn.isEnabled() and not widget.refine_force_btn.isEnabled()
    assert widget._worker.done.callbacks == [widget._show_results]
    widget._worker = None


def audit_failure(case):
    paths, rp, record = case
    record.update(workflow='hybrid', version=9, status='fallback', selected=str(paths[0]),
                  diagnosis={'has_people': True}, targeted_check={'decision': 'skip'},
                  stages=[{'key': 'background', 'kind': 'image', 'status': 'returned',
                           'input': str(paths[0]), 'outputs': [str(paths[2])], 'prompt': 'saved prompt',
                           'options': {'aspect_ratio': '3:2'}},
                          {'key': 'background-audit', 'kind': 'text', 'status': 'failed',
                           'system': 'saved audit', 'user': 'saved task',
                           'images': [str(paths[0]), str(paths[0]), str(paths[2])], 'error': 'TLS EOF'}])
    dump(rp, record)
    dump(paths[0].parent / 'pipeline-steps/run/pipeline-manifest.json',
         {'items': [{'source': str(paths[0]), 'final': str(paths[3]), 'steps': []}]})
    return paths, rp, record


def test_returned_background_audit_failure_is_recoverable_without_repainting(widget, case, monkeypatch):
    paths, rp, record = audit_failure(case)
    widget._refine_record_path = str(rp)  # Explicit load used to skip stage-list population.
    widget._show_results([str(paths[3])])
    assert widget.refine_stage_combo.currentData() == 'background'
    assert widget.refine_continue_btn.isEnabled()
    assert '接口失败' in widget.refine_status_view.toPlainText()
    calls = []
    monkeypatch.setattr(widget, 'retry_refine', lambda **kw: calls.append(kw))
    widget.refine_continue_btn.click()
    assert calls == [{'audit_only': True}]
    plan = state.resume_plan(rp, 'background', audit_only=True)
    assert plan['audit_only'] and plan['remaining'] == ('background', 'clothing', 'whole')
    record['stages'][1].update(status='returned', report={'verdict': 'fail'})
    dump(rp, record)
    with pytest.raises(ValueError, match='没有可复审'):
        state.resume_plan(rp, 'background', audit_only=True)


def test_failed_audit_finishing_live_record_updates_default_stage(widget, case):
    paths, rp, record = audit_failure(case)
    widget._show_results([str(paths[3])])
    record['status'] = 'running'
    dump(rp, record)
    widget.refresh_refine_status()
    widget.refine_stage_combo.setCurrentIndex(widget.refine_stage_combo.findData('targeted'))
    record['status'] = 'fallback'
    dump(rp, record)
    widget.refresh_refine_status()
    assert widget.refine_stage_combo.currentData() == 'background'


def test_explicit_product_load_keeps_product_and_refine_record_on_timer(widget, case, monkeypatch):
    paths, rp, _ = audit_failure(case)
    monkeypatch.setattr(tab.QFileDialog, 'getOpenFileName', lambda *a, **kw: (str(paths[3]), ''))
    widget.load_refine_product()
    assert widget.result_list.currentItem().data(tab.Qt.ItemDataRole.UserRole) == str(paths[3])
    widget.refresh_refine_status()
    assert widget._refine_record_path == str(rp)
    assert widget.refine_stage_combo.currentData() == 'background'
    assert widget.refine_continue_btn.isEnabled()


def test_result_click_and_preview_click_open_zoomable_image_with_exact_path(widget, case):
    from PyQt6.QtTest import QTest
    from PyQt6.QtCore import Qt
    path = str(case[0][3])
    widget._show_results([path])
    widget.result_list.itemClicked.emit(widget.result_list.currentItem())
    viewer = widget._image_dialog
    assert viewer.isVisible() and viewer.path_edit.text() == os.path.abspath(path)
    viewer.zoom(2)
    assert viewer.image_label.pixmap().width() == viewer.original.width() * 2
    viewer.close()
    widget.show_preview(str(case[0][2]))
    QTest.mouseClick(widget.preview_label, Qt.MouseButton.LeftButton)
    assert viewer.isVisible() and viewer.path_edit.text() == os.path.abspath(case[0][2])
    viewer.close()


def test_live_audit_updates_main_task_status(widget, case):
    paths, rp, record = case
    record.update(status='running')
    record['stages'][-1]['status'] = 'requested'
    dump(rp, record)
    widget._show_results([str(paths[3])])
    class Worker:
        def isRunning(self): return True
    widget._worker = Worker()
    widget._refine_live_path = str(paths[0])
    widget.refresh_refine_status(record_path=str(rp))
    assert '背景审计' in widget.status_label.text()
    assert '完成' not in widget.status_label.text()
    widget._worker = None


def test_refine_display_prefers_chinese_details_without_changing_verdict(case):
    _, _, record = case
    report = record['stages'][-1]['report']
    report['display_zh'] = {'summary': '背景未通过', 'details': '窗框仍有碎纹。'}
    text = state.status_text(record)
    assert '窗框仍有碎纹。' in text
    assert 'Window fragments remain.' not in text
    assert report['verdict'] == 'fail'
