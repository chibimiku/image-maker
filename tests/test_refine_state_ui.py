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
