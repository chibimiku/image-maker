import json
from pathlib import Path

import pytest
from PIL import Image

from utils import gpt_image_refine as refine
from modules.others import api_backend as api


def passed(**changes):
    report = dict.fromkeys(refine.AUDIT_FIELDS, True)
    report.update(verdict='pass', confidence=0.95, unacceptable_changes=[])
    report.update(changes)
    return report


@pytest.mark.parametrize('field', refine.AUDIT_FIELDS)
def test_each_fidelity_dimension_is_required(field):
    assert not refine.audit_passes(passed(**{field: False}))
    report = passed()
    del report[field]
    assert not refine.audit_passes(report)


@pytest.mark.parametrize('confidence', [True, float('nan'), 0.79, 1.1])
def test_invalid_confidence_cannot_pass(confidence):
    assert not refine.audit_passes(passed(confidence=confidence))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    paths = [str(tmp_path / (name + '.png')) for name in ['source', 'person', 'background']]
    for path in paths:
        Image.new('RGB', (120, 80), 'white').save(path)
    diag = dict(has_people=True, anatomy_ok=True, confidence=0.95,
                identity_anchors='dark olive green iris, low brightness, metal armour',
                person_tasks='preserve the large broken chest opening; join existing narrow trim',
                background_tasks='merge stone surface micro noise into existing slabs')
    calls = []
    texts = []
    reports = [diag, passed(), passed()]
    def text(*args):
        texts.append(args)
        value = reports.pop(0)
        if isinstance(value, Exception):
            raise value
        return dict(report=value, raw=json.dumps(value), model='test')
    def image(**kwargs):
        calls.append(kwargs)
        return [paths[len(calls)]]
    monkeypatch.setattr(refine, 'text_request', text)
    monkeypatch.setattr(api, 'generate_image_repaint', image)
    options = dict(model='test-image', api_type='aigc2d', resolution='2K',
                   aspect_ratio='3:2', save_sub_dir=str(tmp_path))
    return paths, diag, calls, texts, reports, options, tmp_path


def run(setup, **kwargs):
    paths, _, _, _, _, options, _ = setup
    kwargs.setdefault('log_callback', print)
    return refine.refine_image(paths[0], image_options=options, **kwargs)


def record(setup):
    return json.loads(next(setup[-1].glob('*-refine.json')).read_text(encoding='utf-8'))


def test_person_then_background_order_and_single_source_contract(setup):
    paths, _, calls, texts, _, _, _ = setup
    assert run(setup) == paths[2]
    assert [c['source_paths'] for c in calls] == [[paths[0]], [paths[1]]]
    assert len(texts) == 3
    assert texts[1][3] == [paths[0], paths[0], paths[1]]
    assert texts[2][3] == paths
    for call in calls:
        assert call['extra_reference_paths'] == [] and call['repeat'] == 1
        assert call['request_timeout'] == 360 and call['retry_limit'] == 0
        assert call['refine'] is False and call['content_guard'] is False
        assert call['use_detail_suffix'] is False and call['prompt_suffix'] == ''
    value = record(setup)
    assert value['selected'] == paths[2]
    assert len(value['stages']) == 5
    assert all(s.get('raw') for s in value['stages'] if s['kind'] == 'text')


def test_source_extra_hand_stops_before_paid_image(setup):
    paths, diag, calls, _, _, _, _ = setup
    diag['anatomy_ok'] = False
    assert run(setup) == paths[0]
    assert calls == []
    assert record(setup)['status'] == 'fallback'


def test_person_fail_never_sends_background(setup):
    paths, _, calls, texts, reports, _, _ = setup
    reports[1] = passed(anatomy_ok=False)
    assert run(setup) == paths[0]
    assert len(calls) == 1 and len(texts) == 2
    assert record(setup)['stages'][1]['status'] == 'rejected'
    assert Path(paths[1]).exists()


@pytest.mark.parametrize('field', ['iris_hue_preserved', 'iris_value_preserved', 'iris_chroma_preserved'])
def test_background_colour_drift_returns_accepted_person(setup, field):
    paths, _, _, _, reports, _, _ = setup
    reports[2] = passed(**{field: False})
    assert run(setup) == paths[1]
    assert record(setup)['stages'][3]['status'] == 'rejected'
    assert Path(paths[2]).exists()


@pytest.mark.parametrize('stage', [0, 1, 2])
def test_text_timeout_does_not_retry_or_adopt_unaudited_image(setup, stage):
    paths, _, calls, texts, reports, _, _ = setup
    reports[stage] = TimeoutError('read timeout')
    assert run(setup) == paths[1 if stage == 2 else 0]
    assert len(calls) == stage and len(texts) == stage + 1
    assert record(setup)['status'] == 'fallback'


def test_wrong_aspect_keeps_candidate_and_stops_audit(setup):
    paths, _, calls, texts, _, _, _ = setup
    Image.new('RGB', (80, 80), 'white').save(paths[1])
    assert run(setup) == paths[0]
    assert len(calls) == 1 and len(texts) == 1
    assert Path(paths[1]).exists()


def test_transient_record_lock_never_repeats_paid_requests(setup, monkeypatch):
    replace = refine.os.replace
    attempts = []
    def locked_once(a, b):
        attempts.append((a, b))
        if len(attempts) == 1:
            raise PermissionError('temporary indexer lock')
        return replace(a, b)
    monkeypatch.setattr(refine.os, 'replace', locked_once)
    assert run(setup) == setup[0][2]
    assert len(setup[2]) == 2 and len(setup[3]) == 3
    assert record(setup)['selected'] == setup[0][2]


def test_cancel_after_diagnosis_never_sends_image(setup):
    paths, _, calls, _, _, _, _ = setup
    assert run(setup, cancel_check=lambda: True) == paths[0]
    assert calls == []


def test_api_refine_follows_each_source_aspect(tmp_path, monkeypatch):
    paths = []
    for name, size in [('wide', (120, 80)), ('tall', (80, 120))]:
        p = str(tmp_path / (name + '.png'))
        Image.new('RGB', size).save(p)
        paths.append(p)
    calls = []
    monkeypatch.setattr(refine, 'refine_image', lambda source, **kw: calls.append((source, kw)) or source)
    assert api.generate_image_repaint(paths, refine=True, aspect_ratio='auto', repeat=9,
                                      extra_reference_paths=paths) == paths
    assert [kw['image_options']['aspect_ratio'] for _, kw in calls] == ['3:2', '2:3']


def test_pipeline_refine_does_not_repeat_wardrobe(tmp_path, monkeypatch):
    from utils import post_process as pp
    source = str(tmp_path / 'source.png')
    Image.new('RGB', (120, 80)).save(source)
    calls = []
    monkeypatch.setattr(api, 'generate_image_repaint', lambda **kw: calls.append(kw) or [source])
    steps = dict(repaint=dict(enabled=True, refine=True, content_guard=True),
                 wardrobe=dict(enabled=True))
    assert pp.run_pipeline([source], steps, final_dir=str(tmp_path / 'final'),
                           work_dir=str(tmp_path / 'work'))
    assert len(calls) == 1 and calls[0]['refine'] is True
    assert steps['wardrobe']['enabled'] is True  # caller's saved recipe stays intact


def test_cost_counts_only_two_images_three_texts():
    from utils.cost_estimate import estimate_pipeline
    est = estimate_pipeline(1000, include_repaint=True, include_content_guard=True,
                            include_refine=True, include_wardrobe=True, include_local=False,
                            pricing={'models': {}})
    by_key = {s['key']: s for s in est['steps']}
    assert by_key['refine']['call_count'] == 2 and by_key['refine_audit']['call_count'] == 3
    assert not {'content_cleanup', 'content_audit', 'wardrobe', 'repaint'} & by_key.keys()
