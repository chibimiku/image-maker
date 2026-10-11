import json
from pathlib import Path

import pytest
from PIL import Image

from utils import gpt_image_refine as refine
from modules.others import api_backend as api
BACKEND_REPAINT = api.generate_image_repaint


def passed(**changes):
    report = dict.fromkeys(refine.AUDIT_FIELDS, True)
    report.update(verdict='pass', confidence=0.95, unacceptable_changes=[],
                  background_repair_completed=True,
                  background_repair_evidence=['Named contours are connected and noise reduced.'],
                  clothing_repair_completed=True, clothing_repair_evidence=['Broken waist patches became coherent motifs.'],
                  hair_state='clear', hair_repair_needed=False,
                  hair_evidence=['Visible locks have continuous scalp-to-tip connections.'])
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


@pytest.mark.parametrize('change', [
    {'background_repair_completed': False}, {'background_repair_completed': None},
    {'background_repair_completed': 1}, {'background_repair_evidence': []},
    {'background_repair_evidence': ['']}, {'background_repair_evidence': 'clear'},
])
def test_background_preservation_alone_cannot_adopt_unfinished_repair(setup, change):
    paths, _, calls, texts, reports, _, _ = setup
    reports[2] = passed(**change)
    assert run(setup) == paths[1]
    assert len(calls) == 2 and len(texts) == 3
    assert record(setup)['stages'][3]['status'] == 'rejected'


def test_missing_background_effectiveness_cannot_pass():
    report = passed()
    del report['background_repair_completed']
    assert not refine.background_repair_passes(report)


def test_accepted_minor_variations_do_not_disable_fidelity_requirements():
    report = passed(accepted_variations=['Slight warmth shift within naturally shaded brown irises.'])
    assert refine.audit_passes(report)
    report['iris_hue_preserved'] = False
    assert not refine.audit_passes(report)
    report['iris_hue_preserved'] = True
    report['unacceptable_changes'] = ['Green iris replaced by blue.']
    assert not refine.audit_passes(report)


def test_audit_receives_measured_geometry_for_each_supplied_image(setup):
    paths, _, _, texts, _, _, _ = setup
    run(setup)
    payload = json.loads(texts[1][2])
    assert payload['stage'] == 'person'
    assert texts[1][3] == [paths[0], paths[0], paths[1]]
    assert payload['measured_image_geometry'] == [
        {'image': i, 'width': 120, 'height': 80, 'width_over_height': 1.5}
        for i in (1, 2, 3)
    ]


def test_audit_geometry_uses_exif_display_orientation(tmp_path):
    path = tmp_path / 'rotated.jpg'
    image = Image.new('RGB', (120, 80))
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    assert refine.audit_geometry([path]) == [
        {'image': 1, 'width': 80, 'height': 120, 'width_over_height': 0.666667}
    ]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    # Existing fallback tests exercise exhausted calls; dedicated retry tests opt in.
    monkeypatch.setattr(refine, 'AUDIT_RETRY_LIMIT', 0)
    paths = [str(tmp_path / (name + '.png')) for name in ['source', 'person', 'background', 'ornaments', 'lighting', 'hair']]
    for path in paths:
        Image.new('RGB', (120, 80), 'white').save(path)
    diag = dict(has_people=True, anatomy_ok=True, confidence=0.95,
                identity_anchors='dark olive green iris, low brightness, metal armour',
                person_tasks='preserve the large broken chest opening; join existing narrow trim',
                background_tasks='merge stone surface micro noise into existing slabs')
    calls = []
    texts = []
    reports = [diag, passed(), passed(), passed(), passed()]
    def text(*args):
        texts.append(args)
        value = reports.pop(0)
        if isinstance(value, Exception):
            raise value
        return dict(report=value, raw=json.dumps(value), model='test')
    def image(**kwargs):
        calls.append(kwargs)
        stage = kwargs['file_prefix'].rsplit('-', 1)[-1]
        return [paths[{'clothing': 3, 'targeted': 2, 'whole': 1, 'person': 1, 'background': 2, 'ornaments': 3, 'lighting': 4, 'hair': 5}[stage]]]
    monkeypatch.setattr(refine, 'text_request', text)
    monkeypatch.setattr(api, 'generate_image_repaint', image)
    from utils import gpt_image_optimize as optimize
    monkeypatch.setattr(optimize, 'load_config', lambda: dict(
        refine_workflow='targeted_whole', refine_background_model='test-pro', refine_clothing_model='test-flash'))
    options = dict(model='test-image', api_type='aigc2d', resolution='2K',
                   aspect_ratio='3:2', save_sub_dir=str(tmp_path))
    return paths, diag, calls, texts, reports, options, tmp_path


def run(setup, **kwargs):
    paths, _, _, _, _, options, _ = setup
    kwargs.setdefault('log_callback', print)
    # Preserve coverage of historical v5 staged execution and its checkpoint contract.
    return refine._refine_legacy_image(paths[0], image_options=options, **kwargs)


def record(setup):
    return json.loads(next(setup[-1].glob('*-refine.json')).read_text(encoding='utf-8'))


def run_whole(setup, report=None, **kwargs):
    paths, diag, _, _, reports, options, _ = setup
    reports[:] = [diag, passed() if report is None else report]
    return refine.refine_image(paths[0], image_options=options, **kwargs)


def test_whole_pipeline_uses_one_combined_edit_and_two_text_calls(setup):
    paths, _, calls, texts, _, _, _ = setup
    assert run_whole(setup) == paths[1]
    r = record(setup)
    assert r['workflow'] == 'targeted_whole' and r['version'] == 7
    assert r['image_limit'] == 2 and r['text_limit'] == 3
    assert [s['key'] for s in r['stages']] == ['diagnosis', 'whole', 'whole-audit']
    assert r['status'] == 'finished' and r['stages'][1]['status'] == 'accepted'
    assert len(calls) == 1 and len(texts) == 2
    request = calls[0]
    assert request['source_paths'] == [paths[0]] and request['extra_reference_paths'] == []
    assert request['prompt'] == refine.read_prompt_file('gpt-image-optimize/refine-whole.md').strip()
    assert not request['use_detail_suffix'] and request['prompt_suffix'] == ''
    assert not request['refine'] and request['retry_limit'] == 0 and request['repeat'] == 1
    payload = json.loads(texts[1][2])
    assert payload['stage'] == 'whole'
    assert [m['image'] for m in payload['measured_image_geometry']] == [1, 2, 3]


@pytest.mark.parametrize('report', [
    passed(verdict='fail', iris_hue_preserved=False, unacceptable_changes=['Green eyes changed to blue.']),
    passed(hair_state='fragmented', hair_repair_needed=True),
    passed(hair_state='uncertain', hair_repair_needed=None),
    TimeoutError('whole audit timeout'),
])
def test_whole_failure_preserves_input_and_never_dispatches_followup_repairs(setup, report):
    paths, _, calls, texts, _, _, _ = setup
    assert run_whole(setup, report) == paths[0]
    assert len(calls) == 1 and len(texts) == 2
    r = record(setup)
    assert r['status'] in ('stopped', 'fallback') and r['selected'] == paths[0]
    assert r['stages'][1]['outputs'] == [paths[1]]
    if isinstance(report, Exception):
        assert r['stages'][2]['status'] == 'failed'


def test_whole_diagnosis_uncertainty_never_sends_an_image(setup):
    setup[1]['anatomy_ok'] = False
    assert run_whole(setup) == setup[0][0]
    assert not setup[2] and len(setup[3]) == 1


def test_whole_no_output_never_retries_or_sends_audit(setup, monkeypatch):
    calls = []
    monkeypatch.setattr(api, 'generate_image_repaint', lambda **kw: calls.append(kw) or [])
    assert run_whole(setup) == setup[0][0]
    assert len(calls) == 1 and len(setup[3]) == 1
    assert record(setup)['stages'][1]['status'] == 'no_output'


def test_whole_retry_reuses_diagnosis_and_original_stage_input(setup):
    paths, _, calls, texts, reports, _, _ = setup
    run_whole(setup, passed(verdict='fail'))
    parent = str(next(setup[-1].glob('*-refine.json')))
    calls.clear(); texts.clear(); reports[:] = [passed()]
    assert refine.refine_image('', image_options={}, resume_record=parent, start_stage='whole') == paths[1]
    assert len(calls) == 1 and len(texts) == 1
    assert calls[0]['source_paths'] == [paths[0]]
    assert json.loads(Path(parent).read_text(encoding='utf-8'))['selected'] == paths[0]


def test_whole_manual_adoption_sends_no_requests_or_old_followup_stages(setup):
    paths, _, calls, texts, _, _, _ = setup
    run_whole(setup, passed(verdict='fail'))
    parent = str(next(setup[-1].glob('*-refine.json')))
    calls.clear(); texts.clear()
    assert refine.refine_image('', image_options={}, resume_record=parent,
                              start_stage='whole', force_candidate=True) == paths[1]
    assert not calls and not texts
    child = next(json.loads(p.read_text(encoding='utf-8')) for p in setup[-1].glob('*-refine.json') if str(p) != parent)
    assert child['manual_override']['old_status'] == 'rejected'
    assert child['status'] == 'finished' and child['completed_before'] == ['whole']
    assert json.loads(Path(parent).read_text(encoding='utf-8'))['stages'][1]['status'] == 'rejected'


def add_repair_targets(diag):
    diag.update(targeted_repair_needed=True, repair_targets=[dict(
        region='stone parapet', defect='broken visible seam', evidence='Two matching ends leave a short gap.',
        repair='Join the short gap along the existing route.', preserve='Keep stone texture and original width.')])


def test_targeted_branch_pass_feeds_fixed_whole_step_and_keeps_source_anchor(setup):
    paths, diag, calls, texts, reports, options, _ = setup
    add_repair_targets(diag)
    reports[:] = [diag, passed(targeted_repair_completed=True,
                              targeted_repair_evidence=['Short gap joined.']), passed()]
    assert refine.refine_image(paths[0], image_options=options) == paths[1]
    assert [c['source_paths'] for c in calls] == [[paths[0]], [paths[2]]]
    assert 'stone parapet' in calls[0]['prompt'] and 'Short gap' not in calls[1]['prompt']
    assert calls[1]['prompt'] == refine.read_prompt_file('gpt-image-optimize/refine-whole.md').strip()
    assert texts[2][3] == [paths[0], paths[2], paths[1]]
    r = record(setup)
    assert [s['key'] for s in r['stages']] == ['diagnosis', 'targeted', 'targeted-audit', 'whole', 'whole-audit']
    assert r['status'] == 'finished'


@pytest.mark.parametrize('audit', [
    passed(verdict='fail'), passed(targeted_repair_completed=False), passed(),
    TimeoutError('targeted audit timeout'),
])
def test_targeted_rejection_or_audit_failure_keeps_source_then_runs_fixed_whole(setup, audit):
    paths, diag, calls, texts, reports, options, _ = setup
    add_repair_targets(diag)
    reports[:] = [diag, audit, passed()]
    assert refine.refine_image(paths[0], image_options=options) == paths[1]
    assert [c['source_paths'] for c in calls] == [[paths[0]], [paths[0]]]
    r = record(setup)
    assert r['stages'][1]['status'] == 'rejected' and r['status'] == 'finished'


def test_targeted_no_image_still_runs_whole_without_retry(setup, monkeypatch):
    paths, diag, calls, texts, reports, options, _ = setup
    add_repair_targets(diag)
    reports[:] = [diag, passed()]
    def image(**kw):
        calls.append(kw)
        return [] if kw['file_prefix'].endswith('-targeted') else [paths[1]]
    monkeypatch.setattr(api, 'generate_image_repaint', image)
    assert refine.refine_image(paths[0], image_options=options) == paths[1]
    assert len(calls) == 2 and len(texts) == 2
    assert calls[1]['source_paths'] == [paths[0]]


@pytest.mark.parametrize('changes', [
    {'targeted_repair_needed': False, 'repair_targets': []},
    {'targeted_repair_needed': None, 'repair_targets': []},
    {'targeted_repair_needed': True, 'repair_targets': [{'region': 'hair'}]},
])
def test_no_evidenced_target_never_sends_targeted_repair(setup, changes):
    setup[1].update(changes)
    assert run_whole(setup) == setup[0][1]
    assert len(setup[2]) == 1 and record(setup)['targeted_check']['decision'] == 'skip'


def test_whole_retry_after_targeted_success_does_not_repeat_targeted_step(setup):
    paths, diag, calls, texts, reports, options, _ = setup
    add_repair_targets(diag)
    reports[:] = [diag, passed(targeted_repair_completed=True,
                              targeted_repair_evidence=['Gap joined.']), passed(verdict='fail')]
    refine.refine_image(paths[0], image_options=options)
    parent = str(next(setup[-1].glob('*-refine.json')))
    calls.clear(); texts.clear(); reports[:] = [passed()]
    assert refine.refine_image('', image_options={}, resume_record=parent, start_stage='whole') == paths[1]
    assert len(calls) == 1 and len(texts) == 1
    assert calls[0]['source_paths'] == [paths[2]]


def test_v6_whole_record_retry_does_not_acquire_new_targeted_stage(setup):
    paths, diag, calls, texts, reports, _, _ = setup
    run_whole(setup, passed(verdict='fail'))
    parent = next(setup[-1].glob('*-refine.json'))
    r = json.loads(parent.read_text(encoding='utf-8'))
    add_repair_targets(r['diagnosis'])
    r.update(version=6, workflow='whole')
    parent.write_text(json.dumps(r), encoding='utf-8')
    calls.clear(); texts.clear(); reports[:] = [passed()]
    assert refine.refine_image('', image_options={}, resume_record=str(parent)) == paths[1]
    assert len(calls) == 1 and len(texts) == 1 and calls[0]['file_prefix'].endswith('-whole')


def test_person_background_ornaments_lighting_order_and_single_source_contract(setup):
    paths, _, calls, texts, _, _, _ = setup
    assert run(setup) == paths[4]
    assert [c['source_paths'] for c in calls] == [[paths[0]], [paths[1]], [paths[2]], [paths[3]]]
    assert len(texts) == 5
    assert texts[1][3] == [paths[0], paths[0], paths[1]]
    assert texts[2][3] == paths[:3]
    assert texts[3][3] == [paths[0], paths[2], paths[3]]
    assert texts[4][3] == [paths[0], paths[3], paths[4]]
    for call in calls:
        assert call['extra_reference_paths'] == [] and call['repeat'] == 1
        assert call['request_timeout'] == 360 and call['retry_limit'] == 0
        assert call['refine'] is False and call['content_guard'] is False
        assert call['use_detail_suffix'] is False and call['prompt_suffix'] == ''
    value = record(setup)
    assert value['selected'] == paths[4]
    assert len(value['stages']) == 9
    assert all(s.get('raw') for s in value['stages'] if s['kind'] == 'text')


def test_source_extra_hand_stops_before_paid_image(setup):
    paths, diag, calls, _, _, _, _ = setup
    diag['anatomy_ok'] = False
    assert run(setup) == paths[0]
    assert calls == []
    assert record(setup)['status'] == 'fallback'


def test_ornaments_failure_returns_accepted_background(setup):
    paths, _, calls, texts, reports, _, _ = setup
    reports[3] = passed(stage_scope_preserved=False)
    assert run(setup) == paths[2]
    assert len(calls) == 3 and len(texts) == 4
    assert record(setup)['stages'][5]['status'] == 'rejected'
    assert Path(paths[3]).exists()


def test_lighting_failure_returns_accepted_ornaments(setup):
    paths, _, calls, texts, reports, _, _ = setup
    reports[4] = passed(stage_scope_preserved=False)
    assert run(setup) == paths[3]
    assert len(calls) == 4 and len(texts) == 5
    assert record(setup)['stages'][7]['status'] == 'rejected'
    assert Path(paths[4]).exists()


def test_lighting_runs_when_material_tasks_are_empty(setup):
    paths, diag, calls, texts, _, _, _ = setup
    diag.update(has_people=False, person_tasks='', background_tasks='')
    assert run(setup) == paths[4]
    assert len(calls) == 1 and len(texts) == 2
    assert calls[0]['source_paths'] == [paths[0]]
    assert record(setup)['stages'][1]['key'] == 'lighting'
    assert len(calls[0]['prompt'].split()) <= 100
    assert 'identity_anchors' not in calls[0]['prompt']


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


@pytest.mark.parametrize('stage', [0, 1, 2, 3, 4])
def test_text_timeout_does_not_retry_or_adopt_unaudited_image(setup, stage):
    paths, _, calls, texts, reports, _, _ = setup
    reports[stage] = TimeoutError('read timeout')
    assert run(setup) == paths[max(0, stage - 1)]
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
    assert run(setup) == setup[0][4]
    assert len(setup[2]) == 4 and len(setup[3]) == 5
    assert record(setup)['selected'] == setup[0][4]


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


def test_cost_counts_conditional_targeted_whole_and_three_text_calls():
    from utils.cost_estimate import estimate_pipeline
    est = estimate_pipeline(1000, include_repaint=True, include_content_guard=True,
                            include_refine=True, include_wardrobe=True, include_local=False,
                            pricing={'models': {}})
    by_key = {s['key']: s for s in est['steps']}
    assert by_key['refine']['call_count'] == 2 and by_key['refine_audit']['call_count'] == refine.MAX_TEXT_ATTEMPTS
    assert by_key['refine_background']['call_count'] == by_key['refine_clothing']['call_count'] == 1
    assert not {'content_cleanup', 'content_audit', 'wardrobe', 'repaint'} & by_key.keys()


def garment(material='leather', **changes):
    return dict(region='jacket', material=material, evidence='visible grain and broad highlights',
                confidence=0.9, **changes)


def hybrid_setup(setup, monkeypatch):
    from utils import gpt_image_optimize as optimize
    monkeypatch.setattr(optimize, 'load_config', lambda: dict(
        refine_workflow='hybrid', refine_background_model='chosen-pro', refine_clothing_model='chosen-flash'))
    setup[4][:] = [setup[1], passed(), passed(), passed()]


@pytest.mark.parametrize('audit_result', [passed(), passed(verdict='fail'), TimeoutError('again')])
def test_resume_failed_background_audit_reuses_image_and_request_before_followup(setup, monkeypatch, audit_result):
    hybrid_setup(setup, monkeypatch)
    paths, diag, calls, texts, reports, options, folder = setup
    reports[:] = [diag, TimeoutError('TLS EOF')]
    assert refine.refine_image(paths[0], image_options=options) == paths[0]
    parent = next(folder.glob('*-refine.json'))
    original = json.loads(parent.read_text(encoding='utf-8'))
    original['stages'][-1].update(system='Frozen audit system', user='Frozen audit task')
    parent.write_text(json.dumps(original), encoding='utf-8')
    calls.clear(); texts.clear()
    reports[:] = [audit_result, passed(), passed()]
    result = refine.refine_image('', image_options={}, resume_record=str(parent),
                                 start_stage='background', resume_audit_only=True)
    child_path = next(p for p in folder.glob('*-refine.json') if p != parent)
    child = json.loads(child_path.read_text(encoding='utf-8'))
    assert json.loads(parent.read_text(encoding='utf-8')) == original
    assert child['stages'][0]['reused_from_record'] == str(parent)
    assert child['stages'][0]['outputs'] == [paths[2]]
    assert texts[0][1].startswith(original['stages'][-1]['system'] + '\n\nBILINGUAL DISPLAY OUTPUT:')
    assert texts[0][1].count('BILINGUAL DISPLAY OUTPUT:') == 1
    assert texts[0][2:4] == (original['stages'][-1]['user'], original['stages'][-1]['images'])
    assert child['manual_override'] is None
    if isinstance(audit_result, dict) and audit_result['verdict'] == 'pass':
        assert child['status'] == 'finished' and result == paths[1]
        assert [c['source_paths'] for c in calls] == [[paths[2]], [paths[3]]]
        assert len(texts) == 3
    else:
        assert not calls and result == paths[0]
        assert child['status'] in ('fallback', 'stopped')


def test_old_hybrid_not_a_hair_decision_answer_requires_new_audit_without_new_background(setup, monkeypatch):
    hybrid_setup(setup, monkeypatch)
    paths, diag, calls, texts, reports, options, folder = setup
    reports[:] = [diag, passed(hair_state='uncertain', hair_repair_needed=None,
                              hair_evidence=['Not a hair-repair decision stage; retained hairstyle.'])]
    assert refine.refine_image(paths[0], image_options=options) == paths[0]
    parent = next(folder.glob('*-refine.json'))
    old = parent.read_bytes()
    calls.clear(); texts.clear(); reports[:] = [passed(), passed(), passed()]
    assert refine.refine_image('', image_options={}, resume_record=str(parent),
                              start_stage='background', resume_audit_only=True) == paths[1]
    child = json.loads(next(p for p in folder.glob('*-refine.json') if p != parent).read_text(encoding='utf-8'))
    assert len(calls) == 2 and len(texts) == 3
    assert child['stages'][1]['scope_clarification']
    assert 'Do not return uncertain/null merely because hair' in texts[0][1]
    assert child['status'] == 'finished' and parent.read_bytes() == old


def test_explicit_clothing_redraw_uses_prior_rejection_without_attaching_rejected_image(setup, monkeypatch):
    hybrid_setup(setup, monkeypatch)
    paths, diag, calls, texts, reports, options, folder = setup
    reports[:] = [diag, passed(), passed(verdict='fail', stage_scope_preserved=False,
                                       unacceptable_changes=['Unsupported chest cross-lacing.'])]
    assert refine.refine_image(paths[0], image_options=options) == paths[2]
    parent = next(folder.glob('*-refine.json'))
    old = parent.read_bytes()
    calls.clear(); texts.clear(); reports[:] = [passed(), passed()]
    assert refine.refine_image('', image_options={}, resume_record=str(parent),
                              start_stage='clothing') == paths[1]
    assert len(calls) == 2 and len(texts) == 2
    assert calls[0]['source_paths'] == [paths[2]]
    assert 'Unsupported chest cross-lacing.' in calls[0]['prompt']
    assert 'explicit retry from the trusted input' in calls[0]['prompt']
    assert 'Unsupported chest cross-lacing.' not in calls[1]['prompt']
    assert parent.read_bytes() == old


@pytest.mark.parametrize('failures', [1, 3, 4])
def test_audit_transport_retry_preserves_candidate_and_continues_only_after_return(setup, monkeypatch, failures):
    hybrid_setup(setup, monkeypatch)
    monkeypatch.setattr(refine, 'AUDIT_RETRY_LIMIT', 3)
    monkeypatch.setattr(refine.time, 'sleep', lambda _: None)
    paths, diag, calls, texts, reports, _, _ = setup
    reports[:] = [diag] + [TimeoutError('read timeout')] * failures + [passed(), passed(), passed()]
    result = refine.refine_image(paths[0], image_options=setup[5])
    r = record(setup)
    entry = r['stages'][2]  # background audit after source diagnosis and one image
    assert len(entry['attempts']) == min(failures + 1, 4)
    assert all(texts[i][1:4] == texts[1][1:4] for i in range(1, min(failures + 2, 5)))
    if failures < 4:
        assert result == paths[1] and r['status'] == 'finished'
        assert len(calls) == 3
        assert entry['status'] == 'returned' and 'error' not in entry
    else:
        assert result == paths[0] and r['status'] == 'fallback'
        assert len(calls) == 1 and len(texts) == 5
        assert entry['status'] == 'failed'


@pytest.mark.parametrize('error', [ValueError('invalid JSON'), RuntimeError('budget exhausted')])
def test_non_transport_errors_do_not_retry(setup, monkeypatch, error):
    hybrid_setup(setup, monkeypatch)
    monkeypatch.setattr(refine, 'AUDIT_RETRY_LIMIT', 3)
    setup[4][:] = [setup[1], error]
    assert refine.refine_image(setup[0][0], image_options=setup[5]) == setup[0][0]
    assert len(setup[3]) == 2 and len(setup[2]) == 1


def test_returned_fail_is_not_retried_even_after_transport_recovery(setup, monkeypatch):
    hybrid_setup(setup, monkeypatch)
    monkeypatch.setattr(refine, 'AUDIT_RETRY_LIMIT', 3)
    monkeypatch.setattr(refine.time, 'sleep', lambda _: None)
    setup[4][:] = [setup[1], TimeoutError('timeout'), passed(verdict='fail', anatomy_ok=False)]
    assert refine.refine_image(setup[0][0], image_options=setup[5]) == setup[0][0]
    assert len(setup[3]) == 3 and len(setup[2]) == 1
    assert record(setup)['stages'][2]['report']['verdict'] == 'fail'


@pytest.mark.parametrize('status,expected', [(401, False), (400, False), (429, True), (503, True)])
def test_audit_http_retry_classification(status, expected):
    import requests
    response = requests.Response(); response.status_code = status
    assert refine._retryable_audit_error(requests.HTTPError(response=response)) is expected


def test_cancellation_during_audit_backoff_prevents_next_request(monkeypatch):
    calls = []; saved = []
    def fail(*args):
        calls.append(args)
        raise TimeoutError('timeout')
    monkeypatch.setattr(refine, 'text_request', fail)
    entry = dict(key='whole-audit', system='system', user='task', images=['existing.jpg'])
    with pytest.raises(RuntimeError, match='cancelled'):
        refine._request_recorded_text(entry, 'whole-audit', lambda: saved.append(entry['status']),
                                      lambda _: None, lambda: bool(calls))
    assert len(calls) == 1 and entry['status'] == 'failed'


def test_backend_dispatches_explicit_stage_models_to_core(setup, monkeypatch):
    captured = []
    monkeypatch.setattr(refine, 'refine_image', lambda source, **kw: captured.append(kw) or source)
    result = BACKEND_REPAINT([setup[0][0]], refine=True, model='manual-model',
                            refine_background_model='explicit-pro', refine_clothing_model='explicit-flash')
    assert result == [setup[0][0]]
    assert captured[0]['image_options']['refine_background_model'] == 'explicit-pro'
    assert captured[0]['image_options']['refine_clothing_model'] == 'explicit-flash'


def test_hybrid_no_image_background_preserves_head_without_clothing_or_retry(setup, monkeypatch):
    hybrid_setup(setup, monkeypatch)
    original = api.generate_image_repaint
    def image(**kw):
        if kw['file_prefix'].endswith('-background'):
            setup[2].append(kw)
            return []
        return original(**kw)
    monkeypatch.setattr(api, 'generate_image_repaint', image)
    assert refine.refine_image(setup[0][0], image_options=setup[5]) == setup[0][0]
    assert len(setup[2]) == 1 and len(setup[3]) == 1


def test_hybrid_force_background_only_runs_clothing_and_preserves_old_rejection(setup, monkeypatch):
    hybrid_setup(setup, monkeypatch)
    setup[4][:] = [setup[1], passed(objects_preserved=False)]
    refine.refine_image(setup[0][0], image_options=setup[5])
    parent = str(next(setup[-1].glob('*-refine.json')))
    setup[2].clear(); setup[3].clear(); setup[4][:] = [passed(), passed()]
    assert refine.refine_image('', image_options={}, resume_record=parent,
                              start_stage='background', force_candidate=True) == setup[0][1]
    assert len(setup[2]) == 2 and len(setup[3]) == 2
    assert setup[2][0]['source_paths'] == [setup[0][2]] and setup[2][0]['model'] == 'chosen-flash'
    old = json.loads(Path(parent).read_text(encoding='utf-8'))
    assert old['stages'][-2]['status'] == 'rejected'


def test_hybrid_routes_models_and_each_trusted_input(setup, monkeypatch):
    hybrid_setup(setup, monkeypatch)
    paths, _, calls, texts, _, options, _ = setup
    assert refine.refine_image(paths[0], image_options=options) == paths[1]
    assert [c['model'] for c in calls] == ['chosen-pro', 'chosen-flash', 'chosen-flash']
    assert [c['source_paths'] for c in calls] == [[paths[0]], [paths[2]], [paths[3]]]
    assert [c['file_prefix'].split('-')[-1] for c in calls] == ['background', 'clothing', 'whole']
    assert len(texts) == 4
    assert texts[-1][3] == [paths[0], paths[3], paths[1]]
    assert '地面布料' in calls[1]['prompt'] and '整体模糊' in calls[0]['prompt']
    assert 'Re-render the lighting' in calls[-1]['prompt']
    assert record(setup)['workflow'] == 'hybrid' and record(setup)['version'] == 9
    assert record(setup)['image_limit'] == 4 and record(setup)['text_limit'] == 5


@pytest.mark.parametrize('audit', [passed(objects_preserved=False),
    passed(background_repair_completed=False), TimeoutError('background timeout')])
def test_hybrid_background_failure_stops_before_clothing(setup, monkeypatch, audit):
    hybrid_setup(setup, monkeypatch)
    setup[4][:] = [setup[1], audit]
    assert refine.refine_image(setup[0][0], image_options=setup[5]) == setup[0][0]
    assert len(setup[2]) == 1 and len(setup[3]) == 2


@pytest.mark.parametrize('audit', [passed(objects_preserved=False),
    passed(clothing_repair_completed=False), passed(clothing_repair_evidence=[]),
    passed(hair_state='fragmented', hair_repair_needed=True), TimeoutError('clothing timeout')])
def test_hybrid_clothing_failure_preserves_background(setup, monkeypatch, audit):
    hybrid_setup(setup, monkeypatch)
    setup[4][2] = audit
    assert refine.refine_image(setup[0][0], image_options=setup[5]) == setup[0][2]
    assert len(setup[2]) == 2 and len(setup[3]) == 3


def test_hybrid_no_people_skips_clothing_but_cleans_background(setup, monkeypatch):
    hybrid_setup(setup, monkeypatch)
    setup[1]['has_people'] = False
    assert refine.refine_image(setup[0][0], image_options=setup[5]) == setup[0][1]
    assert len(setup[2]) == 2 and len(setup[3]) == 3


def test_hybrid_resume_freezes_stage_models_and_reuses_accepted_head(setup, monkeypatch):
    hybrid_setup(setup, monkeypatch)
    setup[4][:] = [setup[1], passed(), passed(objects_preserved=False)]
    refine.refine_image(setup[0][0], image_options=setup[5])
    parent = str(next(setup[-1].glob('*-refine.json')))
    from utils import gpt_image_optimize as optimize
    monkeypatch.setattr(optimize, 'load_config', lambda: {'refine_background_model': 'new-pro',
                                                       'refine_clothing_model': 'new-flash'})
    setup[2].clear(); setup[3].clear(); setup[4][:] = [passed(), passed()]
    assert refine.refine_image('', image_options={}, resume_record=parent, start_stage='clothing') == setup[0][1]
    assert [c['model'] for c in setup[2]] == ['chosen-flash', 'chosen-flash']
    assert setup[2][0]['source_paths'] == [setup[0][2]]
    assert len(setup[3]) == 2


def test_hybrid_explicit_model_overrides_and_targeted_path(setup, monkeypatch):
    hybrid_setup(setup, monkeypatch)
    add_repair_targets(setup[1])
    setup[4][:] = [setup[1], passed(targeted_repair_completed=True,
                                  targeted_repair_evidence=['Gap joined']), passed(), passed(), passed()]
    setup[5].update(refine_background_model='custom-pro', refine_clothing_model='custom-flash')
    assert refine.refine_image(setup[0][0], image_options=setup[5]) == setup[0][1]
    assert [c['model'] for c in setup[2]] == ['custom-flash', 'custom-pro', 'custom-flash', 'custom-flash']
    assert len(setup[2]) == 4 and len(setup[3]) == 5
    assert all('refine_background_model' not in c for c in setup[2])


def test_hybrid_final_whole_rejection_preserves_clothing_without_retry(setup, monkeypatch):
    hybrid_setup(setup, monkeypatch)
    setup[4][-1] = passed(objects_preserved=False)
    assert refine.refine_image(setup[0][0], image_options=setup[5]) == setup[0][3]
    assert len(setup[2]) == 3 and len(setup[3]) == 4
    assert record(setup)['stopped_at'] == 'whole'


def test_historical_hybrid_v8_resume_keeps_original_order(setup, monkeypatch):
    hybrid_setup(setup, monkeypatch)
    refine.refine_image(setup[0][0], image_options=setup[5])
    parent = next(setup[-1].glob('*-refine.json'))
    old = json.loads(parent.read_text(encoding='utf-8'))
    old['version'] = 8
    whole = next(s for s in old['stages'] if s['key'] == 'whole')
    whole['input'] = setup[0][0]
    whole['options']['source_paths'] = [setup[0][0]]
    parent.write_text(json.dumps(old), encoding='utf-8')
    setup[2].clear(); setup[3].clear(); setup[4][:] = [passed(), passed(), passed()]
    assert refine.refine_image('', image_options={}, resume_record=str(parent), start_stage='whole') == setup[0][3]
    assert [c['file_prefix'].split('-')[-1] for c in setup[2]] == ['whole', 'background', 'clothing']


def test_material_guidance_reaches_whole_prompt_without_extra_calls(setup):
    setup[1]['garment_materials'] = [garment(), garment('silk')]
    run_whole(setup)
    suffix = record(setup)['material_guidance']['prompt']
    assert suffix.count('.') == 2
    assert 'leather with fine grain' in suffix and 'silk with smooth flowing folds' in suffix
    assert setup[2][0]['prompt'].endswith(suffix)
    assert json.loads(setup[3][-1][2])['permitted_tasks'] == setup[2][0]['prompt']
    assert len(setup[2]) == 1 and len(setup[3]) == 2


@pytest.mark.parametrize('entry', [
    {'material': 'unknown'}, {'material': ['leather']}, {'confidence': 0.7},
    {'confidence': True}, {'confidence': float('nan')}, {'evidence': ''}, {'region': None},
])
def test_uncertain_or_invalid_garment_material_does_not_add_instructions(entry):
    candidate = garment()
    candidate.update(entry)
    result = refine.garment_material_guidance(dict(confidence=0.95, has_people=True,
                                                 garment_materials=[candidate]))
    assert result == {'materials': [], 'prompt': ''}


def test_material_rules_ignore_nonpeople_and_deduplicate_classifications():
    report = dict(confidence=0.95, has_people=False, garment_materials=[garment()])
    assert not refine.garment_material_guidance(report)['prompt']
    report.update(has_people=True, garment_materials=[garment(), garment()])
    assert refine.garment_material_guidance(report)['prompt'].count('leather with') == 1


def test_material_suffix_only_added_to_final_whole_step(setup):
    add_repair_targets(setup[1])
    setup[1]['garment_materials'] = [garment('silk')]
    setup[4][:] = [setup[1], passed(targeted_repair_completed=True,
                                  targeted_repair_evidence=['gap repaired']), passed()]
    refine.refine_image(setup[0][0], image_options=setup[5])
    assert 'silk with smooth' not in setup[2][0]['prompt']
    assert 'silk with smooth' in setup[2][1]['prompt']


def test_whole_material_retry_reuses_diagnosis_without_new_detection(setup):
    setup[1]['garment_materials'] = [garment()]
    run_whole(setup, passed(verdict='fail'))
    parent = str(next(setup[-1].glob('*-refine.json')))
    original = setup[2][0]['prompt']
    setup[2].clear(); setup[3].clear(); setup[4][:] = [passed()]
    refine.refine_image('', image_options={}, resume_record=parent, start_stage='whole')
    assert setup[2][0]['prompt'] == original
    assert len(setup[2]) == 1 and len(setup[3]) == 1


def test_fragmented_hair_inserts_one_repair_before_background(setup):
    paths, _, calls, texts, reports, _, _ = setup
    reports[1] = passed(hair_state='fragmented', hair_repair_needed=True,
                        hair_evidence=['Left person has disconnected locks behind shoulder.'])
    reports.insert(2, passed())
    assert run(setup) == paths[4]
    assert [c['source_paths'] for c in calls] == [[paths[0]], [paths[1]], [paths[5]], [paths[2]], [paths[3]]]
    assert len(calls) == 5 and len(texts) == 6
    assert texts[2][3] == [paths[0], paths[1], paths[5]]
    assert texts[3][3] == [paths[0], paths[5], paths[2]]
    assert 'AND hair continuity' in calls[0]['prompt']
    assert 'clothing, background and lighting' in calls[1]['prompt']
    assert 'Specific garment' not in calls[1]['prompt']
    assert record(setup)['hair_check']['decision'] == 'repair'
    assert record(setup)['version'] == 5
    assert record(setup)['image_limit'] == 5 and record(setup)['text_limit'] == 6


@pytest.mark.parametrize('state', ['clear', 'not_visible'])
def test_coherent_or_invisible_hair_does_not_add_paid_call(setup, state):
    _, _, calls, texts, reports, _, _ = setup
    reports[1] = passed(hair_state=state)
    assert run(setup) == setup[0][4]
    assert len(calls) == 4 and len(texts) == 5
    assert record(setup)['hair_check']['decision'] == 'skip'
    assert not any(s['key'] in ('hair', 'hair-audit') for s in record(setup)['stages'])


@pytest.mark.parametrize('change', [
    {'hair_state': 'uncertain', 'hair_repair_needed': None},
    {'hair_state': 'clear', 'hair_repair_needed': True},
    {'hair_state': 'fragmented', 'hair_repair_needed': False},
    {'hair_state': None}, {'hair_evidence': []}, {'hair_evidence': ['']},
    {'hair_evidence': 'looks good'}, {'hair_repair_needed': 0},
])
def test_uncertain_hair_check_keeps_audited_person_and_stops(setup, change):
    paths, _, calls, texts, reports, _, _ = setup
    reports[1] = passed(**change)
    assert run(setup) == paths[1]
    assert len(calls) == 1 and len(texts) == 2
    assert record(setup)['status'] == 'fallback'


def test_person_fidelity_failure_never_triggers_hair_repair(setup):
    paths, _, calls, texts, reports, _, _ = setup
    reports[1] = passed(anatomy_ok=False, hair_state='fragmented', hair_repair_needed=True)
    assert run(setup) == paths[0]
    assert len(calls) == 1 and len(texts) == 2
    assert 'hair_check' not in record(setup)


@pytest.mark.parametrize('hair_audit', [
    passed(stage_scope_preserved=False),
    passed(hair_state='fragmented', hair_repair_needed=True),
    passed(hair_state='uncertain', hair_repair_needed=None),
    TimeoutError('hair audit timeout'),
])
def test_hair_failure_keeps_person_never_retries_or_sends_background(setup, hair_audit):
    paths, _, calls, texts, reports, _, _ = setup
    reports[1] = passed(hair_state='fragmented', hair_repair_needed=True)
    reports.insert(2, hair_audit)
    assert run(setup) == paths[1]
    assert len(calls) == 2 and len(texts) == 3
    assert Path(paths[5]).exists()  # rejected candidate remains reviewable


def test_hair_request_no_output_stops_without_audit_or_retry(setup, monkeypatch):
    paths, _, calls, texts, reports, _, _ = setup
    reports[1] = passed(hair_state='fragmented', hair_repair_needed=True)
    original = api.generate_image_repaint
    def no_hair(**kwargs):
        if kwargs['file_prefix'].endswith('-hair'):
            calls.append(kwargs)
            return []
        return original(**kwargs)
    monkeypatch.setattr(api, 'generate_image_repaint', no_hair)
    assert run(setup) == paths[1]
    assert len(calls) == 2 and len(texts) == 2


def test_no_people_never_checks_or_repairs_hair(setup):
    _, diag, calls, texts, reports, _, _ = setup
    diag.update(has_people=False, person_tasks='')
    reports[:] = [diag, passed(), passed()]
    assert run(setup) == setup[0][4]
    assert len(calls) == 2 and len(texts) == 3
    assert 'hair_check' not in record(setup)


def test_audit_scope_explicitly_allows_person_hair_reconnection(setup):
    run(setup)
    permitted = json.loads(setup[3][1][2])
    assert permitted['stage'] == 'person'
    assert 'Hair continuity tasks:' in permitted['permitted_tasks']
    assert 'This step repairs ONLY garment' not in permitted['permitted_tasks']


def test_retry_background_uses_original_source_and_accepted_person_without_diagnosis(setup):
    paths, _, calls, texts, reports, options, _ = setup
    reports[2] = passed(stage_scope_preserved=False)
    assert run(setup) == paths[1]
    parent = str(next(setup[-1].glob('*-refine.json')))
    calls.clear(); texts.clear(); reports[:] = [passed(), passed(), passed()]
    assert refine.refine_image('', image_options={}, resume_record=parent, start_stage='background') == paths[4]
    assert len(calls) == 3 and len(texts) == 3
    assert calls[0]['source_paths'] == [paths[1]]
    assert texts[0][3] == [paths[0], paths[1], paths[2]]
    assert all(not c['refine'] and c['retry_limit'] == 0 for c in calls)


def test_force_background_records_manual_override_and_only_runs_later_stages(setup):
    paths, _, calls, texts, reports, _, _ = setup
    reports[2] = passed(stage_scope_preserved=False)
    run(setup)
    parent = str(next(setup[-1].glob('*-refine.json')))
    calls.clear(); texts.clear(); reports[:] = [passed(), passed()]
    assert refine.refine_image('', image_options={}, resume_record=parent,
                              start_stage='background', force_candidate=True) == paths[4]
    assert len(calls) == 2 and len(texts) == 2
    assert calls[0]['source_paths'] == [paths[2]]
    child = next(json.loads(p.read_text(encoding='utf-8')) for p in setup[-1].glob('*-refine.json') if str(p) != parent)
    assert child['manual_override']['old_status'] == 'rejected'
    assert child['manual_override']['candidate'] == paths[2]
    assert child['parent_record'] == parent
    assert json.loads(Path(parent).read_text(encoding='utf-8'))['selected'] == paths[1]


def test_resume_source_hash_change_never_calls_api(setup):
    run(setup)
    parent = str(next(setup[-1].glob('*-refine.json')))
    setup[2].clear(); setup[3].clear()
    Path(setup[0][0]).write_bytes(b'changed')
    with pytest.raises(ValueError, match='源图'):
        refine.refine_image('', image_options={}, resume_record=parent, start_stage='background')
    assert not setup[2] and not setup[3]


def test_force_missing_candidate_never_calls_api(setup):
    run(setup)
    parent = str(next(setup[-1].glob('*-refine.json')))
    Path(setup[0][2]).unlink()
    setup[2].clear(); setup[3].clear()
    with pytest.raises(ValueError, match='不存在'):
        refine.refine_image('', image_options={}, resume_record=parent, start_stage='background', force_candidate=True)
    assert not setup[2] and not setup[3]


def test_explicit_manual_hair_retry_runs_even_when_previous_check_skipped_it(setup):
    paths, _, calls, texts, reports, _, _ = setup
    run(setup)
    parent = str(next(setup[-1].glob('*-refine.json')))
    calls.clear(); texts.clear(); reports[:] = [passed(), passed(), passed(), passed()]
    refine.refine_image('', image_options={}, resume_record=parent, start_stage='hair')
    assert calls[0]['file_prefix'].endswith('-hair')
    assert len(calls) == 4 and len(texts) == 4
