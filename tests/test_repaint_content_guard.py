import json
from pathlib import Path

import pytest
from PIL import Image
from utils import repaint_content_guard as guard
from modules.others import api_backend as api


def report(**changes):
    value = dict.fromkeys(guard.CORE_FIELDS, True)
    value.update(confidence=0.99, global_replacement=False, verdict='pass',
                 eligible_for_local_cleanup=False, added_elements=[], missing_or_replaced_elements=[],
                 rendering=dict(source_medium_preserved=True, uniform_ink_or_cel_conversion=False))
    value.update(changes)
    return value


@pytest.fixture
def images(tmp_path):
    paths = [str(tmp_path / name) for name in ('source.png', 'result.png', 'clean.png')]
    for path in paths:
        Image.new('RGB', (120, 80), 'white').save(path)
    return paths


@pytest.mark.parametrize('changes', [dict(global_replacement=True), dict(pose_preserved=False),
    dict(rendering={}), dict(rendering=dict(source_medium_preserved=False)), dict(confidence=0.5), dict(confidence=True), dict(confidence=float('nan')),
    dict(missing_or_replaced_elements=[{'element': 'chair'}]), dict(verdict='fail'),
    dict(added_elements=[{'element': 'flower'}])])
def test_unsafe_reports_fallback(changes):
    assert guard.assess_report(report(**changes)) == 'fallback'


def test_global_replacement_never_sends_cleanup(images, monkeypatch):
    before, after, _ = images
    monkeypatch.setattr(guard, 'audit_pair', lambda *a, **k: {'report': report(global_replacement=True)})
    monkeypatch.setattr(api, 'generate_image_repaint', lambda **k: pytest.fail('must not clean global replacement'))
    logs = []
    assert guard.guard_repaint(before, after, log_callback=logs.append) == before
    assert Path(after + '-content-audit.json').exists(), logs
    record = json.loads(Path(after + '-content-audit.json').read_text(encoding='utf-8'))
    assert record['action'] == 'fallback'
    assert all(Path(p).exists() for p in images)


@pytest.mark.parametrize('verified', [True, False])
def test_cleanup_runs_once_and_reaudits_original(images, monkeypatch, verified):
    before, after, cleaned = images
    addition = dict(element='flower', location='chest', source_evidence='plain cloth',
                    removal_instruction='remove flower', confidence=0.99)
    audits = []
    def audit(a, b, **kwargs):
        audits.append((a, b))
        return {'report': report(eligible_for_local_cleanup=True, added_elements=[addition], verdict='fail') if b == after else report(pose_preserved=verified)}
    monkeypatch.setattr(guard, 'audit_pair', audit)
    calls = []
    monkeypatch.setattr(api, 'generate_image_repaint', lambda **kw: calls.append(kw) or [cleaned])
    assert guard.guard_repaint(before, after) == (cleaned if verified else before)
    assert audits == [(before, after), (before, cleaned)]
    assert len(calls) == 1
    assert calls[0]['source_paths'] == [before]
    assert calls[0]['extra_reference_paths'] == [after]
    assert calls[0]['content_guard'] is False
    assert calls[0]['request_timeout'] == 360 and calls[0]['retry_limit'] == 0


@pytest.mark.parametrize('failure', ['empty', 'exception', 'wrong_aspect'])
def test_request_failures_preserve_every_source(images, monkeypatch, failure):
    before, after, _ = images
    def generate(**kw):
        if failure == 'exception':
            raise TimeoutError('network timeout')
        if failure == 'wrong_aspect':
            Image.new('RGB', (80, 80)).save(after)
            return [after]
        return []
    monkeypatch.setattr(api, 'generate_image_aigc2d', generate)
    assert api.generate_image_repaint([before, images[2]], content_guard=True, request_timeout=360,
                                     retry_limit=0, aspect_ratio='3:2') == [before, images[2]]


def test_success_audits_and_disables_repeat(images, monkeypatch):
    before, after, _ = images
    calls = []
    monkeypatch.setattr(api, 'generate_image_aigc2d', lambda **kw: calls.append(kw) or [after])
    monkeypatch.setattr(guard, 'audit_pair', lambda *a, **k: {'report': report()})
    assert api.generate_image_repaint([before], content_guard=True, request_timeout=360,
        retry_limit=0, repeat=3, aspect_ratio='3:2') == [after]
    assert len(calls) == 1 and calls[0]['request_timeout'] == 360 and calls[0]['retry_limit'] == 0


def test_audit_error_and_cancel_do_not_cleanup(images, monkeypatch):
    def audit(*a, **kw):
        raise RuntimeError('audit unavailable')
    monkeypatch.setattr(guard, 'audit_pair', audit)
    monkeypatch.setattr(api, 'generate_image_repaint', lambda **kw: pytest.fail('unexpected request'))
    assert guard.guard_repaint(*images[:2]) == images[0]


def test_transport_timeout_retry_are_per_call(images, monkeypatch, tmp_path):
    import requests
    config = {'api_key': 'test', 'base_url': 'https://example.invalid/v1beta/models/', 'timeout': 120, 'max_retries': 3}
    monkeypatch.setattr(api, 'get_api_config', lambda **kw: config)
    calls = []
    def post(*args, **kwargs):
        calls.append(kwargs)
        raise requests.exceptions.ReadTimeout('timeout')
    monkeypatch.setattr(api.requests, 'post', post)
    api.generate_image_aigc2d('repair', image_paths=[images[0]], request_timeout=360,
        retry_limit=0, save_sub_dir=str(tmp_path))
    assert len(calls) == 1 and calls[0]['timeout'] == 360
    assert config['timeout'] == 120 and config['max_retries'] == 3


def test_effective_request_source_only_even_with_old_reference_flags(images):
    from utils.post_process import assemble_repaint_request
    value = assemble_repaint_request(images[0], {'content_guard': True, 'reference_mode': 'both',
        'text_without_image': True, 'scope': 'full'}, firmware='flat cel shading',
        style_ref_path=images[1], style_clauses=['STYLE SENTINEL'])
    assert value['extra_reference_paths'] == []
    assert 'STYLE SENTINEL' not in value['prompt'] and 'flat cel shading' not in value['prompt']
    assert '{source_contract}' not in value['prompt'] and '{style_role}' not in value['prompt']
    assert value['call_kwargs']['content_guard'] is True


def test_cancel_before_cleanup_sends_nothing(images, monkeypatch):
    addition = dict(element='flower', location='chest', source_evidence='cloth', removal_instruction='remove', confidence=0.99)
    monkeypatch.setattr(guard, 'audit_pair', lambda *a, **kw: {'report': report(added_elements=[addition], eligible_for_local_cleanup=True)})
    monkeypatch.setattr(api, 'generate_image_repaint', lambda **kw: pytest.fail('cancelled cleanup'))
    assert guard.guard_repaint(*images[:2], cancel_check=lambda: True) == images[0]


def test_audit_uses_source_result_order(images, monkeypatch):
    from utils import analysis_gpt_prompt as text_api
    monkeypatch.setattr(text_api, 'load_text_api_config', lambda: dict(base_url='unused', api_key='test', model='test-vision'))
    calls = []
    monkeypatch.setattr(text_api, 'call_text_model', lambda *a, **kw: calls.append(kw) or json.dumps(report()))
    result = guard.audit_pair(*images[:2])
    assert result['report']['same_artwork'] is True
    assert calls[0]['image_paths'] == images[:2] and calls[0]['timeout'] == 180


def test_cost_includes_conditional_guard_calls():
    from utils.cost_estimate import estimate_pipeline
    estimate = estimate_pipeline(1000, include_repaint=True, include_content_guard=True,
                                 include_local=False, pricing={'models': {}})
    assert {'content_audit', 'content_cleanup'} <= {step['key'] for step in estimate['steps']}
