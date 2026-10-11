import json
import os
from pathlib import Path

import pytest
from PIL import Image

from utils import repaint_experiment as experiment
from utils import send_budget
from modules.others import api_backend


@pytest.fixture
def plan(tmp_path):
    source = tmp_path / 'source.png'
    Image.new('RGB', (120, 80)).save(source)
    prompt = tmp_path / 'prompt.txt'
    prompt.write_text('preserve input', encoding='utf-8')
    data = dict(output_dir=str(tmp_path / 'output'), run_id='unit', model='fixture-model',
                resolution='2K', aspect_ratio='3:2', api_type='aigc2d', text_model='fixture-text',
                image_limit=1, text_limit=1,
                frozen_hashes={str(p): experiment.digest(p) for p in [source, prompt]},
                jobs={'A': dict(source=str(source), prompt_file=str(prompt))},
                audits={'audit': dict(system_file=str(prompt), user_file=str(prompt), images=[str(source)])})
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    return path, data, source, prompt


def test_changed_frozen_input_never_calls_api(plan, monkeypatch):
    path, _, _, prompt = plan
    prompt.write_text('changed')
    monkeypatch.setattr(api_backend, 'generate_image_repaint', lambda **kw: pytest.fail('sent'))
    with pytest.raises(ValueError, match='Frozen input changed'):
        experiment.run_plan(path, job='A')


def test_dry_run_has_no_attempt_record_or_api(plan, monkeypatch):
    path, data, _, _ = plan
    monkeypatch.setattr(api_backend, 'generate_image_repaint', lambda **kw: pytest.fail('sent'))
    request = experiment.run_plan(path, job='A', dry_run=True)
    assert request['retry_limit'] == 0 and request['extra_reference_paths'] == []
    assert not Path(data['output_dir']).exists()


def test_attempt_cannot_be_resent_and_environment_restored(plan, monkeypatch):
    path, _, source, _ = plan
    calls = []
    monkeypatch.setenv(send_budget.ENV_DIR, 'previous')
    monkeypatch.setattr(api_backend, 'generate_image_repaint', lambda **kw: calls.append(kw) or [str(source)])
    assert experiment.run_plan(path, job='A')['status'] == 'returned'
    assert os.environ[send_budget.ENV_DIR] == 'previous'
    with pytest.raises(FileExistsError):
        experiment.run_plan(path, job='A')
    assert len(calls) == 1


def test_unknown_image_failure_preserves_attempt_without_retry(plan, monkeypatch):
    path, data, _, _ = plan
    def fail(**kwargs):
        raise TimeoutError('read timeout')
    monkeypatch.setattr(api_backend, 'generate_image_repaint', fail)
    result = experiment.run_plan(path, job='A')
    assert result['status'] == 'failed' and result['billed'] == 'unknown'
    assert (Path(data['output_dir']) / 'A.request.json').exists()
    with pytest.raises(FileExistsError):
        experiment.run_plan(path, job='A')


def test_text_failure_uses_one_slot_and_cannot_retry(plan, monkeypatch):
    from utils import analysis_gpt_prompt
    path, data, _, _ = plan
    monkeypatch.setattr(analysis_gpt_prompt, 'load_text_api_config', lambda: dict(model='fixture-text', base_url='fixture', api_key='fixture'))
    def fail(*args, **kwargs):
        raise TimeoutError('read timeout')
    monkeypatch.setattr(analysis_gpt_prompt, 'call_text_model', fail)
    assert experiment.run_plan(path, audit='audit')['status'] == 'failed'
    assert len(list((Path(data['output_dir']) / 'budget/text-slots').glob('*.json'))) == 1
    with pytest.raises(FileExistsError):
        experiment.run_plan(path, audit='audit')


def test_job_selector_is_required(plan):
    with pytest.raises(ValueError, match='exactly one'):
        experiment.run_plan(plan[0])


def test_two_image_roles_preserve_uploaded_order(plan, monkeypatch):
    path, data, source, prompt = plan
    reference = source.with_name('style.png')
    Image.new('RGB', (120, 80), 'blue').save(reference)
    data['jobs']['A']['extra_reference_paths'] = [str(reference)]
    data['frozen_hashes'][str(reference)] = experiment.digest(reference)
    path.write_text(json.dumps(data), encoding='utf-8')
    calls = []
    monkeypatch.setattr(api_backend, 'generate_image_repaint', lambda **kw: calls.append(kw) or [str(source)])
    assert experiment.run_plan(path, job='A')['status'] == 'returned'
    assert calls[0]['source_paths'] + calls[0]['extra_reference_paths'] == [str(source), str(reference)]


def test_gpt_reserves_one_slot_disables_retries_and_freezes_prompt(plan, monkeypatch):
    path, data, source, prompt = plan
    data['jobs']['A'] = dict(kind='gpt', model='fixture-gpt', size='1536x1024',
                            quality='high', output_format='png', api_type='fixture-gpt',
                            image_paths=[str(source)], prompt_file=str(prompt), mode='edit')
    path.write_text(json.dumps(data), encoding='utf-8')
    calls = []
    def generate(**kwargs):
        assert os.environ[api_backend.IMAGE_RETRY_ENV] == '0'
        calls.append(kwargs)
        return [str(source)]
    monkeypatch.setattr(api_backend, 'generate_image_aigc2d_gpt', generate)
    assert experiment.run_plan(path, job='A')['status'] == 'returned'
    assert len(calls) == 1 and calls[0]['prompt'] == prompt.read_text()
    rows = [json.loads(line) for line in (Path(data['output_dir']) / 'budget/images.jsonl').read_text().splitlines()]
    assert [row['event'] for row in rows] == ['reserved', 'settled']


def test_region_measurement_excludes_marked_highlight_and_never_adopts(tmp_path):
    source = tmp_path / 'region.png'
    im = Image.new('RGB', (5, 5), 'green')
    im.putpixel((2, 2), (255, 255, 255))
    im.save(source)
    manifest = {'fixture': dict(path=str(source), sha256=experiment.digest(source))}
    regions = {'canvas': [5, 5], 'preview_box': [0, 0, 5, 5],
               'regions': {'fixture': {'include': [[[0, 0], [4, 0], [4, 4], [0, 4]]],
                                        'exclude': [[[2, 2], [2, 2], [2, 2]]]}}}
    result = experiment.measure_annotated_regions(manifest, regions, tmp_path / 'metrics')['fixture']
    assert result['sample_pixels'] == 24
    assert result['saturation_median'] == 1
    assert result['automatic_fidelity_verdict'] is None
