import json
from pathlib import Path
from PIL import Image
from utils import refine_delivery as delivery
from utils import audit_display


def test_bilingual_display_is_independent_of_gate_fields():
    from utils.first_image_audit import normalize_audit
    report = {'content_ok': True, 'style_ok': True, 'copied_from_reference': False,
              'verdict': 'pass', 'confidence': .95, 'content_issues': [], 'style_issues': [],
              'copied_elements': [], 'uncertain': [], 'summary': 'All checks passed.',
              'display_zh': {'summary': '检查通过', 'details': '没有发现重大问题。', 'verdict': 'fail'}}
    normalized = normalize_audit(report)
    assert normalized['verdict'] == 'pass'
    assert audit_display.summary(normalized) == '检查通过'
    assert 'verdict' not in normalized['display_zh']
    assert audit_display.summary({'summary': 'legacy'}) == 'legacy'


def test_saved_recipe_excludes_paid_repaint(tmp_path):
    root = tmp_path / '20261011'
    manifest = root / 'pipeline-steps/run/pipeline-manifest.json'
    manifest.parent.mkdir(parents=True)
    source = str(root / 'gpt-image-2/source.png')
    final = str(root / 'final.png')
    manifest.write_text(json.dumps({'recipe': {'steps': {'ink': {'enabled': True, 'amount': .6},
                                      'repaint': {'enabled': True}}, 'style_ref_path': ''},
                                   'items': [{'source': source, 'final': final}]}))
    assert delivery.delivery_recipe(final, {'source': source})['steps'] == {'ink': {'enabled': True, 'amount': .6}}


def test_finished_without_recipe_exports_final_png(tmp_path, monkeypatch):
    import utils.post_process as post
    root = tmp_path / '20261011'
    root.mkdir()
    image = root / 'candidate.jpg'
    Image.new('RGB', (20, 30), 'red').save(image)
    record = root / 'test-refine.json'
    record.write_text(json.dumps({'source': str(root / 'original.png'), 'selected': str(image),
                                  'status': 'finished', 'stages': []}))
    monkeypatch.setattr(post, 'date_output_dir', lambda: str(tmp_path / 'delivery'))
    monkeypatch.setattr('utils.output_isolation.resolve_output_target', lambda directory, name: (directory, name))
    outputs = delivery.deliver_refine(str(image), str(record), {'steps': {}})
    assert '-final-refine.png' in outputs[0]
    assert Image.open(outputs[0]).size == (20, 30)
    assert len(list((tmp_path / 'delivery').glob('pipeline-steps/**/pipeline-manifest.json'))) == 1
    record.write_text(json.dumps({'source': str(root / 'original.png'), 'selected': str(image),
                                  'status': 'stopped', 'stages': []}))
    assert delivery.deliver_refine(str(image), str(record), {'steps': {'ink': {'enabled': True}}}) == [str(image)]
