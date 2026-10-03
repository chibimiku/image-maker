import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from utils import color_experiment as ce


def spec_for_test():
    return {'subject': 'test', 'rendering': 'style', 'model': 'gemini-test', 'api_type': 'aigc2d',
            'resolution': '1K', 'aspect_ratio': '3:2', 'repeats': 1,
            'profiles': [{'id': 'baseline', 'clauses': []}, {'id': 'second', 'clauses': ['color']}]}


class ColorExperimentTest(unittest.TestCase):
    def test_only_color_block_changes_between_groups(self):
        s = ce.load_spec(ce.BASE / 'prompts/color-knowledge/pilot-spring-fishing.json')
        baseline = ce.build_request(s, s['profiles'][0])
        for p in s['profiles'][1:]:
            request = ce.build_request(s, p)
            self.assertEqual(request['prompt'].split('\n\nCOLOR PLAN:')[0], baseline['prompt'])
            self.assertEqual({k: v for k, v in request.items() if k != 'prompt'},
                             {k: v for k, v in baseline.items() if k != 'prompt'})

    def test_paid_successes_and_failures_are_not_automatically_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / 'data/test-result/run'
            calls = []

            def generate(**kwargs):
                calls.append(kwargs)
                if len(calls) == 2:
                    return []
                image = Path(kwargs['save_sub_dir']) / 'image.png'
                Image.new('RGB', (20, 20)).save(image)
                return [str(image)]

            with patch.object(ce, 'BASE', root), patch.object(ce, 'preflight', return_value={'model': 'gemini-test'}), patch('modules.others.api_backend.generate_image_aigc2d', side_effect=generate):
                s = spec_for_test()
                first = ce.run_pilot(s, out, log=lambda _: None)
                self.assertEqual([x['status'] for x in first['samples']], ['success', 'failed'])
                ce.run_pilot(s, out, log=lambda _: None)
                self.assertEqual(len(calls), 2)
                resumed = ce.run_pilot(s, out, retry_failed=True, log=lambda _: None)
                self.assertEqual(len(calls), 3)
                self.assertEqual(len(resumed['previous_attempts']), 1)
                with self.assertRaisesRegex(ValueError, 'Protocol changed'):
                    ce.run_pilot(dict(s, subject='changed'), out, log=lambda _: None)
                self.assertEqual(len(calls), 3)

    def test_pilot_rejects_publication_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'test-result'):
                ce.run_pilot(spec_for_test(), temp)

    def test_unknown_automatic_selection_is_not_accepted(self):
        s = ce.load_spec(ce.BASE / 'prompts/color-knowledge/pilot-spring-fishing.json')
        with tempfile.TemporaryDirectory() as temp:
            with patch('utils.analysis_gpt_prompt.call_text_model', return_value=json.dumps({'selected_profile_id': 'invented', 'reason': 'x', 'confidence': 1})):
                with self.assertRaisesRegex(ValueError, 'unknown profile'):
                    ce.select_profile(s, temp, text_cfg={'model': 'test', 'api_key': 'secret', 'base_url': 'https://example.test'})
            self.assertFalse((Path(temp) / 'auto-selection.json').exists())
            self.assertNotIn('secret', (Path(temp) / 'auto-selection.request.json').read_text(encoding='utf-8'))

    def test_evaluation_requires_complete_image_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            image = out / 'input.png'
            Image.new('RGB', (20, 20)).save(image)
            manifest = {'protocol': {'spec': {'subject': 'one person'}},
                        'samples': [{'id': 'baseline-1', 'status': 'success', 'outputs': [str(image)]}]}
            (out / 'experiment.json').write_text(json.dumps(manifest), encoding='utf-8')
            with patch('utils.analysis_gpt_prompt.call_text_model', return_value='{"images": []}'):
                with self.assertRaisesRegex(ValueError, 'cover each image'):
                    ce.evaluate_pilot(out, text_cfg={'model': 'test', 'api_key': 'secret', 'base_url': 'https://example.test'})
            self.assertFalse((out / 'visual-review.json').exists())

    def test_successful_selection_reuses_paid_response(self):
        s = ce.load_spec(ce.BASE / 'prompts/color-knowledge/pilot-spring-fishing.json')
        response = {'selected_profile_id': 'baseline', 'reason': 'compatible', 'confidence': 0.5, 'alternatives': []}
        with tempfile.TemporaryDirectory() as temp:
            with patch('utils.analysis_gpt_prompt.call_text_model', return_value=json.dumps(response)) as call:
                cfg = {'model': 'test', 'api_key': 'secret', 'base_url': 'https://example.test'}
                ce.select_profile(s, temp, text_cfg=cfg)
                ce.select_profile(s, temp, text_cfg=cfg)
                self.assertEqual(call.call_count, 1)


if __name__ == '__main__':
    unittest.main()
