# -*- coding: utf-8 -*-
"""第二阶段（E0 + A）离线检查：多模型分发、保护块一致、缓存、预算与评价校验。

全部用例不联网：生图后端与视觉文字接口都被替换成桩函数。
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from utils import color_experiment as ce

SPEC_PATH = Path(ce.BASE) / 'prompts/color-knowledge/stage2-spring-fishing.json'
TEXT_CFG = {'model': 'text-test', 'api_key': 'topsecret', 'base_url': 'https://example.test'}


def real_spec():
    return ce.load_suite_spec(SPEC_PATH)


def make_image(path, size=(24, 16), color=(120, 160, 120)):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, color).save(path)
    return str(path)


class Stage2SpecTest(unittest.TestCase):
    def test_g0_matches_first_round_baseline_and_k_is_identical(self):
        spec = real_spec()
        first_round = json.loads((Path(ce.BASE) / 'prompts/color-knowledge/pilot-spring-fishing.json').read_text(encoding='utf-8'))
        baseline = first_round['subject'] + '\n\n' + first_round['rendering']
        prompts = {g['id']: ce.build_group_prompt(spec, g) for g in spec['groups']}
        self.assertEqual(prompts['G0'], baseline)
        self.assertNotIn('COLOR PLAN:', prompts['G1'])
        self.assertNotIn('PRESERVE:', prompts['G0'])
        for group_id in ('G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7'):
            self.assertEqual(prompts[group_id].split('PRESERVE:\n')[1], spec['protection_block'])
        for group_id in ('G0', 'G1', 'G2'):
            prefix = prompts[group_id].split('\n\nCOLOR PLAN:')[0].split('\n\nPRESERVE:')[0]
            self.assertEqual(prefix, baseline)

    def test_channel_parameters_match_the_frozen_protocol(self):
        spec = real_spec()
        gemini = ce.build_suite_request(spec, spec['groups'][2], 'gemini-flash')
        gpt = ce.build_suite_request(spec, spec['groups'][2], 'gpt-image-2')
        self.assertEqual((gemini['model'], gemini['resolution'], gemini['aspect_ratio']), ('gemini-3.1-flash-image-preview', '1K', '3:2'))
        self.assertFalse(gemini['face_quality_boost'])
        self.assertNotIn('size', gemini)
        self.assertEqual((gpt['model'], gpt['size'], gpt['quality'], gpt['n']), ('gpt-image-2', '1536x1024', 'medium', 1))
        self.assertEqual(gpt['mode'], 'generate')
        self.assertNotIn('face_quality_boost', gpt)
        self.assertEqual(gemini['prompt'], gpt['prompt'])

    def test_schedule_is_pre_registered_and_rotates_model_order(self):
        spec = real_spec()
        first = ce.build_schedule(spec)
        second = ce.build_schedule(spec)
        self.assertEqual(first, second)
        self.assertEqual(sum(len(r['order']) for r in first['rounds']), 48)
        self.assertEqual([r['model_start_order'] for r in first['rounds']],
                         [['gemini-flash', 'gpt-image-2'], ['gpt-image-2', 'gemini-flash'], ['gemini-flash', 'gpt-image-2']])
        ids = [item['sample_id'] for r in first['rounds'] for item in r['order']]
        self.assertEqual(len(ids), len(set(ids)))

    def test_invalid_specs_are_refused(self):
        spec = real_spec()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'spec.json'
            for broken, message in (
                    (dict(spec, rounds=2), 'three rounds'),
                    (dict(spec, groups=[dict(g, protection=True) if g['id'] == 'G0' else g for g in spec['groups']]), 'only G0'),
                    (dict(spec, channels={'gemini-flash': spec['channels']['gemini-flash']}), 'two channels'),
                    (dict(spec, evaluation=dict(spec['evaluation'], comparisons_per_model=[['G0', 'G1']])), 'comparisons'),
                    (dict(spec, evaluation=dict(spec['evaluation'], vision_call_caps={'calibration': 99})), 'caps')):
                path.write_text(json.dumps(broken, ensure_ascii=False), encoding='utf-8')
                with self.assertRaises(ValueError, msg=message):
                    ce.load_suite_spec(path)


class Stage2RunTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out = self.root / 'data/test-result/run'
        self.patches = [patch.object(ce, 'BASE', self.root)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temp.cleanup()

    def test_samples_are_dispatched_to_the_right_backend_per_channel(self):
        spec = real_spec()
        calls = {'gemini': [], 'gpt': []}

        def fake_gemini(**kwargs):
            calls['gemini'].append(kwargs)
            return {'saved_files': [make_image(Path(kwargs['save_sub_dir']) / 'gemini.png')], 'server_response_raw': {'model': 'gemini-3.1-flash-image-preview'}}

        def fake_gpt(**kwargs):
            calls['gpt'].append(kwargs)
            return {'saved_files': [make_image(Path(kwargs['save_sub_dir']) / 'gpt.png')], 'server_response_raw': {'model': 'gpt-image-2', 'data': [{'b64_json': 'x' * 4000}]}}

        with patch.object(ce, 'suite_preflight', return_value={'host': 'new.aigc2d.com'}), \
             patch('modules.others.api_backend.generate_image_aigc2d', side_effect=fake_gemini), \
             patch('modules.others.api_backend.generate_image_aigc2d_gpt', side_effect=fake_gpt):
            manifest = ce.run_suite(spec, self.out, rounds=[1], log=lambda _: None)
        self.assertEqual(len(calls['gemini']), 8)
        self.assertEqual(len(calls['gpt']), 8)
        self.assertEqual(len(manifest['samples']), 16)
        self.assertTrue(all(s['status'] == 'success' for s in manifest['samples']))
        for call in calls['gemini']:
            self.assertEqual(call['resolution'], '1K')
            self.assertFalse(call['face_quality_boost'])
            self.assertNotIn('size', call)
        for call in calls['gpt']:
            self.assertEqual((call['size'], call['quality'], call['n']), ('1536x1024', 'medium', 1))
            self.assertEqual(call['mode'], 'generate')
        self.assertNotIn('topsecret', (self.out / 'suite.json').read_text(encoding='utf-8'))

    def test_recorded_server_meta_never_keeps_base64_or_urls(self):
        meta = ce._redact_server_meta({'server_response_raw': {'model': 'gpt-image-2', 'data': [
            {'b64_json': 'A' * 5000, 'url': 'https://example.test/signed?token=abc'}]}, 'raw_text': 'revised prompt'})
        blob = json.dumps(meta)
        self.assertNotIn('AAAA', blob)
        self.assertNotIn('token=abc', blob)
        self.assertNotIn('b64_json', blob)
        self.assertEqual(meta['model'], 'gpt-image-2')

    def test_paid_success_is_cached_and_failures_are_not_retried(self):
        spec = real_spec()
        calls = []

        def fake_generate(request, folder, sample_id, log):
            calls.append(sample_id)
            if len(calls) == 2:
                raise RuntimeError('boom')
            return [make_image(folder / 'img.png')], {'model': request['model']}

        with patch.object(ce, 'suite_preflight', return_value={}), patch.object(ce, '_generate_suite_sample', side_effect=fake_generate):
            first = ce.run_suite(spec, self.out, rounds=[1], channels=['gemini-flash'], log=lambda _: None)
            self.assertEqual(len(calls), 8)
            self.assertEqual(len([s for s in first['samples'] if s['status'] == 'success']), 7)
            self.assertEqual([s['failure_type'] for s in first['samples'] if s['status'] != 'success'], ['RuntimeError'])
            ce.run_suite(spec, self.out, rounds=[1], channels=['gemini-flash'], log=lambda _: None)
            self.assertEqual(len(calls), 8)
            resumed = ce.run_suite(spec, self.out, rounds=[1], channels=['gemini-flash'], retry_failed=True, log=lambda _: None)
            self.assertEqual(len(calls), 9)
            self.assertEqual(len(resumed['previous_attempts']), 1)
            self.assertEqual(resumed['previous_attempts'][0]['status'], 'request_failed')

    def test_two_consecutive_failures_stop_that_model_block_only(self):
        spec = real_spec()
        calls = {'gemini-flash': 0, 'gpt-image-2': 0}

        def fake_generate(request, folder, sample_id, log):
            calls[request['channel']] += 1
            if request['channel'] == 'gemini-flash':
                raise RuntimeError('service down')
            return [make_image(folder / 'img.png')], {}

        with patch.object(ce, 'suite_preflight', return_value={}), patch.object(ce, '_generate_suite_sample', side_effect=fake_generate):
            manifest = ce.run_suite(spec, self.out, rounds=[1], log=lambda _: None)
        self.assertEqual(calls['gemini-flash'], 2)
        self.assertEqual(calls['gpt-image-2'], 8)
        self.assertTrue(manifest['node_blocks']['gemini-flash']['aborted'])
        self.assertNotIn('gpt-image-2', manifest['node_blocks'])

    def test_aborted_block_waits_for_an_explicit_user_resume(self):
        spec = real_spec()

        def always_fail(request, folder, sample_id, log):
            raise RuntimeError('upstream unavailable')

        with patch.object(ce, 'suite_preflight', return_value={}), \
             patch.object(ce, '_generate_suite_sample', side_effect=always_fail):
            manifest = ce.run_suite(spec, self.out, rounds=[1], channels=['gpt-image-2'], log=lambda _: None)
        self.assertTrue(manifest['node_blocks']['gpt-image-2']['aborted'])
        resumed_calls = []

        def ok(request, folder, sample_id, log):
            resumed_calls.append(sample_id)
            return [make_image(Path(folder) / 'img.png')], {}

        with patch.object(ce, 'suite_preflight', return_value={}), \
             patch.object(ce, '_generate_suite_sample', side_effect=ok):
            ce.run_suite(spec, self.out, rounds=[1], channels=['gpt-image-2'], retry_failed=True, log=lambda _: None)
            self.assertEqual(resumed_calls, [])
            resumed = ce.resume_block(self.out, 'gpt-image-2', note='user authorised one resume')
            self.assertTrue(resumed['resumed'])
            self.assertFalse(ce.resume_block(self.out, 'gpt-image-2')['resumed'])
            manifest = ce.run_suite(spec, self.out, rounds=[1], channels=['gpt-image-2'],
                                    retry_failed=True, log=lambda _: None)
        self.assertEqual(len(resumed_calls), 8)
        self.assertEqual(len([s for s in manifest['samples'] if s['status'] == 'success']), 8)
        self.assertEqual(len(manifest['block_history']), 1)
        self.assertEqual(manifest['previous_attempts'][0]['status'], 'request_failed')

    def test_protocol_change_and_publication_directory_are_refused(self):
        spec = real_spec()
        with patch.object(ce, 'suite_preflight', return_value={}), \
             patch.object(ce, '_generate_suite_sample', return_value=([], {})):
            ce.run_suite(spec, self.out, rounds=[1], channels=['gemini-flash'], log=lambda _: None)
        with patch.object(ce, 'suite_preflight', return_value={}), \
             patch.object(ce, '_generate_suite_sample', return_value=([], {})):
            with self.assertRaisesRegex(ValueError, 'Protocol changed'):
                ce.run_suite(dict(spec, subject=spec['subject'] + ' changed'), self.out, rounds=[1],
                             channels=['gemini-flash'], log=lambda _: None)
        with self.assertRaisesRegex(ValueError, 'test-result'):
            ce.run_suite(spec, self.root / 'other', rounds=[1], log=lambda _: None)


class Stage2VisionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out = self.root / 'data/test-result/run'
        self.out.mkdir(parents=True)
        self.image = self.out / 'img.png'
        make_image(self.image)
        self.patches = [patch.object(ce, 'BASE', self.root)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temp.cleanup()

    def test_vision_budget_stops_at_the_pre_registered_cap(self):
        calls = []

        def fake_call(*args, **kwargs):
            calls.append(kwargs)
            return json.dumps({'images': [{'id': 'V01', 'content_status': 'pass'}]})

        with patch('utils.analysis_gpt_prompt.call_text_model', side_effect=fake_call):
            for index in range(ce.SUITE_CAPS['calibration']):
                ce._vision_json_call(self.out, purpose='calibration', name=f'call-{index}', system='s', user='u',
                                     images=[self.image], cfg=TEXT_CFG, log=lambda _: None)
            with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
                ce._vision_json_call(self.out, purpose='calibration', name='call-over', system='s', user='u',
                                     images=[self.image], cfg=TEXT_CFG, log=lambda _: None)
        self.assertEqual(len(calls), ce.SUITE_CAPS['calibration'])
        self.assertEqual(ce.vision_call_counts(self.out, 'calibration'), ce.SUITE_CAPS['calibration'])

    def test_cached_vision_call_is_not_repeated(self):
        with patch('utils.analysis_gpt_prompt.call_text_model', return_value='{"ok": 1}') as call:
            for _ in range(2):
                ce._vision_json_call(self.out, purpose='calibration', name='same', system='s', user='u',
                                     images=[self.image], cfg=TEXT_CFG, log=lambda _: None)
            self.assertEqual(call.call_count, 1)

    def test_credentials_never_reach_the_snapshots(self):
        with patch('utils.analysis_gpt_prompt.call_text_model', return_value='{"ok": 1}'):
            ce._vision_json_call(self.out, purpose='calibration', name='named', system='s', user='u',
                                 images=[self.image], cfg=TEXT_CFG, log=lambda _: None)
        for path in self.out.rglob('*'):
            if path.is_file():
                self.assertNotIn('topsecret', path.read_text(encoding='utf-8', errors='ignore'))


class Stage2EvaluationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out = self.root / 'data/test-result/run'
        self.samples = []
        for model_key in ce.SUITE_MODELS:
            for group_id in ce.SUITE_GROUPS:
                for round_number in (1, 2, 3):
                    sample_id = f'{model_key}-{group_id}-r{round_number}'
                    image = make_image(self.out / 'samples' / model_key / sample_id / 'img.png')
                    self.samples.append({'id': sample_id, 'model_key': model_key, 'group_id': group_id,
                                         'round': round_number, 'status': 'success', 'outputs': [image]})
        (self.out).mkdir(parents=True, exist_ok=True)
        spec = real_spec()
        (self.out / 'suite.json').write_text(json.dumps(
            {'spec_id': 'spring-fishing-v2', 'samples': self.samples, 'protocol_hash': 'test',
             'protocol': {'spec': spec, 'schedule_seed': spec['schedule_seed']},
             'schedule': ce.build_schedule(spec), 'node_blocks': {}}), encoding='utf-8')
        self.patches = [patch.object(ce, 'BASE', self.root)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temp.cleanup()

    def _stub_compare(self, winner='tie'):
        def fake_call(base_url, api_key, model, system, user, **kwargs):
            order = json.loads(user)['image_order']
            rows = [{'id': row['id'], 'group': row['group'], 'content_status': 'pass',
                     'content_issues': [], 'uncertain': []} for row in order]
            return json.dumps({'images': rows,
                               'dimensions': {key: 'tie' for key in ce.SUITE_COMPARE_DIMENSIONS},
                               'overall': winner, 'overall_reason': '无可见差异',
                               'cost_of_colour_gain': '没有以内容为代价换取颜色',
                               'region_evidence': ['背景绿色区域', '人物发色区域'], 'limitations': []}, ensure_ascii=False)
        return fake_call

    def test_main_comparisons_follow_the_pre_registered_plan(self):
        spec = real_spec()
        with patch('utils.analysis_gpt_prompt.call_text_model', side_effect=self._stub_compare()) as call:
            ce.compare_groups(self.out, spec, text_cfg=TEXT_CFG, log=lambda _: None)
            self.assertEqual(call.call_count, 16)
            ce.compare_groups(self.out, spec, text_cfg=TEXT_CFG, log=lambda _: None)
            self.assertEqual(call.call_count, 16)
        plan = json.loads((self.out / 'comparisons' / 'plan.json').read_text(encoding='utf-8'))
        self.assertEqual(len(plan['swap_picks']), 2)
        for pick in plan['swap_picks']:
            self.assertIn(pick['model_key'], ce.SUITE_MODELS)
            self.assertIn(pick['pair'], plan['comparisons_per_model'])
        matrix = ce.comparison_matrix(self.out, spec)
        self.assertEqual(len(matrix['comparisons']), 16)
        self.assertEqual(len(matrix['swap_checks']), 2)
        self.assertTrue(all(not check['position_flip'] for check in matrix['swap_checks']))

    def test_comparison_must_cover_every_supplied_image(self):
        spec = real_spec()

        def incomplete(base_url, api_key, model, system, user, **kwargs):
            order = json.loads(user)['image_order'][:1]
            rows = [{'id': row['id'], 'group': row['group'], 'content_status': 'pass',
                     'content_issues': [], 'uncertain': []} for row in order]
            return json.dumps({'images': rows,
                               'dimensions': {key: 'tie' for key in ce.SUITE_COMPARE_DIMENSIONS},
                               'overall': 'tie', 'overall_reason': 'x', 'cost_of_colour_gain': 'x',
                               'region_evidence': ['x'], 'limitations': []}, ensure_ascii=False)

        with patch('utils.analysis_gpt_prompt.call_text_model', side_effect=incomplete):
            with self.assertRaisesRegex(ValueError, 'cover each neutral id'):
                ce.compare_groups(self.out, spec, text_cfg=TEXT_CFG, log=lambda _: None)

    def test_invalid_dimension_answer_is_refused(self):
        spec = real_spec()

        def bad_answer(base_url, api_key, model, system, user, **kwargs):
            order = json.loads(user)['image_order']
            rows = [{'id': row['id'], 'group': row['group'], 'content_status': 'pass',
                     'content_issues': [], 'uncertain': []} for row in order]
            dims = {key: 'tie' for key in ce.SUITE_COMPARE_DIMENSIONS}
            dims['subject_focus'] = 'better'
            return json.dumps({'images': rows, 'dimensions': dims, 'overall': 'tie', 'overall_reason': 'x',
                               'cost_of_colour_gain': 'x', 'region_evidence': ['x'], 'limitations': []}, ensure_ascii=False)

        with patch('utils.analysis_gpt_prompt.call_text_model', side_effect=bad_answer):
            with self.assertRaisesRegex(ValueError, 'Invalid dimension answer'):
                ce.compare_groups(self.out, spec, text_cfg=TEXT_CFG, log=lambda _: None)

    def test_conformance_picks_a_content_qualified_candidate_and_asks_about_execution(self):
        spec = real_spec()

        def compare_with_a_winner(base_url, api_key, model, system, user, **kwargs):
            order = json.loads(user)['image_order']
            rows = []
            for row in order:
                status = 'fail' if row['group'] == 'B' else 'pass'
                rows.append({'id': row['id'], 'group': row['group'], 'content_status': status,
                             'content_issues': ['双人'] if status == 'fail' else [], 'uncertain': []})
            return json.dumps({'images': rows,
                               'dimensions': {key: 'A' for key in ce.SUITE_COMPARE_DIMENSIONS},
                               'overall': 'A', 'overall_reason': 'A 组更清楚',
                               'cost_of_colour_gain': '无明显代价', 'region_evidence': ['背景', '裙摆'],
                               'limitations': []}, ensure_ascii=False)

        def conformance(base_url, api_key, model, system, user, **kwargs):
            payload = json.loads(user)
            return json.dumps({'clauses': [{'index': i + 1, 'executed': 'yes', 'regions': ['背景'],
                                            'evidence': ['背景整体偏绿']} for i in range(len(payload['clauses']))],
                               'protected_colors': [{'name': name, 'status': 'retained', 'evidence': ['可见']}
                                                    for name in payload['protected_colours']],
                               'conflicts': [], 'summary': '条款基本执行', 'limitations': []}, ensure_ascii=False)

        with patch('utils.analysis_gpt_prompt.call_text_model', side_effect=compare_with_a_winner):
            ce.compare_groups(self.out, spec, text_cfg=TEXT_CFG, log=lambda _: None)
        with patch('utils.analysis_gpt_prompt.call_text_model', side_effect=conformance) as call:
            result = ce.evaluate_conformance(self.out, spec, text_cfg=TEXT_CFG, log=lambda _: None)
            self.assertEqual(call.call_count, 2)
        for model_key in ce.SUITE_MODELS:
            record = result[model_key]
            self.assertFalse(record.get('skipped'))
            self.assertTrue(record['group_id'].startswith('G'))
            self.assertEqual(len(record['response']['clauses']), len(ce.group_by_id(spec, record['group_id'])['clauses']))
            self.assertEqual(ce.vision_call_counts(self.out, 'conformance'), 2)

    def test_conformance_is_skipped_without_a_content_qualified_candidate(self):
        spec = real_spec()

        def compare_all_fail(base_url, api_key, model, system, user, **kwargs):
            order = json.loads(user)['image_order']
            rows = [{'id': row['id'], 'group': row['group'], 'content_status': 'fail',
                     'content_issues': ['内容不达标'], 'uncertain': []} for row in order]
            return json.dumps({'images': rows,
                               'dimensions': {key: 'tie' for key in ce.SUITE_COMPARE_DIMENSIONS},
                               'overall': 'tie', 'overall_reason': 'x', 'cost_of_colour_gain': 'x',
                               'region_evidence': ['x'], 'limitations': []}, ensure_ascii=False)

        with patch('utils.analysis_gpt_prompt.call_text_model', side_effect=compare_all_fail):
            ce.compare_groups(self.out, spec, text_cfg=TEXT_CFG, log=lambda _: None)
        with patch('utils.analysis_gpt_prompt.call_text_model', side_effect=AssertionError('must not be called')):
            result = ce.evaluate_conformance(self.out, spec, text_cfg=TEXT_CFG, log=lambda _: None)
        self.assertTrue(all(result[key]['skipped'] for key in ce.SUITE_MODELS))
        self.assertEqual(ce.vision_call_counts(self.out, 'conformance'), 0)

    def test_swap_recheck_mirrors_the_stored_main_comparison(self):
        spec = real_spec()
        with patch('utils.analysis_gpt_prompt.call_text_model', side_effect=self._stub_compare()):
            ce.compare_groups(self.out, spec, model_key='gemini-flash', text_cfg=TEXT_CFG, log=lambda _: None)
            plan = json.loads((self.out / 'comparisons' / 'plan.json').read_text(encoding='utf-8'))
            picked = next(item for item in plan['swap_picks'] if item['model_key'] == 'gemini-flash')
            main_name = f"gemini-flash-{picked['pair'][0]}vs{picked['pair'][1]}"
            main = json.loads((self.out / 'comparisons' / f'{main_name}.json').read_text(encoding='utf-8'))
            mirror = ce._mirror_mapping(self.out, main_name)
            self.assertEqual(mirror['group_letter'],
                             {g: ('B' if l == 'A' else 'A') for g, l in main['mapping']['group_letter'].items()})
            self.assertEqual([row['sample_id'] for row in mirror['order']],
                             list(reversed([row['sample_id'] for row in main['mapping']['order']])))
            for main_row, mirror_row in zip(list(reversed(main['mapping']['order'])), mirror['order']):
                self.assertEqual(main_row['sample_id'], mirror_row['sample_id'])
                self.assertEqual(mirror_row['group'], 'B' if main_row['group'] == 'A' else 'A')
            self.assertEqual(sorted(mirror['group_letter']), sorted(main['mapping']['group_letter']))
            ce.compare_groups(self.out, spec, model_key='gemini-flash', text_cfg=TEXT_CFG, log=lambda _: None,
                              recheck=True)
        matrix = ce.comparison_matrix(self.out, spec)
        check = next(c for c in matrix['swap_checks'] if c['groups'] == list(picked['pair']))
        self.assertTrue(check['corrected_mirror'])
        self.assertIn('mirror', check['used_record'])
        self.assertTrue((self.out / 'comparisons' / 'plan-amendment.json').is_file())

    def test_summary_writes_an_offline_gallery_and_plan_cards(self):
        spec = real_spec()
        result = ce.summarize_suite(self.out, spec)
        gallery = Path(result['gallery']).read_text(encoding='utf-8')
        self.assertIn('方案卡', gallery)
        self.assertIn('data:image/jpeg;base64,', gallery)
        self.assertIn('保护块 K', gallery)
        cards = Path(result['plan_cards']).read_text(encoding='utf-8')
        self.assertIn('精确请求顺序', cards)
        self.assertIn(ce.SUITE_GROUPS[-1], cards)
        metrics = json.loads((self.out / 'metrics.json').read_text(encoding='utf-8'))
        self.assertEqual(len(metrics['images']), 48)
        self.assertEqual(len(metrics['group_means']), 16)


class E0VerdictTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out = self.root / 'data/test-result/run'
        self.out.mkdir(parents=True)
        self.patches = [patch.object(ce, 'BASE', self.root)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temp.cleanup()

    def _records(self, overalls):
        """两个交换调用：第二份把两侧组别对调（同一对图的 A/B 顺序互换）。"""
        letters = [('V01', 'V02'), ('V02', 'V01')]
        records = []
        for index, overall in enumerate(overalls):
            records.append({'name': f'swap-{index + 1}', 'response': {'overall': overall},
                            'mapping': {'letter_group': {'A': letters[index][0], 'B': letters[index][1]}}})
        return records

    def test_missing_double_person_detection_fails_calibration(self):
        spec = real_spec()
        plan = {'alias_of': {'soft-river-rose-1': 'V04'}}
        content = {'response': {'images': [
            {'id': 'V04', 'content_status': 'pass', 'content_issues': [], 'uncertain': [], 'observations': ['单人']}]}}
        aa = [{'name': 'aa-1', 'response': {'overall': 'tie', 'dimensions': {}}, 'mapping': {'letter_group': {'A': 'V01', 'B': 'V01'}}}]
        verdict = ce.e0_verdict(self.out, spec, plan, content, aa, self._records(['tie', 'tie']), [])
        self.assertEqual(verdict['status'], 'failed')
        self.assertIn('六图内容检查未识别柔蓝第一张的双人错误', verdict['failed_reasons'])

    def test_non_tie_aa_and_position_flip_fail_calibration(self):
        spec = real_spec()
        plan = {'alias_of': {'soft-river-rose-1': 'V04'}}
        content = {'response': {'images': [
            {'id': 'V04', 'content_status': 'fail', 'content_issues': ['出现两位粉发女性'],
             'uncertain': [], 'observations': []}]}}
        aa = [{'name': 'aa-1', 'response': {'overall': 'A', 'dimensions': {}}, 'mapping': {'letter_group': {'A': 'V01', 'B': 'V01'}}}]
        verdict = ce.e0_verdict(self.out, spec, plan, content, aa, self._records(['A', 'A']), [])
        self.assertEqual(verdict['status'], 'failed')
        self.assertIn('A/A 出现实质偏好', ' '.join(verdict['failed_reasons']))
        self.assertIn('顺序交换', ' '.join(verdict['failed_reasons']))

    def test_consistent_swap_and_tie_aa_pass(self):
        spec = real_spec()
        plan = {'alias_of': {'soft-river-rose-1': 'V04'}}
        content = {'response': {'images': [
            {'id': 'V04', 'content_status': 'fail', 'content_issues': ['画面里是两个人'],
             'uncertain': [], 'observations': []}]}}
        aa = [{'name': 'aa-1', 'response': {'overall': 'uncertain', 'dimensions': {}}, 'mapping': {'letter_group': {'A': 'V02', 'B': 'V02'}}}]
        verdict = ce.e0_verdict(self.out, spec, plan, content, aa, self._records(['A', 'B']), [])
        self.assertEqual(verdict['status'], 'passed')
        self.assertTrue(verdict['swap_consistent'])


TONE_SPEC_PATH = Path(ce.BASE) / 'prompts/color-knowledge/stage2-tone-effect.json'


class ToneEffectTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.out = self.root / 'data/test-result/tone-run'
        self.spec = ce.load_suite_spec(TONE_SPEC_PATH)
        self.samples = []
        for round_number in (1, 2, 3):
            for group in self.spec['groups']:
                sample_id = f'gemini-flash-{group["id"]}-r{round_number}'
                image = make_image(self.out / 'samples' / 'gemini-flash' / sample_id / 'img.png')
                self.samples.append({'id': sample_id, 'model_key': 'gemini-flash', 'group_id': group['id'],
                                     'round': round_number, 'status': 'success', 'outputs': [image]})
        (self.out).mkdir(parents=True, exist_ok=True)
        (self.out / 'suite.json').write_text(json.dumps(
            {'spec_id': self.spec['id'], 'samples': self.samples, 'protocol_hash': 'tone-test',
             'protocol': {'spec': self.spec, 'schedule_seed': self.spec['schedule_seed']},
             'schedule': ce.build_schedule(self.spec), 'node_blocks': {}}), encoding='utf-8')
        self.patches = [patch.object(ce, 'BASE', self.root)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temp.cleanup()

    def test_tone_spec_keeps_one_variable_and_the_same_protection_block(self):
        spec = self.spec
        self.assertEqual([g['id'] for g in spec['groups']], ['T0', 'T1', 'T2', 'T3', 'T4'])
        prompts = {g['id']: ce.build_group_prompt(spec, g) for g in spec['groups']}
        baseline = spec['subject'] + '\n\n' + spec['rendering']
        self.assertEqual(prompts['T0'], baseline)
        self.assertNotIn('PRESERVE:', prompts['T0'])
        for group_id in ('T1', 'T2', 'T3', 'T4'):
            self.assertEqual(prompts[group_id].split('PRESERVE:\n')[1], spec['protection_block'])
            self.assertEqual(prompts[group_id].split('\n\nCOLOR PLAN:')[0], baseline)
        self.assertIn('cool color palette', prompts['T1'])
        self.assertIn('warm color palette', prompts['T3'])
        self.assertEqual(len(ce.build_schedule(spec)['rounds']), 3)
        gemini = ce.build_suite_request(spec, spec['groups'][1], 'gemini-flash')
        self.assertEqual((gemini['resolution'], gemini['aspect_ratio']), ('1K', '3:2'))

    def test_tone_spec_rejects_treatments_without_the_protection_block(self):
        spec = json.loads(TONE_SPEC_PATH.read_text(encoding='utf-8'))
        broken = dict(spec, groups=[dict(g) for g in spec['groups']])
        broken['groups'][1]['protection'] = False
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'spec.json'
            path.write_text(json.dumps(broken, ensure_ascii=False), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'protection block'):
                ce.load_suite_spec(path)
            broken2 = dict(spec, groups=[dict(g) for g in spec['groups']])
            broken2['groups'][0]['clauses'] = ['colour']
            path.write_text(json.dumps(broken2, ensure_ascii=False), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'baseline'):
                ce.load_suite_spec(path)

    def test_hue_metrics_read_cool_and_warm_images(self):
        cool = make_image(self.root / 'cool.png', size=(40, 30), color=(60, 90, 200))
        warm = make_image(self.root / 'warm.png', size=(40, 30), color=(230, 150, 60))
        cool_row = ce._hue_metrics(cool)
        warm_row = ce._hue_metrics(warm)
        self.assertGreater(cool_row['cool_fraction_of_colorful'], 0.99)
        self.assertLess(cool_row['warm_fraction_of_colorful'], 0.01)
        self.assertGreater(warm_row['warm_fraction_of_colorful'], 0.99)
        self.assertLess(warm_row['cool_fraction_of_colorful'], 0.01)
        self.assertTrue(200 <= cool_row['mean_hue_circular'] <= 240)
        self.assertTrue(20 <= warm_row['mean_hue_circular'] <= 45)

    def _stub(self, tone_for_group, label_map=None):
        order = [s['id'] for s in self.samples]
        groups_of = {s['id']: s['group_id'] for s in self.samples}
        state = {'call': 0}

        def fake(base_url, api_key, model, system, user, **kwargs):
            rows_payload = json.loads(user)['image_order']
            chunk = state['call']
            state['call'] += 1
            rows = []
            for index, row in enumerate(rows_payload):
                sample_id = order[chunk * ce.TONE_CHUNK + index]
                group_id = groups_of[sample_id]
                tone = tone_for_group.get(group_id, 'neutral') if label_map is None else label_map
                rows.append({'id': row['id'], 'content_status': 'pass', 'content_issues': [], 'uncertain': [],
                             'subject_colours': 'retained', 'tone_reading': tone,
                             'tone_evidence': ['河面与阴影偏蓝'], 'cool_regions': ['河面'], 'warm_regions': ['鞋']})
            return json.dumps({'images': rows, 'summary': 'ok', 'limitations': []}, ensure_ascii=False)
        return fake

    def test_tone_readings_are_aggregated_per_group(self):
        with patch('utils.analysis_gpt_prompt.call_text_model', side_effect=self._stub({'T1': 'cool', 'T2': 'very_cool', 'T3': 'warm'})):
            summary = ce.evaluate_tone(self.out, self.spec, text_cfg=TEXT_CFG, log=lambda _: None)
        per_group = {row['group_id']: row for row in summary['per_group']}
        self.assertEqual(per_group['T2']['mean_tone_score'], -2.0)
        self.assertEqual(per_group['T3']['mean_tone_score'], 1.0)
        self.assertEqual(per_group['T0']['mean_tone_score'], 0.0)
        self.assertEqual(per_group['T1']['content_pass'], 3)
        self.assertEqual(ce.vision_call_counts(self.out, 'tone_reading'), 3)
        result = ce.summarize_tone(self.out, self.spec)
        gallery = Path(result['gallery']).read_text(encoding='utf-8')
        self.assertIn('冷暖读数', gallery)
        self.assertIn('T4', gallery)

    def test_tone_reading_rejects_unknown_labels(self):
        with patch('utils.analysis_gpt_prompt.call_text_model', side_effect=self._stub({}, label_map='quite_cold')):
            with self.assertRaisesRegex(ValueError, 'Invalid tone_reading'):
                ce.evaluate_tone(self.out, self.spec, text_cfg=TEXT_CFG, log=lambda _: None)


if __name__ == '__main__':
    unittest.main()
