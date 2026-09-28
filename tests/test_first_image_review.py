import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from utils.first_image_review import generate_first_image, is_moderation_block, apply_safe_plan, safe_repaint_firmware
from utils.generation_checkpoint import GenerationCheckpoint

BLOCK = {"saved_files": [], "server_response_raw": {"error": {"code": "moderation_blocked"}}}

class FirstReviewTests(unittest.TestCase):
    def test_timeout_reserves_conditional_restore_and_prompt_revision(self):
        from modules.image_analysis.single_analyzer import SingleAnalyzerWidget
        budget = SingleAnalyzerWidget._pipeline_timeout_budget
        self.assertEqual(budget(None, 120, {'repaint': {'enabled': True}}), 2160)
        self.assertEqual(budget(None, 120, {'repaint': {'enabled': False}}, {'style_ref_path': 'ref'}), 2400)
        self.assertEqual(budget(None, 120, {'repaint': {'enabled': False}}, {}), 1440)

    def test_gpt_retry_omits_reference_but_gemini_restores_technique_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            ref = str(Path(directory) / 'style.jpg'); Path(ref).touch()
            first = str(Path(directory) / 'first.png'); Path(first).touch()
            cp = GenerationCheckpoint(str(Path(directory) / 'checkpoint.json'))
            payload = {'prompt': 'old', 'style_ref_path': ref, 'image_paths': [ref],
                       'generation_clauses': ['old outfit override'], 'skip_repaint': True}
            steps = {'repaint': {'enabled': False}, 'tone': {'enabled': True}}
            plan = {'version': 3, 'retry_allowed': True, 'changes': ['retain covered dress'],
                    'prompt': 'covered fashion illustration', 'reference_policy': 'none',
                    'gemini_reference_policy': 'style_only', 'safe_rendering_clauses': []}
            gen = Mock(side_effect=[BLOCK, {'saved_files': [first]}])
            with patch('utils.first_image_review.review_request', return_value=plan):
                generate_first_image(gen, prompt='old', image_paths=[ref], checkpoint=cp,
                    plan_callback=lambda p: apply_safe_plan(payload, steps, p))
            self.assertEqual(gen.call_args.kwargs['image_paths'], [])
            self.assertEqual(payload['image_paths'], [])
            self.assertEqual(payload['style_ref_path'], ref)
            self.assertTrue(steps['repaint']['enabled'])
            self.assertEqual(steps['repaint']['reference_mode'], 'style')
            self.assertNotIn('generation_clauses', payload)
            self.assertFalse(payload['skip_quality_refine'])
            self.assertIn('IMAGE 2', safe_repaint_firmware(payload, 'old'))
            preserved = {'style_ref_path': ref}
            old_steps = {'repaint': {'enabled': False}}
            apply_safe_plan(preserved, old_steps, plan, restore_reference=False)
            self.assertFalse(old_steps['repaint']['enabled'])

    def test_worker_fallback_uses_style_transfer_then_caches_paid_stages(self):
        from PIL import Image
        from modules.image_analysis.single_analyzer import GptImageGenWorkerThread
        with tempfile.TemporaryDirectory() as directory:
            first = str(Path(directory) / 'first.png'); ref = str(Path(directory) / 'ref.png')
            for path in (first, ref):
                Image.new('RGB', (32, 48), 'white').save(path)
            cp_path = str(Path(directory) / 'checkpoint.json')
            plan = {'version': 3, 'retry_allowed': True, 'changes': ['covered original outfit'],
                    'prompt': 'covered dress illustration', 'reference_policy': 'none',
                    'gemini_reference_policy': 'style_only', 'safe_rendering_clauses': []}
            kwargs = dict(request_payload={'prompt': 'old theme', 'image_paths': [ref], 'style_ref_path': ref},
                steps={'repaint': {'enabled': False}}, analysis_result={'task_hash': '12345678'},
                style_ref_path=ref, checkpoint_path=cp_path)
            clean = {'needs_refine': False, 'structural_issues': [], 'needs_review': False}
            with patch('utils.first_image_review.review_request', return_value=plan), \
                 patch('modules.others.api_backend.generate_image_aigc2d_gpt', side_effect=[BLOCK, {'saved_files': [first]}]) as gen, \
                 patch('utils.analysis_gen.run_gpt_image_pipeline', return_value=[first]) as repaint, \
                 patch('utils.identity_audit.audit_image_identity', return_value={'mismatch': False, 'differences': []}) as identity, \
                 patch('utils.refine_quality.audit_refine_quality', return_value=clean), \
                 patch('utils.refine_quality.audit_hand_quality', return_value=clean), \
                 patch('utils.analysis_gen.publish_final_output', return_value=first):
                worker = GptImageGenWorkerThread(**kwargs); worker.run()
                self.assertEqual(worker.last_status, 'success')
                self.assertEqual(gen.call_args.kwargs['image_paths'], [])
                self.assertEqual(repaint.call_args.kwargs['style_ref_path'], ref)
                self.assertIn('fully clothed', repaint.call_args.kwargs['firmware'])
                self.assertNotIn('old theme', repaint.call_args.kwargs['firmware'])
                resumed = GptImageGenWorkerThread(**kwargs); resumed.run()
                self.assertEqual(resumed.last_status, 'success')
                self.assertEqual(gen.call_count, 2)
                repaint.assert_called_once(); identity.assert_called_once()

    def test_gpt_only_safe_alternative_still_runs_identity_gate(self):
        from PIL import Image
        from modules.image_analysis.single_analyzer import GptImageGenWorkerThread
        with tempfile.TemporaryDirectory() as directory:
            first = str(Path(directory) / 'first.png')
            Image.new('RGB', (32, 48), 'white').save(first)
            checkpoint_path = str(Path(directory) / 'checkpoint.json')
            cp = GenerationCheckpoint(checkpoint_path)
            cp.begin('first', []); cp.complete('first', [first])
            cp.data['first_safe_review'] = {'outputs': [first], 'plan': {
                'retry_allowed': True, 'prompt': 'covered dress illustration',
                'reference_policy': 'none', 'safe_rendering_clauses': []}}
            cp.save()
            worker = GptImageGenWorkerThread({'prompt': 'old', 'skip_identity_refine': True},
                steps={'repaint': {'enabled': False}}, analysis_result={'task_hash': '12345678'},
                checkpoint_path=checkpoint_path)
            with patch('utils.identity_audit.audit_image_identity', return_value={
                    'mismatch': False, 'differences': [], 'severity': 'none'}) as audit, \
                 patch('utils.refine_quality.audit_hand_quality', return_value={
                    'needs_refine': False, 'structural_issues': [], 'needs_review': False}), \
                 patch('utils.analysis_gen.publish_final_output', return_value=first), \
                 patch('modules.others.api_backend.generate_image_aigc2d_gpt', side_effect=AssertionError('must reuse first')):
                worker.run()
            audit.assert_called_once()
            self.assertEqual(audit.call_args.kwargs['expected_prompt'], 'covered dress illustration')
            self.assertEqual(worker.last_status, 'success')

    def test_explicit_blocks_only(self):
        self.assertTrue(is_moderation_block(BLOCK))
        self.assertFalse(is_moderation_block({"server_response_raw": {"error": {"code": "503"}}}))
        self.assertFalse(is_moderation_block([]))

    def test_ordinary_failure_does_not_analyze(self):
        with patch('utils.first_image_review.review_request') as review:
            self.assertEqual(generate_first_image(Mock(return_value=[]), prompt='original'), ([], 'original'))
            review.assert_not_called()

    def test_decline_does_not_retry(self):
        gen = Mock(return_value=BLOCK)
        with tempfile.TemporaryDirectory() as directory:
            cp = GenerationCheckpoint(str(Path(directory) / 'checkpoint.json'))
            with patch('utils.first_image_review.review_request', return_value={"retry_allowed": False, "version": 2}):
                generate_first_image(gen, prompt='original', checkpoint=cp)
        self.assertEqual(gen.call_count, 1)

    def test_one_revision_and_cached_paid_output(self):
        with tempfile.TemporaryDirectory() as directory:
            image = str(Path(directory) / 'result.png'); Path(image).touch()
            cp = GenerationCheckpoint(str(Path(directory) / 'checkpoint.json'))
            plan = {"retry_allowed": True, "changes": ["remove unsupported clothing change"], "prompt": "covered fashion illustration"}
            gen = Mock(side_effect=[BLOCK, {"saved_files": [image]}])
            with patch('utils.first_image_review.review_request', return_value=plan) as review:
                result = generate_first_image(gen, prompt='original', checkpoint=cp)
                self.assertEqual(result, ([image], plan['prompt']))
                self.assertEqual(generate_first_image(gen, prompt='original', checkpoint=cp), result)
                self.assertEqual(gen.call_count, 2); self.assertEqual(review.call_count, 1)
            cp.reset_from('first')
            self.assertNotIn('first_safe_review', cp.data)

    def test_second_block_not_repeated_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            cp = GenerationCheckpoint(str(Path(directory) / 'checkpoint.json'))
            gen = Mock(return_value=BLOCK)
            with patch('utils.first_image_review.review_request', return_value={"retry_allowed": True, "version": 2, "changes": ['remove invention'], "prompt": 'safe revised'}):
                generate_first_image(gen, prompt='original', checkpoint=cp)
                generate_first_image(gen, prompt='original', checkpoint=cp)
            self.assertEqual(gen.call_count, 2)

    def test_unsafe_references_removed_from_retry_and_subsequent_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            cp = GenerationCheckpoint(str(Path(directory) / 'checkpoint.json'))
            plan = {"retry_allowed": True, "version": 2, "changes": ['keep original covered dress'],
                    "prompt": 'covered source outfit illustration', "reference_policy": 'none',
                    "safe_rendering_clauses": ['watercolor texture']}
            payload = {"image_paths": ['unsafe.jpg'], "style_ref_path": 'unsafe.jpg',
                       "style_text": 'old instructions', "generation_clauses": ['old override']}
            steps = {"repaint": {"reference_mode": 'style'}, "tone": {"enabled": True}}
            gen = Mock(side_effect=[BLOCK, []])
            with patch('utils.first_image_review.review_request', return_value=plan):
                generate_first_image(gen, prompt='old', image_paths=['unsafe.jpg'], checkpoint=cp,
                    plan_callback=lambda p: apply_safe_plan(payload, steps, p))
            self.assertEqual(gen.call_args.kwargs['image_paths'], [])
            self.assertEqual(payload['style_ref_path'], '')
            self.assertNotIn('generation_clauses', payload)
            self.assertNotIn('style_text', payload)
            self.assertFalse(payload['skip_identity_refine'])
            self.assertEqual(steps['repaint']['reference_mode'], 'none')
            self.assertFalse(steps['tone']['enabled'])

    def test_old_declined_plan_researched_again_without_resending_original(self):
        with tempfile.TemporaryDirectory() as directory:
            cp = GenerationCheckpoint(str(Path(directory) / 'checkpoint.json'))
            cp.data['first_safe_review'] = {'blocked': True, 'plan': {'retry_allowed': False}}
            plan = {'version': 2, 'retry_allowed': True, 'changes': ['original covered outfit'],
                    'reference_policy': 'none', 'prompt': 'ordinary fashion illustration'}
            gen = Mock(return_value=[])
            with patch('utils.first_image_review.review_request', return_value=plan) as review:
                generate_first_image(gen, prompt='original', checkpoint=cp)
            review.assert_called_once()
            gen.assert_called_once()
            self.assertEqual(gen.call_args.kwargs['prompt'], plan['prompt'])
