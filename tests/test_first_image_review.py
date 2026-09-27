import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from utils.first_image_review import generate_first_image, is_moderation_block, apply_safe_plan
from utils.generation_checkpoint import GenerationCheckpoint

BLOCK = {"saved_files": [], "server_response_raw": {"error": {"code": "moderation_blocked"}}}

class FirstReviewTests(unittest.TestCase):
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
