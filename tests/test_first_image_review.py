import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from utils.first_image_review import generate_first_image, is_moderation_block
from utils.generation_checkpoint import GenerationCheckpoint

BLOCK = {"saved_files": [], "server_response_raw": {"error": {"code": "moderation_blocked"}}}

class FirstReviewTests(unittest.TestCase):
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
            with patch('utils.first_image_review.review_request', return_value={"retry_allowed": False}):
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
            with patch('utils.first_image_review.review_request', return_value={"retry_allowed": True, "changes": ['remove invention'], "prompt": 'safe revised'}):
                generate_first_image(gen, prompt='original', checkpoint=cp)
                generate_first_image(gen, prompt='original', checkpoint=cp)
            self.assertEqual(gen.call_count, 2)
