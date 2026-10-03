"""Quality evidence must survive errors; only minor rendering can be waived."""
import json

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from utils.refine_quality import (final_quality_decision, quality_failure_details,
                                  record_quality_audit, review_final_candidate)


EMPTY_FINDINGS = {"structural_issues": [], "line_issues": [], "background_drift": [], "style_gaps": []}


def finding(key="style_gaps", severity="minor"):
    """schema 完整的审计结论（第四轮 P0.2 起门禁要求结论字段齐全）。

    ``finding()`` 原来只给一个缺陷集合；真实的 ``normalize_quality_audit`` 会输出四个集合，
    再加归属结论。这里补齐成真实口径，避免用例靠「缺字段默认通过」才成立。
    """
    item = {"needs_refine": True, "severity": severity, "confidence": .9,
            "ownership_uncertain": [], "needs_review": False, **EMPTY_FINDINGS}
    item[key] = [{"region": "skirt", "aspect": "shading", "observed": "pale contours",
                  "candidate": "too glossy", "repair": "reduce bloom", "confidence": .86}]
    return item


def complete_identity(mismatch=False, sha="same"):
    """身份审计的 schema 完整版：合法 boolean mismatch + 合法严重度 + 差异集合。"""
    return {"mismatch": bool(mismatch), "severity": "major" if mismatch else "none",
            "confidence": .9, "differences": [] if not mismatch else [
                {"feature": "hair_colour", "expected": "black", "observed": "blue",
                 "correction": "restore black", "confidence": .95}],
            "stable_anchors": [], "summary": "", "candidate_sha256": sha}


def complete_anatomy(sha="same"):
    return {"needs_refine": False, "severity": "none", "ownership_uncertain": [],
            "needs_review": False, "candidate_sha256": sha, **EMPTY_FINDINGS}


class QualityReportingTests(unittest.TestCase):
    def test_only_minor_rendering_is_warning(self):
        for key in ['style_gaps', 'line_issues']:
            with self.subTest(key=key):
                self.assertEqual(final_quality_decision(finding(key))['action'], 'accept_with_warning')
                for severity in ['major', 'unknown']:
                    self.assertEqual(final_quality_decision(finding(key, severity))['action'], 'repair')

    def test_structure_and_background_remain_blocking_even_when_minor(self):
        for key in ['structural_issues', 'background_drift']:
            self.assertEqual(final_quality_decision(finding(key))['action'], 'repair')

    def test_uncertainty_and_failed_audit_cannot_be_waived(self):
        for extra in [{'needs_review': True}, {'audit_error': '503'},
                      {'ownership_uncertain': [{'region': 'leg', 'reason': 'unknown owner'}]}]:
            self.assertEqual(final_quality_decision({**finding(), **extra})['action'], 'review')

    def test_record_contains_evidence_warning_and_readable_sidecar(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'audit.json'
            logs = []
            record_quality_audit(finding(), str(path), logs.append, final=True)
            data = json.loads(path.read_text(encoding='utf-8'))
            self.assertTrue(data['needs_refine'])  # Preserve original model evidence.
            self.assertEqual(data['gate_decision']['action'], 'accept_with_warning')
            report = path.with_suffix('.txt').read_text(encoding='utf-8')
            for text in ['skirt', '0.86', 'reduce bloom']:
                self.assertIn(text, report)
            self.assertIn(str(path), logs[0])

    def test_minor_final_candidate_does_not_pay_for_another_repaint(self):
        with tempfile.TemporaryDirectory() as folder:
            source = str(Path(folder) / 'source.png')
            Image.new('RGB', (32, 32)).save(source)
            with patch('modules.others.api_backend.generate_image_repaint') as repaint:
                selected, audit = review_final_candidate(source, lambda p: finding(), folder)
                repaint.assert_not_called()
            self.assertEqual(selected, source)
            self.assertEqual(audit['gate_decision']['action'], 'accept_with_warning')

    def test_exhausted_repair_error_names_defect_and_audit_path(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('utils.gpt_image_optimize.load_config', return_value={'final_candidate_repair': {'max_repairs': 0}}):
                with self.assertRaises(RuntimeError) as exc:
                    review_final_candidate('unused.png', lambda p: finding('structural_issues'), folder)
            for text in ['skirt', 'pale contours', 'final-quality-audit-0.json']:
                self.assertIn(text, str(exc.exception))

    def test_bound_final_warning_is_accepted_by_shared_gate(self):
        from utils.gate_status import _verdict
        audit = {**finding(), 'candidate_sha256': 'same'}
        verdict = _verdict(audit, 'final_review', 'same')
        self.assertEqual(verdict['verdict'], 'pass')
        self.assertTrue(verdict['warning'])
        self.assertEqual(_verdict(audit, 'final_review', 'changed')['verdict'], 'stale')
        self.assertEqual(_verdict(audit, 'anatomy', 'same')['verdict'], 'fail')

    def test_details_keep_background_before_and_after(self):
        audit = {'background_drift': [{'region': 'wall', 'original': 'yellow',
                 'candidate': 'white', 'repair': 'restore yellow', 'confidence': .95}]}
        details = quality_failure_details(audit)
        for text in ['wall', 'white', 'restore yellow']:
            self.assertIn(text, details)

    def test_shared_quality_gate_uses_reaudit_and_keeps_upstream_failure(self):
        from utils.gate_status import _verdict, evaluate_final_gate
        quality = {'before': finding('structural_issues', 'major'), 'after': finding()}
        self.assertEqual(_verdict(quality, 'quality', 'same')['verdict'], 'pass')
        self.assertEqual(_verdict({'after': finding('background_drift')}, 'quality', 'same')['verdict'], 'fail')
        gate = evaluate_final_gate(final_image='image.png', final_sha256='same', quality=quality,
            identity=complete_identity(), anatomy=complete_anatomy(),
            final_review={**finding(), 'candidate_sha256': 'same'})
        self.assertEqual(gate['status'], 'complete')
        gate = evaluate_final_gate(final_image='image.png', final_sha256='same', quality=quality,
            quality_audit_error='major drift', upstream_failed=True,
            identity=complete_identity(), anatomy=complete_anatomy(),
            final_review={**finding(), 'candidate_sha256': 'same'})
        self.assertEqual(gate['status'], 'review_required')

    def test_queue_tooltip_retains_failure_evidence_and_checkpoint(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from PyQt6.QtCore import Qt
        from modules.image_analysis.single_analyzer import SingleAnalyzerWidget
        item = Mock()
        item.data.return_value = 'task'
        history_list = Mock()
        history_list.count.return_value = 1
        history_list.item.return_value = item
        widget = SimpleNamespace(_analysis_history={'task': {'pipeline_error': 'skirt: disconnected hems',
            'generation_checkpoints': ['checkpoint.json']}}, history_list=history_list,
            _refresh_history_status_text=lambda record: '失败', _status_to_color=lambda status: Qt.GlobalColor.red,
            log_msg=Mock())
        SingleAnalyzerWidget._refresh_history_item(widget, 'task')
        tooltip = item.setToolTip.call_args.args[0]
        self.assertIn('disconnected hems', tooltip)
        self.assertIn('checkpoint.json', tooltip)
