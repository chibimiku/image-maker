"""Practical delivery gates: minor warnings do not spend another image request."""
from utils import first_image_audit as fa
import unittest


def report(severity="minor", confidence=0.9, verdict="fail", kind="content"):
    return fa.normalize_audit({
        "content_ok": kind != "content", "style_ok": kind != "style",
        "copied_from_reference": False, "verdict": verdict,
        "style_failure_kind": "incompatible_rendering_family"
                              if kind == "style" and severity == "major" else "none",
        "content_issues": [{"element": "side_braids", "observed": "one braid",
                            "severity": severity, "confidence": confidence}]
                          if kind == "content" else [],
        "style_issues": [{"element": "rendering", "observed": "less sparkle",
                          "severity": severity, "confidence": confidence}]
                        if kind == "style" else [],
    })


def test_minor_fail_label_is_warning_not_redraw_or_suspension():
    audit = report()
    assert audit["verdict"] == "uncertain"
    assert len(audit["warnings"]) == 1
    assert not fa.should_retry(audit)[0]
    assert not fa.suspended([{"audit": audit}])


def test_low_confidence_major_fail_label_does_not_bypass_gate():
    audit = report(severity="major", confidence=0.4)
    assert not fa.should_retry(audit)[0]
    assert not fa.suspended([{"audit": audit}])


def test_minor_style_limitations_can_pass_with_warnings():
    audit = report(verdict="pass", kind="style")
    audit["content_ok"] = audit["style_ok"] = True
    assert audit["verdict"] == "pass"
    assert len(audit["warnings"]) == 1
    assert not fa.should_retry(audit)[0]


def test_major_content_and_style_still_block_even_if_flags_claim_ok():
    for kind in ("content", "style"):
        audit = report(severity="major", kind=kind)
        audit["content_ok"] = audit["style_ok"] = True
        assert fa.should_retry(audit)[0]
        assert fa.suspended([{"audit": audit}])


def test_minor_warning_delivers_after_one_generation():
    calls = []
    outcome = fa.generate_with_audit(
        lambda prompt, index: calls.append(index) or ["existing.jpg"],
        prompt="content", audit=lambda *args: report(), max_attempts=2)
    assert calls == [1]
    assert outcome["published"] == "existing.jpg"
    assert not outcome["suspended"]


def test_template_defines_minor_pass_and_major_style_evidence():
    from utils.prompt_loader import read_prompt_file
    prompt = read_prompt_file(fa.AUDIT_SYSTEM_FILE)
    assert "minor warnings are allowed" in prompt
    assert "Splitting these" in prompt
    assert "missing secondary braid" in prompt


def test_style_major_label_without_incompatible_family_only_warns():
    audit = fa.normalize_audit({
        "content_ok": True, "style_ok": False, "verdict": "fail",
        "copied_from_reference": False, "style_failure_kind": "none",
        "style_issues": [{"element": "line", "observed": "stronger contours",
                          "severity": "major", "confidence": 0.95},
                         {"element": "texture", "observed": "smoother shading",
                          "severity": "major", "confidence": 0.95}]})
    assert not fa.should_retry(audit)[0]
    assert not fa.suspended([{"audit": audit}])
    assert len(audit["warnings"]) == 2


def test_smile_opening_cannot_be_promoted_to_major_but_eye_colour_still_blocks():
    payload = {"content_ok": False, "style_ok": True,
               "copied_from_reference": False, "verdict": "fail",
               "content_issues": [{"element": "facial_expression",
                                   "expected": "gentle open-mouth smile",
                                   "observed": "gentle closed-mouth smile",
                                   "severity": "major", "confidence": 0.95}]}
    audit = fa.normalize_audit(payload)
    assert not fa.should_retry(audit)[0]
    payload["content_issues"].append({"element": "eye_colour", "expected": "black",
                                      "observed": "light purple", "severity": "major",
                                      "confidence": 0.95})
    audit = fa.normalize_audit(payload)
    assert len(audit["major_issues"]) == 1
    assert fa.should_retry(audit)[0]


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name, fn in globals().items()
                              if name.startswith("test_") and callable(fn))
