from utils.identity_audit import (build_identity_correction_prompt, identity_gate_action,
                                  normalize_audit, should_correct)


def test_normalize_audit_drops_low_confidence_and_builds_targeted_prompt():
    audit = normalize_audit({"mismatch": True, "severity": "major", "confidence": .91,
        "stable_anchors": ["warm brown hair", "brown eyes"],
        "differences": [
            {"feature": "Eye Color", "expected": "brown", "observed": "green",
             "correction": "Restore both irises to brown.", "confidence": .96},
            {"feature": "hair", "expected": "black", "observed": "navy",
             "correction": "", "confidence": .4}]})
    assert audit["mismatch"] is True and len(audit["differences"]) == 1
    assert should_correct(audit)
    prompt = build_identity_correction_prompt(audit)
    assert "Restore both irises to brown" in prompt
    assert "warm brown hair" in prompt and "PASS 1 OF 2" in prompt
    assert "changing ONLY" in prompt and "Preserve the exact face geometry" in prompt


def test_empty_or_low_confidence_audit_does_not_trigger_edit():
    audit = normalize_audit({"mismatch": True, "differences": [
        {"feature": "eyes", "expected": "blue", "observed": "green", "confidence": .4}]})
    assert audit["mismatch"] is False
    assert should_correct(audit) is False
    assert build_identity_correction_prompt(audit) == ""


def test_identity_gate_corrects_broad_major_drift_without_fallback():
    broad = {"mismatch": True, "severity": "major",
             "differences": [{"confidence": .99}] * 3}
    local = {"mismatch": True, "severity": "major",
             "differences": [{"confidence": .99}]}
    assert identity_gate_action(broad) == "correct"
    assert identity_gate_action(local) == "correct"
    assert identity_gate_action({"audit_error": "400"}) == "review"


def test_structural_identity_difference_gets_local_geometry_exception():
    audit = normalize_audit({"mismatch": True, "severity": "major", "confidence": .99,
        "stable_anchors": ["short layered skirt"],
        "differences": [{"feature": "skirt_length", "expected": "short", "observed": "floor length",
                         "correction": "Shorten the skirt.", "confidence": .99}]})
    prompt = build_identity_correction_prompt(audit, iteration=2, max_iterations=2)
    assert "STRUCTURAL EXCEPTION" in prompt
    assert "obsolete trailing fabric" in prompt
    assert "PASS 2 OF 2" in prompt
