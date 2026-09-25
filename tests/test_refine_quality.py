from utils.refine_quality import (build_quality_correction_prompt, normalize_quality_audit,
                                  should_refine_quality)


def test_quality_audit_keeps_only_confident_complete_findings():
    value = normalize_quality_audit({
        "needs_refine": True,
        "severity": "major",
        "structural_issues": [
            {"region": "right hand", "observed": "fused fingers", "repair": "separate five fingers", "confidence": .9},
            {"region": "shoe", "observed": "", "repair": "fix", "confidence": .9},
        ],
        "background_drift": [{"region": "wall", "original": "yellow wall", "candidate": "white wall",
                               "repair": "restore yellow wall", "confidence": .8}],
    })
    assert value["needs_refine"] is True
    assert len(value["structural_issues"]) == 1
    assert len(value["background_drift"]) == 1


def test_quality_correction_prompt_assigns_three_image_roles():
    prompt = build_quality_correction_prompt({
        "structural_issues": [{"region": "left hand", "repair": "redraw readable fingers"}],
        "background_drift": [{"region": "right wall", "original": "warm yellow wall",
                               "repair": "match Image 2 exactly"}],
        "style_gaps": [{"aspect": "line character", "repair": "use tapered salmon contours"}],
        "protected_features": ["face", "dress design"],
    })
    assert "Image 2 = ORIGINAL GPT FIRST-PASS IMAGE" in prompt
    assert "Image 3" in prompt
    assert "redraw readable fingers" in prompt
    assert "warm yellow wall" in prompt
    assert "face" in prompt
    assert "When Image 3 is absent" in prompt


def test_quality_gate_requires_confident_material_finding():
    assert should_refine_quality({"needs_refine": True,
                                  "line_issues": [{"confidence": .8}]}) is True
    assert should_refine_quality({"needs_refine": True,
                                  "line_issues": [{"confidence": .7}]}) is False
    assert should_refine_quality({"needs_refine": False,
                                  "style_gaps": [{"confidence": .95}]}) is False
