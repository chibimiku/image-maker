from utils.refine_quality import (build_quality_correction_prompt, normalize_quality_audit,
                                  should_refine_quality)


def test_quality_correction_allows_explicit_proportion_repair():
    prompt = build_quality_correction_prompt({
        "structural_issues": [{"region": "legs", "repair": "Shorten both legs to restore 7-head balance."}],
        "line_issues": [], "background_drift": [], "style_gaps": [], "protected_features": []})
    assert "Shorten both legs" in prompt
    assert "proportions or limb lengths" in prompt


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


def test_repair_guard_does_not_transfer_palette_or_blank_space():
    from utils.refine_quality import repair_style_guard
    guard = repair_style_guard({"prompt": "Palette: pink hair\nLighting: soft\n"
                               "Brushwork: layered watercolor\nEdges: fine ink\nTexture: paper\n"
                               "Composition density: half blank\nDetail level: sparse\nAvoid: flat fills\n\n"
                               "brown-haired girl seated in an ivory dress"})
    assert "layered watercolor" in guard
    assert "pink hair" not in guard and "half blank" not in guard
    assert "brown-haired girl" not in guard


def test_anatomy_audit_handles_contradictory_approval_and_keeps_inventory(monkeypatch, tmp_path):
    import json
    from PIL import Image
    from utils import refine_quality as quality
    candidate = str(tmp_path / "two.png")
    Image.new("RGB", (900, 600), "white").save(candidate)
    captured = []
    def model(*args, **kwargs):
        assert "correct TOTAL number of legs" in args[3]
        assert "occluded or off-frame" in args[3]
        assert len(kwargs["image_paths"]) == 4
        assert "continuous RIGHT lower-body crop" in args[3]
        captured.extend(kwargs["image_paths"])
        return json.dumps({"needs_refine": False,
            "structural_issues": [{"region": "right black-dressed character's third leg",
                "observed": "three pelvis-to-foot chains", "repair": "remove duplicate chain", "confidence": .95}],
            "character_limb_inventory": [{"owner": "right black-dressed", "legs": "three visible chains"}]})
    monkeypatch.setattr(quality, "call_text_model", model)
    audit = quality.audit_hand_quality(candidate, text_cfg={"base_url": "test", "api_key": "", "model": "test"})
    assert quality.should_refine_quality(audit)
    assert audit["character_limb_inventory"][0]["owner"] == "right black-dressed"
    assert all(not __import__("os").path.exists(path) for path in captured)


def test_final_audit_detects_style_defect_despite_false_flag(monkeypatch, tmp_path):
    import json
    from PIL import Image
    from utils import refine_quality as quality
    candidate = str(tmp_path / "candidate.png")
    Image.new("RGB", (900, 600), "white").save(candidate)
    def model(*args, **kwargs):
        assert len(kwargs["image_paths"]) == 4
        assert "visible foot to its owner" in args[3]
        assert "raw global mean brightness" in args[3]
        assert "layered gradients" in args[4]
        return json.dumps({"needs_refine": False, "structural_issues": [], "line_issues": [],
                           "background_drift": [], "style_gaps": [{"aspect": "shading",
                           "candidate": "flat vector", "target": "layered gradients",
                           "repair": "restore chromatic gradients", "confidence": .95}]})
    monkeypatch.setattr(quality, "call_text_model", model)
    audit = quality.audit_refine_quality(candidate, candidate, candidate, final_review=True,
        style_targets="layered gradients", text_cfg={"base_url": "test", "api_key": "", "model": "test"})
    assert quality.should_refine_quality(audit)


def test_quality_audit_rejects_missing_lists(monkeypatch, tmp_path):
    import pytest
    from PIL import Image
    from utils import refine_quality as quality
    candidate = str(tmp_path / "candidate.png")
    Image.new("RGB", (900, 600), "white").save(candidate)
    monkeypatch.setattr(quality, "call_text_model", lambda *a, **kw: '{}')
    with pytest.raises(ValueError, match="缺陷列表"):
        quality.audit_refine_quality(candidate, candidate, candidate,
            text_cfg={"base_url": "test", "api_key": "", "model": "test"})


def test_anatomy_uncertain_visible_segment_requires_review(monkeypatch, tmp_path):
    import json
    from PIL import Image
    from utils import refine_quality as quality
    candidate = str(tmp_path / "candidate.png")
    Image.new("RGB", (900, 600), "white").save(candidate)
    monkeypatch.setattr(quality, "call_text_model", lambda *a, **kw: json.dumps({
        "structural_issues": [], "ownership_uncertain": [{"region": "far right thigh", "reason": "unknown owner"}]}))
    audit = quality.audit_hand_quality(candidate, text_cfg={"base_url": "test", "api_key": "", "model": "test"})
    assert audit["needs_review"]
    assert not quality.should_refine_quality(audit)  # Do not guess a repair.
