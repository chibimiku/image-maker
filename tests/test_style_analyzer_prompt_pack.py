# -*- coding: utf-8 -*-
from modules.image_analysis.style_analyzer import (
    STYLE_ITER_VARIANTS_PROMPT_FILE,
    StyleIterativeWorkerThread,
    _crop_image_regions,
    format_style_prompt_package,
    get_style_analyzer_missing_prompt_files,
    normalize_style_prompt_package,
)
from PIL import Image


def test_style_analyzer_has_channel_variant_prompt_file():
    assert STYLE_ITER_VARIANTS_PROMPT_FILE not in get_style_analyzer_missing_prompt_files()


def test_format_style_prompt_package_exposes_all_channels():
    text = format_style_prompt_package("MASTER", {
        "gemini_full_prompt": "GEMINI",
        "gpt_image_prompt": "Palette: pale",
        "gemini_repaint_clauses": ["Repair lines."],
        "face_hair_clauses": ["Use sharp eyes."],
        "optional_motifs": ["Sparse butterflies near the frame edge."],
        "negative_rules": ["No copied identity."],
        "usage_profiles": {"portrait_focus": "Prioritize eyes."},
        "evidence_summary": {"variable_traits": ["Background density varies."]},
    })
    for expected in ("MASTER", "GEMINI", "Palette: pale", "Repair lines.",
                     "Use sharp eyes.", "Sparse butterflies", "No copied identity.", "portrait_focus",
                     "Background density varies."):
        assert expected in text


def test_normalize_prompt_package_builds_app_ready_style_entry():
    package = normalize_style_prompt_package({
        "gemini_full_prompt": "FULL GEMINI",
        "gpt_image_prompt": (
            "Palette: muted jewel tones\nLighting: soft side light\n"
            "Brushwork: layered opaque strokes\nEdges: tapered coloured contours\n"
            "Texture: fine paper grain\nComposition density: balanced negative space\n"
            "Detail level: selective focal detail\nAvoid: global haze and copied content"
        ),
        "gemini_repaint_clauses": "Keep primary contours continuous.",
        "face_hair_clauses": ["Use grouped locks."],
        "optional_motifs": ["Sparse butterflies at the frame edge."],
    }, "MASTER")
    assert package["gpt_image_prompt_valid"] is True
    assert package["gemini_repaint_clauses"] == ["Keep primary contours continuous."]
    assert package["style_entry"] == {
        "prompt": "FULL GEMINI",
        "prompt_gpt": package["gpt_image_prompt"],
        "repaint_clauses": ["Keep primary contours continuous."],
        "face_hair_clauses": ["Use grouped locks."],
        "motif_clauses": ["Sparse butterflies at the frame edge."],
        "motif_enabled": True,
        "enabled": True,
    }


def test_normalize_prompt_package_does_not_export_invalid_gpt_prompt():
    package = normalize_style_prompt_package({"gpt_image_prompt": "Palette: blue"}, "MASTER")
    assert package["gpt_image_prompt_valid"] is False
    assert package["style_entry"]["prompt_gpt"] == ""
    assert package["style_entry"]["prompt"] == "MASTER"


def test_state_rebuild_preserves_test_images_prompt_pack_and_created_at(tmp_path):
    worker = StyleIterativeWorkerThread(
        image_paths=[], api_key="key", base_url="https://example.invalid/v1",
        model_name="model", output_dir=str(tmp_path), file_prefix="style")
    previous = {
        "created_at": "2026-09-20 12:00:00",
        "test_images": {"round_1": {"generated_files": ["one.png"]}},
        "prompt_variants": {"gemini_full_prompt": "FULL"},
    }
    state = worker._build_state(2, [{"step": 1}], previous)
    assert state["created_at"] == previous["created_at"]
    assert state["test_images"] == previous["test_images"]
    assert state["prompt_variants"] == previous["prompt_variants"]


def test_crop_names_do_not_collide_for_same_basename_in_different_dirs(tmp_path):
    first = tmp_path / "a" / "same.png"
    second = tmp_path / "b" / "same.png"
    first.parent.mkdir()
    second.parent.mkdir()
    Image.new("RGB", (100, 140), "white").save(first)
    Image.new("RGB", (100, 140), "black").save(second)
    out = tmp_path / "crops"
    out.mkdir()
    first_crops = _crop_image_regions(str(first), str(out), crop_count=1)
    second_crops = _crop_image_regions(str(second), str(out), crop_count=1)
    assert first_crops[0] != second_crops[0]
    assert len(list(out.glob("*.jpg"))) == 2


def test_round_test_generation_returns_gemini_gpt_and_repaint(monkeypatch, tmp_path):
    import modules.image_analysis.style_analyzer as sa
    import utils.analysis_gen as ag
    import utils.styles as styles

    ref = tmp_path / "ref.png"
    Image.new("RGB", (64, 96), "white").save(ref)
    gemini = tmp_path / "gemini.png"
    first = tmp_path / "gpt.png"
    repainted = tmp_path / "gpt-repainted.png"

    monkeypatch.setattr(sa, "get_api_config", lambda api_type: {
        "model": "gemini-model" if api_type == "aigc2d" else "gpt-model"})
    calls = {}
    monkeypatch.setattr(sa, "generate_image_aigc2d", lambda **kwargs: calls.setdefault("gemini", kwargs) and [str(gemini)])
    monkeypatch.setattr(sa, "generate_image_aigc2d_gpt", lambda **kwargs: calls.setdefault("gpt", kwargs) and [str(first)])
    monkeypatch.setattr(styles, "compose_style_prompt", lambda *args, **kwargs: "gemini prompt")
    monkeypatch.setattr(ag, "build_gpt_image_request", lambda *args, **kwargs: {
        "prompt": "gpt prompt", "image_paths": [str(ref)]})
    monkeypatch.setattr(ag, "run_gpt_image_pipeline", lambda *args, **kwargs: [str(repainted)])

    worker = StyleIterativeWorkerThread(
        image_paths=[str(ref)], api_key="key", base_url="https://example.invalid/v1",
        model_name="text", output_dir=str(tmp_path), enable_test_gen=True,
        test_prompt="fixed subject", test_style_ref_path=str(ref))
    package = {
        "gemini_full_prompt": "full",
        "gpt_image_prompt": (
            "Palette: muted jewel tones\nLighting: soft side light\n"
            "Brushwork: layered opaque strokes\nEdges: tapered coloured contours\n"
            "Texture: fine paper grain\nComposition density: balanced negative space\n"
            "Detail level: selective focal detail\nAvoid: global haze and copied content"),
        "gemini_repaint_clauses": ["Keep contours continuous."],
    }
    result = worker._generate_test_image("MASTER", 1, "unused.json", package)
    assert result == {
        "gemini_direct": [str(gemini)],
        "gpt_first_pass": [str(first)],
        "gpt_repainted": [str(repainted)],
    }
    assert calls["gemini"]["aspect_ratio"] == "2:3"
    assert calls["gpt"]["aspect_ratio"] == "2:3"
