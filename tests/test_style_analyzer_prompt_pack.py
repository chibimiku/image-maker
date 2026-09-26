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
        "negative_rules": ["No copied identity."],
        "usage_profiles": {"portrait_focus": "Prioritize eyes."},
        "evidence_summary": {"variable_traits": ["Background density varies."]},
    })
    for expected in ("MASTER", "GEMINI", "Palette: pale", "Repair lines.",
                     "Use sharp eyes.", "No copied identity.", "portrait_focus",
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
    }, "MASTER")
    assert package["gpt_image_prompt_valid"] is True
    assert package["gemini_repaint_clauses"] == ["Keep primary contours continuous."]
    assert package["style_entry"] == {
        "prompt": "FULL GEMINI",
        "prompt_gpt": package["gpt_image_prompt"],
        "repaint_clauses": ["Keep primary contours continuous."],
        "face_hair_clauses": ["Use grouped locks."],
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
