# -*- coding: utf-8 -*-
from modules.image_analysis.style_analyzer import (
    STYLE_ITER_VARIANTS_PROMPT_FILE,
    format_style_prompt_package,
    get_style_analyzer_missing_prompt_files,
)


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
