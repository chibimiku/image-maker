# -*- coding: utf-8 -*-
"""gpt-image 短版画风说明（prompt_gpt）的回归用例。

覆盖：
- prompt_gpt 的读写与 normalize_style_entry / build_style_entry 的向后兼容
- gpt-image 通道的取用优先级（prompt_gpt > prompt_compressed > prompt）
- 字段版转换规则的校验（缺字段 / 太长 / 主体词泄漏 / 非字段行）
- 宽松修复（超长时的字段级收口）
- build_ref_gen_params 在 gpt-image 通道下确实改用 prompt_gpt
"""
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.style_gpt import (  # noqa: E402
    FIELD_KEYS,
    build_conversion_prompts,
    compose_prompt_gpt,
    format_prompt_gpt,
    is_gpt_image_api_type,
    parse_fields,
    repair_prompt_gpt,
    repair_prompt_gpt_lenient,
    resolve_style_prompt,
    style_prompt_gpt,
    validate_prompt_gpt,
)
from utils.styles import (  # noqa: E402
    MODE_OFF,
    build_ref_gen_params,
    build_style_entry,
    normalize_style_entry,
    style_prompt_compressed,
)

GOOD_FIELDS = {
    "Palette": "pale pastel, very low saturation, clean white, cool shadows",
    "Lighting": "bright high-key, soft even ambience, minimal contrast",
    "Brushwork": "transparent watercolour washes, delicate fine lineart",
    "Edges": "soft thin contours, often lost where washes fade",
    "Texture": "subtle paper grain, tiny floating droplets and bubbles",
    "Composition density": "low, generous empty space around the subject",
    "Detail level": "restrained, crisp accents only at focal points",
    "Avoid": "heavy shading, thick outlines, saturated dark colour",
}
GOOD_TEXT = format_prompt_gpt(GOOD_FIELDS)


def test_api_type_detection():
    assert is_gpt_image_api_type("aigc-2d-gpt")
    assert is_gpt_image_api_type("AIGC_2D_GPT")
    assert is_gpt_image_api_type("gpt-image-2")
    assert not is_gpt_image_api_type("aigc2d")
    assert not is_gpt_image_api_type("")
    assert not is_gpt_image_api_type(None)


def test_normalize_style_entry_reads_prompt_gpt():
    entry = normalize_style_entry({"prompt": "full", "prompt_gpt": "short"})
    assert entry["prompt"] == "full"
    assert entry["prompt_gpt"] == "short"
    # 别名与旧格式
    assert normalize_style_entry({"prompt": "x", "prompt_short": "y"})["prompt_gpt"] == "y"
    assert normalize_style_entry("just a string")["prompt_gpt"] == ""
    assert normalize_style_entry(None)["prompt_gpt"] == ""


def test_build_style_entry_roundtrip():
    entry = build_style_entry("full", ref_image="r.png", prompt_compressed="c", prompt_gpt="g")
    assert entry == {"prompt": "full", "ref_image": "r.png", "prompt_compressed": "c", "prompt_gpt": "g"}
    # 空字段省略，保持旧文件形态
    assert build_style_entry("full") == {"prompt": "full"}


def test_style_prompt_gpt_and_resolve_priority():
    styles = {"tid": {"prompt": "FULL", "prompt_compressed": "COMP", "prompt_gpt": "GPT"}}
    assert style_prompt_gpt(styles, "tid") == "GPT"
    # gpt-image 通道：prompt_gpt 优先
    assert resolve_style_prompt(styles, "tid", api_type="aigc-2d-gpt") == "GPT"
    # 其它通道（gemini）：仍然用全量
    assert resolve_style_prompt(styles, "tid", api_type="aigc2d") == "FULL"
    # 缺 prompt_gpt 时 gpt 通道退回压缩版，再退回全量
    no_gpt = {"tid": {"prompt": "FULL", "prompt_compressed": "COMP"}}
    assert resolve_style_prompt(no_gpt, "tid", api_type="aigc-2d-gpt") == "COMP"
    only_full = {"tid": {"prompt": "FULL"}}
    assert resolve_style_prompt(only_full, "tid", api_type="aigc-2d-gpt") == "FULL"
    assert resolve_style_prompt({}, "missing", api_type="aigc-2d-gpt") == ""


def test_compose_prompt_gpt_contains_subject_and_role_split():
    text = compose_prompt_gpt(GOOD_TEXT, "A girl stands in shallow water.")
    assert GOOD_TEXT in text
    assert "A girl stands in shallow water." in text
    assert "rendering only" in text
    assert len(text) > len(GOOD_TEXT)


def test_parse_and_format_fields_are_stable():
    assert parse_fields(GOOD_TEXT) == GOOD_FIELDS
    assert format_prompt_gpt(GOOD_FIELDS) == GOOD_TEXT
    # 大小写与全角冒号容错
    loose = "palette：warm pastel\nLighting: soft key"
    parsed = parse_fields(loose)
    assert parsed["Palette"] == "warm pastel"
    assert parsed["Lighting"] == "soft key"


def test_validate_accepts_good_text():
    ok, errors = validate_prompt_gpt(GOOD_TEXT)
    assert ok, errors


@pytest.mark.parametrize(
    "broken,expect",
    [
        (GOOD_TEXT.replace("Edges: soft thin contours, often lost where washes fade\n", ""), "缺字段"),
        (GOOD_TEXT.replace("Palette: pale pastel", "Palette: the girl with pink hair has", 1), "主体"),
        ("Some prose without fields at all", "缺字段"),
    ],
)
def test_validate_rejects_bad_text(broken, expect):
    ok, errors = validate_prompt_gpt(broken)
    assert not ok, errors
    assert any(expect in e for e in errors), errors


def test_duplicate_field_keeps_last_value():
    duplicated = GOOD_TEXT + "\nPalette: warm cream, mid saturation"
    ok, errors = validate_prompt_gpt(duplicated)
    assert ok, errors
    assert parse_fields(duplicated)["Palette"] == "warm cream, mid saturation"
    # 重排后只剩一份（最后出现的值）
    assert repair_prompt_gpt(duplicated).count("Palette:") == 1


def test_validate_rejects_subject_words():
    bad = format_prompt_gpt({**GOOD_FIELDS, "Palette": "pink hair, glossy eyes, saturated"})
    ok, errors = validate_prompt_gpt(bad)
    assert not ok
    assert any("主体" in e for e in errors)


def test_validate_rejects_too_long():
    long_text = format_prompt_gpt({k: "very long value, " * 12 for k in FIELD_KEYS})
    ok, errors = validate_prompt_gpt(long_text)
    assert not ok
    assert any("太长" in e for e in errors)


def test_repair_lenient_shortens_total():
    long_text = format_prompt_gpt({k: "very long value, " * 12 for k in FIELD_KEYS})
    repaired = repair_prompt_gpt_lenient(long_text)
    ok, errors = validate_prompt_gpt(repaired)
    assert ok, errors
    assert len(repaired) < len(long_text)


def test_repair_keeps_field_order():
    shuffled = "Lighting: soft key\nPalette: pale pastel\n" + "\n".join(
        f"{k}: value {i}" for i, k in enumerate(FIELD_KEYS[2:])
    )
    repaired = repair_prompt_gpt(shuffled)
    keys = [line.split(":", 1)[0] for line in repaired.splitlines()]
    assert keys == list(FIELD_KEYS)


def test_build_ref_gen_params_uses_prompt_gpt_for_gpt_channel(tmp_path):
    ref = tmp_path / "style.png"
    ref.write_bytes(b"fake")
    styles = {
        "tid": {
            "prompt": "FULL SPEC",
            "prompt_compressed": "COMPRESSED SPEC",
            "prompt_gpt": GOOD_TEXT,
            "ref_image": str(ref),
        }
    }
    head_gpt, post_gpt, refs = build_ref_gen_params(styles, "tid", MODE_OFF, api_type="aigc-2d-gpt")
    assert head_gpt == GOOD_TEXT
    assert post_gpt == ""
    assert refs == []

    head, _post, _refs = build_ref_gen_params(styles, "tid", MODE_OFF, api_type="aigc2d")
    assert head == "FULL SPEC"

    # priority 模式下 gpt 通道也用短版（并不再叠加压缩版）
    head_prio, _post, refs_prio = build_ref_gen_params(styles, "tid", "priority", api_type="aigc-2d-gpt")
    assert GOOD_TEXT in head_prio
    assert "COMPRESSED SPEC" not in head_prio
    assert refs_prio == [str(ref)]


def test_conversion_prompts_mention_8_fields():
    system, user = build_conversion_prompts("tid", "long spec text", has_image=True)
    for key in FIELD_KEYS:
        assert key in system
    assert "long spec text" in user
    assert "AUTHORITATIVE" in user
    assert "<subject>" not in user
