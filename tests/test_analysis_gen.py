# -*- coding: utf-8 -*-
"""「分析→gpt-image」请求组装与工序流水线（utils/analysis_gen.py）。"""
import json
import os
import sys

import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from utils import analysis_gen as ag  # noqa: E402
from utils import post_process as pp  # noqa: E402


def test_resolve_content_prefers_short_field():
    data = {"gpt_image_prompt_short": "SHORT", "gpt_image_prompt": "FULL",
            "english_description": "LONG DESC"}
    assert ag.resolve_content_text(data, "short") == "SHORT"
    assert ag.resolve_content_text(data, "full") == "FULL"


def test_generation_manifest_keeps_reproducible_request_without_api_secrets(tmp_path):
    source = tmp_path / "first.png"
    source.write_bytes(b"first image")
    ref = tmp_path / "reference.png"
    ref.write_bytes(b"style image")
    path = ag.save_generation_manifest(str(source), {
        "prompt": "exact prompt", "style_name": "style", "image_paths": [str(ref)],
        "api_key": "secret-not-for-manifest"}, model="test-model", size="1024x1536",
        quality="high", mode="generate", steps={"repaint": {"reference_mode": "none"}},
        firmware="exact firmware", outputs=["final.png"])
    body = open(path, encoding="utf-8").read()
    snapshot = json.loads(body)
    assert snapshot["prompt"] == "exact prompt"
    assert snapshot["firmware_text"] == "exact firmware"
    assert snapshot["outputs"] == ["final.png"]
    assert len(snapshot["references"][0]["sha256"]) == 64
    assert "secret-not-for-manifest" not in body


def test_resolve_content_falls_back_and_trims():
    long_desc = "sentence one. " * 400
    text = ag.resolve_content_text({"english_description": long_desc}, "short")
    assert len(text) <= ag.SHORT_FIELD_MAX_CHARS
    assert not text.endswith(",")


def test_build_request_without_style_ref_has_no_exclusion():
    data = {"gpt_image_prompt_short": "a seated woman in a white dress"}
    req = ag.build_gpt_image_request(data, style_text="Palette: pale", style_ref_path="")
    assert req["image_paths"] == []
    assert req["prompt"].startswith("Palette: pale")
    assert "a seated woman" in req["prompt"]
    assert ag.STYLE_REF_EXCLUSION not in req["prompt"]


def test_build_request_with_style_ref_appends_exclusion(tmp_path):
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"\x89PNG")
    req = ag.build_gpt_image_request({"gpt_image_prompt_short": "content"}, style_text="Palette: pale",
                                     style_ref_path=str(ref), user_hint="keep the ruffles")
    assert req["image_paths"] == [str(ref)]          # 只挂画风参考图，不挂原分析图
    assert ag.STYLE_REF_EXCLUSION in req["prompt"]
    assert "keep the ruffles" in req["prompt"]
    assert req["content_chars"] == len("content")


def test_pipeline_steps_from_flags():
    steps = ag.pipeline_steps_from_flags(repaint=True, structure=True, local=True,
                                        structure_strength=0.35, local_region="face", local_feather=64)
    assert steps["repaint"]["enabled"] is True
    assert steps["structure"]["strength"] == 0.35
    assert steps["local"]["region"] == "face" and steps["local"]["feather"] == 64
    off = ag.pipeline_steps_from_flags()
    assert not any(off[k]["enabled"] for k in off)


def _ref(tmp_path):
    from utils.styles import ref_image_valid
    ref = tmp_path / "style-ref.png"
    ref.write_bytes(b"\x89PNG")
    assert ref_image_valid(str(ref))
    return ref


def test_derive_style_clauses_from_prompt_gpt_fields():
    """没有手写 repaint_clauses 的画风：按自己的 prompt_gpt 字段派生条款（不新增内容）。"""
    from utils.style_gpt import derive_style_clauses
    body = ("Palette: cobalt, pearl white\nLighting: high-key cool key\nBrushwork: smooth cel base\n"
            "Edges: tapered navy linework\nTexture: satin sheen\n"
            "Composition density: centred figure\nDetail level: crisp focal rendering\n"
            "Avoid: pure black outlines")
    clauses = derive_style_clauses(body)
    text = " ".join(clauses)
    assert 8 <= len(clauses) <= 12
    assert "cobalt, pearl white" in text                 # 取自画风自己的调色板
    assert "tapered navy linework" in text               # 线条画法
    assert "pure black outlines" in text                 # 避免项
    assert "never turns pale, grey or washed out" in text        # 通用：防发灰
    assert "Preserve real chroma" in text                        # 通用：保彩度
    assert derive_style_clauses("") == []


def test_resolve_style_clauses_prefers_entry_then_derives():
    from utils.style_gpt import resolve_style_clauses
    written = ["Use clean tapered linework.", "Keep deep blue anchors."]
    assert resolve_style_clauses({"repaint_clauses": written, "prompt_gpt": "Palette: x"}) == (written, "entry")
    clauses, source = resolve_style_clauses({"prompt_gpt": "Palette: cobalt"})
    assert source == "derived" and clauses and "cobalt" in clauses[0]
    assert resolve_style_clauses({}) == ([], "none")
    assert resolve_style_clauses(None) == ([], "none")


def test_neutral_repaint_clauses_omit_palette_and_content_fields():
    from utils.style_gpt import resolve_neutral_repaint_clauses
    entry = {"prompt_gpt": (
        "Palette: sapphire, turquoise, rose pink\nLighting: cool blue key\n"
        "Brushwork: smooth cel base, transparent glazes\nEdges: tapered navy linework\n"
        "Texture: satin sheen, restrained grain\nComposition density: centred figure\n"
        "Detail level: ribbons, shoes and eyelashes\nAvoid: uniform black outlines")}
    clauses, source = resolve_neutral_repaint_clauses(entry)
    text = " ".join(clauses)
    assert source == "neutral-derived"
    assert "smooth cel base" in text and "tapered navy linework" in text
    assert "sapphire" not in text and "turquoise" not in text and "rose pink" not in text
    assert "cool blue key" not in text and "centred figure" not in text
    assert "ribbons, shoes and eyelashes" not in text
    assert "Ignore its exact palette" in text


def test_neutral_style_reference_uses_colour_agnostic_role(tmp_path, monkeypatch):
    from PIL import Image
    from modules.others import api_backend
    source, reference, output = [tmp_path / name for name in ("source.png", "style.png", "output.png")]
    for path in (source, reference, output):
        Image.new("RGB", (80, 120), "white").save(path)
    seen = {}
    monkeypatch.setattr(api_backend, "generate_image_repaint",
                        lambda **kwargs: seen.update(kwargs) or [str(output)])
    steps = ag.pipeline_steps_from_flags(repaint=True, repaint_ref_mode="style_neutral")
    ag.run_gpt_image_pipeline([str(source)], steps, firmware="FIRMWARE",
        style_ref_path=str(reference), style_clauses=["NEUTRAL SENTINEL"],
        final_dir=str(tmp_path / "final"), work_dir=str(tmp_path / "work"))
    assert seen["extra_reference_paths"] == [str(reference)]
    assert pp.STYLE_REF_ROLE_NEUTRAL_IN_REPAINT in seen["prompt"]
    assert pp.STYLE_REF_ROLE_IN_REPAINT not in seen["prompt"]
    assert "NEUTRAL SENTINEL" in seen["prompt"]


def test_build_first_pass_request_orders_style_content_exclusion_clauses(tmp_path):
    """首图组装顺序：画风 → 内容 → 排除句 → RENDERING LANGUAGE；参考图只挂画风图。"""
    ref = _ref(tmp_path)
    styles = {"tid": {"prompt": "FULL", "prompt_gpt": "Palette: cobalt, pearl white",
                      "ref_image": str(ref)}}
    req = ag.build_first_pass_request(styles, "tid", {"gpt_image_prompt_short": "content anchor"},
                                      content_text="a seated figure")
    prompt = req["prompt"]
    assert prompt.index("Palette: cobalt") < prompt.index("a seated figure") < prompt.index(ag.STYLE_REF_EXCLUSION)
    assert prompt.rstrip().endswith(req["clauses"][-1])          # 条款在最后
    assert "RENDERING LANGUAGE (follow exactly):" in prompt
    assert req["image_paths"] == [str(ref)]                      # 不挂分析素材图
    assert req["clauses_source"] == "derived"


def test_build_first_pass_request_uses_written_clauses(tmp_path):
    ref = _ref(tmp_path)
    written = ["Use clean tapered linework with dark navy-blue contours."]
    styles = {"tinkle": {"prompt": "FULL", "prompt_gpt": "Palette: sapphire",
                         "ref_image": str(ref), "repaint_clauses": written}}
    req = ag.build_first_pass_request(styles, "tinkle", {"english_description": "content"}, tier="short")
    assert req["clauses_source"] == "entry"
    assert written[0] in req["prompt"]


def test_build_first_pass_request_carries_style_quality_refine_override(tmp_path):
    ref = _ref(tmp_path)
    styles = {"iris": {"prompt_gpt": "Palette: slate blue", "ref_image": str(ref),
                       "skip_quality_refine": True}}
    req = ag.build_first_pass_request(styles, "iris", {"gpt_image_prompt": "content"})
    assert req["skip_quality_refine"] is True
    assert ag.build_first_pass_request({}, "", {"gpt_image_prompt": "content"})[
        "skip_quality_refine"] is False


def test_build_first_pass_request_carries_face_hair_refine_override(tmp_path):
    ref = _ref(tmp_path)
    styles = {"puracotte": {"prompt_gpt": "Palette: lavender", "ref_image": str(ref),
                            "face_hair_refine": True}}
    req = ag.build_first_pass_request(styles, "puracotte", {"gpt_image_prompt": "content"})
    assert req["face_hair_refine"] is True


def test_build_first_pass_request_separates_generation_and_repaint_clauses(tmp_path):
    ref = _ref(tmp_path)
    styles = {"chibi": {"prompt_gpt": "Palette: slate blue", "ref_image": str(ref),
                         "repaint_clauses": ["REPAINT ONLY"],
                         "generation_clauses": ["RECOMPOSE AS CHIBI"],
                         "identity_correction_clauses": ["DO NOT BRIGHTEN"],
                         "post_adjustment": {"brightness_scale": 0.9}}}
    req = ag.build_first_pass_request(styles, "chibi", {"gpt_image_prompt": "content"})
    assert "REPAINT ONLY" in req["prompt"] and "RECOMPOSE AS CHIBI" in req["prompt"]
    assert req["clauses"] == ["REPAINT ONLY"]
    assert req["generation_clauses"] == ["RECOMPOSE AS CHIBI"]
    assert req["identity_correction_clauses"] == ["DO NOT BRIGHTEN"]
    assert req["post_adjustment"] == {"brightness_scale": 0.9}


def test_style_post_adjustment_applies_channel_and_line_settings(tmp_path):
    import cv2
    import numpy as np

    source = tmp_path / "source.png"
    canvas = np.full((180, 120, 3), 220, np.uint8)
    cv2.rectangle(canvas, (25, 20), (95, 160), (130, 150, 180), 3)
    cv2.imwrite(str(source), canvas)
    out = ag.apply_style_post_adjustment(str(source), {
        "channel_gains": {"red": 1.1, "green": 1.0, "blue": 0.8},
        "brightness_scale": 0.9,
        "structure": {"enabled": True, "strength": 0.8, "min_len": 30,
                      "darken": 0.5, "thin": False, "dilate": 1},
    }, output_dir=str(tmp_path / "out"))

    adjusted = cv2.imread(out)
    assert adjusted is not None
    assert adjusted[..., 2].mean() > adjusted[..., 0].mean()
    assert adjusted.mean() < canvas.mean()


def test_face_hair_refine_uses_style_reference_and_source_ratio(tmp_path, monkeypatch):
    from PIL import Image
    from modules.others import api_backend
    source, reference, output = [tmp_path / name for name in ("source.png", "style.png", "output.png")]
    Image.new("RGB", (800, 1200), "white").save(source)
    Image.new("RGB", (500, 500), "pink").save(reference)
    Image.new("RGB", (800, 1200), "white").save(output)
    seen = {}

    def repaint(*args, **kwargs):
        seen.update(kwargs)
        return [str(output)]

    monkeypatch.setattr(api_backend, "generate_image_repaint", repaint)
    result = ag.run_face_hair_style_refine(
        str(source), str(reference), style_clauses=["Use oversized jewel-like irises."],
        output_dir=str(tmp_path))
    assert result == [str(output)]
    assert seen["extra_reference_paths"] == [str(reference)]
    assert seen["aspect_ratio"] == "2:3"
    assert "FACE AND HAIR STYLE-GRAMMAR REVISION" in seen["prompt"]
    assert "Use oversized jewel-like irises." in seen["prompt"]


def test_build_first_pass_request_without_style_has_no_clause_block():
    req = ag.build_first_pass_request({}, "", {"gpt_image_prompt_short": "content"})
    assert "RENDERING LANGUAGE" not in req["prompt"]
    assert req["clauses"] == [] and req["clauses_source"] == "none"
    assert req["image_paths"] == []


def test_source_only_repaint_does_not_describe_absent_style_image(tmp_path, monkeypatch):
    from PIL import Image
    from modules.others import api_backend
    source, reference, output = [tmp_path / name for name in ("source.png", "style.png", "output.png")]
    for path in (source, reference, output):
        Image.new("RGB", (80, 120), "white").save(path)
    seen = {}
    def repaint(**kwargs):
        seen.update(kwargs)
        return [str(output)]
    monkeypatch.setattr(api_backend, "generate_image_repaint", repaint)
    steps = ag.pipeline_steps_from_flags(repaint=True, repaint_ref_mode="none", repaint_scope="full")
    ag.run_gpt_image_pipeline([str(source)], steps, firmware="PRESERVE SOURCE",
        style_ref_path=str(reference), style_clauses=["STYLE SENTINEL"],
        final_dir=str(tmp_path / "final"), work_dir=str(tmp_path / "work"))
    assert seen["source_paths"] == [str(source)]
    assert seen["extra_reference_paths"] == []
    assert "STYLE SENTINEL" not in seen["prompt"]
    assert pp.STYLE_REF_ROLE_IN_REPAINT not in seen["prompt"]
    assert "IDENTITY AND CONTENT LOCK" in seen["prompt"]
    assert seen["save_sub_dir"] == str(tmp_path / "final")


def test_run_gpt_image_pipeline_structure_only(tmp_path, monkeypatch):
    """只勾结构线叠加时不调任何 API。"""
    import cv2
    import numpy as np
    img = tmp_path / "img.png"
    canvas = np.full((360, 240, 3), 240, np.uint8)
    cv2.rectangle(canvas, (40, 60), (200, 300), (200, 190, 180), -1)
    for x in range(40, 200, 16):
        cv2.line(canvas, (x, 60), (x + 8, 60), (110, 100, 95), 2)
    cv2.imwrite(str(img), canvas)
    steps = ag.pipeline_steps_from_flags(structure=True)
    calls = []
    monkeypatch.setattr(ag, "run_gpt_image_pipeline", ag.run_gpt_image_pipeline)  # no-op
    from modules.others import api_backend
    monkeypatch.setattr(api_backend, "generate_image_repaint",
                        lambda **kw: calls.append(kw) or [])   # 不应该被调用
    out = ag.run_gpt_image_pipeline([str(img)], steps)
    assert out and out[0].endswith(".png") and "-sline50" in out[0]
    assert calls == []


def test_run_gpt_image_pipeline_repaint_then_structure(tmp_path, monkeypatch):
    """勾了重绘 → 先调 Gemini 重绘（带 use_detail_suffix=False），再叠加结构线。"""
    import cv2
    import numpy as np
    img = tmp_path / "img2.png"
    cv2.imwrite(str(img), np.full((240, 160, 3), 220, np.uint8))
    repainted = tmp_path / "repainted.png"
    cv2.imwrite(str(repainted), np.full((240, 160, 3), 210, np.uint8))
    seen = {}

    def _fake_repaint(**kwargs):
        seen.update(kwargs)
        return [str(repainted)]

    from modules.others import api_backend
    monkeypatch.setattr(api_backend, "generate_image_repaint", _fake_repaint)
    steps = ag.pipeline_steps_from_flags(repaint=True, structure=True)
    out = ag.run_gpt_image_pipeline([str(img)], steps, firmware="FIRMWARE",
                                    final_dir=str(tmp_path / "final"), work_dir=str(tmp_path / "work"))
    assert seen["use_detail_suffix"] is False
    # 默认 full 不追加局部范围句；身份/内容锁仍必须存在。
    assert seen["prompt"].startswith("FIRMWARE")
    assert "IDENTITY AND CONTENT LOCK" in seen["prompt"]
    assert seen["source_paths"] == [str(img)]
    assert seen["extra_reference_paths"] == []
    # 最终产物名带 -final- 标记（工序串说明经过了哪些处理），中间产物在 work 目录里
    assert out and "-final-rp+sline50" in os.path.basename(out[0])
    assert os.path.isfile(os.path.join(str(tmp_path / "work"), "pipeline-manifest.json"))


def test_content_reference_is_first_and_style_last(tmp_path):
    """挂内容参考图时：内容图在前、画风图最后，并写入 IMAGE ROLES 角色分工。"""
    content_img = tmp_path / "source.png"
    style_img = tmp_path / "style.png"
    import cv2
    import numpy as np
    cv2.imwrite(str(content_img), np.full((400, 300, 3), 200, np.uint8))
    cv2.imwrite(str(style_img), np.full((400, 300, 3), 150, np.uint8))
    req = ag.build_gpt_image_request({"gpt_image_prompt_short": "a girl in a blue dress"},
                                     style_text="Palette: blue", style_ref_path=str(style_img),
                                     content_image_path=str(content_img))
    assert req["image_paths"] == [str(content_img), str(style_img)]      # 内容图在前、画风图最后
    assert req["has_content_ref"] is True
    assert "IMAGE ROLES" in req["prompt"]
    assert "CONTENT REFERENCES" in req["prompt"]


def test_without_content_reference_only_style_image(tmp_path):
    style_img = tmp_path / "style2.png"
    import cv2
    import numpy as np
    cv2.imwrite(str(style_img), np.full((400, 300, 3), 150, np.uint8))
    req = ag.build_gpt_image_request({"gpt_image_prompt_short": "a girl"},
                                     style_text="Palette: blue", style_ref_path=str(style_img))
    assert req["image_paths"] == [str(style_img)]
    assert req["has_content_ref"] is False
    assert "IMAGE ROLES" not in req["prompt"]


# ------------------------- 提示词顺序：画风 prompt_gpt 在最前，内容描述（与 Gemini 同源）在后

def test_style_text_comes_first_then_content(tmp_path):
    """画风说明拼在最前面，后面接分析描述（用户 2026-09-24 要求）。"""
    style_img = tmp_path / "style.png"
    import cv2
    import numpy as np
    cv2.imwrite(str(style_img), np.full((400, 300, 3), 150, np.uint8))
    content = "one girl seated on a chair, ivory rococo dress, antique salon, full-body vertical 2:3"
    req = ag.build_gpt_image_request({"gpt_image_prompt_short": "SHORT-ANCHOR"},
                                     style_text="Palette: blue", style_ref_path=str(style_img),
                                     content_text=content)
    assert req["prompt"].startswith("Palette: blue")          # 画风在最前
    assert content in req["prompt"]                            # 内容描述紧随其后
    assert "SHORT-ANCHOR" not in req["prompt"]                 # 显式给 content_text 时不再用 gpt 短锚
    assert req["content_chars"] == len(content)
    assert ag.__dict__.get("RECOMPOSE_CLAUSE") is None         # 不再有「重新构图」条款


def test_content_text_falls_back_to_gpt_fields(tmp_path):
    """没给 content_text 的老调用方：退回 gpt 专用字段（CLI 仍可这样用）。"""
    req = ag.build_gpt_image_request({"gpt_image_prompt_short": "a girl seated on a chair"},
                                     style_text="Palette: blue")
    assert "a girl seated on a chair" in req["prompt"]


def test_content_reference_still_supported_but_not_default(tmp_path):
    """明确要"描图"时仍可挂分析原图（GUI 默认不挂）。"""
    content_img = tmp_path / "src.png"
    import cv2
    import numpy as np
    cv2.imwrite(str(content_img), np.full((400, 300, 3), 200, np.uint8))
    req = ag.build_gpt_image_request({"gpt_image_prompt_short": "a girl"},
                                     content_image_path=str(content_img))
    assert req["image_paths"] == [str(content_img)]
    assert req["has_content_ref"] is True


def test_pipeline_steps_use_2k_local_without_detail_boost():
    """局部重绘统一 2K：不再把细节区自动升到 4K（用户 2026-09-24 要求）。"""
    steps = ag.pipeline_steps_from_flags(repaint=True, structure=True, local=True,
                                         local_region="subject_no_face")
    assert steps["local"]["resolution"] == "2K"
    assert steps["local"]["detail_boost"] is False


def test_pipeline_steps_repaint_uses_complete_style_reference_by_default():
    """线条优先默认把完整画风图作为 Gemini 第二张参考。"""
    steps = ag.pipeline_steps_from_flags(repaint=True)
    assert steps["repaint"]["reference_mode"] == "style"
    assert steps["repaint"]["scope"] == "full"
    steps2 = ag.pipeline_steps_from_flags(repaint=True, repaint_ref_mode="line_anchor")
    assert steps2["repaint"]["reference_mode"] == "line_anchor"
