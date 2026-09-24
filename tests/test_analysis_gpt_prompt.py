# -*- coding: utf-8 -*-
"""分析产物的 gpt-image 专用短提示词字段（utils/analysis_gpt_prompt.py）与落盘行为。"""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from modules.image_analysis.analysis_pipeline import save_result_to_source  # noqa: E402
from utils import analysis_gpt_prompt as agp  # noqa: E402


def test_sanitize_strips_fences_and_labels():
    raw = "```\nShort prompt: a young woman, white dress, seated\n```"
    out = agp.sanitize_short_prompt(raw, max_chars=900)
    assert out == "a young woman, white dress, seated"


def test_sanitize_truncates_on_sentence_boundary():
    text = " ".join(["clause number %d with padding words" % i for i in range(60)])
    out = agp.sanitize_short_prompt(text, max_chars=120)
    assert len(out) <= 120
    assert not out.endswith(",")


def test_compose_and_length_warning():
    composed = agp.compose_gpt_image_prompt("STYLE BLOCK", "CONTENT BLOCK")
    assert composed == "STYLE BLOCK\n\nCONTENT BLOCK"
    # 挂画风参考图时追加"别搬参考图角色特征"的排除句
    with_excl = agp.compose_gpt_image_prompt("STYLE BLOCK", "CONTENT BLOCK", style_ref_exclusion=True)
    assert composed in with_excl
    assert agp.STYLE_REF_EXCLUSION in with_excl
    assert agp.composed_length_warning("x" * 100) == ""
    warn = agp.composed_length_warning("x" * (agp.COMPOSED_WARN_CHARS + 1))
    assert "画风参考图会失效" in warn


def test_short_tier_uses_short_prompt_file_and_cap(monkeypatch):
    captured = {}

    def _fake_call(base_url, api_key, model, system_prompt, user_prompt, **kwargs):
        captured["system"] = system_prompt
        return "short content anchor " * 40

    monkeypatch.setattr(agp, "call_text_model", _fake_call)
    out = agp.build_gpt_image_prompt("d" * 5000,
                                     text_cfg={"base_url": "u", "api_key": "k", "model": "m"},
                                     max_chars=9999, tier="short")
    assert len(out) <= agp.SHORT_FIELD_MAX_CHARS
    assert "VERY SHORT" in captured["system"]


def test_resolve_content_field_falls_back():
    assert agp.resolve_content_field({"gpt_image_prompt": "full"}, "short") == "full"
    assert agp.resolve_content_field({"gpt_image_prompt_short": "short"}, "full") == "short"
    assert agp.resolve_content_field({"gpt_image_prompt": "full", "gpt_image_prompt_short": "s"}, "short") == "s"


def test_build_gpt_image_prompt_uses_model(monkeypatch):
    captured = {}

    def _fake_call(base_url, api_key, model, system_prompt, user_prompt, **kwargs):
        captured["model"] = model
        captured["user"] = user_prompt
        return "a seated woman, white lace dress, antique room"

    monkeypatch.setattr(agp, "call_text_model", _fake_call)
    out = agp.build_gpt_image_prompt(
        "LONG DESCRIPTION " * 50,
        text_cfg={"base_url": "https://example.com/v1", "api_key": "k", "model": "m"},
        max_chars=200)
    assert out.startswith("a seated woman")
    assert captured["model"] == "m"
    assert "LONG DESCRIPTION" in captured["user"]


def test_build_requires_text_config(monkeypatch):
    monkeypatch.setattr(agp, "load_text_api_config", lambda *a, **k: {"base_url": "", "api_key": "", "model": ""})
    try:
        agp.build_gpt_image_prompt("desc", text_cfg={"base_url": "", "api_key": "", "model": ""})
    except RuntimeError as exc:
        assert "文本 API" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("应当抛错")


def test_save_result_writes_gpt_image_prompt_file(tmp_path):
    img = tmp_path / "src.jpg"
    img.write_bytes(b"\xff\xd8\xff")
    result = {
        "english_description": "a woman in a white dress",
        "aspect_ratio": "2:3",
        "gpt_image_prompt": "a woman, white dress, seated, antique room",
        "japanese_title": "テスト",
    }
    json_path = save_result_to_source(result, str(img))
    assert os.path.isfile(json_path)
    saved = json.load(open(json_path, encoding="utf-8"))
    assert saved["gpt_image_prompt"] == result["gpt_image_prompt"]
    txts = [p for p in os.listdir(tmp_path) if p.endswith("-gpt-image-prompts.txt")]
    assert len(txts) == 1
    assert "antique room" in open(tmp_path / txts[0], encoding="utf-8").read()


def test_save_result_without_field_writes_no_extra_file(tmp_path):
    img = tmp_path / "src2.jpg"
    img.write_bytes(b"\xff\xd8\xff")
    save_result_to_source({"english_description": "x", "aspect_ratio": "1:1"}, str(img))
    assert not [p for p in os.listdir(tmp_path) if p.endswith("-gpt-image-prompts.txt")]
