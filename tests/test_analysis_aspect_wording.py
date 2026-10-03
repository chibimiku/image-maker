# -*- coding: utf-8 -*-
"""内容锚里的画幅措辞对齐（只服务分析链路）。

起因（2026-10-01，634dfc91 实测）：源图是 2000x1024 的横图，但分析产物正文开头写着
"Vertical 2:3 illustration"（Step 1 的 vision 把横图说成竖构图，Step 2 的精修又把推断出来的
2:3 写进正文）。首图请求的 size 其实是横向的 1536x1024，可实测文本能压过 size ——
全项目 5 个「请求横向」的首图里，唯二出成竖图的两个正文都写着竖幅措辞。

这组用例守住三件事：
1. 下发前按**本次生效尺寸**改写正文里的画幅声明（手动选定的尺寸同样生效）；
2. 普通描述里的 vertical/horizontal 不许被误伤；
3. 落盘的 `gpt_image_prompt` 按**源图实测比例**改写，画风参考图的朝向永不参与。
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import utils.analysis_gen as ag  # noqa: E402
import utils.analysis_gpt_prompt as agp  # noqa: E402
from utils.aspect_wording import align_aspect_wording, aspect_label  # noqa: E402

# 634dfc91 真实产物前缀（假画幅 + 后面真的有一个 "vertical ribbons"）
FAKE_VERTICAL = (
    "Vertical 2:3 illustration, one woman centered-right, three-quarter view toward viewer, "
    "very long black hair with deep-brown sheen, high traditional bun, floral ornaments, "
    "large blue waist bow with long vertical ribbons; seated or kneeling, lower body concealed. "
    "Mountains, flowering branches, clouds and curling water motifs surround her."
)


def test_aspect_label_reads_sizes_and_ratios():
    assert aspect_label(size="1536x1024") == ("Horizontal", "3:2")
    assert aspect_label(size="1024x1536") == ("Vertical", "2:3")
    assert aspect_label(size="1024x1024") == ("Square", "1:1")
    assert aspect_label(ratio="16:9") == ("Horizontal", "16:9")
    assert aspect_label(ratio="9:16") == ("Vertical", "9:16")
    assert aspect_label(size="16:9") == ("Horizontal", "16:9")   # 调用方把比例塞进 size 也认
    assert aspect_label() == ("", "")
    assert aspect_label(size="怪东西") == ("", "")


def test_landscape_size_rewrites_fake_vertical_head():
    out, changes = align_aspect_wording(FAKE_VERTICAL, size="1536x1024")
    assert out.startswith("Horizontal 3:2 illustration,")
    assert "Vertical 2:3" not in out
    assert changes == ["Vertical 2:3 → Horizontal 3:2"]


def test_unrelated_vertical_phrases_survive():
    """`long vertical ribbons` 不是画幅声明，一个字都不许动。"""
    out, _ = align_aspect_wording(FAKE_VERTICAL, size="1536x1024")
    assert "large blue waist bow with long vertical ribbons" in out
    assert "three-quarter view toward viewer" in out


def test_matching_size_leaves_the_text_alone():
    out, changes = align_aspect_wording(FAKE_VERTICAL, size="1024x1536")
    assert out == FAKE_VERTICAL
    assert changes == []


def test_mid_sentence_wording_gets_lowercase_replacement():
    text = ("Single girl in an enchanted rose garden at dawn, off-center upper-right in a "
            "portrait 2:3 composition, elevated close three-quarter view.")
    out, changes = align_aspect_wording(text, size="1536x1024")
    assert "in a horizontal 3:2 composition" in out
    assert "portrait 2:3" not in out
    assert changes == ["portrait 2:3 → horizontal 3:2"]


def test_landscape_claim_is_rewritten_for_portrait_canvas():
    text = ("16:9 landscape illustration scene, two female characters seated opposite across "
            "a cafe table beside a broad nighttime window.")
    out, _ = align_aspect_wording(text, size="1024x1536")
    assert out.startswith("Vertical 2:3 illustration scene,")
    assert "landscape" not in out


def test_ratio_argument_rewrites_frame_and_orientation():
    text = "This vertical illustration, balanced vertical tableau in a 2:3 frame."
    out, changes = align_aspect_wording(text, ratio="16:9")
    assert out == "This horizontal illustration, balanced horizontal tableau in a 16:9 frame."
    assert len(changes) == 3


def test_unknown_size_leaves_text_untouched():
    out, changes = align_aspect_wording(FAKE_VERTICAL, size="")
    assert out == FAKE_VERTICAL and changes == []
    out, changes = align_aspect_wording(FAKE_VERTICAL, ratio="说不清")
    assert out == FAKE_VERTICAL and changes == []


def test_settled_wording_outside_the_window_is_not_touched():
    """画幅声明只可能出现在开头；400 字符以外的同形文本不碰，免得误伤正常的画面描述。"""
    text = "A woman seated in a garden. " + ("filler sentence about brushes and washes. " * 14) \
           + "vertical 2:3 frame near the tail"
    out, changes = align_aspect_wording(text, size="1536x1024")
    assert changes == []


# ---------------- 组装点：build_first_pass_request ----------------

def _styles():
    return {"tid": {"prompt_gpt": "Palette: pale", "ref_image": ""}}


def test_build_first_pass_request_aligns_content_anchor():
    req = ag.build_first_pass_request(_styles(), "tid",
                                      {"gpt_image_prompt": FAKE_VERTICAL},
                                      tier="short", size="1536x1024")
    assert req["prompt"].startswith("Palette: pale")
    assert "Horizontal 3:2 illustration," in req["prompt"]
    assert "Vertical 2:3" not in req["prompt"]
    assert req["requested_size"] == "1536x1024"
    assert req["aspect_alignment"] == ["Vertical 2:3 → Horizontal 3:2"]


def test_build_first_pass_request_without_size_keeps_the_original_text():
    req = ag.build_first_pass_request(_styles(), "tid",
                                      {"gpt_image_prompt": FAKE_VERTICAL}, tier="short")
    assert "Vertical 2:3 illustration," in req["prompt"]
    assert req["aspect_alignment"] == []


def test_build_first_pass_request_aligns_when_content_comes_from_the_result():
    """没显式传 content_text 时也要对齐（默认走产物字段那条路）。"""
    req = ag.build_first_pass_request({}, "", {"gpt_image_prompt": FAKE_VERTICAL},
                                      tier="short", size="1024x1536")
    assert "Vertical 2:3 illustration," in req["prompt"]


# ---------------- 落盘：build_gpt_image_prompt 按源图实测比例 ----------------

def test_gpt_image_prompt_is_aligned_with_the_measured_source_ratio(monkeypatch):
    monkeypatch.setattr(agp, "call_text_model", lambda *a, **k: FAKE_VERTICAL)
    out = agp.build_gpt_image_prompt(
        "long description", text_cfg={"base_url": "http://x", "api_key": "k", "model": "m"},
        aspect_ratio="16:9")
    assert out.startswith("Horizontal 16:9 illustration,")
    assert "Vertical 2:3" not in out


def test_gpt_image_prompt_keeps_wording_when_source_is_portrait(monkeypatch):
    monkeypatch.setattr(agp, "call_text_model",
                        lambda *a, **k: "Vertical 2:3 illustration, one woman centered-right.")
    out = agp.build_gpt_image_prompt(
        "long description", text_cfg={"base_url": "http://x", "api_key": "k", "model": "m"},
        aspect_ratio="2:3")
    assert out.startswith("Vertical 2:3 illustration,")
