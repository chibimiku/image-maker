# -*- coding: utf-8 -*-
"""画风自带「材质/表面条款」（`gemini_surface_clauses`）的离线回归。

需求（用户 2026-10-06）：**只有 `satou_kuuki-style-v2` 默认就要有皮革质感**，别的画风一个字都不许加。
实现落点：画风条目 → `build_ref_gen_params` 第四个返回值 → 图后「材质权威」块
（`prompts/gemini-style-surface-authority.md`），**必须排在内容锁之前**（图后执行要求），
且只改表面响应、不改几何/身份/配色。
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from utils.analysis_gen import build_gemini_content_postamble, build_surface_authority
from utils.styles import (build_ref_gen_params, gemini_priority_profile_for,
                          normalize_style_entry, surface_clauses_for)

STYLES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "conf", "config-styles.json")


def _styles():
    with open(STYLES_PATH, encoding="utf-8") as handle:
        return json.load(handle)


class SurfaceClausesTests(unittest.TestCase):
    def test_only_satou_style_ships_surface_clauses(self):
        styles = _styles()
        carrying = sorted(name for name, entry in styles.items()
                          if isinstance(entry, dict) and entry.get("gemini_surface_clauses"))
        self.assertEqual(carrying, ["satou_kuuki-style-v2"])

    def test_satou_clauses_are_material_only(self):
        clauses = surface_clauses_for(_styles(), "satou_kuuki-style-v2")
        self.assertEqual(len(clauses), 3)
        blob = " ".join(clauses).lower()
        self.assertIn("patent leather", blob)          # 默认高光皮革
        self.assertIn("wet-sheen", blob)               # 该画风本就有的湿亮族
        self.assertIn("matte", blob)                   # 明确保护内容里的哑光材质
        for lock in ("identity", "composition", "aspect"):   # 不含任何几何/构图改写
            self.assertNotIn(lock, blob)

    def test_other_styles_and_unknown_names_return_nothing(self):
        styles = _styles()
        for name in ("myc0t0xin", "cccccanh", "puracotte-style-v2", "nope"):
            self.assertEqual(surface_clauses_for(styles, name), [], name)
        self.assertEqual(surface_clauses_for({}, "satou_kuuki-style-v2"), [])

    def test_normalize_keeps_and_caps_the_field(self):
        entry = normalize_style_entry({"prompt": "x",
                                       "gemini_surface_clauses": [" a ", "", "b"]})
        self.assertEqual(entry["gemini_surface_clauses"], ["a", "b"])
        self.assertEqual(normalize_style_entry("plain")["gemini_surface_clauses"], [])

    def test_build_ref_gen_params_keeps_its_three_value_signature(self):
        """20 多处调用方按 3 个值解包 —— 材质块只能拼进 `post`，不能改返回值形状。"""
        styles = _styles()
        head, post, refs = build_ref_gen_params(styles, "satou_kuuki-style-v2", "priority")
        self.assertTrue(refs)
        self.assertIn("MATERIAL / SURFACE AUTHORITY", post)
        self.assertIn("patent leather", post.lower())
        # 条款不能混进图前的风格说明里（那会去和图争权威）
        self.assertNotIn("patent leather", head.lower())

    def test_satou_post_puts_the_material_block_after_the_style_post(self):
        styles = _styles()
        _head, post, _refs = build_ref_gen_params(styles, "satou_kuuki-style-v2", "priority")
        self.assertLess(post.index("MATERIAL / SURFACE AUTHORITY"), len(post))


class SurfaceAuthorityBlockTests(unittest.TestCase):
    def test_empty_clauses_produce_no_block(self):
        self.assertEqual(build_surface_authority([]), "")
        self.assertEqual(build_surface_authority(None), "")
        self.assertEqual(build_surface_authority(["  ", ""]), "")

    def test_block_scopes_itself(self):
        block = build_surface_authority(surface_clauses_for(_styles(), "satou_kuuki-style-v2"))
        lowered = block.lower()
        self.assertIn("material / surface authority", lowered)
        self.assertIn("patent leather", lowered)
        # 只改表面：几何/身份/配色/构图不许动；内容是唯一真值
        self.assertIn("only where the content already uses that material", lowered)
        self.assertIn("do not add, remove, recolour or redesign anything", lowered)
        self.assertIn("content explicitly states a different surface treatment", lowered)
        self.assertIn("never let this block produce a material the content does not contain", lowered)

    def test_block_sits_before_the_content_lock_in_the_postamble(self):
        styles = _styles()
        _head, post, _refs = build_ref_gen_params(styles, "satou_kuuki-style-v2", "priority")
        combined = build_gemini_content_postamble("CONTENT-HERE", post)
        self.assertLess(combined.index("MATERIAL / SURFACE AUTHORITY"),
                        combined.index("CONTENT AUTHORITY"),
                        "材质块必须排在内容锁之前（图后先执行要求、再重申内容）")
        self.assertIn("CONTENT-HERE", combined)

    def test_style_without_clauses_produces_the_same_postamble_as_before(self):
        styles = _styles()
        _head, post, _refs = build_ref_gen_params(styles, "myc0t0xin", "priority")
        self.assertNotIn("MATERIAL / SURFACE AUTHORITY", post)
        combined = build_gemini_content_postamble("CONTENT-HERE", post)
        self.assertNotIn("MATERIAL / SURFACE AUTHORITY", combined)
        self.assertIn("CONTENT-HERE", combined)

    def test_gpt_channel_also_gets_the_style_finish(self):
        """gpt 通道同理：材质块在 post 里，图后的执行要求对两条通道都成立。"""
        styles = _styles()
        _head, post, _refs = build_ref_gen_params(styles, "satou_kuuki-style-v2", "priority",
                                                  api_type="aigc-2d-gpt")
        self.assertIn("MATERIAL / SURFACE AUTHORITY", post)


# The opt-in profile must transfer paint without leaking into GPT/other modes.


def _priority_styles(tmp_path):
    ref = tmp_path / "reference.jpg"
    ref.write_bytes(b"test reference")
    return {"puracotte": {"prompt": "legacy", "prompt_gpt": "gpt legacy",
            "ref_image": str(ref), "gemini_priority_profile": {
                "head_file": "gemini-style-profiles/puracotte-v2-head.md",
                "post_file": "gemini-style-profiles/puracotte-v2-post.md",
                "face_quality_boost": False}}}


def test_priority_profile_assembles_whole_scene_contract(tmp_path):
    styles = _priority_styles(tmp_path)
    head, post, refs = build_ref_gen_params(styles, "puracotte", "priority", "aigc2d")
    content = 'Red coat, black hair, amber eyes, a gold necklace and a sign reading "OPEN".'
    combined = build_gemini_content_postamble(content, post)
    assert refs == [styles["puracotte"]["ref_image"]]
    assert head and "Scene surfaces:" in combined
    assert "{content}" not in combined and content in combined
    assert "CONTENT AUTHORITY" not in combined  # no legacy content-wins override
    assert "User-requested jewellery, text and objects take precedence" in combined
    assert "opaque materials retain their opacity" in combined
    assert "deep navy" not in combined and "chestnut" not in combined
    assert gemini_priority_profile_for(styles, "puracotte", "priority", "aigc2d")["face_quality_boost"] is False


@pytest.mark.parametrize("mode,api", [("off", "aigc2d"), ("head", "aigc2d"),
    ("interleave", "aigc2d"), ("priority", "aigc-2d-gpt")])
def test_priority_profile_preserves_other_modes(tmp_path, mode, api):
    styles = _priority_styles(tmp_path)
    expected = {"puracotte": dict(styles["puracotte"])}
    expected["puracotte"].pop("gemini_priority_profile")
    assert gemini_priority_profile_for(styles, "puracotte", mode, api) == {}
    assert build_ref_gen_params(styles, "puracotte", mode, api) == build_ref_gen_params(expected, "puracotte", mode, api)


def test_priority_profile_missing_reference_uses_legacy(tmp_path):
    styles = _priority_styles(tmp_path)
    Path(styles["puracotte"]["ref_image"]).unlink()
    assert gemini_priority_profile_for(styles, "puracotte", "priority", "aigc2d") == {}
    assert build_ref_gen_params(styles, "puracotte", "priority", "aigc2d") == ("gpt legacy", "", [])


def test_only_puracotte_ships_priority_profile():
    carrying = [name for name, entry in _styles().items()
                if isinstance(entry, dict) and entry.get("gemini_priority_profile")]
    assert carrying == ["puracotte-style-v2"]


@pytest.mark.parametrize("boost", [False, True])
def test_worker_preserves_face_boost_in_plain_and_audit_paths(tmp_path, boost):
    from modules.image_analysis import single_analyzer as sa
    worker = sa.ImageGenWorkerThread(prompt="content", model_name="model", aspect_ratio="2:3",
        instructions="head", api_type="aigc2d", post_instructions="post",
        face_quality_boost=boost, publish_dir=str(tmp_path), publish_name="test.jpg")
    with patch.object(sa, "generate_image_aigc2d", return_value=[]) as gen:
        worker._call_generator("content")
        assert gen.call_args.kwargs["face_quality_boost"] is boost
        worker.run()
        assert gen.call_args.kwargs["face_quality_boost"] is boost
        def fake_audit(generate, **kwargs):
            generate("retry content", 2)
            return {"published": ""}
        with patch("utils.first_image_audit.generate_with_audit", side_effect=fake_audit):
            worker._run_with_audit()
        assert gen.call_args.kwargs["face_quality_boost"] is boost
        assert gen.call_args.kwargs["prompt"] == "retry content"


if __name__ == "__main__":
    unittest.main()
