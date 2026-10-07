# -*- coding: utf-8 -*-
"""「材质表面」通道的通用性回归。

用户 2026-10-06 的要求：这套词汇与固件**必须跨任务通用** —— 别处任务里要处理的
可能是皮夹克、皮手套、漆皮包、丝袜、金属扣，不是靴子。所以这里锁三件事：

1. 内置材质的默认目标与条款里**不出现具体物件名**（靴子/裙子/楼梯/衣袖…），
   否则换个任务就会指错对象；
2. 目标永远带"其余材质一律不动"的兜底句，避免材质通道顺手改整幅画面；
3. 自定义材质规格（`--material-spec`）能覆盖任意新材质，不必改固件或工具代码。
"""
from __future__ import annotations

import re
import unittest

from utils.gpt_image_optimize import (
    DEFAULT_MATERIAL_TARGETS, MATERIALS, MATERIAL_SPEC_KEYS, build_material_prompt,
    describe_material, material_names, normalize_material_spec, resolve_material_info,
)

# 本项目已出现过或极易被写死的物件名（任何一条出现在默认目标里都算回归）
OBJECT_WORDS = (
    "boot", "shoe", "heel", "toe", "sock", "stocking", "thigh-high",
    "dress", "skirt", "blouse", "sleeve", "collar", "apron", "ribbon", "bow",
    "glove", "jacket", "coat", "belt", "bag", "hat", "headband",
    "stair", "railing", "curtain", "floor", "wall", "window", "chair", "table", "plank",
    "pom-pom", "pompom", "plush toy", "stuffed animal",
)


def _hits(text: str) -> list:
    """命中写死的物件名（连字符复合词不算：`clear-coat`、`patent-leather` 是材质词）。"""
    lowered = str(text or "").lower()
    return [word for word in OBJECT_WORDS
            if re.search(rf"(?<![a-z-]){re.escape(word)}(?![a-z-])", lowered)]


class MaterialVocabularyTests(unittest.TestCase):
    def test_every_material_has_label_material_and_clauses(self):
        self.assertTrue(material_names())
        for key, info in MATERIALS.items():
            self.assertTrue(info["label"], key)
            self.assertTrue(info["material"], key)
            self.assertTrue(info["clauses"], key)
            self.assertTrue(describe_material(key).startswith(key))

    def test_default_targets_never_name_an_object(self):
        for key, text in DEFAULT_MATERIAL_TARGETS.items():
            self.assertEqual(_hits(text), [], f"{key} 的默认目标写死了物件名：{_hits(text)}")

    def test_default_targets_always_protect_everything_else(self):
        for key, text in DEFAULT_MATERIAL_TARGETS.items():
            self.assertIn("leave every other material", str(text).lower(),
                          f"{key} 的默认目标缺少「其余材质不动」的兜底句")

    def test_clauses_never_name_an_object(self):
        for key, info in MATERIALS.items():
            self.assertEqual(_hits(" ".join(info["clauses"])), [],
                             f"{key} 的条款写死了物件名：{_hits(' '.join(info['clauses']))}")

    def test_unknown_builtin_material_is_rejected_loudly(self):
        with self.assertRaises(ValueError):
            build_material_prompt("nope")
        self.assertEqual(describe_material("nope"), "")
        self.assertEqual(describe_material(None), "")


class MaterialPromptTests(unittest.TestCase):
    def test_prompt_for_any_material_is_object_free_and_self_limiting(self):
        for key in material_names():
            prompt = build_material_prompt(key)
            self.assertEqual(_hits(prompt), [], f"{key} 的最终提示词里出现了物件名：{_hits(prompt)}")
            lowered = prompt.lower()
            # 锁：目标只按材质外观指认 + 其余保持原样 + 不许改几何/配色/构图
            self.assertIn("how the surface is already rendered", lowered, key)
            self.assertIn("leave every other material exactly as it is", lowered, key)
            for lock in ("identity", "composition", "aspect ratio", "base colour"):
                self.assertIn(lock, lowered, f"{key} 缺少 {lock} 锁")
            self.assertLess(len(prompt), 5000, f"{key} 的提示词过长，容易压过画面约束")

    def test_explicit_targets_override_the_default(self):
        prompt = build_material_prompt("metal", targets="the buckles and studs only")
        self.assertIn("the buckles and studs only", prompt)
        self.assertNotIn("every surface in the image that is already drawn as metal", prompt)

    def test_two_materials_produce_different_prompts(self):
        self.assertNotEqual(build_material_prompt("patent_leather"), build_material_prompt("fur"))


class MaterialSpecTests(unittest.TestCase):
    def test_custom_spec_from_string(self):
        spec = normalize_material_spec(
            "material=eggshell-thin lacquer; label=蛋壳漆; clauses=Keep the crackle fine|Never turn it glossy")
        self.assertEqual(spec["material"], "eggshell-thin lacquer")
        self.assertEqual(spec["label"], "蛋壳漆")
        self.assertEqual(spec["clauses"], ["Keep the crackle fine", "Never turn it glossy"])
        self.assertTrue(spec["custom"])

    def test_custom_spec_needs_a_material_line(self):
        with self.assertRaises(ValueError):
            normalize_material_spec("label=只有中文名")

    def test_custom_spec_rejects_unknown_keys(self):
        with self.assertRaises(ValueError):
            normalize_material_spec("material=x; colour=pink")
        self.assertNotIn("colour", MATERIAL_SPEC_KEYS)

    def test_custom_spec_drives_a_prompt_without_code_changes(self):
        prompt = build_material_prompt(
            spec={"material": "aged bronze", "clauses": ["Add a green patina in the recesses only."]})
        self.assertIn("aged bronze", prompt)
        self.assertIn("green patina in the recesses", prompt)
        # 自定义材质也要吃默认兜底句（按材质外观指认 + 其余不动）
        self.assertIn("every surface in the image that is already drawn as this material", prompt)
        self.assertIn("leave every other material", prompt.lower())

    def test_material_and_spec_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            resolve_material_info("fur", "material=matte paper")
        with self.assertRaises(ValueError):
            resolve_material_info("", "")


def test_cli_passes_explicit_timeout_to_transport(tmp_path, monkeypatch):
    from tools import repaint_material as cli
    image = tmp_path / "source.png"
    image.write_bytes(b"test-image")
    calls = []
    monkeypatch.setattr(cli, "generate_image_repaint", lambda *a, **kw: calls.append(kw) or [str(image)])
    assert cli.run(cli.parse_args(["--image", str(image), "--material", "metal", "--timeout", "42"])) == 0
    assert calls[0]["request_timeout"] == 42


if __name__ == "__main__":
    unittest.main()
