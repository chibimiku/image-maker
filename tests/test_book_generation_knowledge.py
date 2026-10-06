import copy
import json
from pathlib import Path
import unittest

from utils.color_extract import build_generation_knowledge, compile_generation_palette


ROOT = Path(__file__).resolve().parents[1]


class GenerationKnowledgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.knowledge = build_generation_knowledge(str(ROOT / "data/color-knowledge"),
            str(ROOT / "docs/261003-color-improve/book-html/pages"))
        cls.selection = json.loads((ROOT / "prompts/color-knowledge/book-generation-example-v1.json").read_text(encoding="utf-8"))

    def compile(self, **changes):
        selection = copy.deepcopy(self.selection)
        selection.update(changes)
        return compile_generation_palette(self.knowledge, selection)

    def test_source_counts_and_unique_ids(self):
        self.assertEqual((self.knowledge["count"], self.knowledge["teaching_count"]), (59, 28))
        self.assertEqual(len({p["id"] for p in self.knowledge["palettes"]}), 59)

    def test_no_invented_teaching_hex_or_roles(self):
        teaching = [p for p in self.knowledge["palettes"] if p["origin"] == "book_teaching"]
        self.assertTrue(all(c["hex"] is None for p in teaching for c in p["colours"]))
        self.assertEqual(sum(c["role"] is not None for p in teaching for c in p["colours"]), 8)

    def test_twelve_qualitative_tones(self):
        self.assertEqual(len(self.knowledge["tones"]), 12)
        self.assertTrue(all(t["numeric_thresholds"] is None for t in self.knowledge["tones"]))
        self.assertIn("NCD Vp", self.compile(tone_code="Vp")["prompt"])

    def test_compile_is_repeatable_and_scoped(self):
        result = self.compile()
        self.assertEqual(result, self.compile())
        self.assertIn("both shoe bows -> 橙", result["prompt"])
        self.assertNotIn("60-70%", result["prompt"])
        self.assertEqual(result["status"], "compiled_not_generated")

    def test_disabled_is_byte_identical(self):
        self.assertEqual(self.compile(enabled=False)["prompt"], self.selection["prompt"])

    def test_composition_guard_is_opt_in_and_hashed(self):
        old = self.compile()
        guarded = self.compile(composition_guard="single-scene-v1")
        self.assertEqual(old["contract"], guarded["contract"])
        self.assertTrue(guarded["prompt"].startswith(old["prompt"]))
        self.assertNotEqual(old["prompt_hash"], guarded["prompt_hash"])
        self.assertIn("preserve its actual number of people", guarded["prompt"])
        self.assertEqual(self.compile(enabled=False, composition_guard="single-scene-v1")["prompt"], self.selection["prompt"])

    def test_unknown_composition_guard_rejected(self):
        with self.assertRaises(ValueError):
            self.compile(composition_guard="guess-count")

    def test_style_image_disallows_selection(self):
        with self.assertRaises(ValueError):
            self.compile(style_reference_images=["tid.png"])

    def test_protected_or_nonexistent_region_is_rejected(self):
        for regions in (["skin"], ["new flower"]):
            with self.assertRaises(ValueError):
                self.compile(authorized_regions=regions)

    def test_missing_binding_or_roles_is_rejected(self):
        with self.assertRaises(ValueError):
            self.compile(bindings=[])
        bindings = copy.deepcopy(self.selection["bindings"])
        bindings[0]["role"] = "accent"
        with self.assertRaises(ValueError):
            self.compile(bindings=bindings)

    def test_interior_ratios_do_not_leak(self):
        with self.assertRaises(ValueError):
            self.compile(area_mode="interior-three-level")

    def test_unknown_tone_is_rejected(self):
        with self.assertRaises(ValueError):
            self.compile(tone_code="invented")

    def test_frozen_knowledge_tampering_is_rejected(self):
        knowledge = copy.deepcopy(self.knowledge)
        knowledge["tones"][0]["description"] = "changed"
        with self.assertRaises(ValueError):
            compile_generation_palette(knowledge, self.selection)

    def test_reference_state_must_be_explicit_list(self):
        with self.assertRaises(ValueError):
            self.compile(style_reference_images="")
        selection = copy.deepcopy(self.selection)
        del selection["style_reference_images"]
        with self.assertRaises(ValueError):
            compile_generation_palette(self.knowledge, selection)

    def test_all_existing_rules_are_retrieval_only(self):
        self.assertEqual(len(self.knowledge["retrieval_rules"]), 430)
        self.assertFalse(self.knowledge["existing_clause_index"]["auto_inject"])

    def test_duplicate_binding_is_rejected(self):
        bindings = copy.deepcopy(self.selection["bindings"])
        bindings.append(bindings[0])
        with self.assertRaises(ValueError):
            self.compile(bindings=bindings)

    def test_equal_and_interior_modes_compile_when_explicit(self):
        palette = self.knowledge["palettes"][0]
        regions = ["existing region " + str(i) for i in range(5)]
        for mode, roles in (("equal-opposition", ["family-a", "family-b", "family-a", "family-b", "family-a"]),
                            ("interior-three-level", ["base", "auxiliary", "accent", "auxiliary", "accent"])):
            result = self.compile(palette_id=palette["id"], existing_regions=regions,
                authorized_regions=regions, protected_regions=[], scene_kind="interior", area_mode=mode,
                bindings=[{"region": region, "role": role, "colour_order": c["order"]}
                          for region, role, c in zip(regions, roles, palette["colours"])])
            self.assertEqual("60-70%" in result["prompt"], mode == "interior-three-level")

    def test_palette_cannot_silently_drop_colours(self):
        with self.assertRaises(ValueError):
            self.compile(bindings=self.selection["bindings"][:2])


if __name__ == "__main__":
    unittest.main()
