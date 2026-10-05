"""Offline generation wiring and clothing-audit contracts (also runnable with unittest)."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("IMAGE_MAKER_TEST_OUTPUT", "1")

from PIL import Image
from utils.wardrobe import build_wardrobe_spec, apply_wardrobe, audit_wardrobe, record_wardrobe_audit
from utils.analysis_gen import build_first_pass_request


def _historic_pack_metadata(path):
    """Check frozen request contracts without requiring local art-image files.

    Real attachment bytes are checked by the CLI preflight; these unit tests
    exercise the historical manifest and composition, using its pinned digests.
    """
    from utils.wardrobe_experiment import load_frozen_generation_pack
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    refs = {p: h for r in raw["requests"] for p, h in zip(r["image_paths"], r["reference_sha256"])}
    with patch("utils.wardrobe_experiment.sha256_file", side_effect=lambda p: refs[p]):
        return load_frozen_generation_pack(path)


class WardrobeContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "candidate.png"
        Image.new("RGB", (100, 140), "red").save(self.image)
        self.spec = build_wardrobe_spec("lolita")

    def test_source_analysis_unchanged_and_size_and_style_independent(self):
        source = {"gpt_image_prompt": "Long-haired woman wearing a red tracksuit in a library."}
        original = copy.deepcopy(source)
        styles = {"ink": {"prompt_gpt": "Palette: red\nBrushwork: ink", "skip_repaint": False}}
        result = build_first_pass_request(styles, "ink", source, size="1536x1024", wardrobe=self.spec)
        self.assertEqual(source, original)
        self.assertEqual(result["requested_size"], "1536x1024")
        self.assertEqual(result["style_name"], "ink")
        self.assertEqual(result["image_paths"], [])
        self.assertFalse(result["skip_identity_refine"])
        self.assertFalse(result["skip_repaint"])
        self.assertIn("Japanese Lolita fashion", result["prompt"])
        self.assertIn("red tracksuit", result["prompt"])
        self.assertEqual(result["wardrobe"], self.spec)
        self.assertIn("NEW outfit", result["identity_correction_clauses"][-1].replace("newly generated clothing", "NEW outfit"))

    def test_unspecified_clothing_still_receives_design_without_age_requirement(self):
        result = build_first_pass_request({}, "", {"gpt_image_prompt": "A long-haired girl."}, wardrobe=self.spec)
        self.assertIn("When clothing is unspecified", result["prompt"])
        self.assertNotIn("age 21", result["prompt"])
        self.assertNotIn("ADULT", result["prompt"])

    def test_off_is_byte_identical_and_does_not_load_templates(self):
        with patch("utils.wardrobe.read_prompt_file", side_effect=AssertionError("off loaded templates")):
            self.assertEqual(build_wardrobe_spec(), {})
            self.assertEqual(apply_wardrobe("original\n", {}), "original\n")
        base = build_first_pass_request({}, "", {"gpt_image_prompt": "A person"})
        off = build_first_pass_request({}, "", {"gpt_image_prompt": "A person"}, wardrobe={})
        self.assertEqual(base, off)

    def test_policies_and_explicit_constraints_survive_generation(self):
        for policy in ("fill_missing", "reinterpret", "replace"):
            with self.subTest(policy=policy):
                spec = build_wardrobe_spec("lolita", policy)
                result = build_first_pass_request({}, "", {}, content_text="Keep my red coat.", wardrobe=spec)
                self.assertIn("Keep my red coat.", result["prompt"])
                self.assertIn("requirements take precedence", result["prompt"])
                self.assertIn("Application policy: " + policy, result["prompt"])

    def test_snapshot_tampering_and_invalid_names_rejected(self):
        broken = {**self.spec, "prompt": "different"}
        with self.assertRaises(ValueError):
            apply_wardrobe("source", broken)
        for name, policy in (("missing", "reinterpret"), ("lolita", "bad")):
            with self.assertRaises(ValueError):
                build_wardrobe_spec(name, policy)

    def test_prompt_stays_compact(self):
        self.assertLess(len(self.spec["prompt"]), 4000)

    def test_preserve_and_adapt_rules_scope_complete_design_vocabulary(self):
        for name in ("lolita-sweet", "cute-lingerie"):
            fill = build_wardrobe_spec(name, "fill_missing")["prompt"]
            self.assertLess(fill.index("PRESERVE:"), fill.index("CONDITIONAL DESIGN VOCABULARY"))
            self.assertIn("skip this entire vocabulary", fill)
            self.assertIn("not missing clothing information", fill)
            adapt = build_wardrobe_spec(name, "reinterpret")["prompt"]
            self.assertIn("no added skirt or apron", adapt)
            self.assertIn("does not become underwear", adapt)
            replace = build_wardrobe_spec(name, "replace")["prompt"]
            self.assertIn("If the whole outfit is protected, skip all target design", replace)

    def test_lingerie_uses_adult_clothing_only_on_both_generation_paths(self):
        from utils.styles import build_ref_gen_params
        spec = build_wardrobe_spec("cute-lingerie")
        content = "A 32-year-old adult woman with brown skin wearing an orange hoodie and jeans."
        styles = {"ink": {"prompt_gpt": "Brushwork: ink", "skip_identity_refine": True}}
        request = build_first_pass_request(styles, "ink", {}, content_text=content, wardrobe=spec)
        self.assertEqual(request["image_paths"], [])
        self.assertFalse(request["skip_identity_refine"])
        self.assertIn("matching high-waisted briefs", request["prompt"])
        self.assertIn("opaque fabric", request["prompt"])
        self.assertIn("explicitly states an adult age of 21 or older", spec["prompt"])
        self.assertLess(spec["prompt"].index("SUBJECT ELIGIBILITY"), spec["prompt"].index("COMPLETE:"))
        self.assertNotIn("Palette:", spec["prompt"])
        self.assertNotIn("cute-lingerie-wardrobe.png", request["prompt"])
        _, _, refs = build_ref_gen_params({}, "", "interleave")
        self.assertEqual(refs, [])
        self.assertIn(spec["prompt"], apply_wardrobe(content, spec))
        with patch("utils.wardrobe.read_prompt_file", side_effect=AssertionError("history reloaded")):
            self.assertEqual(apply_wardrobe(content, spec), content + "\n\n" + spec["prompt"])

    def test_wardrobe_audit_understands_preserve_and_lingerie_targets(self):
        from utils.prompt_loader import read_prompt_file
        rule = read_prompt_file("wardrobe/audit.md")
        self.assertIn("correctly preserved non-Lolita outfit can pass", rule)
        self.assertIn("underwear requires covered connected upper and lower garments", rule)

    def test_direct_output_snapshot_binds_actual_prompt_image_and_frozen_rules(self):
        from utils.wardrobe import record_wardrobe_outputs
        record_wardrobe_outputs([str(self.image)], self.spec, "actual request")
        record = json.loads(self.image.with_suffix(".wardrobe.json").read_text(encoding="utf-8"))
        self.assertEqual(record["wardrobe"], self.spec)
        self.assertFalse(record["effect_reviewed"])
        self.assertEqual(record["image_sha256"], __import__("hashlib").sha256(self.image.read_bytes()).hexdigest())
        self.assertEqual(record["prompt_sha256"], __import__("hashlib").sha256(b"actual request").hexdigest())
        with patch("utils.wardrobe.read_prompt_file", side_effect=AssertionError("off read rules")):
            record_wardrobe_outputs(["missing.png"], {}, "unchanged")

    def test_prompt_file_already_containing_snapshot_is_not_injected_twice(self):
        prompt = apply_wardrobe("body", self.spec)
        self.assertEqual(apply_wardrobe(prompt, self.spec), prompt)
        with self.assertRaisesRegex(ValueError, "重复"):
            apply_wardrobe(prompt + "\n" + self.spec["prompt"], self.spec)

    def test_direct_cli_wardrobe_and_both_sites_preserve_user_images_and_save_snapshot(self):
        import tools.gpt_image2_gen as cli
        config = self.root / "cli.json"
        config.write_text("{}", encoding="utf-8")
        for site in cli.SITE_CHOICES:
            with patch.object(cli, "get_api_config", return_value={"api_key": "unused", "model": "test"}), \
                 patch.object(cli, "_run_one", return_value=[str(self.image)]) as run, \
                 patch.object(cli, "_emit"):
                self.assertEqual(cli.main(["--config", str(config), "--site", site, "--prompt", "A 32-year-old adult woman.",
                                           "--wardrobe", "cute-lingerie", "--image", str(self.image)]), 0)
            args, _, _, prompt, _ = run.call_args.args
            self.assertEqual(args.image, [str(self.image)])
            self.assertEqual(args.wardrobe_policy, "replace")
            self.assertIn("matching high-waisted briefs", prompt)
            self.assertTrue(self.image.with_suffix(".wardrobe.json").exists())
        with patch.object(cli, "_emit"), patch.object(cli, "_run_one", side_effect=AssertionError("sent")):
            self.assertEqual(cli.main(["--list-wardrobes", "--config", "missing.json"]), 0)
            self.assertEqual(cli.main(["--repaint", "--wardrobe", "lolita"]), 2)

    def test_five_families_are_independent_of_art_and_default_to_complete_conversion(self):
        from utils.wardrobe import wardrobe_presets
        for family in ("classical", "sweet", "gothic", "chinese", "japanese"):
            spec = build_wardrobe_spec("lolita-" + family)
            result = build_first_pass_request({}, "", {"gpt_image_prompt": "A long-haired girl."}, wardrobe=spec)
            self.assertEqual(spec["policy"], "replace")
            self.assertEqual(result["image_paths"], [])
            self.assertEqual(result["style_name"], "")
            self.assertEqual(result["prompt"].count(spec["prompt"]), 1)
            self.assertIn("do not merely add lace", spec["prompt"])
            self.assertEqual(wardrobe_presets()[spec["name"]]["provenance"], "manual_visual_reference_review")

    def test_old_clothing_presets_are_hidden_and_conflicting_authorities_rejected(self):
        from utils.styles import enabled_style_names
        styles = {"ink": {"prompt_gpt": "Brushwork: ink"},
                  "cute-lingerie-wardrobe": {"enabled": True},
                  "custom-clothes": {"kind": "wardrobe"}}
        self.assertEqual(enabled_style_names(styles), ["ink"])
        for name in ("cute-lingerie-wardrobe", "custom-clothes"):
            with self.assertRaisesRegex(ValueError, "不能"):
                build_first_pass_request(styles, name, {}, wardrobe=self.spec)

    def test_wardrobe_does_not_inherit_blanket_identity_exemption_from_art(self):
        styles = {"ink": {"prompt_gpt": "Palette: pink\nBrushwork: ink", "skip_identity_refine": True}}
        result = build_first_pass_request(styles, "ink", {}, content_text="Keep a red garment and black hair.", wardrobe=self.spec)
        self.assertFalse(result["skip_identity_refine"])
        self.assertIn("Keep a red garment and black hair", result["prompt"])
        self.assertIn("Style palettes cannot override explicit colours", result["prompt"])

    def test_legacy_snapshot_is_reused_without_reading_current_rules(self):
        with patch("utils.wardrobe.read_prompt_file", side_effect=AssertionError("history reloaded rules")):
            self.assertIn(self.spec["prompt"], apply_wardrobe("historic", self.spec))

    def test_repairs_protect_current_outfit_without_reissuing_replacement(self):
        from utils.refine_quality import repair_style_guard
        guard = repair_style_guard({"wardrobe": self.spec})
        self.assertIn("current image", guard)
        self.assertIn("Do not restore", guard)
        self.assertNotIn("Application policy", guard)

    def _audit(self, response):
        with patch("utils.analysis_gpt_prompt.load_text_api_config", return_value={"base_url": "unused", "api_key": "unused", "model": "test"}), \
             patch("utils.send_budget.guarded_text_call", return_value=json.dumps(response)) as call:
            result = audit_wardrobe(str(self.image), self.spec, "Keep red fabric.")
        self.assertIn("Keep red fabric", call.call_args.args[6])
        self.assertEqual(result["candidate_sha256"], __import__("utils.refine_quality", fromlist=["file_sha256"]).file_sha256(str(self.image)))
        return result

    @staticmethod
    def _valid():
        return {"conforms": True, "uncertain": False, "summary": "裙型与层次符合目标",
                "checks": [{"dimension": dim, "result": "pass", "evidence": "可见结构连接完整"}
                           for dim in ("silhouette", "construction", "trim", "coordination", "explicit_constraints")]}

    def test_valid_audit_binds_pixels_and_target(self):
        result = self._audit(self._valid())
        self.assertTrue(result["accepted"])
        self.assertEqual(result["wardrobe_sha256"], self.spec["prompt_sha256"])

    def test_missing_string_bool_uncertainty_and_contradictions_never_pass(self):
        for response in ({}, {**self._valid(), "conforms": "true"},
                         {**self._valid(), "uncertain": True},
                         {**self._valid(), "checks": [{"dimension": dim, "result": "fail", "evidence": "裙型不符"}
                                                      for dim in ("silhouette", "construction", "trim", "coordination", "explicit_constraints")]}):
            with self.subTest(response=response):
                self.assertFalse(self._audit(response)["accepted"])

    def test_unobservable_clothing_is_not_a_success(self):
        response = self._valid()
        for check in response["checks"]:
            check["result"] = "not_visible"
        self.assertFalse(self._audit(response)["accepted"])

    def test_rejected_audit_saved_with_readable_reason_before_raising(self):
        target = self.root / "wardrobe-audit.json"
        with self.assertRaisesRegex(RuntimeError, "无法确认"):
            record_wardrobe_audit({"accepted": False, "summary": "无法确认衣装", "checks": []}, str(target))
        self.assertTrue(target.exists())
        self.assertIn("无法确认衣装", target.with_suffix(".txt").read_text(encoding="utf-8"))

    def test_identity_audit_receives_authorized_scope(self):
        from utils.identity_audit import audit_image_identity
        with patch("utils.identity_audit.guarded_text_call", return_value=json.dumps({"mismatch": False, "severity": "none", "differences": []})) as call:
            audit_image_identity(str(self.image), {}, text_cfg={"base_url": "unused", "api_key": "unused", "model": "test"},
                                 expected_prompt="brown hair, violet eyes, original school uniform", wardrobe=self.spec)
        user = call.call_args.args[6]
        self.assertIn("AUTHORIZED CLOTHING SCOPE", user)
        self.assertIn("Other identity facts remain fully protected", user)

    def test_manifest_and_legacy_import_preserve_wardrobe_snapshot(self):
        from utils.analysis_gen import save_generation_manifest
        from utils.generation_checkpoint import import_generation_checkpoint
        request = build_first_pass_request({}, "", {"gpt_image_prompt": "Long hair."}, wardrobe=self.spec)
        path = save_generation_manifest(str(self.image), request, model="test", size="1024x1536", quality="high", mode="generate", steps={}, firmware="")
        _, checkpoint = import_generation_checkpoint(path, {})
        recovered = checkpoint["snapshot"]["request_payload"]
        self.assertEqual(recovered["wardrobe"], self.spec)
        self.assertEqual(recovered["prompt"], request["prompt"])

    def test_failed_clothing_gate_blocks_publish_and_resume_reuses_paid_work(self):
        import modules.image_analysis.single_analyzer as sa
        request = build_first_pass_request({}, "", {"gpt_image_prompt": "Long hair."}, wardrobe=self.spec)
        checkpoint = str(self.root / "generation-checkpoint.json")
        def worker(analysis_result=None):
            return sa.GptImageGenWorkerThread(request_payload=request, steps={},
                        analysis_result={"task_hash": "abc12345"} if analysis_result is None else analysis_result,
                        checkpoint_path=checkpoint)
        audit = {"needs_refine": False, "severity": "none", "structural_issues": [], "line_issues": [],
                 "background_drift": [], "style_gaps": [], "ownership_uncertain": [], "summary": "结构通过"}
        with patch("modules.others.api_backend.generate_image_aigc2d_gpt", return_value=[str(self.image)]) as generate, \
             patch("utils.refine_quality.audit_hand_quality", return_value=audit), \
             patch("utils.wardrobe.audit_wardrobe", return_value={"accepted": False, "summary": "目标裙型不符", "checks": []}) as clothing, \
             patch("utils.analysis_gen.publish_final_output") as publish:
            first = worker()
            first.run()
            self.assertEqual(first.last_status, "error")
            self.assertEqual(first.failed_stage, "final_review")
            second = worker()
            second.run()
            self.assertEqual(second.last_status, "error")
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(clothing.call_count, 1)
            publish.assert_not_called()
            # Even without a loaded analysis JSON, the independent wardrobe gate applies.
            request_without_analysis = worker({})
            request_without_analysis.run()
            self.assertEqual(request_without_analysis.last_status, "error")
            publish.assert_not_called()
        self.assertTrue(self.image.exists())
        self.assertIn("目标裙型不符", (self.root / "wardrobe-audit.txt").read_text(encoding="utf-8"))

    def test_analysis_ui_memory_merges_other_settings_and_restores_wardrobe(self):
        from modules.image_analysis.single_analyzer import save_analysis_gpt_ui, load_analysis_gpt_ui
        config = self.root / "config.json"
        config.write_text('{"unrelated":{"keep":true}}', encoding="utf-8")
        selection = {"name": "lolita", "policy": "fill_missing"}
        save_analysis_gpt_ui({"wardrobe_selection": selection}, str(config))
        self.assertEqual(load_analysis_gpt_ui(str(config))["wardrobe_selection"], selection)
        self.assertEqual(json.loads(config.read_text(encoding="utf-8"))["unrelated"], {"keep": True})


class FrozenWardrobeGeneration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pack = json.loads(Path("prompts/wardrobe/round4/generation-pack.json").read_text(encoding="utf-8"))
        self.pack["requests"] = [next(row for row in self.pack["requests"] if row["channel"] == "gpt")]
        self.pack["generation_http_attempts"] = 1
        self.path = self.root / "pack.json"
        self._save()

    def _save(self):
        from utils.wardrobe_experiment import frozen_pack_digest
        self.pack["sha256"] = frozen_pack_digest(self.pack)
        self.path.write_text(json.dumps(self.pack), encoding="utf-8")

    def _run(self, flag):
        import tools.wardrobe_experiment as cli
        with patch.object(cli, "BASE_DIR", str(self.root)):
            return cli.main(["--frozen-pack", str(self.path), "--run-dir", str(self.root / "data/test-result/run"), flag])

    def test_actual_pack_covers_five_families_both_channels_and_art_compatibility(self):
        from utils.wardrobe_experiment import load_frozen_generation_pack
        pack = _historic_pack_metadata("prompts/wardrobe/round4/generation-pack.json")
        self.assertEqual(len(pack["requests"]), 40)
        for channel in ("gpt", "gemini"):
            rows = [row for row in pack["requests"] if row["channel"] == channel]
            self.assertEqual(len(rows), 20)
            self.assertEqual({row["family"] for row in rows}, {"classical", "sweet", "gothic", "chinese", "japanese"})
            self.assertEqual(sum(bool(row["art_style"]) for row in rows), 2)
            self.assertTrue(all(not row["image_paths"] for row in rows if not row["art_style"]))

    def test_round5_covers_legacy_target_and_isolates_art_image_attachment(self):
        from utils.wardrobe_experiment import load_frozen_generation_pack
        pack = _historic_pack_metadata("prompts/wardrobe/round5/generation-pack.json")
        self.assertEqual(len(pack["requests"]), 28)
        for channel in ("gpt", "gemini"):
            rows = [r for r in pack["requests"] if r["channel"] == channel]
            self.assertEqual(len(rows), 14)
            self.assertEqual(sum(r["wardrobe"]["name"] == "cute-lingerie" for r in rows), 5)
            for style in ("waterink-style", "kishida-mel-style"):
                a, b = [r for r in rows if r["art_style"] == style]
                self.assertEqual(a["prompt"], b["prompt"])
                self.assertEqual(a["params"], b["params"])
                self.assertEqual(len(a["image_paths"]), 1)
                self.assertEqual(b["image_paths"], [])

    def test_prepare_is_network_free_and_creates_no_run_directory(self):
        with patch("modules.others.api_backend.generate_image_aigc2d_gpt", side_effect=AssertionError("HTTP")) as gen:
            self.assertEqual(self._run("--prepare"), 0)
            gen.assert_not_called()
        self.assertFalse((self.root / "data/test-result/run").exists())

    def test_duplicate_rule_and_extra_images_are_rejected_before_output(self):
        row = self.pack["requests"][0]
        row["params"]["n"] = 2
        self._save()
        self.assertEqual(self._run("--prepare"), 1)
        row["params"]["n"] = 1
        row["prompt"] += "\n" + row["wardrobe"]["prompt"]
        from utils.wardrobe_experiment import sha256_text
        row["prompt_sha256"] = sha256_text(row["prompt"])
        self._save()
        self.assertEqual(self._run("--prepare"), 1)
        self.assertFalse((self.root / "data/test-result/run").exists())

    def test_success_reuses_bound_pixels_and_changed_output_is_not_regenerated(self):
        image = self.root / "result.png"
        Image.new("RGB", (20, 30), "red").save(image)
        with patch("modules.others.api_backend.generate_image_aigc2d_gpt", return_value=[str(image)]) as gen:
            self.assertEqual(self._run("--generate"), 0)
            self.assertEqual(self._run("--generate"), 0)
            self.assertEqual(gen.call_count, 1)
            Image.new("RGB", (20, 30), "blue").save(image)
            self.assertEqual(self._run("--generate"), 1)
            self.assertEqual(gen.call_count, 1)

    def test_failed_attempt_is_recorded_and_not_automatically_recharged(self):
        with patch("modules.others.api_backend.generate_image_aigc2d_gpt", side_effect=RuntimeError("503")) as gen:
            self.assertEqual(self._run("--generate"), 1)
            self.assertEqual(self._run("--generate"), 1)
            self.assertEqual(gen.call_count, 1)
        sample = json.loads((self.root / "data/test-result/run/samples.json").read_text(encoding="utf-8"))["samples"][0]
        self.assertEqual(sample["status"], "error")
        self.assertIn("503", sample["error"])


class _Signal:
    def connect(self, *_args):
        pass


class _FakeWorker:
    last = None

    def __init__(self, *args, **kwargs):
        type(self).last = self
        self.kwargs = kwargs
        self.log_signal = _Signal()
        self.finish_signal = _Signal()
        self.finished = _Signal()

    def start(self):
        pass


class WardrobeUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        import modules.image_generation.gpt_image2_tab as gt
        cfg_patch = patch.object(gt, "get_api_config", return_value={"api_key": "unused", "model": "gpt-image-2",
            "base_url": "http://unused", "quality": "high", "output_format": "png"})
        cfg_patch.start()
        self.addCleanup(cfg_patch.stop)

    def _analyzer(self):
        import modules.image_analysis.single_analyzer as sa
        self.sa = sa
        for p in (patch.object(sa, "analysis_gpt_ui_path", return_value=str(self.root / "config.json")),
                  patch.object(sa, "list_esrgan_models", return_value=[]),
                  patch.object(sa, "WorkerThread", _FakeWorker),
                  patch.object(sa, "GptImageGenWorkerThread", _FakeWorker),
                  patch.object(sa, "ImageGenWorkerThread", _FakeWorker)):
            p.start()
            self.addCleanup(p.stop)
        cfg = {"base_url": "http://unused", "api_key": "test", "model": "test", "api_type": "aigc2d"}
        w = sa.SingleAnalyzerWidget(lambda nsfw=False: ("http://unused", "test", "test"), lambda: cfg, lambda: {}, lambda: None,
                                   upscale_options_getter_func=lambda: {}, persist_ui_state=False)
        w._start_image_gen_runtime = lambda *_args: None
        self.addCleanup(w.deleteLater)
        return w

    def test_selector_separates_painting_and_wardrobe_with_disabled_off_policy(self):
        from utils.wardrobe_widget import WardrobeSelector
        w = WardrobeSelector()
        self.addCleanup(w.deleteLater)
        self.assertEqual(w.snapshot(), {})
        self.assertFalse(w.policy_combo.isEnabled())
        w.restore({"name": "lolita", "policy": "replace"})
        self.assertTrue(w.policy_combo.isEnabled())
        self.assertEqual(w.snapshot()["policy"], "replace")

    def test_lingerie_selector_and_analysis_entrypoints_use_clothing_snapshot(self):
        w = self._analyzer()
        w.wardrobe_selector.restore({"name": "cute-lingerie", "policy": "replace"})
        spec = w.wardrobe_selector.snapshot()
        bundle = {"task_hash": "abc12345", "wardrobe": spec, "style_name": "",
                  "original_prompt": "A 32-year-old adult woman in an orange hoodie.",
                  "refined_prompt": "A 32-year-old adult woman in an orange hoodie.", "aspect_ratio": "2:3"}
        w._start_gpt_image_thread("refined", bundle, "abc12345", "", {})
        request = _FakeWorker.last.kwargs["request_payload"]
        self.assertEqual(request["wardrobe"], spec)
        self.assertEqual(request["image_paths"], [])
        self.assertIn("matching high-waisted briefs", request["prompt"])
        w.trigger_image_generation("refined", prompt_bundle=bundle, channel="gemini")
        self.assertIn(spec["prompt"], _FakeWorker.last.kwargs["prompt"])
        self.assertEqual(bundle["original_prompt"], "A 32-year-old adult woman in an orange hoodie.")
        w._active_img_threads.clear()

    def test_lingerie_direct_gpt_tab_supports_both_sites_without_legacy_reference(self):
        import modules.image_generation.gpt_image2_tab as gt
        cfg = self.root / "config.json"
        cfg.write_text("{}", encoding="utf-8")
        with patch.object(gt, "CONFIG_IMAGE_FILE", str(cfg)), patch.object(gt, "save_repaint_config"):
            tab = gt.GptImage2Widget()
            self.addCleanup(tab.deleteLater)
            tab.prompt_edit.setPlainText("A 32-year-old adult woman in an orange hoodie.")
            tab.wardrobe_selector.restore({"name": "cute-lingerie", "policy": "replace"})
            for site in (gt.SITE_AIGC2D, gt.SITE_AUTODL):
                tab.site_combo.setCurrentIndex(tab.site_combo.findData(site))
                _, params = tab.build_request()
                self.assertEqual(params["image_paths"], [])
                self.assertIn("matching high-waisted briefs", params["prompt"])
            repaint = tab.build_repaint_request([], wardrobe=tab.wardrobe_selector.snapshot())
            self.assertIn("newly generated clothing", repaint["prompt"])
            self.assertNotIn("Application policy", repaint["prompt"])

    def test_gemini_art_reference_does_not_repeat_complete_wardrobe_design(self):
        w = self._analyzer()
        spec = build_wardrobe_spec("lolita-sweet")
        ref = self.root / "art.png"
        Image.new("RGB", (20, 30)).save(ref)
        bundle = {"task_hash": "abc12345", "wardrobe": spec, "style_name": "",
                  "original_prompt": "An adult woman.", "refined_prompt": "An adult woman.", "aspect_ratio": "2:3"}
        with patch.object(self.sa, "build_ref_gen_params", return_value=("paint", "reference guard", [str(ref)])):
            w.trigger_image_generation("refined", prompt_bundle=bundle, channel="gemini")
        params = _FakeWorker.last.kwargs
        self.assertEqual(params["prompt"].count(spec["prompt"]), 1)
        self.assertNotIn(spec["prompt"], params["post_instructions"])
        self.assertIn("separately stated wardrobe policy", params["post_instructions"])
        w._active_img_threads.clear()

    def test_direct_worker_keeps_snapshot_in_generation_and_repaint_outputs(self):
        import modules.image_generation.gpt_image2_tab as gt
        from utils.wardrobe import wardrobe_continuity
        spec = build_wardrobe_spec("cute-lingerie")
        source, final = self.root / "first.png", self.root / "final.png"
        Image.new("RGB", (20, 30)).save(source)
        Image.new("RGB", (40, 60)).save(final)
        body = "Actual generation request."
        repair = wardrobe_continuity(spec)
        with patch.object(gt, "generate_image_repaint", return_value=[str(final)]):
            worker = gt.GptImage2Worker(lambda **kw: [str(source)], {"prompt": body},
                repaint_params={"prompt": repair}, wardrobe_spec=spec)
            worker.run()
        for path, stage, prompt in ((source, "generation", body), (final, "repaint", repair)):
            record = json.loads(path.with_suffix(".wardrobe.json").read_text(encoding="utf-8"))
            self.assertEqual(record["wardrobe"], spec)
            self.assertEqual(record["stage"], stage)
            self.assertEqual(record["prompt"], prompt)

    def test_queued_analysis_freezes_wardrobe_before_user_changes_selection(self):
        w = self._analyzer()
        image = self.root / "source.png"
        Image.new("RGB", (10, 20)).save(image)
        w.wardrobe_selector.restore({"name": "lolita"})
        thread = w._launch_analysis_task(str(image))
        self.assertIsNotNone(thread)
        w.wardrobe_selector.restore({})
        self.assertEqual(thread.meta_wardrobe["name"], "lolita")
        self.assertNotIn("wardrobe", thread.kwargs)  # analysis itself remains factual
        w._active_analysis_threads.clear()

    def test_analysis_gpt_and_gemini_entrypoints_use_same_independent_target(self):
        w = self._analyzer()
        spec = build_wardrobe_spec("lolita")
        bundle = {"task_hash": "abc12345", "wardrobe": spec, "style_name": "",
                  "original_prompt": "Long hair, red school uniform.", "refined_prompt": "Long hair, red school uniform.", "aspect_ratio": "2:3"}
        w._start_gpt_image_thread("refined", bundle, "abc12345", "", {})
        self.assertEqual(_FakeWorker.last.kwargs["request_payload"]["wardrobe"], spec)
        self.assertIn("Japanese Lolita fashion", _FakeWorker.last.kwargs["request_payload"]["prompt"])
        w.trigger_image_generation("refined", prompt_bundle=bundle, channel="gemini")
        self.assertIn("Japanese Lolita fashion", _FakeWorker.last.kwargs["prompt"])
        self.assertEqual(bundle["refined_prompt"], "Long hair, red school uniform.")
        w._active_img_threads.clear()

    def test_gpt_image_tab_two_sites_and_edit_apply_preset_without_new_reference(self):
        import modules.image_generation.gpt_image2_tab as gt
        cfg = self.root / "config.json"
        cfg.write_text("{}", encoding="utf-8")
        with patch.object(gt, "CONFIG_IMAGE_FILE", str(cfg)), patch.object(gt, "save_repaint_config"):
            tab = gt.GptImage2Widget()
            self.addCleanup(tab.deleteLater)
            tab.prompt_edit.setPlainText("A long-haired girl.")
            tab.wardrobe_selector.restore({"name": "lolita"})
            for site in (gt.SITE_AIGC2D, gt.SITE_AUTODL):
                tab.site_combo.setCurrentIndex(tab.site_combo.findData(site))
                _, params = tab.build_request()
                self.assertIn("Japanese Lolita fashion", params["prompt"])
                self.assertEqual(params["image_paths"], [])
            tab.mode_combo.setCurrentText(gt.MODE_EDIT)
            _, params = tab.build_request()
            self.assertIn("Japanese Lolita fashion", params["prompt"])
            tab.prompt_edit.setPlainText("red school uniform")
            repaint = tab.build_repaint_request([], wardrobe=tab.wardrobe_selector.snapshot())
            self.assertNotIn("red school uniform", repaint["prompt"])
            self.assertIn("newly generated clothing", repaint["prompt"])
            tab.save_defaults()
            self.assertEqual(json.loads(cfg.read_text(encoding="utf-8"))[gt.CONFIG_NODE]["wardrobe_selection"]["name"], "lolita")
            tab.mode_combo.setCurrentText(gt.MODE_REPAINT)
            self.assertFalse(tab.wardrobe_selector.isEnabled())
            tab.resize(1100, 750)
            tab.show()
            self.app.processEvents()
            self.assertLess(tab.minimumSizeHint().width(), 1100)
            tab.close()


if __name__ == "__main__":
    unittest.main()
