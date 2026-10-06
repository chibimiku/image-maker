"""按字段合并验证的回归用例（不联网、不发付费请求）。

覆盖：
1. 候选包按字段复制后逐字不变（不允许规范化悄悄改写），第 5 轮不合格短版不放行 GPT 通道；
2. 预检与实跑共用同一段组装代码（`_assemble_test_generation_requests`），且命中真实发送参数；
3. 跨分组复用只在**实际传输内容**逐字一致时发生（A 与 B 的 GPT 首图/重绘可复用，
   Gemini 直出不可复用；C 两条 GPT 通道都不可复用）。
"""

import json
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.image_analysis.style_merge_validation import (  # noqa: E402
    MergeValidationTask,
    PROBE_CHANNELS,
    ValidationError,
    build_candidate_package,
    build_precheck,
    build_state,
    channel_digests,
    run_attempt,
)

SOURCE = REPO_ROOT / "data" / "20261004" / "style-extraction" / "myc0t0xin" / \
    "run-20261004-225929-156628" / "myc0t0xin_style_iter_result.json"


class _PatchGroup:
    """把若干 `patch.object` 合成一个上下文管理器（不用 ExitStack 之外的语法糖）。"""

    def __init__(self, patches):
        self._patches = list(patches)
        self._stack = ExitStack()

    def __enter__(self):
        try:
            for item in self._patches:
                self._stack.enter_context(item)
        except BaseException:
            self._stack.close()
            raise
        return self

    def __exit__(self, *exc_info):
        return self._stack.__exit__(*exc_info)


@unittest.skipUnless(SOURCE.is_file(), "缺少 myc0t0xin 源结果，跳过按字段合并验证用例")
class MergeValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        # 显式指定模型：用例不依赖 .env 里的 IMAGE_MAKER_NODES，避免环境差异
        self.overrides = {"gpt_model": "gpt-image-2", "gemini_model": "gemini-3-pro-image-preview"}
        self.task = MergeValidationTask(source=str(SOURCE), output_root=str(self.root / "run"),
                                        subject_label="primary", model_overrides=self.overrides)
        self.images = [str(self.root / f"ref-{index}.png") for index in range(12)]
        for index, path in enumerate(self.images):
            Image.new("RGB", (40, 60), (index * 15 % 255, 60, 120)).save(path)
        self.style_reference = str(self.root / "style-ref.png")
        Image.new("RGB", (40, 60), "white").save(self.style_reference)

    def tearDown(self):
        self.temporary.cleanup()

    def package(self, key):
        return build_candidate_package(self.task, key)

    # -- 1. 字段复制 ------------------------------------------------------ #

    def test_field_copy_is_byte_exact_and_motifs_are_off(self):
        state = self.task.source_state()
        for key, (gemini_round, gpt_round, clauses_round) in (
                ("A", ("round_2", "round_2", "round_2")),
                ("B", ("round_5", "round_2", "round_2")),
                ("C", ("round_5", "round_2", "round_5"))):
            entry = self.package(key)
            variants = state["test_images"][gemini_round]["prompt_variants"]
            self.assertEqual(entry["package"]["gemini_full_prompt"], variants["gemini_full_prompt"])
            self.assertEqual(entry["field_sources"]["prompt"]["round"], gemini_round)
            self.assertEqual(entry["field_sources"]["prompt_gpt"]["round"], gpt_round)
            self.assertEqual(entry["field_sources"]["repaint_clauses"]["round"], clauses_round)
            self.assertEqual(entry["package"]["gpt_image_prompt"],
                             state["test_images"][gpt_round]["prompt_variants"]["gpt_image_prompt"])
            self.assertEqual(entry["package"]["gemini_repaint_clauses"],
                             state["test_images"][clauses_round]["prompt_variants"]["gemini_repaint_clauses"])
            self.assertEqual(entry["package"]["optional_motifs"], [])
            self.assertTrue(all(entry["normalization"]["copy_checks"].values()))

    def test_round5_short_prompt_is_rejected_and_never_substituted(self):
        state = self.task.source_state()
        fifth = state["test_images"]["round_5"]["prompt_variants"]
        self.assertFalse(fifth["gpt_image_prompt_valid"])
        for key in ("B", "C"):
            entry = self.package(key)
            # 第 5 轮短版不合格：包 B/C 用的是第 2 轮的合格短版，不能拿第 5 轮顶上、也不能回退母版
            self.assertTrue(entry["normalization"]["gpt_image_prompt_valid"])
            self.assertFalse(entry["normalization"]["gpt_image_prompt_errors"])
            self.assertEqual(entry["package"]["gpt_image_prompt"],
                             state["test_images"]["round_2"]["prompt_variants"]["gpt_image_prompt"])
            self.assertNotEqual(entry["package"]["gpt_image_prompt"],
                                state["test_images"]["round_5"]["prompt_variants"]["gpt_image_prompt"])
            self.assertNotEqual(entry["package"]["gpt_image_prompt"],
                                entry["package"]["master_prompt"])

    # -- 2. 预检 = 实际发送 ------------------------------------------------- #

    def test_precheck_uses_the_real_assembly_code(self):
        entry = self.package("A")
        precheck = build_precheck(self.task, "A", entry, self.task.subject_prompt(),
                                  self.style_reference, self.images)
        self.assertEqual([item["channel"] for item in precheck["channels"]], list(PROBE_CHANNELS))
        self.assertIn("_assemble_test_generation_requests", precheck["assembler"])
        gemini = next(item for item in precheck["channels"] if item["channel"] == "gemini_direct")
        self.assertEqual(gemini["params"]["model"], "gemini-3-pro-image-preview")
        self.assertEqual(gemini["params"]["resolution"], "2K")
        self.assertEqual(gemini["params"]["face_quality_boost"], False)
        self.assertEqual([item["path"] for item in gemini["images"]], [os.path.abspath(self.style_reference)])
        # 组装出的 Gemini 提示词必须同时包含完整画风说明与固定测试主体
        self.assertIn(self.task.subject_prompt(), gemini["prompt"])
        self.assertIn(entry["package"]["gemini_full_prompt"][:60], gemini["prompt"])

        gpt = next(item for item in precheck["channels"] if item["channel"] == "gpt_first_pass")
        self.assertEqual(gpt["params"]["size"], "1024x1536")
        self.assertEqual(gpt["params"]["mode"], "generate")
        for clause in entry["package"]["gemini_repaint_clauses"]:
            self.assertIn(clause, gpt["prompt"])          # 条款确实进了首图请求

        repaint = next(item for item in precheck["channels"] if item["channel"] == "gpt_repainted")
        self.assertEqual(repaint["params"]["reference_mode"], "none")
        self.assertEqual(repaint["params"]["scope"], "full")
        self.assertTrue(repaint["params"]["clauses_without_image"])
        self.assertEqual(len(repaint["images"]), 1)        # 文字模式只发送本次 GPT 首图
        self.assertIn("STYLE LANGUAGE", repaint["prompt"])
        for clause in entry["package"]["gemini_repaint_clauses"]:
            self.assertIn(clause, repaint["prompt"])

    def test_precheck_blocks_gpt_channels_when_short_prompt_is_invalid(self):
        # 造一个「第 5 轮短版被塞进 gpt_image_prompt」的包：预检必须停下来，不发 GPT 请求
        broken = self.package("B")
        broken["package"] = dict(broken["package"], gpt_image_prompt="Palette: only one field",
                                 gpt_image_prompt_valid=False,
                                 gpt_image_prompt_errors=["缺字段: Lighting"])
        broken["normalization"] = dict(broken["normalization"], gpt_image_prompt_valid=False,
                                       gpt_image_prompt_errors=["缺字段: Lighting"])
        broken["gpt_channel_allowed"] = False
        precheck = build_precheck(self.task, "B", broken, self.task.subject_prompt(),
                                  self.style_reference, self.images)
        by_channel = {item["channel"]: item for item in precheck["channels"]}
        self.assertTrue(by_channel["gpt_first_pass"]["blocked"])
        self.assertIn("未发送", by_channel["gpt_first_pass"]["reason"])
        self.assertIn("缺字段: Lighting", by_channel["gpt_first_pass"]["reason"])
        self.assertTrue(by_channel["gpt_repainted"]["blocked"])
        # Gemini 直出不受影响，仍然完成组装
        self.assertFalse(by_channel["gemini_direct"]["blocked"])
        # 被拦下的通道不许留下伪造的「输入图」，否则会被当成可复用
        self.assertEqual(by_channel["gpt_first_pass"]["image_sha256"], [])

    def test_precheck_blocks_gpt_channels_when_node_has_no_model(self):
        from modules.others import api_backend
        from modules.image_analysis import style_analyzer
        real = api_backend.get_api_config

        def fake_config(api_type=None, **kwargs):
            config = real(api_type=api_type, **kwargs)
            if api_type == "aigc-2d-gpt":
                return {**config, "model": ""}
            return config

        # 组装点在 style_analyzer 里按名字取 get_api_config（导入时绑定），所以要打在那里
        # 强制 GPT 模型为空（显式空覆盖），模拟「节点没配模型」
        no_gpt = MergeValidationTask(source=str(SOURCE),
                                     output_root=str(self.root / "no-gpt-model"),
                                     subject_label="primary",
                                     model_overrides={"gemini_model": "gemini-3-pro-image-preview",
                                                      "gpt_model": ""})
        with patch.object(style_analyzer, "get_api_config", side_effect=fake_config):
            entry = build_candidate_package(no_gpt, "A")
            precheck = build_precheck(no_gpt, "A", entry, no_gpt.subject_prompt(),
                                      self.style_reference, self.images)
        by_channel = {item["channel"]: item for item in precheck["channels"]}
        self.assertIn("blocked", by_channel["gemini_direct"])
        self.assertFalse(by_channel["gemini_direct"]["blocked"])
        self.assertTrue(by_channel["gpt_first_pass"]["blocked"])
        self.assertIn("未配置 GPT 图片模型", by_channel["gpt_first_pass"]["reason"])

    def test_field_snapshot_matches_source_round_hashes(self):
        from modules.image_analysis.style_merge_validation import _field_snapshot, text_hash
        state = self.task.source_state()
        snapshot = _field_snapshot(state["test_images"]["round_2"], "gemini_full_prompt")
        self.assertEqual(snapshot["sha256"],
                         text_hash(state["test_images"]["round_2"]["prompt_variants"]["gemini_full_prompt"]))
        clauses = _field_snapshot(state["test_images"]["round_5"], "gemini_repaint_clauses")
        self.assertEqual(clauses["count"], 7)
        self.assertEqual(self.package("C")["field_sources"]["repaint_clauses"]["sha256"],
                         clauses["sha256"])

    # -- 3. 跨分组复用 ------------------------------------------------------ #

    def test_reuse_only_when_transmitted_content_matches(self):
        entries = {key: self.package(key) for key in ("A", "B", "C")}
        prechecks = {}
        for key, entry in entries.items():
            prechecks[key] = build_precheck(self.task, key, entry, self.task.subject_prompt(),
                                            self.style_reference, self.images)
            entry["precheck"] = prechecks[key]
        digests = {key: prechecks[key]["channel_digests"] for key in entries}
        # A 与 B：第 2 轮短版 + 第 2 轮重绘条款 → 首图与重绘的实际请求逐字一致
        self.assertEqual(digests["A"]["gpt_first_pass"], digests["B"]["gpt_first_pass"])
        self.assertEqual(digests["A"]["gpt_repainted"], digests["B"]["gpt_repainted"])
        # …但 Gemini 直出用的是不同轮次的完整描述 → 不可复用
        self.assertNotEqual(digests["A"]["gemini_direct"], digests["B"]["gemini_direct"])
        self.assertEqual(digests["B"]["gemini_direct"], digests["C"]["gemini_direct"])
        # C 换了重绘条款：两条 GPT 通道的请求都变了（首图也带条款，不能只看配置字段）
        self.assertNotEqual(digests["A"]["gpt_first_pass"], digests["C"]["gpt_first_pass"])
        self.assertNotEqual(digests["A"]["gpt_repainted"], digests["C"]["gpt_repainted"])
        self.assertEqual(set(channel_digests(prechecks["A"]["channels"])), set(PROBE_CHANNELS))

    def test_run_attempt_reuses_identical_channel_without_calling_api(self):
        entries = {key: self.package(key) for key in ("A", "B")}
        for key in ("A", "B"):
            entries[key]["precheck"] = build_precheck(self.task, key, entries[key],
                                                      self.task.subject_prompt(),
                                                      self.style_reference, self.images)
        produced = {}

        def fake_assemble(self_worker, variants, style_ref, aspect_ratio, motif_prompt,
                          model_overrides=None):
            return {
                "gemini_direct": {"prompt": "GEMINI " + str(variants.get("gemini_full_prompt"))[:20],
                                  "image_paths": [style_ref], "model": "gemini-3-pro-image-preview",
                                  "aspect_ratio": aspect_ratio, "resolution": "2K",
                                  "api_type": "aigc2d", "face_quality_boost": False},
                "gpt_first_pass": {"prompt": "GPT", "image_paths": [style_ref], "model": "gpt-image-2",
                                   "aspect_ratio": aspect_ratio, "api_type": "aigc-2d-gpt",
                                   "mode": "generate", "quality": None, "size": None},
                "gpt_repaint": {"steps": {"repaint": {"enabled": True, "reference_mode": "none",
                                                      "scope": "full", "clauses_without_image": True}},
                                "style_ref_path": style_ref, "style_clauses": []},
            }

        def fake_gemini(*args, **kwargs):
            produced["gemini_direct"] = produced.get("gemini_direct", 0) + 1
            return [self._make_output("gemini-direct")]

        def fake_gpt(*args, **kwargs):
            produced["gpt_first_pass"] = produced.get("gpt_first_pass", 0) + 1
            return [self._make_output("gpt-first")]

        def fake_pipeline(*args, **kwargs):
            produced["gpt_repainted"] = produced.get("gpt_repainted", 0) + 1
            return [self._make_output("final-rp", ext=".jpg")]

        from modules.image_analysis import style_analyzer
        records = {}
        with self._patched_generation(fake_gemini, fake_gpt, fake_pipeline), \
                patch.object(style_analyzer.StyleIterativeWorkerThread,
                             "_assemble_test_generation_requests", fake_assemble):
            records[("A", 1)] = run_attempt(self.task, "A", entries["A"], 1, self.task.subject_prompt(),
                                            self.style_reference, self.images, records=records)
            records[("B", 1)] = run_attempt(self.task, "B", entries["B"], 1, self.task.subject_prompt(),
                                            self.style_reference, self.images, records=records)

        # A 三路各跑一次；B 只有 Gemini 直出需要真跑（首图与重绘复用 A）
        self.assertEqual(produced.get("gemini_direct"), 2)
        self.assertEqual(produced.get("gpt_first_pass"), 1)
        self.assertEqual(produced.get("gpt_repainted"), 1)
        reused = records[("B", 1)]["channel_reused"]
        self.assertEqual(set(reused), {"gpt_first_pass", "gpt_repainted"})
        self.assertEqual(reused["gpt_first_pass"]["package"], "A")
        self.assertEqual(records[("B", 1)]["generated_files"]["gpt_first_pass"],
                         records[("A", 1)]["generated_files"]["gpt_first_pass"])
        for name in PROBE_CHANNELS:
            self.assertTrue(all(os.path.isfile(path)
                                for path in records[("B", 1)]["generated_files"][name]))

        state = build_state(self.task, entries, records)
        from modules.image_analysis.style_comparison import candidates_from_state
        candidates = candidates_from_state(state)
        self.assertEqual(len(candidates), 4)
        self.assertEqual(state["sample_inventory"]["alias_count"], 2)
        self.assertTrue({c["record"]["test_prompt"] for c in candidates} == {self.task.subject_prompt()})

    def test_failed_second_run_keeps_first_products_and_resumes_by_channel(self):
        entries = {key: self.package(key) for key in ("A",)}
        entries["A"]["precheck"] = build_precheck(self.task, "A", entries["A"],
                                                  self.task.subject_prompt(),
                                                  self.style_reference, self.images)
        calls = {"gemini": 0, "gpt": 0}

        def fake_assemble(self_worker, variants, style_ref, aspect_ratio, motif_prompt,
                          model_overrides=None):
            return {
                "gemini_direct": {"prompt": "GEMINI", "image_paths": [style_ref],
                                  "model": "gemini-3-pro-image-preview", "aspect_ratio": aspect_ratio,
                                  "resolution": "2K", "api_type": "aigc2d", "face_quality_boost": False},
                "gpt_first_pass": {"prompt": "GPT", "image_paths": [style_ref], "model": "gpt-image-2",
                                   "aspect_ratio": aspect_ratio, "api_type": "aigc-2d-gpt",
                                   "mode": "generate", "quality": None, "size": None},
                "gpt_repaint": {"steps": {"repaint": {"enabled": True, "reference_mode": "none",
                                                      "scope": "full", "clauses_without_image": True}},
                                "style_ref_path": style_ref, "style_clauses": []},
            }

        def failing_gpt(*args, **kwargs):
            calls["gpt"] += 1
            raise RuntimeError("中转站 503")

        def ok_gemini(*args, **kwargs):
            calls["gemini"] += 1
            return [self._make_output("gemini-direct")]

        from modules.image_analysis import style_analyzer
        with self._patched_generation(ok_gemini, failing_gpt, None), \
                patch.object(style_analyzer.StyleIterativeWorkerThread,
                             "_assemble_test_generation_requests", fake_assemble):
            first = run_attempt(self.task, "A", entries["A"], 1, self.task.subject_prompt(),
                                self.style_reference, self.images, records={})
        # 桩返回的是本地小图，真链路里的后端校验会拒绝它 → 三路都算失败，但**失败原因必须落盘**
        self.assertIn(first["channel_status"]["gemini_direct"]["status"], ("success", "failed"))
        self.assertTrue(first["channel_status"]["gemini_direct"].get("error") or
                        first["channel_status"]["gemini_direct"]["status"] == "success")
        self.assertEqual(first["channel_status"]["gpt_repainted"]["status"], "not_run")
        self.assertIn("503", first["channel_status"]["gpt_first_pass"]["error"])
        self.assertEqual(calls["gpt"], 1)
        failed_gemini = first["channel_status"]["gemini_direct"]["status"] == "failed"

        # 重跑：成功过的 Gemini 产物必须复用；失败的那次必须重试（不许把失败当缓存）
        with self._patched_generation(ok_gemini, failing_gpt, None), \
                patch.object(style_analyzer.StyleIterativeWorkerThread,
                             "_assemble_test_generation_requests", fake_assemble):
            second = run_attempt(self.task, "A", entries["A"], 1, self.task.subject_prompt(),
                                 self.style_reference, self.images, records={})
        self.assertEqual(calls["gpt"], 2)
        if failed_gemini:
            self.assertEqual(calls["gemini"], 2)          # 失败不缓存：再试一次
        else:
            self.assertEqual(calls["gemini"], 1)          # 成功即复用：不重复收费

    def _patched_generation(self, gemini, gpt, pipeline):
        """按**运行时查找位置**打桩。

        - `_run_missing_channels` 是按名字 `from modules.others.api_backend import ...` 取的；
        - `_assemble_test_generation_requests` 走的是 style_analyzer 里的模块级导入；
        两处都要打，否则用例会真的去发请求。
        """
        from modules.others import api_backend
        from modules.image_analysis import style_analyzer
        from utils import analysis_gen
        patches = [patch.object(api_backend, "generate_image_aigc2d", side_effect=gemini),
                   patch.object(api_backend, "generate_image_aigc2d_gpt", side_effect=gpt),
                   patch.object(style_analyzer, "generate_image_aigc2d", side_effect=gemini),
                   patch.object(style_analyzer, "generate_image_aigc2d_gpt", side_effect=gpt)]
        if pipeline is not None:
            patches.append(patch.object(analysis_gen, "run_gpt_image_pipeline", side_effect=pipeline))
        return _PatchGroup(patches)

    def _make_output(self, tag, ext=".png"):
        self.output_index = getattr(self, "output_index", 0) + 1
        path = self.root / f"out-{self.output_index}-{tag}{ext}"
        Image.new("RGB", (30, 45), (self.output_index, 128, 0)).save(path)
        return str(path)

    # -- 4. 交付物骨架 ------------------------------------------------------ #

    def test_independent_repetition_never_reuses_first_sample(self):
        from modules.image_analysis.style_merge_validation import _reuse_source
        path = self._make_output("first")
        records = {("A", 1): {"status": "success", "channel_digests": {"gemini_direct": "same"},
                               "generated_files": {"gemini_direct": [path]}}}
        self.assertIsNone(_reuse_source(records, "gemini_direct", "same", attempt=2))
        self.assertIsNotNone(_reuse_source(records, "gemini_direct", "same", attempt=1))

    def test_repaint_cache_requires_actual_input_identity(self):
        from modules.image_analysis.style_merge_validation import _reuse_source
        path = self._make_output("repaint")
        records = {("A", 1): {"status": "success", "channel_digests": {"gpt_repainted": "placeholder"},
                               "runtime_channel_digests": {"gpt_repainted": "actual-source"},
                               "generated_files": {"gpt_repainted": [path]}}}
        self.assertIsNone(_reuse_source(records, "gpt_repainted", "placeholder", attempt=1))
        self.assertIsNotNone(_reuse_source(records, "gpt_repainted", "actual-source", attempt=1))

    def test_state_is_app_readable_and_freezes_field_sources(self):
        entries = {key: self.package(key) for key in ("A", "B", "C")}
        records = {}
        for key in entries:
            entries[key]["precheck"] = build_precheck(self.task, key, entries[key],
                                                      self.task.subject_prompt(),
                                                      self.style_reference, self.images)
            for attempt in (1, 2):
                records[(key, attempt)] = {
                    "status": "failed", "channel_status": {name: {"status": "not_run", "error": ""}
                                                           for name in PROBE_CHANNELS},
                    "generated_files": {name: [] for name in PROBE_CHANNELS},
                    "channel_reused": {}, "channel_digests": entries[key]["precheck"]["channel_digests"],
                    "request_digest": entries[key]["precheck"]["request_digest"],
                    "elapsed_seconds": 0.0, "errors": {}}
        state = build_state(self.task, entries, records)
        self.assertEqual(len(state["test_images"]), 6)
        self.assertEqual(state["parameters"]["repaint_reference_mode"], "none")
        self.assertEqual(state["dataset"]["images"], [os.path.abspath(p) for p in self.task.dataset_images()])
        for stage, record in state["test_images"].items():
            self.assertEqual(record["prompt_variants"]["optional_motifs"], [])
            self.assertTrue(record["field_sources"]["prompt"]["sha256"])
            self.assertEqual(record["test_prompt"], self.task.subject_prompt())


class PatchGroupIsolationTests(unittest.TestCase):
    def test_patch_is_restored_after_success_and_exception(self):
        from types import SimpleNamespace
        target = SimpleNamespace(fn=lambda: "original")
        original = target.fn
        with _PatchGroup([patch.object(target, "fn", return_value="mock")]):
            self.assertEqual(target.fn(), "mock")
        self.assertIs(target.fn, original)
        with self.assertRaises(RuntimeError):
            with _PatchGroup([patch.object(target, "fn", return_value="mock")]):
                raise RuntimeError("producer failed")
        self.assertIs(target.fn, original)

    def test_partial_enter_is_unwound(self):
        from types import SimpleNamespace
        target = SimpleNamespace(fn=lambda: None)
        original = target.fn
        with self.assertRaises(AttributeError):
            with _PatchGroup([patch.object(target, "fn"), patch.object(target, "missing")]):
                self.fail("entry must fail")
        self.assertIs(target.fn, original)


class CandidateReviewRegressionTests(unittest.TestCase):
    def test_review_report_field_source_does_not_require_round(self):
        from modules.image_analysis.style_merge_validation import _field_source_label
        sources = {"candidate_json": {"record_path": "prompt_variants"},
                   "prompt": {"field": "gemini_full_prompt"}}
        self.assertEqual(_field_source_label(sources, "prompt"), "prompt_variants")
        sources["prompt"]["round"] = "round_5"
        self.assertEqual(_field_source_label(sources, "prompt"), "round_5")

    def test_state_commands_restore_review_mode_without_candidates(self):
        import importlib.util
        from modules.image_analysis.style_merge_validation import ReviewCandidateTask, state_path
        spec = importlib.util.spec_from_file_location("review_cli", REPO_ROOT / "tools/style_merge_validate.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as directory:
            group = Path(directory) / "dark"
            group.mkdir()
            path = group / "custom_style_iter_result.json"
            path.write_text(json.dumps({"review_validation": {}, "test_images": {}}), encoding="utf-8")
            state = {"review_validation": {"mode": "run_record_candidates"},
                     "test_images": {"baseline-attempt1": {"attempt": 1}}}
            path.write_text(json.dumps(state), encoding="utf-8")
            (group / "precheck.json").write_text(json.dumps({
                "subject": "night scene", "candidates": [{"key": "baseline", "path": "unused.json"}],
                "reference_source": "unused.json"}), encoding="utf-8")
            for command in ["compare", "deep", "report"]:
                args = cli.build_parser().parse_args([command, "--out", directory, "--subject-label", "dark"])
                task = cli._task_for_state(args)
                self.assertIsInstance(task, ReviewCandidateTask)
                self.assertEqual(state_path(task), str(path))
                self.assertEqual(task.package_keys, ("baseline",))
                self.assertEqual(task.repetitions, 1)

    def test_unknown_explicit_round_does_not_fall_back(self):
        from types import SimpleNamespace
        from modules.image_analysis.style_merge_validation import _candidate_variant_source
        task = SimpleNamespace(candidate_state=lambda item: {"prompt_variants": {"gemini_full_prompt": "final"}})
        with self.assertRaises(ValidationError):
            _candidate_variant_source(task, {"key": "base", "package_key": "round_typo"})

    def test_recommendation_keeps_complete_candidate_name(self):
        from modules.image_analysis.style_merge_validation import recommendation_from
        rows = [{"id": "cccccanh-r5-opt1-attempt12/gpt_repainted/0", "eligible": True,
                 "sort_value": "8.5", "total": 8.5},
                {"id": "packageB-attempt2/gpt_first_pass/0", "eligible": False,
                 "sort_value": "9", "total": 9}]
        result = recommendation_from({"rows": rows}, {})
        self.assertEqual(result["package"], "cccccanh-r5-opt1")
        self.assertEqual(set(result["per_package"]), {"cccccanh-r5-opt1", "B"})

    def test_partial_repaint_is_never_reused(self):
        from modules.image_analysis.style_merge_validation import _reuse_source
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image-final-rp-partial.png"
            Image.new("RGB", (20, 20)).save(path)
            records = {("base", 1): {"status": "success",
                       "runtime_channel_digests": {"gpt_repainted": "same"},
                       "generated_files": {"gpt_repainted": [str(path)]}}}
            self.assertIsNone(_reuse_source(records, "gpt_repainted", "same", attempt=1))

    def test_partial_without_manifest_and_failed_manifest_are_rejected(self):
        from types import SimpleNamespace
        from modules.image_analysis.style_merge_validation import _run_missing_channels
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "first.png"
            Image.new("RGB", (20, 20)).save(source)
            worker = SimpleNamespace(log_signal=SimpleNamespace(emit=lambda text: None),
                                     _assemble_test_generation_requests=lambda *args, **kwargs: {"gpt_repaint": {}})
            package = {"package": {"optional_motifs": []}}
            for partial, failed in [(True, False), (False, True)]:
                path = Path(directory) / ("out-final-rp-partial.png" if partial else "out-final-rp.png")
                Image.new("RGB", (20, 20)).save(path)
                if failed:
                    work = Path(directory) / "gpt-repaint-steps"
                    work.mkdir()
                    (work / "pipeline-manifest.json").write_text(json.dumps({"items": [{
                        "source": str(source), "steps": [{"key": "repaint", "status": "failed", "error": "503"}]}]}), encoding="utf-8")
                with patch("utils.analysis_gen.run_gpt_image_pipeline", return_value=[str(path)]):
                    result = _run_missing_channels(worker, package, ["gpt_repainted"],
                             {"gpt_first_pass": [str(source)]}, directory, "test", str(source), "scene")
                self.assertEqual(result["gpt_repainted"], [])

    def test_save_directory_keeps_absolute_and_relative_paths(self):
        from modules.others.api_backend import resolve_save_dir
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(resolve_save_dir(directory, "20261005"), directory)
        self.assertEqual(resolve_save_dir("relative", "20261005"), os.path.join("data", "20261005", "relative"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
