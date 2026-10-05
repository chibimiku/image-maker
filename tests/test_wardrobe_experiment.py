# -*- coding: utf-8 -*-
"""衣装实验族的离线契约测试（不联网、不出图）。

覆盖第二轮任务书的硬性要求：
- 预算按「发送前扣减」，失败/超时/未知都算，绝不先发再判；
- 崩溃恢复把 inflight 视为已消耗，且不能被重新加载的上限重置；
- 保留分支/近景分支的确定性检查（无目标衣装设计块、无全身裙鞋、裁剪下缘显式、无占位符）；
- 解析请求剔除 expected；解析输出必须真布尔、ID 不重不漏、四字段一致。
"""
import json
import os
import tempfile
import unittest

from utils.wardrobe_experiment import (
    BudgetExceeded,
    ExperimentLedger,
    added_blocks,
    build_pairs_exact_balance,
    build_request_order,
    calibration_ok_with_overrides,
    candidate_readiness,
    check_ledger_budgets,
    compose_all,
    content_block_order,
    default_budgets,
    deterministic_prompt_checks,
    gold_routes,
    load_character_test,
    normalize_budgets,
    read_text,
    resolve_run_dir,
    resolver_request_payload,
    scoring_conditions,
    scrub_request_snapshot,
    sha256_text,
    snapshot_hash_check,
    spec_hash,
    validate_character_spec,
    validate_resolver_output,
)


class LedgerBudgetContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = os.path.join(self.temp.name, "ledger.json")

    def test_attempt_is_counted_before_it_is_sent(self):
        ledger = ExperimentLedger.load(self.path, {"generation_http_attempts": 2})
        attempt = ledger.plan_attempt("generation", slot="C03-gpt-r1-v3_routed")
        self.assertEqual(ledger.spent("generation"), 1)
        self.assertEqual(ledger.remaining("generation"), 1)
        self.assertEqual(attempt["status"], "inflight")
        # reload 后仍然是已消耗：不能靠重启把预算刷回去
        again = ExperimentLedger.load(self.path, {"generation_http_attempts": 2})
        self.assertEqual(again.spent("generation"), 1)

    def test_budget_is_enforced_before_dispatch(self):
        ledger = ExperimentLedger.load(self.path, {"text_http_attempts": 1})
        ledger.plan_attempt("text", slot="resolver-batch")
        with self.assertRaises(BudgetExceeded):
            ledger.plan_attempt("text", slot="review")
        self.assertEqual(ledger.spent("text"), 1)

    def test_failed_and_unknown_attempts_stay_counted(self):
        ledger = ExperimentLedger.load(self.path, {"vision_http_attempts": 3})
        first = ledger.plan_attempt("vision", slot="cal-C03")
        ledger.finish_attempt(first, status="error", http_status=400, error="content_filter")
        second = ledger.plan_attempt("vision", slot="cal-C06")
        # 第二次没来得及记录结果 -> 崩溃
        self.assertEqual(ledger.spent("vision"), 2)
        recovered = ledger.recover_inflight()
        self.assertEqual([item["slot"] for item in recovered], ["cal-C06"])
        self.assertEqual(ledger.spent("vision"), 2)
        self.assertEqual(ledger.remaining("vision"), 1)
        self.assertEqual(second["status"], "unknown")

    def test_reload_can_only_tighten_budgets(self):
        ledger = ExperimentLedger.load(self.path, {"generation_http_attempts": 4})
        ledger.plan_attempt("generation", slot="a")
        ledger.plan_attempt("generation", slot="b")
        again = ExperimentLedger.load(self.path, {"generation_http_attempts": 2})
        self.assertEqual(again.data["budgets"]["generation_http_attempts"], 2)
        self.assertEqual(again.spent("generation"), 2)
        self.assertEqual(again.remaining("generation"), 0)
        with self.assertRaises(BudgetExceeded):
            again.plan_attempt("generation", slot="c")

    def test_slot_attempt_index_tracks_retries_within_a_slot(self):
        ledger = ExperimentLedger.load(self.path, {"generation_http_attempts": 4})
        first = ledger.plan_attempt("generation", slot="C03-gpt-r1-v3_routed")
        ledger.finish_attempt(first, status="error", error="timeout")
        second = ledger.plan_attempt("generation", slot="C03-gpt-r1-v3_routed")
        self.assertEqual(second["slot_attempt"], 2)
        self.assertEqual(ledger.spent("generation"), 2)

    def test_summary_reports_limits_spent_and_remaining(self):
        ledger = ExperimentLedger.load(self.path, {"generation_http_attempts": 32,
                                                   "vision_http_attempts": 20,
                                                   "text_http_attempts": 2,
                                                   "model_list_http_attempts": 1})
        ledger.plan_attempt("model_list", slot="aigc2d")
        summary = ledger.summary()
        self.assertEqual(summary["model_list"]["limit"], 1)
        self.assertEqual(summary["model_list"]["remaining"], 0)
        self.assertEqual(summary["generation"]["spent"], 0)

    def test_normalize_budgets_defaults_and_negatives(self):
        self.assertEqual(normalize_budgets(None), default_budgets())
        budgets = normalize_budgets({"generation_http_attempts": -5, "unknown_key": 99})
        self.assertEqual(budgets["generation_http_attempts"], 0)
        self.assertNotIn("unknown_key", budgets)

    def test_unknown_attempt_kind_is_rejected(self):
        ledger = ExperimentLedger.load(self.path)
        with self.assertRaises(ValueError):
            ledger.plan_attempt("images", slot="x")

    def test_stop_reason_is_persisted(self):
        ledger = ExperimentLedger.load(self.path)
        ledger.mark_skipped("resolver mismatch")
        again = ExperimentLedger.load(self.path)
        self.assertEqual(again.data["stop_reason"], "resolver mismatch")

    def test_granted_budget_raises_limit_without_resetting_spend(self):
        ledger = ExperimentLedger.load(self.path, {"text_http_attempts": 2})
        ledger.plan_attempt("text", slot="review")
        granted = ledger.grant_budget("text", 3, authorized_by="user", reason="授权追加审查额度")
        self.assertEqual(granted["previous_limit"], 2)
        self.assertEqual(granted["new_limit"], 3)
        self.assertEqual(granted["spent"], 1)
        self.assertEqual(granted["remaining"], 2)
        # 重新加载后：已消耗仍是 1，上限是授权的 3，追加记录留痕
        again = ExperimentLedger.load(self.path, {"text_http_attempts": 2},
                                      trust_persisted_budgets=True)
        self.assertEqual(again.spent("text"), 1)
        self.assertEqual(again.data["budgets"]["text_http_attempts"], 3)
        # 不信任持久化上限时仍按老口径收紧（防止拿新上限抹掉已消耗）
        clamped = ExperimentLedger.load(self.path, {"text_http_attempts": 2})
        self.assertEqual(clamped.data["budgets"]["text_http_attempts"], 2)
        self.assertEqual(clamped.spent("text"), 1)
        self.assertEqual(again.data["budget_grants"][0]["authorized_by"], "user")
        self.assertEqual(again.data["budget_grants"][0]["reason"], "授权追加审查额度")
        # 不能靠「追加」把上限调低来伪造额度
        with self.assertRaises(ValueError):
            again.grant_budget("text", 2, authorized_by="user", reason="降回去")
        with self.assertRaises(ValueError):
            again.grant_budget("text", 4, authorized_by="", reason="缺授权人")

    def test_granted_limit_survives_a_spec_based_reload(self):
        """追加额度不能被 spec 上限压回去（评分中途被压回 25 就少跑了 6 次）。"""
        ledger = ExperimentLedger.load(self.path, {"vision_http_attempts": 20})
        ledger.plan_attempt("vision", slot="cal")
        ledger.grant_budget("vision", 31, authorized_by="user", reason="校准用掉更多槽位")
        reloaded = ExperimentLedger.load(self.path, {"vision_http_attempts": 20},
                                        trust_persisted_budgets=True)
        self.assertEqual(reloaded.granted_limit("vision"), 31)
        for kind in ("text", "generation", "vision", "model_list"):
            reloaded.assert_budget_floor(kind)
        self.assertEqual(reloaded.data["budgets"]["vision_http_attempts"], 31)
        self.assertEqual(reloaded.spent("vision"), 1)
        # 收紧路径（不信任持久化）后也必须能抬回来，而不是丢掉授权
        clamped = ExperimentLedger.load(self.path, {"vision_http_attempts": 20})
        self.assertEqual(clamped.data["budgets"]["vision_http_attempts"], 20)
        clamped.assert_budget_floor("vision")
        self.assertEqual(clamped.data["budgets"]["vision_http_attempts"], 31)
        self.assertEqual(clamped.spent("vision"), 1)

    def test_gate_override_is_kept_apart_from_budget_grants(self):
        """放行记录不能混进额度列表（否则出现 budget/new_limit 全是 None 的条目）。"""
        ledger = ExperimentLedger.load(self.path, {"vision_http_attempts": 20})
        ledger.record_override("calibrate", authorized_by="user", reason="接受裁剪读数偏差",
                               assumed_hypothesis="读数带一个躯干段偏差",
                               known_risk="framing 低可信", accepted_at="2026-10-05")
        again = ExperimentLedger.load(self.path, {"vision_http_attempts": 20},
                                      trust_persisted_budgets=True)
        self.assertEqual(again.data.get("budget_grants") or [], [])
        self.assertEqual(len(again.data["gate_overrides"]), 1)
        self.assertEqual(again.data["gate_overrides"][0]["kind"], "calibrate")
        self.assertEqual(again.granted_limit("vision"), 20)

    def test_ledger_budget_check_accepts_a_granted_limit(self):
        """追加额度是账本上限高于 spec 的**唯一**合法理由，检查只认「账本 < spec」。"""
        spec_budgets = {"text_http_attempts": 2, "generation_http_attempts": 32,
                        "vision_http_attempts": 20, "model_list_http_attempts": 1}
        granted = {**spec_budgets, "text_http_attempts": 3}
        self.assertEqual(check_ledger_budgets(granted, spec_budgets), [])

    def test_granted_text_budget_is_not_swallowed_by_the_per_stage_guard(self):
        """`send_budget` 的「每阶段 2 次」护栏必须跟随本账本上限，不能吞掉追加额度。"""
        from utils import send_budget
        with tempfile.TemporaryDirectory() as tmp:
            env_backup = {key: os.environ.get(key) for key in
                          (send_budget.ENV_DIR, send_budget.ENV_LIMIT, send_budget.ENV_TEXT_PER_STAGE)}
            try:
                os.environ[send_budget.ENV_DIR] = os.path.join(tmp, "budget-slots", "text")
                os.environ[send_budget.ENV_LIMIT] = "3"
                os.environ[send_budget.ENV_TEXT_PER_STAGE] = "3"
                for _ in range(3):
                    send_budget.reserve_text_attempt(stage="text:review", run_id="review")
                with self.assertRaises(send_budget.SendBudgetExhausted):
                    send_budget.reserve_text_attempt(stage="text:review", run_id="review")
                # 默认护栏（2 次）在第三个槽位就会拦下——所以工具必须显式对齐上限
                self.assertEqual(send_budget.DEFAULT_TEXT_PER_STAGE, 2)
            finally:
                for key, value in env_backup.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value


class SnapshotContracts(unittest.TestCase):
    def test_hash_check_detects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "C03-v3_routed.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("hello")
            manifest = {"snapshots": [{"file": "C03-v3_routed.txt", "case_id": "C03",
                                       "variant": "v3_routed", "sha256": sha256_text("hello"),
                                       "characters": 5}]}
            self.assertTrue(snapshot_hash_check(manifest, tmp)["ok"])
            manifest["snapshots"][0]["sha256"] = sha256_text("other")
            result = snapshot_hash_check(manifest, tmp)
            self.assertFalse(result["ok"])
            self.assertFalse(result["rows"][0]["match"])

    def test_preserve_branch_rejects_target_design_block(self):
        base = "base prompt\n\ncontent"
        leaked = base + "\n\nAUTHORIZED CLOTHING ACTION: PRESERVE.\n\nJapanese Lolita fashion: design an OP dress with layered skirt volume."
        result = deterministic_prompt_checks(leaked, base, variant="v3_routed",
                                             gold_action="preserve", framing="full")
        self.assertFalse(result["ok"])
        self.assertTrue(any(issue["check"] == "preserve_no_target_design_block"
                            for issue in result["issues"]))

    def test_preserve_branch_accepts_clean_addition(self):
        base = "base prompt\n\ncontent"
        clean = base + "\n\nAUTHORIZED CLOTHING ACTION: PRESERVE.\nKeep the supplied clothing category and cut."
        result = deterministic_prompt_checks(clean, base, variant="v3_routed",
                                             gold_action="preserve", framing="full")
        self.assertTrue(result["ok"], result["issues"])

    def test_close_portrait_requires_crop_boundary_and_no_full_outfit(self):
        base = "base prompt\n\ncontent"
        bad = base + "\n\nAUTHORIZED CLOTHING ACTION: FILL VISIBLE UNSPECIFIED CLOTHING.\nDraw a full-body outfit with shoes."
        result = deterministic_prompt_checks(bad, base, variant="v3_routed",
                                             gold_action="fill", framing="upper_chest")
        checks = {issue["check"] for issue in result["issues"]}
        self.assertIn("close_portrait_no_full_outfit", checks)
        self.assertIn("crop_boundary_explicit", checks)
        good = base + ("\n\nAUTHORIZED CLOTHING ACTION: FILL VISIBLE UNSPECIFIED CLOTHING.\n"
                       "Only visible neckline details.\n\n"
                       "FINAL FRAMING REQUIREMENT: close portrait, the lower frame edge cuts across "
                       "the upper chest, above the bust midpoint.")
        self.assertTrue(deterministic_prompt_checks(good, base, variant="v3_routed",
                                                    gold_action="fill", framing="upper_chest")["ok"])

    def test_expectation_leak_and_placeholder_are_rejected(self):
        base = "base"
        leaked = base + "\n\n{" + "design" + "} expected pass score"
        result = deterministic_prompt_checks(leaked, base, variant="v3_routed",
                                             gold_action="preserve", framing="full")
        checks = {issue["check"] for issue in result["issues"]}
        self.assertIn("no_expectation_leak", checks)
        self.assertIn("no_placeholder", checks)

    def test_added_blocks_splits_only_the_extra_part(self):
        base = "rendering\n\ncontent"
        full = base + "\n\nBLOCK ONE: x\n\nBLOCK TWO: y"
        blocks = added_blocks(full, base)
        self.assertEqual(len(blocks), 2)
        self.assertTrue(blocks[1].startswith("BLOCK TWO"))


class ResolverContracts(unittest.TestCase):
    def setUp(self):
        self.cases = [
            {"id": "R01", "policy": "reinterpret", "source_facts": "Red school uniform.",
             "user_request": "Keep the original uniform.", "expected": {"action": "preserve"}},
            {"id": "R07", "policy": "fill_missing", "source_facts": "No source image; no clothing supplied.",
             "user_request": "A long-haired woman, full body.",
             "expected": {"action": "fill"}},
        ]

    def test_request_payload_strips_expected_fields(self):
        payload = resolver_request_payload(self.cases, "system prompt")
        self.assertNotIn("expected", payload["user"])
        self.assertEqual(payload["case_count"], 2)
        self.assertTrue(payload["expected_fields_stripped"])
        body = json.loads(payload["user"])
        for case in body["cases"]:
            self.assertNotIn("expected", case)
            self.assertEqual(sorted(case.keys()), ["id", "policy", "source_facts", "user_request"])

    def test_validator_requires_real_booleans_and_all_ids(self):
        raw = json.dumps({"decisions": [
            {"id": "R01", "action": "preserve", "clothing_presence": "provided",
             "explicit_keep_original": "true", "framing": "full"}]})
        result = validate_resolver_output(raw, self.cases)
        self.assertFalse(result["valid"])
        self.assertTrue(any("真布尔" in issue for row in result["rows"] for issue in row["issues"]))
        self.assertTrue(any("缺少 ID" in issue for issue in result["issues"]))

    def test_validator_accepts_matching_decisions(self):
        raw = json.dumps({"decisions": [
            {"id": "R01", "action": "preserve", "clothing_presence": "provided",
             "explicit_keep_original": True, "framing": "full"},
            {"id": "R07", "action": "fill", "clothing_presence": "unspecified",
             "explicit_keep_original": False, "framing": "full"}]})
        expected = {"R01": {"action": "preserve", "clothing_presence": "provided",
                            "explicit_keep_original": True, "framing": "full"},
                    "R07": {"action": "fill", "clothing_presence": "unspecified",
                            "explicit_keep_original": False, "framing": "full"}}
        result = validate_resolver_output(raw, self.cases, expected_by_id=expected)
        self.assertTrue(result["valid"], result["issues"])
        self.assertEqual(result["matched"], 2)

    def test_validator_without_gold_reports_no_match_count(self):
        raw = json.dumps({"decisions": [
            {"id": "R01", "action": "preserve", "clothing_presence": "provided",
             "explicit_keep_original": True, "framing": "full"}]})
        result = validate_resolver_output(raw, self.cases)
        self.assertEqual(result["matched"], 0)
        self.assertTrue(all(row.get("matches") is None for row in result["rows"]))

    def test_validator_flags_duplicate_ids(self):
        raw = json.dumps({"decisions": [
            {"id": "R01", "action": "preserve", "clothing_presence": "provided",
             "explicit_keep_original": True, "framing": "full"},
            {"id": "R01", "action": "fill", "clothing_presence": "unspecified",
             "explicit_keep_original": False, "framing": "full"}]})
        result = validate_resolver_output(raw, self.cases)
        self.assertTrue(any("重复" in issue for issue in result["issues"]))

    def test_gold_routes_use_preregistered_actions(self):
        plan = {"cases": [{"case_id": "C03", "gold_action": "preserve", "gold_presence": "provided",
                           "framing": "full", "role": "explicit_preservation", "repeats": 2}]}
        routes = gold_routes(plan)
        self.assertEqual(routes["C03"]["gold_action"], "preserve")
        self.assertEqual(routes["C03"]["repeats"], 2)


class SnapshotSerializationContracts(unittest.TestCase):
    def test_scrub_removes_secrets_and_hashes_long_blobs(self):
        payload = {"Authorization": "Bearer sk-secret", "data": "x" * 5000,
                   "prompt": "keep me"}
        result = scrub_request_snapshot(payload, secret_values=["sk-secret"])
        body = result["redacted"]
        self.assertEqual(body["Authorization"], "<REDACTED>")
        self.assertEqual(body["prompt"], "keep me")
        self.assertIn("__sha256__", body["data"])
        self.assertNotIn("sk-secret", json.dumps(result, ensure_ascii=False))

    def test_content_block_order_records_labels_and_image_hashes(self):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "IMAGE LABEL: [A] first"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "IMAGE LABEL: [B] second"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}}]}]
        order = content_block_order(messages)
        self.assertEqual([item["type"] for item in order],
                         ["text", "image_url", "text", "image_url"])
        self.assertEqual([item.get("label") for item in order if item["type"] == "text"], ["A", "B"])
        self.assertTrue(all(item.get("sha256") for item in order))


class Round3SpecContracts(unittest.TestCase):
    """Round3 准备阶段的离线契约：spec 校验、组装、门禁、配对平衡、评分条件。"""

    SPEC = os.path.join("prompts", "wardrobe", "round3", "spec")
    BASE_PACK = os.path.join("prompts", "wardrobe", "round3", "base-pack")

    def test_real_round3_spec_validates_and_composes(self):
        validation = validate_character_spec(self.SPEC)
        self.assertTrue(validation["ok"], validation["issues"])
        self.assertEqual(validation["case_count"], 6)
        self.assertEqual(validation["planned_generation_slots"], 32)
        composed = compose_all(self.SPEC)
        self.assertEqual(len(composed["rows"]), 12)
        for row in composed["rows"]:
            self.assertGreaterEqual(row["characters"], 400)
            self.assertNotIn("{", row["prompt"], f"{row['file']} 残留占位符")

    def test_round3_candidate_passes_strict_and_control_defects_are_expected(self):
        composed = compose_all(self.SPEC)
        for row in composed["rows"]:
            base = read_text(os.path.join(self.BASE_PACK, f"{row['case_id']}-base.txt"))
            strict = row["variant"] == "v4_candidate"
            result = deterministic_prompt_checks(row["prompt"], base, variant=row["variant"],
                                                 gold_action=row["gold_action"],
                                                 framing=row["framing"], strict=strict)
            self.assertTrue(result["ok"], f"{row['file']} 未通过: {result['issues']}")
            if not strict and result["issues"]:
                self.assertTrue(all(issue["severity"] == "defect_expected"
                                    for issue in result["issues"]))

    def test_missing_case_field_fails_validation(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(self.SPEC, os.path.join(tmp, "spec"))
            shutil.copytree(os.path.join("prompts", "wardrobe", "round3", "variants"),
                            os.path.join(tmp, "variants"))
            shutil.copytree(os.path.join("prompts", "wardrobe", "round3", "characters"),
                            os.path.join(tmp, "characters"))
            shutil.copy2(os.path.join("prompts", "wardrobe", "round3", "characters.json"),
                         os.path.join(tmp, "characters.json"))
            spec_plan = os.path.join(tmp, "spec", "plan.json")
            plan = json.load(open(spec_plan, encoding="utf-8"))
            plan["base_pack"] = os.path.abspath(self.BASE_PACK)
            plan["characters_file"] = os.path.join(tmp, "characters.json")
            plan["cases"][0].pop("framing")
            json.dump(plan, open(spec_plan, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            validation = validate_character_spec(os.path.join(tmp, "spec"))
            self.assertFalse(validation["ok"])
            self.assertTrue(any("缺字段" in issue for issue in validation["issues"]))

    def test_wrong_enum_and_slot_mismatch_fail_closed(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(self.SPEC, os.path.join(tmp, "spec"))
            shutil.copytree(os.path.join("prompts", "wardrobe", "round3", "variants"),
                            os.path.join(tmp, "variants"))
            shutil.copytree(os.path.join("prompts", "wardrobe", "round3", "characters"),
                            os.path.join(tmp, "characters"))
            shutil.copy2(os.path.join("prompts", "wardrobe", "round3", "characters.json"),
                         os.path.join(tmp, "characters.json"))
            spec_plan = os.path.join(tmp, "spec", "plan.json")
            plan = json.load(open(spec_plan, encoding="utf-8"))
            plan["base_pack"] = os.path.abspath(self.BASE_PACK)
            plan["characters_file"] = os.path.join(tmp, "characters.json")
            plan["cases"][0]["framing"] = "waist_up"          # 非法枚举
            plan["budgets"]["generation_http_attempts"] = 31  # 与槽位不符
            json.dump(plan, open(spec_plan, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            validation = validate_character_spec(os.path.join(tmp, "spec"))
            self.assertFalse(validation["ok"])
            self.assertTrue(any("framing 非法" in issue for issue in validation["issues"]))
            self.assertTrue(any("不一致" in issue for issue in validation["issues"]))

    def test_pairs_are_exactly_balanced_and_slot_count_matches_spec(self):
        doc = load_character_test(self.SPEC)
        order = build_request_order(doc["plan"]["cases"], doc["plan"]["channels"],
                                    ["v3_control", "v4_candidate"], seed=20261005)
        self.assertEqual(len(order), 32)
        self.assertEqual(len({row["sample_id"] for row in order}), 32)
        pairs = build_pairs_exact_balance(order, seed=20261005)
        self.assertEqual(pairs["balance"]["pairs"], 16)
        self.assertEqual(pairs["balance"]["a_control"], 8)
        self.assertEqual(pairs["balance"]["a_candidate"], 8)
        self.assertTrue(pairs["balance"]["exact_balance"])
        for pair in pairs["pairs"]:
            self.assertEqual(set(pair["assignment"].values()), {"v3_control", "v4_candidate"})

    def test_candidate_gate_blocks_every_generation_entry(self):
        hash_now = spec_hash(self.SPEC)
        blocked = candidate_readiness(self.SPEC, {"prepare_ok": True, "review": {}, "calibrate": {}})
        self.assertFalse(blocked["ready"])
        self.assertTrue(any("候选审查" in reason for reason in blocked["reasons"]))
        self.assertEqual(blocked["control_id"], "v3_control")
        self.assertEqual(blocked["candidate_id"], "v4_candidate")
        partial = candidate_readiness(self.SPEC, {
            "prepare_ok": True, "spec_hash": hash_now,
            "review": {"candidate_ready": True, "blocking": [], "spec_hash": hash_now},
            "calibrate": {"ok": False}})
        self.assertFalse(partial["ready"])
        self.assertTrue(any("校准" in reason for reason in partial["reasons"]))
        ready = candidate_readiness(self.SPEC, {
            "prepare_ok": True, "spec_hash": hash_now,
            "review": {"candidate_ready": True, "blocking": [], "spec_hash": hash_now},
            "calibrate": {"ok": True}})
        self.assertTrue(ready["ready"], ready["reasons"])

    def test_gate_ignores_control_defects_but_not_candidate_blockers(self):
        hash_now = spec_hash(self.SPEC)
        state = {"prepare_ok": True, "spec_hash": hash_now,
                 "review": {"candidate_ready": True, "blocking": [], "spec_hash": hash_now},
                 "calibrate": {"ok": True}}
        self.assertTrue(candidate_readiness(self.SPEC, state)["ready"])
        state["review"] = {"candidate_ready": False, "blocking": [{"file": "T05-v4_candidate.txt"}],
                           "spec_hash": hash_now}
        gate = candidate_readiness(self.SPEC, state)
        self.assertFalse(gate["ready"])
        self.assertTrue(any("blocking" in reason for reason in gate["reasons"]))
        state["review"] = {"candidate_ready": False, "blocking": [], "spec_hash": hash_now}
        self.assertFalse(candidate_readiness(self.SPEC, state)["ready"])

    def test_calibration_override_needs_a_complete_record(self):
        """放行 ≠ 通过：缺授权人/假设/已知风险任何一项都不算放行。"""
        calibration = {"ok": False}
        self.assertFalse(calibration_ok_with_overrides(calibration)["ok"])
        calibration["calibration_overrides"] = [{"kind": "calibrate", "authorized_by": "user",
                                                 "reason": "接受风险"}]
        self.assertFalse(calibration_ok_with_overrides(calibration)["ok"])
        calibration["calibration_overrides"] = [{
            "kind": "calibrate", "authorized_by": "user", "reason": "用户接受已知风险继续",
            "assumed_hypothesis": "裁剪位置读数会附带偏差",
            "known_risk": "framing 分项低可信", "accepted_at": "2026-10-05"}]
        verdict = calibration_ok_with_overrides(calibration)
        self.assertTrue(verdict["ok"])
        self.assertTrue(verdict["overridden"])

    def test_gate_is_ready_only_with_a_recorded_override(self):
        hash_now = spec_hash(self.SPEC)
        state = {"prepare_ok": True, "spec_hash": hash_now,
                 "review": {"candidate_ready": True, "blocking": [], "spec_hash": hash_now},
                 "calibrate": {"ok": False}}
        gate = candidate_readiness(self.SPEC, state)
        self.assertFalse(gate["ready"])
        self.assertTrue(any("校准" in reason for reason in gate["reasons"]))
        state["calibrate"] = {"ok": False, "calibration_overrides": [{
            "kind": "calibrate", "authorized_by": "user", "reason": "接受风险",
            "assumed_hypothesis": "读数带偏差", "known_risk": "framing 低可信",
            "accepted_at": "2026-10-05"}]}
        gate = candidate_readiness(self.SPEC, state)
        self.assertTrue(gate["ready"], gate["reasons"])
        self.assertTrue(gate["overrides"], "放行必须出现在门禁结果里，供报告引用")

    def test_review_from_older_materials_is_not_reused(self):
        current = spec_hash(self.SPEC)
        state = {"prepare_ok": True, "spec_hash": current,
                 "review": {"candidate_ready": True, "blocking": [],
                            "spec_hash": "0" * 64},
                 "calibrate": {"ok": True}}
        gate = candidate_readiness(self.SPEC, state)
        self.assertFalse(gate["ready"])
        self.assertTrue(any("旧素材" in reason for reason in gate["reasons"]))
        state["review"]["spec_hash"] = current
        self.assertTrue(candidate_readiness(self.SPEC, state)["ready"])

    def test_scoring_conditions_carry_real_evidence(self):
        doc = load_character_test(self.SPEC)
        case = doc["plan"]["cases"][0]
        character = doc["by_case"][case["case_id"]]
        conditions = scoring_conditions(case, character)
        self.assertEqual(conditions["identity"], character["identity"])
        self.assertEqual(conditions["source_clothing"], character["source_clothing"])
        self.assertEqual(conditions["user_request"], character["request"])
        self.assertEqual(conditions["protected"], character["protected"])
        self.assertNotIn("variant", json.dumps(conditions))
        self.assertNotIn("v3_control", json.dumps(conditions))
        self.assertTrue(conditions["note"])


class CalibrationRuleContracts(unittest.TestCase):
    def _run(self):
        from tools.wardrobe_experiment import validate_calibration
        return validate_calibration

    def _term(self):
        from tools.wardrobe_experiment import _term_present
        return _term_present
    def test_calibration_without_rules_fails_closed(self):
        validate_calibration = self._run()
        raw = json.dumps({"observations": [
            {"label": "A", "clothing": "red uniform", "lower_frame_edge": "below shoes"},
            {"label": "B", "clothing": "burgundy dress", "lower_frame_edge": "below shoes"}]})
        result = validate_calibration("CAL-X", raw, "", rules=None)
        self.assertFalse(result["ok"])
        self.assertTrue(any("禁止默认通过" in issue for issue in result["issues"]))

    def test_calibration_applies_terms_and_forbidden_words(self):
        validate_calibration = self._run()
        rules = {"A_terms": ["uniform"], "B_terms": ["dress"], "both_forbidden": ["pink"]}
        good = json.dumps({"observations": [
            {"label": "A", "clothing": "red school uniform blazer and skirt",
             "lower_frame_edge": "below the shoes"},
            {"label": "B", "clothing": "burgundy dress with lace",
             "lower_frame_edge": "below the shoes"}]})
        self.assertTrue(validate_calibration("CAL-CAT", good, "", rules=rules)["ok"])
        bad = json.dumps({"observations": [
            {"label": "A", "clothing": "pink uniform", "lower_frame_edge": "x"},
            {"label": "B", "clothing": "dress", "lower_frame_edge": "x"}]})
        result = validate_calibration("CAL-CAT", bad, "", rules=rules)
        self.assertFalse(result["ok"])
        self.assertTrue(any("pink" in issue for issue in result["issues"]))

    def test_calibration_terms_match_words_not_bare_substrings(self):
        """`red` 不能命中 `colored`/`patterned`：裸子串会把纯黑图判成读出红色。"""
        validate_calibration = self._run()
        rules = {"both_forbidden": ["red"]}
        black = json.dumps({"observations": [
            {"label": "A", "clothing": "black blazer, black trousers", "lower_frame_edge": "shoes"},
            {"label": "B", "clothing": "black dress with patterned lace tights",
             "lower_frame_edge": "shoes"}]})
        self.assertTrue(validate_calibration("CAL-ACC", black, "", rules=rules)["ok"])
        red = json.dumps({"observations": [
            {"label": "A", "clothing": "black blazer, black trousers", "lower_frame_edge": "shoes"},
            {"label": "B", "clothing": "a red dress", "lower_frame_edge": "shoes"}]})
        self.assertFalse(validate_calibration("CAL-ACC", red, "", rules=rules)["ok"])
        # 多词短语仍按连续子串匹配
        term_present = self._term()
        self.assertTrue(term_present("lace dress", "black lace dress"))
        self.assertTrue(term_present("crescent", "silver-toned crescent-shaped brooch"))
        self.assertFalse(term_present("red", "patterned lace tights"))

    def test_calibration_crop_requires_distinct_edges(self):
        validate_calibration = self._run()
        rules = {"require_distinct_edges": True}
        same = json.dumps({"observations": [
            {"label": "A", "clothing": "blouse", "lower_frame_edge": "upper chest"},
            {"label": "B", "clothing": "blouse", "lower_frame_edge": "upper chest"}]})
        self.assertFalse(validate_calibration("CAL-CROP", same, "", rules=rules)["ok"])
        distinct = json.dumps({"observations": [
            {"label": "A", "clothing": "blouse", "lower_frame_edge": "above the bust midpoint"},
            {"label": "B", "clothing": "dress", "lower_frame_edge": "down to the waist"}]})
        self.assertTrue(validate_calibration("CAL-CROP", distinct, "", rules=rules)["ok"])
        rules_with_terms = {"require_distinct_edges": True, "must_use_terms": ["chest"]}
        no_position = json.dumps({"observations": [
            {"label": "A", "clothing": "blouse", "lower_frame_edge": "just below the collar"},
            {"label": "B", "clothing": "dress", "lower_frame_edge": "down to the hem"}]})
        result = validate_calibration("CAL-CROP", no_position, "", rules=rules_with_terms)
        self.assertFalse(result["ok"])
        self.assertTrue(any("裁剪位置描述" in issue for issue in result["issues"]))


class RunDirRecoveryContracts(unittest.TestCase):
    """跨天复跑必须锁定「原目录 + 原账本」：不能因为换了一天就多出一份预算。"""

    def test_explicit_run_dir_wins_over_the_date_derived_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "20261005", "wardrobe-round3-prep")
            os.makedirs(run_dir)
            with open(os.path.join(run_dir, "ledger.json"), "w", encoding="utf-8") as handle:
                json.dump({"budgets": {"text_http_attempts": 2}}, handle)
            resolved = resolve_run_dir(tmp, "wardrobe-round3-prep", run_dir)
            self.assertEqual(resolved, os.path.abspath(run_dir))
            self.assertNotIn(os.path.join("20261005", "20261005"), resolved)

    def test_explicit_run_dir_without_ledger_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = os.path.join(tmp, "wardrobe-round3-prep")
            os.makedirs(empty)
            with self.assertRaises(FileNotFoundError):
                resolve_run_dir(tmp, "wardrobe-round3-prep", empty)

    def test_missing_explicit_run_dir_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                resolve_run_dir(tmp, "wardrobe-round3-prep", os.path.join(tmp, "nope"))

    def test_default_path_still_uses_the_current_day(self):
        import time
        with tempfile.TemporaryDirectory() as tmp:
            resolved = resolve_run_dir(tmp, "wardrobe-round3-prep")
            self.assertEqual(os.path.dirname(resolved), os.path.join(os.path.abspath(tmp),
                                                                     time.strftime("%Y%m%d")))
            self.assertTrue(resolved.endswith("wardrobe-round3-prep"))

    def test_ledger_budget_change_is_detected_against_spec(self):
        spec_budgets = {"text_http_attempts": 2, "generation_http_attempts": 32,
                        "vision_http_attempts": 20, "model_list_http_attempts": 1}
        self.assertEqual(check_ledger_budgets(spec_budgets, spec_budgets), [])
        issues = check_ledger_budgets({**spec_budgets, "generation_http_attempts": 8}, spec_budgets)
        self.assertEqual([item["budget"] for item in issues], ["generation_http_attempts"])

    def test_cli_uses_explicit_run_dir_and_never_resets_budgets(self):
        """CLI 层回归：--run-dir 指向原账本；上限被改动时宁可报错也不继续。"""
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "wardrobe-round3-prep")
            os.makedirs(run_dir)
            ledger_path = os.path.join(run_dir, "ledger.json")
            ledger = ExperimentLedger.load(ledger_path, {
                "text_http_attempts": 2, "generation_http_attempts": 32,
                "vision_http_attempts": 20, "model_list_http_attempts": 1})
            ledger.plan_attempt("text", slot="review")
            before = json.load(open(ledger_path, encoding="utf-8"))
            base = [sys.executable, os.path.join("tools", "wardrobe_experiment.py"),
                    "--spec", os.path.join("prompts", "wardrobe", "round3", "spec"),
                    "--pack", os.path.join("prompts", "wardrobe", "round3", "pack"),
                    "--run-id", "wardrobe-round3-prep", "--run-dir", run_dir]
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            ok = subprocess.run(base + ["--status"], capture_output=True, text=True,
                                encoding="utf-8", errors="replace", cwd=os.getcwd(), env=env)
            self.assertEqual(ok.returncode, 0, ok.stderr)
            self.assertEqual(json.loads(ok.stdout)["out_dir"], os.path.abspath(run_dir))
            self.assertEqual(json.loads(ok.stdout)["budgets"]["text"]["spent"], 1)
            self.assertEqual(json.loads(ok.stdout)["budgets"]["generation"]["limit"], 32)
            # 账本上限被悄悄改**小**（换账本式重置）时必须报错停下
            tampered = {**before, "budgets": {**before["budgets"], "generation_http_attempts": 8}}
            json.dump(tampered, open(ledger_path, "w", encoding="utf-8"), ensure_ascii=False)
            blocked = subprocess.run(base + ["--status"], capture_output=True, text=True,
                                     encoding="utf-8", errors="replace", cwd=os.getcwd(), env=env)
            self.assertEqual(blocked.returncode, 1)
            output = (blocked.stdout or "") + (blocked.stderr or "")
            self.assertIn("拒绝继续", output)
            self.assertIn("spec=32", output)
            json.dump(before, open(ledger_path, "w", encoding="utf-8"), ensure_ascii=False)

    def test_cli_grant_budget_records_authorization_and_raises_limit(self):
        """追加额度也必须走工具、留痕、且不动已消耗。"""
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = os.path.join(tmp, "wardrobe-round3-prep")
            os.makedirs(run_dir)
            ledger_path = os.path.join(run_dir, "ledger.json")
            ledger = ExperimentLedger.load(ledger_path, {
                "text_http_attempts": 2, "generation_http_attempts": 32,
                "vision_http_attempts": 20, "model_list_http_attempts": 1})
            ledger.plan_attempt("text", slot="review")
            base = [sys.executable, os.path.join("tools", "wardrobe_experiment.py"),
                    "--spec", os.path.join("prompts", "wardrobe", "round3", "spec"),
                    "--pack", os.path.join("prompts", "wardrobe", "round3", "pack"),
                    "--run-id", "wardrobe-round3-prep", "--run-dir", run_dir,
                    "--grant-budget", "text", "--grant-limit", "3",
                    "--authorized-by", "user", "--grant-reason", "授权追加审查额度",
                    "--grant-source", "conversation 2026-10-05"]
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            grant = subprocess.run(base, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", cwd=os.getcwd(), env=env)
            self.assertEqual(grant.returncode, 0, grant.stderr)
            data = json.load(open(ledger_path, encoding="utf-8"))
            self.assertEqual(data["budgets"]["text_http_attempts"], 3)
            self.assertEqual(data["budget_grants"][0]["authorized_by"], "user")
            self.assertEqual(data["budget_grants"][0]["spent_at_grant"], 1)
            status = subprocess.run(base[:base.index("--grant-budget")] + ["--status"],
                                    capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", cwd=os.getcwd(), env=env)
            self.assertEqual(status.returncode, 0, status.stderr)
            self.assertEqual(json.loads(status.stdout)["budgets"]["text"]["limit"], 3)
            self.assertEqual(json.loads(status.stdout)["budgets"]["text"]["spent"], 1)


class Round3BaselineRegression(unittest.TestCase):
    def test_control_preserves_real_round2_crop_and_preservation(self):
        from pathlib import Path
        root = Path("prompts/wardrobe")
        self.assertEqual((root/"round2/preserve.md").read_text("utf-8"), (root/"round3/variants/preserve.md").read_text("utf-8"))
        crop = (root/"round3/variants/frame-bust-v3.md").read_text("utf-8")
        self.assertIn("above the bust midpoint", crop)
        self.assertIn("Do not show the lower bodice", crop)

    def test_control_visible_redesign_authorization(self):
        from utils.wardrobe_experiment import compose_all
        rows = compose_all("prompts/wardrobe/round3/spec")["rows"]
        text = next(row["prompt"] for row in rows if row["case_id"] == "T06" and row["variant"] == "v3_control")
        self.assertIn("AUTHORIZED CLOTHING ACTION: REDESIGN VISIBLE CLOTHING", text)
        self.assertNotIn("FILL VISIBLE UNSPECIFIED", text)


if __name__ == "__main__":
    unittest.main()
