"""固定色板执行验证（color-fixed-palette-v1）的离线回归用例。

全部离线：只读 spec / 组装请求 / 校验响应形状，不联网、不写 data/ 的产物目录。
这些用例保护协议里最容易退化的几件事：
① 两版条款必须表达同一套颜色与区域（公平对照，不靠删掉约束制造优势）；
② 生成调用数上界 = 计划产物数（按目标数生成 + 关闭底层重试）；
③ 评审按组计（每组 3 张一次调用），额外次数有上限；
④ 响应校验拦住漏图、漏条款与非法状态；
⑤ 预检对「替代色被显式禁用」「保护块写清只允许既有区域改色」的判断。
"""
import json
from pathlib import Path

import pytest

from utils.color_experiment import (PALETTE_CALL_ENV_TARGET, PALETTE_RETRY_ENV, PALETTE_STATES,
                                    load_palette_spec, palette_generation_preflight,
                                    palette_group_checks, palette_group_specs, palette_preflight,
                                    palette_request_preview, palette_variant_pairs,
                                    validate_palette_review)

RUN_DIR = Path("data/test-result/20261004/color-fixed-palette-v1")


@pytest.fixture(scope="module")
def spec():
    return load_palette_spec()


def test_spec_has_two_scales_and_one_channel(spec):
    assert list(spec["channels"]) == ["gemini-flash"]
    assert set(spec["scale_options"]) == {"full", "reduced"}
    assert spec["scale_options"]["full"]["images"] == 21
    assert spec["scale_options"]["reduced"]["images"] == 9
    assert spec["scale_options"]["reduced"]["groups"] == ["P1-simple", "P1-knowledge", "B0"]


def test_full_matrix_is_seven_groups_of_three(spec):
    option, groups = palette_group_specs(spec, "full")
    assert len(groups) == 7
    assert option["images"] == 7 * 3
    assert [g["id"] for g in groups] == ["P1-simple", "P1-knowledge", "P2-simple", "P2-knowledge",
                                        "P3-simple", "P3-knowledge", "B0"]
    reduced_option, reduced_groups = palette_group_specs(spec, "reduced")
    assert [g["id"] for g in reduced_groups] == ["P1-simple", "P1-knowledge", "B0"]
    assert reduced_option["review_calls_planned"] == 3


def test_variant_pairs_cover_same_palette_and_regions(spec):
    pairs = palette_variant_pairs(spec)
    assert {pair["palette_id"] for pair in pairs} == {"P1", "P2", "P3"}
    for pair in pairs:
        simple, knowledge = pair["simple"], pair["knowledge"]
        assert simple and knowledge, f"{pair['palette_id']} 缺一个版本"
        assert len(simple["clauses"]) == len(knowledge["clauses"])
        simple_text = " ".join(simple["clauses"]).lower()
        knowledge_text = " ".join(knowledge["clauses"]).lower()
        for keyword in ("waistband", "ribbon bows"):
            assert keyword in simple_text and keyword in knowledge_text, f"{pair['palette_id']}: 区域分配不一致"
        assert ("ivory-white dress" in simple_text) == ("ivory-white dress" in knowledge_text)
        assert ("second person" in simple_text) == ("second person" in knowledge_text)


def test_baseline_group_carries_no_colour_clauses(spec):
    baseline = [group for group in spec["groups"] if group["id"] == "B0"]
    assert len(baseline) == 1
    assert not baseline[0]["clauses"]
    assert baseline[0]["palette_id"] is None


def test_accent_must_be_named_and_region_assigned(spec):
    for group in spec["groups"]:
        findings = palette_group_checks(spec, group)
        assert not findings["problems"], f"{group['id']}: {findings['problems']}"
    p1 = next(group for group in spec["groups"] if group["id"] == "P1-simple")
    assert "red" in " ".join(p1["clauses"]).lower()


def test_explicitly_forbidden_substitute_is_not_a_problem(spec):
    """P1 写了「do not replace it with pink」：禁用替代色应被识别为通过，不是问题。"""
    p1 = next(group for group in spec["groups"] if group["id"] == "P1-simple")
    findings = palette_group_checks(spec, p1)
    assert not findings["problems"]
    assert any("替换色" in note for note in findings["notes"])


def test_preflight_passes_offline_without_touching_the_network(tmp_path):
    out = Path("data/test-result/20261004") / f"color-fixed-palette-preflight-{tmp_path.name[:6]}"
    try:
        result = palette_preflight(str(out), scale="full")
        assert result["no_network"] is True
        assert result["status"] in ("passed", "problems")
        assert result["budget"]["planned_images"] == 21
        assert result["budget"]["generation_calls_upper_bound"] == 21
        assert result["budget"]["retry_env"] == {PALETTE_RETRY_ENV: PALETTE_CALL_ENV_TARGET}
        assert (out / "palette-preflight.json").is_file()
    finally:
        for path in sorted(out.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        out.rmdir()


def test_generation_plan_counts_calls_by_image(spec):
    plan = palette_generation_preflight(spec, str(RUN_DIR), "full")
    assert plan["images"] == 21
    assert plan["generation_calls_upper_bound"] == plan["images"]
    assert len(plan["samples"]) == 21
    assert len({sample["sample_id"] for sample in plan["samples"]}) == 21
    reduced = palette_generation_preflight(spec, str(RUN_DIR), "reduced")
    assert reduced["images"] == 9
    assert reduced["review_calls_planned"] == 3 and reduced["review_calls_contingency"] == 1


def test_request_preview_matches_the_generation_path(spec):
    preview = palette_request_preview(spec)
    assert preview["count"] == 7
    for row in preview["prompts"]:
        assert row["prompt"] == row["request"]["prompt"]
        assert "PRESERVE:" in row["prompt"], "所有组都必须带同一份保护块"
        if row["clauses"]:
            assert "COLOUR PLAN:" in row["prompt"]
        else:
            assert "COLOUR PLAN:" not in row["prompt"]
    assert preview["length_chars"]["max"] < 3000, "请求过长会压低参考图的相对权重；固定色板不挂参考图，但仍设上限"


def test_protection_block_allows_recolouring_only_existing_regions(spec):
    protection = spec["protection_block"].lower()
    assert "recolour only" in protection
    assert "waist ribbon sash" in protection and "ribbon bows" in protection
    assert "do not add objects" in protection
    assert "do not add a second person" in protection


def _valid_review_response(ids, clause_count):
    return {"images": [{"id": item_id, "colour_status": "conform", "colour_evidence": ["可见"],
                        "clauses": [{"index": index, "status": "conform", "regions": ["环境"],
                                     "observed_colours": ["冷蓝"], "evidence": ["可见"]}
                                    for index in range(1, clause_count + 1)],
                        "region_status": "conform", "region_evidence": ["点缀色只在腰带与鞋蝴蝶结"],
                        "extra_colour_areas": [], "protected_colours_ok": "true",
                        "protection_issues": [], "content_ok": "true", "content_issues": [],
                        "method_notes": [], "unverifiable_reason": []} for item_id in ids],
            "summary": "ok", "limitations": []}


def test_review_validator_accepts_a_complete_response():
    value = _valid_review_response(["Q01", "Q02", "Q03"], 7)
    validate_palette_review(value, ["Q01", "Q02", "Q03"], 7)


def test_review_validator_rejects_missing_image_or_clause_or_bad_status():
    missing_image = _valid_review_response(["Q01", "Q02"], 7)
    with pytest.raises(ValueError):
        validate_palette_review(missing_image, ["Q01", "Q02", "Q03"], 7)

    missing_clause = _valid_review_response(["Q01"], 7)
    missing_clause["images"][0]["clauses"].pop()
    with pytest.raises(ValueError):
        validate_palette_review(missing_clause, ["Q01"], 7)

    bad_status = _valid_review_response(["Q01"], 7)
    bad_status["images"][0]["colour_status"] = "good"
    with pytest.raises(ValueError):
        validate_palette_review(bad_status, ["Q01"], 7)

    zero_based = _valid_review_response(["Q01"], 1)
    zero_based["images"][0]["clauses"][0]["index"] = 0
    validate_palette_review(zero_based, ["Q01"], 1)
    assert zero_based["images"][0]["clauses"][0]["index"] == 1


def test_review_validator_accepts_zero_based_run_of_indices():
    """0 起连号 `[0,1,…,6]` 也必须整体挪到 1 起：旧判据「有 0 且没有 1」会漏掉这种。

    2026-10-04 实测：三组评审因为这条判据被误判为不合法，后来靠离线回收才用上。
    """
    value = _valid_review_response(["Q01"], 7)
    for entry in value["images"][0]["clauses"]:
        entry["index"] -= 1
    assert [entry["index"] for entry in value["images"][0]["clauses"]] == list(range(7))
    validate_palette_review(value, ["Q01"], 7)
    assert [entry["index"] for entry in value["images"][0]["clauses"]] == list(range(1, 8))


def test_review_validator_accepts_the_clauses_key_alias():
    """条款清单写在 `clauses` 或 `items` 都接受（模型两种都出现过）。"""
    value = _valid_review_response(["Q01"], 3)
    value["images"][0]["items"] = value["images"][0].pop("clauses")
    validate_palette_review(value, ["Q01"], 3)


def test_palette_review_recover_needs_products_and_makes_no_calls(tmp_path):
    """离线回收：没有产物时不能猜映射，必须按失败记录（且全程不发网络调用）。"""
    fixtures = Path("data/test-result/20261004/color-fixed-palette-v1")
    if not (fixtures / "review" / "P2-simple.raw.txt").is_file():
        pytest.skip("color-fixed-palette-v1 的评审证据不在工作区")
    out = Path("data/test-result/20261004") / f"color-fixed-palette-probe-{tmp_path.name[:6]}"
    (out / "review").mkdir(parents=True, exist_ok=True)
    (out / "review" / "P2-simple.raw.txt").write_text(
        (fixtures / "review" / "P2-simple.raw.txt").read_text(encoding="utf-8"), encoding="utf-8")
    try:
        from utils.color_experiment import palette_review_recover
        result = palette_review_recover(str(out), names=["P2-simple"], scale="full")
        assert result["recovered"] == []
        assert result["still_unusable"] and "没有成功产物" in result["still_unusable"][0]["reason"]
    finally:
        for path in sorted(out.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        out.rmdir()


def test_protocol_status_enum_is_explicit(spec):
    assert list(spec["evaluation"]["statuses"]) == list(PALETTE_STATES)
    assert spec["evaluation"]["promotion_rule"].startswith("一个版本 3/3 配色符合")
    assert set(spec["evaluation"]["requirement_types"]) == {"dominant", "regional", "exclusive"}
    assert "本协议没有任何组属于这一类" in spec["evaluation"]["requirement_types"]["exclusive"]


def test_palette_out_dir_stays_inside_test_result():
    from utils.color_experiment import _isolated_out
    with pytest.raises(ValueError):
        _isolated_out(str(Path("data") / "20261004" / "color-fixed-palette-v1"))


def test_spec_is_valid_json_without_hardcoded_secrets(spec):
    text = json.dumps(spec, ensure_ascii=False)
    assert "sk-" not in text and "Bearer " not in text
    assert spec["protection_block_id"] == "M"
    assert spec["prompt_layout"]["headers"]["color_plan"] == "COLOUR PLAN:"
