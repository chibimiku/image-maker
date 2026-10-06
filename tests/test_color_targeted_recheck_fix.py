"""定向只读文本复核（retention targeted recheck）的离线回归用例。

只读 + 离线：不联网、不收费、不改写任何已交付目录。
锁定三件事：
① 三条冻结请求的计划 hash 与旧运行完全一致（协议不得边跑边改）；
② 冻结请求的图序、角色、图片 hash 与源实验产物 hash 对得上；
③ 输入副本校验与响应校验的口径正确（少一项、非法枚举、乱做因果归因都要被拒）。
"""
import hashlib
import json
from pathlib import Path

import pytest

from utils import color_experiment as ce

V1 = Path("data/test-result/20261004/color-palette-targeted-recheck-v1")
V2 = Path("data/test-result/20261004/color-palette-targeted-recheck-v2")
V2_SPEC = "prompts/color-knowledge/fixed-palette-retention-v2.json"
FROZEN_PLAN_HASH = "4c8d9bf87474bf68c3b02a18dc50dae43a2d4e263d223542aab84376973fb351"


def _plan(base):
    path = base / "recheck-preflight.json"
    if not path.is_file():
        pytest.skip(f"{base} 的冻结计划不在工作区")
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_plan_hash_is_unchanged_across_runs():
    """两次运行必须共用同一份冻结计划：hash 不一致就说明协议被改过。"""
    old, new = _plan(V1), _plan(V2)
    assert old["plan_hash"] == new["plan_hash"] == FROZEN_PLAN_HASH
    assert new["budget"] == {"image_calls": 0, "text_calls_max": 3, "automatic_retries": 0}
    assert new["execution_authorized"] is False, "预检文件本身不授权执行"


def test_frozen_requests_keep_order_roles_and_hashes():
    plan = _plan(V2)
    by_id = {row["id"]: row for row in plan["requests"]}
    assert sorted(by_id) == ["boundary-B-r1", "colour-A", "colour-B"]
    assert [im["role"] for im in by_id["colour-A"]["images"]] == [
        "Q01_source_full", "Q01_candidate_full", "Q02_source_full", "Q02_candidate_full",
        "style_reference_full"]
    assert [im["role"] for im in by_id["colour-B"]["images"]] == [
        "Q01_source_full", "Q01_candidate_full", "Q02_source_full", "Q02_candidate_full",
        "style_reference_full"]
    assert [im["role"] for im in by_id["boundary-B-r1"]["images"]] == ["source_full", "candidate_full"]
    # 三条请求的 system/user 与 request hash 都在冻结时就固定
    for row in plan["requests"]:
        assert row["system_sha256"] and row["user_sha256"] and row["request_hash"]
        assert row["model"] and row["endpoint"] and row["max_completion_tokens"] == 4000
        # request hash 必须覆盖提示词与图片清单（改任一项都会变）
        body = {key: row[key] for key in
                ("id", "purpose", "system_prompt_file", "user", "images", "model", "endpoint",
                 "max_completion_tokens")}
        first = ce.digest(body)
        body["user"] = body["user"] + " "
        assert ce.digest(body) != first, "request hash 必须真的覆盖请求内容"
        body["user"] = row["user"]
        body["images"] = body["images"][:-1]
        assert ce.digest(body) != first, "request hash 必须覆盖图片清单"


def test_frozen_images_match_source_evidence_hashes():
    """冻结请求里的图片 hash 必须等于源实验记录里的产物 hash（不能指向别的东西）。"""
    plan = _plan(V2)
    for row in plan["requests"]:
        for entry in row["images"]:
            path = Path(entry["path"])
            if not path.is_file():
                pytest.skip(f"{path} 不在工作区")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], path.name


def test_evidence_copies_match_frozen_hashes():
    if not (V2 / "input-evidence").is_dir():
        pytest.skip("本轮输入副本不在工作区")
    plan = _plan(V2)
    for row in plan["requests"]:
        for index, entry in enumerate(row["images"], 1):
            copy = V2 / "input-evidence" / row["id"] / f"{index:02d}{Path(entry['path']).suffix}"
            assert copy.is_file(), copy
            assert hashlib.sha256(copy.read_bytes()).hexdigest() == entry["sha256"]


def test_validation_rejects_incomplete_or_illegal_colour_answers():
    good = {"images": [
        {"id": "Q01", "main_colour_status": "conform", "colour_status": "conform",
         "accent_waistband": "present", "accent_left_shoe_bow": "present",
         "accent_right_shoe_bow": "present", "protected_colours_status": "retained",
         "reference_influence": {"status": "none_observed", "causal_attribution": "not_established"}},
        {"id": "Q02", "main_colour_status": "partial", "colour_status": "partial",
         "accent_waistband": "present", "accent_left_shoe_bow": "present",
         "accent_right_shoe_bow": "unverifiable", "protected_colours_status": "retained",
         "reference_influence": {"status": "suspected", "causal_attribution": "not_established"}}]}
    ce.validate_targeted_recheck(good, "colour-A")            # 合法
    with pytest.raises(ValueError):
        ce.validate_targeted_recheck({"images": good["images"][:1]}, "colour-A")   # 少一张
    illegal = json.loads(json.dumps(good))
    illegal["images"][0]["colour_status"] = "great"
    with pytest.raises(ValueError):
        ce.validate_targeted_recheck(illegal, "colour-A")
    missing_bow = json.loads(json.dumps(good))
    del missing_bow["images"][0]["accent_right_shoe_bow"]
    with pytest.raises(ValueError):
        ce.validate_targeted_recheck(missing_bow, "colour-A")
    causal = json.loads(json.dumps(good))
    causal["images"][1]["reference_influence"]["causal_attribution"] = "proven"
    with pytest.raises(ValueError):
        ce.validate_targeted_recheck(causal, "colour-A")


def test_validation_rejects_incomplete_boundary_answers():
    good = {"candidate_shoes": [
        {"screen_side": "left", "status": "contained"},
        {"screen_side": "right", "status": "touching_frame"}],
        "scope": "framing_diagnostic_only_not_production_gate"}
    ce.validate_targeted_recheck(good, "boundary-B-r1")
    with pytest.raises(ValueError):
        ce.validate_targeted_recheck({"candidate_shoes": good["candidate_shoes"][:1],
                                      "scope": good["scope"]}, "boundary-B-r1")
    illegal = json.loads(json.dumps(good))
    illegal["candidate_shoes"][0]["status"] = "cropped"
    with pytest.raises(ValueError):
        ce.validate_targeted_recheck(illegal, "boundary-B-r1")
    override = json.loads(json.dumps(good))
    override["scope"] = "production_final_review"
    with pytest.raises(ValueError):
        ce.validate_targeted_recheck(override, "boundary-B-r1")


def test_run_records_do_not_claim_production_gate_changes():
    path = V2 / "recheck-results.json"
    if not path.is_file():
        pytest.skip("本轮运行汇总不在工作区")
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["production_gates_changed"] is False
    assert summary["image_calls"] == 0
    assert summary["text_reservations"] <= summary["authorized_text_limit"] == 3
    for row in summary["reviews"]:
        assert row["status"] in ("success", "failed_or_unknown")
        if row["status"] != "success":
            assert row.get("billing", "").startswith("unknown"), "失败必须记费用未知，不能宣称免费"
