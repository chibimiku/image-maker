"""跨主题固定配色（cross-theme-palette-v1）的离线回归用例。

全部离线：mock 掉生图调用与文本评审，不联网、不收费、不写交付目录。

覆盖用户点名要求的六件事：
① 实际发送的 prompt 里真的含配色合同；
② 保护色与授权改色冲突时停下（不靠关掉门禁绕过）；
③ 合同/条款/主题/模型参数变化会让相关缓存失效；
④ 任务快照可恢复（已成功的复用，不重发）；
⑤ 有有效画风参考图时禁止注入配色；
⑥ 未启用配色时原请求逐字不变。
"""
import json
import os
from pathlib import Path

import pytest

from utils import color_experiment as ce

SPEC = ce.CROSS_THEME_SPEC_PATH


@pytest.fixture(autouse=True)
def _test_output_root(tmp_path, monkeypatch):
    root = tmp_path / "test-result-root"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(ce.TEST_OUTPUT_ROOT_ENV, str(root))
    return root


@pytest.fixture()
def spec():
    return ce.load_cross_theme_spec(SPEC)


def _out(tmp_path):
    return Path(os.environ[ce.TEST_OUTPUT_ROOT_ENV]) / "cross"


# --- 合同与计划 ---------------------------------------------------------------

def test_contract_has_all_structured_fields(spec):
    contract = ce.cross_theme_contract(spec)
    block = contract["contract"]
    for key in ("contract_id", "version", "authorized_recolor_regions",
                "protected_in_every_theme", "forbidden_additions", "negative",
                "allowed_neutrals", "colors"):
        assert key in block, key
    assert contract["contract_hash"]
    assert sorted(block["authorized_recolor_regions"]) == sorted(ce.CROSS_THEME_ACCENT_REGIONS)
    for scheme_id, colours in block["colors"].items():
        assert colours["main"]["hex"] and colours["accent"]["hex"]
        assert colours["main"]["role"] and colours["accent"]["role"]


def test_plan_is_sixteen_requests_with_two_controls_per_theme(spec, tmp_path):
    plan = ce.cross_theme_plan(str(_out(tmp_path)), spec_path=SPEC, log=lambda *a, **k: None)
    rows = plan["requests"]
    assert len(rows) == 16
    from collections import Counter
    per_theme = Counter(row["theme"] for row in rows)
    assert set(per_theme.values()) == {8}
    controls = [row for row in rows if row["is_control"]]
    assert len(controls) == 4
    assert all(row["repeat"] in (1, 2) for row in rows)
    assert all(row["reference_images"] == [] and row["image_paths"] == [] for row in rows)


def test_theme_regions_exist_and_accent_never_adds_objects(spec):
    for theme in spec["themes"]:
        existing = set(theme["existing_regions"])
        for region in ce.CROSS_THEME_ACCENT_REGIONS:
            assert region in theme["region_mapping"], region
            assert region in existing
        assert "environment_dominant_region" in theme["region_mapping"]
        # 点缀只能落在既有区域：绑定条款必须明确禁止新增区域/物件
        prompt = ce.cross_theme_prompt(spec, theme, spec["schemes"][0])
        assert "Do not create any new region or object" in prompt
        assert "already present" in prompt


# --- ① 实际发送的 prompt 含合同 -----------------------------------------------

def test_sent_prompt_contains_contract_and_region_binding(spec):
    for row in ce.cross_theme_requests(spec):
        if row["is_control"]:
            assert "COLOUR PLAN:" not in row["prompt"]
            assert "PRESERVE:" in row["prompt"]
            continue
        scheme = next(s for s in spec["schemes"] if s["id"] == row["scheme"])
        assert "COLOUR PLAN:" in row["prompt"]
        for clause in scheme["clauses"]:
            assert clause in row["prompt"], clause[:60]
        assert f'give the environment a {scheme["main"]["name"]}'.lower() in row["prompt"].lower() \
            or scheme["main"]["name"] in row["prompt"].lower()
        assert scheme["accent"]["name"] in row["prompt"].lower()
        assert row["colour_plan_present"] is True
        assert "PRESERVE:" in row["prompt"]
        assert row["prompt_sha256"] == ce._sha256_str(row["prompt"])


def test_mock_capture_shows_prompt_actually_sent(spec, tmp_path, monkeypatch):
    captured = []

    def fake_generate(**kwargs):
        captured.append(kwargs)
        stage = Path(kwargs["save_sub_dir"])
        stage.mkdir(parents=True, exist_ok=True)
        path = stage / f"{kwargs['file_prefix']}-fake.png"
        from PIL import Image
        Image.new("RGB", (32, 24), (180, 210, 235)).save(path)
        return [str(path)]

    monkeypatch.setattr("modules.others.api_backend.generate_image_aigc2d", fake_generate)
    out = _out(tmp_path)
    monkeypatch.setenv(ce.send_budget_env()[0], str(out / "send-budget"))
    monkeypatch.setenv(ce.send_budget_env()[1], "16")
    monkeypatch.setenv("IMAGE_MAKER_IMAGE_MAX_RETRIES", "0")
    result = ce.cross_theme_run(str(out), spec_path=SPEC, log=lambda *a, **k: None)
    assert result["stopped"] is None
    assert len(captured) == 16
    for kwargs in captured:
        assert kwargs["image_paths"] is None, "本轮不附任何参考图"
        assert "PRESERVE:" in kwargs["prompt"]
    controls = [kwargs for kwargs in captured if "COLOUR PLAN:" not in kwargs["prompt"]]
    assert len(controls) == 4, "无配色对照的正文里不应出现 COLOUR PLAN"


# --- ② 冲突拦截 ---------------------------------------------------------------

def test_conflict_between_protected_and_authorized_stops_the_run(spec):
    report = ce.cross_theme_conflict_report(spec)
    assert report["status"] == "ok"
    broken = json.loads(json.dumps(spec))
    broken["themes"][0]["protected_regions"] = list(broken["themes"][0]["protected_regions"]) + \
        ["waist_band_region"]
    report = ce.cross_theme_conflict_report(broken)
    assert report["status"] == "conflict"
    assert any("重叠" in problem for problem in report["problems"])
    # 冲突报告只说明问题，不动任何门禁开关
    assert "不通过关闭或放宽身份/质量门禁绕过" in report["note"]


def test_conflict_blocks_execution_before_any_request(tmp_path, monkeypatch, spec):
    broken = json.loads(json.dumps(spec))
    # 把某个主题的落点区域映射到不存在的区域 → 冲突检查必须拦下（映射完整性）
    broken["themes"][1]["region_mapping"]["waist_band_region"]["maps_to"] = "region_that_does_not_exist"
    broken_path = Path(os.environ[ce.TEST_OUTPUT_ROOT_ENV]) / "broken-spec.json"
    broken_path.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
    calls = []
    monkeypatch.setattr("modules.others.api_backend.generate_image_aigc2d",
                        lambda **kwargs: calls.append(kwargs) or [])
    preflight = ce.cross_theme_preflight(str(_out(tmp_path)), spec_path=str(broken_path),
                                         log=lambda *a, **k: None)
    assert preflight["status"] == "problems"
    assert any("不在该主题已有区域里" in problem for problem in preflight["problems"])
    assert not calls, "冲突未解决前不允许发出任何请求"


def test_malformed_region_mapping_is_rejected_early(spec, tmp_path):
    """区域映射必须是 {maps_to, describes}：写错结构要在加载阶段就报错，不能悄悄继续。"""
    broken = json.loads(json.dumps(spec))
    broken["themes"][0]["region_mapping"]["waist_band_region"] = "just a string"
    path = Path(os.environ[ce.TEST_OUTPUT_ROOT_ENV]) / "malformed-spec.json"
    path.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        ce.load_cross_theme_spec(str(path))


def test_conflict_protected_overlap_blocks_execution(tmp_path, monkeypatch, spec):
    broken = json.loads(json.dumps(spec))
    broken["themes"][0]["protected_regions"] = list(broken["themes"][0]["protected_regions"]) + \
        ["left_shoe_bow_region"]
    broken_path = Path(os.environ[ce.TEST_OUTPUT_ROOT_ENV]) / "overlap-spec.json"
    broken_path.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
    calls = []
    monkeypatch.setattr("modules.others.api_backend.generate_image_aigc2d",
                        lambda **kwargs: calls.append(kwargs) or [])
    result = ce.cross_theme_run(str(_out(tmp_path)), spec_path=str(broken_path), log=lambda *a, **k: None)
    assert result["stopped"] == "preflight_problems"
    assert calls == [], "保护色与授权改色冲突时，必须先停下说明而不是继续发请求"


# --- ③ 合同变化使缓存失效 -----------------------------------------------------

def test_contract_or_clause_change_invalidates_cache(spec, tmp_path):
    out = _out(tmp_path)
    preflight = ce.cross_theme_preflight(str(out), spec_path=SPEC, log=lambda *a, **k: None)
    frozen = ce.cross_theme_freeze(str(out), preflight, log=lambda *a, **k: None)
    assert ce.cross_theme_generation_preflight(spec, frozen) == []
    assert ce.cross_theme_cache_eligibility(frozen, spec=spec, request_id="x")["eligible"] is True

    changed = json.loads(json.dumps(spec))
    changed["schemes"][0]["clauses"][0] = changed["schemes"][0]["clauses"][0] + " Extra sentence."
    differences = ce.cross_theme_generation_preflight(changed, frozen)
    assert any("条款变化" in row for row in differences), differences
    state = ce.cross_theme_cache_eligibility(frozen, spec=changed, request_id="x")
    assert state["eligible"] is False and state["state"] == "invalidated"

    changed_contract = json.loads(json.dumps(spec))
    changed_contract["contract"]["authorized_recolor_regions"] = ["waist_band_region"]
    assert any("合同内容变化" in row
               for row in ce.cross_theme_generation_preflight(changed_contract, frozen))

    changed_theme = json.loads(json.dumps(spec))
    changed_theme["themes"][0]["subject"] = changed_theme["themes"][0]["subject"] + " Extra."
    assert any("主题文本变化" in row
               for row in ce.cross_theme_generation_preflight(changed_theme, frozen))


def test_frozen_run_refuses_a_changed_protocol(spec, tmp_path, monkeypatch):
    out = _out(tmp_path)
    preflight = ce.cross_theme_preflight(str(out), spec_path=SPEC, log=lambda *a, **k: None)
    ce.cross_theme_freeze(str(out), preflight, log=lambda *a, **k: None)
    changed = json.loads(json.dumps(spec))
    changed["prompt_layout"]["protection_block"] = changed["prompt_layout"]["protection_block"] + " Extra."
    changed_path = Path(os.environ[ce.TEST_OUTPUT_ROOT_ENV]) / "changed-spec.json"
    changed_path.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
    calls = []
    monkeypatch.setattr("modules.others.api_backend.generate_image_aigc2d",
                        lambda **kwargs: calls.append(kwargs) or [])
    with pytest.raises(ValueError):
        ce.cross_theme_run(str(out), spec_path=str(changed_path), log=lambda *a, **k: None)
    assert calls == []


# --- ④ 任务快照可恢复 ---------------------------------------------------------

def test_snapshot_marks_pending_and_resumes_without_resending(spec, tmp_path, monkeypatch):
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs["file_prefix"])
        stage = Path(kwargs["save_sub_dir"])
        stage.mkdir(parents=True, exist_ok=True)
        path = stage / f"{kwargs['file_prefix']}-fake.png"
        from PIL import Image
        Image.new("RGB", (32, 24), (200, 220, 235)).save(path)
        return [str(path)]

    monkeypatch.setattr("modules.others.api_backend.generate_image_aigc2d", fake_generate)
    out = _out(tmp_path)
    snap = ce.cross_theme_snapshot(str(out), spec=spec, log=lambda *a, **k: None)
    assert snap["done"] == 0 and snap["total"] == 16
    assert all(row["status"] == "pending" for row in snap["rows"])

    monkeypatch.setenv(ce.send_budget_env()[0], str(out / "send-budget"))
    monkeypatch.setenv(ce.send_budget_env()[1], "16")
    monkeypatch.setenv("IMAGE_MAKER_IMAGE_MAX_RETRIES", "0")
    first = ce.cross_theme_run(str(out), spec_path=SPEC, log=lambda *a, **k: None)
    assert len(calls) == 16
    assert first["summary"]["generation_success"] == 16
    snap2 = ce.cross_theme_snapshot(str(out), spec=spec, log=lambda *a, **k: None)
    assert snap2["done"] == 16
    # 再次运行：全部命中已成功产物，不再发送任何请求
    calls.clear()
    second = ce.cross_theme_run(str(out), spec_path=SPEC, log=lambda *a, **k: None)
    assert calls == [], "已成功的样本必须复用，不重发"
    assert second["summary"]["generation_success"] == 16


# --- ⑤⑥ 参考图与未启用配色 ----------------------------------------------------

def test_colour_injection_is_refused_when_a_style_reference_exists(spec, tmp_path):
    ref = Path(os.environ[ce.TEST_OUTPUT_ROOT_ENV]) / "style-ref.jpg"
    from PIL import Image
    Image.new("RGB", (16, 16), (120, 120, 140)).save(ref)
    allowed, reason = ce.cross_theme_colour_injection_allowed(style_ref_path=str(ref))
    assert allowed is False and "禁止" in reason
    allowed, _ = ce.cross_theme_colour_injection_allowed(style_ref_clauses=["some clause"])
    assert allowed is False
    allowed, _ = ce.cross_theme_colour_injection_allowed(request={"extra_reference_paths": ["x.png"]})
    assert allowed is False
    allowed, reason = ce.cross_theme_colour_injection_allowed()
    assert allowed is True and "允许" in reason


def test_disabled_colour_plan_keeps_the_original_request_byte_identical():
    original = {"prompt": "An anime illustration of one young woman. PRESERVE: keep everything."}
    before = json.dumps(original, ensure_ascii=False, sort_keys=True)
    passthrough = ce.cross_theme_disabled_request(original)
    assert passthrough["prompt"] == original["prompt"]
    assert passthrough["prompt_sha256"] == ce._sha256_str(original["prompt"])
    assert passthrough["extra_calls"] == 0
    assert json.dumps(original, ensure_ascii=False, sort_keys=True) == before, "原请求不得被改写"


def test_control_group_body_has_no_colour_plan(spec):
    control = next(row for row in ce.cross_theme_requests(spec) if row["is_control"])
    assert control["colour_plan_present"] is False
    assert "COLOUR PLAN:" not in control["prompt"]
    assert control["prompt"].count("PRESERVE:") == 1


# --- 评审校验 -----------------------------------------------------------------

def test_review_validator_requires_three_accent_regions_and_no_aesthetic_score():
    good = {"images": [{"id": "Q01", "main_colour_status": "conform", "accent_status": "conform",
                        "colour_status": "conform", "waist_band_region": "present",
                        "left_shoe_bow_region": "present", "right_shoe_bow_region": "unverifiable",
                        "protected_colours_ok": "true"}]}
    ce.validate_cross_theme_review(good, ["Q01"])
    missing = json.loads(json.dumps(good))
    del missing["images"][0]["right_shoe_bow_region"]
    with pytest.raises(ValueError):
        ce.validate_cross_theme_review(missing, ["Q01"])
    illegal = json.loads(json.dumps(good))
    illegal["images"][0]["colour_status"] = "beautiful"
    with pytest.raises(ValueError):
        ce.validate_cross_theme_review(illegal, ["Q01"])
    scored = json.loads(json.dumps(good))
    scored["images"][0]["aesthetic_score"] = 9
    with pytest.raises(ValueError):
        ce.validate_cross_theme_review(scored, ["Q01"])
    extra = json.loads(json.dumps(good))
    extra["images"].append(dict(good["images"][0], id="Q02"))
    with pytest.raises(ValueError):
        ce.validate_cross_theme_review(extra, ["Q01"])
