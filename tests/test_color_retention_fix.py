"""固定色板重绘保留验证（v2 修复版）的离线回归用例。

全部离线：mock 掉后端图片调用与四处门禁审计，不联网、不收费、不改写任何已交付实验。

覆盖的正是 v1 栽过跟头的地方：
① B 的配色合同必须写进**真正发送**的 `call_kwargs.prompt`（v1 只改顶层 prompt，
   四次实际发送的正文都是同一条 11541 字符、未带合同）；
② mock 捕获实际后端调用参数：A 不含追加合同、B 完整含合同，其余输入与参数一致；
③ 冻结记录不可被续跑覆盖，模型/节点/参数/提示词/参考图或传输字节 hash 变化就拒绝复用；
④ 成功产物与成功审计按候选 hash + 请求 hash + 审计协议复用：续跑零调用、
   「只补审计」不触发重绘；
⑤ 门禁预算按 4 候选 × 4 项 = 16 次计，未执行/预算阻止/审计错误/明确不通过分开记录；
⑥ 人体门禁看结构缺陷、结论有效性与归属不确定，不自行放宽。
"""
import json
import os
from pathlib import Path
import uuid

import pytest

from utils import color_experiment as ce

V2_SPEC = "prompts/color-knowledge/fixed-palette-retention-v2.json"
V1_SPEC = "prompts/color-knowledge/fixed-palette-retention-v1.json"
V1_RUN = Path("data/test-result/20261004/color-palette-retention-v1")


def _tmp_run(tmp_path, name="run"):
    """测试用的实验目录：在测试输出根之下（root/run），绝不碰交付目录。"""
    root = Path(os.environ.get(ce.TEST_OUTPUT_ROOT_ENV) or tmp_path)
    return root / name


def _fake_dispatch(record, outputs):
    """替换 dispatch_repaint_request：记录被调用时的真实参数，并写出假产物。"""
    def fake(request, log_callback=None):
        record.append(json.loads(json.dumps(request)))
        folder = Path(request["call_kwargs"]["save_sub_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{request['call_kwargs']['file_prefix']}-fake.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + request["sent_prompt_sha256"].encode()[:32])
        return [str(path)]
    return fake


@pytest.fixture(autouse=True)
def _test_output_root(tmp_path, monkeypatch):
    """测试专用输出根：实验代码只在 pytest 显式设置时才用临时目录，生产仍受 data/test-result 约束。"""
    root = tmp_path / "test-result-root"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(ce.TEST_OUTPUT_ROOT_ENV, str(root))
    return root


@pytest.fixture()
def v2_spec():
    return ce.load_retention_spec(V2_SPEC)


def _prepare_env(out: Path, spec) -> Path:
    """准备好预算目录与限流环境（真实运行里由 run_retention 做的同一件事）。"""
    out.mkdir(parents=True, exist_ok=True)
    ce._retention_prepare_env(out, spec)
    return out


def _write_png(path: Path, size=(8, 12), colour=(180, 210, 235)):
    """写一张真 PNG（门禁/逐张记录要读图片尺寸，假字节会被 PIL 拒绝）。"""
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path)
    return path


# --- ① / ② 合同真的进到发送参数 -------------------------------------------------

def test_contract_is_written_into_the_sent_call_kwargs(v2_spec, tmp_path):
    requests = ce.retention_requests(v2_spec, out_dir=_tmp_run(tmp_path))
    contract = ce.retention_contract_text()
    for request in requests:
        top = request["prompt"]
        sent = request["call_kwargs"]["prompt"]
        assert top == sent, "顶层 prompt 与实际发送的 call_kwargs.prompt 必须完全一致"
        assert request["prompt_matches_sent"] is True
        assert request["prompt_sha256"] == request["sent_prompt_sha256"]
        assert request["prompt_chars"] == request["sent_prompt_chars"]
        has_contract = "SOURCE COLOUR CONTRACT" in sent and "STYLE MIGRATION PERMITTED" in sent
        assert has_contract is (request["branch"] == "B")
        if request["branch"] == "B":
            assert contract in sent
            # 合同必须落在正文末尾（A 的正文逐字不动）
            assert sent.rstrip().endswith(contract.splitlines()[-1].strip())


def test_branch_b_is_a_plus_contract_only(v2_spec, tmp_path):
    diffs = ce.retention_request_diff(ce.retention_requests(v2_spec, out_dir=_tmp_run(tmp_path)))
    assert len(diffs) == 2
    contract = ce.retention_contract_text()
    for row in diffs:
        assert row["b_equals_a_plus_contract"] is True
        assert row["same_inputs_and_params"] is True, "除合同与落盘标识外，A/B 输入与参数必须一致"
        assert row["delta_chars"] == len(contract) + 3, \
            "B 只比 A 多出「空行 + 合同 + 结尾换行」"
        assert row["b_prompt_sha256"] == row["b_sent_prompt_sha256"]
        assert row["b_top_matches_sent"] is True


def test_mock_captures_what_the_backend_receives(v2_spec, tmp_path, monkeypatch):
    """mock 后端：确认 A 没有合同、B 有完整合同，且除合同外参数一致。"""
    captured = []
    monkeypatch.setattr("utils.post_process.dispatch_repaint_request",
                        _fake_dispatch(captured, None))
    out = _prepare_env(_tmp_run(tmp_path), v2_spec)
    for request in ce.retention_requests(v2_spec, out_dir=out):
        ce._retention_repaint_stage(out, v2_spec, request, log=lambda *a, **k: None)
    assert len(captured) == 4
    by_branch = {}
    for request in captured:
        by_branch.setdefault(request["branch"], []).append(request)
    contract = ce.retention_contract_text()
    for request in by_branch["A"]:
        assert "SOURCE COLOUR CONTRACT" not in request["call_kwargs"]["prompt"]
        assert request["colour_contract_appended"] is False
    for request in by_branch["B"]:
        assert contract in request["call_kwargs"]["prompt"]
        assert request["colour_contract_appended"] is True
    for sample_id in ("gemini-flash-P1-knowledge-r1", "gemini-flash-P1-knowledge-r2"):
        a = next(r for r in by_branch["A"] if r["sample_id"] == sample_id)
        b = next(r for r in by_branch["B"] if r["sample_id"] == sample_id)
        for key in ("model", "api_type", "resolution", "aspect_ratio", "repeat", "use_detail_suffix"):
            assert a["call_kwargs"][key] == b["call_kwargs"][key], key
        assert [r["transmitted_sha256"] for r in a["reference_images"]] == \
               [r["transmitted_sha256"] for r in b["reference_images"]]


# --- ③ 冻结核对 ---------------------------------------------------------------

def test_freeze_rejects_reuse_when_contract_or_prompt_changes(v2_spec, tmp_path, monkeypatch):
    out = _tmp_run(tmp_path)
    out.mkdir(parents=True, exist_ok=True)
    preflight = ce.retention_preflight(str(out), spec_path=V2_SPEC, log=lambda *a, **k: None)
    frozen = ce.retention_freeze(str(out), preflight, log=lambda *a, **k: None)
    assert ce._retention_freeze_check(out, frozen) == []

    tampered = json.loads(json.dumps(frozen))
    tampered["contract"]["text_sha256"] = "0" * 64
    differences = ce._retention_freeze_check(out, tampered)
    assert any("合同" in row for row in differences)

    tampered2 = json.loads(json.dumps(frozen))
    first_key = sorted(tampered2["request_hashes"])[0]
    tampered2["request_hashes"][first_key] = "f" * 64
    differences2 = ce._retention_freeze_check(out, tampered2)
    assert any("请求 hash 变化" in row for row in differences2), differences2

    monkeypatch.setattr(ce, "retention_requests", lambda spec, **kwargs: [])
    differences3 = ce._retention_freeze_check(out, frozen)
    assert differences3, "请求集合变化必须被发现"


def test_run_stops_when_the_a_baseline_does_not_match(v2_spec, tmp_path, monkeypatch):
    out = _tmp_run(tmp_path)
    check = ce.retention_a_baseline_check(v2_spec, out_dir=out)
    assert check["status"] in ("ok", "mismatch")
    broken = {**check, "enabled": True, "status": "mismatch", "mismatches": ["prompt hash 不一致"]}
    monkeypatch.setattr(ce, "retention_a_baseline_check", lambda spec, **kwargs: broken)
    result = ce._retention_setup_candidates(out, v2_spec, only_b=True, log=lambda *a, **k: None)
    assert result["stopped"] == "a_baseline_mismatch"
    assert not (out / "stages").exists() or not list((out / "stages").glob("*-initial-repaint"))


# --- ④ 复用：续跑零调用、只补审计零生图 -----------------------------------------

def _write_candidate(out: Path, branch, sample_id, text="candidate"):
    folder = out / "stages" / f"{branch}-{sample_id}-initial-repaint"
    folder.mkdir(parents=True, exist_ok=True)
    image = _write_png(folder / f"{branch}-{sample_id}.png")
    record = {"stage": "initial-repaint", "branch": branch, "sample_id": sample_id,
              "status": "success", "outputs": [str(image)],
              "output_sha256": {image.name: ce._sha256_file(image)},
              "sent_prompt_sha256": text, "request_id": f"{branch}-{sample_id}"}
    (folder / "result.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return image


def test_audit_only_makes_no_image_calls(v2_spec, tmp_path, monkeypatch):
    out = _tmp_run(tmp_path)
    _write_candidate(out, "A", "gemini-flash-P1-knowledge-r1")
    _write_candidate(out, "A", "gemini-flash-P1-knowledge-r2")

    def boom(*args, **kwargs):
        raise AssertionError("只补审计不允许触发任何图片生成")

    monkeypatch.setattr("utils.post_process.dispatch_repaint_request", boom)
    monkeypatch.setattr("utils.post_process.assemble_repaint_request", boom)
    calls = []

    def fake_identity(candidate, analysis, **kwargs):
        calls.append("identity")
        return {"mismatch": False, "severity": "none", "confidence": 0.9, "differences": []}

    monkeypatch.setattr("utils.identity_audit.audit_image_identity", fake_identity)
    monkeypatch.setattr("utils.refine_quality.audit_refine_quality",
                        lambda *a, **k: {"structural_issues": [], "line_issues": [],
                                         "background_drift": [], "style_gaps": [],
                                         "needs_refine": False, "severity": "none", "confidence": 0.9})
    monkeypatch.setattr("utils.refine_quality.audit_hand_quality",
                        lambda *a, **k: {"structural_issues": [], "character_limb_inventory": [],
                                         "summary": "ok", "needs_refine": False,
                                         "ownership_uncertain": []})
    monkeypatch.setattr("utils.refine_quality.record_quality_audit", lambda *a, **k: None)
    monkeypatch.setattr(ce, "RETENTION_AUDIT_KINDS", ("identity", "quality", "hands", "final_review"))

    monkeypatch.setenv("IMAGE_MAKER_SEND_BUDGET_DIR", str(out / "send-budget"))
    monkeypatch.setenv("IMAGE_MAKER_IMAGE_MAX_RETRIES", "0")
    monkeypatch.setenv("IMAGE_MAKER_SEND_BUDGET_LIMIT", "3")
    result = ce.retention_audit_only(str(out), spec_path=V2_SPEC, log=lambda *a, **k: None)
    assert result["image_requests"] == 0
    assert len(result["fresh"]) == 8, "两个 A 候选 × 四项审计 = 8 次新调用"
    assert result["text_requests"] == 8
    assert len(calls) == 2


def test_audit_reuse_matches_candidate_hash_and_skips_calls(tmp_path, monkeypatch):
    legacy = _tmp_run(tmp_path, "legacy")
    image = _write_candidate(legacy, "A", "gemini-flash-P1-knowledge-r1")
    sha = ce._sha256_file(image)
    gates = legacy / "stages" / "A-gemini-flash-P1-knowledge-r1-gates"
    gates.mkdir(parents=True, exist_ok=True)
    (gates / "gates.json").write_text(json.dumps({
        "branch": "A", "sample_id": "gemini-flash-P1-knowledge-r1", "candidate_sha256": sha,
        "audits": {"identity": {"mismatch": False, "severity": "none", "confidence": 0.9,
                                "differences": [], "gate_action": "accept"}},
        "final_decision": {"action": "accept"}, "available_after_gates": True}, ensure_ascii=False),
        encoding="utf-8")
    spec = {"id": "t", "version": 2,
            "sources": [{"sample_id": "gemini-flash-P1-knowledge-r1",
                         "image": "data/test-result/20261004/color-fixed-palette-v1/samples/"
                                  "gemini-flash/gemini-flash-P1-knowledge-r1/"
                                  "test-gemini-flash-P1-knowledge-r1_144341-6bef63.png"}],
            "style": {"ref_image": "data/style-ref/tid-fullbody-background-candidate.jpg",
                      "reference_mode": "style", "scope": "full"},
            "generation": {"resolution": "2K", "repeat": 1, "use_detail_suffix": False},
            "audit_reuse": {"allowed_sources": [str(legacy)]}}
    reuse = ce._retention_reuse_map(_tmp_run(tmp_path, "out"), spec, None, reuse_from=str(legacy))
    assert ("A", "gemini-flash-P1-knowledge-r1", "identity", sha) in reuse
    # 旧记录没有完整指纹：可以复用，但资格必须记「未验证」
    assert reuse[("A", "gemini-flash-P1-knowledge-r1", "identity", sha)]["reuse_eligibility"]["state"] \
        == "unverified_fingerprint"

    state = {"reuse": reuse, "reused": [], "fresh": [], "blocked": [], "errored": [],
             "mode": "audit_only", "used_text": 0, "text_limit": 16}
    audit, source = ce._retention_audit_call(state, spec, kind="identity", branch="A",
                                             sample_id="gemini-flash-P1-knowledge-r1",
                                             candidate_sha=sha,
                                             call=lambda: pytest.fail("命中复用就不该再调用"),
                                             log=lambda *a, **k: None)
    assert source == "reused"
    assert audit["gate_action"] == "accept"
    assert state["used_text"] == 0, "复用不得消耗文本预算"

    audit2, source2 = ce._retention_audit_call(state, spec, kind="identity", branch="A",
                                               sample_id="gemini-flash-P1-knowledge-r1",
                                               candidate_sha="0" * 64,
                                               call=lambda: {"mismatch": True}, log=lambda *a, **k: None)
    assert source2 == "fresh", "候选 hash 不匹配就不能复用"


def test_unknown_reuse_source_is_refused(tmp_path, monkeypatch):
    legacy = _tmp_run(tmp_path, "legacy")
    legacy.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        ce._retention_reuse_map(_tmp_run(tmp_path, "out"),
                                {"id": "t", "audit_reuse": {"allowed_sources": ["data/test-result/别的目录"]}},
                                None, reuse_from=str(legacy))


# --- ⑤ / ⑥ 门禁预算与人体判据 --------------------------------------------------

def test_gate_budget_counts_four_audits_per_candidate(v2_spec):
    budget = ce.retention_generation_budget(v2_spec, planned_images=2)
    assert budget["text_gates_needed_full"] == 16, "4 候选 × 4 项 = 16，不是 8"
    assert budget["text_planned_now"] == 10
    assert int(v2_spec["budget"]["text_hard_limit"]) == 20
    assert int(v2_spec["budget"]["image_hard_limit"]) == 3


def test_budget_blocked_audits_are_recorded_separately(tmp_path):
    state = {"reuse": {}, "reused": [], "fresh": [], "blocked": [], "errored": [],
             "mode": "audit_only", "used_text": 1, "text_limit": 1}
    spec = {"id": "t"}
    audit, source = ce._retention_audit_call(state, spec, kind="quality", branch="B",
                                             sample_id="s", candidate_sha="a" * 64,
                                             call=lambda: pytest.fail("预算用满不该调用"),
                                             log=lambda *a, **k: None)
    assert source == "budget_blocked"
    assert audit["budget_blocked"] is True
    assert audit["candidate_sha256"] == "a" * 64
    assert state["blocked"] and not state["fresh"]


def _gates(audits, decision=None):
    return {"audits": audits, "final_decision": decision or {"action": "accept"}}


def test_hands_gate_uses_production_criteria_not_just_errors(v2_spec):
    from utils.refine_quality import final_quality_decision, should_refine_quality
    clean = {"structural_issues": [], "character_limb_inventory": [],
             "needs_refine": False, "ownership_uncertain": [], "conclusion_valid": True,
             "conclusion_missing": False, "hands_clear": True}
    outcomes = ce._retention_gate_verdicts(
        _gates({"hands": clean}), v2_spec, should_refine_quality, final_quality_decision)
    assert outcomes["hands"] == "pass"

    defective = dict(clean, structural_issues=[{"region": "left hand", "observed": "六指"}],
                     needs_refine=True, hands_clear=False)
    outcomes = ce._retention_gate_verdicts(
        _gates({"hands": defective}), v2_spec, should_refine_quality, final_quality_decision)
    assert outcomes["hands"] == "fail", "明确结构缺陷必须判不通过"

    uncertain = dict(clean, ownership_uncertain=[{"region": "hands"}], hands_clear=False)
    outcomes = ce._retention_gate_verdicts(
        _gates({"hands": uncertain}), v2_spec, should_refine_quality, final_quality_decision)
    assert outcomes["hands"] == "fail", "归属不确定必须判不通过"

    missing = dict(clean, conclusion_valid=False, conclusion_missing=True, hands_clear=False)
    outcomes = ce._retention_gate_verdicts(
        _gates({"hands": missing}), v2_spec, should_refine_quality, final_quality_decision)
    assert outcomes["hands"] == "fail", "缺结论不能算通过"

    errored = {"audit_error": "boom", "candidate_sha256": "b" * 64}
    outcomes = ce._retention_gate_verdicts(
        _gates({"hands": errored}), v2_spec, should_refine_quality, final_quality_decision)
    assert outcomes["hands"] == "audit_error"


def test_legacy_hand_audit_without_hands_clear_is_recomputed_not_failed(v2_spec):
    """复用来的旧人体审计没有 `hands_clear` 字段：必须按同一判据重算，不能当成不通过。"""
    from utils.refine_quality import final_quality_decision, should_refine_quality
    legacy_hands = {"structural_issues": [], "character_limb_inventory": [],
                    "needs_refine": False, "ownership_uncertain": [],
                    "conclusion_valid": True, "conclusion_missing": False}
    assert "hands_clear" not in legacy_hands
    assert ce._retention_hands_clear(legacy_hands) is True
    gates = _gates({"hands": legacy_hands})
    outcomes = ce._retention_gate_verdicts(gates, v2_spec, should_refine_quality, final_quality_decision)
    assert outcomes["hands"] == "pass", "缺字段 ≠ 明确不通过"

    state = {"reuse": {("A", "s", "hands", "c" * 64): {"audit": legacy_hands, "source": "legacy",
                                                       "final_decision": None}},
             "reused": [], "fresh": [], "blocked": [], "errored": [],
             "mode": "audit_only", "used_text": 0, "text_limit": 16}
    audit, source = ce._retention_audit_call(state, {"id": "t"}, kind="hands", branch="A", sample_id="s",
                                             candidate_sha="c" * 64,
                                             call=lambda: pytest.fail("复用命中不该再调用"),
                                             log=lambda *a, **k: None)
    assert source == "reused" and audit["hands_clear"] is True


def test_gate_verdicts_keep_all_four_kinds_distinct(v2_spec):
    from utils.refine_quality import final_quality_decision, should_refine_quality
    audits = {
        "identity": {"mismatch": False, "severity": "none", "confidence": 0.9,
                     "differences": [], "gate_action": "accept"},
        "quality": {"structural_issues": [], "line_issues": [], "background_drift": [],
                    "style_gaps": [], "needs_refine": False, "severity": "none"},
        "hands": {"structural_issues": [], "needs_refine": False, "ownership_uncertain": [],
                  "conclusion_valid": True, "conclusion_missing": False, "hands_clear": True},
        "final_review": {"structural_issues": [], "line_issues": [], "background_drift": [],
                         "style_gaps": [], "needs_refine": False, "severity": "none"},
    }
    gates = _gates(audits, {"action": "accept_with_warning"})
    outcomes = ce._retention_gate_verdicts(gates, v2_spec, should_refine_quality, final_quality_decision)
    assert set(outcomes) == {"identity", "quality", "hands", "final_review"}
    assert all(value == "pass" for value in outcomes.values())


# --- ⑦ 逐张记录与配色评审校验 ---------------------------------------------------

def test_colour_review_validator_rejects_incomplete_answers():
    good = {"images": [{"id": "Q01", "main_colour_status": "conform", "accent_status": "conform",
                        "colour_status": "conform", "accent_waistband": "present",
                        "accent_left_shoe_bow": "present", "accent_right_shoe_bow": "present",
                        "protected_colours_ok": "true", "leak": {"status": "none"}}]}
    ce.validate_retention_colour_review(good, ["Q01"])
    with pytest.raises(ValueError):
        ce.validate_retention_colour_review({"images": []}, ["Q01"])
    bad = json.loads(json.dumps(good))
    bad["images"][0]["colour_status"] = "good"
    with pytest.raises(ValueError):
        ce.validate_retention_colour_review(bad, ["Q01"])
    no_place = json.loads(json.dumps(good))
    del no_place["images"][0]["accent_right_shoe_bow"]
    with pytest.raises(ValueError):
        ce.validate_retention_colour_review(no_place, ["Q01"])


def test_per_image_rows_carry_region_placement_and_aspect(v2_spec, tmp_path):
    out = _tmp_run(tmp_path)
    image = _write_candidate(out, "B", "gemini-flash-P1-knowledge-r1")
    review = out / "review"
    review.mkdir(parents=True, exist_ok=True)
    (review / "colour-B.json").write_text(json.dumps({
        "name": "colour-B", "mapping": {"Q01": "gemini-flash-P1-knowledge-r1"},
        "candidate_hashes": {"Q01": ce._sha256_file(image)},
        "response": {"images": [{"id": "Q01", "colour_status": "partial",
                                 "main_colour_status": "partial", "accent_status": "conform",
                                 "accent_waistband": "present", "accent_left_shoe_bow": "present",
                                 "accent_right_shoe_bow": "present", "protected_colours_ok": "true",
                                 "new_blocks": [{"colour": "warm green", "area": "bank"}],
                                 "content_changes": [{"aspect": "framing", "change": "landscape to portrait"}],
                                 "leak": {"status": "suspected"}}]}}, ensure_ascii=False), encoding="utf-8")
    rows = ce.retention_per_image(str(out), spec=v2_spec)
    row = rows[0]
    assert row["colour_review"]["accent_left_shoe_bow"] == "present"
    assert row["colour_review"]["new_blocks"] and row["colour_review"]["content_changes"]
    assert row["aspect"]["orientation"] in ("portrait", "landscape", "square")
    assert row["candidate_sha256"] == ce._sha256_file(image)


# --- v1 事故留档 ---------------------------------------------------------------

def test_v1_b_never_carried_the_contract_and_is_labelled_legacy():
    """v1 的 B 顶层写了合同、实际发送没带：必须留档且明确不可当 B 用。"""
    if not V1_RUN.is_dir():
        pytest.skip("v1 实验目录不在工作区")
    spec = ce.load_retention_spec(V1_SPEC)
    rows = ce.retention_legacy_b_candidates(V1_RUN, spec)
    assert rows, "应能列出 v1 的 B 留档"
    for row in rows:
        assert row["usable_as_b"] is False
        assert row["top_prompt_chars"] > row["sent_prompt_chars"]
        assert row["contract_in_sent_prompt"] is False
        assert row["top_sha256"] != row["sent_sha256"]


def test_v2_contract_source_and_runtime_text_agree():
    """合同来源文档与运行时正文必须指向同一套条款（数量与关键行都对得上）。"""
    source = Path("prompts/color-knowledge/repaint-colour-contract-v1-source.md").read_text(encoding="utf-8")
    text = ce.retention_contract_text()
    assert "SOURCE COLOUR CONTRACT" in text and "INHERITED FROM THE FIRST IMAGE" in text
    assert "STYLE MIGRATION PERMITTED" in text and "WHEN THE TWO DISAGREE" in text
    for key in ("cool blue", "BOTH shoes", "silver-grey hair", "second person",
                "line work, brushwork, edges"):
        assert key.split(",")[0] in text
    assert "不涉及颜色" in source and "不继承" in source, \
        "来源文档必须写明 M 的画法/光照条款不在继承范围内"