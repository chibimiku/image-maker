"""retention-v2 离线修正交付的回归用例（全部离线，mock，绝不写已交付实验目录）。

覆盖本轮要求的几件事：
① 缓存指纹规范：复用键包含候选/首图/参考图/期望规格/提示词/模型/参数/种类/协议版本；
   旧记录缺指纹只能记「复用资格未验证」，不补造指纹、不自动联网补审计；
② 配色评审缓存不能只看「结果文件在不在」，必须重算签名比对；
③ 离线修正：只读旧证据，不改旧文件，生成逐项门禁 / 逐张 / 汇总 / 页面口径一致的交付；
④ 事实分歧留档：原始模型判断与目视观察分开，不因目视观察放宽门禁；
⑤ 泄漏口径：没有实际附参考图时，参考颜色泄漏记不可验证；「未观察到偏色」不等于「证明没有泄漏」。
"""
import json
from pathlib import Path

import pytest

from utils import color_experiment as ce

V2_SPEC = "prompts/color-knowledge/fixed-palette-retention-v2.json"
V2_RUN = Path("data/test-result/20261004/color-palette-retention-v2")


@pytest.fixture(autouse=True)
def _test_output_root(tmp_path, monkeypatch):
    root = tmp_path / "test-result-root"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(ce.TEST_OUTPUT_ROOT_ENV, str(root))
    return root


@pytest.fixture()
def v2_spec():
    return ce.load_retention_spec(V2_SPEC)


@pytest.fixture()
def spec_fixture(tmp_path):
    """把源图与首图请求文本复制到临时目录，并给出指向副本的 spec。

    这样指纹用例完全在临时目录里跑，不读也不写交付目录。
    """
    import os
    import shutil
    root = Path(os.environ[ce.TEST_OUTPUT_ROOT_ENV])
    spec = ce.load_retention_spec(V2_SPEC)
    sources = []
    for item in spec["sources"]:
        source = Path(item["image"])
        if not source.is_file():
            pytest.skip("首图不在工作区")
        image_copy = root / "fixtures" / item["sample_id"] / source.name
        image_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, image_copy)
        prompt_source = Path(spec["source_run"]) / "samples" / "gemini-flash" / item["sample_id"] / "prompt.txt"
        prompt_copy = image_copy.parent / "prompt.txt"
        if prompt_source.is_file():
            shutil.copyfile(prompt_source, prompt_copy)
        sources.append({**item, "image": str(image_copy)})
    style_copy = root / "fixtures" / Path(spec["style"]["ref_image"]).name
    style_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(spec["style"]["ref_image"], style_copy)
    test_spec = {**spec, "sources": sources,
                 "source_run": str(root / "fixtures" / "source-run"),
                 "style": {**spec["style"], "ref_image": str(style_copy)}}
    return test_spec


# --- ① 指纹规范 --------------------------------------------------------------

def test_fingerprint_covers_every_reuse_relevant_input(spec_fixture, tmp_path):
    fingerprint = ce.retention_audit_fingerprint(tmp_path, spec_fixture, kind="identity", branch="B",
                                                 sample_id="gemini-flash-P1-knowledge-r1",
                                                 candidate_sha="a" * 64)
    for key in ("protocol_version", "audit_kind", "candidate_sha256", "source_image_sha256",
                "style_reference_sha256", "expected_spec_sha256", "prompt_sha256", "model",
                "params_sha256"):
        assert key in fingerprint, key
    assert fingerprint["candidate_sha256"] == "a" * 64
    assert fingerprint["source_image_sha256"], "首图 hash 必须进指纹"
    assert fingerprint["style_reference_sha256"], "画风参考图 hash 必须进指纹"
    assert fingerprint["expected_spec_sha256"], "期望规格 hash 必须进指纹"
    assert fingerprint["system_prompt_sha256"], "审计 system prompt hash 必须进指纹"
    assert fingerprint["params_sha256"] and fingerprint["protocol_version"]
    assert fingerprint["audit_kind"] == "identity"


def test_missing_fingerprint_is_unverified_not_fabricated(spec_fixture, tmp_path):
    current = ce.retention_audit_fingerprint(tmp_path, spec_fixture, kind="quality", branch="A",
                                             sample_id="gemini-flash-P1-knowledge-r1",
                                             candidate_sha="b" * 64)
    legacy = {"kind": "quality", "candidate_sha256": "b" * 64}   # 旧记录：只有候选 hash
    stored = ce.retention_record_fingerprint(legacy, spec_fixture)
    verdict = ce.retention_reuse_eligibility(current, stored)
    assert verdict["state"] == "unverified_fingerprint"
    assert verdict["eligible"] is False
    assert "prompt_sha256" in verdict["missing"] and "model" in verdict["missing"]
    assert stored.get("prompt_sha256") is None and stored.get("model") is None, "不得补造旧记录没有的指纹"


def test_complete_match_is_verified_and_mismatch_is_refused(spec_fixture):
    """指纹完整时完全匹配才算可复用；任一相关输入/协议变化都必须拒绝。

    `spec_fixture` 会在临时输出根下准备好源图与首图请求文本，所以这里不再请求 `tmp_path`
    （请求它会换掉同一个测试的输出根，导致指纹里的路径与文件不一致）。
    """
    import os
    root = Path(os.environ[ce.TEST_OUTPUT_ROOT_ENV])
    current = ce.retention_audit_fingerprint(root, spec_fixture, kind="hands", branch="A",
                                             sample_id="gemini-flash-P1-knowledge-r1",
                                             candidate_sha="c" * 64)
    # 模型是否可用取决于本机文本接口配置；这里只补一个明确的哨兵值，
    # 免得「离线跑测试时没有文本配置」把指纹完整性判成缺项。
    current["model"] = current.get("model") or "model-not-configured-in-tests"
    stored = {key: current.get(key) for key in ce.RETENTION_FINGERPRINT_FIELDS}
    assert ce.retention_fingerprint_completeness(stored) == [], "指纹必须完整"
    assert ce.retention_reuse_eligibility(current, stored)["state"] == "verified_match"
    for key in ("candidate_sha256", "source_image_sha256", "style_reference_sha256",
                "expected_spec_sha256", "prompt_sha256", "model", "params_sha256",
                "audit_kind", "protocol_version", "system_prompt_sha256"):
        changed = dict(stored)
        changed[key] = "x" * 8
        verdict = ce.retention_reuse_eligibility(current, changed)
        assert verdict["state"] == "mismatch" and verdict["eligible"] is False, key
        assert key in verdict["differences"]


def test_reuse_scan_marks_legacy_records_unverified(tmp_path, v2_spec):
    """旧实验的成功审计：候选 hash 对得上，但缺指纹 → 复用资格未验证。"""
    src = Path("data/test-result/20261004/color-palette-retention-v1")
    if not src.is_dir():
        pytest.skip("v1 证据不在工作区")
    scan = ce._retention_reuse_scan(src, v2_spec)
    assert scan, "应能扫描到 v1 的成功审计"
    states = {value["reuse_eligibility"]["state"] for value in scan.values()}
    assert states == {"unverified_fingerprint"}, states
    for key, value in scan.items():
        assert value["reuse_eligibility"]["eligible"] is False
        assert value["reuse_eligibility"]["missing"]


def test_reuse_call_records_eligibility_without_blocking(v2_spec):
    state = {"reuse": {("A", "s", "hands", "d" * 64): {
        "audit": {"structural_issues": [], "needs_refine": False, "ownership_uncertain": [],
                  "conclusion_valid": True, "conclusion_missing": False},
        "source": "legacy",
        "reuse_eligibility": {"state": "unverified_fingerprint", "eligible": False,
                              "missing": ["prompt_sha256"]}}},
        "reused": [], "fresh": [], "blocked": [], "errored": [],
        "mode": "audit_only", "used_text": 0, "text_limit": 16}
    audit, source = ce._retention_audit_call(state, {"id": "t"}, kind="hands", branch="A", sample_id="s",
                                             candidate_sha="d" * 64,
                                             call=lambda: pytest.fail("复用命中不该再调用"),
                                             log=lambda *a, **k: None)
    assert source == "reused" and audit["hands_clear"] is True
    assert state["reuse_states"][0]["state"] == "unverified_fingerprint"


# --- ② 配色评审缓存 ----------------------------------------------------------

def test_colour_review_cache_requires_a_matching_signature(v2_spec, tmp_path):
    rows = [{"sample_id": "gemini-flash-P1-knowledge-r1", "branch": "B"}]
    images = [{"path": "a.png", "role": "first_image", "sha256": "1" * 64},
              {"path": "b.png", "role": "candidate", "sha256": "2" * 64},
              {"path": "c.png", "role": "style_reference", "sha256": "3" * 64}]
    cfg = {"model": "text-model"}
    target = tmp_path / "colour-B.json"
    assert ce.retention_review_cache_state(target, v2_spec, system_text="S", cfg=cfg,
                                           expected_images=images, rows=rows)["state"] == "absent"
    target.write_text(json.dumps({"name": "colour-B"}), encoding="utf-8")
    state = ce.retention_review_cache_state(target, v2_spec, system_text="S", cfg=cfg,
                                            expected_images=images, rows=rows)
    assert state["state"] == "unverified_fingerprint" and state["eligible"] is False

    signature = ce.retention_colour_review_signature(v2_spec, rows, "S", cfg, images=images)["signature"]
    target.write_text(json.dumps({"name": "colour-B", "request_signature": signature}), encoding="utf-8")
    assert ce.retention_review_cache_state(target, v2_spec, system_text="S", cfg=cfg,
                                           expected_images=images, rows=rows)["state"] == "verified_match"
    # 任何一项变化都不能复用
    for mutate in ({"system_text": "S2"}, {"cfg": {"model": "other"}},
                   {"expected_images": images[:2]},
                   {"expected_images": [dict(images[0], sha256="9" * 64)] + images[1:]}):
        state = ce.retention_review_cache_state(target, v2_spec,
                                                system_text=mutate.get("system_text", "S"),
                                                cfg=mutate.get("cfg", cfg),
                                                expected_images=mutate.get("expected_images", images),
                                                rows=rows)
        assert state["state"] == "mismatch" and state["eligible"] is False, mutate


# --- ③④ 离线修正交付 ---------------------------------------------------------

def _copy_run(tmp_path):
    """把 v2 的门禁与评审记录复制一份到临时目录，绝不改交付目录。"""
    import os
    import shutil
    if not V2_RUN.is_dir():
        pytest.skip("v2 证据不在工作区")
    root = Path(os.environ[ce.TEST_OUTPUT_ROOT_ENV])
    dst = root / "v2-copy"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("retention-summary.json", "retention-ledger.json", "retention-frozen.json"):
        if (V2_RUN / name).is_file():
            shutil.copyfile(V2_RUN / name, dst / name)
    for pattern in ("stages/*-gates/gates.json", "stages/*-gates/hands.json",
                    "stages/*-gates/quality.json", "stages/*-gates/identity.json",
                    "stages/*-gates/final-review.json", "stages/*-initial-repaint/result.json",
                    "review/colour-*.json"):
        for path in V2_RUN.glob(pattern):
            target = dst / path.relative_to(V2_RUN)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    return dst


def test_corrections_unifies_gate_files_without_touching_evidence(tmp_path, v2_spec):
    src = _copy_run(tmp_path)
    before = {path: path.stat().st_mtime_ns for path in src.rglob("*.json")}
    out = tmp_path / "corrections"
    summary = ce.retention_corrections(str(out), spec_path=V2_SPEC, src_dir=str(src),
                                       evidence=False, log=lambda *a, **k: None)
    assert summary["no_calls"] is True and summary["no_images"] is True
    gates = json.loads((out / "corrected-gates.json").read_text(encoding="utf-8"))["rows"]
    per_image = json.loads((out / "corrected-per-image.json").read_text(encoding="utf-8"))["rows"]
    assert len(gates) == 4
    for row in gates:
        other = next(item for item in per_image
                     if item["branch"] == row["branch"] and item["sample_id"] == row["sample_id"])
        assert row["final_status"] == other["final_status"]
        for kind, value in row["verdicts"].items():
            assert value["original_model_verdict"], "原始模型判断必须留档"
            assert value["recomputed_offline"] in ce.RETENTION_AUDIT_VERDICT_STATES
            assert value["final_status"] in ce.RETENTION_AUDIT_VERDICT_STATES
            assert value["fingerprint_completeness"]["missing"], "旧记录确实缺指纹"
            assert "source" in other["verdicts"][kind]["source_audit_file"] or \
                   other["verdicts"][kind]["source_audit_file"], "必须引用旧证据路径"
    after = {path: path.stat().st_mtime_ns for path in src.rglob("*.json")}
    assert before == after, "离线修正不得改动旧证据文件"


def test_corrections_hands_pass_matches_recomputation(tmp_path):
    """两张 A 的 hands 在旧 gates.json 里是 fail（旧口径），重算后是 pass：交付里必须统一成重算值。"""
    src = _copy_run(tmp_path)
    out = tmp_path / "corrections"
    ce.retention_corrections(str(out), spec_path=V2_SPEC, src_dir=str(src), evidence=False,
                             log=lambda *a, **k: None)
    rows = json.loads((out / "corrected-per-image.json").read_text(encoding="utf-8"))["rows"]
    a_rows = [row for row in rows if row["branch"] == "A"]
    assert a_rows
    for row in a_rows:
        hands = row["verdicts"]["hands"]
        assert hands["recomputed_offline"] == "pass"
        assert hands["final_status"] == "pass"
        assert hands["audit_sources"] == "reused"
    # 旧的原始记录保持原样（不被改写）
    old = json.loads((src / "stages/A-gemini-flash-P1-knowledge-r1-gates/gates.json").read_text(encoding="utf-8"))
    assert old["outcomes"]["hands"] == "fail", "旧证据保留原判断，不改写"


def test_corrections_leak_is_unverifiable_and_original_claim_kept(tmp_path):
    src = _copy_run(tmp_path)
    out = tmp_path / "corrections"
    ce.retention_corrections(str(out), spec_path=V2_SPEC, src_dir=str(src), evidence=False,
                             log=lambda *a, **k: None)
    rows = json.loads((out / "corrected-per-image.json").read_text(encoding="utf-8"))["rows"]
    for row in rows:
        review = row.get("colour_review")
        if not review:
            continue
        assert review["model_saw_style_reference_image"] is False
        assert review["leak"] == "unverifiable"
        assert review["original_model_leak_claim"] in ("none", "suspected", "unverifiable")
        assert "unverifiable" in review["leak_scope"]


def test_verification_passes_on_generated_delivery(tmp_path):
    src = _copy_run(tmp_path)
    out = tmp_path / "corrections"
    ce.retention_corrections(str(out), spec_path=V2_SPEC, src_dir=str(src), evidence=False,
                             log=lambda *a, **k: None)
    ce.write_retention_corrections_page(str(out), spec_path=V2_SPEC, log=lambda *a, **k: None)
    result = ce.retention_verify_corrections(str(out), spec_path=V2_SPEC, log=lambda *a, **k: None)
    assert result["status"] == "passed", result["problems"]


def test_verification_catches_an_injected_inconsistency(tmp_path):
    src = _copy_run(tmp_path)
    out = tmp_path / "corrections"
    ce.retention_corrections(str(out), spec_path=V2_SPEC, src_dir=str(src), evidence=False,
                             log=lambda *a, **k: None)
    ce.write_retention_corrections_page(str(out), spec_path=V2_SPEC, log=lambda *a, **k: None)
    data = json.loads((out / "corrected-per-image.json").read_text(encoding="utf-8"))
    data["rows"][0]["final_status"] = "available"
    (out / "corrected-per-image.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    result = ce.retention_verify_corrections(str(out), spec_path=V2_SPEC, log=lambda *a, **k: None)
    assert result["status"] == "problems"
    assert any("不一致" in problem or "未通过项" in problem for problem in result["problems"])


def test_shoe_region_evidence_is_read_only_and_measured(tmp_path):
    candidate = None
    for path in V2_RUN.glob("stages/B-*-initial-repaint/*.jpg"):
        candidate = path
        break
    if candidate is None:
        pytest.skip("v2 的 B 候选不在工作区")
    before = candidate.stat().st_mtime_ns
    evidence = ce.retention_shoe_region_evidence(candidate, log=lambda *a, **k: None)
    assert evidence["found"] is True
    assert evidence["gap_to_bottom_px"] >= 0
    assert evidence["sha256"] and evidence["dark_region_pixels"] > 0
    assert candidate.stat().st_mtime_ns == before, "证据测量不得改动原图"


def test_recheck_preview_lists_roles_and_makes_no_call(tmp_path, v2_spec):
    preview = ce.retention_recheck_preview(str(tmp_path), spec_path=V2_SPEC,
                                           log=lambda *a, **k: None)
    assert preview["calls_made"] == 0
    roles = [row["role"] for row in preview["image_layout"]]
    assert roles[0] == "first_image" and "candidate" in roles and "style_reference" in roles
    assert any("参考图" in row for row in preview["requirements"])
    assert any("画幅" in row for row in preview["requirements"])


def test_targeted_recheck_preflight_contains_real_reference_and_full_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(ce, "_text_cfg", lambda: {"model": "offline-review", "base_url": "https://example.invalid/v1"})
    out = tmp_path / "targeted-recheck"
    plan = ce.retention_recheck_preflight(out, spec_path=V2_SPEC, log=lambda *args: None)
    assert plan["calls_made"] == 0 and plan["execution_authorized"] is False
    assert plan["budget"]["text_calls_max"] == 3 and plan["budget"]["image_calls"] == 0
    for row in plan["requests"][:2]:
        assert len(row["images"]) == 5
        assert row["images"][-1]["role"] == "style_reference_full"
        assert all(image["sha256"] and image["data_url_sha256"] for image in row["images"])
        assert not any("detail" in image["role"] for image in row["images"])
    assert len(plan["requests"][2]["images"]) == 2
    assert "not_production_gate" in plan["requests"][2]["purpose"]


def test_targeted_recheck_preflight_refuses_changed_model_without_overwrite(tmp_path, monkeypatch):
    cfg = {"model": "offline-review", "base_url": "https://example.invalid/v1"}
    monkeypatch.setattr(ce, "_text_cfg", lambda: cfg)
    out = tmp_path / "targeted-recheck"
    original = ce.retention_recheck_preflight(out, spec_path=V2_SPEC, log=lambda *args: None)
    target = out / "recheck-preflight.json"
    before = target.read_bytes()
    assert ce.retention_recheck_preflight(out, spec_path=V2_SPEC, log=lambda *args: None) == original
    cfg["model"] = "changed-review"
    with pytest.raises(ValueError, match="Frozen recheck plan changed"):
        ce.retention_recheck_preflight(out, spec_path=V2_SPEC, log=lambda *args: None)
    assert target.read_bytes() == before
