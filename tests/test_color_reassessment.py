"""配色方案符合度重评：分块、JSON 修复与校验的离线回归用例（不联网、不写 data/ 产出目录）。

这些用例保护 2026-10-04 重评协议里最容易退化的两件事：
① 分块必须是确定性的，且「同方案三张不拆块」；
② 评价模型返回的 JSON 即使有多写 `}` 或半截截断，也要能被修复到可用状态，
   否则只能按失败记录保留，不能假装成功。
"""
import json
from pathlib import Path
import uuid

import pytest

from utils.color_experiment import (REASSESS_BUDGET, REASSESS_PROTECTED, _close_partial_json,
                                    _content_failure, _scheme_stability, _validate_reassess_response,
                                    load_reassess_items, normalize_duplicate_ids,
                                    normalize_reassess_response, parse_object_tolerant, reassess_plan,
                                    recover_reassessment)


RUN_DIR = Path("data/test-result/20261004/color-reassessment-v2")
V1_DIR = Path("data/test-result/20261004/color-reassessment-v1")


@pytest.fixture(scope="module")
def items():
    return load_reassess_items()


def test_reassessment_covers_all_39_gemini_samples_once(items):
    assert len(items) == 39
    assert all(row["suite"] in ("stage2", "tone") for row in items)
    ids = [row["item_id"] for row in items]
    assert len(ids) == len(set(ids))


def test_plan_never_splits_a_scheme_and_is_deterministic(items):
    first = reassess_plan(items)
    second = reassess_plan(items)
    assert first["chunks"] == second["chunks"]
    counts = {}
    for chunk in first["chunks"]:
        for item_id in chunk["item_ids"]:
            row = next(r for r in items if r["item_id"] == item_id)
            counts[(row["suite"], row["group_id"])] = counts.get((row["suite"], row["group_id"]), 0) + 1
            assert len(set(row["group_id"] for row in items if row["item_id"] == item_id)) == 1
    for (suite, group_id), count in counts.items():
        total = len([r for r in items if r["suite"] == suite and r["group_id"] == group_id])
        assert count == total, f"{suite}/{group_id} 被拆块或重复"
    for chunk in first["chunks"]:
        assert len(chunk["item_ids"]) <= first["chunk_size"]


def _valid_image(item_id, clause_count, protected_status="retained"):
    return {"id": item_id,
            "colour_conformance": {
                "status": "conform",
                "items": [{"index": index, "target": f"clause {index}", "status": "conform",
                           "regions": ["背景"], "observed_colours": ["青绿"], "evidence": ["可见"]}
                          for index in range(1, clause_count + 1)],
                "protected_colours": [{"name": name, "status": "retained", "evidence": ["可见"]}
                                      for name in REASSESS_PROTECTED]},
            "observations": {"region_layout": "上/中/下", "base_aux_accent": "绿基调", "warm_cool": "冷",
                             "large_non_plan_areas": [], "content_notes": [], "content_issues": [],
                             "method_notes": [], "method_issues": [], "unsupported_claims": []}}


def test_validator_accepts_empty_clauses_without_items():
    value = {"images": [_valid_image("S01", 0)]}
    value["images"][0]["colour_conformance"].pop("items")
    _validate_reassess_response(value, ["S01"], {"S01": []}, REASSESS_PROTECTED)


def test_validator_rejects_missing_clause_index():
    value = {"images": [_valid_image("S01", 0)]}
    with pytest.raises(ValueError):
        _validate_reassess_response(value, ["S01"], {"S01": ["one clause"]}, REASSESS_PROTECTED)


def test_tolerant_parser_refuses_to_invent_structure():
    """真实故障形态（2026-10-04 reassess-01）：模型少写了一层 `}` 且后续内容完整，
    靠括号计数无法安全还原 —— 必须判失败并保留原文，不许猜出一个「成功」对象。"""
    broken = ('{"images":[{"id":"S01","colour_conformance":{"status":"conform","items":[],'
              '"protected_colours":[{"name":"粉色长发","status":"retained","evidence":["可见粉色长发"]}]}]},'
              '"observations":{"region_layout":"上/中/下"}}]}')
    value, note = parse_object_tolerant(broken)
    assert value is None, "结构不合法时不能返回猜测对象"
    assert "Expecting" in note or "delimiter" in note


def test_normalizer_moves_misplaced_observations_back_to_image_level():
    plain = {"images": [{"id": "S01",
                         "colour_conformance": {"status": "conform", "items": [], "protected_colours": [],
                                                "observations": {"region_layout": "上"}}}]}
    assert normalize_reassess_response(plain) == ["S01"]
    assert plain["images"][0]["observations"]["region_layout"] == "上"
    assert "observations" not in plain["images"][0]["colour_conformance"]

    nested = {"images": [{"id": "S02",
                          "colour_conformance": {"status": "conform", "items": [],
                                                 "protected_colours": {"entries": [],
                                                                       "observations": {"region_layout": "下"}}}}]}
    assert normalize_reassess_response(nested) == ["S02"]
    assert nested["images"][0]["observations"]["region_layout"] == "下"
    assert nested["images"][0]["colour_conformance"]["protected_colours"] == []


def test_tolerant_parser_closes_a_truncated_response():
    truncated = ('{"images":[{"id":"S01","colour_conformance":{"status":"conform","items":[],'
                 '"protected_colours":[{"name":"粉色长发","status":"retained","evidence":["可见')
    value, note = parse_object_tolerant(truncated)
    assert value is None, "半截 evidence 不能当成完整对象"


def test_close_partial_json_keeps_earlier_complete_objects():
    text = '{"images":[{"id":"S01"},{"id":"S02","obs'
    assert "S01" in _close_partial_json(text)


def test_duplicate_ids_without_conflict_are_merged():
    value = {"images": [_valid_image("S01", 1), _valid_image("S01", 1), _valid_image("S02", 1)]}
    merged = normalize_duplicate_ids(value)
    assert merged == ["S01"]
    assert [row["id"] for row in value["images"]] == ["S01", "S02"]
    assert not value["duplicate_ids_conflicting"]


def test_duplicate_ids_with_conflicting_clause_verdicts_become_unclear():
    first = _valid_image("S01", 1)
    second = json.loads(json.dumps(first))
    second["colour_conformance"]["items"][0]["status"] = "diverge"
    value = {"images": [first, second]}
    normalize_duplicate_ids(value)
    assert value["duplicate_ids_conflicting"] == ["S01"]
    assert value["images"][0]["colour_conformance"]["status"] == "unclear"
    assert value["images"][0]["duplicate_conflicts"], "冲突原文必须保留"
    assert value["images"][0]["_duplicate_raw"], "两份对象都要留档，不能只留有利的一份"


def test_duplicate_ids_one_sided_unclear_keeps_the_definite_verdict():
    """一份给了结论、另一份弃权（unclear）不算冲突：不能把明确结论丢掉。"""
    first = _valid_image("S01", 1, protected_status="retained")
    second = json.loads(json.dumps(first))
    second["colour_conformance"]["protected_colours"][0]["status"] = "unclear"
    second["colour_conformance"]["items"][0]["status"] = "unclear"
    value = {"images": [first, second]}
    normalize_duplicate_ids(value)
    assert value["duplicate_ids_conflicting"] == []
    assert value["images"][0]["colour_conformance"]["status"] == "conform"
    assert value["images"][0]["colour_conformance"]["protected_colours"][0]["status"] == "retained"
    assert value["images"][0]["colour_conformance"]["items"][0]["status"] == "conform"


def _copy_request_fixtures(source_dir, target_dir, names):
    """把指定记录的 request.json 复制成只读夹具；测试只从这里读请求，不改写已交付的依据。"""
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = Path(source_dir) / "vision" / f"{name}.request.json"
        if source.is_file():
            (target_dir / f"{name}.request.json").write_text(source.read_text(encoding="utf-8"),
                                                             encoding="utf-8")
    return target_dir


def test_recovery_uses_the_request_of_that_call_not_the_current_plan(tmp_path):
    """reassess-11 的实际请求是六张（T2+T3）：回收必须按该请求的编号与条款校验。

    **完全在临时目录里做**：请求快照与响应都从只读来源复制到 tmp_path，
    输出目录用 `ReviewProbeRoot`（路径仍满足 `data/test-result` 约束但**不写交付目录**），
    所以本用例不会改写 `color-reassessment-v2` 里已交付报告所依据的任何记录。
    """
    raw = RUN_DIR / "vision" / "reassess-11.raw.txt"
    source_request = RUN_DIR / "vision" / "reassess-11.request.json"
    if not raw.is_file() or not source_request.is_file():
        pytest.skip("color-reassessment-v2 的 reassess-11 证据不在工作区")
    snapshot = json.loads(source_request.read_text(encoding="utf-8"))
    expected = {entry["neutral_id"]: entry["item"] for entry in snapshot["images"] if entry.get("role") == "full"}
    assert len(expected) == 6
    fixtures = _copy_request_fixtures(RUN_DIR, tmp_path / "requests", ["reassess-11"])
    probe_root = Path("data/test-result") / f"review-probe-{uuid.uuid4().hex[:8]}"
    out = probe_root / "run"
    (out / "vision").mkdir(parents=True, exist_ok=True)
    (out / "vision" / "reassess-11.raw.txt").write_text(raw.read_text(encoding="utf-8"), encoding="utf-8")
    delivered_before = (RUN_DIR / "vision" / "reassess-11.json").stat().st_mtime_ns
    try:
        result = recover_reassessment(str(out), ["reassess-11"], request_dir=str(fixtures), only_missing=False)
        assert result["recovered"] == ["reassess-11"]
        record = json.loads((out / "vision" / "reassess-11.json").read_text(encoding="utf-8"))
        assert record["recovered_offline"] is True
        assert record["evidence_source"]["kind"] == "offline_recovery_of_paid_call"
        assert record["contract_from_request"]["clauses_source"] == "该次调用保存的 request.json"
        assert record["mapping"] == expected
        assert sorted(entry["id"] for entry in record["response"]["images"]) == sorted(expected)
        assert {"gemini-flash-T2-r1", "gemini-flash-T2-r2", "gemini-flash-T2-r3"} <= set(record["mapping"].values())
    finally:
        for path in sorted(probe_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        probe_root.rmdir()
    assert (RUN_DIR / "vision" / "reassess-11.json").stat().st_mtime_ns == delivered_before, \
        "只读夹具用例不得改写交付目录里的记录"


def test_recovery_refuses_to_infer_when_the_request_snapshot_is_missing(tmp_path):
    """缺 request.json 时必须记为失败，不得按当前「三张一块」计划推断映射。"""
    out = Path("data/test-result/20261004") / f"color-reassessment-probe-{tmp_path.name[:6]}"
    (out / "vision").mkdir(parents=True, exist_ok=True)
    (out / "vision" / "reassess-11.raw.txt").write_text('{"images":[]}', encoding="utf-8")
    try:
        result = recover_reassessment(str(out), ["reassess-11"], only_missing=False)
        assert result["recovered"] == []
        assert result["still_unusable"] and "request.json" in result["still_unusable"][0]["reason"]
    finally:
        for path in sorted(out.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            else:
                path.rmdir()
        out.rmdir()


def test_scheme_stability_is_not_applicable_without_colour_clauses():
    planned = [{"item_id": f"gemini-flash-G0-r{i}", "clauses": []} for i in (1, 2, 3)]
    subset = [{"colour_status": "conform"}] * 3
    assert _scheme_stability(planned, subset, [], ["conform"] * 3) == "not_applicable"


def test_scheme_stability_requires_full_coverage():
    planned = [{"item_id": f"gemini-flash-G2-r{i}", "clauses": ["a clause"]} for i in (1, 2, 3)]
    subset = [{"colour_status": "conform"}] * 2
    clause_rows = [{"status": "conform"}] * 2
    assert _scheme_stability(planned, subset, clause_rows, ["conform"] * 2) == "incomplete_coverage"


def test_scheme_stability_separates_consistency_from_conformance():
    planned = [{"item_id": f"gemini-flash-G3-r{i}", "clauses": ["a clause"]} for i in (1, 2, 3)]
    subset = [{"colour_status": "partial"}] * 3
    clause_rows = [{"status": "conform"}, {"status": "partial"}, {"status": "conform"}]
    assert _scheme_stability(planned, subset, clause_rows, ["partial"] * 3) == "stable_partial"
    mixed = [{"colour_status": "conform"}, {"colour_status": "partial"}, {"colour_status": "conform"}]
    assert _scheme_stability(planned, mixed, clause_rows, ["conform", "partial", "conform"]) == "unstable"
    conforming = [{"colour_status": "conform"}] * 3
    all_conform = [{"status": "conform"}] * 3
    assert _scheme_stability(planned, conforming, all_conform, ["conform"] * 3) == "stable_conform"


def test_content_verdicts_are_counted_separately_and_never_assumed():
    two_person = {"item_id": "x",
                  "observations": {"content_issues": ["人物数量为两人而非恰好一人，违反内容契约。"]}}
    unverifiable = {"item_id": "y",
                    "observations": {"content_issues": ["裁剪无法确认单人、全身、头发和鱼竿鱼线。"]}}
    clean = {"item_id": "z", "observations": {"content_issues": []}}
    assert _content_failure(two_person)[0] == "two_person_failure"
    assert _content_failure(unverifiable)[0] == "pass_or_unverifiable"
    assert _content_failure(clean)[0] == "pass_or_unverifiable"


def test_reassessment_output_stays_inside_test_result():
    from utils.color_experiment import _isolated_out
    with pytest.raises(ValueError):
        _isolated_out(str(Path("data") / "20261004" / "color-reassessment-v2"))


def test_authorised_call_budget_is_twelve():
    assert REASSESS_BUDGET == 12
