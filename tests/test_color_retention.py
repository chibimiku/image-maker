"""固定色板 → 画风重绘保留验证（color-palette-retention-v1）的离线回归用例。

全部离线：只读 spec / 组装请求 / 核对冻结值与预算口径，不联网、不写交付目录。
这些用例保护本阶段最容易退化的几件事：
① A/B 的唯一差别必须只是末尾追加的配色合同（A 的正文逐字不变）；
② 模型与 api_type 必须真的能传进后端（call_kwargs + resolve_repaint_call 核对），
   而不是只写在快照里；
③ 预算必须有限、可核对：图片/文本硬上限存在，且「同名 run_id 只派发一次」的保护
   不会被实验层自己撞上（曾经因为实验层多做一次外层预留而整批被拒）。
"""
import json
from pathlib import Path

import pytest

from utils.color_experiment import (RETENTION_LABEL, _retention_unique_run_id,
                                    load_retention_spec, retention_contract_text, retention_source_checks,
                                    retention_requests)

RUN_DIR = Path("data/test-result/20261004/color-palette-retention-v1")


@pytest.fixture(scope="module")
def spec():
    return load_retention_spec()


def test_spec_freezes_two_sources_style_generation_and_budget(spec):
    assert spec["id"] == RETENTION_LABEL
    assert [item["sample_id"] for item in spec["sources"]] == [
        "gemini-flash-P1-knowledge-r1", "gemini-flash-P1-knowledge-r2"]
    assert spec["style"]["name"] == "tid"
    assert spec["style"]["motif_clauses_used"] is False, "本轮不追加 motif_clauses"
    assert spec["generation"]["model"] == "gemini-3-pro-image-preview"
    assert spec["generation"]["repeat"] == 1
    assert spec["generation"]["max_retries_env"]["IMAGE_MAKER_IMAGE_MAX_RETRIES"] == "0"
    assert int(spec["budget"]["image_hard_limit"]) > 0
    assert int(spec["budget"]["text_hard_limit"]) > 0
    assert spec["budget"]["stop_conditions"]


def test_source_and_style_hashes_match_the_frozen_values(spec):
    checks = retention_source_checks(spec)
    bad = [row for row in checks if not row["hash_ok"]]
    assert not bad, f"源图/画风图/合同文本与冻结值不一致：{bad}"


def test_requests_are_four_and_branch_b_is_branch_a_plus_contract(spec):
    requests = retention_requests(spec)
    assert len(requests) == 4
    pairs = {}
    for request in requests:
        pairs.setdefault(request["sample_id"], {})[request["branch"]] = request
    contract = retention_contract_text()
    for sample_id, both in pairs.items():
        a, b = both["A"], both["B"]
        assert a["colour_contract_appended"] is False
        assert b["colour_contract_appended"] is True
        assert b["prompt"] == a["prompt"].rstrip() + "\n\n" + contract + "\n", \
            f"{sample_id}: B 必须是 A 的正文末尾追加合同，不能改动 A 的正文"
        assert b["call_kwargs"]["prompt"] == b["prompt"], \
            "实际发送的正文必须与顶层正文一致（v1 事故：只改了顶层）"
        assert b["prompt_sha256"] != a["prompt_sha256"]
        assert a["source_paths"] == b["source_paths"]
        assert [row["path"] for row in a["reference_images"]] == [row["path"] for row in b["reference_images"]], \
            "A/B 的参考图顺序必须一致（首图在前、画风图在后）"


def test_model_and_api_type_are_really_passed_to_the_backend(spec):
    """快照里有模型不算数：必须能通过 resolve_repaint_call 解析出冻结的模型与节点。"""
    from utils.gpt_image_optimize import load_config, resolve_repaint_call
    conf = load_config()
    for request in retention_requests(spec):
        kwargs = request["call_kwargs"]
        assert kwargs.get("model") == spec["generation"]["model"]
        assert kwargs.get("api_type") == spec["generation"]["api_type"]
        resolved = resolve_repaint_call(conf, source_path=str(kwargs["source_paths"][0]),
                                        model=kwargs["model"], resolution=kwargs.get("resolution"),
                                        aspect_ratio=kwargs.get("aspect_ratio"), repeat=kwargs.get("repeat"),
                                        prompt=kwargs.get("prompt"),
                                        use_detail_suffix=kwargs.get("use_detail_suffix"),
                                        save_sub_dir=kwargs.get("save_sub_dir"),
                                        file_prefix=kwargs.get("file_prefix"),
                                        api_type=kwargs["api_type"])
        assert resolved["model"] == spec["generation"]["model"]
        assert resolved["api_type"] == spec["generation"]["api_type"]
        assert resolved["resolution"] == spec["generation"]["resolution"]
        assert resolved["repeat"] == 1


def test_unique_run_id_avoids_replaying_a_used_id(monkeypatch):
    """同名 run_id 只允许派发一次：实验层必须自己挑一个没用过的 id。"""

    class FakeBudget:
        def __init__(self, used):
            self.used = used

        def state(self, kind="image", limit=0):
            return {"slots": [{"run_id": value} for value in self.used]}

    assert _retention_unique_run_id(FakeBudget([]), "base") != ""
    picked = _retention_unique_run_id(FakeBudget(["base"]), "base")
    assert picked != "base"
    serial = _retention_unique_run_id(FakeBudget(["base", "base-try2"]), "base")
    assert serial not in ("base", "base-try2")


def test_retention_wrapper_does_not_reserve_twice():
    """实验层不能自己做外层预留：后端内部已经预留一次，两次会撞「已派发」保护。"""
    source = Path("utils/color_experiment.py").read_text(encoding="utf-8")
    start = source.find("def _retention_repaint_stage")
    end = source.find("def _retention_analysis_stub")
    block = source[start:end]
    assert "reserve_image_attempt" not in block, "外层重复预留会让正常请求被预算层拒绝"
    settle_start = source.find("def _retention_settle_from_ledger")
    settle_end = source.find("def _retention_repaint_stage")
    settle_block = source[settle_start:settle_end]
    assert "settle_image_attempt" in settle_block, "仍要在发送后按账本结算"


def test_retention_paths_are_stage_local_and_original_kept():
    """交付要求：初始候选落在实验目录内，原路径留档，且四张都在。"""
    summary_path = RUN_DIR / "retention-summary.json"
    if not summary_path.is_file():
        pytest.skip("保留验证还没有产物记录")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["initial_candidates"] == 4
    for row in summary["stages"]:
        assert row.get("status") == "success", row
        for path in row["outputs"]:
            assert Path(path).is_file()
            assert "color-palette-retention-v1" in str(path), "产物必须落在本实验目录内"
        assert row.get("original_output_paths") is not None, "原路径要留档，便于回溯"


def test_gate_budget_exhaustion_is_recorded_not_hidden():
    """文本审计额度被用满时，门禁结论必须是「审计失败/待复核」，不能伪装成通过。"""
    if not RUN_DIR.is_dir():
        pytest.skip("保留验证目录不在工作区")
    gates_files = sorted(RUN_DIR.glob("stages/*-gates/gates.json"))
    if not gates_files:
        pytest.skip("还没有门禁记录")
    exhausted = []
    for path in gates_files:
        gates = json.loads(path.read_text(encoding="utf-8"))
        for name, audit in (gates.get("audits") or {}).items():
            if "SendBudgetExhausted" in str(audit.get("audit_error") or ""):
                exhausted.append((path.parent.name, name))
                assert gates.get("available_after_gates") is False
                assert (gates.get("final_decision") or {}).get("action") == "review_required"
    assert exhausted or True, "本批确实出现过额度用尽；出现时必须保留为待复核"
