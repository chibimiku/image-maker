import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.image_analysis.history_pickup import (AnalysisRerunDialog,
    HistoryPickupDialog, available_prompt_types, branch_from_artifact,
    default_rerun_targets, discover_generation_checkpoints, list_process_artifacts)
from utils.analysis_gen import publish_final_output


@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_pickup_lists_artifacts_and_branches_from_failed_final_review(tmp_path):
    process = tmp_path / "data" / "20260928" / "analysis-gpt-image" / "run"
    process.mkdir(parents=True)
    first = process / "65ebea5b_first.png"
    candidate = process / "65ebea5b-final-rescue-1.jpg"
    first.write_bytes(b"first")
    candidate.write_bytes(b"candidate")
    audit = process / "final-quality-audit-1.json"
    audit.write_text("{}", encoding="utf-8")
    checkpoint = process / "generation-checkpoint.json"
    checkpoint.write_text(json.dumps({
        "snapshot": {"file_prefix": "65ebea5b"}, "status": "error",
        "current_stage": "final_review", "last_outputs": [str(candidate)],
        "stages": {"first": {"status": "success", "outputs": [str(first)]},
                   "hands": {"status": "success", "outputs": [str(first)]},
                   "final_review": {"status": "error"}},
        "operations": {"final_review-v2-repair-1": [str(candidate)]},
    }), encoding="utf-8")

    artifacts = list_process_artifacts(checkpoint)
    assert {Path(item["path"]).name for item in artifacts} >= {
        first.name, candidate.name, audit.name, checkpoint.name}
    selected = next(item for item in artifacts if item["path"] == str(candidate))
    assert selected["stage"] == "final_review"

    branch_path, next_stage = branch_from_artifact(checkpoint, candidate, selected["stage"])
    branch = json.loads(Path(branch_path).read_text(encoding="utf-8"))
    assert next_stage == "final_review"
    assert branch["stages"]["hands"]["outputs"] == [str(candidate)]
    assert "final_review" not in branch["stages"]
    assert json.loads(checkpoint.read_text(encoding="utf-8"))["status"] == "error"


def test_manual_publication_preserves_task_hash_and_date(tmp_path):
    process = tmp_path / "data" / "20260928" / "analysis-gpt-image" / "run"
    process.mkdir(parents=True)
    candidate = process / "65ebea5b-final-rescue.jpg"
    candidate.write_bytes(b"image")
    date_dir = process.parent.parent
    published = publish_final_output(str(candidate), process_dir=str(process),
        final_dir=str(date_dir), task_hash="65ebea5b", prefer_symlink=True)
    assert os.path.isfile(published)
    assert Path(published).parent == date_dir
    assert Path(published).name.startswith("65ebea5b_")
    assert Path(published).read_bytes() == b"image"


def test_discovers_saved_checkpoints_without_json_picker(tmp_path):
    first = tmp_path / "data" / "20260927" / "analysis-gpt-image" / "aaaa1111-old" / "generation-checkpoint.json"
    matched = tmp_path / "data" / "20260928" / "analysis-gpt-image" / "65ebea5b-renian" / "generation-checkpoint.json"
    for path in (first, matched):
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
    paths = discover_generation_checkpoints(tmp_path / "data", "65ebea5b")
    assert paths == [str(matched.resolve())]
    assert discover_generation_checkpoints(tmp_path / "data", "no-match") == []
    assert set(discover_generation_checkpoints(tmp_path / "data")) == {
        str(matched.resolve()), str(first.resolve())}


def test_available_prompt_types_reads_record_and_result_json():
    record = {"result_json": {"original_english_description": "a girl",
                              "english_description": "a girl, refined"}}
    assert available_prompt_types(record) == ["refined", "original"]
    # 记录上的提示词优先，结果 JSON 缺失也不影响
    assert available_prompt_types({"original_prompt": "a girl"}) == ["original"]
    assert available_prompt_types({"result_json": {"english_description": "  "}}) == []
    assert available_prompt_types(None) == []


def test_default_rerun_targets_reuses_recorded_prompt_types():
    available = ["refined", "original"]
    # 没记过 → 优化提示词
    assert default_rerun_targets({}, available) == ["refined"]
    # 记过 → 沿用上次实际跑的（两种都跑过就两种都重跑）
    assert default_rerun_targets(
        {"generation_params": {"prompt_types": ["original"]}}, available) == ["original"]
    assert default_rerun_targets(
        {"generation_params": {"prompt_types": ["original", "refined"]}}, available) == ["original", "refined"]
    # 记录的提示词这次没有文本 → 回退到可用的那种
    assert default_rerun_targets(
        {"generation_params": {"prompt_types": ["original"]}}, ["refined"]) == ["refined"]
    assert default_rerun_targets({}, []) == []


# --------------------------------------------------------------------------
# 「重新分析」出口：分析产物丢了也要给用户一条路
# --------------------------------------------------------------------------


def test_analysis_rerun_dialog_defaults_to_analyze_only_and_gates_generate(qapp):
    dialog = AnalysisRerunDialog(["来源图：a.png", "生图通道：Gemini"], "gemini")
    assert dialog.mode == "analyze"
    assert dialog.selected_targets() == ["refined"]     # 默认勾选优化提示词
    assert dialog.gen_targets == []

    dialog.analyze_and_generate.setChecked(True)
    assert dialog.mode == "generate"
    dialog.accept()
    assert dialog.gen_targets == ["refined"]

    # 要求生图却一种都没勾 → 不关闭（避免建出一条空任务）
    dialog2 = AnalysisRerunDialog([], "gemini")
    dialog2.analyze_and_generate.setChecked(True)
    for box in dialog2.prompt_boxes.values():
        box.setChecked(False)
    dialog2.accept()
    assert dialog2.result() == 0
    dialog.close()
    dialog2.close()


def test_analysis_rerun_dialog_remembers_gpt_channel(qapp):
    dialog = AnalysisRerunDialog([], "gpt-image")
    assert dialog.channel == "gpt-image"
    dialog.close()


def test_history_pickup_dialog_offers_rerun_analysis_without_a_selected_image(qapp, tmp_path):
    """断点里没有可用的图时，也要能点「重新分析」退出（不依赖选中的过程图）。"""
    process = tmp_path / "data" / "20260928" / "analysis-gpt-image" / "65ebea5b-run"
    process.mkdir(parents=True)
    audit = process / "final-quality-audit-1.json"
    audit.write_text("{}", encoding="utf-8")
    checkpoint = process / "generation-checkpoint.json"
    checkpoint.write_text(json.dumps({"snapshot": {}, "status": "error",
                                      "current_stage": "final_review", "last_outputs": [],
                                      "stages": {}, "operations": {}}), encoding="utf-8")

    dialog = HistoryPickupDialog([str(checkpoint)])
    # 断点里只有审计/文本文件 → 没有默认选中项，续跑与发布按钮都是灰的
    assert dialog.selected_artifact is None
    assert dialog.resume_button.isEnabled() is False
    assert dialog.publish_button.isEnabled() is False
    dialog.rerun_button.click()
    assert dialog.action == "rerun-analysis"
    assert dialog.result() == int(dialog.DialogCode.Accepted)
    dialog.close()
