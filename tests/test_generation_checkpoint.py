"""Recovery must preserve successful paid calls and report failed stages."""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PyQt6.QtWidgets import QApplication

from utils.generation_checkpoint import GenerationCheckpoint
from modules.image_analysis.single_analyzer import GptImageGenWorkerThread


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def image(tmp_path, name):
    target = str(tmp_path / name)
    Image.new("RGB", (90, 60), "white").save(target)
    return target


def test_worker_resumes_repaint_without_second_gpt(app, monkeypatch, tmp_path):
    import modules.others.api_backend as backend
    import utils.analysis_gen as pipeline
    first = image(tmp_path, "first.png")
    final = image(tmp_path, "final.png")
    calls = []
    monkeypatch.setattr(backend, "generate_image_aigc2d_gpt", lambda **kw: calls.append("gpt") or [first])
    def repaint(*args, **kwargs):
        calls.append("repaint")
        if calls.count("repaint") == 1:
            raise RuntimeError("network failure")
        return [final]
    monkeypatch.setattr(pipeline, "run_gpt_image_pipeline", repaint)
    monkeypatch.setattr(pipeline, "publish_final_output", lambda path, **kw: path)
    cp = str(tmp_path / "checkpoint.json")
    settings = dict(request_payload={"prompt": "test", "skip_quality_refine": True},
                    steps={"repaint": {"enabled": True}}, checkpoint_path=cp)
    worker = GptImageGenWorkerThread(**settings)
    outputs = []
    worker.finish_signal.connect(outputs.append)
    worker.run()
    assert worker.last_status == "error"
    assert worker.failed_stage == "pipeline"
    assert outputs == [[]]
    data = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert data["stages"]["first"]["outputs"] == [first]
    worker = GptImageGenWorkerThread(**data["snapshot"], checkpoint_path=cp)
    worker.run()
    assert worker.last_status == "success"
    assert calls == ["gpt", "repaint", "repaint"]


def test_successful_correction_reused_after_followup_audit_failure(app, monkeypatch, tmp_path):
    import modules.others.api_backend as backend
    import utils.analysis_gen as pipeline
    import utils.refine_quality as quality
    first = image(tmp_path, "first.png")
    repainted = image(tmp_path, "repaint.png")
    corrected = image(tmp_path, "corrected.png")
    style = image(tmp_path, "style.png")
    calls = []
    monkeypatch.setattr(backend, "generate_image_aigc2d_gpt", lambda **kw: calls.append("gpt") or [first])
    monkeypatch.setattr(pipeline, "run_gpt_image_pipeline", lambda *a, **kw: calls.append("pipeline") or [repainted])
    monkeypatch.setattr(pipeline, "publish_final_output", lambda path, **kw: path)
    monkeypatch.setattr(backend, "generate_image_repaint", lambda *a, **kw: calls.append("correction") or [corrected])
    attempts = []
    def audit(original, candidate, *a, **kw):
        attempts.append(candidate)
        if candidate == corrected and attempts.count(corrected) == 1:
            raise RuntimeError("audit timeout")
        return {"needs_refine": candidate != corrected,
                "structural_issues": [{"confidence": 1, "region": "legs", "repair": "shorten"}]}
    monkeypatch.setattr(quality, "audit_refine_quality", audit)
    monkeypatch.setattr(quality, "build_quality_correction_prompt", lambda a: "fix")
    cp = str(tmp_path / "checkpoint.json")
    worker = GptImageGenWorkerThread(request_payload={"prompt": "test"},
        steps={"repaint": {"enabled": True}}, style_ref_path=style, checkpoint_path=cp)
    worker.run()
    assert worker.last_status == "error"
    assert worker.failed_stage == "quality"
    data = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    worker = GptImageGenWorkerThread(**data["snapshot"], checkpoint_path=cp)
    worker.run()
    assert worker.last_status == "success"
    assert calls == ["gpt", "pipeline", "correction"]
    assert attempts == [repainted, corrected, corrected]


def test_checkpoint_invalid_artifact_invalidates_downstream(tmp_path):
    cp = GenerationCheckpoint(str(tmp_path / "checkpoint.json"))
    first = image(tmp_path, "first.png")
    final = image(tmp_path, "final.png")
    cp.begin("first", [])
    cp.complete("first", [first])
    cp.begin("pipeline", [first])
    cp.complete("pipeline", [final])
    os.remove(first)
    needed, _ = cp.begin("first", [])
    assert needed
    assert "pipeline" not in cp.data["stages"]


def test_hand_gate_after_identity_despite_skip_quality_and_reuses_paid_repair(app, monkeypatch, tmp_path):
    import modules.others.api_backend as backend
    import utils.analysis_gen as pipeline
    import utils.refine_quality as quality
    import utils.identity_audit as identity
    first = image(tmp_path, "first.png")
    repainted = image(tmp_path, "repaint.png")
    corrected = image(tmp_path, "hands.png")
    events = []
    monkeypatch.setattr(backend, "generate_image_aigc2d_gpt", lambda **kw: [first])
    monkeypatch.setattr(pipeline, "run_gpt_image_pipeline", lambda *a, **kw: [repainted])
    monkeypatch.setattr(identity, "audit_image_identity", lambda *a, **kw: events.append("identity") or {"mismatch": False})
    monkeypatch.setattr(identity, "identity_gate_action", lambda a: "accept")
    monkeypatch.setattr(pipeline, "publish_final_output", lambda path, **kw: events.append("publish") or path)
    monkeypatch.setattr(quality, "audit_refine_quality", lambda *a, **kw: pytest.fail("style audit must stay skipped"))
    attempts = []
    def audit(path):
        events.append("hands")
        attempts.append(path)
        if path == corrected and attempts.count(corrected) == 1:
            raise RuntimeError("followup audit timeout")
        return {"needs_refine": path != corrected, "structural_issues":
                [{"region": "left hand", "repair": "separate fused digits", "confidence": 0.99}] if path != corrected else []}
    monkeypatch.setattr(quality, "audit_hand_quality", audit)
    paid = []
    def repair(paths, **kw):
        paid.append(paths)
        assert "extra_reference_paths" not in kw
        assert "five anatomical digits" in kw["prompt"]
        assert not kw["use_detail_suffix"]
        return [corrected]
    monkeypatch.setattr(backend, "generate_image_repaint", repair)
    cp = str(tmp_path / "checkpoint.json")
    settings = dict(request_payload={"prompt": "test", "skip_quality_refine": True},
                    steps={"repaint": {"enabled": True}}, analysis_result={"description": "girl"},
                    checkpoint_path=cp)
    worker = GptImageGenWorkerThread(**settings)
    worker.run()
    assert worker.last_status == "error" and worker.failed_stage == "hands"
    assert events[:2] == ["identity", "hands"]
    assert "publish" not in events
    worker = GptImageGenWorkerThread(**settings)
    worker.run()
    assert worker.last_status == "success"
    assert paid == [[repainted]]
    # 第二次运行时「修复前的复审」已由断点缓存（不再调用接口），实际发生的审计是：
    # ① 修手后复审计一次（首次调用故意超时，不写缓存，所以重跑要重来）；
    # ② 人体门禁结论通过；
    # ③ 最终复核再审一次实际要发布的像素（final_review 独立成 stage 之后新增的那道）。
    assert attempts == [repainted, corrected, corrected, corrected]
    assert events[-1] == "publish"


def test_hand_gate_blocks_persistent_defects(app, monkeypatch, tmp_path):
    import utils.refine_quality as quality
    import modules.others.api_backend as backend
    source = image(tmp_path, "source.png")
    cp = GenerationCheckpoint(str(tmp_path / "checkpoint.json"))
    for stage in ("first", "pipeline", "face", "quality", "identity"):
        cp.begin(stage, [source])
        cp.complete(stage, [source])
    monkeypatch.setattr(quality, "audit_hand_quality", lambda p: {
        "needs_refine": True, "structural_issues": [{"region": "hand", "repair": "fix thumb", "confidence": 1}]})
    calls = []
    monkeypatch.setattr(backend, "generate_image_repaint", lambda *a, **kw: calls.append(kw) or [source])
    worker = GptImageGenWorkerThread(request_payload={"prompt": "test", "skip_quality_refine": True},
        steps={"repaint": {"enabled": True}}, analysis_result={"description": "girl"}, checkpoint_path=cp.path)
    worker.run()
    assert worker.last_status == "error" and worker.failed_stage == "hands"
    assert len(calls) == 2
    assert "publish" not in worker.checkpoint.data["stages"]


def test_final_review_invalidates_old_publish_and_blocks_style_drift(app, monkeypatch, tmp_path):
    import utils.refine_quality as quality
    import utils.analysis_gen as pipeline
    import utils.gpt_image_optimize as optimize
    monkeypatch.setattr(optimize, "load_config", lambda: {"final_candidate_repair": {"max_repairs": 0}})
    source = image(tmp_path, "source.png")
    style = image(tmp_path, "style.png")
    old_final = image(tmp_path, "published.png")
    cp = GenerationCheckpoint(str(tmp_path / "checkpoint.json"))
    for stage in ("first", "pipeline", "face", "quality", "identity", "hands", "adjust", "publish"):
        cp.begin(stage, [source])
        cp.complete(stage, [old_final] if stage == "publish" else [source])
    from utils.generation_checkpoint import ANATOMY_GATE_VERSION
    cp.data["anatomy_gate_version"] = ANATOMY_GATE_VERSION
    cp.save()
    def audit(*args, **kwargs):
        assert args[1] == source
        assert kwargs["final_review"]
        return {"needs_refine": True, "severity": "major",
                "style_gaps": [{"confidence": .99}]}
    monkeypatch.setattr(quality, "audit_refine_quality", audit)
    monkeypatch.setattr(pipeline, "publish_final_output", lambda *a, **kw: pytest.fail("must not publish"))
    worker = GptImageGenWorkerThread(request_payload={"prompt": "test", "skip_quality_refine": True},
        steps={"repaint": {"enabled": True}}, analysis_result={"description": "girl"},
        style_ref_path=style, checkpoint_path=cp.path)
    worker.run()
    assert worker.last_status == "error" and worker.failed_stage == "final_review"
    assert "publish" not in worker.checkpoint.data["stages"]
    assert os.path.isfile(old_final)  # Keep the original attempt.


def test_major_quality_drift_stops_before_identity(app, monkeypatch, tmp_path):
    import modules.others.api_backend as backend
    import utils.analysis_gen as pipeline
    import utils.refine_quality as quality
    import utils.identity_audit as identity
    first = image(tmp_path, "first.png")
    repaint = image(tmp_path, "repaint.png")
    bad = image(tmp_path, "standing-maid.png")
    style = image(tmp_path, "style.png")
    monkeypatch.setattr(backend, "generate_image_aigc2d_gpt", lambda **kw: [first])
    monkeypatch.setattr(pipeline, "run_gpt_image_pipeline", lambda *a, **kw: [repaint])
    monkeypatch.setattr(backend, "generate_image_repaint", lambda *a, **kw: [bad])
    monkeypatch.setattr(quality, "audit_refine_quality", lambda *a, **kw: {
        "needs_refine": True, "severity": "major" if a[1] == bad else "minor",
        "structural_issues": [{"region": "whole pose", "repair": "restore airborne pose", "confidence": .99}]})
    monkeypatch.setattr(identity, "audit_image_identity", lambda *a, **kw: pytest.fail("must stop before identity"))
    worker = GptImageGenWorkerThread(request_payload={"prompt": "test"},
        steps={"repaint": {"enabled": True}}, analysis_result={"description": "girl"},
        style_ref_path=style, checkpoint_path=str(tmp_path / "checkpoint.json"))
    worker.run()
    assert worker.last_status == "error" and worker.failed_stage == "quality"
    assert "identity" not in worker.checkpoint.data["stages"]


def test_gpt_only_style_checks_anatomy_without_forcing_repaint(app, monkeypatch, tmp_path):
    import modules.others.api_backend as backend
    import utils.refine_quality as quality
    source = image(tmp_path, "source.png")
    monkeypatch.setattr(backend, "generate_image_aigc2d_gpt", lambda **kw: [source])
    monkeypatch.setattr(backend, "generate_image_repaint", lambda *a, **kw: pytest.fail("repaint is disabled"))
    monkeypatch.setattr(quality, "audit_hand_quality", lambda p: {
        "needs_refine": True, "structural_issues": [{"region": "third leg", "repair": "remove extra leg", "confidence": .95}]})
    worker = GptImageGenWorkerThread(request_payload={"prompt": "test", "skip_repaint": True},
        steps={"repaint": {"enabled": False}}, analysis_result={"description": "girl"},
        checkpoint_path=str(tmp_path / "checkpoint.json"))
    worker.run()
    assert worker.last_status == "error" and worker.failed_stage == "hands"
    assert "publish" not in worker.checkpoint.data["stages"]


def test_identity_residual_blocks_publish_unless_variation_authorized(app, monkeypatch, tmp_path):
    import modules.others.api_backend as backend
    import utils.analysis_gen as pipeline
    import utils.identity_audit as identity
    import utils.refine_quality as quality
    source = image(tmp_path, "source.png")
    monkeypatch.setattr(backend, "generate_image_aigc2d_gpt", lambda **kw: [source])
    monkeypatch.setattr(pipeline, "run_gpt_image_pipeline", lambda *a, **kw: [source])
    monkeypatch.setattr(pipeline, "publish_final_output", lambda p, **kw: p)
    monkeypatch.setattr(backend, "generate_image_repaint", lambda *a, **kw: [source])
    monkeypatch.setattr(identity, "audit_image_identity", lambda *a, **kw: {
        "mismatch": True, "severity": "minor", "differences": [
            {"feature": "hair_colour", "expected": "blonde", "observed": "brown",
             "correction": "restore blonde hair", "confidence": .99}]})
    monkeypatch.setattr(quality, "audit_hand_quality", lambda p: {"needs_refine": False, "structural_issues": []})
    for authorized in (False, True):
        worker = GptImageGenWorkerThread(request_payload={"prompt": "test", "skip_quality_refine": True,
            "skip_identity_refine": authorized}, steps={"repaint": {"enabled": True}},
            analysis_result={"description": "girl"}, checkpoint_path=str(tmp_path / f"cp-{authorized}.json"))
        worker.run()
        assert worker.last_status == ("success" if authorized else "error")
        if not authorized:
            assert worker.failed_stage == "identity"
            assert "publish" not in worker.checkpoint.data["stages"]


def test_strict_pipeline_detects_swallowed_failure(monkeypatch, tmp_path):
    from utils import post_process as pp
    from utils.analysis_gen import run_gpt_image_pipeline
    source = image(tmp_path, "source.png")
    monkeypatch.setattr(pp, "run_pipeline", lambda *a, **kw: [source])
    monkeypatch.setattr(pp, "pipeline_failures", lambda d: [{"step": "repaint", "error": "timeout"}])
    with pytest.raises(RuntimeError, match="repaint: timeout"):
        run_gpt_image_pipeline([source], {"repaint": {"enabled": True}}, work_dir=str(tmp_path), strict=True)


def test_import_legacy_first_image(tmp_path):
    from utils.generation_checkpoint import import_generation_checkpoint
    first = image(tmp_path, "abcd1234_first.png")
    manifest = tmp_path / "first.png.request.json"
    manifest.write_text(json.dumps({"first_image": first, "prompt": "original prompt",
        "style_name": "test", "steps": {"repaint": {"enabled": True}},
        "size": "1536x1024", "quality": "medium"}), encoding="utf-8")
    path, data = import_generation_checkpoint(str(manifest))
    assert os.path.isfile(path)
    assert data["snapshot"]["request_payload"]["prompt"] == "original prompt"
    assert data["snapshot"]["size"] == "1536x1024"
    assert data["stages"]["first"]["outputs"] == [first]
    assert data["imported_legacy"]


def test_manual_repaint_failure_resumes_same_input_and_workdir(app, monkeypatch, tmp_path):
    import utils.analysis_gen as pipeline
    import modules.others.api_backend as backend
    first = image(tmp_path, "first.png")
    current = image(tmp_path, "current.png")
    output = image(tmp_path, "new.png")
    cp = GenerationCheckpoint(str(tmp_path / "checkpoint.json"))
    settings = dict(request_payload={"prompt": "test", "manual_repaint_input": current,
                                    "skip_quality_refine": True},
                    steps={"repaint": {"enabled": True, "scope": "manual"}})
    cp.data["snapshot"] = settings
    cp.begin("first", [])
    cp.complete("first", [first])
    calls = []
    def repaint(paths, *a, **kw):
        calls.append((list(paths), kw["work_dir"]))
        if len(calls) == 1:
            raise RuntimeError("failure")
        return [output]
    monkeypatch.setattr(backend, "generate_image_aigc2d_gpt", lambda **kw: pytest.fail("GPT must not run"))
    monkeypatch.setattr(pipeline, "run_gpt_image_pipeline", repaint)
    monkeypatch.setattr(pipeline, "publish_final_output", lambda path, **kw: path)
    worker = GptImageGenWorkerThread(**settings, checkpoint_path=cp.path, restart_stage="pipeline")
    worker.run()
    assert worker.last_status == "error"
    data = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    worker = GptImageGenWorkerThread(**data["snapshot"], checkpoint_path=cp.path)
    worker.run()
    assert worker.last_status == "success"
    assert calls[0] == calls[1]
    assert calls[0][0] == [current]
