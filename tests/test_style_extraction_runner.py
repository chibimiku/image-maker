from modules.image_analysis.style_extraction_runner import channel_selections
import pytest


@pytest.mark.parametrize("fallback", [False, True])
def test_controlled_test_checkpoints_prevent_resending_and_reject_fallback(monkeypatch, tmp_path, fallback):
    import json
    from PIL import Image
    import modules.others.api_backend as backend
    import utils.analysis_gen as generation
    from modules.image_analysis.style_extraction_runner import run_controlled_style_tests
    monkeypatch.chdir(tmp_path)
    for path in ("conf/config.json", "prompts/gpt-image-optimize/config.json"):
        p = tmp_path / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}", encoding="utf-8")
    ref = tmp_path / "reference.png"
    Image.new("RGB", (20, 30), "white").save(ref)
    styles = tmp_path / "styles.json"
    styles.write_text(json.dumps({"sample": {"prompt": "fine ink", "prompt_gpt": "drawing grammar",
        "ref_image": str(ref), "repaint_clauses": ["keep content"],
        "motif_clauses": ["botanical accent"], "motif_enabled": False}}), encoding="utf-8")
    subject = tmp_path / "subject.txt"
    subject.write_text("Original adult in a bookshop", encoding="utf-8")
    spec = {"styles_file": str(styles), "style_name": "sample", "test_prompt_file": str(subject),
            "output_dir": str(tmp_path / "out"), "image_paths": [str(ref)], "aspect_ratio": "2:3",
            "conditions": [{"id": "off", "motif_enabled": False}]}
    monkeypatch.setattr(backend, "get_api_config", lambda **kw: {"api_key": "fake", "model": "configured", "quality": "max"})
    monkeypatch.setattr(generation, "build_first_pass_request", lambda *a, **kw: {"prompt": "same subject", "image_paths": [str(ref)]})
    calls = []
    def image_call(channel, **kw):
        calls.append(channel)
        path = tmp_path / (channel + ".png")
        Image.new("RGB", (20, 30), "red" if channel == "gemini" else "blue").save(path)
        return [str(path)]
    monkeypatch.setattr(backend, "generate_image_aigc2d", lambda **kw: image_call("gemini", **kw))
    monkeypatch.setattr(backend, "generate_image_aigc2d_gpt", lambda **kw: image_call("gpt", **kw))
    def repaint(paths, steps, **kw):
        calls.append("repaint")
        assert kw["strict"] is True
        if fallback:
            return paths
        path = tmp_path / "repaint.png"
        Image.new("RGB", (20, 30), "green").save(path)
        return [str(path)]
    monkeypatch.setattr(generation, "run_gpt_image_pipeline", repaint)
    result = run_controlled_style_tests(spec)
    assert result["successful_stages"] == (2 if fallback else 3)
    assert result["stages"]["off/gpt_repainted"]["status"] == ("failed" if fallback else "succeeded")
    run_controlled_style_tests(spec)
    assert calls == ["gemini", "gpt", "repaint"]


def test_feature_brief_remains_in_round_history():
    from modules.image_analysis.style_extraction_runner import ObservedStyleWorker
    worker = ObservedStyleWorker([], "key", "https://example.invalid/v1", "text",
                                 comparison_config=("https://example.invalid/v1", "key", "text"))
    worker.revision_guidance = "Change eye drawing grammar while preserving adult identity."
    assert worker.revision_guidance in worker._build_iteration_history_text([])


def test_channels_choose_different_rounds_and_keep_exact_tested_package():
    def record(label):
        return {"generated_files": {"gemini_direct": [label + "g.png"],
                                   "gpt_first_pass": [label + "p.png"]},
                "prompt_variants": {"marker": label}}
    state = {"test_images": {"round_1": record("early"), "round_5": record("late")}}
    ranking = {"rows": [
        {"id": "round_1/gemini_direct/0", "eligible": True, "sort_value": "9"},
        {"id": "round_5/gpt_first_pass/0", "eligible": True, "sort_value": "8"},
        {"id": "round_1/gpt_first_pass/0", "eligible": True, "sort_value": "7"},
        {"id": "round_5/gemini_direct/0", "eligible": True, "sort_value": "6"}]}
    result = channel_selections(state, ranking)
    assert result["gemini_direct"]["best_id"] == "round_1/gemini_direct/0"
    assert result["gpt_first_pass"]["best_id"] == "round_5/gpt_first_pass/0"
    assert result["gemini_direct"]["tested_record"]["prompt_variants"]["marker"] == "early"
    assert result["gpt_first_pass"]["tested_record"]["prompt_variants"]["marker"] == "late"
    assert result["gpt_repainted"]["best_id"] is None


def test_excluded_higher_score_cannot_win_and_ties_remain_manual():
    state = {"test_images": {"round_1": {"generated_files": {
        "gpt_first_pass": ["bad.png", "a.png", "b.png"]}}}}
    ranking = {"rows": [
        {"id": "round_1/gpt_first_pass/0", "eligible": False, "sort_value": "10"},
        {"id": "round_1/gpt_first_pass/1", "eligible": True, "sort_value": "8.0"},
        {"id": "round_1/gpt_first_pass/2", "eligible": True, "sort_value": "8.00"}]}
    result = channel_selections(state, ranking)["gpt_first_pass"]
    assert result["best_id"] == "round_1/gpt_first_pass/1"
    assert result["status"] == "manual_review_tie"
    assert len(result["tied_best_ids"]) == 2


def test_aggregate_preserves_ancestor_scores_and_remaps_previous_final(monkeypatch, tmp_path):
    import json
    import modules.image_analysis.style_extraction_runner as runner
    from modules.image_analysis.style_comparison import DIMENSIONS
    def scored(candidate_id):
        return {"id": candidate_id, **{key: 8 for key in (*DIMENSIONS, "subject_fidelity")},
                "reason": "source comparison", "dimension_evidence": {
                    key: dict(reference="ink", candidate="ink", difference="minor") for key in DIMENSIONS},
                "gates": dict(reference_content_copied=False, major_structure_defect=False,
                              explicit_subject_mismatch=False, uncertain=False)}
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"rows": [scored("round_3/gemini_direct/0"),
                                         scored("final/gpt_first_pass/0")]}), encoding="utf-8")
    state = {"test_images": {"round_3": {"generated_files": {"gemini_direct": ["old.jpg"]}},
                            "final_baseline_round_7": {"generated_files": {"gpt_first_pass": ["final.png"]}}},
             "baseline_final_stages": ["final_baseline_round_5", "final_baseline_round_7"]}
    worker = runner.ObservedStyleWorker([], "key", "https://example.invalid/v1", "text",
                                       output_dir=str(tmp_path),
                                       comparison_config=("https://example.invalid/v1", "key", "text"))
    worker.baseline_ranking_path = str(prior)
    monkeypatch.setattr(runner, "comparison_inputs", lambda *a, **kw: (None, None, None, "digest"))
    monkeypatch.setattr(runner, "write_comparison_report", lambda *a, **kw: None)
    worker.aggregate(state)
    result = json.loads((tmp_path / "all-stage-ranking.json").read_text(encoding="utf-8"))
    assert result["coverage"] == {"evaluated": 2, "expected": 2, "missing": []}
    assert {row["id"] for row in result["rows"]} == {"round_3/gemini_direct/0", "final_baseline_round_7/gpt_first_pass/0"}


@pytest.mark.parametrize("stage,missing_gemini", [("round_1", False), ("final", False), ("final", True)])
def test_supplement_keeps_gemini_and_original_blocked_record(monkeypatch, tmp_path, stage, missing_gemini):
    import json
    import modules.image_analysis.style_extraction_runner as runner
    import modules.others.api_backend as backend
    import utils.analysis_gen as generation
    prompt = ("Palette: clear local colours\nLighting: soft tinted shadows\n"
              "Brushwork: polished manga colour with transparent washes\n"
              "Edges: thin tapered contours\nTexture: subtle pigment grain\n"
              "Composition density: open areas with focal detail\n"
              "Detail level: selective fine ink marks\nAvoid: plastic gloss and muddy blending")
    original = {"prompts_used": "master", "prompt_variants": {"gpt_image_prompt": prompt},
                "generated_files": {"gemini_direct": ["existing-gemini.jpg"],
                                    "gpt_first_pass": [], "gpt_repainted": []},
                "style_reference": "reference.png", "round": 1}
    if missing_gemini:
        original["generated_files"]["gemini_direct"] = []
    state = {"test_images": {"round_1": original}, "dataset": {"images": ["reference.png"]}}
    if stage == "final":
        state["test_images"] = {}
        state["final_test_images"] = original
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    worker = runner.ObservedStyleWorker(["reference.png"], "key", "https://example.invalid/v1", "text",
                                       output_dir=str(tmp_path), img_aspect_ratio="2:3",
                                       comparison_config=("https://example.invalid/v1", "key", "text"))
    def assemble(package, reference, ratio, motif_prompt):
        assert "manga" not in package["gpt_image_prompt"]
        return {"gpt_first_pass": {"prompt": package["gpt_image_prompt"],
                "image_paths": [reference], "model": "configured"},
                "gemini_direct": {"prompt": "render", "image_paths": [reference],
                                  "model": "configured", "resolution": "2K"},
                "gpt_repaint": {"steps": {}, "style_ref_path": reference, "style_clauses": []}}
    monkeypatch.setattr(worker, "_assemble_test_generation_requests", assemble)
    monkeypatch.setattr(backend, "generate_image_aigc2d_gpt", lambda **kwargs: ["new-gpt.png"])
    gemini_calls = []
    def generate_gemini(**kwargs):
        gemini_calls.append(kwargs["retry_limit"])
        return ["new-gemini.jpg"]
    monkeypatch.setattr(backend, "generate_image_aigc2d", generate_gemini)
    monkeypatch.setattr(generation, "run_gpt_image_pipeline", lambda *args, **kwargs: ["new-repaint.jpg"])
    observed = []
    monkeypatch.setattr(runner, "compare_state_in_process", lambda p, *args, **kwargs:
                        observed.append(json.loads(__import__("pathlib").Path(p).read_text(encoding="utf-8"))))
    monkeypatch.setattr(worker, "aggregate", lambda state: None)
    result = worker.complete_missing_gpt(path, {stage: {"manga colour": "illustrated colour"}}, missing_gemini)
    record = result["final_test_images"] if stage == "final" else result["test_images"][stage]
    assert record["generated_files"]["gemini_direct"] == (["new-gemini.jpg"] if missing_gemini else ["existing-gemini.jpg"])
    assert gemini_calls == ([0] if missing_gemini else [])
    assert record["generated_files"]["gpt_first_pass"] == ["new-gpt.png"]
    assert record["generated_files"]["gpt_repainted"] == ["new-repaint.jpg"]
    observed_record = observed[0]["final_test_images"] if stage == "final" else observed[0]["test_images"][stage]
    assert observed_record["generated_files"]["gemini_direct"] == (["new-gemini.jpg"] if missing_gemini else [])
    backup = json.loads((tmp_path / "supplements" / stage / "original-record.json").read_text(encoding="utf-8"))
    assert backup["prompt_variants"]["gpt_image_prompt"] == prompt
