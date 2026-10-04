# -*- coding: utf-8 -*-
from modules.image_analysis.style_analyzer import (
    STYLE_ITER_VARIANTS_PROMPT_FILE,
    StyleIterativeWorkerThread,
    _crop_image_regions,
    format_style_prompt_package,
    get_style_analyzer_missing_prompt_files,
    normalize_style_prompt_package,
)
from PIL import Image


def test_style_analyzer_has_channel_variant_prompt_file():
    assert STYLE_ITER_VARIANTS_PROMPT_FILE not in get_style_analyzer_missing_prompt_files()


def test_format_style_prompt_package_exposes_all_channels():
    text = format_style_prompt_package("MASTER", {
        "gemini_full_prompt": "GEMINI",
        "gpt_image_prompt": "Palette: pale",
        "gemini_repaint_clauses": ["Repair lines."],
        "face_hair_clauses": ["Use sharp eyes."],
        "optional_motifs": ["Sparse butterflies near the frame edge."],
        "negative_rules": ["No copied identity."],
        "usage_profiles": {"portrait_focus": "Prioritize eyes."},
        "evidence_summary": {"variable_traits": ["Background density varies."]},
    })
    for expected in ("MASTER", "GEMINI", "Palette: pale", "Repair lines.",
                     "Use sharp eyes.", "Sparse butterflies", "No copied identity.", "portrait_focus",
                     "Background density varies."):
        assert expected in text


def test_normalize_prompt_package_builds_app_ready_style_entry():
    package = normalize_style_prompt_package({
        "gemini_full_prompt": "FULL GEMINI",
        "gpt_image_prompt": (
            "Palette: muted jewel tones\nLighting: soft side light\n"
            "Brushwork: layered opaque strokes\nEdges: tapered coloured contours\n"
            "Texture: fine paper grain\nComposition density: balanced negative space\n"
            "Detail level: selective focal detail\nAvoid: global haze and copied content"
        ),
        "gemini_repaint_clauses": "Keep primary contours continuous.",
        "face_hair_clauses": ["Use grouped locks."],
        "optional_motifs": ["Sparse butterflies at the frame edge."],
    }, "MASTER")
    assert package["gpt_image_prompt_valid"] is True
    assert package["gemini_repaint_clauses"] == ["Keep primary contours continuous."]
    assert package["style_entry"] == {
        "prompt": "FULL GEMINI",
        "prompt_gpt": package["gpt_image_prompt"],
        "repaint_clauses": ["Keep primary contours continuous."],
        "motif_clauses": ["Sparse butterflies at the frame edge."],
        "motif_enabled": True,
        "enabled": True,
    }


def test_normalize_prompt_package_does_not_export_invalid_gpt_prompt():
    package = normalize_style_prompt_package({"gpt_image_prompt": "Palette: blue"}, "MASTER")
    assert package["gpt_image_prompt_valid"] is False
    assert package["style_entry"]["prompt_gpt"] == ""
    assert package["style_entry"]["prompt"] == "MASTER"


def test_normalize_prompt_package_allows_style_rendering_terms_for_face_and_hair():
    package = normalize_style_prompt_package({
        "gpt_image_prompt": (
            "Palette: luminous warm-cool harmony with translucent blush\n"
            "Lighting: soft directional light with coloured bounce\n"
            "Brushwork: grouped hair masses and clean iris gradients\n"
            "Edges: tapered coloured contours around eyes and focal forms\n"
            "Texture: controlled pigment pooling and smooth skin planes\n"
            "Composition density: balanced negative space\n"
            "Detail level: organized catchlights and selective strands\n"
            "Avoid: copied identity and noisy micro-detail"
        )
    }, "MASTER")
    assert package["gpt_image_prompt_valid"] is True
    assert package["style_entry"]["prompt_gpt"]


def test_normalize_prompt_package_rejects_specific_sampled_hair_and_eye_colours():
    package = normalize_style_prompt_package({
        "gpt_image_prompt": (
            "Palette: blue hair and violet eyes\nLighting: soft side light\n"
            "Brushwork: layered opaque strokes\nEdges: tapered coloured contours\n"
            "Texture: fine paper grain\nComposition density: balanced negative space\n"
            "Detail level: selective focal detail\nAvoid: global haze"
        )
    }, "MASTER")
    assert package["gpt_image_prompt_valid"] is False
    assert package["style_entry"]["prompt_gpt"] == ""


def test_state_rebuild_preserves_test_images_prompt_pack_and_created_at(tmp_path):
    worker = StyleIterativeWorkerThread(
        image_paths=[], api_key="key", base_url="https://example.invalid/v1",
        model_name="model", output_dir=str(tmp_path), file_prefix="style")
    previous = {
        "created_at": "2026-09-20 12:00:00",
        "test_images": {"round_1": {"generated_files": ["one.png"]}},
        "prompt_variants": {"gemini_full_prompt": "FULL"},
        "final_test_images": {"status": "partial", "generated_files": {"gemini_direct": ["final.png"]}},
    }
    state = worker._build_state(2, [{"step": 1}], previous)
    assert state["created_at"] == previous["created_at"]
    assert state["test_images"] == previous["test_images"]
    assert state["prompt_variants"] == previous["prompt_variants"]
    assert state["final_test_images"] == previous["final_test_images"]


def test_crop_names_do_not_collide_for_same_basename_in_different_dirs(tmp_path):
    first = tmp_path / "a" / "same.png"
    second = tmp_path / "b" / "same.png"
    first.parent.mkdir()
    second.parent.mkdir()
    Image.new("RGB", (100, 140), "white").save(first)
    Image.new("RGB", (100, 140), "black").save(second)
    out = tmp_path / "crops"
    out.mkdir()
    first_crops = _crop_image_regions(str(first), str(out), crop_count=1)
    second_crops = _crop_image_regions(str(second), str(out), crop_count=1)
    assert first_crops[0] != second_crops[0]
    assert len(list(out.glob("*.jpg"))) == 2


def test_round_test_generation_returns_gemini_gpt_and_repaint(monkeypatch, tmp_path):
    import modules.image_analysis.style_analyzer as sa
    import utils.analysis_gen as ag
    import utils.styles as styles

    ref = tmp_path / "ref.png"
    Image.new("RGB", (64, 96), "white").save(ref)
    gemini = tmp_path / "gemini.png"
    first = tmp_path / "gpt.png"
    repainted = tmp_path / "gpt-repainted.png"

    monkeypatch.setattr(sa, "get_api_config", lambda api_type: {
        "model": "gemini-model" if api_type == "aigc2d" else "gpt-model"})
    calls = {}
    monkeypatch.setattr(sa, "generate_image_aigc2d", lambda **kwargs: calls.setdefault("gemini", kwargs) and [str(gemini)])
    monkeypatch.setattr(sa, "generate_image_aigc2d_gpt", lambda **kwargs: calls.setdefault("gpt", kwargs) and [str(first)])
    monkeypatch.setattr(styles, "compose_style_prompt", lambda *args, **kwargs: "gemini prompt")
    monkeypatch.setattr(ag, "build_gpt_image_request", lambda *args, **kwargs: {
        "prompt": "gpt prompt", "image_paths": [str(ref)]})
    monkeypatch.setattr(ag, "run_gpt_image_pipeline", lambda *args, **kwargs: [str(repainted)])

    worker = StyleIterativeWorkerThread(
        image_paths=[str(ref)], api_key="key", base_url="https://example.invalid/v1",
        model_name="text", output_dir=str(tmp_path), enable_test_gen=True,
        test_prompt="fixed subject", test_style_ref_path=str(ref))
    package = {
        "gemini_full_prompt": "full",
        "gpt_image_prompt": (
            "Palette: muted jewel tones\nLighting: soft side light\n"
            "Brushwork: layered opaque strokes\nEdges: tapered coloured contours\n"
            "Texture: fine paper grain\nComposition density: balanced negative space\n"
            "Detail level: selective focal detail\nAvoid: global haze and copied content"),
        "gemini_repaint_clauses": ["Keep contours continuous."],
    }
    result = worker._generate_test_image("MASTER", 1, "unused.json", package)
    assert result == {
        "gemini_direct": [str(gemini)],
        "gpt_first_pass": [str(first)],
        "gpt_repainted": [str(repainted)],
    }
    assert calls["gemini"]["aspect_ratio"] == "2:3"
    assert calls["gpt"]["aspect_ratio"] == "2:3"

    worker._generate_test_image("FINAL MASTER", 1, "unused.json", package, stage="final")
    assert (tmp_path / "test-generations" / "final").is_dir()


def test_gemini_config_failure_does_not_prevent_gpt_test(monkeypatch, tmp_path):
    import modules.image_analysis.style_analyzer as sa
    import utils.analysis_gen as ag

    def config(api_type):
        if api_type == "aigc2d":
            raise RuntimeError("Gemini unavailable")
        return {"model": "gpt-model"}

    monkeypatch.setattr(sa, "get_api_config", config)
    monkeypatch.setattr(sa, "generate_image_aigc2d_gpt", lambda **kwargs: ["gpt.png"])
    monkeypatch.setattr(ag, "build_gpt_image_request", lambda *args, **kwargs: {
        "prompt": "gpt prompt", "image_paths": []})
    monkeypatch.setattr(ag, "run_gpt_image_pipeline", lambda *args, **kwargs: ["repaint.png"])
    worker = StyleIterativeWorkerThread(
        [], "key", "https://example.invalid/v1", "text", output_dir=str(tmp_path),
        test_prompt="subject")
    outputs = worker._generate_test_image("MASTER", 1, "unused", stage="final")
    record = worker._test_image_record("MASTER", {}, outputs)
    assert outputs["gpt_first_pass"] == ["gpt.png"]
    assert outputs["gpt_repainted"] == ["repaint.png"]
    assert record["status"] == "partial"
    assert "Gemini unavailable" in record["channel_status"]["gemini_direct"]["error"]


def test_empty_output_is_failed_and_repaint_requires_first_pass(tmp_path):
    worker = StyleIterativeWorkerThread([], "key", "https://example.invalid/v1", "text",
                                       output_dir=str(tmp_path))
    record = worker._test_image_record("MASTER", {}, {
        "gemini_direct": [], "gpt_first_pass": [], "gpt_repainted": []})
    assert record["status"] == "failed"
    assert record["channel_status"]["gemini_direct"]["status"] == "failed"
    assert record["channel_status"]["gpt_first_pass"]["status"] == "failed"
    assert record["channel_status"]["gpt_repainted"]["status"] == "not_run"


def _stub_style_steps(monkeypatch, worker):
    import modules.image_analysis.style_analyzer as sa

    monkeypatch.setattr(sa, "OpenAI", lambda **kwargs: object())
    monkeypatch.setattr(worker, "_step_commonality_extraction", lambda client, its, r, p: (
        its + [{"type": "commonality_extraction", "round": r, "art_style_prompts": "ROUND"}], "ROUND"))
    monkeypatch.setattr(worker, "_step_local_refinement", lambda client, its, p, r: (
        its + [{"type": "local_merge", "prompts_after": "LOCAL"}], "LOCAL"))
    monkeypatch.setattr(worker, "_step_final_review", lambda client, its, p, r: (
        its + [{"type": "final_review", "prompts_after": "FINAL"}], "FINAL"))
    monkeypatch.setattr(worker, "_step_prompt_variants", lambda client, p: {"gemini_full_prompt": p})


def test_run_tests_final_prompt_after_local_refinement_and_review(monkeypatch, tmp_path):
    import json

    worker = StyleIterativeWorkerThread(
        [], "key", "https://example.invalid/v1", "text", total_rounds=1,
        images_per_round=0, output_dir=str(tmp_path), enable_test_gen=True, test_prompt="subject")
    _stub_style_steps(monkeypatch, worker)
    calls = []

    def generate(prompt, round_num, output_path, prompt_variants=None, stage="round"):
        calls.append((stage, prompt, prompt_variants))
        return {"gemini_direct": [f"{stage}.png"], "gpt_first_pass": [], "gpt_repainted": []}

    monkeypatch.setattr(worker, "_generate_test_image", generate)
    finished = []
    worker.finish_signal.connect(lambda status, text, path: finished.append((status, text)))
    worker.run()
    assert [(stage, prompt) for stage, prompt, _ in calls] == [("round", "ROUND"), ("final", "FINAL")]
    assert calls[-1][2]["gemini_full_prompt"] == "FINAL"
    state = json.loads((tmp_path / "style_style_iter_result.json").read_text(encoding="utf-8"))
    assert state["test_images"]["round_1"]["prompts_used"] == "ROUND"
    assert state["final_test_images"]["prompts_used"] == "FINAL"
    assert state["final_test_images"]["status"] == "partial"
    assert "终审版本测试" in finished[0][1]
    assert "final.png" in finished[0][1]
    assert "部分成功" in finished[0][1]


def test_disabled_generation_explicitly_marks_final_version_untested(monkeypatch, tmp_path):
    import json

    worker = StyleIterativeWorkerThread(
        [], "key", "https://example.invalid/v1", "text", total_rounds=1,
        images_per_round=0, output_dir=str(tmp_path), enable_test_gen=False)
    _stub_style_steps(monkeypatch, worker)
    monkeypatch.setattr(worker, "_generate_test_image", lambda *args, **kwargs: (
        (_ for _ in ()).throw(AssertionError("Generation must stay disabled"))))
    worker.run()
    state = json.loads((tmp_path / "style_style_iter_result.json").read_text(encoding="utf-8"))
    assert state["final_test_images"]["status"] == "not_run"


def test_gui_does_not_show_partial_final_test_as_full_success(tmp_path):
    import json
    from types import SimpleNamespace
    from modules.image_analysis.style_analyzer import StyleAnalyzerWidget

    class Label:
        def setText(self, text):
            self.text = text

        def setPlainText(self, text):
            self.text = text

        def setEnabled(self, enabled):
            self.enabled = enabled

    path = tmp_path / "result.json"
    path.write_text(json.dumps({"final_test_images": {"status": "partial"}}), encoding="utf-8")
    states = []
    widget = SimpleNamespace(
        thread=None, set_running_state=lambda value: None,
        deep_table=SimpleNamespace(setRowCount=lambda count: None), deep_details=Label(),
        output_path_label=Label(), open_output_dir_btn=Label(), result_edit=Label(),
        progress_label=Label(), set_task_state=lambda status, text: states.append((status, text)),
        log_msg=lambda text: None)
    StyleAnalyzerWidget.on_analysis_finished(widget, "success", "PROMPT PACK", str(path))
    assert states[0][0] == "error"
    assert "部分成功" in states[0][1]
    assert "终审版本测试" in widget.progress_label.text
    assert widget.result_edit.text == "PROMPT PACK"
