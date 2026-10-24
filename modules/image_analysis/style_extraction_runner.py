"""Headless orchestration of the GUI extractor with per-stage visual observations."""
import copy
import json
from decimal import Decimal
from pathlib import Path

from modules.image_analysis.style_analyzer import StyleIterativeWorkerThread
from modules.image_analysis.style_comparison import (
    calculate_ranking, candidates_from_state, compare_state_in_process, comparison_inputs,
    write_comparison_report,
)
from modules.image_analysis.style_deep_comparison import atomic_json


def run_controlled_style_tests(spec, dry_run=False):
    """Test an existing style with explicit motif toggles, without re-extraction.

    Each paid stage is checkpointed before dispatch; ambiguous interrupted stages
    are never automatically resent. Successful outputs are reused on resumption.
    """
    from modules.others.api_backend import (get_api_config, generate_image_aigc2d,
                                            generate_image_aigc2d_gpt)
    from utils.analysis_gen import build_first_pass_request, run_gpt_image_pipeline
    from utils.styles import compose_style_prompt, style_motif_prompt
    import hashlib
    from PIL import Image

    styles = json.loads(Path(spec["styles_file"]).read_text(encoding="utf-8"))
    name = spec["style_name"]
    entry = copy.deepcopy(styles[name])
    subject = Path(spec["test_prompt_file"]).read_text(encoding="utf-8")
    reference = str(Path(entry["ref_image"]).resolve(strict=True))
    root = Path(spec["output_dir"]).resolve()
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    frozen_paths = [spec["styles_file"], spec["test_prompt_file"], reference,
                    "conf/config.json", "prompts/gpt-image-optimize/config.json"] + spec["image_paths"]
    frozen = {str(Path(p).resolve()): digest(p) for p in frozen_paths}
    gemini_cfg, gpt_cfg = (get_api_config(api_type=node) for node in ("aigc2d", "aigc-2d-gpt"))
    if not all(c.get("api_key") and c.get("model") for c in (gemini_cfg, gpt_cfg)):
        raise ValueError("Missing configured image model or credentials")
    conditions = spec["conditions"]
    ids = [c["id"] for c in conditions]
    if len(set(ids)) != len(ids) or any(not __import__("re").fullmatch(r"[a-zA-Z0-9_-]+", i) for i in ids):
        raise ValueError("Condition IDs must be unique safe names")
    plan = {"frozen_inputs": frozen, "style_entry": entry, "subject": subject,
            "models": {"gemini": gemini_cfg["model"], "gpt": gpt_cfg["model"],
                       "quality": gpt_cfg.get("quality")}, "cases": {}}
    for condition in conditions:
        variant = copy.deepcopy(styles)
        variant[name]["motif_enabled"] = bool(condition["motif_enabled"])
        motif = style_motif_prompt(variant, name)
        first = build_first_pass_request(variant, name, {}, content_text=subject)
        plan["cases"][condition["id"]] = {
            "gemini_prompt": compose_style_prompt("\n\n".join(x for x in (entry["prompt"], motif) if x),
                                                  subject, style_ref_attached=True),
            "gpt_request": first, "motif_enabled": condition["motif_enabled"]}
    if dry_run:
        return {"planned_image_stages": len(conditions) * 3, "models": plan["models"],
                "conditions": ids, "frozen_input_count": len(frozen)}
    root.mkdir(parents=True, exist_ok=True)
    plan_path, state_path = root / "controlled-plan.json", root / "controlled-state.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Frozen controlled-test plan changed; refusing resumption")
    else:
        atomic_json(str(plan_path), plan)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"stages": {}}
    def log(message):
        print(message, flush=True)
        with (root / "controlled-run.log").open("a", encoding="utf-8") as target:
            target.write(message + "\n")
    def dispatch(stage, action, source=None):
        old = state["stages"].get(stage)
        if old:
            if old["status"] == "succeeded":
                if any(digest(p) != h for p, h in old["output_hashes"].items()):
                    raise ValueError("Successful stage output changed")
                return old["outputs"]
            log(f"[REUSE STATUS] {stage}: {old['status']}; not resent")
            return []
        if any(digest(p) != h for p, h in frozen.items()):
            raise ValueError("Frozen input changed before dispatch")
        state["stages"][stage] = {"status": "dispatched", "outputs": []}
        atomic_json(str(state_path), state)
        log("[DISPATCH] " + stage)
        try:
            outputs = action() or []
            if not outputs:
                raise RuntimeError("No image returned")
            for path in outputs:
                with Image.open(path) as im:
                    im.verify()
            if source and any(digest(p) == digest(source) for p in outputs):
                raise RuntimeError("Repaint returned its source fallback; not a new repaint")
            state["stages"][stage].update(status="succeeded", outputs=outputs,
                                          output_hashes={p: digest(p) for p in outputs})
        except Exception as exc:
            state["stages"][stage].update(status="failed", error=f"{type(exc).__name__}: {exc}")
            outputs = []
        atomic_json(str(state_path), state)
        log(f"[STAGE] {stage}: {state['stages'][stage]['status']}")
        return outputs
    for case_id, case in plan["cases"].items():
        directory = root / case_id
        directory.mkdir(exist_ok=True)
        dispatch(case_id + "/gemini_direct", lambda: generate_image_aigc2d(
            prompt=case["gemini_prompt"], image_paths=[reference], model=gemini_cfg["model"],
            aspect_ratio=spec["aspect_ratio"], resolution="2K", api_type="aigc2d",
            save_sub_dir=str(directory), file_prefix="gemini", face_quality_boost=False,
            retry_limit=0, log_callback=log))
        first = dispatch(case_id + "/gpt_first_pass", lambda: generate_image_aigc2d_gpt(
            prompt=case["gpt_request"]["prompt"], image_paths=case["gpt_request"]["image_paths"],
            model=gpt_cfg["model"], quality=gpt_cfg.get("quality"), aspect_ratio=spec["aspect_ratio"],
            api_type="aigc-2d-gpt", save_sub_dir=str(directory), file_prefix="gpt", mode="generate",
            log_callback=log))
        if first:
            from utils.post_process import default_pipeline
            steps = default_pipeline()
            for value in steps.values():
                if isinstance(value, dict):
                    value["enabled"] = False
            steps["repaint"].update(enabled=True, scope="full", reference_mode=entry.get("repaint_reference_mode", "none"),
                                     clauses_without_image=True)
            dispatch(case_id + "/gpt_repainted", lambda: run_gpt_image_pipeline(
                first, steps, final_dir=str(directory), work_dir=str(directory / "repaint-steps"),
                style_ref_path=reference, style_clauses=entry["repaint_clauses"], strict=True,
                log_callback=log), source=first[0])
    state["expected_stages"] = len(conditions) * 3
    state["successful_stages"] = sum(v["status"] == "succeeded" for v in state["stages"].values())
    atomic_json(str(state_path), state)
    return state


def channel_selections(state, ranking):
    """Select independently, preserving each winner's exact tested prompt package."""
    candidates = {c["id"]: c for c in candidates_from_state(state)}
    selected = {}
    for channel in ("gemini_direct", "gpt_first_pass", "gpt_repainted"):
        rows = [r for r in ranking["rows"]
                if r["id"] in candidates and candidates[r["id"]]["channel"] == channel]
        eligible = [r for r in rows if r["eligible"]]
        winner = eligible[0] if eligible else None
        tied = [r["id"] for r in eligible if winner and Decimal(r["sort_value"]) == Decimal(winner["sort_value"])]
        selected[channel] = {
            "best_id": winner["id"] if winner else None,
            "tied_best_ids": tied,
            "status": "manual_review_tie" if len(tied) > 1 else "ranked" if winner else "no_eligible",
            "ranking": rows,
            "tested_record": copy.deepcopy(candidates[winner["id"]]["record"]) if winner else None,
        }
    return selected


class ObservedStyleWorker(StyleIterativeWorkerThread):
    """Observe actual output after every round without running local neural metrics."""

    def __init__(self, *args, comparison_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.comparison_config = comparison_config
        self.observations = {}

    def _build_iteration_history_text(self, iterations):
        text = super()._build_iteration_history_text(iterations)
        if getattr(self, "revision_guidance", ""):
            text += ("\n\nFEATURE REVISION BRIEF TO VERIFY AGAINST THE SOURCE REFERENCES:\n"
                     + self.revision_guidance)
        if self.observations:
            last = next(reversed(self.observations.values()))
            text += "\n\nOBSERVATIONS OF THE PREVIOUS ACTUAL TEST OUTPUTS:\n"
            text += json.dumps(last, ensure_ascii=False)
            text += ("\nUse these as regression observations, not new reference-style evidence. "
                     "Keep Gemini and GPT findings separate. A later round need not be better. "
                     "Never copy reference characters, cover typography, borders or publisher layouts.")
        return text

    def _step_prompt_variants(self, client, current_prompts):
        guidance = getattr(self, "revision_guidance", "")
        if not guidance:
            return super()._step_prompt_variants(client, current_prompts)
        package = super()._step_prompt_variants(
            client, current_prompts + "\n\nPACKAGING GUIDANCE: Retain the verified discriminative rendering "
            "instructions above in each channel variant. Preserve identity without freezing the old eye "
            "geometry or facial drawing abstraction. Verify this feature brief against the attached sources:\n" + guidance)
        package["variant_revision_guidance"] = guidance
        return package

    def _write_round_report(self, state, round_num, label="round"):
        stage = "final" if label == "final" else f"round_{round_num}"
        snapshot = copy.deepcopy(state)
        snapshot.pop("automatic_comparison", None)
        snapshot.pop("automatic_comparison_status", None)
        snapshot["test_images"] = {} if label == "final" else {
            stage: state["test_images"][stage]}
        snapshot["final_test_images"] = state.get("final_test_images", {}) if label == "final" else {}
        directory = Path(self.output_dir) / "observations" / stage
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "state.json"
        atomic_json(str(path), snapshot)
        try:
            result = compare_state_in_process(
                str(path), self.comparison_config, timeout=self.timeout_seconds,
                progress=self.log_signal.emit,
                cancelled=lambda: self._cancel_requested or self.isInterruptionRequested(),
                title=f"{stage} — actual output observations")
            notes = {r["id"]: {"eligible": r["eligible"], "gates": r["gates"],
                     "reason": r["reason"], "style_score": r["style_score"]}
                     for r in result["rows"]}
            self.observations[stage] = notes
            atomic_json(str(Path(self.output_dir) / "round-observations.json"), self.observations)
            self.log_signal.emit(f"[OBSERVATION {stage}] " + json.dumps(notes, ensure_ascii=False))
            self.aggregate(state)
            return result["report_path"]
        except Exception as exc:
            self.log_signal.emit(f"[OBSERVATION FAILED {stage}] {type(exc).__name__}: {exc}")
            self.observations[stage] = {"status": "failed", "error": str(exc)}
            atomic_json(str(Path(self.output_dir) / "round-observations.json"), self.observations)
            return ""

    def aggregate(self, state):
        candidates = candidates_from_state(state)
        assessments = []
        # A continuation can itself have several ancestors. Its immediate parent's
        # per-round directory alone does not cover those earlier successful tests.
        baseline_ranking = getattr(self, "baseline_ranking_path", "")
        if baseline_ranking:
            from modules.image_analysis.style_comparison import DIMENSIONS
            prior = json.loads(Path(baseline_ranking).read_text(encoding="utf-8"))
            for row in prior["rows"]:
                assessment = {key: row[key] for key in (*DIMENSIONS, "subject_fidelity", "id", "reason", "dimension_evidence")}
                assessment.update(row["gates"])
                if assessment["id"].startswith("final/") and state.get("baseline_final_stages"):
                    assessment["id"] = state["baseline_final_stages"][-1] + assessment["id"][5:]
                assessments.append(assessment)
        sources = []
        if getattr(self, "baseline_observation_dir", ""):
            sources.append((Path(self.baseline_observation_dir), True))
        sources.append((Path(self.output_dir) / "observations", False))
        for directory, baseline in sources:
            for path in sorted(directory.glob("*/automatic-comparison.json")):
                result = json.loads(path.read_text(encoding="utf-8"))
                for assessment in result.get("assessments", []):
                    assessment = copy.deepcopy(assessment)
                    if baseline and assessment["id"].startswith("final/") and state.get("baseline_final_stages"):
                        assessment["id"] = state["baseline_final_stages"][-1] + assessment["id"][5:]
                    assessments.append(assessment)
        by_id = {row["id"]: row for row in assessments}
        covered = [c for c in candidates if c["id"] in by_id]
        if not covered:
            return
        ranking = calculate_ranking(covered, [by_id[c["id"]] for c in covered])
        ranking.update(input_hash=comparison_inputs(state, self.comparison_config[2],
                       self.comparison_config[0], candidates=covered)[3],
                       model=self.comparison_config[2], endpoint=self.comparison_config[0])
        ranking["coverage"] = {"evaluated": len(covered), "expected": len(candidates),
                               "missing": [c["id"] for c in candidates if c["id"] not in by_id]}
        ranking["evaluation_note"] = "Same references and rubric per stage; single samples, no repeated-seed stability claim."
        atomic_json(str(Path(self.output_dir) / "channel-selections.json"), channel_selections(state, ranking))
        atomic_json(str(Path(self.output_dir) / "all-stage-ranking.json"), ranking)
        write_comparison_report(state, ranking, self.output_dir,
                                output_name="all-stage-comparison.html", title="All stages — independent channel selection")

    def complete_missing_gpt(self, state_path, repairs, retry_missing_gemini=False):
        """Complete skipped GPT tests using explicit lexical repairs; retain originals."""
        from modules.image_analysis.style_analyzer import normalize_style_prompt_package
        from modules.others.api_backend import generate_image_aigc2d_gpt, generate_image_aigc2d
        from utils.analysis_gen import run_gpt_image_pipeline
        from utils.styles import motif_prompt_from_clauses
        state_path = Path(state_path)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for stage, replacements in repairs.items():
            record = state["final_test_images"] if stage == "final" else state["test_images"][stage]
            if record["generated_files"].get("gpt_first_pass") and record["generated_files"].get("gpt_repainted"):
                continue
            directory = Path(self.output_dir).resolve() / "supplements" / stage
            directory.mkdir(parents=True, exist_ok=True)
            backup = directory / "original-record.json"
            if not backup.exists():
                atomic_json(str(backup), record)
            package = copy.deepcopy(record["prompt_variants"])
            before = package["gpt_image_prompt"]
            after = before
            for old, new in replacements.items():
                if old not in after and new not in after:
                    raise ValueError(f"Repair text missing in {stage}: {old}")
                after = after.replace(old, new)
            package["gpt_image_prompt"] = after
            package = normalize_style_prompt_package(package, record["prompts_used"])
            if not package["gpt_image_prompt_valid"]:
                raise ValueError(str(package["gpt_image_prompt_errors"]))
            record["prompt_variants"] = package
            record["validation_repair"] = {"before": before, "after": after,
                "replacements": replacements, "original_record": str(backup),
                "note": "Lexical repair only; original blocked GPT request was not sent. Existing successful outputs are reused; any missing Gemini recovery is recorded separately."}
            assembled = self._assemble_test_generation_requests(
                package, record["style_reference"], self.img_aspect_ratio,
                motif_prompt_from_clauses(package.get("optional_motifs") or []))
            if assembled.get("gpt_error"):
                raise ValueError(assembled["gpt_error"])
            payload = assembled["gpt_first_pass"]
            outputs = record["generated_files"]
            gemini_added = False
            if retry_missing_gemini and not outputs.get("gemini_direct"):
                gemini = assembled["gemini_direct"]
                outputs["gemini_direct"] = generate_image_aigc2d(
                    prompt=gemini["prompt"], image_paths=gemini["image_paths"],
                    model=gemini["model"], aspect_ratio=self.img_aspect_ratio,
                    resolution=gemini["resolution"], api_type="aigc2d",
                    save_sub_dir=str(directory), file_prefix=stage + "-gemini-supplement",
                    face_quality_boost=False, retry_limit=0, log_callback=self.log_signal.emit) or []
                gemini_added = bool(outputs["gemini_direct"])
                atomic_json(str(state_path), state)
            if not outputs.get("gpt_first_pass"):
                outputs["gpt_first_pass"] = generate_image_aigc2d_gpt(
                    prompt=payload["prompt"], image_paths=payload["image_paths"], model=payload["model"],
                    aspect_ratio=self.img_aspect_ratio, api_type="aigc-2d-gpt",
                    save_sub_dir=str(directory), file_prefix=stage + "-gpt-supplement", mode="generate",
                    log_callback=self.log_signal.emit) or []
                atomic_json(str(state_path), state)
            if outputs.get("gpt_first_pass") and not outputs.get("gpt_repainted"):
                repaint = assembled["gpt_repaint"]
                outputs["gpt_repainted"] = run_gpt_image_pipeline(
                    outputs["gpt_first_pass"], repaint["steps"], final_dir=str(directory),
                    work_dir=str(directory / "repaint-steps"), style_ref_path=repaint["style_ref_path"],
                    style_clauses=repaint["style_clauses"], log_callback=self.log_signal.emit) or []
            updated = self._test_image_record(record["prompts_used"], package, outputs, record.get("round"))
            updated["validation_repair"] = record["validation_repair"]
            if stage == "final":
                state["final_test_images"] = updated
            else:
                state["test_images"][stage] = updated
            atomic_json(str(state_path), state)
            snapshot = copy.deepcopy(state)
            observed = copy.deepcopy(updated)
            if not gemini_added:
                observed["generated_files"]["gemini_direct"] = []
            snapshot["test_images"] = {} if stage == "final" else {stage: observed}
            snapshot["final_test_images"] = observed if stage == "final" else {}
            evaluation = Path(self.output_dir) / "observations" / (stage + "-supplement") / "state.json"
            atomic_json(str(evaluation), snapshot)
            compare_state_in_process(str(evaluation), self.comparison_config, timeout=self.timeout_seconds,
                                     progress=self.log_signal.emit, title=stage + " — supplementary GPT tests")
            self.aggregate(state)
            self.log_signal.emit(f"[SUPPLEMENT COMPLETE] {stage}")
        return state
