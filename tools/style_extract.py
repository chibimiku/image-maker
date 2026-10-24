"""Run the existing multi-image style extraction workflow from a JSON specification."""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--controlled-tests", action="store_true",
                        help="Compare an existing preset's explicit motif conditions without extraction")
    parser.add_argument("--complete-missing", action="store_true",
                        help="Complete skipped GPT tests using spec.lexical_repairs, without repeating extraction")
    args = parser.parse_args()
    os.chdir(BASE)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtCore import QCoreApplication
    from modules.others.api_backend import load_config, resolve_text_api_key, get_api_config
    from modules.image_analysis.style_extraction_runner import ObservedStyleWorker
    from modules.image_analysis.style_analyzer import get_style_analyzer_missing_prompt_files
    app = QCoreApplication.instance() or QCoreApplication([])
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    if args.controlled_tests:
        from modules.image_analysis.style_extraction_runner import run_controlled_style_tests
        result = run_controlled_style_tests(spec, dry_run=args.dry_run)
        print(json.dumps(result if args.dry_run else {k: result[k] for k in
                         ("successful_stages", "expected_stages")}, ensure_ascii=False, indent=2))
        return 0 if args.dry_run or result["successful_stages"] == result["expected_stages"] else 1
    config = load_config()
    text_config = (config["base_url"], resolve_text_api_key(config), config["model"])
    if not all(text_config) or get_style_analyzer_missing_prompt_files():
        raise ValueError("Missing text credentials, model, or prompt templates")
    paths = [str(Path(p).resolve(strict=True)) for p in spec["image_paths"]]
    if len(paths) < 2 or not 1 <= spec["rounds"] <= 50:
        raise ValueError("Need at least two references and 1–50 rounds")
    reference = str(Path(spec["style_reference"]).resolve(strict=True))
    if reference not in paths:
        raise ValueError("Test style reference must be in the frozen dataset")
    random.seed(spec.get("sampling_seed", 0))
    existing = json.loads(Path(spec["resume_state"]).read_text(encoding="utf-8")) if spec.get("resume_state") else None
    worker = ObservedStyleWorker(
        paths, text_config[1], text_config[0], text_config[2],
        total_rounds=spec["rounds"], images_per_round=spec.get("images_per_round", 4),
        existing_state=existing,
        output_dir=spec["output_dir"], timeout_seconds=max(600, int(config.get("text_analysis_timeout_seconds", 600))),
        enable_test_gen=True, test_prompt=Path(spec["test_prompt_file"]).read_text(encoding="utf-8"),
        seed_prompts=Path(spec["seed_prompt_file"]).read_text(encoding="utf-8") if spec.get("seed_prompt_file") else "",
        test_style_ref_path=reference, img_aspect_ratio=spec.get("aspect_ratio", "auto"),
        file_prefix=spec["name"], comparison_config=text_config,
        dataset_selection={"source": str(Path(args.spec).resolve()), "frozen_images": [
            {"path": p, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in paths]})
    if spec.get("revision_guidance_file"):
        worker.revision_guidance = Path(spec["revision_guidance_file"]).read_text(encoding="utf-8")
    worker.baseline_observation_dir = spec.get("baseline_observation_dir", "")
    worker.baseline_ranking_path = spec.get("baseline_ranking_path", "")
    for node in ("aigc2d", "aigc-2d-gpt"):
        if not get_api_config(api_type=node).get("api_key"):
            raise ValueError(f"Missing image credentials: {node}")
    preflight = {"spec": spec, "text_model": text_config[2], "image_models": {
        n: {k: get_api_config(api_type=n).get(k) for k in ("model", "quality")}
        for n in ("aigc2d", "aigc-2d-gpt")}, "dataset": worker.dataset_selection,
        "planned_image_stages": (spec["rounds"] + 1) * 3}
    if args.dry_run:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0
    directory = Path(worker.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "preflight.json").exists() and not args.complete_missing:
        raise ValueError("Output directory already used; choose a fresh run directory")
    if not args.complete_missing:
        (directory / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2), encoding="utf-8")
    def log(message):
        print(message, flush=True)
        with (directory / "run.log").open("a", encoding="utf-8") as target:
            target.write(message + "\n")
    worker.log_signal.connect(log)
    if args.complete_missing:
        worker.complete_missing_gpt(directory / (spec["name"] + "_style_iter_result.json"),
                                    spec["lexical_repairs"], bool(spec.get("retry_missing_gemini")))
        return 0
    status = []
    worker.finish_signal.connect(lambda result, text, path: status.append(result))
    worker.run()
    path = directory / (spec["name"] + "_style_iter_result.json")
    if path.is_file():
        worker.aggregate(json.loads(path.read_text(encoding="utf-8")))
    return 0 if status == ["success"] else 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
