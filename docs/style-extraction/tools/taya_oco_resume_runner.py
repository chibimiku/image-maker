"""taya_oco 画风提取续跑的无头 runner（scratch 运行脚本，不修改 App 业务代码）。

直接复用业务模块：
  * modules.image_analysis.style_analyzer.StyleIterativeWorkerThread  （续跑 3 轮 + 局部细化 + 终审 + 提示词包 + 三路测试生图）
  * modules.image_analysis.style_comparison.StyleComparisonWorker / calculate_ranking / comparison_inputs
      / write_comparison_report / candidates_from_state
  * modules.image_analysis.style_deep_comparison / tools/style_metrics_verify.py（本地深度指标）
  * modules.others.api_backend（文本/图片配置与密钥解析）

子命令：
  preflight --source-json J [--out F]
  resume    --source-json J --output-dir D [--total-rounds 3] [--images-per-round 4] ...
  compare   --state S [--batch-size N] [--timeout 900]
  deep      --state S [--device npu]
  entries   --state S --out F

规则：
  * 只在工作区写文件；拒绝在 IMAGE_MAKER_TEST_OUTPUT=1 下运行（避免产出被改道）。
  * 不打印任何密钥值，只打印 key 来源标签。
  * 输出目录必须与源 JSON 所在目录不同（独立目录、不覆盖原记录）。
"""
import argparse
import copy
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG_FILE = ROOT / "conf" / "config.json"
RUNNER_ID = "cache/temp/taya_oco_resume_runner.py"


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #

def fail(message, code=2):
    print(f"[错误] {message}", flush=True)
    raise SystemExit(code)


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text):
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return str(path)


def guard_test_isolation():
    if os.environ.get("IMAGE_MAKER_TEST_OUTPUT"):
        fail("IMAGE_MAKER_TEST_OUTPUT 已设置；续跑必须写入真实产出目录，请先清除该变量。")


def text_api_config():
    """与 app.get_text_config 同源：配置文件给 base_url/model，密钥走环境变量解析。"""
    from modules.others.api_backend import resolve_text_api_key, text_api_key_source
    config = read_json(CONFIG_FILE) if CONFIG_FILE.is_file() else {}
    base_url = str(config.get("base_url", "") or "").strip()
    model = str(config.get("model", "") or "").strip()
    key = resolve_text_api_key(config)
    if not base_url or not model:
        fail("conf/config.json 缺少 base_url 或 model（文本分析）")
    if not key:
        fail("文本分析 API Key 未解析到（环境变量 / .env / 配置）")
    return base_url, key, model, text_api_key_source(config)


def current_prompts_of(iterations):
    """与 StyleIterativeWorkerThread._get_current_prompts 同规则的自查副本（只用于核对）。"""
    for item in reversed(iterations or []):
        kind = item.get("type")
        if kind in ("final_review", "local_merge", "refinement_check"):
            return item.get("prompts_after", "")
        if kind == "commonality_extraction":
            return item.get("art_style_prompts", "")
        if kind in ("imported_prompts", "selected_version_seed"):
            return item.get("art_style_prompts", "")
    return ""


def candidate_paths(state):
    from modules.image_analysis.style_comparison import candidates_from_state
    return candidates_from_state(state)


def ensure_source_isolated(source_json, output_dir, state_path):
    """续跑输出必须独立且不覆盖原记录：状态文件名不得与源 JSON 相同。

    允许输出目录位于源 JSON 的父目录之下（任务要求 `data/<日期>/style-extraction/...`），
    只禁止「同一个结果文件被写两次」以及「输出目录就是源 JSON 本身」。
    """
    source_json = Path(source_json).resolve()
    if Path(output_dir).resolve() == source_json:
        fail("输出目录不能是源 JSON 文件")
    if Path(state_path).resolve() == source_json:
        fail("输出状态文件会覆盖源 JSON，请更换 --output-dir 或 --file-prefix")


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #

def collect_preflight(source_json, total_rounds, images_per_round):
    source_json = Path(source_json).resolve()
    state = read_json(source_json)
    iterations = state.get("iterations") or []
    numeric = [it.get("round") for it in iterations if isinstance(it.get("round"), (int, float))]
    last_round = int(max(numeric)) if numeric else 0
    restored = current_prompts_of(iterations)
    final_prompts = str(state.get("final_art_style_prompts") or "")
    images = [p for p in (state.get("dataset") or {}).get("images") or []]
    missing_images = [p for p in images if not os.path.isfile(p)]
    stages = sorted((state.get("test_images") or {}).keys())
    candidates = candidate_paths(state)
    missing_candidates = [c["id"] for c in candidates if not os.path.isfile(c["path"])]
    from modules.image_analysis.style_analyzer import (
        StyleIterativeWorkerThread, get_style_analyzer_missing_prompt_files)
    worker = StyleIterativeWorkerThread([], "", "", "")
    restored_by_worker = worker._get_current_prompts(iterations)
    prompt_files = {}
    for name in ("style-iter-reconcile.md", "style-iter-refine.md", "style-iter-local-extract.md",
                 "style-iter-local-merge.md", "style-iter-final-review.md", "style-iter-variants.md"):
        prompt_files[name] = text_sha256(_read_prompt(name))
    return {
        "source_json": str(source_json),
        "source_sha256": file_sha256(source_json),
        "source_size": source_json.stat().st_size,
        "source_updated_at": state.get("updated_at"),
        "source_version": state.get("version"),
        "file_prefix": state.get("file_prefix"),
        "iterations": len(iterations),
        "completed_rounds": sorted({int(r) for r in numeric}),
        "last_numeric_round": last_round,
        "next_round": last_round + 1,
        "test_images_stages": stages,
        "candidates": len(candidates),
        "missing_candidates": missing_candidates,
        "dataset_images": len(images),
        "missing_images": missing_images,
        "test_prompt": (state.get("final_test_images") or {}).get("test_prompt"),
        "style_reference": (state.get("final_test_images") or {}).get("style_reference"),
        "style_reference_exists": os.path.isfile((state.get("final_test_images") or {}).get("style_reference") or ""),
        "final_art_style_prompts_len": len(final_prompts),
        "final_art_style_prompts_sha256": text_sha256(final_prompts),
        "worker_restored_prompts_len": len(restored_by_worker),
        "worker_restored_prompts_sha256": text_sha256(restored_by_worker),
        "restored_matches_final": restored_by_worker == final_prompts,
        "final_test_status": (state.get("final_test_images") or {}).get("status"),
        "baseline_comparison": {
            "version": (state.get("automatic_comparison") or {}).get("version"),
            "best_id": (state.get("automatic_comparison") or {}).get("best_id"),
            "rows": len((state.get("automatic_comparison") or {}).get("rows") or []),
        },
        "deep_feature_comparison": bool(state.get("deep_feature_comparison")),
        "planned_rounds": list(range(last_round + 1, last_round + 1 + int(total_rounds))),
        "images_per_round": int(images_per_round),
        "missing_prompt_files": get_style_analyzer_missing_prompt_files(),
        "prompt_file_sha256": prompt_files,
        "checked_at": now(),
        "runner": RUNNER_ID,
    }


def _read_prompt(name):
    from utils.prompt_loader import read_prompt_file
    return read_prompt_file(name)


def cmd_preflight(args):
    info = collect_preflight(args.source_json, args.total_rounds, args.images_per_round)
    text = json.dumps(info, ensure_ascii=False, indent=2)
    if args.out:
        write_json(args.out, info)
    print(text, flush=True)
    problems = []
    if info["missing_images"]:
        problems.append(f"源图缺失 {len(info['missing_images'])} 张")
    if info["missing_candidates"]:
        problems.append(f"候选图缺失 {len(info['missing_candidates'])} 个")
    if not info["restored_matches_final"]:
        problems.append("_get_current_prompts 恢复正文与 final_art_style_prompts 不一致")
    if info["missing_prompt_files"]:
        problems.append(f"提示词模板缺失：{info['missing_prompt_files']}")
    print("[preflight] " + ("存在问题: " + "; ".join(problems) if problems else "全部检查通过"),
          flush=True)
    return 0


# --------------------------------------------------------------------------- #
# resume
# --------------------------------------------------------------------------- #

def cmd_resume(args):
    guard_test_isolation()
    source_json = Path(args.source_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"{args.file_prefix}_style_iter_result.json"
    ensure_source_isolated(source_json, output_dir, state_path)

    baseline = read_json(source_json)
    baseline_final = copy.deepcopy(baseline.get("final_test_images") or {})
    baseline_comparison = copy.deepcopy(baseline.get("automatic_comparison") or {})
    baseline_dataset = copy.deepcopy(baseline.get("dataset_selection"))
    baseline_iterations = len(baseline.get("iterations") or [])
    baseline_prompts = str(baseline.get("final_art_style_prompts") or "")

    baseline_numeric = sorted({int(it["round"]) for it in (baseline.get("iterations") or [])
                               if isinstance(it.get("round"), (int, float))})
    last_round = max(baseline_numeric) if baseline_numeric else 0
    seed = copy.deepcopy(baseline)
    seed.pop("automatic_comparison", None)
    seed.pop("deep_feature_comparison", None)
    seed["resume_origin"] = {
        "mode": "resume_full_existing_state",
        "source_json": str(source_json),
        "source_sha256": file_sha256(source_json),
        "source_updated_at": baseline.get("updated_at"),
        "source_iterations": baseline_iterations,
        "source_completed_rounds": baseline_numeric,
        "source_final_art_style_prompts_sha256": text_sha256(baseline_prompts),
        "source_final_test_status": baseline_final.get("status"),
        "source_baseline_candidate": baseline_comparison.get("best_id"),
        "source_baseline_comparison_version": baseline_comparison.get("version"),
        "baseline_stage_key": "final_v1_baseline",
        "new_rounds": list(range(last_round + 1, last_round + 1 + int(args.total_rounds))),
        "requested_total_rounds": int(args.total_rounds),
        "images_per_round": int(args.images_per_round),
        "created_at": now(),
        "runner": RUNNER_ID,
    }

    write_json(output_dir / "resume-seed-state.json", seed)
    write_json(output_dir / "baseline-snapshots" / "final-test-images-baseline.json", baseline_final)
    write_json(output_dir / "baseline-snapshots" / "automatic-comparison-v1-baseline.json",
               baseline_comparison)

    preflight = collect_preflight(source_json, args.total_rounds, args.images_per_round)
    preflight.update({
        "output_dir": str(output_dir),
        "state_path": str(state_path),
        "started_at": now(),
        "test_generation": bool(args.test_gen),
        "text_timeout_seconds": int(args.text_timeout),
        "resume_origin": seed["resume_origin"],
    })
    write_json(output_dir / "resume-manifest.json", preflight)
    print(f"[resume] 源: {source_json}", flush=True)
    print(f"[resume] SHA-256: {preflight['source_sha256']}", flush=True)
    print(f"[resume] 已恢复基线正文 {len(baseline_prompts)} 字符，"
          f"一致性={preflight['restored_matches_final']}", flush=True)
    print(f"[resume] 新增轮次 {preflight['planned_rounds']}，每轮 {args.images_per_round} 张", flush=True)
    print(f"[resume] 输出: {output_dir}", flush=True)

    base_url, api_key, model, key_source = text_api_config()
    print(f"[resume] 文本端点 {base_url}，模型 {model}，key 来源 {key_source}", flush=True)

    from modules.image_analysis.style_analyzer import StyleIterativeWorkerThread
    worker = StyleIterativeWorkerThread(
        image_paths=[str(p) for p in (baseline.get("dataset") or {}).get("images") or []],
        api_key=api_key,
        base_url=base_url,
        model_name=model,
        total_rounds=int(args.total_rounds),
        images_per_round=int(args.images_per_round),
        existing_state=seed,
        output_dir=str(output_dir),
        timeout_seconds=int(args.text_timeout),
        enable_test_gen=bool(args.test_gen),
        test_prompt=args.test_prompt or (baseline_final.get("test_prompt") or ""),
        test_style_ref_path=args.test_ref or (baseline_final.get("style_reference") or ""),
        img_aspect_ratio="auto",
        file_prefix=args.file_prefix,
        seed_prompts="",
        dataset_selection=baseline_dataset,
    )

    log_path = output_dir / "resume-run.log"
    log_handle = open(log_path, "a", encoding="utf-8")
    finished = {"status": None, "text": "", "path": ""}

    def log(message):
        line = str(message)
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    worker.log_signal.connect(log)
    worker.progress_signal.connect(lambda text: print(f"[进度] {text}", flush=True))
    worker.finish_signal.connect(lambda status, text, path: finished.update(
        status=status, text=text, path=path))
    log_handle.write(f"\n===== resume run 开始 {now()} =====\n")

    worker.run()  # 直接同步调用同一业务线程的 run()，不修改 App 代码
    log_handle.write(f"===== resume run 结束 {finished['status']} {now()} =====\n")
    log_handle.close()
    print(f"[resume] 线程结束状态: {finished['status']}", flush=True)

    if not state_path.is_file():
        fail("续跑未产生状态 JSON，后续步骤中止", 3)

    result = read_json(state_path)
    new_iterations = result.get("iterations") or []
    relabeled = []
    if args.relabel_new_rounds:
        last_round = max([int(it["round"]) for it in new_iterations
                          if isinstance(it.get("round"), (int, float))] or [args.total_rounds])
        prefix = f"round-{int(args.total_rounds)}-"
        for index, item in enumerate(new_iterations):
            if index < baseline_iterations:
                continue
            label = item.get("round")
            if isinstance(label, str) and label.startswith(prefix):
                item["round"] = f"round-{last_round}-" + label[len(prefix):]
                relabeled.append({"step": item.get("step"), "from": label, "to": item["round"]})

    # 旧终审作为独立 stage 纳入比较，绝不让新的 final 覆盖它。
    result.setdefault("test_images", {})
    result["test_images"]["final_v1_baseline"] = copy.deepcopy(baseline_final)
    result["final_test_images_baseline"] = copy.deepcopy(baseline_final)
    if baseline_comparison:
        result["automatic_comparison_v1_baseline"] = copy.deepcopy(baseline_comparison)
    result["resume_provenance"] = {
        **(result.get("resume_origin") or {}),
        "baseline_iterations": baseline_iterations,
        "new_iterations": max(0, len(new_iterations) - baseline_iterations),
        "relabeled_iterations": relabeled,
        "baseline_prompt_sha256": text_sha256(baseline_prompts),
        "introduced_keys": ["test_images.final_v1_baseline", "final_test_images_baseline",
                            "automatic_comparison_v1_baseline"],
        "finished_at": now(),
        "thread_status": finished["status"],
        "log": str(log_path),
    }
    result["resume_origin"] = seed["resume_origin"]
    write_json(state_path, result)

    summary = {
        "state_path": str(state_path),
        "thread_status": finished["status"],
        "iterations_total": len(new_iterations),
        "iterations_new": max(0, len(new_iterations) - baseline_iterations),
        "test_images_stages": sorted((result.get("test_images") or {}).keys()),
        "final_status": (result.get("final_test_images") or {}).get("status"),
        "final_channel_status": (result.get("final_test_images") or {}).get("channel_status"),
        "new_final_art_style_prompts_len": len(str(result.get("final_art_style_prompts") or "")),
        "relabeled_iterations": relabeled,
        "finished_at": now(),
    }
    write_json(output_dir / "resume-summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if finished["status"] == "success" else 4


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #

def _drive_comparison(state_path, timeout):
    from modules.image_analysis.style_comparison import StyleComparisonWorker
    base_url, api_key, model, key_source = text_api_config()
    print(f"[compare] 端点 {base_url}，模型 {model}，key 来源 {key_source}", flush=True)
    outcome = {"result": {}, "error": ""}

    def store(result, error):
        outcome["result"] = result or {}
        outcome["error"] = error or ""

    worker = StyleComparisonWorker(str(state_path), (base_url, api_key, str(model)), timeout=timeout)
    worker.completed.connect(store)
    worker.run()
    return outcome["result"], outcome["error"]


def _write_batch_state(state, batch_id, candidates, directory):
    from modules.image_analysis.style_comparison import candidates_from_state
    subset = {key: copy.deepcopy(value) for key, value in state.items()
              if key not in ("test_images", "final_test_images")}
    records = {}
    for candidate in candidates:
        stage, channel = candidate["stage"], candidate["channel"]
        if stage == "final":
            records.setdefault("__final__", {})
            entry = records["__final__"]
        else:
            entry = records.setdefault(stage, {})
        entry.setdefault("generated_files", {}).setdefault(channel, []).append(candidate["path"])
        for key, value in candidate["record"].items():
            if key != "generated_files":
                entry.setdefault(key, copy.deepcopy(value))
    final_record = records.pop("__final__", None)
    subset["test_images"] = records
    if final_record:
        subset["final_test_images"] = final_record
    subset.pop("automatic_comparison", None)
    subset.pop("deep_feature_comparison", None)
    path = Path(directory) / f"{batch_id}.json"
    write_json(path, subset)
    return path, candidates_from_state(subset)


def cmd_compare(args):
    guard_test_isolation()
    from modules.image_analysis.style_comparison import (
        calculate_ranking, comparison_inputs, write_comparison_report)
    state_path = Path(args.state).resolve()
    state = read_json(state_path)
    candidates = candidate_paths(state)
    print(f"[compare] 候选 {len(candidates)} 个，v2 量表，源图 {len((state.get('dataset') or {}).get('images') or [])} 张",
          flush=True)
    result, error = _drive_comparison(state_path, args.timeout)
    batches = []
    if error or not result.get("rows"):
        print(f"[compare] 单次比较未完成：{error or '空结果'}", flush=True)
        if not args.batch_size:
            fail(f"比较失败且未指定 --batch-size：{error}", 5)
        print(f"[compare] 改为按 --batch-size {args.batch_size} 分组重评，"
              f"同一模型/同一量表，分组代价写进结果", flush=True)
        groups = [candidates[index:index + args.batch_size]
                  for index in range(0, len(candidates), args.batch_size)]
        merged = []
        batch_dir = state_path.parent / "comparison-batches"
        batch_dir.mkdir(exist_ok=True)
        for index, group in enumerate(groups, start=1):
            batch_id = "batch-%s" % chr(ord("a") + index - 1)
            batch_path, batch_candidates = _write_batch_state(state, batch_id, group, batch_dir)
            if [c["id"] for c in batch_candidates] != [c["id"] for c in group]:
                fail(f"{batch_id} 的候选 ID 与全量集合不一致，拒绝合并来自不同配对的结果", 6)
            batch_result, batch_error = _drive_comparison(batch_path, args.timeout)
            batch_info = {"batch": batch_id, "state": str(batch_path),
                          "candidate_ids": [c["id"] for c in batch_candidates],
                          "status": "failed" if batch_error else "success",
                          "error": batch_error, "input_hash": batch_result.get("input_hash"),
                          "model": batch_result.get("model"), "endpoint": batch_result.get("endpoint")}
            batches.append(batch_info)
            if batch_error:
                print(f"[compare] {batch_id} 失败：{batch_error}", flush=True)
                continue
            merged.extend(batch_result.get("assessments") or [])
            print(f"[compare] {batch_id} 完成 {len(batch_result.get('rows') or [])} 个候选", flush=True)
        if len(merged) != len(candidates):
            fail(f"分组重评未覆盖全部候选：{len(merged)}/{len(candidates)}", 6)
        base_url, _, model, _ = text_api_config()
        _, _, _, digest = comparison_inputs(state, model, base_url)
        result = calculate_ranking(candidates, merged)
        result.update(input_hash=digest, assessments=merged, model=model, endpoint=base_url,
                      cached=False, batched=True, batches=batches)
        result["formula"] = result.get("formula", "")
    latest = read_json(state_path)
    latest["automatic_comparison"] = result
    result["report_path"] = write_comparison_report(latest, result, str(state_path.parent))
    write_json(state_path, latest)
    write_json(state_path.parent / "automatic-comparison.json", result)
    summary = {
        "version": result.get("version"), "best_id": result.get("best_id"),
        "eligible": [row["id"] for row in result.get("rows", []) if row.get("eligible")],
        "rows": len(result.get("rows") or []), "batched": bool(result.get("batched")),
        "cached": bool(result.get("cached")), "input_hash": result.get("input_hash"),
        "model": result.get("model"),
        "endpoint": result.get("endpoint"), "report_path": result.get("report_path"),
        "top": [{"id": row["id"], "total": row["total"], "style_score": row["style_score"],
                 "subject_fidelity": row["subject_fidelity"], "eligible": row["eligible"],
                 "gates": row["gates"]} for row in (result.get("rows") or [])],
    }
    write_json(state_path.parent / "comparison-summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("version", "best_id", "rows", "batched", "report_path")},
                     ensure_ascii=False, indent=2), flush=True)
    for row in summary["top"]:
        print(f"  {row['id']:28s} total={row['total']:<6} style={row['style_score']:<6} "
              f"subject={row['subject_fidelity']:<5} eligible={row['eligible']}", flush=True)
    return 0


def cmd_compare_subset(args):
    """补充比较：只取部分候选（例如全部 gemini_direct）单独一次评分。

    与主比较集分开存放，分数不跨集合混排；用于检查候选较多时模型是否只按通道给分。
    """
    guard_test_isolation()
    state_path = Path(args.state).resolve()
    state = read_json(state_path)
    candidates = [c for c in candidate_paths(state) if c["channel"] == args.channel]
    if not candidates:
        fail(f"没有匹配的候选：channel={args.channel}")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    subset_path, subset_candidates = _write_batch_state(state, args.label, candidates, out_dir)
    if [c["id"] for c in subset_candidates] != [c["id"] for c in candidates]:
        fail("子集候选 ID 与全量集合不一致", 6)
    print(f"[compare-subset] {args.label}: {len(candidates)} 个候选，"
          f"{len((state.get('dataset') or {}).get('images') or [])} 张源图", flush=True)
    result, error = _drive_comparison(subset_path, args.timeout)
    if error:
        fail(f"补充比较失败：{error}", 5)
    summary = {
        "label": args.label,
        "subset_state": str(subset_path),
        "channel": args.channel,
        "cached": bool(result.get("cached")),
        "version": result.get("version"),
        "best_id": result.get("best_id"),
        "input_hash": result.get("input_hash"),
        "report_path": result.get("report_path"),
        "rows": [{"id": row["id"], "total": row["total"], "style_score": row["style_score"],
                  "subject_fidelity": row["subject_fidelity"], "eligible": row["eligible"],
                  "gates": row["gates"], "group_scores": row.get("group_scores")}
                 for row in result.get("rows") or []],
    }
    write_json(out_dir / f"{args.label}-summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("label", "best_id", "cached", "report_path")},
                     ensure_ascii=False, indent=2), flush=True)
    for row in summary["rows"]:
        print(f"  {row['id']:28s} total={row['total']:<6} style={row['style_score']:<6} "
              f"subject={row['subject_fidelity']:<5} eligible={row['eligible']}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# deep（Intel NPU；不自动转 CUDA）
# --------------------------------------------------------------------------- #

def cmd_deep(args):
    guard_test_isolation()
    state_path = Path(args.state).resolve()
    log_path = state_path.parent / "deep-feature-comparison.log"
    python = Path(sys.executable)
    command = [str(python), "-u", str(ROOT / "tools" / "style_metrics_verify.py"),
               "--comparison-state", str(state_path), "--device", args.device]
    print("[deep] " + " ".join(command), flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=str(ROOT), env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            log.write(line)
            log.flush()
            if line.startswith("PROGRESS"):
                print(line.rstrip(), flush=True)
        process.wait()
    print(f"[deep] 退出码 {process.returncode}；日志 {log_path}", flush=True)
    if process.returncode != 0:
        print((log_path.read_text(encoding="utf-8", errors="replace")[-3000:]), flush=True)
        return process.returncode
    state = read_json(state_path)
    deep = state.get("deep_feature_comparison") or {}
    print(json.dumps({"status": deep.get("status"), "backend": deep.get("backend"),
                      "rows": len(deep.get("rows") or []),
                      "report_path": deep.get("report_path"),
                      "engine_json": deep.get("engine_json"),
                      "self_check": deep.get("self_check")}, ensure_ascii=False, indent=2), flush=True)
    return 0


# --------------------------------------------------------------------------- #
# entries（待审查 style_entry 候选，不写画风配置）
# --------------------------------------------------------------------------- #

def cmd_entries(args):
    state_path = Path(args.state).resolve()
    state = read_json(state_path)
    source = (state.get("resume_origin") or {}).get("source_json")
    baseline = read_json(source) if source and os.path.isfile(source) else {}
    comparison = state.get("automatic_comparison") or {}
    deep = state.get("deep_feature_comparison") or {}
    rows = {row["id"]: row for row in comparison.get("rows") or []}
    deep_rows = {row["id"]: row["summary"] for row in deep.get("rows") or []}
    records = dict(state.get("test_images") or {})
    records["final"] = state.get("final_test_images") or {}
    baseline_variants = baseline.get("prompt_variants") or {}
    baseline_final = baseline.get("final_test_images") or {}

    entries = []

    def add(stage, label, variants, prompts_used, channel_id, origin):
        if not variants:
            return
        style_entry = variants.get("style_entry") or {}
        candidates = sorted((cid for cid in rows if cid.startswith(stage + "/")),
                            key=lambda cid: (0 if "gemini_direct" in cid else 1 if "gpt_first_pass" in cid else 2))
        best_row = None
        for cid in candidates:
            if rows[cid].get("eligible"):
                best_row = cid
                break
        chosen = channel_id or (best_row or (candidates[0] if candidates else None))
        row = rows.get(chosen) if chosen else None
        entry = {
            "stage": stage,
            "label": label,
            "origin": origin,
            "candidate_id": chosen,
            "master_prompt_len": len(str(prompts_used or "")),
            "master_prompt_sha256": text_sha256(prompts_used),
            "gpt_image_prompt_len": len(str(style_entry.get("prompt_gpt") or "")),
            "gpt_image_prompt_valid": variants.get("gpt_image_prompt_valid"),
            "gpt_image_prompt_errors": variants.get("gpt_image_prompt_errors"),
            "repaint_clauses": len(style_entry.get("repaint_clauses") or []),
            "face_hair_clauses": len(variants.get("face_hair_clauses") or []),
            "motif_clauses": len(style_entry.get("motif_clauses") or []),
            "vision": ({key: row.get(key) for key in
                        ("total", "style_score", "subject_fidelity", "eligible", "gates", "reason",
                         "group_scores")} if row else None),
            "deep": deep_rows.get(chosen) if chosen else None,
            "style_entry": style_entry,
        }
        entries.append(entry)

    add("final", "新增终审（第 4–6 轮后）", state.get("prompt_variants"),
        state.get("final_art_style_prompts"), None, "本次新增终审")
    for stage in sorted(k for k in records if k.startswith("round_")):
        number = int(stage.split("_")[1])
        if number < 4:
            continue
        record = records[stage]
        add(stage, f"新增第 {number} 轮", record.get("prompt_variants"), record.get("prompts_used"),
            None, "本次新增轮次")
    records_baseline_final = records.get("final_v1_baseline") or baseline_final
    add("final_v1_baseline", "旧终审基线（第 1–3 轮后）", records_baseline_final.get("prompt_variants"),
        records_baseline_final.get("prompts_used"), None, "基线快照")
    for stage in sorted(k for k in records if k.startswith("round_")):
        number = int(stage.split("_")[1])
        if number > 3:
            continue
        record = records[stage]
        add(stage, f"基线第 {number} 轮", record.get("prompt_variants"), record.get("prompts_used"),
            None, "基线快照")
    if baseline_variants:
        add("baseline_final_alt", "基线终审（原 JSON 的 prompt_variants，仅提示词）",
            baseline_variants, baseline.get("final_art_style_prompts"), None, "基线原记录")

    payload = {
        "created_at": now(),
        "state_json": str(state_path),
        "state_sha256": file_sha256(state_path),
        "comparison_version": comparison.get("version"),
        "comparison_best_id": comparison.get("best_id"),
        "deep_metrics_status": deep.get("status"),
        "deep_metrics_backend": deep.get("backend"),
        "applied_to_style_config": False,
        "note": ("仅生成待人工复核的候选 style_entry；未写入 conf/config-styles.json，"
                 "未覆盖已安装的 taya_oco，未发布图片。"),
        "entries": entries,
    }
    write_json(args.out, payload)
    print(json.dumps([{k: e[k] for k in ("stage", "candidate_id", "gpt_image_prompt_len",
                                         "gpt_image_prompt_valid")} for e in entries],
                     ensure_ascii=False, indent=2), flush=True)
    return 0


# --------------------------------------------------------------------------- #

def build_parser():
    parser = argparse.ArgumentParser(description="taya_oco 画风提取续跑无头 runner")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("preflight")
    check.add_argument("--source-json", required=True)
    check.add_argument("--total-rounds", type=int, default=3)
    check.add_argument("--images-per-round", type=int, default=4)
    check.add_argument("--out")
    check.set_defaults(func=cmd_preflight)

    resume = sub.add_parser("resume")
    resume.add_argument("--source-json", required=True)
    resume.add_argument("--output-dir", required=True)
    resume.add_argument("--file-prefix", default="taya_oco")
    resume.add_argument("--total-rounds", type=int, default=3)
    resume.add_argument("--images-per-round", type=int, default=4)
    resume.add_argument("--test-prompt", default="")
    resume.add_argument("--test-ref", default="")
    resume.add_argument("--text-timeout", type=int, default=600)
    resume.add_argument("--no-test-gen", dest="test_gen", action="store_false")
    resume.add_argument("--relabel-new-rounds", action="store_true", default=True)
    resume.set_defaults(func=cmd_resume, test_gen=True)

    compare = sub.add_parser("compare")
    compare.add_argument("--state", required=True)
    compare.add_argument("--timeout", type=int, default=900)
    compare.add_argument("--batch-size", type=int, default=12,
                         help="单次调用失败后，按该数量分组重评（同一模型/量表）")
    compare.set_defaults(func=cmd_compare)

    subset = sub.add_parser("compare-subset")
    subset.add_argument("--state", required=True)
    subset.add_argument("--out-dir", required=True)
    subset.add_argument("--label", default="gemini-direct-only")
    subset.add_argument("--channel", default="gemini_direct")
    subset.add_argument("--timeout", type=int, default=900)
    subset.set_defaults(func=cmd_compare_subset)

    deep = sub.add_parser("deep")
    deep.add_argument("--state", required=True)
    deep.add_argument("--device", default="npu")
    deep.set_defaults(func=cmd_deep)

    entries = sub.add_parser("entries")
    entries.add_argument("--state", required=True)
    entries.add_argument("--out", required=True)
    entries.set_defaults(func=cmd_entries)
    return parser


def main(argv=None):
    os.chdir(ROOT)
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
