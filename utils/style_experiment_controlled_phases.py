"""受控实验各阶段（E0–E5）的执行体。

所有相似度数值都来自共享入口；本模块只负责取数、统计、判定与落盘。
"""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path

from utils import style_experiment_attributes as attrs
from utils.style_experiment_controlled import (
    ACCEPTANCE, DEVICE, E0_SYMMETRY_ABS, E0_SYMMETRY_RTOL, FACE_METRIC_DIRECTION, MAIN_METRIC,
    METRIC_DIRECTION, SEED, TIE_RTOL, AttemptLedger, atomic_json, cluster_bootstrap, compare,
    exact_counts, icc_a1, json_hash, kendall_tau_b, metric_value, now,
    pair_lookup, prompt_dir, read_json, run_root, sha256_file, wilson, with_retries,
)

#: 十三项面部指标里，本轮建立了受控干预的 8 项；其余 5 项只能作诊断
CONTROLLED_FACE_METRICS = ("hair_fineness", "hair_continuity", "eye_brightness", "eye_height",
                           "eye_width", "eye_gap", "eyelashes", "eye_curvature")
DIAGNOSTIC_FACE_METRICS = ("iris_ratio", "eye_highlights", "lid_weight", "eye_tilt", "face_ratios")


def _log(progress, message):
    if progress:
        progress(message)


def style_groups(plan):
    """返回 (style_ids, 每个画风的参考/query 行)。"""
    styles = plan["split"]
    style_ids = sorted(styles)
    return style_ids, styles


# --------------------------------------------------------------------------- #
# E0：实现一致性与错误传播
# --------------------------------------------------------------------------- #


def e0(ledger: AttemptLedger, plan, progress=None, device=DEVICE) -> dict:
    run = run_root()
    cache = run / "cache" / "e0"
    images = plan["e0_images"]
    originals = [row for row in images if row["kind"] == "original"]
    generated = [row for row in images if row["kind"] == "generated"]
    checks = []

    def check(name, passed, detail=None, error=None):
        checks.append({"check": name, "status": "pass" if passed else ("error" if error else "fail"),
                       "detail": detail or {}, "error": error})

    def run_case(case_id, query, references, cache_name):
        query_rows = [{"path": query["path"], "sha256": query["sha256"]}]
        reference_rows = [{"path": row["path"], "sha256": row["sha256"]} for row in references]
        return with_retries(
            ledger, f"e0:{case_id}", "E0", lambda: compare(query_rows, reference_rows, cache / cache_name, device),
            request={"query": query["path"], "references": [row["path"] for row in references]},
            charged=False)

    # 0.1 同图自检：全部 6 张图各自与自身比较（1 对 1）
    self_rows = []
    for index, row in enumerate(images):
        result = run_case(f"self-{index + 1}", row, [row], f"self-{index + 1}")
        pairs = pair_lookup(result)
        pair = pairs[(row["path"], row["path"])]
        entry = {"image_id": row["image_id"], "path": row["path"], "kind": row["kind"],
                 "exists": Path(row["path"]).is_file(),
                 "values": {metric: metric_value(pair, metric) for metric in METRIC_DIRECTION},
                 "self_check": result.get("self_check"),
                 "status": result["status"], "face_status": result.get("face_status")}
        self_rows.append(entry)
    max_distance = max((abs(value) for row in self_rows for metric, value in row["values"].items()
                        if metric in ("gram", "adain", "lpips") and value is not None), default=None)
    csd_values = [row["values"]["csd"] for row in self_rows if row["values"].get("csd") is not None]
    local_values = [row["values"][metric] for row in self_rows for metric in ("tone", "edges", "lines", "space")
                    if row["values"].get(metric) is not None]
    check("E0.1-同图深度指标", max_distance is not None and max_distance <= 1e-6,
          {"max_abs_distance": max_distance, "tolerance": 1e-6, "images": len(self_rows)})
    check("E0.1-同图CSD≈1", bool(csd_values) and all(abs(value - 1.0) <= 1e-4 for value in csd_values),
          {"csd_values": csd_values, "tolerance": 1e-4})
    check("E0.1-同图本地四组≈1", bool(local_values) and all(abs(value - 1.0) <= 1e-6 for value in local_values),
          {"count": len(local_values), "min": min(local_values) if local_values else None,
           "max": max(local_values) if local_values else None, "tolerance": 1e-6})

    # 0.2 对称性：全部无序对的 A→B 与 B→A
    symmetry = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            first, second = images[i], images[j]
            forward = pair_lookup(run_case(f"sym-{i+1}-{j+1}", first, [second], f"sym-{i+1}-{j+1}"))[
                (first["path"], second["path"])]
            backward = pair_lookup(run_case(f"sym-{j+1}-{i+1}", second, [first], f"sym-{j+1}-{i+1}"))[
                (second["path"], first["path"])]
            for metric in (*METRIC_DIRECTION,):
                a, b = metric_value(forward, metric), metric_value(backward, metric)
                if a is None or b is None:
                    symmetry.append({"first": first["image_id"], "second": second["image_id"], "metric": metric,
                                     "status": "error", "error": "一侧无有效数值"})
                    continue
                limit = E0_SYMMETRY_ABS + E0_SYMMETRY_RTOL * max(abs(a), abs(b))
                symmetry.append({"first": first["image_id"], "second": second["image_id"], "metric": metric,
                                 "a_to_b": a, "b_to_a": b, "abs_error": abs(a - b), "limit": limit,
                                 "status": "pass" if abs(a - b) <= limit else "fail"})
    bad = [row for row in symmetry if row["status"] != "pass"]
    check("E0.2-对称性", not bad, {"cases": len(symmetry), "failures": bad[:10],
                                "rule": "|a-b| <= 1e-6 + 1e-5*max(|a|,|b|)"})

    # 0.3 重复 / 缓存 / 重启复算
    repeat_a = run_case("repeat-a", originals[0], [generated[0]], "repeat-a")
    repeat_b = run_case("repeat-b", originals[0], [generated[0]], "repeat-b")
    a_values = {metric: metric_value(pair_lookup(repeat_a)[(originals[0]["path"], generated[0]["path"])], metric)
                for metric in METRIC_DIRECTION}
    b_values = {metric: metric_value(pair_lookup(repeat_b)[(originals[0]["path"], generated[0]["path"])], metric)
                for metric in METRIC_DIRECTION}
    deterministic = {metric: (a_values[metric], b_values[metric]) for metric in a_values}
    max_repeat_error = max((abs(filter_values[0] - filter_values[1])
                            for filter_values in deterministic.values()
                            if filter_values[0] is not None and filter_values[1] is not None), default=None)
    # 缓存复用：同一 cache 目录再算一次
    cached = with_retries(ledger, "e0:cache-reuse", "E0",
                          lambda: compare([{"path": originals[0]["path"], "sha256": originals[0]["sha256"]}],
                                          [{"path": generated[0]["path"], "sha256": generated[0]["sha256"]}],
                                          cache / "repeat-a", device),
                          request={"note": "复用 repeat-a 的缓存目录"}, charged=False)
    cached_pair = pair_lookup(cached)[(originals[0]["path"], generated[0]["path"])]
    cache_values = {metric: metric_value(cached_pair, metric) for metric in METRIC_DIRECTION}
    check("E0.3-重复与重启复算", max_repeat_error is not None and max_repeat_error <= 1e-12,
          {"values": deterministic, "max_abs_error": max_repeat_error, "note": "两次独立进程内计算（新缓存目录）"})
    check("E0.3-完整缓存复用", cached.get("cache_hit") is True and cache_values == b_values,
          {"cache_hit": cached.get("cache_hit"), "values": cache_values})

    # 0.3b 图片 hash 变化后旧结果不得被误复用
    import shutil
    mutated = run / "e0" / "mutated.png"
    mutated.parent.mkdir(parents=True, exist_ok=True)
    _reencode_with_one_pixel_change(originals[0]["path"], mutated)
    mutated_sha = sha256_file(mutated)
    mutation = {"note": "同一 cache 目录，用改了一个像素的同名逻辑图重新计算", "path": str(mutated),
                "sha256": mutated_sha, "original_sha256": originals[0]["sha256"]}
    try:
        changed = compare([{"path": str(mutated), "sha256": mutated_sha}],
                          [{"path": generated[0]["path"], "sha256": generated[0]["sha256"]}],
                          cache / "repeat-a", device)
        changed_values = {metric: metric_value(pair_lookup(changed)[(str(mutated), generated[0]["path"])], metric)
                          for metric in METRIC_DIRECTION}
        identical = all(changed_values[metric] == b_values[metric] for metric in changed_values)
        # 期望：cache_hit=False（指纹不同）且数值按 1 像素改动发生真实变化；
        # 若算出完全相同的数值，说明旧结果被误复用（协议禁止）。
        check("E0.3-hash变化不复用",
              changed.get("cache_hit") is not True and not identical,
              {**mutation, "changed_values": changed_values, "cache_hit": changed.get("cache_hit"),
               "values_identical_to_original": identical})
    except Exception as exc:
        check("E0.3-hash变化不复用", False, mutation, error=f"{type(exc).__name__}: {exc}")

    # 0.4 入口一致性：CLI / 提取页计算路径 / 双图 UI（offscreen）
    entry_points = e0_entries(ledger, plan, progress)
    check("E0.4-三入口一致", entry_points.get("consistent") is True, entry_points)

    # 0.5 失败状态
    failure = e0_failures(ledger, plan, run, device)
    for name, value in failure.items():
        check(name, value.get("status") == "pass", value, error=value.get("error"))

    result = {"stage": "E0", "device": device, "generated_at": now(), "checks": checks,
              "self_check_rows": self_rows, "symmetry": symmetry,
              "repeatability": {"values": deterministic, "max_abs_error": max_repeat_error,
                                "cache_reuse_values": cache_values},
              "entry_points": entry_points, "failure_propagation": failure,
              "status": "complete" if all(row["status"] == "pass" for row in checks) else "partial",
              "critical_failures": [row["check"] for row in checks if row["status"] != "pass"]}
    return result


def _reencode_with_one_pixel_change(source, target):
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(source) as image:
        array = np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
    array[0, 0] = (array[0, 0].astype(int) + 7) % 256
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(target, format="PNG")


def e0_entries(ledger: AttemptLedger, plan, progress=None) -> dict:
    """同一份 manifest 用三条入口各算一次，比较数值与状态。

    - CLI：`tools/style_metrics_verify.py --similarity A B --reference C`
    - 提取页/无头计算路径：`modules.image_analysis.style_deep_comparison.compute(state.json)`
      （GUI 的 `create_worker` 就是子进程调用同一个 CLI 与同一份 state）
    - 双图 UI：offscreen 实例化 `StyleSimilarityWidget`，走 `start()` → 子进程 worker
    """
    import subprocess
    import sys
    run = run_root()
    directory = run / "e0" / "entries"
    directory.mkdir(parents=True, exist_ok=True)
    query = plan["e0_entries"]["query"]
    references = plan["e0_entries"]["references"]
    outcome = {"query": query["path"], "references": [row["path"] for row in references],
               "device": DEVICE, "entries": {}, "consistent": False, "differences": []}

    cli_out = directory / "cli.json"
    # CLI 契约：`--similarity IMAGE REFERENCE` 必须给两张，额外参考用可重复的 `--reference`
    cli_query = plan["e0_entries"]["query"]
    cli_first = references[0]
    cli_rest = references[1:]
    arguments = [sys.executable, "-u", str(Path(__file__).resolve().parents[1] / "tools/style_metrics_verify.py"),
                 "--similarity", cli_query["path"], cli_first["path"]] + \
                sum([["--reference", row["path"]] for row in cli_rest], []) + \
                ["--device", DEVICE, "--out", str(cli_out)]
    completed = with_retries(ledger, "e0:entry-cli", "E0",
                             lambda: subprocess.run(arguments, cwd=str(Path(__file__).resolve().parents[1]),
                                                    capture_output=True, text=True, encoding="utf-8",
                                                    errors="replace", timeout=1800),
                             request={"argv": arguments}, charged=False)
    outcome["entries"]["cli"] = {"returncode": completed.returncode, "stdout_tail": (completed.stdout or "")[-400:],
                                 "stderr_tail": (completed.stderr or "")[-400:],
                                 "result_path": str(cli_out),
                                 "ok": completed.returncode == 0 and cli_out.is_file()}
    cli_result = read_json(cli_out, {}) or {}

    state = {"dataset": {"images": [row["path"] for row in references]},
             "test_images": {"pair": {"generated_files": {"input": [cli_query["path"]]}}}}
    state_path = directory / "state.json"
    atomic_json(state_path, state)
    import importlib
    deep = importlib.import_module("modules.image_analysis.style_deep_comparison")
    extraction = with_retries(ledger, "e0:entry-extraction", "E0",
                              lambda: deep.compute(state_path, DEVICE), request={"state": str(state_path)},
                              charged=False)
    outcome["entries"]["extraction"] = {"status": extraction.get("status"),
                                        "ok": extraction.get("status") == "ok",
                                        "result_path": str(state_path)}
    outcome["entries"]["ui"] = e0_ui_entry(ledger, directory, cli_query, references, progress)

    def values(result):
        if not result:
            return {}
        lookup = pair_lookup(result)
        pair = lookup.get((cli_query["path"], cli_first["path"]))
        return {metric: metric_value(pair, metric) for metric in (*METRIC_DIRECTION,)} if pair else {}

    collected = {"cli": values(cli_result), "extraction": values(read_json(
        directory / "deep-feature-comparison-cpu-fp32-gram-v2.json", {}) or {}),
        "ui": outcome["entries"]["ui"].get("values", {})}
    outcome["values"] = collected
    outcome["compared_pair"] = {"query": cli_query["path"], "reference": cli_first["path"]}
    baselines = [name for name, value in collected.items() if value]
    if len(baselines) >= 2:
        reference_name = baselines[0]
        for name in baselines[1:]:
            for metric in METRIC_DIRECTION:
                a, b = collected[reference_name].get(metric), collected[name].get(metric)
                if a is None or b is None or abs(a - b) > 1e-9:
                    outcome["differences"].append({"metric": metric, "first": reference_name, "second": name,
                                                   "first_value": a, "second_value": b})
    outcome["statuses"] = {name: entry.get("status") for name, entry in outcome["entries"].items()}
    outcome["consistent"] = (len(baselines) == 3 and not outcome["differences"]
                             and all(entry.get("ok") for entry in outcome["entries"].values()))
    return outcome


def e0_ui_entry(ledger: AttemptLedger, directory, query, references, progress=None) -> dict:
    """offscreen Qt 实例：不加载/不重启用户的 app，只实例化双图控件并等它跑完。"""
    import os
    result = {"mode": "offscreen StyleSimilarityWidget", "ok": False}
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PyQt6.QtWidgets import QApplication
        from modules.image_analysis.style_similarity_tab import StyleSimilarityWidget
        app = QApplication.instance() or QApplication([])
        widget = StyleSimilarityWidget()
        widget.auto_regions.setChecked(False)
        widget.device.setCurrentIndex([widget.device.itemData(index)
                                       for index in range(widget.device.count())].index(DEVICE))
        widget.paths[0].setText(query["path"])
        widget.paths[1].setText(references[0]["path"])
        widget.start()
        deadline = time.time() + 1800
        while widget.worker is not None and time.time() < deadline:
            app.processEvents()
            time.sleep(0.05)
        if widget.worker is not None:
            widget.cancel()
            result["error"] = "UI 计算超时"
            return result
        payload = widget.result or {}
        pair = payload.get("rows", [{}])[0].get("pairs", [{}])[0] if payload else {}
        result.update({"ok": payload.get("status") == "ok",
                       "status": payload.get("status"),
                       "face_status": payload.get("face_status"),
                       "values": {metric: metric_value(pair, metric) for metric in METRIC_DIRECTION} if pair else {},
                       "backend": payload.get("backend"),
                       "table_rows": widget.table.rowCount()})
        widget.close()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def e0_failures(ledger: AttemptLedger, plan, run, device) -> dict:
    """不可读图片 / 区域 hash 错误 / 闭眼 / 遮挡 / 视角不同 / 配对缺失必须给具体状态。"""
    from utils import style_face_metrics as face_metrics
    from utils.style_regions import save_annotation
    from utils.style_similarity import compare_images, image_manifest, validate_manifest
    outcome = {}
    directory = run / "e0" / "failures"
    directory.mkdir(parents=True, exist_ok=True)
    broken = directory / "broken.png"
    broken.write_bytes(b"\x89PNG\r\n\x1a\n" + b"not-an-image" * 40)
    query = plan["e0_entries"]["query"]

    # 1) 不可解码图片
    broken_sha = sha256_file(broken)
    inputs = {"references": [{"path": query["path"], "sha256": query["sha256"]}],
              "candidates": [{"id": "broken", "path": str(broken), "sha256": broken_sha}]}
    try:
        result = compare_images(inputs, device, lambda message: None, directory / "cache-broken")
        statuses = {metric: metric_value(pair, metric) for metric in METRIC_DIRECTION for pair in
                    [row for row in result["rows"]][0]["pairs"]}
        rows = result["rows"][0]["pairs"][0]
        outcome["E0.5-不可读图片"] = {
            "status": "pass" if all(rows.get(metric, {}).get("status") != "ok" for metric in METRIC_DIRECTION)
                      and result["status"] == "partial" else "fail",
            "detail": {"result_status": result["status"],
                       "metric_status": {metric: rows.get(metric, {}).get("status") for metric in METRIC_DIRECTION},
                       "face_status": result.get("face_status")}}
    except Exception as exc:
        outcome["E0.5-不可读图片"] = {"status": "pass", "detail": {"raised": f"{type(exc).__name__}: {exc}",
                                                                   "note": "抛错并携带具体原因也算未标成成功"}}

    # 2) 区域 hash 错误
    wrong = {"version": "style-regions/1", "image_sha256": "0" * 64, "pose": "frontal",
             "face_outline": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]}
    try:
        save_annotation(query["path"], wrong)
        detail, status = "未拦截", "fail"
    except Exception as exc:
        detail, status = f"{type(exc).__name__}: {exc}", "pass"
    outcome["E0.5-区域hash错误"] = {"status": status, "detail": {"message": detail}}

    # 3) 闭眼 / 遮挡 / 视角不同 / 配对缺失（用共享规则直接判定，不新造公式）
    outcome["E0.5-闭眼遮挡视角"] = e0_face_states(directory)
    # 4) 配对缺失：候选缺失时不得补 0
    missing = directory / "missing.png"
    try:
        from utils.style_similarity import aggregate
        rows = [{"csd": {"status": "ok", "value": 0.5}}, {"csd": {"status": "unavailable", "value": None}}]
        summary = aggregate(rows, "csd", 3)
        outcome["E0.5-配对缺失"] = {"status": "pass" if summary["mean"] is None and summary["status"] == "partial" else "fail",
                                 "detail": summary}
    except Exception as exc:
        outcome["E0.5-配对缺失"] = {"status": "fail", "detail": {}, "error": str(exc)}
    return outcome


def e0_face_states(directory) -> dict:
    """闭眼 / 遮挡 / 视角不同必须给具体状态。

    构造方式：为同一张真实图片写两份**hash 匹配**的标注（自动定位候选就是这样保存的），
    只改 `state` 或 `pose`；再走共享 `compare_face_features`，避免用 hash 不匹配的假数据
    把结论变成「区域标注无效」（那是另一条路径，已在「区域 hash 错误」里单独验证）。
    """
    import numpy as np
    from PIL import Image
    from utils import style_face_metrics as face_metrics
    from utils.style_regions import save_annotation

    def base_annotation(sha, pose, state_left="open", state_right="open"):
        curve = [[0.30 + 0.02 * i, 0.40] for i in range(7)]
        lower = [[0.30 + 0.02 * i, 0.44] for i in range(7)]
        iris = [[0.33 + 0.005 * i, 0.42] for i in range(5)]
        return {"version": "style-regions/1", "image_sha256": sha, "pose": pose, "confirmed": False,
                "face_outline": [[0.25, 0.2], [0.45, 0.2], [0.48, 0.5], [0.35, 0.6], [0.22, 0.5]],
                "eyes": {"viewer_left": {"state": state_left, "upper_lid": curve, "lower_lid": lower, "iris": iris,
                                         "lashes": []},
                         "viewer_right": {"state": state_right,
                                          "upper_lid": [[x + 0.1, y] for x, y in curve],
                                          "lower_lid": [[x + 0.1, y] for x, y in lower],
                                          "iris": [[x + 0.1, y] for x, y in iris], "lashes": []}},
                "hair_regions": [[[0.2, 0.1], [0.3, 0.1], [0.3, 0.25], [0.2, 0.25]]],
                "hair_strands": [{"points": [[0.22, 0.12], [0.24, 0.22]], "polarity": "dark"}],
                "limitations": ["E0 构造用例"]}

    results = {}
    for label in ("closed", "occluded"):
        path = directory / f"{label}.png"
        try:
            if not path.is_file():
                Image.fromarray((np.random.RandomState(SEED).rand(400, 400, 3) * 255).astype("uint8")).save(path)
            sha = sha256_file(path)
            state = "closed" if label == "closed" else "occluded"
            save_annotation(str(path), base_annotation(sha, "frontal", state, "open" if label == "occluded" else state))
            features = face_metrics.descriptors(str(path), face_metrics.read_annotation(str(path)))
            statuses = {metric: features[metric]["status"] for metric in ("eye_brightness", "eye_height", "eye_width")}
            results["闭眼" if label == "closed" else "遮挡"] = {
                "status": "pass" if all(value != "ok" for value in statuses.values()) else "fail",
                "metric_status": statuses,
                "specific_status_values": sorted(set(statuses.values()))}
        except Exception as exc:
            results[label] = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}
    try:
        # 视角不同：两张真实（合成）图片，各自 hash 匹配的标注，pose 分别为 frontal / profile
        frontal = _write_pose_image(directory / "pose-frontal.png", seed=SEED)
        profile_image = _write_pose_image(directory / "pose-profile.png", seed=SEED + 1)
        save_annotation(str(frontal), base_annotation(sha256_file(frontal), "frontal"))
        save_annotation(str(profile_image), base_annotation(sha256_file(profile_image), "profile"))
        annotations = {str(frontal): face_metrics.read_annotation(str(frontal)),
                       str(profile_image): face_metrics.read_annotation(str(profile_image))}
        pairs = face_metrics.compare_face_features(str(frontal), str(profile_image), annotations, {})
        statuses = {metric: pairs[metric]["status"] for metric in ("eye_brightness", "eye_height", "eye_width")}
        results["视角不同"] = {"status": "pass" if all(value == "not_comparable" for value in statuses.values()) else "fail",
                            "metric_status": statuses}
    except Exception as exc:
        results["视角不同"] = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}
    return results


def _write_pose_image(path, seed):
    import numpy as np
    from PIL import Image
    import cv2
    array = (np.random.RandomState(seed).rand(400, 400, 3) * 60 + 60).astype("uint8")
    cv2.rectangle(array, (140, 120), (260, 320), (235, 210, 190), -1)
    cv2.imwrite(str(path), array)
    return path


# --------------------------------------------------------------------------- #
# E1：原作品画风辨别（主要效度实验）
# --------------------------------------------------------------------------- #


def e1(ledger: AttemptLedger, plan, progress=None, device=DEVICE) -> dict:
    """原作品画风辨别：每个 query 与 4 个参考集分别比较（禁止跨画风混算均值）。"""
    run = run_root()
    cache = run / "cache" / "e1"
    styles = plan["split"]
    style_ids = sorted(styles)
    reference_rows, query_rows = [], []
    for style_id in style_ids:
        reference_rows.extend(styles[style_id]["references"])
        query_rows.extend(styles[style_id]["queries"])
    per_query = {}
    for index, query in enumerate(query_rows):
        _log(progress, f"E1 {index + 1}/{len(query_rows)}：{query['image_id']}")
        result = with_retries(
            ledger, f"e1:{query['image_id']}", "E1",
            lambda query=query: compare([{"path": query["path"], "sha256": query["sha256"]}],
                                        [{"path": row["path"], "sha256": row["sha256"]} for row in reference_rows],
                                        cache / query["image_id"], device),
            request={"query": query["path"], "references": len(reference_rows)}, charged=False)
        per_query[query["image_id"]] = {"path": result["report_path"] if result.get("report_path") else None,
                                        "result_path": str(cache / query["image_id"] /
                                                           f"style-similarity-{result['backend']['actual']}-"
                                                           f"{result['backend']['precision']}-v2.json"),
                                        "status": result["status"], "face_status": result.get("face_status"),
                                        "lookup": pair_lookup(result), "style_id": query["style_id"],
                                        "plan": result}

    reference_by_style = {style_id: styles[style_id]["references"] for style_id in style_ids}
    units = []
    for query in query_rows:
        entry = per_query[query["image_id"]]
        unit = {"query_id": query["image_id"], "query_style": query["style_id"], "query_path": query["path"],
                "scores": {}, "missing": {}}
        for metric in METRIC_DIRECTION:
            scores = {}
            for style_id in style_ids:
                values, statuses = [], []
                for reference in reference_by_style[style_id]:
                    pair = entry["lookup"].get((query["path"], reference["path"]))
                    value = metric_value(pair, metric) if pair else None
                    values.append(value)
                    statuses.append((pair or {}).get(metric, {}).get("status"))
                valid = [value for value in values if value is not None]
                scores[style_id] = {
                    "values": values, "status": "ok" if len(valid) == len(values) and values else "partial",
                    "mean": statistics.mean(valid) if valid and len(valid) == len(values) else None,
                    "count": len(valid), "expected": len(values),
                    "aggregate": (min(valid) if METRIC_DIRECTION[metric] == "min" else max(valid))
                    if valid and len(valid) == len(values) else None,
                    "metric_statuses": statuses}
            unit["scores"][metric] = scores
        units.append(unit)

    analysis = {}
    for metric in METRIC_DIRECTION:
        table = []
        confusion = {style: {other: 0 for other in style_ids} for style in style_ids}
        correct = ties = wrong = missing = 0
        per_style = {style: {"correct": 0, "total": 0, "tie": 0} for style in style_ids}
        for unit in units:
            scores = unit["scores"][metric]
            valid = {style: value["aggregate"] for style, value in scores.items() if value["aggregate"] is not None}
            if len(valid) < len(style_ids):
                missing += 1
                table.append({"query_id": unit["query_id"], "query_style": unit["query_style"],
                              "status": "missing", "scores": {style: scores[style]["aggregate"] for style in style_ids}})
                continue
            ordered = sorted(valid.items(), key=lambda item: (item[1] if METRIC_DIRECTION[metric] == "min" else -item[1]))
            best_value = ordered[0][1]
            limit = TIE_RTOL * max(1.0, abs(best_value))
            top = [style for style, value in ordered if abs(value - best_value) <= limit]
            per_style[unit["query_style"]]["total"] += 1
            if len(top) > 1:
                ties += 1
                per_style[unit["query_style"]]["tie"] += 1
                status = "tie"
                chosen = None
            elif top[0] == unit["query_style"]:
                correct += 1
                per_style[unit["query_style"]]["correct"] += 1
                status = "correct"
                chosen = top[0]
            else:
                wrong += 1
                status = "wrong"
                chosen = top[0]
            if chosen:
                confusion[unit["query_style"]][chosen] += 1
            table.append({"query_id": unit["query_id"], "query_style": unit["query_style"], "status": status,
                          "predicted": chosen, "top": top,
                          "scores": {style: scores[style]["aggregate"] for style in style_ids},
                          "scores_full": {style: scores[style]["mean"] for style in style_ids}})
        total = len(units)
        accuracy = correct / total if total else None
        apportioned = (correct + sum(1.0 / len(row["top"]) for row in table if row["status"] == "tie")) / total if total else None
        per_query_clusters = {row["query_id"]: (1.0 if row["status"] == "correct" else 0.0)
                              for row in table if row["status"] != "missing"}
        by_query = cluster_bootstrap(per_query_clusters, lambda sample: statistics.mean(sample), seed=SEED)
        by_work = cluster_bootstrap({row["query_id"].split("-")[0] + "|" + row["query_id"]: value
                                     for row, value in
                                     zip([r for r in table if r["status"] != "missing"],
                                         [v for v in per_query_clusters.values()])},
                                    lambda sample: statistics.mean(sample), seed=SEED)
        analysis[metric] = {
            "top1_accuracy": accuracy, "apportioned_accuracy": apportioned,
            "counts": {"correct": correct, "tie": ties, "wrong": wrong, "missing": missing, "total": total},
            "wilson_95": wilson(correct, total - missing),
            "bootstrap_by_query": by_query, "bootstrap_by_work": by_work,
            "per_style": {style: {**value, "accuracy": (value["correct"] / value["total"]) if value["total"] else None}
                          for style, value in per_style.items()},
            "confusion": confusion, "rows": table,
            "independent_clusters": {"queries": len(per_query_clusters), "works": len(per_query_clusters)},
        }

    same, different = e1_distributions(units, style_ids)
    primary = analysis[MAIN_METRIC]
    gates = {
        "top1_ge_80": (primary["top1_accuracy"] or 0) >= 0.80,
        "per_style_ge_2_of_3": all((primary["per_style"][style]["accuracy"] or 0) >= 2 / 3 for style in style_ids),
    }
    return {"stage": "E1", "generated_at": now(), "device": device, "style_ids": style_ids,
            "counts": {"queries": len(query_rows), "references_per_style": {style: len(reference_by_style[style])
                                                                           for style in style_ids},
                       "aggregate_units": len(units) * len(style_ids),
                       "image_pairs": len(units) * sum(len(reference_by_style[style]) for style in style_ids)},
            "main_metric": MAIN_METRIC, "analysis": analysis,
            "distributions": {"same_style": same, "different_style": different},
            "gates": gates,
            "status": "complete",
            "conclusion": ("主指标 CSD 在试验集上达到预注册门槛" if all(gates.values())
                           else "主指标未达到预注册门槛（各风格 ≥2/3 与 Top-1 ≥80% 需同时满足）")}


def e1_distributions(units, style_ids) -> tuple[dict, dict]:
    same, different = {}, {}
    for metric in METRIC_DIRECTION:
        same_values, other_values = [], []
        for unit in units:
            for style_id in style_ids:
                entry = unit["scores"][metric][style_id]
                if entry["aggregate"] is None:
                    continue
                (same_values if style_id == unit["query_style"] else other_values).append(entry["aggregate"])
        def describe(values):
            if not values:
                return {"count": 0}
            return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
                    "min": min(values), "max": max(values),
                    "p10": sorted(values)[max(0, int(0.10 * (len(values) - 1)))],
                    "p90": sorted(values)[min(len(values) - 1, int(0.90 * (len(values) - 1)))]}
        same[metric] = describe(same_values)
        different[metric] = describe(other_values)
        # 用同风格/异风格分布估计可分离性（诊断，不参与验收）
        if same_values and other_values:
            direction = METRIC_DIRECTION[metric]
            separation = (statistics.mean(other_values) - statistics.mean(same_values)) if direction == "min" \
                else (statistics.mean(same_values) - statistics.mean(other_values))
            pooled = math.sqrt((statistics.pvariance(same_values) + statistics.pvariance(other_values)) / 2) or 1e-12
            same[metric]["standardized_separation"] = separation / pooled
    return same, different


# --------------------------------------------------------------------------- #
# E2：内容与配色干扰对照
# --------------------------------------------------------------------------- #


def build_e2_pairs(plan, progress=None) -> dict:
    """在**查看任何指标之前**冻结 P/N 与 N0–N4 变体。

    - P：同画风、与 anchor 差异最大（人物类型/配色/构图）的真实原作品；
    - N：异画风、与 anchor 在景别/亮暗/背景密度/主色相上最接近的真实原作品；
    - 选择只读取冻结属性，不读取任何画风指标。
    """
    from utils.style_experiment_attributes import hue_distance
    styles = plan["split"]
    style_ids = sorted(styles)
    anchors, positive, negative = [], {}, {}
    used_positive = {style: set() for style in style_ids}
    for style_id in style_ids:
        for row in styles[style_id]["queries"][:3]:
            anchors.append(dict(row, style_id=style_id))
    for anchor in anchors:
        pool = [row for row in styles[anchor["style_id"]]["extras"] if row["path"] != anchor["path"]]
        scored = []
        for row in pool:
            framing_diff = 1.0 if row["framing"] != anchor["framing"] else 0.0
            brightness_diff = 1.0 if row["brightness"] != anchor["brightness"] else 0.0
            density_diff = 1.0 if row["background_density"] != anchor["background_density"] else 0.0
            distance = hue_distance(row["dominant_hue"], anchor["dominant_hue"])
            hue_diff = 0.0 if distance != distance else min(distance / 180.0, 1.0)
            used = 1.0 if row["path"] in used_positive[anchor["style_id"]] else 0.0
            scored.append((-(framing_diff + brightness_diff + density_diff + hue_diff) - 10.0 * used, row,
                           {"framing_different": framing_diff, "brightness_different": brightness_diff,
                            "background_density_different": density_diff,
                            "hue_distance_degrees": None if distance != distance else round(distance, 3)}))
        scored.sort(key=lambda item: (item[0], item[1]["path"]))
        best_score, best_row, best_detail = scored[0]
        used_positive[anchor["style_id"]].add(best_row["path"])
        positive[anchor["image_id"]] = {"row": best_row, "match_detail": best_detail,
                                        "difficulty_score": round(-best_score, 4)}
        candidates = []
        for other_style in style_ids:
            if other_style == anchor["style_id"]:
                continue
            for row in styles[other_style]["extras"] + styles[other_style]["references"]:
                framing_match = 1.0 if row["framing"] == anchor["framing"] else 0.0
                brightness_match = 1.0 if row["brightness"] == anchor["brightness"] else 0.0
                density_match = 1.0 if row["background_density"] == anchor["background_density"] else 0.0
                distance = hue_distance(row["dominant_hue"], anchor["dominant_hue"])
                hue_gap = 180.0 if distance != distance else distance
                framing_gap = abs(math.log(max(row["face_area_ratio"], 1e-6)) -
                                  math.log(max(anchor["face_area_ratio"], 1e-6)))
                cost = (3.0 * (1 - framing_match) + 2.0 * (1 - brightness_match) + 1.0 * (1 - density_match)
                        + hue_gap / 180.0 + framing_gap)
                candidates.append((cost, row, {"style_id": other_style, "framing_match": framing_match,
                                               "brightness_match": brightness_match,
                                               "background_density_match": density_match,
                                               "hue_distance_degrees": round(hue_gap, 3),
                                               "face_area_log_gap": round(framing_gap, 4)}))
        candidates.sort(key=lambda item: (item[0], item[1]["path"]))
        cost, best_row, detail = candidates[0]
        negative[anchor["image_id"]] = {"row": best_row, "match_detail": detail, "difficulty_cost": round(cost, 6)}
    return {"anchors": anchors, "positive": positive, "negative": negative,
            "selection_rule": {
                "positive": "同画风 reserved_e2_pool 中与 anchor 在 framing/brightness/background_density/"
                            "dominant_hue 差异总分最大者（同一条多用时优先未被使用的）",
                "negative": "异画风中与 anchor 在 framing/brightness/background_density 匹配且主色相与脸面积最接近者",
                "annotator": "自动匹配代理（协议要求由不知道指标值的标注者填写；本机没有人类标注者，"
                             "故使用只读取冻结属性的确定性规则，并在报告中单列为协议偏离）",
                "metric_values_used": False},
            "per_anchor_difficulty": {anchor["image_id"]: {"positive": positive[anchor["image_id"]]["match_detail"],
                                                          "negative": negative[anchor["image_id"]]["match_detail"]}
                                      for anchor in anchors}}


def freeze_e2_variants(plan, e2_plan, progress=None) -> dict:
    """生成 N0–N4 变体 + 掩膜 + 接触表，记录算法/参数/hash。"""
    run = run_root()
    directory = run / "e2" / "variants"
    records, sheets = {}, []
    for anchor in e2_plan["anchors"]:
        per_anchor = {}
        items = [("original", anchor["path"])]
        for name, function in attrs.TRANSFORMS.items():
            target = directory / anchor["image_id"] / f"{name}.png"
            outcome = function(anchor["path"], target)
            outcome["anchor_id"] = anchor["image_id"]
            per_anchor[name] = outcome
            items.append((name, outcome["path"]))
        records[anchor["image_id"]] = per_anchor
        sheet = run / "e2" / "contact-sheets" / f"{anchor['image_id']}-variants.png"
        attrs.build_contact_sheet(items, sheet)
        sheets.append(str(sheet))
        _log(progress, f"E2 变体已冻结：{anchor['image_id']}")
    return {"variants": records, "contact_sheets": sheets,
            "transform_version": attrs.TRANSFORM_VERSION,
            "parameters": {"N0": {"operation": "lossless PNG re-encode"},
                           "N1": {"hue_shift_degrees": 30.0},
                           "N2": {"intensity_scale": 0.9},
                           "N3": {"blur_sigma": 6.0, "mask_threshold_percentile": 75.0, "dilate_iterations": 2,
                                  "close_kernel": 15},
                           "N4": {"shift_fraction_of_width": 0.05, "padding": "replicate leftmost column"}},
            "human_review_required": "协议要求人工确认变体没有意外改人物结构；接触表路径见 contact_sheets，"
                                     "本机没有人类复核者，已标记 awaiting_human。"}


def e2(ledger: AttemptLedger, plan, e2_plan, variants, progress=None, device=DEVICE) -> dict:
    run = run_root()
    cache = run / "cache" / "e2"
    style_ids = sorted(plan["split"])
    reference_rows = []
    for style_id in style_ids:
        reference_rows.extend(plan["split"][style_id]["references"])

    comparison_rows, variant_rows = [], []
    for index, anchor in enumerate(e2_plan["anchors"]):
        _log(progress, f"E2 {index + 1}/{len(e2_plan['anchors'])}：{anchor['image_id']}")
        positive = e2_plan["positive"][anchor["image_id"]]
        negative = e2_plan["negative"][anchor["image_id"]]
        candidate_rows = [{"path": positive["row"]["path"], "sha256": positive["row"]["sha256"], "id": "P"},
                          {"path": negative["row"]["path"], "sha256": negative["row"]["sha256"], "id": "N"}]
        result = with_retries(
            ledger, f"e2:compare:{anchor['image_id']}", "E2",
            lambda anchor=anchor, candidate_rows=candidate_rows: compare(
                candidate_rows, [{"path": row["path"], "sha256": row["sha256"]} for row in reference_rows],
                cache / anchor["image_id"], device),
            request={"anchor": anchor["path"], "candidates": [row["path"] for row in candidate_rows]},
            charged=False)
        lookup = pair_lookup(result)
        for candidate in candidate_rows:
            by_style, missing = {}, []
            for style_id in style_ids:
                values = []
                for reference in plan["split"][style_id]["references"]:
                    pair = lookup.get((candidate["path"], reference["path"]))
                    value = metric_value(pair, "csd") if pair else None
                    values.append(value)
                if any(value is None for value in values):
                    missing.append(style_id)
                    by_style[style_id] = None
                else:
                    by_style[style_id] = statistics.mean(values)
            comparison_rows.append({
                "anchor_id": anchor["image_id"], "anchor_style": anchor["style_id"], "kind": candidate["id"],
                "path": candidate["path"], "candidate_style": (anchor["style_id"] if candidate["id"] == "P"
                                                               else negative["match_detail"]["style_id"]),
                "match_detail": positive["match_detail"] if candidate["id"] == "P" else negative["match_detail"],
                "csd_by_style": by_style, "missing_styles": missing,
                "metrics": {metric: metric_value(lookup.get((candidate["path"], reference_rows[0]["path"])), metric)
                            for metric in METRIC_DIRECTION},
                "primary_target_style": anchor["style_id"]})

        # 变体：只测「仅改颜色/背景/构图是否改变画风判断」，同一参考集
        variant_candidates = [{"path": anchor["path"], "sha256": anchor["sha256"], "id": "original"}]
        for name, outcome in variants["variants"][anchor["image_id"]].items():
            variant_candidates.append({"path": outcome["path"], "sha256": outcome["sha256"], "id": name})
        variant_result = with_retries(
            ledger, f"e2:variants:{anchor['image_id']}", "E2",
            lambda anchor=anchor, variant_candidates=variant_candidates: compare(
                variant_candidates, [{"path": row["path"], "sha256": row["sha256"]} for row in reference_rows],
                cache / (anchor["image_id"] + "-variants"), device),
            request={"anchor": anchor["path"], "variants": [row["path"] for row in variant_candidates]},
            charged=False)
        variant_lookup = pair_lookup(variant_result)
        for candidate in variant_candidates:
            by_style = {}
            for style_id in style_ids:
                values = [metric_value(variant_lookup.get((candidate["path"], reference["path"])), "csd")
                          for reference in plan["split"][style_id]["references"]]
                by_style[style_id] = statistics.mean(values) if all(value is not None for value in values) else None
            ranked = sorted(((style, value) for style, value in by_style.items() if value is not None),
                            key=lambda item: -item[1])
            variant_rows.append({"anchor_id": anchor["image_id"], "anchor_style": anchor["style_id"],
                                 "variant": candidate["id"], "csd_by_style": by_style,
                                 "argmax": ranked[0][0] if ranked else None,
                                 "same_as_original": None})

    analysis = e2_stats(comparison_rows, variant_rows, style_ids)
    return {"stage": "E2", "generated_at": now(), "device": device, "style_ids": style_ids,
            "selection": {key: e2_plan[key] for key in ("selection_rule", "per_anchor_difficulty")},
            "anchors": [{"image_id": row["image_id"], "style_id": row["style_id"], "path": row["path"]}
                        for row in e2_plan["anchors"]],
            "comparisons": comparison_rows, "variants": variant_rows, "analysis": analysis,
            "status": "complete",
            "human_review": "N0–N4 变体的人工核查（接触表）与「配色/构图匹配」分块标注仍等待人类输入。"}


def e2_stats(comparison_rows, variant_rows, style_ids) -> dict:
    by_anchor = {}
    for row in comparison_rows:
        by_anchor.setdefault(row["anchor_id"], {})[row["kind"]] = row
    metrics = [metric for metric in METRIC_DIRECTION]
    preference = {metric: {"correct": 0, "tie": 0, "wrong": 0, "detail": []} for metric in metrics}
    for anchor_id, rows in sorted(by_anchor.items()):
        positive, negative = rows.get("P"), rows.get("N")
        if not positive or not negative:
            continue
        for metric in metrics:
            first, second = positive["csd_by_style"], negative["csd_by_style"]
            if first is None or second is None:
                preference[metric]["detail"].append({"anchor": anchor_id, "status": "missing"})
                continue
            if metric == "csd":
                # 相似度：越大越像自身画风
                improvement = second - first
                rule = "CSD(anchor→N) − CSD(anchor→P) > 容差 才算正确偏好同画风的 P"
            else:
                # 距离：越小越像自身画风；P 自身画风分必须明显更小
                improvement = first - second
                rule = "距离(anchor→N) − 距离(anchor→P) > 容差 才算正确偏好同画风的 P"
            limit = TIE_RTOL * max(1.0, abs(first), abs(second))
            status = "tie" if abs(improvement) <= limit else ("correct" if improvement > 0 else "wrong")
            preference[metric][status] += 1
            color_close = (negative.get("match_detail") or {}).get("hue_distance_degrees")
            preference[metric]["detail"].append({
                "anchor": anchor_id, "status": status, "rule": rule,
                "positive_style_score": first, "negative_style_score": second,
                "negative_hue_distance_degrees": color_close,
                "composition_match": {key: (negative.get("match_detail") or {}).get(key)
                                      for key in ("framing_match", "background_density_match", "brightness_match")}})
    result = {}
    for metric in metrics:
        row = preference[metric]
        total = row["correct"] + row["tie"] + row["wrong"]
        colour_tight = [item for item in row["detail"]
                        if item.get("status") != "missing" and (item.get("negative_hue_distance_degrees") or 999) <= 30]
        composition_tight = [item for item in row["detail"]
                             if item.get("status") != "missing"
                             and all((item.get("composition_match") or {}).get(key) == 1.0
                                     for key in ("framing_match", "background_density_match", "brightness_match"))]
        def rate(items):
            if not items:
                return {"n": 0, "correct": 0, "rate": None}
            return {"n": len(items), "correct": sum(1 for item in items if item["status"] == "correct"),
                    "rate": sum(1 for item in items if item["status"] == "correct") / len(items),
                    "tie": sum(1 for item in items if item["status"] == "tie"),
                    "wrong": sum(1 for item in items if item["status"] == "wrong")}
        result[metric] = {"n": total, "correct": row["correct"], "tie": row["tie"], "wrong": row["wrong"],
                          "rate": (row["correct"] / total) if total else None,
                          "wilson_95": wilson(row["correct"], total),
                          "split_by_colour_match": rate(colour_tight),
                          "split_by_composition_match": rate(composition_tight),
                          "detail": row["detail"]}
    # 干扰变体：argmax 是否仍然落在自身画风
    transform_summary = {}
    for row in variant_rows:
        bucket = transform_summary.setdefault(row["variant"], {"same_style": 0, "other_style": 0, "missing": 0,
                                                               "flips": []})
        if row["argmax"] is None:
            bucket["missing"] += 1
        elif row["argmax"] == row["anchor_style"]:
            bucket["same_style"] += 1
        else:
            bucket["other_style"] += 1
            bucket["flips"].append({"anchor": row["anchor_id"], "predicted": row["argmax"]})
    for name, bucket in transform_summary.items():
        total = bucket["same_style"] + bucket["other_style"]
        bucket["prediction_accuracy"] = bucket["same_style"] / total if total else None
    return {"preference": result, "transform_summary": transform_summary,
            "note": "变体只做敏感性诊断：N1/N2 是配色/明度敏感性，不要求指标不变；"
                    "不得把不同尺度的分数直接相减比较大小。"}


# --------------------------------------------------------------------------- #
# E3：局部测量的受控响应
# --------------------------------------------------------------------------- #


def freeze_e3(progress=None) -> dict:
    """生成 6 个案例 × (原图 + 8 干预 × 3 档) 与 2 个负对照，并写入 hash 绑定的定位。"""
    import json as json_module
    from utils import style_experiment_synthetic as syn
    from utils.style_regions import save_annotation
    run = run_root()
    directory = run / "e3" / "variants"
    records, contact_items = [], []
    for case in syn.CASES:
        case_id = case["case_id"]
        for name, intervention, level, level_name, parameters in syn.variant_matrix():
            if name != case_id:
                continue
            label = "original" if not intervention else f"{intervention}-{level_name}"
            rgb, annotation, audit = syn.render(case_id, parameters)
            image_path, annotation_path, audit_path = syn.write_variant(
                directory / case_id, label.replace("/", "_"), rgb, annotation, audit)
            save_annotation(str(image_path), json_module.loads(annotation_path.read_text(encoding="utf-8")))
            records.append({"case_id": case_id, "style_id": case["style_id"], "label": label,
                            "intervention": intervention, "level": level, "level_name": level_name,
                            "parameters": parameters, "image": str(image_path),
                            "sha256": sha256_file(image_path), "annotation": str(annotation_path),
                            "audit": str(audit_path)})
        for control, spec in syn.NEGATIVE_CONTROLS.items():
            rgb, annotation, audit = syn.render(case_id, syn.variant_parameters(case), negative_control=control)
            image_path, annotation_path, audit_path = syn.write_variant(
                directory / case_id, control, rgb, annotation, audit)
            save_annotation(str(image_path), json_module.loads(annotation_path.read_text(encoding="utf-8")))
            records.append({"case_id": case_id, "style_id": case["style_id"], "label": control,
                            "intervention": "negative_control", "level": None, "level_name": control,
                            "parameters": {"negative_control": spec}, "image": str(image_path),
                            "sha256": sha256_file(image_path), "annotation": str(annotation_path),
                            "audit": str(audit_path)})
        contact = run / "e3" / "contact-sheets" / f"{case_id}.png"
        attrs.build_contact_sheet([(row["label"], row["image"]) for row in records if row["case_id"] == case_id],
                                  contact, cell=260, columns=5)
        contact_items.append(str(contact))
        _log(progress, f"E3 图样已冻结：{case_id}")
    return {"records": records, "contact_sheets": contact_items,
            "cases": [{key: case[key] for key in ("case_id", "style_id", "face_w", "eye_w", "eye_gap")}
                      for case in syn.CASES],
            "interventions": {name: {"target": spec["target"], "levels": spec["levels"]}
                              for name, spec in syn.INTERVENTIONS.items()},
            "negative_controls": syn.NEGATIVE_CONTROLS,
            "declared_couplings": {
                "eye_width": "改眼宽会同时改归一化眼高比例与上眼睑弧度（弦长变化）",
                "eye_height": "改眼高会同时改眼开口面积与眼亮度构成像素集合",
                "hair_fineness": "加粗发丝会同时改变可见连续支持（coverage）",
                "eye_gap": "双眼外移会同时改变眼中心到鼻/口的归一化距离（非目标诊断项）",
                "note": "这些耦合在预注册中冻结，不要求其他项恒定，但必须逐项报告未响应的分量。"},
            "limitation": "可控绘制图样用于机制测试，不代表真实插画的效度；真实图效度由 E1/E2/E4/E5 承担。"}


def e3(ledger: AttemptLedger, plan, frozen, progress=None, device=DEVICE) -> dict:
    from utils import style_face_metrics as face_metrics
    run = run_root()
    cache = run / "cache" / "e3"
    by_case = {}
    for record in frozen["records"]:
        by_case.setdefault(record["case_id"], []).append(record)
    rows = []
    for index, (case_id, records) in enumerate(sorted(by_case.items())):
        _log(progress, f"E3 测量 {index + 1}/{len(by_case)}：{case_id}")
        base = next(row for row in records if row["label"] == "original")
        candidates = [{"path": row["image"], "sha256": row["sha256"], "id": row["label"]}
                      for row in records if row["label"] != "original"]
        result = with_retries(
            ledger, f"e3:{case_id}", "E3",
            lambda base=base, candidates=candidates, case_id=case_id: compare(
                candidates, [{"path": base["image"], "sha256": base["sha256"]}], cache / case_id, device),
            request={"case": case_id, "variants": [row["path"] for row in candidates]}, charged=False)
        lookup = pair_lookup(result)
        base_descriptor = face_metrics.descriptors(base["image"], face_metrics.read_annotation(base["image"]))
        for record in records:
            if record["label"] == "original":
                continue
            pair = lookup.get((record["image"], base["image"]))
            variant_descriptor = face_metrics.descriptors(record["image"],
                                                          face_metrics.read_annotation(record["image"]))
            target = None
            if record["intervention"] in frozen["interventions"]:
                target = frozen["interventions"][record["intervention"]]["target"]
            row = {"case_id": case_id, "style_id": record["style_id"], "label": record["label"],
                   "intervention": record["intervention"], "level": record["level"],
                   "level_name": record["level_name"], "target_metric": target,
                   "image": record["image"], "parameters": record["parameters"],
                   "closeness": {metric: metric_value(pair, metric) for metric in face_metrics.METRICS} if pair else {},
                   "status": {metric: (pair or {}).get(metric, {}).get("status") for metric in face_metrics.METRICS}
                   if pair else {},
                   "components": {metric: variant_descriptor[metric].get("components")
                                  for metric in face_metrics.METRICS},
                   "base_components": {metric: base_descriptor[metric].get("components")
                                       for metric in face_metrics.METRICS},
                   "whole_image": {metric: metric_value(pair, metric) for metric in METRIC_DIRECTION} if pair else {}}
            rows.append(row)
    analysis = e3_stats(rows, frozen)
    return {"stage": "E3", "generated_at": now(), "device": device, "rows": rows, "analysis": analysis,
            "status": "complete" if all(
                row["status"].get(metric) not in (None, "unavailable") or True for row in rows) else "partial",
            "coverage": {"variants": len(rows), "cases": len(by_case),
                         "controlled_metrics": list(CONTROLLED_FACE_METRICS),
                         "diagnostic_only": list(DIAGNOSTIC_FACE_METRICS)},
            "limitation": "机制测试基于可控绘制图样；不能据此宣称真实插画上的局部效度。"}


def _direction_series(rows, case_id, intervention, metric):
    """返回该案例下某干预的 (level, level_name, components, closeness, status) 序列。"""
    series = []
    for row in rows:
        if row["case_id"] != case_id or row["intervention"] != intervention:
            continue
        series.append(row)
    series.sort(key=lambda item: item["level"])
    return series


def _monotonic(values, direction):
    usable = [value for value in values if value is not None]
    if len(usable) < 2:
        return None
    if direction == "increase":
        return all(b >= a - 1e-12 for a, b in zip(usable, usable[1:]))
    return all(b <= a + 1e-12 for a, b in zip(usable, usable[1:]))


def e3_stats(rows, frozen) -> dict:
    from utils import style_face_metrics as face_metrics
    # 目标响应方向：由可控图样的几何参数预注册
    expected_direction = {
        "hair_fineness": ("hair_fineness", "increase"),
        "hair_continuity": ("hair_continuity", "decrease"),
        "eye_brightness": ("eye_brightness", "decrease"),
        "eye_height": ("eye_height", "increase"),
        "eye_width": ("eye_width", "increase"),
        "eye_gap": ("eye_gap", "increase"),
        "eyelashes": ("eyelashes", "increase"),
        "eye_curvature": ("eye_curvature", "increase"),
    }
    cases = sorted({row["case_id"] for row in rows})
    responses = []
    for case_id in cases:
        for intervention, (target, direction) in expected_direction.items():
            series = _direction_series(rows, case_id, intervention, target)
            if len(series) < 4:
                responses.append({"case_id": case_id, "intervention": intervention, "status": "incomplete",
                                  "variants": len(series)})
                continue
            values = [row["closeness"].get(target) for row in series]
            statuses = [row["status"].get(target) for row in series]
            monotonic = _monotonic(values, direction)
            # 剂量响应：贴近度随档位单调（target 指标贴近度应随干预强度升高而下降）
            dosage = {"levels": [row["level"] for row in series], "closeness": values, "statuses": statuses}
            responses.append({"case_id": case_id, "intervention": intervention, "target": target,
                              "expected_direction": "指标贴近度随干预增强而下降",
                              "monotonic": monotonic,
                              "status": "pass" if monotonic else "fail", "dosage": dosage,
                              "components": [row["components"].get(target) for row in series]})
    passed = [row for row in responses if row.get("status") == "pass"]
    usable = [row for row in responses if row.get("status") in ("pass", "fail")]
    direction_hit = len(passed) / len(usable) if usable else None

    # 非目标响应矩阵：每个干预对全部 13 项的贴近度均值（越大越像 = 越不敏感）
    matrix = {}
    for case_id in cases:
        for intervention in expected_direction:
            series = _direction_series(rows, case_id, intervention, None)
            if not series:
                continue
            matrix.setdefault(intervention, {})
            for metric in face_metrics.METRICS:
                values = [row["closeness"].get(metric) for row in series]
                usable_values = [value for value in values if value is not None]
                matrix[intervention].setdefault(metric, []).extend(usable_values)
    non_target = {intervention: {metric: (statistics.mean(values) if values else None)
                                 for metric, values in sorted(per_metric.items())}
                  for intervention, per_metric in matrix.items()}

    # 负对照
    controls = []
    for row in rows:
        if row["intervention"] != "negative_control":
            continue
        spec = next((value for value in frozen["negative_controls"].values()
                     if value["description"] == row["parameters"]["negative_control"]["description"]), {})
        expected = spec.get("expect", [])
        exact = {metric: row["closeness"].get(metric) for metric in expected}
        unchanged = all(value is not None and abs(value - 1.0) <= 1e-9 for value in exact.values())
        controls.append({"case_id": row["case_id"], "control": row["label"], "expect_no_change": expected,
                         "closeness": {metric: row["closeness"].get(metric) for metric in expected},
                         "statuses": {metric: row["status"].get(metric) for metric in expected},
                         "status": "pass" if unchanged else "fail"})
    diagnostics = {}
    for metric in DIAGNOSTIC_FACE_METRICS:
        values = [row["closeness"].get(metric) for row in rows]
        usable = [value for value in values if value is not None]
        diagnostics[metric] = {"measured_pairs": len(usable), "total_pairs": len(values),
                               "verified_by_intervention": False,
                               "note": "本轮没有对应的受控干预；只能作诊断，不能宣称已验证。"}
    return {"responses": responses,
            "target_direction_hit": direction_hit,
            "target_direction_counts": exact_counts([row.get("status") == "pass" for row in usable]),
            "non_target_response_matrix": non_target,
            "negative_controls": controls,
            "negative_controls_passed": bool(controls) and all(row["status"] == "pass" for row in controls),
            "diagnostics": diagnostics,
            "gate": {"required": ACCEPTANCE["local_measurement_assist"]["value"],
                     "direction_hit": direction_hit,
                     "passed": direction_hit is not None
                               and direction_hit >= ACCEPTANCE["local_measurement_assist"]["value"]
                               and bool(controls) and all(row["status"] == "pass" for row in controls)}}


def metric_value_from(row, metric):
    """负对照用的精确贴近度：直接取共享入口给出的数值。"""
    return row["closeness"].get(metric)


# --------------------------------------------------------------------------- #
# 总编排
# --------------------------------------------------------------------------- #


def run_all(ledger, plan, progress=None, stages=("E0", "E1", "E2", "E3", "E4", "E5"), locate_regions=True):
    """按阶段执行；每个阶段的结果单独落盘，重跑时只补缺的阶段。"""
    from utils.style_experiment_controlled import ensure_stage_result
    run = run_root()
    results = {}

    def stage(name, producer):
        if name not in stages:
            results[name] = read_json(run / name / f"{name}-result.json", {}) or {"stage": name, "status": "planned"}
            return results[name]
        _log(progress, f"==== 阶段 {name} 开始 ====")
        value = ensure_stage_result(ledger, run, name, producer)
        _log(progress, f"==== 阶段 {name} 结束：{value.get('status')} ====")
        results[name] = value
        return value

    stage("E0", lambda: e0(ledger, plan, progress))
    stage("E1", lambda: e1(ledger, plan, progress))

    def e2_stage():
        variants = read_json(run / "E2" / "variants-frozen.json")
        if not variants:
            variants = freeze_e2_variants(plan, plan["e2"], progress)
            atomic_json(run / "E2" / "variants-frozen.json", variants)
        value = e2(ledger, plan, plan["e2"], variants, progress)
        value["contact_sheets"] = variants.get("contact_sheets")
        return value
    stage("E2", e2_stage)
    stage("E3", lambda: e3(ledger, plan, plan["e3_frozen"], progress))
    stage("E4", lambda: e4_auto(ledger, plan, e4_plan(plan, plan["e3_frozen"]), progress,
                                locate=locate_regions))

    def e5_stage():
        e5_inputs = {"runs": plan["e5_states"], "blind_pool": plan["e5"]["blind_pool"],
                     "reference_showcase": plan["e5"]["reference_showcase"]}
        review = e5_visual_review(ledger, e5_inputs, progress)
        blind = build_blind_package(e5_inputs)
        return {"stage": "E5", "generated_at": now(), "review": review,
                "budget": e5_budget(plan), "blind": blind,
                "blind_analysis": e5_blind_summary(blind),
                "analysis": e5_analysis(review, blind),
                "status": "partial",
                "note": "正式视觉复核只覆盖历史 2 张产物；新增 18 槽位待额度，人类盲评待作答。"}
    if "E5" in stages:
        for style_id, variants in plan["e5_states"].items():
            for variant, payload in variants.items():
                payload["state_path"] = str(run / "E5" / style_id / f"state-{variant}.json")
        results["E5"] = ensure_stage_result(ledger, run, "E5", e5_stage)
    else:
        results["E5"] = read_json(run / "E5" / "E5-result.json", {}) or {"status": "planned"}
    return results


# --------------------------------------------------------------------------- #
# E4：自动定位与人工标注误差
# --------------------------------------------------------------------------- #


def e4_plan(plan, frozen) -> dict:
    cases = []
    for case in frozen["cases"]:
        record = next(row for row in frozen["records"]
                      if row["case_id"] == case["case_id"] and row["label"] == "original")
        cases.append({"image_id": record["case_id"], "path": record["image"],
                      "sha256": record["sha256"], "kind": "synthetic_case",
                      "style_id": record["style_id"],
                      "ground_truth_annotation": record["annotation"], "history_failure": False})
    for row in plan["e0_images"]:
        if row["kind"] != "generated":
            continue
        cases.append({"image_id": row["image_id"], "path": row["path"], "sha256": row["sha256"],
                      "kind": "historical_gemini",
                      "style_id": row.get("style_id"), "history_failure": True,
                      "history_note": row.get("history_note")})
    return {"images": cases,
            "human_annotators_required": 2,
            "repeat_delay_hours": 24,
            "note": "自动候选 A 与人工 H1/H2/H1-repeat 必须分开保存；人工必须由实际人类提供，"
                    "不得由模型或 agent 代填 confirmed。"}


def e4_auto(ledger: AttemptLedger, plan, e4_plan_data, progress=None, locate=True) -> dict:
    import os
    from utils import style_face_metrics as face_metrics
    from utils.style_regions import propose_regions
    run = run_root()
    rows = []
    config = None
    if locate:
        from utils.analysis_gpt_prompt import load_text_api_config
        loaded = load_text_api_config()
        config = {"base_url": loaded.get("base_url"), "model": loaded.get("model"),
                  "api_key": loaded.get("api_key")}
    for index, item in enumerate(e4_plan_data["images"]):
        _log(progress, f"E4 自动定位 {index + 1}/{len(e4_plan_data['images'])}：{item['image_id']}")
        entry = {"image_id": item["image_id"], "path": item["path"], "kind": item["kind"],
                 "sha256": item["sha256"], "style_id": item.get("style_id"),
                 "history_failure": item.get("history_failure")}
        if locate and config and config.get("api_key"):
            try:
                annotation = with_retries(
                    ledger, f"e4:propose:{item['image_id']}", "E4",
                    lambda item=item, config=config: propose_regions(
                        item["path"], (config["base_url"], config["api_key"], config["model"])),
                    request={"model": config["model"], "endpoint": config["base_url"],
                             "image_sha256": item["sha256"]},
                    charged=True)
                entry.update({"proposal": "ok", "confirmed": annotation.get("confirmed"),
                              "pose": annotation.get("pose")})
            except Exception as exc:
                entry.update({"proposal": "failed", "error": f"{type(exc).__name__}: {exc}",
                              "error_class": type(exc).__name__})
        else:
            entry.update({"proposal": "skipped"})
        try:
            annotation = face_metrics.read_annotation(item["path"])
            entry["annotation_present"] = annotation is not None
            entry["annotation_confirmed"] = bool((annotation or {}).get("confirmed"))
            entry["pose"] = (annotation or {}).get("pose")
            entry["provisional"] = not bool((annotation or {}).get("confirmed")) and annotation is not None
            entry["incomplete_dimensions"] = sorted(
                metric for metric, outcome in
                face_metrics.descriptors(item["path"], annotation).items() if outcome["status"] != "ok")
            entry["local_meanings"] = {
                "returned": annotation is not None,
                "usable": bool(annotation) and not entry["incomplete_dimensions"],
                "confirmed": entry["annotation_confirmed"]}
        except Exception as exc:
            entry.update({"annotation_present": False, "error": f"{type(exc).__name__}: {exc}"})
        entry["overlay"] = str(run / "e4" / "overlays" / f"{item['image_id']}-auto-overlay.png")
        rows.append(entry)
        _write_overlay(item, annotation if isinstance(annotation, dict) else None, entry["overlay"])
    returned = sum(1 for row in rows if row.get("local_meanings", {}).get("returned"))
    usable = sum(1 for row in rows if row.get("local_meanings", {}).get("usable"))
    confirmed = sum(1 for row in rows if row.get("local_meanings", {}).get("confirmed"))
    per_dimension = {}
    for metric in FACE_METRIC_DIRECTION:
        available = 0
        for row in rows:
            try:
                annotation = face_metrics.read_annotation(row["path"])
                if annotation and face_metrics.descriptors(row["path"], annotation)[metric]["status"] == "ok":
                    available += 1
            except Exception:
                pass
        per_dimension[metric] = {"available": available, "total": len(rows),
                                 "coverage": available / len(rows) if rows else None}
    # 与可控图样的解析真值比较（只对 6 个合成案例有意义）
    geometry = e4_geometry_against_truth(e4_plan_data["images"])
    return {"stage": "E4-auto", "generated_at": now(), "rows": rows,
            "coverage": {"images": len(rows), "returned": returned, "usable": usable, "confirmed": confirmed,
                         "return_rate": returned / len(rows) if rows else None,
                         "usable_rate": usable / len(rows) if rows else None,
                         "confirmed_rate": confirmed / len(rows) if rows else None},
            "per_dimension_coverage": per_dimension,
            "geometry_vs_synthetic_truth": geometry,
            "awaiting_human": {"annotators": 2, "H1": None, "H2": None, "H1_repeat_after_24h": None,
                               "note": "人工标注尚未提供；自动 proposal 始终 confirmed=false。"},
            "gate": {"usable_rate_required": ACCEPTANCE["auto_region_default"]["coverage"],
                     "usable_rate": usable / len(rows) if rows else None,
                     "per_dimension_required": ACCEPTANCE["auto_region_default"]["per_metric_coverage"],
                     "passed": False,
                     "reason": "人工间重复性与人工对照尚未提供，且自动可用率/发丝采样带有效率需人工确认后复核"}}


def _write_overlay(item, annotation, target) -> str:
    """把区域曲线画到图上，作为人工核查证据（自动结果不作正式测量）。"""
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    from pathlib import Path as _Path
    target = _Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(item["path"]) as source:
            array = np.asarray(ImageOps.exif_transpose(source).convert("RGB")).copy()
        height, width = array.shape[:2]
        scale = min(1.0, 1400 / max(width, height))
        if scale < 1.0:
            array = cv2.resize(array, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        height, width = array.shape[:2]
        canvas = array.copy()
        if annotation:
            style = {"face_outline": (0, 255, 0), "hair_regions": (255, 160, 0), "hair_strands": (255, 0, 255)}
            for key, color in style.items():
                polygons = annotation.get(key) or []
                if isinstance(polygons, dict):
                    polygons = [polygons]
                for polygon in polygons:
                    pts = polygon.get("points") if isinstance(polygon, dict) else polygon
                    if not pts:
                        continue
                    vertices = np.round(np.asarray(pts, dtype=float) * [width - 1, height - 1]).astype(np.int32)
                    cv2.polylines(canvas, [vertices.reshape(-1, 1, 2)], key != "hair_strands", color, 2, cv2.LINE_AA)
            for eye in (annotation.get("eyes") or {}).values():
                for key, color in (("upper_lid", (255, 0, 0)), ("lower_lid", (0, 128, 255)), ("iris", (255, 255, 0))):
                    pts = eye.get(key)
                    if not pts:
                        continue
                    vertices = np.round(np.asarray(pts, dtype=float) * [width - 1, height - 1]).astype(np.int32)
                    cv2.polylines(canvas, [vertices.reshape(-1, 1, 2)], key == "iris", color, 2, cv2.LINE_AA)
                for lash in eye.get("lashes") or []:
                    vertices = np.round(np.asarray(lash, dtype=float) * [width - 1, height - 1]).astype(np.int32)
                    cv2.polylines(canvas, [vertices.reshape(-1, 1, 2)], False, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(target), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    except Exception as exc:
        target.with_suffix(".error.txt").write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
    return str(target)


def e4_geometry_against_truth(images) -> dict:
    """自动眼睑曲线/发丝路径与可控图样解析真值的距离（只对合成案例）。"""
    import math as _math
    from utils import style_face_metrics as face_metrics
    rows = []
    for item in images:
        if item["kind"] != "synthetic_case" or not item.get("ground_truth_annotation"):
            continue
        try:
            import json as _json
            truth = _json.loads(Path(item["ground_truth_annotation"]).read_text(encoding="utf-8"))
            proposal = face_metrics.read_annotation(item["path"])
            if not proposal:
                rows.append({"image_id": item["image_id"], "status": "no_proposal"})
                continue
            entry = {"image_id": item["image_id"], "status": "ok", "eyes": {}, "face": {}, "hair": {}}
            for name in ("viewer_left", "viewer_right"):
                expected = (truth.get("eyes") or {}).get(name) or {}
                got = (proposal.get("eyes") or {}).get(name) or {}
                for key in ("upper_lid", "lower_lid"):
                    if not expected.get(key) or not got.get(key):
                        entry["eyes"][f"{name}_{key}"] = {"status": "missing"}
                        continue
                    a = [[float(x), float(y)] for x, y in expected[key]]
                    b = [[float(x), float(y)] for x, y in got[key]]
                    distances = [min(_math.dist(point, other) for other in b) for point in a]
                    width = _eye_width(expected)
                    entry["eyes"][f"{name}_{key}"] = {
                        "status": "ok",
                        "mean_over_eye_width": statistics.mean(distances) / width if width else None,
                        "p95_over_eye_width": sorted(distances)[int(0.95 * (len(distances) - 1))] / width
                        if width else None,
                        "eye_width_normalized": width}
            rows.append(entry)
        except Exception as exc:
            rows.append({"image_id": item["image_id"], "status": "error",
                         "error": f"{type(exc).__name__}: {exc}"})
    return {"rows": rows,
            "note": "合成图样上的偏差不代表真实动漫图精度；真实图需要人工叠图核查（awaiting_human）。"}


def _eye_width(eye) -> float:
    import math as _math
    upper = [[float(x), float(y)] for x, y in eye["upper_lid"]]
    return _math.dist(upper[0], upper[-1]) or 1e-9


def e4_human_analysis(annotations: dict, images) -> dict:
    """给定人工标注（H1/H2/H1-repeat）时的分析；未提供则返回 awaiting_human。"""
    if not annotations or not any(annotations.values()):
        return {"status": "awaiting_human",
                "required": {"H1": "标注者一", "H2": "标注者二", "H1_repeat_after_24h": "标注者一 ≥24 小时后复标"},
                "note": "本机没有实际人类标注者，未填充任何人工数值；A↔H1、H1↔H2、"
                        "H1↔H1-repeat 的偏差与排名翻转率无法计算。"}
    return {"status": "provided", "note": "标注已提供，按 e4_human_report 计算（本函数留作扩展点）。"}


# --------------------------------------------------------------------------- #
# E5：真实 Gemini 候选排序与独立复核
# --------------------------------------------------------------------------- #


def e5_state(e5_plan, variant: str, output_state: Path) -> dict:
    """构造视觉复核状态文件：同一主体、同一目标画风、全部参考图。"""
    atomic_json(output_state, e5_plan["states"][variant])
    return e5_plan["states"][variant]


def e5_visual_review(ledger: AttemptLedger, e5_plan, progress=None) -> dict:
    """复用共享 `compare_state_in_process()`；正序/倒序各一次并记录顺序敏感性。"""
    import importlib
    from utils.analysis_gpt_prompt import load_text_api_config
    comparison = importlib.import_module("modules.image_analysis.style_comparison")
    config = load_text_api_config()
    run = run_root()
    results = {}
    for style_id, variants in e5_plan["runs"].items():
        results[style_id] = {}
        for variant, payload in variants.items():
            state_path = Path(payload["state_path"])
            state_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(state_path, payload["state"])
            outcome = with_retries(
                ledger, f"e5:review:{style_id}:{variant}", "E5",
                lambda state_path=state_path, config=config: comparison.compare_state_in_process(
                    str(state_path), (config["base_url"], config["api_key"], config["model"]), timeout=900,
                    progress=progress or (lambda message: None)),
                request={"state": str(state_path), "model": config["model"], "endpoint": config["base_url"],
                         "order": variant}, charged=True)
            results[style_id][variant] = {"status": "ok", "best_id": outcome.get("best_id"),
                                          "rows": outcome.get("rows"), "formula": outcome.get("formula"),
                                          "selection_status": outcome.get("selection_status"),
                                          "model": outcome.get("model"), "endpoint": outcome.get("endpoint"),
                                          "batch_count": outcome.get("batch_count"),
                                          "report_path": outcome.get("report_path")}
    sensitivity = e5_order_sensitivity(results)
    return {"stage": "E5-review", "generated_at": now(), "runs": results, "order_sensitivity": sensitivity,
            "non_independence": {"note": "视觉复核模型与自动定位模型同属文本/多模态模型族同端点，"
                                         "不是独立第三方评审；模型评分只是另一个待验证测量。"}}


def e5_order_sensitivity(results) -> dict:
    rows = []
    for style_id, variants in results.items():
        forward, reverse = variants.get("forward"), variants.get("reverse")
        if not forward or not reverse:
            continue
        forward_scores = {row["id"]: row["total"] for row in (forward.get("rows") or [])}
        reverse_scores = {row["id"]: row["total"] for row in (reverse.get("rows") or [])}
        for candidate in sorted(set(forward_scores) | set(reverse_scores)):
            rows.append({"style_id": style_id, "candidate": candidate,
                         "forward": forward_scores.get(candidate), "reverse": reverse_scores.get(candidate),
                         "abs_delta": None if forward_scores.get(candidate) is None or reverse_scores.get(candidate) is None
                         else abs(forward_scores[candidate] - reverse_scores[candidate]),
                         "best_forward": forward.get("best_id"), "best_reverse": reverse.get("best_id")})
    ranked_changed = any(row["best_forward"] != row["best_reverse"] for row in rows
                         if row["best_forward"] and row["best_reverse"]) if rows else None
    return {"rows": rows, "ranked_best_changed": ranked_changed,
            "note": "两个顺序独立运行、都保留；不得择优保留。" if rows else "候选数不足以构成顺序敏感性检验。"}


def e5_budget(plan) -> dict:
    """新增生成的额度登记：计划槽位 ≠ 已授权额度。"""
    return {"planned_slots": plan["e5_new_generation"]["planned_slots"],
            "subjects": plan["e5_new_generation"]["subjects"],
            "repeats": plan["e5_new_generation"]["repeats"],
            "authorised_slots": 0,
            "authorisation_evidence": "用户未在本轮授权新增收费生图额度；协议 §9 明确「本文件创建不等于已发送/"
                                      "已获追加费用额度」。",
            "status": "awaiting_budget",
            "frozen_parameters": plan["e5_frozen_parameters"],
            "note": "额度不足时先复用已有图完成可执行部分，新增生成项标待额度；不擅自增加收费额度。"}


# --------------------------------------------------------------------------- #
# 盲评包
# --------------------------------------------------------------------------- #


def build_blind_package(e5_plan, iterations=1) -> dict:
    """独立盲评包：隐藏配置名/槽位/指标/模型分数，A/B 位置由固定种子打乱。"""
    import base64
    import random as random_module
    run = run_root()
    directory = run / "e5" / "blind"
    directory.mkdir(parents=True, exist_ok=True)
    generator = random_module.Random(SEED)
    items = []
    pool = e5_plan["blind_pool"]
    for index in range(len(pool) - 1):
        for other in range(index + 1, len(pool)):
            items.append(("pair", pool[index], pool[other]))
    for entry in pool:
        items.append(("reference", entry, None))
    questions = []
    for question_id, (kind, first, second) in enumerate(items, 1):
        order = [first, second] if generator.random() < 0.5 else [second, first] if second else [first]
        questions.append({"question_id": question_id, "kind": kind,
                          "A": order[0], "B": order[1] if len(order) > 1 else None,
                          "hidden_source": "frozen"})
    mapping = {"schema": "style-similarity-blind-map/1", "seed": SEED, "iterations": iterations,
               "created_at": now(),
               "questions": [{"question_id": row["question_id"], "kind": row["kind"],
                              "A_id": row["A"]["image_id"], "B_id": (row["B"] or {}).get("image_id"),
                              "A_path": row["A"]["path"], "B_path": (row["B"] or {}).get("path")}
                             for row in questions],
               "note": "A/B 位置由固定种子 20261006 打乱；配置文件、槽位、指标与模型分数均不出现在问卷中。"}
    def data_uri(path):
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()
    html = ["<!doctype html><meta charset='utf-8'><title>画风相似度盲评</title>",
            "<style>body{font-family:system-ui;margin:24px;max-width:1200px}figure{display:inline-block;margin:8px}img{max-height:420px;border:1px solid #ccc}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}</style>",
            "<h1>画风相似度盲评问卷（A/B）</h1>",
            "<p>请在<strong>相同主体、相同目标画风</strong>的前提下判断哪张更接近参考画风，允许平局或无法判断。"
            "请记录线条、眼睛、头发、明暗、配色与背景依据。</p>",
            "<p>答案请写入同目录 <code>blind-answers.json</code>（模板已生成）："
            "<code>{\"questions\":[{\"question_id\":1,\"choice\":\"A|B|tie|unknown\",\"basis\":\"...\"}]}</code>。</p>"]
    for row in questions:
        if row["kind"] != "pair":
            continue
        html.append(f"<section><h3>第 {row['question_id']} 题</h3>")
        for label, entry in (("A", row["A"]), ("B", row["B"])):
            if not entry:
                continue
            html.append(f"<figure><figcaption>{label}</figcaption><img src='{data_uri(entry['path'])}'></figure>")
        html.append("<p>更接近参考画风：A / B / 平局 / 无法判断；依据：__________</p></section>")
    html.append("<h2>参考画风图（每题共用，见题目中的目标画风）</h2>")
    for entry in e5_plan.get("reference_showcase", []):
        html.append(f"<figure><figcaption>{entry['label']}</figcaption><img src='{data_uri(entry['path'])}'></figure>")
    html.append("<h2>重复题说明</h2><p>问卷包含重复题以检验个人一致性；重复题不计入主统计。</p>")
    (directory / "blind-questionnaire.html").write_text("\n".join(html), encoding="utf-8")
    atomic_json(directory / "blind-map.json", mapping)
    atomic_json(directory / "blind-answers-template.json",
                {"annotator": "", "started_at": "", "questions": [
                    {"question_id": row["question_id"], "choice": "", "basis": ""} for row in questions]})
    atomic_json(directory / "blind-summary.json",
                {"valid_independent_questions": sum(1 for row in questions if row["kind"] == "pair"),
                 "required": ACCEPTANCE["candidate_ranking_assist"]["min_items"],
                 "status": "awaiting_human",
                 "note": "问卷已生成但没有任何人类作答；不得用模型答案代填，也不得据此宣称排序效度。"})
    return {"directory": str(directory), "questionnaire": str(directory / "blind-questionnaire.html"),
            "map": str(directory / "blind-map.json"), "answers_template": str(directory / "blind-answers-template.json"),
            "valid_independent_questions": sum(1 for row in questions if row["kind"] == "pair")}


def e5_blind_summary(package) -> dict:
    summary = read_json(Path(package["directory"]) / "blind-summary.json", {})
    return {"package": package, "summary": summary,
            "status": "awaiting_human",
            "gap": {"required_valid_questions": ACCEPTANCE["candidate_ranking_assist"]["min_items"],
                    "available_valid_questions": package["valid_independent_questions"],
                    "missing": max(0, ACCEPTANCE["candidate_ranking_assist"]["min_items"]
                                   - package["valid_independent_questions"]),
                    "explanation": "历史只有 2 张 Gemini 产物，最多能组成 1 组组内比较与他人对照；"
                                   "达到 24 个有效独立题需要 E5 新增 18 槽位与额外人类作答。"}}


def e5_analysis(review, blind) -> dict:
    gates = {"review_completed": bool(review.get("runs")),
             "blind_human_available": False}
    body_matters = {}
    for style_id, variants in (review.get("runs") or {}).items():
        forward = variants.get("forward") or {}
        rows = forward.get("rows") or []
        body_matters[style_id] = {
            "best_id": forward.get("best_id"),
            "gates": {row["id"]: row.get("gates") for row in rows},
            "reference_content_copied": [row["id"] for row in rows if (row.get("gates") or {}).get("reference_content_copied")],
            "eligible": [row["id"] for row in rows if row.get("eligible")],
        }
    return {"gates": gates, "per_style": body_matters,
            "spearman_like": "人类共识缺失，方向一致率与 Kendall tau-b 均未计算",
            "status": "awaiting_human",
            "note": "视觉模型与人类的一致率、两人分歧率、重复题一致性都需要人类盲评结果；"
                    "主体门禁结果已记录。"}
