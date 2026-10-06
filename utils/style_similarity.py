"""Shared read-only image style measurements. No Qt, API calls or training."""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics

VERSION = "style-similarity/2"
METRICS = ("gram", "adain", "lpips", "csd")
LOCAL_METRICS = ("tone", "edges", "lines", "space")
ALL_METRICS = METRICS + LOCAL_METRICS
LIMITS = ("CSD 越大越接近；Gram、AdaIN、LPIPS 越小越接近。全部参考图等权，报告算术平均、"
          "中位数与范围；缺失任意配对则均值不参与比较。不换算百分比、不合并视觉总分。"
          "LPIPS 受人物、姿势和构图影响；VGG/LPIPS 方形拉伸，CSD 中心裁剪会遗漏边缘。"
          "NPU FP16 与 CUDA/CPU FP32 分开保存。CSD 上游权重有复现声明，本项目没有人类标注校准。"
          "Gram v2：G=FFᵀ/(CHW)，每层 ||G₁−G₂||²/4，"
          "Gatys 单层归一化、层权重全为1；不得与额外通道归一化的legacy v1标量混用。当前提示词迭代没有 LoRA 训练 loss。本地四组为渲染统计贴近度，受内容/构图影响；不与深度指标合成总分。")

def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate(pairs, metric, expected):
    valid = [p[metric]["value"] for p in pairs if p[metric]["status"] == "ok"
             and isinstance(p[metric]["value"], (int, float)) and math.isfinite(p[metric]["value"])]
    complete = expected > 0 and len(pairs) == expected and len(valid) == expected
    return {"count": len(valid), "expected": expected, "status": "ok" if complete else "partial",
            "mean": statistics.mean(valid) if complete else None,
            "median": statistics.median(valid) if complete else None,
            "min": min(valid) if valid else None, "max": max(valid) if valid else None}


def statistics_distance(first, second, metric):
    """batch=1 压缩统计量；保持部署版本的归一化和减法精度。"""
    import torch
    ga, sa = first
    gb, sb = second
    if metric == "gram":
        layers = {k: float(((ga[k] - gb[k]).double() ** 2).sum() / 4) for k in ga}
    else:
        layers = {k: float(torch.linalg.vector_norm(sa[k][0].double() - sb[k][0].double()) + torch.linalg.vector_norm(sa[k][1].double() - sb[k][1].double())) for k in sa}
    return sum(layers.values()), layers


def resolve_backend(requested):
    from utils.style_metrics import devices
    if requested != "auto-npu":
        return devices.resolve_device(requested)
    failures = {}
    for choice in ("npu", "cuda", "cpu"):
        try:
            backend = devices.resolve_device(choice)
            backend.requested = "auto-npu"
            backend.fallback = choice != "npu"
            backend.note = f"NPU 优先选择 {choice}" + ("；前序设备不可用：" + str(failures) if failures else "")
            backend.evidence = {**backend.evidence, "preceding_device_errors": failures}
            return backend
        except devices.DeviceUnavailableError as exc:
            failures[choice] = str(exc)
    raise devices.DeviceUnavailableError(str(failures))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


LOCAL_KEYS = {
    "tone": ("tone_range_p90_p10", "local_rms_contrast", "mid_band_ratio_90_200", "clip_high_ratio_250"),
    "edges": ("edge_share_coarse", "edge_share_mid", "edge_share_fine"),
    "lines": ("line_long_ratio", "line_frag_ratio", "line_endpoint_density", "line_avg_len"),
    "space": ("largest_clean_space_ratio", "clean_space_ratio", "subject_mid_ratio"),
}
LABELS = {"gram": "Gram ↓", "adain": "AdaIN ↓", "lpips": "LPIPS ↓", "csd": "CSD ↑",
          "tone": "明度层次 ↑", "edges": "边缘比例 ↑", "lines": "线条连续性 ↑", "space": "负空间 ↑"}


def metric_contract():
    return {"version": VERSION, "labels": LABELS, "deep": list(METRICS),
            "local": {"version": "symmetric-render/1", "keys": LOCAL_KEYS,
                      "preprocessing": "EXIF transpose, RGB, longest side 512px bicubic, preserve aspect",
                      "formula": "mean(1 - abs(a-b)/(abs(a)+abs(b))); both zero => 1",
                      "weights": "equal within each group; no combined score",
                      "skeleton": "deterministic morphological erosion/opening"}}


def validate_manifest(inputs):
    if not inputs.get("references") or not inputs.get("candidates"):
        raise ValueError("需要至少一张候选图和一张参考图")
    ids = [row["id"] for row in inputs["candidates"]]
    if len(set(ids)) != len(ids):
        raise ValueError("候选 ID 重复")
    for row in inputs["references"] + inputs["candidates"]:
        if file_hash(row["path"]) != row["sha256"]:
            raise ValueError("图片已变化：" + row["path"])


def image_manifest(candidates, references):
    def entry(path):
        path = str(Path(path).resolve())
        return {"path": path, "sha256": file_hash(path)}
    return {"references": [entry(p) for p in references],
            "candidates": [{"id": f"image_{i}", **entry(p)} for i, p in enumerate(candidates, 1)]}


def local_pair(first, second, cache=None):
    import numpy as np
    import cv2
    from PIL import Image, ImageOps
    from utils import style_render_metrics as render
    cache = cache if cache is not None else {}
    def stats(path):
        if path not in cache:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                width, height = image.size
                scale = 512 / max(width, height)
                image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BICUBIC)
                gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            cache[path] = {**render.tone_stats(gray), **render.multiscale_edge_ratio(gray),
                           **render.line_continuity(gray, deterministic=True), **render.space_stats(gray)}
        return cache[path]
    try:
        a, b = stats(first), stats(second)
        result = {}
        for group, keys in LOCAL_KEYS.items():
            components = {k: 1.0 if abs(a[k]) + abs(b[k]) == 0 else
                          1 - abs(a[k] - b[k]) / (abs(a[k]) + abs(b[k])) for k in keys}
            result[group] = {"status": "ok", "value": statistics.mean(components.values()),
                             "detail": {"first": {k: a[k] for k in keys}, "reference": {k: b[k] for k in keys},
                                        "components": components, "formula_version": "symmetric-render/1"}}
        return result
    except Exception as exc:
        return {m: {"status": "error", "value": None, "error": str(exc)} for m in LOCAL_METRICS}


def compare_images(inputs, requested="auto-npu", progress=lambda message: None, cache_directory=None):
    # 子进程先导入 torch；避免 Qt/ORT 先加载造成 Windows DLL 初始化失败。
    import torch
    from utils.style_metrics import devices, imaging, runner, inventory, gram, adain
    from utils.style_metrics import csd_metric, lpips_metric
    from utils.style_metrics.config import ONNX_CACHE, VGG_INPUT_SIZE, VGG19_GRAM_LAYERS, VGG19_ADAIN_LAYERS, GRAM_FORMULA_VERSION
    from utils.style_metrics.metadata import model_for, preprocessing_for
    validate_manifest(inputs)
    from utils import style_face_metrics as face_metrics
    image_paths = [r["path"] for r in inputs["references"] + inputs["candidates"]]
    regions_snapshot = face_metrics.annotation_snapshot(image_paths)
    annotations = {}
    for path in image_paths:
        try:
            annotations[path] = face_metrics.read_annotation(path)
        except Exception as exc:
            annotations[path] = {"annotation_error": str(exc)}
    face_cache = {}
    progress("验证设备与本地权重…")
    backend = resolve_backend(requested)
    weights = inventory.list_inventory(keys=["vgg19", "lpips_alex", "lpips_alex_trunk", "csd", "clip_vit_l14"], verify=True)
    models = {m: {"model": model_for(m), "preprocessing": preprocessing_for(m)} for m in METRICS}
    sources = {p.name: file_hash(p) for p in (Path(__file__).resolve().parent / "style_metrics").glob("*.py")}
    sources["style_render_metrics.py"] = file_hash(Path(__file__).with_name("style_render_metrics.py"))
    runtime = {}
    for package in ("torch", "torchvision", "numpy", "Pillow", "lpips", "clip", "openvino"):
        try:
            runtime[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            runtime[package] = "unavailable"
    signature = {"version": VERSION, "inputs": inputs, "device": backend.actual, "precision": backend.precision,
                 "models": models, "weights": weights, "sources": sources, "adapter": file_hash(__file__),
                 "runtime": runtime, "device_name": backend.device_name, "regions": regions_snapshot,
                 "face_code": file_hash(Path(face_metrics.__file__))}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    output = Path(cache_directory) / f"style-similarity-{backend.actual}-{backend.precision}-v2.json" if cache_directory else None
    if output and output.exists():
        cached = json.loads(output.read_text(encoding="utf-8"))
        if cached.get("input_hash") == fingerprint and cached.get("status") == "ok":
            progress("复用完整深度指标缓存")
            result = cached
            result["backend"] = backend.as_dict()
            result["cache_hit"] = True
            validate_manifest(inputs)
            if face_metrics.annotation_snapshot(image_paths) != regions_snapshot:
                raise ValueError("读取缓存期间区域标注变化，结果未写回")
            return result
    unavailable = inventory.preflight(METRICS)
    for m in METRICS:
        for key in inventory.METRIC_WEIGHTS[m]:
            if weights[key].get("sha256_matches") is not True:
                unavailable.setdefault(m, []).append({"error": f"{key} 权重 SHA-256 不匹配或缺失"})
    npu_encoder = None
    npu_lpips = None
    csd = None
    init_errors = {}
    init_outcomes = {}
    if backend.is_npu and "gram" not in unavailable:
        try:
            from utils.style_metrics.openvino_backend import NPUEncoder, export_vgg19_onnx
            npu_encoder = NPUEncoder(export_vgg19_onnx(ONNX_CACHE / f"vgg19_features_{VGG_INPUT_SIZE}.onnx"))
            npu_encoder.compile()
        except Exception as exc:
            init_errors.update(gram=exc, adain=exc)
    if "lpips" not in unavailable and backend.is_npu:
        try:
            npu_lpips = lpips_metric.NPULPIPSMetric()
        except Exception as exc:
            init_errors["lpips"] = exc
    if "csd" not in unavailable:
        try:
            csd = csd_metric.NPUCSDMetric() if backend.is_npu else csd_metric.get_metric(device=devices.torch_device(backend), precision=backend.precision)
        except Exception as exc:
            init_errors["csd"] = exc
    descriptor_cache, stats_cache = {}, {}
    def stats(path):
        if path not in stats_cache:
            image = imaging.load_rgb(path)
            feats = runner.vgg_features_npu(npu_encoder, [image]) if backend.is_npu else runner.vgg_features_torch([image], backend)
            stats_cache[path] = ({k: gram.gram_matrix(feats[k]) for k in VGG19_GRAM_LAYERS},
                                 {k: adain.feature_stats(feats[k]) for k in VGG19_ADAIN_LAYERS})
        return stats_cache[path]
    def descriptor(path):
        if path not in descriptor_cache:
            descriptor_cache[path] = torch.nn.functional.normalize(csd.descriptor([imaging.load_rgb(path)]).float().cpu(), dim=1)
        return descriptor_cache[path]
    self_check = {}
    self_path = inputs["references"][0]["path"]
    self_image = imaging.load_rgb(self_path)
    for m in METRICS:
        if m in unavailable or m in init_errors:
            self_check[m] = {"status": "unavailable", "value": None}
            continue
        try:
            if m in ("gram", "adain"):
                value, _ = statistics_distance(stats(self_path), stats(self_path), m)
            elif m == "csd":
                value = float((descriptor(self_path) ** 2).sum())
            elif backend.is_npu:
                value, _ = npu_lpips.distance_images(self_image, self_image)
            else:
                outcome = runner.run_lpips(self_image, self_image, backend)
                if outcome.status != "ok":
                    init_outcomes[m] = outcome.as_dict()
                    self_check[m] = {"status": outcome.status, "value": None, "error": outcome.error}
                    continue
                value = outcome.value
            target = 1.0 if m == "csd" else 0.0
            tolerance = runner.SELF_TOLERANCE[runner.METRIC_KIND[m]]
            if not math.isfinite(value) or abs(value - target) > tolerance:
                raise ValueError(f"同图自比失败：{value}；期望 {target} ± {tolerance}")
            self_check[m] = {"status": "ok", "value": value, "target": target, "tolerance": tolerance}
        except Exception as exc:
            init_errors[m] = exc
            self_check[m] = {"status": "failed_self_check", "value": None, "error": str(exc)}
    local_cache = {}
    rows = []
    total = len(inputs["candidates"]) * len(inputs["references"])
    done = 0
    for candidate in inputs["candidates"]:
        pairs = []
        for reference in inputs["references"]:
            pair = {"reference": reference["path"], "reference_sha256": reference["sha256"]}
            for m in METRICS:
                try:
                    if m in unavailable:
                        pair[m] = {"status": "unavailable", "value": None, "error": unavailable[m]}
                        continue
                    if m in init_outcomes:
                        pair[m] = dict(init_outcomes[m])
                        continue
                    if m in init_errors:
                        raise init_errors[m]
                    details = {}
                    if m in ("gram", "adain"):
                        value, details = statistics_distance(stats(candidate["path"]), stats(reference["path"]), m)
                    elif m == "lpips":
                        a, b = imaging.load_rgb(candidate["path"]), imaging.load_rgb(reference["path"])
                        if backend.is_npu:
                            value, layers = npu_lpips.distance_images(a, b)
                            details = {"layers": layers}
                        else:
                            outcome = runner.run_lpips(a, b, backend)
                            if outcome.status != "ok":
                                pair[m] = outcome.as_dict()
                                continue
                            value, details = outcome.value, outcome.detail
                    else:
                        value = float((descriptor(candidate["path"]) * descriptor(reference["path"])).sum())
                    if not math.isfinite(value):
                        raise ValueError("输出不是有限数值")
                    pair[m] = {"status": "ok", "value": value, "detail": details}
                except Exception as exc:
                    pair[m] = runner.outcome_from_error(m, exc).as_dict()
            pair.update(local_pair(candidate["path"], reference["path"], local_cache))
            pair["face_features"] = face_metrics.compare_face_features(candidate["path"], reference["path"], annotations, face_cache)
            pairs.append(pair)
            done += 1
            progress(f"{done}/{total} · {candidate['id']}")
        rows.append({**candidate, "pairs": pairs, "summary": {m: aggregate(pairs, m, len(inputs["references"])) for m in ALL_METRICS},
                     "face_summary": {m: aggregate([p["face_features"] for p in pairs], m, len(inputs["references"])) for m in face_metrics.METRICS}})
    result = {"version": VERSION, "input_hash": fingerprint, "inputs": inputs, "backend": backend.as_dict(),
              "models": models, "weights": weights, "gram_formula_version": GRAM_FORMULA_VERSION, "cache_hit": False, "rows": rows, "limits": LIMITS,
              "runtime": runtime, "self_check": self_check,
              "status": "ok" if all(r["summary"][m]["status"] == "ok" for r in rows for m in ALL_METRICS) else "partial"}
    if backend.is_npu:
        result["execution"] = {"vgg": npu_encoder.describe() if npu_encoder else None,
                               "lpips": npu_lpips.describe() if npu_lpips else None,
                               "csd": csd.describe() if csd else None,
                               "split": "编码器在 NPU；Gram/AdaIN 统计与 CSD 点积在 CPU"}
    validate_manifest(inputs)
    if face_metrics.annotation_snapshot(image_paths) != regions_snapshot:
        raise ValueError("计算期间区域标注变化，结果未写回")
    face_outcomes = [outcome for row in rows for pair in row["pairs"] for outcome in pair["face_features"].values()]
    result["face_status"] = "ok" if all(o["status"] == "ok" for o in face_outcomes) else "provisional" if any(o["status"] == "provisional" for o in face_outcomes) else "partial"
    for row in rows:
        for metric, summary in row["face_summary"].items():
            summary["provisional_count"] = sum(p["face_features"][metric]["status"] == "provisional" for p in row["pairs"])
    result["region_geometry"] = {path: {key: value for key, value in (annotation or {}).items() if key in (
        "version", "image_sha256", "confirmed", "pose", "target_description", "face_outline", "eyes", "hair_regions", "hair_strands", "nose", "mouth", "source", "model", "proposal_hash", "limitations", "annotation_error")}
        for path, annotation in annotations.items()}
    result["region_annotations"] = regions_snapshot
    result["face_contract"] = {"version": face_metrics.VERSION, "labels": face_metrics.LABELS, "formula": "symmetric component closeness, equal within group", "provisional_excluded_from_means": True}
    result["metric_contract"] = metric_contract()
    if output:
        atomic_json(output, result)
    return result
