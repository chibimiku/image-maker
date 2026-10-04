"""本地深度指标对照；计算在独立进程，GUI 不加载 torch/OpenVINO。"""
import base64
import hashlib
import html
import importlib.metadata
import json
import math
import mimetypes
import os
from pathlib import Path
import statistics
import subprocess
import sys

VERSION = "style-deep-comparison-v1"
METRICS = ("gram", "adain", "lpips", "csd")
LIMITS = ("CSD 越大越接近；Gram、AdaIN、LPIPS 越小越接近。全部参考图等权，报告算术平均、"
          "中位数与范围；缺失任意配对则均值不参与比较。不换算百分比、不合并视觉总分。"
          "LPIPS 受人物、姿势和构图影响；VGG/LPIPS 方形拉伸，CSD 中心裁剪会遗漏边缘。"
          "NPU FP16 与 CUDA/CPU FP32 分开保存。CSD 上游权重有复现声明，本项目没有人类标注校准。"
          "Gram 沿用已部署实现：G=FFᵀ/(CHW)，每层 ||G₁−G₂||²/(4C²)，"
          "额外通道归一化使其数值不等同于 Gatys 原论文 loss。当前提示词迭代没有 LoRA 训练 loss。")


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_manifest(state):
    from modules.image_analysis.style_comparison import candidates_from_state
    refs = state.get("dataset", {}).get("images") or []
    candidates = candidates_from_state(state)
    if not refs or not candidates:
        raise ValueError("需要画风源图及已生成的候选图")
    if len({c["id"] for c in candidates}) != len(candidates):
        raise ValueError("候选 ID 重复")
    return {"references": [{"path": p, "sha256": file_hash(p)} for p in refs],
            "candidates": [{"id": c["id"], "path": c["path"], "sha256": file_hash(c["path"])} for c in candidates]}


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
        layers = {k: float(((ga[k] - gb[k]).double() ** 2).sum() / (4 * ga[k].shape[-1] ** 2)) for k in ga}
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


def write_report(result, directory):
    def esc(value):
        return html.escape(str(value))
    def picture(path):
        data = base64.b64encode(Path(path).read_bytes()).decode()
        mime = mimetypes.guess_type(path)[0] or "image/png"
        return f'<img style="max-height:240px;max-width:260px" src="data:{mime};base64,{data}">'
    parts = ['<!doctype html><meta charset="utf-8"><title>画风深度指标报告</title>',
             '<style>body{font-family:system-ui;margin:24px}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
             '<h1>画风深度指标报告</h1><p>' + esc(LIMITS) + '</p>',
             '<pre>' + esc(json.dumps({k: result.get(k) for k in ("version", "backend", "input_hash", "models", "weights", "training_parameters", "self_check", "execution")}, ensure_ascii=False, indent=2)) + '</pre>',
             '<h2>源图</h2>' + ''.join(picture(r["path"]) for r in result["inputs"]["references"]),
             '<h2>全部候选 · 全参考集等权均值</h2><table><tr><th>候选</th><th>CSD ↑</th><th>Gram ↓</th><th>AdaIN ↓</th><th>LPIPS ↓</th></tr>']
    for row in result["rows"]:
        parts.append('<tr><td>' + esc(row["id"]) + '</td>' + ''.join(
            '<td>' + (format(row["summary"][m]["mean"], '.8g') if row["summary"][m]["mean"] is not None else '缺失') + '</td>'
            for m in ("csd", "gram", "adain", "lpips")) + '</tr>')
    parts.append('</table>')
    for row in result["rows"]:
        parts.extend(['<h2>' + esc(row["id"]) + '</h2>', picture(row["path"]),
                      '<details><summary>逐张源图数值、分层参数、错误与覆盖率</summary><pre>' + esc(json.dumps(row, ensure_ascii=False, indent=2)) + '</pre></details>'])
    target = Path(directory) / "deep-feature-comparison.html"
    target.write_text('\n'.join(parts), encoding="utf-8")
    return str(target.resolve())


def compute(state_path, requested="auto", progress=lambda message: None):
    # 子进程先导入 torch；避免 Qt/ORT 先加载造成 Windows DLL 初始化失败。
    import torch
    from utils.style_metrics import devices, imaging, runner, inventory, gram, adain
    from utils.style_metrics import csd_metric, lpips_metric
    from utils.style_metrics.config import ONNX_CACHE, VGG_INPUT_SIZE, VGG19_GRAM_LAYERS, VGG19_ADAIN_LAYERS
    from tools.style_metrics_verify import model_for, preprocessing_for
    state_path = Path(state_path).resolve()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    inputs = input_manifest(state)
    progress("验证设备与本地权重…")
    backend = resolve_backend(requested)
    weights = inventory.list_inventory(keys=["vgg19", "lpips_alex", "lpips_alex_trunk", "csd", "clip_vit_l14"], verify=True)
    models = {m: {"model": model_for(m), "preprocessing": preprocessing_for(m)} for m in METRICS}
    sources = {p.name: file_hash(p) for p in (Path(__file__).resolve().parents[2] / "utils/style_metrics").glob("*.py")}
    runtime = {}
    for package in ("torch", "torchvision", "numpy", "Pillow", "lpips", "clip", "openvino"):
        try:
            runtime[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            runtime[package] = "unavailable"
    signature = {"version": VERSION, "inputs": inputs, "device": backend.actual, "precision": backend.precision,
                 "models": models, "weights": weights, "sources": sources, "adapter": file_hash(__file__),
                 "runtime": runtime, "device_name": backend.device_name}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    output = state_path.parent / f"deep-feature-comparison-{backend.actual}-{backend.precision}.json"
    if output.exists():
        cached = json.loads(output.read_text(encoding="utf-8"))
        if cached.get("input_hash") == fingerprint and cached.get("status") == "ok":
            progress("复用完整深度指标缓存")
            result = cached
            result["backend"] = backend.as_dict()
            result["cache_hit"] = True
            return publish(state_path, inputs, result, output)
    unavailable = inventory.preflight(METRICS)
    for m in METRICS:
        for key in inventory.METRIC_WEIGHTS[m]:
            if weights[key].get("sha256_matches") is not True:
                unavailable.setdefault(m, []).append({"error": f"{key} 权重 SHA-256 不匹配或缺失"})
    npu_encoder = None
    npu_lpips = None
    csd = None
    init_errors = {}
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
                    raise RuntimeError(outcome.error)
                value = outcome.value
            target = 1.0 if m == "csd" else 0.0
            tolerance = runner.SELF_TOLERANCE[runner.METRIC_KIND[m]]
            if not math.isfinite(value) or abs(value - target) > tolerance:
                raise ValueError(f"同图自比失败：{value}；期望 {target} ± {tolerance}")
            self_check[m] = {"status": "ok", "value": value, "target": target, "tolerance": tolerance}
        except Exception as exc:
            init_errors[m] = exc
            self_check[m] = {"status": "failed_self_check", "value": None, "error": str(exc)}
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
            pairs.append(pair)
            done += 1
            progress(f"{done}/{total} · {candidate['id']}")
        rows.append({**candidate, "pairs": pairs, "summary": {m: aggregate(pairs, m, len(inputs["references"])) for m in METRICS}})
    result = {"version": VERSION, "input_hash": fingerprint, "inputs": inputs, "backend": backend.as_dict(),
              "models": models, "weights": weights, "cache_hit": False, "rows": rows, "limits": LIMITS,
              "runtime": runtime, "self_check": self_check,
              "status": "ok" if all(r["summary"][m]["status"] == "ok" for r in rows for m in METRICS) else "partial"}
    if backend.is_npu:
        result["execution"] = {"vgg": npu_encoder.describe() if npu_encoder else None,
                               "lpips": npu_lpips.describe() if npu_lpips else None,
                               "csd": csd.describe() if csd else None,
                               "split": "编码器在 NPU；Gram/AdaIN 统计与 CSD 点积在 CPU"}
    return publish(state_path, inputs, result, output)


def publish(state_path, inputs, result, output):
    latest = json.loads(state_path.read_text(encoding="utf-8"))
    if input_manifest(latest) != inputs:
        raise ValueError("计算期间源图或候选列表变化，结果未写回")
    result["training_parameters"] = latest.get("parameters", {})
    result["report_path"] = write_report(result, state_path.parent)
    atomic_json(output, result)
    latest["deep_feature_comparison"] = result
    if latest.get("automatic_comparison"):
        from modules.image_analysis.style_comparison import write_comparison_report
        write_comparison_report(latest, latest["automatic_comparison"], str(state_path.parent))
    atomic_json(state_path, latest)
    return result


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def create_worker(state_path, device, parent=None):
    from PyQt6.QtCore import QThread, pyqtSignal
    class Worker(QThread):
        completed = pyqtSignal(object, str)
        progress = pyqtSignal(str)
        def run(self):
            root = Path(__file__).resolve().parents[2]
            log_path = Path(state_path).resolve().parent / "deep-feature-comparison.log"
            process = None
            try:
                python = Path(sys.executable)
                if python.name.lower() == "pythonw.exe":
                    python = python.with_name("python.exe")
                with log_path.open("w", encoding="utf-8") as log:
                    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
                    process = subprocess.Popen([str(python), "-u", str(root / "tools/style_metrics_verify.py"),
                                                "--comparison-state", str(Path(state_path).resolve()), "--device", device],
                                               cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    offset = 0
                    while process.poll() is None:
                        if self.isInterruptionRequested():
                            process.terminate()
                            process.wait()
                            raise RuntimeError("已取消深度指标计算；原结果保留")
                        with log_path.open(encoding="utf-8", errors="replace") as reader:
                            reader.seek(offset)
                            lines = reader.read().splitlines()
                            offset = reader.tell()
                        for line in lines:
                            if line.startswith("PROGRESS "):
                                self.progress.emit(line[9:])
                        self.msleep(200)
                if process.returncode:
                    raise RuntimeError(f"深度指标进程退出码 {process.returncode}；详情见 {log_path}\n" + log_path.read_text(encoding="utf-8", errors="replace")[-2500:])
                latest = json.loads(Path(state_path).read_text(encoding="utf-8"))
                self.completed.emit(latest["deep_feature_comparison"], "")
            except Exception as exc:
                self.completed.emit({}, str(exc))
            finally:
                if process is not None and process.poll() is None:
                    process.terminate()
                    process.wait()
    return Worker(parent)
