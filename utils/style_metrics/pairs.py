"""Read-only measurements of explicit image pairs, separate from calibration."""
import hashlib
import math
from pathlib import Path

from . import imaging, inventory, runner
from .config import GRAM_FORMULA_VERSION, ONNX_CACHE, VGG_INPUT_SIZE


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_pairs(pairs, metrics, backend):
    """Every requested pair is measured; missing weights never become zeros."""
    pre = inventory.preflight(metrics)
    images, features, engines, records = {}, {}, {}, []

    def image(path):
        path = str(Path(path).resolve())
        if path not in images:
            images[path] = imaging.load_rgb(path)
        return images[path]

    def features_for(path):
        key = str(Path(path).resolve())
        if key not in features:
            if backend.actual == "npu":
                if "vgg" not in engines:
                    from .openvino_backend import NPUEncoder, export_vgg19_onnx
                    encoder = NPUEncoder(export_vgg19_onnx(ONNX_CACHE / f"vgg19_features_{VGG_INPUT_SIZE}.onnx"))
                    encoder.compile()
                    engines["vgg"] = encoder
                features[key] = runner.vgg_features_npu(engines["vgg"], [image(path)])
            else:
                features[key] = runner.vgg_features_torch([image(path)], backend)
        return features[key]

    def measure(metric, first, second):
        if metric in ("gram", "adain"):
            fn = runner.run_gram if metric == "gram" else runner.run_adain
            return fn(features_for(first), features_for(second))
        if metric == "lpips" and backend.actual == "npu":
            if metric not in engines:
                from .lpips_metric import NPULPIPSMetric
                engines[metric] = NPULPIPSMetric()
            value, layers = engines[metric].distance_images(image(first), image(second))
            return runner.MetricOutcome(metric, "ok", value, {"per_layer": layers, "execution": engines[metric].describe()})
        fn = runner.run_lpips if metric == "lpips" else runner.run_csd
        return fn(image(first), image(second), backend)

    self_checks, self_sources = {}, {}
    for index, (first, second) in enumerate(pairs, 1):
        hashes = [file_digest(first), file_digest(second)]
        for metric in metrics:
            metadata = {"pair_id": f"P{index}", "pair": [str(Path(first).resolve()), str(Path(second).resolve())],
                        "image_sha256": hashes, "requested_device": backend.requested,
                        "actual_device": backend.actual, "backend": backend.backend, "precision": backend.precision}
            if metric == "gram":
                metadata["formula_version"] = GRAM_FORMULA_VERSION
            try:
                if metric in pre:
                    outcome = runner.MetricOutcome(metric, "unavailable", None,
                                                  error="Missing weights; no value produced")
                else:
                    outcome = measure(metric, first, second)
                row = {**outcome.as_dict(), **metadata, "validation": {}}
                if metric in pre:
                    row["remediation"] = pre[metric]
                    row["validation"]["note"] = "权重未就位，未产出任何数值（不做0分处理）"
                elif outcome.status == "ok":
                    if metric not in self_checks:
                        # Identity check uses first vs itself, NEVER first vs second.
                        self_checks[metric] = measure(metric, first, first)
                        self_sources[metric] = str(Path(first).resolve())
                    identity = self_checks[metric]
                    expected = 0.0 if runner.METRIC_KIND[metric] == "distance" else 1.0
                    limit = runner.SELF_TOLERANCE[runner.METRIC_KIND[metric]]
                    passed = identity.status == "ok" and identity.value is not None and math.isfinite(identity.value) and abs(identity.value - expected) <= limit
                    row["validation"] = {"self_check": {"source": self_sources[metric], "comparison": "source vs itself", "value": identity.value, "expected": expected, "passed": passed},
                                         "finite": outcome.value is not None and math.isfinite(outcome.value),
                                         "note": "真实配对不要求距离为0、相似度为1，也不要求人为soft/hard单调排序"}
                    if not row["validation"]["finite"]:
                        row.update(status="error", error="Non-finite pair value", value=None)
                    elif not passed:
                        row.update(status="failed_self_check", error="Identity check failed")
                if metric == "gram":
                    row["formula_version"] = GRAM_FORMULA_VERSION
            except Exception as exc:
                row = {**runner.outcome_from_error(metric, exc).as_dict(), **metadata,
                       "validation": {"note": "配对计算失败，其他配对继续"}}
            records.append(row)
    return records
