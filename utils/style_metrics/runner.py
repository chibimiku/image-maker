"""统一的指标运行器：预处理 → 编码器 → 距离/相似度 → 验证记录。

一条硬规则：**Gram / AdaIN / LPIPS 是距离指标（越小越像），CSD 是相似度指标（越大越像）**，
分别报告，不做「换算成百分比再平均」这类未经校准的合成。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image

from . import adain as adain_mod
from . import gram as gram_mod
from .config import (
    CSD_INPUT_SIZE,
    LPIPS_INPUT_SIZE,
    VGG19_ADAIN_LAYERS,
    VGG19_GRAM_LAYERS,
    VGG_INPUT_SIZE,
)
from .devices import Backend, torch_device
from .imaging import to_imagenet_vgg, to_lpips
from .inventory import WeightsMissing

METRIC_KIND = {
    "gram": "distance",
    "adain": "distance",
    "lpips": "distance",
    "csd": "similarity",
}

#: 提前声明的验证容差（**先声明后验证**，不许事后放宽）。
#: 判据为 ``|a - b| <= atol + rtol * |b|``；b 为 CPU fp32 基准。
TOLERANCES: dict[tuple[str, str], dict[str, float]] = {
    ("gram", "fp32"): {"rtol": 1e-3, "atol": 1e-6},
    ("gram", "fp16"): {"rtol": 5e-2, "atol": 1e-4},
    ("adain", "fp32"): {"rtol": 1e-3, "atol": 1e-6},
    ("adain", "fp16"): {"rtol": 5e-2, "atol": 1e-4},
    ("lpips", "fp32"): {"rtol": 1e-3, "atol": 1e-5},
    ("lpips", "fp16"): {"rtol": 1e-2, "atol": 1e-3},
    ("csd", "fp32"): {"rtol": 1e-3, "atol": 1e-5},
    ("csd", "fp16"): {"rtol": 1e-2, "atol": 1e-3},
}

#: 同图自比的上界（数学上应为 0 / 1，留一点数值余量）。
SELF_TOLERANCE = {"distance": 1e-6, "similarity": 1e-4}


def tolerance_for(metric: str, precision: str) -> dict[str, float]:
    key = (metric, "fp16" if precision.startswith("fp16") else "fp32")
    return TOLERANCES.get(key, TOLERANCES[(metric, "fp32")])


def within_tolerance(value: float, baseline: float, tol: dict[str, float]) -> tuple[bool, float]:
    abs_err = abs(value - baseline)
    limit = tol["atol"] + tol["rtol"] * abs(baseline)
    return abs_err <= limit, limit


# --------------------------------------------------------------------------- #
# VGG 特征（cpu / cuda）
# --------------------------------------------------------------------------- #


def vgg_features_torch(
    images: Sequence[Image.Image],
    backend: Backend,
    size: int = VGG_INPUT_SIZE,
) -> dict[str, torch.Tensor]:
    from .vgg_encoder import get_encoder

    device = torch_device(backend)
    encoder = get_encoder(device, backend.precision)
    batch = to_imagenet_vgg(images, size).to(device)
    if backend.precision == "fp16":
        batch = batch.half()
    with torch.no_grad():
        feats = encoder(batch)
    return {k: v.float().cpu() for k, v in feats.items()}


def vgg_features_npu(
    encoder,
    images: Sequence[Image.Image],
    size: int = VGG_INPUT_SIZE,
) -> dict[str, torch.Tensor]:
    from .imaging import to_imagenet_vgg_numpy

    out: dict[str, torch.Tensor] = {}
    for i, img in enumerate(images):
        arr = to_imagenet_vgg_numpy([img], size)
        res = encoder.infer(arr)
        for k, v in res.items():
            out.setdefault(k, []).append(torch.from_numpy(np.asarray(v, dtype=np.float32)))
        del i
    return {k: torch.cat(v, dim=0) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# 单指标计算
# --------------------------------------------------------------------------- #


@dataclass
class MetricOutcome:
    metric: str
    status: str
    value: float | None = None
    detail: dict = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "metric": self.metric,
            "kind": METRIC_KIND.get(self.metric, "unknown"),
            "status": self.status,
            "value": self.value,
            "detail": self.detail,
            "error": self.error,
        }


def run_gram(feats_a, feats_b, layers: Iterable[str] | None = None) -> MetricOutcome:
    try:
        res = gram_mod.gram_distance(feats_a, feats_b, layers or VGG19_GRAM_LAYERS)
        return MetricOutcome(
            "gram", "ok", res["distance"], {"layers": res["layers"], "summary": res["summary"]}
        )
    except Exception as exc:
        return outcome_from_error("gram", exc)


def run_adain(feats_a, feats_b, layers: Iterable[str] | None = None) -> MetricOutcome:
    try:
        res = adain_mod.adain_distance(feats_a, feats_b, layers or VGG19_ADAIN_LAYERS)
        return MetricOutcome("adain", "ok", res["distance"], {"layers": res["layers"]})
    except Exception as exc:
        return outcome_from_error("adain", exc)


def run_lpips(
    img_a: Image.Image,
    img_b: Image.Image,
    backend: Backend,
    size: int = LPIPS_INPUT_SIZE,
) -> MetricOutcome:
    try:
        from .lpips_metric import get_metric

        device = torch_device(backend) or torch.device("cpu")
        metric = get_metric(device=device, precision=backend.precision)
        value, per_layer = metric.distance_images(img_a, img_b, size)
        return MetricOutcome(
            "lpips",
            "ok",
            value,
            {"per_layer": per_layer, "layer_names": ["relu1", "relu2", "relu3", "relu4", "relu5"]},
        )
    except Exception as exc:
        return outcome_from_error("lpips", exc)


def run_csd(img_a: Image.Image, img_b: Image.Image, backend: Backend) -> MetricOutcome:
    try:
        from . import csd_metric

        return csd_metric.compare(img_a, img_b, backend)
    except WeightsMissing as exc:
        return MetricOutcome("csd", "unavailable", None, exc.as_dict(), error=str(exc))
    except Exception as exc:
        return MetricOutcome("csd", "error", None, error=f"{type(exc).__name__}: {exc}")


def outcome_from_error(metric: str, exc: BaseException) -> MetricOutcome:
    """把异常映射成统一的 MetricOutcome。

    权重缺失 → ``unavailable``（附修复信息，不冒充「计算失败」也不冒充「0 分」）；
    其它 → ``error``。
    """
    if isinstance(exc, WeightsMissing):
        return MetricOutcome(metric, "unavailable", None, exc.as_dict(), error=str(exc))
    return MetricOutcome(metric, "error", None, error=f"{type(exc).__name__}: {exc}")

# --------------------------------------------------------------------------- #
# 计时
# --------------------------------------------------------------------------- #


def time_call(fn, warmup: int = 2, repeat: int = 5, sync=None) -> dict:
    """固定输入下的耗时；GPU 计时正确同步，报告预热后的中位数。"""
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    samples = []
    for _ in range(repeat):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "warmup_runs": warmup,
        "repeat": repeat,
        "median_ms": samples[len(samples) // 2],
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "samples_ms": samples,
    }


def cuda_sync(backend: Backend):
    if backend.actual == "cuda":
        return lambda: torch.cuda.synchronize()
    return None


def cuda_peak_memory_mb(backend: Backend) -> float | None:
    if backend.actual != "cuda":
        return None
    try:
        return torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
    except Exception:
        return None
